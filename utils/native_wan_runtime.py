"""Isolated native Wan I2V compatibility, without importing compact adapters.

Learned interfaces and solver are unchanged. Restore upstream FP32 modulation;
store only autocast Linear/Conv weights in BF16, retaining FP32 time/head/norms.
NPU attention API: https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000520.html
Upstream: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
"""
import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
              softmax_scale=None, q_scale=None, causal=False,
              window_size=(-1, -1), deterministic=False,
              dtype=torch.float16, version=None):
    if dropout_p or window_size != (-1, -1):
        raise ValueError('native diagnostic supports inference/full attention only')
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError('expected BLHD attention tensors')
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError('batch/head dimensions differ')
    output = torch.zeros_like(q)
    qlens = [q.shape[1]] * len(q) if q_lens is None else torch.as_tensor(q_lens).cpu().tolist()
    klens = [k.shape[1]] * len(k) if k_lens is None else torch.as_tensor(k_lens).cpu().tolist()
    if len(qlens) != len(q) or len(klens) != len(k):
        raise ValueError('length count differs')
    scale = q.shape[-1] ** -.5 if softmax_scale is None else softmax_scale
    for i, (nq, nk) in enumerate(zip(qlens, klens)):
        if not 0 < nq <= q.shape[1] or not 0 < nk <= k.shape[1]:
            raise ValueError('invalid attention lengths')
        values = [a[i:i+1, :n].transpose(1, 2).contiguous()
                  for a, n in ((q, nq), (k, nk), (v, nk))]
        if q.device.type == 'npu':
            values = [a.to(dtype=a.dtype if a.dtype in (torch.float16, torch.bfloat16) else dtype)
                      for a in values]
        qi, ki, vi = [a.to(values[-1].dtype) for a in values]
        if q_scale is not None:
            qi = qi * q_scale
        blocked = None
        if causal:
            # FlashAttention uses bottom-right aligned causal masks for unequal lengths.
            blocked = (torch.arange(nk, device=q.device)[None, :] >
                       torch.arange(nq, device=q.device)[:, None] + nk - nq)
        if q.device.type == 'npu':
            import torch_npu
            value = torch_npu.npu_fusion_attention(
                qi, ki, vi, q.shape[2], input_layout='BNSD',
                atten_mask=blocked, scale=scale, keep_prob=1.,
                pre_tockens=2147483647, next_tockens=2147483647)[0]
        else:
            value = F.scaled_dot_product_attention(qi, ki, vi,
                attn_mask=None if blocked is None else ~blocked, scale=scale)
        output[i:i+1, :nq] = value.transpose(1, 2).to(output.dtype)
    return output


def rope_params(max_seq_len, dim, theta=10000):
    # Construct native phases in CPU float64; keep real pairs on the NPU.
    phase = torch.outer(torch.arange(max_seq_len, dtype=torch.float64),
                        theta ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    return torch.stack((phase.cos(), phase.sin()), -1).float()


def rope_apply(x, grid_sizes, freqs):
    n, c = x.shape[2], x.shape[3] // 2
    parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    results = []
    for i, (t, h, w) in enumerate(grid_sizes.tolist()):
        count = t * h * w
        phase = torch.cat((
            parts[0][:t].view(t, 1, 1, -1, 2).expand(t, h, w, -1, 2),
            parts[1][:h].view(1, h, 1, -1, 2).expand(t, h, w, -1, 2),
            parts[2][:w].view(1, 1, w, -1, 2).expand(t, h, w, -1, 2)), dim=3).reshape(count, 1, c, 2)
        pairs = x[i, :count].float().reshape(count, n, c, 2)
        real, imag = pairs.unbind(-1)
        cos, sin = phase.unbind(-1)
        rotated = torch.stack((real*cos-imag*sin, real*sin+imag*cos), -1).flatten(2)
        results.append(torch.cat((rotated, x[i, count:].float()), 0))
    return torch.stack(results)


def native_block_forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
    if e.dtype != torch.float32:
        raise ValueError('native modulation requires FP32 time embedding')
    e = (self.modulation.float() + e).chunk(6, dim=1)
    y = self.self_attn(self.norm1(x).float() * (1+e[1]) + e[0], seq_lens, grid_sizes, freqs)
    x = x.float() + y.float() * e[2]
    x = x + self.cross_attn(self.norm3(x), context, context_lens)
    y = self.ffn(self.norm2(x).float() * (1+e[4]) + e[3])
    return x + y.float() * e[5]


def compact_storage(model):
    changed = []
    for name, module in model.named_modules():
        if name.split('.')[0] in ('time_embedding', 'time_projection', 'head'):
            continue
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv3d, torch.nn.Conv2d)):
            module.to(dtype=torch.bfloat16)
            changed.append(name)
    return changed


def load_pipeline(checkpoint_dir, device_id=0):
    # Package shells avoid wan.__init__ importing VACE/FLF dependencies.
    root = Path(__file__).resolve().parents[1] / 'Wan2.1' / 'wan'
    prefix = '_vggae_native_wan'
    for suffix in ('', '.modules', '.distributed', '.utils', '.configs'):
        name = prefix + suffix
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(root.joinpath(*suffix.strip('.').split('.'))) if suffix else str(root)]
            sys.modules[name] = module
    attn = importlib.import_module(prefix + '.modules.attention')
    attn.flash_attention = attention
    model_module = importlib.import_module(prefix + '.modules.model')
    model_module.rope_params = rope_params
    model_module.rope_apply = rope_apply
    model_module.WanAttentionBlock.forward = native_block_forward
    pipeline_module = importlib.import_module(prefix + '.image2video')
    config = importlib.import_module(prefix + '.configs.wan_i2v_14B').i2v_14B
    pipeline = pipeline_module.WanI2V(config, checkpoint_dir, device_id=device_id,
        rank=0, t5_cpu=True, init_on_cpu=True, dit_fsdp=False, t5_fsdp=False, use_usp=False)
    storage = compact_storage(pipeline.model)
    return pipeline, dict(runtime='native-i2v-ascend-v1', storage_bf16_modules=storage,
        preserved_fp32=['time_embedding', 'time_projection', 'head', 'norms', 'modulation'],
        modulation='upstream FP32', rope='real FP32 pairs from CPU FP64 phases',
        attention='npu_fusion_attention BNSD; exact per-sample valid lengths')


def attention_gate(device):
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        generator = torch.Generator().manual_seed(197)
        q, k, v = [torch.randn(shape, generator=generator).to(dtype)
                   for shape in ((2, 17, 2, 64), (2, 23, 2, 64), (2, 23, 2, 64))]
        for causal in (False, True):
            kw = dict(q_lens=[17, 11], k_lens=[23, 19], causal=causal, softmax_scale=.13)
            reference = attention(q.float(), k.float(), v.float(), **kw)
            actual = attention(q.to(device), k.to(device), v.to(device), **kw).float().cpu()
            rms = float((reference-actual).square().mean().sqrt())
            maximum = float((reference-actual).abs().max())
            if not torch.isfinite(actual).all() or rms > .015 or maximum > .08:
                raise RuntimeError(f'native attention gate failed: {dtype} {causal} {rms} {maximum}')
            rows.append(dict(dtype=str(dtype), causal=causal, rms=rms, max_abs=maximum))
    return rows
