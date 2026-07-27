#!/usr/bin/env python3
"""E7: dual-stream latent diffusion smoke probe (online encoding).

Trains a small CompactLatentDiT with OT-CFM flow matching directly on the
frozen dual-stream latent (z_geo || z_tex from the E5/R5 checkpoint), encoding
video online through the frozen StreamVGGT + compressor + tex_encoder. No
latent cache, no Wan — the question is only: can diffusion learn to produce
coherent latents in this space at all, and does the ABSOLUTE target beat the
RESIDUAL target as the diffusability spectra predict?

Arms (--target_mode):
  absolute  x1 = normalized z_1..z_7, cond = normalized z_0 (clean past).
  residual  x1 = normalized (z_t - z_0), cond = normalized z_0 (old contract).

Both arms share shapes [B, 7, G*G, geo+tex], the same DiT, and the same
normalization machinery (per-frame per-channel stats measured at startup and
frozen into the checkpoint), so the comparison isolates the parameterization.

Success metrics (logged every eval):
  eval/velocity_mse           flow loss on held-out clips
  eval/gen_std_ratio          std(sampled x1) / std(target x1) — under-
                              dispersion shows as << 1 (the old Wan symptom)
  sampled latents saved as .pt (denormalized, with z0 + stats) for offline
  RGB decoding on the H200 box (DualStreamDecoder interpolate ops are the
  NPU-risky path; keep them off the 910B).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.video_dataset import SpatialVidDataset, collate_fn
from data.loader_utils import multiprocessing_loader_kwargs
from data.token_utils import strip_special_tokens
from models.compact_dit import CompactLatentDiT
from models.dpt_latent_decoder import CompactCompressor
from models.flow_matching import OTCFM
from models.texture_encoder import TextureEncoder
from streamvggt.models.streamvggt import StreamVGGT
from utils.device import (
    configure_backend_compatibility, get_device, get_device_name,
    manual_seed_all, resolve_dtype,
)
from utils.distributed import is_main_process, setup_ddp
from utils.encoder_loader import load_encoder_checkpoint
from utils.training import (
    EMA, append_metrics, atomic_torch_save,
    build_optimizer, build_scheduler, capture_rng_state, restore_rng_state,
)


def parse_args():
    p = argparse.ArgumentParser(description='E7 dual-latent diffusion smoke')
    p.add_argument('--target_mode', type=str, required=True,
                   choices=['absolute', 'residual'])
    p.add_argument('--generator', type=str, default='dit',
                   choices=['dit', 'wan'],
                   help='dit = from-scratch CompactLatentDiT (E7 baseline); '
                        'wan = pretrained Wan backbone via WanCompactAdapter '
                        '(same forward interface, one-variable comparison)')
    p.add_argument('--wan_ckpt_dir', type=str, default='',
                   help='Wan2.1 checkpoint dir (required for --generator wan)')
    p.add_argument('--train_qkv_last_n', type=int, default=0,
                   help='wan: unfreeze QKV of the last N blocks; 0 = all '
                        '(affordable on 1.3B, unlike the 14B last-4 limit)')
    p.add_argument('--train_ffn_last_n', type=int, default=0,
                   help='wan: unfreeze FFN of the last N blocks. E8 showed '
                        'frozen FFNs bottleneck the latent translation '
                        '(fast early drop then plateau above from-scratch)')
    p.add_argument('--pseudo_text_tokens', type=int, default=8,
                   help='wan: learned null-prompt tokens so cross-attention '
                        'keeps running as in pretraining (0 = skip the '
                        'layer, the E8-v1 mistake)')
    p.add_argument('--adapter_lr', type=float, default=1e-4,
                   help='wan: lr for the fresh adapter layers '
                        '(input/output/time/i0/pseudo-context)')
    p.add_argument('--wan_lr', type=float, default=1e-5,
                   help='wan: lr for pretrained Wan parameters '
                        '(modulation/time path/QKV/FFN)')
    p.add_argument('--wan_freeze_steps', type=int, default=500,
                   help='wan: mute gradients into pretrained Wan params for '
                        'the first N optimizer steps so the random adapters '
                        'settle before touching the prior')
    p.add_argument('--time_shift_alpha', type=float, default=1.0,
                   help='RAE dimension-dependent time shift '
                        "t' = a*t/(1+(a-1)*t); 1.0 = off. Shifts training "
                        'time toward high noise, needed as token dim grows')
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--video_root', type=str, required=True)
    p.add_argument('--eval_csv', type=str, required=True)
    p.add_argument('--encoder_ckpt', type=str, required=True)
    p.add_argument('--dual_ae_ckpt', type=str, required=True,
                   help='E5/R5 dual-stream checkpoint (compressor + '
                        'tex_encoder are loaded frozen from it)')

    p.add_argument('--model_dim', type=int, default=768)
    p.add_argument('--spatial_depth', type=int, default=8)
    p.add_argument('--temporal_depth', type=int, default=4)
    p.add_argument('--num_heads', type=int, default=12)
    p.add_argument('--clean_frame0', action='store_true',
                   help='prepend clean normalized z0 to every DiT temporal '
                        'block and predict velocity only for future frames')
    p.add_argument('--block_schedule', type=str, default='phased',
                   choices=['phased', 'interleaved'])
    p.add_argument('--time_scale', type=float, default=1.0)
    p.add_argument('--lambda_motion', type=float, default=0.0)
    p.add_argument('--lambda_accel', type=float, default=0.0)
    p.add_argument('--lambda_geo_motion', type=float, default=0.0)
    p.add_argument('--aux_warmup_steps', type=int, default=1000)
    p.add_argument('--aux_ramp_steps', type=int, default=1000)
    p.add_argument('--aux_t_min', type=float, default=0.6)

    p.add_argument('--max_steps', type=int, default=6000)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--warmup_steps', type=int, default=300)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--stat_batches', type=int, default=16,
                   help='batches per rank used to measure latent stats')

    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--target_size', type=int, default=518)
    p.add_argument('--clip_duration_seconds', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--decode_retries', type=int, default=8)
    p.add_argument('--max_videos', type=int, default=0)

    p.add_argument('--eval_clips', type=int, default=16)
    p.add_argument('--sample_clips', type=int, default=4)
    p.add_argument('--sample_steps', type=int, default=30)

    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--save_every', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dtype', type=str, default='bf16')
    p.add_argument('--local_rank', type=int, default=0)
    return p.parse_args()


def load_dual_ae(args, device, dtype):
    """Frozen AE from an E5/R5 checkpoint, or an R6 bottleneck checkpoint.

    R6 checkpoints (args.has_bottleneck) additionally carry a
    LatentBottleneck; diffusion then runs in the COMPRESSED space
    (comp_dim tokens) and the decode path expands first.
    """
    from models.dual_stream_decoder import DualStreamDecoder
    from models.latent_bottleneck import LatentBottleneck
    ckpt = torch.load(args.dual_ae_ckpt, map_location='cpu', weights_only=False)
    ck = ckpt.get('args', {})
    grid = int(ck.get('latent_grid', 18))
    geo_dim = int(ck.get('geo_dim', 256))
    tex_dim = int(ck.get('tex_dim', 256))
    has_bottleneck = bool(ck.get('has_bottleneck', False))

    encoder = StreamVGGT(
        img_size=args.target_size, patch_size=14, embed_dim=1024)
    load_encoder_checkpoint(encoder, args.encoder_ckpt,
                            verbose=is_main_process())
    encoder = encoder.to(device=device, dtype=dtype).eval()

    compressor = CompactCompressor(
        levels=list(ck.get('levels', [4, 11, 17, 23])),
        token_dim=int(ck.get('token_dim', 2048)), cdim=geo_dim,
        latent_grid=grid, input_grid=args.target_size // 14)
    tex_encoder = TextureEncoder(
        out_dim=tex_dim, out_grid=grid,
        base_ch=int(ck.get('tex_base_ch', 64)), img_size=args.target_size,
        pack_mode=ck.get('tex_pack', 'avgpool'))
    decoder = DualStreamDecoder(
        geo_dim=geo_dim, tex_dim=tex_dim,
        base_dim=int(ck.get('decoder_base_dim', 384)),
        img_size=args.target_size, latent_grid=grid, use_checkpoint=False)
    bottleneck = LatentBottleneck(
        geo_dim + tex_dim, int(ck.get('comp_dim', 128))) \
        if has_bottleneck else None
    state = ckpt['model']
    modules = [(compressor, 'compressor.'), (tex_encoder, 'tex_encoder.'),
               (decoder, 'decoder.')]
    if bottleneck is not None:
        modules.append((bottleneck, 'bottleneck.'))
    for module, prefix in modules:
        sub = {k[len(prefix):]: v for k, v in state.items()
               if k.startswith(prefix)}
        if not sub:
            raise RuntimeError(
                f'{args.dual_ae_ckpt} has no {prefix}* keys — wrong '
                f'checkpoint type for this stage')
        module.load_state_dict(sub)  # strict

    compressor = compressor.to(device).eval()
    tex_encoder = tex_encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    frozen = [encoder, compressor, tex_encoder, decoder]
    if bottleneck is not None:
        bottleneck = bottleneck.to(device).eval()
        frozen.append(bottleneck)
    for m in frozen:
        for prm in m.parameters():
            prm.requires_grad_(False)
    latent_dim = bottleneck.comp_dim if bottleneck is not None \
        else geo_dim + tex_dim
    if is_main_process():
        print(f'  dual-AE: grid={grid} geo={geo_dim} tex={tex_dim} '
              f'bottleneck={"none" if bottleneck is None else latent_dim} '
              f'-> diffusion latent_dim={latent_dim}')
    return (encoder, compressor, tex_encoder, decoder, bottleneck,
            grid, latent_dim)


@torch.no_grad()
def encode_tokens(encoder, compressor, tex_encoder, frames, dtype,
                  bottleneck=None):
    """frames [B,S,3,H,W] -> tokens [B,S,N,D] fp32 (unnormalized).

    D = geo+tex (512) without a bottleneck, comp_dim (128) with one —
    diffusion always sees the space the decoder path starts from.
    """
    tokens_list, psi = encoder(frames.to(dtype))
    stripped = strip_special_tokens(tokens_list, psi)
    z_geo = compressor([t.float() for t in stripped])       # [B,S,C,G,G]
    z_geo = z_geo.permute(0, 1, 3, 4, 2).contiguous().float()
    z_tex = tex_encoder(frames.float()).float()             # [B,S,G,G,C]
    z = torch.cat([z_geo, z_tex], dim=-1)                   # [B,S,G,G,2C]
    if bottleneck is not None:
        z = bottleneck.encode(z)
    B, S, G, _, C = z.shape
    return z.reshape(B, S, G * G, C)


@torch.no_grad()
def measure_stats(loader, encoder, compressor, tex_encoder, args, device,
                  dtype, use_ddp, bottleneck=None):
    """Per-frame per-channel mean/std for absolute z and residual (z_t-z_0).

    Measured over stat_batches per rank and all-reduced, then frozen into the
    checkpoint: the normalization contract must not drift on resume.
    """
    S, C = args.seq_len, None
    sums = sq = None
    rsums = rsq = None
    count = 0
    it = iter(loader)
    for _ in range(args.stat_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        frames = batch['frames'].to(device)
        z = encode_tokens(encoder, compressor, tex_encoder, frames, dtype,
                          bottleneck)
        B, S, N, C = z.shape
        r = z[:, 1:] - z[:, :1]
        # float32 accumulation: NPU float64 support is poor and HCCL cannot
        # all_reduce fp64. Values are ~1e-1 scale over ~1e4 rows per rank,
        # well within fp32 precision.
        flat = z.permute(1, 0, 2, 3).reshape(S, -1, C).float()
        rflat = r.permute(1, 0, 2, 3).reshape(S - 1, -1, C).float()
        if sums is None:
            sums = flat.sum(1)
            sq = (flat ** 2).sum(1)
            rsums = rflat.sum(1)
            rsq = (rflat ** 2).sum(1)
        else:
            sums += flat.sum(1)
            sq += (flat ** 2).sum(1)
            rsums += rflat.sum(1)
            rsq += (rflat ** 2).sum(1)
        count += B * N
    if use_ddp:
        cnt = torch.tensor([count], dtype=torch.float32, device=device)
        for t in (sums, sq, rsums, rsq):
            dist.all_reduce(t)
        dist.all_reduce(cnt)
        count = int(cnt.item())
    mean = (sums / count).float()
    std = ((sq / count).float() - mean ** 2).clamp(min=1e-8).sqrt()
    rmean = (rsums / count).float()
    rstd = ((rsq / count).float() - rmean ** 2).clamp(min=1e-8).sqrt()
    return {'abs_mean': mean.cpu(), 'abs_std': std.cpu(),
            'res_mean': rmean.cpu(), 'res_std': rstd.cpu(),
            'count': count}


def build_targets(z, stats, target_mode, device):
    """z [B,S,N,C] unnormalized -> (x1 [B,S-1,N,C], cond [B,1,N,C])."""
    am = stats['abs_mean'].to(device)   # [S, C]
    asd = stats['abs_std'].to(device)
    cond = (z[:, :1] - am[0]) / asd[0]
    if target_mode == 'absolute':
        m = am[1:].unsqueeze(0).unsqueeze(2)    # [1, S-1, 1, C]
        s = asd[1:].unsqueeze(0).unsqueeze(2)
        x1 = (z[:, 1:] - m) / s
    else:
        r = z[:, 1:] - z[:, :1]
        m = stats['res_mean'].to(device).unsqueeze(0).unsqueeze(2)
        s = stats['res_std'].to(device).unsqueeze(0).unsqueeze(2)
        x1 = (r - m) / s
    return x1, cond


def denorm_to_full(x1, stats, target_mode, z0, device):
    """Normalized x1 [B,S-1,N,C] -> absolute latents [B,S,N,C] incl. z0."""
    if target_mode == 'absolute':
        m = stats['abs_mean'][1:].unsqueeze(0).unsqueeze(2).to(device)
        s = stats['abs_std'][1:].unsqueeze(0).unsqueeze(2).to(device)
        z_future = x1 * s + m
    else:
        m = stats['res_mean'].unsqueeze(0).unsqueeze(2).to(device)
        s = stats['res_std'].unsqueeze(0).unsqueeze(2).to(device)
        z_future = z0 + (x1 * s + m)
    return torch.cat([z0, z_future], dim=1)


def masked_auxiliary_loss(error, t, threshold):
    """Mean a per-element error over high-t samples without slicing ranks."""
    per_sample = error.float().reshape(error.shape[0], -1).mean(dim=1)
    weights = (t.float() >= threshold).to(per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp(min=1.0)


def auxiliary_scale(step, warmup_steps, ramp_steps):
    if step < warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, (step - warmup_steps + 1) / ramp_steps)


def motion_auxiliary_losses(x1_pred, z, stats, t, bottleneck,
                            geo_dim, t_min):
    """Trajectory losses in unnormalized compressed and expanded geo space."""
    mean = stats['abs_mean'][1:].to(x1_pred.device)
    std = stats['abs_std'][1:].to(x1_pred.device)
    future_pred = (
        x1_pred.float() * std.unsqueeze(0).unsqueeze(2)
        + mean.unsqueeze(0).unsqueeze(2))
    pred = torch.cat([z[:, :1].float(), future_pred], dim=1)
    target = z.float()

    pred_motion = pred[:, 1:] - pred[:, :-1]
    target_motion = target[:, 1:] - target[:, :-1]
    motion = masked_auxiliary_loss(
        (pred_motion - target_motion).square(), t, t_min)

    pred_accel = pred_motion[:, 1:] - pred_motion[:, :-1]
    target_accel = target_motion[:, 1:] - target_motion[:, :-1]
    accel = masked_auxiliary_loss(
        (pred_accel - target_accel).square(), t, t_min)

    if bottleneck is not None:
        pred_geo = bottleneck.decode(pred)[..., :geo_dim]
        target_geo = bottleneck.decode(target)[..., :geo_dim]
    else:
        pred_geo, target_geo = pred[..., :geo_dim], target[..., :geo_dim]
    pred_geo_motion = pred_geo[:, 1:] - pred_geo[:, :-1]
    target_geo_motion = target_geo[:, 1:] - target_geo[:, :-1]
    geo_motion = masked_auxiliary_loss(
        (pred_geo_motion - target_geo_motion).square(), t, t_min)
    return motion, accel, geo_motion


@contextlib.contextmanager
def ema_weights(model, ema):
    """Temporarily evaluate with EMA parameters, then restore train weights."""
    parameters = dict(model.named_parameters())
    backup = {}
    try:
        for name, shadow in ema.shadow.items():
            if name not in parameters:
                continue
            parameter = parameters[name]
            backup[name] = parameter.detach().clone()
            parameter.data.copy_(shadow.to(
                device=parameter.device, dtype=parameter.dtype))
        yield
    finally:
        for name, value in backup.items():
            parameters[name].data.copy_(value)


# --------------------------------------------------------------------------
# Wan adapter support (patterns proven on the cluster by
# train_cached_wan_compact_diffusion.py): the full Wan backbone is loaded
# from --wan_ckpt_dir on every run; checkpoints carry ONLY the trainable
# parameters (adapters + modulation + time path + unfrozen QKV), and the
# EMA shadows only those. Frozen parameters are cast to bf16 to halve
# resident memory; trainable ones stay fp32 (so the adapter's lazy
# time-embedding float32 conversion is a no-op and cannot break DDP
# gradient buckets).
# --------------------------------------------------------------------------

def cast_frozen_parameters(model, dtype):
    for parameter in model.parameters():
        if parameter.requires_grad or not parameter.is_floating_point():
            continue
        parameter.data = parameter.data.to(dtype=dtype)


def trainable_state_dict(model):
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad}
    return {name: value.detach().cpu()
            for name, value in model.state_dict().items()
            if name in trainable}


def load_trainable_state_dict(model, state_dict, label):
    parameters = dict(model.named_parameters())
    missing = []
    for name, parameter in parameters.items():
        if not parameter.requires_grad:
            continue
        if name not in state_dict:
            missing.append(name)
            continue
        parameter.data.copy_(state_dict[name].to(
            device=parameter.device, dtype=parameter.dtype))
    unexpected = sorted(set(state_dict) - set(parameters))
    if missing or unexpected:
        raise ValueError(
            f'{label} trainable state mismatch: '
            f'missing={missing[:8]}, unexpected={unexpected[:8]}')


class TrainableEMA:
    """EMA over trainable parameters only (the Wan backbone is excluded)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad}

    def update(self, model):
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name] = self.shadow[name].to(parameter.device)
            self.shadow[name].mul_(self.decay).add_(
                parameter.detach().float(), alpha=1 - self.decay)

    def state_dict(self):
        return {name: value.detach().cpu()
                for name, value in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {name: value.detach().float().clone()
                       for name, value in state_dict.items()}

    def to(self, device):
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self


def build_wan_optimizer(model, adapter_lr, wan_lr, wd):
    """Two-speed AdamW: fresh adapter layers learn fast, pretrained Wan
    parameters move slowly. The single-LR compromise (E8-v1, 5e-5 for both)
    starved the translation layers while over-driving the prior — visible
    as clipped grad_norm plus an early plateau."""
    groups = {('adapter', True): [], ('adapter', False): [],
              ('wan', True): [], ('wan', False): []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        family = 'wan' if name.startswith('wan.') else 'adapter'
        decay = not (p.ndim < 2 or 'norm' in name or 'bias' in name
                     or 'modulation' in name or 'pseudo_context' in name)
        groups[(family, decay)].append(p)
    param_groups = []
    for (family, decay), params in groups.items():
        if not params:
            continue
        param_groups.append({
            'params': params,
            'lr': adapter_lr if family == 'adapter' else wan_lr,
            'weight_decay': wd if decay else 0.0,
            'name': family,
        })
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)


def shift_time(t, alpha):
    """RAE dimension-dependent shift t' = a*t/(1+(a-1)*t) (identity at a=1).

    Pushes training/sampling time toward the high-noise end; RAE found the
    correction increasingly important as token dimensionality grows."""
    if alpha == 1.0:
        return t
    return alpha * t / (1.0 + (alpha - 1.0) * t)


class ShiftedOTCFM(OTCFM):
    """OT-CFM with the RAE time shift applied to the sampled training t."""

    def __init__(self, model, time_shift_alpha=1.0):
        super().__init__(model)
        self.time_shift_alpha = time_shift_alpha

    def compute_loss(self, x1, cond=None, text_emb=None,
                     return_outputs=False):
        x0 = torch.randn_like(x1)
        t = torch.rand((x1.shape[0],), device=x1.device)
        t = shift_time(t, self.time_shift_alpha).to(dtype=x1.dtype)
        t_expand = t.view(-1, *([1] * (x1.dim() - 1)))
        xt = (1 - t_expand) * x0 + t_expand * x1
        v_target = x1 - x0
        v_pred = self.model(xt, t, cond=cond, text_emb=text_emb)
        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        if not return_outputs:
            return loss
        x1_pred = xt + (1 - t_expand) * v_pred
        return {'loss': loss, 't': t, 'xt': xt, 'v_pred': v_pred,
                'v_target': v_target, 'x1_pred': x1_pred}


def main():
    args = parse_args()
    if args.clean_frame0 and args.generator != 'dit':
        raise ValueError('--clean_frame0 currently supports --generator dit only')
    if args.clean_frame0 and args.target_mode != 'absolute':
        raise ValueError('--clean_frame0 requires --target_mode absolute')
    aux_enabled = any(value > 0 for value in (
        args.lambda_motion, args.lambda_accel, args.lambda_geo_motion))
    if aux_enabled and not args.clean_frame0:
        raise ValueError('motion auxiliaries require --clean_frame0')
    if any(value < 0 for value in (
            args.lambda_motion, args.lambda_accel,
            args.lambda_geo_motion, args.aux_t_min)):
        raise ValueError('motion auxiliary weights and aux_t_min must be nonnegative')
    if args.aux_warmup_steps < 0 or args.aux_ramp_steps < 0:
        raise ValueError('auxiliary warmup/ramp steps must be nonnegative')
    use_ddp, rank, local_rank, world_size = setup_ddp()
    main_process = is_main_process()
    device = get_device(local_rank)
    device_type = get_device_name()
    dtype = resolve_dtype(args.dtype)
    configure_backend_compatibility(device_type)
    manual_seed_all(args.seed + rank)

    if main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
        print(f'E7 diffusion smoke: target_mode={args.target_mode} '
              f'device={device_type} world={world_size} dtype={args.dtype}')

    encoder, compressor, tex_encoder, decoder, bottleneck, grid, latent_dim = \
        load_dual_ae(args, device, dtype)
    num_tokens = grid * grid
    geo_latent_dim = (
        bottleneck.full_dim // 2 if bottleneck is not None else latent_dim // 2)
    future = args.seq_len - 1

    if args.generator == 'wan':
        if not args.wan_ckpt_dir or not os.path.isdir(args.wan_ckpt_dir):
            raise FileNotFoundError(
                f'--generator wan requires --wan_ckpt_dir, '
                f'got {args.wan_ckpt_dir!r}')
        # Lazy import: pulls the vendored Wan2.1 package only on this path.
        from models.wan_compact_adapter import WanCompactAdapter
        core = WanCompactAdapter(
            args.wan_ckpt_dir, latent_dim=latent_dim, latent_grid=grid,
            seq_len=future, i0_condition=True, train_text_adapter=False,
            train_qkv=True, train_qkv_last_n=args.train_qkv_last_n,
            train_ffn_last_n=args.train_ffn_last_n,
            num_pseudo_text=args.pseudo_text_tokens)
        # Frozen backbone -> bf16 (memory); trainable stay fp32.
        cast_frozen_parameters(core, dtype)
        core = core.to(device)
        ema = TrainableEMA(core, decay=args.ema_decay)
        # Wan matmuls need autocast (frozen bf16 x fp32 adapters); the DiT
        # baseline stays pure fp32 so E7-a numbers remain comparable.
        amp_ctx = lambda: torch.autocast(device_type=device_type, dtype=dtype)
    else:
        core = CompactLatentDiT(
            latent_dim=latent_dim, num_tokens=num_tokens,
            model_dim=args.model_dim, spatial_depth=args.spatial_depth,
            temporal_depth=args.temporal_depth, num_heads=args.num_heads,
            seq_len=future, text_cond=False,
            i0_condition=not args.clean_frame0,
            clean_frame0=args.clean_frame0,
            block_schedule=args.block_schedule,
            time_scale=args.time_scale).to(device)
        ema = EMA(core, decay=args.ema_decay, dtype=torch.float32).to(device)
        amp_ctx = contextlib.nullcontext

    n_params = sum(p.numel() for p in core.parameters())
    n_train = sum(p.numel() for p in core.parameters() if p.requires_grad)
    if args.generator == 'wan' and n_train > 1.5e9:
        raise RuntimeError(
            f'{n_train / 1e9:.2f}B trainable parameters would need a '
            f'{n_train * 4 / 1e9:.1f}GB DDP gradient bucket — this is the '
            f'known 14B full-QKV OOM. Use the Wan2.1-T2V-1.3B checkpoint, '
            f'or --train_qkv_last_n 4 for larger backbones.')
    if main_process:
        condition_mode = 'clean-frame0' if args.clean_frame0 else 'additive-i0'
        print(f'  generator={args.generator}: {n_train / 1e6:.1f}M trainable '
              f'/ {n_params / 1e6:.1f}M total  '
              f'tokens/frame={num_tokens} latent_dim={latent_dim} '
              f'condition={condition_mode} schedule={args.block_schedule} '
              f'time_scale={args.time_scale:g}')

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            core, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False)
    else:
        model = core
    cfm = ShiftedOTCFM(model, time_shift_alpha=args.time_shift_alpha)
    if args.generator == 'wan':
        optimizer = build_wan_optimizer(
            core, args.adapter_lr, args.wan_lr, args.wd)
    else:
        optimizer = build_optimizer(core, lr=args.lr, wd=args.wd)
    scheduler = build_scheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=args.max_steps)

    dataset = SpatialVidDataset(
        csv_path=args.csv, video_root=args.video_root,
        seq_len=args.seq_len, target_size=args.target_size,
        max_videos=args.max_videos,
        clip_duration_seconds=args.clip_duration_seconds,
        decode_retries=args.decode_retries)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) \
        if use_ddp else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device_type == 'cuda', drop_last=True,
        **multiprocessing_loader_kwargs(args.num_workers))

    eval_loader = None
    if main_process:
        eval_dataset = SpatialVidDataset(
            csv_path=args.eval_csv, video_root=args.video_root,
            seq_len=args.seq_len, target_size=args.target_size,
            max_videos=args.eval_clips, temporal_jitter=False,
            clip_duration_seconds=args.clip_duration_seconds,
            decode_retries=args.decode_retries)
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                 num_workers=0, collate_fn=collate_fn)

    # ---- Resume / stats (the normalization contract lives in the ckpt) ----
    global_step = 0
    stats = None
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        defaults = {
            'generator': 'dit', 'clean_frame0': False,
            'block_schedule': 'phased', 'time_scale': 1.0,
            'lambda_motion': 0.0, 'lambda_accel': 0.0,
            'lambda_geo_motion': 0.0, 'aux_warmup_steps': 1000,
            'aux_ramp_steps': 1000, 'aux_t_min': 0.6,
        }
        for key in ('target_mode', 'generator', 'clean_frame0',
                    'block_schedule', 'time_scale', 'lambda_motion',
                    'lambda_accel', 'lambda_geo_motion',
                    'aux_warmup_steps', 'aux_ramp_steps', 'aux_t_min'):
            if ckpt['args'].get(key, defaults.get(key)) != getattr(args, key):
                raise RuntimeError(
                    f'resume {key} {ckpt["args"].get(key)} != '
                    f'requested {getattr(args, key)}')
        if args.generator == 'wan':
            # Checkpoints carry trainable params only; the frozen backbone
            # was already loaded fresh from --wan_ckpt_dir.
            load_trainable_state_dict(core, ckpt['model'], 'model')
        else:
            core.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema']); ema = ema.to(device)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        global_step = ckpt.get('global_step', 0)
        stats = ckpt['latent_stats']
        restore_rng_state(ckpt.get('rng'))
        if main_process:
            print(f'Resumed from {args.resume} @ step {global_step}')
    if stats is None:
        if main_process:
            print(f'Measuring latent stats over {args.stat_batches} '
                  f'batches/rank ...')
        stats = measure_stats(loader, encoder, compressor, tex_encoder,
                              args, device, dtype, use_ddp, bottleneck)
        if main_process:
            print(f'  stats over {stats["count"]} tokens; '
                  f'abs_std mean={stats["abs_std"].mean():.4f} '
                  f'res_std mean={stats["res_std"].mean():.4f}')

    writer = SummaryWriter(os.path.join(args.output_dir, 'tb')) \
        if main_process else None
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')
    # Windowed throughput counters (reset at every log line): the cluster
    # convention is tokens/s/npu measured over the recent window, not a
    # run-lifetime average.
    window_tokens = 0
    window_start = time.time()

    def save_ckpt(step):
        payload = {
            'model': (trainable_state_dict(core) if args.generator == 'wan'
                      else core.state_dict()),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'global_step': step, 'args': vars(args),
            'latent_stats': stats, 'rng': capture_rng_state(),
        }
        atomic_torch_save(payload, os.path.join(
            args.output_dir, f'checkpoint_step{step:07d}.pt'))
        atomic_torch_save(payload, os.path.join(
            args.output_dir, 'checkpoint_latest.pt'))
        print(f'  saved checkpoint @ step {step}')

    @torch.no_grad()
    def run_eval(step):
        core.eval()
        vmse, n = 0.0, 0
        vmse_buckets = {value: 0.0 for value in (0.1, 0.3, 0.5, 0.7, 0.9)}
        gen_stds, tgt_stds = [], []
        motion_ratios, motion_cosines = [], []
        motion_ratio_frames, motion_cosine_frames = [], []
        latent_mse_frames = []
        saved = 0
        for i, batch in enumerate(eval_loader):
            frames = batch['frames'].to(device)
            z = encode_tokens(encoder, compressor, tex_encoder, frames, dtype,
                              bottleneck)
            x1, cond = build_targets(z, stats, args.target_mode, device)
            eval_generator = torch.Generator(device='cpu')
            eval_generator.manual_seed(args.seed + 1009 * i)
            x0 = torch.randn(
                x1.shape, generator=eval_generator, dtype=torch.float32,
                device='cpu').to(device=device, dtype=x1.dtype)
            t = torch.rand(
                (x1.shape[0],), generator=eval_generator,
                dtype=torch.float32, device='cpu').to(
                    device=device, dtype=x1.dtype)
            te = t.view(-1, 1, 1, 1)
            with amp_ctx():
                v = core((1 - te) * x0 + te * x1, t, cond=cond)
            vmse += torch.nn.functional.mse_loss(v.float(), x1 - x0).item()
            for bucket in vmse_buckets:
                bucket_t = torch.full_like(t, bucket)
                bucket_te = bucket_t.view(-1, 1, 1, 1)
                with amp_ctx():
                    bucket_v = core(
                        (1 - bucket_te) * x0 + bucket_te * x1,
                        bucket_t, cond=cond)
                vmse_buckets[bucket] += torch.nn.functional.mse_loss(
                    bucket_v.float(), x1 - x0).item()
            n += 1
            if saved < args.sample_clips:
                sample_generator = torch.Generator(device='cpu')
                sample_generator.manual_seed(args.seed + 1009 * i + 17)
                sample_noise = torch.randn(
                    x1.shape, generator=sample_generator,
                    dtype=torch.float32, device='cpu').to(
                        device=device, dtype=cond.dtype)
                gen = cfm_sample(cond, x1.shape, sample_noise)
                gen_stds.append(gen.float().std().item())
                tgt_stds.append(x1.float().std().item())
                # z0-copy guard in the unnormalized latent space. Magnitude
                # alone can be correct while motion direction is wrong, so log
                # both aggregate and per-horizon values.
                z_gen = denorm_to_full(gen, stats, args.target_mode,
                                       z[:, :1], device)
                d_gen = (z_gen[:, 1:] - z[:, :1]).flatten(2)
                d_tgt = (z[:, 1:] - z[:, :1]).flatten(2)
                ratio_by_frame = (
                    d_gen.norm(dim=2) / d_tgt.norm(dim=2).clamp(min=1e-8)
                ).mean(0)
                cosine_by_frame = torch.nn.functional.cosine_similarity(
                    d_gen, d_tgt, dim=2).mean(0)
                motion_ratio_frames.append(ratio_by_frame.cpu())
                motion_cosine_frames.append(cosine_by_frame.cpu())
                latent_mse_frames.append(
                    (z_gen[:, 1:] - z[:, 1:]).square().flatten(2).mean(2)
                    .mean(0).cpu())
                motion_ratios.append(ratio_by_frame.mean().item())
                motion_cosines.append(cosine_by_frame.mean().item())
                torch.save(
                    {'sampled_x1': gen.float().cpu(),
                     'target_x1': x1.float().cpu(),
                     'z0_unnorm': z[:, :1].float().cpu(),
                     'stats': stats, 'target_mode': args.target_mode,
                     'clean_frame0': args.clean_frame0,
                     'grid': grid, 'step': step},
                    os.path.join(args.output_dir, 'samples',
                                 f'step{step:07d}_clip{i}.pt'))
                if saved == 0:
                    save_preview(step, frames, z, gen)
                saved += 1
        row = {
            'step': step, 'target_mode': args.target_mode,
            'eval/weights': 'ema',
            'eval/velocity_mse': vmse / max(n, 1),
            'eval/gen_std': float(sum(gen_stds) / max(len(gen_stds), 1)),
            'eval/target_std': float(sum(tgt_stds) / max(len(tgt_stds), 1)),
        }
        for bucket, value in vmse_buckets.items():
            row[f'eval/velocity_mse_t{bucket:.1f}'] = value / max(n, 1)
        row['eval/gen_std_ratio'] = (
            row['eval/gen_std'] / max(row['eval/target_std'], 1e-8))
        row['eval/motion_ratio'] = float(
            sum(motion_ratios) / max(len(motion_ratios), 1))
        row['eval/motion_cosine'] = float(
            sum(motion_cosines) / max(len(motion_cosines), 1))
        if motion_ratio_frames:
            ratio_frames = torch.stack(motion_ratio_frames).mean(0)
            cosine_frames = torch.stack(motion_cosine_frames).mean(0)
            for frame_idx, (ratio, cosine, latent_mse) in enumerate(
                    zip(ratio_frames, cosine_frames,
                        torch.stack(latent_mse_frames).mean(0)), start=1):
                row[f'eval/motion_ratio_f{frame_idx}'] = float(ratio)
                row[f'eval/motion_cosine_f{frame_idx}'] = float(cosine)
                row[f'eval/latent_mse_f{frame_idx}'] = float(latent_mse)
        append_metrics(metrics_path, row)
        if writer:
            for k, v_ in row.items():
                if isinstance(v_, (int, float)):
                    writer.add_scalar(k, v_, step)
        print(f'  [eval] step {step}: vmse={row["eval/velocity_mse"]:.4f} '
              f'gen_std_ratio={row["eval/gen_std_ratio"]:.3f} '
              f'motion_ratio={row["eval/motion_ratio"]:.3f} '
              f'motion_cosine={row["eval/motion_cosine"]:.3f}')
        core.train()

    preview_ok = [True]

    @torch.no_grad()
    def save_preview(step, frames, z, gen_x1):
        """PNG grid (GT | AE recon | generated) for direct viewing; guarded —
        decoder interpolate is the historically NPU-fragile op, so one
        failure disables previews (the .pt latents still get saved and can
        be decoded on the H200 box)."""
        if not preview_ok[0]:
            return
        try:
            import numpy as np
            from PIL import Image
            B, S, N, C = z.shape
            z_gen = denorm_to_full(gen_x1, stats, args.target_mode,
                                   z[:, :1], device)
            def decode(zz):
                zz = zz.reshape(1, S, grid, grid, C)
                if bottleneck is not None:
                    zz = bottleneck.decode(zz.float())
                full = zz.shape[-1]
                geo_dim = full // 2
                rgb = decoder(zz[..., :geo_dim].float(),
                              zz[..., geo_dim:].float(),
                              frames_chunk_size=4)
                return rgb[..., :3].clamp(0, 1)[0]        # [S,H,W,3]
            recon = decode(z[:1])
            genv = decode(z_gen[:1])
            gt = frames[0].float().clamp(0, 1).permute(0, 2, 3, 1)
            rows = torch.cat(
                [torch.cat([gt[s], recon[s], genv[s]], dim=1)
                 for s in range(S)], dim=0)
            img = (rows.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(
                args.output_dir, 'samples', f'step{step:07d}_preview.png'))
        except Exception as exc:
            preview_ok[0] = False
            print(f'  [WARN] preview decode failed ({exc}); previews '
                  f'disabled, decode the saved .pt on the H200 box instead')

    @torch.no_grad()
    def cfm_sample(cond, shape, initial_noise=None):
        z = initial_noise.clone() if initial_noise is not None else torch.randn(
            shape, device=device, dtype=cond.dtype)
        n_steps = args.sample_steps
        # Integrate on the SHIFTED time grid (must match the training-time
        # distribution): uniform grid in u, warped t = shift(u), step dt
        # taken from consecutive warped points.
        import numpy as _np
        us = _np.linspace(0.0, 1.0, n_steps + 1)
        ts = [shift_time(float(u), args.time_shift_alpha) for u in us]
        for i in range(n_steps):
            t = torch.full((shape[0],), ts[i], device=device,
                           dtype=cond.dtype)
            dt = ts[i + 1] - ts[i]
            with amp_ctx():
                v = core(z, t, cond=cond)
            z = z + v.float() * dt
        return z

    if main_process:
        print(f'Training: {args.max_steps} steps, '
              f'{len(loader)} batches/epoch/rank')
    model.train()
    epoch = 0
    optimizer.zero_grad(set_to_none=True)
    done = False
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(loader, disable=not main_process, desc=f'ep{epoch}')
        for it_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(device)
            z = encode_tokens(encoder, compressor, tex_encoder, frames, dtype,
                              bottleneck)
            x1, cond = build_targets(z, stats, args.target_mode, device)
            with amp_ctx():
                flow_out = cfm.compute_loss(
                    x1, cond=cond, return_outputs=True)
                flow_loss = flow_out['loss']
                zero = flow_loss.new_zeros(())
                motion_loss = accel_loss = geo_motion_loss = zero
                aux_scale = auxiliary_scale(
                    global_step, args.aux_warmup_steps,
                    args.aux_ramp_steps)
                if aux_enabled and aux_scale > 0:
                    motion_loss, accel_loss, geo_motion_loss = \
                        motion_auxiliary_losses(
                            flow_out['x1_pred'], z, stats, flow_out['t'],
                            bottleneck, geo_latent_dim, args.aux_t_min)
                total_loss = (
                    flow_loss
                    + aux_scale * args.lambda_motion * motion_loss
                    + aux_scale * args.lambda_accel * accel_loss
                    + aux_scale * args.lambda_geo_motion * geo_motion_loss)
                loss = total_loss / args.accum_steps
            loss.backward()
            if (it_idx + 1) % args.accum_steps != 0:
                continue
            if args.generator == 'wan' and global_step < args.wan_freeze_steps:
                # Adapter warm-up: drop gradients into the pretrained prior
                # until the random translation layers have settled (DDP-safe:
                # grads are reduced normally, then discarded before the step).
                for name, p in core.named_parameters():
                    if name.startswith('wan.') and p.grad is not None:
                        p.grad = None
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in core.parameters() if p.requires_grad],
                args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(core)
            global_step += 1
            # DI_throughput: tokens/s/npu over the window since the last log
            # line (cluster monitoring convention). tokens = latent tokens
            # consumed per optimizer step per rank: batch x accum x (S-1
            # future frames) x N tokens/frame.
            window_tokens += (x1.shape[0] * args.accum_steps
                              * x1.shape[1] * x1.shape[2])

            if main_process and global_step % args.log_every == 0:
                now = time.time()
                di_throughput = window_tokens / max(now - window_start, 1e-12)
                window_tokens = 0
                window_start = now
                row = {
                    'step': global_step, 'target_mode': args.target_mode,
                    'train/loss': float(total_loss.item()),
                    'train/flow_loss': float(flow_loss.item()),
                    'train/motion_loss': float(motion_loss.item()),
                    'train/accel_loss': float(accel_loss.item()),
                    'train/geo_motion_loss': float(geo_motion_loss.item()),
                    'train/aux_scale': float(aux_scale),
                    'train/grad_norm': float(grad_norm),
                    'train/lr': scheduler.get_last_lr()[0],
                    'DI_throughput': di_throughput,
                }
                append_metrics(metrics_path, row)
                if writer:
                    for k, v_ in row.items():
                        if isinstance(v_, (int, float)):
                            writer.add_scalar(k, v_, global_step)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                pbar.write(
                    f'{timestamp}: [train {epoch + 1} '
                    f'{global_step}/{args.max_steps}] '
                    f'loss: {row["train/loss"]:.4f} | '
                    f'DI_throughput: {di_throughput:.2f} tokens/s/npu')
                pbar.set_postfix({
                    'loss': f'{row["train/loss"]:.4f}',
                    'DI_throughput': f'{di_throughput:.2f}',
                })

            should_eval = global_step % args.eval_every == 0
            should_save = global_step % args.save_every == 0
            if should_eval or should_save:
                if use_ddp:
                    dist.barrier()
                if main_process:
                    if should_eval:
                        with ema_weights(core, ema):
                            run_eval(global_step)
                    if should_save:
                        save_ckpt(global_step)
                if use_ddp:
                    dist.barrier()
            if global_step >= args.max_steps:
                done = True
                break
        epoch += 1

    if use_ddp:
        dist.barrier()
    if main_process:
        with ema_weights(core, ema):
            run_eval(global_step)
        save_ckpt(global_step)
        if writer:
            writer.close()
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
