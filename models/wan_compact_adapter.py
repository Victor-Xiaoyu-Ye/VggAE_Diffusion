"""Wan2.1 adapter for compact latent z_g.

Key design (from verification experiments + all prior learnings):
  1. DUAL time conditioning: concat at input (V1/V3: works) + adaLN in blocks (Wan native)
  2. Trainable: adapters + modulation + time_emb + optional last-N QKV
  3. Frozen: FFN, cross-attn, norms, RoPE freqs
  4. Uniform time sampling in flow matching (debug finding: t near 1 critical)

Input:  z_g_flat [B, S, N, latent_dim]  where N=latent_grid²
Output: v_pred [B, S, N, latent_dim]   predicted velocity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import math
import importlib.util
import sys, os
import types

try:
    torch.get_autocast_dtype("npu")
    _NPU_AUTOCAST_AVAILABLE = True
except RuntimeError:
    _NPU_AUTOCAST_AVAILABLE = False

if not _NPU_AUTOCAST_AVAILABLE:
    if not hasattr(torch, "npu"):
        class _DummyNPU:
            @staticmethod
            def current_device():
                return 0

            @staticmethod
            def is_available():
                return False

        torch.npu = _DummyNPU()

    _original_autocast = amp.autocast

    def _autocast_without_npu(*args, **kwargs):
        if kwargs.get("device_type") == "npu":
            kwargs = dict(kwargs)
            kwargs["device_type"] = "cuda" if torch.cuda.is_available() else "cpu"
        elif args and args[0] == "npu":
            args = ("cuda" if torch.cuda.is_available() else "cpu",) + args[1:]
        return _original_autocast(*args, **kwargs)

    amp.autocast = _autocast_without_npu

_wan_root = os.path.join(os.path.dirname(__file__), '..', 'Wan2.1')
_wan_modules_root = os.path.join(_wan_root, "wan", "modules")
_wan_direct_package = "_vgg_ae_wan_modules"
if _wan_direct_package not in sys.modules:
    package = types.ModuleType(_wan_direct_package)
    package.__path__ = [_wan_modules_root]
    sys.modules[_wan_direct_package] = package
_wan_model_name = f"{_wan_direct_package}.model"
if _wan_model_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _wan_model_name, os.path.join(_wan_modules_root, "model.py"))
    wan_model_module = importlib.util.module_from_spec(spec)
    sys.modules[_wan_model_name] = wan_model_module
    spec.loader.exec_module(wan_model_module)
else:
    wan_model_module = sys.modules[_wan_model_name]
WanModel = wan_model_module.WanModel
sinusoidal_embedding_1d = wan_model_module.sinusoidal_embedding_1d


class WanCompactAdapter(nn.Module):
    """Wan backbone adapted for compact latent flow matching."""

    def __init__(self, wan_checkpoint_dir, latent_dim=768, latent_grid=18,
                 seq_len=8, wan_dim=None, freq_dim=None, num_heads=None,
                 i0_condition=False, train_text_adapter=False,
                 train_qkv=True, train_qkv_last_n=0, train_ffn_last_n=0,
                 num_pseudo_text=0):
        super().__init__()

        # Load pretrained Wan backbone. Do not hard-code the 1.3B dimensions:
        # the scale script can point this adapter at larger Wan checkpoints.
        self.wan = WanModel.from_pretrained(wan_checkpoint_dir)
        inferred_wan_dim = int(getattr(self.wan, "dim"))
        inferred_freq_dim = int(getattr(self.wan, "freq_dim"))
        inferred_heads = int(getattr(self.wan, "num_heads"))
        if wan_dim is not None and int(wan_dim) != inferred_wan_dim:
            raise ValueError(
                f"wan_dim={wan_dim} does not match checkpoint dim "
                f"{inferred_wan_dim}")
        if freq_dim is not None and int(freq_dim) != inferred_freq_dim:
            raise ValueError(
                f"freq_dim={freq_dim} does not match checkpoint freq_dim "
                f"{inferred_freq_dim}")
        if num_heads is not None and int(num_heads) != inferred_heads:
            raise ValueError(
                f"num_heads={num_heads} does not match checkpoint heads "
                f"{inferred_heads}")
        self.wan_dim = inferred_wan_dim
        self.freq_dim = inferred_freq_dim
        self.num_heads = inferred_heads

        self.latent_dim = latent_dim
        self.latent_grid = latent_grid
        self.num_tokens = latent_grid ** 2
        self.seq_len = seq_len
        self.i0_condition = i0_condition
        self.train_text_adapter = train_text_adapter
        self.train_qkv = train_qkv
        self.train_qkv_last_n = int(train_qkv_last_n)
        self.train_ffn_last_n = int(train_ffn_last_n)
        self.num_pseudo_text = int(num_pseudo_text)

        # Learned pseudo-text context (E8-v1 finding): Wan pretraining ALWAYS
        # ran cross-attention — even the CFG unconditional branch encodes the
        # empty prompt, it never skips the layer. Passing context=None (v1)
        # removed a computation every block co-adapted with, shifting the
        # residual-stream distribution from block 0 on (fast early gain from
        # modulation rescaling, then a plateau above from-scratch). These
        # trainable tokens act as a learned null prompt, fed directly at
        # wan_dim (bypassing the UMT5 text_embedding projection).
        if self.num_pseudo_text > 0:
            self.pseudo_context = nn.Parameter(
                torch.randn(self.num_pseudo_text, self.wan_dim) * 0.02)

        # ---- Input: latent_dim → wan_dim with concat time injection ----
        self.time_concat_dim = 256
        self.time_concat_mlp = nn.Sequential(
            nn.Linear(self.freq_dim, self.time_concat_dim * 2),
            nn.SiLU(),
            nn.Linear(self.time_concat_dim * 2, self.time_concat_dim),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + self.time_concat_dim, self.wan_dim),
            nn.LayerNorm(self.wan_dim),
            nn.SiLU(),
            nn.Linear(self.wan_dim, self.wan_dim),
        )
        if i0_condition:
            self.i0_proj = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, self.wan_dim),
                nn.SiLU(),
                nn.Linear(self.wan_dim, self.wan_dim),
            )

        # ---- Output: wan_dim → latent_dim (zero-init) ----
        self.output_norm = nn.LayerNorm(self.wan_dim)
        self.output_proj = nn.Linear(self.wan_dim, latent_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # ---- Text projection: legacy CLIP 768 → native Wan dim ----
        self.text_proj = nn.Sequential(
            nn.Linear(768, self.wan_dim),
            nn.GELU(),
            nn.Linear(self.wan_dim, self.wan_dim),
        )

        # ---- Trainable parameter setup ----
        self._freeze_all()
        self._unfreeze_trainable()

        self._time_emb_converted = False

    def _freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def _unfreeze_trainable(self):
        """Unfreeze adapters, modulation, time path, and optional last-N QKV."""
        # Adapter layers
        for p in self.time_concat_mlp.parameters():
            p.requires_grad_(True)
        for p in self.input_proj.parameters():
            p.requires_grad_(True)
        if self.i0_condition:
            for p in self.i0_proj.parameters():
                p.requires_grad_(True)
        for p in self.output_norm.parameters():
            p.requires_grad_(True)
        for p in self.output_proj.parameters():
            p.requires_grad_(True)
        if self.num_pseudo_text > 0:
            self.pseudo_context.requires_grad_(True)
        if self.train_text_adapter:
            for p in self.text_proj.parameters():
                p.requires_grad_(True)

        # Wan time pathway
        for p in self.wan.time_embedding.parameters():
            p.requires_grad_(True)
        for p in self.wan.time_projection.parameters():
            p.requires_grad_(True)

        # Wan blocks: modulation + optionally QKV. On 14B, full-QKV DDP is too
        # large without FSDP/ZeRO, so callers can unfreeze only the last N
        # blocks while still adapting high-level temporal dynamics.
        num_blocks = len(self.wan.blocks)
        if self.train_qkv:
            if self.train_qkv_last_n <= 0:
                qkv_start = 0
            else:
                qkv_start = max(0, num_blocks - self.train_qkv_last_n)
        else:
            qkv_start = num_blocks
        self.qkv_trainable_blocks = max(0, num_blocks - qkv_start)
        # FFN unfreeze (E8 finding): with FFN frozen, the Wan arm showed the
        # adapter-bottleneck signature — fast early drop from modulation
        # adaptation, then a plateau ABOVE the from-scratch baseline. The
        # frozen FFNs (2/3 of params) hold Wan's feature language and cannot
        # be re-aimed at the dual-stream latent through QKV alone.
        if self.train_ffn_last_n > 0:
            ffn_start = max(0, num_blocks - self.train_ffn_last_n)
        else:
            ffn_start = num_blocks
        self.ffn_trainable_blocks = max(0, num_blocks - ffn_start)
        for index, blk in enumerate(self.wan.blocks):
            blk.modulation.requires_grad_(True)
            if index >= qkv_start:
                for name in ['q', 'k', 'v']:
                    attn_module = getattr(blk.self_attn, name)
                    for p in attn_module.parameters():
                        p.requires_grad_(True)
            if index >= ffn_start:
                for p in blk.ffn.parameters():
                    p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"WanCompactAdapter: {trainable/1e6:.1f}M trainable / "
            f"{total/1e9:.2f}B total "
            f"(qkv_blocks={self.qkv_trainable_blocks}/{len(self.wan.blocks)}, "
            f"ffn_blocks={self.ffn_trainable_blocks}/{len(self.wan.blocks)})")

    def _ensure_time_emb_float32(self):
        if self._time_emb_converted:
            return
        for m in [self.wan.time_embedding, self.wan.time_projection]:
            for p in m.parameters():
                p.data = p.data.float()
        self._time_emb_converted = True

    def _time_embed(self, t):
        """t: [B] in Wan's native [0, 1000] range."""
        self._ensure_time_emb_float32()
        with amp.autocast(device_type=t.device.type, enabled=False):
            e = sinusoidal_embedding_1d(self.freq_dim, t).float().to(device=t.device)
            e = self.wan.time_embedding(e.float())
            e = self.wan.time_projection(e.float())
        return e.unflatten(1, (6, self.wan_dim)).float()

    def _concat_time_embed(self, t):
        """t: [B] in [0, 1] → concat time embedding [B, time_concat_dim]."""
        half = self.freq_dim // 2
        emb = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) *
            (-math.log(10000) / (half - 1))
        )
        emb = (t.float() * 1000).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.time_concat_mlp(emb)

    def forward(self, z, t, cond=None, text_emb=None):
        """Flow matching forward.

        Args:
            z: [B, S, N, latent_dim] noisy latent
            t: [B] flow time in [0, 1]
            cond: optional first-frame compact latent [B, 1, N, latent_dim]
            text_emb: CLIP [B, L, 768] or native UMT5 [B, L, 4096]

        Returns:
            v: [B, S, N, latent_dim] predicted velocity
        """
        B, S, N, D = z.shape

        # ---- 1. Concat time at input (convert to model dtype for fp32 backbone) ----
        model_dtype = next(self.input_proj.parameters()).dtype
        t_concat = self._concat_time_embed(t)  # [B, time_concat_dim], float32
        t_concat = t_concat.unsqueeze(1).unsqueeze(1).expand(B, S, N, -1)
        x = torch.cat([z.to(dtype=model_dtype), t_concat], dim=-1)  # [B, S, N, D+time_dim]
        x = self.input_proj(x)  # [B, S, N, wan_dim]
        if self.i0_condition:
            if cond is None:
                raise ValueError("I0-conditioned WanCompactAdapter requires cond")
            if cond.dim() != 4 or cond.shape[0] != B or cond.shape[2:] != (N, D):
                raise ValueError(
                    f"Expected cond [B, T, N, D] compatible with {z.shape}, "
                    f"got {cond.shape}")
            i0_context = self.i0_proj(
                cond.to(dtype=model_dtype).mean(dim=1)).unsqueeze(1)
            x = x + i0_context.expand(B, S, N, self.wan_dim)
        x = x.reshape(B, S * N, self.wan_dim)  # [B, S*N, wan_dim]

        # ---- 2. Wan time embedding (adaLN) ----
        t_wan = (t * 1000).to(device=z.device)
        e = self._time_embed(t_wan)  # [B, 6, wan_dim]

        # ---- 3. Grid setup for 3D RoPE ----
        grid_sizes = torch.tensor(
            [[S, self.latent_grid, self.latent_grid]],
            device=z.device, dtype=torch.long
        ).repeat(B, 1)
        seq_lens = torch.full((B,), S * N, device=z.device, dtype=torch.long)

        # ---- 4. Text conditioning ----
        context, context_lens = None, None
        if text_emb is not None:
            if text_emb.shape[-1] == self.wan.text_dim:
                context = self.wan.text_embedding(text_emb.to(x.dtype))
            elif text_emb.shape[-1] == 768:
                context = self.text_proj(text_emb.to(x.dtype))
            else:
                raise ValueError(
                    f"Expected text dim 768 (legacy CLIP) or "
                    f"{self.wan.text_dim} (native UMT5), got {text_emb.shape[-1]}")
            context_lens = torch.full((B,), context.shape[1], device=x.device, dtype=torch.long)
        elif self.num_pseudo_text > 0:
            # Learned null prompt: keep cross-attention RUNNING (as in all of
            # Wan's pretraining) instead of skipping the layer.
            context = self.pseudo_context.to(x.dtype).unsqueeze(0).expand(
                B, -1, -1)
            context_lens = torch.full(
                (B,), self.num_pseudo_text, device=x.device, dtype=torch.long)

        # ---- 5. Wan DiT blocks ----
        if self.wan.freqs.device != x.device:
            self.wan.freqs = self.wan.freqs.to(x.device)

        def _block_fn(x, e, seq_lens, grid_sizes, freqs, context, context_lens, block):
            e_dtype = x.dtype
            e6 = (block.modulation.to(e_dtype) + e.to(e_dtype)).chunk(6, dim=1)
            # Self-attention
            y = block.self_attn(
                block.norm1(x) * (1 + e6[1]) + e6[0],
                seq_lens, grid_sizes, freqs)
            x = x + y * e6[2].to(x.dtype)
            # Cross-attention (text)
            if context is not None:
                x = x + block.cross_attn(block.norm3(x), context, context_lens)
            # FFN
            y = block.ffn(block.norm2(x) * (1 + e6[4].to(x.dtype)) + e6[3].to(x.dtype))
            x = x + y * e6[5].to(x.dtype)
            return x

        for block in self.wan.blocks:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(
                    _block_fn, x, e, seq_lens, grid_sizes, self.wan.freqs,
                    context, context_lens, block, use_reentrant=False)
            else:
                x = _block_fn(x, e, seq_lens, grid_sizes, self.wan.freqs,
                             context, context_lens, block)

        # ---- 6. Output projection ----
        x = self.output_norm(x)
        x = self.output_proj(x)  # [B, S*N, latent_dim]
        x = x.to(dtype=z.dtype)  # back to input dtype (bf16)
        x = x.reshape(B, S, N, self.latent_dim)

        return x
