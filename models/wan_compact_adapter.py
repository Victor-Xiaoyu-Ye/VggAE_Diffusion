"""Wan2.1 1.3B adapter for compact latent z_g.

Key design (from verification experiments + all prior learnings):
  1. DUAL time conditioning: concat at input (V1/V3: works) + adaLN in blocks (Wan native)
  2. Trainable: modulation + time_emb + QKV (~215M params)
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
import sys, os

_wan_root = os.path.join(os.path.dirname(__file__), '..', 'Wan2.1')
if _wan_root not in sys.path:
    sys.path.insert(0, _wan_root)
from wan.modules.model import WanModel, sinusoidal_embedding_1d


class WanCompactAdapter(nn.Module):
    """Wan backbone adapted for compact latent flow matching."""

    def __init__(self, wan_checkpoint_dir, latent_dim=768, latent_grid=18,
                 seq_len=8, wan_dim=1536, freq_dim=256, num_heads=12,
                 i0_condition=False):
        super().__init__()

        # Load pretrained Wan 1.3B
        self.wan = WanModel.from_pretrained(wan_checkpoint_dir)
        self.wan_dim = wan_dim
        self.freq_dim = freq_dim

        self.latent_dim = latent_dim
        self.latent_grid = latent_grid
        self.num_tokens = latent_grid ** 2
        self.seq_len = seq_len
        self.i0_condition = i0_condition

        # ---- Input: latent_dim → wan_dim with concat time injection ----
        self.time_concat_dim = 256
        self.time_concat_mlp = nn.Sequential(
            nn.Linear(freq_dim, self.time_concat_dim * 2),
            nn.SiLU(),
            nn.Linear(self.time_concat_dim * 2, self.time_concat_dim),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + self.time_concat_dim, wan_dim),
            nn.LayerNorm(wan_dim),
            nn.SiLU(),
            nn.Linear(wan_dim, wan_dim),
        )
        if i0_condition:
            self.i0_proj = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, wan_dim),
                nn.SiLU(),
                nn.Linear(wan_dim, wan_dim),
            )

        # ---- Output: wan_dim → latent_dim (zero-init) ----
        self.output_norm = nn.LayerNorm(wan_dim)
        self.output_proj = nn.Linear(wan_dim, latent_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # ---- Text projection: CLIP 768 → Wan 1536 ----
        self.text_proj = nn.Sequential(
            nn.Linear(768, wan_dim),
            nn.GELU(),
            nn.Linear(wan_dim, wan_dim),
        )

        # ---- Trainable parameter setup ----
        self._freeze_all()
        self._unfreeze_trainable()

        self._time_emb_converted = False

    def _freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def _unfreeze_trainable(self):
        """Unfreeze: modulation + time_emb + QKV + adapters (~215M)."""
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
        for p in self.text_proj.parameters():
            p.requires_grad_(True)

        # Wan time pathway
        for p in self.wan.time_embedding.parameters():
            p.requires_grad_(True)
        for p in self.wan.time_projection.parameters():
            p.requires_grad_(True)

        # Wan blocks: modulation + QKV
        for blk in self.wan.blocks:
            blk.modulation.requires_grad_(True)
            for name in ['q', 'k', 'v']:
                attn_module = getattr(blk.self_attn, name)
                for p in attn_module.parameters():
                    p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"WanCompactAdapter: {trainable/1e6:.1f}M trainable / {total/1e9:.2f}B total")

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
        with amp.autocast(device_type='cuda', enabled=False):
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
