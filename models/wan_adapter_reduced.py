"""Wan2.1 1.3B adapter with dimensionality reduction (channel + spatial).

Reduces VGGT token space from 22M dims → 660K dims (34×):
  2048-dim → Linear → 256-dim  (channel reduction)
  37×37 grid → avg_pool 2× → 18×18  (spatial reduction)

External interface unchanged: input/output are [B, S, 1369, 2048].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp

import sys, os
_wan_root = os.path.join(os.path.dirname(__file__), '..', 'Wan2.1')
if _wan_root not in sys.path:
    sys.path.insert(0, _wan_root)
from wan.modules.model import WanModel, sinusoidal_embedding_1d


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=64, alpha=128):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        in_dim, out_dim = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)
        self.scaling = alpha / rank

    def forward(self, x):
        y = self.linear(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T
        return y + delta * self.scaling


class WanVGGTAdapterReduced(nn.Module):
    """Wan adapter with channel + spatial dimensionality reduction."""

    def __init__(self, wan_checkpoint_dir, vggt_token_dim=2048, reduced_dim=256,
                 seq_len=8, img_size=518, patch_size=14, downsample=2,
                 lora_rank=64, lora_alpha=128):
        super().__init__()
        self.wan = WanModel.from_pretrained(wan_checkpoint_dir)
        wan_dim = self.wan.dim          # 1536
        freq_dim = self.wan.freq_dim    # 256

        self.seq_len = seq_len
        self.downsample = downsample
        self.full_grid = img_size // patch_size  # 37
        self.reduced_grid = self.full_grid // downsample  # 18
        self.full_patches = self.full_grid ** 2  # 1369
        self.reduced_patches = self.reduced_grid ** 2  # 324
        self.token_dim = vggt_token_dim
        self.wan_dim = wan_dim
        self.freq_dim = freq_dim
        self.reduced_dim = reduced_dim
        self.lora_rank = lora_rank

        # ---- Channel input adapter: 2048 → reduced_dim ----
        self.input_proj = nn.Sequential(
            nn.Linear(vggt_token_dim, reduced_dim * 2),
            nn.LayerNorm(reduced_dim * 2),
            nn.SiLU(),
            nn.Linear(reduced_dim * 2, reduced_dim),
        )

        # ---- Wan adapter: reduced_dim → wan_dim ----
        self.wan_input = nn.Sequential(
            nn.Linear(reduced_dim, wan_dim),
            nn.LayerNorm(wan_dim),
            nn.SiLU(),
            nn.Linear(wan_dim, wan_dim),
        )

        # ---- Wan adapter: wan_dim → reduced_dim ----
        self.wan_output_norm = nn.LayerNorm(wan_dim)
        self.wan_output = nn.Linear(wan_dim, reduced_dim)
        nn.init.zeros_(self.wan_output.weight)
        nn.init.zeros_(self.wan_output.bias)

        # ---- Channel output adapter: reduced_dim → 2048 ----
        self.output_proj = nn.Sequential(
            nn.Linear(reduced_dim, reduced_dim * 2),
            nn.LayerNorm(reduced_dim * 2),
            nn.SiLU(),
            nn.Linear(reduced_dim * 2, vggt_token_dim),
        )
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        # ---- Text projection: CLIP 768 → Wan 1536 ----
        self.text_proj = nn.Sequential(
            nn.Linear(768, wan_dim),
            nn.GELU(),
            nn.Linear(wan_dim, wan_dim),
        )

        # ---- Mode: LoRA vs full fine-tune ----
        if lora_rank > 0:
            self._freeze_wan()
            self._apply_lora(lora_rank, lora_alpha)
        # lora_rank == 0: full fine-tune (don't freeze)
        # lora_rank == -1: head-only (freeze wan, no lora)

        self._time_emb_converted = False

    def _freeze_wan(self):
        for p in self.wan.parameters():
            p.requires_grad_(False)

    def _apply_lora(self, rank, alpha):
        for blk in self.wan.blocks:
            attn = blk.self_attn
            attn.q = LoRALinear(attn.q, rank=rank, alpha=alpha)
            attn.k = LoRALinear(attn.k, rank=rank, alpha=alpha)
            attn.v = LoRALinear(attn.v, rank=rank, alpha=alpha)

    def _ensure_time_emb_float32(self):
        if self._time_emb_converted:
            return
        for m in [self.wan.time_embedding, self.wan.time_projection]:
            for p in m.parameters():
                p.data = p.data.float()
        self._time_emb_converted = True

    def _time_embed(self, t):
        self._ensure_time_emb_float32()
        with amp.autocast(device_type=z.device.type, enabled=False):
            e = sinusoidal_embedding_1d(self.freq_dim, t).float().to(device=t.device)
            e = self.wan.time_embedding(e.float())
            e = self.wan.time_projection(e.float())
        return e.unflatten(1, (6, self.wan_dim)).float()

    def set_lora_trainable(self):
        if self.lora_rank <= 0:
            return
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.input_proj.parameters():
            p.requires_grad_(True)
        for p in self.wan_input.parameters():
            p.requires_grad_(True)
        for p in self.wan_output_norm.parameters():
            p.requires_grad_(True)
        for p in self.wan_output.parameters():
            p.requires_grad_(True)
        for p in self.output_proj.parameters():
            p.requires_grad_(True)
        for p in self.text_proj.parameters():
            p.requires_grad_(True)
        for n, p in self.named_parameters():
            if 'lora_A' in n or 'lora_B' in n:
                p.requires_grad_(True)
        for p in self.wan.time_embedding.parameters():
            p.requires_grad_(True)
        for p in self.wan.time_projection.parameters():
            p.requires_grad_(True)

    def _spatial_pool(self, x):
        """Average pool: [B, S, full_G², D] → [B, S, reduced_G², D]."""
        B, S, N, D = x.shape
        full_G = self.full_grid
        reduced_G = self.reduced_grid
        x = x.reshape(B, S, full_G, full_G, D).permute(0, 1, 4, 2, 3).contiguous()
        x = F.adaptive_avg_pool2d(x.reshape(B * S, D, full_G, full_G), (reduced_G, reduced_G))
        x = x.reshape(B, S, D, reduced_G, reduced_G).permute(0, 1, 3, 4, 2).contiguous()
        return x.reshape(B, S, reduced_G * reduced_G, D)

    def _spatial_unpool(self, x):
        """Nearest unpool: [B, S, (G/s)², D] → [B, S, full_G², D]."""
        B, S, N, D = x.shape
        G = int(N ** 0.5)
        full_G = self.full_grid  # 37 (may not be an exact multiple of downsample)
        x = x.reshape(B, S, G, G, D).permute(0, 1, 4, 2, 3).contiguous()
        x = F.interpolate(x.reshape(B * S, D, G, G),
                          size=(full_G, full_G), mode='nearest')
        x = x.reshape(B, S, D, full_G, full_G).permute(0, 1, 3, 4, 2).contiguous()
        return x.reshape(B, S, full_G * full_G, D)

    def forward(self, z, t, cond=None, text_emb=None):
        B, S, N_full, D_full = z.shape

        # ---- Input: channel project (2048→256) ----
        x = self.input_proj(z)  # [B, S, 1369, 256]

        # ---- Spatial pool (1369→324) ----
        x = self._spatial_pool(x)  # [B, S, 324, 256]

        # ---- Wan input adapter (256→1536) ----
        x = self.wan_input(x)  # [B, S, 324, 1536]
        B2, S2, N2, D2 = x.shape
        x = x.reshape(B2, S2 * N2, D2)  # [B, S*N, wan_dim]

        # ---- Time embedding ----
        t_wan = (t * 1000).to(device=z.device)
        e = self._time_embed(t_wan)

        # ---- Grid setup ----
        grid_sizes = torch.tensor(
            [[S2, self.reduced_grid, self.reduced_grid]],
            device=z.device, dtype=torch.long
        ).repeat(B2, 1)
        seq_lens = torch.full((B2,), S2 * N2, device=z.device, dtype=torch.long)

        # ---- Text conditioning ----
        context, context_lens = None, None
        if text_emb is not None:
            context = self.text_proj(text_emb.to(x.dtype))
            context_lens = torch.full((B2,), context.shape[1], device=x.device, dtype=torch.long)

        # ---- Wan DiT blocks ----
        if self.wan.freqs.device != x.device:
            self.wan.freqs = self.wan.freqs.to(x.device)

        def _block_fn(x, e, seq_lens, grid_sizes, freqs, context, context_lens, block):
            e_dtype = x.dtype
            e6 = (block.modulation.to(e_dtype) + e.to(e_dtype)).chunk(6, dim=1)
            y = block.self_attn(block.norm1(x) * (1 + e6[1]) + e6[0], seq_lens, grid_sizes, freqs)
            x = x + y * e6[2].to(x.dtype)
            if context is not None:
                x = x + block.cross_attn(block.norm3(x), context, context_lens)
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

        # ---- Wan output adapter (1536→256) ----
        x = self.wan_output_norm(x)
        x = self.wan_output(x)  # [B, S*N, 256]
        x = x.reshape(B2, S2, N2, self.reduced_dim)  # [B, S, 324, 256]

        # ---- Spatial unpool (324→1369) ----
        x = self._spatial_unpool(x)  # [B, S, 1369, 256]

        # ---- Output: channel unproject (256→2048) ----
        x = self.output_proj(x)  # [B, S, 1369, 2048]

        return x
