"""Compact latent DiT: lightweight diffusion backbone for z_g space.

Design principles (informed by verification experiments):
  - Concat time conditioning (V3: adaLN kills time sensitivity)
  - Factorized spatial-temporal attention (V4, V7: minimal gains from interaction)
  - Wide DDT head (RAE principle: high-dim latent needs wide input/output projection)
  - Lightweight overall (V1: token space is learnable without heavy backbone)

Architecture:
  z_g [B,S,N,D] where N=latent_grid², D=latent_dim
  → Wide input head: D → 4*D → model_dim
  → Spatial blocks (within-frame self-attention)
  → Temporal blocks (cross-frame at each position)
  → Wide output head: model_dim → 4*D → D (zero-init)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WideHead(nn.Module):
    """Wide DDT-style input/output projection (from RAE paper).

    High-dimensional latent channels need wide projection to avoid
    information bottleneck at the very first/last layer.
    """

    def __init__(self, in_dim, model_dim, expansion=4):
        super().__init__()
        hidden = in_dim * expansion
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, model_dim),
        )

    def forward(self, x):
        return self.net(x)


class DiTBlock(nn.Module):
    """Transformer block with pre-norm and residual connections."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        # Self-attention
        y = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + y
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class CompactLatentDiT(nn.Module):
    """Lightweight DiT for flow matching in compact latent space z_g.

    Args:
        latent_dim: token dimension in z_g space (512)
        num_tokens: number of tokens per frame (324 = 18×18)
        model_dim: internal transformer dimension
        spatial_depth: number of spatial attention blocks
        temporal_depth: number of temporal attention blocks
        num_heads: attention heads
        seq_len: number of frames
        time_emb_dim: time embedding dimension
        text_cond: if True, include text cross-attention
    """

    def __init__(self, latent_dim=512, num_tokens=324, model_dim=768,
                 spatial_depth=8, temporal_depth=4, num_heads=12,
                 seq_len=8, time_emb_dim=256, text_cond=True,
                 text_dim=768):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens
        self.model_dim = model_dim
        self.seq_len = seq_len
        self.time_emb_dim = time_emb_dim
        self.text_cond = text_cond

        # ---- Time embedding (sinusoidal + MLP) ----
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # ---- Wide input head ----
        self.input_head = WideHead(latent_dim + model_dim, model_dim, expansion=4)

        # ---- Positional embeddings ----
        self.spatial_pos = nn.Parameter(
            torch.randn(1, 1, num_tokens, model_dim) * 0.02)
        self.temporal_pos = nn.Parameter(
            torch.randn(1, seq_len, 1, model_dim) * 0.02)

        # ---- Transformer blocks ----
        self.spatial_blocks = nn.ModuleList([
            DiTBlock(model_dim, num_heads) for _ in range(spatial_depth)
        ])
        self.temporal_blocks = nn.ModuleList([
            DiTBlock(model_dim, num_heads) for _ in range(temporal_depth)
        ])

        # ---- Text cross-attention (optional) ----
        if text_cond:
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
            )
            self.cross_attn_blocks = nn.ModuleList([
                nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
                for _ in range(min(4, spatial_depth))
            ])

        # ---- Wide output head (zero-init) ----
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_head = WideHead(model_dim, latent_dim, expansion=4)
        # Zero-init the final linear layer
        nn.init.zeros_(self.output_head.net[-1].weight)
        nn.init.zeros_(self.output_head.net[-1].bias)

    def _time_embed(self, t):
        """t: [B] in [0, 1], returns [B, model_dim]."""
        half = self.time_emb_dim // 2
        emb = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) *
            (-math.log(10000) / (half - 1))
        )
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.time_mlp(emb.to(t.dtype))

    def forward(self, z, t, cond=None, text_emb=None):
        """Flow matching forward.

        Args:
            z: [B, S, N, D] noisy latent tokens
            t: [B] flow time in [0, 1]
            cond: unused (API compat)
            text_emb: [B, L, text_dim] optional text conditioning

        Returns:
            v: [B, S, N, D] predicted velocity
        """
        B, S, N, D = z.shape

        # Time embedding
        t_emb = self._time_embed(t)  # [B, model_dim]
        t_emb = t_emb.unsqueeze(1).unsqueeze(1).expand(B, S, N, -1)  # [B, S, N, model_dim]

        # Concat token + time → wide input projection
        x = torch.cat([z, t_emb], dim=-1)  # [B, S, N, D + model_dim]
        x = self.input_head(x)  # [B, S, N, model_dim]

        # Add positional embeddings
        x = x + self.spatial_pos[:, :, :N, :] + self.temporal_pos[:, :S, :, :]

        # Spatial blocks (within-frame attention)
        for i, block in enumerate(self.spatial_blocks):
            x_flat = x.reshape(B * S, N, self.model_dim)
            x_flat = block(x_flat)
            x = x_flat.reshape(B, S, N, self.model_dim)

            # Text cross-attention every few spatial blocks
            if self.text_cond and text_emb is not None and i % 2 == 0:
                cross_idx = i // 2
                if cross_idx < len(self.cross_attn_blocks):
                    # Project text and apply cross-attention per frame
                    context = self.text_proj(text_emb.to(x.dtype))  # [B, L, model_dim]
                    x_flat = x.reshape(B * S, N, self.model_dim)
                    context_expanded = context.repeat_interleave(S, dim=0)  # [B*S, L, model_dim]
                    ca_out, _ = self.cross_attn_blocks[cross_idx](
                        x_flat, context_expanded, context_expanded)
                    x = (x_flat + ca_out).reshape(B, S, N, self.model_dim)

        # Temporal blocks (cross-frame attention at each spatial position)
        for block in self.temporal_blocks:
            x_flat = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, S, self.model_dim)
            x_flat = block(x_flat)
            x = x_flat.reshape(B, N, S, self.model_dim).permute(0, 2, 1, 3).contiguous()

        # Wide output head (zero-init)
        x = self.output_norm(x)
        v = self.output_head(x)  # [B, S, N, D]

        return v
