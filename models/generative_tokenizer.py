"""Generative Tokenizer A: transforms StreamVGGT features into compact generative latent.

Implements the RAE-style reshaping of pretrained feature space:
  StreamVGGT levels [4,11,17,23] × [B,S,1369,2048]
  → per-level LayerNorm
  → 1x1 projection 2048→latent_dim
  → gated multi-layer fusion
  → spatial downsample 37→latent_grid
  → temporal mixer (lightweight)
  → z_g: [B, S, latent_grid, latent_grid, latent_dim]

Design rationale (informed by verification experiments):
  - Concat time conditioning (V3: adaLN kills time sensitivity)
  - Gated fusion instead of simple mean (V5: mean loses information)
  - Spatial downsample (V4: spatial interaction adds little, compress aggressively)
  - Light temporal mixing (V7: temporal attention adds 1.2%, keep minimal)
  - Wide projections (RAE principle: high-dim latent needs wide heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerLevelNorm(nn.Module):
    """Per-level LayerNorm + learnable affine (whitening adaptation)."""

    def __init__(self, dim=2048, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        # Small learnable correction on top of normalization
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.norm(x) * self.scale + self.bias


class GatedLevelFusion(nn.Module):
    """Gated fusion of multiple encoder levels (RAEv2-style).

    Instead of simple mean, learns per-level gates conditioned on the
    concatenated features. This preserves level-specific information
    that simple averaging would destroy.
    """

    def __init__(self, num_levels=4, dim=512):
        super().__init__()
        self.num_levels = num_levels
        # Gate network: concat all levels → softmax weights per level
        self.gate_net = nn.Sequential(
            nn.Linear(dim * num_levels, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, num_levels),
        )

    def forward(self, level_features):
        """level_features: list of [B, S, N, dim] tensors."""
        stacked = torch.stack(level_features, dim=0)  # [L, B, S, N, dim]
        concat = torch.cat(level_features, dim=-1)  # [B, S, N, dim*L]
        gates = self.gate_net(concat)  # [B, S, N, L]
        gates = F.softmax(gates, dim=-1)  # softmax over levels
        # gates: [B, S, N, L] → [L, B, S, N, 1] for broadcasting with stacked
        gates = gates.permute(3, 0, 1, 2).unsqueeze(-1)  # [L, B, S, N, 1]
        fused = (stacked * gates).sum(dim=0)  # [B, S, N, dim]
        return fused


class SpatialCompressor(nn.Module):
    """Spatial downsampling with learnable pre-pool conv.

    37×37 → 18×18 via stride-2 conv (learned) then bilinear adjustment.
    """

    def __init__(self, dim=512, target_grid=18, input_grid=37):
        super().__init__()
        self.target_grid = target_grid
        self.input_grid = input_grid
        # Pre-pool feature refinement
        self.pre_pool = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
        )

    def forward(self, x):
        """x: [B*S, dim, H, W] → [B*S, dim, target, target]."""
        x = self.pre_pool(x)
        x = F.adaptive_avg_pool2d(x, (self.target_grid, self.target_grid))
        return x


class TemporalMixer(nn.Module):
    """Lightweight temporal smoothing via depthwise conv1d + residual.

    Kept intentionally minimal based on V7 finding (temporal attention adds 1.2%).
    """

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: [B*N, S, dim] → [B*N, S, dim]."""
        residual = x
        x = self.norm(x)
        # Conv1d along temporal dim: [B*N, dim, S]
        x_t = x.permute(0, 2, 1)
        x_t = self.conv(x_t)
        x_t = x_t.permute(0, 2, 1)
        # Gated residual
        gate = self.gate(residual)
        return residual + gate * x_t


class GenerativeTokenizer(nn.Module):
    """RAE-style tokenizer: StreamVGGT features → compact generative latent.

    Args:
        token_dim: StreamVGGT token dimension (2048)
        latent_dim: compact latent dimension (512)
        latent_grid: spatial resolution of latent (18)
        levels: which DPT levels to use
        seq_len: number of frames
        input_grid: original token grid size (37)
    """

    def __init__(self, token_dim=2048, latent_dim=512, latent_grid=18,
                 levels=(4, 11, 17, 23), seq_len=8, input_grid=37):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_grid = latent_grid
        self.levels = levels
        self.num_levels = len(levels)
        self.seq_len = seq_len
        self.input_grid = input_grid
        self.num_tokens = input_grid ** 2  # 1369
        self.num_latent_tokens = latent_grid ** 2  # 324

        # Per-level normalization + projection
        self.level_norms = nn.ModuleDict({
            str(lvl): PerLevelNorm(token_dim) for lvl in levels
        })
        self.level_projs = nn.ModuleDict({
            str(lvl): nn.Linear(token_dim, latent_dim) for lvl in levels
        })

        # Gated fusion
        self.fusion = GatedLevelFusion(len(levels), latent_dim)

        # Spatial compression (applied per-frame)
        self.spatial_compressor = SpatialCompressor(latent_dim, latent_grid, input_grid)

        # Temporal mixing (applied per-position)
        self.temporal_mixer = TemporalMixer(latent_dim)

        # Output refinement (small conv after all processing)
        self.output_refine = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
        )

    def forward(self, tokens_list):
        """Convert StreamVGGT token list to compact generative latent.

        Args:
            tokens_list: list of 24 tensors [B, S, N, 2048] (raw, unnormalized)

        Returns:
            z_g: [B, S, latent_grid, latent_grid, latent_dim]
            z_g_flat: [B, S, num_latent_tokens, latent_dim] (for diffusion)
        """
        B, S, N = tokens_list[self.levels[0]].shape[:3]

        # 1. Per-level norm + project
        projected = []
        for lvl in self.levels:
            x = tokens_list[lvl]  # [B, S, N, 2048]
            x = self.level_norms[str(lvl)](x)
            x = self.level_projs[str(lvl)](x)  # [B, S, N, latent_dim]
            projected.append(x)

        # 2. Gated fusion
        z = self.fusion(projected)  # [B, S, N, latent_dim]

        # 3. Reshape to spatial grid
        G = self.input_grid  # 37
        z = z.reshape(B, S, G, G, self.latent_dim)  # [B, S, 37, 37, latent_dim]

        # 4. Spatial compression (per-frame)
        z_flat = z.reshape(B * S, G, G, self.latent_dim).contiguous().permute(0, 3, 1, 2)
        z_flat = self.spatial_compressor(z_flat)  # [B*S, latent_dim, 18, 18]
        z = z_flat.reshape(B, S, self.latent_dim, self.latent_grid, self.latent_grid)
        z = z.permute(0, 1, 3, 4, 2).contiguous()  # [B, S, 18, 18, latent_dim]

        # 5. Temporal mixing (per-position)
        z_reshaped = z.reshape(B, S, self.num_latent_tokens, self.latent_dim)
        z_temporal = z_reshaped.permute(0, 2, 1, 3).contiguous().reshape(B * self.num_latent_tokens, S, self.latent_dim)
        z_temporal = self.temporal_mixer(z_temporal)
        z_temporal = z_temporal.reshape(B, self.num_latent_tokens, S, self.latent_dim).permute(0, 2, 1, 3).contiguous()
        z = z_temporal.reshape(B, S, self.latent_grid, self.latent_grid, self.latent_dim)

        # 6. Output refinement
        z_out = z.reshape(B * S, self.latent_grid, self.latent_grid, self.latent_dim).permute(0, 3, 1, 2).contiguous()
        z_out = self.output_refine(z_out)
        z_out = z_out.permute(0, 2, 3, 1).contiguous().reshape(B, S, self.latent_grid, self.latent_grid, self.latent_dim)

        # Flat version for diffusion
        z_flat_out = z_out.reshape(B, S, self.num_latent_tokens, self.latent_dim)

        return z_out, z_flat_out
