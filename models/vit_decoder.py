"""ViT-based RGB decoder (GLD channel-concat style).

Design (matching GLD's approach):
  1. Channel concat: 4 levels × 2048 → [N, 8192] per patch
  2. Linear(8192, decoder_dim) — fuses levels in one projection
  3. + spatial position embedding → [N, decoder_dim] per patch
  4. Transformer blocks (cross-patch attention, 1369 tokens/frame)
  5. LayerNorm → Linear(decoder_dim, P*P*3) → RGB patches
  6. Optional depth head: Linear(decoder_dim, P*P) → depth patches

Key difference from previous token-concat design:
  - 1369 tokens/frame instead of 5476 (4× fewer)
  - Cross-level fusion via Linear projection, not attention
  - No level embeddings, no mask tokens, no learned fusion MLP
"""

import torch
import torch.nn as nn

from streamvggt.layers.block import Block


def _patches_to_image(patches, B, S, N, patch_size, grid, channels):
    """Reshape [B*S, N, P*P*C] → [B, S, C, H, W]."""
    x = patches.reshape(B, S, N, patch_size, patch_size, channels)
    x = x.reshape(B, S, grid, grid, patch_size, patch_size, channels)
    x = x.permute(0, 1, 6, 2, 3, 4, 5).contiguous()
    return x.reshape(B, S, channels, grid * patch_size, grid * patch_size)


class ViTDecoder(nn.Module):
    """GLD-style ViT decoder: channel-concat multi-level features, transformer, RGB+depth."""

    def __init__(
        self,
        dim=2048,
        decoder_dim=512,
        num_levels=4,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        patch_size=14,
        img_size=518,
        output_dim=3,
        output_depth=False,
    ):
        super().__init__()
        self.dim = dim
        self.decoder_dim = decoder_dim
        self.num_levels = num_levels
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.output_dim = output_dim
        self.output_depth = output_depth

        # Input projection: concat levels in channel dim → decoder_dim
        # GLD: Linear(8192, 1152) for VGGT with 2048-dim features
        self.input_proj = nn.Linear(num_levels * dim, decoder_dim)

        # Spatial position embeddings (sin-cos style, learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, decoder_dim) * 0.02)

        # Transformer blocks — cross-patch attention
        self.blocks = nn.ModuleList([
            Block(
                dim=decoder_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_norm=False,
                rope=None,
            )
            for _ in range(depth)
        ])

        self.output_norm = nn.LayerNorm(decoder_dim)

        # RGB head: decoder_dim → P*P*3
        patch_pixels = patch_size * patch_size * output_dim
        self.output_proj = nn.Linear(decoder_dim, patch_pixels)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Depth head: decoder_dim → P*P*1
        if output_depth:
            self.depth_output_proj = nn.Linear(decoder_dim, patch_size * patch_size)
            nn.init.zeros_(self.depth_output_proj.weight)
            nn.init.zeros_(self.depth_output_proj.bias)

    def _build_input(self, tokens_list, B, S, N):
        """Channel-concatenate 4 DPT levels → project → [B, S, N, decoder_dim]."""
        DPT_LEVELS = [4, 11, 17, 23]

        level_feats = []
        for lvl in DPT_LEVELS:
            t = tokens_list[lvl]  # [B, S, N, 2048]
            # Zero levels remain zero — Linear naturally ignores zero channels
            level_feats.append(t)

        # Channel concat: [B, S, N, L*2048]
        x = torch.cat(level_feats, dim=-1)

        # Linear projection fuses levels
        x = self.input_proj(x)  # [B, S, N, decoder_dim]

        # Add spatial position embedding
        x = x + self.pos_embed[:, :N, :]

        return x

    def forward(self, tokens_list, images, patch_start_idx=0, frames_chunk_size=8):
        """Forward pass. Matches DPTHead signature for drop-in use.

        Returns:
            rgb:        [B, S, H, W, 3]
            conf:       [B, S, H, W]
            (depth):    [B, S, H, W]     if output_depth
            (depth_conf): [B, S, H, W]   if output_depth
        """
        B, S, _, H, W = images.shape
        N = self.num_patches

        # Build input: channel-concat + Linear projection
        x = self._build_input(tokens_list, B, S, N)  # [B, S, N, decoder_dim]

        # Process frames in chunks for memory
        all_rgb = []
        all_depth = [] if self.output_depth else None

        for start in range(0, S, frames_chunk_size):
            end = min(start + frames_chunk_size, S)
            chunk = x[:, start:end]                           # [B, cS, N, D]
            cS = chunk.shape[1]
            chunk = chunk.reshape(B * cS, N, self.decoder_dim)

            # Transformer blocks
            for blk in self.blocks:
                chunk = blk(chunk)
            chunk = self.output_norm(chunk)

            # RGB output
            rgb_patches = self.output_proj(chunk)             # [B*cS, N, P*P*3]
            pp = self.patch_size
            grid = int(N ** 0.5)
            rgb = _patches_to_image(rgb_patches, B, cS, N, pp, grid, self.output_dim)
            all_rgb.append(rgb)

            if self.output_depth:
                depth_patches = self.depth_output_proj(chunk) # [B*cS, N, P*P]
                depth = _patches_to_image(depth_patches, B, cS, N, pp, grid, 1)
                all_depth.append(depth)

        rgb = torch.cat(all_rgb, dim=1)                       # [B, S, 3, H, W]
        rgb = rgb.permute(0, 1, 3, 4, 2).contiguous()        # [B, S, H, W, 3]
        conf = torch.ones(B, S, H, W, device=rgb.device, dtype=rgb.dtype)

        if not self.output_depth:
            return rgb, conf

        depth = torch.cat(all_depth, dim=1)                   # [B, S, 1, H, W]
        depth = depth.squeeze(2)                              # [B, S, H, W]
        depth_conf = torch.ones(B, S, H, W, device=rgb.device, dtype=rgb.dtype)
        return rgb, conf, depth, depth_conf
