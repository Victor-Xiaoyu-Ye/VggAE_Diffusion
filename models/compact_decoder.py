"""Compact Decoder G: renders compact generative latent to RGB + depth video.

Two architectures, auto-detected from checkpoint or selected via config:
  - 'v1': bilinear upsample, 1 ResBlock/stage, 1 temporal block (exp-1 baseline)
  - 'v2': pixel-shuffle upsample, 2 ResBlocks/stage, 2 temporal blocks, deeper refine (big)

Usage:
  decoder = CompactDecoder(base_dim=384, version='v2')  # big
  decoder = CompactDecoder(base_dim=256, version='v1')  # exp-1 compatible
  decoder = CompactDecoder(base_dim=256)                 # auto (v1 for base_dim<=256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


# ---------------------------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------------------------

def _gn(ch):
    for g in [32, 16, 8, 4, 1]:
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, bias=False)
        self.norm = _gn(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch, kernel=3):
        super().__init__()
        self.block1 = ConvBlock(ch, ch, kernel)
        self.block2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel, padding=kernel // 2, bias=False),
            _gn(ch),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.block1(x)
        x = self.block2(x)
        return self.act(x + residual)


class BilinearUpsample(nn.Module):
    """Bilinear 2× upsample + conv (v1 style)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, kernel=3)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        return self.conv(x)


class PixelShuffleUpsample(nn.Module):
    """Pixel-shuffle 2× upsample (v2 style, learned)."""
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.shuffle(self.conv(x))


class UpsampleStage(nn.Module):
    """Upsample + N ResBlocks (v2 structure)."""
    def __init__(self, in_ch, out_ch, num_resblocks, use_pixel_shuffle=True):
        super().__init__()
        if use_pixel_shuffle:
            self.upsample = PixelShuffleUpsample(in_ch, out_ch)
        else:
            self.upsample = BilinearUpsample(in_ch, out_ch)
        self.resblocks = nn.Sequential(*[
            ResBlock(out_ch) for _ in range(num_resblocks)
        ])

    def forward(self, x):
        x = self.upsample(x)
        x = self.resblocks(x)
        return x


class TemporalAttnBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x, B, S):
        BS, C, H, W = x.shape
        x_t = x.reshape(B, S, C, H * W).permute(0, 3, 1, 2).contiguous().reshape(B * H * W, S, C)
        x_t = self.norm(x_t)
        x_t, _ = self.attn(x_t, x_t, x_t)
        x = x_t.reshape(B, H * W, S, C).permute(0, 2, 3, 1).contiguous().reshape(B * S, C, H, W)
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class CompactDecoder(nn.Module):
    """Decoder from compact latent z_g to RGB + depth video.

    Two versions:
      v1: bilinear upsample, 1 ResBlock/stage, 1 temporal block, simple refine
      v2: pixel-shuffle upsample, 2 ResBlocks/stage, 2 temporal blocks, deeper refine

    Auto-detection: set version='auto' to infer from base_dim (<=256 → v1, >=384 → v2).
    """

    def __init__(self, latent_dim=512, base_dim=256, output_dim=3,
                 output_depth=False, img_size=518, latent_grid=18,
                 num_resblocks=None, use_pixel_shuffle=None,
                 num_temporal_blocks=None, use_checkpoint=False,
                 version='auto'):
        super().__init__()

        # Auto-detect version
        if version == 'auto':
            version = 'v1' if base_dim <= 256 else 'v2'
        self.version = version

        # Version-specific defaults
        if version == 'v1':
            num_resblocks = num_resblocks if num_resblocks is not None else 1
            use_pixel_shuffle = use_pixel_shuffle if use_pixel_shuffle is not None else False
            num_temporal_blocks = num_temporal_blocks if num_temporal_blocks is not None else 1
        else:  # v2
            num_resblocks = num_resblocks if num_resblocks is not None else 2
            use_pixel_shuffle = use_pixel_shuffle if use_pixel_shuffle is not None else True
            num_temporal_blocks = num_temporal_blocks if num_temporal_blocks is not None else 2

        self.latent_dim = latent_dim
        self.base_dim = base_dim
        self.output_depth = output_depth
        self.img_size = img_size
        self.latent_grid = latent_grid
        self.num_temporal_blocks = num_temporal_blocks
        self.use_checkpoint = use_checkpoint

        C0 = base_dim * 2
        C1 = base_dim
        C2 = base_dim
        C3 = base_dim // 2
        C4 = base_dim // 4

        # ---- Spatial stem ----
        self.stem = nn.Sequential(
            ConvBlock(latent_dim, C0, kernel=3),
            ResBlock(C0),
            ConvBlock(C0, C0, kernel=3),
            ResBlock(C0),
        )

        # ---- Upsampling stages ----
        if version == 'v1':
            # exp-1: flat Sequential(UpsampleBlock, ResBlock) per stage
            self.up0 = nn.Sequential(BilinearUpsample(C0, C0), ResBlock(C0))
            self.up1 = nn.Sequential(BilinearUpsample(C0, C1), ResBlock(C1))
            self.up2 = nn.Sequential(BilinearUpsample(C1, C2), ResBlock(C2))
            self.up3 = nn.Sequential(BilinearUpsample(C2, C3), ResBlock(C3))
            self.up4 = nn.Sequential(BilinearUpsample(C3, C4), ResBlock(C4))
        else:
            # v2: UpsampleStage (PixelShuffleUpsample + N ResBlocks)
            self.up0 = UpsampleStage(C0, C0, num_resblocks, use_pixel_shuffle)
            self.up1 = UpsampleStage(C0, C1, num_resblocks, use_pixel_shuffle)
            self.up2 = UpsampleStage(C1, C2, num_resblocks, use_pixel_shuffle)
            self.up3 = UpsampleStage(C2, C3, num_resblocks, use_pixel_shuffle)
            self.up4 = UpsampleStage(C3, C4, num_resblocks, use_pixel_shuffle)

        # ---- Temporal attention ----
        if version == 'v1':
            self.temporal_attn = TemporalAttnBlock(C0, num_heads=4)
        else:
            self.temporal_low = TemporalAttnBlock(C0, num_heads=4)
            if num_temporal_blocks >= 2:
                self.temporal_mid = TemporalAttnBlock(C1, num_heads=4)

        # ---- Final refine ----
        if version == 'v1':
            self.final_refine = nn.Sequential(
                ConvBlock(C4, C4, kernel=3),
                ResBlock(C4),
            )
        else:
            self.final_refine = nn.Sequential(
                ConvBlock(C4, C4, kernel=3),
                ResBlock(C4),
                ConvBlock(C4, C4, kernel=3),
                ResBlock(C4),
            )

        # ---- Output heads ----
        if version == 'v1':
            # exp-1 style: Conv2d → SiLU → Conv2d → Sigmoid
            self.rgb_head = nn.Sequential(
                nn.Conv2d(C4, 64, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, output_dim, 3, padding=1),
                nn.Sigmoid(),
            )
        else:
            # v2 style: ConvBlock → Conv2d → Sigmoid
            self.rgb_head = nn.Sequential(
                ConvBlock(C4, 64, kernel=3),
                nn.Conv2d(64, output_dim, 3, padding=1),
                nn.Sigmoid(),
            )

        if output_depth:
            self.depth_head = nn.Sequential(
                nn.Conv2d(C4, 64, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 1, 3, padding=1),
                nn.Sigmoid(),
            )

    def _stage_forward(self, stage, x):
        if self.use_checkpoint and self.training:
            return torch_checkpoint(stage, x, use_reentrant=False)
        return stage(x)

    def forward(self, z_g, patch_start_idx=0, frames_chunk_size=None):
        B, S, Hg, Wg, C = z_g.shape
        if frames_chunk_size is not None and S > frames_chunk_size:
            all_chunks = []
            for start in range(0, S, frames_chunk_size):
                end = min(start + frames_chunk_size, S)
                chunk = z_g[:, start:end]
                chunk_preds = self._forward_impl(chunk)[0]
                all_chunks.append(chunk_preds)
            preds = torch.cat(all_chunks, dim=1)
            if hasattr(self, 'depth_head'):
                return preds, None, None, None
            return preds, None
        return self._forward_impl(z_g)

    def _forward_impl(self, z_g):
        B, S, Hg, Wg, C = z_g.shape
        x = z_g.permute(0, 1, 4, 2, 3).contiguous().reshape(B * S, C, Hg, Wg)

        x = self.stem(x)

        x = self._stage_forward(self.up0, x)
        if hasattr(self, 'temporal_low'):
            x = self.temporal_low(x, B, S)
        else:
            x = self.temporal_attn(x, B, S)

        x = self._stage_forward(self.up1, x)
        if hasattr(self, 'temporal_mid'):
            x = self.temporal_mid(x, B, S)

        x = self._stage_forward(self.up2, x)
        x = self._stage_forward(self.up3, x)
        x = self._stage_forward(self.up4, x)

        x = self._stage_forward(self.final_refine, x)

        rgb = self.rgb_head(x)
        if rgb.shape[-1] != self.img_size:
            rgb = F.interpolate(rgb, size=(self.img_size, self.img_size),
                                mode='bilinear', align_corners=False)
        rgb = rgb.reshape(B, S, 3 if rgb.shape[1] == 3 else rgb.shape[1],
                         self.img_size, self.img_size)
        if rgb.shape[2] != 3:
            rgb = rgb[:, :, :3]
        rgb = rgb.permute(0, 1, 3, 4, 2).contiguous()

        depth = None
        if hasattr(self, 'depth_head'):
            d = self.depth_head(x)
            if d.shape[-1] != self.img_size:
                d = F.interpolate(d, size=(self.img_size, self.img_size),
                                  mode='bilinear', align_corners=False)
            depth = d.reshape(B, S, 1, self.img_size, self.img_size).permute(0, 1, 3, 4, 2).contiguous()
            conf = torch.ones_like(depth)
            return (torch.cat([rgb, depth], dim=-1), depth, conf, None)

        return (rgb, None)
