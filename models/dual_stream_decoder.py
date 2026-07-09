"""Dual-stream decoder: reconstruct RGB from (z_geo, z_tex) only.

Contract
--------
- Inputs are compact latents only. No RGB / I0 skip into the decoder.
  That keeps reconstruction honest and makes both streams diffusion targets.
- z_geo carries layout / depth / camera (from frozen StreamVGGT + compressor).
- z_tex carries appearance packed by TextureEncoder (multi-scale -> grid).
- Fusion is early concat at the stem (same spatial grid), then a CompactDecoder-
  style progressive upsample. Early concat is preferred over late SPADE/cross-
  attn here because both streams already live on the same GxG grid; late
  injection would reintroduce I0-style shortcuts without adding capacity.

Shapes
------
  z_geo: [B, S, G, G, geo_dim]
  z_tex: [B, S, G, G, tex_dim]
  out:   [B, S, H, W, 3]  (BHWC, sigmoid RGB)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def _gn(ch: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.norm = _gn(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block1 = ConvBlock(ch, ch)
        self.block2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            _gn(ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block2(self.block1(x)))


class BilinearUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        return self.conv(x)


class UpsampleStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_resblocks: int = 2):
        super().__init__()
        # resize-conv (not PixelShuffle) — E3 showed PixelShuffle adds grid texture
        self.upsample = BilinearUpsample(in_ch, out_ch)
        self.resblocks = nn.Sequential(*[ResBlock(out_ch) for _ in range(num_resblocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(self.upsample(x))


class TemporalAttnBlock(nn.Module):
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, B: int, S: int) -> torch.Tensor:
        BS, C, H, W = x.shape
        x_t = x.reshape(B, S, C, H * W).permute(0, 3, 1, 2).contiguous()
        x_t = x_t.reshape(B * H * W, S, C)
        x_t = self.norm(x_t)
        x_t, _ = self.attn(x_t, x_t, x_t)
        return x_t.reshape(B, H * W, S, C).permute(0, 2, 3, 1).contiguous().reshape(
            B * S, C, H, W)


class DualStreamDecoder(nn.Module):
    """Decode (z_geo, z_tex) -> RGB without RGB skip connections."""

    def __init__(
        self,
        geo_dim: int = 256,
        tex_dim: int = 256,
        base_dim: int = 384,
        img_size: int = 518,
        latent_grid: int = 18,
        num_resblocks: int = 2,
        num_temporal_blocks: int = 2,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.geo_dim = geo_dim
        self.tex_dim = tex_dim
        self.base_dim = base_dim
        self.img_size = img_size
        self.latent_grid = latent_grid
        self.use_checkpoint = use_checkpoint
        self.num_temporal_blocks = num_temporal_blocks

        C0 = base_dim * 2
        C1 = base_dim
        C2 = base_dim
        C3 = base_dim // 2
        C4 = base_dim // 4
        self.C0, self.C1 = C0, C1

        in_ch = geo_dim + tex_dim
        self.stem = nn.Sequential(
            ConvBlock(in_ch, C0),
            ResBlock(C0),
            ConvBlock(C0, C0),
            ResBlock(C0),
        )
        self.up0 = UpsampleStage(C0, C0, num_resblocks)
        self.up1 = UpsampleStage(C0, C1, num_resblocks)
        self.up2 = UpsampleStage(C1, C2, num_resblocks)
        self.up3 = UpsampleStage(C2, C3, num_resblocks)
        self.up4 = UpsampleStage(C3, C4, num_resblocks)

        self.temporal_low = TemporalAttnBlock(C0, num_heads=4)
        if num_temporal_blocks >= 2:
            self.temporal_mid = TemporalAttnBlock(C1, num_heads=4)

        self.final_refine = nn.Sequential(
            ConvBlock(C4, C4),
            ResBlock(C4),
            ConvBlock(C4, C4),
            ResBlock(C4),
        )
        self.rgb_head = nn.Sequential(
            ConvBlock(C4, 64),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def _stage(self, stage: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch_checkpoint(stage, x, use_reentrant=False)
        return stage(x)

    def forward(
        self,
        z_geo: torch.Tensor,
        z_tex: torch.Tensor,
        frames_chunk_size=None,
    ) -> torch.Tensor:
        """Returns RGB [B, S, H, W, 3]."""
        if z_geo.shape[:4] != z_tex.shape[:4]:
            raise ValueError(
                f"z_geo/z_tex spatial mismatch: {tuple(z_geo.shape)} vs "
                f"{tuple(z_tex.shape)}")
        B, S = z_geo.shape[:2]
        if frames_chunk_size is not None and frames_chunk_size > 0 and S > frames_chunk_size:
            chunks = []
            for start in range(0, S, frames_chunk_size):
                end = min(start + frames_chunk_size, S)
                chunks.append(self._forward_impl(
                    z_geo[:, start:end], z_tex[:, start:end]))
            return torch.cat(chunks, dim=1)
        return self._forward_impl(z_geo, z_tex)

    def _forward_impl(self, z_geo: torch.Tensor, z_tex: torch.Tensor) -> torch.Tensor:
        B, S, Hg, Wg, _ = z_geo.shape
        geo = z_geo.permute(0, 1, 4, 2, 3).contiguous().reshape(B * S, self.geo_dim, Hg, Wg)
        tex = z_tex.permute(0, 1, 4, 2, 3).contiguous().reshape(B * S, self.tex_dim, Hg, Wg)
        x = torch.cat([geo, tex], dim=1)

        x = self.stem(x)
        x = self._stage(self.up0, x)
        x = self.temporal_low(x, B, S)
        x = self._stage(self.up1, x)
        if hasattr(self, 'temporal_mid'):
            x = self.temporal_mid(x, B, S)
        x = self._stage(self.up2, x)
        x = self._stage(self.up3, x)
        x = self._stage(self.up4, x)
        x = self.final_refine(x)
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(
                x, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False)
        rgb = self.rgb_head(x)  # [B*S, 3, H, W]
        return rgb.reshape(B, S, 3, self.img_size, self.img_size).permute(
            0, 1, 3, 4, 2).contiguous()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
