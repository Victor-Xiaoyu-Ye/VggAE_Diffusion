"""Per-frame TextureEncoder: RGB -> compact appearance latent z_tex.

Why this exists
---------------
E1/E4 showed frozen StreamVGGT features plateau at ~20 PSNR on SpatialVID,
with or without a DPT decoder. Geometry features do not carry RGB high-frequency.
Appearance must come from an explicit RGB encoder, not from VGGT shallow levels
and not from a TexturePredictor(z_geo) (that path is information-theoretically
blocked by the same ~20 PSNR ceiling).

Design (reconstruction-first, diffusion-ready)
----------------------------------------------
- Per-frame CNN (no I0 shortcut): every frame has its own z_tex.
- Multi-scale packing into ONE compact grid matching z_geo (default 18x18):
  early stages (high-freq) and late stages (color/structure) are pooled to the
  target grid and fused. This is how high-frequency survives the bottleneck
  without RGB skip connections that would cheat reconstruction / break diffusion.
- Output z_tex [B, S, G, G, tex_dim] is the diffusion target for appearance.
  Decoder must reconstruct from (z_geo, z_tex) alone — no RGB bypass.

Deprecated
----------
TexturePredictor(z_geo -> z_tex) is kept only as a stub that raises, so old
imports fail loudly instead of silently training an impossible mapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.block = nn.Sequential(
            ConvBlock(ch, ch),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            _gn(ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TextureEncoder(nn.Module):
    """RGB -> compact multi-scale-packed appearance latent.

    Args:
        out_dim: z_tex channels (default 256; pair with geo cdim=256).
        out_grid: spatial grid, must match z_geo (default 18).
        base_ch: width of the first stage (default 64).
        img_size: expected input resolution (518); used only for docs/asserts.

    Forward:
        frames [B, S, 3, H, W] -> z_tex [B, S, G, G, out_dim]
    """

    def __init__(
        self,
        out_dim: int = 256,
        out_grid: int = 18,
        base_ch: int = 64,
        img_size: int = 518,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.out_grid = out_grid
        self.base_ch = base_ch
        self.img_size = img_size

        c0, c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4, base_ch * 4

        # 518 -> ~259
        self.stage0 = nn.Sequential(
            ConvBlock(3, c0, kernel=5, stride=2),
            ResBlock(c0),
            ConvBlock(c0, c0),
        )
        # ~259 -> ~130
        self.stage1 = nn.Sequential(
            ConvBlock(c0, c1, stride=2),
            ResBlock(c1),
            ConvBlock(c1, c1),
        )
        # ~130 -> ~65
        self.stage2 = nn.Sequential(
            ConvBlock(c1, c2, stride=2),
            ResBlock(c2),
            ResBlock(c2),
        )
        # ~65 -> ~33
        self.stage3 = nn.Sequential(
            ConvBlock(c2, c3, stride=2),
            ResBlock(c3),
            ResBlock(c3),
        )

        # Per-scale 1x1 before pooling so each scale contributes a clean slice.
        self.to_grid = nn.ModuleList([
            nn.Conv2d(c0, c0, 1, bias=False),
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.Conv2d(c3, c3, 1, bias=False),
        ])
        fused_ch = c0 + c1 + c2 + c3
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_ch, out_dim, 1, bias=False),
            _gn(out_dim),
            nn.SiLU(inplace=True),
            ResBlock(out_dim),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
        )

        # Zero-init last conv so early training is dominated by z_geo path
        # when jointly trained; TextureEncoder still learns via residual.
        nn.init.zeros_(self.fuse[-1].weight)
        if self.fuse[-1].bias is not None:
            nn.init.zeros_(self.fuse[-1].bias)

    def _pool_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.out_grid or x.shape[-1] != self.out_grid:
            x = F.adaptive_avg_pool2d(x, (self.out_grid, self.out_grid))
        return x

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, S, 3, H, W] -> z_tex: [B, S, G, G, out_dim]."""
        if frames.dim() != 5:
            raise ValueError(f"expected [B,S,3,H,W], got {tuple(frames.shape)}")
        B, S, C, H, W = frames.shape
        x = frames.reshape(B * S, C, H, W)

        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

        packed = torch.cat([
            self._pool_to_grid(self.to_grid[0](s0)),
            self._pool_to_grid(self.to_grid[1](s1)),
            self._pool_to_grid(self.to_grid[2](s2)),
            self._pool_to_grid(self.to_grid[3](s3)),
        ], dim=1)
        z = self.fuse(packed)  # [B*S, out_dim, G, G]
        z = z.reshape(B, S, self.out_dim, self.out_grid, self.out_grid)
        return z.permute(0, 1, 3, 4, 2).contiguous()  # [B, S, G, G, out_dim]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TexturePredictor(nn.Module):
    """DEPRECATED — do not use.

    Predicting z_tex from z_geo alone is blocked by the E1/E4 ceiling (~20 PSNR):
    VGGT features do not contain the RGB high-frequency that z_tex must carry.
    Appearance for generation must be produced by the diffusion model (or another
    generative path), not by a deterministic map from geometry.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError(
            "TexturePredictor(z_geo->z_tex) is deprecated and disabled. "
            "E1/E4 showed VGGT features lack RGB high-freq (~20 PSNR ceiling). "
            "Use TextureEncoder(RGB)->z_tex for reconstruction, and generate "
            "z_tex with diffusion at inference time."
        )
