"""Lightweight texture encoder: captures color + high-freq details from RGB.

Works alongside frozen StreamVGGT (geometry) to provide appearance information
that pure geometric features lack. Designed to be small and shallow:
  - 3 conv stages: 518→144→72→18
  - ~3M parameters total
  - Output: z_tex [B, S, 18, 18, 128]

Also includes TexturePredictor: small conv that maps z_geo → z_tex,
used during inference when real RGB is unavailable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch):
    for g in [16, 8, 4, 1]:
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


class TextureEncoder(nn.Module):
    """Lightweight encoder: RGB → texture latent z_tex.

    Processes each frame independently. 3 stages of stride-2 convs
    to match the compact latent grid (18×18).

    Args:
        img_size: input resolution (518)
        out_dim: z_tex channels (128)
        out_grid: target grid size (18)
        base_ch: base channel count (32)
    """

    def __init__(self, img_size=518, out_dim=128, out_grid=18, base_ch=32):
        super().__init__()
        self.out_dim = out_dim
        self.out_grid = out_grid
        self.base_ch = base_ch

        # 518 → 144  (stride=4 via 2x stride-2)
        self.stage0 = nn.Sequential(
            ConvBlock(3, base_ch, kernel=5, stride=2),     # 518→259
            ConvBlock(base_ch, base_ch * 2, kernel=3, stride=2),  # 259→129
        )
        # 129 → 72
        self.stage1 = nn.Sequential(
            ConvBlock(base_ch * 2, base_ch * 4, kernel=3, stride=2),  # 129→64
            ConvBlock(base_ch * 4, base_ch * 4, kernel=3),
            ConvBlock(base_ch * 4, base_ch * 4, kernel=3, stride=2),  # 64→32
        )
        # 72 → 36 → 18
        self.stage2 = nn.Sequential(
            ConvBlock(base_ch * 4, base_ch * 8, kernel=3, stride=2),  # 32→16
            ConvBlock(base_ch * 8, base_ch * 8, kernel=3, stride=2),  # 16→18 → pad/slice
        )
        # Output projection to out_dim
        self.output = nn.Sequential(
            ConvBlock(base_ch * 8, out_dim, kernel=3),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
        )

    def forward(self, frames):
        """frames: [B, S, 3, H, W] → z_tex: [B, S, Hg, Wg, out_dim]"""
        B, S, C, H, W = frames.shape
        x = frames.reshape(B * S, C, H, W)

        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)

        # Adjust to target grid via adaptive pool or interpolate
        if x.shape[-1] != self.out_grid:
            x = F.adaptive_avg_pool2d(x, (self.out_grid, self.out_grid))

        x = self.output(x)  # [B*S, out_dim, Hg, Wg]
        x = x.reshape(B, S, self.out_dim, self.out_grid, self.out_grid)
        x = x.permute(0, 1, 3, 4, 2)  # [B, S, Hg, Wg, out_dim]
        return x


class TexturePredictor(nn.Module):
    """Predicts z_tex from z_geo during inference (no RGB available).

    Lightweight conv net operating on the compact latent grid.
    z_geo [B,S,18,18,512] → z_tex [B,S,18,18,128]
    """

    def __init__(self, geo_dim=512, tex_dim=128, grid=18, base_ch=128):
        super().__init__()
        self.grid = grid
        self.tex_dim = tex_dim

        self.net = nn.Sequential(
            nn.Conv2d(geo_dim, base_ch * 2, 3, padding=1),
            _gn(base_ch * 2),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            _gn(base_ch * 2),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            _gn(base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, tex_dim, 3, padding=1),
        )

    def forward(self, z_geo):
        """z_geo: [B, S, Hg, Wg, geo_dim] → z_tex: [B, S, Hg, Wg, tex_dim]"""
        B, S, Hg, Wg, D = z_geo.shape
        x = z_geo.permute(0, 1, 4, 2, 3).reshape(B * S, D, Hg, Wg)
        x = self.net(x)
        x = x.reshape(B, S, self.tex_dim, Hg, Wg).permute(0, 1, 3, 4, 2)
        return x
