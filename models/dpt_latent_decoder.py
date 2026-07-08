"""Compact latent autoencoder with a DPT-style decoder (E4).

Motivation
----------
4DLangRecon reconstructs sharp RGB from frozen StreamVGGT tokens using the
VGGT-native DPTHead (multi-layer projection + RefineNet progressive
upsampling), reaching l1 ~0.03 on a small dataset. Our earlier probes used a
generic conv decoder and a gated-fusion tokenizer and capped at ~20 PSNR.
This module ports the proven DPT decoding machinery into our compact-latent
setting so we can test whether a compact latent (needed for diffusion) plus a
strong DPT-style decoder can reconstruct well.

Two pieces:
- CompactCompressor: frozen VGGT 4-level tokens -> compact latent z
  [B, S, cdim, g, g] (default 256 x 18 x 18). Deliberately simple
  (per-level LayerNorm + linear + concat + conv + pool); it does NOT use the
  gated fusion / temporal mixer that earlier probes showed to be a negative
  optimization.
- DPTLatentDecoder: z [B, S, cdim, g, g] -> RGB [B, S, 3, H, W] using the DPT
  RefineNet FeatureFusionBlock progressive upsampling.

The DPT building blocks are imported from the vendored StreamVGGT head so the
upsampling behaviour matches the proven implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from streamvggt.heads.dpt_head import _make_fusion_block, custom_interpolate


def _gn(ch):
    for g in [32, 16, 8, 4, 1]:
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.norm = _gn(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CompactCompressor(nn.Module):
    """Frozen VGGT multi-level tokens -> compact latent z [B, S, cdim, g, g].

    Args:
        levels: DPT level indices to consume (patch tokens, special tokens
            already stripped upstream).
        token_dim: per-token channel dim (2048).
        cdim: compact latent channel dim (256).
        latent_grid: compact spatial grid (18).
        input_grid: source token grid (37 for 518/14).
    """

    def __init__(self, levels=(4, 11, 17, 23), token_dim=2048, cdim=256,
                 latent_grid=18, input_grid=37):
        super().__init__()
        self.levels = list(levels)
        self.cdim = cdim
        self.latent_grid = latent_grid
        self.input_grid = input_grid
        self.norms = nn.ModuleDict({
            str(lvl): nn.LayerNorm(token_dim) for lvl in self.levels})
        self.projs = nn.ModuleDict({
            str(lvl): nn.Linear(token_dim, cdim) for lvl in self.levels})
        # Fuse concatenated levels -> cdim (concat preserves per-level info
        # better than the mean/gated fusion used before).
        self.fuse = nn.Conv2d(cdim * len(self.levels), cdim, kernel_size=1)
        self.pre_pool = nn.Sequential(
            nn.Conv2d(cdim, cdim, 3, padding=1), _gn(cdim), nn.ReLU(inplace=True))
        self.post = nn.Sequential(
            nn.Conv2d(cdim, cdim, 3, padding=1), _gn(cdim), nn.ReLU(inplace=True),
            nn.Conv2d(cdim, cdim, 3, padding=1))

    def forward(self, tokens_list):
        """tokens_list: list of tensors [B, S, N, token_dim] (special tokens
        already stripped). Returns z [B, S, cdim, g, g]."""
        feats = []
        for lvl in self.levels:
            x = self.norms[str(lvl)](tokens_list[lvl])
            x = self.projs[str(lvl)](x)  # [B, S, N, cdim]
            feats.append(x)
        B, S, N, _ = feats[0].shape
        g = self.input_grid
        x = torch.cat(feats, dim=-1)  # [B, S, N, cdim*L]
        x = x.reshape(B * S, g, g, self.cdim * len(self.levels)).permute(0, 3, 1, 2)
        x = self.fuse(x)                          # [B*S, cdim, g, g]
        x = self.pre_pool(x)
        x = F.adaptive_avg_pool2d(x, (self.latent_grid, self.latent_grid))
        x = self.post(x)
        return x.reshape(B, S, self.cdim, self.latent_grid, self.latent_grid)


class DPTLatentDecoder(nn.Module):
    """Compact latent z [B, S, cdim, g, g] -> RGB [B, S, 3, H, W].

    Builds a 4-scale feature pyramid from z and runs the DPT RefineNet
    FeatureFusionBlock progressive upsampling, then an output conv + sigmoid.

    Args:
        cdim: compact latent channel dim (256).
        latent_grid: compact spatial grid (18).
        features: RefineNet working channels (256, DPT default).
        img_size: output resolution (518).
        use_checkpoint: gradient checkpoint the pyramid + fusion.
    """

    def __init__(self, cdim=256, latent_grid=18, features=256, img_size=518,
                 use_checkpoint=True):
        super().__init__()
        self.latent_grid = latent_grid
        self.features = features
        self.img_size = img_size
        self.use_checkpoint = use_checkpoint

        g = latent_grid
        # Pyramid target sizes (coarse -> fine): g, 2g, 4g, 8g.
        self.pyr_sizes = [g, g * 2, g * 4, g * 8]
        self.proj4 = _ConvBlock(cdim, features)   # coarsest, at g
        self.proj3 = _ConvBlock(cdim, features)   # at 2g
        self.proj2 = _ConvBlock(cdim, features)   # at 4g
        self.proj1 = _ConvBlock(cdim, features)   # finest, at 8g

        self.refinenet4 = _make_fusion_block(features, has_residual=False)
        self.refinenet3 = _make_fusion_block(features)
        self.refinenet2 = _make_fusion_block(features)
        self.refinenet1 = _make_fusion_block(features)

        self.output_conv1 = nn.Conv2d(features, features // 2, 3, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),
        )

    def _build_pyramid(self, z):
        """z [BS, cdim, g, g] -> 4 feature maps at [g, 2g, 4g, 8g]."""
        g = self.latent_grid
        l4 = self.proj4(z)
        l3 = self.proj3(custom_interpolate(z, size=(g * 2, g * 2), mode="bilinear", align_corners=True))
        l2 = self.proj2(custom_interpolate(z, size=(g * 4, g * 4), mode="bilinear", align_corners=True))
        l1 = self.proj1(custom_interpolate(z, size=(g * 8, g * 8), mode="bilinear", align_corners=True))
        return l1, l2, l3, l4

    def _decode_chunk(self, z):
        """z [B, S, cdim, g, g] -> RGB [B, S, 3, H, W]."""
        B, S, C, H, W = z.shape
        x = z.reshape(B * S, C, H, W)
        l1, l2, l3, l4 = self._build_pyramid(x)

        out = self.refinenet4(l4, size=l3.shape[2:])
        out = self.refinenet3(out, l3, size=l2.shape[2:])
        out = self.refinenet2(out, l2, size=l1.shape[2:])
        out = self.refinenet1(out, l1)              # ~2x -> 16g
        out = self.output_conv1(out)
        rgb = self.output_conv2(out)
        if rgb.shape[-1] != self.img_size:
            rgb = F.interpolate(rgb, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)
        rgb = rgb.reshape(B, S, 3, self.img_size, self.img_size)
        return rgb.permute(0, 1, 3, 4, 2).contiguous()  # [B, S, H, W, 3]

    def _decode_chunk_ckpt(self, z):
        if self.use_checkpoint and self.training:
            return torch_checkpoint(self._decode_chunk, z, use_reentrant=False)
        return self._decode_chunk(z)

    def forward(self, z, frames_chunk_size=None):
        """z [B, S, cdim, g, g] -> RGB [B, S, H, W, 3]."""
        B, S = z.shape[0], z.shape[1]
        if frames_chunk_size is None or S <= frames_chunk_size:
            return self._decode_chunk_ckpt(z)
        outs = []
        for start in range(0, S, frames_chunk_size):
            end = min(start + frames_chunk_size, S)
            outs.append(self._decode_chunk_ckpt(z[:, start:end]))
        return torch.cat(outs, dim=1)
