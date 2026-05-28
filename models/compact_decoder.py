"""Compact Decoder G: renders compact generative latent to RGB + depth video.

High-capacity version with:
  - Pixel-shuffle upsampling (learned, better than bilinear)
  - 2 ResBlocks per resolution stage
  - Temporal attention at 2 scales (low-res + mid-res)
  - Deeper final refinement
  - Configurable base_dim (384 for high quality, 256 for speed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------------------------

def _gn(ch):
    """Safe GroupNorm: pick group count that divides channels."""
    for g in [32, 16, 8, 4, 1]:
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class ConvBlock(nn.Module):
    """Conv2d → GroupNorm → SiLU."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, bias=False)
        self.norm = _gn(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block: ConvBlock → Conv + residual."""

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


class PixelShuffleUpsample(nn.Module):
    """Upsample 2× via Conv → PixelShuffle (learned, better than bilinear)."""

    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(in_ch, out_ch * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.shuffle(self.conv(x))


class UpsampleStage(nn.Module):
    """Upsample + 2×ResBlock (one full resolution stage)."""

    def __init__(self, in_ch, out_ch, num_resblocks=2, use_pixel_shuffle=True):
        super().__init__()
        if use_pixel_shuffle:
            self.upsample = PixelShuffleUpsample(in_ch, out_ch)
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
                ConvBlock(in_ch, out_ch, kernel=3),
            )
        self.resblocks = nn.Sequential(*[
            ResBlock(out_ch) for _ in range(num_resblocks)
        ])

    def forward(self, x):
        x = self.upsample(x)
        x = self.resblocks(x)
        return x


class TemporalAttnBlock(nn.Module):
    """Temporal self-attention: attend across frames per spatial position."""

    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x, B, S):
        """x: [B*S, C, H, W] → [B*S, C, H, W] with temporal mixing."""
        BS, C, H, W = x.shape
        # [B*S, C, H, W] → [B*H*W, S, C]
        x_t = x.reshape(B, S, C, H * W).permute(0, 3, 1, 2).contiguous().reshape(B * H * W, S, C)
        x_t = self.norm(x_t)
        x_t, _ = self.attn(x_t, x_t, x_t)
        # [B*H*W, S, C] → [B*S, C, H, W]
        x = x_t.reshape(B, H * W, S, C).permute(0, 2, 3, 1).contiguous().reshape(B * S, C, H, W)
        return x


# ---------------------------------------------------------------------------
# High-Capacity Decoder
# ---------------------------------------------------------------------------

class CompactDecoder(nn.Module):
    """Decoder from compact latent z_g to RGB + depth video.

    Args:
        latent_dim: input latent channels (512)
        base_dim: base channel width (384 for high quality, 256 for speed)
        output_dim: RGB channels (3)
        output_depth: whether to produce depth head
        img_size: output resolution (518)
        latent_grid: input spatial grid size (18)
        num_resblocks: ResBlocks per upsample stage (2)
    """

    def __init__(self, latent_dim=512, base_dim=384, output_dim=3,
                 output_depth=True, img_size=518, latent_grid=18,
                 num_resblocks=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_dim = base_dim
        self.output_depth = output_depth
        self.img_size = img_size
        self.latent_grid = latent_grid

        C0 = base_dim * 2   # 768
        C1 = base_dim        # 384
        C2 = base_dim        # 384
        C3 = base_dim // 2   # 192
        C4 = base_dim // 4   # 96

        # ---- Spatial stem: 18×18 ----
        self.stem = nn.Sequential(
            ConvBlock(latent_dim, C0, kernel=3),
            ResBlock(C0),
            ConvBlock(C0, C0, kernel=3),
            ResBlock(C0),
        )

        # ---- Upsampling stages ----
        # Stage 0: 18 → 36
        self.up0 = UpsampleStage(C0, C0, num_resblocks)
        # Stage 1: 36 → 72
        self.up1 = UpsampleStage(C0, C1, num_resblocks)
        # Stage 2: 72 → 148 (close to 144, but 148×2=296, 296×2=592≈518)
        self.up2 = UpsampleStage(C1, C2, num_resblocks)
        # Stage 3: 148 → 296
        self.up3 = UpsampleStage(C2, C3, num_resblocks)
        # Stage 4: 296 → 592
        self.up4 = UpsampleStage(C3, C4, num_resblocks)

        # ---- Temporal attention at 2 scales ----
        self.temporal_low = TemporalAttnBlock(C0, num_heads=4)   # at 36×36
        self.temporal_mid = TemporalAttnBlock(C1, num_heads=4)   # at 72×72

        # ---- Final refine ----
        self.final_refine = nn.Sequential(
            ConvBlock(C4, C4, kernel=3),
            ResBlock(C4),
            ConvBlock(C4, C4, kernel=3),
            ResBlock(C4),
        )

        # ---- Output heads ----
        head_in = C4
        self.rgb_head = nn.Sequential(
            ConvBlock(head_in, 64, kernel=3),
            nn.Conv2d(64, output_dim, 3, padding=1),
            nn.Sigmoid(),
        )

        if output_depth:
            self.depth_head = nn.Sequential(
                ConvBlock(head_in, 64, kernel=3),
                nn.Conv2d(64, 1, 3, padding=1),
                nn.Sigmoid(),
            )

    def forward(self, z_g, patch_start_idx=0, frames_chunk_size=None):
        """Decode compact latent to RGB (+ depth) video frames.

        Args:
            z_g: [B, S, H_g, W_g, C] compact latent
            patch_start_idx: ignored (DPTHead API compat)
            frames_chunk_size: max frames to process at once (None = all)

        Returns:
            preds: [B, S, H, W, 3+1] (BHWC, for DPTHead compat)
        """
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

        # [B, S, H, W, C] → [B*S, C, H, W]
        x = z_g.permute(0, 1, 4, 2, 3).contiguous().reshape(B * S, C, Hg, Wg)

        # Stem: 18×18
        x = self.stem(x)

        # Stage 0: 18→36
        x = self.up0(x)
        x = self.temporal_low(x, B, S)

        # Stage 1: 36→72
        x = self.up1(x)
        x = self.temporal_mid(x, B, S)

        # Stage 2-4: 72→148→296→592
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        # Final refine
        x = self.final_refine(x)

        # RGB head
        rgb = self.rgb_head(x)

        # Resize to img_size (592 → 518)
        if rgb.shape[-1] != self.img_size:
            rgb = F.interpolate(rgb, size=(self.img_size, self.img_size),
                                mode='bilinear', align_corners=False)

        # [B*S, 3, H, W] → [B, S, H, W, 3]
        rgb = rgb.reshape(B, S, 3, self.img_size, self.img_size).permute(0, 1, 3, 4, 2).contiguous()

        # Depth head
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
