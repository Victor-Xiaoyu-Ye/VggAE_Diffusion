"""I_0-conditioned decoder for appearance-guided video reconstruction.

Architecture:
  z_geo [B, S, 18, 18, 512] + I_0 features {f36, f72, f144}
  → stem + upsample stages
  → up0/up1: Cross-Attention (Q=z_geo, K,V=I_0_feat) — learned warping
  → up2: SPADE (z_geo modulates I_0 feat) — local texture mapping
  → up3/up4: No I_0 conditioning — disocclusion inpainting
  → RGB head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def load_i0_decoder_state_dict(decoder, state_dict):
    """Load current or pre-f144 decoder weights without hiding real mismatches."""
    load_info = decoder.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = ("spade2.", "app144_proj.")
    invalid_missing = [
        key for key in load_info.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if invalid_missing or load_info.unexpected_keys:
        raise RuntimeError(
            "I0 decoder checkpoint mismatch: "
            f"missing={invalid_missing}, "
            f"unexpected={load_info.unexpected_keys}")
    return load_info


def _gn(ch):
    for g in [32, 16, 8, 4, 1]:
        if ch % g == 0: return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, bias=False)
        self.norm = _gn(out_ch)
        self.act = nn.SiLU()
    def forward(self, x): return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch, kernel=3):
        super().__init__()
        self.block1 = ConvBlock(ch, ch, kernel)
        self.block2 = nn.Sequential(nn.Conv2d(ch, ch, kernel, padding=kernel // 2, bias=False), _gn(ch))
        self.act = nn.SiLU()
    def forward(self, x):
        r = x; x = self.block1(x); x = self.block2(x); return self.act(x + r)


class CrossAttnBlock(nn.Module):
    """Cross-attention: z_geo tokens query I_0 appearance features.

    z_geo: [B*S, H_g*W_g, C_geo] → Q
    I_0_feat: [1, H_f*W_f, C_app] → K, V   (1=batch, shared across S frames)
    """

    def __init__(self, geo_ch, app_ch, num_heads=8):
        super().__init__()
        if geo_ch % num_heads != 0:
            raise ValueError(f"geo_ch={geo_ch} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        head_dim = geo_ch // num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(geo_ch, geo_ch)
        self.k_proj = nn.Linear(app_ch, geo_ch)
        self.v_proj = nn.Linear(app_ch, geo_ch)
        self.out_proj = nn.Linear(geo_ch, geo_ch)

    def forward(self, z_geo_flat, I_0_feat):
        """z_geo_flat: [B*S, N_g, C_g], I_0_feat: [B*S, N_f, C_a]"""
        Q = self.q_proj(z_geo_flat)
        K = self.k_proj(I_0_feat)
        V = self.v_proj(I_0_feat)

        BS, Ng, Cg = Q.shape; _, Nf, _ = K.shape
        Q = Q.view(BS, Ng, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(BS, Nf, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(BS, Nf, self.num_heads, self.head_dim).transpose(1, 2)

        # Fused SDPA avoids materializing the very large 72x72 attention matrix.
        x = F.scaled_dot_product_attention(Q, K, V)
        x = x.transpose(1, 2).reshape(BS, Ng, Cg)
        return z_geo_flat + self.out_proj(x)


class SPADEBlock(nn.Module):
    """Spatially-Adaptive Denormalization: z_geo modulates I_0 features.

    γ, β = Conv(z_geo); output = γ * GroupNorm(I_0_feat) + β
    """

    def __init__(self, geo_ch, app_ch):
        super().__init__()
        self.norm = _gn(app_ch)
        self.gamma = nn.Conv2d(geo_ch, app_ch, 3, padding=1)
        self.beta = nn.Conv2d(geo_ch, app_ch, 3, padding=1)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

    def forward(self, z_geo_grid, I_0_feat):
        """z_geo_grid: [B*S, C_g, H, W], I_0_feat: [B*S, C_a, H, W]"""
        g = self.gamma(z_geo_grid)      # [BS, C_a, H, W]
        b = self.beta(z_geo_grid)        # [BS, C_a, H, W]
        return self.norm(I_0_feat) * (1 + g) + b


class TemporalAttnBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x, B, S):
        BS, C, H, W = x.shape
        residual = x
        x_t = x.reshape(B, S, C, H * W).permute(0, 3, 1, 2).contiguous().reshape(B * H * W, S, C)
        x_t = self.norm(x_t); x_t, _ = self.attn(x_t, x_t, x_t)
        x_t = x_t.reshape(B, H * W, S, C).permute(0, 2, 3, 1).contiguous().reshape(B * S, C, H, W)
        return residual + x_t


class I0ConditionalDecoder(nn.Module):
    """Decoder conditioned on I_0 appearance features.

    Takes z_geo [B, S, Hg, Wg, C] + I_0 features {f36, f72, f144}
    Produces RGB [B, S, 518, 518, 3].
    """

    def __init__(self, latent_dim=512, base_dim=384, img_size=518, latent_grid=18,
                 num_resblocks=2, use_pixel_shuffle=True, num_temporal_blocks=2,
                 use_checkpoint=True):
        super().__init__()
        self.base_dim = base_dim
        self.img_size = img_size
        self.output_depth = False
        self.use_checkpoint = use_checkpoint

        self.C0 = base_dim * 2; self.C1 = base_dim; self.C2 = base_dim
        self.C3 = base_dim // 2; self.C4 = base_dim // 4

        def _upsample(in_ch, out_ch):
            if use_pixel_shuffle:
                return nn.Sequential(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1), nn.PixelShuffle(2))
            return nn.Sequential(nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
                                ConvBlock(in_ch, out_ch, 3))

        def _stage(in_ch, out_ch):
            layers = [_upsample(in_ch, out_ch)]
            for _ in range(num_resblocks): layers.append(ResBlock(out_ch))
            return nn.Sequential(*layers)

        # ---- Stem ----
        self.stem = nn.Sequential(ConvBlock(latent_dim, self.C0, 3), ResBlock(self.C0),
                                  ConvBlock(self.C0, self.C0, 3), ResBlock(self.C0))

        # ---- Upsampling with I_0 conditioning ----
        self.up0 = _stage(self.C0, self.C0)
        self.cross_attn0 = CrossAttnBlock(self.C0, 128, num_heads=8)

        self.up1 = _stage(self.C0, self.C1)
        self.cross_attn1 = CrossAttnBlock(self.C1, 64, num_heads=8)

        self.up2 = _stage(self.C1, self.C2)
        self.spade2 = SPADEBlock(self.C2, 32)
        self.app144_proj = nn.Conv2d(32, self.C2, 1)
        # Old checkpoints start with identical behavior and can be fine-tuned.
        nn.init.zeros_(self.app144_proj.weight)
        nn.init.zeros_(self.app144_proj.bias)

        self.up3 = _stage(self.C2, self.C3)
        self.up4 = _stage(self.C3, self.C4)

        # ---- Temporal ----
        self.temporal_low = TemporalAttnBlock(self.C0, num_heads=4)
        if num_temporal_blocks >= 2: self.temporal_mid = TemporalAttnBlock(self.C1, num_heads=4)

        # ---- Final refine ----
        self.final_refine = nn.Sequential(ConvBlock(self.C4, self.C4, 3), ResBlock(self.C4),
                                          ConvBlock(self.C4, self.C4, 3), ResBlock(self.C4))

        # ---- RGB head ----
        self.rgb_head = nn.Sequential(ConvBlock(self.C4, 64, 3), nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid())

    def _stage_forward(self, stage, x):
        if self.use_checkpoint and self.training: return torch_checkpoint(stage, x, use_reentrant=False)
        return stage(x)

    def forward(self, z_geo, I_0_feats, patch_start_idx=0, frames_chunk_size=None):
        """z_geo: [B, S, Hg, Wg, C], I_0_feats: dict from AppearanceCNN {'f36','f72','f144'}"""
        # Convert I_0 features to match z_geo dtype (AppCNN is bf16, decoder is fp32)
        I_0_feats = {k: v.to(dtype=z_geo.dtype) for k, v in I_0_feats.items()}
        B, S, Hg, Wg, C = z_geo.shape
        x = z_geo.permute(0, 1, 4, 2, 3).contiguous().reshape(B * S, C, Hg, Wg)
        x = self.stem(x)

        # Helper: expand I_0 features from [B, C, H, W] to match z_geo [B*S, C, H, W]
        def _expand_I0(feat):
            return feat.unsqueeze(1).expand(B, S, *feat.shape[1:]).reshape(B * S, *feat.shape[1:])

        # up0 + cross-attn with f36
        x = self._stage_forward(self.up0, x)  # [B*S, C0, 36, 36]
        x = self.temporal_low(x, B, S)
        N_g = 36 * 36
        x_flat = x.reshape(B * S, self.C0, N_g).transpose(1, 2)  # [B*S, 1296, C0]
        f36 = _expand_I0(I_0_feats['f36'])  # [B*S, 128, 36, 36]
        f36_flat = f36.reshape(B * S, 128, N_g).transpose(1, 2)  # [B*S, 1296, 128]
        x_flat = self.cross_attn0(x_flat, f36_flat)
        x = x_flat.transpose(1, 2).reshape(B * S, self.C0, 36, 36)

        # up1 + cross-attn with f72
        x = self._stage_forward(self.up1, x)  # [B*S, C1, 72, 72]
        if hasattr(self, 'temporal_mid'): x = self.temporal_mid(x, B, S)
        N_g1 = 72 * 72
        x_flat = x.reshape(B * S, self.C1, N_g1).transpose(1, 2)  # [B*S, 5184, C1]
        f72 = _expand_I0(I_0_feats['f72'])
        f72_flat = f72.reshape(B * S, 64, N_g1).transpose(1, 2)  # [B*S, 5184, 64]
        x_flat = self.cross_attn1(x_flat, f72_flat)
        x = x_flat.transpose(1, 2).reshape(B * S, self.C1, 72, 72)

        # up2 + high-frequency appearance modulation at 144x144
        x = self._stage_forward(self.up2, x)  # [B*S, C2, 144, 144]
        f144 = _expand_I0(I_0_feats['f144'])
        x = x + self.app144_proj(self.spade2(x, f144))

        # up3 + up4: no I_0 conditioning (disocclusion inpainting)
        x = self._stage_forward(self.up3, x)
        x = self._stage_forward(self.up4, x)

        x = self._stage_forward(self.final_refine, x)

        rgb = self.rgb_head(x)
        if rgb.shape[-1] != self.img_size:
            rgb = F.interpolate(rgb, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        rgb = rgb.reshape(B, S, 3, self.img_size, self.img_size).permute(0, 1, 3, 4, 2).contiguous()
        return (rgb, None)
