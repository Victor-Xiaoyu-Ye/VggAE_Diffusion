"""Lightweight Appearance CNN: extracts multi-scale color/texture features from I_0.

~1.5M parameters. Outputs 3 scales for cross-attention / SPADE injection into decoder.
"""
import torch
import torch.nn as nn


def _gn(ch):
    for g in [16, 8, 4, 1]:
        if ch % g == 0: return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2, bias=False)
        self.norm = _gn(out_ch)
        self.act = nn.SiLU()
    def forward(self, x): return self.act(self.norm(self.conv(x)))


class AppearanceCNN(nn.Module):
    """Extract multi-scale appearance features from I_0.

    I_0 [B, 3, 518, 518] → {f36, f72, f144}
      f36: [B, 128, 36, 36] — global structure (for cross-attn warp)
      f72: [B, 64, 72, 72]   — mid-level detail (for cross-attn refine)
      f144: [B, 32, 144, 144] — local texture (for SPADE modulation)
    """

    def __init__(self, base_ch=16):
        super().__init__()
        # 518 → 259 (stride 2)
        self.stage0 = nn.Sequential(ConvBlock(3, base_ch, stride=1), ConvBlock(base_ch, base_ch * 2, stride=2))
        # 259 → 129 (stride 2)
        self.stage1 = nn.Sequential(ConvBlock(base_ch * 2, base_ch * 3, stride=2), ConvBlock(base_ch * 3, base_ch * 4))
        # 129 → 64 (stride 2)
        self.stage2 = nn.Sequential(ConvBlock(base_ch * 4, base_ch * 5, stride=2), ConvBlock(base_ch * 5, base_ch * 6))
        # 64 → 32 (stride 2)
        self.stage3 = ConvBlock(base_ch * 6, base_ch * 8, stride=2)

        # Output projections
        self.to_f144 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            _gn(base_ch * 4), nn.SiLU(),
            nn.Conv2d(base_ch * 4, 32, 1),
        )
        self.to_f72 = nn.Sequential(
            nn.Conv2d(base_ch * 6, base_ch * 6, 3, padding=1),
            _gn(base_ch * 6), nn.SiLU(),
            nn.Conv2d(base_ch * 6, 64, 1),
        )
        self.to_f36 = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 3, padding=1),
            _gn(base_ch * 8), nn.SiLU(),
            nn.Conv2d(base_ch * 8, 128, 1),
        )

    def forward(self, I_0):
        """I_0: [B, 3, 518, 518] → dict of multi-scale features."""
        s0 = self.stage0(I_0)      # [B, 32, 259, 259]
        s1 = self.stage1(s0)       # [B, 64, 129, 129]
        s2 = self.stage2(s1)       # [B, 96, 64, 64]
        s3 = self.stage3(s2)       # [B, 128, 32, 32]

        def _resize(feat, target):
            if feat.shape[-1] != target:
                feat = nn.functional.interpolate(feat, size=(target, target), mode='bilinear', align_corners=False)
            return feat

        return {
            'f144': _resize(self.to_f144(s1), 144),   # [B, 32, 144, 144]
            'f72':  _resize(self.to_f72(s2), 72),      # [B, 64, 72, 72]
            'f36':  _resize(self.to_f36(s3), 36),      # [B, 128, 36, 36]
        }
