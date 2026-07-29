"""Causal local spatiotemporal compression for dual StreamVGGT latents."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def _gn(channels):
    for groups in (32, 16, 8, 4, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class CausalConv3d(nn.Conv3d):
    """Conv3d with left-only temporal padding and symmetric spatial padding."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, bias=True):
        kernel = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
        stride = (stride,) * 3 if isinstance(stride, int) else stride
        dilation = (dilation,) * 3 if isinstance(dilation, int) else dilation
        super().__init__(in_channels, out_channels, kernel, stride=stride,
                         dilation=dilation, padding=0, bias=bias)
        self.temporal_pad = dilation[0] * (kernel[0] - 1)
        self.height_pad = dilation[1] * (kernel[1] - 1) // 2
        self.width_pad = dilation[2] * (kernel[2] - 1) // 2

    def forward(self, x):
        x = F.pad(x, (
            self.width_pad, self.width_pad,
            self.height_pad, self.height_pad,
            self.temporal_pad, 0,
        ))
        return super().forward(x)


class CausalResBlock3d(nn.Module):
    def __init__(self, channels, dilation=1, zero_init=False,
                 use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = _gn(channels)
        self.conv1 = CausalConv3d(
            channels, channels, 3, dilation=(dilation, 1, 1), bias=False)
        self.norm2 = _gn(channels)
        self.conv2 = CausalConv3d(channels, channels, 3, bias=False)
        if zero_init:
            nn.init.zeros_(self.conv2.weight)

    def _forward(self, x):
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return residual + x

    def forward(self, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return torch_checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class CausalTemporalDownsample(nn.Module):
    """Map 1+r*k frames to 1+k causal latent frames."""

    def __init__(self, channels=512, factor=2, depth=3):
        super().__init__()
        if factor not in (2, 4):
            raise ValueError(f"factor must be 2 or 4, got {factor}")
        self.channels = channels
        self.factor = factor
        self.anchor = nn.Conv2d(channels, channels, 1)
        self.blocks = nn.Sequential(*[
            CausalResBlock3d(channels, dilation=2 ** min(i, 2), zero_init=True)
            for i in range(depth)
        ])
        self.fold = nn.Conv3d(
            channels, channels, kernel_size=(factor, 1, 1),
            stride=(factor, 1, 1), bias=True)
        self._init_mean_fold()

    def _init_mean_fold(self):
        nn.init.dirac_(self.anchor.weight)
        nn.init.zeros_(self.anchor.bias)
        nn.init.zeros_(self.fold.weight)
        nn.init.zeros_(self.fold.bias)
        with torch.no_grad():
            for channel in range(self.channels):
                self.fold.weight[channel, channel, :, 0, 0] = \
                    1.0 / self.factor

    def encoded_length(self, frames):
        if (frames - 1) % self.factor:
            raise ValueError(
                f"frames={frames} must satisfy 1 + {self.factor}*k")
        return 1 + (frames - 1) // self.factor

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"expected [B,C,T,H,W], got {tuple(x.shape)}")
        self.encoded_length(x.shape[2])
        anchor = self.anchor(x[:, :, 0]).unsqueeze(2)
        tail = self.blocks(x[:, :, 1:])
        tail = self.fold(tail)
        return torch.cat([anchor, tail], dim=2)


class CausalTemporalUpsample(nn.Module):
    """Expand 1+k latent frames to 1+r*k ordered frame features."""

    def __init__(self, channels=512, factor=2, depth=3):
        super().__init__()
        if factor not in (2, 4):
            raise ValueError(f"factor must be 2 or 4, got {factor}")
        self.channels = channels
        self.factor = factor
        self.anchor = nn.Conv2d(channels, channels, 1)
        self.expand = nn.Conv3d(channels, channels * factor, 1)
        self.blocks = nn.Sequential(*[
            CausalResBlock3d(channels, dilation=2 ** min(i, 2), zero_init=True)
            for i in range(depth)
        ])
        self._init_replicate()

    def _init_replicate(self):
        nn.init.dirac_(self.anchor.weight)
        nn.init.zeros_(self.anchor.bias)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)
        with torch.no_grad():
            for channel in range(self.channels):
                for subframe in range(self.factor):
                    output = channel * self.factor + subframe
                    self.expand.weight[output, channel, 0, 0, 0] = 1.0

    def decoded_length(self, latent_frames):
        return 1 + self.factor * (latent_frames - 1)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"expected [B,C,T,H,W], got {tuple(x.shape)}")
        anchor = self.anchor(x[:, :, 0]).unsqueeze(2)
        tail = self.expand(x[:, :, 1:])
        B, CR, K, H, W = tail.shape
        tail = tail.reshape(
            B, self.channels, self.factor, K, H, W)
        tail = tail.permute(0, 1, 3, 2, 4, 5).reshape(
            B, self.channels, K * self.factor, H, W)
        tail = self.blocks(tail)
        return torch.cat([anchor, tail], dim=2)


class CausalSpatiotemporalCodec(nn.Module):
    def __init__(self, channels=512, factor=2, depth=3):
        super().__init__()
        self.factor = factor
        self.encoder = CausalTemporalDownsample(channels, factor, depth)
        self.decoder = CausalTemporalUpsample(channels, factor, depth)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
