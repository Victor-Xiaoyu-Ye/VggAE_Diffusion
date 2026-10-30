"""Matched feature-only video codec for DINO/StreamVGGT comparisons.

This is a new controlled codec, not a reproduction of V-RAE or a reuse of an
accepted Scene/R7 checkpoint. Both arms have one learned latent; there is no
separate RGB texture encoder or RGB skip. The encoder is framewise after the
frozen feature extraction; the RGB decoder uses bidirectional time attention
inside a fixed window. Never use decoded prefix pixels as future-free inputs.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.dual_stream_decoder import (ConvBlock, ResBlock, UpsampleStage,
                                       TemporalAttnBlock)


def _validate_frames(frames):
    if frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError('Expected RGB frames [B,T,3,H,W]')


def _patch_tensor(value):
    if isinstance(value, dict):
        value = value['x_norm_patchtokens']
    if not torch.is_tensor(value) or value.ndim != 3:
        raise ValueError('DINO patch embedding must produce [B*T,N,D] tokens')
    return value


@torch.no_grad()
def extract_matched_features(backbone, frames, layer=-1):
    """Extract both arms from the *same* frozen StreamVGGT image forward.

    Returns dino [B,T,N,D] and stream [B,T,N,2D]. The latter concatenates
    frame/global tokens at ``layer`` and excludes all camera/register tokens.
    The hook is scoped to this call, also on exceptions. Callers select AMP;
    outputs retain backbone dtype. No checkpoint download happens here.
    """
    _validate_frames(frames)
    backbone.eval().requires_grad_(False)
    captured = []

    def capture(_module, _args, output):
        captured.append(_patch_tensor(output).detach())

    hook = backbone.aggregator.patch_embed.register_forward_hook(capture)
    try:
        values, start = backbone(frames)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise ValueError('Expected exactly one patch embedding call per window')
    b, t = frames.shape[:2]
    patch = captured[0]
    if patch.shape[0] != b*t:
        raise ValueError('DINO output batch differs from the input window')
    dino = patch.reshape(b, t, patch.shape[1], patch.shape[2])
    stream = values[layer][:, :, int(start):].detach()
    if (stream.ndim != 4 or stream.shape[:3] != dino.shape[:3]
            or stream.shape[-1] != 2*dino.shape[-1]):
        raise ValueError('Stream frame/global concatenation does not match DINO patches')
    return {'dino': dino, 'stream': stream}


class FrozenMatchedFeatures(nn.Module):
    """Frozen, offline feature view over an already-loaded StreamVGGT model."""

    def __init__(self, backbone, source='dino', layer=-1):
        super().__init__()
        if source not in ('dino', 'stream'):
            raise ValueError('source must be dino or stream')
        self.backbone = backbone.eval().requires_grad_(False)
        self.source, self.layer = source, int(layer)
        self.train(False)

    def train(self, mode=True):
        # Parent .train() must not enable dropout/stochastic depth in targets.
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, frames):
        _validate_frames(frames)
        if self.source == 'stream':
            values, start = self.backbone(frames)
            return values[self.layer][:, :, int(start):].detach()
        agg = self.backbone.aggregator
        b, t, c, h, w = frames.shape
        normalized = ((frames - agg._resnet_mean.to(frames.device))
                      / agg._resnet_std.to(frames.device))
        patch = _patch_tensor(agg.patch_embed(normalized.reshape(b*t, c, h, w)))
        return patch.reshape(b, t, patch.shape[1], patch.shape[2]).detach()


class _VideoDecoder(nn.Module):
    """Same convolutional/time-attention decoder for both frozen encoders."""

    def __init__(self, channels, grid, size, width, temporal_blocks, resblocks,
                 use_checkpoint):
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError('decoder_width must be a positive multiple of 8')
        if temporal_blocks not in (0, 1, 2):
            raise ValueError('decoder_temporal_blocks must be 0, 1 or 2')
        if resblocks < 0:
            raise ValueError('decoder_resblocks must be nonnegative')
        self.grid, self.size = grid, size
        self.use_checkpoint = bool(use_checkpoint)
        self.num_temporal_blocks = temporal_blocks
        # At 252px, avoid allocating full-resolution 576px intermediate maps.
        stages = max(1, math.ceil(math.log2(max(1., size/grid))))
        if temporal_blocks > stages:
            raise ValueError('Temporal blocks exceed available upsampling stages')
        self.stem = nn.Sequential(ConvBlock(channels, width*2), ResBlock(width*2))
        widths = [width*2] + [max(width//4, width//(2**(i-1)))
                             for i in range(1, stages)]
        previous = width*2
        self.stages = nn.ModuleList()
        self.temporal = nn.ModuleDict()
        for i, current in enumerate(widths):
            self.stages.append(UpsampleStage(previous, current, resblocks))
            if i < temporal_blocks:
                self.temporal[str(i)] = TemporalAttnBlock(current, num_heads=4)
            previous = current
        self.rgb = nn.Sequential(ResBlock(previous), nn.Conv2d(previous, 3, 3, padding=1),
                                 nn.Sigmoid())

    def _run(self, module, x):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, z):
        b, t, _, c = z.shape
        x = z.reshape(b*t, self.grid, self.grid, c).permute(0, 3, 1, 2)
        x = self._run(self.stem, x)
        for i, stage in enumerate(self.stages):
            x = self._run(stage, x)
            if str(i) in self.temporal:
                module = self.temporal[str(i)]
                if self.use_checkpoint and self.training and torch.is_grad_enabled():
                    x = checkpoint(module, x, b, t, use_reentrant=False)
                else:
                    x = module(x, b, t)
        x = self._run(self.rgb, x)
        if x.shape[-2:] != (self.size, self.size):
            x = F.interpolate(x, (self.size, self.size), mode='bilinear', align_corners=False)
        return x.reshape(b, t, 3, self.size, self.size).permute(0, 1, 3, 4, 2)


class MatchedVideoRAE(nn.Module):
    """One framewise bottleneck plus a matched fixed-window video decoder.

    Input channels differ between DINO (1024) and Stream (2048), so the encoder
    projection and feature readout parameter counts differ and must be reported.
    Decoder construction precedes those projections: with the same RNG seed its
    initial values are exactly paired across arms.
    """
    def __init__(self, input_dim, channels=256, grid=18, size=252,
                 decoder_width=192, decoder_temporal_blocks=2,
                 decoder_resblocks=2, use_checkpoint=True):
        super().__init__()
        for name, value in dict(input_dim=input_dim, channels=channels, grid=grid,
                                size=size).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        self.input_dim, self.channels = input_dim, channels
        self.grid, self.size = grid, size
        self.args = dict(input_dim=input_dim, channels=channels, grid=grid, size=size,
            decoder_width=decoder_width, decoder_temporal_blocks=decoder_temporal_blocks,
            decoder_resblocks=decoder_resblocks, use_checkpoint=bool(use_checkpoint))
        self.decoder = _VideoDecoder(channels, grid, size, decoder_width,
            decoder_temporal_blocks, decoder_resblocks, use_checkpoint)
        self.project = nn.Linear(input_dim, channels)
        self.restore_features = nn.Linear(channels, input_dim)

    def specification(self):
        return dict(schema='matched-video-rae-v1', **self.args, temporal_factor=1,
                    rgb_skip=False, feature_normalization='per-token-layernorm-before-pool',
                    decoder_causal=False)

    def feature_targets(self, features):
        if features.ndim != 4 or features.shape[-1] != self.input_dim:
            raise ValueError('Expected features [B,T,N,input_dim]')
        b, t, n, c = features.shape
        side = math.isqrt(n)
        if side*side != n:
            raise ValueError('Feature patch grid must be square, with special tokens removed')
        normalized = F.layer_norm(features.float(), (c,))
        if side == self.grid:
            return normalized
        pooled = F.adaptive_avg_pool2d(normalized.reshape(b*t, side, side, c)
                                      .permute(0, 3, 1, 2), self.grid)
        return pooled.permute(0, 2, 3, 1).reshape(b, t, self.grid*self.grid, c)

    def encode(self, features):
        return self.project(self.feature_targets(features))

    def decode(self, z):
        if z.ndim == 5:
            if z.shape[2:4] != (self.grid, self.grid):
                raise ValueError('Spatial latent shape does not match codec grid')
            z = z.flatten(2, 3)
        if z.ndim != 4 or z.shape[-2:] != (self.grid*self.grid, self.channels):
            raise ValueError('Expected latent [B,T,grid²,channels] or [B,T,grid,grid,channels]')
        return self.decoder(z)

    def forward(self, features):
        targets = self.feature_targets(features)
        z = self.project(targets)
        return self.decode(z), z, self.restore_features(z), targets.detach()
