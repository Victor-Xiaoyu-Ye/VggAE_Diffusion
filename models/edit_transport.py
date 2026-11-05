"""Learn a decoder residual around a reference edit, not its replacement.

The RGB optical-flow proposal is an exact bypass. A frozen RAE renders a
learned latent correction only where source correspondence is uncertain. The
observed first-frame edit is never projected back through the RAE. Source video
and one edited first frame are legal inputs; edited future frames are not.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _channel_scale(scale, channels, value):
    result = torch.as_tensor(scale, device=value.device, dtype=value.dtype)
    if result.numel() != channels:
        raise ValueError(f'Expected {channels} latent channel scales')
    # Statistics are supplied by the training-source bank, never by targets.
    return result.reshape(1, 1, 1, 1, channels).clamp_min(1e-5)


def _check_inputs(source_z, anchor_z, source, proposal, confidence, masks):
    if source_z.ndim != 5:
        raise ValueError('source_z must be [B,T,G,G,C]')
    batch, frames, height, width, _ = source_z.shape
    if height != width or frames < 1:
        raise ValueError('Expected a square source grid and at least one frame')
    if anchor_z.shape != source_z[:, :1].shape:
        raise ValueError('anchor_z must contain exactly one edited first-frame state')
    if source.ndim != 5 or source.shape[:3] != (batch, frames, 3):
        raise ValueError('source RGB must be [B,T,3,H,W]')
    if proposal.shape != source.shape:
        raise ValueError('Source and RGB proposal must be aligned')
    expected = (batch, frames, 1, *source.shape[-2:])
    if confidence.shape != expected or masks.shape != expected:
        raise ValueError('Confidence and support must be [B,T,1,H,W]')


class _TransportBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(width)
        self.spatial = nn.Sequential(nn.Conv2d(width, width, 3, padding=1,
                                               groups=width), nn.GELU(),
                                     nn.Conv2d(width, width, 1))
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal = nn.MultiheadAttention(width, heads, dropout=dropout,
                                              batch_first=True)
        self.query_norm = nn.LayerNorm(width)
        self.anchor_norm = nn.LayerNorm(width)
        self.anchor = nn.MultiheadAttention(width, heads, dropout=dropout,
                                            batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(width * 4, width))

    def forward(self, tokens, anchor, grid):
        batch, frames, positions, channels = tokens.shape
        spatial = self.spatial_norm(tokens).reshape(batch * frames, grid, grid,
                                                    channels).permute(0, 3, 1, 2)
        spatial = self.spatial(spatial).permute(0, 2, 3, 1).reshape_as(tokens)
        tokens = tokens + spatial
        temporal = self.temporal_norm(tokens).permute(0, 2, 1, 3).reshape(
            batch * positions, frames, channels)
        temporal = self.temporal(temporal, temporal, temporal,
                                 need_weights=False)[0]
        tokens = tokens + temporal.reshape(batch, positions, frames, channels).permute(0, 2, 1, 3)
        query = self.query_norm(tokens).reshape(batch, frames * positions, channels)
        memory = self.anchor_norm(anchor)
        tokens = tokens + self.anchor(query, memory, memory,
                                      need_weights=False)[0].reshape_as(tokens)
        return tokens + self.ff(self.ff_norm(tokens))


class EditTransport(nn.Module):
    """Small, noncausal transport network over an already observed source video.

    forward(source_z, anchor_z, source, proposal, confidence, masks, scale)
    returns a standardized latent residual [B,T,G,G,C]. Only the edited anchor
    is encoded as an edit condition; no edited future latent is an input.
    A zero-initialized output head makes the initial reference exactly the RGB
    proposal. The source clip can be bidirectional because this is editing an
    observed clip, not future prediction.
    """
    def __init__(self, cfg, input_channels=256):
        super().__init__()
        self.input_channels = int(input_channels)
        self.width = int(cfg.get('width', 192))
        heads = int(cfg.get('heads', 6))
        depth = int(cfg.get('depth', 4))
        self.max_frames = int(cfg.get('max_frames', 17))
        self.delta_limit = float(cfg.get('delta_limit', .75))
        self.activation_checkpoint = bool(cfg.get('activation_checkpoint', False))
        dropout = float(cfg.get('dropout', 0.))
        if self.width < 1 or heads < 1 or self.width % heads or depth < 1:
            raise ValueError('Positive width/depth and head-divisible width required')
        if self.max_frames < 1 or self.delta_limit <= 0:
            raise ValueError('Positive frame capacity and delta limit required')
        self.source_projection = nn.Sequential(nn.LayerNorm(input_channels),
                                               nn.Linear(input_channels, self.width))
        self.anchor_projection = nn.Sequential(nn.LayerNorm(2 * input_channels),
                                               nn.Linear(2 * input_channels, self.width))
        # Source RGB, transported RGB edit, confidence, support = 8 channels.
        self.pixel_projection = nn.Linear(8, self.width)
        self.spatial_position = nn.Sequential(nn.Linear(2, self.width), nn.GELU(),
                                              nn.Linear(self.width, self.width))
        self.time_position = nn.Parameter(torch.zeros(1, self.max_frames, 1, self.width))
        nn.init.normal_(self.time_position, std=.02)
        self.blocks = nn.ModuleList([_TransportBlock(self.width, heads, dropout)
                                    for _ in range(depth)])
        self.output_norm = nn.LayerNorm(self.width)
        self.output = nn.Linear(self.width, input_channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, source_z, anchor_z, source, proposal, confidence, masks, scale):
        _check_inputs(source_z, anchor_z, source, proposal, confidence, masks)
        batch, frames, grid, _, channels = source_z.shape
        if channels != self.input_channels or frames > self.max_frames:
            raise ValueError('Source channels/frame count exceed model configuration')
        scale = _channel_scale(scale, channels, source_z)
        normalized = source_z / scale
        tokens = self.source_projection(normalized).reshape(batch, frames, grid * grid, self.width)
        pixel = torch.cat((source, proposal - source, confidence.clamp(0, 1),
                           masks.clamp(0, 1)), dim=2)
        pixel = F.adaptive_avg_pool2d(pixel.flatten(0, 1), (grid, grid))
        pixel = pixel.permute(0, 2, 3, 1).reshape(batch, frames, grid * grid, 8)
        pixels = self.pixel_projection(pixel.to(tokens.dtype))
        axis = torch.linspace(-1, 1, grid, device=tokens.device, dtype=tokens.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        position = self.spatial_position(torch.stack((xx, yy), dim=-1).reshape(1, grid * grid, 2))
        memory_input = torch.cat((anchor_z / scale,
                                  (anchor_z - source_z[:, :1]) / scale), dim=-1)
        memory = self.anchor_projection(memory_input).reshape(batch, grid * grid, self.width)
        memory = memory + pixels[:, 0] + position
        tokens = tokens + pixels + position[:, None] + self.time_position[:, :frames]
        for block in self.blocks:
            if self.activation_checkpoint and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                tokens = checkpoint(block, tokens, memory, grid, use_reentrant=False)
            else:
                tokens = block(tokens, memory, grid)
        delta = self.delta_limit * torch.tanh(self.output(self.output_norm(tokens)))
        delta = delta.reshape(batch, frames, grid, grid, channels)
        # The observed anchor is a boundary condition, never a predicted state.
        return torch.cat((torch.zeros_like(delta[:, :1]), delta[:, 1:]), dim=1)


def decoded_reference_frame(decode, source_z, delta, scale, source, proposal,
                            confidence, masks, frame, cfg, base=None):
    """Return (reference, diagnostics) for one frame of a batched source clip.

    The caller freezes decoder weights, but must NOT disable gradients around
    this function during training. Gradients through D(z + scale * delta) train
    the transport network. D(z) is detached, so reconstruction bias cancels.
    A high source-flow confidence protects an observed effect from correction;
    low confidence permits a residual within dilated support. Renderer support
    is returned separately: multiplying by confidence again would discard the
    uncertain-area corrections this network is meant to learn.
    """
    if delta.shape != source_z.shape:
        raise ValueError('Latent residual must match source states')
    _check_inputs(source_z, source_z[:, :1], source, proposal, confidence, masks)
    if not 0 <= int(frame) < source_z.shape[1]:
        raise ValueError('Frame index outside source window')
    frame = int(frame)
    trusted = float(cfg.get('trusted_confidence', .95))
    strength = float(cfg.get('residual_strength', 1.))
    dilation = int(cfg.get('support_dilation', 9))
    if not 0 < trusted <= 1 or strength < 0 or dilation < 1 or dilation % 2 != 1:
        raise ValueError('Valid confidence threshold, strength and odd support dilation required')
    support = F.max_pool2d(masks[:, frame].clamp(0, 1), dilation, stride=1,
                           padding=dilation // 2)
    gate = support * (1 - confidence[:, frame].clamp(0, 1) / trusted).clamp(0, 1)
    if frame == 0:
        zero = torch.zeros_like(proposal[:, frame])
        return proposal[:, frame], dict(gate=torch.zeros_like(gate),
                                        editable_support=support,
                                        raw_residual=zero, applied_residual=zero)
    z = source_z[:, frame:frame + 1].detach()
    if base is None:
        with torch.no_grad():
            base = decode(z)
    base = base.detach().float()
    edited = decode(z + _channel_scale(scale, z.shape[-1], z) * delta[:, frame:frame + 1])
    if edited.shape != proposal[:, frame].shape or base.shape != edited.shape:
        raise ValueError('Decoded RGB must match the reference RGB canvas')
    raw = edited.float() - base
    correction = strength * gate * raw
    reference = (proposal[:, frame] + correction).clamp(0, 1)
    return reference, dict(gate=gate, editable_support=support,
                           raw_residual=raw, applied_residual=correction)
