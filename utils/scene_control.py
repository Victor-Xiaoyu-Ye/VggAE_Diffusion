"""Stage56 fixed-window controls and observed-only texture transport.

Future camera/track maps are EXPLICIT supplied controls, not autonomous future
predictions. Appearance is sampled only from the observed prefix. Query IDs are
sparse point correspondences, never semantic instance labels. All latent
targets, source statistics and SceneFlow token positions remain unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.scene_flow import SceneFlow
from models.r7_window_dit import position, sinusoid
from utils.scene_prefix import PrefixFlowNet


CONTROL_ARMS = ('baseline', 'motion', 'reference')
CONTROL_SCHEMA = 'scene_control_v1'


def _tensor(value, ndim, device, name):
    value = torch.as_tensor(value, device=device, dtype=torch.float32)
    if value.ndim == ndim - 1:
        value = value.unsqueeze(0)
    if value.ndim != ndim:
        raise ValueError(f'{name} must be a {ndim}-dimensional batched tensor')
    return value


def _xy_weights(xy, grid):
    """Bilinear grid-center weights; normalized RGB coordinates are in [0,1].

    Borders use border padding, matching align_corners=False interpolation.
    Out-of-image and nonfinite points have zero support, rather than being
    clamped into a valid edge token. No grid_sample NPU kernel is required.
    """
    valid = torch.isfinite(xy).all(-1) & (xy >= 0).all(-1) & (xy <= 1).all(-1)
    safe = torch.nan_to_num(xy, nan=0., posinf=0., neginf=0.).clamp(0., 1.)
    coord = safe * grid - .5
    low = coord.floor().long()
    fraction = coord - low
    entries = []
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        x = (low[..., 0] + dx).clamp(0, grid - 1)
        y = (low[..., 1] + dy).clamp(0, grid - 1)
        weight = (fraction[..., 0] if dx else 1 - fraction[..., 0])
        weight = weight * (fraction[..., 1] if dy else 1 - fraction[..., 1])
        entries.append((y * grid + x, weight * valid))
    return entries, valid


def transport_prefix_texture(observed, prefix_tracks, prefix_track_weight,
                             destination_tracks, destination_weight, *,
                             texture_start=None):
    """Transport observed texture features to prescribed destinations.

    Args use [B,H,N,C], [B,H,Q,2], [B,H,Q], [B,F,Q,2], [B,F,Q].
    Texture channels are the trailing half unless explicitly supplied. For
    each query use its latest visible independent-prefix observation. Geometry
    channels are never transported; future latents/RGB are not arguments.
    Return [B,F,N,C_texture+1], with a final support/confidence channel in [0,1].
    Visibility gates both source and destination; holes are exactly zero.
    """
    if not torch.is_tensor(observed) or observed.ndim != 4:
        raise ValueError('observed must be [B,H,N,C]')
    b, h, n, channels = observed.shape
    grid = math.isqrt(n)
    if grid * grid != n or h < 1:
        raise ValueError('observed must contain a square token grid and nonempty prefix')
    start = channels // 2 if texture_start is None else texture_start
    if not isinstance(start, int) or not 0 < start < channels:
        raise ValueError('texture_start must leave nonempty geometry and texture channels')
    src = _tensor(prefix_tracks, 4, observed.device, 'prefix_tracks')
    src_w = _tensor(prefix_track_weight, 3, observed.device, 'prefix_track_weight')
    dst = _tensor(destination_tracks, 4, observed.device, 'destination_tracks')
    dst_w = _tensor(destination_weight, 3, observed.device, 'destination_weight')
    if src.shape[:2] != (b, h) or src.shape[-1] != 2:
        raise ValueError('prefix tracks must match the observed prefix')
    q = src.shape[2]
    if src_w.shape != (b, h, q) or dst.shape[0] != b or dst.shape[2:] != (q, 2):
        raise ValueError('prefix and destination tracks must preserve query IDs')
    if dst_w.shape != dst.shape[:-1]:
        raise ValueError('destination weights must match destination tracks')
    f, ctex = dst.shape[1], channels - start
    result = observed.new_zeros((b, f, n, ctex + 1), dtype=torch.float32)
    if q == 0:
        return result
    # Invalid confidence is absence, not evidence of a valid trajectory.
    src_w = torch.nan_to_num(src_w, nan=0., posinf=0., neginf=0.).clamp(0, 1)
    dst_w = torch.nan_to_num(dst_w, nan=0., posinf=0., neginf=0.).clamp(0, 1)
    src_entries, src_valid = _xy_weights(src, grid)
    src_w = src_w * src_valid
    times = torch.arange(h, device=observed.device)[None, :, None]
    latest = torch.where(src_w > 0, times, -1).amax(1)
    source_exists = latest >= 0
    latest_safe = latest.clamp_min(0)
    features = observed[..., start:].float()
    sampled = features.new_zeros((b, h, q, ctex))
    for index, weight in src_entries:
        sampled = sampled + features.gather(2, index[..., None].expand(b, h, q, ctex)) * weight[..., None]
    chosen = sampled.gather(1, latest_safe[:, None, :, None].expand(b, 1, q, ctex)).squeeze(1)
    chosen_weight = src_w.gather(1, latest_safe[:, None]).squeeze(1) * source_exists
    dst_entries, _ = _xy_weights(dst, grid)
    numerator = features.new_zeros((b, f, n, ctex))
    denominator = features.new_zeros((b, f, n, 1))
    for index, bilinear_weight in dst_entries:
        weight = bilinear_weight * dst_w * chosen_weight[:, None]
        numerator.scatter_add_(2, index[..., None].expand(b, f, q, ctex),
                               chosen[:, None].expand(b, f, q, ctex) * weight[..., None])
        denominator.scatter_add_(2, index[..., None], weight[..., None])
    result[..., :ctex] = numerator / denominator.clamp_min(1e-8)
    result[..., ctex:] = denominator.clamp(0, 1)
    return result


class ControlledSceneFlow(SceneFlow):
    """Original state names plus a zero-initialized per-token control stem.

    Existing self-attention mixes these spatial/temporal maps. Actual camera
    controls enter only the new stem: the warm-start camera MLP still receives
    zeros exactly as stage53/54 did. Therefore arbitrary source camera biases
    cannot break initialization equivalence. Both control arms have identical
    parameter shapes; motion-only receives zeros for reference channels.
    """
    def __init__(self, **model_args):
        super().__init__(**model_args)
        self.texture_channels = self.channels - self.channels // 2
        self.control_channels = 16 + self.texture_channels + 1
        hidden = min(128, self.args['width'])
        self.control = nn.Sequential(nn.Linear(self.control_channels, hidden),
                                     nn.SiLU(), nn.Linear(hidden, self.args['width']))
        nn.init.zeros_(self.control[-1].weight)
        nn.init.zeros_(self.control[-1].bias)

    def forward(self, noisy, u, anchor, text=None, text_valid=None,
                ref_present=None, rays=None, controls=None):
        b, t, n, c = noisy.shape
        if n != self.grid ** 2 or c != self.channels:
            raise ValueError('SceneFlow latent layout mismatch')
        if rays is not None:
            raise ValueError('Pass stage56 camera rays inside controls, not the legacy camera path')
        if controls is None:
            controls = noisy.new_zeros((b, t, n, self.control_channels))
        if controls.shape != (b, t, n, self.control_channels):
            raise ValueError('control features must match every original SceneFlow token slot')
        pos = position(t, self.grid, self.args['width']).to(noisy.device)
        x = self.input(noisy.float().flatten(1, 2)) + pos
        x = x + self.camera(noisy.new_zeros(b, t * n, 8).float())
        # Always execute the stem, even for absent control, so DDP sees every
        # new parameter. No mutable forward hook or global conditioning state.
        x = x + self.control(controls.float().flatten(1, 2))
        memory = self.ref(anchor[:, 0].float())
        if ref_present is None:
            ref_present = torch.ones(b, device=noisy.device, dtype=torch.bool)
        memory = torch.where(ref_present[:, None, None], memory, self.null_ref)
        memory = memory + position(1, self.grid, self.args['width'], anchor=True).to(noisy.device)
        context = self.text(text.float()) if self.text is not None else None
        time = self.time(sinusoid(u * 1000, 256))[:, None]
        for blocks in (self.trunk, self.wide):
            for block in blocks:
                if self.training and self.checkpoint_blocks:
                    x = checkpoint(block, x, time, memory, context, text_valid, use_reentrant=False)
                else:
                    x = block(x, time, memory, context, text_valid)
            if blocks is self.trunk:
                x, memory, time = self.widen(x), self.ref_widen(memory), self.time_widen(time)
                if context is not None:
                    context = self.text_widen(context)
        return self.output(x).reshape(b, t, n, c).float()


def create_control_model(model_args, arm):
    if arm not in CONTROL_ARMS:
        raise ValueError(f'Unknown control arm: {arm}')
    return SceneFlow(**model_args) if arm == 'baseline' else ControlledSceneFlow(**model_args)


def load_warm(model, state):
    """Require every original weight; permit missing new control weights only.

    A stage56 resume must still use ordinary strict load_state_dict, not this
    warm-start helper. Missing source keys or unexpected keys are never hidden.
    """
    base = getattr(model, 'module', model)
    expected = base.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    mismatched = [key for key in set(expected) & set(state)
                  if expected[key].shape != state[key].shape]
    allowed_missing = {key for key in expected if key.startswith('control.')}
    if missing - allowed_missing or unexpected or mismatched:
        raise ValueError(f'Invalid warm start: missing={sorted(missing - allowed_missing)}, '
                         f'unexpected={sorted(unexpected)}, mismatched={sorted(mismatched)}')
    if missing and missing != allowed_missing:
        raise ValueError('A warm start must include either the whole control stem or none of it')
    if missing:
        # Evaluation can temporarily reuse a trained core to repair the source
        # baseline. A missing adapter must restore source behavior, not leave
        # the trained adapter silently active under a source-baseline label.
        nn.init.zeros_(base.control[-1].weight)
        nn.init.zeros_(base.control[-1].bias)
    return base.load_state_dict(state, strict=not missing)


def control_features(batch, sample, arm):
    """Build fixed maps without reading future latent or RGB appearance values."""
    if arm not in CONTROL_ARMS:
        raise ValueError(f'Unknown control arm: {arm}')
    observed = torch.cat((batch.anchor, batch.known), 1).detach()
    b, h, n, channels = observed.shape
    t = h + batch.target.shape[1]
    ctex = channels - channels // 2
    features = observed.new_zeros((b, t - 1, n, 16 + ctex + 1), dtype=torch.float32)
    if arm == 'baseline':
        return features
    controls = sample.get('controls') if isinstance(sample, Mapping) else None
    if controls is None:
        return features
    if not isinstance(controls, Mapping):
        raise ValueError('sample controls must be a mapping')
    for key, offset in (('rays', 0), ('motion', 8)):
        if key not in controls:
            continue
        value = _tensor(controls[key], 4, observed.device, key)
        if value.shape != (b, t, n, 8):
            raise ValueError(f'{key} must preserve full cached frame and token indexing')
        if not torch.isfinite(value).all():
            raise ValueError(f'{key} contains nonfinite control values')
        features[..., offset:offset + 8] = value[:, 1:]
    if arm == 'reference':
        names = ('prefix_tracks', 'prefix_track_weight', 'tracks', 'track_weight')
        present = [key in controls for key in names]
        if any(present) and not all(present):
            raise ValueError('Reference transport requires all independent-prefix and destination tracks')
        if all(present):
            tracks = _tensor(controls['tracks'], 4, observed.device, 'tracks')
            weights = _tensor(controls['track_weight'], 3, observed.device, 'track_weight')
            if tracks.shape[:2] != (b, t) or weights.shape[:2] != (b, t):
                raise ValueError('Full track controls must preserve all original frames')
            transported = transport_prefix_texture(observed,
                controls['prefix_tracks'], controls['prefix_track_weight'],
                tracks[:, h:], weights[:, h:])
            features[:, h - 1:, :, 16:] = transported
    return features.detach()


class ControlledPrefixFlowNet(PrefixFlowNet):
    def __init__(self, model, known_prefix, controls, **kwargs):
        super().__init__(model, known_prefix, **kwargs)
        self.register_buffer('controls', controls, persistent=False)

    def forward(self, noisy_future, u, anchor, text=None, text_valid=None):
        b, future, n, c = noisy_future.shape
        known = self.known_prefix
        if known.shape != (b, self.model_frames - future, n, c) or anchor.shape != (b, 1, n, c):
            raise ValueError('Controlled prefix must preserve original SceneFlow slots')
        full = torch.cat((known, noisy_future), dim=1)
        null_text = noisy_future.new_zeros((b, 1, self.text_dim))
        valid = torch.ones((b, 1), device=noisy_future.device, dtype=torch.bool)
        present = torch.ones(b, device=noisy_future.device, dtype=torch.bool)
        output = self.model(full, u, anchor, null_text, valid,
                            ref_present=present, rays=None, controls=self.controls)
        if output.shape != full.shape:
            raise ValueError('SceneFlow must return every original slot')
        return output[:, known.shape[1]:]


def bind_control(model, batch, sample, arm, training=False, *, model_frames=None):
    """Ephemeral wrapper for WindowFlow and DDP; optimizer/checkpoint the core.

    Training currently uses exactly the same supplied controls as evaluation.
    Sparse/editable control curricula are deliberately not silently enabled.
    """
    del training
    if arm not in CONTROL_ARMS:
        raise ValueError(f'Unknown control arm: {arm}')
    slots = batch.known.shape[1] + batch.target.shape[1] if model_frames is None else model_frames
    if arm == 'baseline':
        return PrefixFlowNet(model, batch.known, model_frames=slots)
    return ControlledPrefixFlowNet(model, batch.known, control_features(batch, sample, arm),
                                   model_frames=slots)
