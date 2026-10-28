"""Paired future-feature oracle for unchanged C256 SceneFlow diffusion.

This is a diagnostic upper-bound experiment, not deployable geometry control.
The first 128 cached channels come from the geometry branch, but can encode
appearance. They are not calibrated depth, pure geometry or predicted future
features. Both arms predict every C256 output channel with the original joint
objective; supplied features are never copied/clamped into denoising outputs.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import random

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.r7_window_dit import position, sinusoid
from models.scene_flow import SceneFlow
from utils.scene_prefix import PrefixFlowNet


GEO_ARMS = ('baseline', 'oracle_geo')
GEO_VARIANTS = ('correct', 'zero', 'shuffle')
GEO_SCHEMA = 'scene_future_geo_oracle_v1'
GEO_MODEL_FAMILY = 'GeoConditionedSceneFlow'
GEO_CHANNELS = 128
LATENT_CHANNELS = 256


def _choice(value, allowed, name):
    if value not in allowed:
        raise ValueError(f'Unknown {name}: {value}')


def _statistics(mean, std, *, device=None):
    mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(std, dtype=torch.float32, device=device)
    if mean.shape != (LATENT_CHANNELS,) or std.shape != mean.shape:
        raise ValueError('mean/std must be the original C256 source-cache vectors')
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError('source-cache mean/std must be finite')
    # Exactly the existing prefix training normalization, not window statistics.
    return mean, std.clamp_min(1e-4)


class GeoConditionedSceneFlow(SceneFlow):
    """Source state names plus one identical 129-channel adapter in both arms."""

    def __init__(self, **model_args):
        if model_args.get('channels', LATENT_CHANNELS) != LATENT_CHANNELS:
            raise ValueError('The future-geo oracle requires the original C256 representation')
        super().__init__(**model_args)
        hidden = min(128, self.args['width'])
        self.geo_condition = nn.Sequential(
            nn.Linear(GEO_CHANNELS + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, self.args['width']))
        nn.init.zeros_(self.geo_condition[-1].weight)
        nn.init.zeros_(self.geo_condition[-1].bias)

    def condition_delta(self, features):
        # Always traverse the adapter for DDP, including baseline/absent slots.
        # Gating after its biases makes absence an exact zero after training too.
        return self.geo_condition(features.float()) * features[..., -1:].float()

    def forward(self, noisy, u, anchor, text=None, text_valid=None,
                ref_present=None, rays=None, geo_features=None):
        b, t, n, c = noisy.shape
        if n != self.grid ** 2 or c != self.channels:
            raise ValueError('SceneFlow latent layout mismatch')
        if rays is not None:
            raise ValueError('The paired geo experiment retains the source zero-camera path')
        if geo_features is None:
            geo_features = noisy.new_zeros((b, t, n, GEO_CHANNELS + 1))
        if geo_features.shape != (b, t, n, GEO_CHANNELS + 1):
            raise ValueError('Future-geo features must preserve every original token slot')
        pos = position(t, self.grid, self.args['width']).to(noisy.device)
        x = self.input(noisy.float().flatten(1, 2)) + pos
        x = x + self.camera(noisy.new_zeros(b, t * n, 8).float())
        x = x + self.condition_delta(geo_features.flatten(1, 2))
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


def create_geo_model(model_args, arm):
    _choice(arm, GEO_ARMS, 'geo arm')
    # No arm-dependent initialization, layer or RNG draw.
    return GeoConditionedSceneFlow(**model_args)


def load_warm(model, state):
    """Strict backbone coverage; the entire new adapter may be absent only.

    Metadata validation belongs to GeoConditionRuntime before tensor loading.
    Full geo checkpoints are loaded strictly; ordinary resume must not drop
    adapter keys and call this helper as though it were a source checkpoint.
    """
    base = getattr(model, 'module', model)
    expected = base.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    mismatched = [key for key in set(expected) & set(state)
                  if expected[key].shape != state[key].shape]
    adapter = {key for key in expected if key.startswith('geo_condition.')}
    if missing - adapter or unexpected or mismatched:
        raise ValueError(f'Invalid geo warm start: missing={sorted(missing - adapter)}, '
                         f'unexpected={sorted(unexpected)}, mismatched={sorted(mismatched)}')
    if missing and missing != adapter:
        raise ValueError('A warm start must include the whole geo adapter or none of it')
    if missing:
        # Source-baseline replay can reuse a trained instance. Reset its final
        # projection so that a previously learned oracle is not left active.
        nn.init.zeros_(base.geo_condition[-1].weight)
        nn.init.zeros_(base.geo_condition[-1].bias)
    return base.load_state_dict(state, strict=not missing)


def geo_condition_features(batch, sample, arm, mean, std, history=4, *, variant='correct'):
    """Build [B,T-1,N,129] from original cached future geo and fixed stats.

    Correct uses ``sample['target'][:,history:,:,:128]`` from the original
    full-window causal encoding. Shuffle requires an explicit, unnormalized
    ``sample['donor_geo']`` of shape [F,N,128] or [B,F,N,128]. The caller must
    choose another case in the same cache. This helper never encodes a future
    subclip independently and never reads future RGB/depth/texture values.
    """
    _choice(arm, GEO_ARMS, 'geo arm')
    _choice(variant, GEO_VARIANTS, 'geo variant')
    if isinstance(history, bool) or not isinstance(history, int) or history < 1:
        raise ValueError('history must be a positive integer')
    b, f, n, c = batch.target.shape
    if c != LATENT_CHANNELS or batch.known.shape != (b, history - 1, n, c):
        raise ValueError('Prefix batch must preserve C256 and original history slots')
    if batch.anchor.shape != (b, 1, n, c):
        raise ValueError('Prefix batch must retain the original anchor')
    features = batch.anchor.new_zeros((b, history - 1 + f, n, GEO_CHANNELS + 1),
                                     dtype=torch.float32)
    # Do not even inspect target/donor fields for baseline and zero variants.
    if arm == 'baseline' or variant == 'zero':
        return features
    if not isinstance(sample, Mapping):
        raise ValueError('An oracle sample must preserve original cached target metadata')
    if variant == 'shuffle':
        geo = sample.get('donor_geo')
        if not isinstance(geo, torch.Tensor) or geo.ndim not in (3, 4):
            raise ValueError('shuffle requires explicit unnormalized donor_geo [F,N,128] or [B,F,N,128]')
        geo = geo.unsqueeze(0) if geo.ndim == 3 else geo
    else:
        target = sample.get('target')
        if not isinstance(target, torch.Tensor) or target.ndim not in (3, 4):
            raise ValueError('oracle_geo requires the original cached target tensor')
        target = target.unsqueeze(0) if target.ndim == 3 else target
        if target.shape != (b, history + f, n, c):
            raise ValueError('Target must preserve all original cached frames and C256 channels')
        # Slice before converting/copying. Other appearance channels never
        # enter the condition-building computation.
        geo = target[:, history:, :, :GEO_CHANNELS]
    if geo.shape != (b, f, n, GEO_CHANNELS):
        raise ValueError('Future geo must match future frame/token slots and contain exactly 128 channels')
    mean, std = _statistics(mean, std, device=features.device)
    geo = geo.detach().to(device=features.device, dtype=torch.float32)
    if not torch.isfinite(geo).all():
        raise ValueError('Supplied future geo contains nonfinite values')
    features[:, history - 1:, :, :GEO_CHANNELS] = (geo - mean[:GEO_CHANNELS]) / std[:GEO_CHANNELS]
    features[:, history - 1:, :, -1] = 1.
    return features.detach()


class GeoPrefixFlowNet(PrefixFlowNet):
    def __init__(self, model, known_prefix, geo_features, **kwargs):
        super().__init__(model, known_prefix, **kwargs)
        self.register_buffer('geo_features', geo_features, persistent=False)

    def forward(self, noisy_future, u, anchor, text=None, text_valid=None):
        b, future, n, c = noisy_future.shape
        known = self.known_prefix
        if known.shape != (b, self.model_frames - future, n, c) or anchor.shape != (b, 1, n, c):
            raise ValueError('Geo prefix must preserve original SceneFlow slots')
        full = torch.cat((known, noisy_future), dim=1)
        null_text = noisy_future.new_zeros((b, 1, self.text_dim))
        valid = torch.ones((b, 1), device=noisy_future.device, dtype=torch.bool)
        present = torch.ones(b, device=noisy_future.device, dtype=torch.bool)
        output = self.model(full, u, anchor, null_text, valid,
                            ref_present=present, rays=None, geo_features=self.geo_features)
        if output.shape != full.shape:
            raise ValueError('SceneFlow must predict all C256 channels at every original slot')
        return output[:, known.shape[1]:]


def bind_geo_condition(model, batch, sample, arm, mean, std, history=4,
                       training=False, *, variant='correct', model_frames=None):
    if training and variant != 'correct':
        raise ValueError('zero/shuffle are evaluation ablations, not training conditions')
    slots = batch.known.shape[1] + batch.target.shape[1] if model_frames is None else model_frames
    features = geo_condition_features(batch, sample, arm, mean, std, history, variant=variant)
    return GeoPrefixFlowNet(model, batch.known, features, model_frames=slots)


class GeoConditionRuntime:
    """Explicit train_scene_prefix runtime, paired with original source stats."""

    def __init__(self, arm, mean, std, history=4, control_lr_multiplier=10., *, variant='correct'):
        _choice(arm, GEO_ARMS, 'geo arm')
        _choice(variant, GEO_VARIANTS, 'geo variant')
        if isinstance(history, bool) or not isinstance(history, int) or history < 1:
            raise ValueError('history must be a positive integer')
        if not math.isfinite(float(control_lr_multiplier)) or float(control_lr_multiplier) <= 0:
            raise ValueError('Control learning-rate multiplier must be positive and finite')
        self.arm, self.history, self.variant = arm, history, variant
        self.control_lr_multiplier = float(control_lr_multiplier)
        mean, std = _statistics(mean, std, device='cpu')
        self.mean, self.std = mean.detach().clone(), std.detach().clone()
        serialized = json.dumps(dict(mean=self.mean.tolist(), std=self.std.tolist()),
                                sort_keys=True, separators=(',', ':'), allow_nan=False)
        self.stats_sha256 = hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def make_model(self, model_args):
        return create_geo_model(model_args, self.arm)

    def load_warm(self, model, state):
        return load_warm(model, state)

    def checkpoint_metadata(self):
        return dict(model_family=GEO_MODEL_FAMILY, geo_schema=GEO_SCHEMA,
            experiment_arm=self.arm, history=self.history,
            geo_condition_channels=GEO_CHANNELS, geo_presence_channels=1,
            geo_condition_stats_sha256=self.stats_sha256,
            control_lr_multiplier=self.control_lr_multiplier,
            prediction_channels=LATENT_CHANNELS,
            output_policy='predict all C256 with original joint objective; no GT copy or clamp',
            condition_policy='original full-window causal cache future geo128; prefix zero; fixed source stats'
                             if self.arm == 'oracle_geo' else 'observed prefix only; identical inactive adapter',
            oracle_diagnostic=True, deployable_future_condition=False,
            future_geo_may_contain_appearance=True,
            pure_geometry_claim=False)

    def validate_checkpoint_metadata(self, payload, *, resume):
        if not isinstance(payload, Mapping) or not isinstance(payload.get('model'), Mapping):
            raise ValueError('Checkpoint must contain a model state dictionary')
        family = payload.get('model_family')
        keys = payload['model']
        has_geo = any(key.startswith('geo_condition.') for key in keys)
        if family == GEO_MODEL_FAMILY:
            expected = self.checkpoint_metadata()
            for key in ('geo_schema', 'experiment_arm', 'history',
                        'geo_condition_stats_sha256', 'control_lr_multiplier'):
                if payload.get(key) != expected[key]:
                    raise ValueError(f'Geo checkpoint {key} differs from this runtime')
            if not has_geo:
                raise ValueError('Geo checkpoint is missing its adapter; not a valid resume')
        elif resume:
            raise ValueError('Geo resume requires its explicit model family and adapter metadata')
        elif family not in (None, 'SceneFlow') or has_geo or any(key.startswith('control.') for key in keys):
            raise ValueError('Warm source must be an ordinary SceneFlow or a matching geo checkpoint')
        elif any(key in payload for key in ('geo_schema', 'control_schema')):
            raise ValueError('Ordinary SceneFlow source must not carry another experiment schema')

    def optimizer_groups(self, model):
        base = getattr(model, 'module', model)
        backbone, adapter = [], []
        for name, param in base.named_parameters():
            (adapter if name.startswith('geo_condition.') else backbone).append(param)
        return [dict(params=backbone, lr_scale=1., name='backbone'),
                dict(params=adapter, lr_scale=self.control_lr_multiplier, name='geo_condition')]

    def _variant(self, sample):
        # Even malformed future metadata is irrelevant to an observed-only arm.
        variant = 'zero' if self.arm == 'baseline' else sample.get('geo_condition_variant', self.variant)
        _choice(variant, GEO_VARIANTS, 'geo variant')
        return variant

    def bind(self, model, batch, sample, training=False):
        variant = self._variant(sample)
        if training and self.arm != 'baseline' and variant != 'correct':
            raise ValueError('Training requires correct oracle conditions, not zero/shuffle ablations')
        return bind_geo_condition(model, batch, sample, self.arm, self.mean, self.std,
            self.history, training=training, variant='correct' if self.arm == 'baseline' else variant)

    @staticmethod
    def seed_training(seed, rank):
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)

    @torch.no_grad()
    def training_metrics(self, model, net, noisy):
        base = getattr(model, 'module', model)
        features = net.geo_features.float()
        injected = base.condition_delta(features.flatten(1, 2))
        original = base.input(torch.cat((net.known_prefix, noisy), 1).float().flatten(1, 2))
        rms = lambda value: float(value.float().square().mean().sqrt())
        return dict(geo_condition_rms=rms(features[..., :GEO_CHANNELS]),
                    geo_condition_presence_fraction=float(features[..., -1].mean()),
                    geo_condition_injection_rms=rms(injected),
                    geo_condition_to_input_rms_ratio=rms(injected) / max(rms(original), 1e-8),
                    control_lr_multiplier=self.control_lr_multiplier)

    def annotate(self, row, sample):
        variant = self._variant(sample)
        supplied = self.arm == 'oracle_geo' and variant != 'zero'
        row.update(experiment_arm=self.arm, geo_condition_variant=variant,
            source_eval_index=sample.get('source_eval_index'),
            task='future-feature oracle diagnostic' if self.arm == 'oracle_geo' else 'observed-prefix completion',
            future_control_supplied=supplied, future_geo_may_contain_appearance=True,
            deployable_future_condition=False, pure_geometry_claim=False,
            condition_provenance='original full-window causal cache; no independent future encoding',
            condition_source='different-case future geo128' if variant == 'shuffle' else
                             'same-case future geo128' if supplied else 'none',
            donor_case_id=sample.get('donor_case_id') if variant == 'shuffle' else None,
            condition_policy='No future texture channels, RGB or depth are supplied; geo features can contain appearance.',
            output_policy='All C256 channels predicted; no GT geometry output copy or clamp')
        if 'supervision' in row:
            row['supervision'] = 'Original joint C256 future denoising target; oracle separately receives future geo128.'
