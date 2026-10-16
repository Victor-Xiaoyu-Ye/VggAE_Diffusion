"""Observed-prefix completion using unchanged SceneFlow token positions.

The normal source cache contains z0..z8. z0 is the independent anchor and
z1..z8 retain their original causal encoder context. With four observed frames,
the denoiser still receives eight slots: clean z1..z3 followed by five noisy
future slots. Only those five future outputs are trained or integrated.

This module does not establish encoder prefix invariance: that is a separate
real-encoder check. Never rebase an arbitrary interior cache slice as a new
independent anchor. Keep the source cache's normalization for a warm start.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class PrefixBatch:
    """Normalized [B, T, N, C] tensors; ``known`` excludes the anchor slot."""

    anchor: torch.Tensor
    known: torch.Tensor
    target: torch.Tensor

    @property
    def history_frames(self):
        return 1 + self.known.shape[1]


def _batched(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim not in (3, 4):
        raise ValueError(f'{name} must be a [T,N,C] or [B,T,N,C] tensor')
    return value.unsqueeze(0) if value.ndim == 3 else value


def prepare_prefix(sample, mean, std, hist=4, *, device=None):
    """Split cached latents into observed-only conditions and future targets.

    ``sample`` is either the full cached target tensor or a mapping containing
    ``target`` and optionally its separately stored ``anchor``. Both unbatched
    and stacked cache tensors are accepted. Statistics are the *original*
    per-channel cache statistics, not statistics computed from this window or
    from a newly selected domain. Output tensors are float32, on ``device``
    when provided or on the target tensor's device otherwise.

    No captions, rays, or other sample metadata enter the conditions. Reading
    future RGB/latents here is only for the returned training target; the
    anchor and known prefix have no dependency on those future values.
    """
    target = sample['target'] if isinstance(sample, Mapping) else sample
    target = _batched(target, 'target')
    if isinstance(hist, bool) or not isinstance(hist, int) or not 1 <= hist < target.shape[1]:
        raise ValueError('hist must leave at least one observed and one future frame')
    anchor = sample.get('anchor', target[:, :1]) if isinstance(sample, Mapping) else target[:, :1]
    anchor = _batched(anchor, 'anchor')
    if anchor.shape != (target.shape[0], 1, target.shape[2], target.shape[3]):
        raise ValueError('anchor must contain exactly the original first frame for every sample')
    target = target.to(device=device, dtype=torch.float32)
    anchor = anchor.to(device=target.device, dtype=torch.float32)
    mean = torch.as_tensor(mean, device=target.device, dtype=torch.float32)
    std = torch.as_tensor(std, device=target.device, dtype=torch.float32)
    if mean.ndim != 1 or std.ndim != 1 or mean.numel() != target.shape[-1] or std.shape != mean.shape:
        raise ValueError('mean/std must be the source cache per-channel vectors')
    # Match the existing Scene trainer and replay normalization exactly.
    std = std.clamp_min(1e-4)
    normalized = (target - mean) / std
    return PrefixBatch(anchor=(anchor - mean) / std,
                       known=normalized[:, 1:hist], target=normalized[:, hist:])


class PrefixFlowNet(nn.Module):
    """Bind observed slots around an ordinary SceneFlow or an existing DDP.

    Construct per batch with ``PrefixFlowNet(core, batch.known)`` and call with
    ``(noisy_future, u, batch.anchor)``. The forward signature also accepts the
    text arguments passed by WindowFlow.sample, but intentionally ignores them:
    this task always uses one zero text token, a present reference, and no rays.

    Use the original model/core for the optimizer, no_sync, and checkpoints;
    this is an ephemeral condition wrapper, not a new model checkpoint schema.
    ``model_frames=8`` preserves every position used by the source model.
    """

    def __init__(self, model, known_prefix, text_dim=None, model_frames=8):
        super().__init__()
        if not isinstance(known_prefix, torch.Tensor) or known_prefix.ndim != 4:
            raise ValueError('known_prefix must be a normalized [B,H-1,N,C] tensor')
        if model_frames < 1 or known_prefix.shape[1] >= model_frames:
            raise ValueError('known_prefix must leave at least one model slot for generation')
        base = getattr(model, 'module', model)
        if text_dim is None:
            text_dim = getattr(base, 'args', {}).get('text_dim', 4096)
        if isinstance(text_dim, bool) or not isinstance(text_dim, int) or text_dim < 0:
            raise ValueError('text_dim must be a nonnegative integer')
        self.model = model
        self.register_buffer('known_prefix', known_prefix, persistent=False)
        self.text_dim = text_dim
        self.model_frames = model_frames

    def forward(self, noisy_future, u, anchor, text=None, text_valid=None):
        if noisy_future.ndim != 4 or anchor.ndim != 4:
            raise ValueError('future and anchor must be [B,T,N,C] tensors')
        b, future, n, c = noisy_future.shape
        known = self.known_prefix
        if known.shape[0] != b or known.shape[2:] != (n, c) or anchor.shape != (b, 1, n, c):
            raise ValueError('known prefix, anchor and future layouts must match')
        if known.shape[1] + future != self.model_frames:
            raise ValueError('prefix plus future must preserve the original model slot count')
        if known.device != noisy_future.device or anchor.device != noisy_future.device:
            raise ValueError('known prefix, anchor and future must be on the same device')
        full = torch.cat((known, noisy_future), dim=1)
        null_text = noisy_future.new_zeros((b, 1, self.text_dim))
        valid = torch.ones((b, 1), device=noisy_future.device, dtype=torch.bool)
        present = torch.ones(b, device=noisy_future.device, dtype=torch.bool)
        output = self.model(full, u, anchor, null_text, valid,
                            ref_present=present, rays=None)
        if output.shape != full.shape:
            raise ValueError('SceneFlow must return one prediction for every original slot')
        return output[:, known.shape[1]:]


def conservative_lr(step, max_steps, peak_lr=1e-5, min_lr=2e-6, warmup_steps=200):
    """Step-based warmup/cosine LR; ``step`` is the next zero-based update.

    The last planned update reaches min_lr. Wall time, IO, and validation do
    not enter the schedule. The caller must pin max_steps/warmup in the run
    contract and resume the update index rather than restart the schedule.
    For a one-update rehearsal the only update uses peak_lr.
    """
    if any(isinstance(v, bool) or not isinstance(v, int)
           for v in (step, max_steps, warmup_steps)):
        raise ValueError('step, max_steps and warmup_steps must be integers')
    if step < 0 or max_steps < 1 or warmup_steps < 0:
        raise ValueError('invalid schedule update counts')
    if not all(math.isfinite(v) for v in (peak_lr, min_lr)) or not 0 <= min_lr <= peak_lr or peak_lr <= 0:
        raise ValueError('learning rates must be finite and 0 <= min_lr <= peak_lr')
    if step >= max_steps:
        return float(min_lr)
    if max_steps == 1:
        return float(peak_lr)
    warmup = min(warmup_steps, max_steps - 1)
    if step < warmup:
        return float(peak_lr * (step + 1) / warmup)
    decay_steps = max_steps - warmup
    fraction = 1. if decay_steps == 1 else (step - warmup) / (decay_steps - 1)
    return float(min_lr + (peak_lr - min_lr) * (1 + math.cos(math.pi * fraction)) / 2)
