"""Diagnostics at the R7 diffusion boundary (never a geometry-quality gate)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize(x, stats):
    return (x.float() - stats['mean'][None, :, None]) / stats['std'][None, :, None]


def inverse(x, stats):
    return x.float() * stats['std'][None, :, None] + stats['mean'][None, :, None]


def fit_statistics(values, floor=1e-4):
    """Exact train-only moments, reduced over samples and space on CPU FP64."""
    if floor <= 0 or values.ndim != 4 or not torch.isfinite(values).all():
        raise ValueError('finite [B,T,N,C] train values and positive std floor required')
    x = values.detach().cpu().double()
    mean = x.mean((0, 2))
    variance = (x - mean[None, :, None]).square().mean((0, 2))
    std = variance.sqrt()
    return {'mean': mean.float(), 'std': std.clamp_min(floor).float(),
            'count': x.shape[0] * x.shape[2], 'std_floor': float(floor),
            'floored_channels': int((std < floor).sum())}


def to_device(stats, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in stats.items()}


def denoising_baselines(noisy, target, t):
    """Actual same-noise errors; division baseline is undefined at data-time zero."""
    row = {'mean_x0_mse': float(target.float().square().mean()),
           'identity_x0_mse': float(F.mse_loss(noisy.float(), target.float()))}
    if float(t) > 0:
        row['rescaled_identity_x0_mse'] = float(F.mse_loss(
            noisy.float() / float(t), target.float()))
        row['rescaled_identity_expected_mse'] = ((1 - float(t)) / float(t)) ** 2
    else:
        row['rescaled_identity_x0_mse'] = None
        row['rescaled_identity_expected_mse'] = None
    return row


def _delta_metrics(prediction, target):
    p = prediction.diff(dim=1).flatten(2).float()
    y = target.diff(dim=1).flatten(2).float()
    pn, yn = p.norm(dim=-1), y.norm(dim=-1)
    valid = yn > 1e-8
    rows = {}
    for i in range(y.shape[1]):
        mask = valid[:, i]
        rows[f'chunk{i+1}/valid_motion_fraction'] = float(mask.float().mean())
        rows[f'chunk{i+1}/motion_cosine'] = (float(F.cosine_similarity(
            p[:, i], y[:, i], dim=-1)[mask].mean()) if mask.any() else None)
        rows[f'chunk{i+1}/motion_ratio'] = (float(
            (pn[:, i] / yn[:, i].clamp_min(1e-8))[mask].mean()) if mask.any() else None)
    return rows


def generation_metrics(generated, target, anchor, target_stats, anchor_stats):
    """Raw inputs. Centered motion is diagnostic, not physical correspondence.

    The first difference still shares the observed anchor. A high first-chunk
    cosine, even after centering, is not evidence of successful motion generation.
    Std ratios measure within-tensor variation, NOT conditional seed diversity.
    """
    g, y, c = generated.float(), target.float(), anchor.float()
    gn, yn = normalize(g, target_stats), normalize(y, target_stats)
    mu = target_stats['mean'][None, :, None].expand_as(y)
    cm = anchor_stats['mean'][None, :, None]
    row = {'raw_mse': float(F.mse_loss(g, y)),
           'normalized_mse': float(F.mse_loss(gn, yn)),
           'raw_std_ratio': float(g.std(unbiased=False) / y.std(unbiased=False).clamp_min(1e-8)),
           'normalized_std_ratio': float(gn.std(unbiased=False) / yn.std(unbiased=False).clamp_min(1e-8)),
           'mean_only_normalized_mse': float(yn.square().mean())}
    truth = torch.cat((c, y), 1)
    for prefix, pred, tgt in (
            ('raw', torch.cat((c, g), 1), truth),
            ('mean_only', torch.cat((c, mu), 1), truth),
            ('position_centered', torch.cat((c-cm, g-mu), 1),
             torch.cat((c-cm, y-mu), 1))):
        row.update({f'{prefix}/{k}': v for k, v in _delta_metrics(pred, tgt).items()})
    return row


def complete_prefix(anchor, future, full_frames):
    """Fill only with generated candidates; there is no ground-truth suffix input."""
    if (anchor.ndim != 4 or anchor.shape[1] != 1 or future.ndim != 4
            or anchor.shape[0] != future.shape[0]
            or anchor.shape[2:] != future.shape[2:]
            or not 1 <= future.shape[1] < full_frames):
        raise ValueError('invalid clean-anchor/candidate-prefix layout')
    suffix = future[:, -1:].expand(-1, full_frames-1-future.shape[1], -1, -1)
    return torch.cat((anchor, future, suffix), 1)
