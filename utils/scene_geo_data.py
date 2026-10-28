"""Small, reproducible street-window selection; no new representation or labels."""
from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping

import torch


def order_key(seed, value):
    return hashlib.sha256(f'{seed}:{value}'.encode('utf8')).hexdigest()


def validate_window(sample, allowed_ids, channels, views=9, *, evaluation=False, grid=None):
    if not isinstance(sample, Mapping):
        raise ValueError('Window must be a mapping')
    if sample.get('dataset') != 'spatialvid' or sample.get('id') not in allowed_ids:
        raise ValueError('Window is outside the declared source split')
    target, anchor = sample.get('target'), sample.get('anchor')
    if (not torch.is_tensor(target) or target.ndim != 3 or target.shape[0] != views
            or target.shape[-1] != channels or not torch.isfinite(target).all()
            or not torch.is_tensor(anchor) or anchor.shape != target[:1].shape
            or not torch.equal(anchor, target[:1])):
        raise ValueError('Invalid original latent/independent-anchor contract')
    if grid is not None and target.shape[1] != grid**2:
        raise ValueError('Latent token grid differs from the source model')
    if len(sample.get('frame_ids', [])) != views:
        raise ValueError('Missing original frame indices')
    if evaluation:
        raw = sample.get('raw')
        if not torch.is_tensor(raw) or raw.dtype != torch.uint8 or raw.shape[:2] != (views, 3):
            raise ValueError('Evaluation must retain sealed uint8 RGB')


def activity_score(sample, std):
    """Temporal texture change, NOT an object motion/semantic mask."""
    z = sample['target'].float()
    half = z.shape[-1] // 2
    sigma = torch.as_tensor(std).float().clamp_min(1e-4)[half:]
    return float((z[1:, :, half:] - z[:-1, :, half:]).abs().div(sigma).mean())


def choose_windows(records, count, seed, high_change_fraction=.5):
    if type(count) is not int or count < 1 or not 0 <= high_change_fraction <= 1:
        raise ValueError('Invalid subset size or high-change fraction')
    unique = {}
    for record in records:
        if not math.isfinite(record['activity']):
            continue
        key = (record['id'], tuple(record['frame_ids']))
        unique.setdefault(key, record)
    pool = list(unique.values())
    count = min(count, len(pool))
    high = sorted(pool, key=lambda r: (-r['activity'], order_key(seed, r['key'])))[:int(count*high_change_fraction)]
    used = {r['key'] for r in high}
    rest = sorted((r for r in pool if r['key'] not in used), key=lambda r: order_key(seed, r['key']))
    chosen = [dict(r, selection_stratum='high_latent_change') for r in high]
    chosen += [dict(r, selection_stratum='uniform_remaining_candidates') for r in rest[:count-len(high)]]
    # Source locality avoids repeatedly staging a large source shard.
    return sorted(chosen, key=lambda r: (r['file'], r['index']))


def compact_sample(sample):
    # Clone to avoid serializing an entire original storage through a tensor view.
    return {k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in sample.items() if k in ('id', 'dataset', 'target', 'anchor', 'frame_ids',
                'fps', 'nominal_fps', 'source_fps', 'time_status')}
