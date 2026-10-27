"""Measure whether supplied controls contain signal; never choose thresholds.

Individual unsupported windows remain legal. An entirely unsupported condition
must be reported as absent, not mistaken for a completed motion/reference test.
Nonzero counts do not certify correspondence accuracy or generated image quality.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


SCHEMA = 'scene-control-readiness-v1'


def _np(value):
    return value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _score_statistics(value, threshold):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = np.isfinite(value)
    valid = value[finite]
    return dict(count=int(value.size), finite_count=int(finite.sum()),
        quantiles={str(q): float(np.percentile(valid, q)) if valid.size else None
                   for q in (0, 1, 50, 99, 100)},
        exact_zero_count=int((value == 0).sum()), exact_one_count=int((value == 1).sum()),
        historical_threshold=threshold, pass_count=int((finite & (value >= threshold)).sum()))


def raw_teacher_scores(teacher):
    """Small full/prefix score distributions from ACTUAL saved teacher outputs."""
    if teacher is None:
        return dict(available=False, reason='teacher_scores_not_supplied_or_preserved')
    result = dict(available=True, source='actual_full_and_independent_prefix_teacher',
                  threshold_note='Historical qualification values, not newly calibrated cutoffs.')
    for context in ('full', 'prefix'):
        value = teacher[context]
        visibility, confidence = _np(value['visibility']), _np(value['track_confidence'])
        if visibility.ndim != 2 or visibility.shape != confidence.shape:
            raise ValueError('Teacher scores must have matching [T,Q] shape')
        result[context] = dict(frames=int(visibility.shape[0]), queries=int(visibility.shape[1]),
            per_frame=[dict(frame_index=i,
                visibility=_score_statistics(visibility[i], .5),
                track_confidence=_score_statistics(confidence[i], .2))
                for i in range(len(visibility))])
    return result


def _positive_tracks(coordinates, weights):
    return (np.isfinite(coordinates).all(-1) & (coordinates >= 0).all(-1)
            & (coordinates <= 1).all(-1) & np.isfinite(weights) & (weights > 0))


def assess_controls(controls, *, split=None, window_key=None, teacher=None):
    """Return JSON counts from a current or legacy prepared control dictionary.

Coordinates are normalized exactly as consumed by transport_prefix_texture.
Reference eligibility means a valid observed source and future destination for
the same query; it does not inspect future appearance or assume track accuracy.
"""
    if not isinstance(controls, Mapping):
        raise ValueError('Prepared controls dictionary is required for readiness')
    xy, weight = _np(controls['tracks']), _np(controls['track_weight'])
    pxy, pweight = _np(controls['prefix_tracks']), _np(controls['prefix_track_weight'])
    motion, rays = _np(controls['motion']), _np(controls['rays'])
    if xy.ndim != 3 or xy.shape[-1] != 2 or weight.shape != xy.shape[:-1]:
        raise ValueError('Control tracks must be [T,Q,2] with [T,Q] weights')
    if pxy.ndim != 3 or pxy.shape[-1] != 2 or pweight.shape != pxy.shape[:-1]:
        raise ValueError('Prefix tracks must be [H,Q,2] with [H,Q] weights')
    t, q = weight.shape
    h = pweight.shape[0]
    if not 0 < h < t or pweight.shape[1] != q:
        raise ValueError('Control prefix length/query count differs from full window')
    if (motion.ndim != 3 or rays.shape != motion.shape or motion.shape[0] != t
            or motion.shape[-1] != 8):
        raise ValueError('Expected matching camera/motion maps [T,N,8]')
    full = _positive_tracks(xy, weight)
    prefix = _positive_tracks(pxy, pweight)
    future = full[h:]
    source = prefix.any(axis=0)
    eligible = future & source[None]
    motion_support = np.isfinite(motion[..., 2]) & (motion[..., 2] > 0)
    camera_present = np.isfinite(rays).all(-1) & (rays[..., -1] > 0)
    stored_scores = controls.get('metadata', {}).get('raw_teacher_scores')
    scores = (raw_teacher_scores(teacher) if teacher is not None else
              stored_scores if isinstance(stored_scores, Mapping) else raw_teacher_scores(None))
    return dict(schema=SCHEMA, split=split, window_key=window_key,
        frames=t, history=h, queries=q,
        track_positive=int(full.sum()), track_total=int(full.size),
        prefix_track_positive=int(prefix.sum()), prefix_track_total=int(prefix.size),
        future_track_positive=int(future.sum()), future_track_total=int(future.size),
        future_motion_supported_tokens=int(motion_support[h:].sum()),
        future_motion_total_tokens=int(motion_support[h:].size),
        reference_source_queries=int(source.sum()),
        reference_eligible_pairs=int(eligible.sum()), reference_total_pairs=int(eligible.size),
        future_camera_present_tokens=int(camera_present[h:].sum()),
        future_camera_total_tokens=int(camera_present[h:].size),
        camera_all_finite=bool(np.isfinite(rays).all()),
        future_motion_present=bool(future.any() and motion_support[h:].any()),
        reference_eligible=bool(eligible.any()),
        raw_teacher_scores=scores,
        note='Support only, not track accuracy, calibrated confidence or generation quality.')


def scan_prepared_payload(payload, *, split, history=None):
    """Scan old/new shard tensors without needing their original teacher files.

No metadata field is required in legacy samples. Missing/malformed controls are
reported as errors rather than fabricating zero-valued control measurements.
"""
    if split not in ('train', 'eval'):
        raise ValueError('Readiness split must be train or eval')
    records = []
    for sample in payload.get('samples', []):
        candidate = sample.get('controls')
        metadata = candidate.get('metadata', {}) if isinstance(candidate, Mapping) else {}
        key = metadata.get('window_key')
        if key is None:
            key = sample.get('window_key')
        try:
            row = assess_controls(sample.get('controls'), split=split, window_key=key)
            if history is not None and row['history'] != history:
                raise ValueError('Prepared history differs from requested readiness scan')
            row.update(sample_id=sample.get('id'), frame_ids=sample.get('frame_ids'))
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            row = dict(schema=SCHEMA, split=split, window_key=key,
                       sample_id=sample.get('id'), error=type(exc).__name__+': '+str(exc))
        records.append(row)
    return records


def summarize(records):
    """Aggregate train/eval separately; allow a subset of zero-support windows."""
    records = list(records)
    result = dict(schema=SCHEMA, total_records=len(records), splits={},
        policy='No per-window quality gate. Dataset-wide absent conditions invalidate that comparison.',
        note='NONZERO_SUPPORT is necessary, not sufficient for correct tracking or useful generation.')
    counts = ('track_positive', 'track_total', 'prefix_track_positive', 'prefix_track_total',
              'future_track_positive', 'future_track_total', 'future_motion_supported_tokens',
              'future_motion_total_tokens', 'reference_source_queries', 'reference_eligible_pairs',
              'reference_total_pairs', 'future_camera_present_tokens', 'future_camera_total_tokens')
    for split in ('train', 'eval'):
        selected = [r for r in records if r.get('split') == split]
        valid = [r for r in selected if not r.get('error')]
        errors = [r for r in selected if r.get('error')]
        row = dict(windows=len(selected), assessed_windows=len(valid), invalid_records=len(errors),
            error_examples=[dict(window_key=r.get('window_key'), error=r['error']) for r in errors[:8]],
            zero_motion_windows=sum(not r['future_motion_present'] for r in valid),
            zero_reference_windows=sum(not r['reference_eligible'] for r in valid),
            raw_teacher_score_windows=sum(bool(r.get('raw_teacher_scores', {}).get('available')) for r in valid),
            **{key:sum(int(r[key]) for r in valid) for key in counts})
        def status(nonzero):
            if errors:
                return 'UNAVAILABLE_INVALID_RECORDS'
            if not valid:
                return 'UNAVAILABLE_EMPTY_SPLIT'
            return 'NONZERO_SUPPORT' if nonzero else 'INVALID_ZERO_SUPPORT'
        row['motion_status'] = status(any(r['future_motion_present'] for r in valid))
        row['reference_status'] = status(any(r['reference_eligible'] for r in valid))
        row['camera_status'] = status(any(r['future_camera_present_tokens'] > 0 and r['camera_all_finite'] for r in valid))
        result['splits'][split] = row
    unknown = [r.get('split') for r in records if r.get('split') not in ('train', 'eval')]
    if unknown:
        raise ValueError('Every readiness record must explicitly identify train/eval split')
    return result
