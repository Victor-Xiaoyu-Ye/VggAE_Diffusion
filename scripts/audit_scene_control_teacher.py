#!/usr/bin/env python3
"""Read saved stage56 teachers on CPU; never rerun or alter training/controls.

Only load trusted project-produced torch files. A sibling .pt.json receipt is
checked when nonempty; missing/empty receipts are explicitly unverified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


VISIBILITY_MIN = 0.5
CONFIDENCE_MIN = 0.2


def receipt_status(path):
    path = Path(path)
    receipt_path = Path(str(path) + '.json')
    if not path.is_file():
        raise FileNotFoundError(path)
    if not receipt_path.exists() or receipt_path.stat().st_size == 0:
        return dict(verified=False, status='empty_receipt' if receipt_path.exists()
                    else 'missing_receipt', receipt=str(receipt_path))
    try:
        receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
        expected_size, expected_sha = receipt['size'], receipt['sha256']
        if type(expected_size) is not int or expected_size < 0:
            raise ValueError('Invalid receipt size')
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64
                or any(c not in '0123456789abcdefABCDEF' for c in expected_sha)):
            raise ValueError('Invalid receipt SHA256')
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f'Invalid nonempty receipt: {receipt_path}') from exc
    if path.stat().st_size != expected_size:
        raise ValueError(f'Receipt size mismatch: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected_sha.lower():
        raise ValueError(f'Receipt SHA256 mismatch: {path}')
    return dict(verified=True, status='size_and_sha256_verified',
                receipt=str(receipt_path), sha256=digest.hexdigest(),
                identity=receipt.get('identity'))


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def statistics(value):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = np.isfinite(value)
    selected = value[finite]
    result = dict(count=int(value.size), finite_count=int(finite.sum()),
                  finite_fraction=float(finite.mean()) if value.size else None)
    result['quantiles'] = {str(q): float(np.percentile(selected, q)) if selected.size else None
                           for q in (0, 1, 5, 50, 95, 99, 100)}
    result['exact_zero_fraction'] = float((value == 0).mean()) if value.size else None
    result['exact_one_fraction'] = float((value == 1).mean()) if value.size else None
    result['outside_probability_range_count'] = int((finite & ((value < 0) | (value > 1))).sum())
    return result


def inspect_context(teacher, frame_ids=None):
    tracks = array(teacher['tracks']).astype(np.float32)
    visibility = array(teacher['visibility']).astype(np.float32)
    confidence = array(teacher['track_confidence']).astype(np.float32)
    queries = array(teacher['query_points']).astype(np.float32)
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError('tracks must be [T,Q,2]')
    t, q, _ = tracks.shape
    if not t or not q or visibility.shape != (t, q) or confidence.shape != (t, q):
        raise ValueError('Expected nonempty matching [T,Q] confidence/visibility')
    if queries.shape != (q, 2):
        raise ValueError('query_points must match track query IDs')
    size = teacher.get('image_size_hw')
    if size is None:
        size = array(teacher['depth']).shape[-2:]
    if len(size) != 2 or any(int(v) != v or int(v) < 2 for v in size):
        raise ValueError('Invalid teacher image_size_hw')
    h, w = map(int, size)
    ref = int(teacher.get('reference_frame', 0))
    if not 0 <= ref < t:
        raise ValueError('Invalid reference_frame')
    if frame_ids is not None and len(frame_ids) != t:
        raise ValueError('provenance frame_ids length does not match teacher')
    coord_finite = np.isfinite(tracks).all(-1)
    inside = (coord_finite & (tracks[..., 0] >= 0) & (tracks[..., 0] <= w - 1)
              & (tracks[..., 1] >= 0) & (tracks[..., 1] <= h - 1))
    vis_ok = np.isfinite(visibility) & (visibility >= VISIBILITY_MIN)
    conf_ok = np.isfinite(confidence) & (confidence >= CONFIDENCE_MIN)
    with np.errstate(invalid='ignore', over='ignore'):
        product = visibility * confidence
    qualifies = inside & vis_ok & conf_ok & np.isfinite(product)
    weights = np.where(qualifies, np.clip(product, 0, 1), 0)
    accepted = weights > 0
    frames = []
    for i in range(t):
        frames.append(dict(frame_index=i, source_frame_id=frame_ids[i] if frame_ids else None,
            visibility=statistics(visibility[i]), track_confidence=statistics(confidence[i]),
            coordinates=dict(finite_count=int(coord_finite[i].sum()), inside_count=int(inside[i].sum()),
                x_quantiles=statistics(tracks[i, :, 0])['quantiles'],
                y_quantiles=statistics(tracks[i, :, 1])['quantiles']),
            visibility_pass_count=int(vis_ok[i].sum()), confidence_pass_count=int(conf_ok[i].sum()),
            filter_funnel=dict(total=q, finite_coordinates=int(coord_finite[i].sum()),
                inside=int(inside[i].sum()), inside_and_visibility=int((inside[i] & vis_ok[i]).sum()),
                inside_and_both_scores=int((inside[i] & vis_ok[i] & conf_ok[i]).sum()),
                positive_weight=int(accepted[i].sum())),
            score_rejection=dict(visibility_only=int((~vis_ok[i] & conf_ok[i]).sum()),
                confidence_only=int((vis_ok[i] & ~conf_ok[i]).sum()),
                both=int((~vis_ok[i] & ~conf_ok[i]).sum())),
            positive_weight_fraction=float(accepted[i].mean())))
    delta = tracks[ref] - queries
    error = np.linalg.norm(delta.astype(np.float64), axis=-1)
    query_finite = np.isfinite(queries).all(-1)
    reference_check = dict(reference_frame=ref, query_finite_count=int(query_finite.sum()),
        error_pixels=statistics(error)['quantiles'],
        matches_queries=bool(np.isfinite(error).all() and (error <= 1e-4).all()))
    sweep = [dict(visibility_min=v, confidence_min=c,
                  positive_weight_fraction=float((inside & np.isfinite(product) & (product > 0)
                      & (visibility >= v) & (confidence >= c)).mean()))
             for v in (0.1, 0.5, 0.9) for c in (0.05, 0.2, 0.5)]
    report = dict(frames=t, queries=q, image_size_hw=[h, w], per_frame=frames,
        positive_weight_count=int(accepted.sum()), positive_weight_fraction=float(accepted.mean()),
        reference_query_check=reference_check, threshold_sensitivity=sweep,
        sensitivity_note='Diagnostic only; no threshold is selected or changed.',
        teacher=teacher.get('teacher', {}))
    compact = dict(tracks=tracks, visibility=visibility, track_confidence=confidence,
                   query_points=queries, accepted=accepted)
    return report, compact


def audit_payload(payload):
    if not isinstance(payload, dict) or not all(k in payload for k in ('full', 'prefix')):
        raise ValueError('Expected stage56 payload containing full and prefix teachers')
    provenance = payload.get('provenance', {})
    frame_ids = provenance.get('frame_ids')
    history = array(payload['prefix']['tracks']).shape[0]
    full, full_values = inspect_context(payload['full'], frame_ids)
    prefix, prefix_values = inspect_context(payload['prefix'], frame_ids[:history] if frame_ids else None)
    if history >= full['frames'] or full['image_size_hw'] != prefix['image_size_hw']:
        raise ValueError('Invalid independent prefix length or image size')
    if not np.array_equal(full_values['query_points'], prefix_values['query_points']):
        raise ValueError('Full and prefix query identities differ')
    source = prefix_values['accepted'].any(axis=0)
    destination = full_values['accepted'][history:]
    usable = destination & source[None]
    intersection = dict(observed_frames=history, future_frame_count=len(destination),
        observed_source_queries=int(source.sum()), future_positive_weights=int(destination.sum()),
        source_destination_pairs=int(usable.sum()),
        source_destination_pair_fraction=float(usable.mean()),
        future_frames=[dict(frame_index=history+i,
            source_frame_id=frame_ids[history+i] if frame_ids else None,
            destination_queries=int(destination[i].sum()), usable_source_queries=int(usable[i].sum()))
            for i in range(len(destination))],
        note='Query correspondence eligibility only; this does not certify texture or motion accuracy.')
    findings = []
    if not full['positive_weight_count']:
        findings.append('FULL_TRACK_WEIGHTS_ALL_ZERO')
    if not prefix['positive_weight_count']:
        findings.append('PREFIX_TRACK_WEIGHTS_ALL_ZERO')
    if not usable.any():
        findings.append('NO_OBSERVED_TO_FUTURE_REFERENCE_SUPPORT')
    for name, report in (('full', full), ('prefix', prefix)):
        if not report['reference_query_check']['matches_queries']:
            findings.append(name.upper() + '_REFERENCE_QUERY_MISMATCH')
        for frame in report['per_frame']:
            if any(frame[key]['finite_fraction'] != 1 or frame[key]['outside_probability_range_count']
                   for key in ('visibility', 'track_confidence')):
                findings.append(name.upper() + '_INVALID_PROBABILITY_VALUES')
                break
    report = dict(window_key=payload.get('window_key'), provenance=provenance, full=full, prefix=prefix,
                  reference_eligibility=intersection, findings=findings,
                  conclusion='ANOMALY_REQUIRES_REVIEW' if findings else 'NONZERO_SUPPORT_REQUIRES_QUALITY_REVIEW')
    compact = {name+'_'+key: value for name, values in (('full', full_values), ('prefix', prefix_values))
               for key, value in values.items()}
    return report, compact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-file', action='append', required=True, help='Repeat for trusted local teacher.pt files')
    parser.add_argument('--output', required=True, help='Destination JSON report (no training files are changed)')
    parser.add_argument('--export-npz', action='store_true', help='Also save compact tracks/scores beside report')
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    inputs = [Path(p).resolve() for p in args.teacher_file]
    if output in inputs or output in [Path(str(p)+'.json') for p in inputs]:
        raise ValueError('Output must not replace an input or its receipt')
    cases, exports = [], []
    for index, path in enumerate(inputs):
        verification = receipt_status(path)
        # str is required by torch_npu's mmap-compatible loader.
        payload = torch.load(str(path), map_location='cpu', weights_only=False)
        report, compact = audit_payload(payload)
        report.update(input_file=str(path), receipt_verification=verification)
        if args.export_npz:
            export = output.with_name(output.stem + f'.case{index:03d}.npz')
            if export in inputs:
                raise ValueError('NPZ output must not replace an input')
            report['compact_npz'] = str(export)
            exports.append((export, compact))
        cases.append(report)
    result = dict(schema='scene-control-teacher-audit-v1', device='cpu',
        thresholds=dict(visibility=VISIBILITY_MIN, track_confidence=CONFIDENCE_MIN),
        note='Read-only numeric audit, not teacher accuracy validation or a training gate.', cases=cases)
    # Validate serializability before creating any output; never emit NaN JSON.
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    for export, values in exports:
        np.savez_compressed(export, **values)
    output.write_text(serialized, encoding='utf-8')
    print(json.dumps(dict(report=str(output), cases=len(cases),
        verified=sum(c['receipt_verification']['verified'] for c in cases),
        anomalies=sum(bool(c['findings']) for c in cases)), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
