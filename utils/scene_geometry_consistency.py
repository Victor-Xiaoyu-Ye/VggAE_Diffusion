"""Read-only representation causality and geometry diagnostics.

No fitting here changes a production representation or a camera prediction.
The depth scale fit is a descriptive residual decomposition, never calibration.
Real RGB tracks may belong to moving objects and are not static-scene truth.
"""
from __future__ import annotations

import numpy as np
import torch

from utils.scene_trackmask import camera_compensated_track_mask


def as_numpy(value):
    return value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def finite_summary(value):
    value = np.asarray(value, np.float64).reshape(-1)
    finite = value[np.isfinite(value)]
    return dict(count=int(value.size), finite_count=int(finite.size),
                mean=float(finite.mean()) if finite.size else None,
                rms=float(np.sqrt(np.mean(finite ** 2))) if finite.size else None,
                median=float(np.median(finite)) if finite.size else None,
                p95=float(np.quantile(finite, .95)) if finite.size else None,
                maximum=float(finite.max()) if finite.size else None)


def difference(reference, candidate):
    """Exact float64 CPU reductions; original first dimension denotes time."""
    reference, candidate = as_numpy(reference), as_numpy(candidate)
    if reference.shape != candidate.shape or reference.ndim < 2 or not reference.size:
        raise ValueError('Difference needs nonempty matching [T,...] arrays')
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    valid = np.isfinite(delta) & np.isfinite(reference)
    result = dict(shape=list(reference.shape), finite=bool(valid.all()))
    if not valid.all():
        result['nonfinite_count'] = int((~valid).sum())
        return result
    ref_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    rms = float(np.sqrt(np.mean(delta ** 2)))
    result.update(l1=float(np.abs(delta).mean()), rms=rms, max_abs=float(np.abs(delta).max()),
                  reference_rms=ref_rms, relative_rms=rms / max(ref_rms, 1e-12),
                  per_frame_rms=np.sqrt(np.mean(delta.reshape(len(delta), -1) ** 2, axis=1)).tolist())
    return result


def latent_difference(reference, candidate, mean, std):
    reference = torch.as_tensor(reference).float().cpu()
    candidate = torch.as_tensor(candidate).float().cpu()
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[-1] % 2:
        raise ValueError('Latents must be matching [T,N,even_channels]')
    mu = torch.as_tensor(mean).float().reshape(-1)
    sigma = torch.as_tensor(std).float().reshape(-1).clamp_min(1e-4)
    if mu.numel() != reference.shape[-1] or sigma.shape != mu.shape:
        raise ValueError('Use original source per-channel normalization')
    half = reference.shape[-1] // 2
    normalized_reference = (reference - mu) / sigma
    normalized_candidate = (candidate - mu) / sigma
    report, maps = {}, {}
    for name, section in [('joint', slice(None)), ('geo', slice(0, half)), ('texture', slice(half, None))]:
        report[name] = dict(raw=difference(reference[..., section], candidate[..., section]),
            source_standardized=difference(normalized_reference[..., section], normalized_candidate[..., section]))
        maps[name] = (normalized_candidate[..., section] - normalized_reference[..., section]).abs().mean(-1)
    return report, maps


def future_intervention(frames, history=4):
    """Keep the observed prefix byte-for-byte; invert and reverse future RGB."""
    frames = torch.as_tensor(frames)
    if frames.ndim != 4 or not 1 <= history < len(frames):
        raise ValueError('Expected [T,3,H,W] with a nonempty future')
    changed = frames.clone()
    changed[history:] = 1. - frames[history:].flip(0)
    return changed


@torch.no_grad()
def representation_probe(frames, sample, mean, std, encode, decode, *, history=4, on_phase=None):
    """Use caller's production encode/decode; return report and compact tensors.

    ``encode`` returns CPU ``latent[T,N,C]`` and selected-layer ``tokens``
    containing only the common prefix. Tokens are measured, not dumped to disk.
    ``decode`` receives unnormalized latents and returns RGB[T,H,W,3].
    ``on_phase`` is called after every completed forward with compact latents.
    Cached targets remain unchanged, including mismatching cases.
    """
    frames = torch.as_tensor(frames).float().cpu()
    changed = future_intervention(frames, history)
    if not torch.equal(frames[:history], changed[:history]):
        raise AssertionError('Future intervention changed observations')
    variants = [('full', frames), ('prefix', frames[:history]),
                ('future_intervened', changed), ('anchor', frames[:1])]
    encoded, phase_input_frames = {}, {}
    for name, rgb in variants:
        value = encode(rgb)
        latent = torch.as_tensor(value['latent']).detach().float().cpu()
        if latent.ndim != 3 or len(latent) != len(rgb) or not torch.isfinite(latent).all():
            raise ValueError(f'{name}: invalid latent returned by production encoder')
        encoded[name] = dict(latent=latent, tokens=value.get('tokens', {}))
        phase_input_frames[name] = len(rgb)
        if on_phase:
            on_phase(name, dict(latent=latent), dict(input_frames=len(rgb), latent_shape=list(latent.shape)))

    anchor = encoded['anchor']['latent']
    full_raw = encoded['full']['latent']
    prefix_raw = encoded['prefix']['latent']
    altered_raw = encoded['future_intervened']['latent']
    full = torch.cat((anchor, full_raw[1:]), 0)
    prefix = torch.cat((anchor, prefix_raw[1:]), 0)
    altered = torch.cat((anchor, altered_raw[1:]), 0)
    cached = torch.as_tensor(sample['target']).float().cpu()
    cached_anchor = torch.as_tensor(sample.get('anchor', cached[:1])).float().cpu()
    if cached.shape != full.shape or cached_anchor.shape != anchor.shape:
        raise ValueError('Original cached representation layout differs from re-encoding')
    comparisons = {
        'full_vs_independent_prefix_raw': (full_raw[:history], prefix_raw),
        'full_vs_changed_future_raw_prefix': (full_raw[:history], altered_raw[:history]),
        'full_vs_independent_prefix_production_anchor': (full[:history], prefix),
        'cached_vs_reencoded_full': (cached, full),
        'cached_vs_reencoded_bfloat16_storage': (cached, full.bfloat16().float()),
        'cached_anchor_vs_reencoded_anchor': (cached_anchor, anchor),
        'reencoded_storage_quantization_only': (full, full.bfloat16().float()),
    }
    reports, maps = {}, {}
    for name, (a, b) in comparisons.items():
        reports[name], maps[name] = latent_difference(a, b, mean, std)
    token_reports = {}
    for name in ('prefix', 'future_intervened'):
        current = encoded[name]['tokens']; reference = encoded['full']['tokens']
        if set(current) != set(reference):
            raise ValueError('Selected token layers differ between contexts')
        token_reports[name] = {str(layer): difference(reference[layer][:history], current[layer][:history])
                               for layer in reference}
    if on_phase:
        on_phase('representation_comparisons', dict(comparisons=reports,
            common_prefix_tokens=token_reports, difference_maps=maps), dict(input_frames=0))
    # Keep only compact bottlenecks; the much larger intermediate token arrays
    # are released before RGB reconstruction.
    encoded.clear()
    artifacts = dict(full_raw=full_raw, prefix_raw=prefix_raw, future_intervened_raw=altered_raw,
        independent_anchor=anchor, full_production=full, cached_target=cached,
        cached_anchor=cached_anchor, difference_maps=maps)
    reconstructions = {}
    reconstruction_scores = {}
    for name, latent in [('cached', cached), ('reencoded_full', full),
                          ('independent_prefix', prefix), ('changed_future_prefix', altered[:history])]:
        rgb = torch.as_tensor(decode(latent)).detach().float().cpu()
        truth = frames[:len(rgb)].permute(0, 2, 3, 1)
        if rgb.shape != truth.shape or not torch.isfinite(rgb).all():
            raise ValueError(f'{name}: invalid reconstruction output')
        reconstructions[name] = rgb
        metric = difference(truth, rgb)
        mse = metric['rms'] ** 2
        metric['psnr'] = float(-10 * np.log10(max(mse, 1e-12)))
        metric['psnr_uses_unclipped_decoder_output'] = True
        reconstruction_scores[name] = metric
    artifacts['reconstructions'] = reconstructions
    report = dict(schema='scene-representation-consistency-v1', comparisons=reports,
        common_prefix_tokens=token_reports, reconstruction=reconstruction_scores,
        history=history, unique_sealed_RGB_frames=len(frames), encoder_input_frames=sum(phase_input_frames.values()),
        encoder_calls=phase_input_frames, decoder_frame_evaluations=sum(len(v) for v in reconstructions.values()),
        future_intervention='invert_and_reverse_future_RGB_only; synthetic_causality_test',
        observed_prefix_exactly_preserved=True,
        cache_policy='Original cache remains the reference; re-encoding is a separate diagnostic, never a replacement.',
        input_precision='Saved uint8 raw converted to float32 /255. Original pre-save floating input is not retained; quantization/numerical differences are reported, not assumed to be corruption.',
        inference_dtype='Production encoder and AE use their existing autocast policy; source cache stores BF16.',
        quality_status='measured_not_certified')
    return report, artifacts


def single_scale_depth(reference, candidate, valid=None):
    """Fit ONE positive least-squares scalar for the entire supplied volume.

    No offset, per-frame scale, spatial warping, depth clipping or camera update.
    Fit uses candidate observations and is not an inference calibration method.
    """
    a, b = as_numpy(reference).astype(np.float64), as_numpy(candidate).astype(np.float64)
    if a.shape != b.shape or a.ndim < 2:
        raise ValueError('Depth alignment requires matching volumes')
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if valid is not None:
        if np.shape(valid) != a.shape:
            raise ValueError('Depth support shape differs')
        mask &= np.asarray(valid, bool)
    if not mask.any():
        return dict(status='no_positive_depth_support', valid_pixels=0), None
    x, y = a[mask], b[mask]
    scale = float(np.dot(x, y) / np.dot(y, y))
    raw, fitted = y - x, scale * y - x
    denominator = max(float(np.sqrt(np.mean(x ** 2))), 1e-12)
    residual = np.full(a.shape, np.nan, np.float32)
    residual[mask] = fitted
    report = dict(status='measured', valid_pixels=int(mask.sum()), total_pixels=int(mask.size),
        single_candidate_to_reference_scale=scale,
        raw_rms=float(np.sqrt(np.mean(raw ** 2))), aligned_rms=float(np.sqrt(np.mean(fitted ** 2))),
        raw_relative_rms=float(np.sqrt(np.mean(raw ** 2))) / denominator,
        aligned_relative_rms=float(np.sqrt(np.mean(fitted ** 2))) / denominator,
        aligned_absolute_relative_error=float(np.mean(np.abs(fitted) / x)),
        policy='One scalar over the whole compared volume; descriptive fit only; does not change teacher depth or cameras.')
    return report, residual


def observed_camera_gauge(outputs, prefix):
    """Use one independent observed-prefix scale and first-camera gauge."""
    depth = as_numpy(prefix['depth']).astype(np.float64)
    good = np.isfinite(depth) & (depth > 0)
    if not good.any():
        raise ValueError('Independent prefix has no positive depth scale')
    scale = float(np.median(depth[good]))
    reference = np.eye(4); reference[:3] = as_numpy(prefix['extrinsics'])[0]
    extrinsics = as_numpy(outputs['extrinsics']).astype(np.float64)
    matrices = np.repeat(np.eye(4)[None], len(extrinsics), 0); matrices[:, :3] = extrinsics
    relative = reference[None] @ np.linalg.inv(matrices)
    relative[:, :3, 3] /= scale
    return relative, scale


def geometry_probe(full, prefix, tracked, *, repeated=False):
    """Measure LK-to-rigid residuals without calling moving points 'errors'."""
    tracks = as_numpy(tracked['tracks'])
    depth = as_numpy(full['depth'])
    if not np.array_equal(as_numpy(full['query_points']), as_numpy(tracked['query_points'])):
        raise ValueError('LK and geometry query identities differ')
    if not np.array_equal(as_numpy(full['query_points']), as_numpy(prefix['query_points'])):
        raise ValueError('Full and independent-prefix query identities differ')
    gauge, scale = observed_camera_gauge(full, prefix)
    prefix_gauge, _ = observed_camera_gauge(prefix, prefix)
    hist = len(prefix_gauge)
    common = dict(depth=difference(as_numpy(prefix['depth']), depth[:hist]),
                  camera_in_fixed_observed_gauge=difference(prefix_gauge, gauge[:hist]),
                  intrinsics=difference(as_numpy(prefix['intrinsics']), as_numpy(full['intrinsics'])[:hist]))
    evidence = camera_compensated_track_mask(tracks=tracks, depth=depth,
        world_to_camera=as_numpy(full['extrinsics']), intrinsics=as_numpy(full['intrinsics']),
        visibility=as_numpy(tracked['visibility']), track_confidence=as_numpy(tracked['track_confidence']),
        depth_confidence=as_numpy(full['depth_conf']))
    h, w = depth.shape[-2:]
    qualified = ((as_numpy(tracked['visibility']) >= .5) & (as_numpy(tracked['track_confidence']) >= .2)
        & np.isfinite(tracks).all(-1) & (tracks[..., 0] >= 0) & (tracks[..., 0] <= w-1)
        & (tracks[..., 1] >= 0) & (tracks[..., 1] <= h-1))
    queries = as_numpy(tracked['query_points'])
    per_frame = []
    for i in range(len(tracks)):
        support = qualified[i]
        per_frame.append(dict(frame=i, lk_qualified=int(support.sum()), total_queries=len(queries),
            geometry_qualified=int(evidence['valid'][i].sum()), reliable_geometry_comparison=bool(evidence['frame_reliable'][i]),
            observed_track_displacement_px=finite_summary(np.linalg.norm(tracks[i]-queries, axis=-1)[support]),
            rigid_residual_on_lk_support_px=finite_summary(evidence['residual_px'][i, support]),
            reverse_rigid_residual_on_lk_support_px=finite_summary(evidence['reverse_residual_px'][i, support]),
            reason_bit_counts={str(b):int(((evidence['reason_bits'][i]&b)>0).sum()) for b in [1,2,4,8,16,32,64,128]}))
    static_depth = None
    if repeated:
        reference = np.broadcast_to(depth[:1], depth[1:].shape)
        static_depth, _ = single_scale_depth(reference, depth[1:])
        static_depth['reference'] = 'Original first-frame depth; one scalar for all remaining repeated-frame depths.'
        static_depth['raw_observed_scale_normalized_rms'] = static_depth.get('raw_rms', 0) / scale
    compact = {name:torch.from_numpy(evidence[name]) for name in
        ('rigid_tracks','residual_px','reverse_residual_px','valid','labels','track_labels','reason_bits',
         'frame_reliable','thresholds_px','depth_allowance_px')}
    compact['camera_observed_gauge'] = torch.from_numpy(gauge)
    report = dict(schema='scene-geometry-consistency-v1', repeated_input=bool(repeated),
        observed_prefix_depth_scale=scale, gauge_policy='Independent prefix first camera and median positive prefix depth; same gauge for all compared frames.',
        common_prefix=common, per_frame=per_frame, static_single_scale_depth=static_depth,
        motion_diagnostics=evidence['diagnostics'],
        interpretation='Real LK tracks are correspondence evidence, not static ground truth; residuals combine object motion and camera/depth/tracking error. Only repeated RGB is a known zero-image-motion test.',
        calibration_applied=False, quality_status='measured_not_certified')
    return report, compact
