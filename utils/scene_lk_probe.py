"""Independent CPU tracking evidence for diagnosis, never a training default.

The supplied teacher's depth/cameras are retained verbatim. Only trajectories
and their qualification scores are replaced. LK cannot label semantic objects,
resolve long occlusion, or prove geometric correctness; downstream rigidity
must still use the retained depth and cameras. Scores here are documented
engineering quality scores, not calibrated visibility probabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
import torch


@dataclass(frozen=True)
class LKProbeConfig:
    window_size: int = 21
    pyramid_levels: int = 3
    patch_radius: int = 5
    fb_max_pixels: float = 1.0
    photometric_max: float = 0.08
    min_eigenvalue: float = 1e-4
    min_patch_std: float = 5.0 / 255.0


def _array(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _summary(values):
    values = np.asarray(values, np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return dict(count=int(values.size),
                minimum=float(values.min()) if values.size else None,
                median=float(np.median(values)) if values.size else None,
                maximum=float(values.max()) if values.size else None)


def _inside(points, height, width, margin):
    return (np.isfinite(points).all(-1) & (points[:, 0] >= margin)
            & (points[:, 0] <= width - 1 - margin)
            & (points[:, 1] >= margin) & (points[:, 1] <= height - 1 - margin))


def _patch(cv2, gray, point, radius):
    return cv2.getRectSubPix(gray, (2 * radius + 1, 2 * radius + 1),
                            (float(point[0]), float(point[1]))).astype(np.float32) / 255.0


def _centered_patch_error(cv2, previous_gray, current_gray, previous_point, current_point, radius):
    previous = _patch(cv2, previous_gray, previous_point, radius)
    current = _patch(cv2, current_gray, current_point, radius)
    return float(np.mean(np.abs((previous - previous.mean()) - (current - current.mean()))))


def probe_lk(frames, original_teacher, *, config=None):
    """Return ``(teacher_outputs, JSON_report)`` for one full OR prefix pass.

    Input is CPU floating RGB [T,3,H,W] in [0,1]. Tracking starts at the
    original frame-zero queries; failure permanently makes that query unknown.
    The function never changes its input or the existing teacher implementation.
    Missing OpenCV is an explicit ImportError, not fabricated empty predictions.
    """
    import cv2

    started = time.perf_counter()
    config = config or LKProbeConfig()
    if (config.window_size < 3 or config.window_size % 2 != 1
            or config.pyramid_levels < 0 or config.patch_radius < 1
            or min(config.fb_max_pixels, config.photometric_max,
                   config.min_eigenvalue, config.min_patch_std) <= 0):
        raise ValueError('Invalid LK diagnostic configuration')
    if torch.is_tensor(frames) and frames.device.type != 'cpu':
        raise ValueError('LK diagnosis requires CPU RGB frames')
    if torch.is_tensor(frames) and not frames.is_floating_point():
        raise ValueError('Expected finite floating RGB [T,3,H,W] in [0,1], T>=2')
    rgb = _array(frames)
    if (rgb.ndim != 4 or rgb.shape[1] != 3 or len(rgb) < 2
            or not np.issubdtype(rgb.dtype, np.floating)
            or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1):
        raise ValueError('Expected finite floating RGB [T,3,H,W] in [0,1], T>=2')
    t, _, height, width = rgb.shape
    if min(height, width) <= 2 * config.patch_radius + 1:
        raise ValueError('Images are too small for the diagnostic patch')
    if int(original_teacher.get('reference_frame', 0)) != 0:
        raise ValueError('LK diagnostic only supports original reference_frame=0')
    if list(original_teacher.get('image_size_hw', [height, width])) != [height, width]:
        raise ValueError('Teacher and RGB image sizes disagree')
    queries = _array(original_teacher['query_points']).astype(np.float32, copy=True)
    if queries.ndim != 2 or queries.shape[1] != 2 or not len(queries) or not np.isfinite(queries).all():
        raise ValueError('Expected nonempty finite original query_points [Q,2]')
    required_shapes = {'tracks': (t, len(queries), 2), 'depth': (t, height, width),
                       'depth_conf': (t, height, width),
                       'extrinsics': (t, 3, 4), 'intrinsics': (t, 3, 3)}
    for name, shape in required_shapes.items():
        if tuple(original_teacher[name].shape) != shape:
            raise ValueError(f'Teacher {name} does not match the supplied RGB window')

    gray = [cv2.cvtColor(np.rint(frame.transpose(1, 2, 0) * 255).astype(np.uint8),
                         cv2.COLOR_RGB2GRAY) for frame in rgb]
    eigen = [cv2.cornerMinEigenVal(g.astype(np.float32) / 255., blockSize=7, ksize=3)
             for g in gray]
    tracks = np.broadcast_to(queries, (t, len(queries), 2)).copy()
    visibility = np.zeros((t, len(queries)), np.float32)
    confidence = np.zeros_like(visibility)
    alive = _inside(queries, height, width, config.patch_radius)
    initial_inside = int(alive.sum())

    def texture(frame_index, points):
        eig, std = [], []
        for point in points:
            eig.append(float(cv2.getRectSubPix(eigen[frame_index], (1, 1), tuple(map(float, point)))[0, 0]))
            std.append(float(_patch(cv2, gray[frame_index], point, config.patch_radius).std()))
        return np.asarray(eig), np.asarray(std)

    ids = np.flatnonzero(alive)
    eig, std = texture(0, queries[ids])
    textured = (eig >= config.min_eigenvalue) & (std >= config.min_patch_std)
    alive[ids] = textured
    ids = ids[textured]
    visibility[0, ids] = 1.0
    confidence[0, ids] = eig[textured] / (eig[textured] + 4 * config.min_eigenvalue)
    records = [dict(frame_index=0, input_queries=len(queries), inside_patch=initial_inside,
                    connected_queries=int(alive.sum()), rejected_texture=int((~textured).sum()),
                    eigenvalue=_summary(eig), patch_std=_summary(std))]
    errors = []
    for frame_index in range(1, t):
        tracks[frame_index] = tracks[frame_index - 1]
        ids = np.flatnonzero(alive)
        record = dict(frame_index=frame_index, input_connected_queries=int(len(ids)),
                      forward_status=0, backward_status=0, inside_patch=0,
                      forward_backward_pass=0, texture_pass=0, photometric_pass=0,
                      connected_queries=0)
        if not len(ids):
            records.append(record)
            continue
        previous = np.ascontiguousarray(tracks[frame_index - 1, ids, None], dtype=np.float32)
        params = dict(winSize=(config.window_size, config.window_size), maxLevel=config.pyramid_levels,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                      minEigThreshold=config.min_eigenvalue)
        try:
            forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                gray[frame_index - 1], gray[frame_index], previous, None, **params)
            if forward is None or forward_status is None:
                raise RuntimeError('LK forward returned no coordinates/status')
            forward = forward.reshape(-1, 2)
            status = forward_status.reshape(-1).astype(bool)
            record['forward_status'] = int(status.sum())
            inside = _inside(forward, height, width, config.patch_radius)
            candidates = np.flatnonzero(status & inside)
            record['inside_patch'] = int(len(candidates))
            selected = np.zeros(len(ids), bool)
            quality = np.zeros(len(ids), np.float32)
            if len(candidates):
                backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray[frame_index], gray[frame_index - 1],
                    np.ascontiguousarray(forward[candidates, None]), None, **params)
                if backward is None or backward_status is None:
                    raise RuntimeError('LK backward returned no coordinates/status')
                backward = backward.reshape(-1, 2)
                backward_ok = backward_status.reshape(-1).astype(bool) & np.isfinite(backward).all(-1)
                record['backward_status'] = int(backward_ok.sum())
                fb = np.linalg.norm(backward - previous[candidates, 0], axis=-1)
                fb_ok = backward_ok & (fb <= config.fb_max_pixels)
                record['forward_backward_pass'] = int(fb_ok.sum())
                eig, std = texture(frame_index, forward[candidates])
                texture_ok = (eig >= config.min_eigenvalue) & (std >= config.min_patch_std)
                record['texture_pass'] = int((fb_ok & texture_ok).sum())
                photo = np.asarray([_centered_patch_error(cv2,
                    gray[frame_index - 1], gray[frame_index], previous[index, 0], forward[index],
                    config.patch_radius) for index in candidates])
                photo_ok = np.isfinite(photo) & (photo <= config.photometric_max)
                accepted = fb_ok & texture_ok & photo_ok
                record['photometric_pass'] = int(accepted.sum())
                selected[candidates] = accepted
                q = np.minimum.reduce((np.exp(-(fb / config.fb_max_pixels) ** 2),
                    np.exp(-(photo / config.photometric_max) ** 2),
                    eig / (eig + 4 * config.min_eigenvalue)))
                quality[candidates[accepted]] = q[accepted]
                record.update(fb_error_pixels=_summary(fb), centered_patch_mae=_summary(photo),
                              eigenvalue=_summary(eig), patch_std=_summary(std))
            alive[ids] = selected
            good_ids = ids[selected]
            tracks[frame_index, good_ids] = forward[selected]
            visibility[frame_index, good_ids] = 1.0
            confidence[frame_index, good_ids] = np.minimum(
                confidence[frame_index - 1, good_ids], quality[selected])
            record['connected_queries'] = int(len(good_ids))
        except (cv2.error, RuntimeError) as exc:
            alive[ids] = False
            errors.append(dict(frame_index=frame_index, error_type=type(exc).__name__, message=str(exc)))
            record['error'] = str(exc)
        records.append(record)

    outputs = dict(original_teacher)
    source_metadata = dict(original_teacher.get('teacher', {}))
    outputs.update(tracks=torch.from_numpy(tracks), visibility=torch.from_numpy(visibility),
                   track_confidence=torch.from_numpy(confidence),
                   teacher={**source_metadata, 'backend': 'diagnostic_lk',
                            'geometry_backend': source_metadata.get('backend'),
                            'tracking_source': 'observed_rgb_sequential_pyramidal_lk',
                            'diagnostic_only': True, 'tracking_config': asdict(config)})
    report = dict(backend='diagnostic_lk', diagnostic_only=True, opencv_version=cv2.__version__,
        status='completed' if not errors else 'completed_with_tracking_errors', config=asdict(config),
        frames=t, queries=len(queries), image_size_hw=[height, width],
        per_frame=records, errors=errors, elapsed_seconds=time.perf_counter() - started,
        score_semantics='visibility is hard qualified connectivity; confidence is cumulative minimum of texture, FB and centered-patch quality, not calibrated probability',
        failure_semantics='failed queries stay unknown, scores zero, last finite position retained; no re-identification',
        geometry_semantics='original depth/depth_conf/extrinsics/intrinsics retained verbatim; use existing camera-compensated rigidity downstream, never pixel-motion magnitude alone',
        scope='diagnostic alternative only; full-window future tracks are oracle conditions; no semantic labels or demonstrated generation-quality claim',
        qualified_fraction=float(((visibility >= .5) & (confidence >= .2)).mean()),
        confidence=_summary(confidence[visibility > 0]))
    return outputs, report
