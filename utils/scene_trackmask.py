"""Conservative *offline diagnostic* of motion unexplained by a rigid scene.

This is not a motion ground truth or an inference-time condition. Labels use
multiple frames (including future frames), and camera/depth/track errors may be
correlated when all three come from the same teacher. Forward/reverse agreement
rejects some one-sided depth errors, but not mutually consistent depth errors.
Moving-majority scenes
and globally inconsistent cameras deliberately produce UNKNOWN, not a full
foreground mask. Sparse tracks cannot certify unqueried pixels as static.

Camera convention is OpenCV world-to-camera; depth is camera-space z, not ray
length. Confidence cutoffs are configurable heuristics, NOT calibrated
probabilities. Save the raw residuals and threshold sweep when auditing data.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TrackMaskConfig:
    # Pixel thresholds scale with max(H, W), relative to a 518px teacher input.
    base_threshold_px: float = 2.0
    robust_sigma: float = 3.0
    visibility_min: float = 0.5
    track_confidence_min: float = 0.2
    depth_confidence_min: float | None = None
    depth_confidence_quantile: float = 0.05
    depth_edge_relative: float = 0.2
    depth_uncertainty_relative: float = 0.05
    check_occlusion: bool = True
    occlusion_relative: float = 0.1
    min_background_tracks: int = 8
    global_median_factor: float = 4.0
    static_support_factor: float = 2.0
    min_static_support: float = 0.35
    min_dynamic_frames: int = 2
    min_dynamic_fraction: float = 0.5


def _sample(image, pixels):
    """Bilinear sample, bounds/finite support and local relative depth range."""
    height, width = image.shape
    x, y = pixels.T
    inside = np.isfinite(pixels).all(-1) & (x >= 0) & (x <= width-1) & (y >= 0) & (y <= height-1)
    xx = np.clip(np.nan_to_num(x, nan=0., posinf=0., neginf=0.), 0, width-1)
    yy = np.clip(np.nan_to_num(y, nan=0., posinf=0., neginf=0.), 0, height-1)
    x0, y0 = np.floor(xx).astype(np.int64), np.floor(yy).astype(np.int64)
    x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
    corners = np.stack((image[y0, x0], image[y0, x1], image[y1, x0], image[y1, x1]))
    finite = np.isfinite(corners).all(0)
    dx, dy = xx-x0, yy-y0
    value = (corners[0]*(1-dx)*(1-dy) + corners[1]*dx*(1-dy)
             + corners[2]*(1-dx)*dy + corners[3]*dx*dy)
    spread = (corners.max(0)-corners.min(0))/np.maximum(np.abs(value), 1e-12)
    return value, inside & finite, spread


def _project(points, rotation, translation, intrinsic):
    camera = points @ rotation.T + translation
    homogeneous = camera @ intrinsic.T
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(camera).all(-1) & (camera[:, 2] > 0)
    np.divide(homogeneous[:, :2], homogeneous[:, 2:3], out=pixels, where=valid[:, None])
    return pixels, camera[:, 2]


def camera_compensated_track_mask(*, tracks, depth, world_to_camera, intrinsics,
                                  visibility=None, track_confidence=None,
                                  depth_confidence=None, reference_frame=0,
                                  config=None):
    """Return sparse static/dynamic/unknown evidence with auditable intermediates.

    Args:
        tracks: [T,N,2] tracked pixel coordinates in the depth image's resolution.
        depth: [T,H,W] camera-z depth, in the same scale as camera translation.
        world_to_camera: [T,3,4] or [T,4,4] OpenCV extrinsics.
        intrinsics: [T,3,3], matching pixels *after* all resize/crop operations.
        visibility, track_confidence: optional [T,N] teacher outputs.
        depth_confidence: optional [T,H,W]; relative quantile filtering is per
            image because teacher scores are not assumed to be probabilities.
        reference_frame: all tracks must refer to the same physical query points
            visible in this frame. Query multiple reference frames for entrants.

    Returns a dict. Array ``labels[T,N]`` and ``track_labels[N]`` encode -1
    UNKNOWN, 0 STATIC, 1 DYNAMIC. The reference observation itself has no motion
    evidence and remains UNKNOWN. ``dynamic_score`` is a heuristic soft score
    (NaN for invalid observations), not a probability. ``persistent_dynamic``
    and track_labels use the full supplied window. A point with a single large
    residual remains unknown rather than becoming a persistent dynamic object.

    ``reason_bits``: 1 invalid coordinates, 2 low visibility/track confidence,
    4 invalid/low-confidence depth, 8 depth edge, 16 predicted occlusion,
    32 unreliable frame/global camera inconsistency, 64 insufficient or
    nonpersistent temporal evidence, 128 forward/reverse motion disagreement.
    More than one reason can apply, including occlusion and disagreement.
    """
    cfg = config or TrackMaskConfig()
    if not isinstance(cfg, TrackMaskConfig):
        raise TypeError('config must be a TrackMaskConfig')
    if not (0 <= cfg.depth_uncertainty_relative < 1 and 0 <= cfg.depth_confidence_quantile < 1):
        raise ValueError('relative depth uncertainty and confidence quantile must be in [0,1)')
    if cfg.min_dynamic_frames < 1 or cfg.min_background_tracks < 1 or cfg.base_threshold_px <= 0:
        raise ValueError('frame/track counts and base pixel threshold must be positive')
    if not (0 < cfg.min_dynamic_fraction <= 1 and 0 <= cfg.min_static_support <= 1):
        raise ValueError('support fractions must be in [0,1] (dynamic fraction > 0)')
    tracks = np.asarray(tracks, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    w2c = np.asarray(world_to_camera, dtype=np.float64)
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    if tracks.ndim != 3 or tracks.shape[-1] != 2 or min(tracks.shape[:2]) < 1:
        raise ValueError('tracks must be nonempty [T,N,2]')
    frames, points = tracks.shape[:2]
    if depth.ndim != 3 or depth.shape[0] != frames or min(depth.shape[1:]) < 2:
        raise ValueError('depth must be [T,H,W], H,W >= 2')
    if w2c.shape not in ((frames, 3, 4), (frames, 4, 4)) or intrinsic.shape != (frames, 3, 3):
        raise ValueError('world_to_camera must be [T,3,4/4,4] and intrinsics [T,3,3]')
    if not np.isfinite(w2c).all() or not np.isfinite(intrinsic).all():
        raise ValueError('camera matrices must be finite')
    if not 0 <= reference_frame < frames:
        raise ValueError('reference_frame is outside the supplied window')
    for name, value in (('visibility', visibility), ('track_confidence', track_confidence)):
        if value is not None and np.shape(value) != (frames, points):
            raise ValueError(f'{name} must be [T,N]')
    if depth_confidence is not None and np.shape(depth_confidence) != depth.shape:
        raise ValueError('depth_confidence must match depth [T,H,W]')
    height, width = depth.shape[1:]
    ref = int(reference_frame)
    reasons = np.zeros((frames, points), dtype=np.uint16)
    confidence_ok = np.ones((frames, points), dtype=bool)
    for value, minimum in ((visibility, cfg.visibility_min), (track_confidence, cfg.track_confidence_min)):
        if value is not None:
            value = np.asarray(value, dtype=np.float64)
            confidence_ok &= np.isfinite(value) & (value >= minimum)
    confidence_ok &= confidence_ok[ref][None]
    reasons[~confidence_ok] |= 2

    sampled_depth = np.empty((frames, points), dtype=np.float64)
    depth_ok = np.empty((frames, points), dtype=bool)
    depth_edge = np.empty((frames, points), dtype=bool)
    for t in range(frames):
        value, inside, spread = _sample(depth[t], tracks[t])
        sampled_depth[t] = value
        depth_ok[t] = inside & (value > 0)
        depth_edge[t] = spread > cfg.depth_edge_relative
        coordinates_ok = (np.isfinite(tracks[t]).all(-1) & (tracks[t, :, 0] >= 0)
                          & (tracks[t, :, 0] <= width-1) & (tracks[t, :, 1] >= 0)
                          & (tracks[t, :, 1] <= height-1))
        reasons[t, ~coordinates_ok] |= 1
        if depth_confidence is not None:
            confidence_map = np.asarray(depth_confidence[t], dtype=np.float64)
            sampled, finite, _ = _sample(confidence_map, tracks[t])
            finite_map = confidence_map[np.isfinite(confidence_map)]
            minimum = (np.quantile(finite_map, cfg.depth_confidence_quantile)
                       if len(finite_map) else np.inf)
            if cfg.depth_confidence_min is not None:
                minimum = max(minimum, cfg.depth_confidence_min)
            depth_ok[t] &= finite & (sampled >= minimum)
    depth_ok &= depth_ok[ref][None]
    depth_edge |= depth_edge[ref][None]
    reasons[~depth_ok] |= 4
    reasons[depth_edge] |= 8

    pixels_h = np.concatenate((tracks[ref], np.ones((points, 1))), axis=-1)
    rays = pixels_h @ np.linalg.inv(intrinsic[ref]).T
    rays /= rays[:, 2:3]
    camera_points = rays * sampled_depth[ref, :, None]
    # Inverse rotation (rather than transpose) makes convention explicit and
    # tolerates small teacher orthogonality error; invalid cameras fail clearly.
    rotation_inv = np.linalg.inv(w2c[ref, :3, :3])
    world_points = (camera_points-w2c[ref, :3, 3]) @ rotation_inv.T
    rigid = np.empty_like(tracks)
    reverse_rigid = np.empty_like(tracks)
    projected_depth = np.empty((frames, points), dtype=np.float64)
    reverse_projected_depth = np.empty((frames, points), dtype=np.float64)
    depth_allowance = np.zeros((frames, points), dtype=np.float64)
    reverse_depth_allowance = np.zeros((frames, points), dtype=np.float64)
    for t in range(frames):
        rigid[t], projected_depth[t] = _project(world_points, w2c[t, :3, :3], w2c[t, :3, 3], intrinsic[t])
        reference_valid = np.isfinite(rigid[t]).all(-1) & (projected_depth[t] > 0)
        target_depth, inside, _ = _sample(depth[t], rigid[t])
        reasons[t, ~(reference_valid & inside & (target_depth > 0))] |= 4
        if cfg.check_occlusion:
            occluded = (reference_valid & inside & (target_depth > 0)
                        & (target_depth < projected_depth[t]*(1-cfg.occlusion_relative)))
            reasons[t, occluded] |= 16
        for scale in (1-cfg.depth_uncertainty_relative, 1+cfg.depth_uncertainty_relative):
            perturbed = (camera_points*scale-w2c[ref, :3, 3]) @ rotation_inv.T
            projected, _ = _project(perturbed, w2c[t, :3, :3], w2c[t, :3, 3], intrinsic[t])
            depth_allowance[t] = np.maximum(depth_allowance[t], np.linalg.norm(projected-rigid[t], axis=-1))
        # Independent target-frame depth helps distinguish an erroneous query
        # depth from actual motion. Reusing forward-projected z here would make
        # this an algebraic round-trip, providing no independent evidence.
        target_pixels_h = np.concatenate((tracks[t], np.ones((points, 1))), axis=-1)
        target_rays = target_pixels_h @ np.linalg.inv(intrinsic[t]).T
        target_rays /= target_rays[:, 2:3]
        target_camera_points = target_rays*sampled_depth[t, :, None]
        target_rotation_inv = np.linalg.inv(w2c[t, :3, :3])
        target_world = (target_camera_points-w2c[t, :3, 3]) @ target_rotation_inv.T
        reverse_rigid[t], reverse_projected_depth[t] = _project(
            target_world, w2c[ref, :3, :3], w2c[ref, :3, 3], intrinsic[ref])
        reverse_depth, reverse_inside, reverse_edge = _sample(depth[ref], reverse_rigid[t])
        reverse_valid = (np.isfinite(reverse_rigid[t]).all(-1)
                         & (reverse_projected_depth[t] > 0) & reverse_inside & (reverse_depth > 0))
        reasons[t, ~reverse_valid] |= 4
        reasons[t, reverse_edge > cfg.depth_edge_relative] |= 8
        if cfg.check_occlusion:
            reverse_occluded = (reverse_valid
                & (reverse_depth < reverse_projected_depth[t]*(1-cfg.occlusion_relative)))
            reasons[t, reverse_occluded] |= 16
        for scale in (1-cfg.depth_uncertainty_relative, 1+cfg.depth_uncertainty_relative):
            perturbed = (target_camera_points*scale-w2c[t, :3, 3]) @ target_rotation_inv.T
            projected, _ = _project(perturbed, w2c[ref, :3, :3], w2c[ref, :3, 3], intrinsic[ref])
            reverse_depth_allowance[t] = np.maximum(
                reverse_depth_allowance[t], np.linalg.norm(projected-reverse_rigid[t], axis=-1))
    residual = np.linalg.norm(tracks-rigid, axis=-1)
    reverse_residual = np.linalg.norm(tracks[ref][None]-reverse_rigid, axis=-1)
    residual[ref] = np.nan  # The query frame is not a second independent observation.
    reverse_residual[ref] = np.nan
    finite = (np.isfinite(residual) & np.isfinite(depth_allowance)
              & np.isfinite(reverse_residual) & np.isfinite(reverse_depth_allowance))
    valid = (reasons == 0) & finite
    base = max(1.0, cfg.base_threshold_px*max(height, width)/518.)
    thresholds = np.full(frames, base, dtype=np.float64)
    reverse_thresholds = np.full(frames, base, dtype=np.float64)
    frame_reliable = np.zeros(frames, dtype=bool)
    background_median = np.full(frames, np.nan)
    reverse_background_median = np.full(frames, np.nan)
    static_support = np.full(frames, np.nan)
    reverse_static_support = np.full(frames, np.nan)
    corrected = np.maximum(0., residual-depth_allowance)
    reverse_corrected = np.maximum(0., reverse_residual-reverse_depth_allowance)
    for t in range(frames):
        values = corrected[t, valid[t]]
        reverse_values = reverse_corrected[t, valid[t]]
        if t == ref:
            continue
        if len(values) >= cfg.min_background_tracks:
            median = float(np.median(values))
            mad = float(np.median(np.abs(values-median)))
            background_median[t] = median
            static_support[t] = float(np.mean(values <= base*cfg.static_support_factor))
            reverse_median = float(np.median(reverse_values))
            reverse_mad = float(np.median(np.abs(reverse_values-reverse_median)))
            reverse_background_median[t] = reverse_median
            reverse_static_support[t] = float(np.mean(reverse_values <= base*cfg.static_support_factor))
            frame_reliable[t] = (median <= base*cfg.global_median_factor
                                 and static_support[t] >= cfg.min_static_support
                                 and reverse_median <= base*cfg.global_median_factor
                                 and reverse_static_support[t] >= cfg.min_static_support)
            thresholds[t] = max(base, median+cfg.robust_sigma*1.4826*mad)
            reverse_thresholds[t] = max(base, reverse_median+cfg.robust_sigma*1.4826*reverse_mad)
        if not frame_reliable[t]:
            reasons[t] |= 32
            valid[t] = False
    effective_thresholds = thresholds[:, None]+depth_allowance
    reverse_effective_thresholds = reverse_thresholds[:, None]+reverse_depth_allowance
    forward_exceed = residual > effective_thresholds
    reverse_exceed = reverse_residual > reverse_effective_thresholds
    # Preserve both reasons for conflicting depth/occlusion evidence. A point
    # may be occluded under one rigid explanation yet also disagree between
    # forward and reverse tests. Neither interpretation supports a motion label.
    comparable = finite & ((reasons & (1 | 2 | 4 | 8 | 32)) == 0)
    disagreement = comparable & (forward_exceed != reverse_exceed)
    reasons[disagreement] |= 128
    valid[disagreement] = False
    agreement = valid & (forward_exceed == reverse_exceed)
    exceed = valid & forward_exceed & reverse_exceed
    valid_count = valid.sum(0)
    exceed_count = exceed.sum(0)
    persistent = ((exceed_count >= cfg.min_dynamic_frames)
                  & (exceed_count >= cfg.min_dynamic_fraction*np.maximum(valid_count, 1)))
    sufficient = valid_count >= cfg.min_dynamic_frames
    labels = np.full((frames, points), -1, dtype=np.int8)
    labels[valid & sufficient[None] & ~exceed] = 0
    labels[exceed & persistent[None]] = 1
    reasons[valid & (labels < 0)] |= 64
    reasons[ref] |= 64
    score = np.clip(residual/effective_thresholds-.5, 0., 1.)
    score[~valid] = np.nan
    track_labels = np.full(points, -1, dtype=np.int8)
    track_labels[sufficient & (exceed_count == 0)] = 0
    track_labels[persistent] = 1
    diagnostics = {
        'method': 'rigid_reprojection_sparse_tracks_v2_bidirectional',
        'uses_future_frames': True,
        'reference_frame': ref,
        'coordinate_convention': 'OpenCV_world_to_camera_and_camera_z_depth',
        'config': asdict(cfg),
        'frames': frames, 'tracks': points,
        'reliable_comparison_frames': int(frame_reliable.sum()),
        'valid_observations': int(valid.sum()),
        'dynamic_tracks': int(persistent.sum()),
        'static_tracks': int((track_labels == 0).sum()),
        'unknown_tracks': int((track_labels < 0).sum()),
        'bidirectional_disagreements': int(disagreement.sum()),
        'depth_confidence_supplied': depth_confidence is not None,
        'track_confidence_supplied': track_confidence is not None,
        'visibility_supplied': visibility is not None,
        'limitations': ['teacher pseudo-labels, not ground truth',
                        'majority-static and accurate relative camera/depth assumptions',
                        'bidirectional agreement cannot reject correlated teacher depth errors',
                        'query-frame sparse coverage misses entrants and unqueried pixels',
                        'future-derived labels must not be free-generation conditions'],
    }
    return dict(rigid_tracks=rigid.astype(np.float32), residual_px=residual.astype(np.float32),
                reverse_rigid_tracks=reverse_rigid.astype(np.float32),
                reverse_residual_px=reverse_residual.astype(np.float32),
                projected_depth=projected_depth.astype(np.float32), valid=valid,
                reverse_projected_depth=reverse_projected_depth.astype(np.float32),
                labels=labels, dynamic_score=score.astype(np.float32),
                persistent_dynamic=persistent, track_labels=track_labels,
                thresholds_px=thresholds.astype(np.float32),
                reverse_thresholds_px=reverse_thresholds.astype(np.float32),
                effective_thresholds_px=effective_thresholds.astype(np.float32),
                reverse_effective_thresholds_px=reverse_effective_thresholds.astype(np.float32),
                depth_allowance_px=depth_allowance.astype(np.float32),
                reverse_depth_allowance_px=reverse_depth_allowance.astype(np.float32),
                bidirectional_agreement=agreement,
                frame_reliable=frame_reliable, reason_bits=reasons,
                background_median_px=background_median.astype(np.float32),
                reverse_background_median_px=reverse_background_median.astype(np.float32),
                static_support=static_support.astype(np.float32),
                reverse_static_support=reverse_static_support.astype(np.float32),
                diagnostics=diagnostics)
