"""Stage56 offline motion controls for matched, fixed-window generation.

Future cameras/tracks are explicitly supplied control labels. Future depth is
only a teacher validity check, not an input map. Appearance is absent here:
the model obtains it from observed latent frames and independent prefix tracks.
Sparse motion evidence is not semantic or instance segmentation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
import torch

from data.scene_rgb import resize_rgb
from utils.scene_trackmask import TrackMaskConfig, camera_compensated_track_mask


SCHEMA = 'scene_control_v1'
MOTION_CHANNELS = ('dx', 'dy', 'support', 'confidence', 'dynamic', 'static',
                   'unknown', 'reserved_zero')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def window_key(sample):
    """Stable correspondence key; never use sample order as frame provenance."""
    ids = sample.get('frame_ids')
    if not isinstance(ids, (list, tuple)) or not ids or len(set(ids)) != len(ids):
        raise ValueError('Each window needs distinct saved frame_ids')
    return canonical_hash(dict(id=sample['id'], dataset=sample['dataset'], frame_ids=list(ids)))


def select_windows(entries, limit, seed=42):
    """Round robin over hash-ordered source IDs, then hash-ordered windows.

    This spreads a small controlled experiment across source clips instead of
    accidentally selecting many adjacent windows from a few source shards.
    Duplicate (ID, frame_ids) windows keep the earliest source shard/index.
    """
    if limit < 0:
        raise ValueError('Window limit must be nonnegative; zero selects all')
    if limit == 0:
        limit = len(entries)
    groups, seen = {}, set()
    for row in sorted(entries, key=lambda x: (x['file'], x['index'])):
        key = window_key(row)
        if key in seen:
            continue
        seen.add(key)
        groups.setdefault(row['id'], []).append(dict(row, key=key))
    ids = sorted(groups, key=lambda ident: canonical_hash([seed, ident]))
    for values in groups.values():
        values.sort(key=lambda row: canonical_hash([seed, row['key']]))
    selected, level = [], 0
    while len(selected) < limit:
        added = [groups[ident][level] for ident in ids if level < len(groups[ident])]
        if not added:
            break
        selected.extend(added[:limit-len(selected)])
        level += 1
    return selected


def replay_rgb(sample, row, localize, size=518):
    """Decode exact source frames using the production RGB crop/resize function.

    Only an original SpatialVID video is accepted in this experiment; do not
    silently substitute an evaluation preview or independently sample a clip.
    ``localize`` is the bounded remote staging function supplied by the runner.
    """
    if (sample.get('dataset') != 'spatialvid' or row.get('dataset') != 'spatialvid'
            or row.get('id') != sample.get('id') or row.get('kind') in ('wai', 'frames', 'omni')):
        raise ValueError('Control replay requires the matching original SpatialVID video row')
    ids = list(sample['frame_ids'])
    if (any(type(i) is not int or i < 0 for i in ids)
            or ids != sorted(set(ids))):
        raise ValueError('Expected strictly chronological original video frame indices')
    import cv2
    path = localize(row['root'])
    cap = cv2.VideoCapture(str(path))
    values, sizes = [], []
    try:
        if not cap.isOpened():
            raise ValueError('Original source video cannot be opened')
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
        if count <= ids[-1]:
            raise ValueError('Saved frame index exceeds original video length')
        for index in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Original video frame cannot be decoded: {index}')
            h, w = image.shape[:2]
            sizes.append([h, w])
            values.append(resize_rgb(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), size))
    finally:
        cap.release()
    frames = torch.stack(values)
    checksum = hashlib.sha256(frames.numpy().tobytes()).hexdigest()
    provenance = dict(schema='scene-raw-frame-replay-v1', id=sample['id'],
        source_uri=row['root'], frame_ids=ids, source_dimensions_hw=sizes,
        resize=dict(size=size, crop='center_square', interpolation='PIL_BICUBIC',
                    implementation='data.scene_rgb.resize_rgb'),
        rgb_float32_sha256=checksum, fps=fps if np.isfinite(fps) and fps > 0 else None,
        timestamp_note='Frame indices plus nominal decoder FPS; source PTS was not saved by stage53',
        source='original_RGB_video_not_preview')
    if 'raw' in sample:
        raw = torch.as_tensor(sample['raw'])
        recovered = (frames*255).round().to(torch.uint8)
        if raw.shape != recovered.shape or not torch.equal(raw, recovered):
            raise ValueError('Original RGB replay differs from sealed evaluation raw frames')
        provenance['eval_raw_exact_uint8_match'] = True
    return frames, provenance


def _np(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _probability_tracks(teacher, h, w):
    tracks = _np(teacher['tracks']).astype(np.float32)
    weight = (_np(teacher['visibility']).astype(np.float32)
              * _np(teacher['track_confidence']).astype(np.float32))
    inside = (np.isfinite(tracks).all(-1) & (tracks[..., 0] >= 0)
              & (tracks[..., 0] <= w-1) & (tracks[..., 1] >= 0) & (tracks[..., 1] <= h-1))
    good = ((_np(teacher['visibility']) >= .5) & (_np(teacher['track_confidence']) >= .2)
            & np.isfinite(weight) & inside)
    weight = np.where(good, np.clip(weight, 0, 1), 0).astype(np.float32)
    xy = np.nan_to_num(tracks / np.array([w-1, h-1], np.float32), nan=0., posinf=0., neginf=0.)
    return xy, weight


def camera_control_rays(full, prefix, grid):
    """Relative Pluecker rays; scale is OBSERVED prefix depth, never future span."""
    extrinsics = _np(full['extrinsics']).astype(np.float64)
    intrinsics = _np(full['intrinsics']).astype(np.float64)
    depth = _np(prefix['depth']).astype(np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError('Observed prefix contains no usable scale reference')
    scale = float(np.median(depth[valid]))
    h, w = _np(full['depth']).shape[-2:]
    t = len(extrinsics)
    matrices = np.repeat(np.eye(4)[None], t, axis=0)
    matrices[:, :3] = extrinsics[:, :3]
    # Express every camera in the first observed camera coordinates.
    c2w = np.linalg.inv(matrices)
    relative = matrices[0][None] @ c2w
    yy, xx = np.meshgrid((np.arange(grid)+.5)*h/grid-.5,
                         (np.arange(grid)+.5)*w/grid-.5, indexing='ij')
    pixels = np.stack((xx, yy, np.ones_like(xx)), -1).reshape(-1, 3)
    rays = np.zeros((t, grid*grid, 8), np.float32)
    for i in range(t):
        directions = pixels @ np.linalg.inv(intrinsics[i]).T
        directions = directions @ relative[i, :3, :3].T
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        origin = relative[i, :3, 3]/scale
        rays[i, :, :3] = directions
        rays[i, :, 3:6] = np.cross(origin, directions)
        rays[i, :, 6] = float(np.linalg.norm(origin) > 1e-7)
        rays[i, :, 7] = 1.
    if not np.isfinite(rays).all():
        raise ValueError('Teacher cameras produced nonfinite rays')
    return torch.from_numpy(rays), scale


def build_controls(full, prefix, *, history=4, grid=18):
    """Numeric CPU controls and honest sparse support; never read future RGB."""
    t, h, w = _np(full['depth']).shape
    if not 1 <= history < t or len(_np(prefix['depth'])) != history:
        raise ValueError('Independent teacher prefix does not match observed history')
    if _np(prefix['depth']).shape[-2:] != (h, w):
        raise ValueError('Full and prefix teacher image shapes differ')
    if not np.array_equal(_np(full['query_points']), _np(prefix['query_points'])):
        raise ValueError('Full and prefix teacher query identities differ')
    xy, weight = _probability_tracks(full, h, w)
    pxy, pweight = _probability_tracks(prefix, h, w)
    evidence = camera_compensated_track_mask(tracks=_np(full['tracks']), depth=_np(full['depth']),
        world_to_camera=_np(full['extrinsics']), intrinsics=_np(full['intrinsics']),
        visibility=_np(full['visibility']), track_confidence=_np(full['track_confidence']),
        depth_confidence=_np(full['depth_conf']), reference_frame=0, config=TrackMaskConfig())
    rays, scale = camera_control_rays(full, prefix, grid)
    motion = np.zeros((t, grid*grid, len(MOTION_CHANNELS)), np.float32)
    motion[..., 6] = 1.  # Unqueried space is UNKNOWN, not static.
    for frame in range(t):
        den = np.zeros(grid*grid, np.float32)
        sums = np.zeros((grid*grid, 7), np.float32)
        # Bilinear splatting in token-center coordinates. Support is local to
        # observed tracks; no dilation across silhouettes or unknown entrants.
        position = xy[frame]*np.array([grid, grid], np.float32)-.5
        lower = np.floor(position).astype(np.int64)
        fraction = position-lower
        label = evidence['labels'][frame]
        values = np.concatenate((xy[frame]-xy[0], np.ones((len(label), 1), np.float32),
            weight[frame, :, None], (label == 1)[:, None], (label == 0)[:, None],
            (label < 0)[:, None]), axis=1).astype(np.float32)
        for ox, oy in ((0, 0), (0, 1), (1, 0), (1, 1)):
            gx, gy = lower[:, 0]+ox, lower[:, 1]+oy
            ww = weight[frame]*(fraction[:, 0] if ox else 1-fraction[:, 0])*(fraction[:, 1] if oy else 1-fraction[:, 1])
            good = (gx >= 0) & (gx < grid) & (gy >= 0) & (gy < grid) & (ww > 0)
            cells = gy[good]*grid+gx[good]
            np.add.at(den, cells, ww[good])
            np.add.at(sums, cells, ww[good, None]*values[good])
        supported = den > 0
        motion[frame, supported, :7] = sums[supported]/den[supported, None]
        motion[frame, :, 2] = np.clip(den, 0, 1)
    # Dense future depth is NOT an inference control in this task. It is used
    # only to assess teacher track validity/static-motion evidence above.
    result = dict(schema=SCHEMA, rays=rays, motion=torch.from_numpy(motion), tracks=torch.from_numpy(xy),
        track_weight=torch.from_numpy(weight), prefix_tracks=torch.from_numpy(pxy),
        prefix_track_weight=torch.from_numpy(pweight), metadata=dict(schema=SCHEMA,
            history=history, grid=grid, frames=t, scale=scale,
            scale_rule='median_positive_depth_of_independent_observed_prefix',
            motion_channels=list(MOTION_CHANNELS),
            task='ORACLE_FUTURE_CAMERA_SPARSE_TRACK_CONTROLLED_GENERATION',
            appearance_source='observed_latent_prefix_only_independent_prefix_tracking',
            tracks_are_semantic_instances=False,
            tracking_query='frame0_sparse_grid; entrants_not_supported',
            teacher_motion_diagnostics=evidence['diagnostics']))
    validate_controls(result, history=history, grid=grid, frames=t)
    return result


def validate_controls(value, *, history=4, grid=18, frames=9):
    q = value['tracks'].shape[1]
    shapes = dict(rays=(frames, grid*grid, 8), motion=(frames, grid*grid, 8),
        tracks=(frames, q, 2), track_weight=(frames, q), prefix_tracks=(history, q, 2),
        prefix_track_weight=(history, q))
    for name, shape in shapes.items():
        tensor = value[name]
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid control tensor {name}; expected finite {shape}')
    for name in ('track_weight', 'prefix_track_weight'):
        if ((value[name] < 0) | (value[name] > 1)).any():
            raise ValueError(f'{name} must be a probability weight')
    return value
