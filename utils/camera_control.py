"""Explicit SpatialVID camera coordinates and sampled-frame alignment.

This module does not infer frame numbers from annotation row numbers, estimate
metric scale, or use future-video statistics to normalize translation. Inputs
are OpenCV world-to-camera poses in [tx, ty, tz, qx, qy, qz, qw] order and
intrinsics normalized independently by image width and height.
"""
import re

import numpy as np


CAMERA_CONTROL_SCHEMA = 'spatialvid-anchor-camera-v1'


def _frame_indexes(value, name):
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be a nonempty one-dimensional integer array')
    if np.any(array < 0) or np.any(array[1:] <= array[:-1]):
        raise ValueError(f'{name} must be nonnegative and strictly increasing')
    if np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f'{name} exceeds the supported frame-index range')
    return array.astype(np.int64)


def parse_indexes_text(text, pose_count):
    """Read explicit original RGB frame indexes, never inferred row numbers.

    SpatialVID's observed layout is ``ordinal original_frame`` with a comment
    header. Ordinals must be exactly 0..N-1. An explicit one-column frame list
    is also supported, but layouts cannot be mixed. A recognized total-count
    header must agree with the pose count. Unknown layouts fail closed.
    """
    if not isinstance(text, str):
        raise ValueError('indexes text must be decoded text')
    lines = text.lstrip('\ufeff').splitlines()
    values, columns = [], None
    for line in lines:
        token = line.strip()
        if not token:
            continue
        if token.startswith('#'):
            total = re.fullmatch(r'#\s*total\s+([0-9]+)\s+indexes', token)
            if total and int(total.group(1)) != pose_count:
                raise ValueError('indexes header count differs from pose count')
            continue
        fields = token.split()
        if len(fields) not in (1, 2) or any(re.fullmatch(r'[0-9]+', f) is None for f in fields):
            raise ValueError('unknown indexes layout: expected RGB frame or ordinal RGB-frame columns')
        if columns is None:
            columns = len(fields)
        if columns != len(fields):
            raise ValueError('indexes file mixes one-column and two-column layouts')
        if columns == 2 and int(fields[0]) != len(values):
            raise ValueError('pose-row ordinals must be exactly 0..N-1 in order')
        values.append(int(fields[-1]))
    if len(values) != pose_count:
        raise ValueError(f'indexes/poses length mismatch: {len(values)} != {pose_count}')
    return _frame_indexes(values, 'annotation indexes')


def normalize_quaternions(quaternions):
    """Normalize xyzw quaternions, including safely handling scaled inputs."""
    q = np.asarray(quaternions, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError('quaternions must be finite xyzw vectors')
    scale = np.max(np.abs(q), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError('zero quaternion has no camera orientation')
    q = q / scale
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quaternion_to_matrix(quaternions):
    q = normalize_quaternions(quaternions)
    x, y, z, w = np.moveaxis(q, -1, 0)
    values = (
        1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y),
    )
    return np.stack(values, axis=-1).reshape(q.shape[:-1] + (3, 3))


def slerp_xyzw(start, end, weight):
    """Shortest-arc interpolation; quaternion sign cannot reverse the path."""
    a, b = normalize_quaternions(start), normalize_quaternions(end)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError('slerp requires two individual quaternion vectors')
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('slerp weight must lie in [0, 1]')
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    dot = np.clip(dot, -1., 1.)
    if dot > .9995:
        return normalize_quaternions((1-weight)*a + weight*b)
    angle = np.arccos(dot)
    return normalize_quaternions(
        (np.sin((1-weight)*angle)*a + np.sin(weight*angle)*b) / np.sin(angle))


def normalized_intrinsics_to_pixel(intrinsics, width, height):
    """Pure resize preserves normalized K; a crop needs an explicit transform."""
    k = np.asarray(intrinsics, dtype=np.float64)
    if k.ndim < 1 or k.shape[-1] != 4 or not np.isfinite(k).all():
        raise ValueError('intrinsics must be finite [fx, fy, cx, cy] vectors')
    if np.any(k[..., :2] <= 0):
        raise ValueError('focal lengths must be positive')
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError('image width and height must be positive')
    pixels = k * np.array([width, height, width, height])
    result = np.zeros(k.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = pixels[..., 0]
    result[..., 1, 1] = pixels[..., 1]
    result[..., 0, 2] = pixels[..., 2]
    result[..., 1, 2] = pixels[..., 3]
    result[..., 2, 2] = 1
    return result


def build_window_camera_controls(poses, intrinsics, annotation_indexes,
                                 sample_frame_indices, max_interpolation_gap_frames=None):
    """Align explicit pose samples, then express cameras in the first-frame axes.

    Translation interpolation uses camera centers, not the translation part of
    w2c matrices. Rotations use SLERP. Exact annotation matches are preserved;
    interior RGB frames require valid annotations on both sides. Extrapolation
    is prohibited. A caller can bound the permitted annotation gap explicitly.

    The output relative_c2w[t] maps points from camera t into the first sampled
    camera's coordinate system. Its translation remains in the source pose
    units; it is not promised to be metres. Normalized K is unchanged by the
    current direct RGB resize to 518x518. No decoder or trainable model is used.
    """
    indexes = _frame_indexes(annotation_indexes, 'annotation indexes')
    requested = _frame_indexes(sample_frame_indices, 'sample frame indexes')
    p = np.asarray(poses, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if p.shape != (len(indexes), 7) or not np.isfinite(p).all():
        raise ValueError('poses must be finite [N,7] matching annotation indexes')
    if k.shape != (len(indexes), 4):
        raise ValueError('intrinsics must be [N,4] matching annotation indexes')
    normalized_intrinsics_to_pixel(k, 1, 1)  # Validate without rescaling.
    if (max_interpolation_gap_frames is not None and
            (not np.isfinite(max_interpolation_gap_frames) or max_interpolation_gap_frames <= 0)):
        raise ValueError('maximum interpolation gap must be positive')
    if requested[0] < indexes[0] or requested[-1] > indexes[-1]:
        raise ValueError('sample frames lie outside annotated coverage; extrapolation is forbidden')

    q_w2c = normalize_quaternions(p[:, 3:])
    q_c2w = q_w2c * np.array([-1., -1., -1., 1.])
    r_c2w = quaternion_to_matrix(q_c2w)
    centers = -np.einsum('nij,nj->ni', r_c2w, p[:, :3])
    if not np.isfinite(centers).all():
        raise ValueError('camera centers are nonfinite')
    matrices, aligned_k, brackets, weights = [], [], [], []
    for frame in requested:
        upper = int(np.searchsorted(indexes, frame))
        if indexes[upper] == frame:
            lower, weight = upper, 0.
            rotation, center, intr = r_c2w[upper], centers[upper], k[upper]
        else:
            lower = upper - 1
            gap = int(indexes[upper] - indexes[lower])
            if max_interpolation_gap_frames is not None and gap > max_interpolation_gap_frames:
                raise ValueError(f'annotation gap {gap} exceeds the configured interpolation limit')
            weight = float(frame-indexes[lower]) / gap
            rotation = quaternion_to_matrix(slerp_xyzw(q_c2w[lower], q_c2w[upper], weight))
            center = (1-weight)*centers[lower] + weight*centers[upper]
            intr = (1-weight)*k[lower] + weight*k[upper]
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, center
        matrices.append(matrix)
        aligned_k.append(intr)
        brackets.append([int(indexes[lower]), int(indexes[upper])])
        weights.append(weight)
    matrices = np.stack(matrices)
    anchor_inverse = np.eye(4)
    anchor_inverse[:3, :3] = matrices[0, :3, :3].T
    anchor_inverse[:3, 3] = -anchor_inverse[:3, :3] @ matrices[0, :3, 3]
    relative = anchor_inverse[None] @ matrices
    if not np.isfinite(relative).all():
        raise ValueError('relative camera matrices are nonfinite')
    # Remove only floating point roundoff for the algebraically exact anchor.
    relative[0] = np.eye(4)
    return {
        'schema': CAMERA_CONTROL_SCHEMA,
        'relative_c2w': relative,
        'normalized_intrinsics': np.stack(aligned_k),
        'frame_indices': requested,
        'annotation_brackets': np.asarray(brackets, dtype=np.int64),
        'interpolation_weights': np.asarray(weights),
        'contract': {
            'source_pose': 'opencv_w2c_tx_ty_tz_qx_qy_qz_qw',
            'relative_pose': 'inverse(c2w_first_sample) @ c2w_sample',
            'frame_mapping': 'explicit_original_rgb_indices',
            'interpolation': 'camera_center_linear_rotation_slerp_intrinsics_linear_no_extrapolation',
            'translation_units': 'source_annotation_units_unverified',
            'translation_scale': 'unchanged_no_per_window_or_future_normalization',
            'intrinsics': 'fx_fy_cx_cy_normalized_by_width_height',
            'max_interpolation_gap_frames': max_interpolation_gap_frames,
        },
    }


def pack_window_camera_features(controls, fps, translation_scale):
    """Pack the nine-frame window into float32 [9,14] camera conditions.

    Feature order is rotation column 0 (3), rotation column 1 (3), relative
    camera-center translation / translation_scale (3), normalized fx/fy/cx/cy
    (4), and elapsed seconds since the first sampled RGB frame (1).

    translation_scale is an explicit positive scalar calibrated on training
    annotations and then held fixed across videos/evaluation/inference. The
    caller must record it in the experiment contract. Never derive it from the
    current video's future path. This function does not assert metric units.
    """
    for name, value in (('fps', fps), ('translation_scale', translation_scale)):
        if np.ndim(value) != 0 or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a finite positive scalar')
    if controls.get('schema') != CAMERA_CONTROL_SCHEMA:
        raise ValueError('unsupported camera-control schema')
    contract = controls.get('contract', {})
    if contract.get('translation_scale') != 'unchanged_no_per_window_or_future_normalization':
        raise ValueError('camera controls must preserve the unnormalized source translation')
    relative = np.asarray(controls['relative_c2w'], dtype=np.float64)
    intrinsics = np.asarray(controls['normalized_intrinsics'], dtype=np.float64)
    frames = _frame_indexes(controls['frame_indices'], 'sample frame indexes')
    if relative.shape != (9, 4, 4) or intrinsics.shape != (9, 4) or len(frames) != 9:
        raise ValueError('window camera features require exactly nine aligned RGB frames')
    if not np.isfinite(relative).all():
        raise ValueError('camera poses must be finite')
    normalized_intrinsics_to_pixel(intrinsics, 1, 1)
    rotation = relative[:, :3, :3]
    if (not np.allclose(relative[:, 3, :], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(rotation.swapaxes(1, 2) @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError('camera matrices must be rigid transforms with proper rotations')
    rotation_6d = rotation[:, :, :2].transpose(0, 2, 1).reshape(9, 6)
    elapsed_seconds = (frames-frames[0]).astype(np.float64)[:, None] / fps
    features = np.concatenate((rotation_6d, relative[:, :3, 3]/translation_scale,
                               intrinsics, elapsed_seconds), axis=1)
    if not np.isfinite(features).all() or np.any(np.abs(features) > np.finfo(np.float32).max):
        raise ValueError('camera features cannot be represented as finite float32')
    return features.astype(np.float32)
