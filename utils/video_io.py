import random
import tempfile
import zipfile
import os

import torch
import numpy as np
import cv2
import OpenEXR
import Imath


def _compute_frame_indices(total_frames, num_frames, temporal_jitter=True,
                            max_start_frac=0.3, stride_jitter_frac=0.15,
                            max_frame_span=0):
    """Compute frame indices for sampling. Shared by video and depth loading."""
    if total_frames <= num_frames:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)

    if max_frame_span > 0:
        span = min(total_frames, max(max_frame_span, num_frames))
        if temporal_jitter and total_frames > span:
            start_offset = random.randint(0, total_frames - span)
        else:
            start_offset = max(0, (total_frames - span) // 2)
        end_offset = min(total_frames - 1, start_offset + span - 1)
        return np.linspace(
            start_offset, end_offset, num_frames, dtype=int)

    if temporal_jitter:
        max_start = max(1, int(total_frames * max_start_frac))
        start_offset = random.randint(0, min(max_start, total_frames - num_frames))
        available = total_frames - start_offset
        uniform_stride = (available - 1) / max(num_frames - 1, 1)
        jitter_range = uniform_stride * stride_jitter_frac
        indices = []
        pos = float(start_offset)
        for _ in range(num_frames):
            jitter = random.uniform(-jitter_range, jitter_range)
            idx = int(np.clip(pos + jitter, start_offset, total_frames - 1))
            indices.append(idx)
            pos += uniform_stride
        return np.array(indices, dtype=int)

    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


def read_video_frames(video_path, num_frames, target_size=518,
                       temporal_jitter=True, max_start_frac=0.3,
                       stride_jitter_frac=0.15, frame_indices=None,
                       max_frame_span=0):
    """Read N frames from mp4.

    Args:
        video_path: path to mp4 file
        num_frames: number of frames to sample
        target_size: resize target (square)
        temporal_jitter: enable random start offset and stride variation
        max_start_frac: max fraction of video to skip as start offset
        stride_jitter_frac: max fraction of uniform stride to jitter
        frame_indices: pre-computed indices (overrides temporal_jitter)

    Returns: Tensor [N, 3, target_size, target_size] float32 in [0, 1]
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Cannot read video: {video_path}")

    if frame_indices is None:
        indices = _compute_frame_indices(total, num_frames, temporal_jitter,
                                          max_start_frac, stride_jitter_frac,
                                          max_frame_span)
    else:
        indices = frame_indices

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)
        frames.append(frame)
    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1].clone())

    return torch.stack(frames, dim=0)


def read_depth_frames(depth_zip_path, frame_indices, num_frames, target_size=518):
    """Read depth frames from a zip of EXR files.

    Args:
        depth_zip_path: path to {video_id}.zip containing 00000.exr ... 00NNN.exr
        frame_indices: array of frame indices to read
        num_frames: expected number of frames (for padding)
        target_size: resize target (square)

    Returns: Tensor [N, target_size, target_size] float32, or None if file missing
    """
    if not depth_zip_path or not os.path.exists(depth_zip_path):
        return None

    try:
        zf = zipfile.ZipFile(depth_zip_path)
        exr_names = sorted([n for n in zf.namelist() if n.endswith('.exr')])
        if not exr_names:
            zf.close()
            return None

        max_exr_idx = len(exr_names) - 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        frames = []
        for idx in frame_indices:
            exr_idx = min(int(idx), max_exr_idx)
            exr_name = exr_names[exr_idx]
            raw = zf.read(exr_name)

            # Write to temp file for OpenEXR (doesn't support BytesIO)
            with tempfile.NamedTemporaryFile(suffix='.exr') as tmp:
                tmp.write(raw)
                tmp.flush()
                exr = OpenEXR.InputFile(tmp.name)
                ch = list(exr.header()['channels'].keys())[0]
                dw = exr.header()['dataWindow']
                w = dw.max.x - dw.min.x + 1
                h = dw.max.y - dw.min.y + 1
                data = np.frombuffer(exr.channel(ch, pt), dtype=np.float32).reshape(h, w)

            # Resize to target
            frame = cv2.resize(data, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            frames.append(torch.from_numpy(frame))

        zf.close()

        while len(frames) < num_frames:
            frames.append(frames[-1].clone())

        return torch.stack(frames, dim=0)  # [N, H, W]
    except Exception as e:
        return None
