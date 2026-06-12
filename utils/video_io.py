import random
import tempfile
import zipfile
import os

import torch
import numpy as np
import cv2
import OpenEXR
import Imath


class VideoDecodeError(RuntimeError):
    """Raised when a video cannot provide any usable frames."""


def _compute_frame_indices(total_frames, num_frames, temporal_jitter=True,
                            max_start_frac=0.3, stride_jitter_frac=0.15,
                            max_frame_span=0, fps=0.0,
                            clip_duration_seconds=0.0):
    """Compute frame indices for sampling. Shared by video and depth loading."""
    if total_frames <= num_frames:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)

    span = 0
    if clip_duration_seconds > 0:
        if fps <= 0:
            raise ValueError(
                "fps must be positive when clip_duration_seconds is set")
        span = round(clip_duration_seconds * fps) + 1
    elif max_frame_span > 0:
        span = max_frame_span

    if span > 0:
        span = min(total_frames, max(span, num_frames))
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
                       max_frame_span=0, fps=0.0,
                       clip_duration_seconds=0.0, seek_retries=2):
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
    if not cap.isOpened():
        cap.release()
        raise VideoDecodeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise VideoDecodeError(
            f"Video reports no frames: {video_path}")

    if frame_indices is None:
        indices = _compute_frame_indices(total, num_frames, temporal_jitter,
                                          max_start_frac, stride_jitter_frac,
                                          max_frame_span, fps,
                                          clip_duration_seconds)
    else:
        # SpatialVID metadata can disagree with the container's actual frame
        # count. Clamp precomputed indices to the decoder-visible range.
        indices = np.clip(
            np.asarray(frame_indices, dtype=np.int64), 0, total - 1)

    frames = [None] * len(indices)
    try:
        for frame_pos, idx in enumerate(indices):
            for attempt in range(seek_retries + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(
                            frame, (target_size, target_size),
                            interpolation=cv2.INTER_LINEAR)
                    except cv2.error:
                        ret = False
                    else:
                        tensor = torch.from_numpy(frame).float() / 255.0
                        frames[frame_pos] = tensor.permute(2, 0, 1)
                        break
                if attempt < seek_retries:
                    # Reopening the container recovers transient random-seek
                    # failures seen with some SpatialVID mp4 files.
                    cap.release()
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        break
    finally:
        cap.release()

    valid_positions = [
        pos for pos, frame in enumerate(frames) if frame is not None]
    if not valid_positions:
        requested = ",".join(str(int(idx)) for idx in indices)
        raise VideoDecodeError(
            f"Failed to decode requested frames [{requested}] from "
            f"{video_path} (reported_frames={total})")

    # Preserve temporal positions when isolated seeks fail by copying the
    # nearest successfully decoded frame, rather than shortening the clip.
    for pos, frame in enumerate(frames):
        if frame is None:
            nearest = min(valid_positions, key=lambda valid: abs(valid - pos))
            frames[pos] = frames[nearest].clone()

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
