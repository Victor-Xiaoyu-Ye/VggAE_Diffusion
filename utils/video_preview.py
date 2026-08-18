"""Small, failure-tolerant RGB preview writers for video training."""

from __future__ import annotations

import json
import os
from typing import Mapping, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw


def _as_uint8(video: torch.Tensor) -> np.ndarray:
    """Convert one THWC RGB tensor to uint8 without modifying the input."""
    value = torch.as_tensor(video).detach().float().cpu()
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"expected THWC RGB video, got {tuple(value.shape)}")
    return value.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()


def _write_mp4(frames: np.ndarray, path: str, fps: float) -> str:
    import imageio.v2 as imageio

    imageio.mimsave(
        path, list(frames), fps=fps, codec="libx264", pixelformat="yuv420p")
    return path


def save_video_preview(
    output_dir: str,
    stem: str,
    videos: Mapping[str, torch.Tensor],
    *,
    fps: float = 8.0,
    metadata: Optional[Mapping[str, object]] = None,
    save_frames: bool = True,
    save_mp4: bool = True,
) -> dict:
    """Save labelled grids, frame PNGs, optional MP4s, and a JSON manifest.

    ``videos`` preserves insertion order. Each value must be a single THWC RGB
    clip with the same frame count and spatial size. MP4 failures are recorded
    in the manifest and never remove the PNG fallback.
    """
    if not videos:
        raise ValueError("at least one video is required")
    if fps <= 0:
        raise ValueError("fps must be positive")
    os.makedirs(output_dir, exist_ok=True)

    arrays = {label: _as_uint8(video) for label, video in videos.items()}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"preview videos must share one shape, got {sorted(shapes)}")
    frames, height, width, channels = next(iter(shapes))
    if frames < 1 or channels != 3:
        raise ValueError(f"invalid preview video shape {(frames, height, width, channels)}")

    label_width = max(140, min(320, max(len(label) for label in arrays) * 10 + 20))
    grid = Image.new("RGB", (label_width + frames * width, len(arrays) * height), "white")
    draw = ImageDraw.Draw(grid)
    result = {
        "schema": "video-preview-v1",
        "stem": stem,
        "fps": float(fps),
        "frames": int(frames),
        "height": int(height),
        "width": int(width),
        "videos": {},
    }
    if metadata:
        result["metadata"] = dict(metadata)

    for row, (label, array) in enumerate(arrays.items()):
        top = row * height
        draw.text((8, top + 8), label, fill="black")
        for index, frame in enumerate(array):
            grid.paste(Image.fromarray(frame), (label_width + index * width, top))

        entry = {
            "range": [float(array.min()) / 255.0, float(array.max()) / 255.0],
        }
        if save_frames:
            frame_dir = os.path.join(output_dir, f"{stem}_{label.lower()}_pngs")
            os.makedirs(frame_dir, exist_ok=True)
            for index, frame in enumerate(array):
                Image.fromarray(frame).save(
                    os.path.join(frame_dir, f"frame_{index:02d}.png"))
            entry["png_dir"] = frame_dir
        if save_mp4:
            mp4_path = os.path.join(output_dir, f"{stem}_{label.lower()}.mp4")
            try:
                entry["mp4"] = _write_mp4(array, mp4_path, fps)
            except Exception as exc:
                entry["mp4_error"] = repr(exc)
        result["videos"][label] = entry

    grid_path = os.path.join(output_dir, f"{stem}_grid.png")
    grid.save(grid_path)
    result["grid"] = grid_path
    manifest_path = os.path.join(output_dir, f"{stem}_preview.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    result["manifest"] = manifest_path
    return result
