import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.video_io import (
    VideoDecodeError,
    _compute_frame_indices,
    read_depth_frames,
    read_video_frames,
)
from utils.moxing_io import (
    is_remote_path,
    join_remote,
    stage_remote_file,
)


class SpatialVidDataset(Dataset):
    """Dataset for SpatialVid mp4 videos.

    Workers decode mp4 and resize. When depth_root is provided, loads per-frame
    depth from zip archives with frame-level alignment to video frames.
    """

    def __init__(self, csv_path, video_root, seq_len=8, target_size=518,
                 annotation_index_path="", max_videos=0, num_frames_per_video=8,
                 depth_root="", temporal_jitter=True, index_shard_id=0,
                 index_num_shards=1, check_files=True, max_frame_span=0,
                 clip_duration_seconds=0.0, decode_retries=8,
                 clips_per_video=1):
        if index_num_shards < 1:
            raise ValueError("index_num_shards must be >= 1")
        if not 0 <= index_shard_id < index_num_shards:
            raise ValueError(
                f"index_shard_id must be in [0, {index_num_shards}), "
                f"got {index_shard_id}")

        self.target_size = target_size
        self.seq_len = seq_len
        self.num_frames_per_video = num_frames_per_video
        self.max_frame_span = max_frame_span
        self.clip_duration_seconds = clip_duration_seconds
        self.decode_retries = max(0, int(decode_retries))
        self.clips_per_video = max(1, int(clips_per_video))
        self.depth_root = depth_root
        self.remote_video_cache = os.environ.get(
            "MOX_VIDEO_CACHE_DIR",
            "/cache/yexiaoyu/vggae_runtime/cache/videos")
        self.remote_depth_cache = os.environ.get(
            "MOX_DEPTH_CACHE_DIR",
            "/cache/yexiaoyu/vggae_runtime/cache/depth")
        self.remote_cache_max_bytes = int(
            float(os.environ.get("MOX_VIDEO_CACHE_GB", "800"))
            * 1024 ** 3)
        self.remote_download_retries = int(
            os.environ.get("MOX_DOWNLOAD_RETRIES", "3"))
        # Video and depth share one precomputed index list, so jitter remains
        # frame-aligned when depth supervision is enabled.
        self.temporal_jitter = temporal_jitter

        # Load annotation index
        self.annotations = {}
        if annotation_index_path and os.path.exists(annotation_index_path):
            with open(annotation_index_path) as f:
                self.annotations = json.load(f)

        # Parse CSV, build index
        self.index = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                if row_idx % index_num_shards != index_shard_id:
                    continue
                vid_id = row["id"]
                relative_path = row["video path"].replace("videos/", "")
                video_path = (
                    join_remote(video_root, relative_path)
                    if is_remote_path(video_root)
                    else os.path.join(video_root, relative_path)
                )
                if (
                        check_files
                        and not is_remote_path(video_path)
                        and not os.path.exists(video_path)):
                    continue
                num_frames = int(row["num frames"])
                fps = float(row.get("fps", 0) or 0)
                caption = self.annotations.get(vid_id, {}).get("caption", "")

                # Depth zip path: replace videos/ with depths/ and .mp4 with .zip
                depth_zip_path = ""
                if depth_root:
                    depth_rel = relative_path.replace(".mp4", ".zip")
                    depth_zip_path = (
                        join_remote(depth_root, depth_rel)
                        if is_remote_path(depth_root)
                        else os.path.join(depth_root, depth_rel)
                    )

                self.index.append({
                    "video_id": vid_id,
                    "video_path": video_path,
                    "num_frames": num_frames,
                    "fps": fps,
                    "caption": caption,
                    "depth_zip_path": depth_zip_path,
                })
                if max_videos > 0 and len(self.index) >= max_videos:
                    break

        depth_info = f", depth_root={depth_root}" if depth_root else ""
        jitter_info = f", temporal_jitter={self.temporal_jitter}"
        shard_info = (
            f", index_shard={index_shard_id}/{index_num_shards}"
            if index_num_shards > 1 else "")
        print(f"SpatialVidDataset: {len(self.index)} videos, "
              f"{len(self)} clips, clips_per_video={self.clips_per_video}, "
              f"seq_len={seq_len}, "
              f"target_size={target_size}{depth_info}{jitter_info}{shard_info}")

    def __len__(self):
        return len(self.index) * self.clips_per_video

    def _load_entry(self, idx):
        video_idx = idx // self.clips_per_video
        window_index = idx % self.clips_per_video
        entry = self.index[video_idx]
        nf = self.num_frames_per_video

        # Compute frame indices once, use for both video and depth
        indices = _compute_frame_indices(
            entry["num_frames"], nf,
            temporal_jitter=self.temporal_jitter,
            max_frame_span=self.max_frame_span,
            fps=entry["fps"],
            clip_duration_seconds=self.clip_duration_seconds,
            window_index=window_index,
            num_windows=self.clips_per_video,
        )

        video_path = entry["video_path"]
        try:
            video_path = stage_remote_file(
                video_path,
                self.remote_video_cache,
                max_cache_bytes=self.remote_cache_max_bytes,
                retries=self.remote_download_retries,
            )
            frames = read_video_frames(
                video_path, nf, self.target_size,
                temporal_jitter=False, frame_indices=indices,
            )
        except Exception as exc:
            if isinstance(exc, VideoDecodeError):
                raise
            raise VideoDecodeError(
                f"Unable to stage/decode {entry['video_path']}: {exc}") from exc

        depth = None
        if entry["depth_zip_path"]:
            depth_path = entry["depth_zip_path"]
            if is_remote_path(depth_path):
                try:
                    depth_path = stage_remote_file(
                        depth_path,
                        self.remote_depth_cache,
                        max_cache_bytes=self.remote_cache_max_bytes,
                        retries=self.remote_download_retries,
                    )
                except Exception:
                    depth_path = ""
            depth = read_depth_frames(
                depth_path, indices, nf, self.target_size,
            )

        return {
            "frames": frames,         # [S, 3, H, W]
            "caption": entry["caption"],
            "video_id": entry["video_id"],
            "window_index": window_index,
            "depth": depth,           # [S, H, W] or None
        }

    def __getitem__(self, idx):
        if not self.index:
            raise IndexError("SpatialVidDataset is empty")

        requested_idx = int(idx)
        last_error = None
        # A large odd stride avoids repeatedly selecting neighboring rows,
        # which are often produced by the same source and can fail together.
        retry_stride = 104729
        for attempt in range(self.decode_retries + 1):
            candidate_idx = (
                requested_idx + attempt * retry_stride) % len(self)
            try:
                sample = self._load_entry(candidate_idx)
                requested_video_idx = (
                    requested_idx // self.clips_per_video)
                sample["requested_video_id"] = self.index[
                    requested_video_idx]["video_id"]
                sample["decode_replacement"] = int(candidate_idx != requested_idx)
                sample["decode_error"] = "" if last_error is None else str(last_error)
                return sample
            except VideoDecodeError as exc:
                last_error = exc

        requested_path = self.index[
            requested_idx // self.clips_per_video]["video_path"]
        raise VideoDecodeError(
            f"Unable to load {requested_path} or any of "
            f"{self.decode_retries} deterministic replacements. "
            f"Last error: {last_error}")


def collate_fn(batch):
    frames = torch.stack([b["frames"] for b in batch], dim=0)  # [B, S, 3, H, W]
    captions = [b["caption"] for b in batch]
    video_ids = [b["video_id"] for b in batch]
    depths = [b["depth"] for b in batch]
    depth_valid = torch.tensor(
        [depth is not None for depth in depths], dtype=torch.bool)
    depth_tensor = None
    if any(d is not None for d in depths):
        reference = next(depth for depth in depths if depth is not None)
        depth_tensor = torch.stack([
            depth if depth is not None else torch.zeros_like(reference)
            for depth in depths
        ], dim=0)  # [B, S, H, W]
    return {
        "frames": frames,
        "caption": captions,
        "video_id": video_ids,
        "window_index": [
            int(b.get("window_index", 0)) for b in batch
        ],
        "requested_video_id": [
            b.get("requested_video_id", b["video_id"]) for b in batch
        ],
        "decode_replacements": sum(
            int(b.get("decode_replacement", 0)) for b in batch
        ),
        "decode_errors": [
            b.get("decode_error", "") for b in batch
            if b.get("decode_error", "")
        ],
        "depth": depth_tensor,
        "depth_valid": depth_valid,
    }


class CachedTokenDataset(Dataset):
    """Dataset that loads pre-cached tokens from disk."""

    def __init__(self, csv_path, cache_dir, levels, annotation_index_path="",
                 max_videos=0):
        self.cache_dir = cache_dir
        self.extra_captions = {}
        if annotation_index_path and os.path.exists(annotation_index_path):
            with open(annotation_index_path) as f:
                self.extra_captions = json.load(f)

        self.index = []
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row["id"]
                    cache_path = os.path.join(cache_dir, f"{vid}.pt")
                    if os.path.exists(cache_path):
                        caption = self.extra_captions.get(vid, {}).get("caption", "")
                        self.index.append({"video_id": vid, "path": cache_path, "caption": caption})

        if max_videos > 0:
            self.index = self.index[:max_videos]
        print(f"CachedTokenDataset: {len(self)} videos from {cache_dir}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        data = torch.load(entry["path"], map_location="cpu")
        return {
            "tokens": data["tokens"],
            "caption": data.get("caption", entry["caption"]),
            "video_id": entry["video_id"],
        }


def token_collate_fn(batch):
    tokens = torch.stack([b["tokens"] for b in batch], dim=0)
    captions = [b["caption"] for b in batch]
    return {"tokens": tokens, "caption": captions}
