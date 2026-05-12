import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.video_io import read_video_frames, read_depth_frames, _compute_frame_indices


class SpatialVidDataset(Dataset):
    """Dataset for SpatialVid mp4 videos.

    Workers decode mp4 and resize. When depth_root is provided, loads per-frame
    depth from zip archives with frame-level alignment to video frames.
    """

    def __init__(self, csv_path, video_root, seq_len=8, target_size=518,
                 annotation_index_path="", max_videos=0, num_frames_per_video=8,
                 depth_root="", temporal_jitter=True):
        self.target_size = target_size
        self.seq_len = seq_len
        self.num_frames_per_video = num_frames_per_video
        self.depth_root = depth_root
        self.temporal_jitter = temporal_jitter and not depth_root  # disable jitter when depth needed for frame alignment
        # TODO: re-enable jitter with depth once frame index alignment is verified
        if depth_root:
            self.temporal_jitter = False

        # Load annotation index
        self.annotations = {}
        if annotation_index_path and os.path.exists(annotation_index_path):
            with open(annotation_index_path) as f:
                self.annotations = json.load(f)

        # Parse CSV, build index
        self.index = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid_id = row["id"]
                relative_path = row["video path"].replace("videos/", "")
                video_path = os.path.join(video_root, relative_path)
                if not os.path.exists(video_path):
                    continue
                num_frames = int(row["num frames"])
                caption = self.annotations.get(vid_id, {}).get("caption", "")

                # Depth zip path: replace videos/ with depths/ and .mp4 with .zip
                depth_zip_path = ""
                if depth_root:
                    depth_rel = relative_path.replace(".mp4", ".zip")
                    depth_zip_path = os.path.join(depth_root, depth_rel)

                self.index.append({
                    "video_id": vid_id,
                    "video_path": video_path,
                    "num_frames": num_frames,
                    "caption": caption,
                    "depth_zip_path": depth_zip_path,
                })

        if max_videos > 0:
            self.index = self.index[:max_videos]

        depth_info = f", depth_root={depth_root}" if depth_root else ""
        jitter_info = f", temporal_jitter={self.temporal_jitter}"
        print(f"SpatialVidDataset: {len(self)} videos, seq_len={seq_len}, "
              f"target_size={target_size}{depth_info}{jitter_info}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        nf = self.num_frames_per_video

        # Compute frame indices once, use for both video and depth
        indices = _compute_frame_indices(
            entry["num_frames"], nf,
            temporal_jitter=self.temporal_jitter,
        )

        frames = read_video_frames(
            entry["video_path"], nf, self.target_size,
            temporal_jitter=False, frame_indices=indices,
        )

        depth = None
        if entry["depth_zip_path"]:
            depth = read_depth_frames(
                entry["depth_zip_path"], indices, nf, self.target_size,
            )

        return {
            "frames": frames,         # [S, 3, H, W]
            "caption": entry["caption"],
            "video_id": entry["video_id"],
            "depth": depth,           # [S, H, W] or None
        }


def collate_fn(batch):
    frames = torch.stack([b["frames"] for b in batch], dim=0)  # [B, S, 3, H, W]
    captions = [b["caption"] for b in batch]
    video_ids = [b["video_id"] for b in batch]
    depths = [b["depth"] for b in batch]
    depth_tensor = None
    if all(d is not None for d in depths):
        depth_tensor = torch.stack(depths, dim=0)  # [B, S, H, W]
    return {"frames": frames, "caption": captions, "video_id": video_ids, "depth": depth_tensor}


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
