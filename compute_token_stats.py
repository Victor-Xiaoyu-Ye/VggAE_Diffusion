"""Compute per-level channel-wise mean and variance of StreamVGGT tokens from SpatialVid.

Uses Welford's online algorithm so the full dataset never needs to be in memory.

Output (torch.save):
    {"mean_4": tensor[2048], "var_4": tensor[2048],
     "mean_11": tensor[2048], "var_11": tensor[2048], ...}
"""

import argparse
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

from streamvggt.models.streamvggt import StreamVGGT
from data.video_dataset import SpatialVidDataset, collate_fn
from data.token_utils import DPT_LEVELS, strip_special_tokens


def parse_args():
    p = argparse.ArgumentParser(description="Compute token channel stats via Welford's algorithm")
    p.add_argument("--csv_path", type=str, required=True, help="SpatialVid CSV index")
    p.add_argument("--video_root", type=str, required=True, help="Root dir of mp4 videos")
    p.add_argument("--streamvggt_ckpt", type=str, required=True, help="StreamVGGT checkpoint path")
    p.add_argument("--out_path", type=str, required=True, help="Output .pt file for stats")
    p.add_argument("--seq_len", type=int, default=8, help="Number of frames per video sample")
    p.add_argument("--max_batches", type=int, default=2000, help="Max batches (0 = no limit)")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--target_size", type=int, default=518, help="Resize target for frames")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"],
                    help="Compute dtype for encoder forward pass")
    return p.parse_args()


def load_encoder(ckpt_path, device, dtype):
    """Load frozen StreamVGGT encoder."""
    encoder = StreamVGGT().to(device, dtype=dtype)
    encoder.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    encoder.load_state_dict(ckpt, strict=False)
    return encoder


def main():
    args = parse_args()

    # dtype mapping
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    compute_dtype = dtype_map[args.dtype]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, dtype: {compute_dtype}")

    # Load encoder (frozen)
    encoder = load_encoder(args.streamvggt_ckpt, device, compute_dtype)
    print(f"Encoder loaded from {args.streamvggt_ckpt}")

    # Dataset & dataloader -- workers decode mp4, main process runs encoder
    dataset = SpatialVidDataset(
        csv_path=args.csv_path,
        video_root=args.video_root,
        seq_len=args.seq_len,
        target_size=args.target_size,
        num_frames_per_video=args.seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # Welford accumulators per level, per channel (dim 2048)
    # We keep everything in float64 for numerical stability.
    channel_dim = 2048  # 2 * embed_dim (1024)
    count = {lvl: 0 for lvl in DPT_LEVELS}
    mean = {lvl: torch.zeros(channel_dim, dtype=torch.float64, device=device) for lvl in DPT_LEVELS}
    M2 = {lvl: torch.zeros(channel_dim, dtype=torch.float64, device=device) for lvl in DPT_LEVELS}

    batch_idx = 0
    print(f"Starting Welford accumulation (max_batches={args.max_batches}) ...")

    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device, dtype=compute_dtype)  # [B, S, 3, H, W]

            aggregated_tokens_list, patch_start_idx = encoder(frames)
            aggregated_tokens_list = strip_special_tokens(aggregated_tokens_list, patch_start_idx)

            # Update Welford stats for each DPT level
            for lvl in DPT_LEVELS:
                patch_tokens = aggregated_tokens_list[lvl]

                # Flatten all spatial dims to a single sample axis
                # [B, S, num_patches, 2048] -> [N, 2048]
                flat = patch_tokens.reshape(-1, channel_dim).to(torch.float64)

                n = flat.shape[0]
                if n == 0:
                    continue

                # Batch mean
                batch_mean = flat.mean(dim=0)

                # Welford update
                old_count = count[lvl]
                new_count = old_count + n
                delta = batch_mean - mean[lvl]
                mean[lvl] = mean[lvl] + delta * (n / new_count)
                delta2 = batch_mean - mean[lvl]
                M2[lvl] = M2[lvl] + flat.var(dim=0, unbiased=False) * n + delta * delta2 * (old_count * n / new_count)
                count[lvl] = new_count

            batch_idx += 1
            if batch_idx % 50 == 0:
                print(f"  batch {batch_idx} done")

            if 0 < args.max_batches <= batch_idx:
                print(f"Reached max_batches={args.max_batches}, stopping.")
                break

    # Finalize: convert M2 -> population variance
    stats = {}
    for lvl in DPT_LEVELS:
        n = count[lvl]
        if n == 0:
            print(f"WARNING: level {lvl} has 0 samples")
            var = torch.zeros(channel_dim)
        else:
            var = M2[lvl] / n  # population variance (not Bessel-corrected)
        stats[f"mean_{lvl}"] = mean[lvl].float()
        stats[f"var_{lvl}"] = var.float()
        print(f"Level {lvl}: n={n}, mean_range=[{mean[lvl].min():.4f}, {mean[lvl].max():.4f}], "
              f"var_range=[{var.min():.4f}, {var.max():.4f}]")

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save(stats, args.out_path)
    print(f"Stats saved to {args.out_path}")


if __name__ == "__main__":
    main()
