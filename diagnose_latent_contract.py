#!/usr/bin/env python3
"""Measure single-frame versus in-clip latent consistency.

The active diffusion contract conditions on an independently encoded I0 while
future targets come from a complete clip. This diagnostic quantifies how much
the first-frame representation changes between those two encoding paths.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.token_utils import strip_special_tokens
from data.video_dataset import SpatialVidDataset, collate_fn
from models.generative_tokenizer import GenerativeTokenizer
from streamvggt.models.streamvggt import StreamVGGT
from utils.device import get_device, get_device_name, resolve_dtype


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--autoencoder_ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_videos", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--max_frame_span", type=int, default=32)
    parser.add_argument("--clip_duration_seconds", type=float, default=0.0)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--latent_grid", type=int, default=18)
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[4, 11, 17, 23])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--disable_temporal_mixer",
        action="store_true",
        help="Replace the tokenizer temporal mixer with identity for ablation.",
    )
    return parser.parse_args()


def compare_tensors(single, clip):
    single = single.float()
    clip = clip.float()
    diff = single - clip
    single_flat = single.reshape(single.shape[0], -1)
    clip_flat = clip.reshape(clip.shape[0], -1)
    diff_flat = diff.reshape(diff.shape[0], -1)
    return {
        "mae": diff_flat.abs().mean(dim=1),
        "rmse": diff_flat.square().mean(dim=1).sqrt(),
        "relative_l2": (
            diff_flat.norm(dim=1) / clip_flat.norm(dim=1).clamp_min(1e-12)),
        "cosine": F.cosine_similarity(single_flat, clip_flat, dim=1),
        "single_std": single_flat.std(dim=1, unbiased=False),
        "clip_std": clip_flat.std(dim=1, unbiased=False),
    }


def append_metrics(storage, prefix, metrics):
    for name, values in metrics.items():
        storage.setdefault(f"{prefix}/{name}", []).extend(
            values.detach().cpu().tolist())


def summarize(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
    }


def main():
    args = parse_args()
    device_type = get_device_name()
    if device_type == "cpu":
        raise RuntimeError(
            "Latent contract diagnostics require an accelerator")
    device = get_device(0)
    compute_dtype = resolve_dtype("fp16")

    dataset = SpatialVidDataset(
        csv_path=args.csv,
        video_root=args.video_root,
        seq_len=args.seq_len,
        target_size=args.target_size,
        max_videos=args.num_videos,
        num_frames_per_video=args.seq_len,
        temporal_jitter=False,
        max_frame_span=args.max_frame_span,
        clip_duration_seconds=args.clip_duration_seconds,
    )
    if not dataset:
        raise RuntimeError("No valid videos found")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device_type == "cuda",
    )

    encoder = StreamVGGT(
        img_size=args.target_size, patch_size=14, embed_dim=1024)
    encoder.load_state_dict(
        torch.load(args.encoder_ckpt, map_location="cpu"), strict=False)
    encoder = encoder.to(device=device, dtype=compute_dtype).eval()

    tokenizer = GenerativeTokenizer(
        latent_dim=args.latent_dim,
        latent_grid=args.latent_grid,
        levels=args.levels,
        seq_len=args.seq_len,
        input_grid=args.target_size // 14,
    ).to(device=device)
    checkpoint = torch.load(
        args.autoencoder_ckpt, map_location="cpu", weights_only=False)
    tokenizer.load_state_dict(checkpoint["tokenizer"])
    if args.disable_temporal_mixer:
        tokenizer.disable_temporal_mixer = True
    tokenizer.eval()

    metrics = {}
    video_ids = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Diagnosing latent contract"):
            frames = batch["frames"].to(
                device=device, dtype=compute_dtype, non_blocking=True)
            with torch.autocast(
                    device_type=device_type, dtype=compute_dtype):
                clip_tokens, clip_psi = encoder(frames)
                clip_tokens = strip_special_tokens(clip_tokens, clip_psi)
                single_tokens, single_psi = encoder(frames[:, :1])
                single_tokens = strip_special_tokens(
                    single_tokens, single_psi)

            for level in args.levels:
                append_metrics(
                    metrics,
                    f"encoder_level_{level}",
                    compare_tensors(
                        single_tokens[level][:, 0],
                        clip_tokens[level][:, 0],
                    ),
                )

            with torch.autocast(
                    device_type=device_type, dtype=compute_dtype):
                _, clip_flat = tokenizer(clip_tokens)
                _, single_flat = tokenizer(single_tokens)
            compact_metrics = compare_tensors(
                single_flat[:, 0], clip_flat[:, 0])
            append_metrics(metrics, "compact_i0", compact_metrics)

            future_residual = (
                clip_flat[:, 1:]
                - single_flat.expand(-1, clip_flat.shape[1] - 1, -1, -1)
            ).float()
            residual_rms = future_residual.square().mean(
                dim=(1, 2, 3)).sqrt()
            drift_rms = compact_metrics["rmse"]
            metrics.setdefault("future_residual/rms", []).extend(
                residual_rms.cpu().tolist())
            metrics.setdefault(
                "compact_i0/drift_to_future_residual_rms", []).extend(
                    (drift_rms / residual_rms.clamp_min(1e-12)).cpu().tolist())
            video_ids.extend(batch["video_id"])

    report = {
        "num_videos": len(video_ids),
        "video_ids": video_ids,
        "configuration": {
            "seq_len": args.seq_len,
            "target_size": args.target_size,
            "max_frame_span": args.max_frame_span,
            "clip_duration_seconds": args.clip_duration_seconds,
            "latent_dim": args.latent_dim,
            "latent_grid": args.latent_grid,
            "levels": args.levels,
            "disable_temporal_mixer": args.disable_temporal_mixer,
            "encoder_ckpt": os.path.abspath(args.encoder_ckpt),
            "autoencoder_ckpt": os.path.abspath(args.autoencoder_ckpt),
        },
        "metrics": {
            name: summarize(values)
            for name, values in sorted(metrics.items())
        },
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as output_file:
        json.dump(report, output_file, indent=2)

    encoder_rel = [
        report["metrics"][f"encoder_level_{level}/relative_l2"]["mean"]
        for level in args.levels
    ]
    compact_rel = report["metrics"]["compact_i0/relative_l2"]["mean"]
    drift_ratio = report["metrics"][
        "compact_i0/drift_to_future_residual_rms"]["mean"]
    print(f"Mean encoder I0 relative L2: {sum(encoder_rel) / len(encoder_rel):.6f}")
    print(f"Compact I0 relative L2: {compact_rel:.6f}")
    print(f"Compact drift / future residual RMS: {drift_ratio:.6f}")
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
