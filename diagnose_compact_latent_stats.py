#!/usr/bin/env python3
"""Audit compact I0 and future-residual latent distributions."""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
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
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_pt", required=True)
    parser.add_argument("--num_videos", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--max_frame_span", type=int, default=32)
    parser.add_argument("--clip_duration_seconds", type=float, default=0.0)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--latent_grid", type=int, default=18)
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[4, 11, 17, 23])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--cov_samples_per_video", type=int, default=256)
    parser.add_argument(
        "--convergence_points", type=int, nargs="+",
        default=[4, 8, 16, 32, 64],
        help="Video counts at which to snapshot frame/channel mean and std")
    parser.add_argument("--disable_temporal_mixer", action="store_true")
    parser.add_argument(
        "--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    return parser.parse_args()


def quantile_summary(values):
    values = values.detach().double().cpu().flatten()
    quantiles = torch.tensor(
        [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0],
        dtype=torch.float64,
    )
    result = torch.quantile(values, quantiles)
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p01", "p05", "p50", "p95", "p99", "max"),
            result,
        )
    }


class LatentMoments:
    def __init__(self, dim, num_frames, num_tokens):
        self.dim = dim
        self.count = 0
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.sum2 = torch.zeros(dim, dtype=torch.float64)
        self.sum3 = torch.zeros(dim, dtype=torch.float64)
        self.sum4 = torch.zeros(dim, dtype=torch.float64)
        self.max_abs = 0.0
        self.frame_count = torch.zeros(num_frames, dtype=torch.float64)
        self.frame_sum = torch.zeros(num_frames, dtype=torch.float64)
        self.frame_sum2 = torch.zeros(num_frames, dtype=torch.float64)
        self.frame_channel_count = torch.zeros(
            num_frames, 1, dtype=torch.float64)
        self.frame_channel_sum = torch.zeros(
            num_frames, dim, dtype=torch.float64)
        self.frame_channel_sum2 = torch.zeros(
            num_frames, dim, dtype=torch.float64)
        self.spatial_count = torch.zeros(num_tokens, dtype=torch.float64)
        self.spatial_sum2 = torch.zeros(num_tokens, dtype=torch.float64)
        self.cov_count = 0
        self.cov_sum = torch.zeros(dim, dtype=torch.float64)
        self.cov_xtx = torch.zeros(dim, dim, dtype=torch.float64)

    def update(self, tensor, cov_samples):
        # tensor: [B, S, N, D]
        values = tensor.float()
        flat = values.reshape(-1, self.dim)
        self.count += flat.shape[0]
        self.sum += flat.sum(dim=0).double().cpu()
        self.sum2 += flat.square().sum(dim=0).double().cpu()
        self.sum3 += flat.pow(3).sum(dim=0).double().cpu()
        self.sum4 += flat.pow(4).sum(dim=0).double().cpu()
        self.max_abs = max(self.max_abs, float(flat.abs().max()))

        frame_values = values.permute(1, 0, 2, 3).reshape(
            values.shape[1], -1)
        self.frame_count += frame_values.shape[1]
        self.frame_sum += frame_values.sum(dim=1).double().cpu()
        self.frame_sum2 += frame_values.square().sum(dim=1).double().cpu()
        frame_channel_values = values.permute(1, 0, 2, 3)
        self.frame_channel_count += (
            values.shape[0] * values.shape[2])
        self.frame_channel_sum += frame_channel_values.sum(
            dim=(1, 2)).double().cpu()
        self.frame_channel_sum2 += frame_channel_values.square().sum(
            dim=(1, 2)).double().cpu()

        spatial_values = values.permute(2, 0, 1, 3).reshape(
            values.shape[2], -1)
        self.spatial_count += spatial_values.shape[1]
        self.spatial_sum2 += spatial_values.square().sum(dim=1).double().cpu()

        if cov_samples > 0:
            sample_count = min(cov_samples, flat.shape[0])
            indices = torch.linspace(
                0, flat.shape[0] - 1, sample_count,
                device=flat.device,
            ).round().long()
            sample = flat.index_select(0, indices)
            self.cov_count += sample.shape[0]
            self.cov_sum += sample.sum(dim=0).double().cpu()
            self.cov_xtx += (sample.T @ sample).double().cpu()

    def frame_channel_stats(self):
        count = self.frame_channel_count.clamp_min(1)
        mean = self.frame_channel_sum / count
        variance = (
            self.frame_channel_sum2 / count - mean.square()
        ).clamp_min(1e-12)
        return mean.float(), variance.sqrt().float()

    def finalize(self):
        count = max(self.count, 1)
        mean = self.sum / count
        raw2 = self.sum2 / count
        raw3 = self.sum3 / count
        raw4 = self.sum4 / count
        variance = (raw2 - mean.square()).clamp_min(1e-12)
        std = variance.sqrt()
        central3 = raw3 - 3 * mean * raw2 + 2 * mean.pow(3)
        central4 = (
            raw4 - 4 * mean * raw3
            + 6 * mean.square() * raw2 - 3 * mean.pow(4)
        )
        skew = central3 / std.pow(3).clamp_min(1e-12)
        excess_kurtosis = central4 / variance.square().clamp_min(1e-12) - 3

        global_mean = float(self.sum.sum() / (count * self.dim))
        global_second = float(self.sum2.sum() / (count * self.dim))
        global_std = math.sqrt(max(global_second - global_mean ** 2, 0.0))

        frame_mean = self.frame_sum / self.frame_count.clamp_min(1)
        frame_var = (
            self.frame_sum2 / self.frame_count.clamp_min(1)
            - frame_mean.square()
        ).clamp_min(0)
        spatial_rms = (
            self.spatial_sum2 / self.spatial_count.clamp_min(1)
        ).sqrt()
        frame_channel_mean, frame_channel_std = self.frame_channel_stats()

        cov_mean = self.cov_sum / max(self.cov_count, 1)
        covariance = (
            self.cov_xtx / max(self.cov_count, 1)
            - torch.outer(cov_mean, cov_mean)
        )
        covariance = (covariance + covariance.T) * 0.5
        cov_std = covariance.diag().clamp_min(1e-12).sqrt()
        correlation = covariance / torch.outer(cov_std, cov_std)
        eigenvalues = torch.linalg.eigvalsh(correlation).clamp_min(0)
        eigenvalues = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
        nonzero = eigenvalues[eigenvalues > 0]
        effective_rank = float(
            torch.exp(-(nonzero * nonzero.log()).sum()))
        descending = eigenvalues.flip(0)

        std_summary = quantile_summary(std)
        std_ratio = std_summary["p99"] / max(std_summary["p01"], 1e-12)
        mean_over_std = mean.abs() / std.clamp_min(1e-12)
        report = {
            "num_values_per_channel": self.count,
            "covariance_samples": self.cov_count,
            "global_mean": global_mean,
            "global_std": global_std,
            "global_max_abs": self.max_abs,
            "channel_mean": quantile_summary(mean),
            "channel_std": std_summary,
            "channel_std_p99_over_p01": std_ratio,
            "channel_abs_mean_over_std": quantile_summary(mean_over_std),
            "channel_skew": quantile_summary(skew),
            "channel_excess_kurtosis": quantile_summary(excess_kurtosis),
            "fraction_std_below_1e-3": float((std < 1e-3).double().mean()),
            "fraction_std_below_1e-2": float((std < 1e-2).double().mean()),
            "frame_mean": frame_mean.tolist(),
            "frame_std": frame_var.sqrt().tolist(),
            "frame_channel_std": {
                "min": float(frame_channel_std.min()),
                "median": float(frame_channel_std.median()),
                "max": float(frame_channel_std.max()),
            },
            "spatial_rms": {
                **quantile_summary(spatial_rms),
                "coefficient_of_variation": float(
                    spatial_rms.std(unbiased=False)
                    / spatial_rms.mean().clamp_min(1e-12)),
            },
            "channel_correlation": {
                "effective_rank": effective_rank,
                "effective_rank_fraction": effective_rank / self.dim,
                "top1_eigen_fraction": float(descending[:1].sum()),
                "top10_eigen_fraction": float(descending[:10].sum()),
                "top32_eigen_fraction": float(descending[:32].sum()),
            },
        }
        tensors = {
            "mean": mean.float(),
            "std": std.float(),
            "skew": skew.float(),
            "excess_kurtosis": excess_kurtosis.float(),
            "frame_mean": frame_mean.float(),
            "frame_std": frame_var.sqrt().float(),
            "frame_channel_mean": frame_channel_mean,
            "frame_channel_std": frame_channel_std,
            "spatial_rms": spatial_rms.float(),
            "correlation_eigenvalues": eigenvalues.float(),
        }
        return report, tensors


def main():
    args = parse_args()
    device_type = get_device_name()
    device = get_device()
    dtype = resolve_dtype(args.dtype)

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
    encoder = encoder.to(device=device, dtype=dtype).eval()

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

    num_tokens = args.latent_grid ** 2
    cond_stats = LatentMoments(
        args.latent_dim, num_frames=1, num_tokens=num_tokens)
    target_stats = LatentMoments(
        args.latent_dim,
        num_frames=args.seq_len - 1,
        num_tokens=num_tokens,
    )
    video_ids = []
    convergence_points = {
        point for point in args.convergence_points
        if 0 < point <= min(args.num_videos, len(dataset))
    }
    convergence_snapshots = {}
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Auditing compact latent stats"):
            frames = batch["frames"].to(
                device=device, dtype=dtype,
                non_blocking=device_type == "cuda")
            with torch.autocast(
                    device_type=device_type, dtype=dtype,
                    enabled=dtype != torch.float32):
                clip_tokens, clip_psi = encoder(frames)
                clip_tokens = strip_special_tokens(clip_tokens, clip_psi)
                i0_tokens, i0_psi = encoder(frames[:, :1])
                i0_tokens = strip_special_tokens(i0_tokens, i0_psi)
                _, clip_flat = tokenizer(clip_tokens)
                _, i0_flat = tokenizer(i0_tokens)
            target = (
                clip_flat[:, 1:]
                - i0_flat.expand(-1, args.seq_len - 1, -1, -1)
            )
            cond_stats.update(i0_flat, args.cov_samples_per_video)
            target_stats.update(target, args.cov_samples_per_video)
            video_ids.extend(batch["video_id"])
            if len(video_ids) in convergence_points:
                cond_mean, cond_std = cond_stats.frame_channel_stats()
                target_mean, target_std = target_stats.frame_channel_stats()
                convergence_snapshots[len(video_ids)] = {
                    "cond_mean": cond_mean.clone(),
                    "cond_std": cond_std.clone(),
                    "target_mean": target_mean.clone(),
                    "target_std": target_std.clone(),
                }

    cond_report, cond_tensors = cond_stats.finalize()
    target_report, target_tensors = target_stats.finalize()
    convergence = {}
    final_target_mean = target_tensors["frame_channel_mean"]
    final_target_std = target_tensors["frame_channel_std"]
    final_cond_mean = cond_tensors["frame_channel_mean"]
    final_cond_std = cond_tensors["frame_channel_std"]
    for count, snapshot in sorted(convergence_snapshots.items()):
        target_std_relative = (
            (snapshot["target_std"] - final_target_std).abs()
            / final_target_std.clamp_min(1e-6)
        )
        cond_std_relative = (
            (snapshot["cond_std"] - final_cond_std).abs()
            / final_cond_std.clamp_min(1e-6)
        )
        convergence[str(count)] = {
            "target_mean_mae": float(
                (snapshot["target_mean"] - final_target_mean).abs().mean()),
            "target_std_relative_mae": float(
                target_std_relative.mean()),
            "target_std_relative_p95": float(
                torch.quantile(target_std_relative.flatten(), 0.95)),
            "cond_mean_mae": float(
                (snapshot["cond_mean"] - final_cond_mean).abs().mean()),
            "cond_std_relative_mae": float(cond_std_relative.mean()),
            "cond_std_relative_p95": float(
                torch.quantile(cond_std_relative.flatten(), 0.95)),
        }
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
        "cond": cond_report,
        "target": target_report,
        "normalization_convergence_vs_final": convergence,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as output:
        json.dump(report, output, indent=2)
    torch.save(
        {
            "configuration": report["configuration"],
            "cond": cond_tensors,
            "target": target_tensors,
            "normalization_convergence_vs_final": convergence,
        },
        args.output_pt,
    )

    print(
        "Target global std: "
        f"{target_report['global_std']:.6f}; channel std p01/p50/p99: "
        f"{target_report['channel_std']['p01']:.6f}/"
        f"{target_report['channel_std']['p50']:.6f}/"
        f"{target_report['channel_std']['p99']:.6f}")
    print(
        "Target channel std p99/p01: "
        f"{target_report['channel_std_p99_over_p01']:.3f}; "
        "correlation effective rank: "
        f"{target_report['channel_correlation']['effective_rank']:.1f}/"
        f"{args.latent_dim}")
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved tensors: {args.output_pt}")


if __name__ == "__main__":
    main()
