#!/usr/bin/env python3
"""Validate scale-stage checkpoints and latent-cache contracts."""

import argparse
import io
import json
import os

import torch

from utils.moxing_io import (
    is_remote_path,
    join_remote,
    read_bytes,
    read_text,
    remote_exists,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("after_i0", "before_merge", "before_diffusion",
                 "before_sample"),
    )
    parser.add_argument("--encoder_ckpt", default="")
    parser.add_argument("--autoencoder_ckpt", default="")
    parser.add_argument("--i0_decoder_ckpt", default="")
    parser.add_argument("--diffusion_ckpt", default="")
    parser.add_argument("--train_cache_dir", default="")
    parser.add_argument("--eval_cache_dir", default="")
    parser.add_argument("--cache_partitions", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--latent_grid", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=8)
    return parser.parse_args()


def require_file(path, label):
    if not path:
        raise ValueError(f"{label} path is empty")
    exists = remote_exists(path) if is_remote_path(path) else os.path.isfile(path)
    if not exists:
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not is_remote_path(path) and os.path.getsize(path) <= 0:
        raise RuntimeError(f"Empty {label}: {path}")
    print(f"[PASS] {label}: {path}")


def load_torch(path):
    source = io.BytesIO(read_bytes(path)) if is_remote_path(path) else path
    return torch.load(source, map_location="cpu", weights_only=False)


def value(args, name, default=None):
    return args.get(name, default) if isinstance(args, dict) else default


def validate_representation_args(saved, expected, label):
    mismatches = []
    for key, expected_value in expected.items():
        actual = value(saved, key)
        if key == "disable_temporal_mixer":
            actual = bool(actual)
        if actual != expected_value:
            mismatches.append(f"{key}={actual!r}, expected {expected_value!r}")
    if mismatches:
        raise RuntimeError(f"{label} contract mismatch: " + "; ".join(mismatches))
    print(f"[PASS] {label} representation contract")


def validate_autoencoder(path, expected):
    require_file(path, "geometry autoencoder checkpoint")
    checkpoint = load_torch(path)
    for key in ("tokenizer", "decoder", "args"):
        if key not in checkpoint:
            raise KeyError(f"Autoencoder checkpoint missing key: {key}")
    validate_representation_args(
        checkpoint["args"], expected, "geometry autoencoder")


def validate_i0(path, expected):
    require_file(path, "I0 decoder checkpoint")
    checkpoint = load_torch(path)
    for key in ("app_cnn", "decoder", "args"):
        if key not in checkpoint:
            raise KeyError(f"I0 checkpoint missing key: {key}")
    validate_representation_args(checkpoint["args"], expected, "I0 decoder")


def validate_partition(cache_dir, partition_id, partition_count):
    partition = join_remote(
        cache_dir,
        f"part-{partition_id:05d}-of-{partition_count:05d}",
    )
    for name in ("_SUCCESS", "manifest.txt", "stats.pt", "config.json"):
        require_file(join_remote(partition, name), f"cache partition {name}")
    marker = json.loads(read_text(join_remote(partition, "_SUCCESS")))
    if int(marker.get("partition_id", -1)) != partition_id:
        raise RuntimeError(f"Invalid partition marker: {partition}")
    if int(marker.get("num_partitions", -1)) != partition_count:
        raise RuntimeError(f"Partition-count mismatch: {partition}")


def validate_merged_cache(cache_dir, expected, label):
    manifest_path = join_remote(cache_dir, "manifest.txt")
    stats_path = join_remote(cache_dir, "stats.pt")
    require_file(manifest_path, f"{label} manifest")
    require_file(stats_path, f"{label} statistics")

    shards = [
        line.strip() for line in read_text(manifest_path).splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not shards:
        raise RuntimeError(f"{label} manifest contains no shards")
    for shard in (shards[:2] + shards[-1:]):
        require_file(shard, f"{label} latent shard")

    stats = load_torch(stats_path)
    if int(stats.get("normalization_version", -1)) != 2:
        raise RuntimeError(f"{label} uses unsupported normalization stats")
    representation = stats.get("representation") or {}
    validate_representation_args(representation, expected, f"{label} cache")
    for group, frames in (("target", expected["seq_len"] - 1), ("cond", 1)):
        values = stats.get(group) or {}
        mean = values.get("mean")
        std = values.get("std")
        expected_shape = (frames, expected["latent_dim"])
        if not isinstance(mean, torch.Tensor) or tuple(mean.shape) != expected_shape:
            raise RuntimeError(
                f"{label} {group}.mean shape is "
                f"{getattr(mean, 'shape', None)}, expected {expected_shape}")
        if not isinstance(std, torch.Tensor) or tuple(std.shape) != expected_shape:
            raise RuntimeError(
                f"{label} {group}.std shape is "
                f"{getattr(std, 'shape', None)}, expected {expected_shape}")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise RuntimeError(f"{label} {group} statistics contain NaN/Inf")
        if (std <= 0).any():
            raise RuntimeError(f"{label} {group} statistics contain nonpositive std")
    print(
        f"[PASS] {label}: samples={stats.get('num_samples', '?')} "
        f"failed={stats.get('num_failed', '?')} shards={len(shards)}")
    return stats


def validate_diffusion(path, expected):
    require_file(path, "Compact DiT checkpoint")
    checkpoint = load_torch(path)
    for key in ("model", "ema", "args", "normalization"):
        if key not in checkpoint:
            raise KeyError(f"Diffusion checkpoint missing key: {key}")
    diffusion_expected = {
        "latent_dim": expected["latent_dim"],
        "latent_grid": expected["latent_grid"],
        "seq_len": expected["seq_len"] - 1,
    }
    validate_representation_args(
        checkpoint["args"], diffusion_expected, "Compact DiT")


def main():
    args = parse_args()
    expected = {
        "latent_dim": args.latent_dim,
        "latent_grid": args.latent_grid,
        "seq_len": args.seq_len,
        "disable_temporal_mixer": True,
    }

    if args.encoder_ckpt:
        require_file(args.encoder_ckpt, "StreamVGGT checkpoint")
    if args.autoencoder_ckpt:
        validate_autoencoder(args.autoencoder_ckpt, expected)

    if args.stage in ("after_i0", "before_diffusion", "before_sample"):
        validate_i0(args.i0_decoder_ckpt, expected)

    if args.stage == "before_merge":
        for cache_dir, label, partitions in (
                (args.train_cache_dir, "training cache", args.cache_partitions),
                (args.eval_cache_dir, "evaluation cache", 1)):
            if not cache_dir:
                raise ValueError(f"{label} path is empty")
            for partition_id in range(partitions):
                validate_partition(cache_dir, partition_id, partitions)

    if args.stage in ("before_diffusion", "before_sample"):
        train_stats = validate_merged_cache(
            args.train_cache_dir, expected, "training cache")
        eval_stats = validate_merged_cache(
            args.eval_cache_dir, expected, "evaluation cache")
        if train_stats.get("representation") != eval_stats.get("representation"):
            raise RuntimeError(
                "Training and evaluation caches use different representations")

    if args.stage == "before_sample":
        validate_diffusion(args.diffusion_ckpt, expected)

    print(f"\nREADY: scale pipeline check '{args.stage}' passed")


if __name__ == "__main__":
    main()
