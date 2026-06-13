#!/usr/bin/env python3
"""Merge array-job latent manifests and normalization statistics."""

import argparse
import glob
import io
import os
import re

import torch

from utils.moxing_io import (
    copy_file,
    is_remote_path,
    join_remote,
    read_bytes,
    read_text,
    remote_exists,
    write_text,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--expected_partitions", type=int, default=0)
    parser.add_argument("--max_failure_rate", type=float, default=0.01)
    return parser.parse_args()


def merge_moments(entries):
    normalized = []
    for entry in entries:
        mean = entry["mean"].double()
        std = entry["std"].double()
        if mean.ndim == 1:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        count = torch.as_tensor(entry["count"], dtype=torch.float64)
        if count.ndim == 0:
            count = count.repeat(mean.shape[0])
        if count.shape != (mean.shape[0],):
            raise ValueError(
                f"Invalid stats count shape {tuple(count.shape)} for "
                f"mean {tuple(mean.shape)}")
        normalized.append((mean, std, count.unsqueeze(-1)))

    total_count = sum(count for _, _, count in normalized)
    if torch.any(total_count <= 0):
        raise ValueError("Cannot merge empty latent statistics")
    mean = sum(
        entry_mean * count
        for entry_mean, _, count in normalized
    ) / total_count
    second = sum(
        (entry_std.square() + entry_mean.square()) * count
        for entry_mean, entry_std, count in normalized
    ) / total_count
    variance = (second - mean.square()).clamp_min(1e-12)
    return {
        "mean": mean.float(),
        "std": variance.sqrt().float(),
        "count": total_count.squeeze(-1).long(),
    }


def merge_raw_moments(entries):
    required = ("sum", "sum_sq", "count")
    for entry in entries:
        if any(key not in entry for key in required):
            raise ValueError("Raw moments must contain sum, sum_sq, and count")
    total_sum = sum(entry["sum"].double() for entry in entries)
    total_sum_sq = sum(entry["sum_sq"].double() for entry in entries)
    total_count = sum(entry["count"].double() for entry in entries)
    if total_sum.shape != total_sum_sq.shape:
        raise ValueError("Raw moment sum and sum_sq shapes do not match")
    if total_count.shape != (total_sum.shape[0], 1):
        raise ValueError(
            f"Raw moment count shape {tuple(total_count.shape)} is invalid "
            f"for sum {tuple(total_sum.shape)}")
    if torch.any(total_count <= 0):
        raise ValueError("Cannot merge empty raw latent moments")
    mean = total_sum / total_count
    variance = (
        total_sum_sq / total_count - mean.square()
    ).clamp_min(1e-12)
    stats = {
        "mean": mean.float(),
        "std": variance.sqrt().float(),
        "count": total_count.squeeze(-1).long(),
    }
    raw = {
        "sum": total_sum,
        "sum_sq": total_sum_sq,
        "count": total_count,
    }
    return stats, raw


def main():
    args = parse_args()
    remote_cache = is_remote_path(args.cache_dir)
    cache_dir = (
        args.cache_dir.rstrip("/")
        if remote_cache else os.path.abspath(args.cache_dir))
    if remote_cache:
        if args.expected_partitions < 1:
            raise ValueError(
                "--expected_partitions is required for remote OBS caches")
        partition_dirs = [
            join_remote(
                cache_dir,
                f"part-{index:05d}-of-{args.expected_partitions:05d}")
            for index in range(args.expected_partitions)
        ]
    else:
        partition_dirs = sorted(
            glob.glob(os.path.join(cache_dir, "part-*-of-*")))
    if not partition_dirs:
        raise FileNotFoundError(f"No cache partitions found in {cache_dir}")
    if args.expected_partitions and len(partition_dirs) != args.expected_partitions:
        raise RuntimeError(
            f"Expected {args.expected_partitions} partitions, "
            f"found {len(partition_dirs)}")

    partition_pattern = re.compile(r"part-(\d+)-of-(\d+)$")
    parsed_partitions = []
    for partition_dir in partition_dirs:
        match = partition_pattern.search(partition_dir.rstrip("/"))
        if match is None:
            raise RuntimeError(f"Invalid partition directory: {partition_dir}")
        parsed_partitions.append((int(match.group(1)), int(match.group(2))))
    declared_counts = {count for _, count in parsed_partitions}
    if len(declared_counts) != 1:
        raise RuntimeError(
            f"Inconsistent partition counts: {sorted(declared_counts)}")
    declared_count = declared_counts.pop()
    expected_ids = set(range(declared_count))
    actual_ids = {partition_id for partition_id, _ in parsed_partitions}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise RuntimeError(
            f"Partition IDs are not complete: missing={missing}, extra={extra}")
    if args.expected_partitions and declared_count != args.expected_partitions:
        raise RuntimeError(
            f"Partition directories declare {declared_count}, "
            f"expected {args.expected_partitions}")

    target_entries = []
    cond_entries = []
    target_raw_entries = []
    cond_raw_entries = []
    num_samples = 0
    num_failed = 0
    shard_paths = []
    config = None
    representation = None
    for partition_dir in partition_dirs:
        success_path = (
            join_remote(partition_dir, "_SUCCESS")
            if remote_cache else os.path.join(partition_dir, "_SUCCESS"))
        if remote_cache:
            try:
                if not remote_exists(success_path):
                    raise FileNotFoundError(success_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Cache partition is incomplete (missing _SUCCESS): "
                    f"{partition_dir}. Finish or resume "
                    "scripts/scale/03_cache_latents.sh before running merge."
                ) from exc
        elif not os.path.exists(success_path):
            raise RuntimeError(
                f"Cache partition is incomplete (missing _SUCCESS): "
                f"{partition_dir}")
        stats_path = (
            join_remote(partition_dir, "stats.pt")
            if remote_cache else os.path.join(partition_dir, "stats.pt"))
        if not remote_cache and not os.path.exists(stats_path):
            raise FileNotFoundError(f"Missing partition stats: {stats_path}")
        stats_source = (
            io.BytesIO(read_bytes(stats_path))
            if remote_cache else stats_path)
        stats = torch.load(
            stats_source, map_location="cpu", weights_only=False)
        target_entries.append(stats["target"])
        cond_entries.append(stats["cond"])
        raw_moments = stats.get("moments")
        if raw_moments is not None:
            target_raw_entries.append(raw_moments["target"])
            cond_raw_entries.append(raw_moments["cond"])
        num_samples += stats["num_samples"]
        num_failed += stats.get("num_failed", 0)
        config = config or stats.get("config")
        current_representation = stats.get("representation")
        if representation is None:
            representation = current_representation
        elif current_representation != representation:
            raise RuntimeError(
                f"Representation mismatch in {stats_path}. "
                "Do not merge caches from different checkpoints or configs.")
        if remote_cache:
            partition_manifest = join_remote(
                partition_dir, "manifest.txt")
            shard_paths.extend(
                line.strip()
                for line in read_text(partition_manifest).splitlines()
                if line.strip())
        else:
            shard_paths.extend(
                sorted(glob.glob(os.path.join(partition_dir, "*.tar")))
                + sorted(glob.glob(
                    os.path.join(partition_dir, "rank-*", "*.tar"))))
    if not shard_paths:
        raise RuntimeError(f"No tar shards found in {cache_dir}")

    total_attempted = num_samples + num_failed
    failure_rate = num_failed / max(total_attempted, 1)
    if failure_rate > args.max_failure_rate:
        raise RuntimeError(
            f"Cache failure rate {failure_rate:.2%} exceeds "
            f"{args.max_failure_rate:.2%}")

    manifest_path = (
        join_remote(cache_dir, "manifest.txt")
        if remote_cache else os.path.join(cache_dir, "manifest.txt"))
    manifest_lines = []
    for path in shard_paths:
        manifest_lines.append(
            path if remote_cache else os.path.relpath(path, cache_dir))
    if remote_cache:
        write_text(manifest_path, "\n".join(manifest_lines) + "\n")
    else:
        with open(manifest_path, "w") as manifest:
            manifest.write("\n".join(manifest_lines) + "\n")

    if target_raw_entries and len(target_raw_entries) != len(partition_dirs):
        raise RuntimeError(
            "Only some partitions contain raw moments; regenerate or use "
            "a consistent cache version")
    if target_raw_entries:
        target_stats, target_moments = merge_raw_moments(target_raw_entries)
        cond_stats, cond_moments = merge_raw_moments(cond_raw_entries)
    else:
        target_stats = merge_moments(target_entries)
        cond_stats = merge_moments(cond_entries)
        target_moments = None
        cond_moments = None

    merged_stats = {
        "normalization_version": 2,
        "target": target_stats,
        "cond": cond_stats,
        "num_samples": num_samples,
        "num_failed": num_failed,
        "num_shards": len(shard_paths),
        "num_partitions": len(partition_dirs),
        "failure_rate": failure_rate,
        "representation": representation,
        "config": config,
    }
    if target_moments is not None:
        merged_stats["moments"] = {
            "target": target_moments,
            "cond": cond_moments,
        }
    stats_path = (
        join_remote(cache_dir, "stats.pt")
        if remote_cache else os.path.join(cache_dir, "stats.pt"))
    if remote_cache:
        local_output = os.path.join(
            os.environ.get(
                "MOX_METADATA_CACHE_DIR",
                "/cache/yexiaoyu/vggae_runtime/cache/metadata"),
            "merged-stats.pt")
        os.makedirs(os.path.dirname(local_output), exist_ok=True)
        torch.save(merged_stats, local_output)
        copy_file(local_output, stats_path)
    else:
        torch.save(merged_stats, stats_path)
    print(
        f"Merged {len(partition_dirs)} partitions, {len(shard_paths)} tar "
        f"shards, {num_samples} samples, {num_failed} failed")
    print(f"Manifest: {manifest_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
