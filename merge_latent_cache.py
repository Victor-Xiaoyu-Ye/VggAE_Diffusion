#!/usr/bin/env python3
"""Merge array-job latent manifests and normalization statistics."""

import argparse
import glob
import os
import re

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--expected_partitions", type=int, default=0)
    parser.add_argument("--max_failure_rate", type=float, default=0.01)
    return parser.parse_args()


def merge_moments(entries):
    total_count = sum(entry["count"] for entry in entries)
    if total_count <= 0:
        raise ValueError("Cannot merge empty latent statistics")
    mean = sum(entry["mean"].double() * entry["count"] for entry in entries)
    mean = mean / total_count
    second = sum(
        (entry["std"].double().square() + entry["mean"].double().square())
        * entry["count"]
        for entry in entries
    ) / total_count
    variance = (second - mean.square()).clamp_min(1e-12)
    return {
        "mean": mean.float(),
        "std": variance.sqrt().float(),
        "count": total_count,
    }


def main():
    args = parse_args()
    cache_dir = os.path.abspath(args.cache_dir)
    partition_dirs = sorted(glob.glob(os.path.join(cache_dir, "part-*-of-*")))
    if not partition_dirs:
        raise FileNotFoundError(f"No cache partitions found in {cache_dir}")
    if args.expected_partitions and len(partition_dirs) != args.expected_partitions:
        raise RuntimeError(
            f"Expected {args.expected_partitions} partitions, "
            f"found {len(partition_dirs)}")

    partition_pattern = re.compile(r"part-(\d+)-of-(\d+)$")
    parsed_partitions = []
    for partition_dir in partition_dirs:
        match = partition_pattern.search(partition_dir)
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
    num_samples = 0
    num_failed = 0
    shard_paths = []
    config = None
    representation = None
    for partition_dir in partition_dirs:
        stats_path = os.path.join(partition_dir, "stats.pt")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Missing partition stats: {stats_path}")
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        target_entries.append(stats["target"])
        cond_entries.append(stats["cond"])
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
        shard_paths.extend(sorted(glob.glob(os.path.join(partition_dir, "*.tar"))))
    if not shard_paths:
        raise RuntimeError(f"No tar shards found in {cache_dir}")

    total_attempted = num_samples + num_failed
    failure_rate = num_failed / max(total_attempted, 1)
    if failure_rate > args.max_failure_rate:
        raise RuntimeError(
            f"Cache failure rate {failure_rate:.2%} exceeds "
            f"{args.max_failure_rate:.2%}")

    manifest_path = os.path.join(cache_dir, "manifest.txt")
    with open(manifest_path, "w") as manifest:
        for path in shard_paths:
            manifest.write(os.path.relpath(path, cache_dir) + "\n")

    merged_stats = {
        "target": merge_moments(target_entries),
        "cond": merge_moments(cond_entries),
        "num_samples": num_samples,
        "num_failed": num_failed,
        "num_shards": len(shard_paths),
        "num_partitions": len(partition_dirs),
        "failure_rate": failure_rate,
        "representation": representation,
        "config": config,
    }
    stats_path = os.path.join(cache_dir, "stats.pt")
    torch.save(merged_stats, stats_path)
    print(
        f"Merged {len(partition_dirs)} partitions, {len(shard_paths)} tar "
        f"shards, {num_samples} samples, {num_failed} failed")
    print(f"Manifest: {manifest_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
