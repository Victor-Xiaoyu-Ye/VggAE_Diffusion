#!/usr/bin/env python3
"""Finalize a latent-cache partition from rank-level OBS artifacts.

This is a recovery tool for the common case where all ranks finished caching
and uploaded ``manifest-r*.txt`` plus ``moments-r*.pt``, but the torchrun job
failed before global rank 0 wrote the partition ``_SUCCESS`` marker.
"""

import argparse
import io
import json
import os
import tempfile

import torch

from utils.latent_stats import finalize_moments, merge_raw_moments
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
    parser.add_argument("--partition_id", type=int, default=0)
    parser.add_argument("--num_partitions", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=48)
    parser.add_argument("--latent_grid", type=int, default=18)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_torch(path):
    source = io.BytesIO(read_bytes(path)) if is_remote_path(path) else path
    return torch.load(source, map_location="cpu", weights_only=False)


def local_or_remote_exists(path):
    return remote_exists(path) if is_remote_path(path) else os.path.exists(path)


def count_failures(path):
    if not local_or_remote_exists(path):
        return 0
    return sum(1 for line in read_text(path).splitlines() if line.strip())


def main():
    args = parse_args()
    partition = join_remote(
        args.cache_dir,
        f"part-{args.partition_id:05d}-of-{args.num_partitions:05d}",
    )
    success_path = join_remote(partition, "_SUCCESS")
    if local_or_remote_exists(success_path) and not args.force:
        print(f"Partition already complete: {partition}")
        print(read_text(success_path))
        return

    shard_paths = []
    target_moments = []
    cond_moments = []
    num_failed = 0
    contract = None
    missing = []
    status_samples = 0
    status_seen = 0

    for rank in range(args.world_size):
        manifest_path = join_remote(partition, f"manifest-r{rank:05d}.txt")
        moments_path = join_remote(partition, f"moments-r{rank:05d}.pt")
        progress_path = join_remote(partition, f"progress-r{rank:05d}.pt")
        failure_path = join_remote(partition, f"failures-r{rank:05d}.jsonl")
        status_path = join_remote(partition, f"status-r{rank:05d}.json")

        if not local_or_remote_exists(manifest_path):
            missing.append(manifest_path)
            continue
        if not local_or_remote_exists(moments_path):
            missing.append(moments_path)
            continue

        shard_paths.extend(
            line.strip()
            for line in read_text(manifest_path).splitlines()
            if line.strip()
        )
        moments = load_torch(moments_path)
        target_moments.append(moments["target"])
        cond_moments.append(moments["cond"])
        num_failed += count_failures(failure_path)

        if local_or_remote_exists(progress_path):
            progress = load_torch(progress_path)
            current_contract = progress.get("contract")
            if contract is None:
                contract = current_contract
            elif current_contract != contract:
                raise RuntimeError(
                    f"Cache contract mismatch in {progress_path}")

        if local_or_remote_exists(status_path):
            status = json.loads(read_text(status_path))
            status_seen += 1
            status_samples += int(status.get("successful_samples", 0))

    if missing:
        preview = "\n".join(missing[:12])
        raise RuntimeError(
            f"Cannot finalize; missing {len(missing)} rank artifacts. "
            f"First missing paths:\n{preview}")
    if not shard_paths:
        raise RuntimeError(f"No shard paths found under {partition}")

    target_raw = merge_raw_moments(target_moments)
    cond_raw = merge_raw_moments(cond_moments)
    target_stats = finalize_moments(target_raw)
    cond_stats = finalize_moments(cond_raw)

    tokens_per_sample = args.latent_grid ** 2
    inferred_samples = int(target_raw["count"].min().item() // tokens_per_sample)
    num_samples = status_samples if status_seen == args.world_size else inferred_samples
    if num_samples != inferred_samples:
        print(
            "[WARN] status sample count differs from raw moments: "
            f"status={num_samples}, moments={inferred_samples}")

    manifest_path = join_remote(partition, "manifest.txt")
    write_text(manifest_path, "\n".join(shard_paths) + "\n")

    stats = {
        "normalization_version": 2,
        "target": target_stats,
        "cond": cond_stats,
        "moments": {
            "target": target_raw,
            "cond": cond_raw,
        },
        "num_samples": num_samples,
        "num_failed": num_failed,
        "representation": contract,
        "config": {
            "finalized_from_rank_artifacts": True,
            "partition_id": args.partition_id,
            "num_partitions": args.num_partitions,
            "world_size": args.world_size,
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        local_stats = os.path.join(tmpdir, "stats.pt")
        local_config = os.path.join(tmpdir, "config.json")
        torch.save(stats, local_stats)
        with open(local_config, "w") as handle:
            json.dump(stats["config"], handle, indent=2, sort_keys=True)
        copy_file(local_stats, join_remote(partition, "stats.pt"))
        copy_file(local_config, join_remote(partition, "config.json"))

    write_text(success_path, json.dumps({
        "partition_id": args.partition_id,
        "num_partitions": args.num_partitions,
        "num_samples": num_samples,
        "num_failed": num_failed,
        "contract": contract,
        "finalized_from_rank_artifacts": True,
    }, sort_keys=True))

    print(
        f"Finalized {partition}: shards={len(shard_paths)} "
        f"samples={num_samples} failed={num_failed}")
    print(f"Manifest: {manifest_path}")
    print(f"Stats: {join_remote(partition, 'stats.pt')}")
    print(f"Success: {success_path}")


if __name__ == "__main__":
    main()
