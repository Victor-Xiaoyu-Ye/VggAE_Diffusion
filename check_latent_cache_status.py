#!/usr/bin/env python3
"""Inspect distributed latent-cache progress directly on OBS."""

import argparse
import json

from utils.moxing_io import join_remote, read_text, remote_exists


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--partition_id", type=int, default=0)
    parser.add_argument("--num_partitions", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=48)
    return parser.parse_args()


def main():
    args = parse_args()
    partition = join_remote(
        args.cache_dir,
        f"part-{args.partition_id:05d}-of-{args.num_partitions:05d}",
    )
    success_path = join_remote(partition, "_SUCCESS")
    if remote_exists(success_path):
        print(f"COMPLETE: {partition}")
        print(read_text(success_path))
        return

    complete_ranks = 0
    missing_ranks = []
    processed = 0
    total = 0
    failed = 0
    shards = 0
    for rank in range(args.world_size):
        status_path = join_remote(partition, f"status-r{rank:05d}.json")
        if not remote_exists(status_path):
            missing_ranks.append(rank)
            continue
        status = json.loads(read_text(status_path))
        complete_ranks += 1
        processed += int(status.get("processed_items", 0))
        total += int(status.get("total_rank_items", 0))
        failed += int(status.get("failed_samples", 0))
        shards += int(status.get("completed_shards", 0))
        print(
            f"rank={rank:02d} "
            f"processed={status.get('processed_items', 0)}/"
            f"{status.get('total_rank_items', 0)} "
            f"samples={status.get('successful_samples', 0)} "
            f"failed={status.get('failed_samples', 0)} "
            f"shards={status.get('completed_shards', 0)}"
        )

    print(
        f"INCOMPLETE: status_ranks={complete_ranks}/{args.world_size} "
        f"processed={processed}/{total or '?'} failed={failed} shards={shards}"
    )
    if missing_ranks:
        print("missing_status_ranks=" + ",".join(map(str, missing_ranks)))
    print("Resume with: bash scripts/scale/03_cache_latents.sh")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
