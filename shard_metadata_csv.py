#!/usr/bin/env python3
"""Split one large CSV into deterministic row-interleaved job shards."""

import argparse
import csv
import glob
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_shards", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def source_signature(path):
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    os.makedirs(args.output_dir, exist_ok=True)

    final_paths = [
        os.path.join(args.output_dir, f"part-{index:05d}.csv")
        for index in range(args.num_shards)
    ]
    counts_path = os.path.join(args.output_dir, "counts.json")
    expected = {
        "source": source_signature(args.csv),
        "num_shards": args.num_shards,
    }
    if not args.force and os.path.exists(counts_path):
        try:
            with open(counts_path) as counts_file:
                existing_metadata = json.load(counts_file)
            if (
                    existing_metadata.get("configuration") == expected
                    and all(os.path.exists(path) for path in final_paths)):
                print(
                    f"Reusing {args.num_shards} metadata shards in "
                    f"{args.output_dir}")
                return
        except (OSError, json.JSONDecodeError):
            pass
    for stale_path in glob.glob(os.path.join(args.output_dir, "part-*.csv")):
        os.unlink(stale_path)

    temp_paths = [path + ".tmp" for path in final_paths]
    handles = [open(path, "w", newline="") for path in temp_paths]
    counts = [0] * args.num_shards
    try:
        with open(args.csv, newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {args.csv}")
            writers = [
                csv.DictWriter(handle, fieldnames=reader.fieldnames)
                for handle in handles
            ]
            for writer in writers:
                writer.writeheader()
            for row_index, row in enumerate(reader):
                shard_index = row_index % args.num_shards
                writers[shard_index].writerow(row)
                counts[shard_index] += 1
    finally:
        for handle in handles:
            handle.close()

    for temp_path, final_path in zip(temp_paths, final_paths):
        os.replace(temp_path, final_path)
    with open(counts_path + ".tmp", "w") as output:
        json.dump({
            "configuration": expected,
            "counts": counts,
            "total": sum(counts),
        }, output, indent=2)
    os.replace(counts_path + ".tmp", counts_path)
    print(
        f"Wrote {args.num_shards} metadata shards with "
        f"{sum(counts)} total rows to {args.output_dir}")


if __name__ == "__main__":
    main()
