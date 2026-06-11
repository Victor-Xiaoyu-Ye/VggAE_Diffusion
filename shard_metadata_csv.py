#!/usr/bin/env python3
"""Split one large CSV into deterministic row-interleaved job shards."""

import argparse
import csv
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_shards", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    os.makedirs(args.output_dir, exist_ok=True)

    final_paths = [
        os.path.join(args.output_dir, f"part-{index:05d}.csv")
        for index in range(args.num_shards)
    ]
    existing = [path for path in final_paths if os.path.exists(path)]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing CSV shards")

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
    with open(os.path.join(args.output_dir, "counts.json"), "w") as output:
        json.dump({
            "source_csv": os.path.abspath(args.csv),
            "num_shards": args.num_shards,
            "counts": counts,
            "total": sum(counts),
        }, output, indent=2)
    print(
        f"Wrote {args.num_shards} metadata shards with "
        f"{sum(counts)} total rows to {args.output_dir}")


if __name__ == "__main__":
    main()
