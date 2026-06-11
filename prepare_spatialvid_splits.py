#!/usr/bin/env python3
"""Build deterministic SpatialVID train/eval/overfit CSV files.

The source metadata is never modified. A bounded candidate pool is selected by
stable hashing, then candidate video paths are checked before the final,
non-overlapping splits are written.
"""

import argparse
import csv
import hashlib
import heapq
import json
import os
import tempfile


REQUIRED_FIELDS = ("id", "video path", "num frames")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_count", type=int, default=10000)
    parser.add_argument("--eval_count", type=int, default=64)
    parser.add_argument("--overfit_count", type=int, default=1)
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate_multiplier", type=int, default=2)
    parser.add_argument("--write_full_train", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_signature(path):
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def stable_score(seed, row_index, row):
    identity = f"{row.get('id', '')}\0{row.get('video path', '')}\0{row_index}"
    digest = hashlib.blake2b(
        identity.encode("utf-8"),
        digest_size=16,
        person=str(seed).encode("utf-8")[:16],
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def resolve_video_path(video_root, metadata_path):
    metadata_path = metadata_path.strip()
    if os.path.isabs(metadata_path):
        return metadata_path
    normalized = metadata_path.replace("\\", "/")
    if normalized.startswith("videos/"):
        normalized = normalized[len("videos/"):]
    return os.path.join(video_root, normalized)


def expected_manifest(args):
    return {
        "source": file_signature(args.csv),
        "video_root": os.path.abspath(args.video_root),
        "train_count": args.train_count,
        "eval_count": args.eval_count,
        "overfit_count": args.overfit_count,
        "min_frames": args.min_frames,
        "seed": args.seed,
        "candidate_multiplier": args.candidate_multiplier,
    }


def outputs_ready(args, expected):
    manifest_path = os.path.join(args.output_dir, "splits.json")
    required = [
        os.path.join(args.output_dir, "train_10k.csv"),
        os.path.join(args.output_dir, "eval.csv"),
        os.path.join(args.output_dir, "overfit.csv"),
    ]
    if args.write_full_train:
        required.append(os.path.join(args.output_dir, "train_full.csv"))
    if args.force or not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path) as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("configuration") == expected and all(
        os.path.exists(path) for path in required)


def write_csv_atomic(path, fieldnames, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp",
        dir=os.path.dirname(os.path.abspath(path)),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def select_rows(args):
    requested = args.train_count + args.eval_count + args.overfit_count
    if requested < 1:
        raise ValueError("At least one split count must be positive")
    if args.candidate_multiplier < 1:
        raise ValueError("--candidate_multiplier must be >= 1")
    candidate_limit = max(
        requested * args.candidate_multiplier,
        requested + 1024,
    )
    heap = []
    valid_metadata_rows = 0

    with open(args.csv, newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {args.csv}")
        missing = [field for field in REQUIRED_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"SpatialVID CSV is missing required columns: {missing}")
        fieldnames = reader.fieldnames
        for row_index, row in enumerate(reader):
            try:
                num_frames = int(float(row["num frames"]))
            except (TypeError, ValueError):
                continue
            if num_frames < args.min_frames:
                continue
            valid_metadata_rows += 1
            score = stable_score(args.seed, row_index, row)
            item = (-score, row_index, row)
            if len(heap) < candidate_limit:
                heapq.heappush(heap, item)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, item)

    candidates = sorted(
        [(-negative_score, row_index, row)
         for negative_score, row_index, row in heap],
        key=lambda item: (item[0], item[1]),
    )
    selected = []
    missing_files = 0
    for _, _, row in candidates:
        video_path = resolve_video_path(args.video_root, row["video path"])
        if not os.path.isfile(video_path):
            missing_files += 1
            continue
        selected.append(row)
        if len(selected) == requested:
            break
    minimum_required = args.overfit_count + args.eval_count + 1
    if len(selected) < minimum_required:
        raise RuntimeError(
            f"Only found {len(selected)} existing videos; at least "
            f"{minimum_required} are required for overfit/eval/train. "
            f"{missing_files} selected candidates were missing. "
            "Increase --candidate_multiplier or check video_root.")
    return fieldnames, selected, valid_metadata_rows, missing_files


def write_full_train(args, fieldnames, excluded_keys):
    path = os.path.join(args.output_dir, "train_full.csv")
    fd, temp_path = tempfile.mkstemp(
        prefix="train_full.csv.", suffix=".tmp",
        dir=args.output_dir, text=True)
    count = 0
    try:
        with os.fdopen(fd, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            with open(args.csv, newline="") as source:
                reader = csv.DictReader(source)
                for row in reader:
                    key = (row.get("id", ""), row.get("video path", ""))
                    if key in excluded_keys:
                        continue
                    try:
                        if int(float(row["num frames"])) < args.min_frames:
                            continue
                    except (TypeError, ValueError):
                        continue
                    writer.writerow(row)
                    count += 1
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    return count


def main():
    args = parse_args()
    expected = expected_manifest(args)
    if outputs_ready(args, expected):
        print(f"Reusing SpatialVID splits in {args.output_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    fieldnames, selected, valid_rows, missing_files = select_rows(args)
    cursor = 0
    overfit = selected[cursor:cursor + args.overfit_count]
    cursor += args.overfit_count
    evaluation = selected[cursor:cursor + args.eval_count]
    cursor += args.eval_count
    training = selected[cursor:cursor + args.train_count]
    if len(training) < args.train_count:
        print(
            f"[WARN] Requested {args.train_count} training videos but only "
            f"{len(training)} remain after reserving overfit/eval splits.")

    write_csv_atomic(
        os.path.join(args.output_dir, "overfit.csv"), fieldnames, overfit)
    write_csv_atomic(
        os.path.join(args.output_dir, "eval.csv"), fieldnames, evaluation)
    write_csv_atomic(
        os.path.join(args.output_dir, "train_10k.csv"), fieldnames, training)

    full_train_count = None
    if args.write_full_train:
        excluded = {
            (row.get("id", ""), row.get("video path", ""))
            for row in overfit + evaluation
        }
        full_train_count = write_full_train(args, fieldnames, excluded)

    manifest = {
        "configuration": expected,
        "counts": {
            "valid_metadata_rows": valid_rows,
            "train_10k": len(training),
            "eval": len(evaluation),
            "overfit": len(overfit),
            "train_full": full_train_count,
            "missing_candidate_files": missing_files,
        },
        "video_ids": {
            "overfit": [row["id"] for row in overfit],
            "eval": [row["id"] for row in evaluation],
        },
    }
    manifest_path = os.path.join(args.output_dir, "splits.json")
    fd, temp_path = tempfile.mkstemp(
        prefix="splits.json.", suffix=".tmp",
        dir=args.output_dir, text=True)
    with os.fdopen(fd, "w") as output:
        json.dump(manifest, output, indent=2)
    os.replace(temp_path, manifest_path)
    print(
        f"Prepared SpatialVID splits: train={len(training)}, "
        f"eval={len(evaluation)}, overfit={len(overfit)}, "
        f"full_train={full_train_count}")
    print(f"Split manifest: {manifest_path}")


if __name__ == "__main__":
    main()
