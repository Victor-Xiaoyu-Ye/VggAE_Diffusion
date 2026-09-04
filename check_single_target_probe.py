#!/usr/bin/env python3
"""Fail-closed decision gates for the R7 single-target quick ladder."""

from __future__ import annotations

import argparse
import json
import math


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--stage", choices=("overfit", "heldout"), required=True)
    parser.add_argument("--max_latent_mse", type=float, default=0.01)
    parser.add_argument("--min_motion_cosine", type=float, default=0.90)
    parser.add_argument("--min_norm_ratio", type=float, default=0.8)
    parser.add_argument("--max_norm_ratio", type=float, default=1.2)
    parser.add_argument("--min_psnr_gain", type=float, default=0.0)
    parser.add_argument("--min_lpips_gain", type=float, default=0.0)
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def load_latest_eval(path):
    latest = None
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if "eval/latent_mse" in row:
                latest = row
    if latest is None:
        raise RuntimeError(f"no evaluation row found in {path}")
    return latest


def main(argv=None):
    args = parse_args(argv)
    row = load_latest_eval(args.metrics)
    expected_split = "train-overfit" if args.stage == "overfit" else "held-out"
    checks = {
        "split": row.get("eval/split") == expected_split,
        "finite_latent_mse": math.isfinite(float(row["eval/latent_mse"])),
        "latent_mse": float(row["eval/latent_mse"]) <= args.max_latent_mse,
        "motion_cosine": float(row["eval/motion_cosine"]) >= args.min_motion_cosine,
        "norm_ratio_min": float(row["eval/latent_norm_ratio"]) >= args.min_norm_ratio,
        "norm_ratio_max": float(row["eval/latent_norm_ratio"]) <= args.max_norm_ratio,
    }
    gains = {
        "psnr": (float(row["eval/generated_psnr_raw"])
                 - float(row["eval/copy_psnr_raw"])),
    }
    if args.stage == "heldout":
        checks["psnr_gain"] = gains["psnr"] >= args.min_psnr_gain
        if ("eval/generated_lpips_raw" in row
                and "eval/copy_lpips_raw" in row):
            gains["lpips"] = (float(row["eval/copy_lpips_raw"])
                              - float(row["eval/generated_lpips_raw"]))
            checks["lpips_gain"] = gains["lpips"] >= args.min_lpips_gain
        elif args.min_lpips_gain > 0:
            checks["lpips_present"] = False
    decision = {
        "schema": "r7-single-target-decision-v1",
        "stage": args.stage, "passed": all(checks.values()),
        "checks": checks, "gains": gains, "metrics": row,
        "limits": {
            "max_latent_mse": args.max_latent_mse,
            "min_motion_cosine": args.min_motion_cosine,
            "norm_ratio": [args.min_norm_ratio, args.max_norm_ratio],
            "min_psnr_gain": args.min_psnr_gain,
            "min_lpips_gain": args.min_lpips_gain,
        },
    }
    payload = json.dumps(decision, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(payload + "\n")
    print(payload)
    if not decision["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
