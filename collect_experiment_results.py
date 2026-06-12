#!/usr/bin/env python3
"""Collect metrics, checkpoints, and previews into one compact report."""

import argparse
import glob
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", default=[],
        help="Run definition in NAME=PATH form; may be repeated")
    parser.add_argument("--latent_contract", default="")
    parser.add_argument("--compact_stats", default="")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as metrics_file:
        for line in metrics_file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def latest_values(records):
    values = {}
    for record in records:
        for key, value in record.items():
            if key in ("step", "epoch", "final"):
                continue
            if isinstance(value, (int, float)):
                values[key] = value
    return values


def best_values(records):
    maximize = ("psnr",)
    minimize = ("loss", "lpips", "l1", "temporal", "depth", "mse")
    best = {}
    for record in records:
        for key, value in record.items():
            if not isinstance(value, (int, float)):
                continue
            lower_key = key.lower()
            if any(name in lower_key for name in maximize):
                best[key] = max(value, best.get(key, value))
            elif any(name in lower_key for name in minimize):
                best[key] = min(value, best.get(key, value))
    return best


def newest_file(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    paths = [path for path in paths if os.path.isfile(path)]
    if not paths:
        return ""
    return os.path.abspath(max(paths, key=os.path.getmtime))


def inspect_run(name, path):
    path = os.path.abspath(path)
    records = read_jsonl(os.path.join(path, "metrics.jsonl"))
    return {
        "name": name,
        "path": path,
        "exists": os.path.isdir(path),
        "num_metric_records": len(records),
        "latest": latest_values(records),
        "best": best_values(records),
        "checkpoint": newest_file([
            os.path.join(path, "checkpoint_final.pt"),
            os.path.join(path, "checkpoint_*.pt"),
        ]),
        "preview": newest_file([
            os.path.join(path, "samples", "*.png"),
            os.path.join(path, "samples", "*.mp4"),
        ]),
    }


def format_metrics(values):
    if not values:
        return "none"
    selected = []
    preferred = (
        "eval/psnr", "eval/lpips", "eval/reference_psnr",
        "eval/velocity_mse", "eval/generated_std",
        "train/loss", "train/flow_loss", "train/target_std",
    )
    for key in preferred:
        if key in values:
            selected.append(f"`{key}={values[key]:.6g}`")
    if not selected:
        for key in sorted(values)[:8]:
            selected.append(f"`{key}={values[key]:.6g}`")
    return ", ".join(selected)


def main():
    args = parse_args()
    runs = []
    for definition in args.run:
        if "=" not in definition:
            raise ValueError(f"Expected NAME=PATH, got {definition}")
        name, path = definition.split("=", 1)
        runs.append(inspect_run(name, path))

    report = {"runs": runs}
    if args.latent_contract and os.path.exists(args.latent_contract):
        with open(args.latent_contract) as contract_file:
            report["latent_contract"] = json.load(contract_file)
    if args.compact_stats and os.path.exists(args.compact_stats):
        with open(args.compact_stats) as stats_file:
            report["compact_stats"] = json.load(stats_file)

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "results_summary.json")
    markdown_path = os.path.join(args.output_dir, "results_summary.md")
    with open(json_path, "w") as output:
        json.dump(report, output, indent=2)

    lines = ["# Experiment Results", ""]
    contract = report.get("latent_contract", {})
    contract_metrics = contract.get("metrics", {})
    if contract_metrics:
        lines.extend([
            "## Latent Contract",
            "",
            f"- Compact I0 relative L2: "
            f"`{contract_metrics.get('compact_i0/relative_l2', {}).get('mean', float('nan')):.6g}`",
            f"- Drift / future residual RMS: "
            f"`{contract_metrics.get('compact_i0/drift_to_future_residual_rms', {}).get('mean', float('nan')):.6g}`",
            "",
        ])
    compact_stats = report.get("compact_stats", {})
    target_stats = compact_stats.get("target", {})
    if target_stats:
        correlation = target_stats.get("channel_correlation", {})
        stats_config = compact_stats.get("configuration", {})
        lines.extend([
            "## Compact Latent Statistics",
            "",
            f"- Videos: `{compact_stats.get('num_videos', 'unknown')}`",
            f"- Clip duration: "
            f"`{stats_config.get('clip_duration_seconds', 'unknown')}`",
            f"- Temporal mixer disabled: "
            f"`{stats_config.get('disable_temporal_mixer', 'unknown')}`",
            f"- Target global std: "
            f"`{target_stats.get('global_std', float('nan')):.6g}`",
            f"- Channel std p99/p01: "
            f"`{target_stats.get('channel_std_p99_over_p01', float('nan')):.6g}`",
            f"- Correlation effective rank: "
            f"`{correlation.get('effective_rank', float('nan')):.6g}`",
            f"- Future-frame std: "
            f"`{target_stats.get('frame_std', [])}`",
            "",
            "### Normalization Convergence",
            "",
            "| videos | target std rel. MAE | target std rel. p95 |",
            "|---:|---:|---:|",
        ])
        convergence = compact_stats.get(
            "normalization_convergence_vs_final", {})
        for count, values in sorted(
                convergence.items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {count} | "
                f"{values['target_std_relative_mae']:.6g} | "
                f"{values['target_std_relative_p95']:.6g} |")
        lines.append("")
    lines.extend(["## Runs", ""])
    for run in runs:
        lines.extend([
            f"### {run['name']}",
            "",
            f"- Directory: `{run['path']}`",
            f"- Metric records: `{run['num_metric_records']}`",
            f"- Latest: {format_metrics(run['latest'])}",
            f"- Best: {format_metrics(run['best'])}",
            f"- Checkpoint: `{run['checkpoint'] or 'missing'}`",
            f"- Latest preview: `{run['preview'] or 'missing'}`",
            "",
        ])
        if not stats_config.get("disable_temporal_mixer", False):
            lines.extend([
                "> Warning: these statistics use the legacy TemporalMixer "
                "contract and must not be used for the active I0 diffusion.",
                "",
            ])
    with open(markdown_path, "w") as output:
        output.write("\n".join(lines))

    print("\n".join(lines))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
