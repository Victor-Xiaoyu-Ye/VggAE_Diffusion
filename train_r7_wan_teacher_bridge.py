#!/usr/bin/env python3
"""Train the diagnostic R7-to-Wan native-patch bridge with held-out eval."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import torch

from models.r7_wan_teacher_bridge import R7WanTeacherBridge
from utils.device import (configure_backend_compatibility, get_device,
                          get_device_name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_fraction", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_manifest(directory):
    marker = os.path.join(directory, "_SUCCESS.pt")
    if not os.path.isfile(marker):
        raise RuntimeError(f"teacher cache marker missing: {marker}")
    meta = torch.load(marker, map_location="cpu", weights_only=False)
    if meta.get("schema") != "r7-wan-native-teacher-v1":
        raise ValueError(f"unsupported teacher schema: {meta.get('schema')}")
    paths = sorted(glob.glob(os.path.join(directory, "teacher-*.pt")))
    if len(paths) != int(meta["num_shards"]):
        raise RuntimeError(
            f"teacher shard count {len(paths)} != marker {meta['num_shards']}")
    index = []
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        index.extend((path, key) for key in sorted(shard))
        del shard
    if len(index) != int(meta["num_samples"]):
        raise RuntimeError(
            f"teacher sample count {len(index)} != marker {meta['num_samples']}")
    return meta, index


class TeacherStore:
    """Lazy one-shard cache; avoids materializing the full teacher pack."""

    def __init__(self):
        self.path = None
        self.shard = None

    def get(self, item):
        path, key = item
        if path != self.path:
            self.shard = torch.load(path, map_location="cpu", weights_only=False)
            self.path = path
        return self.shard[key]


def split_index(index, seed, fraction):
    order = list(index)
    random.Random(seed).shuffle(order)
    eval_count = max(1, round(len(order) * fraction))
    if eval_count >= len(order):
        eval_count = len(order) - 1
    if eval_count < 1:
        raise RuntimeError("teacher bridge needs at least two samples")
    return order[eval_count:], order[:eval_count]


def stack_batch(store, items, device):
    samples = [store.get(item) for item in items]
    r7 = torch.stack([torch.as_tensor(sample["r7"]).float()
                      for sample in samples]).to(device)
    teacher = torch.cat([sample["wan_patch_hidden"].float()
                         for sample in samples], dim=0).to(device)
    return r7, teacher


def append_metric(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


@torch.no_grad()
def evaluate(model, store, index, batch_size, device):
    model.eval()
    totals = {"loss": 0.0, "feature": 0.0, "mean": 0.0, "variance": 0.0}
    count = 0
    for start in range(0, len(index), batch_size):
        items = index[start:start + batch_size]
        r7, teacher = stack_batch(store, items, device)
        loss, parts = model.alignment_loss(r7, teacher)
        weight = len(items)
        totals["loss"] += float(loss) * weight
        for name, value in parts.items():
            totals[name] += float(value) * weight
        count += weight
    model.train()
    return {name: value / count for name, value in totals.items()}


def main():
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        raise ValueError("steps/batch_size/eval_every must be positive")
    if not 0 < args.eval_fraction < 0.5:
        raise ValueError("eval_fraction must be in (0,0.5)")
    random.seed(args.seed); torch.manual_seed(args.seed)
    device_type = get_device_name()
    configure_backend_compatibility(device_type)
    device = get_device(0)

    meta, index = load_manifest(args.teacher_dir)
    train_index, eval_index = split_index(index, args.seed, args.eval_fraction)
    store = TeacherStore()
    sample = store.get(index[0])
    wan_dim = int(sample["wan_patch_hidden"].shape[1])
    r7_dim = int(sample["r7"].shape[-1])
    model = R7WanTeacherBridge(r7_dim, wan_dim).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=1e-2)
    metrics_path = args.metrics or os.path.join(
        os.path.dirname(os.path.abspath(args.output)), "metrics.jsonl")
    best = float("inf")
    rng = random.Random(args.seed)
    model.train()
    for step in range(1, args.steps + 1):
        chosen = rng.choices(train_index, k=args.batch_size)
        r7_batch, teacher_batch = stack_batch(store, chosen, device)
        loss, parts = model.alignment_loss(r7_batch, teacher_batch)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            result = evaluate(
                model, store, eval_index, args.batch_size, device)
            row = {"step": step, "train/loss": float(loss.detach()),
                   "eval/samples": len(eval_index),
                   **{f"eval/{key}": value for key, value in result.items()}}
            append_metric(metrics_path, row)
            print(row)
            if result["loss"] < best:
                best = result["loss"]
                os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                            exist_ok=True)
                torch.save({
                    "schema": "r7-wan-teacher-bridge-v2",
                    "model": model.state_dict(), "step": step,
                    "eval_loss": best,
                    "num_train": len(train_index),
                    "num_eval": len(eval_index),
                    "r7_dim": r7_dim, "wan_dim": wan_dim,
                    "teacher_meta": meta,
                }, args.output)
    print(f"saved best held-out bridge loss={best:.6f}: {args.output}")


if __name__ == "__main__":
    main()
