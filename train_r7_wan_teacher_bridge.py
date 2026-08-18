#!/usr/bin/env python3
"""Train the R7-to-Wan native-patch bridge on diagnostic teacher shards."""

from __future__ import annotations

import argparse
import glob
import os
import random

import torch

from models.r7_wan_teacher_bridge import R7WanTeacherBridge


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_teacher(directory):
    result = {}
    for path in sorted(glob.glob(os.path.join(directory, "teacher-*.pt"))):
        result.update(torch.load(path, map_location="cpu", weights_only=False))
    if not result:
        raise RuntimeError(f"no teacher shards found in {directory}")
    return result


def main():
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    random.seed(args.seed); torch.manual_seed(args.seed)
    teacher = load_teacher(args.teacher_dir)
    ids = sorted(teacher)
    if not ids:
        raise RuntimeError("teacher cache is empty")
    sample_teacher = teacher[ids[0]]["wan_patch_hidden"]
    sample_r7 = teacher[ids[0]].get("r7")
    if sample_r7 is None:
        raise RuntimeError("teacher cache lacks embedded R7 tensors")
    wan_dim = int(sample_teacher.shape[1])
    model = R7WanTeacherBridge(192, wan_dim).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=1e-2)
    best = float("inf")
    for step in range(1, args.steps + 1):
        chosen = random.choices(ids, k=args.batch_size)
        r7_batch = torch.stack([
            torch.as_tensor(teacher[key]["r7"]).float() for key in chosen])
        teacher_batch = torch.cat([
            teacher[key]["wan_patch_hidden"].float() for key in chosen], dim=0)
        loss, parts = model.alignment_loss(r7_batch, teacher_batch)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        value = float(loss.detach())
        if value < best:
            best = value
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            torch.save({"schema": "r7-wan-teacher-bridge-v1",
                        "model": model.state_dict(), "step": step,
                        "loss": value, "num_samples": len(ids),
                        "r7_dim": 192, "wan_dim": wan_dim}, args.output)
        if step % 100 == 0 or step == 1:
            detail = " ".join(f"{key}={float(val):.5f}"
                              for key, val in parts.items())
            print(f"step={step} loss={value:.6f} {detail}")
    print(f"saved best bridge loss={best:.6f}: {args.output}")


if __name__ == "__main__":
    main()
