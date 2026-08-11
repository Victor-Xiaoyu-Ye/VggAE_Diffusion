#!/usr/bin/env python3
"""Precompute UMT5-xxl text embeddings for the R7 Wan T2V diffusion stage.

Runs once on a single device. Loads Wan's native T5 encoder from the
Wan2.1-T2V-1.3B checkpoint directory, encodes the SpatialVID SceneDescription
caption for every train/eval video id plus the empty CFG prompt, and writes
fp16 ``[L, 4096]`` tensors sharded by video id. The trainer loads the sidecar
instead of holding the 5.7B-parameter encoder in memory.

Output layout under --output_dir:
  embeddings-000000.pt ...  {video_id: fp16 tensor [L, 4096]}
  empty_prompt.pt           fp16 tensor [L0, 4096] for the "" prompt
  index.json                {video_id: shard filename}
  _SUCCESS                  metadata marker (counts, text_len, source files)
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import tempfile
import types

import torch

# Installs the torch.npu / autocast fallbacks the vendored Wan code assumes.
import models.wan_compact_adapter  # noqa: F401  (side effects only)
from data.annotation_index import load_annotation_index
from utils.device import get_device, get_device_name
from utils.moxing_io import copy_file, is_remote_path, join_remote

_WAN_MODULES_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Wan2.1", "wan", "modules")
_PACKAGE = "_vgg_ae_wan_modules"


def _load_wan_module(name):
    full_name = f"{_PACKAGE}.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    if _PACKAGE not in sys.modules:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [_WAN_MODULES_ROOT]
        sys.modules[_PACKAGE] = package
    spec = importlib.util.spec_from_file_location(
        full_name, os.path.join(_WAN_MODULES_ROOT, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wan_ckpt_dir", required=True,
                   help="Wan2.1-T2V-1.3B directory with the T5 checkpoint")
    p.add_argument("--annotation_index", required=True,
                   help="annotation_index.json built by data/annotation_index.py")
    p.add_argument("--csv", action="append", required=True,
                   help="metadata CSV with an 'id' column; repeatable")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--text_len", type=int, default=512,
                   help="tokenizer cap; embeddings are trimmed to true length")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--samples_per_shard", type=int, default=2048)
    p.add_argument("--t5_checkpoint", default="models_t5_umt5-xxl-enc-bf16.pth")
    p.add_argument("--t5_tokenizer", default="google/umt5-xxl")
    return p.parse_args(argv)


def read_video_ids(csv_paths):
    ids = []
    seen = set()
    for path in csv_paths:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                vid = row["id"]
                if vid not in seen:
                    seen.add(vid)
                    ids.append(vid)
    return ids


class ShardWriter:
    def __init__(self, output_dir, samples_per_shard):
        self.output_dir = output_dir
        self.remote = is_remote_path(output_dir)
        self.samples_per_shard = samples_per_shard
        self.staging = tempfile.mkdtemp(prefix="wan_text_emb_")
        self.index = {}
        self.current = {}
        self.shard_id = 0

    def _shard_name(self):
        return f"embeddings-{self.shard_id:06d}.pt"

    def _publish(self, name, payload):
        local = os.path.join(self.staging, name)
        torch.save(payload, local)
        if self.remote:
            copy_file(local, join_remote(self.output_dir, name))
            os.remove(local)
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            os.replace(local, os.path.join(self.output_dir, name))

    def add(self, video_id, tensor):
        self.current[video_id] = tensor
        self.index[video_id] = self._shard_name()
        if len(self.current) >= self.samples_per_shard:
            self.flush()

    def flush(self):
        if not self.current:
            return
        self._publish(self._shard_name(), self.current)
        self.current = {}
        self.shard_id += 1

    def publish_json(self, name, payload):
        local = os.path.join(self.staging, name)
        with open(local, "w") as handle:
            json.dump(payload, handle)
        if self.remote:
            copy_file(local, join_remote(self.output_dir, name))
            os.remove(local)
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            os.replace(local, os.path.join(self.output_dir, name))


def main(argv=None):
    args = parse_args(argv)
    device, device_type = get_device(0), get_device_name()
    t5_module = _load_wan_module("t5")
    checkpoint_path = os.path.join(args.wan_ckpt_dir, args.t5_checkpoint)
    tokenizer_path = os.path.join(args.wan_ckpt_dir, args.t5_tokenizer)
    for path in (checkpoint_path, tokenizer_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Wan T5 asset missing: {path}; the Wan2.1-T2V-1.3B snapshot "
                "must include the umt5-xxl encoder and tokenizer")
    encoder = t5_module.T5EncoderModel(
        text_len=args.text_len,
        dtype=torch.bfloat16 if device_type != "cpu" else torch.float32,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path)

    annotations = load_annotation_index(args.annotation_index)
    video_ids = read_video_ids(args.csv)
    captions = {vid: annotations.get(vid, {}).get("caption", "")
                for vid in video_ids}
    missing = sum(1 for value in captions.values() if not value)
    print(f"encoding {len(video_ids)} captions ({missing} empty) "
          f"text_len={args.text_len}")

    writer = ShardWriter(args.output_dir, args.samples_per_shard)
    with torch.no_grad():
        empty = encoder([""], device)[0]
        writer._publish("empty_prompt.pt",
                        empty.to(device="cpu", dtype=torch.float16))
        batch_ids, batch_texts = [], []

        def flush_batch():
            if not batch_ids:
                return
            embeddings = encoder(batch_texts, device)
            for vid, embedding in zip(batch_ids, embeddings):
                writer.add(vid, embedding.to(
                    device="cpu", dtype=torch.float16))
            batch_ids.clear()
            batch_texts.clear()

        for vid in video_ids:
            caption = captions[vid]
            if not caption:
                # Empty captions reuse empty_prompt.pt; storing them per-video
                # would only duplicate the tensor thousands of times.
                continue
            batch_ids.append(vid)
            batch_texts.append(caption)
            if len(batch_ids) >= args.batch_size:
                flush_batch()
        flush_batch()
    writer.flush()
    writer.publish_json("index.json", writer.index)
    writer.publish_json("_SUCCESS", {
        "schema": "wan-umt5xxl-text-embeddings-v1",
        "text_len": args.text_len,
        "num_videos": len(video_ids),
        "num_encoded": len(writer.index),
        "num_empty_captions": missing,
        "t5_checkpoint": args.t5_checkpoint,
        "csv": args.csv,
    })
    print(f"wrote {len(writer.index)} embeddings + empty prompt to "
          f"{args.output_dir}")


if __name__ == "__main__":
    main()
