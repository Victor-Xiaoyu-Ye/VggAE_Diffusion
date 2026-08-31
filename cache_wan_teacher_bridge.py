#!/usr/bin/env python3
"""Cache frozen Wan-VAE/native-patch teacher tensors for R7 bridge probes.

This is deliberately a small 10K diagnostic artifact, not a production cache.
The Wan VAE and patch embedding are used only while creating the teacher pack;
production R7 inference never loads them.
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.loader_utils import multiprocessing_loader_kwargs
from data.video_dataset import SpatialVidDataset, collate_fn
from models.wan_compact_adapter import WanModel
from streamvggt.models.streamvggt import StreamVGGT
from utils.device import (configure_backend_compatibility, get_device,
                          get_device_name, resolve_dtype)
from utils.encoder_loader import load_encoder_checkpoint
from utils.file_signature import sampled_file_signature
from utils.moxing_io import copy_file, is_remote_path, join_remote
from utils.r7_representation import (encode_dual, load_checkpoint,
                                     load_r7_modules)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--wan_ckpt_dir", required=True)
    parser.add_argument("--wan_vae_ckpt", required=True)
    parser.add_argument("--r7_ckpt", required=True)
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_videos", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=9)
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--clip_duration_seconds", type=float, default=1.0)
    parser.add_argument("--samples_per_shard", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    return parser.parse_args()


def load_wan_vae(path, device, dtype):
    import importlib.util
    import sys
    import types

    package_name = "_vgg_ae_wan_modules"
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Wan2.1", "wan", "modules")
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [root]
        sys.modules[package_name] = package
    module_name = package_name + ".vae"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(root, "vae.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name].WanVAE(
        vae_pth=path, dtype=dtype, device=device)


def publish(payload, output_dir, name):
    local = os.path.abspath(name)
    torch.save(payload, local)
    if is_remote_path(output_dir):
        copy_file(local, join_remote(output_dir, name))
        os.remove(local)
    else:
        os.makedirs(output_dir, exist_ok=True)
        os.replace(local, os.path.join(output_dir, name))


def main():
    args = parse_args()
    if args.max_videos < 1 or args.samples_per_shard < 1:
        raise ValueError("max_videos and samples_per_shard must be positive")
    device_type = get_device_name()
    configure_backend_compatibility(device_type)
    device = get_device(0)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.dtype]

    vae = load_wan_vae(args.wan_vae_ckpt, device, dtype)
    wan = WanModel.from_pretrained(args.wan_ckpt_dir).to(device).eval()
    wan_model_type = str(wan.model_type)
    patch = wan.patch_embedding.to(device).eval()
    for parameter in patch.parameters():
        parameter.requires_grad_(False)
    # Only the frozen native patch interface is needed below; release the 1.3B
    # trunk before loading StreamVGGT and the R7 stack on the same device.
    wan.patch_embedding = torch.nn.Identity()
    del wan
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()
    checkpoint = load_checkpoint(args.r7_ckpt)
    r7_config, compressor, tex_encoder, r7_tokenizer, _, _ = \
        load_r7_modules(checkpoint)
    encoder = StreamVGGT(
        img_size=r7_config.target_size, patch_size=14, embed_dim=1024)
    load_encoder_checkpoint(encoder, args.encoder_ckpt, verbose=True)
    encoder_dtype = resolve_dtype(args.dtype)
    encoder = encoder.to(device=device, dtype=encoder_dtype).eval()
    compressor = compressor.to(device).eval()
    tex_encoder = tex_encoder.to(device).eval()
    r7_tokenizer = r7_tokenizer.to(device).eval()
    for module in (encoder, compressor, tex_encoder, r7_tokenizer):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    dataset = SpatialVidDataset(
        args.csv, args.video_root, seq_len=args.seq_len,
        num_frames_per_video=args.seq_len, target_size=args.target_size,
        clip_duration_seconds=args.clip_duration_seconds,
        temporal_jitter=False, max_videos=args.max_videos)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn, **multiprocessing_loader_kwargs(args.num_workers))

    shard, shard_id, total = {}, 0, 0
    first_shapes = None
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Wan teacher cache"):
            frames = batch["frames"].to(device).float()
            # Wan VAE consumes [-1,1] CTHW videos.
            video = frames[0].permute(1, 0, 2, 3).mul(2).sub(1)
            latent = vae.encode([video])[0]
            hidden = patch(latent.unsqueeze(0).to(patch.weight.dtype))
            geo, texture = encode_dual(
                encoder, compressor, tex_encoder, frames, encoder_dtype)
            r7 = r7_tokenizer.encode(geo, texture)
            r7 = r7.reshape(
                1, r7_config.latent_seq_len,
                r7_config.latent_grid ** 2, r7_config.latent_dim)
            if first_shapes is None:
                first_shapes = {
                    "r7": list(r7.shape[1:]),
                    "wan_latent": list(latent.shape),
                    "wan_patch_hidden": list(hidden.shape[1:]),
                }
            key = str(batch["video_id"][0])
            shard[key] = {
                "r7": r7[0].to(device="cpu", dtype=torch.float16),
                "wan_latent": latent.to(device="cpu", dtype=torch.float16),
                "wan_patch_hidden": hidden.to(
                    device="cpu", dtype=torch.float16),
                "window_index": int(batch.get("window_index", [0])[0]),
            }
            total += 1
            if len(shard) >= args.samples_per_shard:
                publish(shard, args.output_dir, f"teacher-{shard_id:06d}.pt")
                shard, shard_id = {}, shard_id + 1
    if shard:
        publish(shard, args.output_dir, f"teacher-{shard_id:06d}.pt")
        shard_id += 1
    publish({
        "schema": "r7-wan-native-teacher-v1",
        "num_samples": total,
        "num_shards": shard_id,
        "seq_len": args.seq_len,
        "target_size": args.target_size,
        "clip_duration_seconds": args.clip_duration_seconds,
        "wan_model_type": wan_model_type,
        "r7_config": dict(checkpoint["representation_contract"]["config"]),
        "tensor_shapes": first_shapes,
        "signatures": {
            "r7": sampled_file_signature(args.r7_ckpt),
            "wan_vae": sampled_file_signature(args.wan_vae_ckpt),
        },
    }, args.output_dir, "_SUCCESS.pt")
    print(f"cached {total} Wan teacher samples in {shard_id} shards")


if __name__ == "__main__":
    main()
