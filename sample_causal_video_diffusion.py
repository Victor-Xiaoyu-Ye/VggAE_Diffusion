#!/usr/bin/env python3
"""Sample a fixed-window R7 diffusion checkpoint using EMA by default.

Only the single cached condition anchor is consumed.  The model emits exactly
four future chunks; optional strict R7 decoding turns anchor+future into the
complete nine-frame RGB clip.
"""
from __future__ import annotations

import argparse
import os

import torch

from models.compact_dit import CompactLatentDiT
from utils.device import (configure_backend_compatibility, get_device,
                          get_device_name, manual_seed_all)
from train_causal_video_diffusion import (
    CONTEXT_CHUNKS, FUTURE_CHUNKS, LATENT_DIM,
    checked_decode_chunk_size, exact_equal, inverse_fp32, load_torch_artifact,
    normalize_fp32, normalization_signature, sample_shifted, stats_tensors,
    validate_r7_artifact_representation, validate_stats,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sample production R7 diffusion")
    p.add_argument("--checkpoint", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--anchor", default="",
                        help=".pt containing cond [1,N,D] (or a raw tensor)")
    source.add_argument("--manifest", default="",
                        help="cache manifest; deterministically uses --sample_index")
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--output", required=True,
                   help="output latent pack path; its directory also receives RGB PNGs/grid/metrics")
    p.add_argument("--stats", default="",
                   help="optional stats.pt; must exactly match checkpoint normalization")
    p.add_argument("--r7_ckpt", default="", help="strict tokenizer+RGB decoder")
    p.add_argument("--sample_steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="",
                   help="override accelerator device; default uses project resolver")
    p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    p.add_argument("--weights", choices=("ema", "model"), default="ema")
    p.add_argument("--decode_chunk_size", type=int, default=0)
    p.add_argument("--preview_fps", type=float, default=8.0)
    return p.parse_args(argv)


def build_model(config):
    return CompactLatentDiT(
        latent_dim=int(config["latent_dim"]),
        num_tokens=int(config.get("latent_grid", 18)) ** 2,
        model_dim=int(config["model_dim"]),
        spatial_depth=int(config["spatial_depth"]),
        temporal_depth=int(config["temporal_depth"]),
        num_heads=int(config["num_heads"]), seq_len=FUTURE_CHUNKS,
        text_cond=False, i0_condition=False, clean_frame0=True,
        block_schedule="interleaved", time_scale=float(config["time_scale"]))


def load_anchor(path):
    pack = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(pack, dict):
        if "cond" not in pack:
            raise KeyError("anchor pack must contain cond; target/latent histories are ignored")
        anchor = pack["cond"]
    else:
        anchor = pack
    anchor = torch.as_tensor(anchor).float()
    if anchor.ndim == 3: anchor = anchor.unsqueeze(0)
    return anchor


def load_manifest_anchor(manifest, index):
    if index < 0:
        raise ValueError("sample_index must be nonnegative")
    from data.latent_shard_dataset import LatentShardDataset
    dataset = LatentShardDataset(
        manifest, shuffle_buffer=1, seed=0, repeat=False, rank=0, world_size=1)
    for current, sample in enumerate(dataset):
        if current == index:
            anchor = torch.as_tensor(sample["cond"]).float()
            return anchor.unsqueeze(0) if anchor.ndim == 3 else anchor
    raise IndexError(f"sample_index {index} is outside cache manifest")


def save_rgb_outputs(rgb, output_path, fps=8.0, label="generated"):
    from utils.video_preview import save_video_preview

    output_dir = os.path.dirname(os.path.abspath(output_path))
    stem = os.path.splitext(os.path.basename(output_path))[0]
    preview = save_video_preview(
        output_dir, stem, {label: rgb[0]}, fps=fps,
        metadata={"source": os.path.abspath(output_path)},
        save_frames=True, save_mp4=True)
    entry = preview["videos"][label]
    return entry.get("png_dir", ""), preview["grid"], entry.get("mp4", "")


def main(argv=None):
    args = parse_args(argv)
    if args.sample_steps < 1: raise ValueError("--sample_steps must be positive")
    device_type = get_device_name(); configure_backend_compatibility(device_type)
    device = torch.device(args.device) if args.device else get_device(0)
    torch.manual_seed(args.seed); manual_seed_all(args.seed)
    checkpoint = load_torch_artifact(args.checkpoint)
    config = checkpoint["args"]
    if (int(config.get("context_chunks", -1)),
            int(config.get("future_chunks", -1)),
            int(config.get("latent_dim", -1))) != (1, 4, 192):
        raise ValueError("checkpoint is not production fixed-window R7 (1+4,D=192)")
    if checkpoint.get("production_mode") is False:
        print("[WARN] sampling a diagnostic normalization=none checkpoint")
    stats = checkpoint.get("normalization")
    signature = validate_stats(stats, "checkpoint stats")
    if checkpoint.get("normalization_signature") != signature:
        raise ValueError("checkpoint normalization signature is invalid")
    if args.stats:
        supplied = load_torch_artifact(args.stats)
        validate_stats(supplied, "supplied stats")
        if not exact_equal(stats, supplied):
            raise ValueError(
                "supplied stats do not exactly match checkpoint normalization")
    st = stats_tensors(stats, device)

    anchor_raw = (load_anchor(args.anchor) if args.anchor else
                  load_manifest_anchor(args.manifest, args.sample_index))
    grid = int(config.get("latent_grid", 18))
    if tuple(anchor_raw.shape[1:]) != (CONTEXT_CHUNKS, grid*grid, LATENT_DIM):
        raise ValueError(f"anchor must be [B,1,{grid*grid},192], got {tuple(anchor_raw.shape)}")
    anchor_raw = anchor_raw.to(device)
    mode = config.get("normalization_mode", "zscore")
    cond = normalize_fp32(anchor_raw, st["cond"], mode)
    dtype = torch.float32
    model = build_model(config).to(device=device, dtype=dtype)
    state = checkpoint["ema"] if args.weights == "ema" else checkpoint["model"]
    model.load_state_dict(state, strict=True); model.eval()
    cond = cond.to(dtype)
    cpu_generator = torch.Generator(device="cpu"); cpu_generator.manual_seed(args.seed)
    shape = (cond.shape[0], FUTURE_CHUNKS, grid*grid, LATENT_DIM)
    noise = torch.randn(shape, generator=cpu_generator).to(device=device, dtype=dtype)
    with torch.inference_mode():
        sampled_norm = sample_shifted(
            model, cond, shape, args.sample_steps,
            float(config.get("time_shift_alpha", 1.0)), noise)
        future = inverse_fp32(sampled_norm, st["target"], mode)

    output = {"schema": "r7-diffusion-sample-v1",
              "source_checkpoint": args.checkpoint,
              "weights": args.weights, "seed": args.seed,
              "cond_anchor": anchor_raw.cpu(),
              "sampled_future": future.cpu(),
              "sampled_normalized": sampled_norm.float().cpu(),
              "normalization": stats,
              "normalization_signature": signature,
              "representation": checkpoint.get("representation", {})}
    metrics = {
        "weights": args.weights,
        "seed": args.seed,
        "sample_steps": args.sample_steps,
        "normalization_mode": mode,
        "sampled_future_mean": future.float().mean().item(),
        "sampled_future_std": future.float().std().item(),
        "sampled_motion_mean_l2": torch.cat(
            (anchor_raw.float(), future.float()), 1).diff(dim=1).flatten(2)
            .norm(dim=2).mean().item(),
    }
    if args.r7_ckpt:
        from utils.r7_representation import load_r7_modules
        artifact = validate_r7_artifact_representation(
            checkpoint["representation"], args.r7_ckpt)
        r7_config, _, _, tokenizer, decoder, _ = load_r7_modules(artifact)
        decode_chunk_size = checked_decode_chunk_size(
            r7_config, decoder, args.decode_chunk_size)
        tokenizer = tokenizer.to(device).eval(); decoder = decoder.to(device).eval()
        latent = torch.cat((anchor_raw.float(), future.float()), 1)
        latent = latent.reshape(cond.shape[0], 5, grid, grid, LATENT_DIM)
        with torch.inference_mode():
            geo, tex = tokenizer.decode(latent)
            rgb = decoder(geo, tex, frames_chunk_size=decode_chunk_size)[..., :3]
        if rgb.shape[1] != 9:
            raise RuntimeError(f"R7 decode must produce complete 9-frame RGB, got {rgb.shape}")
        output["rgb_9frames"] = rgb.clamp(0, 1).float().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(output, args.output)
    metrics_path = os.path.splitext(os.path.abspath(args.output))[0] + "_metrics.json"
    if "rgb_9frames" in output:
        frame_dir, grid_path, mp4_path = save_rgb_outputs(
            output["rgb_9frames"], args.output, args.preview_fps)
        metrics.update({"png_dir": frame_dir, "grid": grid_path,
                        "mp4": mp4_path})
    import json
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(f"saved 1 anchor + 4 future chunks"
          f"{' and complete 9-frame RGB' if 'rgb_9frames' in output else ''}: {args.output}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
