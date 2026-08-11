#!/usr/bin/env python3
"""Sample an R7 Wan2.1-T2V-1.3B full-finetune checkpoint using EMA by default.

Consumes the single cached condition anchor plus an optional text prompt
(precomputed UMT5-xxl sidecar lookup by video id, or a raw embedding file).
The model predicts x0; sampling integrates the recovered velocity with Euler
flow matching and classifier-free guidance against the empty-prompt embedding.
Optional strict R7 decoding (with the decoder_robust override) turns
anchor+future into the complete nine-frame RGB clip.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from sample_causal_video_diffusion import (
    load_anchor, load_manifest_anchor, save_rgb_outputs)
from train_causal_video_diffusion import (
    CONTEXT_CHUNKS, FUTURE_CHUNKS, LATENT_DIM,
    checked_decode_chunk_size, exact_equal, inverse_fp32, load_torch_artifact,
    normalize_fp32, stats_tensors, validate_r7_artifact_representation,
    validate_stats)
from train_causal_wan_video_diffusion import (
    TextEmbeddingBank, load_robust_decoder, sample_x0_shifted,
    wan_config_snapshot)
from models.wan_compact_adapter import WanCompactAdapter
from utils.device import (configure_backend_compatibility, get_device,
                          get_device_name, manual_seed_all)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sample R7 Wan T2V diffusion")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--wan_ckpt_dir", required=True,
                   help="Wan2.1-T2V-1.3B directory (trunk weights are inside "
                        "the training checkpoint; the dir supplies the config)")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--anchor", default="",
                        help=".pt containing cond [1,N,D] (or a raw tensor)")
    source.add_argument("--manifest", default="",
                        help="cache manifest; deterministically uses --sample_index")
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--stats", default="",
                   help="optional stats.pt; must exactly match checkpoint normalization")
    p.add_argument("--r7_ckpt", default="", help="strict tokenizer+RGB decoder")
    p.add_argument("--decoder_ckpt", default="",
                   help="decoder_robust checkpoint overriding decoder.* weights")
    p.add_argument("--text_embedding_dir", required=True,
                   help="precomputed UMT5-xxl sidecar (index.json + shards)")
    p.add_argument("--prompt_video_id", default="",
                   help="use this video id's precomputed caption embedding; "
                        "empty uses the manifest sample's own id when available")
    p.add_argument("--text_embedding_file", default="",
                   help="raw [L,4096] embedding .pt overriding the sidecar lookup")
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--sample_steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="")
    p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    p.add_argument("--weights", choices=("ema", "model"), default="ema")
    p.add_argument("--decode_chunk_size", type=int, default=0)
    return p.parse_args(argv)


def load_manifest_sample(manifest, index):
    """Anchor plus video id for the sidecar lookup."""
    if index < 0:
        raise ValueError("sample_index must be nonnegative")
    from data.latent_shard_dataset import LatentShardDataset
    dataset = LatentShardDataset(
        manifest, shuffle_buffer=1, seed=0, repeat=False, rank=0, world_size=1)
    for current, sample in enumerate(dataset):
        if current == index:
            anchor = torch.as_tensor(sample["cond"]).float()
            anchor = anchor.unsqueeze(0) if anchor.ndim == 3 else anchor
            return anchor, str(sample.get("video_id", ""))
    raise IndexError(f"sample_index {index} is outside cache manifest")


def main(argv=None):
    args = parse_args(argv)
    if args.sample_steps < 1:
        raise ValueError("--sample_steps must be positive")
    device_type = get_device_name(); configure_backend_compatibility(device_type)
    device = torch.device(args.device) if args.device else get_device(0)
    torch.manual_seed(args.seed); manual_seed_all(args.seed)
    checkpoint = load_torch_artifact(args.checkpoint)
    config = checkpoint["args"]
    architecture = checkpoint.get("architecture") or {}
    if architecture.get("class") != "WanCompactAdapter":
        raise ValueError(
            "checkpoint is not a WanCompactAdapter run; use "
            "sample_causal_video_diffusion.py for CompactLatentDiT checkpoints")
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

    if args.anchor:
        anchor_raw = load_anchor(args.anchor)
        anchor_video_id = ""
    else:
        anchor_raw, anchor_video_id = load_manifest_sample(
            args.manifest, args.sample_index)
    grid = int(config.get("latent_grid", 18))
    if tuple(anchor_raw.shape[1:]) != (CONTEXT_CHUNKS, grid*grid, LATENT_DIM):
        raise ValueError(
            f"anchor must be [B,1,{grid*grid},192], got {tuple(anchor_raw.shape)}")
    anchor_raw = anchor_raw.to(device)
    mode = config.get("normalization_mode", "zscore")
    cond = normalize_fp32(anchor_raw, st["cond"], mode)

    model = WanCompactAdapter(
        args.wan_ckpt_dir, latent_dim=int(config["latent_dim"]),
        latent_grid=grid, seq_len=FUTURE_CHUNKS,
        full_finetune=True, anchor_frame=True).to(
        device=device, dtype=torch.float32)
    saved_wan_config = checkpoint.get("wan_config")
    if saved_wan_config and dict(saved_wan_config) != wan_config_snapshot(model):
        raise ValueError(
            f"--wan_ckpt_dir config {wan_config_snapshot(model)} does not "
            f"match checkpoint wan_config {dict(saved_wan_config)}")
    state = checkpoint["ema"] if args.weights == "ema" else checkpoint["model"]
    model.load_state_dict(state, strict=True); model.eval()

    text_bank = TextEmbeddingBank(args.text_embedding_dir)
    if args.text_embedding_file:
        raw = load_torch_artifact(args.text_embedding_file)
        text_emb = torch.as_tensor(raw).float().unsqueeze(0).to(device)
        prompt_source = args.text_embedding_file
    else:
        video_id = args.prompt_video_id or anchor_video_id
        text_emb = text_bank.batch([video_id], device)
        prompt_source = video_id or "<empty prompt>"
    uncond_emb = text_bank.empty_batch(1, device)

    cond = cond.to(torch.float32)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(args.seed)
    shape = (cond.shape[0], FUTURE_CHUNKS, grid*grid, LATENT_DIM)
    noise = torch.randn(shape, generator=cpu_generator).to(
        device=device, dtype=torch.float32)
    with torch.inference_mode():
        sampled_norm = sample_x0_shifted(
            model, cond, shape, args.sample_steps,
            float(config.get("time_shift_alpha", 1.0)), noise, text_emb,
            uncond_emb, args.cfg_scale, device_type)
        future = inverse_fp32(sampled_norm, st["target"], mode)

    output = {"schema": "r7-wan-diffusion-sample-v1",
              "source_checkpoint": args.checkpoint,
              "weights": args.weights, "seed": args.seed,
              "prompt_source": prompt_source,
              "cfg_scale": args.cfg_scale,
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
        "cfg_scale": args.cfg_scale,
        "prompt_source": prompt_source,
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
        tokenizer = tokenizer.to(device).eval()
        decoder = decoder.to(device).eval()
        if args.decoder_ckpt:
            load_robust_decoder(
                args.decoder_ckpt, checkpoint["representation"], decoder)
        latent = torch.cat((anchor_raw.float(), future.float()), 1)
        latent = latent.reshape(cond.shape[0], 5, grid, grid, LATENT_DIM)
        with torch.inference_mode():
            geo, tex = tokenizer.decode(latent)
            rgb = decoder(geo, tex, frames_chunk_size=decode_chunk_size)[..., :3]
        if rgb.shape[1] != 9:
            raise RuntimeError(
                f"R7 decode must produce complete 9-frame RGB, got {rgb.shape}")
        output["rgb_9frames"] = rgb.clamp(0, 1).float().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(output, args.output)
    metrics_path = os.path.splitext(os.path.abspath(args.output))[0] + "_metrics.json"
    if "rgb_9frames" in output:
        frame_dir, grid_path = save_rgb_outputs(output["rgb_9frames"], args.output)
        metrics.update({"png_dir": frame_dir, "grid": grid_path})
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(f"saved 1 anchor + 4 future chunks"
          f"{' and complete 9-frame RGB' if 'rgb_9frames' in output else ''}: "
          f"{args.output}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
