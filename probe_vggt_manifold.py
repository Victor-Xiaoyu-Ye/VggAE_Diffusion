#!/usr/bin/env python3
"""Measure VGGT/R7 geometry and decoder sensitivity without training."""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.loader_utils import multiprocessing_loader_kwargs
from data.token_utils import DPT_LEVELS, strip_special_tokens
from data.video_dataset import SpatialVidDataset, collate_fn
from streamvggt.models.streamvggt import StreamVGGT
from utils.device import (configure_backend_compatibility, get_device,
                          get_device_name, resolve_dtype)
from utils.encoder_loader import load_encoder_checkpoint
from utils.file_signature import validate_file_signature
from utils.r7_representation import (
    build_contract, load_checkpoint, load_r7_modules, validate_contract)
from utils.video_preview import save_video_preview


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--r7_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--preview_samples", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=9)
    parser.add_argument("--target_index", type=int, default=1)
    parser.add_argument("--perturb_scale", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def centered_unit(x, eps=1e-8):
    centered = x.float() - x.float().mean(dim=-1, keepdim=True)
    return centered / centered.norm(dim=-1, keepdim=True).clamp_min(eps)


def tangent_project(base_unit, value):
    value = value.float() - value.float().mean(dim=-1, keepdim=True)
    return value - (value * base_unit).sum(dim=-1, keepdim=True) * base_unit


def exp_map(base_unit, tangent, eps=1e-8):
    norm = tangent.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.cos(norm) * base_unit + torch.sin(norm) * tangent / norm


def slerp(x, y, amount, eps=1e-7):
    x_u, y_u = centered_unit(x), centered_unit(y)
    cosine = (x_u * y_u).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    angle = torch.acos(cosine)
    denominator = torch.sin(angle).clamp_min(eps)
    result = (torch.sin((1 - amount) * angle) / denominator * x_u
              + torch.sin(amount * angle) / denominator * y_u)
    return result


def token_stats(value):
    value = value.detach().float()
    centered = value - value.mean(dim=-1, keepdim=True)
    norm = centered.norm(dim=-1)
    mean_abs = value.mean(dim=-1).abs()
    return {
        "tokens": int(norm.numel()),
        "channel_mean_abs": float(mean_abs.mean()),
        "centered_norm_mean": float(norm.mean()),
        "centered_norm_std": float(norm.std(unbiased=False)),
        "centered_norm_cv": float(norm.std(unbiased=False) / norm.mean().clamp_min(1e-8)),
    }


def psnr(prediction, target):
    error = F.mse_loss(prediction.float(), target.float()).clamp_min(1e-12)
    return float(-10 * torch.log10(error))


def main(argv=None):
    args = parse_args(argv)
    if args.samples < 1 or args.preview_samples < 0:
        raise ValueError("sample counts must be valid")
    if args.target_index != 1:
        raise ValueError(
            "manifold probe v1 supports only frame 0 -> frame 1 so every "
            "decode uses the exact causal prefix")
    if not 0 < args.perturb_scale <= 0.3:
        raise ValueError("perturb_scale must be in (0, 0.3]")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device_type = get_device_name()
    configure_backend_compatibility(device_type)
    if device_type == "cpu":
        raise RuntimeError("manifold decode probe requires an accelerator")
    device = get_device(0)
    encoder_dtype = resolve_dtype(args.dtype)

    checkpoint = load_checkpoint(args.r7_ckpt)
    config, compressor, tex_encoder, tokenizer, decoder, matched = \
        load_r7_modules(checkpoint)
    checkpoint_metadata = checkpoint.get("representation_contract")
    del checkpoint
    contract = checkpoint_metadata
    if not contract:
        raise ValueError("R7 checkpoint has no representation_contract")
    validate_contract(contract, build_contract(config, include_signatures=False))
    expected_encoder = contract.get("signatures", {}).get("streamvggt")
    if expected_encoder:
        validate_file_signature(
            args.encoder_ckpt, expected_encoder, "StreamVGGT checkpoint")
    if config.temporal_factor != 1 or config.latent_seq_len != args.seq_len:
        raise ValueError("quick probe requires a factor-1 frame-aligned R7 checkpoint")
    encoder = StreamVGGT(
        img_size=config.target_size, patch_size=14, embed_dim=1024)
    load_encoder_checkpoint(encoder, args.encoder_ckpt, verbose=True)
    encoder = encoder.to(device=device, dtype=encoder_dtype).eval()
    compressor = compressor.to(device).eval()
    tex_encoder = tex_encoder.to(device).eval()
    tokenizer = tokenizer.to(device).eval()
    decoder = decoder.to(device).eval()
    for module in (encoder, compressor, tex_encoder, tokenizer, decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    dataset = SpatialVidDataset(
        args.csv, args.video_root, seq_len=args.seq_len,
        num_frames_per_video=args.seq_len, target_size=config.target_size,
        clip_duration_seconds=config.clip_duration_seconds,
        temporal_jitter=False, max_videos=args.samples)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn, **multiprocessing_loader_kwargs(args.num_workers))
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= args.samples:
                break
            frames = batch["frames"].to(device)
            raw_tokens, patch_start = encoder(frames.to(dtype=encoder_dtype))
            patch_tokens = strip_special_tokens(raw_tokens, patch_start)
            geo = compressor([token.float() for token in patch_tokens])
            geo = geo.permute(0, 1, 3, 4, 2).contiguous().float()
            tex = tex_encoder(frames.float()).float()
            latent = tokenizer.encode(geo, tex)
            level_stats = {
                str(level): {
                    "raw": token_stats(
                        patch_tokens[level][:, args.target_index]),
                    "post_compressor_projection": token_stats(
                        compressor.projs[str(level)](
                            compressor.norms[str(level)](
                                patch_tokens[level][
                                    :, args.target_index].float())))
                } for level in DPT_LEVELS
                if str(level) in compressor.norms
            }
            # Six R7 decoder passes follow; release the 24 full-resolution VGGT
            # levels (~GB scale for a nine-frame clip) before those allocations.
            del raw_tokens, patch_tokens, geo, tex
            anchor = latent[:, 0].reshape(
                1, config.latent_grid ** 2, config.latent_dim)
            target = latent[:, args.target_index].reshape(
                1, config.latent_grid ** 2, config.latent_dim)
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed + 1009 * index)
            noise = torch.randn(target.shape, generator=generator).to(device)
            target_float = target.float()
            target_mean = target_float.mean(dim=-1, keepdim=True)
            target_center = target_float - target_mean
            target_radius = target_center.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            unit = target_center / target_radius
            # Compare the same centered direction. The Euclidean displacement
            # uses the exact chord corresponding to the spherical angular step.
            tangent = tangent_project(unit, noise)
            tangent_direction = tangent / tangent.norm(
                dim=-1, keepdim=True).clamp_min(1e-8)
            chord = target_radius * (
                2.0 * np.sin(args.perturb_scale / 2.0))
            euclidean = target_float + tangent_direction * chord
            tangent = tangent_direction * args.perturb_scale
            spherical = exp_map(unit, tangent) * target_radius + target_mean
            linear_mid = 0.5 * anchor.float() + 0.5 * target.float()
            geodesic_mid = slerp(anchor, target, 0.5)
            anchor_center = anchor.float() - anchor.float().mean(dim=-1, keepdim=True)
            target_center = target.float() - target.float().mean(dim=-1, keepdim=True)
            radius = 0.5 * (anchor_center.norm(dim=-1, keepdim=True)
                            + target_center.norm(dim=-1, keepdim=True))
            mean = 0.5 * (anchor.float().mean(dim=-1, keepdim=True)
                          + target.float().mean(dim=-1, keepdim=True))
            geodesic_mid = geodesic_mid * radius + mean

            linear_center = linear_mid - linear_mid.mean(dim=-1, keepdim=True)
            norm_matched_linear_mid = linear_center / linear_center.norm(
                dim=-1, keepdim=True).clamp_min(1e-8) * radius + mean
            candidates = {
                "clean": target.float(),
                "euclidean": euclidean,
                "tangent": spherical,
                "linear_mid": linear_mid,
                "norm_matched_linear_mid": norm_matched_linear_mid,
                "geodesic_mid": geodesic_mid,
            }
            decoded = {}
            grid = config.latent_grid
            for name, value in candidates.items():
                # Preserve the trained full-sequence temporal-attention contract
                # without leaking future targets: repeat the one candidate across
                # all future slots, then score only frame 1. Every arm uses the
                # same suffix-fill rule.
                future = value[:, None].expand(
                    1, config.latent_seq_len - 1, *value.shape[1:])
                sequence = torch.cat((anchor[:, None].float(), future), dim=1)
                sequence = sequence.reshape(
                    1, config.latent_seq_len, grid, grid, config.latent_dim)
                geo_out, tex_out = tokenizer.decode(sequence)
                decoded[name] = decoder(
                    geo_out, tex_out)[:, 1:2, ..., :3].float().clamp(0, 1)
            raw_target = frames[:, args.target_index:args.target_index + 1] \
                .permute(0, 1, 3, 4, 2).float()
            clean_stats = token_stats(target)
            clean_radius = clean_stats["centered_norm_mean"]
            row = {
                "sample": index,
                "video_id": batch["video_id"][0],
                "target_index": args.target_index,
                "r7": token_stats(target),
                "r7_geo": token_stats(target[..., :config.geo_latent_dim]),
                "r7_tex": token_stats(target[..., config.geo_latent_dim:]),
                "decode": {},
                "vggt_levels": level_stats,
            }
            clean = decoded["clean"]
            for name, rgb in decoded.items():
                candidate_stats = token_stats(candidates[name])
                row["decode"][name] = {
                    "psnr_vs_raw": psnr(rgb, raw_target),
                    "psnr_vs_clean_decode": psnr(rgb, clean),
                    "centered_norm_ratio": (
                        candidate_stats["centered_norm_mean"]
                        / max(clean_radius, 1e-8)),
                    "finite": bool(torch.isfinite(rgb).all()),
                }
            rows.append(row)
            if index < args.preview_samples:
                anchor_rgb = frames[:, :1].permute(0, 1, 3, 4, 2).float()
                save_video_preview(
                    args.output_dir, f"sample{index:03d}",
                    {"ANCHOR": anchor_rgb[0], "RAW_TARGET": raw_target[0],
                     "AE_TARGET": clean[0], "EUCLIDEAN": decoded["euclidean"][0],
                     "TANGENT": decoded["tangent"][0],
                     "LINEAR_MID": decoded["linear_mid"][0],
                     "NORM_MATCHED_LINEAR_MID":
                         decoded["norm_matched_linear_mid"][0],
                     "GEODESIC_MID": decoded["geodesic_mid"][0]},
                    fps=1, metadata={"video_id": batch["video_id"][0],
                                     "target_index": args.target_index},
                    save_frames=True, save_mp4=False)

    if not rows:
        raise RuntimeError("probe dataset yielded no samples")
    with open(os.path.join(args.output_dir, "metrics.jsonl"), "w",
              encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def mean(path):
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values))

    summary = {
        "schema": "vggt-r7-manifold-probe-v1",
        "samples": len(rows), "target_index": args.target_index,
        "representation": contract,
        "matched": matched,
        "r7_centered_norm_cv": mean(("r7", "centered_norm_cv")),
        "decode_psnr_vs_clean": {
            name: mean(("decode", name, "psnr_vs_clean_decode"))
            for name in ("euclidean", "tangent", "linear_mid",
                         "norm_matched_linear_mid", "geodesic_mid")},
        "vggt_post_projection_cv": {
            str(level): mean(("vggt_levels", str(level),
                              "post_compressor_projection",
                              "centered_norm_cv"))
            for level in DPT_LEVELS},
        "interpretation_limits": [
            "VGGT levels are statistics-only; no raw-VGGT RGB head is tested.",
            "Euclidean/tangent/geodesic decode comparisons operate in R7 space.",
            "One candidate is repeated across all future slots before full-sequence decode.",
            "LayerNorm fixed-radius behavior alone is not manifold evidence.",
        ],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w",
              encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
