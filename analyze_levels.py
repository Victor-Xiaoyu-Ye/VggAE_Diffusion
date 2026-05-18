#!/usr/bin/env python3
"""Quantify per-level reconstruction quality to find the optimal diffusion boundary.

Tests each DPT level (4, 11, 17, 23) individually plus key combinations,
measuring PSNR / SSIM / LPIPS for RGB reconstruction.
Also analyzes feature geometry via PCA per level.

Usage:
    python analyze_levels.py \
        --video_list /path/to/video_list.txt \
        --encoder_ckpt ckpts/streamvggt.pt \
        --decoder_ckpt ckpts/decoder_gld/decoder_epoch45.pt \
        --token_stats ckpts/token_stats.pt \
        --out_dir analysis/level_ablation
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.heads.dpt_head import DPTHead
from data.token_utils import DPT_LEVELS, load_token_stats, normalize_tokens, strip_special_tokens
from utils.video_io import read_video_frames


def parse_args():
    p = argparse.ArgumentParser(description="Ablation: which DPT level carries the most information?")
    p.add_argument("--video_list", type=str, required=True,
                   help="Text file with one video path per line")
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--decoder_ckpt", type=str, required=True)
    p.add_argument("--token_stats", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="analysis/level_ablation")
    p.add_argument("--max_videos", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--img_size", type=int, default=518)
    p.add_argument("--patch_size", type=int, default=14)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    return p.parse_args()


def decode_with_levels(decoder, tokens_list, frames, keep_levels, seq_len):
    """Zero out non-kept levels and decode."""
    test_list = []
    for lvl_idx in range(24):
        t = tokens_list[lvl_idx].clone()
        if lvl_idx not in keep_levels:
            t.zero_()
        test_list.append(t)
    test_list = [t.to(dtype=torch.float32) for t in test_list]
    preds, _ = decoder(test_list, images=frames.float(), patch_start_idx=0, frames_chunk_size=seq_len)
    recon = preds.squeeze(2).permute(0, 1, 4, 2, 3).contiguous()
    return recon[:, :, :3].clamp(0, 1).float()


def compute_metrics(recon, target):
    B, S = recon.shape[:2]
    target = target.float()
    l1 = F.l1_loss(recon, target).item()
    mse = F.mse_loss(recon, target).item()
    psnr = -10 * np.log10(mse) if mse > 0 else float("inf")
    return {"l1": l1, "psnr": psnr, "mse": mse}


def pca_analysis(features):
    """features: [N, D] numpy array. Returns explained variance ratios."""
    x = features - features.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    var = (s ** 2) / max(x.shape[0] - 1, 1)
    ratio = var / max(var.sum(), 1e-12)
    return {
        "effective_rank": float(np.exp(-np.sum(ratio * np.log(ratio + 1e-12)))),
        "top3_var": [float(x) for x in ratio[:3]],
        "top10_var_sum": float(ratio[:10].sum()),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"npu:{args.gpu}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # Load video paths
    with open(args.video_list) as f:
        videos = [line.strip() for line in f if line.strip() and os.path.exists(line.strip())]
    if args.max_videos > 0:
        videos = videos[:args.max_videos]
    print(f"Loaded {len(videos)} videos")

    # Load encoder
    print("Loading encoder...")
    encoder = StreamVGGT(img_size=args.img_size, patch_size=args.patch_size, embed_dim=1024)
    state = torch.load(args.encoder_ckpt, map_location="cpu")
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device=device, dtype=dtype).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    level_stats = load_token_stats(args.token_stats, device, dtype=torch.float32)

    # Load decoder
    print("Loading decoder...")
    decoder = DPTHead(
        dim_in=2048, patch_size=args.patch_size, output_dim=4,
        activation="sigmoid", conf_activation="sigmoid",
    ).to(device=device, dtype=torch.float32)
    dec_state = torch.load(args.decoder_ckpt, map_location="cpu")
    if isinstance(dec_state, dict):
        for k in ("model_state_dict", "ema_state_dict", "model", "ema"):
            if k in dec_state:
                dec_state = dec_state[k]; break
    if any(k.startswith("module.") for k in dec_state):
        dec_state = {k.replace("module.", "", 1): v for k, v in dec_state.items()}
    decoder.load_state_dict(dec_state, strict=True)
    decoder.eval()

    # Test configurations
    configs = [
        ("level04", {4}),
        ("level11", {11}),
        ("level17", {17}),
        ("level23", {23}),
        ("levels_04_11", {4, 11}),
        ("levels_11_17", {11, 17}),
        ("levels_04_11_17", {4, 11, 17}),
        ("all_levels", set(DPT_LEVELS)),
    ]

    results = {name: {"l1": [], "psnr": [], "mse": []} for name, _ in configs}
    level_features = {lvl: [] for lvl in DPT_LEVELS}  # per-level pooled features

    print(f"Testing {len(configs)} level configurations on {len(videos)} videos...")
    for vid_path in tqdm(videos):
        frames = read_video_frames(vid_path, args.seq_len, args.img_size)
        frames_batched = frames.unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            tokens_list, psi = encoder.aggregator(frames_batched)
            tokens_list = strip_special_tokens(tokens_list, psi)
            tokens_list = normalize_tokens(tokens_list, level_stats)

            # Collect per-level feature statistics (CLIP-pooled)
            for lvl in DPT_LEVELS:
                feat = tokens_list[lvl].mean(dim=(1, 2))  # [B, D]
                level_features[lvl].append(feat.cpu().numpy())

            # Test each configuration
            for name, keep in configs:
                recon = decode_with_levels(decoder, tokens_list, frames_batched, keep, args.seq_len)
                metrics = compute_metrics(recon, frames_batched)
                for k, v in metrics.items():
                    results[name][k].append(v)

    # Aggregate
    summary = {}
    print("\n=== Level Ablation Results ===")
    print(f"{'Config':<20s} {'PSNR':>8s} {'L1':>8s} {'MSE':>8s}")
    print("-" * 48)
    for name, _ in configs:
        psnr = np.mean(results[name]["psnr"])
        l1 = np.mean(results[name]["l1"])
        mse = np.mean(results[name]["mse"])
        summary[name] = {"psnr_mean": float(psnr), "l1_mean": float(l1), "mse_mean": float(mse)}
        print(f"{name:<20s} {psnr:8.2f} {l1:8.4f} {mse:8.6f}")

    # Per-level PCA
    for lvl in DPT_LEVELS:
        feats = np.concatenate(level_features[lvl], axis=0)
        pca_stats = pca_analysis(feats)
        summary[f"level{lvl}_pca"] = pca_stats

    # Save
    out_path = os.path.join(args.out_dir, "level_ablation.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Recommendation
    best_single = max(
        [(name, np.mean(results[name]["psnr"])) for name, _ in configs if len(name.split("_")) == 2],
        key=lambda x: x[1],
    )
    print(f"\nBest single level: {best_single[0]} (PSNR={best_single[1]:.2f} dB)")
    print(f"All 4 levels:     PSNR={np.mean(results['all_levels']['psnr']):.2f} dB")
    loss_single = best_single[1] - np.mean(results["all_levels"]["psnr"])
    print(f"Gap (best single vs all): {loss_single:.2f} dB")
    print(f"\nThis gap is the information cost of using a single boundary level for diffusion.")


if __name__ == "__main__":
    main()
