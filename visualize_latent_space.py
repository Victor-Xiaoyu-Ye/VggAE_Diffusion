#!/usr/bin/env python3
"""Visualize StreamVGGT / decoder / DINO representation geometry.

This is an offline diagnostic script. It does not touch training. Typical use:

  python visualize_latent_space.py \
    --csv /path/to/SpatialVID_HQ_metadata.csv \
    --video_root /path/to/videos \
    --encoder_ckpt /path/to/streamvggt/checkpoints.pth \
    --token_stats ckpts/token_stats.pt \
    --decoder_ckpt ckpts/decoder_gld/decoder_epoch45.pt \
    --with_dino \
    --max_videos 128 \
    --out_dir analysis/latent_vis_level11

It answers three questions:
  1. Does decoder_GLD preserve the StreamVGGT manifold?
     real frames -> StreamVGGT vs decoder recon -> StreamVGGT.
  2. Do generated tokens land near real StreamVGGT tokens?
     optional --generated_token_glob "samples/*/*/tokens.pt".
  3. How different is StreamVGGT sample geometry from DINO sample geometry?
     pairwise distance correlation and kNN-overlap on the same clips.
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.heads.dpt_head import DPTHead
from data.token_utils import DPT_LEVELS, load_token_stats, normalize_tokens, strip_special_tokens
from utils.video_io import read_video_frames


def parse_args():
    p = argparse.ArgumentParser(description="PCA/distance diagnostics for StreamVGGT RAE latents")

    # Data
    p.add_argument("--csv", type=str, default="", help="SpatialVID metadata CSV")
    p.add_argument("--video_root", type=str, default="", help="Root directory for mp4 videos")
    p.add_argument("--video_list", type=str, default="",
                   help="Optional text file with one video path per line. Overrides --csv/--video_root")
    p.add_argument("--annotation_index", type=str, default="", help="Optional annotation index JSON")
    p.add_argument("--max_videos", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--img_size", type=int, default=518)
    p.add_argument("--patch_size", type=int, default=14)

    # StreamVGGT
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--token_stats", type=str, default="",
                   help="Token stats for normalized StreamVGGT features. Required for decoder recon.")
    p.add_argument("--levels", type=str, default="11",
                   help="Comma-separated StreamVGGT DPT levels to pool, e.g. 11 or 4,11,17,23")
    p.add_argument("--pool", type=str, default="clip", choices=["clip", "frame"],
                   help="clip: one vector/video. frame: one vector/frame.")

    # Decoder preservation test
    p.add_argument("--decoder_ckpt", type=str, default="",
                   help="Optional decoder_GLD checkpoint. Enables recon -> re-encode diagnostics.")
    p.add_argument("--decoder_keep_levels", type=str, default="11",
                   help="DPT levels kept when decoding before re-encoding. Use 11 for boundary-only GLD.")

    # DINO comparison
    p.add_argument("--with_dino", action="store_true")
    p.add_argument("--dino_model", type=str, default="facebook/dinov2-base")
    p.add_argument("--dino_pool", type=str, default="cls", choices=["cls", "patch_mean"])

    # Generated-token diagnostics
    p.add_argument("--generated_token_glob", type=str, default="",
                   help="Optional glob of saved generated token .pt files.")

    # Analysis / output
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--out_dir", type=str, default="analysis/latent_vis")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])

    return p.parse_args()


def parse_levels(text):
    return [int(x) for x in text.split(",") if x.strip()]


def load_video_index(args):
    captions = {}
    if args.annotation_index and os.path.exists(args.annotation_index):
        with open(args.annotation_index) as f:
            captions = json.load(f)

    items = []
    if args.video_list:
        with open(args.video_list) as f:
            for line in f:
                path = line.strip()
                if not path:
                    continue
                vid = Path(path).stem
                items.append({"video_id": vid, "video_path": path, "caption": ""})
    else:
        with open(args.csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row["id"]
                path = os.path.join(args.video_root, row["video path"].replace("videos/", ""))
                if not os.path.exists(path):
                    continue
                caption = captions.get(vid, {}).get("caption", "")
                items.append({"video_id": vid, "video_path": path, "caption": caption})

    if args.max_videos > 0:
        items = items[:args.max_videos]
    return items


def load_streamvggt(args, device, dtype):
    encoder = StreamVGGT(img_size=args.img_size, patch_size=args.patch_size, embed_dim=1024)
    state = torch.load(args.encoder_ckpt, map_location="cpu")
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device=device, dtype=dtype).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def load_decoder(path, device, patch_size):
    decoder = DPTHead(
        dim_in=2048,
        patch_size=patch_size,
        output_dim=4,
        activation="sigmoid",
        conf_activation="sigmoid",
    ).to(device=device, dtype=torch.float32)

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "ema_state_dict" in state:
            state = state["ema_state_dict"]
        elif "model" in state:
            state = state["model"]
        elif "ema" in state:
            state = state["ema"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    decoder.load_state_dict(state, strict=True)
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad_(False)
    return decoder


def pool_stream_tokens(tokens_list, levels, pool):
    pooled = []
    for lvl in levels:
        patch_tokens = tokens_list[lvl]  # [B,S,N,D] (already stripped)
        if pool == "clip":
            pooled.append(patch_tokens.mean(dim=(1, 2)))  # [B,D]
        else:
            pooled.append(patch_tokens.mean(dim=2).reshape(-1, patch_tokens.shape[-1]))  # [B*S,D]
    return torch.cat(pooled, dim=-1)


def zero_decoder_levels(tokens_list, keep_levels):
    out = list(tokens_list)
    for lvl in DPT_LEVELS:
        if lvl not in keep_levels:
            out[lvl] = torch.zeros_like(out[lvl])
    return out


@torch.no_grad()
def decode_tokens(decoder, tokens_list, frames, seq_len):
    preds, _ = decoder(
        [t.to(dtype=torch.float32) for t in tokens_list],
        images=frames.float(),
        patch_start_idx=0,
        frames_chunk_size=seq_len,
    )
    recon = preds.squeeze(2).permute(0, 1, 4, 2, 3).contiguous()
    return recon[:, :, :3].clamp(0, 1).float()


def load_dino(args, device):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(args.dino_model)
    model = AutoModel.from_pretrained(args.dino_model).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return processor, model


@torch.no_grad()
def extract_dino(processor, model, frames, args, device):
    from PIL import Image

    frames_np = (frames.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    images = [Image.fromarray(frames_np[i].transpose(1, 2, 0)) for i in range(frames_np.shape[0])]
    inputs = processor(images=images, return_tensors="pt").to(device)
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state.float()
    if args.dino_pool == "cls":
        feats = hidden[:, 0]
    else:
        feats = hidden[:, 1:].mean(dim=1)
    if args.pool == "clip":
        feats = feats.mean(dim=0, keepdim=True)
    return feats.cpu().numpy()


def pool_generated_tokens(path, levels, pool):
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        data = data.get("tokens", data.get("z", data))
    if not torch.is_tensor(data):
        raise ValueError(f"Unsupported generated token payload in {path}")

    # Accept [S,N,D], [L,S,N,D], or [B/L,S,N,D]-like saved tensors.
    if data.dim() == 3:
        z = data
    elif data.dim() == 4:
        z = data[0] if data.shape[0] == 1 or len(levels) == 1 else data[0]
    else:
        raise ValueError(f"Unsupported generated token shape {tuple(data.shape)} in {path}")

    if pool == "clip":
        return z.mean(dim=(0, 1)).numpy()[None]
    return z.mean(dim=1).numpy()


def pca_transform(x, n_components=2):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    coords = xc @ vt[:n_components].T
    var = (s ** 2) / max(x.shape[0] - 1, 1)
    ratio = var / max(var.sum(), 1e-12)
    return coords, ratio, var


def pairwise_dist(x):
    x = np.asarray(x, dtype=np.float64)
    sq = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2 * x @ x.T, 0.0)
    return np.sqrt(d2)


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman_corr(a, b):
    ra = rankdata(a)
    rb = rankdata(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def knn_overlap(d1, d2, k):
    n = d1.shape[0]
    if n <= 1:
        return float("nan")
    k = min(k, n - 1)
    overlaps = []
    for i in range(n):
        n1 = np.argsort(d1[i])[1:k + 1]
        n2 = np.argsort(d2[i])[1:k + 1]
        overlaps.append(len(set(n1).intersection(set(n2))) / k)
    return float(np.mean(overlaps))


def effective_rank(var):
    p = var / max(var.sum(), 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable, only .npz/.json will be saved: {exc}")
        return None


def plot_pca(plt, features, groups, title, path):
    coords, ratio, _ = pca_transform(features, 2)
    plt.figure(figsize=(7, 6))
    group_arr = np.asarray(groups)
    for group in sorted(set(groups)):
        mask = group_arr == group
        plt.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.75, label=group)
    plt.xlabel(f"PC1 ({ratio[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({ratio[1] * 100:.1f}%)")
    plt.title(title)
    plt.legend(markerscale=1.5, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_matrix(plt, mat, title, path):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(mat, cmap="magma")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_spectrum(plt, ratio, title, path, limit=128):
    plt.figure(figsize=(7, 4))
    n = min(limit, len(ratio))
    plt.plot(np.arange(1, n + 1), ratio[:n])
    plt.xlabel("principal component")
    plt.ylabel("explained variance ratio")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    levels = parse_levels(args.levels)
    decoder_keep_levels = set(parse_levels(args.decoder_keep_levels))
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device(args.device if torch.npu.is_available() or args.device == "cpu" else "cpu")

    if args.decoder_ckpt and not args.token_stats:
        raise ValueError("--token_stats is required when --decoder_ckpt is used")

    items = load_video_index(args)
    if not items:
        raise ValueError("No readable videos found")
    print(f"Loaded {len(items)} videos for diagnostics")

    encoder = load_streamvggt(args, device, dtype)
    level_stats = load_token_stats(args.token_stats, device, dtype=torch.float32) if args.token_stats else None
    decoder = load_decoder(args.decoder_ckpt, device, args.patch_size) if args.decoder_ckpt else None
    dino = load_dino(args, device) if args.with_dino else None

    vggt_features = []
    vggt_groups = []
    vggt_ids = []
    dino_features = []
    dino_ids = []

    for idx, item in enumerate(items):
        print(f"[{idx + 1}/{len(items)}] {item['video_id']}")
        frames = read_video_frames(item["video_path"], args.seq_len, args.img_size)
        frames_batched = frames.unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            tokens_list, patch_start_idx = encoder(frames_batched)
            tokens_list = strip_special_tokens(tokens_list, patch_start_idx)
            if level_stats is not None:
                tokens_list = normalize_tokens(tokens_list, level_stats)

            real_feat = pool_stream_tokens(tokens_list, levels, args.pool).cpu().numpy()
            for row_idx, row in enumerate(real_feat):
                suffix = f":f{row_idx:03d}" if args.pool == "frame" else ""
                vggt_features.append(row)
                vggt_groups.append("real_vggt")
                vggt_ids.append(item["video_id"] + suffix)

            if decoder is not None:
                decode_tokens_list = zero_decoder_levels(tokens_list, decoder_keep_levels)
                recon = decode_tokens(decoder, decode_tokens_list, frames_batched, args.seq_len)
                recon_tokens, recon_psi = encoder(recon.to(device=device, dtype=dtype))
                recon_tokens = strip_special_tokens(recon_tokens, recon_psi)
                if level_stats is not None:
                    recon_tokens = normalize_tokens(recon_tokens, level_stats)
                recon_feat = pool_stream_tokens(recon_tokens, levels, args.pool).cpu().numpy()
                for row_idx, row in enumerate(recon_feat):
                    suffix = f":f{row_idx:03d}" if args.pool == "frame" else ""
                    vggt_features.append(row)
                    vggt_groups.append("recon_reencoded_vggt")
                    vggt_ids.append(item["video_id"] + suffix)

        if dino is not None:
            processor, dino_model = dino
            dino_feat = extract_dino(processor, dino_model, frames, args, device)
            for row_idx, row in enumerate(dino_feat):
                suffix = f":f{row_idx:03d}" if args.pool == "frame" else ""
                dino_features.append(row)
                dino_ids.append(item["video_id"] + suffix)

    if args.generated_token_glob:
        gen_paths = sorted(glob.glob(args.generated_token_glob))
        print(f"Loading {len(gen_paths)} generated token files")
        for path in gen_paths:
            gen_feat = pool_generated_tokens(path, levels, args.pool)
            for row_idx, row in enumerate(gen_feat):
                suffix = f":f{row_idx:03d}" if args.pool == "frame" else ""
                vggt_features.append(row)
                vggt_groups.append("generated_tokens")
                vggt_ids.append(Path(path).stem + suffix)

    vggt_features = np.asarray(vggt_features, dtype=np.float32)
    dino_features = np.asarray(dino_features, dtype=np.float32) if dino_features else None
    vggt_groups = np.asarray(vggt_groups)
    vggt_ids = np.asarray(vggt_ids)
    dino_ids = np.asarray(dino_ids) if dino_features is not None else None

    np.savez_compressed(
        os.path.join(args.out_dir, "features.npz"),
        vggt_features=vggt_features,
        vggt_groups=vggt_groups,
        vggt_ids=vggt_ids,
        dino_features=dino_features if dino_features is not None else np.empty((0, 0), dtype=np.float32),
        dino_ids=dino_ids if dino_ids is not None else np.empty((0,), dtype=str),
    )

    _, vggt_ratio, vggt_var = pca_transform(vggt_features, min(vggt_features.shape))
    stats = {
        "num_vggt_points": int(vggt_features.shape[0]),
        "vggt_dim": int(vggt_features.shape[1]),
        "vggt_effective_rank": effective_rank(vggt_var),
        "vggt_top10_explained_variance": [float(x) for x in vggt_ratio[:10]],
        "groups": {g: int((vggt_groups == g).sum()) for g in sorted(set(vggt_groups.tolist()))},
        "levels": levels,
        "pool": args.pool,
        "streamvggt_note": (
            "This script uses StreamVGGT aggregated tokens after causal/global attention. "
            "Unlike single-image VGGT diagnostics, clip-level pooling includes temporal geometry and causal context."
        ),
    }

    real_mask = vggt_groups == "real_vggt"
    if real_mask.sum() > 1:
        real_vggt = vggt_features[real_mask]
        stats["real_vggt_effective_rank"] = effective_rank(pca_transform(real_vggt, min(real_vggt.shape))[2])
        stats["real_vggt_token_norm_mean"] = float(np.linalg.norm(real_vggt, axis=1).mean())
        stats["real_vggt_token_norm_std"] = float(np.linalg.norm(real_vggt, axis=1).std())

    recon_mask = vggt_groups == "recon_reencoded_vggt"
    if real_mask.sum() == recon_mask.sum() and real_mask.sum() > 0:
        real = vggt_features[real_mask]
        recon = vggt_features[recon_mask]
        stats["real_recon_l2_mean"] = float(np.linalg.norm(real - recon, axis=1).mean())
        stats["real_recon_cos_mean"] = float(np.mean(np.sum(real * recon, axis=1) / (
            np.linalg.norm(real, axis=1) * np.linalg.norm(recon, axis=1) + 1e-8
        )))

    if dino_features is not None and real_mask.sum() == dino_features.shape[0] and real_mask.sum() > 2:
        d_vggt = pairwise_dist(vggt_features[real_mask])
        d_dino = pairwise_dist(dino_features)
        tri = np.triu_indices(d_vggt.shape[0], k=1)
        stats["vggt_dino_distance_spearman"] = spearman_corr(d_vggt[tri], d_dino[tri])
        stats[f"vggt_dino_knn_overlap@{args.knn_k}"] = knn_overlap(d_vggt, d_dino, args.knn_k)
        np.save(os.path.join(args.out_dir, "distance_real_vggt.npy"), d_vggt)
        np.save(os.path.join(args.out_dir, "distance_dino.npy"), d_dino)

    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    plt = try_import_matplotlib()
    if plt is not None:
        plot_pca(
            plt,
            vggt_features,
            vggt_groups.tolist(),
            "StreamVGGT latent PCA",
            os.path.join(args.out_dir, "pca_streamvggt.png"),
        )
        plot_spectrum(
            plt,
            vggt_ratio,
            "StreamVGGT PCA spectrum",
            os.path.join(args.out_dir, "pca_spectrum_streamvggt.png"),
        )
        if real_mask.sum() > 1:
            plot_matrix(
                plt,
                pairwise_dist(vggt_features[real_mask]),
                "Real StreamVGGT pairwise distance",
                os.path.join(args.out_dir, "distance_real_streamvggt.png"),
            )
        if dino_features is not None:
            dino_coords, dino_ratio, dino_var = pca_transform(dino_features, min(dino_features.shape))
            stats["dino_effective_rank"] = effective_rank(dino_var)
            stats["dino_top10_explained_variance"] = [float(x) for x in dino_ratio[:10]]
            with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
                json.dump(stats, f, indent=2)

            plt.figure(figsize=(7, 6))
            plt.scatter(dino_coords[:, 0], dino_coords[:, 1], s=18, alpha=0.75)
            plt.xlabel(f"PC1 ({dino_ratio[0] * 100:.1f}%)")
            plt.ylabel(f"PC2 ({dino_ratio[1] * 100:.1f}%)")
            plt.title("DINO feature PCA")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "pca_dino.png"), dpi=180)
            plt.close()

            if dino_features.shape[0] > 1:
                plot_matrix(
                    plt,
                    pairwise_dist(dino_features),
                    "DINO pairwise distance",
                    os.path.join(args.out_dir, "distance_dino.png"),
                )

    print(f"Saved diagnostics to {args.out_dir}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
