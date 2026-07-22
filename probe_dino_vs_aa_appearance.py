#!/usr/bin/env python3
"""Diagnostic: why DINO init does not imply RGB-reconstructable VGGT tokens.

VGGT/StreamVGGT loads DINOv2 as aggregator.patch_embed. That does NOT mean
the tokens we decode (AA frame+global intermediates) still carry DINO-style
appearance. This probe measures, on the same images:

  H1  Architecture: we decode AA levels, not patch_embed DINO tokens.
  H2  Representation drift: cosine(DINO_out, AA_level) is low after geometry AA.
  H3  Color probe: linear map token -> RGB patch mean is better from DINO than AA.
  H4  High-freq probe: corr(||token||, RGB Laplacian) DINO vs AA levels.
  H5  Even pure DINO is not a free RGB codec (VGGT_as_RAE ~14-17 PSNR).

Writes NDJSON to the debug log path and prints a short summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from streamvggt.models.streamvggt import StreamVGGT
from data.token_utils import strip_special_tokens

DEBUG_LOG = Path("/home/yexiaoyu/.cursor/debug-88822c.log")
SESSION_ID = "88822c"


def dlog(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "dino-aa"):
    # #region agent log
    payload = {
        "sessionId": SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(DEBUG_LOG, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    # #endregion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--target_size", type=int, default=518)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=2)
    p.add_argument("--num_batches", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--image_source",
        type=str,
        default="probe_grids",
        choices=["probe_grids", "synthetic"],
    )
    p.add_argument(
        "--grid_glob",
        type=str,
        default="outputs/h200/probes/e1_raw_4lvl_proj512/samples/epoch*.png",
    )
    return p.parse_args()


def make_synthetic_batch(B, S, H, W, device, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    base = torch.rand(B, S, 3, 1, 1, generator=g)
    yy = torch.linspace(0, 1, H).view(1, 1, 1, H, 1)
    xx = torch.linspace(0, 1, W).view(1, 1, 1, 1, W)
    noise = torch.randn(B, S, 3, H, W, generator=g) * 0.15
    cy = ((torch.arange(H) // 8) % 2).float().view(1, 1, 1, H, 1)
    cx = ((torch.arange(W) // 8) % 2).float().view(1, 1, 1, 1, W)
    imgs = (base + 0.35 * yy + 0.35 * xx + noise + 0.2 * cy * cx).clamp(0, 1)
    return imgs.to(device)


def load_orig_frames_from_grids(glob_pat, H, W):
    """E1 grids are S rows of [orig | recon]; take left half of each row."""
    from PIL import Image
    import torchvision.transforms.functional as TF

    frames = []
    for p in sorted(glob.glob(glob_pat))[:8]:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        row_h = w // 2
        if row_h <= 0 or h % row_h != 0:
            left = im.crop((0, 0, w // 2, h))
            frames.append(TF.to_tensor(TF.resize(left, [H, W])))
            continue
        s = h // row_h
        for si in range(s):
            cell = im.crop((0, si * row_h, w // 2, (si + 1) * row_h))
            frames.append(TF.to_tensor(TF.resize(cell, [H, W])))
    if not frames:
        raise RuntimeError(f"no frames from {glob_pat}")
    return frames


def make_batch_from_frames(frames, B, S, device, offset=0):
    need = B * S
    picked = [frames[(offset + i) % len(frames)] for i in range(need)]
    return torch.stack(picked, dim=0).view(B, S, 3, *picked[0].shape[1:]).to(device)


def rgb_patch_mean(images, patch=14):
    BS, C, H, W = images.shape
    gh, gw = H // patch, W // patch
    x = images[:, :, : gh * patch, : gw * patch]
    x = x.reshape(BS, C, gh, patch, gw, patch).mean(dim=(3, 5))
    return x.permute(0, 2, 3, 1).reshape(BS, gh * gw, C)


def rgb_patch_laplacian_energy(images, patch=14):
    BS, C, H, W = images.shape
    y = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
    ker = images.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    lap = F.conv2d(y.unsqueeze(1), ker, padding=1).abs()
    gh, gw = H // patch, W // patch
    lap = lap[:, :, : gh * patch, : gw * patch]
    lap = lap.reshape(BS, 1, gh, patch, gw, patch).mean(dim=(3, 5))
    return lap.reshape(BS, gh * gw)


@torch.no_grad()
def extract_dino_and_aa(encoder, images):
    B, S, C, H, W = images.shape
    agg = encoder.aggregator
    mean = agg._resnet_mean.to(images.device)
    std = agg._resnet_std.to(images.device)
    x = ((images - mean) / std).reshape(B * S, C, H, W)
    dino_out = agg.patch_embed(x)
    if isinstance(dino_out, dict):
        dino_tokens = dino_out["x_norm_patchtokens"]
    else:
        dino_tokens = dino_out
    dino_tokens = dino_tokens.reshape(B, S, dino_tokens.shape[1], dino_tokens.shape[2])
    tokens_list, psi = encoder(images)
    tokens_list = strip_special_tokens(tokens_list, psi)
    return dino_tokens, tokens_list, psi


def fit_linear_rgb_probe(tokens, rgb_mean, ridge=1.0, max_rows=8000):
    """Ridge on PCA-reduced features (top-k) to avoid ill-conditioned 1024/2048 fits."""
    M, N, D = tokens.shape
    X = tokens.reshape(M * N, D).float()
    Y = rgb_mean.reshape(M * N, 3).float()
    if X.shape[0] > max_rows:
        idx = torch.randperm(X.shape[0], device=X.device)[:max_rows]
        X, Y = X[idx], Y[idx]
    # shuffle before split
    perm = torch.randperm(X.shape[0], device=X.device)
    X, Y = X[perm], Y[perm]
    mu, sig = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-4)
    X = (X - mu) / sig
    # PCA via SVD, keep top-64
    k = min(64, X.shape[0] - 1, D)
    # economy SVD on centered X
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    Xp = X @ Vh[:k].T
    n_tr = int(0.7 * Xp.shape[0])
    Xtr, Ytr, Xva, Yva = Xp[:n_tr], Y[:n_tr], Xp[n_tr:], Y[n_tr:]
    XtX = Xtr.T @ Xtr + ridge * torch.eye(k, device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(XtX, Xtr.T @ Ytr)
    pred = Xva @ W
    mse = F.mse_loss(pred, Yva).item()
    ss_res = ((pred - Yva) ** 2).sum().item()
    ss_tot = ((Yva - Yva.mean(0)) ** 2).sum().item()
    return {"mse": mse, "r2": 1.0 - ss_res / max(ss_tot, 1e-8), "pca_k": k}


def token_energy_corr_with_hf(tokens, hf_energy):
    e = tokens.float().norm(dim=-1).reshape(-1)
    h = hf_energy.reshape(-1).float()
    e = (e - e.mean()) / e.std().clamp_min(1e-4)
    h = (h - h.mean()) / h.std().clamp_min(1e-4)
    return (e * h).mean().item()


def mean_cosine(a, b):
    a = F.normalize(a.float(), dim=-1)
    if b.shape[-1] == 2 * a.shape[-1]:
        b1 = F.normalize(b[..., : a.shape[-1]].float(), dim=-1)
        b2 = F.normalize(b[..., a.shape[-1] :].float(), dim=-1)
        c1 = (a * b1).sum(-1).mean().item()
        c2 = (a * b2).sum(-1).mean().item()
        return {"cos_frame_half": c1, "cos_global_half": c2, "cos_mean_half": 0.5 * (c1 + c2)}
    b = F.normalize(b.float(), dim=-1)
    return {"cos": (a * b).sum(-1).mean().item()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    state = torch.load(args.encoder_ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state and not any(
        k.startswith("aggregator.") for k in state
    ):
        state = state["model"]
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    pe = encoder.aggregator.patch_embed
    n_pe = sum(p.numel() for p in pe.parameters())
    n_aa = sum(
        p.numel()
        for n, p in encoder.named_parameters()
        if "frame_blocks" in n or "global_blocks" in n
    )
    dlog(
        "H1",
        "probe_dino_vs_aa_appearance.py:load",
        "Architecture: DINO is patch_embed; decode path uses AA outputs",
        {
            "patch_embed_type": type(pe).__name__,
            "patch_embed_params_M": round(n_pe / 1e6, 2),
            "aa_blocks_params_M": round(n_aa / 1e6, 2),
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "decode_path": "AA intermediates 2048-d, NOT patch_embed alone",
        },
    )

    all_dino, all_aa = [], {4: [], 11: [], 17: [], 23: []}
    all_rgb_mean, all_hf = [], []
    cos_accum = {lvl: [] for lvl in (4, 11, 17, 23)}

    real_frames = None
    if args.image_source == "probe_grids":
        real_frames = load_orig_frames_from_grids(
            args.grid_glob, args.target_size, args.target_size
        )
        dlog(
            "H5",
            "probe_dino_vs_aa_appearance.py:data",
            "Loaded real SpatialVID frames from E1 grids",
            {"n_frames": len(real_frames), "glob": args.grid_glob},
        )

    for bi in range(args.num_batches):
        if args.image_source == "synthetic":
            images = make_synthetic_batch(
                args.batch_size,
                args.seq_len,
                args.target_size,
                args.target_size,
                device,
                args.seed + bi,
            )
        else:
            images = make_batch_from_frames(
                real_frames,
                args.batch_size,
                args.seq_len,
                device,
                offset=bi * args.batch_size * args.seq_len,
            )

        dino_tok, aa_list, psi = extract_dino_and_aa(encoder, images)
        B, S = images.shape[:2]
        flat = images.reshape(B * S, 3, args.target_size, args.target_size)
        rgb_m = rgb_patch_mean(flat)
        hf = rgb_patch_laplacian_energy(flat)

        all_dino.append(dino_tok.cpu())
        all_rgb_mean.append(rgb_m.cpu())
        all_hf.append(hf.cpu())
        for lvl in (4, 11, 17, 23):
            aa = aa_list[lvl].cpu()
            all_aa[lvl].append(aa)
            cos_accum[lvl].append(mean_cosine(dino_tok.cpu(), aa))

        dlog(
            "H1",
            "probe_dino_vs_aa_appearance.py:shapes",
            "Token shapes for one batch",
            {
                "batch": bi,
                "dino_shape": list(dino_tok.shape),
                "aa_level23_shape": list(aa_list[23].shape),
                "psi_special_tokens": int(psi),
            },
        )

    dino_cat = torch.cat(all_dino, dim=0).to(device)
    rgb_cat = torch.cat(all_rgb_mean, dim=0).to(device)
    hf_cat = torch.cat(all_hf, dim=0).to(device)
    Bd, Sd, N, Cd = dino_cat.shape
    dino_flat = dino_cat.reshape(Bd * Sd, N, Cd)
    rgb_flat = rgb_cat
    hf_flat = hf_cat

    probe_dino = fit_linear_rgb_probe(dino_flat, rgb_flat)
    dlog(
        "H3",
        "probe_dino_vs_aa_appearance.py:color_probe_dino",
        "Linear RGB-mean probe from frozen DINO patch_embed tokens",
        probe_dino,
    )
    hf_corr_dino = token_energy_corr_with_hf(dino_flat, hf_flat)
    dlog(
        "H4",
        "probe_dino_vs_aa_appearance.py:hf_corr_dino",
        "Corr(||dino||, RGB Laplacian energy)",
        {"corr": hf_corr_dino},
    )

    summary = {
        "dino_color_r2": probe_dino["r2"],
        "dino_color_mse": probe_dino["mse"],
        "dino_hf_corr": hf_corr_dino,
        "levels": {},
    }

    for lvl in (4, 11, 17, 23):
        aa_cat = torch.cat(all_aa[lvl], dim=0).to(device)
        Ba, Sa, Na, Ca = aa_cat.shape
        aa_flat = aa_cat.reshape(Ba * Sa, Na, Ca)
        aa_half = 0.5 * (aa_flat[..., :Cd] + aa_flat[..., Cd:])
        probe_aa = fit_linear_rgb_probe(aa_half, rgb_flat)
        probe_aa_full = fit_linear_rgb_probe(aa_flat, rgb_flat)
        hf_corr = token_energy_corr_with_hf(aa_flat, hf_flat)
        cos_keys = cos_accum[lvl][0].keys()
        cos_mean = {
            k: sum(d[k] for d in cos_accum[lvl]) / len(cos_accum[lvl]) for k in cos_keys
        }
        level_data = {
            "color_r2_avg_half": probe_aa["r2"],
            "color_mse_avg_half": probe_aa["mse"],
            "color_r2_full2048": probe_aa_full["r2"],
            "color_mse_full2048": probe_aa_full["mse"],
            "hf_corr": hf_corr,
            **{f"mean_{k}": v for k, v in cos_mean.items()},
        }
        summary["levels"][str(lvl)] = level_data
        dlog("H2", f"probe_dino_vs_aa_appearance.py:level{lvl}", f"AA level {lvl}", level_data)
        dlog(
            "H3",
            f"probe_dino_vs_aa_appearance.py:color_aa{lvl}",
            f"Color probe AA{lvl} vs DINO",
            {
                "dino_r2": probe_dino["r2"],
                "aa_r2_full": probe_aa_full["r2"],
                "delta_r2_dino_minus_aa": probe_dino["r2"] - probe_aa_full["r2"],
            },
        )
        dlog(
            "H4",
            f"probe_dino_vs_aa_appearance.py:hf_aa{lvl}",
            f"HF corr AA{lvl} vs DINO",
            {
                "dino_hf_corr": hf_corr_dino,
                "aa_hf_corr": hf_corr,
                "delta_dino_minus_aa": hf_corr_dino - hf_corr,
            },
        )

    dlog(
        "H5",
        "probe_dino_vs_aa_appearance.py:h5",
        "DINO init != RGB codec",
        {
            "vggt_as_rae_dinov2_epoch3_psnr": 17.5,
            "vggt_as_rae_mae_dino_hq_oft_val_psnr": 14.125,
            "e1_streamvggt_aa_ceiling_psnr": 20.60,
            "claim": "Loading DINO weights does not imply decode tokens are appearance-complete",
        },
    )

    out = Path("outputs/h200/probes/dino_vs_aa_appearance_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out} and NDJSON -> {DEBUG_LOG}")


if __name__ == "__main__":
    main()
