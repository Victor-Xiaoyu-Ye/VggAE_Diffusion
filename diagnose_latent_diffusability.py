#!/usr/bin/env python3
"""Measure diffusability statistics of compact latents (SSVAE criteria).

Why
---
MIRA's ablation showed reconstruction PSNR and generation quality are
anti-correlated when the latent manifold is not smooth; SSVAE
(arXiv:2512.05394) identifies two measurable properties that predict how
well a video latent trains a diffusion model:

  1. spatio-temporal frequency spectrum biased toward LOW frequencies;
  2. channel eigenspectrum dominated by FEW modes (low effective rank
     relative to nominal dim).

A z_tex stream engineered to carry high frequency is exactly the kind of
latent that can pass the PSNR gate and still be hard to diffuse. This
script turns "is this latent diffusable" into numbers, measured per stream
(z_geo / z_tex) from an E5 checkpoint, so the E5 decision gate can require
BOTH recon PSNR >= 25 AND sane spectra before the cache rebuild.

Reported per stream
-------------------
- spatial_spectrum: radially averaged 2D FFT power, low/mid/high band split
- temporal_spectrum: per-position FFT power over the 8 frames, DC-removed
- high_freq_energy_ratio: fraction of spatial power above Nyquist/2
- effective_rank: exp(entropy of channel eigenvalue distribution)
- top8_var / top32_var: variance captured by leading modes
- channel_kurtosis: heavy-tailedness (Gaussian ~ 3)

Usage (single H200 card)
------------------------
python diagnose_latent_diffusability.py \
    --ckpt outputs/h200/probes/e5_oracle_s2d/checkpoint_latest.pt \
    --csv  <eval csv> --video_root <videos> \
    --encoder_ckpt <streamvggt ckpt> --output_json <path>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from streamvggt.models.streamvggt import StreamVGGT
from models.dpt_latent_decoder import CompactCompressor
from models.texture_encoder import TextureEncoder
from data.video_dataset import SpatialVidDataset, collate_fn
from data.token_utils import strip_special_tokens
from utils.device import get_device, get_device_name, resolve_dtype
from utils.encoder_loader import load_encoder_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description='Latent diffusability diagnostics')
    p.add_argument('--ckpt', type=str, required=True,
                   help='E5 checkpoint (checkpoint_latest.pt)')
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--video_root', type=str, required=True)
    p.add_argument('--encoder_ckpt', type=str, required=True)
    p.add_argument('--num_clips', type=int, default=64)
    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--target_size', type=int, default=518)
    p.add_argument('--clip_duration_seconds', type=float, default=1.0)
    p.add_argument('--dtype', type=str, default='bf16')
    p.add_argument('--output_json', type=str, required=True)
    return p.parse_args()


def spectrum_stats(z: torch.Tensor) -> dict:
    """z: [N, S, G, G, C] float32. Returns diffusability statistics."""
    N, S, G, _, C = z.shape
    x = z.permute(0, 1, 4, 2, 3).reshape(-1, G, G)  # [N*S*C, G, G]
    x = x - x.mean(dim=(-2, -1), keepdim=True)

    # --- spatial spectrum: radially averaged power ---
    f = torch.fft.rfft2(x.float(), norm='ortho')
    power = f.abs().pow(2).mean(0)  # [G, G//2+1]
    fy = torch.fft.fftfreq(G)[:, None].abs()
    fx = torch.fft.rfftfreq(G)[None, :].abs()
    radius = torch.sqrt(fy ** 2 + fx ** 2)
    total = power.sum().item()
    bands = {}
    for name, lo, hi in (('low', 0.0, 0.125), ('mid', 0.125, 0.25),
                         ('high', 0.25, 1.0)):
        mask = (radius >= lo) & (radius < hi)
        bands[name] = float(power[mask].sum().item() / max(total, 1e-12))

    # --- temporal spectrum over S frames ---
    xt = z.reshape(N, S, -1).float()
    xt = xt - xt.mean(dim=1, keepdim=True)
    ft = torch.fft.rfft(xt, dim=1, norm='ortho').abs().pow(2).mean(dim=(0, 2))
    ft_total = ft.sum().item()
    # bins: 0 is DC (removed by centering, ~0); last bin is Nyquist.
    temporal_high = float(ft[len(ft) // 2:].sum().item() / max(ft_total, 1e-12))

    # --- channel eigenspectrum ---
    flat = z.reshape(-1, C).float()
    flat = flat - flat.mean(0, keepdim=True)
    # SVD on [rows, C]; rows can be large — subsample to bound memory.
    if flat.shape[0] > 200_000:
        idx = torch.randperm(flat.shape[0])[:200_000]
        flat = flat[idx]
    cov = flat.T @ flat / max(flat.shape[0] - 1, 1)
    eig = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0)
    ratio = (eig / eig.sum().clamp(min=1e-12)).cpu().numpy()
    entropy = -np.sum(ratio * np.log(ratio + 1e-12))

    std = flat.std(0)
    zn = (flat - flat.mean(0)) / std.clamp(min=1e-6)
    kurt = float((zn ** 4).mean().item())

    return {
        'spatial_band_low': bands['low'],
        'spatial_band_mid': bands['mid'],
        'spatial_band_high': bands['high'],
        'temporal_high_ratio': temporal_high,
        'effective_rank': float(np.exp(entropy)),
        'nominal_dim': int(C),
        'top8_var': float(ratio[:8].sum()),
        'top32_var': float(ratio[:32].sum()),
        'channel_kurtosis': kurt,
        'channel_std_mean': float(std.mean().item()),
        'channel_std_min': float(std.min().item()),
        'channel_std_max': float(std.max().item()),
    }


def main():
    args = parse_args()
    device = get_device(0)
    dtype = resolve_dtype(args.dtype)
    print(f'Device: {get_device_name(device)}')

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    ck_args = ckpt.get('args', {})
    levels = list(ck_args.get('levels', [4, 11, 17, 23]))
    geo_dim = int(ck_args.get('geo_dim', 256))
    tex_dim = int(ck_args.get('tex_dim', 256))
    latent_grid = int(ck_args.get('latent_grid', 18))
    tex_pack = ck_args.get('tex_pack', 'avgpool')
    tex_base_ch = int(ck_args.get('tex_base_ch', 64))

    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    load_encoder_checkpoint(encoder, args.encoder_ckpt)
    encoder = encoder.to(device=device, dtype=dtype).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    compressor = CompactCompressor(
        levels=levels, token_dim=int(ck_args.get('token_dim', 2048)),
        cdim=geo_dim, latent_grid=latent_grid,
        input_grid=args.target_size // 14)
    tex_encoder = TextureEncoder(
        out_dim=tex_dim, out_grid=latent_grid, base_ch=tex_base_ch,
        img_size=args.target_size, pack_mode=tex_pack)

    state = ckpt['model']
    compressor.load_state_dict(
        {k[len('compressor.'):]: v for k, v in state.items()
         if k.startswith('compressor.')})
    tex_encoder.load_state_dict(
        {k[len('tex_encoder.'):]: v for k, v in state.items()
         if k.startswith('tex_encoder.')})
    compressor = compressor.to(device).eval()
    tex_encoder = tex_encoder.to(device).eval()

    dataset = SpatialVidDataset(
        csv_path=args.csv, video_root=args.video_root,
        seq_len=args.seq_len, target_size=args.target_size,
        max_videos=args.num_clips,
        clip_duration_seconds=args.clip_duration_seconds,
        temporal_jitter=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    geos, texs = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            frames = batch['frames'].to(device=device, dtype=dtype)
            tokens_list, psi = encoder(frames)
            stripped = strip_special_tokens(tokens_list, psi)
            z_geo = compressor([t.float() for t in stripped])
            z_geo = z_geo.permute(0, 1, 3, 4, 2)  # [B,S,G,G,C]
            z_tex = tex_encoder(frames.float())
            geos.append(z_geo.float().cpu())
            texs.append(z_tex.float().cpu())
            if (i + 1) % 16 == 0:
                print(f'  encoded {i + 1}/{len(loader)} clips')

    z_geo = torch.cat(geos, 0)
    z_tex = torch.cat(texs, 0)
    print(f'z_geo {tuple(z_geo.shape)}  z_tex {tuple(z_tex.shape)}')

    report = {
        'ckpt': args.ckpt,
        'tex_pack': tex_pack,
        'num_clips': int(z_geo.shape[0]),
        'z_geo': spectrum_stats(z_geo),
        'z_tex': spectrum_stats(z_tex),
    }
    # Residual stream (what diffusion actually predicts): z_t - z_0.
    resid_geo = z_geo[:, 1:] - z_geo[:, :1]
    resid_tex = z_tex[:, 1:] - z_tex[:, :1]
    report['z_geo_residual'] = spectrum_stats(resid_geo)
    report['z_tex_residual'] = spectrum_stats(resid_tex)

    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f'\nSaved: {args.output_json}')
    print(
        '\nReading the numbers (SSVAE criteria):\n'
        '  GOOD for diffusion: spatial_band_low dominant, temporal_high_ratio\n'
        '  small, top32_var high (few-mode channel spectrum), kurtosis ~3.\n'
        '  BAD: z_tex spatial_band_high >> z_geo band_high with flat\n'
        '  eigenspectrum -> tex stream will be hard to diffuse even if the\n'
        '  PSNR gate passes; prefer match_geo reg / packing changes first.')


if __name__ == '__main__':
    main()
