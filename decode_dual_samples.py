#!/usr/bin/env python3
"""Decode E7 sampled latents (.pt from train_dual_diffusion.py) to RGB grids.

Runs on the H200 box (CUDA) — the 910B trainer deliberately saves raw latents
instead of decoding, keeping DualStreamDecoder's interpolate ops off the NPU.

Usage:
    python decode_dual_samples.py \
        --samples <dir or single .pt> \
        --dual_ae_ckpt <R5 checkpoint> \
        --out_dir <dir>
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from models.dual_stream_decoder import DualStreamDecoder


def parse_args():
    p = argparse.ArgumentParser(description='Decode E7 sampled latents')
    p.add_argument('--samples', type=str, required=True,
                   help='sample .pt file or a directory of them')
    p.add_argument('--dual_ae_ckpt', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def denormalize(x1, stats, target_mode, z0):
    """x1 [B,S-1,N,C] normalized -> absolute latents [B,S,N,C] (with z0)."""
    if target_mode == 'absolute':
        m = stats['abs_mean'][1:].unsqueeze(0).unsqueeze(2)
        s = stats['abs_std'][1:].unsqueeze(0).unsqueeze(2)
        z_future = x1 * s + m
    else:
        m = stats['res_mean'].unsqueeze(0).unsqueeze(2)
        s = stats['res_std'].unsqueeze(0).unsqueeze(2)
        z_future = z0 + (x1 * s + m)
    return torch.cat([z0, z_future], dim=1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.dual_ae_ckpt, map_location='cpu',
                      weights_only=False)
    ck = ckpt.get('args', {})
    grid = int(ck.get('latent_grid', 18))
    geo_dim = int(ck.get('geo_dim', 256))
    tex_dim = int(ck.get('tex_dim', 256))
    decoder = DualStreamDecoder(
        geo_dim=geo_dim, tex_dim=tex_dim,
        base_dim=int(ck.get('decoder_base_dim', 384)),
        img_size=int(ck.get('target_size', 518)), latent_grid=grid,
        use_checkpoint=False)
    decoder.load_state_dict(
        {k[len('decoder.'):]: v for k, v in ckpt['model'].items()
         if k.startswith('decoder.')})
    decoder = decoder.to(device).eval()

    paths = sorted(glob.glob(os.path.join(args.samples, '*.pt'))) \
        if os.path.isdir(args.samples) else [args.samples]
    print(f'{len(paths)} sample files')

    from PIL import Image
    for path in paths:
        pack = torch.load(path, map_location='cpu', weights_only=False)
        name = os.path.splitext(os.path.basename(path))[0]
        for key in ('sampled_x1', 'target_x1'):
            z = denormalize(pack[key], pack['stats'], pack['target_mode'],
                            pack['z0_unnorm'])           # [B,S,N,C]
            B, S, N, C = z.shape
            z = z.reshape(B, S, grid, grid, C).to(device)
            z_geo, z_tex = z[..., :geo_dim], z[..., geo_dim:]
            with torch.no_grad():
                rgb = decoder(z_geo.float(), z_tex.float(),
                              frames_chunk_size=4)       # [B,S,H,W,3]
            rgb = rgb[..., :3].clamp(0, 1)
            rows = torch.cat([rgb[0, s] for s in range(S)], dim=0)
            img = (rows.cpu().numpy() * 255).astype(np.uint8)
            out = os.path.join(args.out_dir, f'{name}_{key}.png')
            Image.fromarray(img).save(out)
            print(f'  {out}')


if __name__ == '__main__':
    main()
