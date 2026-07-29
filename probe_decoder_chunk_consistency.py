#!/usr/bin/env python3
"""Compare full-sequence and chunked behavior of the legacy temporal decoder."""

from __future__ import annotations

import argparse

import torch

from models.dual_stream_decoder import DualStreamDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--latent', required=True)
    parser.add_argument('--chunk', type=int, default=4)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(
        args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint.get('args', {})
    geo_dim = int(config.get('geo_dim', 256))
    tex_dim = int(config.get('tex_dim', 256))
    grid = int(config.get('latent_grid', 18))
    model = DualStreamDecoder(
        geo_dim, tex_dim, int(config.get('decoder_base_dim', 384)),
        int(config.get('target_size', 518)), grid, use_checkpoint=False)
    model.load_state_dict({
        key[len('decoder.'):]: value
        for key, value in checkpoint['model'].items()
        if key.startswith('decoder.')
    }, strict=False)
    model.to(device).eval()
    latent = torch.load(args.latent, map_location='cpu', weights_only=False)
    latent = latent['latent'] if isinstance(latent, dict) else latent
    latent = latent.unsqueeze(0).to(device)
    geo, tex = latent[..., :geo_dim], latent[..., geo_dim:geo_dim + tex_dim]
    with torch.no_grad():
        full = model(geo, tex, None)
        chunked = model(geo, tex, args.chunk)
    difference = (full - chunked).abs().mean((0, 2, 3, 4))
    print({
        'mean_abs': float(difference.mean()),
        'per_frame': [float(value) for value in difference],
    })


if __name__ == '__main__':
    main()
