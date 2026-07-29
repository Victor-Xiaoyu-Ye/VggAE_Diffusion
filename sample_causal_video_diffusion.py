#!/usr/bin/env python3
"""Autoregressive overlapping rollout for an R7 latent world model."""

from __future__ import annotations

import argparse

import torch

from models.compact_dit import CompactLatentDiT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--initial_latent', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--chunks', type=int, default=8)
    parser.add_argument('--sample_steps', type=int, default=30)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(
        args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['args']
    model = CompactLatentDiT(
        latent_dim=config['latent_dim'],
        num_tokens=config.get('latent_grid', 18) ** 2,
        model_dim=config['model_dim'],
        spatial_depth=config['spatial_depth'],
        temporal_depth=config['temporal_depth'],
        num_heads=config['num_heads'],
        seq_len=config['future_chunks'],
        text_cond=False,
        i0_condition=True,
        block_schedule='interleaved',
    ).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    latent = torch.load(
        args.initial_latent, map_location='cpu', weights_only=False)
    latent = latent['latent'] if isinstance(latent, dict) else latent
    history = latent.unsqueeze(0).to(device)
    offset = 0
    with torch.no_grad():
        while history.shape[1] < args.chunks:
            context = history[:, -config['context_chunks']:]
            cond = context.mean(1, keepdim=True)
            shape = (1, config['future_chunks'], *history.shape[2:])
            z = torch.randn(shape, device=device)
            dt = 1 / args.sample_steps
            for index in range(args.sample_steps):
                t = torch.full(
                    (1,), index / args.sample_steps, device=device)
                z = z + dt * model(
                    z, t, cond=cond, temporal_offset=offset)
            history = torch.cat([history, z], dim=1)
            offset += config['future_chunks']
    torch.save({
        'latent': history[0, :args.chunks].cpu(),
        'source_checkpoint': args.checkpoint,
    }, args.output)


if __name__ == '__main__':
    main()
