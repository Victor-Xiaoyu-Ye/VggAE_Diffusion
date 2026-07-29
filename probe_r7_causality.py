#!/usr/bin/env python3
"""Assert R7 temporal shape and causality invariants."""

from __future__ import annotations

import argparse

import torch

from models.causal_temporal_codec import CausalSpatiotemporalCodec
from models.causal_dual_tokenizer import CausalDualTokenizerCore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factors', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--atol', type=float, default=1e-5)
    return parser.parse_args()


def check_codec(factor, atol):
    torch.manual_seed(0)
    codec = CausalSpatiotemporalCodec(channels=16, factor=factor, depth=2).eval()
    for chunks in (1, 2, 4):
        frames = 1 + factor * chunks
        x = torch.randn(2, 16, frames, 5, 5)
        z = codec.encode(x)
        y = codec.decode(z)
        assert z.shape == (2, 16, 1 + chunks, 5, 5)
        assert y.shape == x.shape

        for boundary in range(1, z.shape[2]):
            future_start = 1 + boundary * factor
            changed = x.clone()
            changed[:, :, future_start:] += torch.randn_like(
                changed[:, :, future_start:])
            z_changed = codec.encode(changed)
            torch.testing.assert_close(
                z[:, :, :boundary + 1], z_changed[:, :, :boundary + 1],
                atol=atol, rtol=0)

        for latent_boundary in range(1, z.shape[2]):
            changed = z.clone()
            changed[:, :, latent_boundary + 1:] += torch.randn_like(
                changed[:, :, latent_boundary + 1:])
            y_changed = codec.decode(changed)
            frame_end = 1 + latent_boundary * factor
            torch.testing.assert_close(
                y[:, :, :frame_end], y_changed[:, :, :frame_end],
                atol=atol, rtol=0)

    core = CausalDualTokenizerCore(
        geo_dim=8, tex_dim=8, geo_latent_dim=4, tex_latent_dim=4,
        temporal_factor=factor, temporal_depth=1)
    frames = 1 + 2 * factor
    geo = torch.randn(2, frames, 5, 5, 8)
    tex = torch.randn_like(geo)
    geo_rec, tex_rec, z = core(geo, tex)
    assert geo_rec.shape == geo.shape
    assert tex_rec.shape == tex.shape
    assert z.shape == (2, 3, 5, 5, 8)
    (geo_rec.square().mean() + tex_rec.square().mean() + z.square().mean()).backward()
    unused = [name for name, parameter in core.named_parameters()
              if parameter.requires_grad and parameter.grad is None]
    assert not unused, f'unused parameters: {unused}'
    print(f'factor={factor}: causality, shapes, gradients PASS')


def main():
    args = parse_args()
    for factor in args.factors:
        check_codec(factor, args.atol)


if __name__ == '__main__':
    main()
