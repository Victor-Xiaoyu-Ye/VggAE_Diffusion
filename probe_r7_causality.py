#!/usr/bin/env python3
"""Assert synthetic and optional checkpoint-level R7 causality invariants."""
from __future__ import annotations

import argparse

import torch

from models.causal_dual_tokenizer import CausalDualTokenizerCore
from models.causal_temporal_codec import CausalSpatiotemporalCodec
from utils.r7_representation import (build_contract, load_checkpoint,
                                     load_r7_modules, validate_contract)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--checkpoint", default="",
                        help="strictly validate an R7 checkpoint without data")
    parser.add_argument("--allow_legacy_checkpoint", action="store_true")
    return parser.parse_args()


def check_codec(factor, atol):
    torch.manual_seed(0)
    codec = CausalSpatiotemporalCodec(channels=16, factor=factor, depth=2).eval()
    for chunks in (1, 2, 4):
        frames = 1 + factor * chunks
        x = torch.randn(2, 16, frames, 5, 5)
        z = codec.encode(x); y = codec.decode(z)
        assert z.shape == (2, 16, 1 + chunks, 5, 5)
        assert y.shape == x.shape
        # Encoded prefix cannot depend on a later raw-frame suffix.
        for boundary in range(1, z.shape[2]):
            future_start = 1 + boundary * factor
            changed = x.clone()
            changed[:, :, future_start:] += torch.randn_like(changed[:, :, future_start:])
            changed_z = codec.encode(changed)
            torch.testing.assert_close(z[:, :, :boundary + 1],
                                       changed_z[:, :, :boundary + 1],
                                       atol=atol, rtol=0)
        # Decoded frame prefix cannot depend on a later latent suffix.
        for boundary in range(1, z.shape[2]):
            changed = z.clone()
            changed[:, :, boundary + 1:] += torch.randn_like(changed[:, :, boundary + 1:])
            changed_y = codec.decode(changed)
            frame_end = 1 + boundary * factor
            torch.testing.assert_close(y[:, :, :frame_end],
                                       changed_y[:, :, :frame_end],
                                       atol=atol, rtol=0)

    core = CausalDualTokenizerCore(
        geo_dim=8, tex_dim=8, geo_latent_dim=4, tex_latent_dim=4,
        temporal_factor=factor, temporal_depth=1)
    frames = 1 + 2 * factor
    geo = torch.randn(2, frames, 5, 5, 8); tex = torch.randn_like(geo)
    geo_rec, tex_rec, latent = core(geo, tex)
    assert geo_rec.shape == geo.shape and tex_rec.shape == tex.shape
    assert latent.shape == (2, 3, 5, 5, 8)
    (geo_rec.square().mean() + tex_rec.square().mean()
     + latent.square().mean()).backward()
    unused = [name for name, parameter in core.named_parameters()
              if parameter.requires_grad and parameter.grad is None]
    assert not unused, f"unused parameters: {unused}"
    print(f"factor={factor}: causality, shapes, gradients PASS")


def check_checkpoint(path, atol, allow_legacy):
    checkpoint = load_checkpoint(path)
    config, _, _, tokenizer, _, matched = load_r7_modules(checkpoint)
    saved_contract = checkpoint.get("representation_contract")
    if saved_contract is None:
        if not allow_legacy:
            raise RuntimeError("checkpoint lacks representation_contract; pass "
                               "--allow_legacy_checkpoint after reviewing migration")
    else:
        expected = build_contract(config, include_signatures=False)
        # A training contract may include source signatures; structural fields
        # must remain exact while signatures are not needed by this offline probe.
        expected["signatures"] = {}
        validate_contract(saved_contract, expected)

    tokenizer.eval()
    torch.manual_seed(7)
    geo = torch.randn(1, config.seq_len, config.latent_grid,
                      config.latent_grid, config.geo_dim)
    tex = torch.randn(1, config.seq_len, config.latent_grid,
                      config.latent_grid, config.tex_dim)
    latent = tokenizer.encode(geo, tex)
    expected_shape = (1, config.latent_seq_len, config.latent_grid,
                      config.latent_grid, config.latent_dim)
    assert latent.shape == expected_shape, (latent.shape, expected_shape)
    decoded_geo, decoded_tex = tokenizer.decode(latent)
    assert decoded_geo.shape == geo.shape and decoded_tex.shape == tex.shape

    # Anchor is a strict prefix invariant for both encode and decode.
    changed_geo, changed_tex = geo.clone(), tex.clone()
    changed_geo[:, 1:] += torch.randn_like(changed_geo[:, 1:])
    changed_tex[:, 1:] += torch.randn_like(changed_tex[:, 1:])
    changed_latent = tokenizer.encode(changed_geo, changed_tex)
    torch.testing.assert_close(latent[:, :1], changed_latent[:, :1],
                               atol=atol, rtol=0)
    changed = latent.clone()
    changed[:, 1:] += torch.randn_like(changed[:, 1:])
    changed_geo_dec, changed_tex_dec = tokenizer.decode(changed)
    torch.testing.assert_close(decoded_geo[:, :1], changed_geo_dec[:, :1],
                               atol=atol, rtol=0)
    torch.testing.assert_close(decoded_tex[:, :1], changed_tex_dec[:, :1],
                               atol=atol, rtol=0)

    # Every compressed/decoded prefix obeys the checkpoint factor.
    for boundary in range(1, config.latent_seq_len):
        future_start = 1 + boundary * config.temporal_factor
        changed_geo, changed_tex = geo.clone(), tex.clone()
        changed_geo[:, future_start:] += torch.randn_like(changed_geo[:, future_start:])
        changed_tex[:, future_start:] += torch.randn_like(changed_tex[:, future_start:])
        changed_latent = tokenizer.encode(changed_geo, changed_tex)
        torch.testing.assert_close(latent[:, :boundary + 1],
                                   changed_latent[:, :boundary + 1],
                                   atol=atol, rtol=0)
        changed = latent.clone()
        changed[:, boundary + 1:] += torch.randn_like(changed[:, boundary + 1:])
        changed_geo_dec, changed_tex_dec = tokenizer.decode(changed)
        frame_end = 1 + boundary * config.temporal_factor
        torch.testing.assert_close(decoded_geo[:, :frame_end],
                                   changed_geo_dec[:, :frame_end], atol=atol, rtol=0)
        torch.testing.assert_close(decoded_tex[:, :frame_end],
                                   changed_tex_dec[:, :frame_end], atol=atol, rtol=0)
    print(f"checkpoint={path}: strict keys={matched}, shape={expected_shape}, "
          "anchor/prefix causality PASS")


def main():
    args = parse_args()
    for factor in args.factors:
        check_codec(factor, args.atol)
    if args.checkpoint:
        check_checkpoint(args.checkpoint, args.atol,
                         args.allow_legacy_checkpoint)


if __name__ == "__main__":
    main()
