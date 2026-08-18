#!/usr/bin/env python3
"""CPU contract checks for R7 factor-1, rollout, and preview utilities."""
from __future__ import annotations

import argparse
import os
import tempfile

import torch

from models.causal_temporal_codec import CausalSpatiotemporalCodec
from models.r7_overlap_rollout import (generated_context_mix,
                                       overlapping_windows,
                                       scheduled_context_probability)
from utils.video_preview import save_video_preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require_mp4", action="store_true")
    args = parser.parse_args()

    assert overlapping_windows(4, window=2, overlap=1) == [(0, 2), (1, 3), (2, 4)]
    assert scheduled_context_probability(0, 10, 10, .25) == 0
    assert scheduled_context_probability(20, 10, 10, .25) == .25
    clean = torch.zeros(4, 2, 3); generated = torch.ones_like(clean)
    torch.manual_seed(1)
    mixed = generated_context_mix(clean, generated, 1.0)
    assert torch.equal(mixed, generated)

    codec = CausalSpatiotemporalCodec(8, factor=1, depth=1).eval()
    source = torch.randn(1, 8, 3, 4, 4)
    encoded = codec.encode(source); decoded = codec.decode(encoded)
    assert encoded.shape == source.shape and decoded.shape == source.shape
    changed = source.clone(); changed[:, :, 2:] += 10
    changed_encoded = codec.encode(changed)
    torch.testing.assert_close(encoded[:, :, :2], changed_encoded[:, :, :2])

    frames = torch.rand(5, 8, 8, 3)
    with tempfile.TemporaryDirectory() as directory:
        result = save_video_preview(directory, "smoke", {"VIDEO": frames}, fps=8)
        assert os.path.isfile(result["grid"])
        assert len(os.listdir(result["videos"]["VIDEO"]["png_dir"])) == 5
        if args.require_mp4:
            assert os.path.isfile(result["videos"]["VIDEO"]["mp4"])
    print("R7 factor-1/rollout/preview contracts passed")


if __name__ == "__main__":
    main()
