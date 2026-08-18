#!/usr/bin/env python3
"""Evaluate saved R7 sample packs against AE targets and raw SpatialVID RGB.

The training cache stores only the anchor RGB. This offline suite reopens the
source video using ``video_id``/``window_index`` and the representation sampling
contract, so generated-vs-raw metrics are not confused with generated-vs-AE.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import torch
import torch.nn.functional as F

from utils.r7_representation import load_checkpoint, load_r7_modules
from utils.video_io import _compute_frame_indices, read_video_frames
from utils.video_preview import save_video_preview
from utils.moxing_io import is_remote_path, join_remote, stage_remote_file


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_pack", required=True)
    parser.add_argument("--r7_ckpt", required=True)
    parser.add_argument("--decoder_ckpt", default="")
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    return parser.parse_args()


def metadata_index(path, video_root):
    result = {}
    with open(path, encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            relative = row["video path"].replace("videos/", "")
            result[row["id"]] = {
                "path": (join_remote(video_root, relative)
                         if is_remote_path(video_root)
                         else os.path.join(video_root, relative)),
                "fps": float(row.get("fps", 0) or 0),
                "num_frames": int(row["num frames"]),
            }
    return result


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a.float(), b.float()).clamp_min(1e-12)))


def main():
    args = parse_args()
    pack = torch.load(args.sample_pack, map_location="cpu", weights_only=False)
    artifact = load_checkpoint(args.r7_ckpt)
    config, _, _, tokenizer, decoder, _ = load_r7_modules(artifact)
    if args.decoder_ckpt:
        from train_causal_wan_video_diffusion import load_robust_decoder
        representation = pack.get("representation") or artifact["representation_contract"]
        load_robust_decoder(args.decoder_ckpt, representation, decoder)
    tokenizer.eval(); decoder.eval()
    index = metadata_index(args.metadata_csv, args.video_root)
    rows = []
    os.makedirs(args.output_dir, exist_ok=True)
    with torch.inference_mode():
        for sample_index, sample in enumerate(pack.get("samples", [])):
            video_id = sample["video_id"]
            if video_id not in index:
                raise KeyError(f"video id missing from metadata: {video_id}")
            latent = torch.cat((sample["cond_anchor"], sample["sampled_future"]), 1)
            latent = latent.reshape(1, config.latent_seq_len, config.latent_grid,
                                    config.latent_grid, config.latent_dim)
            geo, tex = tokenizer.decode(latent.float())
            generated = decoder(geo, tex)[..., :3].clamp(0, 1)[0]

            target_latent = torch.cat((sample["cond_anchor"], sample["target_future"]), 1)
            target_latent = target_latent.reshape_as(latent)
            target_geo, target_tex = tokenizer.decode(target_latent.float())
            ae_target = decoder(target_geo, target_tex)[..., :3].clamp(0, 1)[0]

            source = index[video_id]
            path = stage_remote_file(
                source["path"], os.environ.get("MOX_VIDEO_CACHE_DIR", "./video_cache"),
                retries=int(os.environ.get("MOX_DOWNLOAD_RETRIES", "4")))
            window_index = int(sample.get("window_index", 0))
            num_windows = int(sample.get("clips_per_video", 1))
            frame_indices = _compute_frame_indices(
                source["num_frames"], config.seq_len, temporal_jitter=False,
                fps=source["fps"],
                clip_duration_seconds=config.clip_duration_seconds,
                window_index=window_index, num_windows=num_windows)
            raw = read_video_frames(
                path, config.seq_len, config.target_size,
                temporal_jitter=False, frame_indices=frame_indices)
            raw = raw.permute(0, 2, 3, 1)
            row = {
                "sample_index": sample_index, "video_id": video_id,
                "window_index": int(sample.get("window_index", 0)),
                "codec_psnr_vs_raw": psnr(ae_target, raw),
                "generated_psnr_vs_ae": psnr(generated, ae_target),
                "generated_psnr_vs_raw": psnr(generated, raw),
            }
            rows.append(row)
            save_video_preview(
                args.output_dir, f"sample{sample_index:03d}",
                {"RAW": raw, "AE_TARGET": ae_target, "GENERATED": generated},
                fps=args.fps, metadata=row, save_frames=sample_index == 0)
    summary = {
        "schema": "r7-raw-rgb-eval-v1", "samples": rows,
        "mean": {key: sum(row[key] for row in rows) / max(len(rows), 1)
                 for key in ("codec_psnr_vs_raw", "generated_psnr_vs_ae",
                             "generated_psnr_vs_raw")},
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(json.dumps(summary["mean"], sort_keys=True))


if __name__ == "__main__":
    main()
