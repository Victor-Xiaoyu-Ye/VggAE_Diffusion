#!/usr/bin/env python3
"""Diagnose whether AE and I0 decoder use temporal geometry latents.

For each clip, this script compares:
  - target RGB frames
  - AE reconstruction from full per-frame z
  - AE reconstruction from z_0 repeated across all frames
  - I0 reconstruction from full per-frame z, conditioned on frame 0
  - I0 reconstruction from z_0 repeated, conditioned on frame 0

The key signal is temporal_delta_ratio = recon_delta / target_delta.
If full_z and repeat_z0 have similar deltas, the decoder is likely ignoring
time-varying geometry latent and copying appearance/low-frequency content.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.loader_utils import multiprocessing_loader_kwargs
from data.token_utils import strip_special_tokens
from data.video_dataset import SpatialVidDataset, collate_fn
from models.appearance_cnn import AppearanceCNN
from models.compact_decoder import CompactDecoder
from models.generative_tokenizer import GenerativeTokenizer
from models.i0_decoder import I0ConditionalDecoder, load_i0_decoder_state_dict
from streamvggt.models.streamvggt import StreamVGGT
from utils.device import configure_backend_compatibility, get_device, get_device_name, resolve_dtype


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--autoencoder_ckpt", required=True)
    parser.add_argument("--i0_decoder_ckpt", default="")
    parser.add_argument("--output_dir", default="outputs/temporal_diagnostics")
    parser.add_argument("--num_videos", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--clip_duration_seconds", type=float, default=1.0)
    parser.add_argument("--max_frame_span", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--latent_grid", type=int, default=18)
    parser.add_argument("--token_dim", type=int, default=2048)
    parser.add_argument("--levels", type=int, nargs="+", default=[4, 11, 17, 23])
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--use_ema", action="store_true")
    return parser.parse_args()


def split_ema_state(ema_state):
    tokenizer_state = {}
    decoder_state = {}
    for key, value in ema_state.items():
        if key.startswith("0."):
            tokenizer_state[key[2:]] = value
        elif key.startswith("1."):
            decoder_state[key[2:]] = value
    return tokenizer_state, decoder_state


def build_compact_decoder(decoder_state, args):
    output_depth = any(key.startswith("depth_head.") for key in decoder_state)
    base_dim = decoder_state["stem.0.conv.weight"].shape[0] // 2
    version = "v2" if any("upsample" in key for key in decoder_state if key.startswith("up0.")) else "v1"
    if version == "v2":
        return CompactDecoder(
            latent_dim=args.latent_dim, base_dim=base_dim, output_dim=3,
            output_depth=output_depth, img_size=args.target_size,
            latent_grid=args.latent_grid, num_resblocks=2,
            use_pixel_shuffle=True, num_temporal_blocks=2,
            version="v2", use_checkpoint=False)
    return CompactDecoder(
        latent_dim=args.latent_dim, base_dim=base_dim, output_dim=3,
        output_depth=output_depth, img_size=args.target_size,
        latent_grid=args.latent_grid, num_resblocks=1,
        use_pixel_shuffle=False, num_temporal_blocks=1,
        version="v1", use_checkpoint=False)


def load_models(args, device, compute_dtype):
    ae_ckpt = torch.load(args.autoencoder_ckpt, map_location="cpu", weights_only=False)
    ae_args = ae_ckpt.get("args", {})
    for key in ("latent_dim", "latent_grid", "target_size", "seq_len", "token_dim"):
        if key in ae_args:
            setattr(args, key, int(ae_args[key]))
    if "levels" in ae_args:
        args.levels = [int(level) for level in ae_args["levels"]]
    disable_temporal_mixer = bool(ae_args.get("disable_temporal_mixer", False))

    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location="cpu"), strict=False)
    encoder = encoder.to(device=device, dtype=compute_dtype).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    tokenizer = GenerativeTokenizer(
        token_dim=args.token_dim, latent_dim=args.latent_dim,
        latent_grid=args.latent_grid, levels=args.levels,
        seq_len=args.seq_len, input_grid=args.target_size // 14,
        disable_temporal_mixer=disable_temporal_mixer,
    ).to(device=device).eval()
    tokenizer.load_state_dict(ae_ckpt["tokenizer"])
    tokenizer.set_temporal_mixer_enabled(not disable_temporal_mixer)

    compact_decoder = build_compact_decoder(ae_ckpt["decoder"], args).to(device=device).eval()
    compact_decoder.load_state_dict(ae_ckpt["decoder"], strict=False)

    if args.use_ema and "ema" in ae_ckpt:
        tokenizer_ema, decoder_ema = split_ema_state(ae_ckpt["ema"])
        if tokenizer_ema:
            tokenizer.load_state_dict(tokenizer_ema, strict=False)
        if decoder_ema:
            compact_decoder.load_state_dict(decoder_ema, strict=False)

    for module in (tokenizer, compact_decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    i0_models = None
    if args.i0_decoder_ckpt:
        i0_ckpt = torch.load(args.i0_decoder_ckpt, map_location="cpu", weights_only=False)
        i0_app = AppearanceCNN().to(device=device).eval()
        i0_dec = I0ConditionalDecoder(
            latent_dim=args.latent_dim,
            base_dim=int(i0_ckpt.get("args", {}).get("decoder_base_dim", 384)),
            img_size=args.target_size,
            latent_grid=args.latent_grid,
            num_resblocks=int(i0_ckpt.get("args", {}).get("decoder_num_resblocks", 2)),
            use_checkpoint=False,
        ).to(device=device).eval()
        i0_app.load_state_dict(i0_ckpt["app_cnn"])
        load_i0_decoder_state_dict(i0_dec, i0_ckpt["decoder"])
        for module in (i0_app, i0_dec):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        i0_models = (i0_app, i0_dec)

    return encoder, tokenizer, compact_decoder, i0_models


def psnr(pred, target):
    mse = F.mse_loss(pred.float(), target.float()).item()
    return float(-10.0 * np.log10(mse)) if mse > 0 else 100.0


def temporal_delta(x):
    if x.shape[0] < 2:
        return 0.0
    return float((x[1:] - x[:-1]).abs().mean().item())


def temporal_delta_error(pred, target):
    if pred.shape[0] < 2:
        return 0.0
    return float(((pred[1:] - pred[:-1]) - (target[1:] - target[:-1])).abs().mean().item())


def metrics_for(pred, target):
    target_delta = temporal_delta(target)
    pred_delta = temporal_delta(pred)
    return {
        "psnr": psnr(pred, target),
        "l1": float((pred.float() - target.float()).abs().mean().item()),
        "target_temporal_delta": target_delta,
        "pred_temporal_delta": pred_delta,
        "temporal_delta_ratio": pred_delta / max(target_delta, 1e-8),
        "temporal_delta_error": temporal_delta_error(pred, target),
    }


def save_grid(frames, outputs, out_path):
    # rows are frames, columns are target / AE full / AE repeat / I0 full / I0 repeat.
    rows = []
    names = ["target", *outputs.keys()]
    for frame_index in range(frames.shape[0]):
        cols = [frames[frame_index]]
        for value in outputs.values():
            cols.append(value[frame_index])
        rows.append(torch.cat(cols, dim=1))
    grid = torch.cat(rows, dim=0).clamp(0, 1)
    image = (grid.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(image).save(out_path)
    with open(out_path + ".columns.txt", "w") as handle:
        handle.write("\n".join(names) + "\n")


def save_frame_delta_grid(frames, outputs, out_path):
    strips = []
    for source in [frames, *outputs.values()]:
        delta = (source[1:] - source[:-1]).abs()
        # normalize each method consistently enough for visual comparison.
        delta = (delta / delta.max().clamp_min(1e-6)).clamp(0, 1)
        delta = torch.cat([torch.zeros_like(delta[:1]), delta], dim=0)
        strips.append(torch.cat([delta[index] for index in range(delta.shape[0])], dim=0))
    grid = torch.cat(strips, dim=1)
    image = (grid.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(image).save(out_path)


@torch.no_grad()
def main():
    args = parse_args()
    device_type = get_device_name()
    configure_backend_compatibility(device_type)
    if device_type == "cpu":
        raise RuntimeError("Temporal diagnosis requires an accelerator")
    device = get_device(0)
    compute_dtype = resolve_dtype(args.dtype)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    encoder, tokenizer, compact_decoder, i0_models = load_models(args, device, compute_dtype)

    dataset = SpatialVidDataset(
        csv_path=args.csv, video_root=args.video_root,
        seq_len=args.seq_len, target_size=args.target_size,
        max_videos=args.num_videos, num_frames_per_video=args.seq_len,
        temporal_jitter=False, max_frame_span=args.max_frame_span,
        clip_duration_seconds=args.clip_duration_seconds,
        decode_retries=0,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        **multiprocessing_loader_kwargs(args.num_workers),
    )

    records = []
    for index, batch in enumerate(tqdm(loader, desc="Diagnosing")):
        if index >= args.num_videos:
            break
        frames = batch["frames"].to(device=device, dtype=compute_dtype)
        target = frames[0].permute(0, 2, 3, 1).float().cpu()
        with torch.amp.autocast(device_type=device_type, dtype=compute_dtype):
            tokens, psi = encoder(frames)
            tokens = strip_special_tokens(tokens, psi)
            z_full, _ = tokenizer(tokens)
            z_repeat = z_full[:, :1].expand_as(z_full).contiguous()
            ae_full = compact_decoder(z_full)
            ae_repeat = compact_decoder(z_repeat)
        ae_full_rgb = (ae_full[0] if not compact_decoder.output_depth else ae_full[0])[0, ..., :3].clamp(0, 1).float().cpu()
        ae_repeat_rgb = (ae_repeat[0] if not compact_decoder.output_depth else ae_repeat[0])[0, ..., :3].clamp(0, 1).float().cpu()

        outputs = {
            "ae_full_z": ae_full_rgb,
            "ae_repeat_z0": ae_repeat_rgb,
        }

        if i0_models is not None:
            i0_app, i0_dec = i0_models
            with torch.amp.autocast(device_type=device_type, dtype=compute_dtype):
                i0_feats = i0_app(frames[:, 0])
                i0_full, _ = i0_dec(z_full, i0_feats)
                i0_repeat, _ = i0_dec(z_repeat, i0_feats)
            outputs["i0_full_z"] = i0_full[0, ..., :3].clamp(0, 1).float().cpu()
            outputs["i0_repeat_z0"] = i0_repeat[0, ..., :3].clamp(0, 1).float().cpu()

        base = f"clip_{index:04d}_{batch['video_id'][0]}"
        grid_path = os.path.join(args.output_dir, base + "_grid.png")
        delta_path = os.path.join(args.output_dir, base + "_frame_delta.png")
        save_grid(target, outputs, grid_path)
        save_frame_delta_grid(target, outputs, delta_path)

        record = {
            "index": index,
            "video_id": batch["video_id"][0],
            "requested_video_id": batch["requested_video_id"][0],
            "grid": os.path.basename(grid_path),
            "frame_delta_grid": os.path.basename(delta_path),
        }
        for name, value in outputs.items():
            record[name] = metrics_for(value, target)
        records.append(record)

    summary = {"num_videos": len(records), "videos": records}
    for name in records[0].keys() if records else []:
        if name in ("ae_full_z", "ae_repeat_z0", "i0_full_z", "i0_repeat_z0"):
            summary[name] = {}
            for metric in records[0][name]:
                values = [record[name][metric] for record in records]
                summary[name][metric + "_mean"] = float(np.mean(values))
    metrics_path = os.path.join(args.output_dir, "temporal_metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "videos"}, indent=2))
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
