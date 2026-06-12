#!/usr/bin/env python3
"""Sample an I0-conditioned compact latent flow model."""

import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from data.token_utils import strip_special_tokens
from data.video_dataset import SpatialVidDataset
from models.appearance_cnn import AppearanceCNN
from models.compact_dit import CompactLatentDiT
from models.generative_tokenizer import GenerativeTokenizer
from models.i0_decoder import I0ConditionalDecoder, load_i0_decoder_state_dict
from streamvggt.models.streamvggt import StreamVGGT
from utils.device import (
    get_device,
    get_device_name,
    manual_seed_all,
    resolve_dtype,
)
from utils.file_signature import validate_file_signature
from utils.latent_stats import (
    denormalize_latent,
    normalize_latent,
    validate_latent_stats,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--i0_path", default="", help="Optional reference image or video")
    parser.add_argument(
        "--csv", default="", help="Use the first dataset clip when i0_path is empty")
    parser.add_argument("--video_root", default="")
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--autoencoder_ckpt", required=True)
    parser.add_argument("--i0_decoder_ckpt", required=True)
    parser.add_argument("--diffusion_ckpt", required=True)
    parser.add_argument("--out_dir", default="outputs/compact_i0")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--solver", choices=["euler", "midpoint"], default="midpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--no_anchor_first_latent", action="store_true")
    return parser.parse_args()


def load_reference(path, size):
    if path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        reader = imageio.get_reader(path)
        frame = reader.get_data(0)
        reader.close()
        image = Image.fromarray(frame).convert("RGB")
    else:
        image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)


def load_dataset_reference(csv_path, video_root, size):
    if not csv_path or not video_root:
        raise ValueError(
            "Set --i0_path or provide both --csv and --video_root")
    dataset = SpatialVidDataset(
        csv_path=csv_path,
        video_root=video_root,
        seq_len=1,
        target_size=size,
        max_videos=1,
        num_frames_per_video=1,
        temporal_jitter=False,
        check_files=False,
    )
    if not dataset:
        raise RuntimeError(f"No sample found in {csv_path}")
    return dataset[0]["frames"][:1].unsqueeze(0)


def save_outputs(rgb, out_dir, fps):
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    uint8 = (rgb.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    uint8 = uint8.transpose(0, 2, 3, 1)
    for index, frame in enumerate(uint8):
        Image.fromarray(frame).save(os.path.join(frames_dir, f"frame_{index:03d}.png"))
    imageio.mimsave(
        os.path.join(out_dir, "generated.mp4"),
        list(uint8),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
    )


def validate_checkpoint_args(saved_args, expected, checkpoint_name):
    for key, value in expected.items():
        if key not in saved_args:
            continue
        saved = saved_args[key]
        if isinstance(saved, (list, tuple)):
            matches = list(saved) == list(value)
        else:
            matches = saved == value
        if not matches:
            raise ValueError(
                f"{checkpoint_name} mismatch for {key}: "
                f"checkpoint={saved}, expected={value}")


def validate_representation_files(normalization, encoder_path, autoencoder_path):
    representation = normalization.get("representation", {})
    for key, path in (
            ("encoder", encoder_path), ("autoencoder", autoencoder_path)):
        signature = representation.get(key)
        if signature is None:
            continue
        validate_file_signature(path, signature, f"{key} checkpoint")


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    manual_seed_all(args.seed)
    device_type = get_device_name()
    device = get_device()
    dtype = resolve_dtype(args.dtype)

    diffusion_ckpt = torch.load(
        args.diffusion_ckpt, map_location="cpu", weights_only=False)
    diffusion_args = diffusion_ckpt.get("args", {})
    normalization = diffusion_ckpt.get("normalization")
    cached_training = normalization is not None
    future_only_checkpoint = (
        cached_training or "generated_seq_len" in diffusion_args)
    if cached_training and normalization.get("representation"):
        validate_representation_files(
            normalization, args.encoder_ckpt, args.autoencoder_ckpt)
    if diffusion_args.get("text_cond", False):
        raise ValueError("This sampler currently supports the non-text I0 checkpoint")
    if not cached_training and not diffusion_args.get("i0_condition", False):
        raise ValueError("The diffusion checkpoint was not trained with --i0_condition")
    if not cached_training and "latent_scale" not in diffusion_ckpt:
        raise ValueError("The diffusion checkpoint does not contain latent_scale")

    clip_seq_len = int(diffusion_args.get("seq_len", 8))
    seq_len = int(diffusion_args.get("generated_seq_len", clip_seq_len))
    target_size = int(diffusion_args.get("target_size", 518))
    latent_dim = int(diffusion_args.get("latent_dim", 512))
    latent_grid = int(diffusion_args.get("latent_grid", 18))
    levels = diffusion_args.get("levels", [4, 11, 17, 23])
    latent_scale = float(diffusion_ckpt.get("latent_scale", 1.0))
    if normalization is not None:
        validate_latent_stats(
            normalization["target"], seq_len, latent_dim, name="target")
        validate_latent_stats(
            normalization["cond"], 1, latent_dim, name="condition")

    encoder = StreamVGGT(img_size=target_size, patch_size=14, embed_dim=1024)
    load_info = encoder.load_state_dict(
        torch.load(args.encoder_ckpt, map_location="cpu"), strict=False)
    if load_info.missing_keys or load_info.unexpected_keys:
        print(
            f"[WARN] Encoder checkpoint mismatch: missing={len(load_info.missing_keys)}, "
            f"unexpected={len(load_info.unexpected_keys)}")
    encoder = encoder.to(device=device, dtype=dtype).eval()

    tokenizer = GenerativeTokenizer(
        token_dim=int(diffusion_args.get("token_dim", 2048)),
        latent_dim=latent_dim,
        latent_grid=latent_grid,
        levels=levels,
        seq_len=(
            int(normalization.get("representation", {}).get(
                "seq_len", clip_seq_len))
            if cached_training else clip_seq_len),
        input_grid=target_size // 14,
    ).to(device).eval()
    autoencoder_ckpt = torch.load(
        args.autoencoder_ckpt, map_location="cpu", weights_only=False)
    validate_checkpoint_args(
        autoencoder_ckpt.get("args", {}),
        {
            "latent_dim": latent_dim,
            "latent_grid": latent_grid,
            "levels": levels,
            "target_size": target_size,
        },
        "Autoencoder checkpoint",
    )
    tokenizer.load_state_dict(autoencoder_ckpt["tokenizer"])
    tokenizer.disable_temporal_mixer = bool(
        diffusion_args.get(
            "disable_temporal_mixer",
            autoencoder_ckpt.get("args", {}).get(
                "disable_temporal_mixer", False),
        )
    )

    i0_ckpt = torch.load(
        args.i0_decoder_ckpt, map_location="cpu", weights_only=False)
    i0_args = i0_ckpt.get("args", {})
    validate_checkpoint_args(
        i0_args,
        {
            "latent_dim": latent_dim,
            "latent_grid": latent_grid,
            "target_size": target_size,
        },
        "I0 decoder checkpoint",
    )
    app_cnn = AppearanceCNN().to(device=device, dtype=dtype).eval()
    decoder = I0ConditionalDecoder(
        latent_dim=latent_dim,
        base_dim=int(i0_args.get(
            "decoder_base_dim", diffusion_args.get("decoder_base_dim", 384))),
        img_size=target_size,
        latent_grid=latent_grid,
        num_resblocks=int(i0_args.get(
            "decoder_num_resblocks", diffusion_args.get("decoder_num_resblocks", 2))),
        use_checkpoint=False,
    ).to(device).eval()
    app_cnn.load_state_dict(i0_ckpt["app_cnn"])
    load_i0_decoder_state_dict(decoder, i0_ckpt["decoder"])

    model = CompactLatentDiT(
        latent_dim=latent_dim,
        num_tokens=latent_grid ** 2,
        model_dim=int(diffusion_args.get("model_dim", 768)),
        spatial_depth=int(diffusion_args.get("spatial_depth", 8)),
        temporal_depth=int(diffusion_args.get("temporal_depth", 4)),
        num_heads=int(diffusion_args.get("num_heads", 12)),
        seq_len=seq_len,
        text_cond=False,
        i0_condition=True,
        time_scale=float(diffusion_args.get("time_scale", 1.0)),
    ).to(device=device, dtype=dtype).eval()
    model.i0_residual = bool(diffusion_args.get("i0_residual", False))
    model_state = diffusion_ckpt.get("ema", diffusion_ckpt["model"])
    model.load_state_dict(model_state)

    reference = (
        load_reference(args.i0_path, target_size)
        if args.i0_path else
        load_dataset_reference(args.csv, args.video_root, target_size)
    ).to(device=device, dtype=dtype)
    tokens, psi = encoder(reference)
    tokens = strip_special_tokens(tokens, psi)
    i0_grid, i0_flat = tokenizer(tokens)
    if cached_training:
        cond = normalize_latent(
            i0_flat, normalization["cond"]).to(dtype=dtype)
    else:
        cond = (i0_flat * latent_scale).to(dtype=dtype)

    shape = (1, seq_len, latent_grid ** 2, latent_dim)
    z = torch.randn(shape, device=device, dtype=dtype)
    dt = 1.0 / args.num_steps
    for index in range(args.num_steps):
        t = torch.full((1,), index / args.num_steps, device=device, dtype=dtype)
        velocity = model(z, t, cond=cond)
        if args.solver == "midpoint":
            z_mid = z + 0.5 * dt * velocity
            t_mid = torch.full(
                (1,), (index + 0.5) / args.num_steps, device=device, dtype=dtype)
            velocity = model(z_mid, t_mid, cond=cond)
        z = z + dt * velocity

    if cached_training:
        residual = denormalize_latent(z, normalization["target"])
        future = residual + i0_flat.float().expand(
            -1, seq_len, -1, -1)
        future_grid = future.reshape(
            1, seq_len, latent_grid, latent_grid, latent_dim)
        z_grid = torch.cat([i0_grid[:, :1].float(), future_grid], dim=1)
    else:
        if model.i0_residual:
            if future_only_checkpoint:
                future = (
                    z / latent_scale
                    + i0_flat.expand(-1, seq_len, -1, -1))
                future_grid = future.reshape(
                    1, seq_len, latent_grid, latent_grid, latent_dim).float()
                z_grid = torch.cat(
                    [i0_grid[:, :1].float(), future_grid], dim=1)
            else:
                z_grid = (
                    z / latent_scale
                    + i0_flat.expand(-1, seq_len, -1, -1)
                ).reshape(
                    1, seq_len, latent_grid, latent_grid, latent_dim).float()
                if not args.no_anchor_first_latent:
                    z_grid[:, 0] = i0_grid[:, 0].float()
        else:
            z_grid = (z / latent_scale).reshape(
                1, seq_len, latent_grid, latent_grid, latent_dim).float()
            if not args.no_anchor_first_latent:
                z_grid[:, 0] = i0_grid[:, 0].float()

    appearance = app_cnn(reference[:, 0])
    with torch.autocast(
            device_type=device_type, dtype=dtype,
            enabled=dtype != torch.float32):
        predictions, _ = decoder(z_grid, appearance)
    rgb = predictions[0].permute(0, 3, 1, 2).contiguous()
    save_outputs(rgb, args.out_dir, args.fps)
    print(f"Saved {rgb.shape[0]} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
