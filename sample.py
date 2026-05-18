#!/usr/bin/env python3
"""Sampling script for VggAE-Diffusion.

Generates tokens via flow ODE (Euler or midpoint) and decodes to RGB video
using an optional DPTHead decoder.

Supports:
  1. Text-conditional generation with CLIP + classifier-free guidance
  2. Unconditional generation (text_emb=None)
  3. RGB decoding via DPTHead (optional)
  4. Save as PNG frames and/or mp4 video

Pipeline per sample:
  1. Encode text prompt -> text_emb (if text_cond)
  2. z ~ N(0, I), shape [B, T, N, D=token_dim]
     Recommended GLD-style path: one boundary level, so T=seq_len.
  3. Euler ODE: for num_steps, v = model(z, t, text_emb), z += v * dt
  4. If CFG: run twice (conditional + unconditional), interpolate
  5. Build 24-entry aggregated_tokens_list for DPTHead
  7. Run DPTHead -> RGB
  8. Save as PNG + mp4

Usage:
  # Unconditional generation (tokens only)
  python sample.py \
      --flow_ckpt ckpts/diffusion/ema_model.pt \
      --out_dir outputs/samples

  # Text-conditional generation with RGB decoding
  python sample.py \
      --flow_ckpt ckpts/diffusion_level11_gld/checkpoint_final.pt \
      --decoder_ckpt ckpts/decoder_gld/decoder_epoch45.pt \
      --streamvggt_ckpt ckpts/streamvggt.pt \
      --token_stats_path ckpts/token_stats.pt \
      --flow_levels 11 \
      --out_dir outputs/samples --save_format both
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from streamvggt.models.streamvggt import StreamVGGT
from models.video_dit import VideoDiT
from models.flow_matching import OTCFM
from models.clip_encoder import CLIPTextEncoder
from data.token_utils import DEFAULT_BOUNDARY_LEVEL, build_decoder_tokens_from_generated
from utils.decoder_loader import load_decoder


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Sample from VggAE-Diffusion flow model")

    # Checkpoints
    p.add_argument("--flow_ckpt", type=str, required=True,
                    help="Path to flow model (VideoDiT) checkpoint")
    p.add_argument("--decoder_ckpt", type=str, default="",
                    help="Path to DPTHead decoder checkpoint (optional, for RGB output)")
    p.add_argument("--streamvggt_ckpt", type=str, default="",
                    help="Path to StreamVGGT encoder checkpoint (optional, for config reference)")
    p.add_argument("--token_stats_path", type=str, default="",
                    help="Path to token normalization stats .pt file")

    # Flow model levels
    p.add_argument("--flow_levels", type=str, default=str(DEFAULT_BOUNDARY_LEVEL),
                    help="Comma-separated list of DPT level indices used during training")

    # Model architecture
    p.add_argument("--hidden_dim", type=int, default=768,
                    help="VideoDiT hidden dimension")
    p.add_argument("--num_layers", type=int, default=12,
                    help="Number of DiT blocks")
    p.add_argument("--num_heads", type=int, default=12,
                    help="Number of attention heads")
    p.add_argument("--seq_len", type=int, default=8,
                    help="Number of frames (S) per generated video")

    # Text conditioning
    p.add_argument("--text_prompt", type=str, default="The scene depicts a mountain trail winding along a steep slope, surrounded by lush evergreen trees. Distant mountain ranges are visible in the background, bathed in the soft hues of a late afternoon or early morning sky. The lighting is gentle, casting long shadows and creating a serene atmosphere. The trail itself is a dirt path, with wildflowers and vegetation growing along its edges. The overall tone is peaceful and inviting, evoking a sense of tranquility and natural beauty.",
                    help="Text prompt for conditional generation (requires --text_cond)")
    p.add_argument("--guidance_scale", type=float, default=7.5,
                    help="Classifier-free guidance scale (only used with --text_cond)")
    p.add_argument("--text_cond", action="store_true",
                    help="Enable text conditioning (model must be trained with cross-attention)")

    # Sampling
    p.add_argument("--num_samples", type=int, default=1,
                    help="Total number of samples to generate")
    p.add_argument("--num_steps", type=int, default=50,
                    help="Number of ODE integration steps")
    p.add_argument("--solver", type=str, default="euler", choices=["euler", "midpoint"],
                    help="ODE solver: euler or midpoint")

    # Batch / device
    p.add_argument("--batch_size", type=int, default=1,
                    help="Batch size per forward pass")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
    p.add_argument("--dtype", type=str, default="float32",
                    choices=["float32", "bfloat16", "float16"],
                    help="Computation dtype")

    # Output
    p.add_argument("--out_dir", type=str, default="outputs/samples",
                    help="Output directory for generated samples")
    p.add_argument("--save_format", type=str, default="both",
                    choices=["frames", "video", "both"],
                    help="Save format: frames (PNG), video (mp4), or both")

    # Decoder / image parameters
    p.add_argument("--img_size", type=int, default=518,
                    help="Image resolution (used by DPTHead for patch grid)")
    p.add_argument("--patch_size", type=int, default=14,
                    help="Patch size (must match encoder/decoder)")
    p.add_argument("--token_dim", type=int, default=2048,
                    help="Token dimension (must match encoder output)")
    p.add_argument("--num_levels", type=int, default=0,
                    help="Deprecated. Inferred from --flow_levels.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def sample_flow(flow, shape, args, text_emb=None, device="npu"):
    """Run ODE sampling, with optional classifier-free guidance.

    Args:
        flow: OTCFM sampler wrapping the VideoDiT model.
        shape: [B, T, N, D] shape for generated tokens.
        args: Parsed arguments.
        text_emb: [B, 1, text_dim] CLIP text embeddings (None for unconditional).
        device: Torch device.

    Returns:
        z: Generated tokens [B, T, N, D].
    """
    model_dtype = next(flow.model.parameters()).dtype
    z = torch.randn(shape, device=device, dtype=model_dtype)

    dt = torch.tensor(1.0 / args.num_steps, device=device, dtype=model_dtype)

    use_cfg = args.text_cond and text_emb is not None and args.guidance_scale > 1.0

    for i in range(args.num_steps):
        t_val = i / args.num_steps
        t = torch.full((shape[0],), t_val, device=device, dtype=model_dtype)

        if use_cfg:
            # --- Classifier-free guidance ---
            # Conditional forward
            v_cond = flow.model(z, t, text_emb=text_emb)

            # Unconditional forward matches training CFG dropout: zero text tokens.
            v_uncond = flow.model(z, t, text_emb=torch.zeros_like(text_emb))

            # Interpolate
            v = v_uncond + args.guidance_scale * (v_cond - v_uncond)
        else:
            v = flow.model(z, t, text_emb=text_emb)

        if args.solver == "euler":
            z = z + v * dt
        else:
            # Midpoint (improved Euler)
            z_mid = z + 0.5 * dt * v
            t_mid = torch.full((shape[0],), (i + 0.5) / args.num_steps,
                               device=device, dtype=model_dtype)
            if use_cfg:
                v_mid_cond = flow.model(z_mid, t_mid, text_emb=text_emb)
                v_mid_uncond = flow.model(z_mid, t_mid, text_emb=torch.zeros_like(text_emb))
                v_mid = v_mid_uncond + args.guidance_scale * (v_mid_cond - v_mid_uncond)
            else:
                v_mid = flow.model(z_mid, t_mid, text_emb=text_emb)
            z = z + dt * v_mid

    return z


def save_frames(rgb, out_dir, sample_idx):
    """Save RGB frames as PNG files.

    Args:
        rgb: [S, 3, H, W] uint8 tensor or numpy array.
        out_dir: Directory to save frames.
        sample_idx: Sample index for naming.
    """
    frames_dir = os.path.join(out_dir, f"sample_{sample_idx:04d}", "frames")
    os.makedirs(frames_dir, exist_ok=True)

    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()

    S = rgb.shape[0]
    for s in range(S):
        frame = rgb[s]  # [3, H, W]
        # CHW -> HWC for PIL
        frame_hwc = frame.transpose(1, 2, 0)
        from PIL import Image
        Image.fromarray(frame_hwc).save(os.path.join(frames_dir, f"frame_{s:05d}.png"))

    print(f"  Saved {S} frames to {frames_dir}")


def save_video(rgb, out_dir, sample_idx, fps=8):
    """Save RGB frames as mp4 video.

    Args:
        rgb: [S, 3, H, W] uint8 tensor or numpy array.
        out_dir: Directory to save video.
        sample_idx: Sample index for naming.
        fps: Frames per second.
    """
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, f"sample_{sample_idx:04d}.mp4")

    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()

    S = rgb.shape[0]
    H, W = rgb.shape[2], rgb.shape[3]

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    writer = imageio.get_writer(video_path, fps=fps, codec='libx264',
                                 pixelformat='yuv420p')
    for s in range(S):
        frame_hwc = rgb[s].transpose(1, 2, 0)  # [H, W, 3]
        writer.append_data(frame_hwc)
    writer.close()

    print(f"  Saved video to {video_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Parse flow levels
    levels = [int(x) for x in args.flow_levels.split(",")]
    num_levels = len(levels)

    # Device and dtype
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    set_seed(args.seed)

    print("=== VggAE-Diffusion Sampling ===")
    print(f"Device: {device}, dtype: {args.dtype}")
    print(f"Flow levels: {levels}")
    print(f"Model: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, "
          f"num_heads={args.num_heads}, seq_len={args.seq_len}")
    print(f"Sampling: {args.num_steps} steps, solver={args.solver}, "
          f"num_samples={args.num_samples}, batch_size={args.batch_size}")
    print(f"Text conditioning: {args.text_cond}")
    if args.text_cond:
        print(f"  Prompt: '{args.text_prompt}'")
        print(f"  Guidance scale: {args.guidance_scale}")
    print(f"Save format: {args.save_format}")
    print(f"Output dir: {args.out_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build and load flow model (auto-detect VideoDiT vs WanVGGTAdapter)
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading flow model from {args.flow_ckpt} ...")

    ckpt = torch.load(args.flow_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt.get("ema", ckpt))
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    # Auto-detect model type from checkpoint keys
    is_wan = any("wan." in k for k in state.keys())

    if is_wan:
        from models.wan_adapter import WanVGGTAdapter
        has_lora = any("lora_A" in k for k in state.keys())
        lora_r = 64 if has_lora else 0
        model = WanVGGTAdapter(
            wan_checkpoint_dir=os.path.join(os.path.dirname(__file__), "Wan2.1/checkpoints/Wan2.1-T2V-1.3B"),
            vggt_token_dim=args.token_dim, seq_len=args.seq_len,
            img_size=args.img_size, patch_size=args.patch_size,
            lora_rank=lora_r, lora_alpha=128,
        ).to(device=device)
        model.load_state_dict(state, strict=has_lora)
        print(f"  Wan adapter loaded ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params, LoRA={has_lora})")
    else:
        model = VideoDiT(
            token_dim=args.token_dim, hidden_dim=args.hidden_dim,
            num_heads=args.num_heads, num_layers=args.num_layers,
            num_levels=num_levels, seq_len=args.seq_len,
            patch_size=args.patch_size, img_size=args.img_size,
            use_cross_attn=args.text_cond,
        ).to(device=device, dtype=dtype)
        model.load_state_dict(state, strict=True)
        print(f"  VideoDiT loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    model.eval()

    flow = OTCFM(model)

    # ------------------------------------------------------------------
    # 2. Load DPTHead decoder (optional, for RGB output)
    # ------------------------------------------------------------------
    decoder = None

    if args.decoder_ckpt:
        print(f"\n[2/4] Loading decoder from {args.decoder_ckpt} ...")
        decoder = load_decoder(args.decoder_ckpt, device, decoder_type="auto",
                               patch_size=args.patch_size, img_size=args.img_size)
        print(f"  Decoder loaded ({sum(p.numel() for p in decoder.parameters()):,} params)")
    else:
        print(f"\n[2/4] No decoder checkpoint provided -- skipping RGB decoding")

    # ------------------------------------------------------------------
    # 3. Load CLIP text encoder (optional, for text conditioning)
    # ------------------------------------------------------------------
    clip_encoder = None
    text_emb = None

    if args.text_cond:
        print(f"\n[3/4] Loading CLIP text encoder ...")
        clip_encoder = CLIPTextEncoder()

        if args.text_prompt is not None:
            text_emb = clip_encoder([args.text_prompt])  # [1, L, 768]
            text_emb = text_emb.to(device=device, dtype=dtype)
            print(f"  Text prompt encoded: {text_emb.shape}")
        else:
            print("  WARNING: --text_cond set but no --text_prompt given. "
                  "Will do unconditional generation.")
    else:
        print(f"\n[3/4] No text conditioning -- skipping CLIP encoder")

    # ------------------------------------------------------------------
    # 4. Generate samples
    # ------------------------------------------------------------------
    print(f"\n[4/4] Generating {args.num_samples} sample(s) ...")

    # Token dimensions: N = 1369 patches (37x37 grid at 518/14), D = 2048
    num_patches = (args.img_size // args.patch_size) ** 2  # 37 * 37 = 1369
    token_dim = 2048
    total_frames = num_levels * args.seq_len  # single level: this is the true frame count

    sample_idx = 0
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    for batch_i in range(num_batches):
        B = min(args.batch_size, args.num_samples - sample_idx)
        shape = (B, total_frames, num_patches, token_dim)

        print(f"\n  Batch {batch_i + 1}/{num_batches} (B={B}, "
              f"shape={shape}) ...")

        # Replicate text_emb for batch if needed
        batch_text_emb = None
        if text_emb is not None and B > 1:
            batch_text_emb = text_emb.expand(B, -1, -1).contiguous()
        elif text_emb is not None:
            batch_text_emb = text_emb

        # --- ODE sampling ---
        z = sample_flow(flow, shape, args, text_emb=batch_text_emb, device=device)

        # --- Reshape to per-level ---
        # z: [B, num_levels * seq_len, N, D] -> [B, num_levels, seq_len, N, D]
        z_per_level = z.reshape(B, num_levels, args.seq_len, num_patches, token_dim)

        # --- Decode to RGB if decoder available ---
        if decoder is not None:
            print(f"  Decoding tokens to RGB via DPTHead ...")

            # Build 24-entry aggregated_tokens_list (cast to fp32 for decoder)
            aggregated_tokens_list = build_decoder_tokens_from_generated(
                z,
                levels=levels,
                seq_len=args.seq_len,
                dtype=torch.float32,
            )

            # Dummy images tensor for DPTHead (needed for shape reference)
            dummy_H = args.img_size
            dummy_W = args.img_size
            dummy_images = torch.zeros(B, args.seq_len, 3, dummy_H, dummy_W,
                                       device=device, dtype=torch.float32)

            # Run decoder
            with torch.no_grad():
                result = decoder(
                    aggregated_tokens_list,
                    images=dummy_images,
                    patch_start_idx=0,
                    frames_chunk_size=args.seq_len,
                )
                if getattr(decoder, 'output_depth', False):
                    preds, conf, _, _ = result
                else:
                    preds, conf = result
            # preds: [B, S, H, W, 3] -> [B, S, 3, H, W]
            preds = preds.permute(0, 1, 4, 2, 3).contiguous()  # [B, S, 3, H, W]
            preds = preds[:, :, :3]  # RGB only, drop confidence channel

            # Convert to uint8 [0, 255]
            preds = preds.float()
            preds = preds.clamp(0, 1)
            rgb = (preds * 255).round().to(torch.uint8)
        else:
            # No decoder: just save the raw generated tokens
            rgb = None

        # --- Save outputs ---
        for b in range(B):
            print(f"\n  Sample {sample_idx}:")

            if rgb is not None:
                sample_rgb = rgb[b]  # [S, 3, H, W]

                if args.save_format in ("frames", "both"):
                    save_frames(sample_rgb, args.out_dir, sample_idx)

                if args.save_format in ("video", "both"):
                    save_video(sample_rgb, args.out_dir, sample_idx)
            else:
                # Save raw tokens as .pt
                tokens_dir = os.path.join(args.out_dir, f"sample_{sample_idx:04d}")
                os.makedirs(tokens_dir, exist_ok=True)
                tokens_path = os.path.join(tokens_dir, "tokens.pt")
                torch.save(z_per_level[b].cpu(), tokens_path)
                print(f"  Saved raw tokens to {tokens_path} "
                      f"(shape: {list(z_per_level[b].shape)})")

            sample_idx += 1

    print(f"\n=== Done. {sample_idx} sample(s) saved to {args.out_dir} ===")


if __name__ == "__main__":
    main()
