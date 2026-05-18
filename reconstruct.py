#!/usr/bin/env python3
"""Reconstruct real videos through StreamVGGT encoder → DPTHead decoder.

Pipeline:
  1. Decode mp4 → frames [S, 3, 518, 518]
  2. StreamVGGT encoder (frozen) → aggregated tokens
  3. Normalize tokens
  4. DPTHead decoder → RGB reconstruction
  5. Save original + reconstructed as mp4

Usage:
    python reconstruct.py \
        --video_path /path/to/video.mp4 \
        --encoder_ckpt ckpts/streamvggt.pt \
        --decoder_ckpt ckpts/decoder/decoder_best.pt \
        --token_stats ckpts/token_stats.pt \
        --out_dir reconstruct_out/
"""

import argparse
import os

import torch
import numpy as np

from streamvggt.models.streamvggt import StreamVGGT
from data.token_utils import DPT_LEVELS, load_token_stats, normalize_tokens, strip_special_tokens
from utils.video_io import read_video_frames
from utils.decoder_loader import load_decoder


def save_video(frames, path, fps=8):
    """frames: [S, 3, H, W] float32 in [0, 1]"""
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p")
    for s in range(frames.shape[0]):
        frame_hwc = (frames[s].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        writer.append_data(frame_hwc)
    writer.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct video via encoder → decoder")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument("--decoder_ckpt", type=str, required=True)
    parser.add_argument("--token_stats", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="reconstruct_out")
    parser.add_argument("--seq_len", type=int, default=0, help="Max frames (0=no limit)")
    parser.add_argument("--sample_fps", type=float, default=8, help="Sample N frames per second from video")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--keep_levels", type=str, default="",
                        help="Comma-separated DPT levels to keep for decoder eval, e.g. '11'. Empty keeps all.")
    args = parser.parse_args()
    keep_levels = None
    if args.keep_levels:
        keep_levels = {int(x) for x in args.keep_levels.split(",") if x.strip()}

    device = torch.device(f"npu:{args.gpu}")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load encoder (frozen, bf16)
    print("[1/4] Loading StreamVGGT encoder...")
    encoder = StreamVGGT(img_size=args.img_size, patch_size=args.patch_size, embed_dim=1024)
    state = torch.load(args.encoder_ckpt, map_location="cpu")
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device=device, dtype=torch.float16).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # 2. Load token stats
    print("[2/4] Loading token stats...")
    level_stats = load_token_stats(args.token_stats, device, dtype=torch.float32)

    # 3. Load decoder (fp32)
    print("[3/4] Loading decoder...")
    decoder = load_decoder(args.decoder_ckpt, device, decoder_type="auto",
                           patch_size=args.patch_size, img_size=args.img_size)

    # 4. Decode video and reconstruct
    print(f"[4/4] Processing {args.video_path}...")

    # Get video fps and original resolution
    import cv2
    cap = cv2.VideoCapture(args.video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if orig_fps <= 0:
        orig_fps = 24.0
    print(f"  Video: {orig_total} frames at {orig_fps:.1f}fps, resolution {orig_w}x{orig_h}")

    # Determine number of frames to sample
    if args.sample_fps > 0:
        args.seq_len = int(orig_total / orig_fps * args.sample_fps)
        args.seq_len = min(args.seq_len, orig_total)
    elif args.seq_len <= 0:
        args.seq_len = orig_total
    args.seq_len = max(args.seq_len, 1)
    print(f"  Video: {orig_total} frames at {orig_fps:.1f}fps, sampling {args.seq_len} frames ({args.sample_fps} fps)")

    frames = read_video_frames(args.video_path, args.seq_len, args.img_size)
    frames_tensor = frames.unsqueeze(0).to(device=device, dtype=torch.float16)  # [1, S, 3, H, W]
    B, S = frames_tensor.shape[:2]

    with torch.no_grad():
        # Encode
        tokens_list, psi = encoder.aggregator(frames_tensor)
        tokens_list = strip_special_tokens(tokens_list, psi)
        # Normalize
        tokens_list = normalize_tokens(tokens_list, level_stats)
        if keep_levels is not None:
            for lvl in DPT_LEVELS:
                if lvl not in keep_levels:
                    tokens_list[lvl] = torch.zeros_like(tokens_list[lvl])
        tokens_list = [t.to(dtype=torch.float32) for t in tokens_list]
        # Decode
        recon, conf = decoder(
            tokens_list,
            images=frames_tensor.float(),
            patch_start_idx=0,
            frames_chunk_size=S,
        )
        # recon: [B, S, 1, H, W, 4] -> [B, S, 3, H, W]
        # recon: [B, S, H, W, 3] -> [B, S, 3, H, W]
        recon = recon.permute(0, 1, 4, 2, 3).contiguous()
        recon = recon[:, :, :3].clamp(0, 1).float()

    # Compute metrics
    target = frames_tensor.float()
    l1 = (recon - target).abs().mean().item()

    # PSNR
    mse = (recon - target).pow(2).mean().item()
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')

    # SSIM (per-frame, average)
    from skimage.metrics import structural_similarity as ssim_fn
    ssim_vals = []
    for s in range(S):
        orig_np = target[0, s].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        recon_np = recon[0, s].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        ssim_vals.append(ssim_fn(orig_np, recon_np, channel_axis=2, data_range=1.0))
    ssim_avg = np.mean(ssim_vals)

    print(f"\n  === Reconstruction Metrics ===")
    print(f"  L1:   {l1:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim_avg:.4f} (per-frame avg)")
    print(f"  Recon: mean={recon.mean():.4f}, std={recon.std():.4f}, range=[{recon.min():.3f}, {recon.max():.3f}]")
    print(f"  Target: mean={target.mean():.4f}, std={target.std():.4f}, range=[{target.min():.3f}, {target.max():.3f}]")

    # Save original (resized to match reconstructed) and reconstructed
    out_fps = args.sample_fps
    print(f"  Saving at {args.img_size}x{args.img_size} @ {out_fps}fps, {S} frames")
    save_video(target[0], os.path.join(args.out_dir, "original.mp4"), fps=out_fps)
    save_video(recon[0], os.path.join(args.out_dir, "reconstructed.mp4"), fps=out_fps)

    # Side-by-side comparison
    from PIL import Image
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for s in range(S):
        orig_np = (target[0, s].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        recon_np = (recon[0, s].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        combined = np.concatenate([orig_np, recon_np], axis=1)
        Image.fromarray(combined).save(os.path.join(frames_dir, f"frame_{s:03d}.png"))
    print(f"  Side-by-side frames saved to {frames_dir}/")
    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
