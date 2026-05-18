#!/usr/bin/env python3
"""Overfit flow matching model on a single video to debug architecture.

Tests two configurations:
  A. Original: AdaLN-Zero + zero-init output_proj (current)
  B. Fixed:    Standard init output_proj (no zero-init)

Diagnostics at every step:
  - Loss, cosine similarity (v_pred vs v_target)
  - v_pred and v_target statistics
After training: sample tokens, decode to RGB, compare with original.

Usage:
    CUDA_VISIBLE_DEVICES=0 python overfit_single.py
    CUDA_VISIBLE_DEVICES=0 python overfit_single.py --no_zero_init --text_cond
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from streamvggt.models.streamvggt import StreamVGGT
from models.video_dit import VideoDiT
from models.flow_matching import OTCFM
from models.clip_encoder import CLIPTextEncoder
from data.token_utils import load_token_stats, normalize_tokens, select_levels, strip_special_tokens
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
        frame = (frames[s].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        writer.append_data(frame)
    writer.close()


def save_frames(frames, out_dir):
    """frames: [S, 3, H, W] float32 in [0, 1]"""
    os.makedirs(out_dir, exist_ok=True)
    for s in range(frames.shape[0]):
        frame = (frames[s].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        from PIL import Image
        Image.fromarray(frame).save(os.path.join(out_dir, f"frame_{s:03d}.png"))


class VideoDiTNoZeroInit(VideoDiT):
    """Same as VideoDiT but with standard init for output_proj and modulation."""

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "modulation" in name:
                    # Standard init instead of zeros — lets modulation learn immediately
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif "output_proj" in name:
                    # Standard init instead of zeros — lets model predict non-zero immediately
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


def build_aggregated_tokens_list(z, levels, seq_len, device, dtype):
    """Build 24-entry token list for DPTHead (patch tokens only)."""
    num_levels = len(levels)
    B, _, N, D = z.shape
    z_per_level = z.reshape(B, num_levels, seq_len, N, D)

    aggregated_tokens_list = []
    level_set = set(levels)
    for lvl_idx in range(24):
        if lvl_idx in level_set:
            pos = levels.index(lvl_idx)
            aggregated_tokens_list.append(z_per_level[:, pos])
        else:
            aggregated_tokens_list.append(
                torch.zeros(B, seq_len, N, D, device=device, dtype=dtype)
            )
    return aggregated_tokens_list


def main():
    parser = argparse.ArgumentParser(description="Overfit flow matching on single video")
    parser.add_argument("--video_path", type=str, default="",
                        help="Path to a single mp4 video for overfitting")
    parser.add_argument("--encoder_ckpt", type=str, default="",
                        help="Path to StreamVGGT checkpoint")
    parser.add_argument("--decoder_ckpt", type=str, default="",
                        help="Path to decoder checkpoint")
    parser.add_argument("--token_stats", type=str, default="",
                        help="Path to token_stats.pt")
    parser.add_argument("--annotation_index", type=str, default="",
                        help="Path to annotation_index.json (optional)")

    parser.add_argument("--no_zero_init", action="store_true",
                        help="Use standard init instead of zero-init for output_proj and modulation")
    parser.add_argument("--text_cond", action="store_true", help="Enable text conditioning")
    parser.add_argument("--caption", type=str, default="", help="Text prompt (reads from annotation if empty)")

    parser.add_argument("--num_layers", type=int, default=4, help="DiT layers (4 for fast debug)")
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--num_sample_steps", type=int, default=50, help="ODE sampling steps")
    parser.add_argument("--eval_every", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--sample_every", type=int, default=500, help="Sample and decode every N steps")
    parser.add_argument("--save_ckpt", action="store_true", help="Save model checkpoint")
    parser.add_argument("--out_dir", type=str, default="overfit_debug")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"npu:{args.gpu}")
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print(f"Overfit Debug: {'NoZeroInit' if args.no_zero_init else 'Original (Zero-Init)'}")
    print(f"  Layers: {args.num_layers}, Heads: {args.num_heads}, Hidden: {args.hidden_dim}")
    print(f"  LR: {args.lr}, Steps: {args.steps}, Text: {args.text_cond}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load encoder + extract tokens
    # ------------------------------------------------------------------
    print("\n[1/5] Loading encoder and extracting tokens...")
    encoder = StreamVGGT(img_size=518, patch_size=14, embed_dim=1024)
    state = torch.load(args.encoder_ckpt, map_location="cpu")
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device=device, dtype=torch.float16).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    level_stats = load_token_stats(args.token_stats, device, dtype=torch.float32)

    # Load video
    frames = read_video_frames(args.video_path, args.seq_len, 518)
    frames_tensor = frames.unsqueeze(0).to(device=device, dtype=torch.float16)  # [1, S, 3, H, W]

    # Extract tokens
    with torch.no_grad():
        tokens_list, patch_start_idx = encoder(frames_tensor)
        tokens_list = strip_special_tokens(tokens_list, patch_start_idx)
        tokens_list = normalize_tokens(tokens_list, level_stats)

    levels = [4, 11, 17, 23]
    x1 = select_levels(tokens_list, levels=levels)
    x1 = x1.float()  # [1, T=32, N=1369, D=2048]

    print(f"  x1 shape: {x1.shape}")
    print(f"  x1 stats: mean={x1.mean():.4f}, std={x1.std():.4f}, "
          f"min={x1.min():.3f}, max={x1.max():.3f}")

    # Load caption
    caption = args.caption
    if not caption and args.text_cond:
        import json
        vid_id = os.path.basename(args.video_path).replace(".mp4", "")
        with open(args.annotation_index) as f:
            ann = json.load(f)
        caption = ann.get(vid_id, {}).get("caption", "")
        print(f"  Caption from annotation: {caption[:80]}...")

    # Text embedding
    clip_encoder = None
    text_emb = None
    if args.text_cond and caption:
        clip_encoder = CLIPTextEncoder()
        text_emb = clip_encoder([caption]).to(device=device, dtype=torch.float32)
        print(f"  text_emb shape: {text_emb.shape}")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    print("\n[2/5] Building model...")
    ModelClass = VideoDiTNoZeroInit if args.no_zero_init else VideoDiT

    model = ModelClass(
        token_dim=2048,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_levels=4,
        seq_len=args.seq_len,
        patch_size=14,
        img_size=518,
        use_cross_attn=args.text_cond,
        use_checkpoint=False,
    ).to(device=device, dtype=torch.float32)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Check initial output
    with torch.no_grad():
        x0_test = torch.randn_like(x1)
        t_test = torch.tensor([0.5], device=device)
        v_init = model(x0_test, t_test, text_emb=text_emb)
    print(f"  Initial v_pred: mean={v_init.mean():.6f}, std={v_init.std():.6f}")
    if args.no_zero_init:
        print(f"  (NoZeroInit: should be non-zero)")

    flow = OTCFM(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Load decoder once (used for periodic sampling)
    decoder = load_decoder(args.decoder_ckpt, device, decoder_type="auto")

    def do_sample_decode(model, x1_real, step_idx, out_dir, text_emb):
        """Sample tokens, decode to RGB, save comparison."""
        model.eval()
        with torch.no_grad():
            z = torch.randn_like(x1_real)
            dt = 1.0 / args.num_sample_steps
            for i in range(args.num_sample_steps):
                t_val = i / args.num_sample_steps
                t = torch.tensor([t_val], device=device)
                v = model(z, t, text_emb=text_emb)
                z = z + v * dt

            z_gen = z.float()
            cos_f = F.cosine_similarity(
                z_gen.flatten().unsqueeze(0), x1_real.flatten().unsqueeze(0)
            ).item()

            # Decode both
            dummy = torch.zeros(1, args.seq_len, 3, 518, 518, device=device, dtype=torch.float32)
            real_agg = build_aggregated_tokens_list(x1_real, levels, args.seq_len, device, torch.float32)
            gen_agg = build_aggregated_tokens_list(z_gen, levels, args.seq_len, device, torch.float32)

            real_preds, _ = decoder(real_agg, images=dummy, patch_start_idx=0, frames_chunk_size=args.seq_len)
            gen_preds, _ = decoder(gen_agg, images=dummy, patch_start_idx=0, frames_chunk_size=args.seq_len)

            real_rgb = real_preds.squeeze(2).permute(0, 1, 4, 2, 3).contiguous()[:, :, :3].clamp(0, 1)
            gen_rgb = gen_preds.squeeze(2).permute(0, 1, 4, 2, 3).contiguous()[:, :, :3].clamp(0, 1)

        # Save
        s_dir = os.path.join(out_dir, f"step{step_idx:05d}")
        os.makedirs(s_dir, exist_ok=True)
        from PIL import Image as PILImage
        S = real_rgb.shape[1]
        for s_idx in range(S):
            orig_np = (frames_tensor[0, s_idx].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            real_np = (real_rgb[0, s_idx].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            gen_np = (gen_rgb[0, s_idx].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            combined = np.concatenate([orig_np, real_np, gen_np], axis=1)
            PILImage.fromarray(combined).save(os.path.join(s_dir, f"frame_{s_idx:03d}.png"))

        save_video(gen_rgb[0], os.path.join(s_dir, "generated.mp4"))
        save_video(real_rgb[0], os.path.join(s_dir, "real_recon.mp4"))

        model.train()
        return cos_f, z_gen

    # ------------------------------------------------------------------
    # 3. Train loop
    # ------------------------------------------------------------------
    print(f"\n[3/5] Training for {args.steps} steps...")
    train_log = []
    sample_log = []

    for step in range(args.steps):
        model.train()
        optimizer.zero_grad()

        x0 = torch.randn_like(x1)
        t = torch.rand(1, device=device)
        t_expand = t.view(-1, 1, 1, 1)

        xt = (1 - t_expand) * x0 + t_expand * x1
        v_target = x1 - x0

        v_pred = model(xt, t, text_emb=text_emb)

        loss = F.mse_loss(v_pred, v_target)
        loss.backward()
        optimizer.step()

        # Diagnostics
        with torch.no_grad():
            vp_flat = v_pred.float().flatten()
            vt_flat = v_target.float().flatten()
            cos_sim = F.cosine_similarity(vp_flat.unsqueeze(0), vt_flat.unsqueeze(0)).item()

        train_log.append({
            "step": step,
            "loss": loss.item(),
            "cos_sim": cos_sim,
            "v_pred_std": v_pred.float().std().item(),
            "v_target_std": v_target.float().std().item(),
        })

        if step % args.eval_every == 0 or step == args.steps - 1:
            print(f"  step {step:5d}: loss={loss.item():.6f}, cos_sim={cos_sim:.4f}, "
                  f"|v_pred|={v_pred.float().std():.4f}, |v_target|={v_target.float().std():.4f}")

        # Periodic sample + decode
        if args.sample_every > 0 and (step + 1) % args.sample_every == 0:
            print(f"\n  --- Sampling at step {step + 1} ---")
            cos_f, _ = do_sample_decode(model, x1, step + 1, args.out_dir, text_emb)
            print(f"  Token cos_sim (gen vs real): {cos_f:.4f}")
            sample_log.append({"step": step + 1, "token_cos_sim": cos_f})
            torch.npu.empty_cache()

    # ------------------------------------------------------------------
    # 4. Final sample
    # ------------------------------------------------------------------
    print(f"\n[4/5] Final sampling...")
    cos_final, z_gen = do_sample_decode(model, x1, args.steps, args.out_dir, text_emb)

    l2_err = (z_gen - x1).pow(2).mean().item()

    print(f"  Generated tokens: mean={z_gen.mean():.4f}, std={z_gen.std():.4f}, "
          f"range=[{z_gen.min():.3f}, {z_gen.max():.3f}]")
    print(f"  Real tokens:      mean={x1.mean():.4f}, std={x1.std():.4f}, "
          f"range=[{x1.min():.3f}, {x1.max():.3f}]")
    print(f"  Cosine similarity (gen vs real): {cos_final:.4f}")
    print(f"  L2 error (gen vs real): {l2_err:.4f}")

    # ------------------------------------------------------------------
    # 5. Decode final results
    # ------------------------------------------------------------------
    print(f"\n[5/5] Saving final results...")

    with torch.no_grad():
        dummy_images = torch.zeros(1, args.seq_len, 3, 518, 518, device=device, dtype=torch.float32)

        # Decode real tokens (reference)
        real_aggregated = build_aggregated_tokens_list(
            x1, levels, args.seq_len, device, torch.float32
        )
        real_preds, _ = decoder(real_aggregated, images=dummy_images,
                                patch_start_idx=0, frames_chunk_size=args.seq_len)
        real_rgb = real_preds.squeeze(2).permute(0, 1, 4, 2, 3).contiguous()[:, :, :3].clamp(0, 1)

        # Decode generated tokens
        gen_aggregated = build_aggregated_tokens_list(
            z_gen, levels, args.seq_len, device, torch.float32
        )
        gen_preds, _ = decoder(gen_aggregated, images=dummy_images,
                               patch_start_idx=0, frames_chunk_size=args.seq_len)
        gen_rgb = gen_preds.squeeze(2).permute(0, 1, 4, 2, 3).contiguous()[:, :, :3].clamp(0, 1)

    # Save results
    orig_dir = os.path.join(args.out_dir, "original")
    real_dir = os.path.join(args.out_dir, "real_reconstruction")
    gen_dir = os.path.join(args.out_dir, "generated")
    compare_dir = os.path.join(args.out_dir, "compare")

    save_frames(frames_tensor[0].float(), orig_dir)
    save_video(frames_tensor[0].float(), os.path.join(orig_dir, "video.mp4"))

    save_frames(real_rgb[0], real_dir)
    save_video(real_rgb[0], os.path.join(real_dir, "video.mp4"))

    save_frames(gen_rgb[0], gen_dir)
    save_video(gen_rgb[0], os.path.join(gen_dir, "video.mp4"))

    # Side-by-side: original | real_recon | generated
    os.makedirs(compare_dir, exist_ok=True)
    from PIL import Image
    S = gen_rgb.shape[1]
    for s in range(S):
        orig_np = (frames_tensor[0, s].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        real_np = (real_rgb[0, s].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        gen_np = (gen_rgb[0, s].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        combined = np.concatenate([orig_np, real_np, gen_np], axis=1)
        Image.fromarray(combined).save(os.path.join(compare_dir, f"frame_{s:03d}.png"))
    # Side-by-side video
    combined_frames = np.concatenate([
        (frames_tensor[0].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1),
        (real_rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1),
        (gen_rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1),
    ], axis=2)
    save_video(torch.from_numpy(combined_frames.transpose(0, 3, 1, 2)).float() / 255.0,
               os.path.join(compare_dir, "video.mp4"))

    print(f"\n  Results saved to {args.out_dir}/")
    print(f"  compare/: original | real_recon | generated (side-by-side)")

    # Save checkpoint
    if args.save_ckpt:
        ckpt_path = os.path.join(args.out_dir, "overfit_model.pt")
        torch.save({"model": model.state_dict()}, ckpt_path)
        print(f"  Checkpoint: {ckpt_path}")

    # Save training log
    log_path = os.path.join(args.out_dir, "train_log.txt")
    with open(log_path, "w") as f:
        f.write("step,loss,cos_sim,v_pred_std,v_target_std\n")
        for entry in train_log:
            f.write(f"{entry['step']},{entry['loss']:.6f},{entry['cos_sim']:.4f},"
                    f"{entry['v_pred_std']:.4f},{entry['v_target_std']:.4f}\n")
    print(f"  Training log: {log_path}")

    # Save sample log
    sample_path = os.path.join(args.out_dir, "sample_log.txt")
    with open(sample_path, "w") as f:
        f.write("step,token_cos_sim\n")
        for entry in sample_log:
            f.write(f"{entry['step']},{entry['token_cos_sim']:.4f}\n")
    print(f"  Sample log: {sample_path}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"  Final loss: {train_log[-1]['loss']:.6f}")
    print(f"  Final cos_sim: {train_log[-1]['cos_sim']:.4f}")
    print(f"  Token cos_sim (gen vs real): {cos_final:.4f}")
    print(f"  Token L2 error: {l2_err:.4f}")
    print(f"  Config: {'NoZeroInit' if args.no_zero_init else 'Original'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
