#!/usr/bin/env python3
"""Train flow matching (OT-CFM) in StreamVGGT token space.

Phase 2: Train a VideoDiT velocity model with optional CLIP text conditioning
using Optimal Transport Conditional Flow Matching in the aggregated token space
of a frozen StreamVGGT encoder.

Usage:
    # Without text conditioning
    python train_diffusion.py \
        --encoder_ckpt ckpts/streamvggt.pt \
        --token_stats ckpts/token_stats.pt \
        --csv data/spatialvid_train.csv \
        --video_root data/videos \
        --output_dir outputs/diffusion

    # With CLIP text conditioning
    python train_diffusion.py \
        --encoder_ckpt ckpts/streamvggt.pt \
        --token_stats ckpts/token_stats.pt \
        --csv data/spatialvid_train.csv \
        --video_root data/videos \
        --output_dir outputs/diffusion \
        --text_cond
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.heads.dpt_head import DPTHead
from models.video_dit import VideoDiT
from models.flow_matching import OTCFM
from models.clip_encoder import CLIPTextEncoder
from data.video_dataset import SpatialVidDataset, collate_fn
from data.token_utils import (
    DEFAULT_BOUNDARY_LEVEL,
    load_token_stats,
    normalize_tokens,
    select_levels,
    strip_special_tokens,
)
from utils.training import EMA, build_optimizer, build_scheduler
from utils.distributed import setup_ddp, is_main_process


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train OT-CFM diffusion in StreamVGGT token space")

    # Data
    p.add_argument("--csv", type=str, default="", help="Path to training CSV")
    p.add_argument("--video_root", type=str, default="", help="Root directory for videos")
    p.add_argument("--annotation_index", type=str, default="", help="Path to annotation index JSON")
    p.add_argument("--max_videos", type=int, default=0, help="Max number of videos (0 = all)")

    # Encoder
    p.add_argument("--encoder_ckpt", type=str, default="", help="Path to StreamVGGT checkpoint")
    p.add_argument("--token_stats", type=str, default="", help="Path to token normalization stats")
    p.add_argument("--select_levels", type=int, nargs="+", default=[DEFAULT_BOUNDARY_LEVEL],
                    help="DPT levels to diffuse. Recommended: one boundary level, e.g. 11")

    # Text conditioning
    p.add_argument("--text_cond", action="store_true", help="Enable CLIP text conditioning")
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14",
                    help="CLIP text encoder model name")
    p.add_argument("--cfg_dropout", type=float, default=0.1, help="CFG dropout probability")

    # Model
    p.add_argument("--token_dim", type=int, default=2048, help="Input token dimension")
    p.add_argument("--hidden_dim", type=int, default=768, help="Transformer hidden dimension")
    p.add_argument("--num_layers", type=int, default=12, help="Number of DiT blocks")
    p.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    p.add_argument("--num_levels", type=int, default=0,
                   help="Deprecated. Inferred from --select_levels.")
    p.add_argument("--patch_size", type=int, default=14, help="Patch size (must match encoder)")
    p.add_argument("--use_checkpoint", action="store_true", default=True,
                    help="Use gradient checkpointing to save GPU memory")
    p.add_argument("--no_checkpoint", action="store_false", dest="use_checkpoint",
                    help="Disable gradient checkpointing (faster but uses more VRAM)")

    # Frozen decoder-aware auxiliary loss. This keeps online token extraction,
    # but ties token-space training to what decoder_GLD can actually render.
    p.add_argument("--decoder_ckpt", type=str, default="",
                   help="Frozen decoder_GLD checkpoint for optional reconstruction auxiliary loss")
    p.add_argument("--recon_weight", type=float, default=0.0,
                   help="Weight for decoder-space L1 reconstruction auxiliary loss")
    p.add_argument("--recon_every", type=int, default=1,
                   help="Run decoder auxiliary every N batches (0 disables, 1=every batch)")
    p.add_argument("--recon_warmup_steps", type=int, default=0,
                   help="Start decoder auxiliary after this many optimizer steps (0=from start)")
    p.add_argument("--recon_num_frames", type=int, default=4,
                   help="Number of frames decoded for auxiliary loss; 0 means all frames")
    p.add_argument("--recon_t_min", type=float, default=0.25,
                   help="Only apply decoder auxiliary to flow samples with t >= this value")
    p.add_argument("--recon_t_max", type=float, default=1.0,
                   help="Only apply decoder auxiliary to flow samples with t <= this value")
    p.add_argument("--recon_grad_weight", type=float, default=0.05,
                   help="Image-gradient loss weight inside decoder auxiliary")
    p.add_argument("--recon_token_clip", type=float, default=8.0,
                   help="Clamp predicted normalized tokens before decoder auxiliary; <=0 disables")
    p.add_argument("--input_noise", type=float, default=0.0,
                   help="Gaussian noise std added to x1 tokens during diffusion training (regularization)")

    # Training
    p.add_argument("--batch_size", type=int, default=2, help="Per-GPU batch size")
    p.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    p.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    p.add_argument("--warmup_steps", type=int, default=2000, help="Linear warmup steps")
    p.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    p.add_argument("--use_bf16", action="store_true", default=True, help="Use BF16 mixed precision")
    p.add_argument("--no_bf16", action="store_true", help="Disable BF16 mixed precision")

    # Dataset
    p.add_argument("--seq_len", type=int, default=8, help="Number of frames per video")
    p.add_argument("--target_size", type=int, default=518, help="Frame resize target")
    p.add_argument("--num_frames_per_video", type=int, default=8, help="Frames to sample per video")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Eval / save
    p.add_argument("--eval_every", type=int, default=10, help="Evaluate every N epochs")
    p.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")

    # Output
    p.add_argument("--output_dir", type=str, default="outputs/diffusion", help="Output directory")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--local_rank", type=int, default=0, help="DDP local rank (set by torchrun)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def image_gradient_loss(recon, target):
    recon_dx = recon[..., :, 1:] - recon[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    recon_dy = recon[..., 1:, :] - recon[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(recon_dx, target_dx) + F.l1_loss(recon_dy, target_dy)


def load_decoder_gld(path, device, output_depth=False):
    """Load a frozen decoder for diffusion auxiliary loss. Auto-detects DPT/ViT."""
    from utils.decoder_loader import load_decoder
    return load_decoder(path, device, decoder_type="auto", output_depth=output_depth)


def build_sparse_decoder_tokens(z, levels, total_levels=24):
    """Build the minimal token list DPTHead needs for generated single-level tokens."""
    if len(levels) != 1:
        raise ValueError("sparse decoder tokens only supports one generated level")

    B, S, N, D = z.shape
    zero_level = torch.zeros(B, S, N, D, device=z.device, dtype=z.dtype)
    tokens = [zero_level.clone() for _ in range(total_levels)]
    tokens[levels[0]] = z
    return tokens


def decoder_auxiliary_loss(decoder, x1_pred, frames, t, args):
    """Decode predicted clean tokens and compare in RGB space.

    This intentionally supports only single-level diffusion. The decoder_GLD
    training path is boundary-level-first.
    """
    if len(args.select_levels) != 1:
        return x1_pred.new_zeros(())

    keep = (t >= args.recon_t_min) & (t <= args.recon_t_max)
    if keep.sum().item() == 0:
        return x1_pred.new_zeros(())

    z = x1_pred[keep]
    target = frames[keep].float()

    if args.recon_num_frames > 0 and args.recon_num_frames < z.shape[1]:
        frame_ids = torch.randperm(z.shape[1], device=z.device)[:args.recon_num_frames].sort().values
        z = z[:, frame_ids]
        target = target[:, frame_ids]

    if args.recon_token_clip > 0:
        z = z.clamp(-args.recon_token_clip, args.recon_token_clip)

    z = z.to(dtype=torch.float32)
    tokens = build_sparse_decoder_tokens(z, args.select_levels)

    with torch.amp.autocast(device_type=target.device.type, enabled=False):
        result = decoder(
            tokens,
            images=target,
            patch_start_idx=0,
            frames_chunk_size=max(1, min(z.shape[1], args.recon_num_frames or z.shape[1])),
        )
        if decoder.output_depth:
            preds, _, _, _ = result
        else:
            preds, _ = result
        recon = preds.squeeze(2).permute(0, 1, 4, 2, 3).contiguous()
        recon = recon[:, :, :3].float()
        l1 = F.l1_loss(recon, target)
        grad = image_gradient_loss(recon, target)

    return l1 + args.recon_grad_weight * grad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # DDP setup
    use_ddp, rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    main_process = is_main_process()

    use_bf16 = args.use_bf16 and not args.no_bf16 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    if main_process:
        print(f"=== OT-CFM Diffusion Training ===")
        print(f"Device: {device}, BF16: {use_bf16}, DDP: {use_ddp} (world_size={world_size})")
        print(f"Diffused DPT levels: {args.select_levels}")
        if len(args.select_levels) > 1:
            print("  WARNING: multi-level diffusion keeps a legacy flattened level*time axis. "
                  "For stable geometry-aware generation, train a single boundary level first.")
        print(f"Text conditioning: {args.text_cond}")
        if args.text_cond:
            print(f"  CFG dropout: {args.cfg_dropout}")
        print(f"Batch size: {args.batch_size} x {args.accum_steps} accum = "
              f"{args.batch_size * args.accum_steps * world_size} effective")

    set_seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load frozen StreamVGGT encoder + token stats
    # ------------------------------------------------------------------
    if main_process:
        print(f"\n[1/6] Loading StreamVGGT encoder from {args.encoder_ckpt} ...")
    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    state_dict = torch.load(args.encoder_ckpt, map_location="cpu")
    encoder.load_state_dict(state_dict, strict=False)
    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    level_stats = load_token_stats(args.token_stats, device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 2. Load CLIP text encoder if --text_cond
    # ------------------------------------------------------------------
    clip_encoder = None
    if args.text_cond:
        if main_process:
            print(f"[2/6] Loading CLIP text encoder ({args.clip_model}) ...")
        clip_encoder = CLIPTextEncoder(model_name=args.clip_model)

    # ------------------------------------------------------------------
    # 3. Build VideoDiT + OTCFM trainer
    # ------------------------------------------------------------------
    if main_process:
        print(f"[3/6] Building VideoDiT "
              f"(token_dim={args.token_dim}, hidden_dim={args.hidden_dim}, "
              f"layers={args.num_layers}, heads={args.num_heads}, "
              f"use_cross_attn={args.text_cond}) ...")

    model_num_levels = len(args.select_levels)
    model = VideoDiT(
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_levels=model_num_levels,
        seq_len=args.seq_len,
        patch_size=args.patch_size,
        img_size=args.target_size,
        use_cross_attn=args.text_cond,
        use_checkpoint=args.use_checkpoint,
    ).to(device=device, dtype=dtype)

    if main_process:
        print(f"  Model latent shape per sample: "
              f"[T={model_num_levels * args.seq_len}, N={(args.target_size // args.patch_size) ** 2}, D={args.token_dim}]")
        print(f"  Trainable parameters: {count_parameters(model):,}")

    flow = OTCFM(model)

    decoder_gld = None
    decoder_aux_enabled = (
        args.decoder_ckpt
        and args.recon_weight > 0
        and args.recon_every > 0
    )
    if decoder_aux_enabled:
        if len(args.select_levels) != 1:
            raise ValueError("decoder auxiliary is only supported for single-level diffusion, e.g. --select_levels 11")
        if main_process:
            print(f"[3.5/6] Loading frozen decoder_GLD from {args.decoder_ckpt} ...")
            print(f"  Decoder aux: weight={args.recon_weight}, every={args.recon_every} batches, "
                  f"warmup={args.recon_warmup_steps} steps, frames={args.recon_num_frames}, "
                  f"t_range=[{args.recon_t_min}, {args.recon_t_max}]")
        decoder_gld = load_decoder_gld(args.decoder_ckpt, device)

    # EMA
    ema = EMA(model, decay=args.ema_decay).to(device)

    # ------------------------------------------------------------------
    # 4. Build dataset + dataloader
    # ------------------------------------------------------------------
    if main_process:
        print(f"[4/6] Building SpatialVidDataset ...")
    dataset = SpatialVidDataset(
        csv_path=args.csv,
        video_root=args.video_root,
        seq_len=args.seq_len,
        target_size=args.target_size,
        annotation_index_path=args.annotation_index,
        max_videos=args.max_videos,
        num_frames_per_video=args.num_frames_per_video,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if use_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # 5. Build optimizer + scheduler
    # ------------------------------------------------------------------
    if main_process:
        print(f"[5/6] Building optimizer (lr={args.lr}, wd={args.wd}) ...")

    optimizer = build_optimizer(model, lr=args.lr, wd=args.wd)

    steps_per_epoch = (len(dataloader) + args.accum_steps - 1) // args.accum_steps
    total_steps = args.epochs * steps_per_epoch

    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=max(total_steps, 1))

    # GradScaler for fp16 fallback (unused with bf16, but harmless to create)
    scaler = GradScaler(enabled=(not use_bf16))

    # ------------------------------------------------------------------
    # 6. Resume if requested
    # ------------------------------------------------------------------
    global_step = 0
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        if main_process:
            print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        ema = ema.to(device)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        if main_process:
            print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    # DDP model wrapper (after loading state_dict)
    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False,
        )
        # Re-wrap flow model reference
        flow.model = model

    # TensorBoard
    writer = None
    if main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if main_process:
        print(f"\n[6/6] Starting training: {args.epochs} epochs, "
              f"{steps_per_epoch} steps/epoch, {total_steps} total steps\n")

    text_dim = clip_encoder.dim if clip_encoder else 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0

        if use_ddp:
            sampler.set_epoch(epoch)

        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            # --- Load frames ---
            frames = batch["frames"].to(device=device, dtype=torch.bfloat16)  # [B, S, 3, H, W]

            # --- Encode with frozen StreamVGGT ---
            with torch.no_grad():
                tokens_list, psi = encoder(frames)
                tokens_list = strip_special_tokens(tokens_list, psi)
                tokens_list = normalize_tokens(tokens_list, level_stats)

            # --- Select DPT levels ---
            x1 = select_levels(tokens_list, levels=args.select_levels)
            x1 = x1.to(dtype=dtype)

            # Input noise regularization
            if args.input_noise > 0:
                x1 = x1 + torch.randn_like(x1) * args.input_noise

            # --- Text conditioning ---
            text_emb = None
            if clip_encoder is not None:
                text_emb = clip_encoder(batch["caption"])  # [B, L, 768]
                text_emb = text_emb.to(device=device, dtype=dtype)

                # CFG dropout: randomly zero out text embeddings during training
                if model.training and args.cfg_dropout > 0:
                    mask = torch.rand(x1.shape[0], device=device) < args.cfg_dropout
                    text_emb = text_emb * (~mask).view(-1, 1, 1).to(dtype=text_emb.dtype)

            use_decoder_aux = (
                decoder_gld is not None
                and global_step >= args.recon_warmup_steps
                and (batch_idx % args.recon_every == 0)
            )
            recon_loss = x1.new_zeros(())

            # --- Forward pass with mixed precision ---
            if use_bf16:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    flow_out = flow.compute_loss(
                        x1,
                        text_emb=text_emb,
                        return_outputs=use_decoder_aux,
                    )
            else:
                flow_out = flow.compute_loss(
                    x1,
                    text_emb=text_emb,
                    return_outputs=use_decoder_aux,
                )

            if use_decoder_aux:
                flow_loss = flow_out["loss"]
                recon_loss = decoder_auxiliary_loss(
                    decoder_gld,
                    flow_out["x1_pred"],
                    frames,
                    flow_out["t"],
                    args,
                )
                loss = flow_loss + args.recon_weight * recon_loss
            else:
                flow_loss = flow_out
                loss = flow_loss

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1

            # --- Backward with gradient accumulation ---
            scaled_loss = loss / args.accum_steps
            if not use_bf16:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # --- Optimizer step (after accumulation) ---
            if (batch_idx + 1) % args.accum_steps == 0:
                if not use_bf16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                # EMA update
                ema.update(model.module if use_ddp else model)

                # Scheduler step
                scheduler.step()
                global_step += 1

                # Update progress bar
                pbar.set_postfix(
                    loss=f"{loss_val:.6f}",
                    flow=f"{flow_loss.item():.6f}",
                    recon=f"{recon_loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

                # Logging
                if main_process and global_step % 50 == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    print(f"  [epoch {epoch} step {global_step}] "
                          f"loss={loss_val:.6f}, flow={flow_loss.item():.6f}, "
                          f"recon={recon_loss.item():.4f}, lr={lr_now:.2e}")
                    if writer is not None:
                        writer.add_scalar("train/loss", loss_val, global_step)
                        writer.add_scalar("train/flow_loss", flow_loss.item(), global_step)
                        writer.add_scalar("train/recon_loss", recon_loss.item(), global_step)
                        writer.add_scalar("train/lr", lr_now, global_step)

        if num_batches > 0 and num_batches % args.accum_steps != 0:
            if not use_bf16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            ema.update(model.module if use_ddp else model)
            scheduler.step()
            global_step += 1

        # --- Epoch summary ---
        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start

        if main_process:
            print(f"\n  Epoch {epoch}/{args.epochs} completed in {epoch_time:.1f}s | "
                  f"avg loss: {avg_loss:.6f} | steps: {global_step}")
            if writer is not None:
                writer.add_scalar("train/epoch_loss", avg_loss, epoch)
                writer.add_scalar("train/epoch_time", epoch_time, epoch)

        # --- Evaluation ---
        if (epoch + 1) % args.eval_every == 0 and main_process:
            print(f"\n  === Evaluation at epoch {epoch} ===")
            model.eval()

            def run_eval():
                loss_sum, n = 0.0, 0
                with torch.no_grad():
                    for ei, eb in enumerate(dataloader):
                        if ei >= 20:
                            break
                        ef = eb["frames"].to(device=device, dtype=torch.bfloat16)
                        tl, psi = encoder(ef)
                        tl = strip_special_tokens(tl, psi)
                        tl = normalize_tokens(tl, level_stats)
                        ex = select_levels(tl, levels=args.select_levels).to(dtype=dtype)
                        et = clip_encoder(eb["caption"]).to(device=device, dtype=dtype) if clip_encoder else None
                        with autocast(device_type="cuda", dtype=torch.bfloat16):
                            el = flow.compute_loss(ex, text_emb=et)
                        loss_sum += el.item()
                        n += 1
                return loss_sum / max(n, 1)

            eval_avg = run_eval()

            # EMA eval: swap weights, eval, swap back
            print(f"  Eval loss (EMA): computing with EMA weights...")
            base_model = model.module if use_ddp else model
            orig_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
            base_model.load_state_dict(ema.state_dict())
            eval_ema_avg = run_eval()
            base_model.load_state_dict(orig_state)
            del orig_state

            print(f"  Eval loss: {eval_avg:.6f} | Eval loss (EMA): {eval_ema_avg:.6f}")
            if writer is not None:
                writer.add_scalar("eval/loss", eval_avg, epoch)
                writer.add_scalar("eval/ema_loss", eval_ema_avg, epoch)

            torch.cuda.empty_cache()
            model.train()

        # --- Save checkpoint ---
        if (epoch + 1) % args.save_every == 0 and main_process:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch:04d}.pt")
            base_model = model.module if use_ddp else model
            torch.save({
                "model": base_model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "args": vars(args),
            }, save_path)
            print(f"  Checkpoint saved: {save_path}")

    # --- Final save ---
    if main_process:
        final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        base_model = model.module if use_ddp else model
        torch.save({
            "model": base_model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": args.epochs - 1,
            "args": vars(args),
        }, final_path)
        print(f"\nTraining complete. Final checkpoint: {final_path}")

        # Save EMA-only weights for inference
        ema_path = os.path.join(args.output_dir, "ema_model.pt")
        torch.save(ema.state_dict(), ema_path)
        print(f"EMA weights saved: {ema_path}")

    if use_ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
