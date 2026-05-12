#!/usr/bin/env python3
"""Train OT-CFM diffusion with Wan2.1 1.3B pretrained DiT backbone.

Phase 2 variant: Uses WanVGGTAdapter instead of from-scratch VideoDiT.
The Wan backbone provides pretrained motion/temporal reasoning;
only the input/output adapters and backbone are fine-tuned.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from streamvggt.models.streamvggt import StreamVGGT
from models.video_dit import VideoDiT  # kept for import compatibility
from models.wan_adapter import WanVGGTAdapter
from models.flow_matching import OTCFM
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
from utils.decoder_loader import load_decoder


def parse_args():
    p = argparse.ArgumentParser(description="Train OT-CFM with Wan2.1 backbone")

    # Data
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--video_root", type=str, default="")
    p.add_argument("--max_videos", type=int, default=0)

    # Checkpoints
    p.add_argument("--encoder_ckpt", type=str, default="")
    p.add_argument("--token_stats", type=str, default="")
    p.add_argument("--wan_ckpt_dir", type=str, required=True,
                   help="Path to Wan2.1 1.3B pretrained checkpoint directory")
    p.add_argument("--select_levels", type=int, nargs="+", default=[DEFAULT_BOUNDARY_LEVEL])

    # Model
    p.add_argument("--token_dim", type=int, default=2048)
    p.add_argument("--patch_size", type=int, default=14)
    p.add_argument("--lora_rank", type=int, default=64, help="LoRA rank (0 = full fine-tune)")
    p.add_argument("--lora_alpha", type=float, default=128)
    p.add_argument("--use_checkpoint", action="store_true", default=True)

    # Frozen decoder auxiliary loss
    p.add_argument("--decoder_ckpt", type=str, default="")
    p.add_argument("--recon_weight", type=float, default=0.05)
    p.add_argument("--recon_every", type=int, default=1)
    p.add_argument("--recon_warmup_steps", type=int, default=0)
    p.add_argument("--recon_num_frames", type=int, default=4)
    p.add_argument("--recon_t_min", type=float, default=0.25)
    p.add_argument("--recon_t_max", type=float, default=1.0)
    p.add_argument("--recon_grad_weight", type=float, default=0.05)
    p.add_argument("--recon_token_clip", type=float, default=8.0)

    # Training
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--accum_steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--input_noise", type=float, default=0.005)
    p.add_argument("--text_cond", action="store_true")
    p.add_argument("--cfg_dropout", type=float, default=0.1)

    # Dataset
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--target_size", type=int, default=518)
    p.add_argument("--num_frames_per_video", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)

    # Eval / save
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=10)

    # Output
    p.add_argument("--output_dir", type=str, default="outputs/diffusion_wan")
    p.add_argument("--resume", type=str, default="")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int, default=0)

    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_gradient_loss(recon, target):
    recon_dx = recon[..., :, 1:] - recon[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    recon_dy = recon[..., 1:, :] - recon[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(recon_dx, target_dx) + F.l1_loss(recon_dy, target_dy)


def build_sparse_decoder_tokens(z, levels, total_levels=24):
    if len(levels) != 1:
        raise ValueError("sparse decoder tokens only supports one generated level")
    B, S, N, D = z.shape
    zero_level = torch.zeros(B, S, N, D, device=z.device, dtype=z.dtype)
    tokens = [zero_level.clone() for _ in range(total_levels)]
    tokens[levels[0]] = z
    return tokens


def decoder_auxiliary_loss(decoder, x1_pred, frames, t, args):
    if len(args.select_levels) != 1:
        return x1_pred.new_zeros(())
    keep = (t >= args.recon_t_min) & (t <= args.recon_t_max)
    if keep.sum().item() == 0:
        return x1_pred.new_zeros(())
    z = x1_pred[keep]
    target = frames[keep].float()
    if args.recon_num_frames > 0 and args.recon_num_frames < z.shape[1]:
        frame_ids = torch.randperm(z.shape[1], device=z.device)[:args.recon_num_frames].sort().values
        z = z[:, frame_ids]; target = target[:, frame_ids]
    if args.recon_token_clip > 0:
        z = z.clamp(-args.recon_token_clip, args.recon_token_clip)
    z = z.to(dtype=torch.float32)
    tokens = build_sparse_decoder_tokens(z, args.select_levels)

    with torch.amp.autocast(device_type=target.device.type, enabled=False):
        result = decoder(tokens, images=target, patch_start_idx=0,
                         frames_chunk_size=max(1, min(z.shape[1], args.recon_num_frames or z.shape[1])))
        if getattr(decoder, 'output_depth', False):
            preds, _, _, _ = result
        else:
            preds, _ = result
        recon = preds.permute(0, 1, 4, 2, 3).contiguous()
        recon = recon[:, :, :3].float()
        l1 = F.l1_loss(recon, target)
        grad = image_gradient_loss(recon, target)
    return l1 + args.recon_grad_weight * grad


def main():
    args = parse_args()

    use_ddp, rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    main_process = is_main_process()

    use_bf16 = args.use_bf16 and not args.no_bf16 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    if main_process:
        print(f"=== OT-CFM Diffusion (Wan2.1 backbone) ===")
        print(f"Device: {device}, BF16: {use_bf16}, DDP: {use_ddp}")
        print(f"Wan ckpt: {args.wan_ckpt_dir}")
        print(f"Diffused DPT levels: {args.select_levels}")

    set_seed(args.seed)

    # ---- 1. Load frozen StreamVGGT encoder ----
    if main_process:
        print(f"\n[1/5] Loading encoder...")
    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    state_dict = torch.load(args.encoder_ckpt, map_location="cpu")
    encoder.load_state_dict(state_dict, strict=False)
    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    level_stats = load_token_stats(args.token_stats, device, dtype=torch.float32)

    # ---- 2. Build Wan adapter ----
    if main_process:
        print(f"[2/5] Loading Wan2.1 backbone from {args.wan_ckpt_dir} ...")
    model = WanVGGTAdapter(
        wan_checkpoint_dir=args.wan_ckpt_dir,
        vggt_token_dim=args.token_dim,
        seq_len=args.seq_len,
        img_size=args.target_size,
        patch_size=args.patch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    ).to(device=device)

    if args.lora_rank > 0:
        model.set_lora_trainable()

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if main_process:
        mode = f"LoRA rank={args.lora_rank}" if args.lora_rank > 0 else "full fine-tune"
        print(f"  Wan adapter ({mode}): {total_p/1e9:.2f}B total, {trainable_p/1e6:.1f}M trainable")

    # Wan blocks require time_embedding in float32; .to(bf16) converts them.
    # Restore float32 for these small modules.
    for m in [model.wan.time_embedding, model.wan.time_projection]:
        for p in m.parameters():
            p.data = p.data.float()

    flow = OTCFM(model)
    ema = EMA(model, decay=args.ema_decay).to(device)

    # ---- 3. CLIP text encoder ----
    clip_encoder = None
    if args.text_cond:
        if main_process:
            print(f"[2.5/5] Loading CLIP text encoder...")
        from models.clip_encoder import CLIPTextEncoder
        clip_encoder = CLIPTextEncoder()

    # ---- 3.5 Optional frozen decoder ----
    decoder_gld = None
    if args.decoder_ckpt and args.recon_weight > 0 and args.recon_every > 0:
        if main_process:
            print(f"[3/5] Loading frozen decoder from {args.decoder_ckpt} ...")
        decoder_gld = load_decoder(args.decoder_ckpt, device, decoder_type="auto")

    # ---- 4. Dataset ----
    if main_process:
        print(f"[3/5] Building dataset...")
    dataset = SpatialVidDataset(
        csv_path=args.csv, video_root=args.video_root,
        seq_len=args.seq_len, target_size=args.target_size,
        max_videos=args.max_videos, num_frames_per_video=args.num_frames_per_video,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if use_ddp else None
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )

    # ---- Eval sampling ----
    eval_frames = None
    for eb in dataloader:
        eval_frames = eb["frames"][:1].clone()
        break

    @torch.no_grad()
    def sample_and_decode(model_ema, step, out_dir):
        """Generate one sample via ODE + decode for visual progress tracking."""
        model_ema.eval()
        # Wan model is float32; cast noise to float32, let autocast handle rest
        z = torch.randn(1, args.seq_len, 1369, args.token_dim, device=device).float()
        dt = 1.0 / 20
        for i in range(20):
            t_val = torch.tensor([i / 20.], device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                v = model_ema(z, t_val)
            z = (z + v * dt).float()
        # Decode with frozen decoder if available
        if decoder_gld is not None:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                tokens_out = build_sparse_decoder_tokens(z.float(), args.select_levels)
                result = decoder_gld(tokens_out, images=eval_frames.float().to(device),
                                     patch_start_idx=0, frames_chunk_size=args.seq_len)
            if getattr(decoder_gld, 'output_depth', False):
                preds, _, _, _ = result
            else:
                preds, _ = result
            recon = preds.permute(0, 1, 4, 2, 3).contiguous().clamp(0, 1)
            # Save comparison: original | reconstructed | generated
            import numpy as np
            from PIL import Image as PImage
            orig = eval_frames[0, 0].to(device).clamp(0, 1)
            rec_frame = recon[0, 0].to(device)
            gen_frame = recon[0, -1].to(device)
            row = torch.cat([orig, rec_frame, gen_frame], dim=2)
            row_np = (row.float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            os.makedirs(out_dir, exist_ok=True)
            PImage.fromarray(row_np).save(os.path.join(out_dir, f"sample_step{step:06d}.png"))
        model_ema.train()

    # ---- 5. Optimizer ----
    if main_process:
        print(f"[4/5] Building optimizer (lr={args.lr}) ...")
    optimizer = build_optimizer(model, lr=args.lr, wd=args.wd)
    steps_per_epoch = (len(dataloader) + args.accum_steps - 1) // args.accum_steps
    total_steps = args.epochs * steps_per_epoch
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=max(total_steps, 1))
    scaler = GradScaler(enabled=(not use_bf16))

    # Resume
    global_step = 0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        if main_process:
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"]); ema = ema.to(device)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )
        flow.model = model

    writer = None
    if main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

    if main_process:
        print(f"\n[5/5] Training: {args.epochs} epochs, {steps_per_epoch} steps/epoch")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        if use_ddp:
            sampler.set_epoch(epoch)
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            frames = batch["frames"].to(device=device, dtype=torch.bfloat16)

            with torch.no_grad():
                tokens_list, psi = encoder(frames)
                tokens_list = strip_special_tokens(tokens_list, psi)
                tokens_list = normalize_tokens(tokens_list, level_stats)

            x1 = select_levels(tokens_list, levels=args.select_levels).to(dtype=dtype)

            if args.input_noise > 0:
                x1 = x1 + torch.randn_like(x1) * args.input_noise

            # Text conditioning
            text_emb = None
            if clip_encoder is not None:
                text_emb = clip_encoder(batch["caption"]).to(device=device, dtype=dtype)
                if args.cfg_dropout > 0:
                    mask = torch.rand(x1.shape[0], device=device) < args.cfg_dropout
                    text_emb = text_emb * (~mask).view(-1, 1, 1).to(dtype=text_emb.dtype)

            use_decoder_aux = (
                decoder_gld is not None and global_step >= args.recon_warmup_steps
                and (batch_idx % args.recon_every == 0)
            )
            recon_loss = x1.new_zeros(())

            if use_bf16:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    flow_out = flow.compute_loss(x1, text_emb=text_emb, return_outputs=use_decoder_aux)
            else:
                flow_out = flow.compute_loss(x1, text_emb=text_emb, return_outputs=use_decoder_aux)

            if use_decoder_aux:
                flow_loss = flow_out["loss"]
                recon_loss = decoder_auxiliary_loss(decoder_gld, flow_out["x1_pred"], frames,
                                                    flow_out["t"], args)
                loss = flow_loss + args.recon_weight * recon_loss
            else:
                loss = flow_out

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1

            scaled_loss = loss / args.accum_steps
            if not use_bf16:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (batch_idx + 1) % args.accum_steps == 0:
                if not use_bf16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model.module if use_ddp else model)
                scheduler.step()
                global_step += 1
                pbar.set_postfix(loss=f"{loss_val:.6f}", flow=f"{flow_loss.item():.6f}" if use_decoder_aux else f"{loss_val:.6f}",
                                 recon=f"{recon_loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

                if main_process and writer is not None and global_step % 50 == 0:
                    writer.add_scalar("train/loss", loss_val, global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        if num_batches > 0 and num_batches % args.accum_steps != 0:
            if not use_bf16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model.module if use_ddp else model)
            scheduler.step()
            global_step += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        if main_process:
            print(f"  Epoch {epoch}/{args.epochs} | avg loss: {avg_loss:.6f} | steps: {global_step}")
            if writer is not None:
                writer.add_scalar("train/epoch_loss", avg_loss, epoch)

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
            print(f"  Checkpoint: {save_path}")
            if main_process:
                try:
                    raw = model.module if use_ddp else model
                    ema_state = {k: v for k, v in ema.state_dict().items()}
                    orig_state = {k: v.clone() for k, v in raw.state_dict().items()}
                    raw.load_state_dict(ema_state)
                    sample_and_decode(raw, global_step, os.path.join(args.output_dir, "samples"))
                    raw.load_state_dict(orig_state)
                    del orig_state, ema_state
                except Exception as e:
                    print(f"  [WARN] Eval sampling failed: {e}")

    if main_process:
        final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        base_model = model.module if use_ddp else model
        torch.save({
            "model": base_model.state_dict(),
            "ema": ema.state_dict(),
            "global_step": global_step,
            "epoch": args.epochs - 1,
            "args": vars(args),
        }, final_path)
        ema_path = os.path.join(args.output_dir, "ema_model.pt")
        torch.save(ema.state_dict(), ema_path)
        print(f"\nDone. Final: {final_path}, EMA: {ema_path}")

    if use_ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
