"""
Phase 1 (ViT variant): Train ViTDecoder (GLD/MAE style) for RGB reconstruction.

Usage:
    python train_decoder.py \
        --data_csv data/train.csv \
        --video_root /path/to/videos \
        --encoder_ckpt ckpts/streamvggt.pt \
        --token_stats ckpts/token_stats.pt \
        --output_dir ckpts/decoder_vit

The DPTHead variant is in train_decoder_dpt.py.
"""

import argparse
import os
import time

import lpips
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from streamvggt.models.streamvggt import StreamVGGT
from models.vit_decoder import ViTDecoder
from data.video_dataset import SpatialVidDataset, collate_fn
from data.token_utils import (
    DEFAULT_BOUNDARY_LEVEL,
    DPT_LEVELS,
    augment_tokens_for_decoder,
    load_token_stats,
    normalize_tokens,
    strip_special_tokens,
)
from torch.utils.tensorboard import SummaryWriter

from utils.training import EMA, build_optimizer, build_scheduler
from utils.distributed import setup_ddp, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViTDecoder on StreamVGGT tokens")

    # Data
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, default="")
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--annotation_index", type=str, default="")

    # Model checkpoints
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument("--token_stats", type=str, required=True)
    parser.add_argument("--decoder_ckpt", type=str, default="")

    # Output
    parser.add_argument("--output_dir", type=str, default="ckpts/decoder_vit")

    # ViTDecoder architecture
    parser.add_argument("--token_dim", type=int, default=2048)
    parser.add_argument("--decoder_dim", type=int, default=512,
                        help="Internal transformer dimension (input projected 2048->decoder_dim)")
    parser.add_argument("--vit_depth", type=int, default=4, help="Transformer blocks per frame")
    parser.add_argument("--vit_heads", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--output_depth", action="store_true",
                        help="Enable depth head alongside RGB head")
    parser.add_argument("--depth_root", type=str, default="",
                        help="Root directory for depth zip/EXR files")
    parser.add_argument("--depth_weight", type=float, default=0.1,
                        help="Weight for depth L1 loss")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lpips_weight", type=float, default=0.1)
    parser.add_argument("--temporal_weight", type=float, default=0.05)
    parser.add_argument("--grad_weight", type=float, default=0.05)
    parser.add_argument("--token_noise_std", type=float, default=0.02)
    parser.add_argument("--level_dropout", type=float, default=0.15)
    parser.add_argument("--boundary_level", type=int, default=DEFAULT_BOUNDARY_LEVEL)
    parser.add_argument("--boundary_only_prob", type=float, default=0.25)

    # Data loading
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_videos", type=int, default=0)

    # Precision
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def temporal_difference_loss(recon, target):
    if recon.shape[1] < 2:
        return recon.new_zeros(())
    recon_dt = recon[:, 1:] - recon[:, :-1]
    target_dt = target[:, 1:] - target[:, :-1]
    return F.l1_loss(recon_dt, target_dt)


def image_gradient_loss(recon, target):
    recon_dx = recon[..., :, 1:] - recon[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    recon_dy = recon[..., 1:, :] - recon[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(recon_dx, target_dx) + F.l1_loss(recon_dy, target_dy)


def build_datasets(args):
    train_dataset = SpatialVidDataset(
        csv_path=args.data_csv, video_root=args.video_root,
        seq_len=args.seq_len, target_size=args.img_size,
        annotation_index_path=args.annotation_index,
        max_videos=args.max_videos, num_frames_per_video=args.seq_len,
        depth_root=args.depth_root,
    )
    val_dataset = None
    if args.val_csv:
        val_dataset = SpatialVidDataset(
            csv_path=args.val_csv, video_root=args.video_root,
            seq_len=args.seq_len, target_size=args.img_size,
            annotation_index_path=args.annotation_index,
            max_videos=args.max_videos, num_frames_per_video=args.seq_len,
            depth_root=args.depth_root,
        )
    return train_dataset, val_dataset


def train_one_epoch(epoch, encoder, decoder, lpips_model, train_loader,
                    optimizer, scheduler, ema, level_stats, device, dtype,
                    args, global_step, use_ddp=False, writer=None):
    decoder.train()
    encoder.eval()

    output_depth = args.output_depth
    total_loss = total_l1 = total_lpips = total_temporal = total_grad = total_depth = 0.0
    num_batches = 0

    optimizer.zero_grad()
    _decoder_raw = decoder.module if use_ddp else decoder

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        frames = batch["frames"].to(device, dtype=torch.bfloat16)
        B, S = frames.shape[:2]

        with torch.no_grad():
            tokens_list, psi = encoder.aggregator(frames)
            tokens_list = strip_special_tokens(tokens_list, psi)

        tokens_list = normalize_tokens(tokens_list, level_stats)
        tokens_list = augment_tokens_for_decoder(
            tokens_list, levels=DPT_LEVELS,
            boundary_level=args.boundary_level,
            level_dropout=args.level_dropout,
            boundary_only_prob=args.boundary_only_prob,
            token_noise_std=args.token_noise_std,
        )
        tokens_list = [t.to(dtype=torch.float32) for t in tokens_list]

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            if output_depth:
                recon, conf, depth_pred, depth_conf = decoder(
                    tokens_list, images=frames.float(),
                    patch_start_idx=0, frames_chunk_size=S)
            else:
                recon, conf = decoder(tokens_list, images=frames.float(),
                                       patch_start_idx=0, frames_chunk_size=S)
            # recon: [B, S, H, W, 3] -> [B, S, 3, H, W]
            recon = recon.permute(0, 1, 4, 2, 3).contiguous().float()
            target = frames.float()

            l1_loss = F.l1_loss(recon, target)
            recon_lpips = (recon * 2 - 1).reshape(B * S, 3, *recon.shape[-2:])
            target_lpips = (target * 2 - 1).reshape(B * S, 3, *target.shape[-2:])
            lpips_val = lpips_model(recon_lpips, target_lpips).mean()
            temporal_loss = temporal_difference_loss(recon, target)
            grad_loss = image_gradient_loss(recon, target)

            depth_loss = recon.new_zeros(())
            if output_depth and batch.get("depth") is not None:
                depth_pred = depth_pred  # [B, S, H, W]
                depth_target = batch["depth"].to(device=device, dtype=torch.float32)
                if depth_target.shape[-1] != depth_pred.shape[-1]:
                    depth_target = F.interpolate(
                        depth_target.reshape(-1, 1, *depth_target.shape[-2:]),
                        size=depth_pred.shape[-2:], mode="bilinear", align_corners=True,
                    ).reshape_as(depth_pred)
                depth_loss = F.l1_loss(depth_pred, depth_target)

            loss = (l1_loss + args.lpips_weight * lpips_val +
                    args.temporal_weight * temporal_loss +
                    args.grad_weight * grad_loss +
                    args.depth_weight * depth_loss)

        loss = loss / args.accum_steps
        loss.backward()

        total_loss += loss.item() * args.accum_steps
        total_l1 += l1_loss.item()
        total_lpips += lpips_val.item()
        total_temporal += temporal_loss.item()
        total_grad += grad_loss.item()
        total_depth += depth_loss.item()
        num_batches += 1

        if (batch_idx + 1) % args.accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(_decoder_raw)
            optimizer.zero_grad()
            global_step += 1

            if is_main_process() and writer is not None and global_step % 10 == 0:
                writer.add_scalar("step/loss", total_loss / max(num_batches, 1), global_step)
                writer.add_scalar("step/lr", scheduler.get_last_lr()[0], global_step)

        if (batch_idx + 1) % 5 == 0:
            pbar.set_postfix(
                loss=f"{total_loss / max(num_batches, 1):.4f}",
                l1=f"{total_l1 / max(num_batches, 1):.4f}",
                temp=f"{total_temporal / max(num_batches, 1):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

    if num_batches % args.accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(_decoder_raw)
        optimizer.zero_grad()
        global_step += 1

    n = max(num_batches, 1)
    return total_loss / n, total_l1 / n, total_lpips / n, total_temporal / n, total_grad / n, total_depth / n, global_step


@torch.no_grad()
def evaluate(encoder, decoder, eval_loader, level_stats, device, dtype, args):
    decoder.eval()
    encoder.eval()
    total_l1 = 0.0
    num_batches = 0

    for batch in tqdm(eval_loader, desc="  Eval", dynamic_ncols=True):
        frames = batch["frames"].to(device, dtype=torch.bfloat16)
        B, S = frames.shape[:2]

        with torch.no_grad():
            tokens_list, psi = encoder.aggregator(frames)
        tokens_list = strip_special_tokens(tokens_list, psi)
        tokens_list = normalize_tokens(tokens_list, level_stats)
        tokens_list = [t.to(dtype=torch.float32) for t in tokens_list]

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            if args.output_depth:
                recon, conf, _, _ = decoder(tokens_list, images=frames.float(),
                                             patch_start_idx=0, frames_chunk_size=S)
            else:
                recon, conf = decoder(tokens_list, images=frames.float(),
                                       patch_start_idx=0, frames_chunk_size=S)
            recon = recon.permute(0, 1, 4, 2, 3).contiguous().float()
            l1_loss = F.l1_loss(recon, frames.float())

        total_l1 += l1_loss.item()
        num_batches += 1

    return total_l1 / max(num_batches, 1)


def save_checkpoint(path, decoder, optimizer, scheduler, ema, epoch, global_step, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch, "global_step": global_step,
        "model_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "args": vars(args),
    }, path)
    print(f"  Saved checkpoint: {path}")


@torch.no_grad()
def save_recon_sample(decoder, encoder, lpips_model, frames, level_stats, path, device, dtype, args):
    """Save reconstruction PNG + metrics (PSNR, LPIPS, SSIM) alongside checkpoint."""
    import numpy as np
    from PIL import Image

    decoder.eval()
    frames = frames.to(device, dtype=torch.bfloat16)
    B, S = frames.shape[:2]
    display_S = min(S, 4)
    display_B = min(B, 2)

    tokens_list, psi = encoder.aggregator(frames)
    tokens_list = strip_special_tokens(tokens_list, psi)
    tokens_list = normalize_tokens(tokens_list, level_stats)
    tokens_list = [t.to(dtype=torch.float32) for t in tokens_list]

    if args.output_depth:
        recon, _, _, _ = decoder(tokens_list, images=frames.float(), patch_start_idx=0, frames_chunk_size=S)
    else:
        recon, _ = decoder(tokens_list, images=frames.float(), patch_start_idx=0, frames_chunk_size=S)
    recon = recon.permute(0, 1, 4, 2, 3).contiguous().clamp(0, 1).float()
    target = frames.float()

    # ---- Metrics ----
    mse = F.mse_loss(recon, target).item()
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    l1 = F.l1_loss(recon, target).item()

    # LPIPS
    recon_lp = (recon * 2 - 1).reshape(B * S, 3, *recon.shape[-2:])
    target_lp = (target * 2 - 1).reshape(B * S, 3, *target.shape[-2:])
    lpips_val = lpips_model(recon_lp, target_lp).mean().item()

    # SSIM (per-frame avg via skimage)
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        ssim_vals = []
        for b in range(display_B):
            for s in range(display_S):
                orig_np = target[b, s].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                rec_np = recon[b, s].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                ssim_vals.append(ssim_fn(orig_np, rec_np, channel_axis=2, data_range=1.0))
        ssim_val = np.mean(ssim_vals)
    except Exception:
        ssim_val = float('nan')

    # Save metrics
    metrics_path = path.replace('.png', '.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"PSNR:  {psnr:.4f} dB\n")
        f.write(f"LPIPS: {lpips_val:.6f}\n")
        f.write(f"SSIM:  {ssim_val:.4f}\n")
        f.write(f"L1:    {l1:.6f}\n")
    print(f"  recon metrics → PSNR={psnr:.2f}dB  LPIPS={lpips_val:.4f}  SSIM={ssim_val:.4f}")

    # Save image grid
    rows = []
    for b in range(display_B):
        row_frames = []
        for s in range(display_S):
            orig = target[b, s].clamp(0, 1)
            rec = recon[b, s]
            pair = torch.cat([orig, rec], dim=2)
            pair_np = (pair.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            row_frames.append(pair_np)
        rows.append(np.concatenate(row_frames, axis=1))
    grid = np.concatenate(rows, axis=0)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(grid).save(path)
    decoder.train()


def load_checkpoint(path, decoder, optimizer, scheduler, ema):
    ckpt = torch.load(path, map_location="cpu")
    decoder.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    ema.load_state_dict(ckpt["ema_state_dict"])
    print(f"  Loaded checkpoint: {path}")
    return ckpt["epoch"], ckpt["global_step"]


def main():
    args = parse_args()
    setup_seed(args.seed)

    use_ddp, rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if use_ddp else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if is_main_process():
        print(f"Training ViTDecoder on {device} (dtype={dtype})")
        print(f"  DDP: {use_ddp}, world_size: {world_size}")
        print(f"  Architecture: token_dim={args.token_dim}, decoder_dim={args.decoder_dim}, "
              f"depth={args.vit_depth}, heads={args.vit_heads}")
        print(f"  Args: {vars(args)}")

    train_dataset, val_dataset = build_datasets(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if use_ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        ) if use_ddp else None
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            sampler=val_sampler, num_workers=args.num_workers,
            pin_memory=True, collate_fn=collate_fn, drop_last=False,
        )

    steps_per_epoch = (len(train_loader) + args.accum_steps - 1) // args.accum_steps
    total_steps = steps_per_epoch * args.epochs

    if is_main_process():
        print(f"  Train batches: {len(train_loader)}, steps/epoch: {steps_per_epoch}, total_steps: {total_steps}")

    # Load frozen encoder
    if is_main_process():
        print("Loading StreamVGGT encoder...")
    encoder = StreamVGGT(img_size=args.img_size, patch_size=args.patch_size,
                          embed_dim=args.token_dim // 2)
    encoder_state = torch.load(args.encoder_ckpt, map_location="cpu")
    encoder.load_state_dict(encoder_state, strict=False)
    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    level_stats = load_token_stats(args.token_stats, device, dtype=torch.float32)

    # Build ViTDecoder
    if is_main_process():
        print("Building ViTDecoder...")
    decoder = ViTDecoder(
        dim=args.token_dim, decoder_dim=args.decoder_dim,
        num_levels=4, depth=args.vit_depth, num_heads=args.vit_heads,
        patch_size=args.patch_size, img_size=args.img_size, output_dim=3,
        output_depth=args.output_depth,
    ).to(device=device, dtype=torch.float32)

    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lpips_model.parameters():
        p.requires_grad = False

    optimizer = build_optimizer(decoder, lr=args.lr, wd=args.wd)
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)
    ema = EMA(decoder, decay=args.ema_decay)

    start_epoch = 0
    global_step = 0
    if args.resume and args.decoder_ckpt and os.path.exists(args.decoder_ckpt):
        start_epoch, global_step = load_checkpoint(args.decoder_ckpt, decoder, optimizer, scheduler, ema)
        start_epoch += 1
        if is_main_process():
            print(f"Resumed from epoch {start_epoch}, global_step {global_step}")

    if use_ddp:
        decoder = nn.parallel.DistributedDataParallel(
            decoder, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )

    writer = None
    if is_main_process():
        print(f"\nStarting training from epoch {start_epoch} to {args.epochs - 1}")
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

    best_val_l1 = float("inf")

    # Capture fixed frames for per-checkpoint reconstruction snapshots
    recon_frames = None
    for fb in train_loader:
        recon_frames = fb["frames"][:2].clone()
        break

    for epoch in range(start_epoch, args.epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        avg_loss, avg_l1, avg_lpips_val, avg_temporal, avg_grad, avg_depth, global_step = train_one_epoch(
            epoch=epoch, encoder=encoder, decoder=decoder, lpips_model=lpips_model,
            train_loader=train_loader, optimizer=optimizer, scheduler=scheduler,
            ema=ema, level_stats=level_stats, device=device, dtype=dtype,
            args=args, global_step=global_step, use_ddp=use_ddp, writer=writer,
        )
        epoch_time = time.time() - epoch_start

        if is_main_process():
            depth_str = f"  depth={avg_depth:.4f}" if args.output_depth else ""
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"loss={avg_loss:.4f}  l1={avg_l1:.4f}  lpips={avg_lpips_val:.4f}  "
                f"temp={avg_temporal:.4f}  grad={avg_grad:.4f}"
                f"{depth_str}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"time={epoch_time:.1f}s  step={global_step}"
            )
            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, epoch)
                writer.add_scalar("train/l1", avg_l1, epoch)
                writer.add_scalar("train/lpips", avg_lpips_val, epoch)
                writer.add_scalar("train/temporal", avg_temporal, epoch)
                writer.add_scalar("train/grad", avg_grad, epoch)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        if val_loader is not None and (epoch + 1) % args.eval_every == 0:
            val_l1 = evaluate(encoder=encoder, decoder=decoder, eval_loader=val_loader,
                              level_stats=level_stats, device=device, dtype=dtype, args=args)
            if is_main_process():
                print(f"  [Eval] val_l1={val_l1:.4f}")
            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                save_checkpoint(os.path.join(args.output_dir, "decoder_best.pt"),
                                decoder, optimizer, scheduler, ema, epoch, global_step, args)

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(os.path.join(args.output_dir, f"decoder_epoch{epoch + 1}.pt"),
                            decoder, optimizer, scheduler, ema, epoch, global_step, args)
            if is_main_process() and recon_frames is not None:
                raw_decoder = decoder.module if use_ddp else decoder
                save_recon_sample(raw_decoder, encoder, lpips_model, recon_frames, level_stats,
                                  os.path.join(args.output_dir, f"recon_epoch{epoch + 1}.png"),
                                  device, dtype, args)

    if is_main_process():
        save_checkpoint(os.path.join(args.output_dir, "decoder_final.pt"),
                        decoder, optimizer, scheduler, ema, args.epochs - 1, global_step, args)
        if recon_frames is not None:
            raw_decoder = decoder.module if use_ddp else decoder
            save_recon_sample(raw_decoder, encoder, lpips_model, recon_frames, level_stats,
                              os.path.join(args.output_dir, "recon_final.png"), device, dtype, args)
        print(f"\nTraining complete. Final checkpoint: {args.output_dir}/decoder_final.pt")


if __name__ == "__main__":
    main()
