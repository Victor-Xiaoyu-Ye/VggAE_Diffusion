"""
Phase 1 (DPTHead variant): Train DPTHead decoder (conv multi-scale fusion) for RGB + optional depth.

Usage:
    python train_decoder_dpt.py \
        --data_csv data/train.csv --video_root ... --encoder_ckpt ... \
        --token_stats ckpts/token_stats.pt --output_dir ckpts/decoder_dpt \
        --output_depth --depth_root ... --depth_weight 0.1

The ViT variant is in train_decoder.py.
"""

import argparse
import os
import time

import lpips
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", message=".*Grad strides do not match bucket view strides.*")
from torch.utils.data import DataLoader

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.heads.dpt_head import DPTHead
from data.video_dataset import SpatialVidDataset, collate_fn
from data.token_utils import (
    DEFAULT_BOUNDARY_LEVEL,
    DPT_LEVELS,
    augment_tokens_for_decoder,
    load_token_stats,
    normalize_tokens,
    strip_special_tokens,
)
from utils.training import EMA, build_optimizer, build_scheduler
from utils.distributed import setup_ddp, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPTHead RGB decoder on StreamVGGT tokens")

    # Data
    parser.add_argument("--data_csv", type=str, required=True,
                        help="Path to training CSV with columns: id, video path, num frames")
    parser.add_argument("--val_csv", type=str, default="",
                        help="Path to validation CSV (optional)")
    parser.add_argument("--video_root", type=str, required=True,
                        help="Root directory for video files")
    parser.add_argument("--annotation_index", type=str, default="",
                        help="Path to annotation index JSON for captions")

    # Model checkpoints
    parser.add_argument("--encoder_ckpt", type=str, required=True,
                        help="Path to StreamVGGT encoder checkpoint")
    parser.add_argument("--token_stats", type=str, required=True,
                        help="Path to token statistics (mean/var per DPT level)")
    parser.add_argument("--decoder_ckpt", type=str, default="",
                        help="Path to resume decoder checkpoint (optional)")

    # Output
    parser.add_argument("--output_dir", type=str, default="ckpts/decoder",
                        help="Directory to save checkpoints and logs")

    # Architecture
    parser.add_argument("--dim_in", type=int, default=2048,
                        help="Input dimension to DPTHead (2 * embed_dim)")
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--output_dim", type=int, default=4,
                        help="Output channels (3 RGB + 1 confidence)")
    parser.add_argument("--output_depth", action="store_true",
                        help="Enable depth head alongside RGB head")
    parser.add_argument("--depth_root", type=str, default="",
                        help="Root directory for depth .npy/.pt files")
    parser.add_argument("--depth_weight", type=float, default=0.1,
                        help="Weight for depth L1 loss")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--features", type=int, default=256,
                        help="DPTHead feature channels (256 default, 512 for big)")
    parser.add_argument("--multi_scale", action="store_true",
                        help="Multi-scale L1 supervision at 518/256/128")
    parser.add_argument("--gan", action="store_true",
                        help="Enable GAN loss (DINO discriminator, epoch 6+)")
    parser.add_argument("--multi_layer_mean", action="store_true",
                        help="RAEv2-style: average levels 4,11,17,23 into one representation")
    parser.add_argument("--mlm_boundary_only", action="store_true",
                        help="Multi-layer mean only replaces boundary level, others keep original")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lpips_weight", type=float, default=0.1,
                        help="Weight for LPIPS loss term")
    parser.add_argument("--temporal_weight", type=float, default=0.05,
                        help="Weight for adjacent-frame temporal difference loss")
    parser.add_argument("--grad_weight", type=float, default=0.05,
                        help="Weight for image-gradient loss to reduce patch/grid artifacts")
    parser.add_argument("--token_noise_std", type=float, default=0.02,
                        help="Gaussian noise std added to normalized patch tokens")
    parser.add_argument("--level_dropout", type=float, default=0.15,
                        help="Probability of dropping each DPT level during decoder training")
    parser.add_argument("--boundary_level", type=int, default=DEFAULT_BOUNDARY_LEVEL,
                        help="Boundary DPT level used by the diffusion model")
    parser.add_argument("--boundary_only_prob", type=float, default=0.25,
                        help="Probability of training decoder with only boundary_level tokens")

    # Data loading
    parser.add_argument("--seq_len", type=int, default=8,
                        help="Number of frames per video sample")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Max videos to use (0 = all)")

    # Precision
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp32"],
                        help="Mixed precision dtype for encoder forward pass")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from decoder_ckpt")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For DDP distributed training")

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


def multi_scale_l1(recon, target, scales=[1.0, 0.5, 0.25]):
    """L1 loss at multiple resolutions for better detail preservation."""
    loss = F.l1_loss(recon, target)
    for s in scales[1:]:
        h, w = int(recon.shape[-2] * s), int(recon.shape[-1] * s)
        if h < 16 or w < 16: break
        rs = F.interpolate(recon.reshape(-1, 3, *recon.shape[-2:]),
                           size=(h, w), mode='bilinear', align_corners=True)
        ts = F.interpolate(target.reshape(-1, 3, *target.shape[-2:]),
                           size=(h, w), mode='bilinear', align_corners=True)
        loss = loss + 0.5 * F.l1_loss(rs, ts)
    return loss


def image_gradient_loss(recon, target):
    recon_dx = recon[..., :, 1:] - recon[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    recon_dy = recon[..., 1:, :] - recon[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(recon_dx, target_dx) + F.l1_loss(recon_dy, target_dy)


def build_datasets(args):
    """Build train and optionally val datasets."""
    train_dataset = SpatialVidDataset(
        csv_path=args.data_csv,
        video_root=args.video_root,
        seq_len=args.seq_len,
        target_size=args.img_size,
        annotation_index_path=args.annotation_index,
        max_videos=args.max_videos,
        num_frames_per_video=args.seq_len,
        depth_root=args.depth_root,
    )
    val_dataset = None
    if args.val_csv:
        val_dataset = SpatialVidDataset(
            csv_path=args.val_csv,
            video_root=args.video_root,
            seq_len=args.seq_len,
            target_size=args.img_size,
            annotation_index_path=args.annotation_index,
            max_videos=args.max_videos,
            num_frames_per_video=args.seq_len,
            depth_root=args.depth_root,
        )
    return train_dataset, val_dataset


def train_one_epoch(
    epoch,
    encoder,
    decoder,
    lpips_model,
    train_loader,
    optimizer,
    scheduler,
    ema,
    level_stats,
    device,
    dtype,
    args,
    global_step,
    use_ddp=False,
    writer=None,
    disc=None,
    disc_opt=None,
):
    decoder.train()
    encoder.eval()

    output_depth = args.output_depth
    total_loss = 0.0
    total_l1 = 0.0
    total_lpips = 0.0
    total_temporal = 0.0
    total_grad = 0.0
    total_depth = 0.0
    num_batches = 0

    optimizer.zero_grad()

    _decoder_raw = decoder.module if use_ddp else decoder

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        frames = batch["frames"].to(device, dtype=torch.bfloat16)  # [B, S, 3, H, W]
        B, S = frames.shape[:2]

        # ---- Encoder forward (frozen, bf16) ----
        with torch.no_grad():
            tokens_list, psi = encoder.aggregator(frames)
            tokens_list = strip_special_tokens(tokens_list, psi)

        # ---- Normalize DPT-level tokens ----
        tokens_list = normalize_tokens(tokens_list, level_stats)
        tokens_list = augment_tokens_for_decoder(
            tokens_list,
            levels=DPT_LEVELS,
            boundary_level=args.boundary_level,
            level_dropout=args.level_dropout,
            boundary_only_prob=args.boundary_only_prob,
            token_noise_std=args.token_noise_std,
        )

        # Multi-layer mean (RAEv2): average levels → replace DPT levels
        if args.multi_layer_mean:
            z_mean = (tokens_list[4] + tokens_list[11] + tokens_list[17] + tokens_list[23]) / 4.0
            if args.mlm_boundary_only:
                tokens_list[DEFAULT_BOUNDARY_LEVEL] = z_mean  # only boundary level
            else:
                for lvl in DPT_LEVELS:
                    tokens_list[lvl] = z_mean

        # ---- Cast to fp32 for DPTHead ----
        tokens_list = [t.to(dtype=torch.float32) for t in tokens_list]

        # ---- Decoder forward (fp32, with autocast) ----
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            if output_depth:
                recon, conf, depth_pred, depth_conf = decoder(
                    tokens_list,
                    images=frames.float(),
                    patch_start_idx=0,
                    frames_chunk_size=S,
                )
            else:
                recon, conf = decoder(
                    tokens_list,
                    images=frames.float(),
                    patch_start_idx=0,
                    frames_chunk_size=S,
                )
            # recon: [B, S, H, W, 3] -> [B, S, 3, H, W]
            recon = recon.permute(0, 1, 4, 2, 3).contiguous()  # [B, S, 3, H, W]
            recon = recon.float()

            target = frames.float()  # [B, S, 3, H, W]

            # L1 loss (multi-scale if enabled)
            if args.multi_scale:
                l1_loss = multi_scale_l1(recon, target)
            else:
                l1_loss = F.l1_loss(recon, target)

            # LPIPS loss (input range [-1, 1])
            recon_lpips = recon * 2 - 1
            target_lpips = target * 2 - 1
            # Reshape for LPIPS: [B*S, 3, H, W]
            recon_lpips = recon_lpips.reshape(B * S, 3, recon.shape[-2], recon.shape[-1])
            target_lpips = target_lpips.reshape(B * S, 3, target.shape[-2], target.shape[-1])
            lpips_val = lpips_model(recon_lpips, target_lpips).mean()

            temporal_loss = temporal_difference_loss(recon, target)
            grad_loss = image_gradient_loss(recon, target)

            depth_loss = recon.new_zeros(())
            if output_depth and "depth" in batch and batch["depth"] is not None:
                # depth_pred: [B, S, H, W, 1] from activate_head
                depth_pred = depth_pred.squeeze(-1)  # [B, S, H, W]
                depth_target = batch["depth"].to(device=device, dtype=torch.float32)
                # Interpolate depth to match prediction resolution if needed
                if depth_target.shape[-1] != depth_pred.shape[-1]:
                    depth_target = F.interpolate(
                        depth_target.reshape(-1, 1, depth_target.shape[-2], depth_target.shape[-1]),
                        size=depth_pred.shape[-2:], mode="bilinear", align_corners=True,
                    ).reshape_as(depth_pred)
                depth_loss = F.l1_loss(depth_pred, depth_target)

            loss = (
                l1_loss
                + args.lpips_weight * lpips_val
                + args.temporal_weight * temporal_loss
                + args.grad_weight * grad_loss
                + args.depth_weight * depth_loss
            )

        # Scale for gradient accumulation
        loss = loss / args.accum_steps

        # GAN loss (starts at epoch 6, GLD-style)
        gan_loss = recon.new_zeros(())
        if args.gan and disc is not None and epoch >= 5:
            from models.dino_disc import hinge_d_loss, vanilla_g_loss, diff_augment
            # Discriminator update on real and fake
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                real_aug = diff_augment((target * 2 - 1).reshape(B*S, 3, *target.shape[-2:]))
                fake_aug = diff_augment((recon.detach() * 2 - 1).reshape(B*S, 3, *recon.shape[-2:]))
                logits_real = disc(real_aug)
                logits_fake = disc(fake_aug)
                d_loss = hinge_d_loss(logits_real, logits_fake)
            disc_opt.zero_grad()
            d_loss.backward()
            disc_opt.step()
            # Generator loss
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                fake_for_g = diff_augment((recon * 2 - 1).reshape(B*S, 3, *recon.shape[-2:]))
                logits_fake_g = disc(fake_for_g)
                gan_loss = vanilla_g_loss(logits_fake_g) * 0.1  # adaptive weight simplified
            loss = loss + gan_loss / args.accum_steps

        loss.backward()

        total_loss += loss.item() * args.accum_steps
        total_l1 += l1_loss.item()
        total_lpips += lpips_val.item()
        total_temporal += temporal_loss.item()
        total_grad += grad_loss.item()
        total_depth += depth_loss.item()
        num_batches += 1

        # Gradient accumulation step
        if (batch_idx + 1) % args.accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(_decoder_raw)
            optimizer.zero_grad()
            global_step += 1

            if is_main_process() and writer is not None and global_step % 10 == 0:
                writer.add_scalar("step/loss", total_loss / max(num_batches, 1), global_step)
                writer.add_scalar("step/lr", scheduler.get_last_lr()[0], global_step)

            # Update progress bar
            if (batch_idx + 1) % 5 == 0:
                postfix = {
                    "loss": f"{total_loss / max(num_batches, 1):.4f}",
                    "l1": f"{total_l1 / max(num_batches, 1):.4f}",
                    "temp": f"{total_temporal / max(num_batches, 1):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
                if output_depth:
                    postfix["depth"] = f"{total_depth / max(num_batches, 1):.4f}"
                pbar.set_postfix(**postfix)

    # Handle remaining gradients if batch count is not divisible by accum_steps
    if num_batches % args.accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(_decoder_raw)
        optimizer.zero_grad()
        global_step += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_l1 = total_l1 / max(num_batches, 1)
    avg_lpips = total_lpips / max(num_batches, 1)
    avg_temporal = total_temporal / max(num_batches, 1)
    avg_grad = total_grad / max(num_batches, 1)
    avg_depth = total_depth / max(num_batches, 1)

    return avg_loss, avg_l1, avg_lpips, avg_temporal, avg_grad, avg_depth, global_step


@torch.no_grad()
def evaluate(
    encoder,
    decoder,
    eval_loader,
    level_stats,
    device,
    dtype,
    args,
):
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
        if args.multi_layer_mean:
            z_mean = (tokens_list[4] + tokens_list[11] + tokens_list[17] + tokens_list[23]) / 4.0
            if args.mlm_boundary_only:
                tokens_list[DEFAULT_BOUNDARY_LEVEL] = z_mean
            else:
                for lvl in DPT_LEVELS:
                    tokens_list[lvl] = z_mean
        tokens_list = [t.to(dtype=torch.float32) for t in tokens_list]

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            if args.output_depth:
                recon, conf, depth_pred, depth_conf = decoder(
                    tokens_list,
                    images=frames.float(),
                    patch_start_idx=0,
                    frames_chunk_size=S,
                )
            else:
                recon, conf = decoder(
                    tokens_list,
                    images=frames.float(),
                    patch_start_idx=0,
                    frames_chunk_size=S,
                )
            # recon: [B, S, H, W, 3]
            recon = recon.permute(0, 1, 4, 2, 3).contiguous()
            recon = recon.float()

            target = frames.float()
            l1_loss = F.l1_loss(recon, target)

        total_l1 += l1_loss.item()
        num_batches += 1

    avg_l1 = total_l1 / max(num_batches, 1)
    return avg_l1


def save_checkpoint(path, decoder, optimizer, scheduler, ema, epoch, global_step, args):
    """Save decoder checkpoint with optimizer, scheduler, and EMA states."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = decoder.module if hasattr(decoder, 'module') else decoder
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, path)
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

    mse = F.mse_loss(recon, target).item()
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    l1 = F.l1_loss(recon, target).item()

    recon_lp = (recon * 2 - 1).reshape(B * S, 3, *recon.shape[-2:])
    target_lp = (target * 2 - 1).reshape(B * S, 3, *target.shape[-2:])
    lpips_val = lpips_model(recon_lp, target_lp).mean().item()

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

    metrics_path = path.replace('.png', '.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"PSNR:  {psnr:.4f} dB\n")
        f.write(f"LPIPS: {lpips_val:.6f}\n")
        f.write(f"SSIM:  {ssim_val:.4f}\n")
        f.write(f"L1:    {l1:.6f}\n")
    print(f"  recon metrics → PSNR={psnr:.2f}dB  LPIPS={lpips_val:.4f}  SSIM={ssim_val:.4f}")

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
    """Load decoder checkpoint and restore optimizer, scheduler, EMA states."""
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    decoder.load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    ema.load_state_dict(ckpt["ema_state_dict"])
    print(f"  Loaded checkpoint: {path}")
    return ckpt["epoch"], ckpt["global_step"]


def main():
    args = parse_args()
    setup_seed(args.seed)

    # ---- DDP setup ----
    use_ddp, rank, local_rank, world_size = setup_ddp()
    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Precision dtype ----
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if is_main_process():
        print(f"Training DPTHead decoder on {device} (dtype={dtype})")
        print(f"  DDP: {use_ddp}, world_size: {world_size}")
        print(f"  Args: {vars(args)}")

    # ---- Build datasets & loaders ----
    train_dataset, val_dataset = build_datasets(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if use_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        ) if use_ddp else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

    # ---- Compute total training steps ----
    steps_per_epoch = len(train_loader) // args.accum_steps
    if len(train_loader) % args.accum_steps != 0:
        steps_per_epoch += 1
    total_steps = steps_per_epoch * args.epochs

    if is_main_process():
        print(f"  Train batches: {len(train_loader)}, steps/epoch: {steps_per_epoch}, total_steps: {total_steps}")

    # ---- Load encoder (frozen) ----
    if is_main_process():
        print("Loading StreamVGGT encoder...")
    encoder = StreamVGGT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.dim_in // 2,
    )
    encoder_state = torch.load(args.encoder_ckpt, map_location="cpu")
    encoder.load_state_dict(encoder_state, strict=False)
    encoder = encoder.to(device=device, dtype=torch.bfloat16)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ---- Load token statistics ----
    if is_main_process():
        print("Loading token statistics...")
    level_stats = load_token_stats(args.token_stats, device, dtype=torch.float32)

    # ---- Build DPTHead decoder (fp32) ----
    if is_main_process():
        print("Building DPTHead decoder...")
    decoder = DPTHead(
        dim_in=args.dim_in,
        patch_size=args.patch_size,
        output_dim=args.output_dim,
        activation="sigmoid",
        conf_activation="sigmoid",
        output_depth=args.output_depth,
        features=args.features,
    ).to(device=device, dtype=torch.float32)

    # ---- Build LPIPS model (frozen) ----
    if is_main_process():
        print("Building LPIPS model...")
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad = False

    # ---- GAN discriminator ----
    disc, disc_opt = None, None
    if args.gan:
        if is_main_process():
            print("Building DINO discriminator...")
        from models.dino_disc import DinoDiscriminator
        disc = DinoDiscriminator(device=device)
        disc_opt = torch.optim.AdamW(disc.parameters(), lr=args.lr * 0.5, betas=(0.0, 0.99))

    # ---- Optimizer & scheduler ----
    optimizer = build_optimizer(decoder, lr=args.lr, wd=args.wd)
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)

    # ---- EMA ----
    ema = EMA(decoder, decay=args.ema_decay)

    # ---- Resume from checkpoint ----
    start_epoch = 0
    global_step = 0
    if args.resume and args.decoder_ckpt and os.path.exists(args.decoder_ckpt):
        start_epoch, global_step = load_checkpoint(args.decoder_ckpt, decoder, optimizer, scheduler, ema)
        start_epoch += 1  # continue from next epoch
        if is_main_process():
            print(f"Resumed from epoch {start_epoch}, global_step {global_step}")

    # ---- DDP wrapper ----
    if use_ddp:
        decoder = nn.parallel.DistributedDataParallel(
            decoder, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )

    # ---- Training loop ----
    writer = None
    if is_main_process():
        print(f"\nStarting training from epoch {start_epoch} to {args.epochs - 1}")
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

    recon_frames = None
    for fb in train_loader:
        recon_frames = fb["frames"][:2].clone()
        break

    best_val_l1 = float("inf")

    for epoch in range(start_epoch, args.epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        avg_loss, avg_l1, avg_lpips_val, avg_temporal, avg_grad, avg_depth, global_step = train_one_epoch(
            epoch=epoch,
            encoder=encoder,
            decoder=decoder,
            lpips_model=lpips_model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            level_stats=level_stats,
            device=device,
            dtype=dtype,
            args=args,
            global_step=global_step,
            use_ddp=use_ddp,
            writer=writer,
            disc=disc,
            disc_opt=disc_opt,
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
                if args.output_depth:
                    writer.add_scalar("train/depth", avg_depth, epoch)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # ---- Evaluation ----
        if val_loader is not None and (epoch + 1) % args.eval_every == 0:
            val_l1 = evaluate(
                encoder=encoder,
                decoder=decoder,
                eval_loader=val_loader,
                level_stats=level_stats,
                device=device,
                dtype=dtype,
                args=args,
            )
            if is_main_process():
                print(f"  [Eval] val_l1={val_l1:.4f}")

            # Save best model
            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                best_path = os.path.join(args.output_dir, "decoder_best.pt")
                save_checkpoint(best_path, decoder, optimizer, scheduler, ema, epoch, global_step, args)
                if is_main_process():
                    print(f"  New best val_l1={val_l1:.4f}")

        # ---- Periodic save ----
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"decoder_epoch{epoch + 1}.pt")
            save_checkpoint(save_path, decoder, optimizer, scheduler, ema, epoch, global_step, args)
            if is_main_process() and recon_frames is not None:
                raw_decoder = decoder.module if use_ddp else decoder
                save_recon_sample(raw_decoder, encoder, lpips_model, recon_frames, level_stats,
                                  os.path.join(args.output_dir, f"recon_epoch{epoch + 1}.png"),
                                  device, dtype, args)

    # ---- Final save ----
    if is_main_process():
        final_path = os.path.join(args.output_dir, "decoder_final.pt")
        save_checkpoint(final_path, decoder, optimizer, scheduler, ema, args.epochs - 1, global_step, args)
        if recon_frames is not None:
            raw_decoder = decoder.module if use_ddp else decoder
            save_recon_sample(raw_decoder, encoder, lpips_model, recon_frames, level_stats,
                              os.path.join(args.output_dir, "recon_final.png"), device, dtype, args)
        print(f"\nTraining complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
