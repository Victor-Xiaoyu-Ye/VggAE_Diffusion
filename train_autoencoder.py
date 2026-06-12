#!/usr/bin/env python3
"""Phase 1: Train Generative Tokenizer A + Compact Decoder G jointly.

This creates the generative latent space z_g that Phase 2 (diffusion) operates on.

Training strategy:
  - Freeze StreamVGGT encoder
  - Tokenizer A compresses VGGT features → compact latent z_g
  - Decoder G reconstructs RGB (+ depth) from z_g
  - Noise augmentation on z_g for decoder robustness
  - Multi-loss: L1 + perceptual + gradient + temporal + depth

The resulting z_g should be:
  - Compact enough for diffusion (512-dim × 18×18 = 166K dims per frame)
  - Smooth enough for flow matching (no high-frequency decoder-sensitive details)
  - Rich enough for the decoder to reconstruct quality video
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from streamvggt.models.streamvggt import StreamVGGT
from models.generative_tokenizer import GenerativeTokenizer
from models.compact_decoder import CompactDecoder
from data.video_dataset import SpatialVidDataset, collate_fn
from data.token_utils import strip_special_tokens
from utils.training import (
    EMA,
    append_metrics,
    atomic_torch_save,
    build_optimizer,
    build_scheduler,
    capture_rng_state,
    restore_rng_state,
)
from utils.distributed import setup_ddp, is_main_process
from utils.device import (
    create_grad_scaler,
    get_device,
    get_device_name,
    manual_seed_all,
    resolve_dtype,
)

# Lazy import for LPIPS
_lpips_fn = None


def get_lpips(device):
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
        for p in _lpips_fn.parameters():
            p.requires_grad_(False)
    return _lpips_fn


def parse_args():
    p = argparse.ArgumentParser(description='Train generative autoencoder')

    # Data
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--video_root', type=str, required=True)
    p.add_argument('--max_videos', type=int, default=0)
    p.add_argument('--annotation_index', type=str, default='')
    p.add_argument('--eval_csv', type=str, default='')
    p.add_argument('--eval_video_root', type=str, default='')

    # Checkpoints
    p.add_argument('--encoder_ckpt', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='ckpts/autoencoder/exp-1')
    p.add_argument('--resume', type=str, default='')

    # Model
    p.add_argument('--latent_dim', type=int, default=512)
    p.add_argument('--latent_grid', type=int, default=18)
    p.add_argument('--token_dim', type=int, default=2048)
    p.add_argument('--levels', type=int, nargs='+', default=[4, 11, 17, 23])
    p.add_argument('--decoder_base_dim', type=int, default=384,
                   help='Decoder channel width (384=high quality, 256=speed)')
    p.add_argument('--decoder_num_resblocks', type=int, default=2)
    p.add_argument('--disable_temporal_mixer', action='store_true')
    p.add_argument('--output_depth', action='store_true', default=False)
    p.add_argument('--depth_root', type=str, default='')
    p.add_argument('--lambda_depth', type=float, default=0.1)

    # Noise augmentation
    p.add_argument('--latent_noise_std', type=float, default=0.05,
                   help='Noise std added to z_g during training for decoder robustness')
    p.add_argument('--latent_noise_warmup', type=int, default=1000,
                   help='Linearly ramp noise from 0→latent_noise_std over N steps')
    # Loss weights
    p.add_argument('--lambda_l1', type=float, default=1.0)
    p.add_argument('--lambda_lpips', type=float, default=1.0,
                   help='LPIPS perceptual loss weight')
    p.add_argument('--lambda_grad', type=float, default=0.05)
    p.add_argument('--lambda_temporal', type=float, default=0.05)
    p.add_argument('--lambda_latent_reg', type=float, default=0.01,
                   help='Aggregate channel mean/std regularization toward N(0,1)')

    # Training
    p.add_argument('--batch_size', type=int, default=10)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--warmup_steps', type=int, default=500)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument(
        '--dtype', type=str, default='fp16',
        choices=['fp16', 'bf16', 'fp32'])

    # Data loading
    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--target_size', type=int, default=518)
    p.add_argument('--num_frames_per_video', type=int, default=8)
    p.add_argument('--max_frame_span', type=int, default=0,
                   help='Sample frames within at most this many source frames')
    p.add_argument('--clip_duration_seconds', type=float, default=0.0,
                   help='Fixed clip duration; takes precedence over frame span')
    p.add_argument('--num_workers', type=int, default=4)

    # Eval
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--local_rank', type=int, default=0)

    return p.parse_args()


@torch.no_grad()
def eval_samples(tokenizer, decoder, encoder, eval_frames, device, out_dir,
                 epoch, compute_dtype, device_type, eval_depth=None,
                 eval_depth_valid=None):
    """Save reconstruction comparison images for visual progress tracking."""
    import numpy as np
    from PIL import Image as PImage

    os.makedirs(out_dir, exist_ok=True)
    tokenizer_was_training = tokenizer.training
    decoder_was_training = decoder.training
    tokenizer.eval()
    decoder.eval()

    frames = eval_frames.to(device=device, dtype=compute_dtype)
    tokens_list, psi = encoder(frames)
    tokens_list = strip_special_tokens(tokens_list, psi)

    if compute_dtype != torch.float32:
        with torch.amp.autocast(
                device_type=device_type, dtype=compute_dtype):
            z_g, _ = tokenizer(tokens_list)
            result = decoder(z_g)
    else:
        z_g, _ = tokenizer(tokens_list)
        result = decoder(z_g)

    pred_depth = None
    if decoder.output_depth:
        preds, pred_depth, _, _ = result
    else:
        preds, _ = result

    recon = preds[..., :3].clamp(0, 1)  # [1, S, H, W, 3]
    orig = frames.permute(0, 1, 3, 4, 2).clamp(0, 1)  # [1, S, H, W, 3]

    # Save per-frame comparison: top=original, bottom=reconstructed
    S = recon.shape[1]
    for s in range(S):
        row = torch.cat([orig[0, s], recon[0, s]], dim=1)  # [H, 2W, 3]
        row_np = (row.float().cpu().numpy() * 255).astype(np.uint8)
        PImage.fromarray(row_np).save(os.path.join(out_dir, f'epoch{epoch:04d}_frame{s}.png'))

    # Also save a grid of all frames
    grid_rows = []
    for s in range(S):
        grid_rows.append(torch.cat([orig[0, s], recon[0, s]], dim=1))
    grid = torch.cat(grid_rows, dim=0)
    grid_np = (grid.float().cpu().numpy() * 255).astype(np.uint8)
    PImage.fromarray(grid_np).save(os.path.join(out_dir, f'epoch{epoch:04d}_grid.png'))

    # Compute PSNR
    mse = F.mse_loss(recon, orig).item()
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    l1 = F.l1_loss(recon, orig).item()
    temporal = temporal_consistency_loss(
        recon.permute(0, 1, 4, 2, 3),
        orig.permute(0, 1, 4, 2, 3),
    ).item()
    metrics = {"psnr": psnr, "l1": l1, "temporal": temporal}
    try:
        lpips_fn = get_lpips(device)
        recon_nchw = recon.permute(0, 1, 4, 2, 3).reshape(
            -1, 3, recon.shape[2], recon.shape[3])
        orig_nchw = orig.permute(0, 1, 4, 2, 3).reshape(
            -1, 3, orig.shape[2], orig.shape[3])
        metrics["lpips"] = lpips_fn(
            recon_nchw * 2 - 1, orig_nchw * 2 - 1).mean().item()
    except Exception:
        pass
    if pred_depth is not None and eval_depth is not None:
        target_depth = eval_depth.to(device=device)
        target_depth, depth_mask = normalize_relative_depth(target_depth)
        if eval_depth_valid is not None:
            depth_mask = depth_mask & eval_depth_valid.to(
                device=device).view(-1, 1, 1, 1)
        if depth_mask.any():
            metrics["depth_l1"] = F.l1_loss(
                pred_depth[..., 0].float()[depth_mask],
                target_depth[depth_mask],
            ).item()
    if tokenizer_was_training:
        tokenizer.train()
    if decoder_was_training:
        decoder.train()
    print(
        f'  Eval PSNR: {psnr:.2f} dB | '
        f'L1: {l1:.5f} | temporal: {temporal:.5f}')
    return metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    manual_seed_all(seed)


def image_gradient_loss(pred, target):
    """Spatial gradient consistency loss."""
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def temporal_consistency_loss(pred, target):
    """First-order temporal smoothness loss."""
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    return F.l1_loss(pred_diff, target_diff)


def latent_regularization(z_g):
    """Match aggregate latent channels to zero mean and unit variance."""
    values = z_g.float().reshape(-1, z_g.shape[-1])
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    return mean.square().mean() + (std - 1).square().mean()


def validate_resume_args(saved_args, args):
    for key in (
            'latent_dim', 'latent_grid', 'token_dim', 'levels', 'seq_len',
            'target_size', 'decoder_base_dim', 'decoder_num_resblocks',
            'output_depth', 'disable_temporal_mixer', 'max_frame_span',
            'clip_duration_seconds'):
        if key not in saved_args:
            if key not in (
                    'disable_temporal_mixer', 'max_frame_span',
                    'clip_duration_seconds'):
                continue
            if key == 'disable_temporal_mixer':
                saved = False
            else:
                saved = 0 if key == 'max_frame_span' else 0.0
        else:
            saved = saved_args[key]
        current = getattr(args, key)
        if isinstance(saved, (list, tuple)):
            matches = list(saved) == list(current)
        else:
            matches = saved == current
        if not matches:
            raise ValueError(
                f'Resume mismatch for {key}: '
                f'checkpoint={saved}, current={current}')


def checkpoint_payload(tokenizer, decoder, ema, optimizer, scheduler, scaler,
                       global_step, epoch, args):
    return {
        'checkpoint_version': 2,
        'tokenizer': tokenizer.state_dict(),
        'decoder': decoder.state_dict(),
        'ema': ema.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
        'rng_state': capture_rng_state(),
        'args': vars(args),
    }


def normalize_relative_depth(depth):
    """Normalize log-depth per frame while preserving relative geometry."""
    normalized = torch.zeros_like(depth, dtype=torch.float32)
    valid = torch.isfinite(depth) & (depth > 0)
    flat_depth = depth.float().reshape(-1, depth.shape[-2] * depth.shape[-1])
    flat_valid = valid.reshape_as(flat_depth)
    flat_normalized = normalized.reshape_as(flat_depth)
    for index in range(flat_depth.shape[0]):
        mask = flat_valid[index]
        if mask.sum() < 16:
            flat_valid[index] = False
            continue
        valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
        sample_stride = max(1, valid_indices.numel() // 4096)
        sampled = flat_depth[
            index, valid_indices[::sample_stride]].clamp_min(1e-6).log()
        sampled_cpu = sampled.detach().cpu()
        low = torch.quantile(sampled_cpu, 0.02).to(flat_depth.device)
        high = torch.quantile(sampled_cpu, 0.98).to(flat_depth.device)
        scale = (high - low).clamp_min(1e-6)
        values = flat_depth[index, mask].clamp_min(1e-6).log()
        flat_normalized[index, mask] = ((values - low) / scale).clamp(0, 1)
    return normalized, valid


def main():
    args = parse_args()
    if min(args.log_every, args.save_every, args.eval_every) < 1:
        raise ValueError('log/save/eval intervals must be positive')

    use_ddp, rank, local_rank, world_size = setup_ddp()
    device_type = get_device_name()
    device = get_device(local_rank)
    main_process = is_main_process()

    dtype = resolve_dtype(args.dtype)
    use_amp = dtype != torch.float32
    use_scaler = dtype == torch.float16

    if main_process:
        print(f'=== Phase 1: Generative Autoencoder Training ===')
        print(f'Device: {device}, dtype: {dtype}, DDP: {use_ddp}')
        print(f'Latent: {args.latent_dim}dim × {args.latent_grid}×{args.latent_grid}')
        print(f'Levels: {args.levels}')
        print(f'Output: {args.output_dir}')

    set_seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. Load frozen encoder ----
    if main_process:
        print(f'\n[1/4] Loading frozen VGGT encoder...')
    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    state = torch.load(args.encoder_ckpt, map_location='cpu')
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device=device, dtype=dtype).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # ---- 2. Build tokenizer + decoder ----
    if main_process:
        print(f'[2/4] Building generative tokenizer + decoder...')

    tokenizer = GenerativeTokenizer(
        token_dim=args.token_dim, latent_dim=args.latent_dim,
        latent_grid=args.latent_grid, levels=args.levels,
        seq_len=args.seq_len, input_grid=args.target_size // 14,
        disable_temporal_mixer=args.disable_temporal_mixer,
    ).to(device=device)

    decoder = CompactDecoder(
        latent_dim=args.latent_dim, base_dim=args.decoder_base_dim,
        output_dim=3, output_depth=args.output_depth,
        img_size=args.target_size, latent_grid=args.latent_grid,
        num_resblocks=args.decoder_num_resblocks,
        use_checkpoint=True,
    ).to(device=device)

    total_p = sum(p.numel() for p in tokenizer.parameters()) + \
              sum(p.numel() for p in decoder.parameters())
    if main_process:
        print(f'  Tokenizer: {sum(p.numel() for p in tokenizer.parameters()) / 1e6:.1f}M params')
        print(f'  Decoder:   {sum(p.numel() for p in decoder.parameters()) / 1e6:.1f}M params')
        print(f'  Total:     {total_p / 1e6:.1f}M params')

    # EMA
    ema = EMA(
        nn.ModuleList([tokenizer, decoder]),
        decay=args.ema_decay,
        dtype=torch.float32,
    ).to(device)

    # ---- 3. Dataset ----
    if main_process:
        print(f'[3/4] Building dataset...')
    dataset = SpatialVidDataset(
        csv_path=args.csv, video_root=args.video_root,
        seq_len=args.seq_len, target_size=args.target_size,
        annotation_index_path=args.annotation_index,
        max_videos=args.max_videos,
        num_frames_per_video=args.num_frames_per_video,
        depth_root=args.depth_root,
        max_frame_span=args.max_frame_span,
        clip_duration_seconds=args.clip_duration_seconds,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if use_ddp else None
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device_type == 'cuda', drop_last=True,
    )

    # Capture a fixed held-out sample for periodic reconstruction tracking.
    eval_frames = None
    eval_depth = None
    eval_depth_valid = None
    if main_process:
        eval_csv = args.eval_csv or args.csv
        eval_video_root = args.eval_video_root or args.video_root
        if not args.eval_csv:
            print('  [WARN] --eval_csv not set; periodic PSNR uses training data')
        eval_dataset = SpatialVidDataset(
            csv_path=eval_csv,
            video_root=eval_video_root,
            seq_len=args.seq_len,
            target_size=args.target_size,
            max_videos=1,
            num_frames_per_video=args.num_frames_per_video,
            depth_root=args.depth_root,
            temporal_jitter=False,
            max_frame_span=args.max_frame_span,
            clip_duration_seconds=args.clip_duration_seconds,
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_fn)
        eval_batch = next(iter(eval_loader))
        eval_frames = eval_batch['frames'].clone()
        eval_depth = (
            eval_batch['depth'].clone()
            if eval_batch['depth'] is not None else None)
        eval_depth_valid = eval_batch['depth_valid'].clone()

    # ---- 4. Optimizer ----
    if main_process:
        print(f'[4/4] Building optimizer...')
    params = list(tokenizer.parameters()) + list(decoder.parameters())
    optimizer = build_optimizer(nn.ModuleList([tokenizer, decoder]), lr=args.lr, wd=args.wd)
    steps_per_epoch = (len(dataloader) + args.accum_steps - 1) // args.accum_steps
    total_steps = args.epochs * steps_per_epoch
    if args.warmup_steps >= max(total_steps, 1):
        raise ValueError(
            f'warmup_steps={args.warmup_steps} must be smaller than '
            f'total_steps={total_steps}')
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=max(total_steps, 1))
    scaler = create_grad_scaler(enabled=use_scaler)

    # Resume
    global_step = 0
    start_epoch = 0
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f'Resume checkpoint not found: {args.resume}')
        if main_process:
            print(f'Resuming from {args.resume}')
        ckpt = torch.load(
            args.resume, map_location='cpu', weights_only=False)
        validate_resume_args(ckpt.get('args', {}), args)
        tokenizer.load_state_dict(ckpt['tokenizer'])
        decoder.load_state_dict(ckpt['decoder'])
        ema.load_state_dict(ckpt['ema']); ema = ema.to(device)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        global_step = ckpt.get('global_step', 0)
        start_epoch = ckpt.get('epoch', 0) + 1
        if rank == 0:
            restore_rng_state(ckpt.get('rng_state'))
        else:
            set_seed(args.seed + rank + global_step * 1009)

    if use_ddp:
        tokenizer = nn.parallel.DistributedDataParallel(
            tokenizer, device_ids=[local_rank], output_device=local_rank)
        decoder = nn.parallel.DistributedDataParallel(
            decoder, device_ids=[local_rank], output_device=local_rank)

    writer = None
    if main_process:
        writer = SummaryWriter(
            log_dir=os.path.join(args.output_dir, 'tb'),
            purge_step=global_step if global_step > 0 else None,
        )
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')
    lpips_available = args.lambda_lpips > 0

    if main_process:
        print(f'\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch')

    for epoch in range(start_epoch, args.epochs):
        tokenizer.train()
        decoder.train()
        epoch_loss = 0.0
        num_batches = 0
        if use_ddp:
            sampler.set_epoch(epoch)
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}', dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(device=device, dtype=dtype)

            # ---- Encode ----
            with torch.no_grad():
                tokens_list, psi = encoder(frames)
                tokens_list = strip_special_tokens(tokens_list, psi)

            noise_std = args.latent_noise_std * min(1.0, global_step / max(1, args.latent_noise_warmup))
            if use_amp:
                with autocast(device_type=device_type, dtype=dtype):
                    z_g, z_g_flat = tokenizer(tokens_list)
                    z_g_noisy = (
                        z_g + torch.randn_like(z_g) * noise_std
                        if noise_std > 0 and tokenizer.training else z_g)
                    result = decoder(z_g_noisy)
            else:
                z_g, z_g_flat = tokenizer(tokens_list)
                z_g_noisy = (
                    z_g + torch.randn_like(z_g) * noise_std
                    if noise_std > 0 and tokenizer.training else z_g)
                result = decoder(z_g_noisy)

            if decoder.module.output_depth if use_ddp else decoder.output_depth:
                preds, pred_depth, pred_conf, _ = result
            else:
                preds, _ = result

            # RGB: decoder outputs BHWC [B,S,H,W,3], frames is BCSHW [B,S,3,H,W]
            pred_rgb = preds[..., :3].permute(0, 1, 4, 2, 3).contiguous().float()  # [B,S,3,H,W]
            target_rgb = frames.float().clamp(0, 1)  # already [B,S,3,H,W]

            # ---- Losses ----
            l1 = F.l1_loss(pred_rgb, target_rgb)
            grad = image_gradient_loss(
                pred_rgb.reshape(-1, *pred_rgb.shape[2:]),
                target_rgb.reshape(-1, *target_rgb.shape[2:]),
            )
            temp = temporal_consistency_loss(pred_rgb, target_rgb)
            reg = latent_regularization(z_g_flat.float())

            # LPIPS perceptual loss
            lpips_loss = z_g.new_zeros(())
            if lpips_available:
                try:
                    lpips_fn = get_lpips(device)
                    # LPIPS expects [B, 3, H, W] with values in [-1, 1] or [0, 1]
                    p_flat = pred_rgb.reshape(-1, 3, pred_rgb.shape[-2], pred_rgb.shape[-1])
                    t_flat = target_rgb.reshape(-1, 3, target_rgb.shape[-2], target_rgb.shape[-1])
                    lpips_loss = lpips_fn(p_flat * 2 - 1, t_flat * 2 - 1).mean()
                except Exception as exc:
                    lpips_available = False
                    if main_process:
                        print(f'  [WARN] LPIPS disabled: {exc}')

            loss = (args.lambda_l1 * l1 +
                    args.lambda_lpips * lpips_loss +
                    args.lambda_grad * grad +
                    args.lambda_temporal * temp +
                    args.lambda_latent_reg * reg)

            # Depth loss (if available)
            depth_loss = z_g.new_zeros(())
            if (args.output_depth and pred_depth is not None
                    and batch['depth'] is not None):
                target_depth = batch['depth'].to(device=device)
                target_depth, depth_mask = normalize_relative_depth(target_depth)
                sample_mask = batch['depth_valid'].to(
                    device=device).view(-1, 1, 1, 1)
                depth_mask = depth_mask & sample_mask
                pred_depth_nchw = pred_depth[..., 0].float()
                if depth_mask.any():
                    depth_loss = F.l1_loss(
                        pred_depth_nchw[depth_mask],
                        target_depth[depth_mask],
                    )
                    loss = loss + args.lambda_depth * depth_loss

            # ---- Backward ----
            scaled_loss = loss / args.accum_steps
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (batch_idx + 1) % args.accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params, args.max_grad_norm)
                    scaler.step(optimizer); scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params, args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(nn.ModuleList([
                    tokenizer.module if use_ddp else tokenizer,
                    decoder.module if use_ddp else decoder,
                ]))
                scheduler.step()
                global_step += 1

                if main_process and writer and global_step % args.log_every == 0:
                    train_metrics = {
                        'step': global_step,
                        'epoch': epoch,
                        'train/loss': loss.item(),
                        'train/l1': l1.item(),
                        'train/lpips': lpips_loss.item(),
                        'train/gradient': grad.item(),
                        'train/temporal': temp.item(),
                        'train/depth': depth_loss.item(),
                        'train/latent_regularization': reg.item(),
                        'train/latent_mean': z_g_flat.float().mean().item(),
                        'train/latent_std': z_g_flat.float().std().item(),
                        'train/latent_noise_std': noise_std,
                        'train/grad_norm': float(grad_norm),
                        'train/lr': optimizer.param_groups[0]['lr'],
                    }
                    for name, value in train_metrics.items():
                        if name.startswith('train/'):
                            writer.add_scalar(name, value, global_step)
                    append_metrics(metrics_path, train_metrics)

                pbar.set_postfix(
                    loss=f'{loss.item():.4f}', l1=f'{l1.item():.4f}',
                    grad=f'{grad.item():.4f}', temp=f'{temp.item():.4f}',
                    depth=f'{depth_loss.item():.4f}',
                    noise=f'{noise_std:.3f}',
                )

            epoch_loss += loss.item()
            num_batches += 1

        # Handle incomplete accumulation
        if num_batches % args.accum_steps != 0:
            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(nn.ModuleList([
                tokenizer.module if use_ddp else tokenizer,
                decoder.module if use_ddp else decoder,
            ]))
            scheduler.step()
            global_step += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        if main_process:
            print(f'  Epoch {epoch} | avg_loss={avg_loss:.4f} | global_step={global_step}')
            if writer:
                writer.add_scalar('train/epoch_loss', avg_loss, epoch)
            append_metrics(metrics_path, {
                'step': global_step,
                'epoch': epoch,
                'train/epoch_loss': avg_loss,
            })

        tok = tokenizer.module if use_ddp else tokenizer
        dec = decoder.module if use_ddp else decoder

        save_due = (epoch + 1) % args.save_every == 0
        eval_due = (epoch + 1) % args.eval_every == 0

        if main_process and save_due:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:04d}.pt')
            atomic_torch_save(
                checkpoint_payload(
                    tok, dec, ema, optimizer, scheduler, scaler,
                    global_step, epoch, args),
                save_path,
            )
            print(f'  Saved: {save_path}')

        should_eval = save_due or eval_due or epoch == args.epochs - 1
        if use_ddp and should_eval:
            torch.distributed.barrier()
        if main_process and eval_frames is not None and should_eval:
            try:
                eval_metrics = eval_samples(
                    tok, dec, encoder, eval_frames, device,
                    os.path.join(args.output_dir, 'samples'),
                    epoch, dtype, device_type,
                    eval_depth=eval_depth,
                    eval_depth_valid=eval_depth_valid)
                eval_record = {
                    'step': global_step,
                    'epoch': epoch,
                    **{
                        f'eval/{name}': value
                        for name, value in eval_metrics.items()
                    },
                }
                if writer:
                    for name, value in eval_metrics.items():
                        writer.add_scalar(f'eval/{name}', value, global_step)
                    writer.flush()
                append_metrics(metrics_path, eval_record)
            except Exception as e:
                print(f'  [WARN] Eval sampling failed: {e}')
        if use_ddp and should_eval:
            torch.distributed.barrier()

    # Final save
    if main_process:
        tok = tokenizer.module if use_ddp else tokenizer
        dec = decoder.module if use_ddp else decoder
        final_path = os.path.join(args.output_dir, 'checkpoint_final.pt')
        atomic_torch_save(
            checkpoint_payload(
                tok, dec, ema, optimizer, scheduler, scaler,
                global_step, args.epochs - 1, args),
            final_path,
        )
        print(f'\nDone. Final: {final_path}')
        if writer:
            writer.flush()
            writer.close()

    if use_ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
