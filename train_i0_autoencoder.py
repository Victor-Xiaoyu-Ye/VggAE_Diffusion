#!/usr/bin/env python3
"""Train an I_0-conditioned autoencoder.

The first frame provides appearance while StreamVGGT latents provide geometry for
the complete clip. Training and inference use the same first-frame condition.

Usage: bash scripts/10k/train_i0_autoencoder.sh
"""

import argparse, os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from streamvggt.models.streamvggt import StreamVGGT
from models.generative_tokenizer import GenerativeTokenizer
from models.appearance_cnn import AppearanceCNN
from models.i0_decoder import (
    I0ConditionalDecoder,
    initialize_i0_decoder_from_compact,
    load_i0_decoder_state_dict,
)
from data.video_dataset import SpatialVidDataset, collate_fn
from data.token_utils import strip_special_tokens
from utils.training import (
    EMA,
    append_metrics,
    atomic_torch_save,
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

_lpips_fn = None
def get_lpips(device):
    global _lpips_fn
    if _lpips_fn is None:
        import lpips; _lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
        for p in _lpips_fn.parameters(): p.requires_grad_(False)
    return _lpips_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--video_root', type=str, required=True)
    p.add_argument('--encoder_ckpt', type=str, required=True)
    p.add_argument('--autoencoder_ckpt', type=str, required=True,
                   help='exp-1-big checkpoint for frozen Tokenizer A')
    p.add_argument('--output_dir', type=str, default='ckpts/i0_autoencoder/exp-1')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--max_videos', type=int, default=0)
    p.add_argument('--eval_csv', type=str, default='')
    p.add_argument('--eval_video_root', type=str, default='')
    p.add_argument('--latent_dim', type=int, default=512); p.add_argument('--latent_grid', type=int, default=18)
    p.add_argument('--decoder_base_dim', type=int, default=384)
    p.add_argument('--decoder_num_resblocks', type=int, default=2)
    p.add_argument('--epochs', type=int, default=50); p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--accum_steps', type=int, default=4); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-2); p.add_argument('--warmup_steps', type=int, default=500)
    p.add_argument(
        '--pretrained_lr_scale', type=float, default=0.1,
        help='Learning-rate multiplier for decoder blocks inherited from the geometry AE')
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--lambda_lpips', type=float, default=1.0)
    p.add_argument(
        '--dtype', type=str, default='fp16',
        choices=['fp16', 'bf16', 'fp32'])
    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--target_size', type=int, default=518); p.add_argument('--num_workers', type=int, default=2)
    p.add_argument(
        '--decode_retries', type=int, default=8,
        help='Number of deterministic replacement videos tried after a decode failure')
    p.add_argument('--max_frame_span', type=int, default=0)
    p.add_argument('--clip_duration_seconds', type=float, default=0.0)
    p.add_argument('--disable_temporal_jitter', action='store_true')
    p.add_argument(
        '--disable_temporal_mixer', action='store_true',
        help='Bypass TemporalMixer in frozen Tokenizer A to preserve I0 latent contract')
    p.add_argument('--save_every', type=int, default=5); p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--local_rank', type=int, default=0)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    manual_seed_all(seed)


@torch.no_grad()
def eval_samples(enc, tok, dec, app_cnn, eval_video, device, out_dir, epoch,
                 compute_dtype, device_type):
    """Save reconstruction: I_0→I_0 (self), I_0→I_t (cross-frame), for multiple t."""
    os.makedirs(out_dir, exist_ok=True)
    decoder_was_training = dec.training
    app_was_training = app_cnn.training
    dec.eval(); app_cnn.eval()
    frames = eval_video.to(device, dtype=compute_dtype)
    B, S = frames.shape[:2]

    # Get z_geo for all frames
    tl, psi = enc(frames); tl = strip_special_tokens(tl, psi)
    z_g, _ = tok(tl)  # [1, S, 18, 18, 512]
    i0_tokens, i0_psi = enc(frames[:, :1])
    i0_tokens = strip_special_tokens(i0_tokens, i0_psi)
    i0_z, _ = tok(i0_tokens)
    z_g = z_g.clone()
    z_g[:, 0] = i0_z[:, 0]

    # I_0 features
    I_0 = frames[:, 0:1, :, :, :]  # [1, 1, 3, H, W]
    I_0_feats = app_cnn(I_0.reshape(1, 3, 518, 518))

    # Decode all S frames conditioned on I_0
    with autocast(
            device_type=device_type, dtype=compute_dtype,
            enabled=compute_dtype != torch.float32):
        result = dec(z_g, I_0_feats)
    preds = result[0] if isinstance(result, tuple) else result
    recon = preds[..., :3].clamp(0, 1)  # [1, S, 518, 518, 3]
    orig = frames.permute(0, 1, 3, 4, 2).clamp(0, 1)  # [1, S, 518, 518, 3]

    # Save I_0 + middle/last frame comparison
    for t_idx in [0, S // 2, S - 1]:
        row = torch.cat([orig[0, t_idx].cpu(), recon[0, t_idx].cpu()], dim=1)
        row_np = (row.numpy() * 255).astype(np.uint8)
        Image.fromarray(row_np).save(os.path.join(out_dir, f'epoch{epoch:04d}_t{t_idx}.png'))

    mse = F.mse_loss(recon, orig).item()
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    l1 = F.l1_loss(recon, orig).item()
    temporal = F.l1_loss(
        recon[:, 1:] - recon[:, :-1],
        orig[:, 1:] - orig[:, :-1],
    ).item() if S > 1 else 0.0
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
    if decoder_was_training:
        dec.train()
    if app_was_training:
        app_cnn.train()
    print(
        f'  Eval PSNR: {psnr:.2f} dB | '
        f'L1: {l1:.5f} | temporal: {temporal:.5f}')
    return metrics


def validate_resume_args(saved_args, args):
    for key in (
            'latent_dim', 'latent_grid', 'seq_len', 'target_size',
            'decoder_base_dim', 'decoder_num_resblocks',
            'pretrained_lr_scale', 'max_frame_span',
            'clip_duration_seconds', 'disable_temporal_mixer'):
        if key in saved_args:
            saved = saved_args[key]
        elif key in (
                'disable_temporal_mixer', 'max_frame_span',
                'clip_duration_seconds'):
            if key == 'disable_temporal_mixer':
                saved = False
            else:
                saved = 0 if key == 'max_frame_span' else 0.0
        else:
            continue
        if saved != getattr(args, key):
            raise ValueError(
                f'Resume mismatch for {key}: '
                f'checkpoint={saved}, current={getattr(args, key)}')


def checkpoint_payload(app_cnn, decoder, ema, optimizer, scheduler, scaler,
                       global_step, epoch, args):
    return {
        'checkpoint_version': 2,
        'app_cnn': app_cnn.state_dict(),
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


def build_i0_optimizer(app_cnn, decoder, lr, wd, pretrained_lr_scale):
    if pretrained_lr_scale < 0:
        raise ValueError('--pretrained_lr_scale must be non-negative')
    adapter_prefixes = (
        'cross_attn0.', 'cross_attn1.', 'spade2.', 'app144_proj.')
    groups = {
        ('adapter', 'decay'): [],
        ('adapter', 'no_decay'): [],
        ('pretrained', 'decay'): [],
        ('pretrained', 'no_decay'): [],
    }

    named_parameters = [
        (f'app_cnn.{name}', parameter, True)
        for name, parameter in app_cnn.named_parameters()
    ]
    named_parameters.extend(
        (f'decoder.{name}', parameter, name.startswith(adapter_prefixes))
        for name, parameter in decoder.named_parameters()
    )
    for name, parameter, is_adapter in named_parameters:
        if not parameter.requires_grad:
            continue
        decay_group = (
            'no_decay'
            if parameter.ndim < 2 or 'norm' in name or 'bias' in name
            else 'decay'
        )
        groups[
            ('adapter' if is_adapter else 'pretrained', decay_group)
        ].append(parameter)

    optimizer_groups = []
    for role in ('adapter', 'pretrained'):
        group_lr = lr if role == 'adapter' else lr * pretrained_lr_scale
        for decay_group in ('decay', 'no_decay'):
            parameters = groups[(role, decay_group)]
            if not parameters:
                continue
            optimizer_groups.append({
                'params': parameters,
                'lr': group_lr,
                'initial_lr': group_lr,
                'weight_decay': wd if decay_group == 'decay' else 0.0,
                'group_name': f'{role}_{decay_group}',
            })
    return torch.optim.AdamW(
        optimizer_groups, betas=(0.9, 0.95), eps=1e-8)


def main():
    args = parse_args()
    if min(args.log_every, args.save_every, args.eval_every) < 1:
        raise ValueError('log/save/eval intervals must be positive')
    use_ddp, rank, local_rank, world_size = setup_ddp()
    device_type = get_device_name()
    device = get_device(local_rank)
    main_process = is_main_process()
    dtype = resolve_dtype(args.dtype)
    use_scaler = dtype == torch.float16

    if main_process:
        print(f'=== I_0-Conditioned Autoencoder Training ===')
        print(f'Device: {device}, Dtype: {dtype}, DDP: {use_ddp}')

    set_seed(args.seed + rank); os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load frozen encoder + tokenizer ----
    if main_process: print('[1/4] Loading frozen StreamVGGT + Tokenizer A...')
    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    load_info = encoder.load_state_dict(
        torch.load(args.encoder_ckpt, map_location='cpu'), strict=False)
    if main_process and (load_info.missing_keys or load_info.unexpected_keys):
        print(f'  Encoder checkpoint mismatch: missing={len(load_info.missing_keys)}, '
              f'unexpected={len(load_info.unexpected_keys)}')
    encoder = encoder.to(device, dtype=dtype).eval()
    for p in encoder.parameters(): p.requires_grad_(False)

    tokenizer = GenerativeTokenizer(levels=[4,11,17,23], seq_len=args.seq_len, input_grid=37,
                                     latent_dim=args.latent_dim, latent_grid=args.latent_grid).to(device)
    ae_ckpt = torch.load(
        args.autoencoder_ckpt, map_location='cpu', weights_only=False)
    tokenizer.load_state_dict(ae_ckpt['tokenizer']); tokenizer.eval()
    tokenizer.disable_temporal_mixer = (
        args.disable_temporal_mixer
        or bool(ae_ckpt.get('args', {}).get('disable_temporal_mixer', False))
    )
    if tokenizer.disable_temporal_mixer:
        if main_process:
            print('  TemporalMixer bypassed for I0 latent contract')
    for p in tokenizer.parameters(): p.requires_grad_(False)

    # ---- Build Appearance CNN + I_0 Decoder ----
    if main_process: print('[2/4] Building Appearance CNN + I_0 Conditional Decoder...')
    app_cnn = AppearanceCNN().to(device=device)
    decoder = I0ConditionalDecoder(
        latent_dim=args.latent_dim, base_dim=args.decoder_base_dim, img_size=args.target_size,
        latent_grid=args.latent_grid, num_resblocks=args.decoder_num_resblocks, use_checkpoint=True,
    ).to(device)
    initialized_keys = initialize_i0_decoder_from_compact(
        decoder, ae_ckpt['decoder'])
    if main_process:
        print(
            f'  Initialized {initialized_keys} shared decoder tensors '
            'from geometry autoencoder.')
    tp = sum(p.numel() for p in app_cnn.parameters()) + sum(p.numel() for p in decoder.parameters())
    if main_process: print(f'  AppCNN: {sum(p.numel() for p in app_cnn.parameters())/1e6:.1f}M, Decoder: {sum(p.numel() for p in decoder.parameters())/1e6:.1f}M, Total: {tp/1e6:.1f}M')

    ema_modules = nn.ModuleList([app_cnn, decoder])
    ema = EMA(
        ema_modules, decay=args.ema_decay, dtype=torch.float32).to(device)

    # ---- Dataset ----
    if main_process: print('[3/4] Building dataset...')
    dataset = SpatialVidDataset(csv_path=args.csv, video_root=args.video_root, seq_len=args.seq_len,
                                 target_size=args.target_size, max_videos=args.max_videos,
                                 num_frames_per_video=args.seq_len,
                                 temporal_jitter=not args.disable_temporal_jitter,
                                 max_frame_span=args.max_frame_span,
                                 clip_duration_seconds=args.clip_duration_seconds,
                                 decode_retries=args.decode_retries)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if use_ddp else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        pin_memory=device_type == 'cuda', drop_last=True)

    # Eval video
    eval_video = None
    if main_process:
        eval_csv = args.eval_csv or args.csv
        eval_video_root = args.eval_video_root or args.video_root
        if not args.eval_csv:
            print('  [WARN] --eval_csv not set; periodic PSNR uses training data')
        eval_dataset = SpatialVidDataset(
            csv_path=eval_csv, video_root=eval_video_root,
            seq_len=args.seq_len, target_size=args.target_size,
            max_videos=1, num_frames_per_video=args.seq_len,
            temporal_jitter=False, max_frame_span=args.max_frame_span,
            clip_duration_seconds=args.clip_duration_seconds)
        eval_loader = DataLoader(
            eval_dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_fn)
        eval_video = next(iter(eval_loader))['frames'].clone()

    # ---- Optimizer ----
    if main_process: print('[4/4] Building optimizer...')
    optimizer = build_i0_optimizer(
        app_cnn, decoder, args.lr, args.wd, args.pretrained_lr_scale)
    # Zero-LR pretrained groups remain in the optimizer for a stable resume
    # contract, but must not dilute adapter gradients during global clipping.
    trainable = [
        parameter
        for group in optimizer.param_groups
        if group['initial_lr'] > 0
        for parameter in group['params']
    ]
    if not trainable:
        raise ValueError('No parameters have a positive learning rate')
    steps_per_epoch = (len(loader) + args.accum_steps - 1) // args.accum_steps
    total_steps = args.epochs * steps_per_epoch
    if args.warmup_steps >= max(total_steps, 1):
        raise ValueError(
            f'warmup_steps={args.warmup_steps} must be smaller than '
            f'total_steps={total_steps}')
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=max(total_steps, 1))
    scaler = create_grad_scaler(enabled=use_scaler)

    global_step, start_epoch = 0, 0
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f'Resume checkpoint not found: {args.resume}')
        if main_process: print(f'Resuming from {args.resume}')
        ckpt = torch.load(
            args.resume, map_location='cpu', weights_only=False)
        validate_resume_args(ckpt.get('args', {}), args)
        app_cnn.load_state_dict(ckpt['app_cnn'])
        load_i0_decoder_state_dict(decoder, ckpt['decoder'])
        ema.load_state_dict(ckpt['ema']); ema = ema.to(device)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        global_step, start_epoch = ckpt.get('global_step', 0), ckpt.get('epoch', 0) + 1
        if rank == 0:
            restore_rng_state(ckpt.get('rng_state'))
        else:
            set_seed(args.seed + rank + global_step * 1009)

    if use_ddp:
        app_cnn = nn.parallel.DistributedDataParallel(app_cnn, device_ids=[local_rank], output_device=local_rank)
        decoder = nn.parallel.DistributedDataParallel(decoder, device_ids=[local_rank], output_device=local_rank)

    writer = (
        SummaryWriter(
            log_dir=os.path.join(args.output_dir, 'tb'),
            purge_step=global_step if global_step > 0 else None,
        )
        if main_process else None
    )
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')
    lpips_available = args.lambda_lpips > 0
    if main_process: print(f'\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch')

    for epoch in range(start_epoch, args.epochs):
        app_cnn.train(); decoder.train()
        epoch_loss, n_batches, epoch_decode_replacements = 0.0, 0, 0
        if use_ddp: sampler.set_epoch(epoch)
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}', dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            epoch_decode_replacements += batch.get(
                'decode_replacements', 0)
            frames = batch['frames'].to(device, dtype=dtype)
            # Keep the conditioning distribution identical at train and inference.
            I_A = frames[:, 0:1]  # [B, 1, 3, H, W]
            I_B_frames = frames  # all frames for geometry

            # ---- Encode geometry from I_B (all frames) ----
            with autocast(
                    device_type=device_type, dtype=dtype,
                    enabled=dtype != torch.float32):
                with torch.no_grad():
                    tl, psi = encoder(I_B_frames)
                    tl = strip_special_tokens(tl, psi)
                    z_g, _ = tokenizer(tl)
                    i0_tokens, i0_psi = encoder(I_A)
                    i0_tokens = strip_special_tokens(i0_tokens, i0_psi)
                    i0_z, _ = tokenizer(i0_tokens)
                    z_g = z_g.clone()
                    z_g[:, 0] = i0_z[:, 0]
                # Call DDP wrappers so gradients synchronize across ranks.
                I_0_feats = app_cnn(I_A[:, 0])
                result = decoder(z_g, I_0_feats)
            preds = result[0] if isinstance(result, tuple) else result

            pred_rgb = preds[..., :3].permute(0, 1, 4, 2, 3).float()
            target = I_B_frames.float().clamp(0, 1)

            l1 = F.l1_loss(pred_rgb, target)
            lpips_loss = l1.new_zeros(())
            if lpips_available:
                try:
                    lp = get_lpips(device)
                    pred_lpips = pred_rgb.reshape(-1, 3, *pred_rgb.shape[-2:]) * 2 - 1
                    target_lpips = target.reshape(-1, 3, *target.shape[-2:]) * 2 - 1
                    lpips_loss = lp(pred_lpips, target_lpips).mean()
                except Exception as exc:
                    lpips_available = False
                    if main_process:
                        print(f'  [WARN] LPIPS disabled: {exc}')

            loss = l1 + args.lambda_lpips * lpips_loss
            scaled_loss = loss / args.accum_steps
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            epoch_loss += loss.item(); n_batches += 1
            pbar.set_postfix(l1=f'{l1.item():.4f}', lpips=f'{lpips_loss.item():.4f}')

            if (batch_idx + 1) % args.accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, args.max_grad_norm)
                if use_scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(ema_modules); scheduler.step(); global_step += 1
                if main_process and global_step % args.log_every == 0:
                    train_metrics = {
                        'step': global_step,
                        'epoch': epoch,
                        'train/loss': loss.item(),
                        'train/l1': l1.item(),
                        'train/lpips': lpips_loss.item(),
                        'train/grad_norm': float(grad_norm),
                        'train/lr_adapter': next(
                            group['lr'] for group in optimizer.param_groups
                            if group['group_name'].startswith('adapter_')),
                        'train/lr_pretrained': next(
                            group['lr'] for group in optimizer.param_groups
                            if group['group_name'].startswith('pretrained_')),
                    }
                    if writer:
                        for name, value in train_metrics.items():
                            if name.startswith('train/'):
                                writer.add_scalar(name, value, global_step)
                    append_metrics(metrics_path, train_metrics)

        # End of epoch
        if n_batches % args.accum_steps != 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            if use_scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(ema_modules); scheduler.step(); global_step += 1

        decode_replacements = torch.tensor(
            epoch_decode_replacements, device=device, dtype=torch.long)
        if use_ddp:
            torch.distributed.all_reduce(
                decode_replacements, op=torch.distributed.ReduceOp.SUM)
        total_decode_replacements = int(decode_replacements.item())
        avg_loss = epoch_loss / max(n_batches, 1)
        if main_process:
            print(
                f'  Epoch {epoch} | avg_loss={avg_loss:.4f} | '
                f'decode_replacements={total_decode_replacements} | '
                f'step={global_step}')
            writer.add_scalar('train/epoch_loss', avg_loss, epoch)
            writer.add_scalar(
                'data/decode_replacements',
                total_decode_replacements, epoch)
            append_metrics(metrics_path, {
                'step': global_step,
                'epoch': epoch,
                'train/epoch_loss': avg_loss,
                'data/decode_replacements': total_decode_replacements,
            })

        ac = app_cnn.module if use_ddp else app_cnn
        dc = decoder.module if use_ddp else decoder
        save_due = (epoch + 1) % args.save_every == 0
        eval_due = (epoch + 1) % args.eval_every == 0
        if main_process and save_due:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:04d}.pt')
            atomic_torch_save(
                checkpoint_payload(
                    ac, dc, ema, optimizer, scheduler, scaler,
                    global_step, epoch, args),
                save_path,
            )
            print(f'  Saved: {save_path}')
        should_eval = save_due or eval_due or epoch == args.epochs - 1
        if use_ddp and should_eval:
            torch.distributed.barrier()
        if main_process and eval_video is not None and should_eval:
            try:
                eval_metrics = eval_samples(
                    encoder, tokenizer, dc, ac, eval_video, device,
                    os.path.join(args.output_dir, 'samples'),
                    epoch, dtype, device_type)
                if writer:
                    for name, value in eval_metrics.items():
                        writer.add_scalar(f'eval/{name}', value, global_step)
                    writer.flush()
                append_metrics(metrics_path, {
                    'step': global_step,
                    'epoch': epoch,
                    **{
                        f'eval/{name}': value
                        for name, value in eval_metrics.items()
                    },
                })
            except Exception as e: print(f'  [WARN] Eval: {e}')
        if use_ddp and should_eval:
            torch.distributed.barrier()

    if main_process:
        ac = app_cnn.module if use_ddp else app_cnn
        dc = decoder.module if use_ddp else decoder
        atomic_torch_save(
            checkpoint_payload(
                ac, dc, ema, optimizer, scheduler, scaler,
                global_step, args.epochs - 1, args),
            os.path.join(args.output_dir, 'checkpoint_final.pt'),
        )
        print(f'\nDone.')
        if writer:
            writer.flush()
            writer.close()
    if use_ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__': main()
