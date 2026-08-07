#!/usr/bin/env python3
"""E4: DPT-style reconstruction probe (compact-latent vs no-bottleneck).

Tests whether the proven VGGT-native DPT decoding (as in 4DLangRecon) works in
our pipeline, and how much a compact-latent bottleneck (needed for diffusion)
costs.

Modes
-----
--bottleneck 0  (control / ceiling on our data)
    frozen VGGT -> DPTHead (output_dim=4, sigmoid) -> RGB.
    Reproduces 4DLangRecon's decoder directly on our SpatialVID data.
    No compact latent. Establishes the reconstruction ceiling with the good
    decoder on our data/distribution.

--bottleneck 1  (production path)
    frozen VGGT -> CompactCompressor -> z [B,S,cdim,g,g]
                -> DPTLatentDecoder -> RGB.
    The compact latent z is what diffusion (Wan adapter) will operate on.

Decision
--------
- bottleneck=0 vs the earlier E1 (~20.6): isolates decoder quality
  (DPT vs the generic ProbeDecoder).
- bottleneck=1 vs bottleneck=0: isolates the cost of the compact bottleneck.
- bottleneck=1 >= 25 PSNR: compact-latent path is viable -> proceed to Wan
  diffusion on z. Otherwise the compression is the bottleneck and the
  tokenizer/latent must be redesigned (less compression / two-stream /
  diffuse on the 37-grid feature directly).

Shares the dataset / DDP / device / eval patterns with probe_feature_recon.py.
"""

import argparse
import json
import os
from contextlib import nullcontext as _nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.heads.dpt_head import DPTHead
from models.dpt_latent_decoder import CompactCompressor, DPTLatentDecoder
from data.video_dataset import SpatialVidDataset, collate_fn
from data.loader_utils import multiprocessing_loader_kwargs
from data.token_utils import strip_special_tokens
from utils.training import (
    EMA, ThroughputMeter, append_metrics, atomic_torch_save,
    build_optimizer, build_scheduler, capture_rng_state, restore_rng_state,
)
from utils.distributed import setup_ddp, is_main_process
from utils.encoder_loader import load_encoder_checkpoint
from utils.device import (
    configure_backend_compatibility, create_grad_scaler, get_device,
    get_device_name, manual_seed_all, resolve_dtype,
)

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
    p = argparse.ArgumentParser(description='E4 DPT reconstruction probe')
    p.add_argument('--bottleneck', type=int, default=1, choices=[0, 1],
                   help='0=DPTHead direct (no compact latent, control), '
                        '1=CompactCompressor + DPTLatentDecoder (production).')
    p.add_argument('--probe_name', type=str, default='e4_dpt_bottleneck1')

    # Data
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--video_root', type=str, required=True)
    p.add_argument('--eval_csv', type=str, default='')
    p.add_argument('--eval_video_root', type=str, default='')
    p.add_argument('--max_videos', type=int, default=0)

    # Encoder / latent
    p.add_argument('--encoder_ckpt', type=str, required=True)
    p.add_argument('--levels', type=int, nargs='+', default=[4, 11, 17, 23])
    p.add_argument('--token_dim', type=int, default=2048)
    p.add_argument('--cdim', type=int, default=256, help='compact latent channels')
    p.add_argument('--latent_grid', type=int, default=18)
    p.add_argument('--dpt_features', type=int, default=256,
                   help='DPT RefineNet working channels')

    # Losses
    p.add_argument('--lambda_l1', type=float, default=1.0)
    p.add_argument('--lambda_lpips', type=float, default=0.1,
                   help='LPIPS weight (4DLangRecon uses 0.1)')

    # Training
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--warmup_steps', type=int, default=200)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--dtype', type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'])
    p.add_argument('--use_checkpoint', type=int, default=1, choices=[0, 1])
    p.add_argument('--frames_chunk_size', type=int, default=4,
                   help='decode frames in chunks (0 = all at once)')

    # Data loading
    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--target_size', type=int, default=518)
    p.add_argument('--num_frames_per_video', type=int, default=8)
    p.add_argument('--max_frame_span', type=int, default=0)
    p.add_argument('--clip_duration_seconds', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--decode_retries', type=int, default=8)

    # IO
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--eval_every', type=int, default=2)
    p.add_argument('--eval_clips', type=int, default=32,
                   help='Number of eval clips; metrics are means over these. '
                        'Single-clip PSNR varies more than the decision gates.')
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--local_rank', type=int, default=0)
    return p.parse_args()


def _run_forward(bottleneck, encoder, compressor, decoder, dpt_head, frames,
                 use_amp, device_type, dtype, chunk, patch_start_idx_holder):
    """Returns pred_rgb [B,S,3,H,W] float and z_flat (or None)."""
    with torch.no_grad():
        tokens_list, psi = encoder(frames)
    patch_start_idx_holder[0] = psi
    ctx = autocast(device_type=device_type, dtype=dtype) if use_amp else _nullcontext()
    if bottleneck == 0:
        # DPTHead consumes raw (unstripped) tokens + patch_start_idx.
        with ctx:
            preds, _ = dpt_head(tokens_list, images=frames,
                                patch_start_idx=psi,
                                frames_chunk_size=(chunk or frames.shape[1]))
        # preds: [B, S, H, W, 3]
        pred_rgb = preds[..., :3].permute(0, 1, 4, 2, 3).contiguous().float()
        return pred_rgb, None
    else:
        stripped = strip_special_tokens(tokens_list, psi)
        with ctx:
            z = compressor(stripped)                  # [B,S,cdim,g,g]
            recon = decoder(z, frames_chunk_size=chunk)  # [B,S,H,W,3]
        pred_rgb = recon[..., :3].permute(0, 1, 4, 2, 3).contiguous().float()
        z_flat = z.flatten(2)  # [B,S,cdim*g*g] for stats
        return pred_rgb, z_flat


class ProbeCore(nn.Module):
    """Single trainable module whose forward covers the whole trainable path.

    DDP must wrap THIS module and every training forward must go through the
    DDP wrapper; calling sub-modules via ``.module`` silently skips the
    gradient all-reduce and multi-GPU training degrades to independent
    single-GPU runs.
    """

    def __init__(self, bottleneck, compressor=None, decoder=None, dpt_head=None):
        super().__init__()
        self.bottleneck = bottleneck
        self.compressor = compressor
        self.decoder = decoder
        self.dpt_head = dpt_head

    def forward(self, encoder, frames, use_amp, device_type, dtype, chunk,
                patch_start_idx_holder):
        return _run_forward(
            self.bottleneck, encoder, self.compressor, self.decoder,
            self.dpt_head, frames, use_amp, device_type, dtype, chunk,
            patch_start_idx_holder)


@torch.no_grad()
def eval_recon(core, encoder, eval_loader, device, out_dir, epoch, dtype,
               device_type, chunk):
    """Evaluate over the whole eval loader; metrics are per-clip means.

    Single-clip PSNR varies by several dB — more than the 2-5 dB gaps the
    probe decision rules discriminate.
    """
    from PIL import Image as PImage
    os.makedirs(out_dir, exist_ok=True)
    was_training = core.training
    core.eval()
    psnrs, l1s, mses, lpips_vals = [], [], [], []
    grid_saved = False
    holder = [0]
    for batch in eval_loader:
        frames = batch['frames'].to(device=device, dtype=dtype)
        pred_rgb, _ = core(
            encoder, frames, use_amp=(dtype != torch.float32),
            device_type=device_type, dtype=dtype, chunk=chunk,
            patch_start_idx_holder=holder)
        recon = pred_rgb.permute(0, 1, 3, 4, 2).clamp(0, 1)  # [B,S,H,W,3]
        orig = frames.permute(0, 1, 3, 4, 2).float().clamp(0, 1)
        if not grid_saved:
            S = recon.shape[1]
            rows = [torch.cat([orig[0, s], recon[0, s]], dim=1) for s in range(S)]
            grid = torch.cat(rows, dim=0)
            PImage.fromarray((grid.float().cpu().numpy() * 255).astype(np.uint8)).save(
                os.path.join(out_dir, f'epoch{epoch:04d}_grid.png'))
            grid_saved = True
        mse = F.mse_loss(recon, orig).item()
        mses.append(mse)
        psnrs.append(-10 * np.log10(max(mse, 1e-10)))
        l1s.append(F.l1_loss(recon, orig).item())
        try:
            lpips_fn = get_lpips(device)
            rn = recon.permute(0, 1, 4, 2, 3).reshape(-1, 3, recon.shape[2], recon.shape[3])
            on = orig.permute(0, 1, 4, 2, 3).reshape(-1, 3, orig.shape[2], orig.shape[3])
            lpips_vals.append(lpips_fn(rn * 2 - 1, on * 2 - 1).mean().item())
        except Exception:
            pass
    core.train(was_training)
    return {
        'psnr': float(np.mean(psnrs)),
        'psnr_std': float(np.std(psnrs)),
        'l1': float(np.mean(l1s)),
        'mse': float(np.mean(mses)),
        'lpips': float(np.mean(lpips_vals)) if lpips_vals else float('nan'),
        'eval_clips': len(psnrs),
    }


def main():
    args = parse_args()
    configure_backend_compatibility()
    use_ddp, rank, local_rank, world_size = setup_ddp()
    device = get_device(local_rank)
    main_process = is_main_process()
    device_type = get_device_name()
    dtype = resolve_dtype(args.dtype)
    use_amp = dtype != torch.float32
    use_scaler = dtype == torch.float16
    scaler = create_grad_scaler(enabled=use_scaler)
    manual_seed_all(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)

    if main_process:
        print(f'E4 probe: bottleneck={args.bottleneck} levels={args.levels} '
              f'cdim={args.cdim} latent_grid={args.latent_grid} '
              f'dpt_features={args.dpt_features}')

    # ---- Frozen encoder ----
    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    load_encoder_checkpoint(encoder, args.encoder_ckpt, verbose=main_process)
    encoder = encoder.to(device=device, dtype=dtype).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    input_grid = args.target_size // 14

    # ---- Trainable modules ----
    compressor = None
    decoder = None
    dpt_head = None
    if args.bottleneck == 0:
        # DPT head kept fp32 (heads run under autocast-off in VGGT); RGB via sigmoid.
        dpt_head = DPTHead(
            dim_in=args.token_dim, patch_size=14, output_dim=4,
            activation='sigmoid', conf_activation='sigmoid',
            features=args.dpt_features)
    else:
        compressor = CompactCompressor(
            levels=args.levels, token_dim=args.token_dim, cdim=args.cdim,
            latent_grid=args.latent_grid, input_grid=input_grid)
        decoder = DPTLatentDecoder(
            cdim=args.cdim, latent_grid=args.latent_grid,
            features=args.dpt_features, img_size=args.target_size,
            use_checkpoint=bool(args.use_checkpoint))
    core = ProbeCore(args.bottleneck, compressor=compressor,
                     decoder=decoder, dpt_head=dpt_head).to(device=device)

    n_params = sum(p.numel() for p in core.parameters())
    if main_process:
        print(f'  Trainable params: {n_params / 1e6:.1f}M')

    ema = EMA(core, decay=args.ema_decay, dtype=torch.float32).to(device)

    if use_ddp:
        # All train forwards go through `model` (the DDP wrapper); calling
        # core directly would silently skip the gradient all-reduce.
        model = nn.parallel.DistributedDataParallel(
            core, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False)
    else:
        model = core

    optimizer = build_optimizer(core, lr=args.lr, wd=args.wd)

    # ---- Dataset ----
    dataset = SpatialVidDataset(
        csv_path=args.csv, video_root=args.video_root,
        seq_len=args.seq_len, target_size=args.target_size,
        max_videos=args.max_videos,
        num_frames_per_video=args.num_frames_per_video,
        max_frame_span=args.max_frame_span,
        clip_duration_seconds=args.clip_duration_seconds,
        decode_retries=args.decode_retries)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if use_ddp else None
    loader_kwargs = multiprocessing_loader_kwargs(args.num_workers)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device_type == 'cuda', drop_last=True, **loader_kwargs)

    steps_per_epoch = (len(dataloader) + args.accum_steps - 1) // args.accum_steps
    total_steps = args.epochs * steps_per_epoch
    warmup = min(args.warmup_steps, max(total_steps - 1, 0))
    scheduler = build_scheduler(optimizer, warmup_steps=warmup, total_steps=max(total_steps, 1))

    # Eval loader (multi-clip; metrics are per-clip means)
    if args.eval_csv:
        eval_dataset = SpatialVidDataset(
            csv_path=args.eval_csv, video_root=args.eval_video_root or args.video_root,
            seq_len=args.seq_len, target_size=args.target_size,
            max_videos=args.eval_clips,
            num_frames_per_video=args.num_frames_per_video,
            max_frame_span=args.max_frame_span,
            clip_duration_seconds=args.clip_duration_seconds,
            decode_retries=args.decode_retries,
            temporal_jitter=False)
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                 num_workers=0, collate_fn=collate_fn)
    else:
        eval_subset = torch.utils.data.Subset(
            dataset, list(range(min(args.eval_clips, len(dataset)))))
        eval_loader = DataLoader(eval_subset, batch_size=1, shuffle=False,
                                 num_workers=0, collate_fn=collate_fn)

    # Resume
    global_step = 0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        core.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema']); ema = ema.to(device)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        global_step = ckpt.get('global_step', 0)
        start_epoch = ckpt.get('epoch', 0) + 1
        restore_rng_state(ckpt.get('rng_state'))
        if main_process:
            print(f'Resumed from {args.resume} at epoch {start_epoch}')

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tb')) if main_process else None
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')
    params = list(core.parameters())
    chunk = args.frames_chunk_size if args.frames_chunk_size > 0 else None

    if main_process:
        print(f'\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch')

    throughput = ThroughputMeter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if use_ddp:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(dataloader, desc=f'ep{epoch}/{args.epochs}',
                    dynamic_ncols=True, disable=not main_process)
        holder = [0]

        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(device=device, dtype=dtype)
            # Latent tokens, not frames, and on every micro-batch: counting only
            # optimizer steps under-reports by accum_steps.
            throughput.update(
                int(frames.shape[0] * frames.shape[1] * args.latent_grid ** 2))
            pred_rgb, z_flat = model(
                encoder, frames, use_amp, device_type, dtype, chunk, holder)
            target_rgb = frames.float().clamp(0, 1)

            l1 = F.l1_loss(pred_rgb, target_rgb)
            lpips_loss = pred_rgb.new_zeros(())
            if args.lambda_lpips > 0:
                try:
                    lpips_fn = get_lpips(device)
                    pf = pred_rgb.reshape(-1, 3, pred_rgb.shape[-2], pred_rgb.shape[-1])
                    tf = target_rgb.reshape(-1, 3, target_rgb.shape[-2], target_rgb.shape[-1])
                    lpips_loss = lpips_fn(pf * 2 - 1, tf * 2 - 1).mean()
                except Exception as exc:
                    if main_process:
                        print(f'  [WARN] LPIPS disabled: {exc}')
            loss = args.lambda_l1 * l1 + args.lambda_lpips * lpips_loss

            scaled = loss / args.accum_steps
            if use_scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            if (batch_idx + 1) % args.accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(core)
                scheduler.step()
                global_step += 1

                if main_process and writer and global_step % args.log_every == 0:
                    m = {
                        'step': global_step, 'epoch': epoch,
                        'probe': args.probe_name, 'bottleneck': args.bottleneck,
                        'train/loss': loss.item(), 'train/l1': l1.item(),
                        'train/lpips': lpips_loss.item(),
                        'train/grad_norm': float(grad_norm),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'DI_throughput': throughput.rate(),
                    }
                    if z_flat is not None:
                        m['train/latent_std'] = z_flat.float().std().item()
                        m['train/latent_mean'] = z_flat.float().mean().item()
                    for k, v in m.items():
                        if k.startswith('train/') or k == 'DI_throughput':
                            writer.add_scalar(k, v, global_step)
                    append_metrics(metrics_path, m)
                if main_process:
                    pbar.set_postfix(loss=f'{loss.item():.4f}', l1=f'{l1.item():.4f}',
                                     lpips=f'{lpips_loss.item():.4f}',
                                     DI=throughput.format())

        if main_process and (epoch + 1) % args.eval_every == 0:
            metrics = eval_recon(
                core, encoder, eval_loader, device,
                os.path.join(args.output_dir, 'samples'),
                epoch, dtype, device_type, chunk)
            metrics.update({'step': global_step, 'epoch': epoch,
                            'probe': args.probe_name, 'bottleneck': args.bottleneck})
            append_metrics(metrics_path, metrics)
            if writer:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        writer.add_scalar(f'eval/{k}', v, global_step)
            print(f'  [eval] ep{epoch}: psnr={metrics["psnr"]:.2f}'
                  f'±{metrics["psnr_std"]:.2f} ({metrics["eval_clips"]} clips) '
                  f'l1={metrics["l1"]:.4f} lpips={metrics.get("lpips", float("nan")):.4f}')

        if main_process and (epoch + 1) % args.save_every == 0:
            payload = {
                'model': core.state_dict(), 'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'global_step': global_step, 'epoch': epoch,
                'rng_state': capture_rng_state(), 'args': vars(args),
            }
            atomic_torch_save(payload, os.path.join(args.output_dir, f'checkpoint_epoch{epoch:04d}.pt'))
            atomic_torch_save(payload, os.path.join(args.output_dir, 'checkpoint_latest.pt'))
            print(f'  Saved checkpoint at epoch {epoch}')

    if main_process:
        print(f'\nDone. Output: {args.output_dir}')
        if writer:
            writer.flush(); writer.close()


if __name__ == '__main__':
    main()
