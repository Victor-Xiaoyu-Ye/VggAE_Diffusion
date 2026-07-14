#!/usr/bin/env python3
"""E5: Dual-stream reconstruction probe (z_geo + z_tex).

Hypothesis (from E1/E4)
-----------------------
Frozen StreamVGGT features plateau at ~20 PSNR on SpatialVID. Appearance
high-frequency is missing from the geometry stream. An explicit per-frame
TextureEncoder(RGB)->z_tex must supply it. Decoder reconstructs from
(z_geo, z_tex) only — no RGB skip — so both streams remain diffusion targets.

Modes
-----
--tex_mode oracle (default)
    z_tex = TextureEncoder(frames). This is the reconstruction CEILING with
    an honest compact appearance latent. Target: PSNR >= 25 (gate for Wan).

--tex_mode zero
    z_tex = zeros. Isolates how much the geo stream alone can do with the
    DualStreamDecoder capacity (should land near E1/E4 ~20 if capacity is
    not the issue).

--tex_mode tex_only
    z_geo = zeros. Proves z_geo is load-bearing: if tex_only ~= oracle, the
    decoder architecturally bypasses geometry (reviewer question #1).

Key ablation flags (2026-07 survey driven)
------------------------------------------
--tex_pack s2d       space-to-depth packing (avg-pool low-passes the high
                     frequency z_tex exists to carry)
--tex_reg_mode match_geo
                     SVG-style: align z_tex channel stats to z_geo's instead
                     of N(0,1) — keeps both streams on one distribution family
                     (diffusability + anti-shortcut)
--lambda_feat > 0    VGGT feature-consistency loss (MIRA P-DINO analogue):
                     re-encode k sampled pred frames through the frozen
                     encoder, L1 on the probe levels vs re-encoded target

Decision
--------
- oracle >= 25 PSNR: dual-stream reconstruction is viable; proceed to train
  diffusion on (z_geo, z_tex) or z_tex conditioned on z_geo.
- oracle still ~20: TextureEncoder capacity / packing is insufficient;
  raise tex_dim / base_ch or keep higher-res tex grid.
- zero << oracle: confirms appearance is carried by z_tex (expected).
- tex_only << oracle: confirms geometry is load-bearing (required).
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext as _nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from streamvggt.models.streamvggt import StreamVGGT
from models.dpt_latent_decoder import CompactCompressor
from models.texture_encoder import TextureEncoder
from models.dual_stream_decoder import DualStreamDecoder
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
    p = argparse.ArgumentParser(description='E5 dual-stream texture recon probe')
    p.add_argument('--tex_mode', type=str, default='oracle',
                   choices=['oracle', 'zero', 'tex_only'],
                   help='oracle=TextureEncoder(RGB); zero=geo-only ablation; '
                        'tex_only=z_geo zeroed (proves geo is load-bearing)')
    p.add_argument('--probe_name', type=str, default='e5_dual_stream_oracle')

    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--video_root', type=str, required=True)
    p.add_argument('--eval_csv', type=str, default='')
    p.add_argument('--eval_video_root', type=str, default='')
    p.add_argument('--max_videos', type=int, default=0)

    p.add_argument('--encoder_ckpt', type=str, required=True)
    p.add_argument('--levels', type=int, nargs='+', default=[4, 11, 17, 23])
    p.add_argument('--token_dim', type=int, default=2048)
    p.add_argument('--geo_dim', type=int, default=256)
    p.add_argument('--tex_dim', type=int, default=256)
    p.add_argument('--tex_base_ch', type=int, default=64)
    p.add_argument('--tex_pack', type=str, default='avgpool',
                   choices=['avgpool', 's2d'],
                   help='z_tex multi-scale packing: avgpool (legacy, '
                        'low-passes high freq) or s2d (space-to-depth, '
                        'lossless; resizes input to 32*grid internally)')
    p.add_argument('--latent_grid', type=int, default=18)
    p.add_argument('--decoder_base_dim', type=int, default=384)

    p.add_argument('--lambda_l1', type=float, default=1.0)
    p.add_argument('--lambda_lpips', type=float, default=0.1)
    p.add_argument('--lambda_feat', type=float, default=0.0,
                   help='VGGT feature-consistency loss weight (MIRA P-DINO '
                        'analogue): re-encode sampled pred frames through '
                        'the frozen encoder, L1 vs target features. '
                        '0 disables (no extra encoder forward).')
    p.add_argument('--feat_frames', type=int, default=2,
                   help='frames per clip re-encoded for the feature loss')
    p.add_argument('--lambda_tex_reg', type=float, default=0.01,
                   help='z_tex channel-stat regularizer weight')
    p.add_argument('--tex_reg_mode', type=str, default='n01',
                   choices=['n01', 'match_geo'],
                   help='n01: pull z_tex channel stats to N(0,1). '
                        'match_geo (SVG-style): align z_tex channel stats '
                        'to the batch stats of z_geo')

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
    p.add_argument('--frames_chunk_size', type=int, default=4)

    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--target_size', type=int, default=518)
    p.add_argument('--num_frames_per_video', type=int, default=8)
    p.add_argument('--max_frame_span', type=int, default=0)
    p.add_argument('--clip_duration_seconds', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--decode_retries', type=int, default=8)

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


def encode_streams(encoder, compressor, tex_encoder, frames, tex_mode, chunk,
                   use_amp, device_type, dtype):
    """Returns z_geo [B,S,G,G,C], z_tex [B,S,G,G,C]."""
    stripped = None
    if tex_mode != 'tex_only':
        with torch.no_grad():
            tokens_list, psi = encoder(frames)
            stripped = strip_special_tokens(tokens_list, psi)

    ctx = autocast(device_type=device_type, dtype=dtype) if use_amp else _nullcontext()
    with ctx:
        if tex_mode == 'tex_only':
            # Skip the compressor forward entirely: z_geo is zeros and the
            # compressor is frozen in this mode (a discarded forward would
            # leave its params unused and break DDP with
            # find_unused_parameters=False).
            B, S = frames.shape[0], frames.shape[1]
            g, c = compressor.latent_grid, compressor.cdim
            z_geo = torch.zeros(B, S, g, g, c,
                                device=frames.device, dtype=frames.dtype)
        else:
            z_geo_nchw = compressor(stripped)  # [B,S,C,G,G]
            z_geo = z_geo_nchw.permute(0, 1, 3, 4, 2).contiguous()
        if tex_mode in ('oracle', 'tex_only'):
            z_tex = tex_encoder(frames)
        else:
            z_tex = torch.zeros(
                z_geo.shape[0], z_geo.shape[1], z_geo.shape[2], z_geo.shape[3],
                tex_encoder.out_dim, device=z_geo.device, dtype=z_geo.dtype)
    return z_geo, z_tex


class DualStreamCore(nn.Module):
    """Wraps the full trainable forward so DDP sees every gradient.

    Calling sub-modules through ``ddp.module`` skips the reducer and the
    gradient all-reduce silently never happens; multi-GPU runs degrade to
    independent single-GPU runs. All training/eval forwards must go through
    this module's forward (i.e. through the DDP wrapper when use_ddp).
    """

    def __init__(self, compressor, tex_encoder, decoder, tex_mode):
        super().__init__()
        self.compressor = compressor
        self.tex_encoder = tex_encoder
        self.decoder = decoder
        self.tex_mode = tex_mode

    def forward(self, encoder, frames, chunk, use_amp, device_type, dtype):
        z_geo, z_tex = encode_streams(
            encoder, self.compressor, self.tex_encoder, frames, self.tex_mode,
            chunk, use_amp, device_type, dtype)
        pred = decode_rgb(self.decoder, z_geo, z_tex, chunk, use_amp,
                          device_type, dtype)
        return pred, z_geo, z_tex


def decode_rgb(decoder, z_geo, z_tex, chunk, use_amp, device_type, dtype):
    ctx = autocast(device_type=device_type, dtype=dtype) if use_amp else _nullcontext()
    with ctx:
        recon = decoder(z_geo, z_tex, frames_chunk_size=chunk)  # [B,S,H,W,3]
    return recon[..., :3].permute(0, 1, 4, 2, 3).contiguous().float()


@torch.no_grad()
def eval_recon(encoder, core, eval_loader, device, out_dir, epoch, dtype,
               device_type, chunk):
    """Evaluate over the whole eval loader (not a single clip).

    Single-clip PSNR varies by several dB — far more than the 2-5 dB gaps
    the probe decision rules discriminate — so metrics are means over all
    eval clips, with per-clip std reported alongside.
    """
    from PIL import Image as PImage
    os.makedirs(out_dir, exist_ok=True)
    was_training = core.training
    core.eval()
    use_amp = dtype != torch.float32

    psnrs, l1s, lpips_vals, mses = [], [], [], []
    grid_saved = False
    for clip_idx, batch in enumerate(eval_loader):
        frames = batch['frames'].to(device=device, dtype=dtype)
        pred, _, _ = core(encoder, frames, chunk, use_amp, device_type, dtype)
        target = frames.float()

        mse = F.mse_loss(pred, target).item()
        mses.append(mse)
        l1s.append(F.l1_loss(pred, target).item())
        psnrs.append(10.0 * np.log10(1.0 / max(mse, 1e-8)))
        try:
            lpips = get_lpips(device)
            B, S = pred.shape[:2]
            lpips_vals.append(lpips(
                pred.reshape(B * S, 3, *pred.shape[-2:]) * 2 - 1,
                target.reshape(B * S, 3, *target.shape[-2:]) * 2 - 1
            ).mean().item())
        except Exception as exc:
            print(f'  [WARN] LPIPS eval skipped: {exc}')

        if not grid_saved:
            S = pred.shape[1]
            rows = []
            for t in range(min(S, 8)):
                gt = (target[0, t].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                pr = (pred[0, t].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                rows.append(np.concatenate([gt, pr], axis=1))
            grid = np.concatenate(rows, axis=0)
            PImage.fromarray(grid).save(
                os.path.join(out_dir, f'epoch{epoch:04d}_grid.png'))
            grid_saved = True

    core.train(was_training)
    return {
        'psnr': float(np.mean(psnrs)),
        'psnr_std': float(np.std(psnrs)),
        'l1': float(np.mean(l1s)),
        'lpips': float(np.mean(lpips_vals)) if lpips_vals else float('nan'),
        'mse': float(np.mean(mses)),
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
        os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
        print(f'E5 probe: tex_mode={args.tex_mode} geo={args.geo_dim} '
              f'tex={args.tex_dim} grid={args.latent_grid} '
              f'device={device_type} dtype={args.dtype}')

    # Frozen VGGT
    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    load_encoder_checkpoint(encoder, args.encoder_ckpt, verbose=main_process)
    encoder = encoder.to(device=device, dtype=dtype).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    input_grid = args.target_size // 14
    compressor = CompactCompressor(
        levels=args.levels, token_dim=args.token_dim, cdim=args.geo_dim,
        latent_grid=args.latent_grid, input_grid=input_grid)
    tex_encoder = TextureEncoder(
        out_dim=args.tex_dim, out_grid=args.latent_grid,
        base_ch=args.tex_base_ch, img_size=args.target_size,
        pack_mode=args.tex_pack)
    decoder = DualStreamDecoder(
        geo_dim=args.geo_dim, tex_dim=args.tex_dim,
        base_dim=args.decoder_base_dim, img_size=args.target_size,
        latent_grid=args.latent_grid,
        use_checkpoint=bool(args.use_checkpoint))

    if args.tex_mode == 'zero':
        for p in tex_encoder.parameters():
            p.requires_grad_(False)
    elif args.tex_mode == 'tex_only':
        # Compressor is bypassed in tex_only forward; freeze so DDP does not
        # expect gradients for it (find_unused_parameters=False).
        for p in compressor.parameters():
            p.requires_grad_(False)

    core = DualStreamCore(compressor, tex_encoder, decoder,
                          args.tex_mode).to(device=device)

    n_params = sum(p.numel() for p in core.parameters() if p.requires_grad)
    if main_process:
        print(f'  Trainable params: {n_params / 1e6:.1f}M '
              f'(tex_enc={tex_encoder.num_parameters()/1e6:.1f}M, '
              f'dec={decoder.num_parameters()/1e6:.1f}M)')

    ema = EMA(core, decay=args.ema_decay, dtype=torch.float32).to(device)
    if use_ddp:
        # All forwards below go through `model` (the DDP wrapper). Calling
        # core/.module directly would silently skip the gradient all-reduce.
        model = nn.parallel.DistributedDataParallel(
            core, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False)
    else:
        model = core

    optimizer = build_optimizer(core, lr=args.lr, wd=args.wd)

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

    eval_loader = None
    if args.eval_csv and main_process:
        eval_dataset = SpatialVidDataset(
            csv_path=args.eval_csv,
            video_root=args.eval_video_root or args.video_root,
            seq_len=args.seq_len, target_size=args.target_size,
            max_videos=args.eval_clips,
            num_frames_per_video=args.num_frames_per_video,
            max_frame_span=args.max_frame_span,
            clip_duration_seconds=args.clip_duration_seconds,
            decode_retries=args.decode_retries,
            temporal_jitter=False)
        # num_workers=0: match E1/E4 eval loaders so all probes score the
        # same deterministic eval windows (worker processes reseed RNG).
        eval_loader = DataLoader(
            eval_dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_fn)

    start_epoch, global_step = 0, 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        core.load_state_dict(ckpt['model'])
        if 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt and use_scaler:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        global_step = int(ckpt.get('global_step', 0))
        if 'rng' in ckpt:
            restore_rng_state(ckpt['rng'])
        if main_process:
            print(f'  Resumed from {args.resume} @ epoch {start_epoch}')

    writer = SummaryWriter(os.path.join(args.output_dir, 'tb')) if main_process else None
    meter = ThroughputMeter()
    chunk = args.frames_chunk_size if args.frames_chunk_size > 0 else None
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')

    if main_process:
        print(f'Training: {args.epochs} epochs, {steps_per_epoch} steps/epoch')

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(dataloader, disable=not main_process, desc=f'ep{epoch}')
        for it, batch in enumerate(pbar):
            frames = batch['frames'].to(device=device, dtype=dtype, non_blocking=True)
            target = frames.float()

            pred, z_geo, z_tex = model(
                encoder, frames, chunk, use_amp, device_type, dtype)

            l1 = F.l1_loss(pred, target)
            loss = args.lambda_l1 * l1
            metrics = {'train/l1': l1.detach()}

            if args.lambda_lpips > 0:
                lpips = get_lpips(device)
                B, S = pred.shape[:2]
                lp = lpips(
                    pred.reshape(B * S, 3, *pred.shape[-2:]) * 2 - 1,
                    target.reshape(B * S, 3, *target.shape[-2:]) * 2 - 1).mean()
                loss = loss + args.lambda_lpips * lp
                metrics['train/lpips'] = lp.detach()

            if args.lambda_feat > 0:
                # VGGT feature-consistency loss (MIRA P-DINO analogue): the
                # frozen encoder is its own perceptual metric, and it directly
                # optimizes what the geometric-consistency eval later measures.
                # Only k frames per clip are re-encoded to bound the cost.
                B, S = pred.shape[:2]
                k = min(args.feat_frames, S)
                idx = torch.randperm(S, device=pred.device)[:k]
                pred_sub = pred[:, idx].clamp(0, 1).to(dtype)
                with torch.no_grad():
                    tgt_tokens, tgt_psi = encoder(frames[:, idx])
                    tgt_feats = strip_special_tokens(tgt_tokens, tgt_psi)
                pred_tokens, pred_psi = encoder(pred_sub)
                pred_feats = strip_special_tokens(pred_tokens, pred_psi)
                feat_loss = sum(
                    F.l1_loss(pred_feats[lvl].float(), tgt_feats[lvl].float())
                    for lvl in args.levels) / len(args.levels)
                loss = loss + args.lambda_feat * feat_loss
                metrics['train/feat'] = feat_loss.detach()

            if args.tex_mode in ('oracle', 'tex_only') and args.lambda_tex_reg > 0:
                # var+eps under sqrt: std() has a NaN gradient at zero
                # variance (e.g. any constant channel), and one NaN here
                # poisons every parameter through clip_grad_norm_.
                flat = z_tex.float().reshape(-1, z_tex.shape[-1])
                ch_std = torch.sqrt(flat.var(0) + 1e-6)
                if args.tex_reg_mode == 'match_geo' and args.tex_mode == 'oracle':
                    # SVG-style: align z_tex channel stats to z_geo's batch
                    # stats so both streams live on one distribution family
                    # (keeps the tex stream from drifting onto its own
                    # manifold, which both hurts diffusability and lets it
                    # shortcut around geometry).
                    gflat = z_geo.float().reshape(-1, z_geo.shape[-1]).detach()
                    g_std = torch.sqrt(gflat.var(0) + 1e-6)
                    tex_reg = (flat.mean(0) - gflat.mean(0)).pow(2).mean() \
                        + (ch_std - g_std).pow(2).mean()
                else:
                    # Channel-wise mean/std toward N(0,1).
                    tex_reg = flat.mean(0).pow(2).mean() \
                        + (ch_std - 1.0).pow(2).mean()
                loss = loss + args.lambda_tex_reg * tex_reg
                metrics['train/tex_reg'] = tex_reg.detach()

            metrics['train/loss'] = loss.detach()
            loss = loss / args.accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (it + 1) % args.accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        core.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        core.parameters(), args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(core)
                global_step += 1
                meter.update(frames.shape[0] * frames.shape[1] * world_size)

                if main_process and global_step % args.log_every == 0:
                    row = {
                        'epoch': epoch, 'step': global_step,
                        'probe': args.probe_name, 'tex_mode': args.tex_mode,
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/grad_norm': float(grad_norm),
                        'train/DI_throughput': meter.rate(),
                    }
                    for k, v in metrics.items():
                        row[k] = float(v.item() if torch.is_tensor(v) else v)
                        if writer is not None:
                            writer.add_scalar(k, row[k], global_step)
                    append_metrics(metrics_path, row)
                    pbar.set_postfix(l1=row['train/l1'], loss=row['train/loss'])

        # End-of-epoch eval
        if main_process and eval_loader is not None and (
                (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1):
            # Eval with current weights (EMA shadow kept for checkpoint only;
            # matches E4 probe pattern — EMA class has no apply/restore API).
            stats = eval_recon(
                encoder, core, eval_loader, device,
                os.path.join(args.output_dir, 'samples'), epoch, dtype,
                device_type, chunk)
            row = {'epoch': epoch, 'step': global_step,
                   'probe': args.probe_name, 'tex_mode': args.tex_mode, **stats}
            append_metrics(metrics_path, row)
            if writer is not None:
                for k, v in stats.items():
                    writer.add_scalar(f'eval/{k}', v, global_step)
            print(f'  Eval ep{epoch}: PSNR={stats["psnr"]:.3f}'
                  f'±{stats["psnr_std"]:.3f} ({stats["eval_clips"]} clips) '
                  f'LPIPS={stats["lpips"]:.4f} L1={stats["l1"]:.4f}')

        if main_process and (
                (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1):
            ckpt = {
                'model': core.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'args': vars(args),
                'rng': capture_rng_state(),
            }
            if use_scaler:
                ckpt['scaler'] = scaler.state_dict()
            path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:04d}.pt')
            atomic_torch_save(ckpt, path)
            atomic_torch_save(ckpt, os.path.join(args.output_dir, 'checkpoint_latest.pt'))
            print(f'  Saved checkpoint at epoch {epoch}')

    if writer is not None:
        writer.close()
    if use_ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
