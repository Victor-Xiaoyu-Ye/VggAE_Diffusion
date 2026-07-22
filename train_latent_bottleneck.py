#!/usr/bin/env python3
"""R6: finetune a MIRA-style latent bottleneck (512 -> comp_dim) on the R5 AE.

Frozen: StreamVGGT encoder, CompactCompressor, TextureEncoder (the R5
tokenizer is the measurement instrument — do not disturb it).
Trained: LatentBottleneck (new) + DualStreamDecoder (finetuned to read
expanded latents; initialized from R5).

Objective: same recon losses as R5 (L1 + LPIPS), plus a small N(0,1)
channel-stat regularizer on the COMPRESSED tokens (that is the future
diffusion space).

Gate (vs R5's 24.41 dB / 0.208 LPIPS): PSNR drop <= 0.5 dB. The output
checkpoint carries the full dual-AE contract (compressor + tex_encoder +
bottleneck + decoder) and is the single artifact the compressed-diffusion
stage loads.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.video_dataset import SpatialVidDataset, collate_fn
from data.loader_utils import multiprocessing_loader_kwargs
from data.token_utils import strip_special_tokens
from models.dpt_latent_decoder import CompactCompressor
from models.dual_stream_decoder import DualStreamDecoder
from models.latent_bottleneck import LatentBottleneck
from models.texture_encoder import TextureEncoder
from streamvggt.models.streamvggt import StreamVGGT
from utils.device import (
    configure_backend_compatibility, get_device, get_device_name,
    manual_seed_all, resolve_dtype,
)
from utils.distributed import is_main_process, setup_ddp
from utils.encoder_loader import load_encoder_checkpoint
from utils.training import (
    append_metrics, atomic_torch_save, build_optimizer, build_scheduler,
    capture_rng_state, restore_rng_state,
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
    p = argparse.ArgumentParser(description='R6 latent bottleneck finetune')
    p.add_argument('--comp_dim', type=int, default=128)
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--video_root', type=str, required=True)
    p.add_argument('--eval_csv', type=str, required=True)
    p.add_argument('--encoder_ckpt', type=str, required=True)
    p.add_argument('--dual_ae_ckpt', type=str, required=True,
                   help='R5 checkpoint (tokenizer frozen, decoder init)')

    p.add_argument('--max_steps', type=int, default=3000)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--bottleneck_lr', type=float, default=2e-4,
                   help='fresh bottleneck learns faster than the decoder')
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--warmup_steps', type=int, default=200)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--lambda_l1', type=float, default=1.0)
    p.add_argument('--lambda_lpips', type=float, default=0.5)
    p.add_argument('--lambda_comp_reg', type=float, default=0.01,
                   help='N(0,1) channel-stat pull on compressed tokens')

    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--target_size', type=int, default=518)
    p.add_argument('--clip_duration_seconds', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--decode_retries', type=int, default=8)
    p.add_argument('--max_videos', type=int, default=0)
    p.add_argument('--eval_clips', type=int, default=32)
    p.add_argument('--frames_chunk_size', type=int, default=4)

    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--save_every', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dtype', type=str, default='bf16')
    p.add_argument('--local_rank', type=int, default=0)
    return p.parse_args()


class R6Core(nn.Module):
    """Trainable core: bottleneck + decoder (DDP wraps this)."""

    def __init__(self, bottleneck, decoder):
        super().__init__()
        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, z_geo, z_tex, chunk):
        z = torch.cat([z_geo, z_tex], dim=-1)          # [B,S,G,G,512]
        z_rec, z_comp = self.bottleneck(z)
        geo_dim = z_geo.shape[-1]
        rgb = self.decoder(z_rec[..., :geo_dim], z_rec[..., geo_dim:],
                           frames_chunk_size=chunk)     # [B,S,H,W,3]
        return rgb, z_comp


def main():
    args = parse_args()
    use_ddp, rank, local_rank, world_size = setup_ddp()
    main_process = is_main_process()
    device = get_device(local_rank)
    device_type = get_device_name()
    dtype = resolve_dtype(args.dtype)
    configure_backend_compatibility(device_type)
    manual_seed_all(args.seed + rank)

    if main_process:
        os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
        print(f'R6 bottleneck finetune: 512 -> {args.comp_dim} '
              f'world={world_size} dtype={args.dtype}')

    # ---- Frozen tokenizer + trainable decoder from R5 ----
    ckpt = torch.load(args.dual_ae_ckpt, map_location='cpu',
                      weights_only=False)
    ck = ckpt.get('args', {})
    grid = int(ck.get('latent_grid', 18))
    geo_dim = int(ck.get('geo_dim', 256))
    tex_dim = int(ck.get('tex_dim', 256))

    encoder = StreamVGGT(img_size=args.target_size, patch_size=14,
                         embed_dim=1024)
    load_encoder_checkpoint(encoder, args.encoder_ckpt, verbose=main_process)
    encoder = encoder.to(device=device, dtype=dtype).eval()

    compressor = CompactCompressor(
        levels=list(ck.get('levels', [4, 11, 17, 23])),
        token_dim=int(ck.get('token_dim', 2048)), cdim=geo_dim,
        latent_grid=grid, input_grid=args.target_size // 14)
    tex_encoder = TextureEncoder(
        out_dim=tex_dim, out_grid=grid,
        base_ch=int(ck.get('tex_base_ch', 64)), img_size=args.target_size,
        pack_mode=ck.get('tex_pack', 'avgpool'))
    decoder = DualStreamDecoder(
        geo_dim=geo_dim, tex_dim=tex_dim,
        base_dim=int(ck.get('decoder_base_dim', 384)),
        img_size=args.target_size, latent_grid=grid,
        use_checkpoint=bool(ck.get('use_checkpoint', 1)))
    state = ckpt['model']
    for module, prefix in ((compressor, 'compressor.'),
                           (tex_encoder, 'tex_encoder.'),
                           (decoder, 'decoder.')):
        sub = {k[len(prefix):]: v for k, v in state.items()
               if k.startswith(prefix)}
        if not sub:
            raise RuntimeError(f'{args.dual_ae_ckpt} lacks {prefix}* keys')
        module.load_state_dict(sub)

    compressor = compressor.to(device).eval()
    tex_encoder = tex_encoder.to(device).eval()
    for m in (encoder, compressor, tex_encoder):
        for prm in m.parameters():
            prm.requires_grad_(False)

    bottleneck = LatentBottleneck(geo_dim + tex_dim, args.comp_dim)
    core = R6Core(bottleneck, decoder).to(device)
    n_train = sum(p.numel() for p in core.parameters() if p.requires_grad)
    if main_process:
        print(f'  trainable {n_train / 1e6:.1f}M '
              f'(bottleneck {bottleneck.num_parameters() / 1e6:.2f}M)')

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            core, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False)
    else:
        model = core

    # Two-speed: fresh bottleneck vs pretrained decoder.
    groups = [
        {'params': list(core.bottleneck.parameters()),
         'lr': args.bottleneck_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in core.decoder.named_parameters()
                    if p.ndim >= 2 and 'norm' not in n and 'bias' not in n],
         'lr': args.lr, 'weight_decay': args.wd},
        {'params': [p for n, p in core.decoder.named_parameters()
                    if p.ndim < 2 or 'norm' in n or 'bias' in n],
         'lr': args.lr, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    scheduler = build_scheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=args.max_steps)

    dataset = SpatialVidDataset(
        csv_path=args.csv, video_root=args.video_root,
        seq_len=args.seq_len, target_size=args.target_size,
        max_videos=args.max_videos,
        clip_duration_seconds=args.clip_duration_seconds,
        decode_retries=args.decode_retries)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) \
        if use_ddp else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device_type == 'cuda', drop_last=True,
        **multiprocessing_loader_kwargs(args.num_workers))

    eval_loader = None
    if main_process:
        eval_dataset = SpatialVidDataset(
            csv_path=args.eval_csv, video_root=args.video_root,
            seq_len=args.seq_len, target_size=args.target_size,
            max_videos=args.eval_clips, temporal_jitter=False,
            clip_duration_seconds=args.clip_duration_seconds,
            decode_retries=args.decode_retries)
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                 num_workers=0, collate_fn=collate_fn)

    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        rck = torch.load(args.resume, map_location='cpu', weights_only=False)
        core.load_state_dict(rck['core'])
        optimizer.load_state_dict(rck['optimizer'])
        scheduler.load_state_dict(rck['scheduler'])
        global_step = rck.get('global_step', 0)
        restore_rng_state(rck.get('rng'))
        if main_process:
            print(f'Resumed from {args.resume} @ step {global_step}')

    writer = SummaryWriter(os.path.join(args.output_dir, 'tb')) \
        if main_process else None
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')
    chunk = args.frames_chunk_size if args.frames_chunk_size > 0 else None

    @torch.no_grad()
    def encode_streams(frames):
        tokens_list, psi = encoder(frames.to(dtype))
        stripped = strip_special_tokens(tokens_list, psi)
        z_geo = compressor([t.float() for t in stripped])
        z_geo = z_geo.permute(0, 1, 3, 4, 2).contiguous().float()
        z_tex = tex_encoder(frames.float()).float()
        return z_geo, z_tex

    def save_ckpt(step):
        # Full dual-AE contract in ONE artifact: the compressed-diffusion
        # stage (and future cache builds) load only this file.
        merged = {}
        for module, prefix in ((compressor, 'compressor.'),
                               (tex_encoder, 'tex_encoder.'),
                               (core.bottleneck, 'bottleneck.'),
                               (core.decoder, 'decoder.')):
            for k, v in module.state_dict().items():
                merged[prefix + k] = v.detach().cpu()
        payload = {
            'model': merged,
            'core': core.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'global_step': step,
            'args': {**ck, **vars(args), 'comp_dim': args.comp_dim,
                     'has_bottleneck': True},
            'rng': capture_rng_state(),
        }
        atomic_torch_save(payload, os.path.join(
            args.output_dir, f'checkpoint_step{step:07d}.pt'))
        atomic_torch_save(payload, os.path.join(
            args.output_dir, 'checkpoint_latest.pt'))
        print(f'  saved checkpoint @ step {step}')

    @torch.no_grad()
    def run_eval(step):
        from PIL import Image as PImage
        core.eval()
        psnrs, l1s, lpips_vals = [], [], []
        grid_saved = False
        for batch in eval_loader:
            frames = batch['frames'].to(device)
            z_geo, z_tex = encode_streams(frames)
            pred, _ = core(z_geo, z_tex, chunk)
            pred = pred[..., :3].clamp(0, 1)
            target = frames.float().clamp(0, 1).permute(0, 1, 3, 4, 2)
            mse = F.mse_loss(pred, target).item()
            psnrs.append(-10 * np.log10(max(mse, 1e-10)))
            l1s.append(F.l1_loss(pred, target).item())
            try:
                lp = get_lpips(device)
                B, S = pred.shape[:2]
                pr = pred.permute(0, 1, 4, 2, 3).reshape(
                    B * S, 3, *pred.shape[2:4])
                tg = target.permute(0, 1, 4, 2, 3).reshape(
                    B * S, 3, *target.shape[2:4])
                lpips_vals.append(lp(pr * 2 - 1, tg * 2 - 1).mean().item())
            except Exception:
                pass
            if not grid_saved:
                S = pred.shape[1]
                rows = torch.cat(
                    [torch.cat([target[0, s], pred[0, s]], dim=1)
                     for s in range(S)], dim=0)
                PImage.fromarray(
                    (rows.cpu().numpy() * 255).astype(np.uint8)).save(
                    os.path.join(args.output_dir, 'samples',
                                 f'step{step:07d}_grid.png'))
                grid_saved = True
        row = {
            'step': step, 'comp_dim': args.comp_dim,
            'psnr': float(np.mean(psnrs)),
            'psnr_std': float(np.std(psnrs)),
            'l1': float(np.mean(l1s)),
            'lpips': float(np.mean(lpips_vals)) if lpips_vals else -1,
            'eval_clips': len(psnrs),
        }
        append_metrics(metrics_path, row)
        if writer:
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f'eval/{k}', v, step)
        print(f'  [eval] step {step}: PSNR={row["psnr"]:.3f}'
              f'±{row["psnr_std"]:.3f} LPIPS={row["lpips"]:.4f}')
        core.train()

    if main_process:
        print(f'Training: {args.max_steps} steps')
    model.train()
    epoch = 0
    window_tokens = 0
    window_start = time.time()
    optimizer.zero_grad(set_to_none=True)
    done = False
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(loader, disable=not main_process, desc=f'ep{epoch}')
        for it_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(device)
            z_geo, z_tex = encode_streams(frames)
            pred, z_comp = model(z_geo, z_tex, chunk)
            pred = pred[..., :3]
            target = frames.float().clamp(0, 1).permute(0, 1, 3, 4, 2)

            l1 = F.l1_loss(pred, target)
            loss = args.lambda_l1 * l1
            if args.lambda_lpips > 0:
                lp = get_lpips(device)
                B, S = pred.shape[:2]
                pr = pred.permute(0, 1, 4, 2, 3).reshape(
                    B * S, 3, *pred.shape[2:4])
                tg = target.permute(0, 1, 4, 2, 3).reshape(
                    B * S, 3, *target.shape[2:4])
                lpv = lp(pr * 2 - 1, tg * 2 - 1).mean()
                loss = loss + args.lambda_lpips * lpv
            if args.lambda_comp_reg > 0:
                flat = z_comp.float().reshape(-1, z_comp.shape[-1])
                ch_std = torch.sqrt(flat.var(0) + 1e-6)
                reg = flat.mean(0).pow(2).mean() + (ch_std - 1.0).pow(2).mean()
                loss = loss + args.lambda_comp_reg * reg

            (loss / args.accum_steps).backward()
            if (it_idx + 1) % args.accum_steps != 0:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in core.parameters() if p.requires_grad],
                args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            window_tokens += (frames.shape[0] * args.accum_steps
                              * args.seq_len * grid * grid)

            if main_process and global_step % args.log_every == 0:
                now = time.time()
                di = window_tokens / max(now - window_start, 1e-12)
                window_tokens = 0
                window_start = now
                row = {'step': global_step, 'train/loss': float(loss.item()),
                       'train/l1': float(l1.item()),
                       'train/grad_norm': float(grad_norm),
                       'train/lr': scheduler.get_last_lr()[0],
                       'DI_throughput': di}
                append_metrics(metrics_path, row)
                if writer:
                    for k, v in row.items():
                        if isinstance(v, (int, float)):
                            writer.add_scalar(k, v, global_step)
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                pbar.write(f'{ts}: [train {epoch + 1} '
                           f'{global_step}/{args.max_steps}] '
                           f'loss: {row["train/loss"]:.4f} | '
                           f'DI_throughput: {di:.2f} tokens/s/npu')
                pbar.set_postfix({'loss': f'{row["train/loss"]:.4f}',
                                  'DI_throughput': f'{di:.2f}'})

            if main_process and global_step % args.eval_every == 0:
                run_eval(global_step)
            if main_process and global_step % args.save_every == 0:
                save_ckpt(global_step)
            if global_step >= args.max_steps:
                done = True
                break
        epoch += 1

    if main_process:
        run_eval(global_step)
        save_ckpt(global_step)
        if writer:
            writer.close()
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
