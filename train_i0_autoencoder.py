#!/usr/bin/env python3
"""Train I_0-conditioned autoencoder with cross-frame appearance sampling.

Key idea: I_A provides appearance, z_geo_B provides geometry.
Decoder learns to warp I_A's texture to I_B's geometry.
Prevents shortcut learning by sampling different frames for appearance vs geometry.

Usage: bash scripts/train_i0_autoencoder.sh
"""

import argparse, os, sys, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from streamvggt.models.streamvggt import StreamVGGT
from models.generative_tokenizer import GenerativeTokenizer
from models.appearance_cnn import AppearanceCNN
from models.i0_decoder import I0ConditionalDecoder
from data.video_dataset import SpatialVidDataset, collate_fn
from data.token_utils import strip_special_tokens
from utils.training import EMA, build_optimizer, build_scheduler
from utils.distributed import setup_ddp, is_main_process

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
    p.add_argument('--latent_dim', type=int, default=512); p.add_argument('--latent_grid', type=int, default=18)
    p.add_argument('--decoder_base_dim', type=int, default=384)
    p.add_argument('--decoder_num_resblocks', type=int, default=2)
    p.add_argument('--epochs', type=int, default=50); p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--accum_steps', type=int, default=4); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-2); p.add_argument('--warmup_steps', type=int, default=500)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--dtype', type=str, default='bf16'); p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--target_size', type=int, default=518); p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--save_every', type=int, default=5); p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--local_rank', type=int, default=0)
    p.add_argument('--cross_frame_gap', type=int, default=4,
                   help='Max frame gap between I_A and I_B during cross-frame training')
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_samples(enc, tok, dec, app_cnn, eval_video, device, out_dir, epoch, use_bf16):
    """Save reconstruction: I_0→I_0 (self), I_0→I_t (cross-frame), for multiple t."""
    os.makedirs(out_dir, exist_ok=True)
    dec.eval(); app_cnn.eval()
    frames = eval_video.to(device, dtype=torch.bfloat16)  # [1, S, 3, 518, 518]
    B, S = frames.shape[:2]

    # Get z_geo for all frames
    tl, psi = enc(frames); tl = strip_special_tokens(tl, psi)
    z_g, _ = tok(tl)  # [1, S, 18, 18, 512]

    # I_0 features
    I_0 = frames[:, 0:1, :, :, :]  # [1, 1, 3, H, W]
    I_0_feats = app_cnn(I_0.reshape(1, 3, 518, 518))

    # Decode all S frames conditioned on I_0
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
    print(f'  Eval PSNR: {psnr:.2f} dB')
    dec.train(); app_cnn.train()


def main():
    args = parse_args()
    use_ddp, rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    main_process = is_main_process()
    use_bf16 = args.dtype == 'bf16' and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    if main_process:
        print(f'=== I_0-Conditioned Autoencoder Training ===')
        print(f'Device: {device}, Dtype: {dtype}, DDP: {use_ddp}')

    set_seed(args.seed); os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load frozen encoder + tokenizer ----
    if main_process: print('[1/4] Loading frozen StreamVGGT + Tokenizer A...')
    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location='cpu'), strict=False)
    encoder = encoder.to(device, dtype=torch.bfloat16).eval()
    for p in encoder.parameters(): p.requires_grad_(False)

    tokenizer = GenerativeTokenizer(levels=[4,11,17,23], seq_len=args.seq_len, input_grid=37,
                                     latent_dim=args.latent_dim, latent_grid=args.latent_grid).to(device)
    ae_ckpt = torch.load(args.autoencoder_ckpt, map_location='cpu')
    tokenizer.load_state_dict(ae_ckpt['tokenizer']); tokenizer.eval()
    for p in tokenizer.parameters(): p.requires_grad_(False)

    # ---- Build Appearance CNN + I_0 Decoder ----
    if main_process: print('[2/4] Building Appearance CNN + I_0 Conditional Decoder...')
    app_cnn = AppearanceCNN().to(device=device, dtype=torch.bfloat16)
    decoder = I0ConditionalDecoder(
        latent_dim=args.latent_dim, base_dim=args.decoder_base_dim, img_size=args.target_size,
        latent_grid=args.latent_grid, num_resblocks=args.decoder_num_resblocks, use_checkpoint=True,
    ).to(device)
    tp = sum(p.numel() for p in app_cnn.parameters()) + sum(p.numel() for p in decoder.parameters())
    if main_process: print(f'  AppCNN: {sum(p.numel() for p in app_cnn.parameters())/1e6:.1f}M, Decoder: {sum(p.numel() for p in decoder.parameters())/1e6:.1f}M, Total: {tp/1e6:.1f}M')

    ema_modules = nn.ModuleList([app_cnn, decoder])
    ema = EMA(ema_modules, decay=0.999).to(device)

    # ---- Dataset ----
    if main_process: print('[3/4] Building dataset...')
    dataset = SpatialVidDataset(csv_path=args.csv, video_root=args.video_root, seq_len=args.seq_len,
                                 target_size=args.target_size, max_videos=args.max_videos,
                                 num_frames_per_video=args.seq_len)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if use_ddp else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=True)

    # Eval video
    eval_video = None
    for eb in loader: eval_video = eb['frames'][:1].clone(); break

    # ---- Optimizer ----
    if main_process: print('[4/4] Building optimizer...')
    trainable = list(app_cnn.parameters()) + list(decoder.parameters())
    optimizer = build_optimizer(nn.ModuleList([app_cnn, decoder]), lr=args.lr, wd=args.wd)
    steps_per_epoch = (len(loader) + args.accum_steps - 1) // args.accum_steps
    total_steps = args.epochs * steps_per_epoch
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=max(total_steps, 1))

    global_step, start_epoch = 0, 0
    if args.resume and os.path.exists(args.resume):
        if main_process: print(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        app_cnn.load_state_dict(ckpt['app_cnn']); decoder.load_state_dict(ckpt['decoder'])
        ema.load_state_dict(ckpt['ema']); ema = ema.to(device)
        if 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['scheduler'])
        global_step, start_epoch = ckpt.get('global_step', 0), ckpt.get('epoch', 0) + 1

    if use_ddp:
        app_cnn = nn.parallel.DistributedDataParallel(app_cnn, device_ids=[local_rank], output_device=local_rank)
        decoder = nn.parallel.DistributedDataParallel(decoder, device_ids=[local_rank], output_device=local_rank)

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tb')) if main_process else None
    if main_process: print(f'\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch')

    for epoch in range(start_epoch, args.epochs):
        app_cnn.train(); decoder.train()
        epoch_loss, n_batches = 0.0, 0
        if use_ddp: sampler.set_epoch(epoch)
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}', dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(device, dtype=torch.bfloat16)  # [B, S, 3, H, W]
            B, S = frames.shape[:2]

            # ---- Cross-frame sampling: I_A ≠ I_B ----
            t_A = torch.randint(0, S, (1,)).item()
            t_B_candidates = [t for t in range(max(0, t_A - args.cross_frame_gap),
                                                min(S, t_A + args.cross_frame_gap + 1)) if t != t_A]
            t_B = t_B_candidates[torch.randint(0, len(t_B_candidates), (1,)).item()] if t_B_candidates else t_A

            I_A = frames[:, t_A:t_A + 1]  # [B, 1, 3, H, W]
            I_B_frames = frames  # all frames for geometry

            # ---- Encode geometry from I_B (all frames) ----
            with torch.no_grad():
                tl, psi = encoder(I_B_frames); tl = strip_special_tokens(tl, psi)
                z_g, _ = tokenizer(tl)  # [B, S, Hg, Wg, C]

            # ---- Extract appearance from I_A ----
            I_0_feats = (app_cnn.module if use_ddp else app_cnn)(I_A[:, 0])  # AppCNN bf16 → {f36,f72,f144} bf16

            # ---- Decode I_B conditioned on I_A appearance ----
            result = (decoder.module if use_ddp else decoder)(z_g, I_0_feats)
            preds = result[0] if isinstance(result, tuple) else result

            pred_rgb = preds[..., :3].permute(0, 1, 4, 2, 3).float()
            target = I_B_frames.float().clamp(0, 1)

            l1 = F.l1_loss(pred_rgb, target)
            lpips_loss = l1.new_zeros(())
            try:
                lp = get_lpips(device)
                lpips_loss = lp(pred_rgb.reshape(-1, 3, *pred_rgb.shape[-2:]),
                               target.reshape(-1, 3, *target.shape[-2:])).mean()
            except: pass

            loss = l1 + lpips_loss
            (loss / args.accum_steps).backward()
            epoch_loss += loss.item(); n_batches += 1
            pbar.set_postfix(l1=f'{l1.item():.4f}', lpips=f'{lpips_loss.item():.4f}')

            if (batch_idx + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                ema.update(ema_modules); scheduler.step(); global_step += 1

        # End of epoch
        if n_batches % args.accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            ema.update(ema_modules); scheduler.step(); global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if main_process:
            print(f'  Epoch {epoch} | avg_loss={avg_loss:.4f} | step={global_step}')
            writer.add_scalar('train/epoch_loss', avg_loss, epoch)

        if main_process and (epoch + 1) % args.save_every == 0:
            ac = app_cnn.module if use_ddp else app_cnn
            dc = decoder.module if use_ddp else decoder
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:04d}.pt')
            torch.save({'app_cnn': ac.state_dict(), 'decoder': dc.state_dict(), 'ema': ema.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'global_step': global_step, 'epoch': epoch, 'args': vars(args)}, save_path)
            print(f'  Saved: {save_path}')
            try:
                eval_samples(encoder, tokenizer, dc, ac, eval_video, device,
                            os.path.join(args.output_dir, 'samples'), epoch, use_bf16)
            except Exception as e: print(f'  [WARN] Eval: {e}')

    if main_process:
        ac = app_cnn.module if use_ddp else app_cnn
        dc = decoder.module if use_ddp else decoder
        torch.save({'app_cnn': ac.state_dict(), 'decoder': dc.state_dict(), 'ema': ema.state_dict(),
                    'global_step': global_step, 'epoch': args.epochs - 1, 'args': vars(args)},
                   os.path.join(args.output_dir, 'checkpoint_final.pt'))
        print(f'\nDone.')
    if use_ddp: torch.distributed.destroy_process_group()


if __name__ == '__main__': main()
