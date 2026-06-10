#!/usr/bin/env python3
"""Phase 2 (Wan backbone): Train flow matching on compact latent z_g using Wan 1.3B.

Key differences from from-scratch DiT:
  - Wan backbone with DUAL time conditioning (concat input + adaLN blocks)
  - Trainable: modulation + time_emb + QKV (~215M)
  - 3D RoPE for video spatial-temporal prior
  - Pretrained text cross-attention
"""

import argparse
import os, sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from streamvggt.models.streamvggt import StreamVGGT
from models.generative_tokenizer import GenerativeTokenizer
from models.compact_decoder import CompactDecoder
from models.wan_compact_adapter import WanCompactAdapter
from models.flow_matching import OTCFM
from data.video_dataset import SpatialVidDataset, collate_fn
from data.token_utils import strip_special_tokens
from utils.training import EMA, build_optimizer, build_scheduler
from utils.distributed import setup_ddp, is_main_process


def parse_args():
    p = argparse.ArgumentParser(description='Train Wan-based flow matching on compact latent')
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--video_root', type=str, required=True)
    p.add_argument('--max_videos', type=int, default=0)
    p.add_argument('--annotation_index', type=str, default='')
    p.add_argument('--encoder_ckpt', type=str, required=True)
    p.add_argument('--autoencoder_ckpt', type=str, required=True)
    p.add_argument('--wan_ckpt_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='ckpts/diffusion_wan_compact/exp-1')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--latent_dim', type=int, default=512)
    p.add_argument('--latent_grid', type=int, default=18)
    p.add_argument('--token_dim', type=int, default=2048)
    p.add_argument('--levels', type=int, nargs='+', default=[4, 11, 17, 23])
    p.add_argument('--decoder_base_dim', type=int, default=384)
    p.add_argument('--decoder_num_resblocks', type=int, default=2)
    p.add_argument('--decoder_pixel_shuffle', action='store_true', default=True)
    p.add_argument('--decoder_temporal_blocks', type=int, default=2)
    p.add_argument('--decoder_version', type=str, default='v2')
    p.add_argument('--text_cond', action='store_true')
    p.add_argument('--cfg_dropout', type=float, default=0.1)
    p.add_argument('--decoder_aux', action='store_true', default=True)
    p.add_argument('--recon_weight', type=float, default=0.05)
    p.add_argument('--recon_every', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--ema_decay', type=float, default=0.9999)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--dtype', type=str, default='bf16')
    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--target_size', type=int, default=518)
    p.add_argument('--num_frames_per_video', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--local_rank', type=int, default=0)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def diffusion_sample(model, decoder, tokenizer, encoder, eval_frames, device,
                     out_dir, step, num_steps=20):
    import numpy as np
    from PIL import Image as PImage
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    model_dtype = next(model.parameters()).dtype
    num_tokens = 18 * 18
    latent_dim = 512

    z = torch.randn(1, 8, num_tokens, latent_dim, device=device, dtype=model_dtype)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_val = torch.full((1,), i / num_steps, device=device, dtype=model_dtype)
        v = model(z, t_val)
        z = (z + v * dt)

    z_g = z.reshape(1, 8, 18, 18, latent_dim).float()
    result = decoder(z_g)
    if getattr(decoder, 'output_depth', False):
        preds, _, _, _ = result
    else:
        preds, _ = result
    gen = preds[..., :3].clamp(0, 1)
    orig = eval_frames.to(device).permute(0, 1, 3, 4, 2).clamp(0, 1)

    gen = gen.cpu(); orig = orig.cpu()
    S = gen.shape[1]; grid_rows = []
    for s in range(S):
        grid_rows.append(torch.cat([orig[0, s], gen[0, s]], dim=1))
    grid = torch.cat(grid_rows, dim=0)
    grid_np = (grid.float().numpy() * 255).astype(np.uint8)
    PImage.fromarray(grid_np).save(os.path.join(out_dir, f'sample_step{step:06d}.png'))
    model.train()


def main():
    args = parse_args()

    use_ddp, rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    main_process = is_main_process()

    use_bf16 = args.dtype == 'bf16' and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    if main_process:
        print(f'=== Phase 2: Wan Compact Diffusion ===')
        print(f'Device: {device}, Dtype: {dtype}, DDP: {use_ddp}')
        print(f'Wan: {args.wan_ckpt_dir}')
        print(f'Autoencoder: {args.autoencoder_ckpt}')

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. Load frozen encoder + tokenizer + decoder ----
    if main_process:
        print(f'\n[1/4] Loading frozen autoencoder...')
    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    state = torch.load(args.encoder_ckpt, map_location='cpu')
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    tokenizer = GenerativeTokenizer(
        token_dim=args.token_dim, latent_dim=args.latent_dim,
        latent_grid=args.latent_grid, levels=args.levels,
        seq_len=args.seq_len, input_grid=args.target_size // 14,
    ).to(device=device)

    decoder = CompactDecoder(
        latent_dim=args.latent_dim, base_dim=args.decoder_base_dim,
        output_dim=3, output_depth=True, img_size=args.target_size,
        latent_grid=args.latent_grid,
        num_resblocks=args.decoder_num_resblocks,
        use_pixel_shuffle=args.decoder_pixel_shuffle,
        num_temporal_blocks=args.decoder_temporal_blocks,
        version=args.decoder_version, use_checkpoint=False,
    ).to(device=device)

    ae_ckpt = torch.load(args.autoencoder_ckpt, map_location='cpu')
    tokenizer.load_state_dict(ae_ckpt['tokenizer'])
    decoder.load_state_dict(ae_ckpt['decoder'], strict=False)
    tokenizer.eval(); decoder.eval()
    for p in tokenizer.parameters(): p.requires_grad_(False)
    for p in decoder.parameters(): p.requires_grad_(False)
    if main_process:
        print(f'  Autoencoder frozen.')

    # ---- 2. Build Wan adapter ----
    if main_process:
        print(f'[2/4] Loading WanCompactAdapter...')
    model = WanCompactAdapter(
        wan_checkpoint_dir=args.wan_ckpt_dir,
        latent_dim=args.latent_dim,
        latent_grid=args.latent_grid,
        seq_len=args.seq_len,
    ).to(device=device)

    # Wan time modules in float32
    for m in [model.wan.time_embedding, model.wan.time_projection]:
        for p in m.parameters():
            p.data = p.data.float()

    flow = OTCFM(model)
    ema = EMA(model, decay=args.ema_decay).to(device)

    # ---- 2.5 CLIP text ----
    clip_encoder = None
    if args.text_cond:
        if main_process:
            print(f'[2.5/4] Loading CLIP text encoder...')
        from models.clip_encoder import CLIPTextEncoder
        clip_encoder = CLIPTextEncoder()

    # ---- 3. Dataset ----
    if main_process:
        print(f'[3/4] Building dataset...')
    dataset = SpatialVidDataset(
        csv_path=args.csv, video_root=args.video_root,
        seq_len=args.seq_len, target_size=args.target_size,
        annotation_index_path=args.annotation_index,
        max_videos=args.max_videos,
        num_frames_per_video=args.num_frames_per_video,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if use_ddp else None
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )

    eval_frames = None
    for eb in dataloader:
        eval_frames = eb['frames'][:1].clone()
        break

    # ---- 4. Optimizer ----
    if main_process:
        print(f'[4/4] Building optimizer...')
    optimizer = build_optimizer(model, lr=args.lr, wd=args.wd)
    steps_per_epoch = (len(dataloader) + args.accum_steps - 1) // args.accum_steps
    total_steps = args.epochs * steps_per_epoch
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=max(total_steps, 1))
    scaler = GradScaler(enabled=False)

    global_step = 0; start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        if main_process:
            print(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema']); ema = ema.to(device)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
        global_step = ckpt.get('global_step', 0)
        start_epoch = ckpt.get('epoch', 0) + 1

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )
        flow.model = model

    writer = None
    if main_process:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tb'))
        print(f'\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0; num_batches = 0
        if use_ddp: sampler.set_epoch(epoch)
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}', dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(device=device, dtype=torch.bfloat16)

            with torch.no_grad():
                tokens_list, psi = encoder(frames)
                tokens_list = strip_special_tokens(tokens_list, psi)
                z_g, z_g_flat = tokenizer(tokens_list)
                x1 = z_g_flat.to(dtype=dtype)

            text_emb = None
            if clip_encoder is not None:
                text_emb = clip_encoder(batch['caption']).to(device=device, dtype=dtype)
                if args.cfg_dropout > 0:
                    mask = torch.rand(x1.shape[0], device=device) < args.cfg_dropout
                    text_emb = text_emb * (~mask).view(-1, 1, 1).to(dtype=text_emb.dtype)

            loss = flow.compute_loss(x1, text_emb=text_emb)

            dec_loss = x1.new_zeros(())
            if args.decoder_aux and (batch_idx % args.recon_every == 0):
                with torch.no_grad():
                    flow_out = flow.compute_loss(x1, text_emb=text_emb, return_outputs=True)
                    z_g_pred = flow_out['x1_pred'].reshape(*z_g.shape).float()
                    result = decoder(z_g_pred)
                if decoder.output_depth:
                    preds, _, _, _ = result
                else:
                    preds, _ = result
                recon = preds[..., :3].permute(0, 1, 4, 2, 3).clamp(0, 1).float()
                dec_loss = F.l1_loss(recon, frames.float().clamp(0, 1))
                loss = loss + args.recon_weight * dec_loss

            (loss / args.accum_steps).backward()
            epoch_loss += loss.item(); num_batches += 1

            if (batch_idx + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                ema.update(model.module if use_ddp else model)
                scheduler.step(); global_step += 1
                pbar.set_postfix(loss=f'{loss.item():.4f}',
                                 dec=f'{dec_loss.item():.4f}' if dec_loss.item() > 0 else '')

                if main_process and writer and global_step % 50 == 0:
                    writer.add_scalar('train/loss', loss.item(), global_step)

        if num_batches % args.accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            ema.update(model.module if use_ddp else model)
            scheduler.step(); global_step += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        if main_process:
            print(f'  Epoch {epoch} | avg_loss={avg_loss:.4f} | steps={global_step}')
            if writer: writer.add_scalar('train/epoch_loss', avg_loss, epoch)

        if main_process and (epoch + 1) % args.save_every == 0:
            base_model = model.module if use_ddp else model
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:04d}.pt')
            torch.save({
                'model': base_model.state_dict(), 'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'global_step': global_step, 'epoch': epoch, 'args': vars(args),
            }, save_path)
            print(f'  Saved: {save_path}')
            if eval_frames is not None:
                try:
                    ema_state = {k: v.clone() for k, v in base_model.state_dict().items()}
                    base_model.load_state_dict({k: v for k, v in ema.state_dict().items()})
                    diffusion_sample(base_model, decoder, tokenizer, encoder,
                                    eval_frames, device,
                                    os.path.join(args.output_dir, 'samples'), global_step)
                    base_model.load_state_dict(ema_state); del ema_state
                except Exception as ex:
                    print(f'  [WARN] Eval sampling failed: {ex}')

    if main_process:
        base_model = model.module if use_ddp else model
        torch.save({
            'model': base_model.state_dict(), 'ema': ema.state_dict(),
            'global_step': global_step, 'epoch': args.epochs - 1, 'args': vars(args),
        }, os.path.join(args.output_dir, 'checkpoint_final.pt'))
        print(f'\nDone.')

    if use_ddp:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
