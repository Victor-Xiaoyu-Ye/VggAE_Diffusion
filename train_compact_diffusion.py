#!/usr/bin/env python3
"""Phase 2: Train flow matching diffusion on compact generative latent z_g.

Uses the frozen Tokenizer A from Phase 1 to produce z_g = A(E(x)),
then trains CompactLatentDiT to model p(z_g) via OT-CFM flow matching.

The compact latent space (512-dim × 18×18 = 166K dims per frame)
is much more tractable than raw VGGT tokens (2048-dim × 37×37 = 2.8M),
enabling stable training with a lightweight DiT backbone.
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
from models.compact_dit import CompactLatentDiT
from models.flow_matching import OTCFM
from data.video_dataset import SpatialVidDataset, collate_fn
from data.token_utils import strip_special_tokens
from utils.training import EMA, build_optimizer, build_scheduler
from utils.distributed import setup_ddp, is_main_process


def parse_args():
    p = argparse.ArgumentParser(description='Train flow matching on compact latent')

    # Data
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--video_root', type=str, required=True)
    p.add_argument('--max_videos', type=int, default=0)
    p.add_argument('--annotation_index', type=str, default='')

    # Checkpoints
    p.add_argument('--encoder_ckpt', type=str, required=True)
    p.add_argument('--autoencoder_ckpt', type=str, required=True,
                   help='Phase 1 checkpoint (contains tokenizer + decoder)')
    p.add_argument('--output_dir', type=str, default='ckpts/diffusion_compact/exp-1')
    p.add_argument('--resume', type=str, default='')

    # Model
    p.add_argument('--latent_dim', type=int, default=512)
    p.add_argument('--latent_grid', type=int, default=18)
    p.add_argument('--model_dim', type=int, default=768)
    p.add_argument('--spatial_depth', type=int, default=8)
    p.add_argument('--temporal_depth', type=int, default=4)
    p.add_argument('--num_heads', type=int, default=12)
    p.add_argument('--token_dim', type=int, default=2048)
    p.add_argument('--levels', type=int, nargs='+', default=[4, 11, 17, 23])
    p.add_argument('--text_cond', action='store_true')
    p.add_argument('--cfg_dropout', type=float, default=0.1)

    # Decoder auxiliary loss
    p.add_argument('--decoder_aux', action='store_true', default=True)
    p.add_argument('--recon_weight', type=float, default=0.05)
    p.add_argument('--recon_every', type=int, default=1)

    # Training
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--ema_decay', type=float, default=0.9999)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--use_bf16', action='store_true', default=True)
    p.add_argument('--input_noise', type=float, default=0.0)

    # Data loading
    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--target_size', type=int, default=518)
    p.add_argument('--num_frames_per_video', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)

    # Eval
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--local_rank', type=int, default=0)

    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    use_ddp, rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    main_process = is_main_process()

    use_bf16 = args.use_bf16 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    if main_process:
        print(f'=== Phase 2: Compact Latent Diffusion ===')
        print(f'Device: {device}, BF16: {use_bf16}, DDP: {use_ddp}')
        print(f'DiT: dim={args.model_dim}, spatial={args.spatial_depth}, temporal={args.temporal_depth}')
        print(f'Latent: {args.latent_dim}dim × {args.latent_grid}×{args.latent_grid}')

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. Load frozen encoder + tokenizer + decoder ----
    if main_process:
        print(f'\n[1/4] Loading frozen encoder + tokenizer...')
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
        latent_dim=args.latent_dim, base_dim=256,
        output_dim=3, output_depth=True,
        img_size=args.target_size, latent_grid=args.latent_grid,
    ).to(device=device)

    # Load Phase 1 weights
    ae_ckpt = torch.load(args.autoencoder_ckpt, map_location='cpu')
    tokenizer.load_state_dict(ae_ckpt['tokenizer'])
    decoder.load_state_dict(ae_ckpt['decoder'])
    tokenizer.eval()
    decoder.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)

    if main_process:
        print(f'  Tokenizer + Decoder frozen.')

    # ---- 2. Build DiT ----
    if main_process:
        print(f'[2/4] Building CompactLatentDiT...')
    num_tokens = args.latent_grid ** 2
    model = CompactLatentDiT(
        latent_dim=args.latent_dim, num_tokens=num_tokens,
        model_dim=args.model_dim, spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth, num_heads=args.num_heads,
        seq_len=args.seq_len, text_cond=args.text_cond,
    ).to(device=device)

    total_p = sum(p.numel() for p in model.parameters())
    if main_process:
        print(f'  DiT params: {total_p / 1e6:.1f}M')

    flow = OTCFM(model)
    ema = EMA(model, decay=args.ema_decay).to(device)

    # ---- 2.5 CLIP text encoder ----
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

    # ---- 4. Optimizer ----
    if main_process:
        print(f'[4/4] Building optimizer (lr={args.lr})...')
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
            print(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema']); ema = ema.to(device)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        global_step = ckpt.get('global_step', 0)
        start_epoch = ckpt.get('epoch', 0) + 1

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=args.text_cond,
        )
        flow.model = model

    # Eval frames
    eval_frames = None
    for eb in dataloader:
        eval_frames = eb['frames'][:1].clone()
        break

    writer = None
    if main_process:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tb'))
        print(f'\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        if use_ddp:
            sampler.set_epoch(epoch)
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}', dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(device=device, dtype=torch.bfloat16)

            # ---- Encode → Tokenize → z_g ----
            with torch.no_grad():
                tokens_list, psi = encoder(frames)
                tokens_list = strip_special_tokens(tokens_list, psi)
                z_g, z_g_flat = tokenizer(tokens_list)
                x1 = z_g_flat.to(dtype=dtype)  # [B, S, N, D] target for flow matching

            # Text conditioning
            text_emb = None
            if clip_encoder is not None:
                text_emb = clip_encoder(batch['caption']).to(device=device, dtype=dtype)
                if args.cfg_dropout > 0:
                    mask = torch.rand(x1.shape[0], device=device) < args.cfg_dropout
                    text_emb = text_emb * (~mask).view(-1, 1, 1).to(dtype=text_emb.dtype)

            # ---- Flow matching loss ----
            if use_bf16:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss = flow.compute_loss(x1, text_emb=text_emb)
            else:
                loss = flow.compute_loss(x1, text_emb=text_emb)

            # Decoder auxiliary loss
            dec_loss = x1.new_zeros(())
            if args.decoder_aux and (batch_idx % args.recon_every == 0):
                with torch.no_grad():
                    flow_out_full = flow.compute_loss(x1, text_emb=text_emb, return_outputs=True)
                    x1_pred = flow_out_full['x1_pred']  # predicted clean z_g
                # Decode predicted z_g
                z_g_pred = x1_pred.reshape(*z_g.shape)
                with torch.no_grad():
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        result = decoder(z_g_pred.float())
                    if decoder.module.output_depth if use_ddp else decoder.output_depth:
                        preds, _, _, _ = result
                    else:
                        preds, _ = result
                # decoder outputs BHWC [B,S,H,W,3], frames is BCSHW [B,S,3,H,W]
                recon = preds[..., :3].permute(0, 1, 4, 2, 3).contiguous().clamp(0, 1).float()
                target = frames.clamp(0, 1).float()  # already [B,S,3,H,W]
                dec_loss = F.l1_loss(recon, target)
                loss = loss + args.recon_weight * dec_loss

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
                pbar.set_postfix(
                    loss=f'{loss_val:.4f}',
                    dec=f'{dec_loss.item():.4f}' if dec_loss.item() > 0 else '',
                )

                if main_process and writer and global_step % 50 == 0:
                    writer.add_scalar('train/loss', loss_val, global_step)
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

        # Handle incomplete accumulation
        if num_batches % args.accum_steps != 0:
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
            print(f'  Epoch {epoch} | avg_loss={avg_loss:.4f} | steps={global_step}')
            if writer:
                writer.add_scalar('train/epoch_loss', avg_loss, epoch)

        # Save
        if main_process and (epoch + 1) % args.save_every == 0:
            base_model = model.module if use_ddp else model
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:04d}.pt')
            torch.save({
                'model': base_model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
                'args': vars(args),
            }, save_path)
            print(f'  Saved: {save_path}')

    if main_process:
        base_model = model.module if use_ddp else model
        final_path = os.path.join(args.output_dir, 'checkpoint_final.pt')
        torch.save({
            'model': base_model.state_dict(),
            'ema': ema.state_dict(),
            'global_step': global_step,
            'epoch': args.epochs - 1,
            'args': vars(args),
        }, final_path)
        print(f'\nDone. Final: {final_path}')

    if use_ddp:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
