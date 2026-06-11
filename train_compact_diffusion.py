#!/usr/bin/env python3
"""Phase 2: Train flow matching diffusion on compact generative latent z_g.

Uses the frozen Tokenizer A from Phase 1 to produce z_g = A(E(x)),
then trains CompactLatentDiT to model p(z_g) via OT-CFM flow matching.

The compact latent space (512-dim × 18×18 = 166K dims per frame)
is much more tractable than raw VGGT tokens (2048-dim × 37×37 = 2.8M),
enabling stable training with a lightweight DiT backbone.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
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


def parse_args():
    p = argparse.ArgumentParser(description='Train flow matching on compact latent')

    # Data
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--video_root', type=str, required=True)
    p.add_argument('--max_videos', type=int, default=0)
    p.add_argument('--annotation_index', type=str, default='')
    p.add_argument('--eval_csv', type=str, default='')
    p.add_argument('--eval_video_root', type=str, default='')

    # Checkpoints
    p.add_argument('--encoder_ckpt', type=str, required=True)
    p.add_argument('--autoencoder_ckpt', type=str, required=True,
                   help='Phase 1 checkpoint (contains tokenizer + decoder)')
    p.add_argument('--i0_decoder_ckpt', type=str, default='',
                   help='Optional I_0 conditional decoder checkpoint for aux loss')
    p.add_argument('--output_dir', type=str, default='ckpts/diffusion_compact/exp-1')
    p.add_argument('--resume', type=str, default='')

    # Model
    p.add_argument('--latent_dim', type=int, default=512)
    p.add_argument('--latent_grid', type=int, default=18)
    p.add_argument('--model_dim', type=int, default=768)
    p.add_argument('--spatial_depth', type=int, default=8)
    p.add_argument('--temporal_depth', type=int, default=4)
    p.add_argument('--num_heads', type=int, default=12)
    p.add_argument('--time_scale', type=float, default=1000.0)
    p.add_argument('--token_dim', type=int, default=2048)
    p.add_argument('--levels', type=int, nargs='+', default=[4, 11, 17, 23])
    p.add_argument('--decoder_base_dim', type=int, default=256)
    p.add_argument('--decoder_num_resblocks', type=int, default=1)
    p.add_argument('--decoder_pixel_shuffle', action='store_true', default=False)
    p.add_argument('--decoder_temporal_blocks', type=int, default=1)
    p.add_argument('--decoder_output_depth', dest='decoder_output_depth',
                   action='store_true')
    p.add_argument('--no_decoder_output_depth', dest='decoder_output_depth',
                   action='store_false')
    p.set_defaults(decoder_output_depth=True)
    p.add_argument('--text_cond', action='store_true')
    p.add_argument('--cfg_dropout', type=float, default=0.1)
    p.add_argument('--i0_condition', action='store_true',
                   help='Condition the flow model on a separately encoded first frame')
    p.add_argument('--i0_residual', action='store_true',
                   help='Model z_video - z_I0 instead of the full video latent')

    # Decoder auxiliary loss
    p.add_argument('--decoder_aux', dest='decoder_aux', action='store_true')
    p.add_argument('--no_decoder_aux', dest='decoder_aux', action='store_false')
    p.set_defaults(decoder_aux=False)
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
    p.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp32'])
    p.add_argument('--rescale', dest='rescale', action='store_true',
                   help='Rescale z_g to std≈1 for stable flow matching')
    p.add_argument('--no_rescale', dest='rescale', action='store_false')
    p.set_defaults(rescale=True)

    # Data loading
    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--target_size', type=int, default=518)
    p.add_argument('--num_frames_per_video', type=int, default=8)
    p.add_argument('--max_frame_span', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--disable_temporal_jitter', action='store_true')

    # Eval
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--local_rank', type=int, default=0)

    return p.parse_args()


@torch.no_grad()
def diffusion_sample(model, decoder, tokenizer, encoder, eval_frames, device,
                     out_dir, step, latent_scale=1.0, num_steps=20,
                     i0_app_cnn=None):
    """Generate samples via ODE integration + decode for visual tracking."""
    import numpy as np
    from PIL import Image as PImage

    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    B = 1
    num_tokens = model.num_tokens
    latent_dim = model.latent_dim
    seq_len = model.seq_len
    latent_grid = int(num_tokens ** 0.5)
    if latent_grid * latent_grid != num_tokens:
        raise ValueError(f'num_tokens={num_tokens} is not a square grid')

    # Sample noise in z_g flat space (match model dtype: bf16)
    model_dtype = next(model.parameters()).dtype
    z = torch.randn(B, seq_len, num_tokens, latent_dim, device=device, dtype=model_dtype)
    cond = None
    i0_flat = None
    if getattr(model, 'i0_condition', False):
        i0_frames = eval_frames[:, :1].to(device=device, dtype=torch.bfloat16)
        i0_tokens, i0_psi = encoder(i0_frames)
        i0_tokens = strip_special_tokens(i0_tokens, i0_psi)
        _, i0_flat = tokenizer(i0_tokens)
        cond = (i0_flat * latent_scale).to(dtype=model_dtype)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_val = torch.full((B,), i / num_steps, device=device, dtype=model_dtype)
        v = model(z, t_val, cond=cond)
        z = (z + v * dt)

    if getattr(model, 'i0_residual', False):
        future = z / latent_scale + i0_flat.expand(-1, seq_len, -1, -1)
        future_grid = future.reshape(
            B, seq_len, latent_grid, latent_grid, latent_dim)
        i0_grid = i0_flat.reshape(
            B, 1, latent_grid, latent_grid, latent_dim)
        z_g = torch.cat([i0_grid, future_grid], dim=1).float()
    else:
        z_g = (z / latent_scale).reshape(
            B, seq_len, latent_grid, latent_grid, latent_dim).float()

    if i0_app_cnn is not None:
        # I_0 conditional decoding: use first frame as appearance reference
        I_0 = eval_frames[:, 0:1].to(device)  # [1, 1, 3, H, W]
        I_0_feats = i0_app_cnn(I_0[:, 0].to(dtype=torch.bfloat16))
        result = decoder(z_g, I_0_feats)
    else:
        result = decoder(z_g)

    if getattr(decoder, 'output_depth', False):
        preds, _, _, _ = result
    else:
        preds, _ = result

    gen = preds[..., :3].clamp(0, 1)  # [1, S, H, W, 3]
    orig = eval_frames.to(device).permute(0, 1, 3, 4, 2).clamp(0, 1)

    # Metrics
    mse = F.mse_loss(gen.float(), orig.float()).item()
    psnr = float(-10 * np.log10(mse) if mse > 0 else 100.0)
    z_mean = z.float().mean().item()
    z_std = z.float().std().item()
    temporal_delta = (
        (gen[:, 1:] - gen[:, :-1]).abs().mean().item()
        if gen.shape[1] > 1 else 0.0)

    # Save comparison grid
    S = gen.shape[1]; grid_rows = []
    for s in range(S):
        grid_rows.append(torch.cat([orig[0, s].cpu(), gen[0, s].cpu()], dim=1))
    grid = torch.cat(grid_rows, dim=0)
    grid_np = (grid.float().cpu().numpy() * 255).astype(np.uint8)
    PImage.fromarray(grid_np).save(os.path.join(out_dir, f'sample_step{step:06d}.png'))

    model.train()
    return {
        'reference_psnr': psnr,
        'latent_mean': z_mean,
        'latent_std': z_std,
        'generated_temporal_delta': temporal_delta,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_saved_args(saved_args, current_args, keys, checkpoint_name):
    for key in keys:
        if key not in saved_args:
            continue
        saved = saved_args[key]
        current = getattr(current_args, key)
        if isinstance(saved, (list, tuple)):
            matches = list(saved) == list(current)
        else:
            matches = saved == current
        if not matches:
            raise ValueError(
                f'{checkpoint_name} mismatch for {key}: '
                f'checkpoint={saved}, current={current}')


def checkpoint_payload(model, ema, optimizer, scheduler, scaler, global_step,
                       epoch, latent_scale, args):
    return {
        'checkpoint_version': 2,
        'model': model.state_dict(),
        'ema': ema.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
        'latent_scale': latent_scale,
        'rng_state': capture_rng_state(),
        'args': vars(args),
    }


def main():
    args = parse_args()
    if min(args.log_every, args.save_every, args.eval_every) < 1:
        raise ValueError('log/save/eval intervals must be positive')
    if args.i0_residual and not args.i0_condition:
        raise ValueError('--i0_residual requires --i0_condition')
    if args.i0_residual and args.seq_len < 2:
        raise ValueError('--i0_residual requires --seq_len >= 2')
    args.generated_seq_len = (
        args.seq_len - 1 if args.i0_residual else args.seq_len)

    use_ddp, rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    main_process = is_main_process()

    use_bf16 = args.dtype == 'bf16' and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    if main_process:
        print(f'=== Phase 2: Compact Latent Diffusion ===')
        print(f'Device: {device}, BF16: {use_bf16}, DDP: {use_ddp}')
        print(f'DiT: dim={args.model_dim}, spatial={args.spatial_depth}, temporal={args.temporal_depth}')
        print(f'Latent: {args.latent_dim}dim × {args.latent_grid}×{args.latent_grid}')

    set_seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. Load frozen encoder + tokenizer + decoder ----
    if main_process:
        print(f'\n[1/4] Loading frozen encoder + tokenizer...')
    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    state = torch.load(args.encoder_ckpt, map_location='cpu')
    load_info = encoder.load_state_dict(state, strict=False)
    if main_process and (load_info.missing_keys or load_info.unexpected_keys):
        print(f'  Encoder checkpoint mismatch: missing={len(load_info.missing_keys)}, '
              f'unexpected={len(load_info.unexpected_keys)}')
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
        output_dim=3, output_depth=args.decoder_output_depth,
        img_size=args.target_size, latent_grid=args.latent_grid,
        num_resblocks=args.decoder_num_resblocks,
        use_pixel_shuffle=args.decoder_pixel_shuffle,
        num_temporal_blocks=args.decoder_temporal_blocks,
        use_checkpoint=False,
    ).to(device=device)

    # Load Phase 1 weights
    ae_ckpt = torch.load(
        args.autoencoder_ckpt, map_location='cpu', weights_only=False)
    validate_saved_args(
        ae_ckpt.get('args', {}), args,
        ('latent_dim', 'latent_grid', 'token_dim', 'levels', 'seq_len',
         'target_size', 'decoder_base_dim', 'decoder_num_resblocks'),
        'Autoencoder checkpoint')
    tokenizer.load_state_dict(ae_ckpt['tokenizer'])
    decoder.load_state_dict(ae_ckpt['decoder'], strict=False)
    # Match checkpoint's actual depth_head status
    has_depth = any('depth_head' in k for k in ae_ckpt['decoder'].keys())
    decoder.output_depth = has_depth
    if not has_depth and hasattr(decoder, 'depth_head'):
        del decoder.depth_head
    tokenizer.eval()
    decoder.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)

    # Optional I_0 conditional decoder for sampling
    i0_decoder = None; i0_app_cnn = None
    if args.i0_decoder_ckpt and os.path.exists(args.i0_decoder_ckpt):
        if main_process: print(f'  Loading I_0 conditional decoder from {args.i0_decoder_ckpt}')
        from models.i0_decoder import (
            I0ConditionalDecoder, load_i0_decoder_state_dict)
        from models.appearance_cnn import AppearanceCNN
        i0_decoder = I0ConditionalDecoder(
            latent_dim=args.latent_dim, base_dim=args.decoder_base_dim, img_size=args.target_size,
            latent_grid=args.latent_grid, num_resblocks=args.decoder_num_resblocks,
            use_checkpoint=False,
        ).to(device=device)
        i0_app_cnn = AppearanceCNN().to(device=device, dtype=torch.bfloat16)
        i0_ckpt = torch.load(
            args.i0_decoder_ckpt, map_location='cpu', weights_only=False)
        validate_saved_args(
            i0_ckpt.get('args', {}), args,
            ('latent_dim', 'latent_grid', 'seq_len', 'target_size',
             'decoder_base_dim', 'decoder_num_resblocks'),
            'I0 decoder checkpoint')
        load_i0_decoder_state_dict(i0_decoder, i0_ckpt['decoder'])
        i0_app_cnn.load_state_dict(i0_ckpt['app_cnn'])
        i0_decoder.eval(); i0_app_cnn.eval()
        for p in i0_decoder.parameters(): p.requires_grad_(False)
        for p in i0_app_cnn.parameters(): p.requires_grad_(False)
        if main_process: print(f'  I_0 decoder loaded.')

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
        seq_len=args.generated_seq_len, text_cond=args.text_cond,
        i0_condition=args.i0_condition, time_scale=args.time_scale,
    ).to(device=device, dtype=dtype)
    model.i0_residual = args.i0_residual

    total_p = sum(p.numel() for p in model.parameters())
    if main_process:
        print(f'  DiT params: {total_p / 1e6:.1f}M')

    flow = OTCFM(model)
    ema = EMA(model, decay=args.ema_decay, dtype=torch.float32).to(device)

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
        temporal_jitter=not args.disable_temporal_jitter,
        max_frame_span=args.max_frame_span,
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
    if args.warmup_steps >= max(total_steps, 1):
        raise ValueError(
            f'warmup_steps={args.warmup_steps} must be smaller than '
            f'total_steps={total_steps}')
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=max(total_steps, 1))
    scaler = GradScaler(enabled=(not use_bf16))

    # Resume
    global_step = 0
    start_epoch = 0
    restored_latent_scale = None
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f'Resume checkpoint not found: {args.resume}')
        if main_process:
            print(f'Resuming from {args.resume}')
        ckpt = torch.load(
            args.resume, map_location='cpu', weights_only=False)
        saved_args = ckpt.get('args', {})
        validate_saved_args(
            saved_args, args,
            ('latent_dim', 'latent_grid', 'model_dim', 'spatial_depth',
             'temporal_depth', 'num_heads', 'token_dim', 'levels',
             'seq_len', 'target_size'),
            'Diffusion resume checkpoint')
        for key in ('i0_condition', 'i0_residual'):
            if bool(saved_args.get(key, False)) != bool(getattr(args, key)):
                raise ValueError(
                    f'Resume mismatch for {key}: checkpoint={saved_args.get(key, False)}, '
                    f'current={getattr(args, key)}')
        saved_generated_seq_len = int(
            saved_args.get(
                'generated_seq_len', saved_args.get('seq_len', args.seq_len)))
        if saved_generated_seq_len != args.generated_seq_len:
            raise ValueError(
                'Resume checkpoint uses a different diffusion target length: '
                f'checkpoint={saved_generated_seq_len}, '
                f'current={args.generated_seq_len}')
        saved_time_scale = float(saved_args.get('time_scale', 1.0))
        if saved_time_scale != args.time_scale:
            raise ValueError(
                f'Resume mismatch for time_scale: '
                f'checkpoint={saved_time_scale}, current={args.time_scale}')
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema']); ema = ema.to(device)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        print(f'  Restored optimizer/scheduler state')
        global_step = ckpt.get('global_step', 0)
        start_epoch = ckpt.get('epoch', 0) + 1
        restored_latent_scale = ckpt.get('latent_scale')
        if rank == 0:
            restore_rng_state(ckpt.get('rng_state'))
        else:
            set_seed(args.seed + rank + global_step * 1009)

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=args.text_cond,
        )
        flow.model = model

    # Eval frames
    # Compute rescale factor from first few batches
    latent_scale = float(restored_latent_scale or 1.0)
    if args.rescale and restored_latent_scale is None:
        if main_process: print('  Computing latent rescale factor...')
        scale_stats = torch.zeros(2, device=device, dtype=torch.float64)
        with torch.no_grad():
            for i, eb in enumerate(dataloader):
                if i >= 10: break
                frames = eb['frames'].to(device=device, dtype=torch.bfloat16)
                tokens_list, psi = encoder(frames)
                tokens_list = strip_special_tokens(tokens_list, psi)
                _, z_g_flat = tokenizer(tokens_list)
                scale_target = z_g_flat
                if args.i0_residual:
                    i0_tokens, i0_psi = encoder(frames[:, :1])
                    i0_tokens = strip_special_tokens(i0_tokens, i0_psi)
                    _, i0_flat = tokenizer(i0_tokens)
                    scale_target = (
                        z_g_flat[:, 1:]
                        - i0_flat.expand(
                            -1, z_g_flat.shape[1] - 1, -1, -1))
                scale_stats[0] += scale_target.float().std().double()
                scale_stats[1] += 1
        if use_ddp:
            dist.all_reduce(scale_stats, op=dist.ReduceOp.SUM)
        if scale_stats[1].item() == 0:
            raise RuntimeError('Cannot compute latent scale from an empty dataloader')
        latent_std = (scale_stats[0] / scale_stats[1]).item()
        latent_scale = 1.0 / max(latent_std, 1e-8)
        if main_process:
            print(f'  Latent scale: {latent_scale:.2f} '
                  f'(diffusion target std={1.0/latent_scale:.4f} → 1.0)')
    elif args.rescale and main_process:
        print(f'  Restored latent scale: {latent_scale:.6f}')

    eval_frames = None
    if main_process:
        eval_csv = args.eval_csv or args.csv
        eval_video_root = args.eval_video_root or args.video_root
        if not args.eval_csv:
            print('  [WARN] --eval_csv not set; sampling uses training data')
        eval_dataset = SpatialVidDataset(
            csv_path=eval_csv,
            video_root=eval_video_root,
            seq_len=args.seq_len,
            target_size=args.target_size,
            max_videos=1,
            num_frames_per_video=args.num_frames_per_video,
            temporal_jitter=False,
            max_frame_span=args.max_frame_span,
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_fn)
        eval_frames = next(iter(eval_loader))['frames'].clone()

    writer = None
    if main_process:
        writer = SummaryWriter(
            log_dir=os.path.join(args.output_dir, 'tb'),
            purge_step=global_step if global_step > 0 else None,
        )
        print(f'\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch')
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')

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
                i0_cond = None
                if args.i0_condition:
                    i0_tokens, i0_psi = encoder(frames[:, :1])
                    i0_tokens = strip_special_tokens(i0_tokens, i0_psi)
                    _, i0_flat = tokenizer(i0_tokens)
                    i0_cond = (i0_flat * latent_scale).to(dtype=dtype)
                diffusion_target = z_g_flat
                if args.i0_residual:
                    diffusion_target = (
                        z_g_flat[:, 1:]
                        - i0_flat.expand(
                            -1, z_g_flat.shape[1] - 1, -1, -1))
                x1 = (diffusion_target * latent_scale).to(dtype=dtype)

            # Text conditioning
            text_emb = None
            if clip_encoder is not None:
                text_emb = clip_encoder(batch['caption']).to(device=device, dtype=dtype)
                if args.cfg_dropout > 0:
                    mask = torch.rand(x1.shape[0], device=device) < args.cfg_dropout
                    text_emb = text_emb * (~mask).view(-1, 1, 1).to(dtype=text_emb.dtype)

            # Decoder auxiliary loss
            dec_loss = x1.new_zeros(())
            use_decoder_aux = args.decoder_aux and (batch_idx % args.recon_every == 0)
            flow_out = flow.compute_loss(
                x1, cond=i0_cond, text_emb=text_emb, return_outputs=use_decoder_aux)
            if use_decoder_aux:
                flow_loss = flow_out['loss']
                loss = flow_loss
                x1_pred = flow_out['x1_pred']
                # Decode predicted z_g (decoder is float32)
                if args.i0_residual:
                    future_flat = (
                        x1_pred / latent_scale
                        + i0_flat.expand_as(x1_pred))
                    future_grid = future_flat.reshape(
                        z_g.shape[0], args.generated_seq_len,
                        args.latent_grid, args.latent_grid, args.latent_dim)
                    i0_grid = i0_flat.reshape(
                        z_g.shape[0], 1, args.latent_grid,
                        args.latent_grid, args.latent_dim)
                    z_g_pred = torch.cat(
                        [i0_grid, future_grid], dim=1).float()
                else:
                    z_g_pred = (
                        x1_pred / latent_scale).reshape(*z_g.shape).float()
                if i0_decoder is not None:
                    with torch.no_grad():
                        i0_feats = i0_app_cnn(frames[:, 0])
                    with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                        result = i0_decoder(z_g_pred, i0_feats)
                    preds, _ = result
                else:
                    with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                        result = decoder(z_g_pred)
                    if decoder.output_depth:
                        preds, _, _, _ = result
                    else:
                        preds, _ = result
                # decoder outputs BHWC [B,S,H,W,3], frames is BCSHW [B,S,3,H,W]
                recon = preds[..., :3].permute(0, 1, 4, 2, 3).contiguous().clamp(0, 1).float()
                target = frames.clamp(0, 1).float()  # already [B,S,3,H,W]
                dec_loss = F.l1_loss(recon, target)
                loss = loss + args.recon_weight * dec_loss
            else:
                flow_loss = flow_out
                loss = flow_loss

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
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer); scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model.module if use_ddp else model)
                scheduler.step()
                global_step += 1
                pbar.set_postfix(
                    loss=f'{loss_val:.4f}',
                    dec=f'{dec_loss.item():.4f}' if dec_loss.item() > 0 else '',
                )

                if main_process and global_step % args.log_every == 0:
                    train_metrics = {
                        'step': global_step,
                        'epoch': epoch,
                        'train/loss': loss_val,
                        'train/flow_loss': flow_loss.item(),
                        'train/decoder_loss': dec_loss.item(),
                        'train/target_mean': x1.float().mean().item(),
                        'train/target_std': x1.float().std().item(),
                        'train/grad_norm': float(grad_norm),
                        'train/lr': optimizer.param_groups[0]['lr'],
                    }
                    if writer:
                        for name, value in train_metrics.items():
                            if name.startswith('train/'):
                                writer.add_scalar(name, value, global_step)
                    append_metrics(metrics_path, train_metrics)

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
            append_metrics(metrics_path, {
                'step': global_step,
                'epoch': epoch,
                'train/epoch_loss': avg_loss,
            })

        base_model = model.module if use_ddp else model
        save_due = (epoch + 1) % args.save_every == 0
        eval_due = (epoch + 1) % args.eval_every == 0
        if main_process and save_due:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:04d}.pt')
            atomic_torch_save(
                checkpoint_payload(
                    base_model, ema, optimizer, scheduler, scaler,
                    global_step, epoch, latent_scale, args),
                save_path,
            )
            print(f'  Saved: {save_path}')

        should_eval = save_due or eval_due or epoch == args.epochs - 1
        if use_ddp and should_eval:
            dist.barrier()
        if main_process and should_eval and eval_frames is not None:
            train_state = {
                key: value.clone()
                for key, value in base_model.state_dict().items()}
            try:
                base_model.load_state_dict({k: v for k, v in ema.state_dict().items()})
                metrics = diffusion_sample(base_model,
                                i0_decoder if i0_decoder is not None else decoder,
                                tokenizer, encoder, eval_frames, device,
                                os.path.join(args.output_dir, 'samples'),
                                global_step, latent_scale,
                                i0_app_cnn=i0_app_cnn)
                if metrics:
                    if writer:
                        for name, value in metrics.items():
                            writer.add_scalar(
                                f'eval/{name}', value, global_step)
                        writer.flush()
                    append_metrics(metrics_path, {
                        'step': global_step,
                        'epoch': epoch,
                        **{
                            f'eval/{name}': value
                            for name, value in metrics.items()
                        },
                    })
                    print(
                        '  Eval: reference PSNR='
                        f'{metrics["reference_psnr"]:.1f} dB, '
                        f'latent_std={metrics["latent_std"]:.4f}')
            except Exception as e:
                print(f'  [WARN] Eval sampling failed: {e}')
            finally:
                base_model.load_state_dict(train_state)
                del train_state
        if use_ddp and should_eval:
            dist.barrier()

    if main_process:
        base_model = model.module if use_ddp else model
        final_path = os.path.join(args.output_dir, 'checkpoint_final.pt')
        atomic_torch_save(
            checkpoint_payload(
                base_model, ema, optimizer, scheduler, scaler,
                global_step, args.epochs - 1, latent_scale, args),
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
