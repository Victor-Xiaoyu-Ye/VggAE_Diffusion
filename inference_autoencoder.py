#!/usr/bin/env python3
"""Autoencoder inference: reconstruct video from compact latent z_g.

Supports both v1 (exp-1) and v2 (big) checkpoints via auto-detection.

Usage:
  bash scripts/10k/inference_autoencoder.sh
"""

import argparse
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from streamvggt.models.streamvggt import StreamVGGT
from models.generative_tokenizer import GenerativeTokenizer
from models.compact_decoder import CompactDecoder
from data.video_dataset import SpatialVidDataset, collate_fn
from data.loader_utils import multiprocessing_loader_kwargs
from data.token_utils import strip_special_tokens
from utils.device import (
    configure_backend_compatibility,
    get_device,
    get_device_name,
    resolve_dtype,
)


def parse_args():
    p = argparse.ArgumentParser(description='Autoencoder inference')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--encoder_ckpt', type=str, required=True)
    p.add_argument('--video_path', type=str, default='')
    p.add_argument('--csv', type=str, default='')
    p.add_argument('--video_root', type=str, default='')
    p.add_argument('--num_videos', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--clip_duration_seconds', type=float, default=0.0)
    p.add_argument('--target_size', type=int, default=518)
    p.add_argument('--output_dir', type=str, default='outputs/autoencoder_inference')
    p.add_argument('--compute_psnr', dest='compute_psnr', action='store_true')
    p.add_argument('--no_compute_psnr', dest='compute_psnr', action='store_false')
    p.set_defaults(compute_psnr=True)
    p.add_argument('--use_ema', dest='use_ema', action='store_true')
    p.add_argument('--no_use_ema', dest='use_ema', action='store_false')
    p.set_defaults(use_ema=True)
    p.add_argument('--latent_dim', type=int, default=512)
    p.add_argument('--latent_grid', type=int, default=18)
    p.add_argument('--token_dim', type=int, default=2048)
    p.add_argument('--levels', type=int, nargs='+', default=[4, 11, 17, 23])
    p.add_argument('--input_grid', type=int, default=37)
    return p.parse_args()


def detect_version(decoder_state):
    """Detect decoder version from checkpoint keys.
    v1 (exp-1): up0.0.conv... (Sequential-style)
    v2 (big):   up0.upsample.conv... (UpsampleStage-style)
    """
    for k in decoder_state.keys():
        if k.startswith('up0.'):
            if 'upsample' in k:
                return 'v2'
            else:
                return 'v1'
    return 'v1'


def build_v1_decoder(base_dim, output_depth, args):
    """exp-1 architecture: bilinear, 1 ResBlock/stage, 1 temporal."""
    return CompactDecoder(
        latent_dim=args.latent_dim, base_dim=base_dim, output_dim=3,
        output_depth=output_depth, img_size=args.target_size,
        latent_grid=args.latent_grid,
        num_resblocks=1, use_pixel_shuffle=False, num_temporal_blocks=1,
        version='v1', use_checkpoint=False,
    )


def build_v2_decoder(base_dim, output_depth, args):
    """Big architecture: pixel-shuffle, 2 ResBlocks/stage, 2 temporal."""
    return CompactDecoder(
        latent_dim=args.latent_dim, base_dim=base_dim, output_dim=3,
        output_depth=output_depth, img_size=args.target_size,
        latent_grid=args.latent_grid,
        num_resblocks=2, use_pixel_shuffle=True, num_temporal_blocks=2,
        version='v2', use_checkpoint=False,
    )


def split_ema_state(ema_state):
    tokenizer_state = {}
    decoder_state = {}
    for key, value in ema_state.items():
        if key.startswith('0.'):
            tokenizer_state[key[2:]] = value
        elif key.startswith('1.'):
            decoder_state[key[2:]] = value
    return tokenizer_state, decoder_state


def load_model(args, device, compute_dtype):
    ckpt = torch.load(
        args.checkpoint, map_location='cpu', weights_only=False)
    checkpoint_args = ckpt.get('args', {})
    for name in (
            'latent_dim', 'latent_grid', 'seq_len', 'target_size',
            'token_dim', 'input_grid'):
        if name in checkpoint_args:
            setattr(args, name, int(checkpoint_args[name]))
    if 'levels' in checkpoint_args:
        args.levels = [int(level) for level in checkpoint_args['levels']]
    print(
        'Checkpoint config: '
        f'latent={args.latent_dim}x{args.latent_grid}x{args.latent_grid}, '
        f'seq_len={args.seq_len}, target_size={args.target_size}, '
        f'levels={args.levels}')

    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    state = torch.load(args.encoder_ckpt, map_location='cpu')
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device=device, dtype=compute_dtype).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    tokenizer = GenerativeTokenizer(
        token_dim=args.token_dim, latent_dim=args.latent_dim,
        latent_grid=args.latent_grid, levels=args.levels,
        seq_len=args.seq_len, input_grid=args.input_grid,
    ).to(device=device).eval()

    tokenizer.load_state_dict(ckpt['tokenizer'])

    # Auto-detect decoder version and base_dim from checkpoint
    dec_state = ckpt['decoder']
    version = detect_version(dec_state)
    output_depth = any(key.startswith('depth_head.') for key in dec_state)
    # Infer base_dim from stem.0.conv weight shape
    base_dim = dec_state['stem.0.conv.weight'].shape[0] // 2
    print(
        f'Detected: version={version}, base_dim={base_dim}, '
        f'output_depth={output_depth}')

    if version == 'v1':
        decoder = build_v1_decoder(base_dim, output_depth, args)
    else:
        decoder = build_v2_decoder(base_dim, output_depth, args)
    decoder = decoder.to(device=device)

    decoder.load_state_dict(dec_state)

    if args.use_ema and 'ema' in ckpt:
        tokenizer_ema, decoder_ema = split_ema_state(ckpt['ema'])
        if tokenizer_ema and decoder_ema:
            tokenizer.load_state_dict(tokenizer_ema, strict=False)
            decoder.load_state_dict(decoder_ema, strict=False)
            print('Loaded EMA weights for tokenizer and decoder.')
        else:
            print('[WARN] EMA state found but could not split ModuleList keys; '
                  'using raw checkpoint weights.')
    elif args.use_ema:
        print('[WARN] --use_ema requested but checkpoint has no ema field; '
              'using raw checkpoint weights.')

    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    return encoder, tokenizer, decoder


@torch.no_grad()
def reconstruct(encoder, tokenizer, decoder, frames, device, compute_dtype,
                device_type):
    with torch.amp.autocast(
            device_type=device_type, dtype=compute_dtype):
        tokens_list, psi = encoder(
            frames.to(device=device, dtype=compute_dtype))
        tokens_list = strip_special_tokens(tokens_list, psi)
        z_g, _ = tokenizer(tokens_list)
        result = decoder(z_g)
    if decoder.output_depth:
        preds, depth, conf, _ = result
    else:
        preds, _ = result
    return preds[..., :3].clamp(0, 1)


def save_comparison_grid(original, reconstructed, out_path):
    S, H, W, C = original.shape
    rows = []
    for s in range(S):
        row = torch.cat([original[s], reconstructed[s]], dim=1)
        rows.append(row)
    grid = torch.cat(rows, dim=0)
    grid_np = (grid.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(grid_np).save(out_path)


def main():
    args = parse_args()
    device_type = get_device_name()
    configure_backend_compatibility(device_type)
    if device_type == 'cpu':
        raise RuntimeError('Autoencoder inference requires an accelerator')
    device = get_device(0)
    compute_dtype = resolve_dtype('fp16')

    print('Loading model...')
    encoder, tokenizer, decoder = load_model(args, device, compute_dtype)
    n_params = sum(p.numel() for p in tokenizer.parameters()) + \
               sum(p.numel() for p in decoder.parameters())
    print(f'Model params: {n_params/1e6:.1f}M')

    os.makedirs(args.output_dir, exist_ok=True)

    if args.video_path:
        from utils.video_io import read_video_frames
        print(f'Processing: {args.video_path}')
        frames = read_video_frames(args.video_path, args.seq_len, args.target_size)
        frames_t = frames.unsqueeze(0)
        original = frames_t[0].permute(0, 2, 3, 1)
        recon = reconstruct(
            encoder, tokenizer, decoder, frames_t, device,
            compute_dtype, device_type)
        recon = recon[0].cpu()
        out_path = os.path.join(args.output_dir, 'reconstruction.png')
        save_comparison_grid(original, recon, out_path)
        print(f'Saved: {out_path}')
        if args.compute_psnr:
            mse = F.mse_loss(recon, original).item()
            psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
            print(f'PSNR: {psnr:.2f} dB')

    elif args.csv:
        dataset = SpatialVidDataset(
            csv_path=args.csv, video_root=args.video_root,
            seq_len=args.seq_len, target_size=args.target_size,
            max_videos=args.num_videos, num_frames_per_video=args.seq_len,
            clip_duration_seconds=args.clip_duration_seconds,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
            **multiprocessing_loader_kwargs(args.num_workers),
        )
        psnr_list = []
        records = []
        for i, batch in enumerate(tqdm(loader, desc='Reconstructing')):
            if i >= args.num_videos:
                break
            frames_t = batch['frames']
            original = frames_t[0].permute(0, 2, 3, 1)
            recon = reconstruct(
                encoder, tokenizer, decoder, frames_t, device,
                compute_dtype, device_type)
            recon = recon[0].cpu()
            out_path = os.path.join(args.output_dir, f'recon_{i:04d}.png')
            save_comparison_grid(original, recon, out_path)
            if args.compute_psnr:
                mse = F.mse_loss(recon, original).item()
                psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
                psnr_list.append(psnr)
            else:
                psnr = None
            records.append({
                'index': i,
                'requested_video_id': batch['requested_video_id'][0],
                'video_id': batch['video_id'][0],
                'decode_replacements': int(batch['decode_replacements']),
                'decode_errors': batch['decode_errors'],
                'psnr': float(psnr) if psnr is not None else None,
                'image': os.path.basename(out_path),
            })
        if psnr_list:
            avg_psnr = np.mean(psnr_list)
            print(f'\nAverage PSNR over {len(psnr_list)} videos: {avg_psnr:.2f} dB')
        else:
            avg_psnr = None
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump({
                'checkpoint': args.checkpoint,
                'num_videos': len(records),
                'avg_psnr': (
                    float(avg_psnr) if avg_psnr is not None else None),
                'decode_replacements': sum(
                    record['decode_replacements'] for record in records),
                'videos': records,
            }, f, indent=2)
    else:
        print('ERROR: specify --video_path or --csv')
        sys.exit(1)


if __name__ == '__main__':
    main()
