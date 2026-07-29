#!/usr/bin/env python3
"""Cache R7 causal tokenizer latents for diffusion training."""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from data.token_utils import strip_special_tokens
from data.video_dataset import SpatialVidDataset, collate_fn
from models.causal_dual_tokenizer import CausalDualTokenizerCore
from models.dpt_latent_decoder import CompactCompressor
from models.texture_encoder import TextureEncoder
from streamvggt.models.streamvggt import StreamVGGT
from utils.encoder_loader import load_encoder_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--video_root', required=True)
    parser.add_argument('--encoder_ckpt', required=True)
    parser.add_argument('--r7_ckpt', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seq_len', type=int, default=9)
    parser.add_argument('--target_size', type=int, default=518)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(
        args.r7_ckpt, map_location='cpu', weights_only=False)
    config = checkpoint['args']
    state = checkpoint['model']
    grid = int(config.get('latent_grid', 18))
    geo_dim = int(config.get('geo_dim', 256))
    tex_dim = int(config.get('tex_dim', 256))
    encoder = StreamVGGT(
        img_size=args.target_size, patch_size=14, embed_dim=1024)
    load_encoder_checkpoint(encoder, args.encoder_ckpt)
    encoder = encoder.to(device).eval()
    compressor = CompactCompressor(
        list(config.get('levels', [4, 11, 17, 23])),
        int(config.get('token_dim', 2048)), geo_dim, grid,
        args.target_size // 14)
    tex_encoder = TextureEncoder(
        tex_dim, grid, int(config.get('tex_base_ch', 64)),
        args.target_size, pack_mode=config.get('tex_pack', 'avgpool'))
    tokenizer = CausalDualTokenizerCore(
        geo_dim, tex_dim, int(config['geo_latent_dim']),
        int(config['tex_latent_dim']), int(config['temporal_factor']),
        int(config['temporal_depth']))
    for module, prefix in (
            (compressor, 'compressor.'), (tex_encoder, 'tex_encoder.'),
            (tokenizer, 'tokenizer.')):
        module.load_state_dict({
            key[len(prefix):]: value for key, value in state.items()
            if key.startswith(prefix)
        }, strict=False)
        module.to(device).eval()

    dataset = SpatialVidDataset(
        args.csv, args.video_root, seq_len=args.seq_len,
        target_size=args.target_size, num_frames_per_video=args.seq_len)
    loader = DataLoader(
        dataset, batch_size=1, num_workers=args.num_workers,
        collate_fn=collate_fn)
    manifest_path = os.path.join(args.output_dir, 'manifest.txt')
    with open(manifest_path, 'w') as manifest, torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            tokens, prefix_index = encoder(frames)
            stripped = strip_special_tokens(tokens, prefix_index)
            geo = compressor([token.float() for token in stripped])
            geo = geo.permute(0, 1, 3, 4, 2).contiguous()
            texture = tex_encoder(frames.float())
            latent = tokenizer.encode(geo, texture).cpu()
            path = os.path.join(
                args.output_dir, f"{batch['video_id'][0]}.pt")
            torch.save({
                'latent': latent[0],
                'video_id': batch['video_id'][0],
                'temporal_factor': config['temporal_factor'],
            }, path)
            manifest.write(path + '\n')


if __name__ == '__main__':
    main()
