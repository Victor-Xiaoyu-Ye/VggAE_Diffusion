#!/usr/bin/env python3
"""Feature-space reconstruction probes (E1 / E2 / baseline).

Diagnostic script for the project's core open question: does the frozen
StreamVGGT feature space carry enough information (in particular RGB
high-frequency) to reconstruct video at high PSNR? The answer determines
whether to build a two-stream latent (z_geo + z_app from shallow VGGT
levels) or to keep a single geometry latent and let the decoder
hallucinate RGB high-frequency via perceptual + adversarial training.

Modes
-----
--mode raw
    Take selected --levels (default 4 11 17 23), concat their raw 2048-dim
    patch tokens along channel at the native 37x37 grid, project to
    --proj_dim, decode 37 -> target_size with a strong resize-conv decoder.
    No tokenizer compression. This is the RGB reconstruction CEILING of the
    frozen feature space. If this cannot exceed ~23 PSNR, the features do
    not carry RGB high-frequency and a two-stream latent from VGGT levels
    will not help.

--mode per_level
    Run ONE level at a time. For each level in --levels, project 2048 ->
    latent_dim, optionally spatial-compress 37 -> --latent_grid, decode.
    Invoke once per level (loop in shell) or use --levels with a single
    value. Measures which level carries the most reconstructable info and
    tests the "deep LayerNorm flattens high-frequency" hypothesis.

--mode compressed
    Full GenerativeTokenizer + CompactDecoder baseline (current pipeline)
    for reference. Uses the same train_autoencoder loss path.

Outputs
-------
- metrics.jsonl with psnr / l1 / lpips / mse per eval epoch
- samples/ reconstruction grids (top=original, bottom=reconstruction)
- checkpoint_latest.pt (tokenizer/probe weights, optimizer, RNG, args)

The probe intentionally shares the dataset / encoder / loss helpers with
train_autoencoder.py so conclusions transfer to the main pipeline.
"""

import argparse
import json
import os
import random
from contextlib import nullcontext as _nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from streamvggt.models.streamvggt import StreamVGGT
from models.generative_tokenizer import GenerativeTokenizer
from models.compact_decoder import CompactDecoder
from data.video_dataset import SpatialVidDataset, collate_fn
from data.loader_utils import multiprocessing_loader_kwargs
from data.token_utils import strip_special_tokens
from utils.training import (
    EMA,
    ThroughputMeter,
    count_latent_tokens,
    append_metrics,
    atomic_torch_save,
    build_optimizer,
    build_scheduler,
    capture_rng_state,
    restore_rng_state,
)
from utils.distributed import setup_ddp, is_main_process
from utils.device import (
    configure_backend_compatibility,
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
        import lpips
        _lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
        for p in _lpips_fn.parameters():
            p.requires_grad_(False)
    return _lpips_fn


# ---------------------------------------------------------------------------
# Probe decoder: flexible input grid + upsample type
# ---------------------------------------------------------------------------

def _gn(ch):
    for g in [32, 16, 8, 4, 1]:
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.norm = _gn(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _ResBlock(nn.Module):
    def __init__(self, ch, kernel=3):
        super().__init__()
        self.block1 = _ConvBlock(ch, ch, kernel)
        self.block2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel, padding=kernel // 2, bias=False), _gn(ch))
        self.act = nn.SiLU()

    def forward(self, x):
        r = x
        x = self.block1(x)
        x = self.block2(x)
        return self.act(x + r)


class _ResizeConvStage(nn.Module):
    """Bilinear 2x upsample + conv (resize-conv, checkerboard-free)."""

    def __init__(self, in_ch, out_ch, num_resblocks=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            _ConvBlock(in_ch, out_ch, 3),
        )
        self.res = nn.Sequential(*[_ResBlock(out_ch) for _ in range(num_resblocks)])

    def forward(self, x):
        return self.res(self.up(x))


class _PixelShuffleStage(nn.Module):
    def __init__(self, in_ch, out_ch, num_resblocks=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, padding=1), nn.PixelShuffle(2))
        self.res = nn.Sequential(*[_ResBlock(out_ch) for _ in range(num_resblocks)])

    def forward(self, x):
        return self.res(self.up(x))


class ProbeDecoder(nn.Module):
    """Flexible decoder for feature-space reconstruction probes.

    Handles arbitrary input_grid (e.g. 37 for raw features, 18 for
    compressed latents) by computing the number of 2x upsample stages
    needed, capped by ``max_feature_grid`` so the largest stored activation
    stays bounded; the remaining resolution is covered by a final bilinear
    interpolate to ``target_size``.

    Memory management (the E1 probe OOMs without these):
    - ``use_checkpoint``: gradient checkpoint each upsample stage and the
      final refine so activations are recomputed in backward instead of
      stored. Matches the production CompactDecoder behaviour.
    - ``frames_chunk_size``: decode the S frames in chunks to bound the
      per-chunk activation memory.

    Args:
        in_dim: input channel dim (proj_dim for raw, latent_dim for others)
        input_grid: spatial size of the input feature map (37 or 18)
        target_size: output RGB resolution (518)
        base_dim: decoder channel width
        num_resblocks: ResBlocks per upsample stage
        use_pixel_shuffle: False = resize-conv (default, avoids checkerboard)
        num_temporal_blocks: temporal attention blocks (0 disables)
        max_feature_grid: largest feature map side before final interpolate.
            Lower values save activation memory at the cost of some spatial
            precision in the last few conv layers.
        use_checkpoint: gradient checkpoint stages + final refine.
    """

    def __init__(self, in_dim, input_grid=37, target_size=518,
                 base_dim=384, num_resblocks=2, use_pixel_shuffle=False,
                 num_temporal_blocks=1, max_feature_grid=296,
                 use_checkpoint=True):
        super().__init__()
        import math
        self.input_grid = input_grid
        self.target_size = target_size
        self.base_dim = base_dim
        self.use_checkpoint = use_checkpoint

        C0 = base_dim * 2
        C1 = base_dim
        C2 = base_dim
        C3 = base_dim // 2
        C4 = base_dim // 4

        # Number of 2x stages so 2^stages * input_grid >= target_size.
        full_stages = max(1, int(math.ceil(math.log2(target_size / input_grid))))
        # Cap by max_feature_grid so the largest stored activation is bounded;
        # the remaining resolution is handled by a final bilinear interpolate.
        cap = 0
        g = input_grid
        for s in range(full_stages):
            g_next = g * 2
            if g_next > max_feature_grid:
                break
            cap = s + 1
            g = g_next
        self.num_stages = max(cap, 1)
        self.last_feature_grid = input_grid * (2 ** self.num_stages)

        stage_dims = [C0, C1, C2, C3, C4]
        Stage = _PixelShuffleStage if use_pixel_shuffle else _ResizeConvStage

        self.stem = nn.Sequential(
            _ConvBlock(in_dim, C0, 3), _ResBlock(C0),
            _ConvBlock(C0, C0, 3), _ResBlock(C0))

        self.stages = nn.ModuleList([
            Stage(C0 if i == 0 else stage_dims[i - 1], stage_dims[i], num_resblocks)
            for i in range(self.num_stages)
        ])

        self.temporal_blocks = nn.ModuleList()
        # Attach temporal attention at the first stages (lowest resolution).
        for i in range(min(num_temporal_blocks, self.num_stages)):
            self.temporal_blocks.append(_TemporalAttn(stage_dims[i]))

        last_dim = stage_dims[self.num_stages - 1]
        self.final_refine = nn.Sequential(
            _ConvBlock(last_dim, C4, 3),
            _ResBlock(C4), _ConvBlock(C4, C4, 3), _ResBlock(C4))
        self.rgb_head = nn.Sequential(
            _ConvBlock(C4, 64, 3), nn.Conv2d(64, 3, 3, padding=1), nn.Sigmoid())

    def _stage_forward(self, stage, x):
        if self.use_checkpoint and self.training:
            return torch_checkpoint(stage, x, use_reentrant=False)
        return stage(x)

    def _decode_chunk(self, feat):
        """feat: [B, S, input_grid, input_grid, in_dim] -> RGB [B, S, H, W, 3]."""
        B, S, H, W, C = feat.shape
        x = feat.permute(0, 1, 4, 2, 3).contiguous().reshape(B * S, C, H, W)
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = self._stage_forward(stage, x)
            if i < len(self.temporal_blocks):
                x = self.temporal_blocks[i](x, B, S)
        x = self._stage_forward(self.final_refine, x)
        rgb = self.rgb_head(x)
        if rgb.shape[-1] != self.target_size:
            rgb = F.interpolate(
                rgb, size=(self.target_size, self.target_size),
                mode='bilinear', align_corners=False)
        rgb = rgb.reshape(B, S, 3, self.target_size, self.target_size)
        return rgb.permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, feat, frames_chunk_size=None):
        """feat: [B, S, input_grid, input_grid, in_dim] -> RGB [B, S, H, W, 3].

        If frames_chunk_size is set and S > frames_chunk_size, decode in
        temporal chunks to bound activation memory, then concatenate.
        """
        B, S = feat.shape[0], feat.shape[1]
        if frames_chunk_size is None or S <= frames_chunk_size:
            return self._decode_chunk(feat)
        outs = []
        for start in range(0, S, frames_chunk_size):
            end = min(start + frames_chunk_size, S)
            outs.append(self._decode_chunk(feat[:, start:end]))
        return torch.cat(outs, dim=1)


class _TemporalAttn(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x, B, S):
        BS, C, H, W = x.shape
        x_t = x.reshape(B, S, C, H * W).permute(0, 3, 1, 2).contiguous().reshape(
            B * H * W, S, C)
        x_t = self.norm(x_t)
        x_t, _ = self.attn(x_t, x_t, x_t)
        x = x_t.reshape(B, H * W, S, C).permute(0, 2, 3, 1).contiguous().reshape(
            B * S, C, H, W)
        return x


# ---------------------------------------------------------------------------
# Feature extractors (raw / per_level / compressed)
# ---------------------------------------------------------------------------

class RawFeatureProjector(nn.Module):
    """Concat selected VGGT levels at native 37x37 grid, project to proj_dim.

    No spatial compression, no temporal mixing. Used for E1 (raw ceiling).
    """

    def __init__(self, levels=(4, 11, 17, 23), token_dim=2048, proj_dim=512):
        super().__init__()
        self.levels = list(levels)
        in_dim = token_dim * len(self.levels)
        # Per-level LayerNorm + concat + linear project to proj_dim.
        self.norms = nn.ModuleDict({
            str(lvl): nn.LayerNorm(token_dim) for lvl in self.levels})
        self.proj = nn.Linear(in_dim, proj_dim)

    def forward(self, tokens_list):
        """tokens_list: list of 24 tensors [B, S, N, 2048] (special tokens stripped).

        Returns [B, S, 37, 37, proj_dim].
        """
        lvl_feats = [self.norms[str(lvl)](tokens_list[lvl]) for lvl in self.levels]
        B, S, N, _ = lvl_feats[0].shape
        grid = int(N ** 0.5)
        concat = torch.cat(lvl_feats, dim=-1)  # [B, S, N, 2048*L]
        z = self.proj(concat)                  # [B, S, N, proj_dim]
        z = z.reshape(B, S, grid, grid, -1)
        return z


class PerLevelProjector(nn.Module):
    """Single VGGT level -> latent_dim, optional spatial compress to latent_grid.

    Used for E2 (per-level info content). One level per module instance.
    """

    def __init__(self, level, token_dim=2048, latent_dim=512,
                 latent_grid=18, input_grid=37):
        super().__init__()
        self.level = level
        self.latent_dim = latent_dim
        self.latent_grid = latent_grid
        self.input_grid = input_grid
        self.norm = nn.LayerNorm(token_dim)
        self.proj = nn.Linear(token_dim, latent_dim)
        # Pre-pool conv + adaptive pool when compressing.
        self.compress = (latent_grid != input_grid)
        if self.compress:
            self.pre_pool = nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
                _gn(latent_dim), nn.SiLU())

    def forward(self, tokens_list):
        x = self.norm(tokens_list[self.level])     # [B, S, N, 2048]
        B, S, N, _ = x.shape
        grid = int(N ** 0.5)
        z = self.proj(x).reshape(B, S, grid, grid, self.latent_dim)
        if self.compress:
            z = z.permute(0, 1, 4, 2, 3).reshape(B * S, self.latent_dim, grid, grid)
            z = self.pre_pool(z)
            z = F.adaptive_avg_pool2d(z, (self.latent_grid, self.latent_grid))
            z = z.reshape(B, S, self.latent_dim, self.latent_grid, self.latent_grid)
            z = z.permute(0, 1, 3, 4, 2).contiguous()
        return z


# ---------------------------------------------------------------------------
# Losses (shared with train_autoencoder)
# ---------------------------------------------------------------------------

def image_gradient_loss(pred, target):
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def temporal_consistency_loss(pred, target):
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    return F.l1_loss(pred_diff, target_diff)


def latent_regularization(z_flat):
    values = z_flat.float().reshape(-1, z_flat.shape[-1])
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    return mean.square().mean() + (std - 1).square().mean()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Feature-space reconstruction probe')

    # Mode
    p.add_argument('--mode', type=str, default='raw',
                   choices=['raw', 'per_level', 'compressed'],
                   help='raw=E1 ceiling, per_level=E2 single level, '
                        'compressed=baseline reference')
    p.add_argument('--probe_name', type=str, default='e1_raw',
                   help='Experiment name (used in output dir + metrics).')

    # Data
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--video_root', type=str, required=True)
    p.add_argument('--eval_csv', type=str, default='')
    p.add_argument('--eval_video_root', type=str, default='')
    p.add_argument('--max_videos', type=int, default=0)

    # Encoder
    p.add_argument('--encoder_ckpt', type=str, required=True)
    p.add_argument('--levels', type=int, nargs='+', default=[4, 11, 17, 23])
    p.add_argument('--token_dim', type=int, default=2048)

    # Raw / per_level model
    p.add_argument('--proj_dim', type=int, default=512,
                   help='raw mode: channel dim after projecting concat features')
    p.add_argument('--latent_dim', type=int, default=512)
    p.add_argument('--latent_grid', type=int, default=18)
    p.add_argument('--per_level', type=int, default=-1,
                   help='per_level mode: single level index. -1 = use --levels[0].')

    # Decoder
    p.add_argument('--decoder_base_dim', type=int, default=384)
    p.add_argument('--decoder_num_resblocks', type=int, default=2)
    p.add_argument('--decoder_use_pixel_shuffle', type=int, default=0,
                   choices=[0, 1],
                   help='0=resize-conv (default, avoids checkerboard), '
                        '1=PixelShuffle.')
    p.add_argument('--num_temporal_blocks', type=int, default=1)
    p.add_argument('--max_feature_grid', type=int, default=296,
                   help='Largest decoder feature map side before final '
                        'bilinear interpolate to target_size. Lower saves '
                        'activation memory. 296 keeps 37->74->148->296 then '
                        'interpolates to 518, avoiding the 592x592 stage.')
    p.add_argument('--use_checkpoint', type=int, default=1, choices=[0, 1],
                   help='Gradient checkpoint decoder stages + final refine. '
                        'On by default; matches production CompactDecoder.')
    p.add_argument('--frames_chunk_size', type=int, default=4,
                   help='Decode the S frames in chunks of this size to bound '
                        'activation memory. 0 = decode all S frames at once.')

    # Losses
    p.add_argument('--lambda_l1', type=float, default=1.0)
    p.add_argument('--lambda_mse', type=float, default=0.5,
                   help='Per-pixel MSE weight; L1 + MSE together drive '
                        'per-pixel reconstruction.')
    p.add_argument('--lambda_lpips', type=float, default=1.0)
    p.add_argument('--lambda_grad', type=float, default=0.05)
    p.add_argument('--lambda_temporal', type=float, default=0.05)
    p.add_argument('--lambda_latent_reg', type=float, default=0.0,
                   help='Latent N(0,1) reg. 0 by default for probes; the '
                        'compressed baseline sets this when needed.')

    # Training
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accum_steps', type=int, default=8)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--warmup_steps', type=int, default=200)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--dtype', type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'])
    p.add_argument('--latent_noise_std', type=float, default=0.0,
                   help='Optional noise on the projected feature for decoder '
                        'robustness; 0 by default for clean probes.')

    # Data loading
    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--target_size', type=int, default=518)
    p.add_argument('--num_frames_per_video', type=int, default=8)
    p.add_argument('--max_frame_span', type=int, default=0)
    p.add_argument('--clip_duration_seconds', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--decode_retries', type=int, default=8)

    # Eval / io
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--eval_every', type=int, default=2)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--local_rank', type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_recon(encoder, projector, decoder, eval_frames, device, out_dir,
               epoch, compute_dtype, device_type, mode, frames_chunk_size=None):
    import numpy as np
    from PIL import Image as PImage
    os.makedirs(out_dir, exist_ok=True)
    projector.eval(); decoder.eval()
    frames = eval_frames.to(device=device, dtype=compute_dtype)
    ctx = autocast(device_type=device_type, dtype=compute_dtype) \
        if compute_dtype != torch.float32 else _nullcontext()
    with ctx:
        tokens_list, psi = encoder(frames)
        tokens_list = strip_special_tokens(tokens_list, psi)
        if mode == 'compressed':
            z_g, _ = projector(tokens_list)
            recon, _ = decoder(z_g, frames_chunk_size=frames_chunk_size)
        else:
            feat = projector(tokens_list)
            recon = decoder(feat, frames_chunk_size=frames_chunk_size)
    recon = recon.clamp(0, 1)
    orig = frames.permute(0, 1, 3, 4, 2).clamp(0, 1)
    S = recon.shape[1]
    rows = [torch.cat([orig[0, s], recon[0, s]], dim=1) for s in range(S)]
    grid = torch.cat(rows, dim=0)
    PImage.fromarray((grid.float().cpu().numpy() * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f'epoch{epoch:04d}_grid.png'))

    mse = F.mse_loss(recon, orig).item()
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    l1 = F.l1_loss(recon, orig).item()
    metrics = {'psnr': psnr, 'l1': l1, 'mse': mse}
    try:
        lpips_fn = get_lpips(device)
        rn = recon.permute(0, 1, 4, 2, 3).reshape(-1, 3, recon.shape[2], recon.shape[3])
        on = orig.permute(0, 1, 4, 2, 3).reshape(-1, 3, orig.shape[2], orig.shape[3])
        metrics['lpips'] = lpips_fn(rn * 2 - 1, on * 2 - 1).mean().item()
    except Exception as exc:
        metrics['lpips'] = float('nan')
    projector.train(); decoder.train()
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        print(f'Probe: mode={args.mode} levels={args.levels} '
              f'proj_dim={args.proj_dim} latent_dim={args.latent_dim} '
              f'latent_grid={args.latent_grid} '
              f'pixel_shuffle={bool(args.decoder_use_pixel_shuffle)}')

    # ---- Encoder ----
    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    state = torch.load(args.encoder_ckpt, map_location='cpu')
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device=device, dtype=dtype).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    input_grid = args.target_size // 14  # 37 for 518

    # ---- Projector + decoder ----
    if args.mode == 'raw':
        projector = RawFeatureProjector(
            levels=args.levels, token_dim=args.token_dim, proj_dim=args.proj_dim
        ).to(device=device)
        decoder = ProbeDecoder(
            in_dim=args.proj_dim, input_grid=input_grid,
            target_size=args.target_size, base_dim=args.decoder_base_dim,
            num_resblocks=args.decoder_num_resblocks,
            use_pixel_shuffle=bool(args.decoder_use_pixel_shuffle),
            num_temporal_blocks=args.num_temporal_blocks,
            max_feature_grid=args.max_feature_grid,
            use_checkpoint=bool(args.use_checkpoint)).to(device=device)
        feat_dim = args.proj_dim
    elif args.mode == 'per_level':
        lvl = args.per_level if args.per_level >= 0 else args.levels[0]
        projector = PerLevelProjector(
            level=lvl, token_dim=args.token_dim, latent_dim=args.latent_dim,
            latent_grid=args.latent_grid, input_grid=input_grid
        ).to(device=device)
        # If compressed, decoder operates on latent_grid; else on input_grid.
        dec_input_grid = args.latent_grid if args.latent_grid != input_grid \
            else input_grid
        decoder = ProbeDecoder(
            in_dim=args.latent_dim, input_grid=dec_input_grid,
            target_size=args.target_size, base_dim=args.decoder_base_dim,
            num_resblocks=args.decoder_num_resblocks,
            use_pixel_shuffle=bool(args.decoder_use_pixel_shuffle),
            num_temporal_blocks=args.num_temporal_blocks,
            max_feature_grid=args.max_feature_grid,
            use_checkpoint=bool(args.use_checkpoint)).to(device=device)
        feat_dim = args.latent_dim
        if main_process:
            print(f'  per_level: probing level {lvl}, '
                  f'dec_input_grid={dec_input_grid}')
    else:  # compressed (baseline)
        projector = GenerativeTokenizer(
            token_dim=args.token_dim, latent_dim=args.latent_dim,
            latent_grid=args.latent_grid, levels=args.levels,
            seq_len=args.seq_len, input_grid=input_grid,
            disable_temporal_mixer=False,
        ).to(device=device)
        decoder = CompactDecoder(
            latent_dim=args.latent_dim, base_dim=args.decoder_base_dim,
            output_dim=3, output_depth=False, img_size=args.target_size,
            latent_grid=args.latent_grid, num_resblocks=args.decoder_num_resblocks,
            use_pixel_shuffle=bool(args.decoder_use_pixel_shuffle),
            use_checkpoint=bool(args.use_checkpoint)).to(device=device)
        feat_dim = args.latent_dim

    total_p = sum(p.numel() for p in projector.parameters()) + \
              sum(p.numel() for p in decoder.parameters())
    if main_process:
        print(f'  Projector: {sum(p.numel() for p in projector.parameters()) / 1e6:.1f}M')
        print(f'  Decoder:   {sum(p.numel() for p in decoder.parameters()) / 1e6:.1f}M')
        print(f'  Total:     {total_p / 1e6:.1f}M')

    ema = EMA(
        nn.ModuleList([projector, decoder]),
        decay=args.ema_decay, dtype=torch.float32).to(device)

    # DDP
    if use_ddp:
        projector = nn.parallel.DistributedDataParallel(
            projector, device_ids=[local_rank], output_device=local_rank)
        decoder = nn.parallel.DistributedDataParallel(
            decoder, device_ids=[local_rank], output_device=local_rank)

    optimizer = build_optimizer(
        nn.ModuleList([
            projector.module if use_ddp else projector,
            decoder.module if use_ddp else decoder]),
        lr=args.lr, wd=args.wd)

    # ---- Dataset ----
    dataset = SpatialVidDataset(
        csv_path=args.csv, video_root=args.video_root,
        seq_len=args.seq_len, target_size=args.target_size,
        max_videos=args.max_videos,
        num_frames_per_video=args.num_frames_per_video,
        max_frame_span=args.max_frame_span,
        clip_duration_seconds=args.clip_duration_seconds,
        decode_retries=args.decode_retries)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) \
        if use_ddp else None
    loader_kwargs = multiprocessing_loader_kwargs(args.num_workers)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=device_type == 'cuda', drop_last=True, **loader_kwargs)

    steps_per_epoch = (len(dataloader) + args.accum_steps - 1) // args.accum_steps
    total_steps = args.epochs * steps_per_epoch
    warmup = min(args.warmup_steps, max(total_steps - 1, 0))
    scheduler = build_scheduler(
        optimizer, warmup_steps=warmup, total_steps=max(total_steps, 1))

    # Eval batch
    eval_frames = None
    if args.eval_csv:
        eval_dataset = SpatialVidDataset(
            csv_path=args.eval_csv,
            video_root=args.eval_video_root or args.video_root,
            seq_len=args.seq_len, target_size=args.target_size,
            max_videos=4, num_frames_per_video=args.num_frames_per_video,
            max_frame_span=args.max_frame_span,
            clip_duration_seconds=args.clip_duration_seconds,
            decode_retries=args.decode_retries)
        eval_loader = DataLoader(
            eval_dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_fn)
        eval_frames = next(iter(eval_loader))['frames'].clone()
    else:
        # Fallback: grab one batch from train as eval visual.
        eval_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=collate_fn)
        eval_frames = next(iter(eval_loader))['frames'].clone()

    # Resume
    global_step = 0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        (projector.module if use_ddp else projector).load_state_dict(ckpt['projector'])
        (decoder.module if use_ddp else decoder).load_state_dict(ckpt['decoder'])
        ema.load_state_dict(ckpt['ema']); ema = ema.to(device)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        global_step = ckpt.get('global_step', 0)
        start_epoch = ckpt.get('epoch', 0) + 1
        restore_rng_state(ckpt.get('rng_state'))
        if main_process:
            print(f'Resumed from {args.resume} at epoch {start_epoch}')

    if main_process:
        writer = SummaryWriter(
            log_dir=os.path.join(args.output_dir, 'tb'),
            purge_step=global_step if global_step > 0 else None)
    else:
        writer = None
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')

    params = list(
        (projector.module if use_ddp else projector).parameters()) + \
        list((decoder.module if use_ddp else decoder).parameters())

    if main_process:
        print(f'\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch')

    proj_mod = projector.module if use_ddp else projector
    dec_mod = decoder.module if use_ddp else decoder

    for epoch in range(start_epoch, args.epochs):
        projector.train(); decoder.train()
        if use_ddp:
            sampler.set_epoch(epoch)
        optimizer.zero_grad()
        throughput_meter = ThroughputMeter()
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}',
                    dynamic_ncols=True, disable=not main_process)

        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(device=device, dtype=dtype)
            with torch.no_grad():
                tokens_list, psi = encoder(frames)
                tokens_list = strip_special_tokens(tokens_list, psi)

            if use_amp:
                ctx = autocast(device_type=device_type, dtype=dtype)
            else:
                ctx = _nullcontext()
            chunk = args.frames_chunk_size if args.frames_chunk_size > 0 else None
            with ctx:
                if args.mode == 'compressed':
                    z_g, z_flat = projector(tokens_list)
                    noise_std = args.latent_noise_std
                    if noise_std > 0 and projector.training:
                        z_g_in = z_g + torch.randn_like(z_g) * noise_std
                    else:
                        z_g_in = z_g
                    preds, _ = dec_mod(z_g_in, frames_chunk_size=chunk)
                else:
                    feat = projector(tokens_list)
                    noise_std = args.latent_noise_std
                    if noise_std > 0 and projector.training:
                        feat = feat + torch.randn_like(feat) * noise_std
                    preds = dec_mod(feat, frames_chunk_size=chunk)
                    z_flat = feat.reshape(feat.shape[0], feat.shape[1], -1, feat.shape[-1])

            pred_rgb = preds[..., :3].permute(0, 1, 4, 2, 3).contiguous().float()
            target_rgb = frames.float().clamp(0, 1)

            l1 = F.l1_loss(pred_rgb, target_rgb)
            mse = F.mse_loss(pred_rgb, target_rgb) if args.lambda_mse > 0 \
                else pred_rgb.new_zeros(())
            grad = image_gradient_loss(
                pred_rgb.reshape(-1, *pred_rgb.shape[2:]),
                target_rgb.reshape(-1, *target_rgb.shape[2:]))
            temp = temporal_consistency_loss(pred_rgb, target_rgb)
            reg = latent_regularization(z_flat) if args.lambda_latent_reg > 0 \
                else pred_rgb.new_zeros(())

            lpips_loss = pred_rgb.new_zeros(())
            if args.lambda_lpips > 0:
                try:
                    lpips_fn = get_lpips(device)
                    p_flat = pred_rgb.reshape(-1, 3, pred_rgb.shape[-2], pred_rgb.shape[-1])
                    t_flat = target_rgb.reshape(-1, 3, target_rgb.shape[-2], target_rgb.shape[-1])
                    lpips_loss = lpips_fn(p_flat * 2 - 1, t_flat * 2 - 1).mean()
                except Exception as exc:
                    if main_process:
                        print(f'  [WARN] LPIPS disabled: {exc}')

            loss = (args.lambda_l1 * l1 + args.lambda_mse * mse +
                    args.lambda_lpips * lpips_loss + args.lambda_grad * grad +
                    args.lambda_temporal * temp + args.lambda_latent_reg * reg)

            scaled = loss / args.accum_steps
            if use_scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            if (batch_idx + 1) % args.accum_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params, args.max_grad_norm)
                    scaler.step(optimizer); scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params, args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(nn.ModuleList([proj_mod, dec_mod]))
                scheduler.step()
                global_step += 1
                throughput_meter.update(count_latent_tokens(z_flat))

                if main_process and writer and global_step % args.log_every == 0:
                    m = {
                        'step': global_step, 'epoch': epoch,
                        'probe': args.probe_name, 'mode': args.mode,
                        'train/loss': loss.item(), 'train/l1': l1.item(),
                        'train/mse': mse.item(),
                        'train/lpips': lpips_loss.item(),
                        'train/gradient': grad.item(),
                        'train/temporal': temp.item(),
                        'train/grad_norm': float(grad_norm),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'train/DI_throughput': throughput_meter.rate(),
                    }
                    for k, v in m.items():
                        if k.startswith('train/'):
                            writer.add_scalar(k, v, global_step)
                    append_metrics(metrics_path, m)
                if main_process:
                    pbar.set_postfix(
                        loss=f'{loss.item():.4f}', l1=f'{l1.item():.4f}',
                        mse=f'{mse.item():.4f}',
                        DI=throughput_meter.format())
            epoch_loss += loss.item()
            num_batches += 1

        if main_process and (epoch + 1) % args.eval_every == 0:
            metrics = eval_recon(
                encoder, proj_mod, dec_mod, eval_frames, device,
                os.path.join(args.output_dir, 'samples'), epoch,
                dtype, device_type, args.mode,
                frames_chunk_size=(args.frames_chunk_size
                                   if args.frames_chunk_size > 0 else None))
            metrics.update({
                'step': global_step, 'epoch': epoch,
                'probe': args.probe_name, 'mode': args.mode,
            })
            append_metrics(metrics_path, metrics)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if writer:
                        writer.add_scalar(f'eval/{k}', v, global_step)
            print(f'  [eval] epoch {epoch}: '
                  f'psnr={metrics["psnr"]:.2f} l1={metrics["l1"]:.4f} '
                  f'lpips={metrics.get("lpips", float("nan")):.4f}')

        if main_process and (epoch + 1) % args.save_every == 0:
            payload = {
                'projector': proj_mod.state_dict(),
                'decoder': dec_mod.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'global_step': global_step, 'epoch': epoch,
                'rng_state': capture_rng_state(), 'args': vars(args),
            }
            atomic_torch_save(
                payload,
                os.path.join(args.output_dir, f'checkpoint_epoch{epoch:04d}.pt'))
            atomic_torch_save(
                payload,
                os.path.join(args.output_dir, 'checkpoint_latest.pt'))
            print(f'  Saved checkpoint at epoch {epoch}')

    if main_process:
        print(f'\nDone. Output: {args.output_dir}')
        if writer:
            writer.flush(); writer.close()


if __name__ == '__main__':
    main()
