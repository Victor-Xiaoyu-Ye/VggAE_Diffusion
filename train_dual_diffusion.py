#!/usr/bin/env python3
"""E7: dual-stream latent diffusion smoke probe (online encoding).

Trains a small CompactLatentDiT with OT-CFM flow matching directly on the
frozen dual-stream latent (z_geo || z_tex from the E5/R5 checkpoint), encoding
video online through the frozen StreamVGGT + compressor + tex_encoder. No
latent cache, no Wan — the question is only: can diffusion learn to produce
coherent latents in this space at all, and does the ABSOLUTE target beat the
RESIDUAL target as the diffusability spectra predict?

Arms (--target_mode):
  absolute  x1 = normalized z_1..z_7, cond = normalized z_0 (clean past).
  residual  x1 = normalized (z_t - z_0), cond = normalized z_0 (old contract).

Both arms share shapes [B, 7, G*G, geo+tex], the same DiT, and the same
normalization machinery (per-frame per-channel stats measured at startup and
frozen into the checkpoint), so the comparison isolates the parameterization.

Success metrics (logged every eval):
  eval/velocity_mse           flow loss on held-out clips
  eval/gen_std_ratio          std(sampled x1) / std(target x1) — under-
                              dispersion shows as << 1 (the old Wan symptom)
  sampled latents saved as .pt (denormalized, with z0 + stats) for offline
  RGB decoding on the H200 box (DualStreamDecoder interpolate ops are the
  NPU-risky path; keep them off the 910B).
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.video_dataset import SpatialVidDataset, collate_fn
from data.loader_utils import multiprocessing_loader_kwargs
from data.token_utils import strip_special_tokens
from models.compact_dit import CompactLatentDiT
from models.dpt_latent_decoder import CompactCompressor
from models.flow_matching import OTCFM
from models.texture_encoder import TextureEncoder
from streamvggt.models.streamvggt import StreamVGGT
from utils.device import (
    configure_backend_compatibility, get_device, get_device_name,
    manual_seed_all, resolve_dtype,
)
from utils.distributed import is_main_process, setup_ddp
from utils.encoder_loader import load_encoder_checkpoint
from utils.training import (
    EMA, ThroughputMeter, append_metrics, atomic_torch_save,
    build_optimizer, build_scheduler, capture_rng_state, restore_rng_state,
)


def parse_args():
    p = argparse.ArgumentParser(description='E7 dual-latent diffusion smoke')
    p.add_argument('--target_mode', type=str, required=True,
                   choices=['absolute', 'residual'])
    p.add_argument('--generator', type=str, default='dit',
                   choices=['dit', 'wan'],
                   help='dit = from-scratch CompactLatentDiT (E7 baseline); '
                        'wan = pretrained Wan backbone via WanCompactAdapter '
                        '(same forward interface, one-variable comparison)')
    p.add_argument('--wan_ckpt_dir', type=str, default='',
                   help='Wan2.1 checkpoint dir (required for --generator wan)')
    p.add_argument('--train_qkv_last_n', type=int, default=0,
                   help='wan: unfreeze QKV of the last N blocks; 0 = all '
                        '(affordable on 1.3B, unlike the 14B last-4 limit)')
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--video_root', type=str, required=True)
    p.add_argument('--eval_csv', type=str, required=True)
    p.add_argument('--encoder_ckpt', type=str, required=True)
    p.add_argument('--dual_ae_ckpt', type=str, required=True,
                   help='E5/R5 dual-stream checkpoint (compressor + '
                        'tex_encoder are loaded frozen from it)')

    p.add_argument('--model_dim', type=int, default=768)
    p.add_argument('--spatial_depth', type=int, default=8)
    p.add_argument('--temporal_depth', type=int, default=4)
    p.add_argument('--num_heads', type=int, default=12)

    p.add_argument('--max_steps', type=int, default=6000)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--accum_steps', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--warmup_steps', type=int, default=300)
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--stat_batches', type=int, default=16,
                   help='batches per rank used to measure latent stats')

    p.add_argument('--seq_len', type=int, default=8)
    p.add_argument('--target_size', type=int, default=518)
    p.add_argument('--clip_duration_seconds', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--decode_retries', type=int, default=8)
    p.add_argument('--max_videos', type=int, default=0)

    p.add_argument('--eval_clips', type=int, default=16)
    p.add_argument('--sample_clips', type=int, default=4)
    p.add_argument('--sample_steps', type=int, default=30)

    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--save_every', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dtype', type=str, default='bf16')
    p.add_argument('--local_rank', type=int, default=0)
    return p.parse_args()


def load_dual_ae(args, device, dtype):
    """Frozen encoder + compressor + tex_encoder + decoder from E5 ckpt."""
    from models.dual_stream_decoder import DualStreamDecoder
    ckpt = torch.load(args.dual_ae_ckpt, map_location='cpu', weights_only=False)
    ck = ckpt.get('args', {})
    grid = int(ck.get('latent_grid', 18))
    geo_dim = int(ck.get('geo_dim', 256))
    tex_dim = int(ck.get('tex_dim', 256))

    encoder = StreamVGGT(
        img_size=args.target_size, patch_size=14, embed_dim=1024)
    load_encoder_checkpoint(encoder, args.encoder_ckpt,
                            verbose=is_main_process())
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
        img_size=args.target_size, latent_grid=grid, use_checkpoint=False)
    state = ckpt['model']
    for module, prefix in ((compressor, 'compressor.'),
                           (tex_encoder, 'tex_encoder.'),
                           (decoder, 'decoder.')):
        sub = {k[len(prefix):]: v for k, v in state.items()
               if k.startswith(prefix)}
        if not sub:
            raise RuntimeError(
                f'{args.dual_ae_ckpt} has no {prefix}* keys — not an E5 '
                f'dual-stream checkpoint')
        module.load_state_dict(sub)  # strict

    compressor = compressor.to(device).eval()
    tex_encoder = tex_encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    for m in (encoder, compressor, tex_encoder, decoder):
        for prm in m.parameters():
            prm.requires_grad_(False)
    return encoder, compressor, tex_encoder, decoder, grid, geo_dim + tex_dim


@torch.no_grad()
def encode_tokens(encoder, compressor, tex_encoder, frames, dtype):
    """frames [B,S,3,H,W] -> tokens [B,S,N,geo+tex] fp32 (unnormalized)."""
    tokens_list, psi = encoder(frames.to(dtype))
    stripped = strip_special_tokens(tokens_list, psi)
    z_geo = compressor([t.float() for t in stripped])       # [B,S,C,G,G]
    z_geo = z_geo.permute(0, 1, 3, 4, 2).contiguous().float()
    z_tex = tex_encoder(frames.float()).float()             # [B,S,G,G,C]
    z = torch.cat([z_geo, z_tex], dim=-1)                   # [B,S,G,G,2C]
    B, S, G, _, C = z.shape
    return z.reshape(B, S, G * G, C)


@torch.no_grad()
def measure_stats(loader, encoder, compressor, tex_encoder, args, device,
                  dtype, use_ddp):
    """Per-frame per-channel mean/std for absolute z and residual (z_t-z_0).

    Measured over stat_batches per rank and all-reduced, then frozen into the
    checkpoint: the normalization contract must not drift on resume.
    """
    S, C = args.seq_len, None
    sums = sq = None
    rsums = rsq = None
    count = 0
    it = iter(loader)
    for _ in range(args.stat_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        frames = batch['frames'].to(device)
        z = encode_tokens(encoder, compressor, tex_encoder, frames, dtype)
        B, S, N, C = z.shape
        r = z[:, 1:] - z[:, :1]
        # float32 accumulation: NPU float64 support is poor and HCCL cannot
        # all_reduce fp64. Values are ~1e-1 scale over ~1e4 rows per rank,
        # well within fp32 precision.
        flat = z.permute(1, 0, 2, 3).reshape(S, -1, C).float()
        rflat = r.permute(1, 0, 2, 3).reshape(S - 1, -1, C).float()
        if sums is None:
            sums = flat.sum(1)
            sq = (flat ** 2).sum(1)
            rsums = rflat.sum(1)
            rsq = (rflat ** 2).sum(1)
        else:
            sums += flat.sum(1)
            sq += (flat ** 2).sum(1)
            rsums += rflat.sum(1)
            rsq += (rflat ** 2).sum(1)
        count += B * N
    if use_ddp:
        cnt = torch.tensor([count], dtype=torch.float32, device=device)
        for t in (sums, sq, rsums, rsq):
            dist.all_reduce(t)
        dist.all_reduce(cnt)
        count = int(cnt.item())
    mean = (sums / count).float()
    std = ((sq / count).float() - mean ** 2).clamp(min=1e-8).sqrt()
    rmean = (rsums / count).float()
    rstd = ((rsq / count).float() - rmean ** 2).clamp(min=1e-8).sqrt()
    return {'abs_mean': mean.cpu(), 'abs_std': std.cpu(),
            'res_mean': rmean.cpu(), 'res_std': rstd.cpu(),
            'count': count}


def build_targets(z, stats, target_mode, device):
    """z [B,S,N,C] unnormalized -> (x1 [B,S-1,N,C], cond [B,1,N,C])."""
    am = stats['abs_mean'].to(device)   # [S, C]
    asd = stats['abs_std'].to(device)
    cond = (z[:, :1] - am[0]) / asd[0]
    if target_mode == 'absolute':
        m = am[1:].unsqueeze(0).unsqueeze(2)    # [1, S-1, 1, C]
        s = asd[1:].unsqueeze(0).unsqueeze(2)
        x1 = (z[:, 1:] - m) / s
    else:
        r = z[:, 1:] - z[:, :1]
        m = stats['res_mean'].to(device).unsqueeze(0).unsqueeze(2)
        s = stats['res_std'].to(device).unsqueeze(0).unsqueeze(2)
        x1 = (r - m) / s
    return x1, cond


def denorm_to_full(x1, stats, target_mode, z0, device):
    """Normalized x1 [B,S-1,N,C] -> absolute latents [B,S,N,C] incl. z0."""
    if target_mode == 'absolute':
        m = stats['abs_mean'][1:].unsqueeze(0).unsqueeze(2).to(device)
        s = stats['abs_std'][1:].unsqueeze(0).unsqueeze(2).to(device)
        z_future = x1 * s + m
    else:
        m = stats['res_mean'].unsqueeze(0).unsqueeze(2).to(device)
        s = stats['res_std'].unsqueeze(0).unsqueeze(2).to(device)
        z_future = z0 + (x1 * s + m)
    return torch.cat([z0, z_future], dim=1)


# --------------------------------------------------------------------------
# Wan adapter support (patterns proven on the cluster by
# train_cached_wan_compact_diffusion.py): the full Wan backbone is loaded
# from --wan_ckpt_dir on every run; checkpoints carry ONLY the trainable
# parameters (adapters + modulation + time path + unfrozen QKV), and the
# EMA shadows only those. Frozen parameters are cast to bf16 to halve
# resident memory; trainable ones stay fp32 (so the adapter's lazy
# time-embedding float32 conversion is a no-op and cannot break DDP
# gradient buckets).
# --------------------------------------------------------------------------

def cast_frozen_parameters(model, dtype):
    for parameter in model.parameters():
        if parameter.requires_grad or not parameter.is_floating_point():
            continue
        parameter.data = parameter.data.to(dtype=dtype)


def trainable_state_dict(model):
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad}
    return {name: value.detach().cpu()
            for name, value in model.state_dict().items()
            if name in trainable}


def load_trainable_state_dict(model, state_dict, label):
    parameters = dict(model.named_parameters())
    missing = []
    for name, parameter in parameters.items():
        if not parameter.requires_grad:
            continue
        if name not in state_dict:
            missing.append(name)
            continue
        parameter.data.copy_(state_dict[name].to(
            device=parameter.device, dtype=parameter.dtype))
    unexpected = sorted(set(state_dict) - set(parameters))
    if missing or unexpected:
        raise ValueError(
            f'{label} trainable state mismatch: '
            f'missing={missing[:8]}, unexpected={unexpected[:8]}')


class TrainableEMA:
    """EMA over trainable parameters only (the Wan backbone is excluded)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad}

    def update(self, model):
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name] = self.shadow[name].to(parameter.device)
            self.shadow[name].mul_(self.decay).add_(
                parameter.detach().float(), alpha=1 - self.decay)

    def state_dict(self):
        return {name: value.detach().cpu()
                for name, value in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {name: value.detach().float().clone()
                       for name, value in state_dict.items()}

    def to(self, device):
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self


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
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
        print(f'E7 diffusion smoke: target_mode={args.target_mode} '
              f'device={device_type} world={world_size} dtype={args.dtype}')

    encoder, compressor, tex_encoder, decoder, grid, latent_dim = load_dual_ae(
        args, device, dtype)
    num_tokens = grid * grid
    future = args.seq_len - 1

    if args.generator == 'wan':
        if not args.wan_ckpt_dir or not os.path.isdir(args.wan_ckpt_dir):
            raise FileNotFoundError(
                f'--generator wan requires --wan_ckpt_dir, '
                f'got {args.wan_ckpt_dir!r}')
        # Lazy import: pulls the vendored Wan2.1 package only on this path.
        from models.wan_compact_adapter import WanCompactAdapter
        core = WanCompactAdapter(
            args.wan_ckpt_dir, latent_dim=latent_dim, latent_grid=grid,
            seq_len=future, i0_condition=True, train_text_adapter=False,
            train_qkv=True, train_qkv_last_n=args.train_qkv_last_n)
        # Frozen backbone -> bf16 (memory); trainable stay fp32.
        cast_frozen_parameters(core, dtype)
        core = core.to(device)
        ema = TrainableEMA(core, decay=args.ema_decay)
        # Wan matmuls need autocast (frozen bf16 x fp32 adapters); the DiT
        # baseline stays pure fp32 so E7-a numbers remain comparable.
        import contextlib
        amp_ctx = lambda: torch.autocast(device_type=device_type, dtype=dtype)
    else:
        core = CompactLatentDiT(
            latent_dim=latent_dim, num_tokens=num_tokens,
            model_dim=args.model_dim, spatial_depth=args.spatial_depth,
            temporal_depth=args.temporal_depth, num_heads=args.num_heads,
            seq_len=future, text_cond=False, i0_condition=True).to(device)
        ema = EMA(core, decay=args.ema_decay, dtype=torch.float32).to(device)
        import contextlib
        amp_ctx = contextlib.nullcontext
    n_params = sum(p.numel() for p in core.parameters())
    n_train = sum(p.numel() for p in core.parameters() if p.requires_grad)
    if args.generator == 'wan' and n_train > 1.5e9:
        raise RuntimeError(
            f'{n_train / 1e9:.2f}B trainable parameters would need a '
            f'{n_train * 4 / 1e9:.1f}GB DDP gradient bucket — this is the '
            f'known 14B full-QKV OOM. Use the Wan2.1-T2V-1.3B checkpoint, '
            f'or --train_qkv_last_n 4 for larger backbones.')
    if main_process:
        print(f'  generator={args.generator}: {n_train / 1e6:.1f}M trainable '
              f'/ {n_params / 1e6:.1f}M total  '
              f'tokens/frame={num_tokens} latent_dim={latent_dim}')

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            core, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False)
    else:
        model = core
    cfm = OTCFM(model)
    optimizer = build_optimizer(core, lr=args.lr, wd=args.wd)
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

    # ---- Resume / stats (the normalization contract lives in the ckpt) ----
    global_step = 0
    stats = None
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        for key in ('target_mode', 'generator'):
            if ckpt['args'].get(key, 'dit' if key == 'generator' else None) \
                    != getattr(args, key):
                raise RuntimeError(
                    f'resume {key} {ckpt["args"].get(key)} != '
                    f'requested {getattr(args, key)}')
        if args.generator == 'wan':
            # Checkpoints carry trainable params only; the frozen backbone
            # was already loaded fresh from --wan_ckpt_dir.
            load_trainable_state_dict(core, ckpt['model'], 'model')
        else:
            core.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema']); ema = ema.to(device)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        global_step = ckpt.get('global_step', 0)
        stats = ckpt['latent_stats']
        restore_rng_state(ckpt.get('rng'))
        if main_process:
            print(f'Resumed from {args.resume} @ step {global_step}')
    if stats is None:
        if main_process:
            print(f'Measuring latent stats over {args.stat_batches} '
                  f'batches/rank ...')
        stats = measure_stats(loader, encoder, compressor, tex_encoder,
                              args, device, dtype, use_ddp)
        if main_process:
            print(f'  stats over {stats["count"]} tokens; '
                  f'abs_std mean={stats["abs_std"].mean():.4f} '
                  f'res_std mean={stats["res_std"].mean():.4f}')

    writer = SummaryWriter(os.path.join(args.output_dir, 'tb')) \
        if main_process else None
    metrics_path = os.path.join(args.output_dir, 'metrics.jsonl')
    meter = ThroughputMeter()

    def save_ckpt(step):
        payload = {
            'model': (trainable_state_dict(core) if args.generator == 'wan'
                      else core.state_dict()),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'global_step': step, 'args': vars(args),
            'latent_stats': stats, 'rng': capture_rng_state(),
        }
        atomic_torch_save(payload, os.path.join(
            args.output_dir, f'checkpoint_step{step:07d}.pt'))
        atomic_torch_save(payload, os.path.join(
            args.output_dir, 'checkpoint_latest.pt'))
        print(f'  saved checkpoint @ step {step}')

    @torch.no_grad()
    def run_eval(step):
        core.eval()
        vmse, n = 0.0, 0
        gen_stds, tgt_stds = [], []
        motion_ratios = []
        saved = 0
        for i, batch in enumerate(eval_loader):
            frames = batch['frames'].to(device)
            z = encode_tokens(encoder, compressor, tex_encoder, frames, dtype)
            x1, cond = build_targets(z, stats, args.target_mode, device)
            x0 = torch.randn_like(x1)
            t = torch.rand((x1.shape[0],), device=device, dtype=x1.dtype)
            te = t.view(-1, 1, 1, 1)
            with amp_ctx():
                v = core((1 - te) * x0 + te * x1, t, cond=cond)
            vmse += torch.nn.functional.mse_loss(v.float(), x1 - x0).item()
            n += 1
            if saved < args.sample_clips:
                gen = cfm_sample(cond, x1.shape)
                gen_stds.append(gen.float().std().item())
                tgt_stds.append(x1.float().std().item())
                # z0-copy guard: motion magnitude ||z_t - z_0|| in the
                # UNNORMALIZED space, generated vs target. ~1 = real
                # dynamics; <<1 = the generator is copying the condition.
                z_gen = denorm_to_full(gen, stats, args.target_mode,
                                       z[:, :1], device)
                m_gen = (z_gen[:, 1:] - z[:, :1]).flatten(2).norm(dim=2)
                m_tgt = (z[:, 1:] - z[:, :1]).flatten(2).norm(dim=2)
                motion_ratios.append(
                    (m_gen / m_tgt.clamp(min=1e-8)).mean().item())
                torch.save(
                    {'sampled_x1': gen.float().cpu(),
                     'target_x1': x1.float().cpu(),
                     'z0_unnorm': z[:, :1].float().cpu(),
                     'stats': stats, 'target_mode': args.target_mode,
                     'grid': grid, 'step': step},
                    os.path.join(args.output_dir, 'samples',
                                 f'step{step:07d}_clip{i}.pt'))
                if saved == 0:
                    save_preview(step, frames, z, gen)
                saved += 1
        row = {
            'step': step, 'target_mode': args.target_mode,
            'eval/velocity_mse': vmse / max(n, 1),
            'eval/gen_std': float(sum(gen_stds) / max(len(gen_stds), 1)),
            'eval/target_std': float(sum(tgt_stds) / max(len(tgt_stds), 1)),
        }
        row['eval/gen_std_ratio'] = (
            row['eval/gen_std'] / max(row['eval/target_std'], 1e-8))
        row['eval/motion_ratio'] = float(
            sum(motion_ratios) / max(len(motion_ratios), 1))
        append_metrics(metrics_path, row)
        if writer:
            for k, v_ in row.items():
                if isinstance(v_, (int, float)):
                    writer.add_scalar(k, v_, step)
        print(f'  [eval] step {step}: vmse={row["eval/velocity_mse"]:.4f} '
              f'gen_std_ratio={row["eval/gen_std_ratio"]:.3f} '
              f'motion_ratio={row["eval/motion_ratio"]:.3f}')
        core.train()

    preview_ok = [True]

    @torch.no_grad()
    def save_preview(step, frames, z, gen_x1):
        """PNG grid (GT | AE recon | generated) for direct viewing; guarded —
        decoder interpolate is the historically NPU-fragile op, so one
        failure disables previews (the .pt latents still get saved and can
        be decoded on the H200 box)."""
        if not preview_ok[0]:
            return
        try:
            import numpy as np
            from PIL import Image
            B, S, N, C = z.shape
            z_gen = denorm_to_full(gen_x1, stats, args.target_mode,
                                   z[:, :1], device)
            def decode(zz):
                zz = zz.reshape(1, S, grid, grid, C)
                geo_dim = C // 2
                rgb = decoder(zz[..., :geo_dim].float(),
                              zz[..., geo_dim:].float(),
                              frames_chunk_size=4)
                return rgb[..., :3].clamp(0, 1)[0]        # [S,H,W,3]
            recon = decode(z[:1])
            genv = decode(z_gen[:1])
            gt = frames[0].float().clamp(0, 1).permute(0, 2, 3, 1)
            rows = torch.cat(
                [torch.cat([gt[s], recon[s], genv[s]], dim=1)
                 for s in range(S)], dim=0)
            img = (rows.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(
                args.output_dir, 'samples', f'step{step:07d}_preview.png'))
        except Exception as exc:
            preview_ok[0] = False
            print(f'  [WARN] preview decode failed ({exc}); previews '
                  f'disabled, decode the saved .pt on the H200 box instead')

    @torch.no_grad()
    def cfm_sample(cond, shape):
        z = torch.randn(shape, device=device, dtype=cond.dtype)
        n_steps = args.sample_steps
        for i in range(n_steps):
            t = torch.full((shape[0],), i / n_steps, device=device,
                           dtype=cond.dtype)
            with amp_ctx():
                v = core(z, t, cond=cond)
            z = z + v.float() / n_steps
        return z

    if main_process:
        print(f'Training: {args.max_steps} steps, '
              f'{len(loader)} batches/epoch/rank')
    model.train()
    epoch = 0
    optimizer.zero_grad(set_to_none=True)
    done = False
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        pbar = tqdm(loader, disable=not main_process, desc=f'ep{epoch}')
        for it_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(device)
            z = encode_tokens(encoder, compressor, tex_encoder, frames, dtype)
            x1, cond = build_targets(z, stats, args.target_mode, device)
            with amp_ctx():
                loss = cfm.compute_loss(x1, cond=cond) / args.accum_steps
            loss.backward()
            if (it_idx + 1) % args.accum_steps != 0:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in core.parameters() if p.requires_grad],
                args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(core)
            global_step += 1
            meter.update(x1.shape[0] * x1.shape[1] * world_size)

            if main_process and global_step % args.log_every == 0:
                row = {
                    'step': global_step, 'target_mode': args.target_mode,
                    'train/loss': float(loss.item() * args.accum_steps),
                    'train/grad_norm': float(grad_norm),
                    'train/lr': scheduler.get_last_lr()[0],
                    'DI_throughput': meter.rate(),
                }
                append_metrics(metrics_path, row)
                if writer:
                    for k, v_ in row.items():
                        if isinstance(v_, (int, float)):
                            writer.add_scalar(k, v_, global_step)
                pbar.set_postfix(
                    loss=f'{row["train/loss"]:.4f}',
                    DI=f'{row["DI_throughput"]:.1f}')

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
