#!/usr/bin/env python3
"""Frozen-R7 prefix flow: explicit train-memory/held-out and online/EMA evaluation."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from train_single_target_probe import FrozenR7, build_loader, psnr
from train_causal_dual_tokenizer import get_lpips, lpips_chunked
from models.r7_flow_probe import (FLOW_SCHEMA, R7FlowProbe, clean_prediction,
                                  network_target, sample_flow)
from utils.latent_generation_metrics import (complete_prefix, denoising_baselines,
    fit_statistics, generation_metrics, inverse, normalize, to_device)
from utils.device import (configure_backend_compatibility, get_device,
                         get_device_name, manual_seed_all, resolve_dtype)
from utils.file_signature import sampled_file_signature
from utils.training import (EMA, append_metrics, atomic_torch_save, build_optimizer,
                            capture_rng_state, restore_rng_state)
from train_causal_video_diffusion import ema_weights, exact_equal
from utils.video_preview import save_video_preview


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('csv', 'eval_csv', 'video_root', 'encoder_ckpt', 'r7_ckpt', 'output_dir'):
        p.add_argument('--'+flag, required=True)
    p.add_argument('--prediction', choices=('plain_x0', 'preconditioned'), default='plain_x0')
    p.add_argument('--future_frames', type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument('--max_samples', type=int, choices=(1, 16, 256), default=1)
    p.add_argument('--eval_samples', type=int, default=16)
    p.add_argument('--train_eval_samples', type=int, default=16)
    p.add_argument('--hidden_dim', type=int, default=384)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--num_heads', type=int, default=6)
    p.add_argument('--max_steps', type=int, default=2000)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--wd', type=float, default=.01)
    p.add_argument('--warmup_steps', type=int, default=100)
    p.add_argument('--std_floor', type=float, default=1e-4)
    p.add_argument('--ema_decay', type=float, default=.999)
    p.add_argument('--eval_ema', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--eval_every', type=int, default=100)
    p.add_argument('--log_every', type=int, default=10)
    p.add_argument('--save_every', type=int, default=500)
    p.add_argument('--sample_steps', type=int, default=30)
    p.add_argument('--sample_seeds', default='42,43,44,45')
    p.add_argument('--preview_clips', type=int, default=1)
    p.add_argument('--eval_lpips', action='store_true')
    p.add_argument('--text_embedding_dir', default='')
    p.add_argument('--allow_no_text', action='store_true', help='explicit uncaptioned diagnostic, never a T2V claim')
    p.add_argument('--resume', default='')
    p.add_argument('--stop_after_steps', type=int, default=0,
                   help='planned interruption for resume smoke; does not change scheduler budget')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--dtype', choices=('fp16', 'bf16', 'fp32'), default='bf16')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args(argv)


@torch.no_grad()
def pack_samples(loader, frozen, frames):
    items = []
    device = next(frozen.decoder.parameters()).device
    for batch in loader:
        if batch['decode_replacements']:
            raise RuntimeError('decode replacement invalidates fixed probe identity')
        rgb = batch['frames'].to(device)
        latent = frozen.encode(rgb).float().cpu()
        for i, vid in enumerate(batch['video_id']):
            items.append({'video_id': vid, 'window_index': int(batch['window_index'][i]),
                'anchor': latent[i, :1], 'target': latent[i, 1:frames+1],
                'full_latent': latent[i],
                'raw_target': (rgb[i, 1:frames+1].clamp(0, 1)*255).round().byte().cpu()})
    if not items or len({x['video_id'] for x in items}) != len(items):
        raise ValueError('empty/duplicate video IDs in probe subset')
    return items


def stack(items, device):
    return (torch.stack([x['anchor'] for x in items]).to(device),
            torch.stack([x['target'] for x in items]).to(device))


def scalar_mean(rows):
    keys = set.intersection(*(set(r) for r in rows)) if rows else set()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys
            if all(isinstance(r[k], (float, int)) and not isinstance(r[k], bool) for r in rows)}


def main(argv=None):
    args = parse_args(argv)
    for k in ('eval_samples', 'train_eval_samples', 'max_steps', 'batch_size',
              'eval_every', 'log_every', 'save_every', 'sample_steps'):
        if getattr(args, k) < 1:
            raise ValueError(f'{k} must be positive')
    if (args.std_floor <= 0 or args.lr <= 0 or args.wd < 0 or args.warmup_steps < 0
            or not 0 <= args.ema_decay < 1 or args.preview_clips < 0 or args.num_workers < 0):
        raise ValueError('invalid optimizer/statistics/preview configuration')
    if not args.text_embedding_dir and not (args.max_samples == 1 or args.allow_no_text):
        raise ValueError('n16+ requires text sidecar or explicit --allow_no_text diagnostic')
    seeds = [int(x) for x in args.sample_seeds.split(',')]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('sample seeds must be distinct')
    outdir = Path(args.output_dir)
    if not args.resume and any(outdir.glob('checkpoint*.pt')):
        raise ValueError('existing checkpoints require explicit --resume/new namespace')
    if not args.resume and (outdir/'metrics.jsonl').exists():
        raise ValueError('existing metrics require explicit resume/new namespace')
    dtype = resolve_dtype(args.dtype)
    backend = get_device_name(); configure_backend_compatibility(backend)
    if backend == 'cpu':
        raise RuntimeError('training requires GPU/NPU; CPU tensor tests are separate')
    device = get_device(0)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    manual_seed_all(args.seed)
    frozen = FrozenR7(args.encoder_ckpt, args.r7_ckpt, device, dtype)
    train_csv = args.eval_csv if args.max_samples == 1 else args.csv
    train_loader = build_loader(train_csv, args.video_root, frozen.config,
                               args.max_samples, 1, args.num_workers, False)
    train = pack_samples(train_loader, frozen, args.future_frames)
    evaluation = []
    if len(train) != args.max_samples:
        raise ValueError('requested training subset could not be materialized completely')
    if args.max_samples > 1:
        evaluation = pack_samples(build_loader(args.eval_csv, args.video_root,
            frozen.config, args.eval_samples, 1, args.num_workers, False), frozen, args.future_frames)
        if len(evaluation) != args.eval_samples:
            raise ValueError('requested held-out subset could not be materialized completely')
        if {x['video_id'] for x in train} & {x['video_id'] for x in evaluation}:
            raise ValueError('train and held-out video IDs overlap')
    del frozen.encoder, frozen.compressor, frozen.tex_encoder, train_loader
    from utils.device import empty_cache
    empty_cache()
    ac, at = stack(train, 'cpu')
    stats = {'anchor': fit_statistics(ac, args.std_floor),
             'target': fit_statistics(at, args.std_floor)}
    del ac, at
    identity = {'train': [(x['video_id'], x['window_index']) for x in train],
                'heldout': [(x['video_id'], x['window_index']) for x in evaluation],
                'sampling': frozen.representation_contract,
                'r7_signature': sampled_file_signature(args.r7_ckpt),
                'encoder_signature': sampled_file_signature(args.encoder_ckpt),
                'prefix_decode': 'candidate-prefix-repeat-last-to-nine'}
    st = {k: to_device(v, device) for k, v in stats.items()}
    bank = None
    if args.text_embedding_dir:
        from train_causal_wan_video_diffusion import TextEmbeddingBank
        bank = TextEmbeddingBank(args.text_embedding_dir)
        absent = [x['video_id'] for x in train+evaluation if x['video_id'] not in bank.embeddings]
        if absent:
            raise ValueError(f'probe captions missing from sidecar: {absent[:8]}')
    digest = hashlib.sha256()
    for item in train+evaluation:
        digest.update(item['video_id'].encode())
        digest.update(item['full_latent'].contiguous().numpy().tobytes())
        digest.update(item['raw_target'].contiguous().numpy().tobytes())
        if bank is not None:
            digest.update(bank.embeddings[item['video_id']].float().contiguous().numpy().tobytes())
    identity['materialized_sha256'] = digest.hexdigest()

    def text(items):
        if bank is None:
            return None, None
        ids = [x['video_id'] for x in items]
        lengths = [bank.embeddings[v].shape[0] for v in ids]
        emb = bank.batch(ids, device)
        mask = torch.arange(emb.shape[1], device=device)[None] >= torch.tensor(lengths, device=device)[:, None]
        return emb, mask

    model_args = dict(latent_dim=frozen.config.latent_dim,
        num_tokens=frozen.config.latent_grid**2, future_frames=args.future_frames,
        hidden_dim=args.hidden_dim, depth=args.depth, num_heads=args.num_heads,
        prediction=args.prediction, text_cond=bank is not None)
    model = R7FlowProbe(**model_args).to(device)
    optimizer = build_optimizer(model, args.lr, args.wd)
    def lr_factor(s):
        if s < args.warmup_steps:
            return (s+1)/max(args.warmup_steps, 1)
        return .1+.9*.5*(1+np.cos(np.pi*min(1., (s-args.warmup_steps)/max(1, args.max_steps-args.warmup_steps))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    ema = EMA(model, args.ema_decay, dtype=torch.float32, warmup=True).to(device)
    # Explicit scaler for fp16; BF16/FP32 use the ordinary optimizer path.
    from utils.device import create_grad_scaler
    scaler = create_grad_scaler(enabled=dtype == torch.float16)
    order = list(range(len(train))); cursor = 0; step = 0; best = float('inf')
    rng = random.Random(args.seed)
    immutable = {k: v for k, v in vars(args).items() if k not in
                 ('resume', 'stop_after_steps', 'output_dir', 'num_workers', 'csv', 'eval_csv', 'video_root',
                  'encoder_ckpt', 'r7_ckpt', 'text_embedding_dir')}
    immutable['resolved_dtype'] = str(dtype)
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if (ckpt.get('schema') != FLOW_SCHEMA or not exact_equal(ckpt['identity'], identity)
                or not exact_equal(ckpt['statistics'], stats) or ckpt['contract'] != immutable):
            raise ValueError('resume representation/data/statistics/objective mismatch')
        model.load_state_dict(ckpt['model'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer']); scheduler.load_state_dict(ckpt['scheduler'])
        ema.load_state_dict(ckpt['ema']); ema.to(device); ema.load_metadata(ckpt['ema_metadata'])
        scaler.load_state_dict(ckpt['scaler'])
        step, best = ckpt['step'], ckpt['best']
        order, cursor = ckpt['order'], ckpt['cursor']; rng.setstate(ckpt['order_rng'])
        resume_rng = ckpt['rng']
        del ckpt
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir/'samples').mkdir(exist_ok=True)
    writer = SummaryWriter(str(outdir/'tb'), purge_step=step or None)
    lpips = get_lpips(device) if args.eval_lpips else None
    if args.resume:
        restore_rng_state(resume_rng)
    def decode(c, z):
        sequence = complete_prefix(c, z, frozen.config.seq_len)
        # Frozen decoder is kept FP32 so this contract is independent of denoiser AMP.
        return frozen.decode_full(sequence)[:, 1:args.future_frames+1]
    def save(kind):
        payload = {'schema': FLOW_SCHEMA, 'step': step, 'best': best,
            'model': model.state_dict(), 'model_args': model_args, 'ema': ema.state_dict(),
            'ema_metadata': ema.metadata(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'rng': capture_rng_state(), 'order': order, 'cursor': cursor,
            'order_rng': rng.getstate(), 'statistics': stats, 'identity': identity,
            'contract': immutable, 'args': vars(args)}
        atomic_torch_save(payload, str(outdir/f'checkpoint_{kind}.pt'))
    def log(row):
        append_metrics(str(outdir/'metrics.jsonl'), row)
        for k, v in row.items():
            if isinstance(v, (int, float)) and k != 'step':
                writer.add_scalar(k, v, step)
        writer.flush()

    @torch.no_grad()
    def evaluate():
        model.eval()
        subsets = [('train-memory', train[:args.train_eval_samples])]
        if evaluation:
            subsets.append(('held-out', evaluation))
        primary = None
        weights = ['model', 'ema'] if args.eval_ema else ['model']
        for weight in weights:
            swap = ema_weights(model, ema) if weight == 'ema' else contextlib.nullcontext()
            with swap:
                for split, items in subsets:
                    rows = []; all_seed_deltas = []; condition_outputs = []
                    for index, item in enumerate(items):
                        c, y = stack([item], device)
                        cn, yn = normalize(c, st['anchor']), normalize(y, st['target'])
                        emb, mask = text([item])
                        common_noise = torch.randn(yn.shape, generator=torch.Generator(device='cpu').manual_seed(seeds[0])).to(device)
                        condition_outputs.append(sample_flow(model, cn, common_noise, args.sample_steps,
                                                             emb, mask, dtype).cpu())
                        raw = item['raw_target'].to(device).float().div(255).permute(0, 2, 3, 1)[None]
                        ae = decode(c, y)
                        copy = decode(c, c.expand(-1, args.future_frames, -1, -1))
                        full = frozen.decode_full(item['full_latent'][None].to(device))[:, 1:args.future_frames+1]
                        noises = []
                        for seed in seeds:
                            gen = torch.Generator(device='cpu').manual_seed(seed+1009*index)
                            noises.append(torch.randn(yn.shape, generator=gen).to(device))
                        for tval in (.1, .3, .5, .7, .9, .99):
                            noisy = (1-tval)*noises[0]+tval*yn
                            t = torch.full((1,), tval, device=device)
                            with torch.autocast(device_type=backend, dtype=dtype, enabled=dtype != torch.float32):
                                output = model(noisy, t, cn, emb, mask)
                            hat = clean_prediction(output, noisy, t, args.prediction)
                            diagnostic = denoising_baselines(noisy, yn, tval)
                            diagnostic.update(x0_mse=float(F.mse_loss(hat, yn)),
                                network_mse=float(F.mse_loss(output, network_target(yn, noises[0], t, args.prediction))))
                            append_metrics(str(outdir/'denoising.jsonl'), dict(step=step, weights=weight,
                                split=split, video_id=item['video_id'], t=tval, **diagnostic))
                        first = None
                        for seed, noise in zip(seeds, noises):
                            gn = sample_flow(model, cn, noise, args.sample_steps, emb, mask, dtype)
                            generated = inverse(gn, st['target'])
                            rgb = decode(c, generated)
                            row = generation_metrics(generated, y, c, st['target'], st['anchor'])
                            row.update(generated_psnr_ae=psnr(rgb, ae), generated_psnr_raw=psnr(rgb, raw),
                                ae_psnr_raw=psnr(ae, raw), full_ae_psnr_raw=psnr(full, raw),
                                copy_psnr_raw=psnr(copy, raw),
                                raw_psnr_gap_to_ae=psnr(ae, raw)-psnr(rgb, raw))
                            if lpips is not None:
                                row.update(generated_lpips_raw=float(lpips_chunked(lpips, rgb, raw, 1, 256)),
                                           ae_lpips_raw=float(lpips_chunked(lpips, ae, raw, 1, 256)),
                                           copy_lpips_raw=float(lpips_chunked(lpips, copy, raw, 1, 256)))
                            if first is None:
                                first = gn.clone()
                            else:
                                all_seed_deltas.append(float(F.mse_loss(gn, first)))
                            rows.append(row)
                            append_metrics(str(outdir/'eval_samples.jsonl'), dict(step=step, weights=weight,
                                split=split, video_id=item['video_id'], seed=seed, **row))
                            if index < args.preview_clips and seed == seeds[0]:
                                save_video_preview(str(outdir/'samples'), f'step{step:07d}_{weight}_{split}_{index:03d}',
                                    {'RAW_TARGET': raw[0], 'AE_TARGET': ae[0], 'FULL_AE': full[0],
                                     'COPY_ANCHOR': copy[0], 'GENERATED': rgb[0]}, fps=8,
                                    metadata={'step': step, 'weights': weight, 'split': split,
                                              'video_id': item['video_id'], 'seed': seed,
                                              'prediction': args.prediction, 'future_frames': args.future_frames},
                                    save_frames=True, save_mp4=args.future_frames > 1)
                    summary = scalar_mean(rows)
                    summary['worst_generated_psnr_ae'] = min(r['generated_psnr_ae'] for r in rows)
                    summary['worst_raw_psnr_gap_to_ae'] = max(r['raw_psnr_gap_to_ae'] for r in rows)
                    summary['seed_pair_mse'] = float(np.mean(all_seed_deltas)) if all_seed_deltas else 0.
                    summary['same_noise_condition_pair_mse'] = (float(np.mean([
                        F.mse_loss(v, condition_outputs[0]).item() for v in condition_outputs[1:]]))
                        if len(condition_outputs) > 1 else 0.)
                    row = {'step': step, 'eval/weights': weight, 'eval/split': split,
                           'eval/clips': len(items), 'eval/seeds': len(seeds),
                           **{'eval/'+k: v for k, v in summary.items()}}
                    log(row)
                    if split == 'train-memory' and weight == 'model':
                        primary = summary
                        (outdir/'memory_status.pending.json').write_text(json.dumps({'schema': FLOW_SCHEMA,
                            'step': step, 'clips': len(items), 'total_train_clips': len(train),
                            'r7_signature': identity['r7_signature'], 'future_frames': args.future_frames,
                            'seeds': seeds, 'prediction': args.prediction,
                            'all_train_evaluated': len(items) == len(train),
                            'passed': (len(items) == len(train)
                                and summary['worst_generated_psnr_ae'] >= 30
                                and summary['worst_raw_psnr_gap_to_ae'] <= .5),
                            'limits': {'min_psnr_vs_ae': 30., 'max_gap_to_ae': .5},
                            'metrics': summary,
                            'meaning': 'training memory only, not held-out video quality'}, indent=2))
                        os.replace(outdir/'memory_status.pending.json', outdir/'memory_status.json')
        model.train()
        return primary['normalized_mse']

    try:
        overflow_retries = 0
        stop_step = min(args.max_steps, args.stop_after_steps) if args.stop_after_steps > 0 else args.max_steps
        while step < stop_step:
            if cursor == 0:
                rng.shuffle(order)
            selected = order[cursor:cursor+args.batch_size]
            cursor += len(selected)
            if cursor >= len(order):
                cursor = 0
            items = [train[i] for i in selected]
            c, y = stack(items, device)
            cn, yn = normalize(c, st['anchor']), normalize(y, st['target'])
            noise = torch.randn_like(yn); t = torch.rand(len(items), device=device)
            te = t[:, None, None, None]; noisy = (1-te)*noise+te*yn
            emb, mask = text(items)
            with torch.autocast(device_type=backend, dtype=dtype, enabled=dtype != torch.float32):
                pred = model(noisy, t, cn, emb, mask)
            loss = F.mse_loss(pred.float(), network_target(yn, noise, t, args.prediction))
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite flow loss')
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scale_before = scaler.get_scale()
            if not torch.isfinite(grad) and not scaler.is_enabled():
                raise FloatingPointError('non-finite gradient norm')
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() < scale_before:
                overflow_retries += 1
                log({'step': step, 'train/skipped_fp16_updates': overflow_retries,
                     'train/grad_scale': scaler.get_scale()})
                if overflow_retries >= 16:
                    raise FloatingPointError('16 consecutive FP16 overflows; no optimizer steps claimed')
                continue
            overflow_retries = 0
            scheduler.step(); ema.update(model)
            step += 1
            if step == 1 or step % args.log_every == 0:
                log({'step': step, 'train/network_loss': float(loss),
                     'train/grad_norm': float(grad), 'train/lr': optimizer.param_groups[0]['lr'],
                     'train/prediction': args.prediction})
            if step % args.eval_every == 0 or step == args.max_steps:
                score = evaluate()
                if score < best:
                    best = score; save('best')
                save('latest')
            if step % args.save_every == 0:
                save(f'step{step:07d}'); save('latest')
        if step == args.max_steps:
            save('final')
        save('latest')
    finally:
        writer.close()


if __name__ == '__main__':
    main()
