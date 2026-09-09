#!/usr/bin/env python3
"""Frozen-AE full-window I2V training, explicit noise-time objective and resume."""
import argparse
import contextlib
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.r7_window_dit import R7WindowDiT
from utils.window_flow import WindowFlow
from utils.window_training import (CaptionBank, collate, digest, inverse, load_artifact,
    normalize, reconcile_history, resume_contract, use_ema, validate_batch, validate_resume, validate_statistics)
from data.latent_shard_dataset import LatentShardDataset, read_shard_manifest
from utils.device import configure_backend_compatibility, get_device, get_device_name, create_grad_scaler
from utils.distributed import setup_ddp
from utils.flow_run_status import atomic_json
from utils.training import (EMA, append_metrics, atomic_torch_save, build_optimizer,
                            capture_rng_state, restore_rng_state)
from utils.file_signature import sampled_file_signature
from utils.moxing_io import read_text


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('manifest', 'stats', 'eval_manifest', 'eval_stats', 'r7_ckpt', 'output_dir'):
        p.add_argument('--'+k, required=True)
    p.add_argument('--text_dir', default='')
    p.add_argument('--no_text', action='store_true', help='explicit uncaptioned I2V arm')
    p.add_argument('--prediction', choices=('x0', 'velocity'), default='x0')
    p.add_argument('--time_shift', type=float, default=1.)
    p.add_argument('--loss_floor', type=float, default=.05)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--depth', type=int, default=12)
    p.add_argument('--heads', type=int, default=12)
    p.add_argument('--aux_layer', type=int, default=0)
    p.add_argument('--aux_weight', type=float, default=0.)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--accum_steps', type=int, default=2)
    p.add_argument('--max_steps', type=int, default=6000)
    p.add_argument('--warmup_steps', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=.01)
    p.add_argument('--ema_decay', type=float, default=.999)
    p.add_argument('--text_dropout', type=float, default=.1)
    p.add_argument('--dtype', choices=('bf16', 'fp16', 'fp32'), default='bf16')
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--save_every', type=int, default=500)
    p.add_argument('--log_every', type=int, default=10)
    p.add_argument('--eval_clips', type=int, default=16)
    p.add_argument('--preview_clips', type=int, default=4)
    p.add_argument('--sample_steps', type=int, default=64)
    p.add_argument('--sample_seeds', default='42,43')
    p.add_argument('--sample_method', choices=('euler', 'heun'), default='euler')
    p.add_argument('--shuffle_buffer', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', default='')
    p.add_argument('--stop_after_steps', type=int, default=0, help='planned pause; scheduler budget unchanged')
    p.add_argument('--cpu_test', action='store_true', help='synthetic tests only: cannot decode real AE')
    p.add_argument('--ae_norm',choices=('legacy','framewise'),required=True)
    p.add_argument('--min_ae_psnr',type=float,default=23.5)
    return p.parse_args(argv)


def run(args):
    if not np.isfinite(args.aux_weight) or args.aux_weight < 0 or bool(args.aux_layer) != bool(args.aux_weight):
        raise ValueError('aux_layer and positive aux_weight must be enabled together')
    if args.aux_layer and (not 1 <= args.aux_layer < args.depth or args.prediction != 'x0'):
        raise ValueError('intermediate clean supervision requires x0 and an intermediate block')
    for k in ('batch_size', 'accum_steps', 'max_steps', 'eval_every', 'save_every', 'log_every', 'eval_clips', 'sample_steps'):
        if getattr(args, k) < 1:
            raise ValueError(k+' must be positive')
    if not 0 <= args.warmup_steps < args.max_steps or not 0 <= args.text_dropout < 1:
        raise ValueError('invalid warmup/dropout')
    if args.lr <= 0 or args.wd < 0 or not 0 <= args.ema_decay < 1 or args.stop_after_steps < 0:
        raise ValueError('invalid optimizer/budget')
    if bool(args.text_dir) == args.no_text:
        raise ValueError('provide --text_dir OR explicit --no_text')
    seeds = [int(x) for x in args.sample_seeds.split(',')]
    if len(seeds) != len(set(seeds)) or args.preview_clips < 0:
        raise ValueError('invalid evaluation settings')
    device_type = get_device_name()
    if args.cpu_test and (device_type != 'cpu' or 'RANK' in os.environ):
        raise ValueError('synthetic CPU mode cannot run on accelerator/DDP')
    if device_type == 'cpu' and not args.cpu_test:
        raise RuntimeError('real training requires GPU/NPU')
    ddp, rank, local, world = (False, 0, 0, 1) if args.cpu_test else setup_ddp()
    device = get_device(local)
    configure_backend_compatibility(device_type)
    # Do not use legacy resolve_dtype(), which silently maps NPU BF16 to FP16.
    dtype = {'bf16':torch.bfloat16, 'fp16':torch.float16, 'fp32':torch.float32}[args.dtype]
    random.seed(args.seed+rank); np.random.seed(args.seed+rank); torch.manual_seed(args.seed+rank)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    consumed = 0
    status = dict(schema='r7-window-run-v1', status='running', step=0, expected_steps=args.max_steps,
                  rank=rank, world_size=world, requested_dtype=args.dtype, actual_dtype=str(dtype))
    status_path = out/f'run_status_rank{rank:03d}.json'
    def progress(phase, **values):
        status.update(phase=phase, updated_unix=time.time(), **values)
        status['consumed_batches']=consumed
        if phase != 'evaluating':
            for key in ('weights','clip','sample_seed'):status.pop(key,None)
        atomic_json(status_path, status)
        if rank == 0:
            atomic_json(out/'run_status.json', status)
            print('[window-status] '+json.dumps(status), flush=True)
    def signal_error(signum, frame):
        raise RuntimeError(f'interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, signal_error); signal.signal(signal.SIGINT, signal_error)
    try:
        if not args.resume and ((out/'metrics.jsonl').exists() or list(out.glob('checkpoint*.pt'))):
            raise ValueError('existing output requires explicit resume/new namespace')
        progress('loading_data_contract')
        stats, evstats = load_artifact(args.stats), load_artifact(args.eval_stats)
        if args.cpu_test:
            cfg = stats['representation']['config']
        else:
            cfg = validate_statistics(stats); validate_statistics(evstats)
            from utils.window_codec import validate_runtime
            if validate_runtime(stats['representation']) != args.ae_norm:
                raise ValueError('requested AE normalization differs from cache runtime')
            validate_runtime(evstats['representation'])
            if not all(s.get('config', {}).get('independent_anchor') for s in (stats, evstats)):
                raise ValueError('window baseline requires independently encoded first-frame conditions; use stage 28')
            if not evstats.get('config',{}).get('store_rgb'):
                raise ValueError('held-out RAW clips are required for the reconstruction quality gate')
        if digest(stats['representation']) != digest(evstats['representation']):
            raise ValueError('train/eval representations differ')
        train_shards, eval_shards = read_shard_manifest(args.manifest), read_shard_manifest(args.eval_manifest)
        if set(train_shards) & set(eval_shards):
            raise ValueError('train and held-out shards overlap')
        factor, channels, grid = cfg['temporal_factor'], cfg['geo_latent_dim']+cfg['tex_latent_dim'], cfg['latent_grid']
        future = (cfg['seq_len']-1)//factor
        tokenizer = decoder = None
        if not args.cpu_test:
            from utils.r7_representation import load_r7_modules, validate_contract
            artifact = load_artifact(args.r7_ckpt)
            validate_contract(stats['representation'], artifact['representation_contract'])
            if sampled_file_signature(args.r7_ckpt) != stats['representation']['signatures']['r7']:
                raise ValueError('cache was produced by a different AE checkpoint')
            actual, _, _, tokenizer, decoder, _ = load_r7_modules(artifact)
            from utils.window_codec import configure_codec
            configure_codec(tokenizer,args.ae_norm)
            for module in (tokenizer, decoder):
                module.to(device).eval().requires_grad_(False)
            if rank == 0:
                print('[AE-baseline]', json.dumps({'step':artifact.get('global_step', artifact.get('step')),
                    'signature':sampled_file_signature(args.r7_ckpt), 'factor':factor,
                    'temporal_norm':args.ae_norm}), flush=True)
            del artifact
        st = {k:{n:stats[k][n].to(device).float() for n in ('mean', 'std')} for k in ('cond', 'target')}
        bank = CaptionBank(args.text_dir) if args.text_dir else None
        identity = dict(representation=stats['representation'], statistics=digest(stats),
            eval_statistics=digest(evstats), train_manifest=train_shards,
            eval_manifest=eval_shards, text_signature=bank.signature if bank else None,
            synthetic_test=args.cpu_test)
        contract = resume_contract(args, identity, world)
        model_args = dict(channels=channels, grid=grid, future=future, temporal_factor=factor,
                         width=args.width, depth=args.depth, heads=args.heads, text_dim=4096 if bank else 0)
        if args.aux_layer: model_args['aux_layer'] = args.aux_layer
        # EMA is constructed before DDP's parameter broadcast, so initialize the
        # same model on every rank; rank-specific noise RNG is set below.
        torch.manual_seed(args.seed)
        core = R7WindowDiT(**model_args).to(device)
        flow = WindowFlow(args.prediction, args.loss_floor, args.time_shift)
        contract['flow'] = flow.contract()
        optimizer = build_optimizer(core, args.lr, args.wd)
        def schedule(s):
            if s < args.warmup_steps:
                return (s+1)/max(args.warmup_steps, 1)
            return .1+.45*(1+np.cos(np.pi*min(1., (s-args.warmup_steps)/max(1,args.max_steps-args.warmup_steps))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        scaler = create_grad_scaler(enabled=dtype == torch.float16)
        ema = EMA(core, args.ema_decay, dtype=torch.float32, warmup=True).to(device)
        step, consumed, best = 0, 0, float('inf')
        resume_rng = None
        if args.resume:
            saved = load_artifact(args.resume); validate_resume(saved, contract)
            core.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
            ema.load_state_dict(saved['ema']); ema.load_metadata(saved['ema_metadata']); ema.to(device)
            step, best = saved['step'], saved['best']
            consumed = saved['consumed_by_rank'][rank]; resume_rng = saved['rng_by_rank'][rank]
            del saved
        # Fixed zero workers and an isolated DataLoader generator make replay of
        # consumed batches deterministic. No claim of O(1) resume; large replay costs I/O.
        dataset = LatentShardDataset(args.manifest, args.shuffle_buffer, args.seed, True, rank, world)
        if len(dataset.shards) < world:
            raise ValueError('not enough train shards for all ranks')
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate,
                            drop_last=True, generator=torch.Generator().manual_seed(args.seed))
        evdata = list(LatentShardDataset(args.eval_manifest, 0, args.seed, False, 0, 1))
        eval_ids = {x.get('video_id') for x in evdata}
        if len(evdata) < args.eval_clips or None in eval_ids or '' in eval_ids:
            raise ValueError('insufficient/invalid held-out cache')
        if bank and eval_ids-set(bank.values):
            raise ValueError('held-out captions missing')
        evbatches = [collate([x]) for x in evdata[:args.eval_clips]]
        del evdata
        model = torch.nn.parallel.DistributedDataParallel(core, device_ids=[local],
            output_device=local, find_unused_parameters=False) if ddp else core
        torch.manual_seed(args.seed+rank)
        if device_type == 'npu': torch.npu.manual_seed_all(args.seed+rank)
        if rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            from utils.window_observability import normalization_report
            atomic_json(out/'normalization_audit.json', normalization_report(stats))
            if args.resume: reconcile_history(out,step)
            writer = SummaryWriter(str(out/'tb'), purge_step=step or None)
            atomic_json(out/'config.json', dict(contract=contract, model=model_args,
                parameters=sum(p.numel() for p in core.parameters()), global_batch=world*args.batch_size*args.accum_steps))
        progress('replaying_data_cursor', step=step, consumed_batches=consumed)
        iterator = iter(loader)
        for i in range(consumed):
            next(iterator)
            if i and i % 1000 == 0:
                progress('replaying_data_cursor', replayed_batches=i)
        if resume_rng is not None:
            restore_rng_state(resume_rng)

        def decode(c, z):
            if args.cpu_test:
                return torch.cat((c, z), 1).reshape(c.shape[0], future+1, grid, grid, channels)[..., :3].sigmoid()
            seq = torch.cat((c, z), 1).reshape(c.shape[0], future+1, grid, grid, channels)
            geo, tex = tokenizer.decode(seq)
            rgb = decoder(geo, tex)[..., :3].float()
            if not torch.isfinite(rgb).all(): raise RuntimeError('nonfinite decoded video')
            return rgb.clamp(0, 1)

        @torch.no_grad()
        def reconstruction_baseline():
            rows = []
            for index, batch in enumerate(evbatches):
                if index % world != rank: continue
                cr, yr = validate_batch(batch, future, grid, channels)
                cr, yr = cr.to(device).float(), yr.to(device).float()
                ae = decode(cr, yr)
                mean = decode(cr, st['target']['mean'][None,:,None].expand_as(yr))
                copied = decode(cr, cr.expand_as(yr))
                row = dict(video_id=batch['video_id'][0],
                    mean_l1_vs_ae=float((mean[:,1:]-ae[:,1:]).abs().mean()),
                    latent_copy_l1_vs_ae=float((copied[:,1:]-ae[:,1:]).abs().mean()))
                videos = dict(ae=ae[0], mean=mean[0], latent_copy_diagnostic=copied[0])
                if 'rgb' in batch:
                    raw=batch['rgb'].to(device).float().div(255).permute(0,1,3,4,2)
                    row['ae_psnr_full_vs_raw'] = float(-10*torch.log10((ae-raw).square().mean().clamp_min(1e-12)))
                    row['rgb_copy_l1_vs_raw'] = float((raw[:,:1]-raw[:,1:]).abs().mean())
                    videos['raw']=raw[0]
                    videos['rgb_copy']=raw[0,:1].expand_as(raw[0])
                if 'anchor_relative_l2' in batch: row['anchor_relative_l2']=batch['anchor_relative_l2'][0]
                rows.append(row)
                if index < args.preview_clips:
                    from utils.video_preview import save_video_preview
                    save_video_preview(str(out/'ae_baseline'),f'clip{index}',videos,
                        metadata=row,fps=8,save_mp4=not args.cpu_test)
            if ddp:
                gathered=[None]*world; dist.all_gather_object(gathered,rows)
                rows=[r for part in gathered for r in part]
            result=dict(ae_signature=identity['representation'].get('signatures'), clips=rows,
                        temporal_norm=args.ae_norm,min_ae_psnr=args.min_ae_psnr)
            if rows and 'ae_psnr_full_vs_raw' in rows[0]:
                result['mean_clip_ae_psnr_full_vs_raw']=float(np.mean([r['ae_psnr_full_vs_raw'] for r in rows]))
            if args.cpu_test:
                result['gate_passed']=True
            else:
                from utils.window_codec import reconstruction_gate
                _,result['gate_passed']=reconstruction_gate([r['ae_psnr_full_vs_raw'] for r in rows],args.min_ae_psnr)
            if rank == 0:
                atomic_json(out/'ae_baseline.json',result)
                print('[AE reconstruction baseline]',json.dumps(result),flush=True)
            if not result['gate_passed']:
                raise RuntimeError('AE replay PSNR below declared minimum; no diffusion updates performed')

        @torch.no_grad()
        def evaluate():
            eval_rng = capture_rng_state()
            core.eval(); rows = []; preview_root = out/'samples'/f'step{step:07d}'
            for label in ('online', 'ema'):
                with use_ema(core, ema) if label == 'ema' else contextlib.nullcontext():
                    for index, batch in enumerate(evbatches):
                        if index % world != rank:
                            continue
                        cr, yr = validate_batch(batch, future, grid, channels)
                        cr, yr = cr.to(device).float(), yr.to(device).float()
                        c, y = normalize(cr, st['cond']), normalize(yr, st['target'])
                        text, valid = bank.batch(batch['video_id'], device) if bank else (None, None)
                        ae = decode(cr, yr)
                        raw = batch.get('rgb')
                        raw = raw.to(device).float().div(255).permute(0,1,3,4,2) if raw is not None else None
                        copy = (raw if raw is not None else ae)[:,:1].expand_as(ae)
                        for seed in seeds:
                            progress('evaluating', step=step, weights=label, clip=index, sample_seed=seed)
                            noise = torch.randn(y.shape, generator=torch.Generator().manual_seed(seed+1009*index)).to(device)
                            generated = flow.sample(core, c, noise, args.sample_steps, dtype, text, valid, args.sample_method)
                            gen = decode(cr, inverse(generated, st['target']))
                            row = dict(step=step, weights=label, video_id=batch['video_id'][0], seed=seed,
                                latent_mse=float((generated-y).square().mean()),
                                rgb_l1_vs_ae=float((gen[:, 1:]-ae[:, 1:]).abs().mean()),
                                rgb_copy_l1_vs_ae=float((copy[:, 1:]-ae[:, 1:]).abs().mean()),
                                generated_motion=float(gen.diff(dim=1).abs().mean()),
                                ae_motion=float(ae.diff(dim=1).abs().mean()))
                            if 'anchor_relative_l2' in batch:
                                row['anchor_relative_l2'] = batch['anchor_relative_l2'][0]
                            if raw is not None:
                                row['rgb_copy_l1_vs_raw'] = float((copy[:,1:]-raw[:,1:]).abs().mean())
                                row['raw_motion'] = float(raw.diff(dim=1).abs().mean())
                                row['rgb_l1_vs_raw'] = float((gen[:, 1:]-raw[:, 1:]).abs().mean())
                                row['ae_l1_vs_raw'] = float((ae[:, 1:]-raw[:, 1:]).abs().mean())
                            row['normalized_target_power'] = float(y.square().mean())
                            row['normalized_generated_power'] = float(generated.square().mean())
                            for value in (.05, .25, .5, .75, .95, 1.):
                                u = torch.full((len(y),), value, device=device)
                                x = (1-value)*y+value*noise
                                with torch.autocast(device_type, dtype=dtype, enabled=dtype != torch.float32):
                                    pred = core(x, u, c, text, valid)
                                row[f'x0_mse_u{value:g}'] = float((flow.clean(pred, x, u)-y).square().mean())
                            append_metrics(out/f'eval_samples_rank{rank:03d}.jsonl', row); rows.append(row)
                            if index < args.preview_clips:
                                from utils.video_preview import save_video_preview
                                videos = dict(ae=ae[0], rgb_copy=copy[0], generated=gen[0])
                                if raw is not None: videos['raw'] = raw[0]
                                save_video_preview(str(preview_root), f'{label}_clip{index}_seed{seed}', videos,
                                    metadata=row, fps=8, save_mp4=not args.cpu_test)
                                atomic_torch_save(dict(cond=cr.cpu(), target=yr.cpu(), generated=inverse(generated, st['target']).cpu(),
                                    video_id=batch['video_id'][0], seed=seed, weights=label),
                                    str(preview_root/f'{label}_clip{index}_seed{seed}.pt'))
            core.train()
            if ddp:
                gathered = [None]*world
                dist.all_gather_object(gathered, rows)
                rows = [r for part in gathered for r in part]
            rows.sort(key=lambda r:(r['weights'],r['video_id'],r['seed']))
            summary = {'step':step}
            for label in ('online', 'ema'):
                group = [r for r in rows if r['weights'] == label]
                for key in ('latent_mse','rgb_l1_vs_ae','generated_motion','ae_motion',
                            'rgb_l1_vs_raw','rgb_copy_l1_vs_raw', 'normalized_target_power',
                            'normalized_generated_power', 'x0_mse_u0.05','x0_mse_u0.25',
                            'x0_mse_u0.5','x0_mse_u0.75','x0_mse_u0.95','x0_mse_u1'):
                    if all(key in r for r in group):
                        summary[f'eval/{label}/{key}'] = float(np.mean([r[key] for r in group]))
            if rank == 0:
                for row in rows: append_metrics(out/'eval_samples.jsonl',row)
                append_metrics(out/'metrics.jsonl', summary)
                for k,v in summary.items():
                    if k != 'step': writer.add_scalar(k,v,step)
                writer.flush()
            restore_rng_state(eval_rng)
            return summary['eval/ema/rgb_l1_vs_ae']

        def save(kind):
            progress('saving:'+kind, step=step)
            rng = capture_rng_state()
            states, cursors = [None]*world, [None]*world
            if ddp:
                dist.all_gather_object(states, rng); dist.all_gather_object(cursors, consumed)
            else: states[0], cursors[0] = rng, consumed
            if rank == 0:
                payload = dict(schema='r7-window-trainer-v1', step=step, best=best,
                    contract=contract, model_args=model_args, model=core.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), ema=ema.state_dict(),
                    ema_metadata=ema.metadata(), rng_by_rank=states, consumed_by_rank=cursors,
                    statistics=stats, args=vars(args))
                atomic_torch_save(payload, str(out/f'checkpoint_{kind}.pt'))
                if kind != 'best_reconstruction':
                    atomic_torch_save(payload, str(out/'checkpoint_latest.pt'))
            if ddp: dist.barrier()

        stop = min(args.max_steps, args.stop_after_steps) if args.stop_after_steps else args.max_steps
        if step > stop:
            raise ValueError('stop budget precedes resumed step')
        progress('reconstruction_baseline',step=step)
        baseline_rng=capture_rng_state()
        reconstruction_baseline()
        restore_rng_state(baseline_rng)
        progress('training', step=step)
        from utils.window_observability import NoiseMeter
        noise_meter = NoiseMeter(device)
        while step < stop:
            started = time.perf_counter(); optimizer.zero_grad(set_to_none=True); loss_sum = 0.
            main_sum = aux_sum = 0.
            for micro in range(args.accum_steps):
                progress('reading_batch', step=step) if step == 0 else None
                batch = next(iterator); consumed += 1
                if set(batch['video_id']) & eval_ids:
                    raise ValueError('training cache overlaps held-out video IDs')
                cr, yr = validate_batch(batch, future, grid, channels)
                c, y = normalize(cr.to(device), st['cond']), normalize(yr.to(device), st['target'])
                text, valid = bank.batch(batch['video_id'], device, args.text_dropout) if bank else (None, None)
                noise, u = torch.randn_like(y), flow.times(len(y), device)
                ue = u[:, None, None, None]; x = (1-ue)*y+ue*noise
                sync = model.no_sync() if ddp and micro < args.accum_steps-1 else contextlib.nullcontext()
                with sync:
                    with torch.autocast(device_type, dtype=dtype, enabled=dtype != torch.float32):
                        if args.aux_layer:
                            prediction, auxiliary = model(x, u, c, text, valid, return_aux=True)
                        else:
                            prediction = model(x, u, c, text, valid)
                    main_loss = flow.loss(prediction, y, noise, u)
                    aux_loss = flow.loss(auxiliary, y, noise, u) if args.aux_layer else main_loss.new_zeros(())
                    loss = main_loss + args.aux_weight*aux_loss
                    if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
                    scaler.scale(loss/args.accum_steps).backward()
                loss_sum += float(loss.detach())/args.accum_steps
                main_sum += float(main_loss.detach())/args.accum_steps
                aux_sum += float(aux_loss.detach())/args.accum_steps
                noise_meter.update(flow, prediction, y, noise, u)
                noise_meter.observe_latents('cond', c)
                noise_meter.observe_latents('target', y)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(core.parameters(), 1.)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite gradient norm')
            scaler.step(optimizer); scaler.update(); scheduler.step(); ema.update(core); step += 1
            if device_type != 'cpu':
                getattr(torch, device_type).synchronize()
            elapsed = time.perf_counter()-started
            measured = torch.tensor([loss_sum, elapsed, main_sum, aux_sum], device=device)
            if ddp:
                dist.all_reduce(measured[:1]); measured[0] /= world
                dist.all_reduce(measured[1:2], op=dist.ReduceOp.MAX)
                dist.all_reduce(measured[2:]); measured[2:] /= world
            log_now = step % args.log_every == 0 or step == 1 or step == stop
            noise_metrics = noise_meter.flush(ddp) if log_now else {}
            if rank == 0 and log_now:
                row = dict(step=step, **{'train/loss':float(measured[0]),'train/grad_norm':float(norm),
                    'train/lr':scheduler.get_last_lr()[0], 'step_seconds':float(measured[1]),
                    'DI_throughput':args.batch_size*args.accum_steps*future*grid*grid/float(measured[1]),
                    'global_batch':args.batch_size*args.accum_steps*world,
                    'samples_seen_global':step*args.batch_size*args.accum_steps*world,
                    'clips_per_second_global':args.batch_size*args.accum_steps*world/float(measured[1]),
                    'consumed_batches_rank0':consumed})
                row.update(noise_metrics)
                row.update({'train/main_loss':float(measured[2]), 'train/aux_loss':float(measured[3]),
                    'train/aux_weighted_loss':args.aux_weight*float(measured[3])})
                if device_type != 'cpu':
                    row['peak_memory_gib'] = getattr(torch, device_type).max_memory_allocated()/2**30
                append_metrics(out/'metrics.jsonl', row); print('[window-train]', row, flush=True)
                for k,v in row.items():
                    if k != 'step': writer.add_scalar(k,v,step)
            if step % args.eval_every == 0 or step == stop:
                # Durable checkpoint BEFORE potentially expensive RGB evaluation.
                save(f'step{step:07d}')
                score = evaluate()
                flag = torch.tensor([float(rank == 0 and score < best)], device=device)
                if rank == 0 and score < best: best = score
                if ddp: dist.broadcast(flag, 0)
                if bool(flag): save('best_reconstruction')
            elif step % args.save_every == 0:
                save(f'step{step:07d}')
            progress('training', step=step) if step % args.log_every == 0 else None
        save('final' if step == args.max_steps else 'paused')
        progress('finished', step=step, status='completed' if step == args.max_steps else 'paused')
        if rank == 0: writer.close()
    except BaseException as exc:
        progress(status.get('phase', 'unknown'), status='failed', error=repr(exc))
        raise
    finally:
        if ddp and dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    run(parse_args())
