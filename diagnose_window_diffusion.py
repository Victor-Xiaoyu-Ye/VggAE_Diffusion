#!/usr/bin/env python3
"""Read-only EMA sampler/generalization/anchor diagnostics; never saves training state."""
import argparse
import itertools
import time
from pathlib import Path
import torch
from torch import distributed as dist
from data.latent_shard_dataset import LatentShardDataset, read_shard_manifest
from models.r7_window_dit import R7WindowDiT
from utils.window_flow import WindowFlow
from utils.window_training import load_artifact, digest, collate, validate_batch, normalize, inverse, validate_statistics
from utils.flow_run_status import atomic_json
from utils.training import append_metrics, atomic_torch_save


ARMS = [('euler64', 'euler', 64, 'normal'), ('euler128', 'euler', 128, 'normal'),
        ('heun64', 'heun', 64, 'normal'), ('zero', 'euler', 64, 'zero'),
        ('shuffle', 'euler', 64, 'shuffle')]


def condition(anchor, donor, mode):
    if mode == 'normal': return anchor
    if mode == 'zero': return torch.zeros_like(anchor)
    if mode == 'shuffle': return donor
    raise ValueError(mode)


def select_samples(manifest, count, seed):
    samples = list(itertools.islice(LatentShardDataset(manifest, 0, seed, False, 0, 1), count))
    ids = [x['video_id'] for x in samples]
    if len(samples) != count or len(set(ids)) != count:
        raise ValueError('insufficient or duplicate diagnostic video IDs')
    return samples


def run(args):
    from utils.device import get_device, get_device_name, configure_backend_compatibility
    from utils.distributed import setup_ddp
    from utils.window_codec import configure_codec, validate_runtime, reconstruction_gate
    from utils.file_signature import sampled_file_signature
    from utils.r7_representation import load_r7_modules, validate_contract
    from utils.video_preview import save_video_preview
    kind = get_device_name()
    if kind == 'cpu': raise RuntimeError('real diagnostics require GPU/NPU')
    ddp, rank, local, world = setup_ddp()
    device = get_device(local); configure_backend_compatibility(kind)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    def status(phase, **extra):
        atomic_json(out/f'diagnostic_status_rank{rank:03d}.json',
                    dict(phase=phase, rank=rank, world_size=world, unix_time=time.time(), **extra))
    try:
        status('loading')
        saved = load_artifact(args.checkpoint)
        if saved.get('schema') != 'r7-window-trainer-v1' or saved['step'] != args.expected_step:
            raise ValueError('checkpoint schema/step mismatch')
        if not saved['args']['no_text']: raise ValueError('this diagnostic requires the uncaptioned baseline')
        stats = saved['statistics']; cfg = validate_statistics(stats)
        mode = validate_runtime(stats['representation'])
        identity = saved['contract']['identity']
        if digest(stats) != identity['statistics']: raise ValueError('checkpoint statistics mismatch')
        for split, manifest in [('train', args.manifest), ('eval', args.eval_manifest)]:
            if read_shard_manifest(manifest) != identity[split+'_manifest']:
                raise ValueError(split+' manifest differs from checkpoint')
        evstats = load_artifact(args.eval_stats)
        if digest(evstats) != identity['eval_statistics']: raise ValueError('eval statistics differ')
        artifact = load_artifact(args.r7_ckpt)
        validate_contract(stats['representation'], artifact['representation_contract'])
        if sampled_file_signature(args.r7_ckpt) != stats['representation']['signatures']['r7']:
            raise ValueError('AE signature mismatch')
        _, _, _, tokenizer, decoder, _ = load_r7_modules(artifact)
        configure_codec(tokenizer, mode)
        for module in (tokenizer, decoder): module.to(device).eval().requires_grad_(False)
        model_args = saved['model_args']
        model = R7WindowDiT(**model_args).to(device).eval().requires_grad_(False)
        model.load_state_dict(saved['model'], strict=True)
        # EMA stores floating tensors only; non-floating buffers retain original values.
        state = model.state_dict()
        expected = {k for k, v in state.items() if v.is_floating_point()}
        if set(saved['ema']) != expected: raise ValueError('incomplete EMA state')
        state.update(saved['ema']); model.load_state_dict(state, strict=True)
        flow = WindowFlow(saved['args']['prediction'], saved['args']['loss_floor'], saved['args']['time_shift'])
        if flow.contract() != saved['contract']['flow']: raise ValueError('flow contract mismatch')
        dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32, 'fp16': torch.float16}[saved['args']['dtype']]
        seed = saved['args']['seed']
        del artifact, saved, state
        st = {k: {n: stats[k][n].to(device).float() for n in ('mean', 'std')} for k in ('cond', 'target')}
        data = {split: select_samples(path, args.clips, seed) for split, path in
                [('train', args.manifest), ('eval', args.eval_manifest)]}
        if {x['video_id'] for x in data['train']} & {x['video_id'] for x in data['eval']}:
            raise ValueError('train/eval diagnostic overlap')
        if rank == 0:
            atomic_json(out/'config.json', dict(args=vars(args), model=model_args, temporal_norm=mode,
                checkpoint_signature=sampled_file_signature(args.checkpoint), arms=ARMS if args.mode == 'standard' else 'see perturbation_contract.json',
                ids={k: [x['video_id'] for x in v] for k, v in data.items()},
                decoder_anchor='original for all arms', zero_condition='zero in normalized space',
                weights='ema', checkpoint_step=args.expected_step))
        if args.mode == 'perturbation':
            from utils.window_perturbation import ALPHAS, audit_sample
            data = {'eval': data['eval']}
            if rank == 0:
                atomic_json(out/'perturbation_contract.json', dict(alphas=ALPHAS,
                    directions=['generated', 'random'], random_matching='RMS per future slot in normalized space',
                    anchor='original native anchor', weights='ema', sampler='euler64',
                    checkpoint_step=args.expected_step, no_training=True))
        if args.mode == 'subspace':
            from utils.window_subspace import fit_basis, audit_sample, ARMS as SUBSPACE_ARMS
            pca_data = select_samples(args.manifest, 64, seed)
            if {x['video_id'] for x in pca_data} & {x['video_id'] for x in data['eval']}:
                raise ValueError('PCA training data overlaps evaluation')
            basis, eigenvalues = fit_basis(pca_data, stats['target'])
            if rank == 0:
                atomic_torch_save(dict(basis=basis, eigenvalues=eigenvalues,
                    video_ids=[x['video_id'] for x in pca_data], statistics_digest=digest(stats)),
                    str(out/'pca_training_basis.pt'))
                atomic_json(out/'subspace_contract.json', dict(oracle=True, no_training=True,
                    pca_source='64 training videos only', arms=SUBSPACE_ARMS,
                    matching='remaining MSE per future slot', frequencies='latent grid cycles/cell; low<=.15 high>.3'))
            del pca_data
            data = {'eval': data['eval']}
        def sync():
            if kind == 'npu': torch.npu.synchronize()
            elif kind == 'cuda': torch.cuda.synchronize()
        @torch.no_grad()
        def decode(c, y):
            seq = torch.cat((c, y), 1).reshape(1, model_args['future']+1, cfg['latent_grid'], cfg['latent_grid'], model_args['channels'])
            geo, tex = tokenizer.decode(seq)
            result = decoder(geo, tex)[..., :3].float()
            if not torch.isfinite(result).all(): raise RuntimeError('nonfinite RGB')
            return result.clamp(0, 1)
        status('ae_gate')
        psnrs = []
        for index, sample in enumerate(data['eval']):
            if index % world != rank: continue
            batch = collate([sample])
            cr, yr = validate_batch(batch, model_args['future'], cfg['latent_grid'], model_args['channels'])
            if 'rgb' not in batch: raise ValueError('eval RAW missing')
            raw = batch['rgb'].to(device).float().div(255).permute(0, 1, 3, 4, 2)
            ae = decode(cr.to(device).float(), yr.to(device).float())
            psnrs.append(float(-10*torch.log10((ae-raw).square().mean().clamp_min(1e-12))))
        if ddp:
            gathered = [None]*world; dist.all_gather_object(gathered, psnrs)
            psnrs = [v for part in gathered for v in part]
        mean_psnr, passed = reconstruction_gate(psnrs, 23.5)
        if rank == 0: atomic_json(out/'ae_gate.json', dict(psnr=mean_psnr, passed=passed, clips=len(psnrs)))
        if not passed: raise RuntimeError('AE gate failed before sampling')
        rows = []
        for split, samples in data.items():
            for index, sample in enumerate(samples):
                if index % world != rank: continue
                batch = collate([sample])
                cr, yr = validate_batch(batch, model_args['future'], cfg['latent_grid'], model_args['channels'])
                cr, yr = cr.to(device).float(), yr.to(device).float()
                c, y = normalize(cr, st['cond']), normalize(yr, st['target'])
                donor_sample = samples[(index+1) % len(samples)]
                donor = normalize(collate([donor_sample])['cond'].to(device).float(), st['cond'])
                with torch.no_grad(): ae = decode(cr, yr)
                raw = batch.get('rgb')
                raw = raw.to(device).float().div(255).permute(0, 1, 3, 4, 2) if raw is not None else None
                if split == 'eval' and raw is None: raise ValueError('eval RAW missing')
                if args.mode in ('perturbation', 'subspace'):
                    rows.extend(audit_sample(model=model, flow=flow, c=c, y=y, cr=cr,
                        ae=ae, raw=raw, stats=st['target'], decode=decode, sync=sync,
                        dtype=dtype, seeds=args.seeds, index=index, video_id=sample['video_id'],
                        out=out, rank=rank, previews=args.previews, status=status,
                        **({'basis': basis} if args.mode == 'subspace' else {})))
                    continue
                for sample_seed in args.seeds:
                    noise = torch.randn(y.shape, generator=torch.Generator().manual_seed(sample_seed+1009*index)).to(device)
                    reference = None
                    for arm, method, steps, ablation in ARMS:
                        status('sampling', split=split, clip=index, seed=sample_seed, arm=arm)
                        sync(); start = time.perf_counter()
                        z = flow.sample(model, condition(c, donor, ablation), noise, steps, dtype, method=method)
                        sync(); elapsed = time.perf_counter()-start
                        gen = decode(cr, inverse(z, st['target']))
                        if reference is None: reference = z.clone()
                        row = dict(split=split, video_id=sample['video_id'], index=index, seed=sample_seed,
                            arm=arm, method=method, steps=steps, nfe=steps if method == 'euler' else 2*steps-1,
                            donor_video_id=donor_sample['video_id'] if ablation == 'shuffle' else None,
                            latent_mse=float((z-y).square().mean()), rgb_l1_vs_ae=float((gen[:,1:]-ae[:,1:]).abs().mean()),
                            delta_latent_vs_euler64=float((z-reference).square().mean()),
                            generated_motion=float(gen.diff(dim=1).abs().mean()), ae_motion=float(ae.diff(dim=1).abs().mean()),
                            sample_seconds=elapsed, DI_throughput=y.shape[1]*y.shape[2]*steps/elapsed,
                            model_token_evals_per_second=y.shape[1]*y.shape[2]*(steps if method == 'euler' else 2*steps-1)/elapsed)
                        if raw is not None:
                            row.update(rgb_l1_vs_raw=float((gen[:,1:]-raw[:,1:]).abs().mean()),
                                rgb_copy_l1_vs_raw=float((raw[:,:1]-raw[:,1:]).abs().mean()),
                                ae_psnr=float(-10*torch.log10((ae-raw).square().mean().clamp_min(1e-12))))
                        append_metrics(out/f'metrics_rank{rank:03d}.jsonl', row); rows.append(row)
                        if index < args.previews:
                            videos = dict(ae=ae[0], generated=gen[0])
                            if raw is not None: videos.update(raw=raw[0], rgb_copy=raw[0,:1].expand_as(raw[0]))
                            name = f'{split}_clip{index}_seed{sample_seed}_{arm}'
                            save_video_preview(str(out/'samples'), name, videos, metadata=row, fps=8, save_mp4=True)
                            atomic_torch_save(dict(generated=inverse(z, st['target']).cpu(), **row), str(out/'samples'/f'{name}.pt'))
        if ddp:
            gathered = [None]*world; dist.all_gather_object(gathered, rows)
            rows = [r for part in gathered for r in part]
        if rank == 0:
            if args.mode == 'standard': arms = [r[0] for r in ARMS]
            elif args.mode == 'subspace': arms = list(SUBSPACE_ARMS)
            else: arms = [f'{direction}_a{alpha:g}' for direction in ('generated', 'random') for alpha in ALPHAS]
            expected_rows = len(data)*args.clips*len(args.seeds)*len(arms)
            if len(rows) != expected_rows: raise ValueError('incomplete diagnostic matrix')
            summary = {}
            for split in data:
                for arm in arms:
                    group = [r for r in rows if r['split'] == split and r['arm'] == arm]
                    summary[split+'/'+arm] = {k: sum(r[k] for r in group)/len(group) for k in group[0]
                        if isinstance(group[0][k], float)}
            atomic_json(out/'summary.json', dict(groups=summary, ae_psnr=mean_psnr, ae_gate_passed=passed,
                complete=True, rows=len(rows), note='Pixel errors and finite perturbation ratios are not perceptual/geometry scores or Jacobian norms. DI is sampler token-steps/s per active device, excludes decode/IO. Sampling time is repeated across perturbation rows; do not sum it.'))
            for row in rows: append_metrics(out/'metrics.jsonl', row)
            if not passed: raise RuntimeError('AE gate failed; diagnostics not quality-valid')
        if ddp: dist.barrier()
        status('completed')
    except Exception as exc:
        status('failed', error=str(exc)); raise
    finally:
        if ddp and dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'r7_ckpt', 'manifest', 'eval_manifest', 'eval_stats', 'output_dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--expected_step', type=int, default=6000)
    p.add_argument('--mode', choices=['standard', 'perturbation', 'subspace'], default='standard')
    p.add_argument('--clips', type=int, default=16)
    p.add_argument('--previews', type=int, default=4)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    args = p.parse_args()
    if args.clips < 2 or args.previews < 0 or len(set(args.seeds)) != len(args.seeds): p.error('invalid clips/previews/seeds')
    run(args)
