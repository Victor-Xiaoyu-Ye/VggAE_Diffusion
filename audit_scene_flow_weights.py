"""Read-only, resumable online versus historical EMA inference from Scene runs.

Uses saved validation RGB/latents/conditions and the frozen AE. No encoder,
optimizer, training data or EMA update is involved. Source artifacts stay intact.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import threading
import time

import torch
import torch.distributed as dist

from models.scene_flow import SceneFlow
from models.scene_rae import SceneRAE
from utils.native_audit_io import recover_case, seal, safe_relative
from utils.scene_run import ArtifactStore, atomic, digest
from utils.training import append_metrics, atomic_torch_save
from utils.window_flow import WindowFlow
from train_scene_pipeline import amp, barrier, distributed, gather, memory_gib, video


SCHEMA = 'scene-weight-audit-v1'


def profile_checkpoints(profile):
    final = dict(label='video_final', stage='video', file='video/checkpoint_final.pt')
    if profile == 'smoke':
        return [final]
    if profile != 'standard':
        raise ValueError('Unknown audit profile')
    return [dict(label='image_final', stage='image', file='image/checkpoint_final.pt'),
            dict(label='video_s9185', stage='video', file='video/weights_step0009185.pt'),
            dict(label='video_s71609', stage='video', file='video/weights_step0071609.pt'), final]


def select_cases(samples, per_domain=2, max_cases=0):
    if per_domain < 0 or max_cases < 0:
        raise ValueError('Case limits must be nonnegative; zero means all')
    counts, selected = Counter(), []
    for index, sample in enumerate(samples):
        source = sample['dataset']
        if per_domain and counts[source] >= per_domain:
            continue
        selected.append((index, sample))
        counts[source] += 1
        if max_cases and len(selected) >= max_cases:
            break
    return selected


def jobs_for_rank(selected, rank, world):
    if world < 1 or not 0 <= rank < world:
        raise ValueError('Invalid rank/world')
    jobs = [(index, sample, weight_source) for index, sample in selected
            for weight_source in ('online', 'legacy_ema')]
    return jobs[rank::world]


@contextmanager
def heartbeat(message, rank):
    stop = threading.Event()
    start = time.monotonic()
    print(f'[scene audit] rank={rank} {message}', flush=True)
    def report():
        while not stop.wait(30):
            print(f'[scene audit] rank={rank} {message}; elapsed={time.monotonic()-start:.0f}s', flush=True)
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def receipt_spec(store, name):
    receipt = store.json(safe_relative(name)+'.json', required=True)
    return dict(file=name, **{k: receipt[k] for k in ('identity', 'size', 'sha256')})


def build_plan(source, args):
    contract = source.json('contract.json', required=True)
    manifest = source.json('cache/complete.json', required=True)
    ae_done = source.json('ae/complete.json', required=True)
    ae = receipt_spec(source, ae_done['selected'])
    evaluation = receipt_spec(source, 'cache/eval.pt')
    if manifest['pipeline'] != contract['identity'] or manifest['ae'] != ae['sha256']:
        raise ValueError('Source cache does not match its pipeline/frozen AE')
    if evaluation['identity'] != manifest['identity'] or ae['identity'] != ae_done['identity']:
        raise ValueError('Source AE/evaluation receipt identity mismatch')
    checkpoints = []
    for entry in profile_checkpoints(args.profile):
        model_info = source.json(entry['stage']+'/model.json', required=True)
        receipt = receipt_spec(source, entry['file'])
        expected = digest(dict(pipeline=contract['identity'], cache=manifest['identity'],
                               stage=entry['stage'], model=model_info['model']))
        if receipt['identity'] != expected:
            raise ValueError('Source flow belongs to another pipeline/cache: '+entry['file'])
        checkpoints.append(dict(entry, receipt=receipt, model_args=model_info['model']))
    plan = dict(schema=SCHEMA, pipeline=contract['identity'], cache_identity=manifest['identity'],
        ae=ae, evaluation=evaluation, mean=manifest['mean'], std=manifest['std'],
        checkpoints=checkpoints, profile=args.profile, steps=args.sample_steps,
        method=args.sample_method, cases_per_domain=args.cases_per_domain, max_cases=args.max_cases,
        seed_rule='101 + original evaluation cache index', cfg=1,
        weight_sources=['online', 'legacy_ema'], quality_policy='report_only')
    return dict(plan, identity=digest(plan))


def broadcast(value, device):
    if not dist.is_initialized():
        return value
    values = [value]
    dist.broadcast_object_list(values, src=0, device=device)
    return values[0]


def stage_asset(source, spec, local_rank, rank):
    """Only each node leader downloads/hashes. Other readers share its mmap file."""
    name = safe_relative(spec['file'])
    if local_rank == 0:
        with heartbeat('staging '+name, rank):
            payload = source.load(name, spec['identity'], required=True)
            del payload
            gc.collect()
            got = source.json(name+'.json', required=True)
            if any(got[k] != spec[k] for k in ('identity', 'sha256', 'size')):
                raise ValueError('Source changed after audit contract was pinned: '+name)
    barrier()
    path = source.local/name
    receipt = json.loads(path.with_name(path.name+'.json').read_text(encoding='utf8'))
    if path.stat().st_size != spec['size'] or receipt['sha256'] != spec['sha256']:
        raise ValueError('Node-local staging does not match pinned receipt: '+name)
    return path


def load_staged(path):
    # Ascend torch_npu mmap requires a string filename, not pathlib.Path.
    return torch.load(str(path), map_location='cpu', weights_only=False, mmap=os.name != 'nt')


@torch.no_grad()
def audit_case(model, rae, sample, index, stage, device, mean, std,
               steps=32, method='euler', progress=None):
    if stage not in ('image', 'video'):
        raise ValueError('Expected image or video')
    image_task = stage == 'image'
    anchor = (sample['anchor'].unsqueeze(0).to(device).float()-mean)/std
    text = sample['text'].unsqueeze(0).to(device).float()
    valid = torch.ones(text.shape[:2], device=device, dtype=torch.bool)
    frames = 1 if image_task else sample['target'].shape[0]-1
    rays = None if image_task else sample['rays'][1:].unsqueeze(0).to(device)
    present = torch.tensor([not image_task], device=device)
    clean = sample['anchor'] if image_task else sample['target'][1:]
    truth_z = (clean.unsqueeze(0).to(device).float()-mean)/std
    generator = torch.Generator(device=device).manual_seed(101+index)
    noise = torch.randn((1, frames, rae.grid**2, rae.channels), device=device, generator=generator)
    trace = []
    def net(z, u, c, t, v):
        return model(z, u, c, t, v, ref_present=present, rays=rays)
    def observer(iteration, u, state, x0):
        if iteration % 8 and iteration != steps-1:
            return
        row = dict(iteration=iteration, noise_time=u,
                   state_std=state.float().std(unbiased=False).item(),
                   x0_std=x0.float().std(unbiased=False).item(),
                   x0_target_mse=(x0.float()-truth_z).square().mean().item())
        trace.append(row)
        if progress is not None:
            progress(row)
    z = WindowFlow().sample(net, anchor, noise, steps=steps, method=method,
        dtype=torch.bfloat16 if device.type != 'cpu' else torch.float32,
        text=text, text_valid=valid, observer=observer)
    latent = z*std+mean
    if not image_task:
        latent = torch.cat((sample['anchor'].unsqueeze(0).to(device).float(), latent), 1)
    with amp(device):
        rgb = rae.decode(latent.reshape(1, -1, rae.grid, rae.grid, rae.channels))[0].float()
        clean_full = sample['anchor'] if image_task else sample['target']
        ae_rgb = rae.decode(clean_full.unsqueeze(0).to(device).float().reshape(
            1, -1, rae.grid, rae.grid, rae.channels))[0].float()
    if not torch.isfinite(rgb).all():
        raise RuntimeError('Nonfinite decoded RGB')
    raw = sample['raw'].float().permute(0, 2, 3, 1).to(device)/255
    if image_task:
        raw, ae_rgb = raw[:1], ae_rgb[:1]
    score = slice(None) if image_task else slice(1, None)
    nfe = steps if method == 'euler' else 2*steps-1
    row = dict(id=sample['id'], dataset=sample['dataset'], case_index=index, seed=101+index,
        raw_l1=(rgb[score]-raw[score]).abs().mean().item(),
        ae_l1=(ae_rgb[score]-raw[score]).abs().mean().item(),
        copy_l1=(raw[:1].expand_as(raw)[score]-raw[score]).abs().mean().item(),
        latent_mse=(z-truth_z).square().mean().item(),
        latent_mean=z.mean().item(), latent_std=z.std(unbiased=False).item(),
        metric_scope='single generated image' if image_task else 'future frames only',
        has_caption=sample['has_caption'], camera_present=bool(rays is not None and rays[..., 7].any()),
        denoiser_evaluations=nfe, denoiser_tokens=nfe*frames*rae.grid**2,
        columns=['RAW', 'AE', 'GENERATED'], cfg=1)
    return row, torch.cat((raw, ae_rgb, rgb), 2).cpu(), z.cpu(), trace


def ensure_disjoint(args):
    def normalized(value):
        if '://' in value:
            return value.rstrip('/')
        return Path(value).resolve().as_posix().rstrip('/').casefold()
    inputs = [normalized(s) for s in [args.source_local, *args.source_root] if s]
    outputs = [normalized(s) for s in [args.output, *args.root] if s]
    for source in inputs:
        for target in outputs:
            if source == target or source.startswith(target+'/') or target.startswith(source+'/'):
                raise ValueError('Audit input and output directories must be separate')


def free_device(device):
    gc.collect()
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()


def summary(rows, expected, identity):
    completed = [r for r in rows if r.get('status') == 'completed']
    errors = [r for r in rows if r.get('status') != 'completed']
    groups = {}
    for row in completed:
        key = row['checkpoint_label']+'/'+row['weight_source']
        groups.setdefault(key, []).append(row)
    means = {key: dict(cases=len(values), **{
        name: sum(v[name] for v in values)/len(values)
        for name in ('raw_l1', 'ae_l1', 'copy_l1', 'latent_mse')}) for key, values in groups.items()}
    return dict(schema=SCHEMA, identity=identity,
        status='completed' if len(completed) == expected and not errors else 'partial',
        expected=expected, completed=len(completed), errors=len(errors),
        remaining=expected-len(completed), means=means, samples=rows,
        note='Review RGB and motion; single-target pixel L1 is not a generative quality pass rate.')


def run(args, device, rank, world):
    ensure_disjoint(args)
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
    node_rank = int(os.environ.get('GROUP_RANK', str(rank//local_world)))
    output = ArtifactStore(args.output, args.root)
    # Downloads belong to the new audit. Never write or remove any source file.
    source = ArtifactStore(Path(args.output)/'_inputs'/f'node{node_rank}',
                           [args.source_local, *args.source_root])
    started, tokens = time.monotonic(), 0
    with heartbeat('reading and pinning source receipts', rank):
        plan = broadcast(build_plan(source, args) if rank == 0 else None, device)
    if rank == 0:
        old = output.json('contract.json')
        if old and old['identity'] != plan['identity']:
            raise ValueError('Audit contract changed; select a fresh SCENE_AUDIT_NAMESPACE')
        output.write_json('contract.json', dict(plan, world=world,
            source_roots=[args.source_local, *args.source_root]))
    barrier()
    eval_path = stage_asset(source, plan['evaluation'], local_rank, rank)
    samples = load_staged(eval_path)['samples']
    selected = select_cases(samples, args.cases_per_domain, args.max_cases)
    if not selected:
        raise ValueError('No validation samples available')
    selected_meta = [dict(index=i, id=s['id'], dataset=s['dataset'], seed=101+i) for i, s in selected]
    if rank == 0:
        output.write_json('selection.json', dict(identity=plan['identity'], samples=selected_meta))
    expected = len(selected)*2*len(plan['checkpoints'])
    jobs = jobs_for_rank(selected, rank, world)
    ae_path = stage_asset(source, plan['ae'], local_rank, rank)
    ae_payload = load_staged(ae_path)
    rae = SceneRAE(ae_payload['spec']['source_config'], ae_payload['spec']['channels'])
    rae.load_state_dict(ae_payload['model'], strict=True)
    rae.eval().requires_grad_(False).to(device)
    del ae_payload
    mean = torch.tensor(plan['mean'], device=device, dtype=torch.float32)
    std = torch.tensor(plan['std'], device=device, dtype=torch.float32).clamp_min(1e-4)
    local_rows = []
    for checkpoint in plan['checkpoints']:
        label = checkpoint['label']
        pending = []
        for index, sample, weight_source in jobs:
            case = f'{label}/{weight_source}/case{index:03d}'
            identity = digest(dict(audit=plan['identity'], checkpoint=checkpoint['receipt']['sha256'],
                case_index=index, id=sample['id'], seed=101+index, weight_source=weight_source))
            receipt_name = case+'/complete.json'
            try:
                recovered = recover_case(output.local, receipt_name, identity, output.roots)
            except RuntimeError as exc:
                print(f'[scene audit] rank={rank} recomputing incomplete case {case}: {exc}', flush=True)
                recovered = None
            if recovered is not None:
                # Recovering a locally committed case also repairs either remote replica.
                output.publish([*recovered['files'], receipt_name])
                local_rows.append(recovered['metadata'])
                print(f'[scene audit] rank={rank} resumed {case}', flush=True)
            else:
                pending.append((index, sample, weight_source, case, identity))
        counts = gather(len(pending), world)
        if any(counts):
            path = stage_asset(source, checkpoint['receipt'], local_rank, rank)
            # No optimizer restored; mmap keeps large source tensor pages shared per node.
            payload = load_staged(path) if pending else None
            if pending:
                if payload['model_args'] != checkpoint['model_args']:
                    raise ValueError('Checkpoint model layout differs from saved model.json')
                model = SceneFlow(**checkpoint['model_args']).eval().requires_grad_(False).to(device)
                step = int(payload['step'])
                stem = Path(checkpoint['file']).stem
                if stem.startswith('weights_step') and step != int(stem[len('weights_step'):]):
                    raise ValueError('Snapshot internal step disagrees with its filename')
                for weight_source in ('online', 'legacy_ema'):
                    arm = [job for job in pending if job[2] == weight_source]
                    if not arm:
                        continue
                    state = payload['model' if weight_source == 'online' else 'ema']
                    stored_dtypes = sorted({str(v.dtype) for v in state.values()})
                    # FP32 module holds original saved values. No EMA update or repair.
                    model.load_state_dict(state, strict=True)
                    for index, sample, _, case, identity in arm:
                        print(f'[scene audit] rank={rank} start {case} step={step} seed={101+index}', flush=True)
                        def progress(row):
                            print(f'[scene audit] rank={rank} {case} sample_step={row["iteration"]+1}/{args.sample_steps} '
                                  f'u={row["noise_time"]:.5f}', flush=True)
                        try:
                            metrics, frames, latent, trace = audit_case(model, rae, sample, index,
                                checkpoint['stage'], device, mean, std, args.sample_steps,
                                args.sample_method, progress)
                            video(output.local/(case+'/preview.mp4'), frames)
                            atomic_torch_save(dict(normalized_generated=latent, trace=trace,
                                mean=mean.cpu(), std=std.cpu(), seed=101+index,
                                identity=identity), str(output.local/(case+'/generated.pt')))
                            tokens += metrics['denoiser_tokens']
                            row = dict(metrics, status='completed', checkpoint_label=label,
                                checkpoint=checkpoint['file'], checkpoint_step=step,
                                checkpoint_sha256=checkpoint['receipt']['sha256'], weight_source=weight_source,
                                stored_dtypes=stored_dtypes, step=step, rank=rank, path=case,
                                DI_throughput=tokens/max(time.monotonic()-started, 1e-6),
                                scope='denoiser token evaluations per active NPU; job wall time including staging/IO; guard excluded',
                                peak_allocated_gib=memory_gib(device))
                            atomic(output.local/(case+'/metrics.json'), row)
                            files = [case+'/'+name for name in ('preview.mp4', 'generated.pt', 'metrics.json')]
                            seal(output.local, case+'/complete.json', identity, files, row)
                            output.publish([*files, case+'/complete.json'])
                            del frames, latent
                        except (RuntimeError, ValueError, OSError) as exc:
                            # Technical per-case failures remain visible and are retried on resume.
                            row = dict(status='error', checkpoint_label=label, weight_source=weight_source,
                                       case_index=index, id=sample['id'], seed=101+index, path=case, error=repr(exc))
                            output.write_json(case+'/error.json', row)
                            local_rows.append(row)
                            print(f'[scene audit] rank={rank} failed {case}: {exc}', flush=True)
                            free_device(device)
                        else:
                            local_rows.append(row)
                            # Logging failure must not turn an already committed
                            # case into a duplicate success+error in the summary.
                            append_metrics(str(output.local/f'metrics-r{rank:03d}.jsonl'), row)
                            output.publish([f'metrics-r{rank:03d}.jsonl'])
                            print(f'DI_throughput: {row["DI_throughput"]:.2f} tokens/s/npu '
                                  f'(rank={rank}, {row["scope"]})', flush=True)
                            print(f'[scene audit] rank={rank} committed {case}', flush=True)
                del state, model
            del payload
            free_device(device)
            barrier()
        barrier()
        if local_rank == 0:
            # Also clean a leftover stage file when every case resumed. This
            # only touches our audit cache, never an original source checkpoint.
            path = source.local/safe_relative(checkpoint['file'])
            if not path.resolve().is_relative_to(source.local.resolve()):
                raise ValueError('Unexpected staging path')
            path.unlink(missing_ok=True)
            path.with_name(path.name+'.json').unlink(missing_ok=True)
        gathered = gather(local_rows, world)
        if rank == 0:
            report = summary([row for rows in gathered for row in rows], expected, plan['identity'])
            output.write_json('summary.json', report)
            print(f'[scene audit] completed={report["completed"]}/{expected} errors={report["errors"]}', flush=True)
    barrier()
    gathered = gather(local_rows, world)
    report = summary([row for rows in gathered for row in rows], expected, plan['identity'])
    if rank == 0:
        output.write_json('complete.json', report)
    barrier()
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-local', required=True)
    p.add_argument('--source-root', action='append', default=[])
    p.add_argument('--output', required=True)
    p.add_argument('--root', action='append', default=[])
    p.add_argument('--profile', choices=['smoke', 'standard'], default='standard')
    p.add_argument('--sample-steps', type=int, default=32)
    p.add_argument('--sample-method', choices=['euler', 'heun'], default='euler')
    p.add_argument('--cases-per-domain', type=int, default=0, help='0 selects all cached cases (default)')
    p.add_argument('--max-cases', type=int, default=0)
    args = p.parse_args(argv)
    if args.sample_steps < 1 or args.cases_per_domain < 0 or args.max_cases < 0:
        p.error('Invalid sampling/case limits')
    return args


def main():
    args = parse_args()
    device, rank, world = distributed()
    try:
        report = run(args, device, rank, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if report['status'] != 'completed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
