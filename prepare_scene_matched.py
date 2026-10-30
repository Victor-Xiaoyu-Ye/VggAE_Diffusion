"""One exact-RGB feature bank and an original-C256 free-generation control.

The two new codec arms share every accepted window and failed-data decision.
Only source train-audit/dev cases retain the separate original-518 RGB/latents.
Neither representation extraction nor the original-source audit trains a model.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F

from audit_scene_end_to_end import code_signature
from audit_scene_flow_weights import heartbeat, load_staged, receipt_spec, stage_asset
from train_scene_pipeline import amp, barrier, distributed, gather, make_encoder, video
from utils.native_audit_io import recover_case, seal
from utils.scene_run import ArtifactStore, digest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare', 'audit'), default='prepare')
    p.add_argument('--config', required=True)
    p.add_argument('--cohort', required=True)
    p.add_argument('--encoder')
    p.add_argument('--source-local', required=True)
    p.add_argument('--source-root', action='append', default=[])
    p.add_argument('--output', required=True)
    p.add_argument('--root', action='append', default=[])
    for name in ('bank', 'generator', 'codec'):
        p.add_argument('--'+name+'-local')
        p.add_argument('--'+name+'-root', action='append', default=[])
    return p


def limits(cfg):
    # Top-level fields are the public launcher contract; nested dataset is also
    # accepted for existing Scene configurations.
    data = cfg.get('dataset', {})
    result = {name: int(cfg.get(name, data.get(name, default))) for name, default in
        (('train_windows', 512), ('dev_cases', 64), ('train_eval_cases', 16),
         ('views', 9), ('size', 252), ('grid', 18), ('history', 4))}
    if any(value < 1 for value in result.values()) or result['history'] >= result['views']:
        raise ValueError('Positive data limits and a nonempty future are required')
    if result['size'] % 14 or result['size']//14 != result['grid']:
        raise ValueError('Matched extraction uses the native patch grid: size must equal grid*14')
    return result


def job_references(manifest, cfg):
    """One immutable job ID per source shard/index, with fixed train probes."""
    spec = limits(cfg)
    jobs = []
    train_index = 0
    for shard in manifest['shards']:
        if len(shard.get('ids', [])) != shard['count']:
            raise ValueError('Prepared source shards must expose every sample ID')
        for index, ident in enumerate(shard['ids']):
            if train_index >= spec['train_windows']:
                break
            jobs.append(dict(split='train', index=train_index, source_file=shard['file'],
                source_index=index, expected_id=ident, source_audit=False,
                file=f'data/train/{train_index:06d}.pt'))
            train_index += 1
        if train_index >= spec['train_windows']:
            break
    # Source compaction can be grouped by activity/shard. Do not accidentally
    # choose a single leading stratum as the whole training-fit diagnostic.
    groups = {}
    for job in jobs:
        groups.setdefault(job['expected_id'], []).append(job)
    seed = int(cfg.get('seed', 42))
    chosen_ids = sorted(groups, key=lambda ident: digest([seed, 'train-probe', ident]))[:spec['train_eval_cases']]
    for ident in chosen_ids:
        min(groups[ident], key=lambda j: digest([seed, 'train-window', j['source_file'], j['source_index']]))['source_audit'] = True
    for index in range(min(spec['dev_cases'], int(manifest['eval_cases']))):
        jobs.append(dict(split='dev', index=index, source_file='data/eval.pt',
            source_index=index, source_audit=True, file=f'data/dev/{index:06d}.pt'))
    if not any(job['split'] == 'train' for job in jobs) or not any(job['split'] == 'dev' for job in jobs):
        raise ValueError('The source bank must contain both train and dev data')
    return jobs


def make_plan(source, cfg, cohort_path, encoder):
    contract = source.json('contract.json', True)
    manifest = source.json('data/cache_complete.json', True)
    if manifest['identity'] != contract['identity']:
        raise ValueError('Prepared source data does not match its contract')
    jobs = job_references(manifest, cfg)
    files = sorted({j['source_file'] for j in jobs})
    receipts = [receipt_spec(source, name) for name in files]
    if any(r['identity'] != contract['identity'] for r in receipts):
        raise ValueError('Prepared source receipts have mixed identities')
    # Full checkpoint hash once per node avoids a path-only encoder identity.
    from utils.native_audit_io import sha256
    if not encoder or not Path(encoder).is_file():
        raise ValueError('An existing local StreamVGGT checkpoint is required')
    fields = dict(schema='scene-matched-feature-bank-v1', config=cfg,
        source_contract=contract, source_manifest_sha=digest(manifest), sources=receipts,
        jobs=jobs, limits=limits(cfg), cohort_sha256=sha256(cohort_path),
        encoder=dict(size=Path(encoder).stat().st_size, sha256=sha256(encoder)),
        code_sha256=code_signature(), feature_dims=dict(dino=1024, stream=2048),
        preprocessing='source518 center crop; torch bicubic antialias resize; round uint8; divide255; shared full/prefix forwards',
        conditioning='Every observed feature slot is replaced by independent-prefix extraction; future RGB cannot enter conditions',
        selection='Reuse stage58 subset; deterministic seeded unique-ID train probes across all selected windows; source-disjoint dev',
        failure_policy='One shared skip for unreadable RGB; model/extraction/publication errors are fatal',
        source_audit_resolution=518)
    return dict(fields, identity=digest(fields))


def pin(store, plan, rank):
    identities = gather(plan['identity'], dist.get_world_size() if dist.is_initialized() else 1)
    if len(set(identities)) != 1:
        raise ValueError('Nodes disagree on pinned encoder/code/config/source identity')
    if rank == 0:
        previous = store.json('contract.json')
        if previous and previous['identity'] != plan['identity']:
            raise ValueError('Changed code/config/source: use a fresh matched namespace')
        store.write_json('contract.json', plan)
    barrier()


def source_raw(sample, row, localize, views):
    if sample['dataset'] != 'spatialvid' or len(sample['frame_ids']) != views:
        raise ValueError('Expected the fixed SpatialVID window from the shared source bank')
    ids = sample['frame_ids']
    if any(type(i) is not int or i < 0 for i in ids) or ids != sorted(set(ids)):
        raise ValueError('Source frame IDs must be strictly increasing original indices')
    if 'raw' in sample:
        raw = torch.as_tensor(sample['raw'])
        if raw.dtype != torch.uint8 or tuple(raw.shape) != (views, 3, 518, 518):
            raise ValueError('Sealed source RGB must be uint8 [T,3,518,518]')
        return raw.clone(), dict(source='sealed_original_518_RGB', frame_ids=ids)
    if row is None:
        raise ValueError('Source ID is absent from the original RGB cohort')
    from utils.scene_control_data import replay_rgb
    rgb, provenance = replay_rgb(sample, row, localize, size=518)
    return (rgb*255).round().to(torch.uint8), provenance


def resize_matched(raw, size):
    """Quantized RGB is simultaneously the feature input and decoder target."""
    return F.interpolate(raw.float()/255., (size, size), mode='bicubic',
        align_corners=False, antialias=True).clamp(0, 1).mul(255).round().to(torch.uint8)


def prefix_safe_features(full, prefix, history):
    """Measure causal consistency and make it exact without quality gating.

    Future RGB may affect a future target, but never an observed condition.
    The learned codec is framewise on its encoder side, so replacing these
    feature slots also protects downstream normalized latent conditions.
    """
    result, report = {}, {}
    for name in ('dino', 'stream'):
        observed = prefix[name]
        expected = full[name][:, :history]
        if observed.shape != expected.shape or observed.shape[1] != history:
            raise ValueError('Independent-prefix feature layout differs: '+name)
        if not torch.isfinite(observed).all():
            raise ValueError('Nonfinite independent-prefix features: '+name)
        delta = expected.float()-observed.float()
        report[name] = dict(relative_rms=float(delta.square().mean().sqrt()/
            observed.float().square().mean().sqrt().clamp_min(1e-12)),
            max_abs=float(delta.abs().max()), independent_prefix_used=True)
        result[name] = torch.cat((observed, full[name][:, history:]), dim=1)
    return result, report


def matched_payload(sample, raw, provenance, features, cfg, source_audit=False):
    spec = limits(cfg)
    payload = dict(id=str(sample['id']), dataset=sample['dataset'],
        frame_ids=list(sample['frame_ids']), raw=resize_matched(raw, spec['size']),
        provenance=provenance)
    for key in ('nominal_fps', 'source_fps', 'fps', 'time_status'):
        if key in sample:
            payload[key] = sample[key]
    for name, dim in (('dino', 1024), ('stream', 2048)):
        value = features[name]
        if tuple(value.shape) != (1, spec['views'], spec['grid']**2, dim):
            raise ValueError(f'Unexpected {name} feature shape: {tuple(value.shape)}')
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite feature extraction: '+name)
        payload[name] = value[0].detach().to(device='cpu', dtype=torch.bfloat16).contiguous()
    if source_audit:
        keys = ('id', 'dataset', 'frame_ids', 'target', 'anchor', 'nominal_fps', 'source_fps', 'fps', 'time_status')
        payload['source_sample'] = {k: (sample[k].detach().cpu().clone()
            if torch.is_tensor(sample[k]) else sample[k]) for k in keys if k in sample}
        payload['source_sample']['raw'] = raw
    return payload


def check_payload(payload, job, cfg):
    spec = limits(cfg)
    if job.get('expected_id', payload['id']) != payload['id']:
        raise ValueError('Source sample ID changed')
    if len(payload['frame_ids']) != spec['views']:
        raise ValueError('Cached paired frame IDs are incomplete')
    if payload['raw'].dtype != torch.uint8 or tuple(payload['raw'].shape) != (spec['views'], 3, spec['size'], spec['size']):
        raise ValueError('Cached paired RGB shape/dtype changed')
    for name, dim in (('dino', 1024), ('stream', 2048)):
        value = payload[name]
        if value.dtype != torch.bfloat16 or tuple(value.shape) != (spec['views'], spec['grid']**2, dim) or not torch.isfinite(value).all():
            raise ValueError('Invalid cached paired feature: '+name)
    if job['source_audit'] and 'source_sample' not in payload:
        raise ValueError('Missing original representation audit input')


def bank_manifest(plan, reports):
    records = sorted([row for report in reports for row in report['records']],
                     key=lambda row: (row['split'], row['index']))
    train = [r for r in records if r['split'] == 'train']
    dev = [r for r in records if r['split'] == 'dev']
    if {r['id'] for r in train} & {r['id'] for r in dev}:
        raise ValueError('Source ID leakage between train and dev')
    if not train or not dev:
        raise RuntimeError('Shared usable train or dev split is empty')
    spec = plan['limits']
    return dict(identity=plan['identity'], schema=plan['schema'], train=train, dev=dev,
        train_eval=[r for r in train if r['source_audit']], feature_dims=plan['feature_dims'],
        views=spec['views'], history=spec['history'], size=spec['size'], grid=spec['grid'],
        bad=[row for report in reports for row in report['bad']], requested=spec,
        source_identity=plan['source_contract']['identity'],
        source_audit_resolution=518, raw_target_resolution=spec['size'],
        note='All arms use the same accepted samples. Source audit and new codecs have different resolutions.')


def prepare(a, cfg, store, source, device, rank, world):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    # Node leaders hash the 5GB encoder once. Other ranks obtain that same plan.
    plan_name = f'prepare/node{int(os.environ.get("GROUP_RANK", 0))}_plan.json'
    if local_rank == 0:
        plan = make_plan(source, cfg, a.cohort, a.encoder)
        from utils.scene_run import atomic
        atomic(store.local/plan_name, plan)
    barrier()
    plan = json.loads((store.local/plan_name).read_text(encoding='utf8'))
    pin(store, plan, rank)
    completed_bank = store.json('data/bank.json')
    if completed_bank and completed_bank['identity'] != plan['identity']:
        raise ValueError('Completed paired bank identity differs from its plan')
    # Once training can consume the bank, its accepted subset is immutable.
    # A previously missing video becoming readable on a retry must not silently
    # change the examples/cursors under an existing codec or DiT checkpoint.
    frozen_skips = {(row['split'],row['index']):row
                    for row in (completed_bank or {}).get('bad',[])}
    paths = {item['file']: stage_asset(source, item, local_rank, rank) for item in plan['sources']}
    cohort = json.loads(Path(a.cohort).read_text(encoding='utf8'))
    rows = {r['id']: r for r in cohort['records']}
    from data.scene_rgb import SceneRGB
    reader = SceneRGB(cohort)
    from models.matched_video_rae import extract_matched_features
    assigned = plan['jobs'][rank::world]
    records, bad, encoder = [], [], None
    current_file, current_samples = None, None
    start = time.monotonic()
    encoded = 0
    processed_new = 0
    for job in assigned:
        previous_skip = frozen_skips.get((job['split'],job['index']))
        if previous_skip is not None:
            bad.append(previous_skip)
            continue
        # Publication and model failures never become a corrupt-data skip.
        payload = store.load(job['file'], plan['identity'])
        if payload is None:
            try:
                if current_file != job['source_file']:
                    current_samples = load_staged(paths[job['source_file']])['samples']
                    current_file = job['source_file']
                sample = current_samples[job['source_index']]
                if sample['id'] != job.get('expected_id', sample['id']):
                    raise ValueError('Pinned source ID differs from the shard sample')
                raw, provenance = source_raw(sample, rows.get(sample['id']), reader.local, plan['limits']['views'])
            except (OSError, RuntimeError, ValueError, KeyError, IndexError, TypeError, EOFError) as exc:
                failure = dict(job, error=repr(exc), phase='source_RGB_read')
                bad.append(failure)
                store.write_json(f'prepare/skips/{job["split"]}_{job["index"]:06d}.json',
                    dict(identity=plan['identity'], **failure))
                print('[matched prepare skip] '+json.dumps(failure), flush=True)
                continue
            if encoder is None:
                with heartbeat('loading existing frozen StreamVGGT for paired extraction', rank):
                    encoder = make_encoder(a.encoder, device)
            rgb = resize_matched(raw, plan['limits']['size']).float().div(255).unsqueeze(0).to(device)
            with torch.no_grad(), amp(device):
                features = extract_matched_features(encoder, rgb)
                prefix = extract_matched_features(encoder, rgb[:, :plan['limits']['history']])
                features, prefix_check = prefix_safe_features(features, prefix, plan['limits']['history'])
            payload = matched_payload(sample, raw, provenance, features, cfg, job['source_audit'])
            payload['prefix_consistency'] = prefix_check
            check_payload(payload, job, cfg)
            store.save(job['file'], payload, plan['identity'])
            encoded += plan['limits']['views']+plan['limits']['history']
            processed_new += 1
            del rgb, features, prefix, raw
        else:
            check_payload(payload, job, cfg)
            store.publish([job['file'], job['file']+'.json'])
        records.append(dict(job, id=payload['id'], dataset=payload['dataset'], frame_ids=payload['frame_ids'],
            prefix_consistency=payload['prefix_consistency']))
        del payload
        elapsed = max(time.monotonic()-start, 1e-6)
        progress = dict(identity=plan['identity'], rank=rank, completed=len(records), skipped=len(bad),
            assigned=len(assigned), elapsed_seconds=elapsed, DI_throughput=encoded/elapsed,
            DI_unit='newly_encoded_input_frames_per_second_per_NPU_including_RGB_IO_and_publication',
            recovered_windows=len(records)-processed_new)
        store.write_json(f'prepare/rank{rank:03d}.json', progress)
        print('[matched prepare] '+json.dumps(progress), flush=True)
    del encoder, current_samples
    gc.collect()
    reports = gather(dict(records=records, bad=bad), world)
    if rank == 0:
        bank = bank_manifest(plan, reports)
        store.write_json('data/bank.json', bank)
        store.write_json('complete.json', dict(identity=plan['identity'], status='completed',
            train=len(bank['train']), dev=len(bank['dev']), train_eval=len(bank['train_eval']),
            skipped=len(bank['bad']), quality_policy='report_only'))
    barrier()


def audit_jobs(bank, seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Audit seeds must be nonempty and distinct')
    return [dict(split=split, entry=entry, seed=int(seed),
        key=f'{split}/case{entry["index"]:06d}/seed{seed}')
        for split in ('train_eval', 'dev') for entry in bank[split] for seed in seeds]


def audit_summary(plan, reports):
    rows = [row for report in reports for row in report]
    failures = [row for row in rows if row['status'] != 'completed']
    summary = dict(identity=plan['identity'], status='completed' if not failures else 'incomplete',
        expected=len(plan['jobs']), completed=len(rows)-len(failures), failed=failures,
        resolution=518, source_weights=plan['weights'],
        condition='observed4 only; independently sampled pure future Gaussian noise',
        quality_policy='report only; fixed training-fit versus source-disjoint reused development comparison')
    for split in ('train_eval', 'dev'):
        values = [row['metrics'] for row in rows if row['status'] == 'completed' and row['split'] == split]
        metrics = ('raw_l1', 'ae_l1', 'copy_l1', 'future_geo_mse', 'future_texture_mse',
                   'generated_motion_l1', 'gt_motion_l1', 'generated_laplacian_abs')
        summary[split] = dict(cases_seeds=len(values), mean={k: sum(v[k] for v in values)/len(values)
            for k in metrics} if values else {})
    return summary


def audit(a, cfg, store, source, device, rank, world):
    if not all((a.bank_local, a.generator_local, a.codec_local)):
        raise ValueError('Audit requires --bank-local, --generator-local and --codec-local')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    inputs = store.local/'_inputs'/f'rank{rank:03d}'
    bank_store = ArtifactStore(inputs/'bank', [a.bank_local, *a.bank_root])
    bank = bank_store.json('data/bank.json', True)
    data_plan = bank_store.json('contract.json', True)
    contract = source.json('contract.json', True)
    if bank['identity'] != data_plan['identity'] or contract != data_plan['source_contract']:
        raise ValueError('Audit sources differ from the paired feature bank')
    seeds = cfg.get('source_audit_seeds', cfg.get('eval_seeds', [101, 202]))
    jobs = audit_jobs(bank, seeds)
    fields = dict(schema='scene-matched-original-source-audit-v1', bank=bank['identity'],
        bank_manifest_sha=digest(bank), weights=contract['source_weights'], ae=contract['ae'],
        model=contract['model'], mean=contract['mean'], std=contract['std'], jobs=jobs,
        history=limits(cfg)['history'], steps=int(cfg.get('sample_steps', 32)),
        code_sha256=code_signature())
    plan = dict(fields, identity=digest(fields))
    pin(store, plan, rank)
    node = int(os.environ.get('GROUP_RANK', 0))
    shared = store.local/'_inputs'/f'node{node}'
    generator = ArtifactStore(shared/'generator', [a.generator_local, *a.generator_root])
    codec = ArtifactStore(shared/'codec', [a.codec_local, *a.codec_root])
    weight_path = stage_asset(generator, plan['weights'], local_rank, rank)
    ae_path = stage_asset(codec, plan['ae'], local_rank, rank)
    model = rae = None
    reports = []
    start = time.monotonic()
    completed_new = 0
    for job in jobs[rank::world]:
        name = job['key']+'/receipt.json'
        old = recover_case(store.local, name, plan['identity'], store.roots)
        if old:
            row = old['metadata']
            store.publish([*old['files'], name])
        else:
            try:
                if model is None:
                    from models.scene_flow import SceneFlow
                    from models.scene_rae import SceneRAE
                    warm = load_staged(weight_path)
                    if warm.get('model_args') != plan['model'] or warm.get('step') != 31069:
                        raise ValueError('Expected the pinned original online step31069 architecture')
                    model = SceneFlow(**plan['model'])
                    model.load_state_dict(warm['model'], strict=True)
                    model = model.eval().requires_grad_(False).to(device)
                    del warm
                    ae = load_staged(ae_path)
                    rae = SceneRAE(ae['spec']['source_config'], ae['spec']['channels'])
                    rae.load_state_dict(ae['model'], strict=True)
                    rae = rae.eval().requires_grad_(False).to(device)
                    del ae
                payload = bank_store.load(job['entry']['file'], bank['identity'], True)
                sample = payload['source_sample']
                if sample['id'] != job['entry']['id'] or sample['frame_ids'] != job['entry']['frame_ids']:
                    raise ValueError('Source audit sample identity differs')
                from utils.scene_prefix_eval import sample_case
                with heartbeat('original source free generation '+job['key'], rank):
                    metrics, preview, latent = sample_case(model, rae, sample,
                        plan['mean'], plan['std'], device, history=plan['history'],
                        steps=plan['steps'], seed=job['seed'])
                row = dict(status='completed', split=job['split'], key=job['key'], metrics=metrics)
            except Exception as exc:
                row = dict(status='failed', split=job['split'], key=job['key'], error=repr(exc))
                store.write_json(job['key']+'/failure.json', dict(identity=plan['identity'], **row))
                reports.append(row)
                print('[matched source audit failure] '+json.dumps(row), flush=True)
                continue
            # Output errors must not be swallowed as inference failures.
            tensor_name = job['key']+'/generated_future.pt'
            store.save(tensor_name, dict(latent=latent, metrics=metrics), plan['identity'])
            movie_name = job['key']+'/raw_ae_generated_copy.mp4'
            from train_scene_prefix import labelled_preview
            video(store.local/movie_name, labelled_preview(preview,
                ['RAW', 'AE', 'GENERATED', 'COPY'], plan['history']), fps=float(metrics['nominal_fps']))
            metrics_name = job['key']+'/metrics.json'
            store.write_json(metrics_name, metrics)
            files = [tensor_name, tensor_name+'.json', movie_name, metrics_name]
            seal(store.local, name, plan['identity'], files, row)
            store.publish([*files, name])
            completed_new += 1
            del payload, sample, preview, latent
        reports.append(row)
        progress = dict(rank=rank, identity=plan['identity'], completed=len(reports),
            elapsed_seconds=time.monotonic()-start,
            DI_throughput=completed_new*(bank['views']-plan['history'])*int(plan['model'].get('grid', 18))**2*plan['steps']/max(time.monotonic()-start, 1e-6),
            DI_unit='newly_sampled_future_tokens_times_denoising_steps_per_second_per_NPU_including_audit_IO')
        store.write_json(f'progress/rank{rank:03d}.json', progress)
        print('[matched source audit] '+json.dumps(progress), flush=True)
    reports = gather(reports, world)
    if rank == 0:
        summary = audit_summary(plan, reports)
        store.write_json('summary.json', summary)
        store.write_json('complete.json', dict(identity=plan['identity'], status=summary['status']))
    barrier()
    if any(row['status'] != 'completed' for report in reports for row in report):
        raise RuntimeError('Some original-source audit cases failed; see summary.json. New codec arms remain independent.')


def main():
    a = parser().parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding='utf8'))
    limits(cfg)
    # Never let a typo direct output into an input store.
    inputs = [a.source_local, a.bank_local, a.generator_local, a.codec_local,
              *a.source_root, *a.bank_root, *a.generator_root, *a.codec_root]
    if set(str(v).rstrip('/\\') for v in [a.output, *a.root]) & set(str(v).rstrip('/\\') for v in inputs if v):
        raise ValueError('Input and output stores must be disjoint')
    device, rank, world = distributed()
    store = ArtifactStore(a.output, a.root)
    node = int(os.environ.get('GROUP_RANK', 0))
    source = ArtifactStore(store.local/'_inputs'/f'node{node}'/'source', [a.source_local, *a.source_root])
    try:
        (prepare if a.stage == 'prepare' else audit)(a, cfg, store, source, device, rank, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
