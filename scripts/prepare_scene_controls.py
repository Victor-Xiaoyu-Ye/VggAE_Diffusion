#!/usr/bin/env python3
"""Resumable offline stage56 controls from exact raw stage53 RGB windows.

Plan and merge run once; work uses RANK/WORLD_SIZE/LOCAL_RANK independently,
without HCCL collectives. The launcher coordinates phase completion and keeps
the accelerator idle guard alive during CPU/IO phases. No public downloads.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import torch

from audit_scene_trackmask import heartbeat
from data.scene_rgb import SceneRGB
from utils.file_signature import sampled_file_signature
from utils.scene_control_data import (SCHEMA, build_controls, canonical_hash, replay_rgb,
                                     select_windows, validate_controls, window_key)
from utils.scene_quality_data import load_street_selection
from utils.scene_run import ArtifactStore, digest
from utils.scene_trackmask_teacher import VGGTTrackTeacher, teacher_source_identity


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('plan', 'work', 'merge'), required=True)
    p.add_argument('--source-local', required=True)
    p.add_argument('--source-root', action='append', default=[])
    p.add_argument('--local', required=True)
    p.add_argument('--root', action='append', default=[])
    p.add_argument('--cohort', required=True)
    p.add_argument('--selection', default=None)
    p.add_argument('--train-windows', type=int, default=4096)
    p.add_argument('--source-shard-limit', type=int, default=0,
                   help='Explicit smoke-only source scan bound; zero scans the full source manifest')
    p.add_argument('--eval-cases', type=int, default=64)
    p.add_argument('--history', type=int, default=4)
    p.add_argument('--grid', type=int, default=18)
    p.add_argument('--size', type=int, default=518)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', '--nodes', dest='workers', type=int,
                   default=int(os.environ.get('WORLD_SIZE', 1)))
    p.add_argument('--worker-rank', '--node-rank', dest='worker_rank', type=int,
                   default=int(os.environ.get('RANK', 0)))
    p.add_argument('--teacher-checkpoint', required=True)
    p.add_argument('--device', default='npu:'+os.environ.get('LOCAL_RANK', '0'))
    p.add_argument('--precision', choices=('float32', 'bfloat16'), default='bfloat16')
    p.add_argument('--query-stride', type=int, default=21)
    p.add_argument('--query-margin', type=int, default=7)
    p.add_argument('--query-chunk-size', type=int, default=0)
    p.add_argument('--raw-cache', default=None)
    return p


@contextmanager
def source_payload(store, name, identity):
    """Trim only staging files newly fetched by this process, never user files."""
    path = store.local/name
    existed = path.exists()
    payload = None
    try:
        payload = store.load(name, identity, required=True)
        yield payload
    finally:
        # torch.load mmap can hold an open mapping on Linux. Removing a new
        # private staging link is safe there; Windows skips open-file cleanup.
        del payload
        if not existed and path.exists():
            try:
                path.unlink()
                path.with_name(path.name+'.json').unlink(missing_ok=True)
            except PermissionError:
                pass


def trim_published(store, name):
    """Retain local intermediates unless both durable destinations accepted it."""
    if len(store.roots) < 2:
        return
    publications = getattr(store, '_control_publications', {})
    if name not in publications:
        # A resume/merge process has no artifact-specific acknowledgement.
        # Re-publish this exact payload before using its status for cleanup.
        store.publish([name, name+'.json'])
        publications[name] = list((store.json(store.status_name) or {}).get('success', []))
        store._control_publications = publications
    if set(publications[name]) != set(store.roots):
        return
    path = store.local/name
    if store.local.resolve() not in path.resolve().parents:
        raise ValueError('Refusing cleanup outside private control output')
    path.unlink(missing_ok=True)
    # Keep the tiny receipt so resumption can recover either durable replica.


def make_contract(a, source):
    previous = source.json('contract.json', True)
    manifest = source.json('data/cache_complete.json', True)
    model = previous.get('model', {})
    source_config = previous.get('config', {})
    if model.get('grid', a.grid) != a.grid:
        raise ValueError('Control token grid must match the frozen source model')
    if source_config.get('history', a.history) != a.history:
        raise ValueError('Observed history must match the fixed prefix source task')
    selection = load_street_selection(a.cohort, a.selection)
    code = teacher_source_identity('streamvggt')
    fields = dict(schema='scene-control-prepare-v1', control_schema=SCHEMA,
        source_prefix=previous['identity'], source_cache_identity=manifest['identity'],
        source_cache_manifest_sha=digest(manifest),
        source_eval_receipt=source.json('data/eval.pt.json', True),
        cohort_sha256=hashlib.sha256(Path(a.cohort).read_bytes()).hexdigest(),
        selection_sha256=selection['selection_sha256'],
        teacher=dict(backend='streamvggt', code_sha256=code['code_sha256'],
            checkpoint=sampled_file_signature(a.teacher_checkpoint), precision=a.precision,
            query_stride=a.query_stride, query_margin=a.query_margin,
            query_chunk_size=a.query_chunk_size),
        config=dict(train_windows=a.train_windows, eval_cases=a.eval_cases,
            history=a.history, grid=a.grid, size=a.size, seed=a.seed, workers=a.workers,
            source_shard_limit=a.source_shard_limit),
        task='explicit_future_camera_and_sparse_tracks; no_future_appearance',
        data='stage53_original_targets_and_normalization_unchanged',
        motion='sparse_teacher_point_evidence_not_instance_or_semantic_masks')
    if previous.get('cohort_sha256') != fields['cohort_sha256']:
        raise ValueError('Cohort differs from the frozen stage53 source')
    # Include this implementation so changed teacher/control semantics require
    # a new namespace, but no host paths or mtimes enter the identity.
    fields['control_source_sha256'] = canonical_hash({str(p.relative_to(PROJECT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__), PROJECT/'utils/scene_control_data.py', PROJECT/'utils/scene_trackmask.py')})
    return dict(fields, identity=digest(fields)), manifest, selection


def progress(store, a, started, **fields):
    elapsed = max(time.monotonic()-started, 1e-6)
    frames = int(fields.pop('neural_frames', 0))
    value = dict(stage='control_prepare', phase=a.stage, worker=a.worker_rank,
                 elapsed_seconds=elapsed, DI_throughput=frames/elapsed,
                 DI_unit='teacher_RGB_frames_per_second', **fields)
    print('[control prepare] '+json.dumps(value, ensure_ascii=False), flush=True)
    store.write_json(f'prepare/progress/{a.stage}_worker{a.worker_rank:03d}.json', value)


def plan(a, source, store, contract, manifest, selection):
    ready = store.json('prepare/plan.json')
    if ready:
        if ready['identity'] != contract['identity']:
            raise ValueError('Control plan identity changed; use a fresh namespace')
        return ready
    old = store.json('contract.json')
    if old and old['identity'] != contract['identity']:
        raise ValueError('Existing control output belongs to a different contract')
    store.write_json('contract.json', contract)
    entries, bad = [], []
    allowed = set(selection['allowed_ids'])
    started = time.monotonic()
    source_shards = manifest['shards'][:a.source_shard_limit] if a.source_shard_limit else manifest['shards']
    for ordinal, shard in enumerate(source_shards):
        try:
            receipt = source.json(shard['file']+'.json', True)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            error = dict(file=shard['file'], error=repr(exc), reason='source_receipt_unavailable')
            bad.append(error)
            store.write_json(f'prepare/index_errors/shard{ordinal:05d}.json', error)
            print('[control source receipt skipped] '+json.dumps(error), flush=True)
            continue
        name = f'prepare/index/shard{ordinal:05d}.json'
        index = store.json(name)
        stamp = dict(contract=contract['identity'], file=shard['file'], sha256=receipt['sha256'])
        if index and index.get('stamp') != stamp:
            raise ValueError('A source shard/index changed during control preparation')
        if index is None:
            rows, error = [], None
            try:
                with heartbeat('index_frozen_latents', ordinal):
                    with source_payload(source, shard['file'], manifest['identity']) as payload:
                        for i, sample in enumerate(payload['samples']):
                            if sample.get('dataset') != 'spatialvid' or sample.get('id') not in allowed:
                                continue
                            window_key(sample)
                            rows.append(dict(file=shard['file'], index=i, id=sample['id'],
                                dataset=sample['dataset'], frame_ids=sample['frame_ids'],
                                source_ordinal=ordinal))
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError) as exc:
                error = repr(exc)
            index = dict(stamp=stamp, entries=rows, error=error)
            store.write_json(name, index)
        entries.extend(index['entries'])
        if index.get('error'):
            bad.append(dict(file=shard['file'], error=index['error']))
        if ordinal % 16 == 0 or ordinal+1 == len(source_shards):
            progress(store, a, started, shards_done=ordinal+1, shards_total=len(source_shards),
                     indexed_windows=len(entries), bad_shards=len(bad))
    chosen = select_windows(entries, a.train_windows, a.seed)
    if not chosen:
        raise RuntimeError('No usable source windows for controlled training')
    grouped = {}
    for row in chosen:
        grouped.setdefault(row['file'], []).append(row)
    jobs = [dict(kind='train', file=name, entries=sorted(values, key=lambda x:x['index']))
            for name, values in sorted(grouped.items())]
    eval_payload = source.load('data/eval.pt', manifest['identity'], required=True)
    evaluation = eval_payload['samples'][:a.eval_cases]
    if not evaluation:
        raise RuntimeError('Source evaluation contains no cases')
    first = evaluation[0]
    if (tuple(first['raw'].shape[-2:]) != (a.size, a.size)
            or first['target'].shape[1] != a.grid*a.grid
            or not 1 <= a.history < len(first['frame_ids'])):
        raise ValueError('Control image/grid/history settings differ from frozen source evaluation')
    for index, sample in enumerate(evaluation):
        jobs.append(dict(kind='eval', index=index, key=window_key(sample), id=sample['id'],
                         frame_ids=sample['frame_ids']))
    for index, job in enumerate(jobs):
        job['job_index'], job['worker'] = index, index % a.workers
    ready = dict(identity=contract['identity'], jobs=jobs, selected_windows=len(chosen),
                 requested_windows=a.train_windows, evaluation_cases=len(evaluation),
                 bad_source_shards=bad, workers=a.workers,
                 selected_pool=dict(source_shard_limit=a.source_shard_limit,
                    scanned_source_shards=len(source_shards), total_source_shards=len(manifest['shards']),
                    eligible_windows=len(entries),
                    scope='bounded_smoke_not_quality_experiment' if a.source_shard_limit else 'full_source_manifest'))
    store.write_json('prepare/plan.json', ready)
    progress(store, a, started, status='planned', selected_windows=len(chosen),
             evaluation_cases=len(evaluation), jobs=len(jobs))
    return ready


def prepare_case(sample, a, store, identity, rows, reader, teacher_factory):
    key = window_key(sample)
    case = f'prepare/cases/{key}'
    done = load_output(store, case+'/controls.pt', identity)
    if done is not None:
        if done['window_key'] != key:
            raise ValueError('Saved controls do not match source window')
        validate_controls(done['controls'], history=a.history, grid=a.grid,
                          frames=len(sample['frame_ids']))
        return done['controls'], 0
    teacher_data = load_output(store, case+'/teacher.pt', identity)
    processed = 0
    if teacher_data is None:
        with heartbeat('raw_exact_frame_replay', key[:12]):
            frames, provenance = replay_rgb(sample, rows[sample['id']], reader.local, a.size)
        teacher = teacher_factory()  # setup errors are handled by outer fatal path
        with heartbeat('full_and_independent_prefix_teacher', key[:12]):
            full = teacher.infer(frames, reference_frame=0)
            prefix = teacher.infer(frames[:a.history], reference_frame=0)
        # Preserve the actual FP32 outputs used for controls. Remote sealed
        # teacher results resume expensive inference before later processing.
        teacher_data = dict(full=full, prefix=prefix, provenance=provenance, window_key=key)
        save_output(store, case+'/teacher.pt', teacher_data, identity)
        processed = len(frames)+a.history
    if teacher_data['window_key'] != key:
        raise ValueError('Teacher receipt belongs to another exact source window')
    controls = build_controls(teacher_data['full'], teacher_data['prefix'],
                              history=a.history, grid=a.grid)
    controls['metadata']['raw_provenance'] = teacher_data['provenance']
    controls['metadata']['window_key'] = key
    save_output(store, case+'/controls.pt', dict(window_key=key, controls=controls), identity)
    trim_published(store, case+'/teacher.pt')
    return controls, processed


class TeacherFailure(RuntimeError):
    pass


class OutputFailure(RuntimeError):
    """A failed output commit/recovery is never a bad-input skip."""


def load_output(store, name, identity):
    try:
        return store.load(name, identity)
    except Exception as exc:
        raise OutputFailure(f'Cannot recover committed control output {name}: {exc}') from exc


def save_output(store, name, value, identity):
    try:
        result = store.save(name, value, identity)
        publications = getattr(store, '_control_publications', {})
        publications[name] = list((store.json(store.status_name) or {}).get('success', []))
        store._control_publications = publications
        return result
    except Exception as exc:
        raise OutputFailure(f'Cannot commit control output {name}: {exc}') from exc


def work(a, source, store, contract, manifest):
    identity = contract['identity']
    planned = store.json('prepare/plan.json', True)
    if planned['identity'] != identity or planned['workers'] != a.workers:
        raise ValueError('Worker settings differ from the committed control plan')
    cohort = json.loads(Path(a.cohort).read_text(encoding='utf8'))
    rows = {r['id']:r for r in cohort['records']}
    reader = SceneRGB(cohort, split='train', cache_dir=a.raw_cache, size=a.size)
    teacher_instance = None
    def teacher_factory():
        nonlocal teacher_instance
        if teacher_instance is None:
            try:
                device = ('npu:'+os.environ.get('LOCAL_RANK', '0') if a.device == 'npu' else a.device)
                teacher_instance = VGGTTrackTeacher(a.teacher_checkpoint, device=device,
                    precision=a.precision, query_stride=a.query_stride, query_margin=a.query_margin,
                    query_chunk_size=a.query_chunk_size, backend='streamvggt')
            except Exception as exc:
                raise TeacherFailure('Full StreamVGGT teacher setup failed: '+repr(exc)) from exc
            store.write_json(f'prepare/teacher/worker{a.worker_rank:03d}.json', teacher_instance.metadata)
        return teacher_instance
    started, neural_frames = time.monotonic(), 0
    completed, errors = [], []
    evaluation = None
    report_name = f'prepare/nodes/worker{a.worker_rank:03d}.json'
    try:
        for job in planned['jobs']:
            if job['worker'] != a.worker_rank:
                continue
            output = f'data/prepared/job{job["job_index"]:05d}.pt'
            payload = load_output(store, output, identity)
            if payload is None:
                if job['kind'] == 'eval':
                    if evaluation is None:
                        evaluation = source.load('data/eval.pt', manifest['identity'], True)['samples']
                    values = [evaluation[job['index']]]
                    payload = prepare_job(values, job, a, store, identity, rows, reader, teacher_factory)
                else:
                    # Read source tensors separately from inference/publication:
                    # only source failures qualify for a shared bad-input skip.
                    try:
                        with source_payload(source, job['file'], manifest['identity']) as loaded:
                            values = [loaded['samples'][entry['index']] for entry in job['entries']]
                            for sample, entry in zip(values, job['entries']):
                                if window_key(sample) != entry['key']:
                                    raise ValueError('Source target window changed after indexing')
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError) as exc:
                        payload = dict(samples=[], neural_frames=0, errors=[dict(
                            id=entry['id'], frame_ids=entry['frame_ids'], file=job['file'],
                            error=repr(exc), kind='train', job_index=job['job_index'])
                            for entry in job['entries']])
                        print('[control source shard skipped] '+json.dumps(payload['errors']), flush=True)
                    else:
                        payload = prepare_job(values, job, a, store, identity, rows, reader, teacher_factory)
                save_output(store, output, payload, identity)
                neural_frames += payload['neural_frames']
            errors.extend(payload['errors'])
            completed.append(dict(file=output, job_index=job['job_index'], kind=job['kind'],
                eval_index=job.get('index'), count=len(payload['samples']),
                sources=[s['dataset'] for s in payload['samples']]))
            trim_published(store, output)
            progress(store, a, started, neural_frames=neural_frames, status='running',
                     jobs_done=len(completed), prepared_windows=sum(x['count'] for x in completed),
                     skipped_windows=len(errors))
        report = dict(identity=identity, status='completed', worker=a.worker_rank,
                      jobs=completed, errors=errors, neural_frames=neural_frames)
        store.write_json(report_name, report)
        return report
    except BaseException as exc:
        store.write_json(report_name, dict(identity=identity, status='failed', worker=a.worker_rank,
            jobs=completed, errors=errors, fatal_error=repr(exc)))
        raise


def prepare_job(values, job, a, store, identity, rows, reader, teacher_factory):
    samples, errors, frames = [], [], 0
    for sample in values:
        try:
            controls, processed = prepare_case(sample, a, store, identity, rows, reader, teacher_factory)
        except (TeacherFailure, OutputFailure):
            raise
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError) as exc:
            message = str(exc).lower()
            # Systematic device failures should not turn the entire dataset
            # into a superficially successful empty/skipped preparation.
            if any(s in message for s in ('out of memory', 'not implemented', 'unsupported',
                                          'does not support', 'no durable output')):
                raise TeacherFailure('Teacher/device/publication failure: '+repr(exc)) from exc
            error = dict(id=sample.get('id'), frame_ids=sample.get('frame_ids'), error=repr(exc),
                         kind=job['kind'], job_index=job['job_index'])
            errors.append(error)
            print('[control window skipped] '+json.dumps(error), flush=True)
            store.write_json(f'prepare/errors/{window_key(sample)}.json', error)
            continue
        # Preserve original target/anchor tensor bytes and eval raw; no encoder,
        # no new normalization, no future appearance in the control dictionary.
        samples.append(dict(sample, controls=controls))
        frames += processed
    return dict(samples=samples, errors=errors, neural_frames=frames)


def merge(a, store, contract, manifest):
    identity = contract['identity']
    plan = store.json('prepare/plan.json', True)
    reports = [store.json(f'prepare/nodes/worker{i:03d}.json') for i in range(a.workers)]
    if any(r is None for r in reports):
        raise SystemExit(75)
    if any(r['identity'] != identity or r['status'] != 'completed' for r in reports):
        raise RuntimeError('Control preparation has a failed or mismatched worker; see node reports')
    jobs = sorted([j for r in reports for j in r['jobs']], key=lambda j:j['job_index'])
    if [j['job_index'] for j in jobs] != list(range(len(plan['jobs']))):
        raise ValueError('Prepared job coverage has duplicates or omissions')
    shards, evaluation, errors = [], [], [e for r in reports for e in r['errors']]
    for job in jobs:
        # Verify payload/receipt before publishing final completeness. This
        # remains bounded to one shard at a time and repairs from either root.
        payload = store.load(job['file'], identity, True)
        if len(payload['samples']) != job['count']:
            raise ValueError('Prepared report and payload sample counts differ')
        if job['kind'] == 'train' and job['count']:
            shards.append(dict(file=job['file'], sources=job['sources'],
                ids=[s['id'] for s in payload['samples']], count=job['count']))
        elif job['kind'] == 'eval':
            evaluation.extend((job['eval_index'], dict(s,source_eval_index=job['eval_index']))
                              for s in payload['samples'])
        trim_published(store, job['file'])
    if not shards or not evaluation:
        raise RuntimeError('No usable controlled train/eval data; inspect preparation errors')
    evaluation.sort(key=lambda row:row[0])
    store.save('data/eval.pt', dict(samples=[s for _, s in evaluation], errors=errors,
        original_eval_indices=[i for i, _ in evaluation], identity=identity), identity)
    ready = dict(identity=identity, source_prefix=contract['source_prefix'],
        source_cache_manifest_sha=contract['source_cache_manifest_sha'],
        control_schema=SCHEMA, shards=shards, samples=sum(s['count'] for s in shards),
        mean=manifest['mean'], std=manifest['std'], normalization='original_source_statistics_unchanged',
        eval_cases=len(evaluation), bad=errors, bad_source_shards=plan['bad_source_shards'],
        requested_windows=a.train_windows, selected_windows=plan['selected_windows'],
        selected_pool=plan['selected_pool'],
        future_control='camera_and_sparse_track_pseudo_labels_no_dense_future_depth_or_RGB')
    store.write_json('data/cache_complete.json', ready)
    print('[control prepare complete] '+json.dumps(dict(train=ready['samples'],
        evaluation=ready['eval_cases'], skipped=len(errors), identity=identity)), flush=True)
    return ready


def main(argv=None):
    a = parser().parse_args(argv)
    if (not 0 <= a.worker_rank < a.workers or a.train_windows < 0 or a.source_shard_limit < 0
            or min(a.eval_cases, a.history, a.grid) < 1):
        raise ValueError('Invalid control preparation counts/rank')
    if Path(a.local).resolve() == Path(a.source_local).resolve() or set(a.root) & set(a.source_root):
        raise ValueError('Controlled output must not overwrite the source prefix run')
    torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', 2))))
    # Source namespace is a read-only replica, never the staging workspace.
    # Per-worker private paths also avoid sibling workers deleting each other's
    # temporary shards while mmap readers are being initialized.
    source = ArtifactStore(Path(a.local)/'_inputs'/f'worker{a.worker_rank:03d}',
                           [a.source_local, *a.source_root])
    store = ArtifactStore(a.local, a.root)
    contract, manifest, selection = make_contract(a, source)
    if a.stage == 'plan':
        plan(a, source, store, contract, manifest, selection)
    elif a.stage == 'work':
        work(a, source, store, contract, manifest)
    else:
        merge(a, store, contract, manifest)


if __name__ == '__main__':
    main()
