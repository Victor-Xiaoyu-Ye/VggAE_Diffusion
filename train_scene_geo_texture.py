"""Stage58 matched prefix-only versus future-geo-conditioned latent diffusion.

The oracle is an upper-bound study, not deployable geometry or pure geometry
truth. Both arms generate every future C256 channel; neither inserts GT into
the decoder. CPU preparation creates one immutable shared data subset first.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from audit_scene_end_to_end import code_signature
from audit_scene_flow_weights import ensure_disjoint, receipt_spec, stage_asset
from train_scene_motion import build_plan as source_plan
from train_scene_pipeline import barrier, distributed
from train_scene_prefix import train
from utils.scene_geo_data import activity_score, choose_windows, compact_sample, order_key, validate_window
from utils.scene_quality_data import load_street_selection
from utils.scene_run import ArtifactStore, StageBudget, digest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--cohort', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--encoder')  # launcher compatibility; training never loads it
    p.add_argument('--source-local', required=True)
    p.add_argument('--source-root', action='append', default=[])
    p.add_argument('--output', required=True)
    p.add_argument('--root', action='append', default=[])
    p.add_argument('--data-local')
    p.add_argument('--data-root', action='append', default=[])
    p.add_argument('--stage', choices=('prepare', 'preflight', 'train'), default='train')
    p.add_argument('--startup-seconds', type=float, default=0.)
    return p


def validate_config(cfg):
    if cfg['history'] != 4 or cfg['views'] != 9 or cfg.get('convergence'):
        raise ValueError('Stage58 requires matched fixed 4->5 windows and equal-update training')
    data = cfg['dataset']
    if not 0 < data['train_windows'] <= data['candidate_windows']:
        raise ValueError('Candidate pool must cover the requested training subset')
    if cfg['max_steps'] < 2 or cfg['accum'] < 1:
        raise ValueError('Need at least the two-update save/restart test')
    if not 0 <= data.get('high_change_fraction', .5) <= 1:
        raise ValueError('Invalid high-change proportion')


def stores(a):
    local = Path(a.output)
    node = int(os.environ.get('GROUP_RANK', os.environ.get('NODE_RANK', 0)))
    output = ArtifactStore(local, a.root)
    source = ArtifactStore(local/'_inputs'/f'node{node}'/'source', [a.source_local, *a.source_root])
    cfg = json.loads(Path(a.config).read_text(encoding='utf8'))
    ns = cfg['codec_namespace']
    if not ns.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Invalid codec namespace')
    codec = ArtifactStore(local/'_inputs'/f'node{node}'/'codec',
        [str(Path(a.source_local).parent/ns/'production')]+
        [r.rstrip('/').rsplit('/', 1)[0]+'/'+ns+'/production' for r in a.source_root])
    return cfg, output, source, codec


def data_plan(source, codec, cfg, selection):
    base, manifest = source_plan(source, codec, cfg, selection, 24)
    base.pop('identity')
    base.update(schema='scene-geo-texture-data-v1', code_sha256=code_signature(),
        selection_note='half high temporal texture change, half uniform remainder of bounded candidate pool; not semantic motion labels',
        condition='original latent cache; no reencoding, controls or changed normalization')
    return dict(base, identity=digest(base)), manifest


def discard_private_input(source, name):
    """Only the private staging payload; original local/remote sources untouched."""
    target = (source.local/name).resolve()
    if source.local.resolve() not in target.parents:
        raise ValueError('Staging cleanup escaped its private root')
    target.unlink(missing_ok=True)
    target.with_name(target.name+'.json').unlink(missing_ok=True)


def prepare(source, codec, store, cfg, selection):
    plan, original = data_plan(source, codec, cfg, selection)
    identity = plan['identity']
    prior = store.json('contract.json')
    if prior and prior['identity'] != identity:
        raise ValueError('Prepared subset source/code/config changed; use a fresh namespace')
    store.write_json('contract.json', plan)
    complete = store.json('data/cache_complete.json')
    if complete and complete['identity'] != identity:
        raise ValueError('Prepared data identity changed')
    if complete:
        # Verify actual durable payloads, not only a completion marker.
        for shard in complete['shards']:
            store.load(shard['file'], identity, True)
            store.publish([shard['file'], shard['file']+'.json'])
        store.load('data/eval.pt', identity, True)
        store.publish(['data/eval.pt', 'data/eval.pt.json'])
        store.write_json('data/cache_complete.json', complete)
        return complete
    start = time.monotonic()
    spec = cfg['dataset']
    allowed = frozenset(selection['allowed_ids'])
    evaluation = source.load(plan['evaluation']['file'], plan['evaluation']['identity'], True)['samples']
    eval_ids = {s['id'] for s in evaluation}
    # The split is checked at sample level even when old shard manifests omit IDs.
    allowed = allowed - eval_ids
    eval_count = int(spec.get('eval_cases', cfg.get('dev_cases', len(evaluation))))
    evaluation = [dict(s, source_eval_index=i) for i, s in enumerate(evaluation[:eval_count])]
    for sample in evaluation:
        validate_window(sample, eval_ids, len(plan['mean']), cfg['views'], evaluation=True,
                        grid=plan['model'].get('grid'))
    store.save('data/eval.pt', dict(samples=evaluation), identity)
    shard_order = sorted(original['shards'], key=lambda s: order_key(cfg['seed'], s['file']))
    if spec.get('source_shard_limit', 0):
        shard_order = shard_order[:spec['source_shard_limit']]
    candidates, scans = [], []
    for index, shard in enumerate(shard_order):
        if len(candidates) >= spec['candidate_windows']:
            break
        scan_name = f'prepare/scans/{index:05d}.json'
        scan = store.json(scan_name)
        if scan and (scan.get('identity') != identity or scan.get('file') != shard['file']):
            raise ValueError('Candidate scan identity changed')
        if not scan:
            scan = dict(identity=identity, file=shard['file'], records=[], bad=[])
            try:
                payload = source.load(shard['file'], original['identity'], True)
                for si, sample in enumerate(payload['samples']):
                    try:
                        validate_window(sample, allowed, len(plan['mean']), cfg['views'], grid=plan['model'].get('grid'))
                        ids = torch.as_tensor(sample['frame_ids']).tolist()
                        scan['records'].append(dict(file=shard['file'], index=si, id=sample['id'],
                            frame_ids=ids, key=digest(dict(id=sample['id'], frames=ids)),
                            activity=activity_score(sample, plan['std'])))
                    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                        scan['bad'].append(dict(index=si, error=repr(exc)))
                del payload
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError) as exc:
                scan['bad'].append(dict(error=repr(exc)))
            finally:
                discard_private_input(source, shard['file'])
            # Output publication failures must not be reported as bad input.
            store.write_json(scan_name, scan)
        scans.append(scan_name)
        remaining = spec['candidate_windows']-len(candidates)
        ordered = sorted(scan['records'], key=lambda r: order_key(cfg['seed'], r['key']))
        candidates.extend(ordered[:remaining])
        elapsed = max(time.monotonic()-start, 1e-6)
        row = dict(phase='candidate_scan', shards=index+1, candidates=len(candidates),
            DI_throughput=len(candidates)*cfg['views']/elapsed,
            DI_unit='accepted_cached_latent_frames_per_second_including_scan_and_publication; no encoder forward',
            elapsed_seconds=elapsed)
        store.write_json('prepare/progress.json', row)
        print('[geo data prepare] '+json.dumps(row), flush=True)
    chosen = choose_windows(candidates, spec['train_windows'], cfg['seed'], spec.get('high_change_fraction', .5))
    if not chosen:
        raise RuntimeError('No usable training windows in candidate source pool')
    store.write_json('prepare/selection.json', dict(identity=identity, requested=spec['train_windows'],
        candidate_count=len(candidates), selected=chosen, scans=scans,
        note=plan['selection_note'], original_cache_manifest_sha=digest(original)))
    groups = defaultdict(list)
    for row in chosen:
        groups[row['file']].append(row)
    shards, bad, accepted = [], [], []
    chunk_size = int(spec.get('shard_samples', 64))
    if chunk_size < 1:
        raise ValueError('Prepared shard size must be positive')
    groups = [(name, rows[i:i+chunk_size]) for name, rows in sorted(groups.items())
              for i in range(0, len(rows), chunk_size)]
    for group_index, (name, rows) in enumerate(groups):
        output_name = f'data/shards/subset{group_index:05d}.pt'
        receipt = store.json(output_name+'.json')
        if receipt:
            payload = store.load(output_name, identity, True)
            samples = payload['samples']
            bad.extend(payload.get('bad', []))
            accepted.extend(payload['selection'])
            store.publish([output_name, output_name+'.json'])
        else:
            samples, selection_rows, local_bad = [], [], []
            try:
                payload = source.load(name, original['identity'], True)
                for row in rows:
                    try:
                        sample = payload['samples'][row['index']]
                        validate_window(sample, allowed, len(plan['mean']), cfg['views'], grid=plan['model'].get('grid'))
                        if sample['id'] != row['id'] or torch.as_tensor(sample['frame_ids']).tolist() != row['frame_ids']:
                            raise ValueError('Scanned sample identity changed')
                        samples.append(compact_sample(sample)); selection_rows.append(row)
                    except (KeyError, TypeError, ValueError, RuntimeError, IndexError) as exc:
                        local_bad.append(dict(file=name, index=row['index'], error=repr(exc)))
                del payload
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError) as exc:
                local_bad.append(dict(file=name, error=repr(exc)))
            finally:
                discard_private_input(source, name)
            store.save(output_name, dict(samples=samples, selection=selection_rows, bad=local_bad), identity)
            bad.extend(local_bad); accepted.extend(selection_rows)
        if samples:
            shards.append(dict(file=output_name, count=len(samples), sources=['spatialvid']*len(samples),
                ids=[s['id'] for s in samples]))
        row = dict(phase='compact_commit', committed_windows=len(accepted),
            DI_throughput=len(accepted)*cfg['views']/max(time.monotonic()-start, 1e-6),
            DI_unit='committed_cached_latent_frames_per_second_including_preparation_IO')
        store.write_json('prepare/progress.json', row)
        print('[geo data prepare] '+json.dumps(row), flush=True)
    if not accepted:
        raise RuntimeError('All selected source payloads failed during compaction')
    manifest = dict(identity=identity, source_prefix=plan['source_prefix'],
        source_cache_manifest_sha=digest(original), shards=shards, samples=len(accepted),
        requested_samples=spec['train_windows'], candidates=len(candidates), eval_cases=len(evaluation),
        mean=plan['mean'], std=plan['std'], selection_sha=digest(accepted), selection=accepted, bad=bad,
        quality_policy='No visual threshold; fewer valid samples are reported and training continues',
        source_weight_sha=plan['source_weights']['sha256'], ae_sha=plan['ae']['sha256'])
    store.write_json('data/cache_complete.json', manifest)
    return manifest


def training_plan(source, codec, data, cfg, selection, world):
    base, original = source_plan(source, codec, cfg, selection, world)
    ready = data.json('data/cache_complete.json', True)
    contract = data.json('contract.json', True)
    common = dict(cfg); common.pop('arm', None)
    expected, _ = data_plan(source, codec, common, selection)
    if contract['identity'] != expected['identity'] or ready['identity'] != contract['identity']:
        raise ValueError('Data subset source/config/code identity differs')
    if ready['source_cache_manifest_sha'] != digest(original) or ready['mean'] != base['mean'] or ready['std'] != base['std']:
        raise ValueError('Prepared subset changed source normalization or cache')
    evaluation = receipt_spec(data, 'data/eval.pt')
    if evaluation['identity'] != ready['identity'] or not ready['samples']:
        raise ValueError('Missing valid matched data')
    base.pop('identity')
    base.update(schema='scene-geo-texture-pilot-v1', code_sha256=code_signature(),
        data_identity=ready['identity'], subset_manifest_sha=digest(ready),
        source_evaluation=base['evaluation'], evaluation=evaluation,
        matched_windows=ready['samples'], comparison='Matched architecture, data, RNG, updates; only future geo input differs',
        condition=('observed prefix only' if cfg['arm']=='baseline' else
            'ORACLE future RGB-encoded geo features; may contain appearance; not deployable'),
        feature_target='Original full C256 jointly predicted; no GT geometry clamping at decoding',
        stopping='Fixed pilot update budget; no automatic visual success claim')
    return dict(base, identity=digest(base)), ready


def main():
    a = parser().parse_args(); ensure_disjoint(a)
    cfg, store, source, codec = stores(a); validate_config(cfg)
    selection = load_street_selection(a.cohort, a.selection)
    if a.stage == 'prepare':
        prepare(source, codec, store, cfg, selection)
        return
    if not a.data_local:
        raise ValueError('Prepare one shared subset before starting matched arms')
    if Path(a.output).resolve() == Path(a.data_local).resolve() or set(a.root) & set(a.data_root):
        raise ValueError('Training output must not overwrite prepared data')
    arm = os.environ.get('SCENE_GEO_ARM', 'baseline')
    from utils.scene_geo_condition import GEO_ARMS, GeoConditionRuntime
    if arm not in GEO_ARMS:
        raise ValueError('Unknown SCENE_GEO_ARM')
    cfg = dict(cfg, arm=arm)
    device, rank, world = distributed()
    if world != 24:
        raise ValueError('Production pilot uses three nodes x eight ranks')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    data = ArtifactStore(store.local/'_inputs'/f'node{rank//8}'/'data', [a.data_local, *a.data_root])
    plan, manifest = training_plan(source, codec, data, cfg, selection, world)
    runtime = GeoConditionRuntime(arm, plan['mean'], plan['std'], history=cfg['history'],
        control_lr_multiplier=cfg['control_lr_multiplier'])
    runtime.seed_training(cfg['seed'], rank)
    if rank == 0:
        prior = store.json('contract.json')
        if prior and prior['identity'] != plan['identity']:
            raise ValueError('Changed experiment: use a fresh namespace')
        store.write_json('contract.json', plan)
    barrier()
    done = store.json('complete.json')
    baseline = store.json(f"eval/{cfg['source_label']}/summary.json")
    if not (done and done.get('evaluation_status') == 'completed' and baseline and baseline.get('status') == 'completed'):
        progress = store.json('progress.json') or {}
        budget = StageBudget(cfg['hours']*3600 if cfg['hours'] else math.inf,
            progress.get('total_spent', 0.)+max(0., a.startup_seconds))
        budget.install_signals()
        ae = stage_asset(codec, plan['ae'], local_rank, rank)
        weights = stage_asset(source, plan['source_weights'], local_rank, rank)
        evaluation = stage_asset(data, plan['evaluation'], local_rank, rank)
        train(a, cfg, plan, source, store, manifest, selection, ae, weights, evaluation,
              budget, device, rank, world, data_store=data, runtime=runtime)
    barrier()
    import train_scene_pipeline as pipeline
    if pipeline.WRITER is not None:
        pipeline.WRITER.close(); pipeline.WRITER = None
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
