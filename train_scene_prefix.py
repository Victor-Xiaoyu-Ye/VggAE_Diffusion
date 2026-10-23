"""48h Scene quality warm-start: frozen codec, street-only, observed prefix.

Source artifacts are read-only. Keep the original eight network slots, fix the
first three to observed latents, and generate only the five unknown slots.
This is a new task/optimizer/EMA, not a resume of the historical training run.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from audit_scene_flow_weights import heartbeat, receipt_spec, stage_asset, load_staged
from data.scene_rgb import SceneRGB, seed_for
from models.scene_flow import SceneFlow
from models.scene_rae import SceneRAE
from train_scene_pipeline import (amp, any_rank, barrier, distributed, gather,
    make_encoder, features, memory_gib, metrics, preview, publish_metrics, wrap)
from utils.scene_run import ArtifactStore, StageBudget, digest
from utils.native_audit_io import recover_case, seal
from utils.scene_prefix import prepare_prefix, PrefixFlowNet, conservative_lr
from utils.scene_prefix_eval import sample_case
from utils.scene_quality_data import load_street_selection, select_street_samples, FilteredLatentSamples
from utils.training import EMA, capture_rng_state, restore_rng_state, append_metrics
from utils.window_flow import WindowFlow


def record_progress(store, budget, **fields):
    previous = store.json('progress.json') or {}
    store.write_json('progress.json', dict(previous, total_spent=budget.elapsed(), **fields))


def labelled_preview(frames, columns, history):
    """Burn column/phase labels into new diagnostic previews, not metric tensors."""
    import cv2
    data=(frames.detach().cpu().float().clamp(0,1).numpy()*255).round().astype(np.uint8)
    width=data.shape[2]//len(columns)
    labelled=np.zeros((len(data),data.shape[1]+32,data.shape[2],3),dtype=np.uint8)
    labelled[:,32:]=data
    for i,frame in enumerate(labelled):
        phase='OBSERVED' if i<history else 'FUTURE'
        for j,column in enumerate(columns):
            cv2.putText(frame,f'{column} | f{i} {phase}',(j*width+5,21),
                cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    return torch.from_numpy(labelled).float()/255.


def trim_local_snapshots(store, keep=2):
    """Bound private disk use only after both durable destinations accepted save."""
    status = store.json(store.status_name) or {}
    if len(store.roots) < 2 or set(status.get('success', [])) != set(store.roots):
        return
    directory = store.local/'train'
    for pattern in ('weights_step*.pt', 'full_step*.pt'):
        for path in sorted(directory.glob(pattern))[:-keep]:
            if directory.resolve() not in path.resolve().parents:
                raise ValueError('Refusing snapshot cleanup outside training directory')
            path.unlink()
            path.with_name(path.name+'.json').unlink(missing_ok=True)


def source_plan(source, cfg, cohort_path, selection, world):
    contract = source.json('contract.json', True)
    manifest = source.json('cache/complete.json', True)
    ae_done = source.json('ae/complete.json', True)
    ae = receipt_spec(source, ae_done['selected'])
    evaluation = receipt_spec(source, 'cache/eval.pt')
    weights = receipt_spec(source, cfg['source_weights'])
    model = source.json('video/model.json', True)['model']
    cohort_sha = hashlib.sha256(Path(cohort_path).read_bytes()).hexdigest()
    if contract.get('cohort_sha256') != cohort_sha:
        raise ValueError('Cohort does not match the original frozen cache')
    if manifest['pipeline'] != contract['identity'] or manifest['ae'] != ae['sha256']:
        raise ValueError('Source cache/AE contract mismatch')
    if evaluation['identity'] != manifest['identity'] or ae['identity'] != ae_done['identity']:
        raise ValueError('Source AE/evaluation receipt identity mismatch')
    expected = digest(dict(pipeline=contract['identity'], cache=manifest['identity'], stage='video', model=model))
    if weights['identity'] != expected:
        raise ValueError('Warm-start weights are not from the pinned source cache/model')
    plan = dict(schema='scene-prefix-quality-v1', config=cfg, world=world,
        source_pipeline=contract['identity'], source_cache=manifest['identity'],
        source_weights=weights, ae=ae, evaluation=evaluation,
        model=model, cohort_sha256=cohort_sha,
        selection_sha256=selection['selection_sha256'],
        street_ids_sha=digest(sorted(selection['allowed_ids'])),
        mean=manifest['mean'], std=manifest['std'],
        warm_start='online model only; fresh Adam and FP32 EMA',
        condition='z0 anchor; clean z1:history; future Gaussian noise; no text/camera',
        forecast='original 9-frame window; approximately 0.625s future, not a long video',
        source_data_exposure='historical pretrained codec/flow already saw these training sources')
    return dict(plan, identity=digest(plan)), manifest


def keep_training_fields(sample):
    # Future labels remain supervised targets, never conditions. Remove optional
    # modalities entirely so they cannot accidentally enter this experiment.
    return {k: sample[k] for k in ('anchor', 'target', 'id', 'dataset', 'frame_ids')}


def compact_cache(source, store, manifest, selection, identity, budget, device, rank, world):
    ready = store.json('data/cache_complete.json')
    if ready:
        if ready['identity'] != identity:
            raise ValueError('Filtered cache identity changed')
        return ready
    sources = manifest['shards']
    state_name = f'prepare/rank{rank:03d}.json'
    state = store.json(state_name) or dict(cursor=0, shards=[], bad=[])
    turns = math.ceil(len(sources)/world)
    while any_rank(state['cursor'] < turns, device):
        if any_rank(budget.due(), device):
            store.write_json(state_name, state)
            raise SystemExit(75)
        index = rank + world*state['cursor']
        if index < len(sources):
            entry = sources[index]
            name = f'data/shards/source{index:05d}.pt'
            receipt = store.json(name+'.json')
            payload = chosen = None
            try:
                if receipt:
                    payload = store.load(name, identity, True)
                    chosen = payload['samples']
                else:
                    with heartbeat('reading frozen shard '+entry['file'], rank):
                        payload = source.load(entry['file'], manifest['identity'], True)
                    chosen = select_street_samples(payload['samples'], selection)
                    chosen = [keep_training_fields(s) for s in chosen]
                for sample in chosen:
                    target, anchor = sample['target'], sample['anchor']
                    if (not torch.is_tensor(target) or not torch.is_tensor(anchor) or
                            target.ndim != 3 or target.shape[0] != 9 or
                            target.shape[2] != len(manifest['mean']) or
                            anchor.shape != target[:1].shape or
                            not torch.equal(anchor, target[:1]) or
                            not torch.isfinite(target).all()):
                        raise ValueError('Invalid cached window tensor/anchor contract')
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError) as exc:
                chosen = None
                state['bad'].append(dict(file=entry['file'], error=repr(exc)))
                print('[street cache skipped] '+json.dumps(state['bad'][-1]), flush=True)
            # Publication errors are not bad-input errors: never acknowledge a
            # cursor advance when no durable replica accepted this output.
            try:
                if chosen:
                    if receipt:
                        store.publish([name, name+'.json'])
                    else:
                        store.save(name, dict(samples=chosen), identity)
                    state['shards'].append(dict(file=name, sources=['spatialvid']*len(chosen)))
            finally:
                del payload, chosen
                # This is the new run's private source staging, never the old
                # production directory or either remote replica.
                local = source.local/entry['file']
                if local.is_file() and source.local.resolve() in local.resolve().parents:
                    local.unlink()
                    local.with_name(local.name+'.json').unlink(missing_ok=True)
        state['cursor'] += 1
        store.write_json(state_name, state)
        if rank == 0:
            record_progress(store, budget, phase='filtering_cache')
        if state['cursor'] % 8 == 0 or state['cursor'] == turns:
            metrics(store, dict(stage='prepare', cursor=state['cursor'], total=turns,
                valid_windows=sum(len(s['sources']) for s in state['shards']),
                bad_shards=len(state['bad']), DI_throughput=0.,
                scope='CPU/IO cache filtering; no neural tokens processed'), rank)
    barrier()
    if rank == 0:
        states = [store.json(f'prepare/rank{r:03d}.json', True) for r in range(world)]
        shards = sorted([s for item in states for s in item['shards']], key=lambda x:x['file'])
        if not shards:
            raise RuntimeError('No valid street windows; cannot train an empty dataset')
        ready = dict(identity=identity, shards=shards, mean=manifest['mean'], std=manifest['std'],
            samples=sum(len(s['sources']) for s in shards), expected_samples=len(selection['allowed_ids'])*4,
            normalization='source statistics unchanged', source_identity=manifest['identity'],
            bad=[s for item in states for s in item['bad']])
        store.write_json('data/cache_complete.json', ready)
    barrier()
    return store.json('data/cache_complete.json', True)


@torch.no_grad()
def encode_eval_sample(encoder, rae, item, device):
    frames = item['frames'].unsqueeze(0).to(device)
    with amp(device):
        z = rae.encode(features(encoder, frames, device), frames).flatten(2,3)[0]
        anchor = rae.encode(features(encoder, frames[:,:1], device), frames[:,:1]).flatten(2,3)[0]
    return dict(anchor=anchor.cpu().bfloat16(), target=torch.cat((anchor,z[1:]),0).cpu().bfloat16(),
        raw=(item['frames']*255).round().to(torch.uint8), id=item['id'], dataset=item['dataset'],
        frame_ids=item['frame_ids'])


@torch.no_grad()
def verify_prefix(encoder, rae, sample, mean, std, device, history):
    frames = sample['raw'].float().unsqueeze(0).to(device)/255
    altered = frames.clone()
    # Fixed deterministic replacement, independent of training/eval RNG.
    altered[:,history:] = 1-frames[:,history:]
    def encode(rgb):
        with amp(device):
            return rae.encode(features(encoder,rgb,device),rgb).flatten(2,3).float()
    full = encode(frames)
    short = encode(frames[:,:history])
    counterfactual = encode(altered)
    future_rms = ((full[:,:history]-counterfactual[:,:history])/std).square().mean().sqrt().item()
    prefix_rms = ((full[:,:history]-short)/std).square().mean().sqrt().item()
    cache_rms = ((full[:,1:]-sample['target'][1:].to(device).float())/std).square().mean().sqrt().item()
    row = dict(future_perturbation_normalized_rms=future_rms, prefix_length_normalized_rms=prefix_rms,
        cached_target_normalized_rms=cache_rms, tolerances=dict(future=.002,prefix=.03,cache=.03),
        note='BF16 numerical contract, not a reconstruction/quality gate')
    row['cache_quantization_warning'] = cache_rms > .03
    row['cache_comparison_note'] = 'Cache used original float RGB; replay uses rounded uint8 RGB. Report only.'
    if not all(math.isfinite(x) for x in (future_rms,prefix_rms,cache_rms)) or future_rms>.002 or prefix_rms>.03:
        raise ValueError('Observed-prefix/cache encoder contract failed: '+json.dumps(row))
    return row


def prepare_evaluation(source, store, plan, cfg, cohort, selection, encoder_path,
                       rae_path, eval_path, device, rank, world):
    if store.json('data/eval.pt.json'):
        return
    old = load_staged(eval_path)['samples']
    val_ids = set(selection['val_ids'])
    old = [keep_training_fields(s) | dict(raw=s['raw']) for s in old
           if s['dataset']=='spatialvid' and s['id'] in val_ids]
    rows = {r['id']:r for r in cohort['records'] if r['split']=='val' and r['id'] in val_ids}
    if any(s['id'] not in rows or s['id'] in selection['allowed_ids'] for s in old):
        raise ValueError('Validation overlaps training or is not in original val split')
    ordered = sorted(rows, key=lambda x:seed_for('prefix-dev',x))
    selected = list(dict.fromkeys([s['id'] for s in old]+ordered))[:cfg['dev_cases']]
    existing = {s['id']:s for s in old}
    results, errors = [], []
    payload = load_staged(rae_path)
    rae = SceneRAE(payload['spec']['source_config'],payload['spec']['channels'])
    rae.load_state_dict(payload['model']); del payload
    rae.eval().requires_grad_(False).to(device)
    encoder = make_encoder(encoder_path, device)
    if rank == 0 and old:
        report = verify_prefix(encoder, rae, old[0], torch.tensor(plan['mean'],device=device),
            torch.tensor(plan['std'],device=device).clamp_min(1e-4),device,cfg['history'])
        store.write_json('prepare/prefix_contract.json',report)
        print('[prefix contract] '+json.dumps(report),flush=True)
    barrier()
    chosen = [rows[i] for i in selected if i not in existing]
    data = (SceneRGB(dict(cohort,records=chosen),split='val',finite=True,windows=1,
        views=cfg['views'],size=rae.config.target_size,grid=rae.grid) if chosen else [])
    encoding_start=time.monotonic(); encoded=0
    for i in range(rank,len(data),world):
        with heartbeat(f'encoding development clip {i+1}/{len(data)}',rank):
            item=data[i]
            if item['error']:
                errors.append(dict(id=item['id'],error=item['error']))
                continue
            try:
                results.append(encode_eval_sample(encoder,rae,item,device))
                encoded+=(cfg['views']+1)*rae.grid**2
                metrics(store,dict(stage='prepare_eval',completed=len(results),
                    DI_throughput=encoded/max(time.monotonic()-encoding_start,1e-6),
                    scope='encoded latent tokens including independent anchor; load+encode wall time'),rank)
            except (OSError,RuntimeError,ValueError) as exc:
                errors.append(dict(id=item['id'],error=repr(exc)))
    store.save(f'data/eval-r{rank:03d}.pt',dict(samples=results,errors=errors),plan['identity'])
    del encoder,rae
    gc.collect()
    if device.type!='cpu':
        getattr(torch,device.type).empty_cache()
    barrier()
    if rank==0:
        all_items=dict(existing); all_errors=[]
        for r in range(world):
            block=store.load(f'data/eval-r{r:03d}.pt',plan['identity'],True)
            all_items.update({s['id']:s for s in block['samples']})
            all_errors+=block['errors']
        values=[all_items[i] for i in selected if i in all_items]
        for item in values:
            item['fps']=rows[item['id']].get('fps')
            item['time_status']='nominal cohort fps; original cache did not save PTS'
        store.save('data/eval.pt',dict(samples=values,errors=all_errors,
            split='original val; not a newly unseen benchmark',requested=len(selected)),plan['identity'])
    barrier()


def evaluate(model, ema, rae, samples, mean, std, store, plan, cfg, device,
             rank, world, step, *, full=False, baseline=False, budget=None, runtime=None):
    label=cfg.get('source_label','source71609') if baseline else f'step{step:07d}'
    selected=samples if full else samples[:cfg['monitor_cases']]
    seeds=cfg['eval_seeds'] if full else cfg['eval_seeds'][:1]
    arms=['online'] if baseline else ['online','ema']
    rng=capture_rng_state(); was_training=model.training
    model.eval()
    rows=[]
    previous=None
    try:
        for arm in arms:
            if arm=='ema':
                previous={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                ema.copy_to(model)
            try:
                jobs=[(i,s,seed) for i,s in enumerate(selected) for seed in seeds]
                for index,sample,seed in jobs[rank::world]:
                    if budget is not None and budget.due():
                        break
                    prefix=f'eval/{label}/{arm}/case{index:03d}_seed{seed}'
                    case_identity=digest(dict(run=plan['identity'],label=label,arm=arm,
                        id=sample['id'],seed=seed))
                    try:
                        receipt=recover_case(store.local,prefix+'.done.json',case_identity,store.roots)
                    except (OSError,ValueError,RuntimeError) as exc:
                        print(f'[prefix eval] recomputing {prefix}: {exc}',flush=True)
                        receipt=None
                    needs_oracle=full and index in cfg.get('oracle_cases',[]) and seed==seeds[0]
                    if receipt and (receipt['metadata'].get('preview_status')!='completed' or
                        (needs_oracle and receipt['metadata'].get('oracle_preview_status')!='completed')):
                        receipt=None  # Repair failed media, including final-only oracle diagnostics.
                    if receipt:
                        store.publish([*receipt['files'],prefix+'.done.json'])
                        rows.append(receipt['metadata'])
                        continue
                    start=time.monotonic()
                    try:
                        row,frames,z=sample_case(model,rae,sample,mean,std,device,
                            history=cfg['history'],steps=cfg['sample_steps'],seed=seed,
                            net_factory=runtime.bind if runtime is not None else None)
                        row.update(step=step,weights=arm,case_index=index,source=label)
                        if runtime is not None:
                            runtime.annotate(row,sample)
                        generated_label=('GEN '+runtime.arm+' '+arm if runtime is not None else 'GENERATED '+arm)
                        display=labelled_preview(frames,['RAW','AE',generated_label,'COPY'],cfg['history']) \
                            if cfg.get('label_previews') else frames
                        files=preview(store,prefix+'.mp4',display)
                        row['preview_status']='completed' if prefix+'.mp4' in files else 'failed'
                        if cfg.get('convergence') and seed==seeds[0]:
                            from utils.scene_motion_eval import probe_case
                            row['denoising_probe']=probe_case(model,rae,sample,mean,std,device,
                                history=cfg['history'],seed=303,
                                net_factory=runtime.bind if runtime is not None else None)
                            if runtime is not None:
                                runtime.annotate(row['denoising_probe'],sample)
                        if needs_oracle:
                            from utils.scene_motion_eval import oracle_branches
                            diagnostic,oracle_rgb=oracle_branches(rae,sample,z,device,history=cfg['history'])
                            row['gt_assisted_diagnostic']=diagnostic
                            if cfg.get('label_previews'):
                                oracle_rgb=labelled_preview(oracle_rgb,
                                    ['RAW','AE','GENERATED','GT GEO + PRED TEX','PRED GEO + GT TEX'],cfg['history'])
                            oracle_file=prefix+'_GT_ASSISTED_ORACLE.mp4'
                            oracle_files=preview(store,oracle_file,oracle_rgb)
                            row['oracle_preview_status']='completed' if oracle_file in oracle_files else 'failed'
                            files+=oracle_files
                        store.save(prefix+'.pt',dict(latent=z,frame_ids=sample['frame_ids'],
                            history=cfg['history'],seed=seed),plan['identity'])
                        store.write_json(prefix+'.json',row)
                        files += [prefix+'.pt',prefix+'.pt.json',prefix+'.json']
                        seal(store.local,prefix+'.done.json',case_identity,files,row)
                        store.publish([*files,prefix+'.done.json'])
                        rows.append(row)
                    except (OSError,ValueError,RuntimeError) as exc:
                        row=dict(id=sample['id'],seed=seed,weights=arm,error=repr(exc))
                        store.write_json(prefix+'.error.json',row); rows.append(row)
                    metrics(store,dict(stage='eval',step=step,case=index,weights=arm,
                        DI_throughput=cfg['sample_steps']*(cfg['views']-cfg['history'])*model.grid**2/max(time.monotonic()-start,1e-6),
                        scope='future sampler tokens/npu; wall time includes decode, probes, oracles and publication'),rank)
            finally:
                if previous is not None:
                    model.load_state_dict(previous); previous=None
        gathered=gather(rows,world)
        all_rows=[row for batch in gathered for row in batch]
        probe_summary={}
        if cfg.get('convergence'):
            for arm in arms:
                probes=[r['denoising_probe'] for r in all_rows if r.get('weights')==arm
                    and 'error' not in r and 'denoising_probe' in r]
                if len(probes)==len(selected) and len(selected)>0:
                    probe_summary[arm]={key:sum(float(r[key]) for r in probes)/len(probes)
                        for key in ('probe_global_mse','probe_change_mse')}
                    probe_summary[arm]['cases']=len(probes)
        if rank==0:
            expected=len(selected)*len(seeds)*len(arms)
            store.write_json(f'eval/{label}/summary.json',dict(samples=all_rows,step=step,
                expected=expected,completed=sum('error' not in r for r in all_rows),
                previews=sum(r.get('preview_status')=='completed' for r in all_rows),
                status='completed' if len(all_rows)==expected and not any(
                    'error' in r or r.get('preview_status')!='completed' or
                    r.get('oracle_preview_status','completed')!='completed' for r in all_rows) else 'partial',
                full=full,quality_policy='report_only',history=cfg['history'],future=cfg['views']-cfg['history'],
                denoising_probe=probe_summary,
                condition_policy=plan.get('condition', 'observed prefix only'),
                experiment_arm=getattr(runtime,'arm',None),
                note='Source and new weights use identical prefix/noise. Observed frames are not generated.'))
    finally:
        if previous is not None:
            model.load_state_dict(previous)
        model.train(was_training); restore_rng_state(rng)
    barrier()
    return probe_summary


def train(a,cfg,plan,source,store,manifest,selection,rae_path,weights_path,eval_path,
          budget,device,rank,world,*,data_store=None,runtime=None):
    identity=plan['identity']
    latest='train/checkpoint_latest.pt'
    if rank==0:
        recovered=store.load(latest,identity)
        if recovered is not None:
            store.publish([latest,latest+'.json'])
        del recovered
    barrier()
    receipt=store.json(latest+'.json')
    saved=None
    if receipt:
        path=stage_asset(store,receipt,int(os.environ.get('LOCAL_RANK',0)),rank)
        saved=load_staged(path)
    model=(SceneFlow(**plan['model']) if runtime is None else runtime.make_model(plan['model'])).to(device)
    if saved:
        model.load_state_dict(saved['model'])
    else:
        warm=load_staged(weights_path)
        if warm.get('model_args') != plan['model']:
            raise ValueError('Warm-start tensor model arguments differ from contract')
        if 'source_weights' in plan:
            expected_step=int(Path(plan['source_weights']['file']).stem.removeprefix('weights_step'))
            if warm.get('step') != expected_step:
                raise ValueError('Warm-start internal step differs from pinned filename')
        if runtime is None:
            model.load_state_dict(warm['model'],strict=True)
        else:
            runtime.load_warm(model,warm['model'])
        del warm
    parameters=model.parameters() if runtime is None else runtime.optimizer_groups(model)
    optimizer=torch.optim.AdamW(parameters,lr=cfg['lr'],betas=(.9,.95),weight_decay=0.)
    ema=EMA(model,decay=.9995,dtype=torch.float32,warmup=True)
    tracker=None
    if cfg.get('convergence'):
        from utils.scene_motion_objective import PlateauTracker
        tracker=PlateauTracker(**cfg['convergence'])
    step=cursor=0; train_spent=0.
    saved_rng=saved_bank=None
    if saved:
        if saved['world']!=world:
            raise ValueError('Exact training resume requires original world size')
        optimizer.load_state_dict(saved['optimizer'])
        ema.load_state_dict(saved['ema']); ema.load_metadata(saved['ema_metadata']); ema.to(device)
        step=saved['step']; cursor=saved['cursors'][rank]; train_spent=saved['train_spent']
        saved_rng=saved['rng'][rank]; saved_bank=saved['bank_states'][rank]
        progress=store.json('progress.json') or {}
        train_spent=max(train_spent,progress.get('train_spent',0.))
        budget.spent=max(budget.spent,saved.get('total_spent',0.))
        if tracker is not None and saved.get('convergence_state'):
            tracker.load_state_dict(saved['convergence_state'])
    del saved
    core=wrap(model,device,world)
    bank=FilteredLatentSamples(data_store or store,manifest,selection,seed=cfg['seed'],
        on_bad_shard=lambda row:append_metrics(str(store.local/f'train/bad-r{rank:03d}.jsonl'),row))
    if saved_bank is not None:
        bank.load_state_dict(saved_bank)
    mean=torch.tensor(plan['mean'],device=device)
    std=torch.tensor(plan['std'],device=device).clamp_min(1e-4)
    payload=load_staged(rae_path)
    rae=SceneRAE(payload['spec']['source_config'],payload['spec']['channels'])
    rae.load_state_dict(payload['model']); del payload
    rae.eval().requires_grad_(False).to(device)
    samples=load_staged(eval_path)['samples']
    if rank==0:
        store.write_json('train/model.json',dict(identity=identity,parameters=sum(p.numel() for p in model.parameters()),
            model=plan['model'],effective_batch=world*cfg['accum'],model_slots=cfg['views']-1,
            history=cfg['history'],generated_frames=cfg['views']-cfg['history'],
            decoder_temporal_blocks=getattr(rae.config,'decoder_temporal_blocks',None),
            schedule='update-based; wall time only stops resource use',
            **(runtime.checkpoint_metadata() if runtime is not None else {})))
    baseline_label=cfg.get('source_label','source71609')
    baseline_summary=store.json(f'eval/{baseline_label}/summary.json')
    if not baseline_summary or ((tracker is not None or runtime is not None)
                                and baseline_summary.get('status')!='completed'):
        original=None
        try:
            if step:
                if tracker is None and runtime is None:
                    raise RuntimeError('Source baseline receipt missing on trained resume; do not relabel new weights')
                original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                warm=load_staged(weights_path)
                if runtime is None:
                    model.load_state_dict(warm['model'],strict=True)
                else:
                    runtime.load_warm(model,warm['model'])
                del warm
            evaluate(model,ema,rae,samples,mean,std,store,plan,cfg,device,rank,world,0,
                full=True,baseline=True,budget=budget,runtime=runtime)
        finally:
            if original is not None:
                model.load_state_dict(original)
    if saved_rng is not None:
        # Model/codec constructors and DDP setup consume RNG. Restore only once
        # all initialization has finished, immediately before the next update.
        restore_rng_state(saved_rng)
    elif runtime is not None:
        # Extra adapters consume constructor RNG. Start all matched arms from
        # the same *training* random stream, independent of model construction.
        runtime.seed_training(cfg['seed'],rank)
    train_clock=StageBudget(cfg['train_hours']*3600 if cfg['train_hours'] else float('inf'),train_spent)
    start=last_save=last_eval=last_weights=last_full=time.monotonic()
    tokens=processed=0
    flow=WindowFlow(time_shift=max(1.,math.sqrt(model.grid**2*model.channels/4096)))
    stop_reason='step_cap'
    stopped=store.json('train/stopped.json') if tracker is not None else None
    if stopped and (stopped['identity']!=identity or stopped['step']!=step):
        raise ValueError('Training stop decision does not match recovered checkpoint')
    if stopped:
        stop_reason=stopped['stop_reason']
    elif tracker is not None and tracker.converged:
        stop_reason='optimization_plateau_requires_visual_review'

    def save(*,final=False,periodic=True,full_snapshot=False):
        nonlocal last_weights,last_full
        cursors=gather(cursor,world); rng=gather(capture_rng_state(),world)
        bank_states=gather(bank.state_dict(),world)
        if rank==0:
            payload=dict(identity=identity,world=world,model=model.state_dict(),model_args=plan['model'],
                optimizer=optimizer.state_dict(),ema=ema.state_dict(),ema_metadata=ema.metadata(),
                step=step,cursors=cursors,rng=rng,bank_states=bank_states,
                train_spent=train_clock.elapsed(),total_spent=budget.elapsed())
            if runtime is not None:
                payload.update(runtime.checkpoint_metadata())
            if tracker is not None:
                payload['convergence_state']=tracker.state_dict()
            with heartbeat('saving complete training recovery point',rank):
                store.save(latest,payload,identity)
                if full_snapshot or final:
                    store.save(f'train/full_step{step:07d}.pt',payload,identity)
                if periodic or final:
                    store.save(f'train/weights_step{step:07d}.pt',dict(model=model.state_dict(),ema=ema.state_dict(),
                        ema_metadata=ema.metadata(),model_args=plan['model'],step=step,
                        **(runtime.checkpoint_metadata() if runtime is not None else {})),identity)
            trim_local_snapshots(store)
            record_progress(store,budget,step=step,train_spent=train_clock.elapsed(),phase='training')
        bad_path=store.local/f'train/bad-r{rank:03d}.jsonl'
        if bad_path.is_file():
            store.publish([bad_path.relative_to(store.local).as_posix()])
        publish_metrics(store,rank,'train')
        barrier()
        if periodic or final:
            last_weights=time.monotonic()
        if full_snapshot or final:
            last_full=time.monotonic()

    if a.stage=='preflight' and step>=2:
        if rank==0:
            store.write_json('preflight_complete.json',dict(step=step,identity=identity))
        return
    while step<cfg['max_steps']:
        if stopped:
            stop_reason=stopped['stop_reason']
            break
        if tracker is not None and tracker.state_dict().get('converged',False):
            stop_reason='optimization_plateau_requires_visual_review'
            break
        stop=budget.elapsed()>=budget.seconds-cfg['reserve_hours']*3600 or budget.interrupted or train_clock.due()
        if any_rank(stop,device):
            stop_reason='interrupted' if budget.interrupted else 'resource_cap'
            break
        optimizer.zero_grad(set_to_none=True)
        total_loss=plain=0.; noise_bins=[0.,0.,0.]; noise_counts=[0,0,0]
        out_std=0.
        for micro in range(cfg['accum']):
            sample=bank.get(rank+world*cursor); cursor+=1
            batch=prepare_prefix(sample,mean,std,cfg['history'],device=device)
            noise=torch.randn_like(batch.target); u=flow.times(1,device)
            noisy=(1-u[:,None,None,None])*batch.target+u[:,None,None,None]*noise
            sync=core.no_sync() if world>1 and micro+1<cfg['accum'] else contextlib.nullcontext()
            with sync,amp(device):
                net=(PrefixFlowNet(core,batch.known) if runtime is None else
                     runtime.bind(core,batch,sample,training=True))
                prediction=net(noisy,u,batch.anchor)
                motion_metrics={}
                if cfg.get('motion_objective') is not None:
                    from utils.scene_motion_objective import motion_weighted_x0_loss
                    last=batch.known[:,-1:] if batch.known.shape[1] else batch.anchor
                    loss,motion_metrics=motion_weighted_x0_loss(prediction,batch.target,last,u,
                        loss_floor=flow.loss_floor,**cfg['motion_objective'])
                else:
                    loss=flow.loss(prediction,batch.target,noise,u)
                if any_rank(not torch.isfinite(loss).item(),device):
                    raise RuntimeError('Nonfinite prefix flow loss')
                (loss/cfg['accum']).backward()
            mse=(prediction.detach().float()-batch.target).square().mean().item()
            slot=min(2,int(u.item()*3)); noise_bins[slot]+=mse; noise_counts[slot]+=1
            total_loss+=loss.item()/cfg['accum']; plain+=mse/cfg['accum']
            out_std+=prediction.detach().float().std(unbiased=False).item()/cfg['accum']
            tokens+=batch.target.shape[1]*model.grid**2
            processed+=(cfg['views']-1)*model.grid**2
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if any_rank(not torch.isfinite(grad).item(),device):
            raise RuntimeError('Nonfinite prefix flow gradient')
        lr=conservative_lr(step,cfg['lr_steps'],cfg['lr'],cfg['min_lr'],cfg['warmup_steps'])
        for group in optimizer.param_groups:
            group['lr']=lr*group.get('lr_scale',1.)
        optimizer.step(); ema.update(model); step+=1
        if step==1 or step%cfg['log_every']==0:
            probe=model.output[-1].weight.detach().float()
            ema_probe=ema.state_dict()['output.1.weight'].to(device).float()
            metrics(store,dict(stage='train',step=step,loss=total_loss,unweighted_x0_mse=plain,
                noise_bin_mse_sum=noise_bins,noise_bin_counts=noise_counts,pre_clip_grad_norm=grad.item(),
                clipped=bool(grad.item()>1),prediction_std=out_std,lr=lr,
                output_weight_norm=probe.norm().item(),output_ema_distance=(probe-ema_probe).norm().item(),
                DI_throughput=tokens/max(time.monotonic()-start,1e-6),
                processed_tokens_per_second_per_npu=processed/max(time.monotonic()-start,1e-6),
                scope='generated future tokens/npu; wall time including IO/save/eval, excludes guard',
                train_hours=train_clock.elapsed()/3600,total_hours=budget.elapsed()/3600,
                peak_allocated_gib=memory_gib(device),
                **(runtime.training_metrics(model,net,noisy) if runtime is not None else {})),rank)
            if motion_metrics:
                metrics(store,dict(stage='motion_objective',step=step,**{
                    k:float(v) for k,v in motion_metrics.items()},
                    DI_throughput=tokens/max(time.monotonic()-start,1e-6)),rank)
        ev=any_rank(step%cfg['eval_every_steps']==0 if cfg.get('eval_every_steps') else
            time.monotonic()-last_eval>=cfg['eval_seconds'],device)
        sv=any_rank(time.monotonic()-last_save>=cfg['save_seconds'],device)
        if ev:
            probe=evaluate(model,ema,rae, samples,mean,std,store,plan,cfg,device,rank,world,step,
                budget=budget,runtime=runtime)
            if tracker is not None and 'ema' in probe:
                convergence=tracker.update(step,dict(x0_mse=probe['ema']['probe_global_mse'],
                    changed_region_x0_mse=probe['ema']['probe_change_mse']))
                if rank==0:
                    store.write_json('train/convergence.json',convergence)
                    print('[optimization convergence] '+json.dumps(convergence),flush=True)
            last_eval=time.monotonic()
        if sv or ev:
            save(periodic=ev or time.monotonic()-last_weights>=cfg['weights_seconds'],
                full_snapshot=time.monotonic()-last_full>=cfg['full_snapshot_seconds'])
            last_save=time.monotonic()
        if a.stage=='preflight' and step>=2:
            # Do not skip an evaluation scheduled at this boundary. Otherwise
            # preflight/resume changes plateau state versus continuous training.
            save(periodic=False,full_snapshot=True)
            if rank==0:
                store.write_json('preflight_complete.json',dict(step=step,identity=identity,
                    note='Full shape optimizer/save complete; launcher starts new process for exact resume'))
            return
        if tracker is not None and ev and 'ema' in probe and convergence.get('converged',False):
            stop_reason='optimization_plateau_requires_visual_review'
            break
    save(final=True,full_snapshot=True)
    if not any_rank(budget.interrupted,device):
        if rank==0 and tracker is not None:
            store.write_json('train/stopped.json',dict(identity=identity,step=step,stop_reason=stop_reason,
                note='Optimizer finished; incomplete evaluation can resume without more updates.'))
        barrier()
        evaluate(model,ema,rae,samples,mean,std,store,plan,cfg,device,rank,world,step,
            full=True,budget=budget,runtime=runtime)
        if rank==0:
            record_progress(store,budget,step=step,train_spent=train_clock.elapsed(),phase='finished')
            evaluation=store.json(f'eval/step{step:07d}/summary.json',True)
            store.write_json('complete.json',dict(identity=identity,step=step,total_spent=budget.elapsed(),
                train_spent=train_clock.elapsed(),quality_policy='report_only',
                evaluation_status=evaluation['status'],
                stop_reason=stop_reason,
                selected='not automatically selected; inspect online/EMA intermediate videos'))
    else:
        raise SystemExit(75)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--cohort',required=True)
    p.add_argument('--selection',required=True); p.add_argument('--encoder',required=True)
    p.add_argument('--source-local',required=True); p.add_argument('--source-root',action='append',default=[])
    p.add_argument('--output',required=True); p.add_argument('--root',action='append',default=[])
    p.add_argument('--stage',choices=['preflight','train'],default='train')
    p.add_argument('--startup-seconds',type=float,default=0.)
    a=p.parse_args()
    from audit_scene_flow_weights import ensure_disjoint
    ensure_disjoint(a)
    cfg=json.loads(Path(a.config).read_text(encoding='utf8'))
    if cfg['views']!=9 or cfg['history']!=4:
        raise ValueError('This quality run is pinned to original9 / observed4 / generated5')
    if cfg['condition_policy']!='observed_prefix_only_no_text_no_camera':
        raise ValueError('Unsupported condition policy')
    cohort=json.loads(Path(a.cohort).read_text(encoding='utf8'))
    selection=load_street_selection(a.cohort,a.selection)
    device,rank,world=distributed()
    local_rank=int(os.environ.get('LOCAL_RANK',0)); node=int(os.environ.get('GROUP_RANK',0))
    random.seed(cfg['seed']+rank); np.random.seed(cfg['seed']+rank); torch.manual_seed(cfg['seed']+rank)
    store=ArtifactStore(a.output,a.root)
    source=ArtifactStore(Path(a.output)/'_inputs'/f'node{node}',[a.source_local,*a.source_root])
    progress=store.json('progress.json') or {}
    budget=StageBudget(cfg['hours']*3600,progress.get('total_spent',0.)+max(0.,a.startup_seconds))
    budget.install_signals()
    plan,manifest=source_plan(source,cfg,a.cohort,selection,world)
    if rank==0:
        old=store.json('contract.json')
        if old and old['identity']!=plan['identity']:
            raise ValueError('Changed source/data/model/config; use a new output namespace')
        store.write_json('contract.json',plan)
    barrier()
    if store.json('complete.json') or (a.stage=='preflight' and store.json('preflight_complete.json')):
        print('[prefix] requested stage already completed',flush=True)
    else:
        ae_path=stage_asset(source,plan['ae'],local_rank,rank)
        eval_path=stage_asset(source,plan['evaluation'],local_rank,rank)
        prepare_evaluation(source,store,plan,cfg,cohort,selection,a.encoder,ae_path,eval_path,device,rank,world)
        filtered=compact_cache(source,store,manifest,selection,plan['identity'],budget,device,rank,world)
        if rank==0:
            record_progress(store,budget,phase='prepared')
        eval_spec=receipt_spec(store,'data/eval.pt')
        eval_path=stage_asset(store,eval_spec,local_rank,rank)
        weights_path=stage_asset(source,plan['source_weights'],local_rank,rank)
        train(a,cfg,plan,source,store,filtered,selection,ae_path,weights_path,eval_path,budget,device,rank,world)
    barrier()
    import train_scene_pipeline as pipeline
    if pipeline.WRITER is not None:
        pipeline.WRITER.close(); pipeline.WRITER=None
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__=='__main__':
    main()
