"""Stage64: train an edit transport module through a frozen, decodable RAE.

Train/monitor: one edited first frame is a declared condition; edited future
frames live in a separate supervision artifact. Dev: only Stage63 legal model/
user references. Native VACE, RGB writeback, and learned RAE writeback share
conditions and seeds. No target video is accessed by prediction or rendering.
"""
import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F

from audit_edit_transport import rae_decoder
from edit_reference_video import (from_square, valid_pixels, publish_case,
    recovered_case, failure, crop_video, crop_float, output_metrics)
from models.edit_transport import EditTransport, decoded_reference_frame
from utils.editing_data import digest, file_hash, rae_canvas
from utils.editing_train import (distributed, barrier, gather, any_rank, progress,
    load_teacher, load_encoder, sample_row, update_ema, write_video)
from utils.scene_run import ArtifactStore, StageBudget
from utils.training import capture_rng_state, restore_rng_state
from utils.native_audit_io import seal, recover_case
from utils.editing_publication import reconcile_case


def code_signature():
    root = Path(__file__).resolve().parent
    names = ('train_edit_transport.py', 'models/edit_transport.py',
             'utils/edit_transport_data.py', 'utils/editing_video_transport.py',
             'utils/vace_edit_runtime.py', 'models/scene_rae.py')
    return {name: file_hash(root / name) for name in names}


def checked_manifest(out, cfg):
    manifest = out.json('manifest.json', True)
    if manifest['config'] != cfg or manifest['code'] != code_signature():
        raise ValueError('Stage64 code/config differs; use a fresh LEARNED_NAMESPACE')
    return manifest


def references_store(a):
    return ArtifactStore(a.references, a.reference_root)


def prepare(a, cfg, out, device, rank, world):
    from utils.edit_transport_data import build_split, prepare_training_pair, prepare_stage63_case
    data = ArtifactStore(a.data, a.data_root)
    bank = data.json('bank.json', True)
    refs = references_store(a)
    reference_manifest = refs.json('manifest.json', True)
    if reference_manifest['bank'] != bank['identity']:
        raise ValueError('Dev references and training bank have different native bank identities')
    split = build_split(bank, cfg)
    teacher = dict(rae=file_hash(a.rae), encoder=file_hash(a.encoder))
    manifest = dict(schema='learned-edit-run-v1', config=cfg, code=code_signature(),
                    bank=bank['identity'], references=reference_manifest['identity'],
                    split=split, teacher=teacher)
    manifest['identity'] = digest(manifest)
    if any(v != manifest['identity'] for v in gather(manifest['identity'], world)):
        raise ValueError('Nodes disagree on Stage64 inputs or weights')
    old = out.json('manifest.json')
    if old and old['identity'] != manifest['identity']:
        raise ValueError('Input identity changed; select a fresh LEARNED_NAMESPACE')
    if rank == 0:
        out.write_json('manifest.json', manifest)
    barrier()
    identity = manifest['identity']
    reports = []
    encoder = rae = None
    started = time.monotonic()
    moments = None
    for row in split['records'][rank::world]:
        key = row['id']
        names = dict(inputs=f'inputs/{key}.pt', supervision=f'supervision/{key}.pt')
        try:
            inputs = out.load(names['inputs'], identity)
            supervision = None
            if row['partition'] != 'dev':
                supervision = out.load(names['supervision'], identity)
            if inputs is None or (row['partition'] != 'dev' and supervision is None):
                if rae is None:
                    rae = load_teacher(a.rae, device)
                    encoder = load_encoder(a.encoder, device)
                if row['partition'] == 'dev':
                    inputs = prepare_stage63_case(refs, key, cfg, rae, encoder, device=device)
                else:
                    sample = data.load(row['file'], bank['identity'], True)
                    inputs, supervision, diagnostics = prepare_training_pair(
                        sample, cfg, rae, encoder, device=device, partition=row['partition'])
                    out.save(names['supervision'], supervision, identity)
                    del sample
                out.save(names['inputs'], inputs, identity)
            report = dict(id=key, partition=row['partition'], group=row['group'],
                          status='completed', inputs=names['inputs'])
            if row['partition'] != 'dev':
                report['supervision'] = names['supervision']
            if row['partition'] == 'train':
                z = inputs['source_z'].double().reshape(-1, inputs['source_z'].shape[-1])
                current = (len(z), z.sum(0), z.square().sum(0))
                moments = current if moments is None else tuple(x+y for x, y in zip(moments, current))
            reports.append(report)
            completed = sum(r['status']=='completed' for r in reports)
            progress(out, f'prepare_r{rank}', dict(phase='prepare', completed=completed, id=key,
                DI_throughput=completed*cfg['frames']/max(time.monotonic()-started, 1e-6),
                throughput_unit='processed source frames/s/rank including encoding, flow and IO'))
        except Exception as exc:
            reports.append(dict(failure(out, 'prepare', row, exc), partition=row['partition']))
    all_rows = sum(gather(reports, world), [])
    moment_rows = gather(None if moments is None else (moments[0], moments[1].tolist(), moments[2].tolist()), world)
    if rank == 0:
        valid = [v for v in moment_rows if v is not None]
        scale = None
        if valid:
            count = sum(v[0] for v in valid)
            total = sum(torch.tensor(v[1], dtype=torch.float64) for v in valid)
            square = sum(torch.tensor(v[2], dtype=torch.float64) for v in valid)
            scale = (square/count-(total/count).square()).clamp_min(0).sqrt().clamp_min(.05).float().tolist()
        out.write_json('prepared.json', dict(identity=identity, requested=len(split['records']),
            records=all_rows, source_scale=scale, scale_source='successful TRAIN source states only'))
    barrier()


def batch_inputs(payload, device):
    return dict(source_z=payload['source_z'][None].to(device).float(),
        anchor_z=payload['anchor_z'][None].to(device).float(),
        source=payload['source'][None].to(device).float()/255,
        proposal=payload['proposal'][None].to(device).float()/255,
        confidence=payload['confidence'][None].to(device).float(),
        masks=payload['masks'][None].to(device).float())


def corrupt_observations(inputs, rng, probability):
    """TRAIN-only missing-observation augmentation; never reads supervision.

    A coherent space/time hole removes a transported observation, not source
    content or the edited first frame. This teaches completion even on clips
    whose classical flow happens to be reliable everywhere.
    """
    if rng.random() >= probability or inputs['source'].shape[1] < 2:
        return inputs
    result = dict(inputs)
    result['proposal'] = inputs['proposal'].clone()
    result['confidence'] = inputs['confidence'].clone()
    _, frames, _, height, width = inputs['source'].shape
    h = max(1, int(height*rng.uniform(.2,.45)))
    w = max(1, int(width*rng.uniform(.2,.45)))
    y, x = rng.randrange(height-h+1), rng.randrange(width-w+1)
    t = rng.randrange(1,frames)
    end = min(frames,t+rng.randrange(1,4))
    region = (slice(None),slice(t,end),slice(None),slice(y,y+h),slice(x,x+w))
    selected = inputs['masks'][region]>.05
    result['proposal'][region] = torch.where(selected,inputs['source'][region],result['proposal'][region])
    result['confidence'][region] = torch.where(selected,torch.zeros_like(result['confidence'][region]),result['confidence'][region])
    return result


def training_loss(model, decoder, inputs, target, scale, cfg, frames, valid_rect=None):
    """Future labels enter only this objective, never the model forward call."""
    delta = model(**inputs, scale=scale)
    losses = []
    diagnostics = dict(edit_l1=0., keep_l1=0., rgb_l1=0., output_l1=0., writable_fraction=0., applied_residual_l1=0.)
    valid = torch.ones_like(inputs['masks'][:,0])
    if valid_rect is not None:
        y,x,h,w = valid_rect
        valid.zero_()
        valid[:,:,y:y+h,x:x+w] = 1
    weight = cfg['training']
    for frame in frames:
        reference, info = decoded_reference_frame(decoder, inputs['source_z'], delta, scale,
            inputs['source'], inputs['proposal'], inputs['confidence'], inputs['masks'], frame, cfg['learned'])
        truth = target[:, frame]
        # Paired future differences are LOSS weights only, never conditioning.
        changed = ((truth-inputs['source'][:, frame]).abs().amax(1, keepdim=True) > .06).float()*valid
        preserved = valid*(1-changed)
        error = (reference-truth).abs().mean(1, keepdim=True)
        edit = (error*changed).sum()/changed.sum().clamp_min(1)
        keep = (error*preserved).sum()/preserved.sum().clamp_min(1)
        losses.append(weight['edit_weight']*edit+weight['keep_weight']*keep)
        diagnostics['edit_l1'] += float(edit.detach())/len(frames)
        diagnostics['keep_l1'] += float(keep.detach())/len(frames)
        diagnostics['output_l1'] += float((error.detach()*valid).sum()/valid.sum().clamp_min(1))/len(frames)
        diagnostics['rgb_l1'] += float(((inputs['proposal'][:, frame]-truth).abs()*valid).sum()/(3*valid.sum()).clamp_min(1))/len(frames)
        diagnostics['writable_fraction'] += float(((info['gate']>0).float()*valid).sum()/valid.sum().clamp_min(1))/len(frames)
        diagnostics['applied_residual_l1'] += float((info['applied_residual'].detach().abs()*valid).sum()/(3*valid.sum()).clamp_min(1))/len(frames)
    loss = torch.stack(losses).mean()+weight['latent_weight']*delta.square().mean()
    diagnostics['loss'] = float(loss.detach())
    return loss, diagnostics


def validate_training(model, decoder, out, prepared, scale, cfg, device, rank, world, step):
    rows = [r for r in prepared['records'] if r['status']=='completed' and r['partition']=='monitor']
    rows = sorted(rows, key=lambda r:r['id'])[:cfg['training']['monitor_cases']]
    reports = []
    model.eval()
    for row in rows[rank::world]:
        try:
            cached = out.load(row['inputs'], prepared['identity'], True)
            inputs = batch_inputs(cached, device)
            truth = out.load(row['supervision'], prepared['identity'], True)['target'][None].to(device).float()/255
            times = sorted(set((max(1, inputs['source'].shape[1]//3), inputs['source'].shape[1]-1)))
            with torch.no_grad():
                _, metrics = training_loss(model, decoder, inputs, truth, scale, cfg, times, cached.get('valid_rect'))
            reports.append(dict(id=row['id'], status='completed', **metrics))
        except Exception as exc:
            reports.append(dict(id=row['id'], status='failed', error=repr(exc)))
    reports = sum(gather(reports, world), [])
    values = [r for r in reports if r['status']=='completed']
    # A missing monitor case cannot silently improve model selection.
    score = sum(r['loss'] for r in values)/len(values) if values and len(values)==len(rows) else None
    result = dict(step=step, score=score, cases=reports, requested=len(rows),
                  protocol='heldout_train_groups_with_edited_FIRST_FRAME_condition',
                  not_evaluated='not text-only instruction following, not a final video benchmark')
    if rank == 0:
        out.write_json(f'monitor/step_{step:06d}.json', result)
    model.train()
    return result


def run_training(cfg, out, prepared, model, decoder, device, rank, world, stop_after=0):
    identity = prepared['identity']
    tc = cfg['training']
    scale = torch.tensor(prepared['source_scale'], device=device)
    rows = sorted([r for r in prepared['records'] if r['status']=='completed' and r['partition']=='train'], key=lambda r:r['id'])
    if not rows:
        raise RuntimeError('No valid TRAIN inputs; RGB/native rendering can still run')
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc['lr'], weight_decay=.01)
    ema = {n:p.detach().float().clone() for n,p in model.named_parameters()}
    state = out.load('checkpoint_latest.pt', identity)
    start = 0
    best = float('inf')
    stale = 0
    converged = False
    updates = 0
    if state:
        if state['world'] != world:
            raise ValueError('Training resume world size changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        ema = {n:v.to(device).float() for n,v in state['ema'].items()}
        restore_rng_state(state['rng'][rank])
        start, best, stale = state['step'], state['best_score'], state['stale']
        converged = state.get('converged', False)
        updates = state.get('updates', start)
    wrapped = (torch.nn.parallel.DistributedDataParallel(model,
        device_ids=[device.index] if device.type!='cpu' else None,
        broadcast_buffers=False) if world>1 else model)
    started = time.monotonic()
    budget = StageBudget(float('inf'))
    budget.install_signals()
    limit = min(stop_after or tc['max_steps'], tc['max_steps'])
    last_step = start
    skipped = 0
    processed_frames = 0
    was_interrupted = False
    tb = None
    if rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(str(out.local/'tb'))
        except ImportError:
            print('[learned edit] tensorboard unavailable; JSONL metrics remain enabled', flush=True)
        out.write_json('training_run.json', dict(identity=identity, trainable_parameters=sum(p.numel() for p in model.parameters()),
            frozen='RAE decoder/StreamVGGT/VACE', config=cfg, world=world, train_cases=len(rows)))

    def save(step):
        rng = gather(capture_rng_state(), world)
        if rank == 0:
            payload = dict(step=step, world=world, model=model.state_dict(), ema=ema,
                optimizer=optimizer.state_dict(), rng=rng, config=cfg, scale=scale.cpu(),
                best_score=best, stale=stale, converged=converged, identity=identity, updates=updates)
            out.save(f'checkpoints/step_{step:06d}.pt', payload, identity)
            out.save('checkpoint_latest.pt', payload, identity)
            if tb:
                tb.flush()
            files = [p.relative_to(out.local).as_posix() for parent in ('logs','tb')
                     for p in (out.local/parent).glob('*') if p.is_file()]
            if files:
                out.publish(files)
        barrier()

    for step in range(start, start if converged else limit):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rate = tc['lr']*min(1., (step+1)/max(1, tc['warmup_steps']))
        for group in optimizer.param_groups:
            group['lr'] = rate
        totals = {}
        failed = False
        for micro in range(tc['accum']):
            row = sample_row(rows, step, micro, rank, world, tc['accum'], cfg['seed'])
            error = None
            try:
                cached = out.load(row['inputs'], identity, True)
                inputs = batch_inputs(cached, device)
                target = out.load(row['supervision'], identity, True)['target'][None].to(device).float()/255
                rng = random.Random(cfg['seed']+step*world*tc['accum']+micro*world+rank)
                if rng.random() < tc.get('identity_fraction', .1):
                    inputs['anchor_z'] = inputs['source_z'][:, :1].clone()
                    inputs['proposal'] = inputs['source'].clone()
                    target = inputs['source'].clone()
                inputs = corrupt_observations(inputs, rng, tc.get('observation_dropout',.3))
                first = rng.randrange(1, inputs['source'].shape[1])
                neighbor = first+1 if first+1<inputs['source'].shape[1] else max(1, first-1)
                times = sorted(set((first, neighbor)))
            except Exception as exc:
                error = repr(exc)
            if any_rank(error is not None, device):
                skipped += 1
                failed = True
                if error:
                    progress(out, f'train_skip_r{rank}', dict(phase='train', step=step, id=row['id'], error=error))
                break
            # No no_grad around the frozen decoder: input gradients train theta.
            loss, metrics = training_loss(wrapped, decoder, inputs, target, scale, cfg, times, cached.get('valid_rect'))
            if any_rank(not bool(torch.isfinite(loss)), device):
                # Finish the DDP graph on every rank with a finite zero loss.
                torch.nan_to_num(loss, nan=0., posinf=0., neginf=0.).mul(0).backward()
                failed = True
                skipped += 1
                break
            (loss/tc['accum']).backward()
            processed_frames += len(times)
            for key,value in metrics.items():
                totals[key] = totals.get(key, 0.)+value/tc['accum']
        if failed:
            optimizer.zero_grad(set_to_none=True)
        else:
            head_grad = sum(float(p.grad.detach().float().square().sum()) for n,p in model.named_parameters()
                            if n.startswith('output.') and p.grad is not None)**.5
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc['grad_clip'])
            totals.update(gradient_norm=float(norm), head_gradient_norm=head_grad)
            if not any_rank(not bool(torch.isfinite(norm)), device):
                optimizer.step()
                update_ema(ema, model, step, tc['ema_decay'])
                updates += 1
            else:
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
        last_step = step+1
        if last_step%tc['log_every']==0 or last_step==1:
            values = gather(totals, world)
            all_frames = sum(gather(processed_frames, world))
            if rank == 0:
                means = {k:sum(r.get(k,0.) for r in values)/world for k in totals}
                progress(out, 'training', dict(phase='train', step=last_step, updates=updates, lr=rate, skipped=skipped, **means,
                    DI_throughput=all_frames/max(time.monotonic()-started,1e-6),
                    throughput_unit='supervised decoder frames/s across ranks including monitor and IO'))
                if tb:
                    for key,value in means.items():tb.add_scalar(key, value, last_step)
        if last_step%tc['eval_every']==0 or last_step==tc['max_steps']:
            online = {k:v.detach().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(ema)
            report = validate_training(model, decoder, out, prepared, scale, cfg, device, rank, world, last_step)
            model.load_state_dict(online)
            score = report['score']
            if score is not None:
                improved = score < best-tc['min_improvement']*max(abs(best) if math.isfinite(best) else 1., .01)
                if improved:
                    best, stale = score, 0
                    if rank == 0:
                        out.save('selected.pt', dict(model={k:v.detach().cpu() for k,v in ema.items()},
                            scale=scale.cpu(), step=last_step, config=cfg, score=score,
                            selection='EMA heldout train-group supervised reference loss'), identity)
                elif last_step>=tc['min_steps']:
                    stale += 1
                converged = last_step>=tc['min_steps'] and stale>=tc['patience']
        interrupted = any_rank(budget.interrupted, device)
        was_interrupted = was_interrupted or interrupted
        if last_step%tc['save_every']==0 or converged or interrupted:
            save(last_step)
        if converged or interrupted:
            break
    save(last_step)
    if rank == 0:
        selected = out.load('selected.pt', identity)
        # A pre-monitor interruption may already have published a fallback.
        # Refresh that fallback after resumed updates; retain a scored best EMA.
        if selected is None or selected.get('score') is None:
            out.save('selected.pt', dict(model={k:v.detach().cpu() for k,v in ema.items()},
                scale=scale.cpu(), step=last_step, config=cfg, score=None,
                selection='final EMA; no complete monitor score'), identity)
        out.write_json('training_summary.json', dict(step=last_step, updates=updates, converged=converged,
            best_score=best if math.isfinite(best) else None, skipped=skipped,
            status=('failed_no_updates' if not updates else 'interrupted' if was_interrupted or
                (stop_after and last_step<tc['max_steps'] and not converged) else 'completed'),
            note='Convergence of supervised transport is not proof of final VACE improvement'))
        if tb:tb.close()
    barrier()


def train(a, cfg, out, device, rank, world):
    manifest = checked_manifest(out, cfg)
    if file_hash(a.rae)!=manifest['teacher']['rae']:
        raise ValueError('Frozen RAE differs from the prepared source state identity')
    prepared = out.json('prepared.json', True)
    if not prepared['source_scale']:
        raise RuntimeError('No source training statistics')
    torch.manual_seed(cfg['seed'])
    model = EditTransport(cfg['learned'], len(prepared['source_scale'])).to(device)
    rae = load_teacher(a.rae, device).float()
    run_training(cfg, out, prepared, model, rae_decoder(rae), device, rank, world, getattr(a,'stop_after',0))


def predict(a, cfg, out, device, rank, world):
    from utils.editing_video_transport import propagate_reference, warp_flow
    from edit_reference_video import tensor_hash
    manifest = checked_manifest(out, cfg)
    identity = manifest['identity']
    refs = references_store(a)
    rows = [r for r in manifest['split']['records'] if r['partition']=='dev']
    reports = []
    model = rae = selected = None
    selected_receipt = out.json('selected.pt.json')
    selected_hash = selected_receipt['sha256'] if selected_receipt else None
    start = time.monotonic()
    for row in rows[rank::world]:
        folder = 'predictions/'+row['id']
        try:
            payload, record = recovered_case(out, folder, identity)
            if payload is not None and (payload.get('selected_hash')!=selected_hash or payload.get('learned') is None):
                payload = record = None
            if record:
                reports.append(record)
                continue
            anchor = refs.load('anchors/'+row['id']+'/result.pt', manifest['references'], True)
            source, spatial = anchor['source'], anchor['spatial']
            if payload is None:
                square, rect = rae_canvas(source, spatial['valid_rect'])
                ar, _ = rae_canvas(anchor['anchor'][None], spatial['valid_rect'])
                support, _ = rae_canvas(anchor['support'][None], spatial['valid_rect'])
                full = propagate_reference(square, ar[0], support[0], cfg['transport'])
                raw = (source+from_square(full['proposal']-square, rect, spatial)).clamp(0,1)
                raw[0] = anchor['anchor']
                k = cfg['learned']['support_dilation']
                square_masks = F.max_pool2d(full['masks'], k, 1, k//2)
                masks = from_square(square_masks, rect, spatial).clamp(0,1)*valid_pixels(source, spatial['valid_rect'])[None]
                learned = None
                diagnostics = {}
                try:
                    data = out.load('inputs/'+row['id']+'.pt', identity, True)
                    # Coarse/full source correspondences differ near occlusions.
                    # Their permitted support union is shared by ALL render arms.
                    coarse_support = F.max_pool2d(data['masks'].float(), k, 1, k//2)
                    for j,t in enumerate(data['observed_indices']):
                        square_masks[t] = torch.maximum(square_masks[t],coarse_support[j])
                        if t+1<len(source):
                            warped = warp_flow(coarse_support[j:j+1],full['flows_backward'][t])[0]
                            square_masks[t+1] = torch.maximum(square_masks[t+1],warped)
                    masks = from_square(square_masks,rect,spatial).clamp(0,1)*valid_pixels(source,spatial['valid_rect'])[None]
                    if selected is None:
                        selected = out.load('selected.pt', identity, True)
                        model = EditTransport(cfg['learned'], len(selected['scale'])).to(device).eval()
                        model.load_state_dict(selected['model'])
                        if file_hash(a.rae)!=manifest['teacher']['rae']:
                            raise ValueError('Frozen RAE differs from training/preparation')
                        rae = load_teacher(a.rae, device).float()
                    inputs = batch_inputs(data, device)
                    scale = selected['scale'].to(device)
                    with torch.no_grad():
                        delta = model(**inputs, scale=scale)
                        if not bool(torch.isfinite(delta).all()):raise ValueError('Nonfinite learned latent correction')
                        calibrated = [decoded_reference_frame(rae_decoder(rae), inputs['source_z'], delta,
                            scale, inputs['source'], inputs['proposal'], inputs['confidence'], inputs['masks'],
                            f, cfg['learned'])[0][0].cpu() for f in range(len(data['source']))]
                    # Transfer only learned corrections, never the source's RAE reconstruction.
                    correction = torch.stack(calibrated)-inputs['proposal'][0].cpu()
                    all_correction = torch.zeros_like(square)
                    for j,t in enumerate(data['observed_indices']):
                        all_correction[t] = correction[j]
                        if t+1<len(source):
                            all_correction[t+1] = warp_flow(correction[j:j+1], full['flows_backward'][t])[0]
                    learned = (source+from_square(full['proposal']+all_correction-square,rect,spatial)).clamp(0,1)
                    learned[0] = anchor['anchor']
                    if not bool(torch.isfinite(learned).all()):raise ValueError('Nonfinite learned reference')
                    diagnostics = dict(selected_step=selected['step'], selection=selected['selection'],
                        correction_l1=float(correction.abs().mean()), anchor_exact=bool(torch.equal(learned[0],anchor['anchor'])))
                except Exception as exc:
                    learned = None
                    diagnostics = dict(learned_error=repr(exc))
                    failure(out, 'prediction_learned', row, exc)
                payload = dict(source=source, anchor=anchor['anchor'], rgb=raw, learned=learned,
                    masks=masks, spatial=spatial, text=anchor['text'], negative=anchor['negative'],
                    provenance=anchor['provenance'], diagnostics=diagnostics, selected_hash=selected_hash)
            meta = dict(id=row['id'], status='completed', phase='predict',
                learned_available=payload['learned'] is not None, provenance=payload['provenance'], **payload['diagnostics'])
            reports.append(publish_case(out,folder,identity,payload,meta,
                [source,payload['rgb'],payload['learned'] if payload['learned'] is not None else source],
                ['SOURCE','RGB FLOW REFERENCE','LEARNED RAE / SOURCE IF FAILED'],spatial['valid_rect'],cfg['fps']))
            completed=sum(r['status']=='completed' for r in reports)
            progress(out,f'predict_r{rank}',dict(phase='predict',completed=completed,
                DI_throughput=completed*cfg['frames']/max(time.monotonic()-start,1e-6),throughput_unit='processed source frames/s/rank including IO'))
        except Exception as exc:reports.append(failure(out,'predict',row,exc))
    reports = sum(gather(reports,world),[])
    if rank==0:out.write_json('phases/predict.json',dict(requested=len(rows),cases=reports))
    barrier()


def render(a, cfg, out, device, rank, world):
    from utils.vace_edit_runtime import load_backbone, load_vae, pack_context, sample_native, decode_native, build_reference_writeback
    from edit_reference_video import tensor_hash
    manifest = checked_manifest(out, cfg)
    identity = manifest['identity']
    rows = [r for r in manifest['split']['records'] if r['partition']=='dev']
    variants = ('native_reference','rgb_writeback','learned_rae_writeback')
    reports = []
    model = vae = None
    start = time.monotonic()
    for row in rows[rank::world]:
        for variant in variants:
            folder = 'render/'+variant+'/'+row['id']
            try:
                data = out.load('predictions/'+row['id']+'/result.pt',identity,True)
                source, masks, anchor = data['source'],data['masks'],data['anchor']
                if variant=='learned_rae_writeback' and data['learned'] is None:
                    raise RuntimeError('Learned prediction failed; native and RGB remain separately usable')
                seed=cfg['seed']+int(digest(row['id'])[:8],16)%1000000
                input_identity=digest([identity,row['id'],seed,tensor_hash(masks),tensor_hash(anchor)])
                reference = None if variant=='native_reference' else data['rgb' if variant=='rgb_writeback' else 'learned']
                generation_identity=digest([input_identity,variant,tensor_hash(reference) if reference is not None else None])
                payload, record = recovered_case(out,folder,identity)
                if payload is not None and payload.get('generation_identity')!=generation_identity:
                    payload = record = None
                if record:
                    reports.append(record)
                    continue
                if payload is None:
                    if model is None:
                        expected_assets = references_store(a).json('manifest.json',True).get('assets',{}).get('model')
                        if expected_assets is not None:
                            from utils.vace_edit_runtime import validate_assets
                            if validate_assets(a.model)!=expected_assets:
                                raise ValueError('Native renderer assets differ from the shared-reference experiment')
                        model=load_backbone(a.model,device)
                        vae=load_vae(a.model,device)
                    events=[]
                    def observer(event):
                        events.append(event)
                        progress(out,f'writeback_r{rank}',dict(phase='writeback',id=row['id'],variant=variant,**event))
                    with torch.no_grad():
                        context=pack_context(vae,(source*2-1).permute(1,0,2,3).to(device),
                            masks.permute(1,0,2,3).to(device),[(anchor*2-1)[:,None].to(device)])
                        callback=None
                        if variant!='native_reference':
                            # Flow reliability is NOT multiplied here: it would erase learned completion.
                            callback=build_reference_writeback(vae,(reference*2-1).permute(1,0,2,3).to(device),cfg,
                                reference_count=1,mask=masks.permute(1,0,2,3).to(device),observer=observer)
                        latent=sample_native(model,context,data['text'].to(device),data['negative'].to(device),cfg,seed,velocity_callback=callback)
                        generated=((decode_native(vae,latent,reference_count=1).float().cpu()+1)/2).permute(1,0,2,3)
                    if generated.shape!=source.shape or not bool(torch.isfinite(generated).all()):
                        raise ValueError('Invalid native output')
                    payload=dict(generated=generated,seed=seed,events=events,
                        input_identity=input_identity,generation_identity=generation_identity)
                meta=dict(id=row['id'],phase='render',status='completed',variant=variant,
                    seed=payload['seed'],input_identity=payload['input_identity'],writeback_events=payload['events'],
                    **output_metrics(crop_float(source,data['spatial']['valid_rect']),
                        crop_float(payload['generated'],data['spatial']['valid_rect']),crop_float(masks,data['spatial']['valid_rect'])))
                reports.append(publish_case(out,folder,identity,payload,meta,
                    [source,anchor[None].expand_as(source),payload['generated']],
                    ['SOURCE','SHARED REFERENCE',variant.upper()],data['spatial']['valid_rect'],cfg['fps']))
                completed=sum(r['status']=='completed' for r in reports)
                progress(out,f'render_r{rank}',dict(phase='render',completed=completed,variant=variant,
                    DI_throughput=completed*cfg['frames']/max(time.monotonic()-start,1e-6),
                    throughput_unit='generated frames/s/rank including rendering and IO'))
            except Exception as exc:reports.append(dict(failure(out,'render_'+variant,row,exc),variant=variant))
    reports=sum(gather(reports,world),[])
    if rank==0:out.write_json('phases/render.json',dict(requested=len(rows)*3,cases=reports))
    barrier()


def summarize(out,cfg):
    from html import escape
    from edit_reference_video import tensor_hash
    manifest=checked_manifest(out,cfg)
    identity=manifest['identity']
    rows=[r for r in manifest['split']['records'] if r['partition']=='dev']
    failures=[]
    synced=[]
    current_renders={}
    for phase in ('predict','render'):
        report=out.json('phases/'+phase+'.json')
        cases=[] if report is None else report['cases']
        if phase=='render':current_renders={(r['id'],r['variant']):r['status'] for r in cases}
        expected=len(rows)*(3 if phase=='render' else 1)
        if len(cases)!=expected:failures.append(dict(phase=phase,error='Coverage mismatch',expected=expected,actual=len(cases)))
        for record in cases:
            if record['status']!='completed':failures.append(record);continue
            if phase=='predict' and not record['learned_available']:failures.append(record)
            try:
                roots=reconcile_case(out,record['receipt'],identity)
                synced.append(dict(receipt=record['receipt'],ok=all(r['ok'] for r in roots)))
            except Exception as exc:synced.append(dict(receipt=record['receipt'],ok=False,error=repr(exc)))
    page=['<!doctype html><meta charset="utf-8"><title>Learned edit transport</title>',
          '<style>body{font:16px sans-serif;max-width:1600px;margin:24px auto}video{max-width:100%}pre{white-space:pre-wrap}</style>',
          '<h1>Native / RGB writeback / learned RAE writeback</h1>',
          '<p>Shared source/reference/mask/seed. No future edited target as inference input. First-frame references are preserved before rendering; final native sampling is not a hard RGB clamp.</p>']
    for row in rows:
        try:
            variants=('native_reference','rgb_writeback','learned_rae_writeback')
            if any(current_renders.get((row['id'],v))!='completed' for v in variants):
                raise ValueError('A current render arm failed or is missing; older files are not a successful comparison')
            data=out.load('predictions/'+row['id']+'/result.pt',identity,True)
            chosen=out.json('selected.pt.json')
            if not chosen or data.get('selected_hash')!=chosen['sha256']:
                raise ValueError('Predictions use an older selection; run predict and render')
            payloads=[out.load('render/'+v+'/'+row['id']+'/result.pt',identity,True) for v in variants]
            if len({p['input_identity'] for p in payloads})!=1:raise ValueError('Unmatched rendering conditions')
            seed=cfg['seed']+int(digest(row['id'])[:8],16)%1000000
            expected=digest([identity,row['id'],seed,tensor_hash(data['masks']),tensor_hash(data['anchor'])])
            for variant,payload in zip(variants,payloads):
                reference=None if variant=='native_reference' else data['rgb' if variant=='rgb_writeback' else 'learned']
                expected_generation=digest([expected,variant,tensor_hash(reference) if reference is not None else None])
                if payload['generation_identity']!=expected_generation:
                    raise ValueError('Render uses an older prediction; run render')
            folder='comparisons/'+row['id']
            receipt=folder+'/receipt.json'
            previous=recover_case(out.local,receipt,identity,out.roots)
            generations=[p['generation_identity'] for p in payloads]
            if previous and previous['metadata'].get('generations')!=generations:previous=None
            if not previous:
                movie=folder+'/native_rgb_learned.mp4'
                write_video(out.local/movie,*[crop_video(p['generated'],data['spatial']['valid_rect']) for p in payloads],
                            cfg['fps'],labels=['NATIVE + REFERENCE','RGB WRITEBACK','LEARNED RAE WRITEBACK'])
                previous=seal(out.local,receipt,identity,[movie],dict(id=row['id'],generations=generations))
            out.publish([*previous['files'],receipt])
            roots=reconcile_case(out,receipt,identity)
            synced.append(dict(receipt=receipt,ok=all(r['ok'] for r in roots)))
            page.extend(['<h2>'+escape(row['id']+' '+row.get('instruction',''))+'</h2>',
                '<video controls preload="metadata" src="comparisons/'+row['id']+'/native_rgb_learned.mp4"></video>'])
        except Exception as exc:failures.append(dict(phase='comparison',id=row['id'],error=repr(exc)))
    prepared=out.json('prepared.json') or {}
    skipped_training=[r for r in prepared.get('records',[]) if r['status']!='completed' and r['partition']!='dev']
    failures.extend(r for r in prepared.get('records',[]) if r['status']!='completed' and r['partition']=='dev')
    training=out.json('training_summary.json')
    if not training or training['status']!='completed':failures.append(dict(phase='train',status=training))
    bad=[r for r in synced if not r['ok']]
    result=dict(status='completed' if not failures and not bad else 'incomplete',cases=len(rows),
        training=training,failures=failures,media_failures=bad,skipped_training_data=skipped_training,
        limitations=['Training monitor uses held-out TRAIN groups and edited first-frame references, not text-only evaluation.',
            'Dev uses existing model/user anchors; incorrect anchors remain end-to-end failures.',
            'RAE-specific advantage is unproven until matched RGB/native videos are reviewed.'])
    out.write_json('summary.json',result)
    out.write_json('media_sync.json',dict(checked=len(synced),failed=bad))
    page.append('<pre>'+escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre>')
    (out.local/'report.html').write_text('\n'.join(page),encoding='utf8')
    out.publish(['report.html'])
    if failures or bad:raise RuntimeError('Incomplete Stage64 cases; successful outputs retained, inspect summary.json')


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('prepare','train','predict','render','summarize'),required=True)
    for name in ('config','output'):p.add_argument('--'+name,required=True)
    for name in ('data','references','rae','encoder','model'):p.add_argument('--'+name,default='')
    for name in ('root','data-root','reference-root'):p.add_argument('--'+name,action='append',default=[])
    p.add_argument('--stop-after',type=int,default=0)
    return p


def main():
    a=build_parser().parse_args()
    cfg=json.loads(Path(a.config).read_text())
    out=ArtifactStore(a.output,a.root)
    if a.phase=='summarize':
        if int(os.environ.get('NODE_RANK','0'))==0:summarize(out,cfg)
        return
    device,rank,world=distributed()
    try:globals()[a.phase](a,cfg,out,device,rank,world)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
