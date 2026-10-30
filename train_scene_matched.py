"""Matched frozen-feature video RAE/flow pilot. No pretrained-generation claim.

Both arms train from scratch on the same sealed RGB/features. Generation uses
only the observed prefix; full future features are training/scoring targets.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from models.matched_video_rae import MatchedVideoRAE
from models.scene_flow import SceneFlow
from train_scene_pipeline import (amp, any_rank, barrier, distributed, gather,
    memory_gib, metrics, preview, publish_metrics, recover_checkpoint, wrap)
from train_scene_prefix import labelled_preview
from utils.scene_prefix import PrefixFlowNet, conservative_lr, prepare_prefix
from utils.scene_prefix_eval import sample_case
from utils.scene_run import ArtifactStore, StageBudget, digest
from utils.training import EMA, capture_rng_state, restore_rng_state
from utils.window_flow import WindowFlow


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--arm', choices=['dino', 'stream'], required=True)
    p.add_argument('--data-local', required=True)
    p.add_argument('--data-root', action='append', default=[])
    p.add_argument('--output', required=True)
    p.add_argument('--root', action='append', default=[])
    p.add_argument('--stage', choices=['codec', 'cache', 'flow'], required=True)
    p.add_argument('--stop-after', type=int, default=0,
                   help='Absolute update limit for restart rehearsal; not part of recipe identity')
    return p


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def codec_args(cfg, dim):
    return dict(input_dim=dim, channels=cfg['channels'], grid=cfg['grid'],
                size=cfg['size'], **cfg.get('decoder', {}))


def source_hash():
    root = Path(__file__).parent
    files = [Path(__file__), root/'models/matched_video_rae.py',
             root/'models/scene_flow.py', root/'models/dual_stream_decoder.py',
             root/'models/r7_window_dit.py', root/'utils/window_flow.py',
             root/'utils/scene_prefix.py', root/'utils/scene_prefix_eval.py',
             root/'utils/scene_losses.py']
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def ordered_entry(entries, index, seed):
    """Epoch permutations shared by arms, independent of rank and constructor RNG."""
    if not entries:
        raise ValueError('No valid training windows')
    epoch, offset = divmod(index, len(entries))
    order = list(range(len(entries)))
    random.Random(seed+epoch).shuffle(order)
    return entries[order[offset]]


class ImmutableBank(ArtifactStore):
    """Reuse the shared node cache and receipts of an identity-pinned bank.

    Unlike a mutable latest checkpoint, these payload names never change after
    preparation. Query both durable mirrors once per receipt, not per update.
    ArtifactStore still verifies local size/hash and repairs damaged payloads.
    """
    def __init__(self, local, roots):
        super().__init__(local, roots)
        self.receipts = {}

    def json(self, name, required=False):
        if name.endswith('.pt.json') and name in self.receipts:
            return self.receipts[name]
        value = super().json(name, required)
        if name.endswith('.pt.json') and value is not None:
            self.receipts[name] = value
        return value


def prune_local_weights(store, stage, keep=2):
    """Only evict our older weight payloads after durable publication succeeds.

    Receipts and remote checkpoints remain; latest full resume and selected
    codec are never candidates. A local-only test/run keeps all its payloads.
    """
    if not store.roots:
        return
    base = (store.local/stage).resolve()
    candidates = sorted(base.glob('weights_step*.pt'))
    for path in candidates[:-keep]:
        if path.resolve().parent != base:
            raise ValueError('Checkpoint eviction escaped its stage directory')
        path.unlink(missing_ok=True)


def synchronized_load(store, entry, identity, device):
    value = None; error = None
    try:
        value = store.load(entry['file'], identity, True)
    except Exception as exc:
        error = repr(exc)
    if any_rank(error is not None, device):
        raise RuntimeError('Committed bank read failed on a rank; retry mirrored payload: '+str(error))
    return value


def load_codec(store, cfg, dim, identity, device):
    saved = store.load('codec/selected.pt', identity, True)
    model = MatchedVideoRAE(**codec_args(cfg, dim)).to(device)
    model.load_state_dict(saved['model'], strict=True)
    return model.eval().requires_grad_(False)


def codec_objective(output, raw, cfg, perceptual=None):
    rgb, z, restored, target = output
    truth = raw.float().permute(0, 1, 3, 4, 2)/255.
    l1 = (rgb.float()-truth).abs().mean()
    feature = (restored.float()-target.float()).square().mean()
    temporal = ((rgb[:, 1:]-rgb[:, :-1]).float()-(truth[:, 1:]-truth[:, :-1])).abs().mean()
    lp = l1.new_zeros(())
    if cfg.get('lpips_weight', .5):
        if perceptual is None:
            raise ValueError('LPIPS objective requires pre-staged offline weights')
        from utils.scene_losses import lpips_chunked
        lp = lpips_chunked(perceptual, rgb.float(), truth, resize=0)
    loss = l1 + cfg.get('feature_weight', .1)*feature + cfg.get('temporal_weight', .1)*temporal + cfg.get('lpips_weight', .5)*lp
    return loss, dict(rgb_l1=float(l1.detach()), feature_mse=float(feature.detach()),
                      temporal_l1=float(temporal.detach()), lpips=float(lp.detach()))


def aggregate_rows(rows, fields):
    result = {}
    for row in rows:
        key = row['split']+'/'+row.get('weights','online')
        result.setdefault(key,[]).append(row)
    return {key:dict(count=len(group),mean={field:sum(row[field] for row in group)/len(group)
                 for field in fields}) for key,group in result.items()}


def eval_codec(model, bank_store, bank, store, cfg, arm, step, device, rank, world, full=False):
    rng = capture_rng_state(); training = model.training; rows = []
    model.eval()
    try:
        for split in ('train_eval', 'dev'):
            entries = bank[split] if full else bank[split][:cfg.get('monitor_cases', 8)]
            for i in range(rank, len(entries), world):
                entry = entries[i]
                with torch.no_grad(), amp(device):
                    sample = bank_store.load(entry['file'], bank['identity'], True)
                    rgb = model(sample[arm].unsqueeze(0).to(device))[0][0].float()
                raw = sample['raw'].to(device).float().permute(0, 2, 3, 1)/255.
                error = (rgb-raw).square().mean(dim=(1,2,3)).clamp_min(1e-10)
                row = dict(id=sample['id'], split=split, step=step,
                           psnr=float((-10*error.log10()).mean()), l1=float((rgb-raw).abs().mean()))
                directory = f'codec/eval/step{step:07d}/{split}/case{i:04d}'
                names = preview(store, directory+'/preview.mp4',
                    labelled_preview(torch.cat((raw,rgb), 2), ['RAW','AE'], cfg['history']))
                store.publish(names)
                row['preview_status'] = 'completed' if names and names[0].endswith('.mp4') else 'failed'
                store.write_json(directory+'/metrics.json', row); rows.append(row)
        all_rows = [r for part in gather(rows, world) for r in part]
        if rank == 0:
            store.write_json(f'codec/eval/step{step:07d}/summary.json', dict(rows=all_rows,
                full=full, status='completed' if all_rows and all(r['preview_status']=='completed' for r in all_rows) else 'partial_media',
                groups=aggregate_rows(all_rows,['psnr','l1']),
                quality_policy='report_only', source='online; matched final-step codec is used by flow'))
    finally:
        model.train(training); restore_rng_state(rng)
    barrier()


def sample_eval(model, ema, codec, latent_store, manifest, store, cfg, step, device, rank, world, full=False):
    rng = capture_rng_state(); training = model.training
    model.eval(); rows = []
    online = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    try:
        for weights in ('online', 'ema'):
            model.load_state_dict(online if weights=='online' else ema.state_dict(), strict=True)
            for split in ('train_eval', 'dev'):
                entries = manifest[split] if full else manifest[split][:cfg.get('monitor_cases', 8)]
                for i in range(rank, len(entries), world):
                    sample = latent_store.load(entries[i]['file'], manifest['identity'], True)
                    for seed in cfg['eval_seeds'] if full else cfg['eval_seeds'][:1]:
                        start = time.monotonic()
                        row, video, z = sample_case(model, codec, sample, manifest['mean'], manifest['std'],
                            device, cfg['history'], cfg['sample_steps'], seed)
                        # These learned channels have no geometry/texture decomposition.
                        first = row.pop('future_geo_mse'); second = row.pop('future_texture_mse')
                        row.update(normalized_latent_mse=(first+second)/2,
                            split=split, weights=weights, step=step, arm=cfg['arm'],
                            metric_note=row['metric_note']+' Latent channels are unified; no geo/texture labels.')
                        directory = f'flow/eval/step{step:07d}/{split}/{weights}/case{i:04d}_seed{seed}'
                        store.save(directory+'/generated.pt', dict(target=z, id=sample['id'],
                            frame_ids=sample['frame_ids'], seed=seed), manifest['identity'])
                        names = preview(store, directory+'/preview.mp4',
                            labelled_preview(video, row['columns'], cfg['history']))
                        store.publish(names)
                        row['preview_status'] = 'completed' if names and names[0].endswith('.mp4') else 'failed'
                        row.update(DI_throughput=cfg['sample_steps']*(cfg['views']-cfg['history'])*cfg['grid']**2/max(time.monotonic()-start,1e-6),
                                   DI_unit='future_token_steps/s/NPU including sample/decode/publication')
                        store.write_json(directory+'/metrics.json', row); rows.append(row)
                        print('[matched sample] '+json.dumps(row), flush=True)
        all_rows = [r for part in gather(rows, world) for r in part]
        if rank == 0:
            store.write_json(f'flow/eval/step{step:07d}/summary.json',dict(rows=all_rows, full=full,
                status='completed' if all_rows and all(r['preview_status']=='completed' for r in all_rows) else 'partial_media',
                groups=aggregate_rows(all_rows,['raw_l1','ae_l1','copy_l1','normalized_latent_mse',
                    'generated_motion_l1','gt_motion_l1','generated_laplacian_abs','gt_laplacian_abs']),
                condition='observed prefix only; future starts from pure Gaussian noise',
                split_note='train_eval intentionally seen; dev is historical held-out development, not a new final test'))
    finally:
        model.load_state_dict(online); model.train(training); restore_rng_state(rng)
    barrier()


@torch.no_grad()
def cache_latents(codec, data, bank, store, cfg, identity, device, rank, world):
    codec_receipt = store.json('codec/selected.pt.json', True)
    cache_id = digest(dict(run=identity, codec=codec_receipt['sha256']))
    entries = []; sums = torch.zeros(cfg['channels'], dtype=torch.float64)
    squares = sums.clone(); count = 0
    start = time.monotonic(); frames = 0
    for split in ('train','dev'):
        for i in range(rank, len(bank[split]), world):
            entry = bank[split][i]; name = f'cache/{split}/{i:06d}.pt'
            payload = store.load(name, cache_id)
            if payload is None:
                source = data.load(entry['file'], bank['identity'], True)
                with amp(device):
                    z = codec.encode(source[cfg['arm']].unsqueeze(0).to(device))[0]
                if not torch.isfinite(z).all():
                    raise RuntimeError('Nonfinite codec latent')
                payload = {k:source[k] for k in ('id','dataset','frame_ids','raw')}
                for key in ('nominal_fps','source_fps','fps','time_status'):
                    if key in source: payload[key] = source[key]
                payload.update(target=z.float().cpu(), anchor=z[:1].float().cpu())
                store.save(name, payload, cache_id)
            if split=='train':
                flat = payload['target'].double().flatten(0,-2)
                sums += flat.sum(0); squares += flat.square().sum(0); count += len(flat)
            entries.append(dict(file=name, id=entry['id'], source_file=entry['file'], split=split, index=i))
            frames += cfg['views']
            metrics(store,dict(stage='cache',cursor=len(entries),DI_throughput=frames*cfg['grid']**2/max(time.monotonic()-start,1e-6),
                scope='encoded latent tokens/s/NPU including reads and dual publication'),rank)
    parts = gather(dict(entries=entries,sum=sums.tolist(),squares=squares.tolist(),count=count),world)
    if rank==0:
        total = sum(p['count'] for p in parts)
        mean = sum(torch.tensor(p['sum'],dtype=torch.float64) for p in parts)/total
        var = sum(torch.tensor(p['squares'],dtype=torch.float64) for p in parts)/total-mean.square()
        result = dict(identity=cache_id,run_identity=identity,codec_sha256=codec_receipt['sha256'],
            mean=mean.tolist(),std=var.clamp_min(1e-8).sqrt().tolist(),statistics='train only, float64 moments')
        all_entries = [e for p in parts for e in p['entries']]
        for split in ('train','dev'):
            result[split]=sorted([e for e in all_entries if e['split']==split],key=lambda e:e['index'])
        selected={e['file'] for e in bank['train_eval']}
        result['train_eval']=[e for e in result['train'] if e['source_file'] in selected]
        store.write_json('cache/complete.json',result)
    publish_metrics(store,rank,'cache'); barrier()


def run_training(a, cfg, bank, data, store, identity, device, rank, world):
    stage = a.stage; spec = cfg['codec' if stage=='codec' else 'flow_training']
    seed_all(cfg['seed'])
    codec = None; manifest = None; perceptual = None
    if stage=='codec':
        model = MatchedVideoRAE(**codec_args(cfg,bank['feature_dims'][a.arm])).to(device)
        entries = bank['train']; inputs = data; input_id = bank['identity']
        if spec.get('lpips_weight',.5):
            from utils.scene_losses import get_lpips
            perceptual = get_lpips(device)
    else:
        codec = load_codec(store,cfg,bank['feature_dims'][a.arm],identity,device)
        manifest = store.json('cache/complete.json',True)
        if manifest['run_identity']!=identity or manifest['codec_sha256']!=store.json('codec/selected.pt.json',True)['sha256']:
            raise ValueError('Latents do not match selected codec')
        seed_all(cfg['seed'])  # Exactly shared DiT initialization, independent of codec construction.
        model = SceneFlow(channels=cfg['channels'],grid=cfg['grid'],**cfg['flow']).to(device)
        entries=manifest['train']; inputs=ImmutableBank(store.local,store.roots); input_id=manifest['identity']
    saved = recover_checkpoint(store,f'{stage}/checkpoint_latest.pt',identity,rank)
    if saved: model.load_state_dict(saved['model'],strict=True)
    core = wrap(model,device,world)
    optimizer = torch.optim.AdamW(model.parameters(),lr=spec['lr'],betas=(.9,.95),weight_decay=.01)
    ema = EMA(model,decay=.9995,dtype=torch.float32,warmup=True)
    step = cursor = 0; spent = 0.
    if saved:
        if saved['world']!=world: raise ValueError('Exact resume requires same world size')
        optimizer.load_state_dict(saved['optimizer']); ema.load_state_dict(saved['ema'])
        ema.load_metadata(saved['ema_metadata']); ema.to(device)
        step=saved['step']; cursor=saved['cursors'][rank]; spent=saved['spent']
        restore_rng_state(saved['rng'][rank])
    else:
        seed_all(cfg['seed']+10000+rank)
    resumed_step = step
    del saved
    if a.stop_after and step>=a.stop_after:
        # A repeated launcher must not run completed-stage evaluations during
        # its two-step restart rehearsal.
        return 0
    budget=StageBudget(float('inf'),spent); budget.install_signals()
    start=time.monotonic(); tokens=0; compute_seconds=0.
    if rank==0:
        store.write_json(f'{stage}/model.json',dict(parameters=sum(p.numel() for p in model.parameters()),
            decoder_parameters=sum(p.numel() for p in model.decoder.parameters()) if stage=='codec' else None,
            effective_batch=world*spec['accum'],world=world,model_args=model.specification() if stage=='codec' else model.args,
            initialization='scratch; paired seed',feature_source=a.arm))

    def save(final=False):
        cursors=gather(cursor,world); rng=gather(capture_rng_state(),world)
        if rank==0:
            payload=dict(identity=identity,stage=stage,world=world,model=model.state_dict(),
                model_args=model.specification() if stage=='codec' else model.args,
                optimizer=optimizer.state_dict(),ema=ema.state_dict(),ema_metadata=ema.metadata(),
                step=step,cursors=cursors,rng=rng,spent=budget.elapsed())
            store.save(f'{stage}/checkpoint_latest.pt',payload,identity)
            store.save(f'{stage}/weights_step{step:07d}.pt',
                {k:v for k,v in payload.items() if k not in ('optimizer','rng','cursors')},identity)
            if final and stage=='codec':
                store.save('codec/selected.pt',dict(model=model.state_dict(),step=step,
                    selection='fixed final online update; same budget across arms'),identity)
            prune_local_weights(store,stage,cfg.get('keep_local_weights',2))
        publish_metrics(store,rank,stage); barrier()

    limit=min(spec['steps'],a.stop_after) if a.stop_after else spec['steps']
    flow=WindowFlow(time_shift=max(1.,math.sqrt(cfg['grid']**2*cfg['channels']/4096)))
    while step<limit:
        if any_rank(budget.interrupted,device):
            save(); return 75
        update_start=time.monotonic()
        optimizer.zero_grad(set_to_none=True); total=0.; extra={}
        for micro in range(spec['accum']):
            entry=ordered_entry(entries,rank+world*cursor,cfg['seed']); cursor+=1
            sample=synchronized_load(inputs,entry,input_id,device)
            sync=core.no_sync() if world>1 and micro+1<spec['accum'] else contextlib.nullcontext()
            with sync,amp(device):
                if stage=='codec':
                    out=core(sample[a.arm].unsqueeze(0).to(device))
                    loss,extra=codec_objective(out,sample['raw'].unsqueeze(0).to(device),spec,perceptual)
                    token_count=cfg['views']*cfg['grid']**2
                else:
                    part=prepare_prefix(sample,manifest['mean'],manifest['std'],cfg['history'],device=device)
                    u=flow.times(1,device); noise=torch.randn_like(part.target)
                    noisy=(1-u[:,None,None,None])*part.target+u[:,None,None,None]*noise
                    pred=PrefixFlowNet(core,part.known,model_frames=cfg['views']-1)(noisy,u,part.anchor)
                    loss=flow.loss(pred,part.target,noise,u)
                    extra=dict(unweighted_x0_mse=float((pred.detach()-part.target).square().mean()),noise_u=float(u))
                    token_count=(cfg['views']-cfg['history'])*cfg['grid']**2
                if any_rank(not torch.isfinite(loss).item(),device): raise RuntimeError('Nonfinite loss')
                (loss/spec['accum']).backward()
            total+=float(loss.detach())/spec['accum']; tokens+=token_count
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if any_rank(not torch.isfinite(grad).item(),device): raise RuntimeError('Nonfinite gradient')
        lr=conservative_lr(step,spec['steps'],spec['lr'],spec.get('min_lr',spec['lr']/10),spec.get('warmup_steps',200))
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.step(); ema.update(model); step+=1
        compute_seconds+=time.monotonic()-update_start
        if step==1 or step%cfg['log_every']==0:
            metrics(store,dict(stage=stage,step=step,loss=total,lr=lr,pre_clip_grad_norm=float(grad),
                seconds_per_update_including_read=compute_seconds/max(step-resumed_step,1),
                estimated_remaining_training_hours=(spec['steps']-step)*compute_seconds/max(step-resumed_step,1)/3600.,
                eta_note='current process mean update/read time; excludes future evaluation and checkpoint publication',
                DI_throughput=tokens/max(time.monotonic()-start,1e-6),
                scope=('reconstructed' if stage=='codec' else 'generated future')+' latent tokens/s/NPU, including stage IO/eval/save; excludes idle guard',
                peak_allocated_gib=memory_gib(device),**extra),rank)
        if step%cfg['save_every_steps']==0: save()
        if step%cfg['eval_every_steps']==0 and step<spec['steps']:
            if stage=='codec': eval_codec(model,data,bank,store,cfg,a.arm,step,device,rank,world)
            else: sample_eval(model,ema,codec,store,manifest,store,cfg,step,device,rank,world)
    complete=step>=spec['steps']
    # Do not serialize an unchanged completed codec again: byte-level provenance
    # must stay stable for its immutable latent bank and the flow resume contract.
    selected_ok = stage!='codec' or store.json('codec/selected.pt.json') is not None
    if step!=resumed_step or resumed_step<spec['steps'] or not selected_ok:
        save(final=complete)
    if complete:
        if stage=='codec': eval_codec(model,data,bank,store,cfg,a.arm,step,device,rank,world,True)
        else: sample_eval(model,ema,codec,store,manifest,store,cfg,step,device,rank,world,True)
        if rank==0:
            evaluation=store.json(f'{stage}/eval/step{step:07d}/summary.json',True)
            store.write_json(f'{stage}/complete.json',dict(identity=identity,step=step,
                stop_reason='matched_step_budget_not_convergence',quality_policy='report_only',
                evaluation_status=evaluation['status'],spent=budget.elapsed()))
    elif rank==0:
        store.write_json(f'{stage}/preflight.json',dict(identity=identity,step=step,status='saved_for_fresh_process_resume'))
    return 0


def main():
    a=parser().parse_args(); cfg=json.loads(Path(a.config).read_text(encoding='utf8'))
    cfg['arm']=a.arm
    device,rank,world=distributed()
    store=ArtifactStore(a.output,a.root)
    # One shared immutable bank per node. Do not replicate all 512 feature
    # windows into eight rank-private caches on every training arm.
    data=ImmutableBank(a.data_local,a.data_root)
    bank=data.json('data/bank.json',True)
    if cfg['views']!=9 or cfg['history']!=4: raise ValueError('Pilot pins four observed and five future frames')
    for key in ('views','size','grid'):
        if bank[key]!=cfg[key]: raise ValueError('Paired bank shape differs from recipe: '+key)
    if {e['id'] for e in bank['train']} & {e['id'] for e in bank['dev']}:
        raise ValueError('Train/dev video identity overlap')
    plan=dict(schema='scene-matched-rae-v1',config=cfg,bank=bank['identity'],
              bank_manifest_sha256=digest(bank),code_sha256=source_hash(),world=world)
    identity=digest(plan); plan['identity']=identity
    previous=store.json('contract.json')
    if previous and previous['identity']!=identity: raise ValueError('Changed matched recipe; use a fresh namespace')
    if rank==0: store.write_json('contract.json',plan)
    barrier()
    try:
        if a.stage=='cache':
            codec=load_codec(store,cfg,bank['feature_dims'][a.arm],identity,device)
            cache_latents(codec,data,bank,store,cfg,identity,device,rank,world)
            code=0
        else:
            code=run_training(a,cfg,bank,data,store,identity,device,rank,world)
    finally:
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
    raise SystemExit(code)


if __name__=='__main__': main()
