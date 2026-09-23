"""Three-node RGB representation -> frozen cache -> image/video flow stages.

Each invocation runs one resumable stage. The shell launcher runs all stages,
including a separate short end-to-end rehearsal first. Quality is report-only.
"""
import argparse
from collections import Counter, OrderedDict
import contextlib
from datetime import timedelta
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
from torch.utils.data import DataLoader
from data.scene_rgb import SceneRGB, CursorSampler, singleton, seed_for
from data.token_utils import strip_special_tokens
from models.scene_rae import SceneRAE, reconstruction_loss, relation_loss
from models.scene_flow import SceneFlow
from scripts.window_run_io import child, atomic
from utils.scene_run import ArtifactStore, StageBudget, digest
from utils.training import EMA, capture_rng_state, restore_rng_state, append_metrics
from utils.device import get_device, configure_backend_compatibility
from utils.file_signature import sampled_file_signature
from utils.window_flow import WindowFlow

WRITER = None


def distributed():
    device = get_device(int(os.environ.get('LOCAL_RANK', 0)))
    if device.type != 'cpu':
        getattr(torch, device.type).set_device(device)
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        backend = dict(npu='hccl', cuda='nccl', cpu='gloo')[device.type]
        dist.init_process_group(backend, timeout=timedelta(seconds=int(os.environ.get('HCCL_EXEC_TIMEOUT','7200'))))
    configure_backend_compatibility(device.type)
    return device, (dist.get_rank() if dist.is_initialized() else 0), (dist.get_world_size() if dist.is_initialized() else 1)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def any_rank(value, device):
    flag = torch.tensor(int(value), device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def gather(value, world):
    if world == 1:
        return [value]
    out = [None]*world
    dist.all_gather_object(out, value)
    return out


def recover_checkpoint(store,name,identity,rank):
    # Republish a leader-local commit before other nodes choose their receipt.
    # This prevents mixing new local optimizer state with an older remote copy
    # after a job stops during publication. Receipt reads choose newest commit.
    if rank==0:
        value=store.load(name,identity)
        if value is not None:
            store.publish([name,name+'.json'])
        del value
    barrier()
    return store.load(name,identity)


def amp(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type != 'cpu')


def wrap(model, device, world):
    if world == 1:
        return model
    return torch.nn.parallel.DistributedDataParallel(model,
        device_ids=[device.index] if device.type != 'cpu' else None,
        find_unused_parameters=True, broadcast_buffers=False)


def make_encoder(path, device):
    from streamvggt.models.streamvggt import StreamVGGT
    from utils.encoder_loader import load_encoder_checkpoint
    encoder = StreamVGGT()
    load_encoder_checkpoint(encoder, path, verbose=False)
    encoder.eval().requires_grad_(False).to(device)
    return encoder


@torch.no_grad()
def features(encoder, frames, device):
    with amp(device):
        values, start = encoder(frames)
    return [x.detach() for x in strip_special_tokens(values, start)]


def loader(data, rank, world, cursor, workers):
    kwargs = dict(num_workers=workers, batch_size=1, collate_fn=singleton,
                  sampler=CursorSampler(len(data), rank, world, cursor))
    if workers:
        kwargs.update(multiprocessing_context='spawn', persistent_workers=True,
                      prefetch_factor=2, timeout=300)
    return DataLoader(data, **kwargs)


def selected_val(cohort, per_domain):
    records, counts = [], Counter()
    for r in sorted(cohort['records'], key=lambda x: seed_for('scene-val', x['id'])):
        if r['split'] == 'val' and counts[r['dataset']] < per_domain:
            records.append(r)
            counts[r['dataset']] += 1
    return dict(cohort, records=records)


def video(path, frames, fps=8):
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (frames.detach().cpu().float().clamp(0, 1).numpy()*255).round().astype(np.uint8)
    imageio.mimwrite(path, data, fps=fps, macro_block_size=1)


def preview(store, name, frames):
    # Rendering is an output convenience; a broken codec must not cancel a
    # valid optimizer run. Tensor metrics and a visible error remain durable.
    try:
        video(store.local/name, frames)
        return [name]
    except (OSError, ValueError, RuntimeError) as exc:
        error=name+'.error.json'
        atomic(store.local/error,dict(error=repr(exc),shape=list(frames.shape)))
        print('[preview warning] '+repr(exc),flush=True)
        return [error]


def metrics(store, row, rank=0):
    global WRITER
    row = dict(row, rank=rank, unix_time=time.time())
    name = f'metrics-r{rank:03d}.jsonl'
    append_metrics(str(store.local/name), row)
    if rank == 0:
        if WRITER is None:
            from torch.utils.tensorboard import SummaryWriter
            WRITER = SummaryWriter(str(store.local/'tb'/row.get('stage','run')))
        for key,value in row.items():
            if isinstance(value,(int,float)) and math.isfinite(float(value)):
                WRITER.add_scalar(key,value,row.get('step',row.get('cursor',0)))
    print(json.dumps(row, ensure_ascii=False), flush=True)
    if 'DI_throughput' in row:
        print(f'DI_throughput: {row["DI_throughput"]:.2f} tokens/s/npu '
              f'(rank={rank}, {row.get("scope", "stage wall time excluding channels")})', flush=True)


def memory_gib(device):
    if device.type == 'cpu':
        return 0.
    return getattr(torch, device.type).max_memory_allocated(device)/1024**3


def publish_metrics(store,rank,stage):
    names=[f'metrics-r{rank:03d}.jsonl',f'{stage}/bad-r{rank:03d}.jsonl']
    if rank==0 and WRITER is not None:
        WRITER.flush()
        names += [p.relative_to(store.local).as_posix() for p in (store.local/'tb').rglob('*') if p.is_file()]
    names=[n for n in names if (store.local/n).exists()]
    if names:
        store.publish(names)


def prune_local_weights(store,stage,keep=8):
    # Remote periodic snapshots are permanent. Bound node-local working storage;
    # never delete selected/best/latest/final checkpoints or any remote object.
    status=store.json(store.status_name)
    if not store.roots or status.get('errors'):
        return
    for file in sorted((store.local/stage).glob('weights_step*.pt'))[:-keep]:
        file.unlink()
        file.with_name(file.name+'.json').unlink(missing_ok=True)


def read_rae(store, device):
    complete = store.json('ae/complete.json', required=True)
    payload = store.load(complete['selected'], complete['identity'], required=True)
    model = SceneRAE(payload['spec']['source_config'], payload['spec']['channels'])
    model.load_state_dict(payload['model'], strict=True)
    return model.eval().requires_grad_(False).to(device), complete


@torch.no_grad()
def eval_ae(model, encoder, data, store, step, device, old=None):
    model.eval()
    rows, files = [], []
    for index in range(len(data)):
        sample = data[index]
        print(f'[AE validation] step={step} case={index+1}/{len(data)} id={sample["id"]}', flush=True)
        if sample['error']:
            rows.append(dict(id=sample['id'], error=sample['error']))
            continue
        frames = sample['frames'].unsqueeze(0).to(device)
        tokens = features(encoder, frames, device)
        single = features(encoder, frames[:, :1], device)
        with amp(device):
            z = model.encode(tokens, frames)
            image_z=model.encode(single,frames[:, :1])
            z = torch.cat((image_z,z[:,1:]),1)
            rgb = model.decode(z).float()
            rgb_image=model.decode(image_z).float()
            target=torch.cat((model.feature_targets(single),model.feature_targets(tokens)[:,1:]),1)
            feature_mse=(model.restore_features(z).float()-target).square().mean().item()
            gram=relation_loss(z,target).item()
        truth = frames.permute(0,1,3,4,2)
        mse = (rgb-truth).square().mean().item()
        row = dict(id=sample['id'], dataset=sample['dataset'], psnr=-10*math.log10(max(mse,1e-12)),
                   l1=(rgb-truth).abs().mean().item(),feature_mse=feature_mse,relation_mse=gram,
                   image_psnr=-10*math.log10(max((rgb_image-truth[:,:1]).square().mean().item(),1e-12)))
        row['joint_score']=row['l1']+.1*feature_mse+.1*gram
        if old is not None:
            comp, tex, tokenizer, decoder = old
            with amp(device):
                geo = comp(tokens).permute(0,1,3,4,2)
                appearance = tex(frames)
                old_z = tokenizer.encode(geo, appearance)
                old_rgb = decoder(*tokenizer.decode(old_z)).float()
            row['old_r7_psnr'] = -10*math.log10(max((old_rgb-truth).square().mean().item(),1e-12))
            row['paired_psnr_delta'] = row['psnr']-row['old_r7_psnr']
        elif step:
            baseline=store.json('ae/eval/step0000000.json')
            reference={r['id']:r for r in baseline['samples']} if baseline else {}
            if 'old_r7_psnr' in reference.get(sample['id'],{}):
                row['paired_psnr_delta']=row['psnr']-reference[sample['id']]['old_r7_psnr']
        name = f'ae/samples/step{step:07d}_{index:02d}.mp4'
        files.extend(preview(store,name,torch.cat((truth[0], rgb[0]), dim=2)))
        rows.append(row)
    domains, joint_scores = {}, {}
    for source in sorted({r.get('dataset') for r in rows if 'psnr' in r}):
        chosen = [r for r in rows if r.get('dataset') == source]
        domains[source] = sum(r['psnr'] for r in chosen)/len(chosen)
        joint_scores[source]=sum(r['joint_score'] for r in chosen)/len(chosen)
    result = dict(step=step, domains=domains, samples=rows, quality_policy='report_only',
                  mean_domain_psnr=sum(domains.values())/max(1,len(domains)),
                  mean_joint_score=sum(joint_scores.values())/max(1,len(joint_scores)))
    name = f'ae/eval/step{step:07d}.json'
    atomic(store.local/name, result)
    store.publish([*files, name])
    model.train()
    return result


def train_ae(a, cfg, cohort, store, identity, device, rank, world):
    from utils.r7_representation import load_checkpoint, load_r7_modules
    from utils.window_codec import configure_codec
    saved = recover_checkpoint(store,'ae/checkpoint_latest.pt',identity,rank)
    if saved:
        model = SceneRAE(saved['spec']['source_config'], saved['spec']['channels'])
        model.load_state_dict(saved['model'])
    else:
        source = load_checkpoint(a.r7)
        model = SceneRAE.from_r7(source, cfg['channels'])
        del source
    if model.config.target_size != cfg['size']:
        raise ValueError('RGB size must match warm-start decoder; use a new architecture for another size')
    model.to(device).train()
    encoder = make_encoder(a.encoder, device)
    from utils.scene_losses import get_lpips
    perceptual = get_lpips(device)  # Required objective asset, checked during rehearsal.
    front, rest = [], []
    for name, parameter in model.named_parameters():
        (front if name.startswith(('compressor.', 'tex_encoder.')) else rest).append(parameter)
    optimizer = torch.optim.AdamW([dict(params=front, lr=cfg['lr_front']),
        dict(params=rest, lr=cfg['lr_ae'])], betas=(.9,.95), weight_decay=.01)
    step, cursor, spent, best, best_joint = 0, 0, 0., -float('inf'), float('inf')
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        step, cursor, spent, best = saved['step'], saved['cursors'][rank], saved['spent'], saved['best']
        best_joint=saved['best_joint']
        restore_rng_state(saved['rng'][rank])
    budget = StageBudget(cfg['ae_hours']*3600, spent)
    budget.install_signals()
    core = wrap(model, device, world)
    data = SceneRGB(cohort, weights=cfg['source_weights'], size=cfg['size'],views=cfg['views'],grid=cfg['grid'])
    val = SceneRGB(selected_val(cohort, cfg['eval_per_domain']), split='val', finite=True, windows=1,
                   size=cfg['size'],views=cfg['views'],grid=cfg['grid'])
    iterator = iter(loader(data, rank, world, cursor, cfg['workers']))
    last_save = last_eval = time.monotonic()
    tokens_count, start, exposure = 0, time.monotonic(), Counter()

    def save(final=False):
        cursors = gather(cursor, world)
        rng = gather(capture_rng_state(), world)
        if rank == 0:
            payload = dict(model=model.state_dict(), spec=model.specification(), optimizer=optimizer.state_dict(),
                step=step, cursors=cursors, rng=rng, spent=budget.elapsed(), best=best,best_joint=best_joint,
                identity=identity, world=world, schedule='wall-budget cosine with optimizer-step warmup')
            store.save('ae/checkpoint_latest.pt', payload, identity)
            if final:
                store.save('ae/checkpoint_final.pt', payload, identity)
            else:
                store.save(f'ae/weights_step{step:07d}.pt', dict(model=model.state_dict(),spec=model.specification(),step=step), identity)
                prune_local_weights(store,'ae')
        publish_metrics(store,rank,'ae')
        barrier()

    if not saved:
        if rank == 0:
            _, comp, tex, tokenizer, decoder, _ = load_r7_modules(load_checkpoint(a.r7))
            configure_codec(tokenizer, 'legacy')
            old = [m.eval().requires_grad_(False).to(device) for m in (comp,tex,tokenizer,decoder)]
            baseline = eval_ae(model, encoder, val, store, step, device, old)
            best = baseline['mean_domain_psnr'] if baseline['domains'] else -float('inf')
            best_joint=baseline['mean_joint_score'] if baseline['domains'] else float('inf')
            store.save('ae/best_reconstruction.pt', dict(model=model.state_dict(), spec=model.specification(),step=step), identity)
            store.save('ae/best_joint.pt', dict(model=model.state_dict(), spec=model.specification(),step=step), identity)
            del old, comp, tex, tokenizer, decoder
            gc.collect()
        barrier()
        save()
    while step < cfg['ae_steps'] and not any_rank(budget.due(), device):
        optimizer.zero_grad(set_to_none=True)
        losses_total = 0.
        component_values = Counter()
        for micro in range(cfg['ae_accum']):
            for attempt in range(128):
                sample = next(iterator)
                cursor += 1
                if sample['error']:
                    append_metrics(str(store.local/f'ae/bad-r{rank:03d}.jsonl'), sample)
                if not any_rank(bool(sample['error']), device):
                    break
            else:
                raise RuntimeError('No computable synchronized RGB batch after 128 attempts; check data access')
            frames = sample['frames'].unsqueeze(0).to(device)
            if step % cfg['image_every'] == 0:
                frames = frames[:, :1]
            teacher = features(encoder, frames, device)
            anchor_teacher = features(encoder,frames[:,:1],device) if frames.shape[1]>1 else None
            warm = min(1., (step+1)/cfg['warmup_steps'])
            fraction = min(1., budget.elapsed()/budget.seconds)
            jitter = .05*min(1., max(0., (fraction-.65)/.1)) if (step+micro)%2 else 0.
            ctx = core.no_sync() if world>1 and micro+1<cfg['ae_accum'] else contextlib.nullcontext()
            with ctx, amp(device):
                rgb, z, restored, target = core(teacher, frames, jitter=jitter, anchor_tokens=anchor_teacher)
                loss, components = reconstruction_loss(rgb, frames, z, restored, target,
                    perceptual, sample['temporal_valid'], relation_weight=.1*warm)
                if any_rank(not torch.isfinite(loss).item(), device):
                    raise RuntimeError('Nonfinite AE objective; last committed checkpoint remains resumable')
                (loss/cfg['ae_accum']).backward()
            losses_total += loss.item()/cfg['ae_accum']
            for key,value in components.items():
                component_values[key] += value.item()/cfg['ae_accum']
            tokens_count += z.shape[0]*z.shape[1]*z.shape[2]*z.shape[3]
            exposure[sample['dataset']] += int(frames.shape[1])
            del teacher, anchor_teacher, rgb, z, restored, target, frames
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if any_rank(not torch.isfinite(norm).item(), device):
            raise RuntimeError('Nonfinite AE gradients')
        for group, lr in zip(optimizer.param_groups, (cfg['lr_front'],cfg['lr_ae'])):
            group['lr'] = lr*warm*(.05+.95*(1+math.cos(math.pi*fraction))/2)
        optimizer.step()
        step += 1
        if a.smoke and step==1:
            save()
            recovered=store.load('ae/checkpoint_latest.pt',identity,required=True)
            model.load_state_dict(recovered['model']); optimizer.load_state_dict(recovered['optimizer'])
            restore_rng_state(recovered['rng'][rank])
            if rank==0:
                store.write_json('ae/resume_rehearsal.json',dict(step=step,optimizer_restored=True,cursors=recovered['cursors']))
            del recovered
        if step % cfg['log_every'] == 0:
            metrics(store, dict(stage='ae', step=step, loss=losses_total, source_frames=dict(exposure),
                loss_components=dict(component_values), jitter_sigma=jitter,
                DI_throughput=tokens_count/max(time.monotonic()-start,1e-6),
                scope='AE output grid tokens, per active NPU, wall time', elapsed_hours=budget.elapsed()/3600,
                peak_allocated_gib=memory_gib(device)), rank)
        evaluate = any_rank(time.monotonic()-last_eval >= cfg['eval_seconds'], device)
        checkpoint_due = any_rank(time.monotonic()-last_save >= cfg['save_seconds'], device)
        if evaluate:
            if rank == 0:
                result = eval_ae(model, encoder, val, store, step, device)
                if result['domains'] and result['mean_domain_psnr'] > best:
                    best = result['mean_domain_psnr']
                    store.save('ae/best_reconstruction.pt', dict(model=model.state_dict(), spec=model.specification(),step=step), identity)
                if result['domains'] and result['mean_joint_score'] < best_joint:
                    best_joint=result['mean_joint_score']
                    store.save('ae/best_joint.pt',dict(model=model.state_dict(),spec=model.specification(),step=step),identity)
            barrier()
            last_eval = time.monotonic()
        if checkpoint_due or evaluate:
            save()
            last_save = time.monotonic()
    if rank == 0:
        result = eval_ae(model, encoder, val, store, step, device)
        if result['domains'] and result['mean_domain_psnr'] > best:
            best = result['mean_domain_psnr']
            store.save('ae/best_reconstruction.pt', dict(model=model.state_dict(),spec=model.specification(),step=step), identity)
        if result['domains'] and result['mean_joint_score'] < best_joint:
            best_joint=result['mean_joint_score']
            store.save('ae/best_joint.pt',dict(model=model.state_dict(),spec=model.specification(),step=step),identity)
    barrier()
    save(final=True)
    interrupted = any_rank(budget.interrupted, device)
    if rank == 0 and not interrupted:
        selected='ae/best_joint.pt' if math.isfinite(best_joint) else 'ae/checkpoint_final.pt'
        store.write_json('ae/complete.json', dict(identity=identity, selected=selected,
            step=step, spent=budget.elapsed(), best_domain_psnr=best, best_joint=best_joint,quality_policy='report_only'))
    barrier()
    if interrupted:
        raise SystemExit(75)


class LatentSamples:
    def __init__(self, store, manifest, weights, seed=42):
        self.store, self.manifest = store, manifest
        self.seed = seed
        self.by_source, self.cache = {}, OrderedDict()
        for shard in manifest['shards']:
            for i, source in enumerate(shard['sources']):
                self.by_source.setdefault(source, []).append((shard['file'], i))
        self.sources = sorted(self.by_source)
        self.weights = [weights.get(s,.05) for s in self.sources]
        if not self.sources:
            raise ValueError('Latent cache contains no computable samples')

    def get(self, cursor):
        rng = random.Random(seed_for(self.seed, cursor))
        source = rng.choices(self.sources, self.weights)[0]
        name, i = rng.choice(self.by_source[source])
        if name not in self.cache:
            self.cache[name] = self.store.load(name, self.manifest['identity'], required=True)['samples']
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        return self.cache[name][i]


def cache(a, cfg, cohort, store, identity, device, rank, world):
    model, ae_complete = read_rae(store, device)
    encoder = make_encoder(a.encoder, device)
    ae_receipt = store.json(ae_complete['selected']+'.json', required=True)
    cache_id = digest(dict(pipeline=identity, ae_sha=ae_receipt['sha256']))
    caption_bank = None
    if a.text:
        from utils.scene_text import SceneCaptionBank
        caption_ids=[r['id'].split(':',1)[1] for r in cohort['records']
                     if r['dataset']=='spatialvid' and r['split']!='test']
        caption_bank = SceneCaptionBank(a.text,caption_ids,store.local/'text_subset')
    saved = store.load(f'cache/state-r{rank:03d}.pt', cache_id)
    state = saved or dict(cursor=0, spent=0., shards=[], count=0,
        sum=torch.zeros(cfg['channels'],dtype=torch.float64), square=torch.zeros(cfg['channels'],dtype=torch.float64))
    budget = StageBudget(cfg['cache_hours']*3600, state['spent'])
    budget.install_signals()
    start, tokens, pending = time.monotonic(), 0, []

    def commit():
        nonlocal pending
        if pending:
            name = f'cache/train/r{rank:03d}-s{len(state["shards"]):06d}.pt'
            store.save(name, dict(samples=pending), cache_id)
            state['shards'].append(dict(file=name, sources=[x['dataset'] for x in pending], count=len(pending)))
            for item in pending:
                # Both single-frame and joint context appear in generation tasks.
                z = torch.cat((item['anchor'],item['target']),0).double().reshape(-1,cfg['channels'])
                state['sum'] += z.sum(0)
                state['square'] += z.square().sum(0)
                state['count'] += len(z)
            pending = []
        state['spent'] = budget.elapsed()
        store.save(f'cache/state-r{rank:03d}.pt', state, cache_id)
        metrics(store, dict(stage='cache', cursor=state['cursor'], shards=len(state['shards']),
            DI_throughput=tokens/max(time.monotonic()-start,1e-6), scope='encoded latent tokens including independent anchor; wall time'), rank)
        publish_metrics(store,rank,'cache')

    @torch.no_grad()
    def encode(sample, raw=False):
        frames = sample['frames'].unsqueeze(0).to(device)
        joint = features(encoder, frames, device)
        with amp(device):
            z = model.encode(joint, frames).flatten(2,3)[0]
        del joint
        single = features(encoder, frames[:, :1], device)
        with amp(device):
            anchor = model.encode(single, frames[:, :1]).flatten(2,3)[0]
        z = torch.cat((anchor,z[1:]),0)
        if not torch.isfinite(z).all() or not torch.isfinite(anchor).all():
            raise ValueError('Nonfinite encoder latents')
        text, text_valid = torch.zeros(1,4096), False
        if caption_bank is not None and sample['id'].startswith('spatialvid:'):
            key = sample['id'].split(':',1)[1]
            if key in caption_bank.values:
                text_valid = caption_bank.values[key] is not None
                if text_valid:
                    text = caption_bank.get(key)
        item = dict(anchor=anchor.cpu().bfloat16(), target=z.cpu().bfloat16(),
                    rays=sample['rays'], id=sample['id'], dataset=sample['dataset'],
                    frame_ids=sample['frame_ids'], text=text.cpu().bfloat16(), has_caption=text_valid)
        if raw:
            item['raw'] = (sample['frames']*255).round().to(torch.uint8)
        return item

    # Validation happens first, so the cache deadline cannot eliminate evaluation.
    if rank == 0:
        val_done = store.json('cache/eval.pt.json')
        if not val_done:
            val = SceneRGB(selected_val(cohort,cfg['eval_per_domain']),split='val',finite=True,windows=1,
                           size=cfg['size'],views=cfg['views'],grid=cfg['grid'])
            values, errors = [], []
            for i in range(len(val)):
                sample = val[i]
                print(f'[cache validation] case={i+1}/{len(val)} id={sample["id"]}', flush=True)
                if sample['error']:
                    errors.append(sample)
                    continue
                values.append(encode(sample,raw=True))
            store.save('cache/eval.pt', dict(samples=values,errors=errors), cache_id)
    barrier()
    data = SceneRGB(cohort, finite=True,windows=cfg['cache_windows'],size=cfg['size'],views=cfg['views'],grid=cfg['grid'])
    iterator = iter(loader(data,rank,world,state['cursor'],cfg['workers']))
    max_items = math.ceil(len(data)/world)
    while any_rank(state['cursor'] < max_items,device) and not any_rank(budget.due(), device):
        sample = next(iterator, None)
        if sample is not None:
            if sample['error']:
                append_metrics(str(store.local/f'cache/bad-r{rank:03d}.jsonl'), sample)
            else:
                item = encode(sample)
                pending.append(item)
                tokens += item['target'].shape[0]*model.grid**2 + model.grid**2
        state['cursor'] = min(max_items,state['cursor']+1)
        # All ranks commit on the same input-cursor interval, even with failures.
        if state['cursor'] % cfg['shard_samples'] == 0:
            commit()
        if a.smoke and state['cursor'] >= 2:
            break
    commit()
    barrier()
    interrupted = any_rank(budget.interrupted, device)
    if rank == 0 and not interrupted:
        states = [store.load(f'cache/state-r{r:03d}.pt', cache_id, required=True) for r in range(world)]
        count = sum(s['count'] for s in states)
        if not count:
            raise RuntimeError('All cache samples failed; no latent target exists')
        mean = sum(s['sum'] for s in states)/count
        std = (sum(s['square'] for s in states)/count-mean.square()).clamp_min(1e-8).sqrt()
        manifest = dict(identity=cache_id, pipeline=identity, ae=ae_receipt['sha256'],
            shards=[v for s in states for v in s['shards']], mean=mean.float().tolist(),
            std=std.float().tolist(), normalization='train per-channel; includes independent anchor',
            complete_source_pass=all(s['cursor']>=max_items for s in states), world=world)
        store.write_json('cache/complete.json', manifest)
    barrier()
    if interrupted:
        raise SystemExit(75)


@torch.no_grad()
def eval_flow(model, ema, rae, samples, mean, std, store, stage, step, device):
    # Copy only on rank0; other ranks wait outside. Restore train weights afterwards.
    previous = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    ema.copy_to(model)
    model.eval()
    rows, files = [], []
    for i, sample in enumerate(samples):
        print(f'[flow validation] stage={stage} step={step} case={i+1}/{len(samples)} id={sample["id"]}',flush=True)
        anchor = (sample['anchor'].unsqueeze(0).to(device).float()-mean)/std
        text = sample['text'].unsqueeze(0).to(device).float()
        valid = torch.ones(text.shape[:2], device=device, dtype=torch.bool)
        image_task = stage == 'image'
        frames = 1 if image_task else sample['target'].shape[0]-1
        rays = None if image_task else sample['rays'][1:].unsqueeze(0).to(device)
        present = torch.tensor([not image_task], device=device)
        def net(z,u,c,t,v):
            return model(z,u,c,t,v,ref_present=present,rays=rays)
        generator = torch.Generator(device=device).manual_seed(101+i)
        noise = torch.randn((1,frames,rae.grid**2,rae.channels),device=device,generator=generator)
        try:
            z = WindowFlow().sample(net, anchor, noise, steps=32,
                dtype=torch.bfloat16 if device.type!='cpu' else torch.float32,text=text,text_valid=valid)
        except RuntimeError as exc:
            if str(exc)!='nonfinite free-sampled latents':
                raise
            rows.append(dict(id=sample['id'],error=str(exc)))
            continue
        raw = sample['raw'].float().permute(0,2,3,1).to(device)/255
        latent = z*std+mean
        if not image_task:
            latent = torch.cat((sample['anchor'].unsqueeze(0).to(device).float(),latent),1)
        with amp(device):
            rgb = rae.decode(latent.reshape(1,-1,rae.grid,rae.grid,rae.channels))[0].float()
            clean = sample['anchor'] if image_task else sample['target']
            ae_rgb = rae.decode(clean.unsqueeze(0).to(device).float().reshape(1,-1,rae.grid,rae.grid,rae.channels))[0].float()
        if image_task:
            raw, ae_rgb = raw[:1], ae_rgb[:1]
        name = f'{stage}/samples/step{step:07d}_{i:02d}.mp4'
        files.extend(preview(store,name,torch.cat((raw,ae_rgb,rgb),2)))
        score=slice(None) if image_task else slice(1,None)
        row = dict(id=sample['id'], dataset=sample['dataset'],
            raw_l1=(rgb[score]-raw[score]).abs().mean().item(),
            ae_l1=(ae_rgb[score]-raw[score]).abs().mean().item(),
            copy_l1=(raw[:1].expand_as(raw)[score]-raw[score]).abs().mean().item(), seed=101+i,
            metric_scope='single generated image' if image_task else 'future frames only',
            has_caption=sample['has_caption'], camera_present=bool(rays is not None and rays[...,7].any()))
        rows.append(row)
    name = f'{stage}/eval/step{step:07d}.json'
    atomic(store.local/name, dict(samples=rows, step=step, quality_policy='report_only', cfg=1,
        columns=['RAW','AE','GENERATED'], note='RGB quality and temporal behavior require visual review'))
    store.publish([*files,name])
    model.load_state_dict(previous)
    model.train()
    return rows


def train_flow(a,cfg,cohort,store,identity,device,rank,world):
    stage = a.stage
    manifest = store.json('cache/complete.json',required=True)
    if manifest['pipeline'] != identity:
        raise ValueError('Frozen cache belongs to another pipeline identity')
    model_args = dict(cfg['flow'], channels=cfg['channels'],grid=cfg['grid'])
    model = SceneFlow(**model_args).to(device)
    flow_id = digest(dict(pipeline=identity,cache=manifest['identity'],stage=stage,model=model_args))
    saved = recover_checkpoint(store,stage+'/checkpoint_latest.pt',flow_id,rank)
    if saved:
        model.load_state_dict(saved['model'])
    elif stage == 'video':
        previous = store.json('image/complete.json',required=True)
        weights = store.load(previous['selected'],previous['identity'],required=True)
        model.load_state_dict(weights['model'])
    core = wrap(model,device,world)
    ema = EMA(model,decay=.9995,dtype=torch.bfloat16,warmup=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['lr_flow'],betas=(.9,.95),weight_decay=0.)
    step, cursor, spent = 0,0,0.
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        ema.load_state_dict(saved['ema']); ema.load_metadata(saved['ema_metadata']); ema.to(device)
        step,cursor,spent = saved['step'],saved['cursors'][rank],saved['spent']
        restore_rng_state(saved['rng'][rank])
    budget = StageBudget(cfg[stage+'_hours']*3600,spent)
    budget.install_signals()
    bank = LatentSamples(store,manifest,cfg['source_weights'],cfg['seed'])
    mean = torch.tensor(manifest['mean'],device=device)
    std = torch.tensor(manifest['std'],device=device).clamp_min(1e-4)
    # GAE release pins this from per-view dimensionality, not total clip length.
    shift = max(1.,math.sqrt(cfg['grid']**2*cfg['channels']/4096))
    flow = WindowFlow(time_shift=shift)
    eval_items = store.load('cache/eval.pt',manifest['identity'],required=True)['samples'] if rank==0 else None
    rae = read_rae(store,device)[0] if rank==0 else None
    last_eval = last_save = start = time.monotonic()
    tokens, exposures = 0,Counter()
    if rank==0:
        store.write_json(stage+'/model.json',dict(parameters=sum(p.numel() for p in model.parameters()),
            model=model_args,flow=flow.contract(), time_shift_basis='one view: grid^2*channels',
            effective_video_batch=world*cfg['flow_accum'],
            effective_image_batch=world*cfg['flow_accum']*cfg['image_batch'], identity=flow_id))

    def save(final=False):
        cursors=gather(cursor,world); rng=gather(capture_rng_state(),world)
        if rank==0:
            payload=dict(model=model.state_dict(),ema=ema.state_dict(),ema_metadata=ema.metadata(),
                optimizer=optimizer.state_dict(),step=step,cursors=cursors,rng=rng,spent=budget.elapsed(),
                identity=flow_id,model_args=model_args,world=world)
            store.save(stage+'/checkpoint_latest.pt',payload,flow_id)
            if final:
                store.save(stage+'/checkpoint_final.pt',payload,flow_id)
            else:
                store.save(f'{stage}/weights_step{step:07d}.pt',dict(model=model.state_dict(),
                    ema=ema.state_dict(),model_args=model_args,step=step),flow_id)
                prune_local_weights(store,stage)
        publish_metrics(store,rank,stage)
        barrier()

    while step<cfg[stage+'_steps'] and not any_rank(budget.due(),device):
        optimizer.zero_grad(set_to_none=True)
        losses=0.
        image_task = stage=='image' or step%cfg['image_every']==0
        for micro in range(cfg['flow_accum']):
            batch_size=cfg['image_batch'] if image_task else 1
            batch=[bank.get(rank+world*(cursor+j)) for j in range(batch_size)]
            cursor+=batch_size
            anchor=(torch.stack([s['anchor'] for s in batch]).to(device).float()-mean)/std
            target=torch.stack([s['anchor'] if image_task else s['target'][1:] for s in batch])
            target=(target.to(device).float()-mean)/std
            texts=[s['text'] if random.random()>=cfg['text_drop'] else torch.zeros(1,4096) for s in batch]
            text=torch.zeros(batch_size,max(len(t) for t in texts),4096,device=device)
            valid=torch.zeros(text.shape[:2],device=device,dtype=torch.bool)
            for j,t in enumerate(texts):
                text[j,:len(t)]=t.to(device); valid[j,:len(t)]=True
            present=torch.tensor([not image_task and random.random()>=cfg['ref_drop'] for _ in batch],device=device)
            rays=None if image_task else batch[0]['rays'][1:].unsqueeze(0).to(device)
            if rays is not None and random.random()<cfg['camera_drop']:
                rays=torch.zeros_like(rays)
            noise=torch.randn_like(target); u=flow.times(batch_size,device)
            noisy=(1-u[:,None,None,None])*target+u[:,None,None,None]*noise
            ctx=core.no_sync() if world>1 and micro+1<cfg['flow_accum'] else contextlib.nullcontext()
            with ctx,amp(device):
                prediction=core(noisy,u,anchor,text,valid,ref_present=present,rays=rays)
                loss=flow.loss(prediction,target,noise,u)
                if any_rank(not torch.isfinite(loss).item(),device):
                    raise RuntimeError('Nonfinite flow loss')
                (loss/cfg['flow_accum']).backward()
            losses+=loss.item()/cfg['flow_accum']
            tokens+=target.shape[0]*target.shape[1]*target.shape[2]
            for sample in batch:
                exposures[sample['dataset']+'/'+('image' if image_task else 'video')]+=1
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if any_rank(not torch.isfinite(norm).item(),device):
            raise RuntimeError('Nonfinite flow gradients')
        fraction=min(1.,budget.elapsed()/budget.seconds)
        lr=cfg['lr_flow']*min(1.,(step+1)/cfg['warmup_steps'])*(.02+.98*(1+math.cos(math.pi*fraction))/2)
        for group in optimizer.param_groups:
            group['lr']=lr
        optimizer.step(); ema.update(model); step+=1
        if a.smoke and step==1:
            save()
            recovered=store.load(stage+'/checkpoint_latest.pt',flow_id,required=True)
            model.load_state_dict(recovered['model']); optimizer.load_state_dict(recovered['optimizer'])
            ema.load_state_dict(recovered['ema']); ema.load_metadata(recovered['ema_metadata']); ema.to(device)
            restore_rng_state(recovered['rng'][rank])
            if rank==0:
                store.write_json(stage+'/resume_rehearsal.json',dict(step=step,optimizer_restored=True,cursors=recovered['cursors']))
            del recovered
        if step%cfg['log_every']==0:
            metrics(store,dict(stage=stage,step=step,loss=losses,lr=lr,source_samples=dict(exposures),
                task='image' if image_task else 'video',DI_throughput=tokens/max(time.monotonic()-start,1e-6),
                scope='noisy target tokens per active NPU; wall time',elapsed_hours=budget.elapsed()/3600,
                peak_allocated_gib=memory_gib(device)),rank)
        ev=any_rank(time.monotonic()-last_eval>=cfg['eval_seconds'],device)
        sv=any_rank(time.monotonic()-last_save>=cfg['save_seconds'],device)
        if ev:
            if rank==0:
                eval_flow(model,ema,rae,eval_items,mean,std,store,stage,step,device)
            barrier(); last_eval=time.monotonic()
        if ev or sv:
            save(); last_save=time.monotonic()
    if rank==0:
        eval_flow(model,ema,rae,eval_items,mean,std,store,stage,step,device)
    barrier(); save(final=True)
    interrupted=any_rank(budget.interrupted,device)
    if rank==0 and not interrupted:
        store.write_json(stage+'/complete.json',dict(identity=flow_id,step=step,
            selected=stage+'/checkpoint_final.pt',spent=budget.elapsed(),quality_policy='report_only'))
    barrier()
    if interrupted:
        raise SystemExit(75)


def main():
    global WRITER
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['ae','cache','image','video'],required=True)
    for name in ('config','cohort','output','encoder','r7'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--root',action='append',default=[])
    p.add_argument('--text',default='')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    cohort=json.loads(Path(a.cohort).read_text(encoding='utf8'))
    device,rank,world=distributed()
    print(f'[scene pipeline] stage={a.stage} rank={rank}/{world} device={device} rehearsal={a.smoke}',flush=True)
    random.seed(cfg['seed']+rank); np.random.seed(cfg['seed']+rank); torch.manual_seed(cfg['seed']+rank)
    if a.smoke:
        cfg.update(ae_steps=2,image_steps=2,video_steps=2,log_every=1,
            eval_per_domain=1,shard_samples=2)
    text_signature=None
    if a.text:
        from utils.moxing_io import read_text
        text_signature=json.loads(read_text(child(a.text,'_SUCCESS')))['index_sha256']
    identity=digest(dict(config=cfg,cohort_sha=hashlib.sha256(Path(a.cohort).read_bytes()).hexdigest(),world=world,
        encoder=sampled_file_signature(a.encoder),source_r7=sampled_file_signature(a.r7),
        text_root=a.text,text_signature=text_signature))
    store=ArtifactStore(a.output,a.root)
    if rank==0:
        old=store.json('contract.json')
        if old and old['identity']!=identity:
            raise ValueError('Resume namespace has different architecture/data; use a new namespace')
        store.write_json('contract.json',dict(identity=identity,config=cfg,world=world,
            quality_policy='report_only',smoke=a.smoke,
            cohort_sha256=hashlib.sha256(Path(a.cohort).read_bytes()).hexdigest(),
            data_counts=cohort.get('counts'),split_scope=cohort.get('split_scope'),
            code_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                         for name in ('train_scene_pipeline.py','models/scene_rae.py','models/scene_flow.py',
                                      'data/scene_rgb.py','utils/scene_run.py','utils/scene_text.py',
                                      'utils/scene_losses.py','utils/window_flow.py')}))
    barrier()
    done=store.json(a.stage+'/complete.json')
    if done:
        print('[stage already completed] '+a.stage,flush=True)
    elif a.stage=='ae':
        train_ae(a,cfg,cohort,store,identity,device,rank,world)
    elif a.stage=='cache':
        cache(a,cfg,cohort,store,identity,device,rank,world)
    else:
        train_flow(a,cfg,cohort,store,identity,device,rank,world)
    barrier()
    if WRITER is not None:
        WRITER.close()
        WRITER=None
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__=='__main__':
    main()
