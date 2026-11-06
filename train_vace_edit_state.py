"""Stage65: train native VACE to consume a shared RGB reference and edit states.

Future edited labels are stored separately and enter only flow supervision.
The RGB and RAE arms share references, sampling seeds and optimization budget.
RAE states enter the VACE control branch on every denoising evaluation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist

from utils.scene_run import ArtifactStore, StageBudget
from utils.editing_data import digest, file_hash
from utils.editing_train import (distributed, barrier, gather, any_rank, progress,
    sample_row, update_ema, load_teacher, load_encoder, write_video)
from utils.training import capture_rng_state, restore_rng_state, append_metrics
from utils.vace_edit_runtime import (load_backbone, load_vae, pack_context,
    decode_native, flow_pair, sample_native, validate_assets)
from utils.vace_state_runtime import (StateConditionedVACE, pack_edit_context,
    prepend_reference_clean, sample_state_native, state_runtime_preflight)
from utils.vace_state_data import build_reference, prepare_rae_state, build_rgb_state
from utils.editing_alignment import balanced_flow_loss
from utils.native_audit_io import seal, recover_case
from utils.editing_publication import reconcile_case
from edit_reference_video import (crop_video, crop_float, output_metrics,
    recovered_case, failure, publish_case, tensor_hash)


def code_signature():
    root = Path(__file__).resolve().parent
    names = ('train_vace_edit_state.py', 'utils/vace_state_runtime.py',
        'utils/vace_state_data.py', 'utils/vace_edit_runtime.py',
        'utils/editing_video_transport.py', 'utils/editing_alignment.py')
    return {n:file_hash(root/n) for n in names}


def store_for(a, name):
    roots='reference_root' if name=='references' else name+'_root'
    return ArtifactStore(getattr(a,name), getattr(a,roots))


def arm_store(out, arm):
    return ArtifactStore(out.local/'arms'/arm,[r.rstrip('/')+'/arms/'+arm for r in out.roots])


def checked_manifest(out,cfg):
    m = out.json('manifest.json', True)
    if m['config'] != cfg or m['code'] != code_signature():
        raise ValueError('Stage65 code/config changed; use a fresh STATE_NAMESPACE')
    return m


def encoded_reference(reference):
    value = dict(reference)
    for name in ('source','reference','anchor'):
        value[name] = (value[name].float().clamp(0,1)*255).round().byte().cpu()
    for name in ('masks','confidence'):
        value[name] = value[name].half().cpu()
    return value


def decoded_reference(reference):
    value = dict(reference)
    for name in ('source','reference','anchor'):
        value[name] = value[name].float()/255
    for name in ('masks','confidence'):
        value[name] = value[name].float()
    return value


def selected_rows(transport, cfg):
    """Preserve Stage64 source-group split; never resplit the reserved tests."""
    rows = transport['split']['records']
    selected = []
    for part in ('train','monitor','dev'):
        subset = sorted([dict(r) for r in rows if r['partition']==part],key=lambda r:r['id'])
        count = cfg.get('train_pairs',0) if part=='train' else (cfg['training']['monitor_cases'] if part=='monitor' else cfg['cases'])
        if count:subset=subset[:count]
        selected.extend(subset)
    assignments={}
    for r in selected:assignments.setdefault(r['group'],set()).add(r['partition'])
    if any(len(v)>1 for v in assignments.values()):raise ValueError('Overlapping source groups')
    return selected


def states(a,cfg,out,device,rank,world):
    bank_store, transport, refs = (store_for(a,n) for n in ('data','transport','references'))
    bank=bank_store.json('bank.json',True)
    old=transport.json('manifest.json',True)
    reference_manifest=refs.json('manifest.json',True)
    if old['bank']!=bank['identity'] or old['references']!=reference_manifest['identity']:
        raise ValueError('Stage60/63/64 inputs belong to different runs')
    local=dict(rae=file_hash(a.rae),encoder=file_hash(a.encoder)) if int(os.environ.get('LOCAL_RANK',0))==0 else None
    signatures=[s for s in gather(local,world) if s is not None]
    if not signatures or any(s!=old['teacher'] for s in signatures):
        raise ValueError('Frozen RAE/encoder differs from cached source states')
    signatures=signatures[0]
    m=dict(schema='vace-edit-state-v1',config=cfg,code=code_signature(),bank=bank['identity'],
        transport=old['identity'],references=reference_manifest['identity'],teacher=signatures,
        assets=bank.get('model_signature'),records=selected_rows(old,cfg))
    m['identity']=digest(m)
    if any(s!=m['identity'] for s in gather(m['identity'],world)):raise ValueError('Nodes disagree on inputs')
    previous=out.json('manifest.json')
    if previous and previous['identity']!=m['identity']:raise ValueError('Inputs changed; choose a fresh STATE_NAMESPACE')
    if rank==0:out.write_json('manifest.json',m)
    barrier()
    reports=[];rae=encoder=None;started=time.monotonic()
    for row in m['records'][rank::world]:
        try:
            filename='states/'+row['id']+'.pt'
            payload=out.load(filename,m['identity'])
            if payload is None:
                source_state=transport.load('inputs/'+row['id']+'.pt',old['identity'],True)
                if row['partition']=='dev':
                    original=refs.load('anchors/'+row['id']+'/result.pt',m['references'],True)
                    if original['provenance'] not in ('model_output','user_edit'):raise ValueError('Invalid dev reference provenance')
                    source,anchor=original['source'],original['anchor']
                    text,negative=original['text'],original['negative']
                    empty=negative
                    provenance=original['provenance']
                else:
                    original=bank_store.load(row['file'],m['bank'],True)
                    source=original['source'].float()/255
                    # Only the declared first edited frame is an input.
                    from edit_reference_video import source_support
                    anchor=original['target'][0].float()/255
                    support,_=source_support(source,anchor,original['timing']['spatial'],None,old['config'])
                    anchor=source[0]*(1-support)+anchor*support
                    text,negative=original['text'],original['negative']
                    empty=original.get('empty',negative)
                    provenance='paired_train_first_frame_reference'
                reference=encoded_reference(build_reference(source_state,source,anchor,cfg))
                if rae is None:
                    rae=load_teacher(a.rae,device).float()
                    encoder=load_encoder(a.encoder,device)
                state=prepare_rae_state(source_state,decoded_reference(reference),rae,encoder,cfg,device=device)
                payload=dict(reference=reference,rae_state=state['state'],calibration=state['calibration'],
                    alignment=state.get('alignment'),text=text.cpu(),negative=negative.cpu(),empty=empty.cpu(),
                    provenance=provenance)
                out.save(filename,payload,m['identity'])
            reports.append(dict(row,status='completed',state_file=filename,calibration=payload['calibration']))
            completed=sum(r['status']=='completed' for r in reports)
            progress(out,f'states_r{rank}',dict(phase='states',completed=completed,id=row['id'],
                DI_throughput=completed*cfg['frames']/max(time.monotonic()-started,1e-6),
                throughput_unit='processed source frames/s/rank including RAE encoding, calibration and IO'))
        except Exception as exc:reports.append(dict(row,**{k:v for k,v in failure(out,'states',row,exc).items() if k!='id'}))
    reports=sum(gather(reports,world),[])
    if rank==0:out.write_json('phases/states.json',dict(requested=len(m['records']),records=reports))
    barrier()


def cache(a,cfg,out,device,rank,world):
    m=checked_manifest(out,cfg);identity=m['identity'];data=store_for(a,'data')
    actual=validate_assets(a.model) if int(os.environ.get('LOCAL_RANK',0))==0 else None
    if m['assets'] is not None and any(v!=m['assets'] for v in gather(actual,world) if v is not None):
        raise ValueError('VACE weights differ from the original native bank')
    requested=out.json('phases/states.json',True)['records']
    rows=[r for r in requested if r['status']=='completed']
    reports=[];vae=None;started=time.monotonic()
    for row in rows[rank::world]:
        try:
            name='native/'+row['id']+'.pt';sup='supervision/'+row['id']+'.pt'
            encoded=out.load(name,identity)
            labels=out.load(sup,identity) if row['partition']!='dev' else None
            if encoded is None or (row['partition']!='dev' and labels is None):
                state=out.load(row['state_file'],identity,True)
                ref=decoded_reference(state['reference']);src=ref['source'];prop=ref['reference'];anchor=ref['anchor']
                if vae is None:vae=load_vae(a.model,device)
                rgb=lambda v:(v*2-1).permute(1,0,2,3).to(device)
                mask=ref['masks'].permute(1,0,2,3).to(device)
                with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type!='cpu'):
                    condition=pack_edit_context(vae,rgb(src),rgb(prop),mask,(anchor*2-1)[:,None].to(device))
                    native=pack_context(vae,rgb(src),mask,[(anchor*2-1)[:,None].to(device)]) if row['partition']=='dev' else None
                    identity_condition=source_clean=None
                    if row['partition']!='dev':
                        identity_condition=pack_edit_context(vae,rgb(src),rgb(src),mask,rgb(src[:1]))
                        source_clean=vae.encode([rgb(src)])[0].cpu().bfloat16()
                encoded=dict(context=condition.cpu().bfloat16(),
                    native_context=native.cpu().bfloat16() if native is not None else None,
                    identity_context=identity_condition.cpu().bfloat16() if identity_condition is not None else None)
                if row['partition']!='dev':
                    original=data.load(row['file'],m['bank'],True)
                    # Original clean is the native VAE edited target. Never a condition.
                    labels=dict(clean=original['clean'].cpu(),identity_clean=source_clean,
                                protocol='future edited target latent is supervision only')
                    out.save(sup,labels,identity)
                out.save(name,encoded,identity)
            reports.append(dict(row,status='completed',cache_file=name,supervision_file=sup if row['partition']!='dev' else None))
            completed=sum(r['status']=='completed' for r in reports)
            progress(out,f'cache_r{rank}',dict(phase='cache',completed=completed,id=row['id'],
                DI_throughput=completed*cfg['frames']/max(time.monotonic()-started,1e-6),
                throughput_unit='cached source frames/s/rank including VAE and IO'))
        except Exception as exc:reports.append(dict(row,**{k:v for k,v in failure(out,'cache',row,exc).items() if k!='id'}))
    reports=sum(gather(reports,world),[])
    if rank==0:out.write_json('phases/cache.json',dict(requested=len(m['records']),records=reports,
        skipped_states=[r for r in requested if r['status']!='completed']))
    barrier()


def load_case(out,row,identity,arm,device,supervised=False,identity_example=False):
    payload=out.load(row['state_file'],identity,True)
    native=out.load(row['cache_file'],identity,True)
    ref=decoded_reference(payload['reference'])
    state=(build_rgb_state(ref) if arm=='rgb' else payload['rae_state'].float())
    if identity_example:
        state=state.clone();half=(state.shape[0]-2)//2;state[half:2*half]=0
    context=native['identity_context' if identity_example else 'context']
    if device.type=='cpu':context=context.float()
    value=dict(context=context.to(device),state=state.to(device),text=payload['text'].to(device),
        negative=payload['negative'].to(device),empty=payload['empty'].to(device),
        masks=ref['masks'].to(device),spatial=ref['spatial'])
    if supervised:
        labels=out.load(row['supervision_file'],identity,True)
        value['clean']=prepend_reference_clean(labels['identity_clean' if identity_example else 'clean'].to(device),value['context'])
    return value


def flow_objective(model,item,sigma,noise,cfg,text=None,state_drop=False):
    x,target=flow_pair(item['clean'][None].float(),noise,sigma)
    state=torch.zeros_like(item['state']) if state_drop else item['state']
    prediction=model(list(x),sigma*1000,[item['context']],[item['text'] if text is None else text],[state])
    # Reference-image latent is conditioning context, not an extra supervised video frame.
    loss,metrics=balanced_flow_loss(prediction[:,:,1:],target[:,:,1:],item['masks'],item['spatial'],
        dict(flow_edit_fraction=cfg['training']['flow_edit_fraction']))
    return loss,metrics


def monitor(model,out,rows,identity,arm,cfg,device,rank,world,step):
    reports=[];model.eval()
    for row in rows[rank::world]:
        try:
            item=load_case(out,row,identity,arm,device,supervised=True)
            values=[]
            for index,level in enumerate(cfg['training']['monitor_sigmas']):
                seed=cfg['seed']+int(digest([row['id'],index,'monitor'])[:8],16)
                gen=torch.Generator(device=device).manual_seed(seed)
                noise=torch.randn((1,*item['clean'].shape),generator=gen,device=device)
                with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type!='cpu'):
                    loss,metrics=flow_objective(model,item,torch.tensor([level],device=device),noise,cfg)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite monitor')
                values.append(dict(sigma=level,loss=float(loss),**metrics))
            reports.append(dict(id=row['id'],status='completed',score=sum(v['loss'] for v in values)/len(values),sigmas=values))
        except Exception as exc:reports.append(dict(id=row['id'],status='failed',error=repr(exc)))
    reports=sum(gather(reports,world),[])
    good=[r['score'] for r in reports if r['status']=='completed']
    score=sum(good)/len(good) if good and len(good)==len(rows) else None
    result=dict(step=step,score=score,cases=reports,requested=len(rows),
        caveat='Held-out train-group fixed-noise native flow loss, not final video quality')
    if rank==0:arm_store(out,arm).write_json(f'monitor/step_{step:06d}.json',result)
    model.train()
    return result


def train_loop(a,cfg,out,m,records,model,device,rank,world):
    arm=a.arm;store=arm_store(out,arm);identity=digest([m['identity'],arm]);tc=cfg['training']
    rows=sorted([r for r in records if r['status']=='completed' and r['partition']=='train'],key=lambda r:r['id'])
    cached={r['id']:r for r in records}
    checks=sorted([cached.get(r['id'],dict(r,status='failed')) for r in m['records']
                   if r['partition']=='monitor'],key=lambda r:r['id'])
    if not rows:raise RuntimeError('No valid training cases')
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=tc['lr'],weight_decay=.01)
    ema={n:p.detach().float().clone() for n,p in model.named_parameters() if p.requires_grad}
    writer=None
    if rank==0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer=SummaryWriter(str(store.local/'tb'))
        except ImportError as exc:
            print('[editing state] TensorBoard unavailable; JSON metrics remain enabled: '+repr(exc),flush=True)
    saved=store.load('checkpoint_latest.pt',identity)
    start=updates=stale=skipped=0;best=float('inf');converged=False
    if saved:
        if saved['world']!=world:raise ValueError('Resume world size changed')
        model.restore(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        ema={n:v.to(device).float() for n,v in saved['ema'].items()}
        start,updates,stale,best,converged=(saved[k] for k in ('step','updates','stale','best','converged'))
        skipped=saved.get('skipped',0)
        restore_rng_state(saved['rng'][rank])
    wrapped=(torch.nn.parallel.DistributedDataParallel(model,device_ids=[device.index] if device.type!='cpu' else None,
        broadcast_buffers=False,init_sync=False) if world>1 else model)
    budget=StageBudget(float('inf'));budget.install_signals()
    begin=time.monotonic();initial_updates=updates;last_step=start;interrupted=False
    limit=min(getattr(a,'stop_after',0) or tc['max_steps'],tc['max_steps'])
    if rank==0:store.write_json('run.json',dict(identity=identity,arm=arm,config=cfg,world=world,
        train_cases=len(rows),monitor_cases=len(checks),trainable=sum(p.numel() for p in params),
        architecture='VACE LoRA plus edit-state control-token bridge; future targets are loss only'))
    def save(step):
        rng=gather(capture_rng_state(),world)
        if rank==0:
            payload=dict(step=step,updates=updates,skipped=skipped,stale=stale,best=best,converged=converged,world=world,
                model=model.deltas(),ema={k:v.detach().cpu() for k,v in ema.items()},optimizer=optimizer.state_dict(),
                rng=rng,config=cfg,arm=arm)
            store.save(f'checkpoints/step_{step:06d}.pt',payload,identity)
            store.save('checkpoint_latest.pt',payload,identity)
            files=[p.relative_to(store.local).as_posix() for p in (store.local/'logs').glob('*') if p.is_file()]
            if writer:writer.flush()
            files.extend(p.relative_to(store.local).as_posix() for p in (store.local/'tb').glob('*') if p.is_file())
            if (store.local/'metrics.jsonl').exists():files.append('metrics.jsonl')
            if files:store.publish(files)
        barrier()
    for step in range(start,start if converged else limit):
        model.train();optimizer.zero_grad(set_to_none=True);totals={};failed=False
        lr=tc['lr']*min(1.,(step+1)/max(1,tc['warmup_steps']))
        for group in optimizer.param_groups:group['lr']=lr
        for micro in range(tc['accum']):
            row=sample_row(rows,step,micro,rank,world,tc['accum'],cfg['seed'])
            error=None
            try:
                rng=random.Random(cfg['seed']+step*world*tc['accum']+micro*world+rank)
                identity_example=rng.random()<tc.get('identity_fraction',0.)
                item=load_case(out,row,m['identity'],arm,device,supervised=True,
                    identity_example=identity_example)
                generator=torch.Generator(device=device).manual_seed(cfg['seed']+step*world*tc['accum']+micro*world+rank)
                noise=torch.randn((1,*item['clean'].shape),generator=generator,device=device)
                u=torch.rand(1,generator=generator,device=device).clamp(1e-4,1-1e-4)
                sigma=cfg['shift']*u/(1+(cfg['shift']-1)*u)
                text_drop=rng.random()<tc.get('text_dropout',.1)
                text=item['empty'] if identity_example or text_drop else item['text']
                state_drop=rng.random()<tc.get('state_dropout',.1)
            except Exception as exc:error=repr(exc)
            if any_rank(error is not None,device):
                failed=True
                if error:progress(store,f'skip_r{rank}',dict(step=step,id=row['id'],error=error))
                break
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type!='cpu'):
                loss,metrics=flow_objective(wrapped,item,sigma,noise,cfg,text,state_drop)
            if any_rank(not bool(torch.isfinite(loss)),device):
                torch.nan_to_num(loss,nan=0.,posinf=0.,neginf=0.).mul(0).backward();failed=True;break
            (loss/tc['accum']).backward()
            for k,v in dict(loss=float(loss.detach()),**metrics).items():totals[k]=totals.get(k,0.)+v/tc['accum']
        if failed:
            optimizer.zero_grad(set_to_none=True);skipped+=1
        else:
            bridge_norm=sum(float(p.grad.detach().float().square().sum()) for n,p in model.named_parameters()
                if n.startswith(('bridge.','state_norm.')) and p.grad is not None)**.5
            norm=torch.nn.utils.clip_grad_norm_(params,tc['grad_clip'])
            totals.update(gradient_norm=float(norm),bridge_gradient_norm=bridge_norm)
            if any_rank(not bool(torch.isfinite(norm)),device):
                optimizer.zero_grad(set_to_none=True);skipped+=1
            else:
                optimizer.step();update_ema(ema,model,step,tc['ema_decay']);updates+=1
        last_step=step+1
        if last_step==1 or last_step%tc['log_every']==0:
            values=gather(totals,world)
            if rank==0:
                keys=set().union(*(v.keys() for v in values))
                means={k:sum(v.get(k,0.) for v in values)/world for k in keys}
                if device.type!='cpu':means['peak_allocated_GiB']=getattr(torch,device.type).max_memory_allocated()/2**30
                metric_row=dict(phase='train',arm=arm,step=last_step,updates=updates,skipped=skipped,lr=lr,**means,
                    DI_throughput=(updates-initial_updates)*world*tc['accum']*cfg['frames']/max(time.monotonic()-begin,1e-6),
                    throughput_unit='successful optimization target frames/s across ranks including monitor, previews and IO')
                progress(store,'training',metric_row)
                append_metrics(str(store.local/'metrics.jsonl'),dict(time=time.time(),**metric_row))
                if writer:
                    for key,value in metric_row.items():
                        if isinstance(value,(int,float)):writer.add_scalar('train/'+key,value,last_step)
        if last_step%tc['eval_every']==0 or last_step==tc['max_steps']:
            online=model.deltas();model.restore(ema)
            report=monitor(model,out,checks,m['identity'],arm,cfg,device,rank,world,last_step)
            model.restore(online);score=report['score']
            if rank==0 and writer and score is not None:writer.add_scalar('monitor/flow_loss',score,last_step)
            if score is not None:
                improved=score<best-tc['min_improvement']*max(abs(best) if math.isfinite(best) else 1.,.01)
                if improved:
                    best,stale=score,0
                    if rank==0:store.save('selected.pt',dict(model={k:v.detach().cpu() for k,v in ema.items()},
                        step=last_step,score=score,selection='EMA heldout train-group fixed-noise flow loss'),identity)
                elif last_step>=tc['min_steps']:stale+=1
                converged=last_step>=tc['min_steps'] and stale>=tc['patience']
        interrupted=any_rank(budget.interrupted,device)
        if last_step%tc['save_every']==0 or converged or interrupted:save(last_step)
        preview=tc.get('preview_every',0)
        if preview and last_step%preview==0 and not interrupted:
            rng_state=capture_rng_state();online=model.deltas();model.restore(ema)
            try:render_records(a,cfg,out,m,records,device,rank,world,model=model,
                checkpoint_tag=f'ema_step_{last_step}',prefix=f'previews/{arm}/s{last_step:06d}',limit=tc.get('preview_cases',8))
            finally:model.restore(online);restore_rng_state(rng_state)
        if interrupted or converged:break
    save(last_step)
    if rank==0:
        selected=store.load('selected.pt',identity)
        if selected is None or selected.get('score') is None:
            store.save('selected.pt',dict(model={k:v.detach().cpu() for k,v in ema.items()},step=last_step,score=None,
                selection='final EMA fallback; no complete monitor score'),identity)
        status='failed_no_updates' if not updates else ('interrupted' if interrupted or (getattr(a,'stop_after',0) and last_step<tc['max_steps'] and not converged) else 'completed')
        store.write_json('training_summary.json',dict(status=status,step=last_step,updates=updates,skipped=skipped,
            converged=converged,best_score=best if math.isfinite(best) else None,
            caveat='Stopping rule, not proof of final video quality; matched maximum budgets may stop at different steps'))
        if writer:writer.close()
    barrier()


def train(a,cfg,out,device,rank,world):
    if a.arm not in ('rgb','rae'):raise ValueError('Only RGB/RAE arms are trained')
    m=checked_manifest(out,cfg)
    actual=validate_assets(a.model) if int(os.environ.get('LOCAL_RANK',0))==0 else None
    if m['assets'] is not None and any(v!=m['assets'] for v in gather(actual,world) if v is not None):
        raise ValueError('Training base weights differ from the condition cache')
    preflight=state_runtime_preflight(device,cfg)
    if rank==0:arm_store(out,a.arm).write_json('preflight.json',preflight)
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    if device.type!='cpu':getattr(torch,device.type).manual_seed_all(cfg['seed'])
    backbone=load_backbone(a.model,device)
    torch.manual_seed(cfg['seed'])
    model=StateConditionedVACE(backbone,cfg,cfg['state_channels'][a.arm]).to(device)
    records=out.json('phases/cache.json',True)['records']
    train_loop(a,cfg,out,m,records,model,device,rank,world)


def publish_lossless(out,folder,identity,payload,fps):
    """Repair interruption after MP4 commit without another expensive generation."""
    import numpy as np
    receipt_name=folder+'/receipt.json';lossless=folder+'/generated_frames.npz'
    receipt=out.json(receipt_name,True)
    if lossless not in receipt['files']:
        np.savez_compressed(out.local/lossless,
            frames=crop_video(payload['generated'],payload['spatial']['valid_rect']).numpy(),fps=fps)
        receipt=seal(out.local,receipt_name,identity,[*receipt['files'],lossless],receipt['metadata'])
    out.publish([*receipt['files'],receipt_name])
    return dict(receipt['metadata'],receipt=receipt_name)


def render_records(a,cfg,out,m,records,device,rank,world,model=None,checkpoint_tag=None,prefix=None,limit=0):
    identity=m['identity'];arm=a.arm;previews=prefix is not None
    prefix=prefix or 'render/'+arm
    if previews and rank==0:
        index=out.json('previews/index.json') or {'prefixes':[]}
        if prefix not in index['prefixes']:index['prefixes'].append(prefix)
        out.write_json('previews/index.json',index)
    rows=sorted([r for r in records if r['partition']=='dev'],key=lambda r:r['id'])
    if limit:rows=rows[:limit]
    if checkpoint_tag is None:
        checkpoint_tag='native' if arm=='native' else arm_store(out,arm).json('selected.pt.json',True)['sha256']
    reports=[];vae=None;started=time.monotonic()
    for row in rows[rank::world]:
        folder=prefix+'/'+row['id']
        try:
            if row['status']!='completed':raise RuntimeError('Input preparation failed for this case')
            state=out.load(row['state_file'],identity,True);native=out.load(row['cache_file'],identity,True)
            ref=decoded_reference(state['reference']);seed=cfg['seed']+int(digest(row['id'])[:8],16)%1000000
            input_identity=digest([identity,row['id'],seed,tensor_hash(state['reference']['source']),
                tensor_hash(state['reference']['reference']),tensor_hash(ref['masks']),tensor_hash(ref['anchor'])])
            generation=digest([input_identity,arm,checkpoint_tag])
            payload,record=recovered_case(out,folder,identity)
            if payload is not None and payload.get('generation_identity')!=generation:payload=record=None
            if record:
                reports.append(publish_lossless(out,folder,identity,payload,cfg['fps']));continue
            if model is None:
                actual=validate_assets(a.model)
                if m['assets'] is not None and actual!=m['assets']:raise ValueError('Native weights changed')
                backbone=load_backbone(a.model,device)
                if arm=='native':model=backbone
                else:
                    model=StateConditionedVACE(backbone,cfg,cfg['state_channels'][arm]).to(device)
                    chosen=arm_store(out,arm).load('selected.pt',digest([identity,arm]),True)
                    model.restore(chosen['model'])
            model.eval()
            if vae is None:vae=load_vae(a.model,device)
            with torch.no_grad():
                if arm=='native':
                    latent=sample_native(model,native['native_context'].to(device),state['text'].to(device),state['negative'].to(device),cfg,seed)
                else:
                    features=build_rgb_state(ref) if arm=='rgb' else state['rae_state']
                    latent=sample_state_native(model,native['context'].to(device),state['text'].to(device),state['negative'].to(device),features.to(device),cfg,seed)
                generated=(decode_native(vae,latent,1).permute(1,0,2,3).float()+1)/2
            if not bool(torch.isfinite(generated).all()):raise FloatingPointError('Nonfinite generated video')
            generated=generated.clamp(0,1).cpu();rect=ref['spatial']['valid_rect']
            payload=dict(generated=generated,spatial=ref['spatial'],input_identity=input_identity,
                generation_identity=generation,checkpoint=checkpoint_tag,seed=seed,reference_count=1)
            metrics=dict(id=row['id'],status='completed',arm=arm,input_identity=input_identity,generation_identity=generation,
                checkpoint=checkpoint_tag,seed=seed,provenance=state['provenance'],
                conditioning='original source reactive + anchor' if arm=='native' else 'shared RGB reference reactive + trained '+arm+' state bridge',
                **output_metrics(crop_float(ref['source'],rect),crop_float(generated,rect),crop_float(ref['masks'],rect)))
            record=publish_case(out,folder,identity,payload,metrics,
                [ref['source'],ref['reference'],generated],['SOURCE','SHARED RGB REFERENCE',arm.upper()+' OUTPUT'],rect,cfg['fps'])
            # Lossless RGB for future quantitative review; MP4 codec error is not a method metric.
            reports.append(publish_lossless(out,folder,identity,payload,cfg['fps']))
            done=sum(r['status']=='completed' for r in reports)
            progress(out,f'render_{arm}_r{rank}',dict(phase='preview' if previews else 'render',arm=arm,completed=done,
                DI_throughput=done*cfg['frames']/max(time.monotonic()-started,1e-6),throughput_unit='generated frames/s/rank including IO'))
        except Exception as exc:reports.append(dict(failure(out,('preview_' if previews else 'render_')+arm,row,exc),arm=arm))
    reports=sum(gather(reports,world),[])
    if rank==0:out.write_json((prefix+'/phase.json') if previews else ('phases/render_'+arm+'.json'),dict(requested=len(rows),cases=reports))
    barrier()
    return reports


def render(a,cfg,out,device,rank,world):
    m=checked_manifest(out,cfg)
    render_records(a,cfg,out,m,out.json('phases/cache.json',True)['records'],device,rank,world)


def summarize(out,cfg):
    from html import escape
    m=checked_manifest(out,cfg);identity=m['identity']
    rows=[r for r in m['records'] if r['partition']=='dev'];failures=[];synced=[];rendered={}
    selected={}
    preview_failures=[]
    for prefix in (out.json('previews/index.json') or {}).get('prefixes',[]):
        preview=out.json(prefix+'/phase.json')
        if not preview:
            preview_failures.append(dict(phase=prefix,error='Preview interrupted before phase receipt'))
            continue
        for r in preview.get('cases',[]):
            if r['status']!='completed':preview_failures.append(r);continue
            try:
                roots=reconcile_case(out,r['receipt'],identity)
                synced.append(dict(receipt=r['receipt'],ok=all(x['ok'] for x in roots)))
            except Exception as exc:synced.append(dict(receipt=r['receipt'],ok=False,error=repr(exc)))
    for arm in ('rgb','rae'):
        store=arm_store(out,arm);selected[arm]=store.json('selected.pt.json')
        status=store.json('training_summary.json')
        if not status or status['status']!='completed':failures.append(dict(arm=arm,phase='train',status=status))
    for arm in ('native','rgb','rae'):
        report=out.json('phases/render_'+arm+'.json') or {}
        cases=report.get('cases',[])
        if len(cases)!=len(rows):failures.append(dict(arm=arm,phase='render',error='Coverage mismatch',actual=len(cases),expected=len(rows)))
        for r in cases:
            rendered[(arm,r['id'])]=r
            if r['status']!='completed':failures.append(r);continue
            try:
                roots=reconcile_case(out,r['receipt'],identity)
                synced.append(dict(receipt=r['receipt'],ok=all(x['ok'] for x in roots)))
            except Exception as exc:synced.append(dict(receipt=r['receipt'],ok=False,error=repr(exc)))
    page=['<!doctype html><meta charset="utf-8"><title>VACE trained edit states</title>',
        '<style>body{font:16px sans-serif;max-width:1500px;margin:24px auto}video{max-width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>Native + reference / trained RGB state / trained RAE state</h1>',
        '<p>Both trained arms share the same RGB reference, masks, seed and native LoRA settings. Native receives the same first-frame reference. No late writeback or future target conditioning.</p>']
    for row in rows:
        try:
            records=[rendered.get((arm,row['id'])) for arm in ('native','rgb','rae')]
            if any(r is None or r['status']!='completed' for r in records):raise ValueError('Missing current render; old files are not completion')
            if len({r['input_identity'] for r in records})!=1:raise ValueError('Unmatched inputs')
            for arm,r in zip(('native','rgb','rae'),records):
                expected='native' if arm=='native' else (selected[arm] or {}).get('sha256')
                if r['checkpoint']!=expected:raise ValueError('Old selected weights rendered; rerun render')
            payloads=[out.load('render/'+arm+'/'+row['id']+'/result.pt',identity,True) for arm in ('native','rgb','rae')]
            generations=[p['generation_identity'] for p in payloads]
            folder='comparisons/'+row['id'];receipt=folder+'/receipt.json'
            previous=recover_case(out.local,receipt,identity,out.roots)
            if previous and previous['metadata'].get('generations')!=generations:previous=None
            if not previous:
                movie=folder+'/native_rgb_rae.mp4';rect=payloads[0]['spatial']['valid_rect']
                write_video(out.local/movie,*[crop_video(p['generated'],rect) for p in payloads],cfg['fps'],
                    labels=['NATIVE + REFERENCE','TRAINED RGB STATE','TRAINED RAE STATE'])
                previous=seal(out.local,receipt,identity,[movie],dict(id=row['id'],generations=generations))
            out.publish([*previous['files'],receipt]);roots=reconcile_case(out,receipt,identity)
            synced.append(dict(receipt=receipt,ok=all(x['ok'] for x in roots)))
            page.extend(['<h2>'+escape(row['id']+' '+row.get('instruction',''))+'</h2>',
                '<video controls preload="metadata" src="comparisons/'+row['id']+'/native_rgb_rae.mp4"></video>'])
        except Exception as exc:failures.append(dict(id=row['id'],phase='comparison',error=repr(exc)))
    cached=out.json('phases/cache.json') or {}
    skipped=[r for r in cached.get('records',[])+cached.get('skipped_states',[]) if r['status']!='completed']
    failures.extend(r for r in skipped if r['partition']=='dev')
    bad=[r for r in synced if not r['ok']]
    summary=dict(status='completed' if not failures and not bad else 'incomplete',cases=len(rows),failures=failures,
        media_failures=bad,preview_failures=preview_failures,skipped_training_data=[r for r in skipped if r['partition']!='dev'],
        training={arm:arm_store(out,arm).json('training_summary.json') for arm in ('rgb','rae')},
        limitations=['Internal dev, not an independent benchmark; wrong anchor semantics remain failures.',
            'RGB has a wider input projection; actual trainable parameter counts are recorded, not falsely called identical.',
            'RAE-specific gain and geometry benefits require actual output review; training flow loss is not a quality score.'])
    out.write_json('summary.json',summary);out.write_json('media_sync.json',dict(checked=len(synced),failed=bad))
    page.append('<pre>'+escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
    (out.local/'report.html').write_text('\n'.join(page),encoding='utf8');out.publish(['report.html'])
    if failures or bad:raise RuntimeError('Incomplete Stage65; inspect summary.json; successful outputs retained')


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('states','cache','train','render','summarize'),required=True)
    p.add_argument('--arm',choices=('native','rgb','rae'),default='native')
    for name in ('config','output'):p.add_argument('--'+name,required=True)
    for name in ('data','transport','references','rae','encoder','model'):p.add_argument('--'+name,default='')
    for name in ('root','data-root','transport-root','reference-root'):p.add_argument('--'+name,action='append',default=[])
    p.add_argument('--stop-after',type=int,default=0)
    return p


def main():
    a=build_parser().parse_args()
    cfg=json.loads(Path(a.config).read_text());out=ArtifactStore(a.output,a.root)
    if a.phase=='summarize':
        if int(os.environ.get('NODE_RANK','0'))==0:summarize(out,cfg)
        return
    device,rank,world=distributed()
    try:globals()[a.phase](a,cfg,out,device,rank,world)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
