"""Native VACE instruction post-training: matched SFT / edited-target RAE arms."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F
from utils.editing_data import digest
from utils.editing_train import (distributed,barrier,gather,any_rank,progress,load_teacher,
    sample_row,update_ema,write_video,pixel_report)
from utils.scene_run import ArtifactStore,StageBudget
from utils.training import append_metrics,capture_rng_state,restore_rng_state
from utils.vace_edit_runtime import (validate_assets,load_backbone,load_vae,EditingModel,
    flow_pair,rae_supervision,sample_native)


def evaluate(a,cfg,bank,data,out,model,device,rank,world,step,variant,identity):
    rows=evaluate_protocol(a,cfg,bank,data,out,model,device,rank,world,step,variant,identity)
    if getattr(a,'controls',None) and (step>=cfg['steps'] or a.arm=='native' or getattr(a,'eval_checkpoint',None)):
        from utils.editing_controls import load_controls
        controls=load_controls(a.controls)
        rows+=evaluate_protocol(a,cfg,bank,data,out,model,device,rank,world,step,variant,identity,controls)
    return rows


def evaluate_protocol(a,cfg,bank,data,out,model,device,rank,world,step,variant,identity,controls=None):
    from utils.native_audit_io import recover_case,seal
    count=cfg['dev_pairs'] if a.arm=='native' or step>=cfg['steps'] else cfg['eval_cases']
    records=sorted([r for r in bank['records'] if r['split']=='dev'],key=lambda r:r['id'])[:count]
    if a.arm=='native':
        records+=[r for r in bank['records'] if r['split']=='benchmark_dev']
    if controls is not None:
        records=[r for r in records if r['split']=='dev' and r['id'] in controls['cases']]
    tag=variant if controls is None else variant+'_source_region'
    rows=[]; vae=None; started=time.monotonic(); completed=0
    model.eval()
    for row in records[rank::world]:
        folder=f'samples/s{step:06d}/{tag}/{row["id"]}'
        contract=digest([identity,step,variant,row['id']] + ([controls['identity']] if controls else []))
        receipt=recover_case(out.local,folder+'/receipt.json',contract,out.roots)
        if receipt:
            # Recovered movies may come from an older job's durable mirror.
            # Publish payloads before the receipt into this job's OUTPUT_URL too.
            out.publish([*receipt['files'],folder+'/receipt.json'])
            rows.append(receipt['metadata']); continue
        try:
            sample=data.load(row['file'],bank['identity'],True)
            if vae is None: vae=load_vae(a.model,device)
            condition=sample['context'].to(device)
            region_mask=None
            if controls is not None:
                from utils.editing_controls import source_box_mask
                from utils.vace_edit_runtime import pack_context
                region_mask=source_box_mask(controls['cases'][row['id']],sample['timing']['spatial'],cfg['frames'])
                context_name='source_region_context/'+digest([bank['identity'],row['id'],controls['identity']])+'.pt'
                context_identity=digest([bank['identity'],controls['identity'],'source-only-vace-context-v1'])
                cached=out.load(context_name,context_identity)
                if cached is None:
                    # Only source RGB and the explicit user box enter the encoder.
                    with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type!='cpu'):
                        encoded=pack_context(vae,sample['source'].permute(1,0,2,3).to(device).float()/127.5-1,
                                             region_mask.to(device))
                    cached=dict(context=encoded.cpu().bfloat16())
                    out.save(context_name,cached,context_identity)
                condition=cached['context'].to(device)
            seed=cfg['seed']+int(digest(row['id'])[:8],16)%1000000
            with torch.no_grad():
                latent=sample_native(model,condition,sample['text'].to(device),
                                     sample['negative'].to(device),cfg,seed)
                if not torch.isfinite(latent).all(): raise ValueError('Nonfinite generated native latent')
                rgb=vae.decode([latent])[0].permute(1,0,2,3)
                rgb=((rgb+1)*127.5).round().clamp(0,255).byte().cpu()
            top,left,height,width=sample['timing']['spatial']['valid_rect']
            source=sample['source'][:,:,top:top+height,left:left+width]
            target=sample['target'][:,:,top:top+height,left:left+width]
            cropped=rgb[:,:,top:top+height,left:left+width]
            result=dict(id=row['id'],step=step,variant=variant,split=row['split'],
                prompt=sample['row']['instruction'],prompt_kind=row['prompt_kind'],seed=seed,
                spatial=sample['timing']['spatial'],**pixel_report(source,target,cropped))
            result['protocol']='text_only' if controls is None else 'text_plus_source_box_diagnostic'
            if controls is not None:
                result['control_identity']=controls['identity']
                result['control']=controls['cases'][row['id']]
                result['control_caveat']='Additional manual source-video region, same for every arm; NOT evidence of stronger text-only instruction following.'
            if cfg.get('region_metrics') and row['split']=='dev':
                from utils.editing_alignment import region_pixel_report
                external=None if region_mask is None else region_mask.permute(1,0,2,3)[:,:,top:top+height,left:left+width]
                result.update(region_pixel_report(source,target,cropped,cfg,external_mask=external))
            if row['split']=='benchmark_dev':
                result['metric_caveat']='Official condition/prompt at reduced pilot geometry; raw_video is not an edited-target ground truth. NOT an official benchmark score.'
            movie=folder+'/source_target_output.mp4'
            labels=(['OFFICIAL SOURCE CONDITION','RAW REFERENCE (not edited GT)','NATIVE VACE OUTPUT']
                    if row['split']=='benchmark_dev' else
                    ['SOURCE' if controls is None else 'SOURCE + PROVIDED BOX','EDITED TARGET',f'{a.arm.upper()} {tag.upper()} step {step}'])
            write_video(out.local/movie,source,target,cropped,sample['timing']['output_fps'],labels=labels)
            out.save(folder+'/generated_latent.pt',dict(latent=latent.cpu(),seed=seed),contract)
            receipt=seal(out.local,folder+'/receipt.json',contract,[movie],result)
        except Exception as exc:
            result=dict(id=row['id'],step=step,variant=variant,error=repr(exc),status='failed')
            out.write_json(folder+'/failure.json',result)
            print('[editing sample failed] '+json.dumps(result),flush=True)
            rows.append(result); continue
        out.publish([movie,folder+'/receipt.json'])
        rows.append(result); completed+=1
        progress(out,f'eval_rank{rank}',dict(phase='sampling',step=step,variant=variant,completed=completed,
            DI_throughput=completed*cfg['frames']/max(1e-6,time.monotonic()-started),throughput_unit='generated frames/s per rank'))
    del vae
    rows=sum(gather(rows,world),[])
    if rank==0: out.write_json(f'evaluation/s{step:06d}_{tag}.json',dict(cases=rows,requested=len(records),
        completed=sum('error' not in r for r in rows),failed=sum('error' in r for r in rows),
        protocol='text_only' if controls is None else 'text_plus_source_box_diagnostic',
        omitted_controls=0 if controls is None else len(controls['cases'])-len(records)))
    barrier()
    return rows


def run_training(a,cfg,bank,data,out,identity,device,rank,world,model,teacher_bank=None,decoder=None):
    """Also exercised with tiny real differentiable modules in CPU contract tests."""
    params=[p for p in model.parameters() if p.requires_grad]
    backbone_params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('readout.')]
    readout_params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('readout.')]
    optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=.01)
    if world>1:
        # Frozen native weights are identical via asset signature; broadcasting
        # 31GB/rank at DDP construction serves no purpose and adds a huge peak.
        wrapped=torch.nn.parallel.DistributedDataParallel(model,
            device_ids=[device.index] if device.type!='cpu' else None,
            broadcast_buffers=False,find_unused_parameters=False,init_sync=False)
    else: wrapped=model
    latest='checkpoint_latest.pt'
    if rank==0:
        pointer=out.json('resume_pointer.pt.json')
        if pointer and pointer['identity']!=identity: raise ValueError('Resume pointer identity differs')
        recovery=pointer['file'] if pointer else latest
        recovered=out.load(recovery,identity)
        if recovered is not None:
            out.publish([recovery,recovery+'.json'])
            if pointer: out.write_json('resume_pointer.pt.json',pointer)
        del recovered
    barrier()
    pointer=out.json('resume_pointer.pt.json')
    if pointer and pointer['identity']!=identity: raise ValueError('Resume pointer identity differs')
    recovery=pointer['file'] if pointer else latest
    recovered=out.load(recovery,identity)
    ema={n:p.detach().float().clone() for n,p in model.named_parameters() if p.requires_grad}
    start=0
    if recovered:
        if recovered['world']!=world: raise ValueError('Resume world size changed; use a new namespace')
        model.restore(recovered['model']); optimizer.load_state_dict(recovered['optimizer'])
        ema={n:v.to(device).float() for n,v in recovered['ema'].items()}
        start=int(recovered['step']); restore_rng_state(recovered['rng'][rank])
        print(f'[editing resume] {a.arm} step={start}; optimizer/FP32 EMA/RNG restored',flush=True)
        del recovered
    rows=sorted([r for r in bank['records'] if r['split']=='train'],key=lambda r:r['id'])
    teacher_map={r['id']:r for r in teacher_bank['records']} if teacher_bank else {}
    tb=None
    if rank==0:
        from torch.utils.tensorboard import SummaryWriter
        tb=SummaryWriter(str(out.local/'tb'))
        out.write_json('run.json',dict(identity=identity,config=cfg,arm=a.arm,world=world,
            trainable=sum(p.numel() for p in params),total=sum(p.numel() for p in model.parameters()),
            native_bank=bank['identity'],teacher=teacher_bank['identity'] if teacher_bank else None))
    budget=StageBudget(float('inf')); budget.install_signals()
    stop=min(a.stop_after or cfg['steps'],cfg['steps'])
    started=time.monotonic(); last_step=start; skipped=0
    last_samples=[]
    def save(step):
        rng=gather(capture_rng_state(),world)
        if rank==0:
            payload=dict(step=step,world=world,model=model.deltas(),
                ema={n:v.detach().cpu() for n,v in ema.items()},optimizer=optimizer.state_dict(),
                rng=rng,config=cfg,arm=a.arm,identity=identity)
            out.save(f'checkpoints/step_{step:06d}.pt',payload,identity)
            out.write_json('resume_pointer.pt.json',dict(file=f'checkpoints/step_{step:06d}.pt',
                step=step,identity=identity,committed_unix_ns=time.time_ns()))
            out.save(latest,payload,identity)
            if tb: tb.flush()
            logs=(['metrics.jsonl'] if (out.local/'metrics.jsonl').is_file() else [])
            logs.extend(p.relative_to(out.local).as_posix() for p in (out.local/'tb').glob('*') if p.is_file())
            out.publish(logs)
        barrier()
    def previews(step):
        rng=capture_rng_state()
        samples=evaluate(a,cfg,bank,data,out,model,device,rank,world,step,'raw',identity)
        online=model.deltas(); model.restore(ema)
        try:
            samples+=evaluate(a,cfg,bank,data,out,model,device,rank,world,step,'ema',identity)
        finally:
            model.restore(online); restore_rng_state(rng)
        return samples
    for step in range(start,stop):
        model.train(); optimizer.zero_grad(set_to_none=True)
        totals={}; batch_failed=False
        warm=min(1.,(step+1)/max(1,cfg['warmup_steps']))
        cosine=.1+.9*.5*(1+math.cos(math.pi*step/max(1,cfg['steps'])))
        for group in optimizer.param_groups: group['lr']=cfg['lr']*warm*cosine
        for micro in range(cfg['accum']):
            row=sample_row(rows,step,micro,rank,world,cfg['accum'],cfg['seed'])
            error=None
            try:
                sample=data.load(row['file'],bank['identity'],True)
                teacher=(data.load(teacher_map[row['id']]['file'],teacher_bank['identity'],True) if teacher_bank else None)
                target=teacher['z'][None].to(device).float() if teacher else None
                support=pixel_support=None
                if a.arm in ('region_sft','rae_balanced'):
                    from utils.editing_alignment import change_support
                    pixel_support=change_support(sample['source'],sample['target'],
                        cfg.get('edit_support_threshold',16/255),cfg.get('edit_support_dilation',5))
                if a.arm=='rae_balanced':
                    from utils.editing_alignment import support_on_teacher
                    support=support_on_teacher(sample['source'],sample['target'],sample['timing']['spatial'],
                        teacher['valid_rect'],cfg['rae_frames'],cfg,pixel_support=pixel_support)
            except Exception as exc:
                error=repr(exc)
                progress(out,f'data_rank{rank}',dict(phase='skip_cached_pair',id=row['id'],error=error,step=step))
            if any_rank(error is not None,device):
                batch_failed=True; break
            # Per-position generator guarantees identical noises/drop decisions
            # between native SFT and RAE arms, including after a process restart.
            generator=torch.Generator(device=device).manual_seed(cfg['seed']+step*world*cfg['accum']+micro*world+rank)
            clean=sample['clean'][None].to(device).float()
            noise=torch.randn(clean.shape,device=device,generator=generator)
            u=torch.rand(1,device=device,generator=generator).clamp(1e-4,1-1e-4)
            sigma=cfg['shift']*u/(1+(cfg['shift']-1)*u)
            x,velocity=flow_pair(clean,noise,sigma)
            drop=bool((torch.rand((),device=device,generator=generator)<cfg['text_dropout']).item())
            detach=(a.arm=='rae_detached' or step<cfg['rae_warmup'])
            sync=wrapped.no_sync() if world>1 and micro+1<cfg['accum'] else nullcontext()
            with sync,torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type!='cpu'):
                pred,z=wrapped(list(x),sigma*1000,[sample['context'].to(device)],
                               [sample['empty' if drop else 'text'].to(device)],detach_readout=detach)
                aux={}
                if pixel_support is not None:
                    from utils.editing_alignment import balanced_flow_loss
                    flow,aux=balanced_flow_loss(pred,velocity,pixel_support,sample['timing']['spatial'],cfg)
                else:
                    flow=F.mse_loss(pred.float(),velocity)
                loss=flow
                if target is not None:
                    addition,rae_aux=rae_supervision(z,target,decoder,sigma,cfg,step%cfg['rae_frames'],teacher.get('valid_rect'),
                        spatial=sample.get('timing',{}).get('spatial'),support=support)
                    aux.update(rae_aux)
                    loss=loss+addition
                    probe=cfg.get('gradient_probe_every',0)
                    if probe and (step+1)%probe==0 and not detach:
                        from utils.editing_alignment import hidden_gradient_diagnostics
                        aux.update(hidden_gradient_diagnostics(flow,addition,model.last_readout_hidden))
                if any_rank(not bool(torch.isfinite(loss).all()),device):
                    raise FloatingPointError('Nonfinite editing loss; see last durable checkpoint')
                (loss/cfg['accum']).backward()
            if hasattr(model,'last_readout_hidden'): model.last_readout_hidden=None
            for k,v in dict(loss=float(loss.detach()),flow=float(flow.detach()),**aux).items():
                totals[k]=totals.get(k,0.)+v/cfg['accum']
        if batch_failed:
            optimizer.zero_grad(set_to_none=True); skipped+=1
            # Do not let a destroyed cache become 1000 pretend optimizer steps.
            if skipped>=10: raise RuntimeError('Ten global batches lost to unavailable cache; restore data first')
            continue
        # A detached/head-only control must not alter LoRA through a shared
        # gradient clipping norm. Clip the two optimizer subspaces separately.
        norm=torch.nn.utils.clip_grad_norm_(backbone_params,1.)
        head_norm=torch.nn.utils.clip_grad_norm_(readout_params,1.) if readout_params else norm.new_zeros(())
        if any_rank(not bool(torch.isfinite(norm+head_norm)),device): raise FloatingPointError('Nonfinite gradient')
        optimizer.step(); update_ema(ema,model,step+1,cfg['ema_decay']); last_step=step+1
        if (step+1)%cfg['log_every']==0 or step==start:
            statistics=gather(totals,world)
            if rank==0:
                stats={k:sum(x[k] for x in statistics)/world for k in totals}
                if 'aux_probe_active' in stats:
                    active=stats['aux_probe_active']
                    for key in ('aux_flow_cosine','aux_flow_grad_ratio'):
                        stats[key]=stats[key]/active if active else 0.
                stats.update(step=step+1,arm=a.arm,lr=optimizer.param_groups[0]['lr'],grad_norm=float(norm),head_grad_norm=float(head_norm),
                    DI_throughput=(step+1-start-skipped)*world*cfg['accum']*cfg['frames']/max(1e-6,time.monotonic()-started),
                    throughput_unit='global target video frames/s including IO, sampling and checkpoints',
                    skipped_global_batches=skipped,rae_to_backbone=bool(teacher_bank and not detach))
                if device.type!='cpu': stats['peak_allocated_GiB']=getattr(torch,device.type).max_memory_allocated()/2**30
                append_metrics(str(out.local/'metrics.jsonl'),stats)
                for k,v in stats.items():
                    if isinstance(v,(int,float)): tb.add_scalar(k,v,step+1)
                tb.flush(); print('[VACE train] '+json.dumps(stats),flush=True)
        due=any_rank(budget.due(),device)
        if (step+1)%cfg['save_every']==0 or step+1==stop or due:
            save(step+1)
        if ((step+1)%cfg['eval_every']==0 or step+1==stop) and not due:
            last_samples=previews(step+1)
        if due: break
    if start>=stop:
        # Recover unfinished final sampling even when all optimizer steps exist.
        last_samples=previews(start)
    if rank==0:
        if tb: tb.close()
        out.write_json('status.json',dict(status='completed' if last_step>=cfg['steps'] else 'paused_at_checkpoint',
            step=last_step,target_steps=cfg['steps'],skipped_global_batches=skipped,
            samples_completed=sum('error' not in r for r in last_samples),
            samples_failed=sum('error' in r for r in last_samples)))
        from utils.moxing_io import copy_directory
        for root in out.roots:
            try: copy_directory(str(out.local/'tb'),root.rstrip('/')+'/tb')
            except Exception as exc: print('[TensorBoard mirror warning] '+repr(exc),flush=True)
    barrier()
    if last_samples and not any('error' not in r for r in last_samples):
        raise RuntimeError('Optimizer state saved, but all requested previews failed; inspect samples/*/failure.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('config','model','data','output'): p.add_argument('--'+flag,required=True)
    p.add_argument('--arm',choices=('native','sft','region_sft','rae','rae_balanced','rae_detached'),required=True)
    p.add_argument('--data-root',action='append',default=[]); p.add_argument('--root',action='append',default=[])
    p.add_argument('--rae'); p.add_argument('--stop-after',type=int,default=0)
    p.add_argument('--controls',help='Optional source-only manual box JSON; evaluated separately at final sampling')
    p.add_argument('--eval-checkpoint',help='Evaluate an existing checkpoint relative to this arm output; no optimizer steps')
    a=p.parse_args(); cfg=json.loads(Path(a.config).read_text())
    device,rank,world=distributed()
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
    data=ArtifactStore(a.data,a.data_root); out=ArtifactStore(a.output,a.root)
    bank=data.json('bank.json',required=True)
    actual=validate_assets(a.model) if int(os.environ.get('LOCAL_RANK',0))==0 else None
    if any(s!=bank['model_signature'] for s in gather(actual,world) if s is not None):
        raise ValueError('Training native weights differ from cached weights')
    for key in ('height','width','max_side','resize_policy','frames','fps','benchmark_fps','rae_frames','negative_prompt'):
        if cfg[key]!=bank['config'][key]: raise ValueError('Cache geometry/text mismatch: '+key)
    teacher_bank=decoder=None
    if a.arm.startswith('rae'):
        teacher_bank=data.json('teacher_bank.json',required=True)
        if teacher_bank['native_identity']!=bank['identity']: raise ValueError('RAE targets belong to another native bank')
        from utils.editing_data import file_hash
        if file_hash(a.rae)!=teacher_bank['signature']['rae']: raise ValueError('Frozen decoder differs from target teacher')
        decoder=load_teacher(a.rae,device)
    identity=digest(dict(schema='vace-edit-train-v1',cfg=cfg,arm=a.arm,bank=bank['identity'],
        teacher=teacher_bank['identity'] if teacher_bank else None,world=world))
    if any(x!=identity for x in gather(identity,world)): raise ValueError('Nodes disagree on run identity')
    model=load_backbone(a.model,device)
    if a.arm=='native':
        results=evaluate(a,cfg,bank,data,out,model,device,rank,world,0,'native',identity)
        if not any('error' not in r for r in results): raise RuntimeError('All native previews failed')
    else:
        torch.manual_seed(cfg['seed'])
        if device.type!='cpu': getattr(torch,device.type).manual_seed_all(cfg['seed'])
        model=EditingModel(model,cfg,teacher_bank is not None).to(device)
        if a.eval_checkpoint:
            saved=out.load(a.eval_checkpoint,identity,True)
            model.restore(saved['model'])
            results=evaluate(a,cfg,bank,data,out,model,device,rank,world,saved['step'],'raw',identity)
            model.restore(saved['ema'])
            results+=evaluate(a,cfg,bank,data,out,model,device,rank,world,saved['step'],'ema',identity)
            if not any('error' not in r for r in results): raise RuntimeError('All checkpoint previews failed')
        else:
            run_training(a,cfg,bank,data,out,identity,device,rank,world,model,teacher_bank,decoder)
    if dist.is_initialized(): dist.destroy_process_group()


if __name__=='__main__': main()
