"""Stage70: final-video calibration attribution and shared-SFT operator tests.

Stage69 is a read-only parent. No training, new anchors, cohort selection or
future-target conditioning. All new states/results live in a fresh namespace.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import torch

from utils.scene_run import ArtifactStore
from utils.editing_data import digest, file_hash
from utils.editing_train import distributed, progress, load_teacher, load_encoder, write_video
from utils.editing_attribution import (ARMS, CALIBRATION_ARMS, OPERATOR_ARMS,
    arm_spec, supported_rms, smooth_effect, fit_video_vae, MagnitudeProjector, NativeEffectFeedback)
from utils.vace_state_data import calibrate_edit_states, align_rae_state
from utils.editing_state_feedback import load_decoder_only, expand_source_correction, map_square_correction
from utils.editing_joint_state import JointEditProjector
from utils.editing_paper_controls import ReferenceSFT, PaperFeedback
from utils.vace_edit_runtime import load_backbone, load_vae, decode_native, validate_assets
from utils.vace_state_runtime import StateConditionedVACE, sample_state_native
from train_vace_edit_state import decoded_reference, arm_store, publish_lossless
from edit_reference_video import (publish_case, recovered_case, crop_video, tensor_hash, failure)
from paper_editing_suite import code_signature as parent_code_signature, release, save_phase
from utils.native_audit_io import seal

REPO=Path(__file__).resolve().parent


def signature():
    return dict(parent_code_signature(), **{n:file_hash(REPO/n) for n in
        ('audit_edit_attribution.py','utils/editing_attribution.py')})


def parent(a):return ArtifactStore(a.parent,a.parent_root)
def states(a):return ArtifactStore(a.states,a.state_root)


def validate_config(cfg):
    if tuple(cfg['arms'])!=ARMS or cfg['frames']!=17 or cfg['fps']!=8:
        raise ValueError('Use the fixed 17-frame/8fps attribution matrix')
    if not cfg['seeds'] or len(set(cfg['seeds']))!=len(cfg['seeds']):raise ValueError('Distinct seeds required')
    if cfg['sample_steps']<2 or cfg['rgb_smoothing_sigma']<=0:raise ValueError('Invalid sampling/smoothing settings')


def checked(a,cfg,out):
    m=out.json('manifest.json',True)
    if m['config']!=cfg or m['code']!=signature():raise ValueError('Stage70 contract changed; use a fresh namespace')
    if parent(a).json('manifest.json',True)['identity']!=m['parent']['identity']:
        raise ValueError('Parent input identity changed')
    return m


def assignments(rows,rank,world):
    # A three-case smoke exercises one device on EACH of the three nodes.
    if len(rows)<world and world>=24:
        return [r for i,r in enumerate(rows) if ((i%3)*8+i//3)%world==rank]
    return rows[rank::world]


def tick(out,phase,rank,rows,started,new_frames):
    progress(out,phase+'_r'+str(rank),dict(phase=phase,attempted=len(rows),
        completed=sum(r.get('status')=='completed' for r in rows),
        DI_throughput=new_frames/max(time.monotonic()-started,1e-6),
        throughput_unit='new completed frames/s/rank including phase IO; cache hits excluded'))


def index(a,cfg,out):
    p=parent(a); pm=p.json('manifest.json',True); cohort=p.json('cohort.json',True)
    ids=cohort['ablation_ids'][:3] if cfg.get('smoke') else cohort['ablation_ids']
    rows=[r for r in cohort['records'] if r['id'] in ids]
    if not rows or len(rows)!=len(ids):raise ValueError('Parent fixed subset incomplete')
    checkpoint={}
    for family,store in [('rae',arm_store(states(a),'rae')),('sft',arm_store(p,'sft'))]:
        checkpoint[family]=store.json('selected.pt.json',True)['sha256']
    if checkpoint['rae']!=pm['selected']['rae']:raise ValueError('RAE selected checkpoint changed')
    m=dict(schema='editing-attribution-v1',config=cfg,code=signature(),parent=pm,
        parent_cohort=cohort['identity'],records=rows,checkpoint=checkpoint,
        protocol='fixed Stage69 subset; source/reference only; no training or new anchor',
        operator_scope='shared reference-SFT without state bridge; initial effect fitting, no online solve; not retrained VAE system',
        smoothing='predeclared spatial sigma; NOT bandwidth-matched or test-target-tuned',
        cost_scope='equal event schedule; measured cost, NOT wall-clock-matched')
    m['identity']=digest(m)
    previous=out.json('manifest.json')
    if previous and previous['identity']!=m['identity']:raise ValueError('Changed inputs/checkpoints; choose a fresh namespace')
    out.write_json('manifest.json',m)
    if a.source:
        from prepare_vace_edit import stage_data
        stage_data(a.source,a.data)
    # One CPU process per node stages the small selected deltas.
    for family,store in [('rae',arm_store(states(a),'rae')),('sft',arm_store(p,'sft'))]:
        identity=pm['stage65'] if family=='rae' else p.json('sft_contract.json',True)['identity']
        store.load('selected.pt',digest([identity,family]),True)
    print('[attribution frozen]',json.dumps(dict(identity=m['identity'],cases=len(rows),arms=list(ARMS))),flush=True)


def valid_mask(inputs):
    source=inputs['source'];v=torch.zeros(len(source),1,*source.shape[-2:])
    y,x,h,w=inputs['valid_rect'];v[:,:,y:y+h,x:x+w]=1
    return v


def prepare(a,cfg,out,device,rank,world):
    from audit_edit_transport import encode_window, rae_decoder
    m=checked(a,cfg,out); p=parent(a); pm=m['parent']; rows=[];frames=0;start=time.monotonic()
    teacher=encoder=None
    if file_hash(a.rae)!=pm['teacher']['rae'] or file_hash(a.encoder)!=pm['teacher']['encoder']:
        raise ValueError('Teacher weights differ from frozen parent')
    for row in assignments(m['records'],rank,world):
        key=row['id'];name='calibration/'+key+'.pt'
        try:
            if out.load(name,m['identity']) is None:
                old=p.load('prepared/'+key+'.pt',pm['identity'],True)
                inputs=old['inputs'];ref=decoded_reference(old['reference'])
                if teacher is None:teacher,encoder=load_teacher(a.rae,device).float(),load_encoder(a.encoder,device)
                source=inputs['source'].float()/255;proposal=inputs['proposal'].float()/255
                source_z=inputs['source_z'].float()
                with torch.no_grad():raw=encode_window(teacher,encoder,proposal[None].to(device))[0].detach().float().cpu()
                # Both branches originate from the SAME raw encoded proposal.
                calibrated,diagnostics=calibrate_edit_states(rae_decoder(teacher),source_z.to(device),raw.to(device),
                    source,proposal,inputs['masks'],inputs['valid_rect'],pm['base_config'])
                if diagnostics['fallback_count']:raise RuntimeError('Initial calibration failed: '+str(diagnostics))
                source_z=source_z.half().float();raw=raw.half().float();calibrated=calibrated.half().float()
                if any(not torch.isfinite(x).all() for x in (source_z,raw,calibrated)):
                    raise FloatingPointError('Calibration state overflows the FP16 cache format')
                prepared=dict(source_z=source_z,raw=raw,calibrated=calibrated,
                    off_state=align_rae_state(source_z,raw,inputs['anchor_z'],inputs,ref).half(),
                    on_state=align_rae_state(source_z,calibrated,inputs['anchor_z'],inputs,ref).half(),
                    diagnostics=diagnostics,source_reference_only=True,
                    parent_state_mae=float((align_rae_state(source_z,calibrated,inputs['anchor_z'],inputs,ref)-old['rae_state'].float()).abs().mean()))
                decode=rae_decoder(teacher); effect=[]
                with torch.no_grad():
                    for z,e in zip(source_z,calibrated):
                        effect.append((decode(e[None,None].to(device))-decode(z[None,None].to(device)))[0].cpu())
                square_effect=torch.stack(effect)
                native_effect=map_square_correction(expand_source_correction(square_effect,inputs,len(ref['source'])),inputs['valid_rect'],ref['spatial'])
                prepared['operator_rae_effect']=native_effect
                out.save(name,prepared,m['identity'])
                out.write_json('calibration/'+key+'.json',dict(id=key,diagnostics=diagnostics,
                    parent_state_mae=prepared['parent_state_mae'],raw_state_sha=tensor_hash(raw),calibrated_state_sha=tensor_hash(calibrated),
                    raw_delta_rms=float((raw-source_z).square().mean().sqrt()),
                    calibrated_delta_rms=float((calibrated-source_z).square().mean().sqrt())))
                frames+=cfg['frames']
            rows.append(dict(id=key,status='completed'))
        except Exception as exc:rows.append(failure(out,'prepare',row,exc))
        tick(out,'prepare',rank,rows,start,frames)
    save_phase(out,'prepare',rows,len(m['records']),rank,world)


def prepare_vae(a,cfg,out,device,rank,world):
    m=checked(a,cfg,out);p=parent(a);rows=[];frames=0;start=time.monotonic();vae=None
    for row in assignments(m['records'],rank,world):
        key=row['id'];name='vae/'+key+'.pt'
        try:
            if out.load(name,m['identity']) is None:
                item=p.load('prepared/'+key+'.pt',m['parent']['identity'],True);ref=decoded_reference(item['reference'])
                if vae is None:
                    if validate_assets(a.model)!=m['parent']['assets']:raise ValueError('Native assets changed')
                    vae=load_vae(a.model,device)
                valid=torch.zeros(len(ref['source']),1,*ref['source'].shape[-2:])
                y,x,h,w=ref['spatial']['valid_rect'];valid[:,:,y:y+h,x:x+w]=1
                effect,diagnostics=fit_video_vae(vae,ref['source'],ref['reference'],valid,cfg['vae_calibration'],device)
                out.save(name,dict(effect=effect,diagnostics=diagnostics),m['identity'])
                out.write_json('vae/'+key+'.json',dict(id=key,**diagnostics))
                frames+=cfg['frames']
            rows.append(dict(id=key,status='completed'))
        except Exception as exc:rows.append(failure(out,'prepare_vae',row,exc));release(device)
        tick(out,'prepare_vae',rank,rows,start,frames)
    save_phase(out,'prepare_vae',rows,len(m['records']),rank,world)


def projector(inputs,cal,decoder,options,device,calibrated,online):
    options=dict(options,device=str(device))
    if not online:options['iterations']=0
    edited=cal['calibrated' if calibrated else 'raw']
    return JointEditProjector(decode=decoder,source_z=cal['source_z'].to(device),
        source=inputs['source'].float().to(device)/255,anchor=inputs['anchor'].float().to(device)/255,
        masks=inputs['masks'].float().to(device),confidence=inputs['confidence'].float().to(device),
        cfg=options,initial_delta=(edited-cal['source_z']).to(device),
        proposal=inputs['proposal'].float().to(device)/255,edges=inputs['edges'],
        scale=inputs['scale'].to(device),valid=valid_mask(inputs).to(device))


def render(a,cfg,out,device,rank,world):
    m=checked(a,cfg,out);p=parent(a);pm=m['parent'];spec=arm_spec(a.arm)
    rows=[];frames=0;start=time.monotonic();model=vae=decoder=None
    settings=dict(pm['base_config'],sample_steps=cfg['sample_steps'])
    for row in assignments(m['records'],rank,world):
        for seed_base in cfg['seeds']:
            key=row['id'];folder=f'render/{a.arm}/seed{seed_base}/{key}'
            feedback=proj=features=latent=None
            try:
                old=p.load('prepared/'+key+'.pt',pm['identity'],True);ref=decoded_reference(old['reference'])
                native=p.load('native/'+key+'.pt',pm['identity'],True);text=p.load('text/'+key+'.pt',pm['identity'],True)
                seed=seed_base+int(digest(key)[:8],16)%1000000
                input_id=digest([pm['identity'],key,seed,tensor_hash(ref['source']),tensor_hash(ref['reference'])])
                generation=digest([m['identity'],input_id,a.arm,m['checkpoint'][spec['family']],settings])
                payload,record=recovered_case(out,folder,m['identity'])
                reused=payload is not None
                if reused and payload['generation_identity']!=generation:raise ValueError('Saved generation identity changed')
                if reused and spec['feedback'] and (not payload['events'] or any(e.get('fallback') for e in payload['events'])):
                    out.write_json(folder+'/previous_failure.json',dict(events=payload['events'],generation_identity=generation))
                    payload=record=None;reused=False
                if payload is None:
                    setup=time.monotonic()
                    # Resolve inputs before loading a 14B model, so a missing VAE
                    # cache only fails its own arm without expensive model setup.
                    cal=None
                    if not spec['operator'] or a.arm=='op_rae':cal=out.load('calibration/'+key+'.pt',m['identity'],True)
                    effect=None
                    if a.arm=='op_vae':effect=out.load('vae/'+key+'.pt',m['identity'],True)['effect']
                    elif a.arm=='op_rae':effect=cal['operator_rae_effect']
                    elif a.arm in ('op_rgb','op_rgb_smooth'):
                        effect=ref['reference']-ref['source']
                        if a.arm=='op_rgb_smooth':effect=smooth_effect(effect,cfg['rgb_smoothing_sigma'])
                    if model is None:
                        if validate_assets(a.model)!=pm['assets']:raise ValueError('Native assets changed')
                        core=load_backbone(a.model,device)
                        if spec['operator']:
                            model=ReferenceSFT(core,settings).to(device);store=arm_store(p,'sft')
                            identity=p.json('sft_contract.json',True)['identity']
                        else:
                            model=StateConditionedVACE(core,settings,settings['state_channels']['rae']).to(device)
                            store=arm_store(states(a),'rae');identity=pm['stage65']
                        chosen=store.load('selected.pt',digest([identity,spec['family']]),True)
                        if store.json('selected.pt.json',True)['sha256']!=m['checkpoint'][spec['family']]:raise ValueError('Selected checkpoint changed')
                        model.restore(chosen['model']);model.eval().requires_grad_(False);del chosen
                    if vae is None:vae=load_vae(a.model,device)
                    events=[]
                    def observer(event):
                        events.append(dict(event));step=event['step_index'];base=folder+f'/feedback/step_{step:03d}'
                        if event.get('applied'):
                            from refine_vace_edit_state import _cpu_snapshot
                            snapshot=_cpu_snapshot(dict(reference=feedback.last_reference,projection=feedback.last_projection,
                                state=feedback.state,event=event,generation_identity=generation))
                            out.save(base+'.pt',snapshot,m['identity'])
                            rect=ref['spatial']['valid_rect']
                            write_video(out.local/(base+'.mp4'),crop_video(ref['source'],rect),crop_video(snapshot['reference']['reference'],rect),
                                cfg['fps'],labels=['SOURCE','LIVE CORRECTION'])
                            seal(out.local,base+'.json',m['identity'],[base+'.mp4'],dict(event=event))
                            out.publish([base+'.mp4',base+'.json'])
                        progress(out,'feedback_r'+str(rank),dict(event,id=key,arm=a.arm))
                    if spec['operator']:
                        if spec['feedback']:feedback=NativeEffectFeedback(vae,ref,effect,cfg['feedback'],observer)
                    else:
                        if decoder is None:
                            if file_hash(a.rae)!=pm['teacher']['rae']:raise ValueError('Decoder checkpoint changed')
                            decoder=load_decoder_only(a.rae,device,True)
                        inputs=dict(old['inputs'],source_z=cal['source_z'])
                        features=cal['on_state' if spec['calibrated'] or spec['fixed_bridge'] else 'off_state'].float().to(device)
                        proj=projector(inputs,cal,decoder,cfg['feedback']['solver'],device,spec['calibrated'],spec['online'])
                        if spec['magnitude']:
                            control=projector(inputs,cal,decoder,cfg['feedback']['solver'],device,True,False)
                            proj=MagnitudeProjector(proj,control,inputs['masks'].float()*valid_mask(inputs))
                        feedback=PaperFeedback(vae,proj,inputs,ref,features,
                            dict(cfg['feedback'],update_bridge_state=not spec['fixed_bridge'],update_native_context=False),'rae',observer=observer)
                    setup_seconds=time.monotonic()-setup
                    if device.type!='cpu':
                        getattr(torch,device.type).synchronize();getattr(torch,device.type).reset_peak_memory_stats()
                    sampling=time.monotonic()
                    with torch.no_grad():
                        latent=sample_state_native(model,native['context'].to(device),text['text'].to(device),text['negative'].to(device),
                            features,settings,seed,use_state=not spec['operator'],velocity_callback=feedback)
                        generated=((decode_native(vae,latent,1).permute(1,0,2,3).float()+1)/2).clamp(0,1).cpu()
                    if device.type!='cpu':getattr(torch,device.type).synchronize()
                    if generated.shape!=ref['source'].shape or not torch.isfinite(generated).all():raise ValueError('Invalid output video')
                    payload=dict(generated=generated,spatial=ref['spatial'],seed=seed,seed_base=seed_base,
                        input_identity=input_id,generation_identity=generation,checkpoint=m['checkpoint'][spec['family']],
                        events=events,setup_seconds=setup_seconds,sampling_seconds=time.monotonic()-sampling,
                        peak_allocated_gib=getattr(torch,device.type).max_memory_allocated()/2**30 if device.type!='cpu' else None)
                bad=[e for e in payload['events'] if e.get('fallback')]
                if spec['feedback'] and not payload['events']:bad=[dict(error='No feedback events')]
                metrics=dict(id=key,group=row['group'],arm=a.arm,seed_base=seed_base,status='completed' if not bad else 'feedback_failed',
                    **{k:payload[k] for k in ('seed','input_identity','generation_identity','checkpoint','sampling_seconds','setup_seconds','peak_allocated_gib')},
                    feedback_events=payload['events'],fallback_count=len(bad),reused=reused)
                if record is None:
                    publish_case(out,folder,m['identity'],payload,metrics,[ref['source'],ref['reference'],payload['generated']],
                        ['SOURCE','SHARED REFERENCE',a.arm],ref['spatial']['valid_rect'],cfg['fps'])
                result=publish_lossless(out,folder,m['identity'],payload,cfg['fps'])
                rows.append(result)
                if not reused and not bad:frames+=cfg['frames']
            except Exception as exc:rows.append(dict(failure(out,'render_'+a.arm,row,exc),arm=a.arm,seed_base=seed_base))
            finally:feedback=proj=features=latent=None;release(device)
            tick(out,'render_'+a.arm,rank,rows,start,frames)
    save_phase(out,'render_'+a.arm,rows,len(m['records'])*len(cfg['seeds']),rank,world)


def evaluate(a,cfg,out,device,rank,world):
    from utils.editing_paper_data import load_evaluation_labels
    from utils.editing_paper_metrics import evaluate_videos,optional_lpips,evaluation_context
    m=checked(a,cfg,out);p=parent(a);pm=m['parent'];rows=[];frames=0;start=time.monotonic()
    bank=ArtifactStore(a.bank,a.bank_root).json('bank.json',True)
    perceptual,status=optional_lpips(device,enabled=cfg['lpips'])
    for row in assignments(m['records'],rank,world):
        try:
            item=p.load(row['source_file'],pm['identity'],True)
            target=load_evaluation_labels(a.data,row,bank,item);rect=item['spatial']['valid_rect']
            source=crop_video(item['source'].float()/255,rect);target=crop_video(target.float()/255,rect)
            context=evaluation_context(source,target)
            for arm in cfg['arms']:
                for seed in cfg['seeds']:
                    name=f'evaluation/{arm}/seed{seed}/{row["id"]}.json'
                    try:
                        record=out.json(name)
                        if not record or record.get('status')!='completed':
                            folder=f'render/{arm}/seed{seed}/{row["id"]}'
                            meta=out.json(folder+'/metrics.json',True)
                            if meta['status']!='completed':raise ValueError('Feedback failed; excluded from success means')
                            payload=out.load(folder+'/result.pt',m['identity'],True)
                            values=evaluate_videos(source,target,crop_video(payload['generated'],rect),perceptual,device,context)
                            record=dict(id=row['id'],group=row['group'],arm=arm,seed_base=seed,status='completed',
                                generation_identity=payload['generation_identity'],lpips=status,metrics=values)
                            out.write_json(name,record);frames+=cfg['frames']
                        rows.append(record)
                    except Exception as exc:rows.append(dict(id=row['id'],arm=arm,seed_base=seed,status='failed',error=repr(exc)))
        except Exception as exc:rows.append(failure(out,'evaluation_source',row,exc))
        tick(out,'evaluate',rank,rows,start,frames)
    save_phase(out,'evaluate',rows,len(m['records'])*len(cfg['arms'])*len(cfg['seeds']),rank,world)


def summarize(a,cfg,out):
    from utils.editing_publication import reconcile_case
    from utils.editing_paper_metrics import aggregate
    m=checked(a,cfg,out);evaluations=(out.json('phases/evaluate.json') or {}).get('records',[])
    tables={};media=[];failures=[];costs={}
    for title,arms,baseline in [('calibration',CALIBRATION_ARMS,'cal_on_online_on'),('operator',OPERATOR_ARMS,'op_rae')]:
        chosen=[r for r in evaluations if r.get('arm') in arms]
        # Existing aggregator uses rae_full as its paired reference key.
        mapped=[dict(r,arm='rae_full' if r['arm']==baseline else r['arm']) for r in chosen]
        tables[title]=dict(reference_arm=baseline,**aggregate(mapped,['rae_full' if arm==baseline else arm for arm in arms],bootstrap=10000))
    for arm in cfg['arms']:
        for row in m['records']:
            for seed in cfg['seeds']:
                folder=f'render/{arm}/seed{seed}/{row["id"]}'
                try:
                    receipt=out.json(folder+'/receipt.json',True)
                    roots=reconcile_case(out,folder+'/receipt.json',m['identity'])
                    if receipt['metadata']['status']!='completed':raise ValueError('Feedback failure')
                    meta=receipt['metadata']
                    costs.setdefault(arm,[]).append({k:meta[k] for k in ('sampling_seconds','setup_seconds','peak_allocated_gib')})
                    for e in receipt['metadata']['feedback_events']:
                        if e.get('applied'):
                            roots+=reconcile_case(out,folder+f'/feedback/step_{e["step_index"]:03d}.json',m['identity'])
                    media.append(dict(arm=arm,id=row['id'],seed=seed,roots=roots,ok=all(r['ok'] for r in roots)))
                except Exception as exc:failures.append(dict(arm=arm,id=row['id'],seed=seed,error=repr(exc)))
    expected=len(cfg['arms'])*len(m['records'])*len(cfg['seeds'])
    successful=sum(r.get('status')=='completed' for r in evaluations)
    summary=dict(expected=expected,evaluated=successful,failures=failures,tables=tables,costs=costs,
        complete=successful==expected and not failures and all(r['ok'] for r in media),
        limitations=[m['operator_scope'],m['smoothing'],m['cost_scope'],'No new human ratings or external benchmark results.'])
    out.write_json('media_sync.json',dict(cases=media,failures=failures))
    out.write_json('summary.json',summary)
    if not summary['complete']:raise RuntimeError('Attribution suite incomplete; successful outputs retained; use resume')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase',required=True,choices=('index','prepare','prepare_vae','render','evaluate','summarize'))
    parser.add_argument('--arm',choices=ARMS)
    for n in ('config','output','parent','states','model','rae','encoder','bank','data','source'):
        parser.add_argument('--'+n,default='')
    for n in ('root','parent-root','state-root','bank-root'):parser.add_argument('--'+n,action='append',default=[])
    a=parser.parse_args();cfg=json.loads(Path(a.config).read_text());validate_config(cfg);out=ArtifactStore(a.output,a.root)
    if a.phase=='index':return index(a,cfg,out)
    if a.phase=='summarize':
        if int(os.environ.get('NODE_RANK','0'))==0:return summarize(a,cfg,out)
        return
    device,rank,world=distributed()
    try:dict(prepare=prepare,prepare_vae=prepare_vae,render=render,evaluate=evaluate)[a.phase](a,cfg,out,device,rank,world)
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
