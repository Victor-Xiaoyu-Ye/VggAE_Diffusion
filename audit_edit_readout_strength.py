"""Stage72: predeclared strength curves, then the remaining shared-SFT sources.

Stage69/70 remote parents are read-only. Reused outputs keep their ORIGINAL
generation identity and hash provenance; a new job identity tracks this suite.
No training, new anchors, outcome-based cohort selection, or target conditioning.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch

import audit_edit_attribution as legacy
from utils.scene_run import ArtifactStore
from utils.editing_data import digest, file_hash
from utils.editing_train import distributed, load_teacher, load_encoder, write_video
from utils.editing_attribution import OPERATOR_ARMS, smooth_effect, fit_video_vae
from utils.editing_readout_strength import (validate_config, specifications, select_cohorts,
    records_for, arms_for, ObservedEffectFeedback, strength_metrics, matched_strength)
from utils.vace_edit_runtime import load_backbone, load_vae, decode_native, validate_assets
from utils.editing_paper_controls import ReferenceSFT
from utils.vace_state_runtime import sample_state_native
from utils.vace_state_data import calibrate_edit_states
from utils.editing_state_feedback import expand_source_correction, map_square_correction
from train_vace_edit_state import decoded_reference, arm_store, publish_lossless
from edit_reference_video import publish_case, recovered_case, crop_video, tensor_hash, failure
from paper_editing_suite import release
from utils.native_audit_io import seal, recover_case

REPO=Path(__file__).resolve().parent
PROTOCOL_KEYS=('frames','fps','seeds','sample_steps','rgb_smoothing_sigma','vae_calibration','feedback')


def signature():
    return dict(legacy.signature(), **{n:file_hash(REPO/n) for n in
        ('audit_edit_readout_strength.py','utils/editing_readout_strength.py')})


def parent(a): return ArtifactStore(a.parent,a.parent_root)
def attribution(a): return ArtifactStore(a.attribution,a.attribution_root)


def check_parent(a,cfg,out):
    p=parent(a); prior=attribution(a)
    pm=p.json('manifest.json',True); old=prior.json('manifest.json',True)
    cohort=p.json('cohort.json',True)
    legacy.check_contract(old,old['config'],out)
    if old['parent']['identity']!=pm['identity'] or old['parent_cohort']!=cohort['identity']:
        raise ValueError('Stage69/70 source identities disagree')
    # Smoke changes sampling length ONLY, and never imports full-run outputs.
    for key in PROTOCOL_KEYS:
        if key=='sample_steps' and cfg.get('smoke'): continue
        if cfg[key]!=old['config'][key]:raise ValueError('Frozen operator protocol changed: '+key)
    expected=[r for r in cohort['records'] if r['id'] in cohort['ablation_ids']]
    if expected!=old['records']:raise ValueError('Stage70 rows differ from fixed parent subset')
    chosen=arm_store(p,'sft').json('selected.pt.json',True)['sha256']
    if chosen!=old['checkpoint']['sft']:raise ValueError('Reference-SFT weights changed')
    return pm,old,cohort


def checked(a,cfg,out):
    m=out.json('manifest.json',True)
    if m['config']!=cfg or m['code']!=signature():raise ValueError('Stage72 contract changed; use a fresh READOUT_NAMESPACE')
    if parent(a).json('manifest.json',True)['identity']!=m['parent']['identity']:
        raise ValueError('Parent source identity changed')
    if attribution(a).json('manifest.json',True)['identity']!=m['attribution']['identity']:
        raise ValueError('Stage70 identity changed')
    return m


def index(a,cfg,out):
    pm,old,cohort=check_parent(a,cfg,out);subset,extension=select_cohorts(cohort,cfg)
    selected={r['id'] for r in subset+extension}
    m=dict(schema=cfg['schema'],config=cfg,code=signature(),parent=pm,attribution=old,
        records=[r for r in cohort['records'] if r['id'] in selected],
        subset_ids=[r['id'] for r in subset],extension_ids=[r['id'] for r in extension],
        checkpoint=old['checkpoint']['sft'],parent_cohort=cohort['identity'],
        protocol='same reference-SFT, no bridge; Stage70 initial readouts, no online solve',
        strength_protocol='fixed alpha .15/.35/.55; post-cap dose and future-only source/reference metrics; no test-selected optimum')
    m['identity']=digest(m)
    previous=out.json('manifest.json')
    if previous and previous!=m:raise ValueError('Changed run contract; choose a fresh READOUT_NAMESPACE')
    out.write_json('manifest.json',m)
    expected=sum(len(arms_for(m,r))*len(cfg['seeds']) for r in m['records'])
    reusable=0 if cfg.get('smoke') else len(subset)*len(OPERATOR_ARMS)*len(cfg['seeds'])
    out.write_json('plan.json',dict(expected_outputs=expected,planned_imports=reusable,
        planned_new_outputs=expected-reusable,subset=len(subset),extension=len(extension),
        weights_trained=False,old_remote_roots_read_only=True))
    # Source archive staging is for evaluation only. Selection/anchors stay fixed.
    if a.source:
        from prepare_vace_edit import stage_data
        stage_data(a.source,a.data)
    identity=parent(a).json('sft_contract.json',True)['identity']
    arm_store(parent(a),'sft').load('selected.pt',digest([identity,'sft']),True)
    print('[readout plan] '+json.dumps(out.json('plan.json')),flush=True)


def finish(out,phase,rows,requested,rank,world):
    all_rows=legacy.finish_phase(out,phase,rows,requested,rank,world)
    if any(r.get('status')!='completed' for r in all_rows) or len(all_rows)!=requested:
        raise RuntimeError('Incomplete '+phase+'; independent arms may continue; successful artifacts retained')


def prepare(a,cfg,out,device,rank,world):
    from audit_edit_transport import encode_window, rae_decoder
    m=checked(a,cfg,out);p=parent(a);prior=attribution(a);pm=m['parent']
    records=records_for(m,a.cohort);reports=[];new_frames=0;start=time.monotonic();teacher=encoder=vae=None
    kind='calibration' if a.phase=='prepare' else 'vae';phase=a.phase+'_'+a.cohort
    for row in legacy.assignments(records,rank,world):
        key=row['id'];name=f'{kind}/{key}.pt'
        try:
            cached=out.load(name,m['identity'])
            if cached is None:
                imported=None
                if key in m['subset_ids']:
                    imported=prior.load(name,m['attribution']['identity'])
                if imported is not None:
                    out.save(name,imported,m['identity'])
                    out.write_json(f'{kind}/{key}.json',dict(id=key,imported=True,
                        origin_identity=m['attribution']['identity'],origin_receipt=prior.json(name+'.json',True)))
                else:
                    old=p.load('prepared/'+key+'.pt',pm['identity'],True);ref=decoded_reference(old['reference'])
                    if kind=='vae':
                        if vae is None:
                            if validate_assets(a.model)!=pm['assets']:raise ValueError('Native VAE assets changed')
                            vae=load_vae(a.model,device)
                        valid=torch.zeros(len(ref['source']),1,*ref['source'].shape[-2:])
                        y,x,h,w=ref['spatial']['valid_rect'];valid[:,:,y:y+h,x:x+w]=1
                        effect,diagnostics=fit_video_vae(vae,ref['source'],ref['reference'],valid,cfg['vae_calibration'],device)
                        payload=dict(effect=effect,diagnostics=diagnostics)
                    else:
                        if teacher is None:
                            if file_hash(a.rae)!=pm['teacher']['rae'] or file_hash(a.encoder)!=pm['teacher']['encoder']:
                                raise ValueError('Frozen trained RAE/encoder weights changed')
                            teacher,encoder=load_teacher(a.rae,device).float(),load_encoder(a.encoder,device)
                        inputs=old['inputs'];source=inputs['source'].float()/255;proposal=inputs['proposal'].float()/255
                        source_z=inputs['source_z'].float()
                        with torch.no_grad():raw=encode_window(teacher,encoder,proposal[None].to(device))[0].detach().float().cpu()
                        calibrated,diagnostics=calibrate_edit_states(rae_decoder(teacher),source_z.to(device),raw.to(device),
                            source,proposal,inputs['masks'],inputs['valid_rect'],pm['base_config'])
                        if diagnostics['fallback_count']:raise RuntimeError('Initial RAE calibration failed: '+str(diagnostics))
                        source_z,raw,calibrated=[v.half().float() for v in (source_z,raw,calibrated)]
                        if any(not torch.isfinite(v).all() for v in (source_z,raw,calibrated)):
                            raise FloatingPointError('RAE FP16 cache overflow')
                        decode=rae_decoder(teacher);effects=[]
                        with torch.no_grad():
                            for z,e in zip(source_z,calibrated):
                                effects.append((decode(e[None,None].to(device))-decode(z[None,None].to(device)))[0].cpu())
                        effect=map_square_correction(expand_source_correction(torch.stack(effects),inputs,len(ref['source'])),
                            inputs['valid_rect'],ref['spatial'])
                        payload=dict(source_z=source_z,raw=raw,calibrated=calibrated,operator_rae_effect=effect,
                            diagnostics=diagnostics,source_reference_only=True)
                    if effect.shape!=ref['source'].shape or not torch.isfinite(effect).all():raise ValueError('Invalid prepared effect')
                    out.save(name,payload,m['identity'])
                    out.write_json(f'{kind}/{key}.json',dict(id=key,imported=False,diagnostics=diagnostics))
                    new_frames+=cfg['frames']
            reports.append(dict(id=key,status='completed'))
        except Exception as exc:reports.append(failure(out,phase,row,exc));release(device)
        legacy.tick(out,phase,rank,reports,start,new_frames)
    finish(out,phase,reports,len(records),rank,world)


def restore_import_media(prior,out,folder,identity,origin):
    """Copy receipt-verified previews; preserve original snapshots in the parent."""
    for event in origin['events']:
        if not event.get('applied'):continue
        stem=folder+f'/feedback/step_{event["step_index"]:03d}'
        receipt=recover_case(prior.local,stem+'.json',origin['identity'],prior.roots)
        if receipt is None:raise FileNotFoundError(stem+'.json')
        for name in receipt['files']:
            target=out.local/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(prior.local/name,target)
        seal(out.local,stem+'.json',identity,list(receipt['files']),dict(event=event,origin_identity=origin['identity']))
        out.publish([*receipt['files'],stem+'.json'])


def import_output(a,cfg,out,m,row,folder,input_id,settings):
    if cfg.get('smoke') or a.arm not in OPERATOR_ARMS or row['id'] not in m['subset_ids']:return None
    prior=attribution(a);old=m['attribution']
    payload=prior.load(folder+'/result.pt',old['identity'])
    if payload is None:return None
    receipt=recover_case(prior.local,folder+'/receipt.json',old['identity'],prior.roots)
    expected=digest([old['identity'],input_id,a.arm,old['checkpoint']['sft'],settings])
    if (receipt is None or receipt['metadata']['status']!='completed' or
            payload['generation_identity']!=expected or payload['input_identity']!=input_id or
            payload['checkpoint']!=m['checkpoint']):raise ValueError('Imported output identity/checkpoint/status mismatch')
    legacy.check_contract(dict(old,code=payload['runtime_code']),old['config'],out)
    if a.arm!='op_sft' and (not payload['events'] or any(e.get('fallback') or not e.get('applied') for e in payload['events'])):
        raise ValueError('Cannot import unsuccessful feedback')
    origin=dict(identity=old['identity'],folder=folder,generation_identity=expected,
        result_receipt=prior.json(folder+'/result.pt.json',True),events=payload['events'],roots=prior.roots,
        intermediate_snapshots='original Stage70 .pt files; verified preview media copied to this run')
    restore_import_media(prior,out,folder,m['identity'],origin)
    return dict(payload,origin=origin)


def render(a,cfg,out,device,rank,world):
    m=checked(a,cfg,out);p=parent(a);pm=m['parent'];spec=specifications(cfg)[a.arm]
    records=records_for(m,a.cohort,a.arm);reports=[];frames=0;start=time.monotonic();model=vae=None
    settings=dict(pm['base_config'],sample_steps=cfg['sample_steps']);phase='render_'+a.cohort+'_'+a.arm
    for row in legacy.assignments(records,rank,world):
        for seed_base in cfg['seeds']:
            key=row['id'];folder=f'render/{a.arm}/seed{seed_base}/{key}';feedback=latent=None
            try:
                old=p.load('prepared/'+key+'.pt',pm['identity'],True);ref=decoded_reference(old['reference'])
                seed=seed_base+int(digest(key)[:8],16)%1000000
                input_id=digest([pm['identity'],key,seed,tensor_hash(ref['source']),tensor_hash(ref['reference'])])
                job=digest([m['identity'],input_id,a.arm,m['checkpoint'],settings])
                payload,record=recovered_case(out,folder,m['identity']);reused=payload is not None
                if reused and payload['job_identity']!=job:raise ValueError('Saved job identity changed')
                if reused and spec['operator']!='op_sft' and (not payload['events'] or any(e.get('fallback') for e in payload['events'])):
                    out.write_json(folder+'/previous_failure.json',dict(events=payload['events']))
                    payload=record=None;reused=False
                if payload is None:payload=import_output(a,cfg,out,m,row,folder,input_id,settings)
                imported=payload is not None and 'origin' in payload
                if payload is None:
                    setup=time.monotonic();operator=spec['operator'];effect=None
                    if operator=='op_vae':effect=out.load('vae/'+key+'.pt',m['identity'],True)['effect']
                    elif operator=='op_rae':effect=out.load('calibration/'+key+'.pt',m['identity'],True)['operator_rae_effect']
                    elif operator in ('op_rgb','op_rgb_smooth'):
                        effect=ref['reference']-ref['source']
                        if operator=='op_rgb_smooth':effect=smooth_effect(effect,cfg['rgb_smoothing_sigma'])
                    native=p.load('native/'+key+'.pt',pm['identity'],True);text=p.load('text/'+key+'.pt',pm['identity'],True)
                    if model is None:
                        if validate_assets(a.model)!=pm['assets']:raise ValueError('Native assets changed')
                        model=ReferenceSFT(load_backbone(a.model,device),settings).to(device)
                        store=arm_store(p,'sft');identity=p.json('sft_contract.json',True)['identity']
                        chosen=store.load('selected.pt',digest([identity,'sft']),True)
                        if store.json('selected.pt.json',True)['sha256']!=m['checkpoint']:raise ValueError('SFT checkpoint changed')
                        model.restore(chosen['model']);model.eval().requires_grad_(False);del chosen
                    if vae is None:vae=load_vae(a.model,device)
                    events=[]
                    def observer(event):
                        events.append(dict(event));stem=folder+f'/feedback/step_{event["step_index"]:03d}'
                        if event.get('applied'):
                            from refine_vace_edit_state import _cpu_snapshot
                            snapshot=_cpu_snapshot(dict(reference=feedback.last_reference,projection=feedback.last_projection,
                                state=None,event=event,generation_identity=job))
                            out.save(stem+'.pt',snapshot,m['identity']);rect=ref['spatial']['valid_rect']
                            write_video(out.local/(stem+'.mp4'),crop_video(ref['source'],rect),crop_video(ref['reference'],rect),
                                crop_video(snapshot['reference']['reference'],rect),fps=cfg['fps'],
                                labels=['SOURCE','SHARED REFERENCE','LIVE CORRECTION'])
                            seal(out.local,stem+'.json',m['identity'],[stem+'.mp4'],dict(event=event))
                            out.publish([stem+'.mp4',stem+'.json'])
                        legacy.progress(out,'feedback_r'+str(rank),dict(event,id=key,arm=a.arm))
                    if effect is not None:
                        feedback=ObservedEffectFeedback(vae,ref,effect,dict(cfg['feedback'],writeback_alpha=spec['alpha']),observer)
                    setup_seconds=time.monotonic()-setup
                    if device.type!='cpu':
                        getattr(torch,device.type).synchronize();getattr(torch,device.type).reset_peak_memory_stats()
                    sampling=time.monotonic()
                    with torch.no_grad():
                        latent=sample_state_native(model,native['context'].to(device),text['text'].to(device),text['negative'].to(device),
                            None,settings,seed,use_state=False,velocity_callback=feedback)
                        generated=((decode_native(vae,latent,1).permute(1,0,2,3).float()+1)/2).clamp(0,1).cpu()
                    if device.type!='cpu':getattr(torch,device.type).synchronize()
                    if generated.shape!=ref['source'].shape or not torch.isfinite(generated).all():raise ValueError('Invalid output video')
                    payload=dict(generated=generated,spatial=ref['spatial'],seed=seed,seed_base=seed_base,
                        input_identity=input_id,generation_identity=job,checkpoint=m['checkpoint'],runtime_code=signature(),events=events,
                        setup_seconds=setup_seconds,sampling_seconds=time.monotonic()-sampling,
                        peak_allocated_gib=getattr(torch,device.type).max_memory_allocated()/2**30 if device.type!='cpu' else None)
                payload['job_identity']=job
                bad=any(e.get('fallback') or not e.get('applied') for e in payload['events'])
                if spec['operator']!='op_sft' and not payload['events']:bad=True
                meta=dict(id=key,group=row['group'],arm=a.arm,operator=spec['operator'],alpha=spec['alpha'],seed_base=seed_base,
                    status='feedback_failed' if bad else 'completed',job_identity=job,imported=imported,reused=reused,
                    origin=payload.get('origin'),feedback_events=payload['events'],strength=strength_metrics(payload['generated'],ref),
                    **{k:payload[k] for k in ('seed','input_identity','generation_identity','checkpoint','sampling_seconds','setup_seconds','peak_allocated_gib')})
                if record is None:
                    publish_case(out,folder,m['identity'],payload,meta,[ref['source'],ref['reference'],payload['generated']],
                        ['SOURCE','SHARED REFERENCE',a.arm],ref['spatial']['valid_rect'],cfg['fps'])
                reports.append(publish_lossless(out,folder,m['identity'],payload,cfg['fps']))
                if not reused and not imported and not bad:frames+=cfg['frames']
            except Exception as exc:reports.append(dict(failure(out,phase,row,exc),arm=a.arm,seed_base=seed_base))
            finally:feedback=latent=None;release(device)
            legacy.tick(out,phase,rank,reports,start,frames)
    finish(out,phase,reports,len(records)*len(cfg['seeds']),rank,world)


def evaluate(a,cfg,out,device,rank,world):
    from utils.editing_paper_data import load_evaluation_labels
    from utils.editing_paper_metrics import evaluate_videos,optional_lpips,evaluation_context
    m=checked(a,cfg,out);p=parent(a);pm=m['parent'];reports=[];frames=0;start=time.monotonic()
    records=records_for(m,a.cohort);bank=ArtifactStore(a.bank,a.bank_root).json('bank.json',True)
    perceptual,status=optional_lpips(device,enabled=cfg['lpips']);phase='evaluate_'+a.cohort
    for row in legacy.assignments(records,rank,world):
        try:
            item=p.load(row['source_file'],pm['identity'],True)
            target=load_evaluation_labels(a.data,row,bank,item);rect=item['spatial']['valid_rect']
            source=crop_video(item['source'].float()/255,rect);target=crop_video(target.float()/255,rect)
            context=evaluation_context(source,target)
            for arm in arms_for(m,row):
                for seed in cfg['seeds']:
                    name=f'evaluation/{arm}/seed{seed}/{row["id"]}.json';folder=f'render/{arm}/seed{seed}/{row["id"]}'
                    try:
                        meta=out.json(folder+'/metrics.json',True)
                        if meta['status']!='completed':raise ValueError('Feedback failed')
                        record=out.json(name)
                        if record and record.get('job_identity')!=meta['job_identity']:raise ValueError('Evaluation generation changed')
                        if (not record or record.get('status')!='completed' or
                                (cfg['lpips'] and record.get('lpips',{}).get('status')!='available')):
                            payload=out.load(folder+'/result.pt',m['identity'],True)
                            values=evaluate_videos(source,target,crop_video(payload['generated'],rect),perceptual,device,context)
                            values.update(meta['strength'])
                            events=[e for e in payload['events'] if e.get('applied')]
                            for key in ('clean_delta_rms','clean_delta_future_rms','capped_fraction'):
                                vals=[e[key] for e in events if key in e]
                                values['feedback_mean_'+key]=sum(vals)/len(vals) if vals else None
                            record=dict(id=row['id'],group=row['group'],arm=arm,seed_base=seed,status='completed',
                                job_identity=meta['job_identity'],generation_identity=payload['generation_identity'],lpips=status,metrics=values)
                            out.write_json(name,record);frames+=cfg['frames']
                        reports.append(record)
                    except Exception as exc:reports.append(dict(id=row['id'],arm=arm,seed_base=seed,status='failed',error=repr(exc)))
        except Exception as exc:reports.append(failure(out,phase,row,exc))
        legacy.tick(out,phase,rank,reports,start,frames)
    finish(out,phase,reports,sum(len(arms_for(m,r))*len(cfg['seeds']) for r in records),rank,world)


def summarize(a,cfg,out):
    from utils.editing_publication import reconcile_case
    from utils.editing_paper_metrics import aggregate
    m=checked(a,cfg,out);evaluations=[];failures=[];media=[];costs={};imports=0
    records=records_for(m,a.cohort);expected=sum(len(arms_for(m,r))*len(cfg['seeds']) for r in records)
    for row in records:
        for arm in arms_for(m,row):
            for seed in cfg['seeds']:
                folder=f'render/{arm}/seed{seed}/{row["id"]}'
                try:
                    receipt=out.json(folder+'/receipt.json',True);meta=receipt['metadata']
                    ev=out.json(f'evaluation/{arm}/seed{seed}/{row["id"]}.json',True)
                    if (meta['status']!='completed' or ev['status']!='completed' or
                            ev['job_identity']!=meta['job_identity']):raise ValueError('Failed or mismatched output/evaluation')
                    evaluations.append(ev)
                    roots=reconcile_case(out,folder+'/receipt.json',m['identity'])
                    for e in meta['feedback_events']:
                        if e.get('applied'):roots+=reconcile_case(out,folder+f'/feedback/step_{e["step_index"]:03d}.json',m['identity'])
                    if not all(r['ok'] for r in roots):raise RuntimeError('Incomplete dual media publication')
                    media.append(dict(arm=arm,id=row['id'],seed=seed,roots=roots,ok=True))
                    imports+=int(meta['imported'])
                    costs.setdefault(arm,[]).append({k:meta[k] for k in ('sampling_seconds','setup_seconds','peak_allocated_gib','imported')})
                except Exception as exc:failures.append(dict(id=row['id'],arm=arm,seed=seed,error=repr(exc)))
                if (len(media)+len(failures))%24==0:print(f'[readout media] {len(media)+len(failures)}/{expected}',flush=True)
    def table(ids):
        mapped=[dict(r,arm='rae_full' if r['arm']=='op_rae' else r['arm']) for r in evaluations
            if r['id'] in ids and r['arm'] in OPERATOR_ARMS]
        return aggregate(mapped,['rae_full',*OPERATOR_ARMS[:-1]],bootstrap=10000)
    subset=[r for r in evaluations if r['id'] in m['subset_ids']]
    specs=specifications(cfg);by_alpha={}
    for alpha in [.15,.35,.55]:
        selected=[dict(r,arm='rae_full' if specs[r['arm']]['operator']=='op_rae' else specs[r['arm']]['operator'])
            for r in subset if r['arm']!='op_sft' and specs[r['arm']]['alpha']==alpha]
        by_alpha[str(alpha)]=aggregate(selected,['rae_full',*OPERATOR_ARMS[1:-1]],bootstrap=10000)
    lpips_missing=[dict(id=r['id'],arm=r['arm']) for r in evaluations if cfg['lpips'] and r['lpips']['status']!='available']
    summary=dict(expected=expected,evaluated=len(evaluations),imported=imports,failures=failures,costs=costs,
        complete=len(evaluations)==expected and not failures and not lpips_missing,missing_lpips=lpips_missing,
        tables=dict(all_sources=table([r['id'] for r in records]),original_subset=table(m['subset_ids']),extension=table(m['extension_ids'])),
        strength_curves=aggregate(subset,list(specifications(cfg)),bootstrap=10000),
        paired_at_fixed_alpha=by_alpha,
        matched_strength=matched_strength(subset,cfg,[r for r in records if r['id'] in m['subset_ids']]),
        limitations=['Fixed source cohort; no external-editor benchmark or human ratings.',
            'Equal alpha is not equal dose; interpolated comparisons are diagnostics, not generated operating points.',
            'Future reference agreement is a proxy, not instruction-following accuracy.',
            'VAE and RAE differ in temporal lattice/capacity; smoothing is not bandwidth matched.',
            'No online state solve or learned bridge; expanded evidence isolates cached readout under shared SFT.'])
    suffix='_subset' if a.cohort=='subset' else ''
    out.write_json('media_sync'+suffix+'.json',dict(cases=media,failures=failures))
    out.write_json('summary'+suffix+'.json',summary)
    out.write_json('evaluations'+suffix+'.json',dict(records=evaluations))
    print('[readout summary] '+json.dumps(dict(expected=expected,evaluated=len(evaluations),complete=summary['complete'],imports=imports)),flush=True)
    if not summary['complete']:raise RuntimeError('Readout suite incomplete; successful results retained; use resume')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase',required=True,choices=('index','prepare','prepare_vae','render','evaluate','summarize'))
    parser.add_argument('--cohort',choices=('all','subset','extension'),default='all')
    parser.add_argument('--arm')
    for n in ('config','output','parent','attribution','model','rae','encoder','bank','data','source'):parser.add_argument('--'+n,default='')
    for n in ('root','parent-root','attribution-root','bank-root'):parser.add_argument('--'+n,action='append',default=[])
    a=parser.parse_args();cfg=json.loads(Path(a.config).read_text());validate_config(cfg);out=ArtifactStore(a.output,a.root)
    if a.arm and a.arm not in specifications(cfg):parser.error('Unknown readout arm')
    if a.phase=='index':return index(a,cfg,out)
    if a.phase=='summarize':
        if int(os.environ.get('NODE_RANK','0'))==0:return summarize(a,cfg,out)
        return
    device,rank,world=distributed()
    try:dict(prepare=prepare,prepare_vae=prepare,render=render,evaluate=evaluate)[a.phase](a,cfg,out,device,rank,world)
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
