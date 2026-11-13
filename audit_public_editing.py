"""Stage71: frozen four-arm VACE gray-to-color transfer, no new training.

The original RGB is an evaluation-only label. Native, SFT, RGB and RAE share
the official gray input, full-frame mask, prompt and generated anchor.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import time

import torch

from utils.scene_run import ArtifactStore
from utils.editing_data import digest, file_hash, rae_canvas
from utils.editing_public_data import (ARMS, audit_index, exposed_benchmark_ids,
    public_source, freeze_public, shared_reference, read_target, color_metrics)
from utils.editing_paper_data import source_fingerprint
from utils.editing_train import distributed, progress, load_teacher, load_encoder, write_video
from utils.editing_paper_controls import ReferenceSFT, PaperFeedback, arm_spec
from utils.editing_joint_state import JointEditProjector
from utils.editing_state_feedback import load_decoder_only
from utils.vace_edit_runtime import (load_backbone, load_vae, load_text, pack_context,
    sample_native, decode_native, validate_assets)
from utils.vace_state_runtime import StateConditionedVACE, pack_edit_context, sample_state_native
from utils.vace_state_data import prepare_rae_state, build_rgb_state
from train_vace_edit_state import encoded_reference, decoded_reference, arm_store, publish_lossless
from edit_reference_video import publish_case, recovered_case, crop_video, failure, tensor_hash
from paper_editing_suite import code_signature as parent_signature, release, save_phase
from utils.native_audit_io import seal

REPO = Path(__file__).resolve().parent


def signature():
    return dict(parent_signature(), **{n:file_hash(REPO/n) for n in
        ('audit_public_editing.py', 'utils/editing_public_data.py', 'utils/editing_public_report.py',
         'audit_edit_transport.py', 'models/scene_rae.py', 'models/dual_stream_decoder.py',
         'utils/scene_run.py', 'utils/native_audit_io.py', 'utils/editing_publication.py')})


def parent(a): return ArtifactStore(a.parent, a.parent_root)
def states(a): return ArtifactStore(a.states, a.state_root)
def cohort(out): return out.json('cohort.json', True)['records']


def checked(a, cfg, out):
    m = out.json('manifest.json', True)
    if m['config'] != cfg or m['code'] != signature():
        raise ValueError('Public run code/config changed; select a fresh PUBLIC_NAMESPACE')
    return m


def assignments(rows, rank, world):
    # Even a three-case smoke reaches one NPU on EACH node.
    if world == 24:
        return [r for i,r in enumerate(rows) if (i%3)*8+(i//3)%8 == rank]
    return rows[rank::world]


def tick(out, phase, rank, rows, started, frames):
    progress(out, phase+'_r'+str(rank), dict(phase=phase, attempted=len(rows),
        completed=sum(r.get('status')=='completed' for r in rows),
        DI_throughput=frames/max(time.monotonic()-started, 1e-6),
        throughput_unit='new completed frames/s/rank including phase IO; cache hits excluded'))


def finish_phase(out, phase, rows, requested, rank, world):
    all_rows = save_phase(out, phase, rows, requested, rank, world)
    if rank == 0:
        print('[public phase] '+json.dumps(dict(phase=phase, requested=requested,
            completed=sum(r.get('status')=='completed' for r in all_rows),
            errors=dict(Counter(r.get('error',r['status']) for r in all_rows
                                if r.get('status')!='completed')))), flush=True)
    return all_rows


def index(a, cfg, out):
    if (tuple(cfg['arms']) != ARMS or cfg['frames'] != 17 or cfg['fps'] != 8
            or not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds'])
            or cfg['sample_steps'] < 2 or not 1 <= cfg['cases'] <= 20):
        raise ValueError('Use the declared 17-frame/8fps four-arm public protocol')
    p = parent(a); pm = p.json('manifest.json', True)
    if any(pm['source_geometry'][key] != cfg[key] for key in ('frames','fps')):
        raise ValueError('Public window must match the frozen training geometry')
    bank = ArtifactStore(a.bank, a.bank_root).json('bank.json', True)
    transport = ArtifactStore(a.transport, a.transport_root)
    if pm['bank'] != bank['identity'] or pm['stage64'] != transport.json('manifest.json',True)['identity']:
        raise ValueError('Frozen source bank/normalization provenance differs')
    sft = arm_store(p, 'sft'); training = sft.json('training_summary.json', True)
    if training.get('status') != 'completed' or not training.get('updates'):
        raise ValueError('Need the completed matched-reference SFT checkpoint')
    checkpoints = {'native':'native'}
    for family, store in [('sft',sft), ('rgb',arm_store(states(a),'rgb')), ('rae',arm_store(states(a),'rae'))]:
        checkpoints[family] = store.json('selected.pt.json', True)['sha256']
        if family != 'sft' and checkpoints[family] != pm['selected'][family]:
            raise ValueError('Frozen '+family+' selected checkpoint changed')
    candidates, labels, audit = audit_index(a.benchmark, a.benchmark_source)
    exposure = p.json('phases/screen.json', True)['records']
    prior_cohort = p.json('cohort.json', True)
    exposure += [dict(r,status='completed') for r in prior_cohort['records']]
    scale = transport.json('prepared.json', True)['source_scale']
    if scale is None: raise ValueError('Missing TRAIN-only source scale')
    previous_ids=exposed_benchmark_ids(bank)
    previous_ids.update(exposed_benchmark_ids(dict(records=exposure)))
    m = dict(schema='editing-public-colorization-v1', config=cfg, code=signature(),
        parent_identity=pm['identity'], stage65=pm['stage65'], bank=pm['bank'],
        sft_identity=p.json('sft_contract.json',True)['identity'],
        base_config=pm['base_config'], teacher=pm['teacher'], assets=pm['assets'],
        source_geometry=pm['source_geometry'], negative_prompt=pm['negative_prompt'],
        source_scale=scale, checkpoint=checkpoints, candidates=candidates,
        evaluation_index_sha256=digest(labels), audit=audit,
        exposed=exposure, old_benchmark_ids=sorted(previous_ids),
        protocol='official V2V/gray; custom physical 17-frame window; source-only cohort; '
                 'frozen Native/SFT/RGB/RAE; shared model anchor; no benchmark training')
    m['identity'] = digest(m)
    previous = out.json('manifest.json')
    if previous and previous != m: raise ValueError('Public input contract changed; use a fresh namespace')
    # Every node verifies the same snapshot; only leader publishes global metadata.
    if int(os.environ.get('NODE_RANK','0')) == 0:
        out.write_json('manifest.json', m)
        out.write_json('evaluation_manifest.json', dict(identity=m['identity'], labels=labels))
        out.write_json('benchmark_audit.json', audit)
    print('[public inputs] '+json.dumps(dict(eligible=len(candidates), domains=audit['eligible_domains'],
        previous_benchmark_inputs=m['old_benchmark_ids'], new_training=False)), flush=True)


def sources(a, cfg, out, device, rank, world):
    m = checked(a,cfg,out); rows=[]; frames=0; started=time.monotonic()
    for row in assignments(m['candidates'],rank,world):
        name='sources/'+row['id']+'.pt'
        try:
            item=out.load(name,m['identity'])
            if item is None:
                item=public_source(a.benchmark,row,m['source_geometry'],a.benchmark_source)
                out.save(name,item,m['identity']); frames+=cfg['frames']
            rows.append(dict(id=row['id'],group=item['group'],status='completed',file=name,
                fingerprint=source_fingerprint(item['source'],item['spatial']['valid_rect'])))
        except Exception as exc: rows.append(failure(out,'sources',row,exc))
        tick(out,'sources',rank,rows,started,frames)
    finish_phase(out,'sources',rows,len(m['candidates']),rank,world)


def select(a, cfg, out):
    if int(os.environ.get('NODE_RANK','0')) != 0: return
    m=checked(a,cfg,out)
    selected=freeze_public(m['candidates'],out.json('phases/sources.json',True)['records'],
        m['exposed'],set(m['old_benchmark_ids']),cfg['cases'],cfg.get('smoke',False),m['audit']['asset_groups'])
    previous=out.json('cohort.json')
    if previous and previous != selected: raise ValueError('Frozen cohort changed; select a new namespace')
    out.write_json('cohort.json',selected)
    print('[public cohort] '+json.dumps({k:v for k,v in selected.items() if k!='records'}),flush=True)
    if not selected['records']: raise RuntimeError('No eligible readable unseen gray cases')


def texts(a,cfg,out,device,rank,world):
    m=checked(a,cfg,out); rows=[]; model=None; frames=0; started=time.monotonic()
    for row in assignments(cohort(out),rank,world):
        try:
            name='text/'+row['id']+'.pt'
            if out.load(name,m['identity']) is None:
                if model is None: model=load_text(a.model,device)
                with torch.no_grad(): values=model([row['instruction'],m['negative_prompt']],device)
                out.save(name,{k:v.cpu().bfloat16() for k,v in zip(('text','negative'),values)},m['identity'])
                frames+=cfg['frames']
            rows.append(dict(id=row['id'],status='completed'))
        except Exception as exc: rows.append(failure(out,'text',row,exc))
        tick(out,'text',rank,rows,started,frames)
    finish_phase(out,'text',rows,len(cohort(out)),rank,world)


def anchors(a,cfg,out,device,rank,world):
    m=checked(a,cfg,out); rows=[]; model=vae=None; frames=0; started=time.monotonic()
    settings=dict(m['base_config'],sample_steps=cfg['sample_steps'])
    for row in assignments(cohort(out),rank,world):
        try:
            folder='anchors/'+row['id']; payload,record=recovered_case(out,folder,m['identity'])
            reused=payload is not None
            if payload is None:
                item=out.load(row['source_file'],m['identity'],True)
                text=out.load('text/'+row['id']+'.pt',m['identity'],True)
                source=item['source'].float()/255; mask=item['native_mask'].float()/255
                if model is None:
                    if validate_assets(a.model)!=m['assets']: raise ValueError('Native assets changed')
                    model,vae=load_backbone(a.model,device),load_vae(a.model,device)
                    model.eval().requires_grad_(False)
                seed=cfg['anchor_seed']+int(digest(row['id'])[:8],16)%1000000
                with torch.no_grad():
                    context=pack_context(vae,(source*2-1).permute(1,0,2,3).to(device),mask.permute(1,0,2,3).to(device))
                    latent=sample_native(model,context,text['text'].to(device),text['negative'].to(device),settings,seed)
                    proposal=((decode_native(vae,latent).permute(1,0,2,3).float()+1)/2).clamp(0,1).cpu()
                if proposal.shape!=source.shape or not torch.isfinite(proposal).all(): raise ValueError('Invalid native anchor proposal')
                support=mask[0]; anchor=source[0]*(1-support)+proposal[0]*support
                payload=dict(source=source,anchor=anchor,support=support,spatial=item['spatial'],
                    native_proposal=proposal,provenance='model_output',seed=seed,source_only=True)
            if record is None:
                record=publish_case(out,folder,m['identity'],payload,
                    dict(id=row['id'],status='completed',provenance='model_output',frontend='frozen_native_vace'),
                    [payload['source'],payload['anchor'][None].expand_as(payload['source']),payload['native_proposal']],
                    ['GRAY SOURCE','SHARED MODEL ANCHOR','FRONTEND'],payload['spatial']['valid_rect'],cfg['fps'])
            rows.append(record)
            if not reused: frames+=cfg['frames']
        except Exception as exc: rows.append(failure(out,'anchors',row,exc))
        tick(out,'anchors',rank,rows,started,frames)
    finish_phase(out,'anchors',rows,len(cohort(out)),rank,world)


def shared(a,cfg,out,device,rank,world):
    m=checked(a,cfg,out); rows=[]; frames=0; started=time.monotonic()
    for row in assignments(cohort(out),rank,world):
        try:
            name='shared/'+row['id']+'.pt'
            if out.load(name,m['identity']) is None:
                anchor=out.load('anchors/'+row['id']+'/result.pt',m['identity'],True)
                if anchor['provenance']!='model_output': raise ValueError('Unexpected anchor provenance')
                source=out.load(row['source_file'],m['identity'],True)
                inputs,ref=shared_reference(anchor['source'],anchor['anchor'],anchor['support'],
                    anchor['spatial'],source['native_mask'],m['base_config'])
                inputs['scale']=torch.tensor(m['source_scale'])
                out.save(name,dict(inputs=inputs,reference=encoded_reference(ref)),m['identity'])
                frames+=cfg['frames']
            rows.append(dict(id=row['id'],status='completed'))
        except Exception as exc: rows.append(failure(out,'shared',row,exc))
        tick(out,'shared',rank,rows,started,frames)
    finish_phase(out,'shared',rows,len(cohort(out)),rank,world)


def cache(a,cfg,out,device,rank,world):
    m=checked(a,cfg,out); rows=[]; vae=None; frames=0; started=time.monotonic()
    for row in assignments(cohort(out),rank,world):
        try:
            name='native/'+row['id']+'.pt'
            if out.load(name,m['identity']) is None:
                ref=decoded_reference(out.load('shared/'+row['id']+'.pt',m['identity'],True)['reference'])
                if vae is None: vae=load_vae(a.model,device)
                rgb=lambda x:(x*2-1).permute(1,0,2,3).to(device)
                mask=ref['masks'].permute(1,0,2,3).to(device); anchor=rgb(ref['anchor'][None])
                with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type!='cpu'):
                    context=pack_edit_context(vae,rgb(ref['source']),rgb(ref['reference']),mask,anchor)
                    native=pack_context(vae,rgb(ref['source']),mask,[anchor])
                out.save(name,dict(context=context.cpu().bfloat16(),native_context=native.cpu().bfloat16()),m['identity'])
                frames+=cfg['frames']
            rows.append(dict(id=row['id'],status='completed'))
        except Exception as exc: rows.append(failure(out,'cache',row,exc))
        tick(out,'cache',rank,rows,started,frames)
    finish_phase(out,'cache',rows,len(cohort(out)),rank,world)


def prepare(a,cfg,out,device,rank,world):
    from audit_edit_transport import encode_window
    m=checked(a,cfg,out); rows=[]; rae=encoder=None; frames=0; started=time.monotonic()
    if file_hash(a.rae)!=m['teacher']['rae'] or file_hash(a.encoder)!=m['teacher']['encoder']:
        raise ValueError('Production teacher weights changed')
    base=copy.deepcopy(m['base_config'])
    if cfg.get('smoke'): base['calibration']['steps']=1
    for row in assignments(cohort(out),rank,world):
        try:
            name='prepared/'+row['id']+'.pt'
            if out.load(name,m['identity']) is None:
                item=out.load('shared/'+row['id']+'.pt',m['identity'],True)
                inputs=item['inputs']; ref=decoded_reference(item['reference'])
                anchor=out.load('anchors/'+row['id']+'/result.pt',m['identity'],True)
                # Match the trained encoder path: float resized source BEFORE uint8 caching.
                square,_=rae_canvas(anchor['source'][inputs['observed_indices']],anchor['spatial']['valid_rect'])
                ar,_=rae_canvas(anchor['anchor'][None],anchor['spatial']['valid_rect'])
                if rae is None: rae,encoder=load_teacher(a.rae,device).float(),load_encoder(a.encoder,device)
                with torch.no_grad():
                    inputs['source_z']=encode_window(rae,encoder,square[None].to(device))[0].detach().cpu().half()
                    inputs['anchor_z']=encode_window(rae,encoder,ar[None].to(device))[0].detach().cpu().half()
                state=prepare_rae_state(inputs,ref,rae,encoder,base,device)
                if state['calibration']['fallback_count']: raise RuntimeError('RAE initial calibration fallback: '+str(state['calibration']))
                inputs['initial_delta']=(state['edited_z'].float()-inputs['source_z'].float()).half()
                out.save(name,dict(inputs=inputs,reference=item['reference'],rae_state=state['state'],
                    calibration=state['calibration'],provenance='model_output'),m['identity'])
                frames+=cfg['frames']
            rows.append(dict(id=row['id'],status='completed'))
        except Exception as exc: rows.append(failure(out,'prepare',row,exc)); release(device)
        tick(out,'prepare',rank,rows,started,frames)
    finish_phase(out,'prepare',rows,len(cohort(out)),rank,world)


def weight_store(a,m,family):
    store=arm_store(parent(a) if family=='sft' else states(a),family)
    identity=digest([m['sft_identity'] if family=='sft' else m['stage65'],family])
    return store,identity


def weights(a,cfg,out):
    m=checked(a,cfg,out); reports=[]
    for family in ('sft','rgb','rae'):
        try:
            store,identity=weight_store(a,m,family)
            store.load('selected.pt',identity,True)
            if store.json('selected.pt.json',True)['sha256']!=m['checkpoint'][family]: raise ValueError('Selected weight changed')
            reports.append(dict(family=family,status='completed'))
        except Exception as exc: reports.append(dict(family=family,status='failed',error=repr(exc)))
    out.write_json('staging/node'+str(os.environ.get('NODE_RANK','0'))+'.json',dict(records=reports))
    if any(r['status']!='completed' for r in reports): raise RuntimeError(str(reports))


def render(a,cfg,out,device,rank,world):
    m=checked(a,cfg,out); spec=arm_spec(a.arm); family=spec['family']
    if a.arm not in ARMS: raise ValueError('Undeclared public arm')
    rows=[]; frames=0; started=time.monotonic(); model=vae=decoder=None
    settings=dict(m['base_config'],sample_steps=cfg['sample_steps'])
    checkpoint=m['checkpoint'][family]
    for row in assignments(cohort(out),rank,world):
        for seed_base in cfg['seeds']:
            key=row['id']; folder=f'render/{a.arm}/seed{seed_base}/{key}'
            feedback=projector=features=latent=None
            try:
                prepared=out.load(('prepared/' if family=='rae' else 'shared/')+key+'.pt',m['identity'],True)
                ref=decoded_reference(prepared['reference'])
                native=out.load('native/'+key+'.pt',m['identity'],True); text=out.load('text/'+key+'.pt',m['identity'],True)
                seed=seed_base+int(digest(key)[:8],16)%1000000
                input_id=digest([m['identity'],key,seed,tensor_hash(ref['source']),tensor_hash(ref['reference']),
                    tensor_hash(ref['anchor']),tensor_hash(ref['masks'])])
                generation=digest([input_id,a.arm,checkpoint,settings])
                payload,record=recovered_case(out,folder,m['identity']); reused=payload is not None
                if reused and payload['generation_identity']!=generation: raise ValueError('Saved generation identity changed')
                if reused and spec['feedback'] and (not payload['events'] or any(e.get('fallback') for e in payload['events'])):
                    out.write_json(folder+'/previous_failure.json',dict(events=payload['events'],generation_identity=generation))
                    payload=record=None; reused=False
                if payload is None:
                    setup=time.monotonic()
                    if model is None:
                        if validate_assets(a.model)!=m['assets']: raise ValueError('Native assets changed')
                        core=load_backbone(a.model,device)
                        if family=='native': model=core
                        else:
                            model=(ReferenceSFT(core,settings) if family=='sft' else
                                StateConditionedVACE(core,settings,settings['state_channels'][family])).to(device)
                            store,identity=weight_store(a,m,family); chosen=store.load('selected.pt',identity,True)
                            if store.json('selected.pt.json',True)['sha256']!=checkpoint: raise ValueError('Selected checkpoint changed')
                            model.restore(chosen['model']); del chosen
                        model.eval().requires_grad_(False)
                    if vae is None: vae=load_vae(a.model,device)
                    events=[]
                    if family in ('rgb','rae'):
                        features=(build_rgb_state(ref) if family=='rgb' else prepared['rae_state'].float()).to(device)
                        inputs=prepared['inputs']; source=inputs['source'].float().to(device)/255
                        valid=source.new_zeros(len(source),1,*source.shape[-2:])
                        y,x,h,w=inputs['valid_rect']; valid[:,:,y:y+h,x:x+w]=1
                        if family=='rae' and decoder is None:
                            if file_hash(a.rae)!=m['teacher']['rae']: raise ValueError('Frozen decoder changed')
                            decoder=load_decoder_only(a.rae,device,True)
                        projector=JointEditProjector(decode=decoder if family=='rae' else None,
                            source_z=inputs['source_z'].to(device) if family=='rae' else None,
                            source=source,anchor=inputs['anchor'].float().to(device)/255,
                            masks=inputs['masks'].float().to(device),confidence=inputs['confidence'].float().to(device),
                            cfg=dict(cfg['feedback']['solver'],device=str(device)),
                            initial_delta=inputs['initial_delta'].to(device) if family=='rae' else None,
                            initial_effect=(inputs['proposal'].float()-inputs['source'].float()).to(device)/255,
                            proposal=inputs['proposal'].float().to(device)/255,edges=inputs['edges'],
                            scale=inputs['scale'].to(device),valid=valid)
                        def observer(event):
                            events.append(dict(event))
                            if event.get('applied'):
                                from refine_vace_edit_state import _cpu_snapshot
                                base=folder+f'/feedback/step_{int(event["step_index"]):03d}'
                                snapshot=_cpu_snapshot(dict(reference=feedback.last_reference,projection=feedback.last_projection,
                                    state=feedback.state,event=event,generation_identity=generation))
                                out.save(base+'.pt',snapshot,m['identity']); rect=ref['spatial']['valid_rect']
                                write_video(out.local/(base+'.mp4'),crop_video(ref['source'],rect),crop_video(ref['reference'],rect),
                                    crop_video(snapshot['reference']['reference'],rect),fps=cfg['fps'],
                                    labels=['GRAY SOURCE','SHARED REFERENCE','LIVE CORRECTION'])
                                seal(out.local,base+'.json',m['identity'],[base+'.mp4'],dict(event=event))
                                out.publish([base+'.mp4',base+'.json'])
                            progress(out,'feedback_r'+str(rank),dict(event,id=key,arm=a.arm,seed_base=seed_base))
                        feedback=PaperFeedback(vae,projector,inputs,ref,features,
                            dict(cfg['feedback'],update_bridge_state=True,update_native_context=False),family,observer=observer)
                    setup_seconds=time.monotonic()-setup
                    if device.type!='cpu':
                        getattr(torch,device.type).synchronize(); getattr(torch,device.type).reset_peak_memory_stats()
                    sampling=time.monotonic()
                    with torch.no_grad():
                        if family=='native':
                            latent=sample_native(model,native['native_context'].to(device),text['text'].to(device),text['negative'].to(device),settings,seed)
                        else:
                            latent=sample_state_native(model,native['context'].to(device),text['text'].to(device),text['negative'].to(device),
                                features,settings,seed,use_state=family!='sft',velocity_callback=feedback)
                        generated=((decode_native(vae,latent,1).permute(1,0,2,3).float()+1)/2).clamp(0,1).cpu()
                    if device.type!='cpu': getattr(torch,device.type).synchronize()
                    if generated.shape!=ref['source'].shape or not torch.isfinite(generated).all(): raise ValueError('Invalid video output')
                    payload=dict(generated=generated,spatial=ref['spatial'],seed=seed,seed_base=seed_base,
                        input_identity=input_id,generation_identity=generation,checkpoint=checkpoint,events=events,
                        runtime_code=signature(),setup_seconds=setup_seconds,sampling_seconds=time.monotonic()-sampling,
                        peak_allocated_gib=getattr(torch,device.type).max_memory_allocated()/2**30 if device.type!='cpu' else None)
                bad=[e for e in payload['events'] if e.get('fallback')]
                if spec['feedback'] and not payload['events']: bad=[dict(error='No scheduled feedback events')]
                metrics=dict(id=key,group=row['group'],domain=row['domain'],arm=a.arm,seed_base=seed_base,
                    status='feedback_failed' if bad else 'completed',feedback_events=payload['events'],fallback_count=len(bad),reused=reused,
                    **{k:payload[k] for k in ('seed','input_identity','generation_identity','checkpoint','sampling_seconds','setup_seconds','peak_allocated_gib')})
                if record is None:
                    publish_case(out,folder,m['identity'],payload,metrics,[ref['source'],ref['reference'],payload['generated']],
                        ['GRAY SOURCE','SHARED REFERENCE',a.arm],ref['spatial']['valid_rect'],cfg['fps'])
                rows.append(publish_lossless(out,folder,m['identity'],payload,cfg['fps']))
                if not reused and not bad: frames+=cfg['frames']
            except Exception as exc: rows.append(dict(failure(out,'render_'+a.arm,row,exc),arm=a.arm,seed_base=seed_base))
            finally: feedback=projector=features=latent=None; release(device)
            tick(out,'render_'+a.arm,rank,rows,started,frames)
    finish_phase(out,'render_'+a.arm,rows,len(cohort(out))*len(cfg['seeds']),rank,world)


def evaluate(a,cfg,out,device,rank,world):
    from utils.editing_paper_metrics import evaluate_videos, optional_lpips, evaluation_context
    m=checked(a,cfg,out); labels=out.json('evaluation_manifest.json',True)
    if labels['identity']!=m['identity'] or digest(labels['labels'])!=m['evaluation_index_sha256']:
        raise ValueError('Evaluation label index changed')
    rows=[]; frames=0; started=time.monotonic()
    perceptual,status=optional_lpips(device,enabled=cfg['lpips'])
    for row in assignments(cohort(out),rank,world):
        try:
            item=out.load(row['source_file'],m['identity'],True)
            target,label=read_target(a.benchmark,labels['labels'][row['id']],item,m['source_geometry'],a.benchmark_source)
            y,x,h,w=item['spatial']['valid_rect']
            source=item['source'][:,:,y:y+h,x:x+w]; target=target[:,:,y:y+h,x:x+w]
            context=evaluation_context(source,target)
            control=dict(evaluate_videos(source,target,source,perceptual,device,context),**color_metrics(source,target,source))
            out.write_json('copy_gray/'+row['id']+'.json',dict(id=row['id'],group=row['group'],domain=row['domain'],
                status='completed',arm='copy_gray',seed_base=cfg['seeds'][0],metrics=control))
        except Exception as exc:
            rows.append(failure(out,'evaluate_source',row,exc)); continue
        for arm in ARMS:
            for seed in cfg['seeds']:
                try:
                    folder=f'render/{arm}/seed{seed}/{row["id"]}'
                    payload=out.load(folder+'/result.pt',m['identity'],True)
                    render_record=out.json(folder+'/metrics.json',True)
                    if render_record['status']!='completed': raise ValueError('Render is not technically complete')
                    name=f'evaluation/{arm}/seed{seed}/{row["id"]}.json'
                    identity=digest([m['identity'],payload['generation_identity'],label,status])
                    record=out.json(name)
                    if record is None or record.get('evaluation_identity')!=identity or record.get('status')!='completed':
                        generated=payload['generated'][:,:,y:y+h,x:x+w]
                        metrics=dict(evaluate_videos(source,target,generated,perceptual,device,context),**color_metrics(source,target,generated))
                        future=evaluate_videos(source[1:],target[1:],generated[1:],perceptual,device)
                        metrics.update({'future_'+k:future[k] for k in ('target_psnr','target_ssim','target_lpips')})
                        record=dict(id=row['id'],group=row['group'],domain=row['domain'],arm=arm,seed_base=seed,status='completed',
                            metrics=metrics,evaluation_identity=identity,input_identity=payload['input_identity'],
                            generation_identity=payload['generation_identity'],target=label,lpips=status)
                        out.write_json(name,record); frames+=cfg['frames']
                    rows.append(record)
                except Exception as exc: rows.append(dict(failure(out,'evaluate_'+arm,row,exc),arm=arm,seed_base=seed))
                tick(out,'evaluate',rank,rows,started,frames)
    finish_phase(out,'evaluate',rows,len(cohort(out))*len(ARMS)*len(cfg['seeds']),rank,world)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=['audit','index','sources','select','text','anchors','shared','cache','prepare','weights','render','evaluate','summarize'])
    for name in ('config','output','parent','states','transport','bank','benchmark','model','rae','encoder'):
        parser.add_argument('--'+name,required=True)
    for name in ('root','parent-root','state-root','transport-root','bank-root'):
        parser.add_argument('--'+name,action='append',default=[])
    parser.add_argument('--benchmark-source',default='')
    parser.add_argument('--arm',choices=ARMS)
    a=parser.parse_args(); cfg=json.loads(Path(a.config).read_text())
    out=ArtifactStore(a.output,a.root)
    if a.phase=='audit':
        _,_,report=audit_index(a.benchmark,a.benchmark_source)
        out.write_json('benchmark_audit.json',report); print(json.dumps(report,indent=2)); return
    if a.phase in ('index','select','weights'): return globals()[a.phase](a,cfg,out)
    if a.phase=='summarize':
        if int(os.environ.get('NODE_RANK','0'))==0:
            from utils.editing_public_report import summarize
            return summarize(a,cfg,out)
        return
    device,rank,world=distributed()
    try: globals()['texts' if a.phase=='text' else a.phase](a,cfg,out,device,rank,world)
    finally:
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__=='__main__': main()
