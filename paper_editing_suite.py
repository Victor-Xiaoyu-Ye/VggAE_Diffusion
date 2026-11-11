"""Stage69: frozen-method paper evidence on three nodes.

Only a matched no-state SFT is newly trained. Main evaluation, input-only
cohort screening, path/graph ablations, metrics and human-review exports have
separate persisted phases. No future edited label enters a sampling API.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch

from utils.scene_run import ArtifactStore
from utils.editing_data import digest, file_hash, rae_canvas
from utils.editing_train import distributed, barrier, gather, progress, load_teacher, load_encoder, write_video
from utils.editing_paper_data import (candidate_rows, load_source, source_fingerprint,
    freeze_cohort, load_evaluation_labels, TrainingCacheView)
from utils.editing_paper_controls import (MAIN_ARMS, ABLATION_ARMS, SECOND_SEED_ARMS,
    ReferenceSFT, arm_spec, intervened_state, intervened_inputs, PaperFeedback)
from utils.vace_edit_runtime import load_backbone, load_vae, load_text, pack_context, sample_native, decode_native, validate_assets
from utils.vace_state_runtime import StateConditionedVACE, pack_edit_context, sample_state_native
from utils.vace_state_data import build_reference, prepare_rae_state, build_rgb_state
from train_vace_edit_state import encoded_reference, decoded_reference, publish_lossless, arm_store, train_loop
from edit_reference_video import source_support, valid_pixels, crop_float, crop_video, publish_case, recovered_case, failure, tensor_hash
from utils.native_audit_io import seal


REPO = Path(__file__).resolve().parent


def code_signature():
    names = ('paper_editing_suite.py', 'utils/editing_paper_data.py',
        'utils/editing_paper_controls.py', 'utils/editing_paper_metrics.py',
        'utils/editing_paper_report.py', 'train_vace_edit_state.py',
        'utils/vace_state_runtime.py', 'utils/vace_state_data.py',
        'utils/editing_state_feedback.py', 'utils/editing_joint_state.py',
        'utils/edit_transport_data.py', 'utils/editing_video_transport.py',
        'utils/vace_edit_runtime.py', 'utils/editing_alignment.py',
        'utils/editing_data.py', 'utils/editing_train.py', 'utils/native_wan_runtime.py',
        'edit_reference_video.py', 'refine_vace_edit_state.py')
    return {n:file_hash(REPO/n) for n in names}


def stores(a):
    return (ArtifactStore(a.bank, a.bank_root), ArtifactStore(a.states, a.state_root),
            ArtifactStore(a.transport, a.transport_root))


def checked(a, cfg, out):
    m = out.json('manifest.json', True)
    if m['config'] != cfg or m['code'] != code_signature():
        raise ValueError('Paper suite code/config changed; use a fresh PAPER_NAMESPACE')
    return m


def cohort(out):
    return out.json('cohort.json', True)


def save_phase(out, phase, records, requested, rank, world):
    all_rows = sum(gather(records, world), [])
    if rank == 0:
        out.write_json('phases/'+phase+'.json', dict(requested=requested, records=all_rows,
            completed=sum(r['status'] == 'completed' for r in all_rows)))
    barrier()
    return all_rows


def tick(out, phase, rank, records, started, frames, **extra):
    count=sum(row.get('status')=='completed' for row in records)
    progress(out, phase+'_r'+str(rank), dict(phase=phase, completed=count,
        attempted=len(records), failed=len(records)-count,
        DI_throughput=count*frames/max(time.monotonic()-started, 1e-6),
        throughput_unit='completed source/output frames/s/rank including phase IO', **extra))


def release(device):
    gc.collect()
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()


def index(a, cfg, out):
    """Run once per node before distributed readers; exact same frozen contract."""
    from prepare_vace_edit import stage_data
    bank_store, state_store, transport = stores(a)
    bank, old, prior = bank_store.json('bank.json', True), state_store.json('manifest.json', True), transport.json('manifest.json', True)
    if cfg['frames'] != bank['config']['frames'] or cfg['fps'] != bank['config']['fps']:
        raise ValueError('Evaluation window must match the frozen training contract')
    if cfg['cases'] < 1 or cfg['seed'] == cfg['second_seed']:
        raise ValueError('Need positive case count and distinct robustness seeds')
    if tuple(cfg['main_arms']) != MAIN_ARMS or tuple(cfg['ablation_arms']) != ABLATION_ARMS:
        raise ValueError('Use the declared paper controls; resume arms with --arm, not by editing the contract')
    if old['bank'] != bank['identity'] or old['transport'] != prior['identity']:
        raise ValueError('Stage60/64/65 source contracts disagree')
    if old['teacher'] != prior['teacher']:
        raise ValueError('Source teacher signatures disagree')
    if not cfg.get('skip_staging', False):
        stage_data(a.source, a.data)
    rows, counts = candidate_rows(bank, cfg['cases'], cfg.get('smoke', False), cfg['seed'])
    selected = {}
    for family in ('rgb', 'rae'):
        receipt = arm_store(state_store, family).json('selected.pt.json')
        selected[family] = receipt['sha256'] if receipt else None
    m = dict(schema='editing-paper-suite-v1', config=cfg, code=code_signature(),
        bank=bank['identity'], stage65=old['identity'], stage64=prior['identity'],
        base_config=old['config'], teacher=old['teacher'], assets=bank['model_signature'],
        selected=selected, candidates=rows, candidate_counts=counts,
        source_geometry={k:bank['config'][k] for k in ('height', 'width', 'max_side', 'frames', 'fps')},
        negative_prompt=bank['config']['negative_prompt'],
        protocol='frozen native frontend shared across all arms; no dataset targets as conditions')
    m['identity'] = digest(m)
    previous = out.json('manifest.json')
    if previous and previous['identity'] != m['identity']:
        raise ValueError('Input/weights changed; use a fresh PAPER_NAMESPACE')
    out.write_json('manifest.json', m)
    print('[paper frozen contract] '+json.dumps(dict(identity=m['identity'], candidates=counts,
        main=list(MAIN_ARMS), new_training='matched reference SFT only')), flush=True)


def screen(a, cfg, out, device, rank, world):
    m = checked(a, cfg, out)
    data, _, _ = stores(a)
    bank = data.json('bank.json', True)
    cached = {r['id']:r for r in bank['records']}
    seen_map = {r['id']:r for r in bank['manifest']['records'] if r['split'] in ('train','dev')}
    seen_map.update({r['id']:r for r in bank['records'] if r['split'] != 'test'})
    if cfg.get('smoke'):
        seen_map = dict(sorted(seen_map.items())[:24])
    records, started = [], time.monotonic()
    for row in sorted(seen_map.values(), key=lambda r:r['id'])[rank::world]:
        name = 'screen/exposed/'+row['id']+'.json'
        try:
            record = out.json(name)
            if not record or record['status'] != 'completed':
                item = None
                if row['id'] in cached:
                    try:
                        old = data.load(cached[row['id']]['file'], m['bank'], True)
                        item = dict(source=old['source'], spatial=old['timing']['spatial'])
                        del old
                    except Exception as exc:
                        print('[paper screen cache fallback] '+row['id']+' '+repr(exc), flush=True)
                if item is None:
                    item = load_source(a.data, row, m['source_geometry'])
                record = dict(id=row['id'], group=row['group'], status='completed',
                    fingerprint=source_fingerprint(item['source'], item['spatial']['valid_rect']))
                out.write_json(name, record)
            records.append(record)
        except Exception as exc:
            records.append(dict(id=row['id'], group=row['group'], status='failed', error=repr(exc)))
        if len(records)%8 == 0 or len(records) == 1:
            tick(out, 'screen', rank, records, started, cfg['frames'], id=row['id'])
    save_phase(out, 'screen', records, len(seen_map), rank, world)


def sources(a, cfg, out, device, rank, world):
    m = checked(a, cfg, out)
    records, started = [], time.monotonic()
    for row in m['candidates'][rank::world]:
        name = 'sources/'+row['id']+'.pt'
        try:
            value = out.load(name, m['identity'])
            if value is None:
                value = load_source(a.data, row, m['source_geometry'])
                out.save(name, value, m['identity'])
            records.append(dict(id=row['id'], status='completed', file=name,
                fingerprint=source_fingerprint(value['source'], value['spatial']['valid_rect'])))
        except Exception as exc:
            records.append(dict(id=row['id'], status='failed', error=repr(exc)))
        tick(out, 'sources', rank, records, started, cfg['frames'], id=row['id'])
    save_phase(out, 'sources', records, len(m['candidates']), rank, world)


def select(a, cfg, out):
    m = checked(a, cfg, out)
    result = freeze_cohort(m['candidates'], out.json('phases/sources.json', True)['records'],
        out.json('phases/screen.json', True)['records'], cfg['cases'], cfg['seed'], cfg.get('smoke', False))
    previous = out.json('cohort.json')
    if previous and previous['identity'] != result['identity']:
        raise ValueError('Frozen cohort changed after input screening; use a fresh namespace')
    out.write_json('cohort.json', result)
    print('[paper cohort] '+json.dumps({k:result[k] for k in ('requested','selected','protocol','exposed_screen_completed')}), flush=True)
    if not result['records']:
        raise RuntimeError('No readable, unexposed source groups; other cached training work may continue')


def texts(a, cfg, out, device, rank, world):
    m = checked(a, cfg, out); rows = cohort(out)['records']
    reports, model, started = [], None, time.monotonic()
    for row in rows[rank::world]:
        try:
            name = 'text/'+row['id']+'.pt'
            if out.load(name, m['identity']) is None:
                if model is None:
                    model = load_text(a.model, device)
                with torch.no_grad():
                    values = model([row['instruction'], m['negative_prompt'], ''], device)
                out.save(name, {k:v.cpu().bfloat16() for k,v in zip(('text','negative','empty'),values)}, m['identity'])
            reports.append(dict(id=row['id'], status='completed'))
        except Exception as exc:
            reports.append(dict(id=row['id'], status='failed', error=repr(exc)))
        tick(out, 'text', rank, reports, started, cfg['frames'], id=row['id'])
    save_phase(out, 'text', reports, len(rows), rank, world)


def anchors(a, cfg, out, device, rank, world):
    m = checked(a, cfg, out); rows = cohort(out)['records']
    reports, model, vae, started = [], None, None, time.monotonic()
    for row in rows[rank::world]:
        try:
            folder = 'anchors/'+row['id']
            payload, record = recovered_case(out, folder, m['identity'])
            if payload is None:
                item = out.load(row['source_file'], m['identity'], True)
                text = out.load('text/'+row['id']+'.pt', m['identity'], True)
                source, spatial = item['source'].float()/255, item['spatial']
                if model is None:
                    if validate_assets(a.model) != m['assets']:
                        raise ValueError('Native model differs from frozen bank assets')
                    model, vae = load_backbone(a.model, device), load_vae(a.model, device)
                model.eval().requires_grad_(False)
                mask = valid_pixels(source, spatial['valid_rect'])[:,None].expand(1,len(source),-1,-1)
                settings = dict(m['base_config'], sample_steps=cfg['sample_steps'])
                seed = cfg['seed']+int(digest(row['id'])[:8],16)%1000000
                with torch.no_grad():
                    context = pack_context(vae,(source*2-1).permute(1,0,2,3).to(device),mask.to(device))
                    latent = sample_native(model,context,text['text'].to(device),text['negative'].to(device),settings,seed)
                    proposal = ((decode_native(vae,latent).float().cpu().permute(1,0,2,3)+1)/2).clamp(0,1)
                if proposal.shape != source.shape or not bool(torch.isfinite(proposal).all()):
                    raise ValueError('Invalid shared native anchor proposal')
                support, diagnostics = source_support(source, proposal[0], spatial, None, cfg['anchor'])
                anchor = source[0]*(1-support)+proposal[0]*support
                payload = dict(source=source, anchor=anchor, support=support, spatial=spatial,
                    native_proposal=proposal, provenance='model_output', text=text['text'],
                    negative=text['negative'], empty=text['empty'], diagnostics=diagnostics)
            if record is None:
                record = publish_case(out,folder,m['identity'],payload,
                    dict(id=row['id'],status='completed',provenance='model_output',frontend='frozen_native_vace'),
                    [payload['source'],payload['anchor'][None].expand_as(payload['source']),payload['native_proposal']],
                    ['SOURCE','SHARED MODEL ANCHOR','FRONTEND OUTPUT'],payload['spatial']['valid_rect'],cfg['fps'])
            reports.append(record)
        except Exception as exc:
            reports.append(failure(out,'anchors',row,exc))
        tick(out,'anchors',rank,reports,started,cfg['frames'],id=row['id'])
    save_phase(out,'anchors',reports,len(rows),rank,world)


def prepare(a, cfg, out, device, rank, world):
    from utils.edit_transport_data import prepare_inputs
    from utils.editing_joint_state import build_source_edges
    from utils.editing_video_transport import propagate_reference
    m = checked(a,cfg,out); rows = cohort(out)['records']
    _,_,transport = stores(a)
    scale = transport.json('prepared.json',True)['source_scale']
    if scale is None or file_hash(a.rae) != m['teacher']['rae'] or file_hash(a.encoder) != m['teacher']['encoder']:
        raise ValueError('Need the frozen production codec and its TRAIN source normalization')
    reports, rae, encoder, started = [], None, None, time.monotonic()
    for row in rows[rank::world]:
        try:
            name = 'prepared/'+row['id']+'.pt'
            if out.load(name,m['identity']) is None:
                anchor = out.load('anchors/'+row['id']+'/result.pt',m['identity'],True)
                if anchor['provenance'] != 'model_output':
                    raise ValueError('Unexpected evaluation anchor provenance')
                if rae is None:
                    rae,encoder = load_teacher(a.rae,device).float(),load_encoder(a.encoder,device)
                inputs = prepare_inputs(anchor['source'],anchor['anchor'],anchor['support'],anchor['spatial'],
                    m['base_config'],rae,encoder,device)
                ref = build_reference(inputs,anchor['source'],anchor['anchor'],m['base_config'])
                state = prepare_rae_state(inputs,ref,rae,encoder,m['base_config'],device)
                square,rect = rae_canvas(ref['source'],ref['spatial']['valid_rect'],size=inputs['source'].shape[-1])
                options = dict(m['base_config']['transport'])
                edges = build_source_edges(square[inputs['observed_indices']],options)
                flows,confidence = [],[]
                for t in range(len(square)-1):
                    edge = propagate_reference(square[t:t+2],square[t],torch.ones_like(square[t,:1]),dict(options,flow_direct_anchor=False))
                    flows.append(edge['flows_backward'][0].half()); confidence.append(edge['confidence'][1].half())
                inputs.update(initial_delta=(state['edited_z'].float()-inputs['source_z'].float()).half(),
                    scale=torch.tensor(scale),edges=edges,full_flows_backward=torch.stack(flows),full_edge_confidence=torch.stack(confidence))
                out.save(name,dict(inputs=inputs,reference=encoded_reference(ref),rae_state=state['state'],
                    calibration=state['calibration'],provenance='model_output'),m['identity'])
            reports.append(dict(id=row['id'],status='completed',file=name))
        except Exception as exc:
            reports.append(failure(out,'prepare',row,exc))
        tick(out,'prepare',rank,reports,started,cfg['frames'],id=row['id'])
    save_phase(out,'prepare',reports,len(rows),rank,world)


def cache(a,cfg,out,device,rank,world):
    m=checked(a,cfg,out); rows=cohort(out)['records']
    reports,vae,started=[],None,time.monotonic()
    for row in rows[rank::world]:
        try:
            name='native/'+row['id']+'.pt'
            if out.load(name,m['identity']) is None:
                item=out.load('prepared/'+row['id']+'.pt',m['identity'],True)
                ref=decoded_reference(item['reference'])
                if vae is None:vae=load_vae(a.model,device)
                rgb=lambda v:(v*2-1).permute(1,0,2,3).to(device)
                mask=ref['masks'].permute(1,0,2,3).to(device)
                anchor=rgb(ref['anchor'][None])
                with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type!='cpu'):
                    edited=pack_edit_context(vae,rgb(ref['source']),rgb(ref['reference']),mask,anchor)
                    native=pack_context(vae,rgb(ref['source']),mask,[anchor])
                out.save(name,dict(context=edited.cpu().bfloat16(),native_context=native.cpu().bfloat16()),m['identity'])
            reports.append(dict(id=row['id'],status='completed'))
        except Exception as exc:reports.append(failure(out,'cache',row,exc))
        tick(out,'cache',rank,reports,started,cfg['frames'],id=row['id'])
    save_phase(out,'cache',reports,len(rows),rank,world)


def sft(a,cfg,out,device,rank,world):
    m=checked(a,cfg,out); _,states,_=stores(a)
    old=states.json('manifest.json',True)
    tc=copy.deepcopy(m['base_config'])
    tc['training']['preview_every']=0
    if cfg.get('smoke'):
        tc['training'].update(max_steps=2,min_steps=0,eval_every=1,save_every=1,log_every=1)
    training=dict(schema='matched-reference-sft-v1',source_state_identity=m['stage65'],
        config=tc,code=m['code'],arm='sft',world=world)
    training['identity']=digest(training)
    if rank==0:out.write_json('sft_contract.json',training)
    barrier()
    view=TrainingCacheView(out,states,m['stage65'],training['identity'])
    records=states.json('phases/cache.json',True)['records']
    actual=validate_assets(a.model) if int(os.environ.get('LOCAL_RANK',0))==0 else None
    if any(v!=m['assets'] for v in gather(actual,world) if v is not None):
        raise ValueError('SFT native weights differ from cached training data')
    torch.manual_seed(tc['seed']); random.seed(tc['seed'])
    if device.type!='cpu':getattr(torch,device.type).manual_seed_all(tc['seed'])
    backbone=load_backbone(a.model,device)
    torch.manual_seed(tc['seed'])
    model=ReferenceSFT(backbone,tc).to(device)
    mm=dict(old,identity=training['identity'])
    train_loop(SimpleNamespace(arm='sft',stop_after=0),tc,view,mm,records,model,device,rank,world)
    if rank==0:
        store=arm_store(out,'sft')
        run=store.json('run.json',True)
        run['architecture']='native LoRA only; shared reference context; no state bridge'
        store.write_json('run.json',run)
        arm_store(out,'sft').write_json('matching.json',dict(source_stage65=m['stage65'],
            train_ids=[r['id'] for r in records if r['status']=='completed' and r['partition']=='train'],
            monitor_ids=[r['id'] for r in old['records'] if r['partition']=='monitor'],
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            architecture='native LoRA only, same reference context and flow supervision; no state bridge',
            uses_future_targets_as_conditions=False,smoke=bool(cfg.get('smoke'))))
    barrier()
    status=arm_store(out,'sft').json('training_summary.json',True)
    if status['status']!='completed' or not status['updates']:
        raise RuntimeError('SFT did not complete successful updates: '+str(status))


def weights(a,cfg,out):
    """One CPU reader per node stages selected deltas before 8 render ranks."""
    m=checked(a,cfg,out); _,states,_=stores(a); records=[]
    for family in ('rgb','rae','sft'):
        try:
            store=arm_store(out if family=='sft' else states,family)
            identity=out.json('sft_contract.json',True)['identity'] if family=='sft' else m['stage65']
            store.load('selected.pt',digest([identity,family]),True)
            if family!='sft' and store.json('selected.pt.json',True)['sha256']!=m['selected'][family]:
                raise ValueError('Frozen checkpoint changed')
            records.append(dict(family=family,status='completed'))
        except Exception as exc:
            records.append(dict(family=family,status='failed',error=repr(exc)))
    out.write_json('staging/node'+str(os.environ.get('NODE_RANK','0'))+'.json',dict(records=records))
    if any(r['status']!='completed' for r in records):
        raise RuntimeError('Some selected deltas unavailable; independent arms can still run: '+str(records))


def selected_tasks(cfg,selected,arm):
    if arm not in cfg['main_arms']+cfg['ablation_arms']:
        raise ValueError('Arm not declared in frozen suite')
    subset=set(selected['ablation_ids'])
    tasks=[]
    for row in selected['records']:
        if arm in cfg['ablation_arms'] and row['id'] not in subset:continue
        seeds=[cfg['seed']]
        if arm in cfg['second_seed_arms'] and row['id'] in subset and not cfg.get('smoke'):
            seeds.append(cfg['second_seed'])
        tasks.extend((row,seed) for seed in seeds)
    return tasks


def render(a,cfg,out,device,rank,world):
    from utils.editing_joint_state import JointEditProjector
    from utils.editing_state_feedback import load_decoder_only
    from edit_reference_video import output_metrics
    m=checked(a,cfg,out); selected=cohort(out); spec=arm_spec(a.arm)
    family=spec['family']; tasks=selected_tasks(cfg,selected,a.arm)
    _,states,_=stores(a)
    chosen=checkpoint=None
    if family in ('rgb','rae'):
        checkpoint=m['selected'][family]
        if checkpoint is None:raise FileNotFoundError('Missing frozen '+family+' checkpoint')
    elif family=='sft':
        status=arm_store(out,'sft').json('training_summary.json',True)
        if status['status']!='completed' or not status['updates']:
            raise RuntimeError('Cannot label an incomplete/zero-update model as trained SFT')
        checkpoint=arm_store(out,'sft').json('selected.pt.json',True)['sha256']
    else:checkpoint='native'
    model=vae=decoder=None; reports=[]; started=time.monotonic()
    settings=dict(m['base_config'],sample_steps=cfg['sample_steps'])
    if spec['sample_steps'] and not cfg.get('smoke'):settings['sample_steps']=spec['sample_steps']
    for row,seed_base in tasks[rank::world]:
        key=row['id']; folder=f'render/{a.arm}/seed{seed_base}/{key}'
        feedback=projector=features=latent=prepared=None
        try:
            prepared=out.load('prepared/'+key+'.pt',m['identity'],True)
            ref=decoded_reference(prepared['reference'])
            native=out.load('native/'+key+'.pt',m['identity'],True)
            text=out.load('text/'+key+'.pt',m['identity'],True)
            seed=seed_base+int(digest(key)[:8],16)%1000000
            input_id=digest([m['identity'],selected['identity'],key,seed,
                tensor_hash(ref['source']),tensor_hash(ref['reference']),tensor_hash(ref['anchor']),tensor_hash(ref['masks'])])
            generation=digest([input_id,a.arm,checkpoint,settings])
            payload,record=recovered_case(out,folder,m['identity'])
            if payload is not None and payload.get('generation_identity')!=generation:
                raise ValueError('Saved output generation contract differs')
            if payload is None:
                load_started=time.monotonic()
                if model is None:
                    if validate_assets(a.model)!=m['assets']:raise ValueError('Native assets changed')
                    core=load_backbone(a.model,device)
                    if family=='native':model=core
                    elif family=='sft':
                        model=ReferenceSFT(core,settings).to(device)
                        contract=out.json('sft_contract.json',True)
                        chosen=arm_store(out,'sft').load('selected.pt',digest([contract['identity'],'sft']),True)
                        model.restore(chosen['model'])
                    else:
                        model=StateConditionedVACE(core,settings,settings['state_channels'][family]).to(device)
                        chosen=arm_store(states,family).load('selected.pt',digest([m['stage65'],family]),True)
                        model.restore(chosen['model'])
                    model.eval().requires_grad_(False)
                if vae is None:vae=load_vae(a.model,device)
                if family in ('rgb','rae'):
                    features=intervened_state(build_rgb_state(ref) if family=='rgb' else prepared['rae_state'].float(),a.arm).to(device)
                else:features=None
                events=[]
                if spec['feedback']:
                    inputs,solver=intervened_inputs(prepared['inputs'],cfg['feedback']['solver'],a.arm)
                    if family=='rae' and decoder is None:
                        if file_hash(a.rae)!=m['teacher']['rae']:raise ValueError('Frozen decoder changed')
                        decoder=load_decoder_only(a.rae,device,True)
                    source=inputs['source'].float().to(device)/255
                    valid=source.new_zeros(len(source),1,*source.shape[-2:])
                    y,x,h,w=inputs['valid_rect']; valid[:,:,y:y+h,x:x+w]=1
                    projector=JointEditProjector(decode=decoder if family=='rae' else None,
                        source_z=inputs['source_z'].to(device) if family=='rae' else None,
                        source=source,anchor=inputs['anchor'].float().to(device)/255,
                        masks=inputs['masks'].float().to(device),confidence=inputs['confidence'].float().to(device),
                        cfg=dict(solver,device=str(device)),initial_delta=inputs['initial_delta'].to(device) if family=='rae' else None,
                        initial_effect=(inputs['proposal'].float()-inputs['source'].float()).to(device)/255,
                        proposal=inputs['proposal'].float().to(device)/255,edges=inputs['edges'],scale=inputs['scale'].to(device),valid=valid)
                    def observer(event):
                        events.append(dict(event))
                        if event.get('applied'):
                            step=int(event['step_index']); base=folder+f'/feedback/step_{step:03d}'
                            from refine_vace_edit_state import _cpu_snapshot
                            snapshot=_cpu_snapshot(dict(reference=feedback.last_reference,state=feedback.state,
                                projection=feedback.last_projection,event=event,generation_identity=generation))
                            out.save(base+'.pt',snapshot,m['identity'])
                            rect=ref['spatial']['valid_rect']
                            write_video(out.local/(base+'.mp4'),crop_video(ref['source'],rect),crop_video(ref['reference'],rect),
                                crop_video(snapshot['reference']['reference'],rect),cfg['fps'],labels=['SOURCE','SHARED REFERENCE','LIVE CORRECTION'])
                            seal(out.local,base+'.json',m['identity'],[base+'.mp4'],dict(id=key,arm=a.arm,event=event))
                            out.publish([base+'.mp4',base+'.json'])
                        progress(out,'feedback_r'+str(rank),dict(event,id=key,experiment_arm=a.arm,seed_base=seed_base))
                    feedback=PaperFeedback(vae,projector,inputs,ref,features,
                        dict(cfg['feedback'],update_bridge_state=spec['bridge'],update_native_context=False),
                        family,apply_velocity=spec['velocity'],observer=observer)
                setup_seconds=time.monotonic()-load_started
                if device.type!='cpu':getattr(torch,device.type).synchronize()
                sample_started=time.monotonic()
                with torch.no_grad():
                    if family=='native':
                        latent=sample_native(model,native[spec['context']].to(device),text['text'].to(device),text['negative'].to(device),settings,seed)
                    else:
                        latent=sample_state_native(model,native['context'].to(device),text['text'].to(device),text['negative'].to(device),
                            features,settings,seed,use_state=spec['use_state'],velocity_callback=feedback)
                    generated=((decode_native(vae,latent,1).permute(1,0,2,3).float()+1)/2).clamp(0,1).cpu()
                if device.type!='cpu':getattr(torch,device.type).synchronize()
                if generated.shape!=ref['source'].shape or not bool(torch.isfinite(generated).all()):raise ValueError('Invalid generated video')
                payload=dict(generated=generated,spatial=ref['spatial'],input_identity=input_id,generation_identity=generation,
                    checkpoint=checkpoint,seed=seed,seed_base=seed_base,events=events,
                    sampling_seconds=time.monotonic()-sample_started,setup_seconds=setup_seconds,
                    sample_steps=settings['sample_steps'],reference_count=1,
                    peak_allocated_gib=(getattr(torch,device.type).max_memory_allocated()/2**30 if device.type!='cpu' else None))
            rect=ref['spatial']['valid_rect']
            metrics=dict(id=key,group=row['group'],status='completed',arm=a.arm,seed_base=seed_base,seed=seed,
                input_identity=input_id,generation_identity=generation,checkpoint=checkpoint,
                feedback_events=payload['events'],sampling_seconds=payload['sampling_seconds'],setup_seconds=payload['setup_seconds'],
                peak_allocated_gib=payload['peak_allocated_gib'],sample_steps=payload['sample_steps'],
                feedback_decoder_evaluations=sum(e.get('solver',{}).get('decoder_evaluations',0) for e in payload['events']),
                feedback_response_evaluations=sum(e.get('solver',{}).get('response_evaluations',0) for e in payload['events']),
                **output_metrics(crop_float(ref['source'],rect),crop_float(payload['generated'],rect),crop_float(ref['masks'],rect)))
            if record is None:
                publish_case(out,folder,m['identity'],payload,metrics,[ref['source'],ref['reference'],payload['generated']],
                    ['SOURCE','SHARED REFERENCE',a.arm],rect,cfg['fps'])
            reports.append(publish_lossless(out,folder,m['identity'],payload,cfg['fps']))
        except Exception as exc:
            reports.append(dict(failure(out,'render_'+a.arm,row,exc),arm=a.arm,seed_base=seed_base))
        finally:
            feedback=projector=features=latent=prepared=None
            release(device)
        tick(out,'render_'+a.arm,rank,reports,started,cfg['frames'],id=key,seed_base=seed_base)
    save_phase(out,'render_'+a.arm,reports,len(tasks),rank,world)


def evaluate(a,cfg,out,device,rank,world):
    from utils.editing_paper_metrics import evaluate_videos, optional_lpips, evaluation_context
    m=checked(a,cfg,out); selected=cohort(out); data,_,_=stores(a)
    bank=data.json('bank.json',True); reports=[]; started=time.monotonic()
    perceptual,perceptual_status=optional_lpips(device,enabled=cfg.get('lpips',True))
    for row in selected['records'][rank::world]:
        try:
            item=out.load(row['source_file'],m['identity'],True)
            target=load_evaluation_labels(a.data,row,bank,item)
            rect=item['spatial']['valid_rect']
            source=crop_video(item['source'].float()/255,rect)
            target=crop_video(target.float()/255,rect)
            context=evaluation_context(source,target)
            for arm in cfg['main_arms']+cfg['ablation_arms']:
                for rr,seed in selected_tasks(cfg,selected,arm):
                    if rr['id']!=row['id']:continue
                    name=f'evaluation/{arm}/seed{seed}/{row["id"]}.json'
                    try:
                        metrics=out.json(name)
                        if not metrics or metrics.get('status')!='completed':
                            payload=out.load(f'render/{arm}/seed{seed}/{row["id"]}/result.pt',m['identity'],True)
                            generated=crop_video(payload['generated'],rect)
                            values=evaluate_videos(source,target,generated,perceptual,device,context)
                            metrics=dict(id=row['id'],group=row['group'],arm=arm,seed_base=seed,status='completed',
                                input_identity=payload['input_identity'],generation_identity=payload['generation_identity'],
                                lpips=perceptual_status,metrics=values)
                            out.write_json(name,metrics)
                        reports.append(metrics)
                    except Exception as exc:
                        reports.append(dict(id=row['id'],group=row['group'],arm=arm,seed_base=seed,status='failed',error=repr(exc)))
        except Exception as exc:
            # One missing target does not suppress another case or generation.
            reports.append(dict(id=row['id'],group=row['group'],status='failed',phase='labels',error=repr(exc)))
        tick(out,'evaluate',rank,reports,started,cfg['frames'],id=row['id'])
    save_phase(out,'evaluate',reports,sum(len(selected_tasks(cfg,selected,arm)) for arm in cfg['main_arms']+cfg['ablation_arms']),rank,world)


def build_parser():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--phase',required=True,choices=('index','screen','sources','select','text','anchors','prepare','cache','sft','weights','render','evaluate','summarize'))
    p.add_argument('--arm',default='native',choices=MAIN_ARMS+ABLATION_ARMS)
    for name in ('config','output'):p.add_argument('--'+name,required=True)
    for name in ('bank','states','transport','model','rae','encoder','data','source'):p.add_argument('--'+name,default='')
    for name in ('root','bank-root','state-root','transport-root'):p.add_argument('--'+name,action='append',default=[])
    return p


def main():
    a=build_parser().parse_args(); cfg=json.loads(Path(a.config).read_text(encoding='utf8'))
    out=ArtifactStore(a.output,a.root)
    if a.phase=='index':return index(a,cfg,out)
    if a.phase=='weights':return weights(a,cfg,out)
    if a.phase=='select':
        if int(os.environ.get('NODE_RANK',os.environ.get('VC_TASK_INDEX',0)))==0:return select(a,cfg,out)
        return
    if a.phase=='summarize':
        if int(os.environ.get('NODE_RANK',os.environ.get('VC_TASK_INDEX',0)))==0:
            from utils.editing_paper_report import summarize
            return summarize(a,cfg,out)
        return
    device,rank,world=distributed()
    dispatch=dict(screen=screen,sources=sources,text=texts,anchors=anchors,prepare=prepare,
                  cache=cache,sft=sft,render=render,evaluate=evaluate)
    try:dispatch[a.phase](a,cfg,out,device,rank,world)
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':
    main()
