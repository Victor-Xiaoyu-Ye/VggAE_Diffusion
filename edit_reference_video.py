"""Stage63: natural-video reference editing, joint RAE calibration, native writeback.

No dataset edited target is used to construct any inference condition. A model
anchor and a user-provided reference are explicitly different input protocols.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn import functional as F
from utils.editing_data import digest, file_hash, rae_canvas
from utils.editing_train import distributed, gather, barrier, progress, load_teacher, load_encoder, write_video
from utils.editing_controls import load_controls, source_box_mask
from utils.scene_run import ArtifactStore
from utils.native_audit_io import recover_case, seal
from utils.editing_publication import reconcile_case
from utils.vace_edit_runtime import load_backbone, load_vae, pack_context, sample_native, decode_native


def code_signature():
    root=Path(__file__).resolve().parent
    return {p:file_hash(root/p) for p in ('edit_reference_video.py','utils/editing_video_transport.py',
        'utils/vace_edit_runtime.py','utils/editing_data.py','models/scene_rae.py')}


def select_rows(bank,count):
    return sorted((r for r in bank['records'] if r['split']=='dev'),key=lambda r:r['id'])[:count]


def stage_file(path,out,label):
    from utils.moxing_io import is_remote_path,copy_file
    if not is_remote_path(path):return str(Path(path).resolve())
    destination=out.local/'inputs'/('rank'+os.environ.get('RANK','0'))/f'{label}_{digest(path)[:12]}'/Path(path).name
    destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_name(destination.name+'.partial')
    copy_file(path,str(partial));os.replace(partial,destination)
    return str(destination)


def read_references(path,out):
    if not path:return {}
    local=Path(stage_file(path,out,'references'))
    value=json.loads(local.read_text(encoding='utf8'))
    if value.get('schema')!='edit-references-v1':raise ValueError('Expected edit-references-v1 manifest')
    result={}
    for key,row in value['cases'].items():
        if row.get('provenance') not in ('user_edit','model_output'):
            raise ValueError('Reference must declare user_edit or model_output, never dataset target')
        name=row['image']
        if not name.startswith('obs://') and not Path(name).is_absolute():
            name=(path.rsplit('/',1)[0]+'/'+name if path.startswith('obs://') else str(local.parent/name))
        name=stage_file(name,out,'reference_'+digest(key)[:16])
        result[key]=dict(row,image=name,sha256=file_hash(name))
    return result


def valid_pixels(source,rect):
    y,x,h,w=rect
    mask=source.new_zeros(1,*source.shape[-2:]);mask[:,y:y+h,x:x+w]=1
    return mask


def reference_image(path,source,rect):
    """Fit a legal reference to the source field of view without aspect stretch."""
    import numpy as np
    from PIL import Image
    raw=torch.from_numpy(np.array(Image.open(path).convert('RGB'),copy=True)).permute(2,0,1).float()/255
    y,x,h,w=rect;rh,rw=raw.shape[-2:]
    # A full-frame edited reference should depict the same field of view.
    # Fit inside it; keep source in uncovered margins rather than create black edits.
    scale=min(h/rh,w/rw);hh,ww=max(1,round(rh*scale)),max(1,round(rw*scale))
    ref=source[0].clone();yy,xx=y+(h-hh)//2,x+(w-ww)//2
    ref[:,yy:yy+hh,xx:xx+ww]=F.interpolate(raw[None],(hh,ww),mode='bilinear',align_corners=False)[0]
    return ref


def source_support(source,anchor,spatial,control,cfg):
    valid=valid_pixels(source,spatial['valid_rect'])
    if control is not None:
        support=source_box_mask(control,spatial,len(source))[:,0]*valid
        origin='manual_source_box'
    else:
        change=(anchor-source[0]).abs().amax(0,keepdim=True)[None]
        support=(F.avg_pool2d(change,5,1,2)[0]>cfg.get('anchor_support_threshold',.06)).float()*valid
        origin='generated_or_user_reference_difference_not_semantic_mask'
    k=int(cfg.get('support_dilation',9));k=max(1,k//2*2+1)
    support=F.max_pool2d(support[None],k,1,k//2)[0]*valid
    # An empty/nonexecuted edit remains a completed failure example, not a new edit.
    return support,dict(support_source=origin,support_fraction=float(support.sum()/valid.sum().clamp_min(1)),
                        anchor_change=float(((anchor-source[0]).abs()*valid).sum()/(3*valid.sum()).clamp_min(1)))


def save_image(path,value):
    from PIL import Image
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    v=(value.detach().cpu().clamp(0,1)*255).round().byte()
    if v.shape[0]==1:v=v.expand(3,-1,-1)
    Image.fromarray(v.permute(1,2,0).numpy()).save(path)


def crop_video(value,rect):
    y,x,h,w=rect
    return (value[:,:,y:y+h,x:x+w].clamp(0,1)*255).round().byte()


def publish_case(out,folder,identity,payload,metadata,panels,labels,rect,fps,images=None):
    out.save(folder+'/result.pt',payload,identity)
    movie=folder+'/comparison.mp4'
    write_video(out.local/movie,*[crop_video(v,rect) for v in panels],fps,labels=labels)
    files=[movie]
    for name,value in (images or {}).items():
        save_image(out.local/folder/name,value);files.append(folder+'/'+name)
    out.write_json(folder+'/metrics.json',metadata);files.append(folder+'/metrics.json')
    receipt=folder+'/receipt.json'
    seal(out.local,receipt,identity,files,metadata);out.publish([*files,receipt])
    return dict(metadata,receipt=receipt)


def recovered_case(out,folder,identity):
    payload=out.load(folder+'/result.pt',identity)
    if payload is None:return None,None
    try:receipt=recover_case(out.local,folder+'/receipt.json',identity,out.roots)
    except RuntimeError as exc:
        print('[reference media rebuild] '+repr(exc),flush=True);receipt=None
    if receipt:
        out.publish([folder+'/result.pt',folder+'/result.pt.json'])
        out.publish([*receipt['files'],folder+'/receipt.json'])
        return payload,dict(receipt['metadata'],receipt=folder+'/receipt.json')
    return payload,None


def finish_phase(out,phase,reports,rank,world,requested):
    all_rows=sum(gather(reports,world),[])
    if rank==0:out.write_json(f'phases/{phase}.json',dict(phase=phase,requested=requested,cases=all_rows))
    barrier()
    if not any(r.get('status')=='completed' for r in all_rows):raise RuntimeError('No completed cases in '+phase)


def failure(out,phase,row,exc):
    value=dict(id=row['id'],phase=phase,status='failed',error=repr(exc))
    out.write_json(f'failures/{phase}/{row["id"]}.json',value)
    print('[reference case failed] '+json.dumps(value),flush=True)
    return value


def backbone(a,device,adapter=False):
    model=load_backbone(a.model,device)
    if adapter and a.adapter:
        from utils.vace_edit_runtime import EditingModel
        value=torch.load(str(a.adapter),map_location='cpu',weights_only=False,mmap=os.name!='nt')
        config=(json.loads(Path(a.adapter_config).read_text()) if a.adapter_config else value['config'])
        model=EditingModel(model,config,with_rae=False).to(device).eval()
        model.restore(value['model']);model.requires_grad_(False)
        del value
    return model


def choose_anchor_adapter(a,out,rank,world):
    """Read-only reuse of fixed Stage61 RAW750; node leaders stage once.

    Auto selection is fixed before seeing any target/evaluation outcome. If an
    old checkpoint is unavailable, all nodes explicitly use native anchors.
    """
    if a.adapter!='auto':
        if a.adapter:a.adapter=stage_file(a.adapter,out,'anchor_adapter')
        return dict(mode='explicit_adapter' if a.adapter else 'native',fallback_errors=[])
    entry=None
    local_world=int(os.environ.get('LOCAL_WORLD_SIZE','8'))
    if int(os.environ.get('LOCAL_RANK','0'))==0:
        roots=[r.rsplit('/',1)[0]+'/vace_edit_aligned_v2' for r in out.roots]
        roots.extend(r.rsplit('/cache_latents/',1)[0]+'/output/scale/vace_edit_aligned_v2'
                     for r in a.data_root if '/cache_latents/' in r)
        cache=ArtifactStore(out.local/'inputs'/'stage61_anchor',roots)
        name='region_sft/checkpoints/step_000750.pt'
        try:
            checkpoint=cache.load(name,required=True)
            if checkpoint.get('step')!=750 or checkpoint.get('arm')!='region_sft':
                raise ValueError('Auto anchor checkpoint is not Stage61 regionSFT RAW750')
            del checkpoint
            entry=dict(path=str(cache.local/name),sha256=file_hash(cache.local/name))
        except Exception as exc:entry=dict(error=repr(exc))
    entries=gather(entry,world)
    leaders=[v for v in entries if v is not None]
    errors=[v['error'] for v in leaders if 'error' in v]
    if errors or not leaders or len({v.get('sha256') for v in leaders})!=1:
        a.adapter=''
        result=dict(mode='native_fallback',fallback_errors=errors or ['Node adapter signatures disagree'])
        print('[reference anchor fallback] '+json.dumps(result),flush=True)
    else:
        leader=(rank//local_world)*local_world
        a.adapter=entries[leader]['path']
        result=dict(mode='stage61_region_sft_raw750',sha256=leaders[0]['sha256'],fallback_errors=[])
    return result


def anchor_phase(a,cfg,data,out,device,rank,world):
    from utils.vace_edit_runtime import validate_assets
    bank=data.json('bank.json',True);rows=select_rows(bank,cfg['cases'])
    controls=load_controls(a.controls);references=read_references(a.references,out)
    adapter_selection=choose_anchor_adapter(a,out,rank,world)
    local=dict(model=validate_assets(a.model),adapter=file_hash(a.adapter) if a.adapter else None)
    if any(v!=local for v in gather(local,world)):raise ValueError('Nodes disagree on native/adapter weights')
    reference_identity={k:{n:v for n,v in r.items() if n!='image'} for k,r in references.items()}
    manifest=dict(schema='reference-video-v1',config=cfg,code=code_signature(),bank=bank['identity'],
        assets=local,anchor_adapter_selection=adapter_selection,controls=controls,reference_identity=reference_identity,cases=rows,
        protocol='model_anchor_or_declared_user_reference_no_dataset_targets')
    contract=dict(manifest,anchor_adapter_selection={k:v for k,v in adapter_selection.items() if k!='fallback_errors'})
    # Transient OBS request IDs/errors are diagnostics, not changed model inputs.
    manifest['identity']=digest(contract)
    if any(v!=manifest['identity'] for v in gather(manifest['identity'],world)):
        raise ValueError('Nodes disagree on source/controls/reference configuration')
    old=out.json('manifest.json')
    if old and old['identity']!=manifest['identity']:raise ValueError('Run identity changed; use a fresh REFERENCE_NAMESPACE')
    if rank==0:out.write_json('manifest.json',manifest)
    barrier();identity=manifest['identity'];model=vae=None;reports=[];start=time.monotonic();done=0
    for row in rows[rank::world]:
        folder='anchors/'+row['id']
        try:
            payload,record=recovered_case(out,folder,identity)
            if record:reports.append(record);continue
            if payload is None:
                sample=data.load(row['file'],bank['identity'],True)
                source=sample['source'].float()/255;spatial=sample['timing']['spatial'];rect=spatial['valid_rect']
                if len(source)!=cfg['frames']:raise ValueError('Source cache frame count differs')
                control=(controls or {}).get('cases',{}).get(row['id'])
                if row['id'] in references:
                    ref=references[row['id']];anchor=reference_image(ref['image'],source,rect)
                    proposal=source.clone();provenance=ref['provenance'];external=True
                else:
                    if model is None:model=backbone(a,device,adapter=True);vae=load_vae(a.model,device)
                    mask=source_box_mask(control,spatial,len(source)) if control else valid_pixels(source,rect)[:,None].expand(1,len(source),-1,-1)
                    with torch.no_grad():
                        context=pack_context(vae,(source*2-1).permute(1,0,2,3).to(device),mask.to(device))
                        seed=cfg['seed']+int(digest(row['id'])[:8],16)%1000000
                        latent=sample_native(model,context,sample['text'].to(device),sample['negative'].to(device),
                            dict(cfg,sample_steps=cfg.get('anchor_steps',cfg['sample_steps'])),seed)
                        proposal=(decode_native(vae,latent).float().cpu().permute(1,0,2,3)+1)/2
                    if proposal.shape!=source.shape or not bool(torch.isfinite(proposal).all()):raise ValueError('Invalid anchor proposal video')
                    anchor=proposal[0].clone();provenance='model_output';external=False
                support,diagnostics=source_support(source,anchor,spatial,control,cfg)
                # Only the intended support contributes to the shared reference;
                # outside it retain source content for all compared methods.
                anchor=source[0]*(1-support)+anchor*support
                payload=dict(source=source,anchor=anchor,support=support,spatial=spatial,
                    native_proposal=proposal,text=sample['text'],negative=sample['negative'],
                    provenance=provenance,external=external,diagnostics=diagnostics)
            source=payload['source'];rect=payload['spatial']['valid_rect']
            meta=dict(id=row['id'],phase='anchors',status='completed',provenance=payload['provenance'],
                external_reference=payload['external'],**payload['diagnostics'])
            reports.append(publish_case(out,folder,identity,payload,meta,
                [source,payload['anchor'][None].expand_as(source),payload['native_proposal']],
                ['SOURCE','SHARED EDIT REFERENCE (fixed)','ANCHOR FRONTEND / SOURCE if external'],rect,cfg['fps'],
                {'anchor.png':payload['anchor'],'support.png':payload['support']}))
            done+=1;progress(out,f'anchors_r{rank}',dict(phase='anchors',completed=done,
                DI_throughput=done*cfg['frames']/max(time.monotonic()-start,1e-6),throughput_unit='processed source frames/s/rank including model-anchor inference and IO'))
        except Exception as exc:reports.append(failure(out,'anchors',row,exc))
    finish_phase(out,'anchors',reports,rank,world,len(rows))


def checked_manifest(out,cfg):
    manifest=out.json('manifest.json',True)
    if manifest['config']!=cfg or manifest['code']!=code_signature():raise ValueError('Code/config changed; use a fresh REFERENCE_NAMESPACE')
    return manifest


def from_square(value,square_rect,spatial):
    y,x,h,w=square_rect;top,left,hh,ww=spatial['valid_rect']
    result=value.new_zeros(len(value),value.shape[1],*spatial['canvas'])
    result[:,:,top:top+hh,left:left+ww]=F.interpolate(value[:,:,y:y+h,x:x+w],(hh,ww),mode='bilinear',align_corners=False)
    return result


def transport_phase(a,cfg,out,device,rank,world):
    from audit_edit_transport import encode_window,rae_decoder
    from utils.editing_video_transport import propagate_reference,calibrate_video,warp_flow
    manifest=checked_manifest(out,cfg);identity=manifest['identity'];rows=manifest['cases']
    sig=dict(rae=file_hash(a.rae),encoder=file_hash(a.encoder))
    if any(v!=sig for v in gather(sig,world)):raise ValueError('Frozen weights differ across ranks')
    previous=out.json('teacher_identity.json')
    if previous and previous!=sig:raise ValueError('Frozen weights changed; choose a fresh namespace')
    if rank==0:out.write_json('teacher_identity.json',sig)
    barrier();rae=encoder=None;reports=[];start=time.monotonic();done=0
    for row in rows[rank::world]:
        folder='transport/'+row['id']
        try:
            payload,record=recovered_case(out,folder,identity)
            if record:reports.append(record);continue
            anchor=out.load('anchors/'+row['id']+'/result.pt',identity,True)
            source=anchor['source'];spatial=anchor['spatial'];rect=spatial['valid_rect']
            if payload is None:
                square,srect=rae_canvas(source,rect)
                ar,_=rae_canvas(anchor['anchor'][None],rect);mask,_=rae_canvas(anchor['support'][None],rect)
                tcfg=cfg['transport']
                full=propagate_reference(square,ar[0],mask[0],tcfg)
                indices=list(range(0,len(source),2))
                coarse=propagate_reference(square[indices],ar[0],mask[0],tcfg)
                encoded_name=folder+'/source_states.pt'
                cached=out.load(encoded_name,identity)
                if cached is None:
                    if rae is None:rae=load_teacher(a.rae,device).float()
                    if encoder is None:encoder=load_encoder(a.encoder,device)
                    z=encode_window(rae,encoder,square[indices][None].to(device))
                    cached=dict(z=z.cpu(),indices=indices,teacher=sig)
                    out.save(encoded_name,cached,identity)
                    encoder=None;gc.collect()
                    if device.type!='cpu':getattr(torch,device.type).empty_cache()
                if cached['teacher']!=sig:raise ValueError('Source states use a different teacher')
                if rae is None:rae=load_teacher(a.rae,device).float()
                z=cached['z'].to(device).float()
                resume=out.load(folder+'/optimizer.pt',identity)
                def save(state):out.save(folder+'/optimizer.pt',state,identity)
                def log(state):progress(out,f'transport_r{rank}',dict(phase='joint_rae',id=row['id'],**state))
                solved=calibrate_video(rae_decoder(rae),z,square[indices].to(device),coarse['proposal'].to(device),
                    coarse['confidence'].to(device),coarse['masks'].to(device),tcfg,
                    checkpoint=save,resume=resume,progress=log,flows_backward=coarse['flows_backward'].to(device))
                effects=solved['effect'].detach().cpu()
                # Preserve full-rate optical-flow proposal; add calibrated
                # corrections at even states, source-warped corrections at odds.
                correction=effects-(coarse['proposal']-square[indices])
                all_correction=torch.zeros_like(square)
                for j,t in enumerate(indices):
                    all_correction[t]=correction[j]
                    if t+1<len(square):
                        all_correction[t+1]=warp_flow(correction[j:j+1],full['flows_backward'][t])[0]
                calibrated=(full['proposal']+all_correction*full['confidence']).clamp(0,1)
                # Transfer only edit differences back to the native canvas.
                # Resizing full RGB here would blur source details even at zero edit.
                proposal=(source+from_square(full['proposal']-square,srect,spatial)).clamp(0,1)
                reference=(source+from_square(calibrated-square,srect,spatial)).clamp(0,1)
                confidence=from_square(full['confidence'],srect,spatial).clamp(0,1)
                masks=from_square(full['masks'],srect,spatial).clamp(0,1)
                # Reinsert exact native source for padded/unsupported content.
                valid=valid_pixels(source,rect)[None]
                reference=reference*valid+source*(1-valid);proposal=proposal*valid+source*(1-valid)
                payload=dict(reference=reference,proposal=proposal,confidence=confidence,masks=masks,
                    delta=solved['delta'].detach().cpu(),history=solved['history'],flow_metadata=full['metadata'],
                    observed_indices=indices,square_rect=srect,teacher=sig)
                del z,solved
            meta=dict(id=row['id'],phase='transport',status='completed',flow=payload['flow_metadata'],
                confidence_mean=float(payload['confidence'].mean()),mask_fraction=float(payload['masks'].mean()),
                calibrated_vs_rgb_l1=float((payload['reference']-payload['proposal']).abs().mean()),
                observed_indices=payload['observed_indices'])
            reports.append(publish_case(out,folder,identity,payload,meta,[source,payload['proposal'],payload['reference']],
                ['SOURCE','SOURCE FLOW EDIT PROPOSAL','JOINT RAE EDIT REFERENCE'],rect,cfg['fps']))
            done+=1;progress(out,f'transport_phase_r{rank}',dict(phase='transport',completed=done,
                DI_throughput=done*cfg['frames']/max(time.monotonic()-start,1e-6),throughput_unit='processed source frames/s/rank including flow, RAE solve and IO'))
        except Exception as exc:reports.append(failure(out,'transport',row,exc))
    finish_phase(out,'transport',reports,rank,world,len(rows))


def output_metrics(source,generated,masks):
    change=(generated-source).abs().mean(1,keepdim=True)
    return dict(source_change=float(change.mean()),edit_region_change=float((change*masks).sum()/masks.sum().clamp_min(1)),
        keep_change=float((change*(1-masks)).sum()/(1-masks).sum().clamp_min(1)),
        output_frame_delta=float((generated[1:]-generated[:-1]).abs().mean()),
        caveat='Source-only diagnostics; low change is not successful editing. No target or geometry quality score.')


def tensor_hash(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def render_phase(a,cfg,out,device,rank,world):
    from utils.vace_edit_runtime import build_reference_writeback
    manifest=checked_manifest(out,cfg);identity=manifest['identity'];rows=manifest['cases']
    model=vae=None;reports=[];start=time.monotonic();done=0
    for row in rows[rank::world]:
        for variant in ('native_reference','rae_writeback'):
            folder=f'render/{variant}/{row["id"]}'
            try:
                transported=out.load('transport/'+row['id']+'/result.pt',identity)
                payload,record=recovered_case(out,folder,identity)
                if payload is not None and transported is not None and payload['mask_origin']!='shared_source_flow':
                    print('[reference regenerate] replacing fallback-mask baseline after transport recovery: '+row['id'],flush=True)
                    payload=record=None
                if record:reports.append(record);continue
                anchor=out.load('anchors/'+row['id']+'/result.pt',identity,True)
                source=anchor['source'];rect=anchor['spatial']['valid_rect']
                if payload is None:
                    if transported is None:
                        if variant=='rae_writeback':raise RuntimeError('RAE transport unavailable; native baseline remains separately usable')
                        # Source-flow support is still available without the RAE teacher.
                        from utils.editing_video_transport import propagate_reference
                        propagated=propagate_reference(source,anchor['anchor'],anchor['support'],cfg['transport'])
                        masks=propagated['masks'];mask_origin='native_resolution_flow_fallback'
                    else:masks=transported['masks'];mask_origin='shared_source_flow'
                    if model is None:model=backbone(a,device);vae=load_vae(a.model,device)
                    with torch.no_grad():
                        context=pack_context(vae,(source*2-1).permute(1,0,2,3).to(device),
                            masks.permute(1,0,2,3).to(device),[(anchor['anchor']*2-1)[:,None].to(device)])
                    events=[]
                    def observer(event):
                        events.append(event);progress(out,f'writeback_r{rank}',dict(phase='native_writeback',id=row['id'],**event))
                    callback=None
                    if variant=='rae_writeback':
                        callback=build_reference_writeback(vae,(transported['reference']*2-1).permute(1,0,2,3).to(device),
                            cfg,reference_count=1,mask=(transported['confidence']*masks).permute(1,0,2,3).to(device),observer=observer)
                    seed=cfg['seed']+int(digest(row['id'])[:8],16)%1000000
                    with torch.no_grad():
                        latent=sample_native(model,context,anchor['text'].to(device),anchor['negative'].to(device),
                            cfg,seed,velocity_callback=callback)
                        generated=((decode_native(vae,latent,reference_count=1).float().cpu()+1)/2).permute(1,0,2,3)
                    if generated.shape!=source.shape or not bool(torch.isfinite(generated).all()):raise ValueError('Invalid rendered video')
                    inputs=digest([identity,row['id'],seed,tensor_hash(masks),tensor_hash(anchor['anchor'])])
                    payload=dict(generated=generated,events=events,masks=masks,mask_origin=mask_origin,seed=seed,input_identity=inputs)
                meta=dict(id=row['id'],phase='render',variant=variant,status='completed',seed=payload['seed'],
                    provenance=anchor['provenance'],mask_origin=payload['mask_origin'],writeback_events=payload['events'],
                    input_identity=payload['input_identity'],
                    **output_metrics(crop_float(source,rect),crop_float(payload['generated'],rect),crop_float(payload['masks'],rect)))
                reports.append(publish_case(out,folder,identity,payload,meta,
                    [source,anchor['anchor'][None].expand_as(source),payload['generated']],
                    ['SOURCE','SHARED EDIT REFERENCE (fixed)',variant.upper()],rect,cfg['fps']))
                done+=1;progress(out,f'render_r{rank}',dict(phase='render',variant=variant,completed=done,
                    DI_throughput=done*cfg['frames']/max(time.monotonic()-start,1e-6),throughput_unit='generated frames/s/rank including decode, writeback and IO'))
            except Exception as exc:
                failed=failure(out,'render_'+variant,row,exc);failed['variant']=variant;reports.append(failed)
    finish_phase(out,'render',reports,rank,world,len(rows)*2)


def crop_float(value,rect):
    y,x,h,w=rect;return value[:,:,y:y+h,x:x+w]


def summarize(out,cfg):
    from html import escape
    manifest=checked_manifest(out,cfg);rows=manifest['cases'];identity=manifest['identity']
    reports={p:out.json(f'phases/{p}.json') for p in ('anchors','transport','render')}
    failures=[];replication=[];completed={}
    for phase,report in reports.items():
        if report is None:failures.append(dict(phase=phase,error='Missing phase summary'));completed[phase]=0;continue
        expected=len(rows)*(2 if phase=='render' else 1)
        cases=report['cases'];completed[phase]=sum(r['status']=='completed' for r in cases)
        if len(cases)!=expected:failures.append(dict(phase=phase,error='Case coverage mismatch',expected=expected,actual=len(cases)))
        failures.extend(r for r in cases if r['status']!='completed')
        for record in cases:
            if record['status']!='completed':continue
            try:
                roots=reconcile_case(out,record['receipt'],identity)
                replication.append(dict(receipt=record['receipt'],roots=roots,ok=all(r['ok'] for r in roots)))
            except Exception as exc:replication.append(dict(receipt=record['receipt'],ok=False,error=repr(exc)))
    gallery=['<!doctype html><meta charset="utf-8"><title>Natural reference video editing</title>',
        '<style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}video{max-width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>Source / shared reference / actual native output</h1>',
        '<p>Native and RAE use identical source, anchor, mask, seed and scheduler. RAE adds calibrated late writeback. No dataset target is a condition. Model anchors can fail; user-reference cases are separate protocols. No SAM2 or learned-flow claim.</p>']
    # Save matched videos as a separate receipt so their recovery never needs a model.
    comparisons=[]
    for row in rows:
        key=row['id'];folder='comparisons/'+key
        try:
            anchor=out.load('anchors/'+key+'/result.pt',identity,True)
            native=out.load('render/native_reference/'+key+'/result.pt',identity,True)
            rae=out.load('render/rae_writeback/'+key+'/result.pt',identity,True)
            if native['input_identity']!=rae['input_identity']:
                raise ValueError('Baseline/full source-reference-mask-seed conditions differ; rerun render')
            receipt=folder+'/receipt.json'
            try:old=recover_case(out.local,receipt,identity,out.roots)
            except RuntimeError:old=None
            if not old:
                movie=folder+'/source_native_rae.mp4';rect=anchor['spatial']['valid_rect']
                write_video(out.local/movie,*[crop_video(v,rect) for v in (anchor['source'],native['generated'],rae['generated'])],
                    cfg['fps'],labels=['SOURCE','NATIVE + SHARED REFERENCE','RAE + ACTUAL WRITEBACK'])
                seal(out.local,receipt,identity,[movie],dict(id=key,provenance=anchor['provenance']))
            record=out.json(receipt,True);out.publish([*record['files'],receipt])
            roots=reconcile_case(out,receipt,identity);replication.append(dict(receipt=receipt,roots=roots,ok=all(r['ok'] for r in roots)))
            delta=float((native['generated']-rae['generated']).abs().mean())
            comparisons.append(dict(id=key,output_difference=delta,writeback_count=len(rae['events']),provenance=anchor['provenance']))
            gallery.extend(['<h2>'+escape(key)+' / '+escape(anchor['provenance'])+'</h2>',
                '<video controls preload="metadata" src="comparisons/'+escape(key)+'/source_native_rae.mp4"></video>'])
        except Exception as exc:failures.append(dict(id=key,phase='comparison',error=repr(exc)))
    bad=[r for r in replication if not r['ok']]
    report=dict(status='completed' if not failures and not bad else 'incomplete',source_requested=len(rows),completed=completed,
        failures=failures,replication_failed=bad,comparisons=comparisons,
        limitations=['No quality guarantee from completion','No unique RAE advantage until matched RGB/VAE writeback ablations',
            'Source-only CPU classical flow; no SAM2 integration','Model-generated anchor quality is part of end-to-end failure'])
    out.write_json('summary.json',report);out.write_json('media_sync.json',dict(status='synced' if not bad else 'incomplete',checked=len(replication),failed=bad))
    gallery.append('<pre>'+escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre>')
    (out.local/'report.html').write_text('\n'.join(gallery),encoding='utf8');out.publish(['report.html'])
    print('[reference summary] '+json.dumps(report),flush=True)
    if report['status']!='completed':raise RuntimeError('Incomplete cases/publication; completed results retained, inspect summary.json')


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('anchors','transport','render','summarize'),required=True)
    for name in ('config','output'):p.add_argument('--'+name,required=True)
    for name in ('data','model','rae','encoder','controls','references','adapter','adapter-config'):p.add_argument('--'+name,default='')
    p.add_argument('--root',action='append',default=[]);p.add_argument('--data-root',action='append',default=[])
    return p


def main():
    a=build_parser().parse_args();cfg=json.loads(Path(a.config).read_text());out=ArtifactStore(a.output,a.root)
    if cfg['frames']!=17:raise ValueError('This first real-video milestone uses the existing 17-frame cache')
    if a.phase=='summarize':
        if int(os.environ.get('NODE_RANK','0'))==0:summarize(out,cfg)
        return
    device,rank,world=distributed()
    try:
        if a.phase=='anchors':anchor_phase(a,cfg,ArtifactStore(a.data,a.data_root),out,device,rank,world)
        elif a.phase=='transport':transport_phase(a,cfg,out,device,rank,world)
        else:render_phase(a,cfg,out,device,rank,world)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
