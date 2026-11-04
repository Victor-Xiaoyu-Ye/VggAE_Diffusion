"""Stage62: frozen-codec edit-effect transport; independent of Stage61 training.

Controlled synthetic 2D correspondences and analytic spatial edits. This is NOT
a text/video generation score, learned optical flow, or native 14B writeback.
"""
import argparse
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn import functional as F
from utils.editing_data import digest,file_hash,rae_canvas
from utils.editing_train import distributed,barrier,gather,progress,load_teacher,load_encoder,write_video
from utils.scene_run import ArtifactStore
from utils.native_audit_io import recover_case,seal
from utils.editing_publication import reconcile_case
from utils.edit_transport_probe import (warp,synthetic_sequence,analytic_edit,observation_mask,
    rgb_completion,effect_metrics,solve_effect)


def code_signature():
    root=Path(__file__).resolve().parent
    return {n:file_hash(root/n) for n in ('audit_edit_transport.py','utils/edit_transport_probe.py',
        'models/scene_rae.py','models/dual_stream_decoder.py','utils/editing_data.py',
        'utils/vace_edit_runtime.py','Wan2.1/wan/modules/vae.py')}


def save_panel(path,images,labels):
    from PIL import Image,ImageDraw
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    cells=[]
    for image,label in zip(images,labels):
        array=(image.detach().cpu()[0].clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
        canvas=Image.fromarray(array)
        ImageDraw.Draw(canvas).text((5,5),label,fill='white',stroke_width=1,stroke_fill='black')
        cells.append(canvas)
    result=Image.new('RGB',(sum(c.width for c in cells),max(c.height for c in cells)))
    offset=0
    for cell in cells:result.paste(cell,(offset,0));offset+=cell.width
    result.save(path)


def select_rows(bank,count):
    return sorted([r for r in bank['records'] if r['split']=='dev'],key=lambda r:r['id'])[:count]


def encode_window(rae,encoder,frames):
    from data.token_utils import strip_special_tokens
    with torch.no_grad(),torch.autocast(frames.device.type,dtype=torch.bfloat16,enabled=frames.device.type!='cpu'):
        tokens,start=encoder(frames)
        return rae.encode(strip_special_tokens(tokens,start),frames).float()


def rae_decoder(rae):
    def decode(z):return rae.decode(z)[:,0].permute(0,3,1,2).float()
    return decode


def anchor_effects(decode,base,delta,strengths):
    """Evaluate EACH actual anchor intervention; don't assume D is linear in dose."""
    with torch.no_grad():
        reference=decode(base).float()
        return {s:(decode(base+s*delta).float()-reference).cpu() for s in strengths}


def prepare(a,cfg,data,out,device,rank,world):
    bank=data.json('bank.json',True);rows=select_rows(bank,cfg['cases'])
    if not rows:raise ValueError('Stage60 bank has no development source videos')
    local=(dict(rae=file_hash(a.rae),encoder=file_hash(a.encoder))
           if int(os.environ.get('LOCAL_RANK',0))==0 else None)
    signatures=[s for s in gather(local,world) if s is not None]
    if any(s!=signatures[0] for s in signatures):raise ValueError('Nodes have different frozen teacher weights')
    manifest=dict(schema='edit-transport-input-v1',config=cfg,bank=bank['identity'],world=world,
        signatures=signatures[0],code=code_signature(),cases=[dict(id=r['id'],file=r['file']) for r in rows])
    identity=digest(manifest);manifest['identity']=identity
    previous=out.json('manifest.json')
    if previous and previous['identity']!=identity:raise ValueError('Run contract changed; choose a new TRANSPORT_NAMESPACE')
    if rank==0:out.write_json('manifest.json',manifest)
    barrier();rae=encoder=None;reports=[];started=time.monotonic();completed=encoded=cache_hits=0
    for row in rows[rank::world]:
        name='source/'+row['id']+'.pt'
        try:
            value=out.load(name,identity)
            if value is None:
                item=data.load(row['file'],bank['identity'],True)
                # Only source and spatial metadata are used. Dataset target is
                # deliberately absent from the synthetic edit/correspondence path.
                square,rect=rae_canvas(item['source'][:1].float()/255,item['timing']['spatial']['valid_rect'])
                frames,_,_=synthetic_sequence(square,rect,cfg)
                edited,effect,support=analytic_edit(square,rect,cfg)
                anchor_window=frames.clone();anchor_window[0]=edited[0]
                if rae is None:rae=load_teacher(a.rae,device)
                if encoder is None:encoder=load_encoder(a.encoder,device)
                z=encode_window(rae,encoder,frames[None].to(device))
                za=encode_window(rae,encoder,anchor_window[None].to(device))
                if z.shape!=(1,cfg['frames'],18,18,256) or not torch.isfinite(z).all() or not torch.isfinite(za).all():
                    raise ValueError('Need production C256, 18x18, one source state per synthetic frame')
                decode=rae_decoder(rae)
                with torch.no_grad():
                    recon=torch.cat([decode(z[:,i:i+1]).cpu() for i in range(cfg['frames'])])
                    anchor_recon=decode(za[:,:1]).cpu()
                if not bool(torch.isfinite(recon).all() and torch.isfinite(anchor_recon).all()):
                    raise FloatingPointError('Nonfinite frozen-decoder reconstruction')
                y,x,h,w=rect
                psnr=float(-10*(recon[0,:,y:y+h,x:x+w]-square[0,:,y:y+h,x:x+w]).square().mean().clamp_min(1e-10).log10())
                value=dict(square=square,rect=rect,z=z.cpu(),delta_anchor=(za[:,:1]-z[:,:1]).cpu(),
                    anchor_effect=anchor_recon-recon[:1],analytic_effect=effect,support=support,
                    source_reconstruction_psnr=psnr,source_row_id=row['id'],
                    protocol='synthetic_2d_source_and_analytic_edit_not_generated_video')
                out.save(name,value,identity)
                encoded+=1
                movie='source_previews/'+row['id']+'.mp4'
                # Nine synthetic poses, not a natural generated video. Known
                # analytic edit is independently specified from decoder output.
                _,grids,_=synthetic_sequence(square,rect,cfg)
                target=torch.cat([(frames[i:i+1]+warp(effect,g)).clamp(0,1) for i,g in enumerate(grids)])
                values=[(v.clamp(0,1)*255).round().byte() for v in (frames,recon,target)]
                write_video(out.local/movie,*values,4,labels=['SYNTHETIC 2D SOURCE','FROZEN AE RECON','ANALYTIC EDIT - NOT GENERATED'])
                seal(out.local,'source_previews/'+row['id']+'.json',identity,[movie],dict(id=row['id'],psnr=psnr))
                out.publish([movie,'source_previews/'+row['id']+'.json'])
                del z,za
            else:
                cache_hits+=1
                out.publish([name,name+'.json'])
                receipt='source_previews/'+row['id']+'.json'
                try:recovered=recover_case(out.local,receipt,identity,out.roots)
                except RuntimeError as exc:
                    print('[transport source preview regenerate] '+repr(exc),flush=True);recovered=None
                if recovered:out.publish([*recovered['files'],receipt])
                else:
                    if rae is None:rae=load_teacher(a.rae,device)
                    frames,grids,_=synthetic_sequence(value['square'],value['rect'],cfg)
                    zs=value['z'].to(device)
                    with torch.no_grad():recon=torch.cat([rae_decoder(rae)(zs[:,i:i+1]).cpu() for i in range(cfg['frames'])])
                    target=torch.cat([(frames[i:i+1]+warp(value['analytic_effect'],g)).clamp(0,1) for i,g in enumerate(grids)])
                    movie='source_previews/'+row['id']+'.mp4'
                    write_video(out.local/movie,*[(v.clamp(0,1)*255).round().byte() for v in (frames,recon,target)],4,
                        labels=['SYNTHETIC 2D SOURCE','FROZEN AE RECON','ANALYTIC EDIT - NOT GENERATED'])
                    seal(out.local,receipt,identity,[movie],dict(id=row['id'],psnr=value['source_reconstruction_psnr']))
                    out.publish([movie,receipt])
            reports.append(dict(id=row['id'],file=name,psnr=value['source_reconstruction_psnr']))
            completed+=1
            progress(out,f'prepare_r{rank}',dict(phase='source_rae_encoding',completed=completed,
                newly_encoded=encoded,cache_hits=cache_hits,
                DI_throughput=encoded*cfg['frames']/max(1e-6,time.monotonic()-started),throughput_unit='newly encoded source frames/s/npu including anchor encode, recon and IO'))
        except Exception as exc:
            reports.append(dict(id=row['id'],error=repr(exc)));out.write_json(f'failures/prepare/{row["id"]}.json',reports[-1])
            print('[transport source failed] '+json.dumps(reports[-1]),flush=True)
    reports=sum(gather(reports,world),[])
    if rank==0:out.write_json('source_bank.json',dict(identity=identity,records=reports,requested=len(rows)))
    barrier()
    if not any('error' not in r for r in reports):raise RuntimeError('No source states could be encoded')


def cell(out,identity,key,metadata,source,truth,valid,observed,decode=None,base=None,scale=None,
         cfg=None,initial=None,gradient_base=None,prediction=None,preview=False,rank=0,observations=None):
    folder='cells/'+key;receipt=folder+'/receipt.json';contract=digest([identity,key])
    try:recovered=recover_case(out.local,receipt,contract,out.roots)
    except RuntimeError as exc:
        print('[transport incomplete cell regenerate] '+repr(exc),flush=True);recovered=None
    if recovered:
        out.publish([*recovered['files'],receipt]);return dict(recovered['metadata'],receipt=receipt)
    checkpoint_name=folder+'/optimizer.pt'
    try:
        if prediction is None:
            resume=out.load(checkpoint_name,contract)
            def checkpoint(state):out.save(checkpoint_name,state,contract)
            def heartbeat(row):progress(out,f'solver_r{rank}',dict(phase='inverse_edit',cell=key,**row))
            inputs=truth*observed if observations is None else observations
            solved=solve_effect(decode,base,inputs,observed,scale,cfg,initial,gradient_base,
                resume,checkpoint,heartbeat)
            prediction=solved['effect'].cpu()
            payload=dict(delta=solved['delta'].detach().cpu(),effect_preview=F.adaptive_avg_pool2d(prediction,(128,128)).half(),history=solved['history'])
            extra={k:solved[k] for k in ('gradient_checked','evaluations','elapsed')}
        else:payload=dict(effect_preview=F.adaptive_avg_pool2d(prediction,(128,128)).half());extra={}
        if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('Nonfinite effect prediction')
        metrics=dict(metadata,**effect_metrics(prediction,truth,valid,observed),**extra,status='completed')
        metrics['evaluation_shape']=list(truth.shape[-2:]);metrics['stored_effect_shape']=[128,128]
        if extra:metrics['zero_input_gradient']=not extra['gradient_checked']
        out.save(folder+'/result.pt',payload,contract);out.write_json(folder+'/metrics.json',metrics)
        files=[folder+'/result.pt',folder+'/result.pt.json',folder+'/metrics.json']
        if preview:
            name=folder+'/comparison.png'
            shown=truth*observed if observations is None else observations
            save_panel(out.local/name,[source,(source+shown).clamp(0,1),(source+truth).clamp(0,1),
                (source+prediction).clamp(0,1)],['SYNTH SOURCE','OBSERVED EFFECT ONLY','KNOWN EFFECT TARGET','READ-WRITE '+metadata['method']])
            files.append(name)
        seal(out.local,receipt,contract,files,metrics);out.publish([*files,receipt])
        return dict(metrics,receipt=receipt)
    except Exception as exc:
        result=dict(metadata,status='failed',error=repr(exc));out.write_json(folder+'/failure.json',result)
        print('[transport cell failed] '+json.dumps(result),flush=True)
        return result


def native_image_codec(vae,image):
    """Wan image-mode T=1 control; keep full 518 canvas by pad-to-528.

    Bypass inference wrapper autocast/in-place clamp, preserving FP32 input
    gradients through the real frozen decoder. model cache resets each call.
    """
    h,w=image.shape[-2:];ph,pw=(-h)%16,(-w)%16;top,left=ph//2,pw//2
    pixels=F.pad(image,(left,pw-left,top,ph-top),value=0)*2-1
    scale=[v.float() for v in vae.scale]
    with torch.no_grad():base=vae.model.encode(pixels[:,:,None],scale).float()
    def decode(z):return ((vae.model.decode(z,scale)[:,:,0].float().clamp(-1,1)+1)/2)[:,:,top:top+h,left:left+w]
    std=base.detach().cpu().flatten(2).std(-1,unbiased=False).clamp_min(.05)[:,:,None,None,None]
    return base,decode,std.to(base.device)


def run_codec(a,cfg,out,device,rank,world):
    manifest=out.json('manifest.json',True);source_bank=out.json('source_bank.json',True)
    if manifest['config']!=cfg or manifest['code']!=code_signature():raise ValueError('Probe code/config changed; select a fresh namespace')
    identity=manifest['identity'];codec=a.phase;model=None;reports=[]
    if codec=='vae':
        signature=file_hash(Path(a.model)/'Wan2.1_VAE.pth') if int(os.environ.get('LOCAL_RANK',0))==0 else None
    else:signature=file_hash(a.rae) if int(os.environ.get('LOCAL_RANK',0))==0 else None
    signatures=[v for v in gather(signature,world) if v is not None]
    if any(v!=signatures[0] for v in signatures):raise ValueError('Codec weights differ across nodes')
    if codec=='rae' and signatures[0]!=manifest['signatures']['rae']:raise ValueError('Production decoder changed')
    phase_identity=digest([identity,codec,signatures[0]])
    def record(result):
        reports.append(result)
        # Small ledgers preserve usable partial results even if another worker
        # dies before the final collective/phase summary.
        out.write_json(f'shards/{codec}_rank{rank:03d}.json',dict(identity=phase_identity,
            manifest_identity=identity,rank=rank,cases=reports))
    for row in [r for r in source_bank['records'] if 'error' not in r][rank::world]:
        item=out.load(row['file'],identity,True);square=item['square']
        frames,grids,visibility=synthetic_sequence(square,item['rect'],cfg)
        if model is None:
            if codec=='rae':model=load_teacher(a.rae,device)
            else:
                from utils.vace_edit_runtime import load_vae
                model=load_vae(a.model,device)
                model.model.float()
        z=item['z'].to(device).float() if codec=='rae' else None
        scale=(item['z'].flatten(0,-2).std(0,unbiased=False).clamp_min(.05).to(device) if codec=='rae' else None)
        anchors=(anchor_effects(rae_decoder(model),z[:,:1],item['delta_anchor'].to(device),cfg['strengths']) if codec=='rae' else None)
        for frame in cfg['target_indices']:
            grid=grids[frame];valid=visibility[frame];src=frames[frame:frame+1]
            if codec=='rae':base=z[:,frame:frame+1];decode=rae_decoder(model)
            else:base,decode,scale=native_image_codec(model,src.to(device))
            with torch.no_grad():codec_reconstruction=decode(base).cpu()
            codec_psnr=float(-10*(((codec_reconstruction-src).square()*valid).sum()/(valid.sum()*3).clamp_min(1)).clamp_min(1e-10).log10())
            for strength in cfg['strengths']:
                suffix=f't{frame:02d}_s{round(strength*100):03d}'
                meta=dict(id=row['id'],codec=codec,frame=frame,strength=strength,latent_parameters=base.numel(),
                    source_rae_frame0_psnr=row['psnr'],codec_reconstruction_psnr=codec_psnr,
                    protocol='controlled_synthetic_2d_effect')
                if codec=='rae':
                    truth=warp(anchors[strength],grid)
                    latent_grid=F.interpolate(grid.permute(0,3,1,2),base.shape[2:4],mode='bilinear',align_corners=False).permute(0,2,3,1)
                    def warp_code(value):return warp(value[:,0].permute(0,3,1,2).cpu(),latent_grid).permute(0,2,3,1)[:,None].to(device)
                    initial=warp_code(item['delta_anchor']*strength)
                    with torch.no_grad():naive=(decode(base+initial)-decode(base)).cpu()
                    # A one-scalar calibration is a strong ordinary control:
                    # state-specific optimization must do more than rescale edits.
                    gain=float((naive*truth*valid).sum()/(naive.square()*valid).sum().clamp_min(1e-12))
                    for method in ('no_edit','rgb_oracle','naive_token','naive_gain_fit','calibrated','fixed_response'):
                        if method=='fixed_response' and not cfg['gradient_control']:continue
                        pred={'no_edit':torch.zeros_like(truth),'rgb_oracle':truth,'naive_token':naive,'naive_gain_fit':naive*gain}.get(method)
                        metrics=dict(meta,scenario='h1_dense_codec_effect',method=method,observation_fraction=1.,noise_std=0.)
                        if method=='naive_gain_fit':metrics['fitted_scalar_gain']=gain
                        record(cell(out,phase_identity,f'{codec}/{row["id"]}/h1_{suffix}_{method}',metrics,
                            src,truth,valid,valid,decode,base,scale,cfg,initial,
                            warp_code(item['z'][:,:1]) if method=='fixed_response' else None,pred,
                            preview=(frame==cfg['target_indices'][-1] and strength==max(cfg['strengths'])),rank=rank))
                truth=warp(item['analytic_effect']*strength,grid)
                support=warp(item['support'],grid)
                for fraction in cfg['observed_fractions']:
                    for noise in cfg.get('noise_stds',[0.,.005]):
                        # Nested observation sets and the same noise realization
                        # across doses make the strength/coverage curves comparable.
                        seed=cfg['seed']+int(digest([row['id'],frame])[:8],16)%1000000
                        mask=observation_mask(valid,fraction,cfg['observation_tiles'],seed,support)
                        perturb=torch.randn(truth.shape,generator=torch.Generator().manual_seed(seed+7))*noise
                        observations=(truth+perturb)*mask
                        # Optimization input is the observed tensor. Evaluation
                        # truth is kept separate, never used for choosing best state.
                        label=suffix+f'_o{round(fraction*100):03d}_n{round(noise*10000):04d}'
                        metrics=dict(meta,scenario='h2_analytic_sparse_effect',method='calibrated',
                            observation_fraction=fraction,noise_std=noise)
                        record(observed_cell(out,phase_identity,f'{codec}/{row["id"]}/h2_{label}',metrics,
                            src,truth,observations,valid,mask,decode,base,scale,cfg,rank,
                            preview=(frame==cfg['target_indices'][-1] and strength==max(cfg['strengths']) and fraction==min(cfg['observed_fractions']) and noise==max(cfg.get('noise_stds',[0.,.005])))))
                        if codec=='rae':
                            candidates=dict(no_edit=torch.zeros_like(truth),sparse_rgb=observations,
                                edge_aware_rgb=rgb_completion(observations.to(device),mask.to(device),src.to(device),cfg['rgb_fill_iterations']).cpu())
                            for method,pred in candidates.items():
                                record(cell(out,phase_identity,f'rgb/{row["id"]}/h2_{label}_{method}',
                                    dict(metrics,codec='rgb',method=method,latent_parameters=0),src,truth,valid,mask,
                                    prediction=pred,preview=False,rank=rank))
            del base
        progress(out,f'codec_r{rank}',dict(phase=codec,id=row['id'],cells=len(reports),
            DI_throughput=sum(r.get('evaluations',0) for r in reports)/max(1e-6,sum(r.get('elapsed',0) for r in reports)),
            throughput_unit='decoder frame evaluations/s/npu, completed optimization cells'))
    reports=sum(gather(reports,world),[])
    if rank==0:out.write_json('phase_summary/'+codec+'.json',dict(identity=phase_identity,cases=reports,
        completed=sum(r['status']=='completed' for r in reports),failed=sum(r['status']=='failed' for r in reports)))
    barrier()
    if any(r['status']=='failed' for r in reports):raise RuntimeError(codec+' cells failed; other codec phase can still run')


def observed_cell(out,identity,key,metadata,source,truth,observations,valid,mask,decode,base,scale,cfg,rank,preview):
    # cell() has separate optimization/evaluation arguments via this adapter.
    return cell(out,identity,key,metadata,source,truth,valid,mask,decode,base,scale,cfg,
                preview=preview,rank=rank,observations=observations)


def summarize(out):
    from collections import defaultdict
    import random
    manifest=out.json('manifest.json',True);bank=out.json('source_bank.json',True);cfg=manifest['config']
    records=[];errors=[];replication=[]
    for codec in ('rae','vae'):
        phase=out.json('phase_summary/'+codec+'.json')
        if phase is None:
            errors.append('Missing '+codec+' phase summary; using committed per-rank partial ledgers')
            for rank in range(manifest.get('world',24)):
                ledger=out.json(f'shards/{codec}_rank{rank:03d}.json')
                if ledger:
                    if ledger['manifest_identity']!=manifest['identity']:raise ValueError('Stale partial ledger')
                    records+=ledger['cases']
            continue
        records+=phase['cases']
        out.write_json('phase_summary/'+codec+'.json',phase)
    for row in bank['records']:
        if 'error' in row:
            errors.append('Source failed '+row['id']);continue
        name='source_previews/'+row['id']+'.json'
        try:replication.append(dict(receipt=name,roots=reconcile_case(out,name,manifest['identity'])))
        except Exception as exc:replication.append(dict(receipt=name,error=repr(exc)))
    for row in records:
        if row['status']!='completed':continue
        try:
            receipt=out.json(row['receipt'],True)
            replication.append(dict(receipt=row['receipt'],roots=reconcile_case(out,row['receipt'],receipt['contract_hash'])))
        except Exception as exc:replication.append(dict(receipt=row['receipt'],error=repr(exc)))
    fields=('effect_mse','effect_l1','observed_l1','heldout_l1','edit_l1','heldout_edit_l1','keep_leakage_l1','effect_gain')
    groups=defaultdict(list)
    for row in records:
        if row['status']=='completed':
            groups[tuple(row[k] for k in ('scenario','codec','method','strength','observation_fraction','noise_std'))].append(row)
    means=[]
    for key,rows in sorted(groups.items()):
        item=dict(zip(('scenario','codec','method','strength','observation_fraction','noise_std'),key),
            cells=len(rows),videos=len({r['id'] for r in rows}))
        for field in fields:
            values=[r[field] for r in rows if r.get(field) is not None]
            item[field]=None if not values else sum(values)/len(values)
        means.append(item)
    comparisons=[]
    def condition(r):return tuple(r[k] for k in ('id','frame','strength','observation_fraction','noise_std'))
    for scenario,bc,bm,mc,mm in (
        ('h1_dense_codec_effect','rae','naive_token','rae','calibrated'),
        ('h1_dense_codec_effect','rae','naive_gain_fit','rae','calibrated'),
        ('h1_dense_codec_effect','rae','fixed_response','rae','calibrated'),
        ('h2_analytic_sparse_effect','rgb','edge_aware_rgb','rae','calibrated'),
        ('h2_analytic_sparse_effect','vae','calibrated','rae','calibrated')):
        left={condition(r):r for r in records if r['status']=='completed' and (r['scenario'],r['codec'],r['method'])==(scenario,bc,bm)}
        right={condition(r):r for r in records if r['status']=='completed' and (r['scenario'],r['codec'],r['method'])==(scenario,mc,mm)}
        for strength in cfg['strengths']:
            for obs in ([1.] if scenario.startswith('h1') else cfg['observed_fractions']):
                for noise in ([0.] if scenario.startswith('h1') else cfg.get('noise_stds',[0.,.005])):
                    left_keys={k for k in left if k[2:]==(strength,obs,noise)}
                    right_keys={k for k in right if k[2:]==(strength,obs,noise)}
                    chosen=sorted(left_keys & right_keys)
                    delta={}
                    for field in fields:
                        per_clip=defaultdict(list)
                        for key in chosen:
                            if left[key].get(field) is not None and right[key].get(field) is not None:
                                per_clip[key[0]].append(right[key][field]-left[key][field])
                        values=[sum(v)/len(v) for v in per_clip.values()]
                        if not values:continue
                        rng=random.Random(42)
                        bootstrap=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(2000))
                        delta[field]=dict(mean=sum(values)/len(values),clips=len(values),descriptive_ci95=[bootstrap[49],bootstrap[1949]])
                    comparisons.append(dict(scenario=scenario,baseline=bc+'/'+bm,method=mc+'/'+mm,
                        strength=strength,observation_fraction=obs,noise_std=noise,paired_cells=len(chosen),
                        missing_baseline=len(right_keys-left_keys),missing_method=len(left_keys-right_keys),delta=delta))
    n=sum('error' not in r for r in bank['records']);pairs=n*len(cfg['target_indices'])*len(cfg['strengths'])
    h2=pairs*len(cfg['observed_fractions'])*len(cfg.get('noise_stds',[0.,.005]))
    expected=dict(rae=pairs*(6 if cfg['gradient_control'] else 5)+h2,vae=h2,rgb=3*h2)
    actual={c:sum(r['codec']==c for r in records) for c in expected}
    if expected!=actual:errors.append('Cell coverage differs: expected='+str(expected)+' actual='+str(actual))
    failed=[r for r in records if r['status']=='failed']
    degraded=[r for r in replication if 'error' in r or not all(t['ok'] for t in r.get('roots',[]))]
    optimized=[r for r in records if r['status']=='completed' and 'gradient_checked' in r]
    gradient_evidence=dict(optimized_cells=len(optimized),nonzero_data_gradients=sum(r['gradient_checked'] for r in optimized),
        zero_data_gradients=sum(not r['gradient_checked'] for r in optimized))
    report=dict(schema='edit-transport-probe-summary-v1',status='completed' if not errors and not failed and not degraded else 'incomplete',
        source_requested=bank['requested'],source_completed=n,expected=expected,actual=actual,failed=failed,errors=errors,
        replication_failed=degraded,groups=means,comparisons=comparisons,gradient_evidence=gradient_evidence,
        interpretation='Negative quality results are valid completed experiments. CI resamples source clips, not cells; one probe seed, no paper-level claim.',
        not_evaluated=['natural 3D/dynamic correspondences','DINO-RAE','text editing','native14B writeback','full video generation'])
    out.write_json('summary.json',report);out.write_json('media_sync.json',dict(status='synced' if not degraded else 'incomplete',checked=len(replication),failed=degraded))
    # Flat per-cell logs for filtering; tensor/optimizer payloads remain separate.
    from html import escape
    gallery=['<!doctype html><meta charset="utf-8"><title>Stage62 controlled decoder probe</title>',
        '<style>body{font:15px sans-serif;background:#171717;color:#eee;margin:24px}img{max-width:100%}pre{white-space:pre-wrap}a{color:#9cd}</style>',
        '<h1>Controlled decoder edit-effect probe</h1><p>Synthetic 2D transforms and analytic color edits. Not generated-video quality or 3D geometry.</p>',
        '<pre>'+escape(json.dumps({k:report[k] for k in ('status','expected','actual','errors','not_evaluated')},indent=2))+'</pre>']
    for row in records:
        if row['status']=='completed':
            receipt=out.json(row['receipt'],True)
            for name in receipt['files']:
                if name.endswith('.png'):gallery+=['<p>'+escape(str({k:row[k] for k in ('id','codec','method','scenario','frame','strength','observation_fraction','noise_std')}))+'</p>',
                    '<img loading="lazy" src="'+escape(name,quote=True)+'">']
    path=out.local/'report.html';path.write_text('\n'.join(gallery),encoding='utf8')
    # Idempotent finalization; don't append the same recovered cells repeatedly.
    metrics=out.local/'metrics.jsonl';metrics.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf8')
    out.publish(['report.html','metrics.jsonl'])
    print('[transport summary] '+json.dumps({k:report[k] for k in ('status','source_completed','actual','errors')}),flush=True)
    if report['status']!='completed':raise RuntimeError('Incomplete probe coverage/publication; see summary.json and rerun original command')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('prepare','rae','vae','summarize'),required=True)
    for flag in ('config','output'):p.add_argument('--'+flag,required=True)
    for flag in ('data','rae','encoder','model'):p.add_argument('--'+flag)
    p.add_argument('--root',action='append',default=[]);p.add_argument('--data-root',action='append',default=[])
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());out=ArtifactStore(a.output,a.root)
    if cfg['decoder_precision']!='fp32' or cfg['effect']!='rgb_difference':
        raise ValueError('Stage62 currently implements FP32 decoder RGB differences only')
    if cfg['frames']!=9 or not cfg['target_indices'] or any(i<=0 or i>=cfg['frames'] for i in cfg['target_indices']):
        raise ValueError('Use 9 source states and target indices in 1..8')
    if a.phase=='summarize':
        if int(os.environ.get('NODE_RANK','0'))==0:summarize(out)
        return
    device,rank,world=distributed()
    try:
        if a.phase=='prepare':prepare(a,cfg,ArtifactStore(a.data,a.data_root),out,device,rank,world)
        else:run_codec(a,cfg,out,device,rank,world)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
