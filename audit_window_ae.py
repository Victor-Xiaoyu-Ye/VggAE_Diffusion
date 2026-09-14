#!/usr/bin/env python3
"""Re-encode identical RAW clips under both historical/current codec semantics."""
import argparse
import json
import signal
import time
import faulthandler
from pathlib import Path
import torch
from utils.device import get_device, get_device_name, configure_backend_compatibility
from utils.window_training import load_artifact
from utils.window_codec import configure_codec, runtime_contract, reconstruction_gate
from utils.file_signature import sampled_file_signature
from utils.flow_run_status import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','stats','r7_ckpt','encoder_ckpt','output_dir'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--selected_norm',choices=('legacy','framewise'),default='legacy')
    p.add_argument('--clips',type=int,default=16)
    p.add_argument('--previews',type=int,default=4)
    p.add_argument('--min_psnr',type=float,default=23.5)
    p.add_argument('--min_gain',type=float,default=2.)
    p.add_argument('--report_only',action='store_true',help='Complete diagnosis without approving training quality')
    p.add_argument('--baseline_json',help='Select the exact video IDs from a previous AE baseline')
    a=p.parse_args()
    baseline = json.loads(Path(a.baseline_json).read_text()) if a.baseline_json else None
    wanted = {r['video_id']: r for r in baseline['clips']} if baseline else None
    if wanted is not None and (len(wanted)!=a.clips or len(baseline['clips'])!=a.clips):
        p.error('baseline must contain exactly --clips unique video IDs')
    if a.clips < 1 or a.previews < 0: p.error('invalid clip count')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'audit.json').exists(): raise FileExistsError('select a fresh audit output directory')
    result=dict(schema='r7-window-ae-audit-v1',status='running',args=vars(a),clips=[])
    def interrupted(signum, frame):raise RuntimeError(f'AE audit interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def record(**kw):
        result.update(kw);atomic_json(out/'audit.json',result)
        print('[AE audit]',json.dumps({k:v for k,v in result.items() if k in
              ('status','phase','completed_clips','elapsed_seconds','DI_throughput','error','mean_psnr')}),flush=True)
    try:
        faulthandler.enable()
        faulthandler.dump_traceback_later(300, repeat=True)
        record(phase='loading')
        if get_device_name()=='cpu': raise RuntimeError('real AE audit requires an accelerator')
        device=get_device(0);configure_backend_compatibility()
        from data.latent_shard_dataset import LatentShardDataset
        from streamvggt.models.streamvggt import StreamVGGT
        from utils.encoder_loader import load_encoder_checkpoint
        from utils.r7_representation import load_r7_modules,validate_contract,encode_dual
        from utils.video_preview import save_video_preview
        artifact=load_artifact(a.r7_ckpt);stats=load_artifact(a.stats)
        validate_contract(stats['representation'],artifact['representation_contract'])
        signatures=stats['representation']['signatures']
        if baseline and (baseline['ae_signature'] != signatures or baseline['temporal_norm'] != a.selected_norm):
            raise ValueError('baseline representation mismatch')
        for key,path in [('r7',a.r7_ckpt),('streamvggt',a.encoder_ckpt)]:
            if sampled_file_signature(path)!=signatures[key]: raise ValueError(key+' signature mismatch')
        config,compressor,texture,codec,decoder,_=load_r7_modules(artifact)
        result['ae_step']=artifact.get('global_step',artifact.get('step'))
        result['signatures']=signatures
        result['runtime']=runtime_contract(a.selected_norm)
        del artifact
        encoder=StreamVGGT(img_size=config.target_size,patch_size=14,embed_dim=1024)
        load_encoder_checkpoint(encoder,a.encoder_ckpt)
        encoder.to(device=device,dtype=torch.float16).eval().requires_grad_(False)
        for m in (compressor,texture,codec,decoder):m.to(device).eval().requires_grad_(False)
        data=LatentShardDataset(a.manifest,0,42,False,0,1)
        started=time.monotonic(); seen=set()
        with torch.no_grad():
            for sample in data:
                if len(result['clips'])>=a.clips:break
                if wanted is not None and sample['video_id'] not in wanted:continue
                if sample['video_id'] in seen:raise ValueError('duplicate selected video ID')
                seen.add(sample['video_id']); index=len(result['clips'])
                if 'rgb' not in sample: raise ValueError('audit needs saved RAW clip, not decoded AE RGB')
                if sample.get('video_id')!=sample.get('requested_video_id',sample.get('video_id')):
                    raise ValueError('replacement video in audit cache')
                frames=sample['rgb'][None].to(device).float()/255
                raw=frames.permute(0,1,3,4,2)
                geo,tex=encode_dual(encoder,compressor,texture,frames,torch.float16)
                g0,t0=encode_dual(encoder,compressor,texture,frames[:,:1],torch.float16)
                row=dict(video_id=sample['video_id'],window_id=sample.get('window_id'),
                         frame_indices=sample.get('frame_indices')); videos={'raw':raw[0]}
                if wanted is not None:
                    row['previous_cache_psnr']=wanted[sample['video_id']]['ae_psnr_full_vs_raw']
                for mode in ('legacy','framewise'):
                    configure_codec(codec,mode)
                    full=codec.encode(geo,tex); single=codec.encode(g0,t0)
                    row[mode+'/anchor_relative_l2']=float((full[:,:1]-single).norm()/full[:,:1].norm().clamp_min(1e-8))
                    z=torch.cat((single,full[:,1:]),1)
                    gr,tr=codec.decode(z);rgb=decoder(gr,tr).float()
                    if not torch.isfinite(rgb).all():raise RuntimeError('nonfinite audit RGB')
                    rgb=rgb.clamp(0,1);error=(rgb-raw).square()
                    row[mode+'/psnr']=float(-10*error.mean().clamp_min(1e-12).log10())
                    row[mode+'/l1']=float((rgb-raw).abs().mean())
                    row[mode+'/frame_psnr']=(-10*error.mean((0,2,3,4)).clamp_min(1e-12).log10()).cpu().tolist()
                    old=sample['target'].to(device).float().reshape_as(z[:,1:])
                    row[mode+'/relative_l2_vs_old_cache']=float((z[:,1:]-old).norm()/old.norm().clamp_min(1e-8))
                    old_anchor=sample['cond'].to(device).float().reshape_as(single)
                    row[mode+'/anchor_relative_l2_vs_cache']=float((single-old_anchor).norm()/old_anchor.norm().clamp_min(1e-8))
                    cached=torch.cat((old_anchor,old),1)
                    cg,ct=codec.decode(cached); cached_rgb=decoder(cg,ct).float().clamp(0,1)
                    if not torch.isfinite(cached_rgb).all():raise RuntimeError('nonfinite cache replay')
                    ce=(cached_rgb-raw).square()
                    row[mode+'/cache_psnr']=float(-10*ce.mean().clamp_min(1e-12).log10())
                    row[mode+'/cache_frame_psnr']=(-10*ce.mean((0,2,3,4)).clamp_min(1e-12).log10()).cpu().tolist()
                    row[mode+'/fresh_vs_cache_rgb_l1']=float((rgb-cached_rgb).abs().mean())
                    if mode==a.selected_norm:videos['cached']=cached_rgb[0]
                    videos[mode]=rgb[0]
                result['clips'].append(row)
                elapsed=time.monotonic()-started
                record(phase='replaying',completed_clips=len(result['clips']),elapsed_seconds=elapsed,
                       DI_throughput=len(result['clips'])*frames.shape[1]/elapsed,
                       throughput_units='source frames/s, includes both norm modes and cache replay')
                if index<a.previews:save_video_preview(str(out/'samples'),f'clip{index}',videos,metadata=row)
        if len(result['clips'])!=a.clips:raise ValueError('insufficient RAW audit clips')
        means={m:sum(x[m+'/psnr'] for x in result['clips'])/a.clips for m in ('legacy','framewise')}
        _,passed=reconstruction_gate([x[a.selected_norm+'/psnr'] for x in result['clips']],a.min_psnr)
        if a.selected_norm=='legacy':passed=passed and means['legacy']-means['framewise']>=a.min_gain
        record(phase='finished',status='completed' if a.report_only else ('passed' if passed else 'failed'),
               quality_gate_passed=passed,training_authorized=False,mean_psnr=means,
               selected_psnr=means[a.selected_norm],gain_legacy_vs_framewise=means['legacy']-means['framewise'])
        if not passed and not a.report_only:raise RuntimeError('AE reconstruction audit gate failed; training must not start')
    except BaseException as e:
        record(status='failed',error=repr(e));raise
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
