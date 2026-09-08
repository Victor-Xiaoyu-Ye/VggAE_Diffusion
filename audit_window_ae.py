#!/usr/bin/env python3
"""Re-encode identical RAW clips under both historical/current codec semantics."""
import argparse
import json
import signal
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
    a=p.parse_args()
    if a.clips < 1 or a.previews < 0: p.error('invalid clip count')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'audit.json').exists(): raise FileExistsError('select a fresh audit output directory')
    result=dict(schema='r7-window-ae-audit-v1',status='running',args=vars(a),clips=[])
    def interrupted(signum, frame):raise RuntimeError(f'AE audit interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def record(**kw):
        result.update(kw);atomic_json(out/'audit.json',result)
        print('[AE audit]',json.dumps({k:v for k,v in result.items() if k not in ('clips','args')}),flush=True)
    try:
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
        with torch.no_grad():
            for index,sample in enumerate(data):
                if index>=a.clips:break
                if 'rgb' not in sample: raise ValueError('audit needs saved RAW clip, not decoded AE RGB')
                if sample.get('video_id')!=sample.get('requested_video_id',sample.get('video_id')):
                    raise ValueError('replacement video in audit cache')
                frames=sample['rgb'][None].to(device).float()/255
                raw=frames.permute(0,1,3,4,2)
                geo,tex=encode_dual(encoder,compressor,texture,frames,torch.float16)
                g0,t0=encode_dual(encoder,compressor,texture,frames[:,:1],torch.float16)
                row=dict(video_id=sample['video_id']); videos={'raw':raw[0]}
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
                    videos[mode]=rgb[0]
                result['clips'].append(row)
                if index<a.previews:save_video_preview(str(out/'samples'),f'clip{index}',videos,metadata=row)
                record(phase='replaying',completed_clips=len(result['clips']))
        if len(result['clips'])!=a.clips:raise ValueError('insufficient RAW audit clips')
        means={m:sum(x[m+'/psnr'] for x in result['clips'])/a.clips for m in ('legacy','framewise')}
        _,passed=reconstruction_gate([x[a.selected_norm+'/psnr'] for x in result['clips']],a.min_psnr)
        if a.selected_norm=='legacy':passed=passed and means['legacy']-means['framewise']>=a.min_gain
        record(phase='finished',status='passed' if passed else 'failed',mean_psnr=means,
               selected_psnr=means[a.selected_norm],gain_legacy_vs_framewise=means['legacy']-means['framewise'])
        if not passed:raise RuntimeError('AE reconstruction audit gate failed; training must not start')
    except BaseException as e:
        record(status='failed',error=repr(e));raise


if __name__=='__main__':main()
