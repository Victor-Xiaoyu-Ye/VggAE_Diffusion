#!/usr/bin/env python3
"""Replay existing R7 Wan x0 checkpoints; no training or production promotion."""
from __future__ import annotations
import argparse
import contextlib
import json
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.latent_shard_dataset import LatentShardDataset
from models.wan_compact_adapter import WanCompactAdapter
from train_causal_wan_video_diffusion import TextEmbeddingBank, wan_config_snapshot
from train_causal_video_diffusion import (checked_decode_chunk_size, exact_equal,
    load_robust_decoder, load_torch_artifact, validate_r7_artifact_representation, validate_stats)
from utils.device import configure_backend_compatibility, get_device, get_device_name, resolve_dtype
from utils.latent_generation_metrics import (normalize, inverse, to_device,
    generation_metrics, denoising_baselines)
from utils.r7_representation import load_r7_modules
from utils.training import append_metrics
from utils.video_preview import save_video_preview


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'wan_ckpt_dir', 'manifest', 'text_embedding_dir', 'output_dir'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--r7_ckpt', default='')
    p.add_argument('--eval_stats', default='', help='defaults to stats.pt beside evaluation manifest; representation must match')
    p.add_argument('--decoder_ckpt', default='')
    p.add_argument('--weights', default='model,ema')
    p.add_argument('--times', default='0.1,0.3,0.5,0.7,0.9,0.99')
    p.add_argument('--cfg_scales', default='1,3')
    p.add_argument('--sample_steps', default='30,60')
    p.add_argument('--sampling_grids', default='trained')
    p.add_argument('--seeds', default='42')
    p.add_argument('--eval_clips', type=int, default=4)
    p.add_argument('--preview_clips', type=int, default=1)
    p.add_argument('--oracle_starts', default='', help='optional data-time starts; NOT free generation')
    p.add_argument('--condition_ablations', action='store_true')
    p.add_argument('--dtype', choices=('fp32','fp16','bf16'), default='bf16')
    p.add_argument('--allow_missing_text', action='store_true')
    return p.parse_args(argv)


def csv_values(value, cast):
    result = [cast(x.strip()) for x in value.split(',') if x.strip()]
    if len(set(result)) != len(result):
        raise ValueError('duplicate matrix entries')
    return result


@torch.no_grad()
def integrate(model, anchor, noise, text, empty, steps, alpha, cfg, dtype, start=0., data=None):
    if not 0 <= start < 1 or (start and data is None):
        raise ValueError('oracle start requires known target and 0<=start<1')
    z = noise.float().clone() if not start else (1-start)*noise.float()+start*data.float()
    u = torch.linspace(0, 1, steps+1, device=noise.device)
    grid = start+(1-start)*(alpha*u/(1+(alpha-1)*u))
    for left,right in zip(grid[:-1],grid[1:]):
        t = torch.full((z.shape[0],), float(left), device=z.device)
        with torch.autocast(device_type=z.device.type, dtype=dtype, enabled=dtype!=torch.float32):
            pred = model(z,t,cond=anchor,text_emb=text).float()
            if cfg != 1:
                null = model(z,t,cond=anchor,text_emb=empty).float()
                pred = null+cfg*(pred-null)
        z = z+(right-left)*(pred-z)/(1-left).clamp_min(1e-6)
    return z


def main(argv=None):
    args=parse_args(argv)
    weights=csv_values(args.weights,str); times=csv_values(args.times,float)
    cfgs=csv_values(args.cfg_scales,float); steps=csv_values(args.sample_steps,int)
    grids=csv_values(args.sampling_grids,str); seeds=csv_values(args.seeds,int)
    starts=csv_values(args.oracle_starts,float)
    if (not all((weights,times,cfgs,steps,grids,seeds)) or set(weights)-{'model','ema'}
            or set(grids)-{'trained','uniform'} or min(steps)<1 or min(cfgs)<0
            or any(not 0<=t<=1 for t in times) or any(not 0<s<1 for s in starts)
            or args.eval_clips<1 or args.preview_clips<0):
        raise ValueError('invalid replay matrix')
    out=Path(args.output_dir)
    if any((out/n).exists() for n in ('metrics.jsonl','summary.json','replay.json')):
        raise ValueError('replay output already exists; choose a fresh namespace')
    if args.decoder_ckpt and not args.r7_ckpt:
        raise ValueError('decoder override requires --r7_ckpt')
    backend=get_device_name();configure_backend_compatibility(backend)
    device=get_device(0);dtype=resolve_dtype(args.dtype)
    ckpt=load_torch_artifact(args.checkpoint)
    architecture=ckpt.get('architecture') or {}; config=ckpt.get('args') or {}
    objective=ckpt.get('objective') or {}
    if architecture.get('class')!='WanCompactAdapter':
        raise ValueError('only explicit WanCompactAdapter checkpoint contracts supported')
    # Old R7 Wan payloads may omit objective schema but name x0 in prediction.
    prediction=objective.get('prediction', architecture.get('prediction', ckpt.get('prediction')))
    schema=str(ckpt.get('schema',''))
    if prediction not in (None,'x0') or (prediction is None and not schema.startswith('r7-wan')):
        raise ValueError('cannot establish x0 checkpoint contract')
    if int(config.get('latent_dim',-1))!=192 or int(config.get('context_chunks',-1))!=1:
        raise ValueError('unsupported latent/anchor contract')
    future=int(config.get('future_chunks',-1)); grid=int(config.get('latent_grid',18))
    if future not in (4,8):
        raise ValueError('expected four/eight future chunks')
    checkargs=types.SimpleNamespace(context_chunks=1,future_chunks=future,latent_dim=192,
                                   temporal_factor=int(config.get('temporal_factor',2)))
    stats=ckpt.get('normalization');signature=validate_stats(stats,'checkpoint',checkargs)
    if ckpt.get('normalization_signature')!=signature:
        raise ValueError('checkpoint normalization signature mismatch')
    if config.get('normalization_mode','zscore')!='zscore':
        raise ValueError('this diagnostic requires the measured zscore contract')
    for w in weights:
        if w not in ckpt:
            raise ValueError(f'checkpoint has no {w} weights; no fallback permitted')
    from utils.moxing_io import is_remote_path
    eval_stats_path = args.eval_stats or (args.manifest.rsplit('/', 1)[0]+'/stats.pt'
        if is_remote_path(args.manifest) else str(Path(args.manifest).with_name('stats.pt')))
    eval_stats = load_torch_artifact(eval_stats_path)
    validate_stats(eval_stats, 'evaluation stats', checkargs)
    if not exact_equal(eval_stats['representation'], stats['representation']):
        raise ValueError('evaluation cache representation does not match checkpoint')
    # Evaluation moments are only a contract check; always normalize with train stats.
    st={k:to_device(stats[k],device) for k in ('cond','target')}
    model=WanCompactAdapter(args.wan_ckpt_dir,latent_dim=192,latent_grid=grid,seq_len=future,
        full_finetune=True,anchor_frame=True,
        anchor_memory=bool(architecture.get('anchor_memory',False)),
        reverse_flow_time=bool(architecture.get('reverse_flow_time',False))).to(device).eval()
    if ckpt.get('wan_config') and ckpt['wan_config']!=wan_config_snapshot(model):
        raise ValueError('Wan checkpoint directory configuration mismatch')
    bank=TextEmbeddingBank(args.text_embedding_dir)
    samples=[]
    for sample in LatentShardDataset(args.manifest,shuffle_buffer=1,seed=0,repeat=False,rank=0,world_size=1):
        samples.append(sample)
        if len(samples)>=args.eval_clips: break
    if len(samples)!=args.eval_clips:
        raise ValueError('evaluation manifest has too few clips')
    if args.condition_ablations and len(samples)<2:
        raise ValueError('condition swapping requires at least two clips')
    for s in samples:
        if tuple(s['cond'].shape)!=(1,grid*grid,192) or tuple(s['target'].shape)!=(future,grid*grid,192):
            raise ValueError('cache member layout mismatch')
        if str(s.get('video_id','')) not in bank.embeddings and not args.allow_missing_text:
            raise ValueError('missing caption; explicitly allow only for an empty-text diagnostic')
    tokenizer=decoder=None
    if args.r7_ckpt:
        artifact=validate_r7_artifact_representation(stats['representation'],args.r7_ckpt)
        rep,compressor,texture,tokenizer,decoder,_=load_r7_modules(artifact)
        del artifact,compressor,texture
        checked_decode_chunk_size(rep,decoder,0)
        tokenizer=tokenizer.to(device).eval();decoder=decoder.to(device).eval()
        if args.decoder_ckpt:
            load_robust_decoder(args.decoder_ckpt,stats['representation'],decoder)
    def decode(c,y):
        if decoder is None: return None
        z=torch.cat((c,y),1).reshape(c.shape[0],1+future,grid,grid,192)
        with torch.autocast(device_type=backend,dtype=dtype,enabled=dtype!=torch.float32):
            geo,tex=tokenizer.decode(z);rgb=decoder(geo,tex)
        return rgb.float().clamp(0,1)
    # Drop training-only CPU optimizer tensors before running the replay matrix.
    for key in ('optimizer', 'scheduler', 'scaler', 'rng_states'):
        ckpt.pop(key, None)
    out.mkdir(parents=True,exist_ok=True)
    info={'checkpoint':args.checkpoint,'checkpoint_step':ckpt.get('step',ckpt.get('global_step')),
          'schema':'r7-wan-denoising-replay-v1','args':vars(args),'architecture':architecture,
          'objective':objective,'normalization_signature':signature,
            'ema_metadata':ckpt.get('ema_metadata'), 'weights_present':weights,
          'resolved_dtype':str(dtype),
          'limits':['AE_TARGET is not raw RGB','oracle starts are not generation',
                    'centered cosine is not physical motion','preserve old time convention']}
    (out/'replay.json').write_text(json.dumps(info,indent=2,default=str))
    rows=[]
    def record(row):
        append_metrics(str(out/'metrics.jsonl'),row);rows.append(row)
    with torch.inference_mode():
        for w in weights:
            model.load_state_dict(ckpt[w],strict=True)
            for index,s in enumerate(samples):
                c=s['cond'][None].to(device).float();y=s['target'][None].to(device).float()
                cn,yn=normalize(c,st['cond']),normalize(y,st['target'])
                vid=str(s.get('video_id','')); text=bank.batch([vid],device);empty=bank.empty_batch(1,device)
                ref=decode(c,y)
                base=dict(weights=w,video_id=vid,sample_index=index)
                noise=torch.randn(yn.shape,generator=torch.Generator(device='cpu').manual_seed(seeds[0]+1009*index)).to(device)
                for tv in times:
                    x=(1-tv)*noise+tv*yn;t=torch.full((1,),tv,device=device)
                    with torch.autocast(device_type=backend,dtype=dtype,enabled=dtype!=torch.float32):
                        pred=model(x,t,cond=cn,text_emb=text).float()
                    row=dict(base,kind='denoising',t=tv,x0_mse=F.mse_loss(pred,yn).item(),
                             **denoising_baselines(x,yn,tv))
                    if ref is not None and index<args.preview_clips:
                        rgb=decode(c,inverse(pred,st['target']))
                        save_video_preview(str(out/'samples'),f'{w}_clip{index}_t{tv:.3f}',
                            {'AE_TARGET':ref[0],'DENOISED':rgb[0]},metadata=row,save_frames=False,save_mp4=False)
                    record(row)
                    if args.condition_ablations:
                        other=samples[(index+1)%len(samples)]
                        oc=normalize(other['cond'][None].to(device).float(),st['cond'])
                        ot=bank.batch([str(other.get('video_id',''))],device)
                        for label,ac,at in [('swapped_anchor',oc,text),('swapped_text',cn,ot)]:
                            with torch.autocast(device_type=backend,dtype=dtype,enabled=dtype!=torch.float32):
                                ab=model(x,t,cond=ac,text_emb=at).float()
                            record(dict(base,kind='denoising_ablation',ablation=label,t=tv,
                                        x0_mse=F.mse_loss(ab,yn).item(),prediction_change=F.mse_loss(ab,pred).item()))
                for seed in seeds:
                    noise=torch.randn(yn.shape,generator=torch.Generator(device='cpu').manual_seed(seed+1009*index)).to(device)
                    for gridname in grids:
                        alpha=float(config.get('time_shift_alpha',1.)) if gridname=='trained' else 1.
                        if alpha<=0: raise ValueError('invalid saved time shift')
                        for count in steps:
                            for cfg in cfgs:
                                gen=integrate(model,cn,noise,text,empty,count,alpha,cfg,dtype)
                                raw=inverse(gen,st['target'])
                                row=dict(base,kind='free_sample',seed=seed,sampling_grid=gridname,
                                    sample_steps=count,cfg_scale=cfg,
                                    **generation_metrics(raw,y,c,st['target'],st['cond']))
                                if ref is not None:
                                    rgb=decode(c,raw)
                                    row['rgb_psnr_vs_ae']=float((-10*torch.log10((rgb[:,1:]-ref[:,1:]).square().mean((2,3,4)).clamp_min(1e-12))).mean())
                                    if index<args.preview_clips:
                                        save_video_preview(str(out/'samples'),f'{w}_clip{index}_seed{seed}_{gridname}_n{count}_cfg{cfg}',
                                            {'AE_TARGET':ref[0],'GENERATED':rgb[0]},metadata=row,save_frames=True,save_mp4=True)
                                record(row)
                    for start in starts:
                        gen=integrate(model,cn,noise,text,empty,steps[0],float(config.get('time_shift_alpha',1.)),
                                      cfgs[0],dtype,start,yn)
                        record(dict(base,kind='oracle_start',start=start,seed=seed,
                            sample_steps=steps[0],cfg_scale=cfgs[0],
                            **generation_metrics(inverse(gen,st['target']),y,c,st['target'],st['cond'])))
    groups={}
    for r in rows:
        key=tuple((k,r[k]) for k in ('kind','weights','t','ablation','seed','sampling_grid','sample_steps','cfg_scale','start') if k in r)
        groups.setdefault(key,[]).append(r)
    summary=[]
    for key,rs in groups.items():
        numeric={k:float(np.mean([r[k] for r in rs])) for k in set.intersection(*(set(r) for r in rs))
                 if all(isinstance(r[k],(int,float)) for r in rs) and k not in dict(key) and k!='sample_index'}
        summary.append(dict(key,clips=len(rs),**numeric))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(f'Replay finished: {out}; no training or gate promotion performed')


if __name__=='__main__':
    main()
