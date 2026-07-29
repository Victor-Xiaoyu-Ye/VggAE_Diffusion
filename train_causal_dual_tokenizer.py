#!/usr/bin/env python3
"""R7 causal spatiotemporal tokenizer training over frozen dual latents."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.loader_utils import multiprocessing_loader_kwargs
from data.token_utils import strip_special_tokens
from data.video_dataset import SpatialVidDataset, collate_fn
from models.causal_dual_tokenizer import CausalDualTokenizerCore
from models.dpt_latent_decoder import CompactCompressor
from models.dual_stream_decoder import DualStreamDecoder
from models.texture_encoder import TextureEncoder
from streamvggt.models.streamvggt import StreamVGGT
from utils.device import (configure_backend_compatibility, get_device,
                          get_device_name, manual_seed_all, resolve_dtype)
from utils.distributed import is_main_process, setup_ddp
from utils.encoder_loader import load_encoder_checkpoint
from utils.training import (append_metrics, atomic_torch_save, build_scheduler,
                            capture_rng_state, restore_rng_state)


def parse_args():
    p = argparse.ArgumentParser(description='R7 causal dual tokenizer')
    p.add_argument('--csv', required=True)
    p.add_argument('--video_root', required=True)
    p.add_argument('--eval_csv', required=True)
    p.add_argument('--encoder_ckpt', required=True)
    p.add_argument('--dual_ae_ckpt', required=True)
    p.add_argument('--temporal_factor', type=int, choices=[2, 4], default=2)
    p.add_argument('--geo_latent_dim', type=int, default=96)
    p.add_argument('--tex_latent_dim', type=int, default=96)
    p.add_argument('--temporal_depth', type=int, default=3)
    p.add_argument('--phase', choices=['codec', 'joint'], default='codec')
    p.add_argument('--seq_len', type=int, default=9)
    p.add_argument('--target_size', type=int, default=518)
    p.add_argument('--clip_duration_seconds', type=float, default=1.0)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--accum_steps', type=int, default=2)
    p.add_argument('--max_steps', type=int, default=3000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--pretrained_lr', type=float, default=5e-5)
    p.add_argument('--wd', type=float, default=1e-2)
    p.add_argument('--warmup_steps', type=int, default=200)
    p.add_argument('--lambda_l1', type=float, default=1.0)
    p.add_argument('--lambda_latent', type=float, default=0.5)
    p.add_argument('--lambda_temporal', type=float, default=0.1)
    p.add_argument('--lambda_accel', type=float, default=0.05)
    p.add_argument('--lambda_geo_motion', type=float, default=0.1)
    p.add_argument('--lambda_comp_reg', type=float, default=0.01)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--decode_retries', type=int, default=8)
    p.add_argument('--eval_clips', type=int, default=16)
    p.add_argument('--frames_chunk_size', type=int, default=3)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--resume', default='')
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--save_every', type=int, default=500)
    p.add_argument('--dtype', default='bf16')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


class R7TrainingCore(nn.Module):
    def __init__(self, tokenizer, decoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.decoder = decoder

    def forward(self, z_geo, z_tex, frames_chunk_size=None):
        geo_rec, tex_rec, z = self.tokenizer(z_geo, z_tex)
        rgb = self.decoder(
            geo_rec, tex_rec, frames_chunk_size=frames_chunk_size)
        return rgb, geo_rec, tex_rec, z


def temporal_loss(pred, target):
    return F.l1_loss(pred[:, 1:] - pred[:, :-1],
                     target[:, 1:] - target[:, :-1])


def acceleration_loss(pred, target):
    pd = pred[:, 1:] - pred[:, :-1]
    td = target[:, 1:] - target[:, :-1]
    return F.l1_loss(pd[:, 1:] - pd[:, :-1], td[:, 1:] - td[:, :-1])


def latent_regularization(z):
    flat = z.float().reshape(-1, z.shape[-1])
    return flat.mean(0).square().mean() + (flat.std(0, unbiased=False) - 1).square().mean()


def main():
    args = parse_args()
    if args.phase == 'joint':
        raise NotImplementedError(
            'joint phase requires compressor/texture encoder inside the DDP '
            'forward; run phase=codec until the joint wrapper is implemented')
    if (args.seq_len - 1) % args.temporal_factor:
        raise ValueError('seq_len must equal 1 + temporal_factor*k')
    use_ddp, rank, local_rank, world_size = setup_ddp()
    main_process = is_main_process()
    device = get_device(local_rank)
    device_type = get_device_name()
    dtype = resolve_dtype(args.dtype)
    configure_backend_compatibility(device_type)
    manual_seed_all(args.seed + rank)
    if main_process:
        os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)

    source = torch.load(args.dual_ae_ckpt, map_location='cpu', weights_only=False)
    ck = source.get('args', {})
    grid = int(ck.get('latent_grid', 18)); geo_dim = int(ck.get('geo_dim', 256)); tex_dim = int(ck.get('tex_dim', 256))
    encoder = StreamVGGT(img_size=args.target_size, patch_size=14, embed_dim=1024)
    load_encoder_checkpoint(encoder, args.encoder_ckpt, verbose=main_process)
    encoder = encoder.to(device=device, dtype=dtype).eval()
    compressor = CompactCompressor(levels=list(ck.get('levels', [4,11,17,23])), token_dim=int(ck.get('token_dim',2048)), cdim=geo_dim, latent_grid=grid, input_grid=args.target_size//14)
    tex_encoder = TextureEncoder(out_dim=tex_dim, out_grid=grid, base_ch=int(ck.get('tex_base_ch',64)), img_size=args.target_size, pack_mode=ck.get('tex_pack','avgpool'))
    decoder = DualStreamDecoder(geo_dim=geo_dim, tex_dim=tex_dim, base_dim=int(ck.get('decoder_base_dim',384)), img_size=args.target_size, latent_grid=grid, num_temporal_blocks=0, use_checkpoint=True)
    state = source['model']
    for module,prefix in ((compressor,'compressor.'),(tex_encoder,'tex_encoder.'),(decoder,'decoder.')):
        sub={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        module.load_state_dict(sub, strict=False)
    compressor=compressor.to(device); tex_encoder=tex_encoder.to(device); decoder=decoder.to(device)
    tokenizer=CausalDualTokenizerCore(geo_dim,tex_dim,args.geo_latent_dim,args.tex_latent_dim,args.temporal_factor,args.temporal_depth).to(device)
    core=R7TrainingCore(tokenizer,decoder).to(device)
    for p in encoder.parameters(): p.requires_grad_(False)
    for module in (compressor,tex_encoder):
        trainable=args.phase=='joint'; module.train(trainable)
        for p in module.parameters(): p.requires_grad_(trainable)
    if args.phase=='codec':
        for p in decoder.parameters(): p.requires_grad_(False)
        # Keep training mode so the frozen spatial decoder checkpoints its
        # activations while gradients still flow back into the tokenizer.
        decoder.train()
    else:
        for p in decoder.parameters(): p.requires_grad_(True)
        decoder.train()
    groups=[{'params':[p for p in tokenizer.parameters() if p.requires_grad],'lr':args.lr,'weight_decay':args.wd}]
    pretrained=[p for p in decoder.parameters() if p.requires_grad]
    if args.phase == 'joint':
        pretrained += [
            p for module in (compressor, tex_encoder)
            for p in module.parameters() if p.requires_grad]
    if pretrained: groups.append({'params':pretrained,'lr':args.pretrained_lr,'weight_decay':args.wd})
    optimizer=torch.optim.AdamW(groups,betas=(0.9,0.95),eps=1e-8)
    scheduler=build_scheduler(optimizer,args.warmup_steps,args.max_steps)
    if use_ddp: model=nn.parallel.DistributedDataParallel(core,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=args.phase=='codec')
    else: model=core

    dataset=SpatialVidDataset(args.csv,args.video_root,seq_len=args.seq_len,target_size=args.target_size,num_frames_per_video=args.seq_len,clip_duration_seconds=args.clip_duration_seconds,decode_retries=args.decode_retries)
    sampler=torch.utils.data.distributed.DistributedSampler(dataset) if use_ddp else None
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=sampler is None,sampler=sampler,num_workers=args.num_workers,collate_fn=collate_fn,drop_last=True,pin_memory=device_type=='cuda',**multiprocessing_loader_kwargs(args.num_workers))
    eval_loader=None
    if main_process:
        ed=SpatialVidDataset(args.eval_csv,args.video_root,seq_len=args.seq_len,target_size=args.target_size,max_videos=args.eval_clips,num_frames_per_video=args.seq_len,temporal_jitter=False,clip_duration_seconds=args.clip_duration_seconds,decode_retries=args.decode_retries)
        eval_loader=DataLoader(ed,batch_size=1,num_workers=0,collate_fn=collate_fn)

    global_step=0
    if args.resume:
        r=torch.load(args.resume,map_location='cpu',weights_only=False); core.load_state_dict(r['core'], strict=False); optimizer.load_state_dict(r['optimizer']); scheduler.load_state_dict(r['scheduler']); global_step=r['global_step']; restore_rng_state(r.get('rng'))
    writer=SummaryWriter(os.path.join(args.output_dir,'tb')) if main_process else None
    metrics_path=os.path.join(args.output_dir,'metrics.jsonl')

    def encode(frames):
        with torch.no_grad():
            tokens,psi=encoder(frames.to(dtype))
        stripped=strip_special_tokens(tokens,psi)
        grad_enabled = args.phase == 'joint' and torch.is_grad_enabled()
        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with context:
            geo=compressor([token.float() for token in stripped])
            geo=geo.permute(0,1,3,4,2).contiguous().float()
            tex=tex_encoder(frames.float()).float()
        return geo,tex

    def save(step):
        merged={}
        modules=((compressor,'compressor.'),(tex_encoder,'tex_encoder.'),(core.tokenizer,'tokenizer.'),(core.decoder,'decoder.'))
        for module,prefix in modules:
            for k,v in module.state_dict().items(): merged[prefix+k]=v.detach().cpu()
        payload={'model':merged,'core':core.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'global_step':step,'args':{**ck,**vars(args),'has_temporal_codec':True,'latent_dim':core.tokenizer.latent_dim},'rng':capture_rng_state()}
        atomic_torch_save(payload,os.path.join(args.output_dir,f'checkpoint_step{step:07d}.pt')); atomic_torch_save(payload,os.path.join(args.output_dir,'checkpoint_latest.pt'))

    @torch.no_grad()
    def evaluate(step):
        core.eval(); psnr=[]; boundary=[]
        for batch in eval_loader:
            frames=batch['frames'].to(device)
            if frames.shape[1] != args.seq_len:
                raise RuntimeError(
                    f'dataset returned {frames.shape[1]} frames, expected '
                    f'{args.seq_len}; set num_frames_per_video=seq_len')
            geo,tex=encode(frames); pred,_,_,_=core(geo,tex,args.frames_chunk_size); target=frames.float().permute(0,1,3,4,2)
            mse=F.mse_loss(pred,target).item(); psnr.append(-10*np.log10(max(mse,1e-10)))
            pd=(pred[:,1:]-pred[:,:-1]).abs().mean((2,3,4)); td=(target[:,1:]-target[:,:-1]).abs().mean((2,3,4)); boundary.append(F.l1_loss(pd,td).item())
        row={'step':step,'psnr':float(np.mean(psnr)),'temporal_error':float(np.mean(boundary))}; append_metrics(metrics_path,row); print('[eval]',row); core.train()

    optimizer.zero_grad(set_to_none=True); epoch=0
    while global_step<args.max_steps:
        if sampler: sampler.set_epoch(epoch)
        for it,batch in enumerate(tqdm(loader,disable=not main_process)):
            frames=batch['frames'].to(device)
            if frames.shape[1] != args.seq_len:
                raise RuntimeError(
                    f'dataset returned {frames.shape[1]} frames, expected '
                    f'{args.seq_len}; set num_frames_per_video=seq_len')
            geo,tex=encode(frames)
            pred,geo_rec,tex_rec,z=model(geo,tex,args.frames_chunk_size); target=frames.float().permute(0,1,3,4,2)
            l1=F.l1_loss(pred,target); latent=F.l1_loss(geo_rec,geo)+F.l1_loss(tex_rec,tex); temp=temporal_loss(pred,target); accel=acceleration_loss(pred,target); geo_motion=temporal_loss(geo_rec,geo); reg=latent_regularization(z)
            total=args.lambda_l1*l1+args.lambda_latent*latent+args.lambda_temporal*temp+args.lambda_accel*accel+args.lambda_geo_motion*geo_motion+args.lambda_comp_reg*reg
            (total/args.accum_steps).backward()
            if (it+1)%args.accum_steps: continue
            grad=torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad],args.max_grad_norm); optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); global_step+=1
            if main_process and global_step%args.log_every==0:
                row={'step':global_step,'train/loss':float(total),'train/l1':float(l1),'train/latent':float(latent),'train/temporal':float(temp),'train/accel':float(accel),'train/geo_motion':float(geo_motion),'train/reg':float(reg),'train/grad_norm':float(grad),'train/lr':scheduler.get_last_lr()[0]}; append_metrics(metrics_path,row); print(datetime.now(),row)
            if global_step%args.eval_every==0:
                if use_ddp: dist.barrier()
                if main_process: evaluate(global_step)
                if use_ddp: dist.barrier()
            if global_step%args.save_every==0 and main_process: save(global_step)
            if global_step>=args.max_steps: break
        epoch+=1
    if use_ddp: dist.barrier()
    if main_process: evaluate(global_step); save(global_step); writer.close() if writer else None
    if use_ddp: dist.barrier(); dist.destroy_process_group()


if __name__=='__main__': main()
