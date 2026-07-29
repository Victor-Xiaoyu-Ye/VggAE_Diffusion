#!/usr/bin/env python3
"""Flow-matching world model for R7 temporally compressed video latents."""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models.compact_dit import CompactLatentDiT
from utils.device import get_device
from utils.distributed import setup_ddp
from utils.training import EMA, atomic_torch_save, build_optimizer, build_scheduler


class R7LatentDataset(Dataset):
    def __init__(self, manifest):
        with open(manifest) as handle:
            self.paths = [line.strip() for line in handle if line.strip()]

    def __len__(self): return len(self.paths)
    def __getitem__(self, index):
        pack = torch.load(self.paths[index], map_location='cpu', weights_only=False)
        return pack['latent'].float()


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True); p.add_argument('--output_dir',required=True)
    p.add_argument('--latent_dim',type=int,default=192); p.add_argument('--latent_grid',type=int,default=18)
    p.add_argument('--context_chunks',type=int,default=1); p.add_argument('--future_chunks',type=int,default=2)
    p.add_argument('--model_dim',type=int,default=1152); p.add_argument('--spatial_depth',type=int,default=10); p.add_argument('--temporal_depth',type=int,default=8); p.add_argument('--num_heads',type=int,default=16)
    p.add_argument('--batch_size',type=int,default=2); p.add_argument('--max_steps',type=int,default=6000); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--warmup_steps',type=int,default=300)
    p.add_argument('--context_noise',type=float,default=0.0); p.add_argument('--num_workers',type=int,default=4); p.add_argument('--save_every',type=int,default=500); p.add_argument('--seed',type=int,default=42); p.add_argument('--resume',default='')
    return p.parse_args()


def main():
    args=parse_args(); torch.manual_seed(args.seed); use_ddp, rank, local_rank, world_size = setup_ddp(); device=get_device(local_rank); os.makedirs(args.output_dir,exist_ok=True)
    dataset=R7LatentDataset(args.manifest)
    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=sampler is None,sampler=sampler,num_workers=args.num_workers,drop_last=True)
    model=CompactLatentDiT(latent_dim=args.latent_dim,num_tokens=args.latent_grid**2,model_dim=args.model_dim,spatial_depth=args.spatial_depth,temporal_depth=args.temporal_depth,num_heads=args.num_heads,seq_len=args.future_chunks,text_cond=False,i0_condition=True,block_schedule='interleaved').to(device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False)
    core = model.module if hasattr(model, 'module') else model
    optimizer=build_optimizer(core,args.lr,1e-2); scheduler=build_scheduler(optimizer,args.warmup_steps,args.max_steps); ema=EMA(core,0.999,dtype=torch.float32).to(device); step=0
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False); core.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler']); ema.load_state_dict(ck['ema']); ema=ema.to(device); step=ck['global_step']
    writer=SummaryWriter(os.path.join(args.output_dir,'tb')) if rank == 0 else None
    epoch=0
    while step<args.max_steps:
        if sampler is not None: sampler.set_epoch(epoch)
        for latent in loader:
            latent=latent.to(device); required=args.context_chunks+args.future_chunks
            if latent.shape[1]<required: raise ValueError(f'latent sequence {latent.shape[1]} < {required}')
            start=torch.randint(0,latent.shape[1]-required+1,(1,)).item(); window=latent[:,start:start+required]
            context=window[:,:args.context_chunks]; target=window[:,args.context_chunks:]
            if args.context_noise>0: context=context+torch.randn_like(context)*args.context_noise
            x0=torch.randn_like(target); t=torch.rand(target.shape[0],device=device); tv=t.view(-1,1,1,1); xt=(1-tv)*x0+tv*target; velocity=target-x0
            cond=context.mean(dim=1,keepdim=True); pred=model(xt,t,cond=cond); loss=F.mse_loss(pred,velocity)
            loss.backward(); torch.nn.utils.clip_grad_norm_(core.parameters(),1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); ema.update(core); step+=1
            if writer: writer.add_scalar('train/loss',loss.item(),step)
            if rank == 0 and (step%args.save_every==0 or step>=args.max_steps):
                payload={'model':core.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'global_step':step,'args':vars(args)}; atomic_torch_save(payload,os.path.join(args.output_dir,'checkpoint_latest.pt'))
            if step>=args.max_steps: break
        epoch += 1
    if writer: writer.close()
    if use_ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__=='__main__': main()
