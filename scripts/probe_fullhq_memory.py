"""Real 24-rank optimizer/EMA/CFG/decode memory acceptance before full caching."""
import argparse
import contextlib
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from models.r7_window_dit import R7WindowDiT
from utils.device import get_device,configure_backend_compatibility
from utils.distributed import setup_ddp
from utils.training import EMA,build_optimizer
from utils.window_training import use_ema
from utils.fullhq_text import TextCFG
from scripts.window_run_io import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--r7_ckpt',required=True);p.add_argument('--output',required=True)
    p.add_argument('--limit_gib',type=float,default=52.);a=p.parse_args()
    ddp,rank,local,world=setup_ddp();device=get_device(local)
    if world!=24 or device.type!='npu':raise RuntimeError('probe requires 3x8 NPU topology')
    configure_backend_compatibility('npu');torch.manual_seed(42)
    from utils.r7_representation import load_r7_modules,load_checkpoint
    from utils.window_codec import configure_codec
    _,comp,tex,codec,decoder,_=load_r7_modules(load_checkpoint(a.r7_ckpt));del comp,tex
    configure_codec(codec,'legacy')
    for module in (codec,decoder):module.to(device).eval().requires_grad_(False)
    core=R7WindowDiT(width=1536,depth=24,heads=24,text_dim=4096).to(device)
    optimizer=build_optimizer(core,1e-4,.01)
    for group in optimizer.param_groups:group['foreach']=False
    ema=EMA(core,.999,dtype=torch.float32,warmup=True).to(device)
    model=torch.nn.parallel.DistributedDataParallel(core,device_ids=[local],gradient_as_bucket_view=True)
    x=torch.randn(1,4,324,192,device=device);anchor=torch.randn(1,1,324,192,device=device)
    text=torch.randn(1,256,4096,device=device);valid=torch.ones(1,256,dtype=torch.bool,device=device)
    times=[]
    for step in range(3):
        start=time.monotonic();optimizer.zero_grad(set_to_none=True)
        for micro in range(2):
            with model.no_sync() if micro==0 else contextlib.nullcontext():
                with torch.autocast('npu',dtype=torch.bfloat16):pred=model(x,torch.ones(1,device=device)*.5,anchor,text,valid)
                (pred.square().mean()/2).backward()
        norm=torch.nn.utils.clip_grad_norm_(core.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('nonfinite synthetic training gradients')
        optimizer.step();ema.update(core);torch.npu.synchronize();times.append(time.monotonic()-start)
        print(f'[fullHQ memory] rank={rank} optimizer_step={step+1}',flush=True)
    core.eval()
    with torch.no_grad(),use_ema(core,ema,cpu_backup=True):
        with torch.autocast('npu',dtype=torch.bfloat16):
            pred=TextCFG(core,text[0,:1].cpu(),3.)(x,torch.ones(1,device=device),anchor,text,valid)
        seq=torch.cat((anchor,pred),1).reshape(1,5,18,18,192)
        geo,texture=codec.decode(seq)
        rgb=decoder(geo,texture,frames_chunk_size=1)
        if not torch.isfinite(rgb).all():raise RuntimeError('nonfinite synthetic decode')
    torch.npu.synchronize()
    peak=torch.tensor([torch.npu.max_memory_allocated()/2**30,torch.npu.max_memory_reserved()/2**30],device=device)
    dist.all_reduce(peak,op=dist.ReduceOp.MAX)
    passed=float(peak[0])<=a.limit_gib
    if rank==0:atomic(Path(a.output)/'memory_probe.json',dict(status='passed' if passed else 'failed',
        parameters=sum(p.numel() for p in core.parameters()),world_size=world,text_len=256,
        max_rank_allocated_gib=float(peak[0]),max_rank_reserved_gib=float(peak[1]),limit_gib=a.limit_gib,
        rank0_step_seconds=times,scope='synthetic optimizer/EMA/CFG/R7 decode; no quality validation'))
    dist.barrier();dist.destroy_process_group()
    if not passed:raise RuntimeError('insufficient memory margin')


if __name__=='__main__':main()
