"""Small shared training/IO helpers. No changes to historical scene trainers."""
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist


def distributed():
    from utils.device import get_device,configure_backend_compatibility
    device=get_device(int(os.environ.get('LOCAL_RANK',0)))
    if device.type!='cpu': getattr(torch,device.type).set_device(device)
    if int(os.environ.get('WORLD_SIZE',1))>1:
        dist.init_process_group(dict(npu='hccl',cuda='nccl',cpu='gloo')[device.type],
                                timeout=timedelta(seconds=7200))
    configure_backend_compatibility(device.type)
    return device,dist.get_rank() if dist.is_initialized() else 0,dist.get_world_size() if dist.is_initialized() else 1


def barrier():
    if dist.is_initialized(): dist.barrier()


def gather(value,world):
    if world==1: return [value]
    values=[None]*world; dist.all_gather_object(values,value); return values


def any_rank(flag,device):
    value=torch.tensor(int(flag),device=device)
    if dist.is_initialized(): dist.all_reduce(value,op=dist.ReduceOp.MAX)
    return bool(value.item())


def progress(store,name,row):
    from utils.training import append_metrics
    row=dict(time=time.time(),**row)
    append_metrics(str(store.local/'logs'/f'{name}.jsonl'),row)
    store.write_json(f'progress/{name}.json',row)
    print('[editing] '+json.dumps(row),flush=True)


def load_teacher(path,device):
    from models.scene_rae import SceneRAE
    value=torch.load(str(path),map_location='cpu',weights_only=False,mmap=os.name!='nt')
    if value['spec'].get('schema')!='scene-rae-v1' or value['spec']['channels']!=256:
        raise ValueError('Need the production Scene C256 AE checkpoint')
    model=SceneRAE(value['spec']['source_config'],256)
    model.load_state_dict(value['model'],strict=True)
    return model.eval().requires_grad_(False).to(device)


def load_encoder(path,device):
    from streamvggt.models.streamvggt import StreamVGGT
    from utils.encoder_loader import load_encoder_checkpoint
    model=StreamVGGT(); load_encoder_checkpoint(model,path,verbose=False)
    return model.eval().requires_grad_(False).to(device)


def sample_row(rows,step,micro,rank,world,accum,seed):
    # Stable shuffled epochs; exact logical cursor derived from global update.
    import random
    cursor=(step*accum+micro)*world+rank
    epoch,offset=divmod(cursor,len(rows))
    order=list(range(len(rows))); random.Random(seed+epoch).shuffle(order)
    return rows[order[offset]]


def update_ema(ema,model,step,decay):
    decay=min(decay,(1+step)/(10+step))
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:
                ema[name].mul_(decay).add_(p.detach().float(),alpha=1-decay)


def write_video(path,source,target,generated,fps,labels=None):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image,ImageDraw
    values=[source,target,generated]
    names=labels or ['SOURCE','TARGET (training specialist)','NATIVE VACE OUTPUT']
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    frames=[]
    for triplet in zip(*values):
        cells=[]
        for value,label in zip(triplet,names):
            arr=value.detach().cpu().permute(1,2,0).numpy().astype('uint8')
            canvas=Image.fromarray(arr); ImageDraw.Draw(canvas).text((6,6),label,fill='white',stroke_width=1,stroke_fill='black')
            cells.append(np.asarray(canvas))
        frame=np.concatenate(cells,axis=1)
        # yuv420 requires even dimensions. Pad at most one pixel, never stretch.
        frames.append(np.pad(frame,((0,frame.shape[0]%2),(0,frame.shape[1]%2),(0,0))))
    imageio.mimwrite(str(path),frames,fps=fps,macro_block_size=1)


def pixel_report(source,target,generated):
    a,b,c=[x.float()/255 for x in (source,target,generated)]
    return dict(target_psnr=float(-10*(c-b).square().mean().clamp_min(1e-10).log10()),
        source_change=float((c-a).abs().mean()),target_change=float((b-a).abs().mean()),
        output_frame_delta=float((c[1:]-c[:-1]).abs().mean()),
        target_frame_delta=float((b[1:]-b[:-1]).abs().mean()),
        metric_caveat='Pixel diagnostics only; PSNR to a specialist target is not editing quality or geometry accuracy.')
