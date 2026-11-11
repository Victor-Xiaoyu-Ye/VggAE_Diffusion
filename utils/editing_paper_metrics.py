"""Independent-of-method output diagnostics, with honest evaluator limits.

Paired targets are dataset-generated references, not unique ground truth.
Their difference masks are evaluation-only and not certified semantic masks.
Farneback is a different source-flow estimator from the method's DIS, not a
ground-truth geometry evaluator. Human edit/quality judgments remain separate.
"""
from pathlib import Path
import math

import numpy as np
import torch


def rgb_array(value):
    v=torch.as_tensor(value).detach().cpu()
    if v.ndim!=4 or v.shape[1]!=3:raise ValueError('Evaluation needs [T,3,H,W]')
    if v.dtype==torch.uint8:v=v.float()/255
    else:v=v.float()
    if not bool(torch.isfinite(v).all()) or bool(((v<0)|(v>1)).any()):
        raise ValueError('Invalid evaluation pixels')
    return v.permute(0,2,3,1).numpy()


def optional_lpips(device, enabled=True):
    if not enabled:return None,dict(status='disabled')
    try:
        # Check the precise torchvision cache BEFORE constructing LPIPS, so
        # the offline cluster never attempts an implicit weight download.
        checkpoint=Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
        if not checkpoint.is_file():raise FileNotFoundError(str(checkpoint))
        import lpips
        learned=Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
        if not learned.is_file():raise FileNotFoundError(str(learned))
        model=lpips.LPIPS(net='alex',version='0.1',verbose=False).eval().requires_grad_(False).to(device)
        return model,dict(status='available',name='LPIPS-Alex-v0.1',weights=str(checkpoint))
    except (ImportError,FileNotFoundError,RuntimeError) as exc:
        return None,dict(status='unavailable',reason=repr(exc),download_attempted=False)


def evaluation_context(source,target):
    import cv2
    src,truth=rgb_array(source),rgb_array(target)
    if src.shape!=truth.shape:raise ValueError('Paired evaluation shapes differ')
    # Fixed before seeing generated outputs. Never use the method's masks.
    changed=np.max(np.abs(truth-src),axis=-1)>8/255
    size=(max(8,round(src.shape[2]*min(1,256/max(src.shape[1:3])))),
          max(8,round(src.shape[1]*min(1,256/max(src.shape[1:3])))))
    small=np.stack([cv2.resize(f,size,interpolation=cv2.INTER_AREA) for f in src])
    masks=np.stack([cv2.resize(f.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0 for f in changed])
    gray=[cv2.cvtColor((f*255).round().astype(np.uint8),cv2.COLOR_RGB2GRAY) for f in small]
    h,w=gray[0].shape; yy,xx=np.mgrid[:h,:w].astype(np.float32)
    maps=[]; magnitudes=[]; residuals=[]
    for i in range(1,len(src)):
        fwd=cv2.calcOpticalFlowFarneback(gray[i-1],gray[i],None,.5,3,15,3,5,1.2,0)
        back=cv2.calcOpticalFlowFarneback(gray[i],gray[i-1],None,.5,3,15,3,5,1.2,0)
        mx,my=xx+back[...,0],yy+back[...,1]
        warp=lambda v:cv2.remap(v,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        inside=(mx>=0)&(mx<=w-1)&(my>=0)&(my<=h-1)
        consistent=np.linalg.norm(back+warp(fwd),axis=-1)<1.5
        photo=np.mean(np.abs(small[i]-warp(small[i-1])),axis=-1)<.12
        valid=inside&consistent&photo
        edit=(masks[i]|(warp(masks[i-1].astype(np.float32))>.5))&valid
        maps.append((mx,my,edit,valid))
        magnitudes.append(float(np.linalg.norm(fwd,axis=-1).mean()))
        median=np.median(fwd.reshape(-1,2),axis=0)
        residuals.append(float(np.linalg.norm(fwd-median,axis=-1).mean()))
    return dict(source=src,target=truth,changed=changed,source_small=small,maps=maps,size=size,
        source_motion_proxy_px=float(np.mean(magnitudes)) if magnitudes else 0.,
        source_nonuniform_motion_proxy_px=float(np.mean(residuals)) if residuals else 0.)


def ssim(a,b):
    import cv2
    scores=[]
    for x,y in zip(a,b):
        ux=cv2.GaussianBlur(x,(11,11),1.5);uy=cv2.GaussianBlur(y,(11,11),1.5)
        vx=cv2.GaussianBlur(x*x,(11,11),1.5)-ux*ux
        vy=cv2.GaussianBlur(y*y,(11,11),1.5)-uy*uy
        xy=cv2.GaussianBlur(x*y,(11,11),1.5)-ux*uy
        value=((2*ux*uy+.01**2)*(2*xy+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
        scores.append(float(value.mean()))
    return float(np.mean(scores))


def _weighted(error,mask):
    count=int(mask.sum())
    return float(error[mask].mean()) if count else None


def evaluate_videos(source,target,generated,perceptual=None,device='cpu',context=None):
    import cv2
    c=context or evaluation_context(source,target)
    src,truth,g=c['source'],c['target'],rgb_array(generated)
    if src.shape!=g.shape:raise ValueError('Generated time/geometry mismatch')
    err=np.abs(g-truth).mean(-1); source_error=np.abs(g-src).mean(-1)
    mse=float(np.square(g-truth).mean())
    motion_den=float(np.abs(src[1:]-src[:-1]).mean()) if len(src)>1 else 0.
    output_delta=float(np.abs(g[1:]-g[:-1]).mean()) if len(src)>1 else 0.
    result=dict(target_psnr=-10*math.log10(max(mse,1e-12)),target_ssim=ssim(g,truth),
        target_l1=float(err.mean()),dataset_edit_l1=_weighted(err,c['changed']),
        dataset_keep_source_l1=_weighted(source_error,~c['changed']),
        dataset_keep_fraction=float((~c['changed']).mean()),
        source_frame_delta=motion_den,output_frame_delta=output_delta,
        frame_delta_ratio=output_delta/motion_den if motion_den>1e-6 else None,
        source_motion_proxy_px=c['source_motion_proxy_px'],
        source_nonuniform_motion_proxy_px=c['source_nonuniform_motion_proxy_px'])
    small=np.stack([cv2.resize(f,c['size'],interpolation=cv2.INTER_AREA) for f in g])
    effect=small-c['source_small']; error_sum=0.; error_count=0; valid_pixels=0; total=0
    for i,(mx,my,edited,valid) in enumerate(c['maps'],1):
        previous=cv2.remap(effect[i-1],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        difference=np.abs(effect[i]-previous).mean(-1)
        error_sum+=float(difference[edited].sum(dtype=np.float64));error_count+=int(edited.sum())
        valid_pixels+=int(valid.sum());total+=valid.size
    result['source_tracked_effect_l1']=error_sum/error_count if error_count else None
    result['source_flow_valid_fraction']=valid_pixels/total if total else None
    if perceptual is not None:
        scores=[]
        with torch.no_grad():
            for x,y in zip(g,truth):
                pair=[torch.from_numpy(np.ascontiguousarray(v)).permute(2,0,1)[None].to(device)*2-1 for v in (x,y)]
                scores.append(float(perceptual(*pair).mean()))
        result['target_lpips']=float(np.mean(scores))
    else:result['target_lpips']=None
    return result


def aggregate(records,arms,bootstrap=2000,seed=42):
    """Average seeds within source first; do not inflate N with frames/seeds."""
    values={}
    for row in records:
        if row.get('status')!='completed' or 'metrics' not in row:continue
        for key,value in row['metrics'].items():
            if isinstance(value,(float,int)) and math.isfinite(value):
                values.setdefault((row['arm'],key),{}).setdefault(row['group'],{})[row['seed_base']]=float(value)
    means={}; paired={}
    for (arm,key),groups in values.items():
        vector=[np.mean(list(v.values())) for v in groups.values()]
        means.setdefault(arm,{})[key]=dict(mean=float(np.mean(vector)),source_groups=len(vector))
    for control in arms:
        if control=='rae_full':continue
        for (arm,key),groups in values.items():
            if arm!='rae_full' or (control,key) not in values:continue
            common=sorted(set(groups)&set(values[(control,key)]))
            if not common:continue
            differences=[]; matched_seeds=[]
            for g in common:
                other=values[(control,key)][g]; seeds=sorted(set(groups[g])&set(other))
                if seeds:
                    differences.append(np.mean([groups[g][s]-other[s] for s in seeds]))
                    matched_seeds.append(len(seeds))
            if not differences:continue
            delta=np.array(differences)
            rng=np.random.default_rng(seed)
            sampled=delta[rng.integers(0,len(delta),size=(bootstrap,len(delta)))].mean(1)
            paired.setdefault(control,{})[key]=dict(rae_minus_control=float(delta.mean()),
                paired_source_groups=len(delta),paired_seed_counts=matched_seeds,
                lower_count=int((delta<0).sum()),higher_count=int((delta>0).sum()),
                bootstrap95=np.quantile(sampled,[.025,.975]).tolist(),
                caveat='Paired intersection; coverage/failures must accompany this comparison')
    return dict(means=means,paired=paired,aggregation='seed mean then source-group bootstrap')
