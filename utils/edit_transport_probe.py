"""Controlled image effects and resumable decoder-input optimization.

CPU builds known correspondences. Optimization uses ordinary decoder reverse
autograd only: no NPU grid_sample backward/JVP or high-dimensional Jacobian.
"""
import math
import time
import torch
from torch.nn import functional as F


def transform_grid(height,width,angle=0.,scale=1.,shift=(0.,0.)):
    """Inverse grid: output pixel centre -> input pixel centre, align_corners=False."""
    yy,xx=torch.meshgrid(torch.arange(height).float(),torch.arange(width).float(),indexing='ij')
    x=xx-(width-1)/2-shift[0];y=yy-(height-1)/2-shift[1]
    c,s=math.cos(math.radians(angle)),math.sin(math.radians(angle))
    px=(c*x+s*y)/scale+(width-1)/2
    py=(-s*x+c*y)/scale+(height-1)/2
    return torch.stack((2*(px+.5)/width-1,2*(py+.5)/height-1),-1)[None]


def warp(value,grid):
    # Only preparation/baselines use this; no accelerator backward requirement.
    return F.grid_sample(value.float(),grid.to(value.device),mode='bilinear',
        padding_mode='zeros',align_corners=False)


def synthetic_sequence(image,rect,cfg):
    """Real source first frame -> known 2D transforms; no edited-target read."""
    h,w=image.shape[-2:];y,x,rh,rw=rect
    valid=torch.zeros(1,1,h,w);valid[:,:,y:y+rh,x:x+rw]=1
    frames=[];grids=[];visibility=[]
    for i in range(cfg['frames']):
        a=i/(cfg['frames']-1)
        grid=transform_grid(h,w,cfg['rotation_degrees']*a,1+cfg['scale_change']*a,
            [v*a for v in cfg['translation_pixels']])
        frames.append(warp(image,grid)[0]);grids.append(grid)
        visibility.append((warp(valid,grid)>.999).float())
    return torch.stack(frames),grids,visibility


def analytic_edit(image,rect,cfg):
    """Soft ellipse is a controlled spatial edit, NOT an object/semantic mask."""
    h,w=image.shape[-2:];top,left,rh,rw=rect
    yy,xx=torch.meshgrid(torch.arange(h).float(),torch.arange(w).float(),indexing='ij')
    radius=((xx-(left+.5*rw))/(.22*rw)).square()+((yy-(top+.55*rh))/(.22*rh)).square()
    support=((1-radius)*8).clamp(0,1)[None,None]
    effect=support*torch.tensor(cfg['edit_rgb']).reshape(1,3,1,1)
    edited=(image+effect).clamp(0,1)
    return edited,edited-image,support


def observation_mask(valid,fraction,tiles,seed,seed_support):
    generator=torch.Generator().manual_seed(seed)
    coarse=(torch.rand(1,1,tiles,tiles,generator=generator)<fraction).float()
    mask=F.interpolate(coarse,size=valid.shape[-2:],mode='nearest')
    # A known planned-edit seed makes the test identifiable, without inspecting
    # any withheld effect values. All methods receive this exact observation set.
    points=torch.nonzero(seed_support[0,0]>.9)
    if len(points):
        y,x=points[len(points)//2].tolist();r=max(2,min(valid.shape[-2:])//32)
        mask[:,:,max(0,y-r):y+r+1,max(0,x-r):x+r+1]=1
    return mask*valid


def rgb_completion(observed,mask,source,iterations=120,observation_weight=8.):
    """Source-edge-aware harmonic fill/denoise with soft observation confidence."""
    # Coarse normalized interpolation initializes holes; source edges then weight
    # a four-neighbour diffusion. No withheld target values enter this routine.
    small_num=F.adaptive_avg_pool2d(observed*mask,(8,8))
    small_den=F.adaptive_avg_pool2d(mask,(8,8))
    average=(observed*mask).sum((2,3),keepdim=True)/mask.sum().clamp_min(1)
    small=torch.where(small_den>0,small_num/small_den.clamp_min(1e-6),average)
    value=F.interpolate(small,source.shape[-2:],mode='bilinear',align_corners=False)
    shifts=((-1,0),(1,0),(0,-1),(0,1));weights=[]
    for dy,dx in shifts:
        neighbour=torch.roll(source,(dy,dx),(-2,-1))
        weight=torch.exp(-(source-neighbour).square().mean(1,keepdim=True)/(.1**2))+.001
        if dy<0:weight[:,:,-1]=0
        if dy>0:weight[:,:,0]=0
        if dx<0:weight[:,:,:,-1]=0
        if dx>0:weight[:,:,:,0]=0
        weights.append(weight)
    denom=sum(weights).clamp_min(1e-6)
    for _ in range(iterations):
        neighbours=sum(w*torch.roll(value,shift,(-2,-1)) for w,shift in zip(weights,shifts))
        value=(observation_weight*mask*observed+neighbours)/(observation_weight*mask+denom)
    return value


def effect_metrics(pred,truth,valid,observed):
    def mean(value,weight):
        n=float(weight.sum())
        return None if n<1e-6 else float((value.mean(1,keepdim=True)*weight).sum()/n)
    roi=(truth.abs().amax(1,keepdim=True)>2/255).float()*valid
    hidden=valid*(1-observed)
    mse=mean((pred-truth).square(),valid)
    gain_denom=(truth.square()*roi).sum()
    return dict(effect_mse=mse,effect_l1=mean((pred-truth).abs(),valid),
        observed_l1=mean((pred-truth).abs(),observed),heldout_l1=mean((pred-truth).abs(),hidden),
        edit_l1=mean((pred-truth).abs(),roi),heldout_edit_l1=mean((pred-truth).abs(),hidden*roi),
        keep_leakage_l1=mean(pred.abs(),valid*(1-roi)),
        target_effect_rms=math.sqrt(mean(truth.square(),valid)),
        predicted_effect_rms=math.sqrt(mean(pred.square(),valid)),
        effect_gain=None if float(gain_denom)<1e-10 else float((pred*truth*roi).sum()/gain_denom),
        observed_fraction=float(observed.sum()/valid.sum().clamp_min(1)),
        metric_caveat='Controlled RGB-effect accuracy; NOT instruction following, real motion or 3D geometry.')


def solve_effect(decode,base,target_observations,mask,scale,cfg,initial=None,
                 gradient_base=None,resume=None,checkpoint=None,heartbeat=None):
    """Only masked observations enter optimization; best selection uses these too.

    gradient_base tests a fixed warped-anchor response (same actual forward,
    approximate Jacobian). Best state/optimizer/step are checkpointed exactly.
    """
    started=time.monotonic()
    with torch.no_grad():reference=decode(base).float().detach()
    scale=scale.to(base.device).float()
    parameter=torch.nn.Parameter(torch.zeros_like(base,dtype=torch.float32))
    if initial is not None:
        with torch.no_grad():parameter.copy_(initial.to(base.device)/scale)
    with torch.no_grad():parameter.clamp_(-cfg['trust'],cfg['trust'])
    optimizer=torch.optim.Adam([parameter],lr=cfg['lr'])
    start=0;history=[];evaluations=1;elapsed_prior=0.
    if resume:
        with torch.no_grad():parameter.copy_(resume['parameter'].to(base.device))
        optimizer.load_state_dict(resume['optimizer']);start=resume['step'];history=resume['history']
        evaluations=resume.get('evaluations',0)+1;elapsed_prior=resume.get('elapsed',0.)
    target=target_observations.to(base.device).float()*mask.to(base.device)
    mask=mask.to(base.device).float()
    denominator=(mask.sum()*target.shape[1]).clamp_min(1)
    energy=(target.square()*mask).sum()/denominator
    energy=energy.clamp_min(1e-6)
    best=resume['best'].to(base.device) if resume else parameter.detach().clone()
    best_value=resume['best_value'] if resume else float('inf')
    gradient_checked=bool(resume and resume.get('gradient_checked'))
    def payload(step):
        return dict(parameter=parameter.detach().cpu(),optimizer=optimizer.state_dict(),step=step,
            best=best.detach().cpu(),best_value=best_value,history=history,evaluations=evaluations,
            elapsed=elapsed_prior+time.monotonic()-started,gradient_checked=gradient_checked)
    for step in range(start,cfg['iterations']+1):
        optimizer.zero_grad(set_to_none=True)
        if gradient_base is None:
            effect=decode(base+scale*parameter).float()-reference;evaluations+=1
        else:
            with torch.no_grad():actual=decode(base+scale*parameter).float()-reference
            proxy=decode(gradient_base+scale*parameter).float();evaluations+=2
            effect=actual+proxy-proxy.detach()
        fit=((effect-target).square()*mask).sum()/denominator/energy
        if step<cfg['iterations'] and not effect.requires_grad:
            raise RuntimeError('Decoder output is detached from the editable input')
        loss=fit+cfg['ridge']*parameter.square().mean()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite decoder inverse objective')
        current=float(loss.detach())
        if current<best_value:best_value=current;best=parameter.detach().clone()
        row=dict(step=step,observed_relative_mse=float(fit.detach()),objective=current,
                 delta_standardized_rms=float(parameter.detach().square().mean().sqrt()))
        if step==start or step%cfg['log_every']==0 or step==cfg['iterations']:
            history.append(row)
            if heartbeat:heartbeat(dict(row,DI_throughput=(evaluations-(resume or {}).get('evaluations',0))/max(1e-6,time.monotonic()-started),
                throughput_unit='decoder forward frames/s/npu including backward and IO time'))
        if step==cfg['iterations']:
            if checkpoint:checkpoint(payload(step))
            break
        if not gradient_checked and (step==start or step%cfg['log_every']==0):
            # The regularizer itself has a parameter gradient, even when the
            # decoder is detached. Verify the OBSERVATION path independently.
            fit_gradient=torch.autograd.grad(fit,parameter,retain_graph=True,allow_unused=True)[0]
            if fit_gradient is None or not bool(torch.isfinite(fit_gradient).all()):
                raise FloatingPointError('Observation objective has no finite decoder-input gradient')
            gradient_checked=float(fit_gradient.abs().sum())>1e-12
        loss.backward()
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError('Decoder must provide a finite gradient to its input')
        optimizer.step()
        with torch.no_grad():parameter.clamp_(-cfg['trust'],cfg['trust'])
        if checkpoint and (step+1)%cfg['checkpoint_every']==0:checkpoint(payload(step+1))
    with torch.no_grad():effect=(decode(base+scale*best).float()-reference).detach();evaluations+=1
    return dict(delta=scale*best,effect=effect,history=history,gradient_checked=gradient_checked,
                evaluations=evaluations,elapsed=elapsed_prior+time.monotonic()-started)
