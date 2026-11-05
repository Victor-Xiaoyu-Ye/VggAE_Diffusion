"""Source-only optical-flow propagation and frozen-decoder video edit calibration.

No target video, depth labels, SAM2 or downloaded optical-flow weights are used.
The support is a user/source/reference-derived region, not dynamic segmentation.
The graph solver keeps one decoder's backward graph resident at a time.
"""
import copy
import time

import numpy as np
import torch
from torch.nn import functional as F


def _stencil(flow):
    """CPU bilinear pull-map stencil, output pixel -> input pixel, zero border."""
    flow=torch.as_tensor(flow,dtype=torch.float32,device='cpu')
    h,w=flow.shape[:2]
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    x=xx+flow[...,0];y=yy+flow[...,1]
    x0=x.floor().long();y0=y.floor().long();dx=x-x0;dy=y-y0
    indices=[];weights=[]
    for ox,oy,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),
                         (0,1,(1-dx)*dy),(1,1,dx*dy)):
        px=x0+ox;py=y0+oy
        valid=(px>=0)&(px<w)&(py>=0)&(py<h)
        indices.append((py.clamp(0,h-1)*w+px.clamp(0,w-1)).flatten())
        weights.append((weight*valid).flatten())
    return indices,weights,(h,w)


def _apply_stencil(value,stencil):
    indices,weights,(h,w)=stencil
    if value.shape[-2:]!=(h,w):raise ValueError('Flow and image canvas differ')
    flat=value.flatten(-2)
    # Ordinary gather backward; no grid_sample backward or forward-mode AD.
    result=sum(flat.index_select(-1,i.to(value.device))*a.to(value.device)
               for i,a in zip(indices,weights))
    return result.reshape(*value.shape[:-2],h,w)


def warp_flow(value,flow):
    """Differentiable bilinear warp by pixel displacement [H,W,2]."""
    return _apply_stencil(value,_stencil(flow))


def _check_video(source,anchor,support):
    if source.ndim!=4 or source.shape[1]!=3 or len(source)<1:
        raise ValueError('Source must be [T,3,H,W]')
    if anchor.shape!=source.shape[1:] or support.shape!=(1,*source.shape[-2:]):
        raise ValueError('Anchor and support must match the source RGB canvas')
    if any(not bool(torch.isfinite(x).all()) for x in (source,anchor,support)):
        raise ValueError('Nonfinite source, anchor or support')


def propagate_reference(source,anchor,support,cfg):
    """Propagate one legal edited first frame over a real observed source video.

    DIS/Farneback uses RGB source frames only. Forward/backward consistency and
    source photometric residual weight both adjacent propagation and independent
    first-frame matches, so a reappearing surface need not inherit a broken chain.
    Direct long-range flow is still a classical approximation, not reidentification.
    Unreliable effect pixels remain the source; masks are approximate flow masks.
    """
    import cv2
    source=source.detach().float().cpu();anchor=anchor.detach().float().cpu()
    support=support.detach().float().cpu().clamp(0,1)
    _check_video(source,anchor,support)
    t,_,h,w=source.shape
    max_side=int(cfg.get('flow_max_side',384))
    ratio=min(1.,max_side/max(h,w))
    fh,fw=max(8,round(h*ratio)),max(8,round(w*ratio))
    rgb=(source.clamp(0,1)*255).round().byte().permute(0,2,3,1).numpy()
    gray=[cv2.resize(cv2.cvtColor(v,cv2.COLOR_RGB2GRAY),(fw,fh),
                     interpolation=cv2.INTER_AREA) for v in rgb]
    requested=cfg.get('flow_backend','dis')
    if requested not in ('dis','farneback'):raise ValueError('Unknown flow_backend')
    backend='dis' if requested=='dis' and hasattr(cv2,'DISOpticalFlow_create') and min(fh,fw)>=16 else 'farneback'
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) if backend=='dis' else None
    fallback_count=0

    def estimate(a,b):
        nonlocal fallback_count
        if dis is not None:
            try:f=dis.calc(gray[a],gray[b],None)
            except cv2.error:
                fallback_count+=1
                f=cv2.calcOpticalFlowFarneback(gray[a],gray[b],None,.5,4,21,4,7,1.5,0)
        else:f=cv2.calcOpticalFlowFarneback(gray[a],gray[b],None,.5,4,21,4,7,1.5,0)
        f=cv2.resize(f,(w,h),interpolation=cv2.INTER_LINEAR)
        f[...,0]*=w/fw;f[...,1]*=h/fh
        if not np.isfinite(f).all():raise FloatingPointError('Nonfinite source optical flow')
        return torch.from_numpy(f.copy()).float()

    fb=float(cfg.get('flow_fb_threshold',1.5));photo=float(cfg.get('flow_photo_threshold',.12))
    minimum=float(cfg.get('flow_min_confidence',.1))
    if fb<=0 or photo<=0 or not 0<=minimum<=1:raise ValueError('Invalid flow confidence setting')

    def connection(a,b):
        forward=estimate(a,b);backward=estimate(b,a)
        stencil=_stencil(backward)
        warped_forward=_apply_stencil(forward.permute(2,0,1),stencil)
        residual=(warped_forward+backward.permute(2,0,1)).square().sum(0,keepdim=True).sqrt()
        photometric=(_apply_stencil(source[a],stencil)-source[b]).abs().mean(0,keepdim=True)
        valid=_apply_stencil(torch.ones(1,h,w),stencil)>.999
        # Motion magnitude is not a foreground/background classifier.
        tolerance=fb+.05*backward.square().sum(-1).sqrt()[None]
        confidence=torch.exp(-(residual/tolerance).square()-(photometric/photo).square())*valid
        return backward,stencil,confidence

    effect=(anchor-source[0])*support
    effects=[effect];confidences=[torch.ones_like(support)];masks=[support];flows=[]
    direct_enabled=bool(cfg.get('flow_direct_anchor',True))
    for i in range(1,t):
        backward,stencil,adj_conf=connection(i-1,i);flows.append(backward)
        chain_conf=_apply_stencil(confidences[-1],stencil)*adj_conf
        chain_effect=_apply_stencil(effects[-1],stencil)
        chain_mask=_apply_stencil(masks[-1],stencil)
        if direct_enabled and i>1:
            _,direct_stencil,direct_conf=connection(0,i)
            direct_effect=_apply_stencil(effect,direct_stencil)
            direct_mask=_apply_stencil(support,direct_stencil)
            denom=(chain_conf+direct_conf).clamp_min(1e-8)
            candidate=(chain_effect*chain_conf+direct_effect*direct_conf)/denom
            mask=(chain_mask*chain_conf+direct_mask*direct_conf)/denom
            conf=torch.maximum(chain_conf,direct_conf)
        else:candidate=chain_effect;mask=chain_mask;conf=chain_conf
        usable=conf>=minimum
        effects.append(candidate*usable);confidences.append(conf*usable)
        masks.append(mask.clamp(0,1))
    return dict(proposal=(source+torch.stack(effects)).clamp(0,1),
        confidence=torch.stack(confidences),masks=torch.stack(masks),
        flows_backward=torch.stack(flows) if flows else torch.empty(0,h,w,2),
        metadata=dict(flow_backend=backend,requested_backend=requested,
            flow_fallback_count=fallback_count,flow_resolution=[fh,fw],
            correspondence_source='source_rgb_only',direct_anchor_matches=direct_enabled,
            support_kind='flow_propagated_input_region_not_semantic_or_dynamic_mask',
            reliable_fraction=[float((v>=minimum).float().mean()) for v in confidences],
            limitations='Classical optical flow can fail on occlusion, textureless surfaces and large motion; not learned tracking or SAM2.'))


def _cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def calibrate_video(decode,z,source,proposal,confidence,masks,cfg,checkpoint=None,
                    resume=None,progress=None,flows_backward=None):
    """Block-coordinate optimization of a shared visible-effect graph.

    The first initialization sweeps fit the dense transported/native proposal;
    subsequent sweeps also penalize BOTH incoming and outgoing flow edges.
    Other frames' decoded effects are detached and updated after each block.
    This is not independent sparse-target inversion. No target-video interface.
    Checkpoints at completed sweeps preserve Adam, current and best graph states.
    """
    started=time.monotonic();device=z.device
    if z.ndim!=5 or z.shape[0]!=1:raise ValueError('z must be [1,T,h,w,c]')
    frames=z.shape[1]
    if source.ndim!=4 or source.shape[0]!=frames or source.shape[1]!=3 or proposal.shape!=source.shape:
        raise ValueError('Source/proposal must match all latent time states')
    if confidence.shape!=masks.shape or masks.shape!=(frames,1,*source.shape[-2:]):
        raise ValueError('Expected aligned confidence/masks [T,1,H,W]')
    if flows_backward is not None and tuple(flows_backward.shape)!=(frames-1,*source.shape[-2:],2):
        raise ValueError('Need backward flow between each pair of calibrated frames')
    if any(not bool(torch.isfinite(v).all()) for v in (z,source,proposal,confidence,masks)):
        raise ValueError('Nonfinite calibration inputs')
    z=z.detach().float();source=source.detach().float().cpu();proposal=proposal.detach().float().cpu()
    masks=masks.detach().float().cpu().clamp(0,1);confidence=confidence.detach().float().cpu().clamp(0,1)
    target=proposal-source
    std=z.flatten(0,3).std(0,unbiased=False).clamp_min(.05).reshape(1,1,1,1,-1)
    parameters=[torch.nn.Parameter(torch.zeros_like(z[:,i:i+1])) for i in range(frames)]
    optimizers=[torch.optim.Adam([p],lr=float(cfg.get('lr',.02))) for p in parameters]
    iterations=int(cfg.get('iterations',12));init_sweeps=int(cfg.get('initialization_sweeps',2))
    trust=float(cfg.get('trust',.75));ridge=float(cfg.get('ridge',1e-4))
    joint=float(cfg.get('joint_weight',.2));keep=float(cfg.get('keep_weight',1.))
    anchor_weight=float(cfg.get('anchor_weight',4.))
    fallback=float(cfg.get('proposal_fallback_weight',.2))
    if iterations<0 or trust<=0 or anchor_weight<=0 or min(ridge,joint,keep,fallback)<0:raise ValueError('Invalid calibration settings')
    stencils=[] if flows_backward is None else [_stencil(f) for f in flows_backward]
    if joint and frames>1 and not stencils:
        raise ValueError('joint_weight requires source flow; independent calibration must explicitly set it to zero')
    with torch.no_grad():base=[decode(z[:,i:i+1]).float().detach().cpu() for i in range(frames)]
    if any(b.shape!=(1,3,*source.shape[-2:]) for b in base):raise ValueError('Decoder RGB canvas differs')
    evaluations=frames;elapsed_prior=0.;history=[];start=0;gradient_checked=False
    best_value=float('inf');best=None
    if resume:
        if resume.get('schema')!='edit-video-calibration-v1' or len(resume['parameters'])!=frames:
            raise ValueError('Incompatible calibration checkpoint')
        start=int(resume['sweep']);history=copy.deepcopy(resume['history'])
        if start>iterations:raise ValueError('Checkpoint already exceeds requested iterations')
        for p,v,opt,state in zip(parameters,resume['parameters'],optimizers,resume['optimizers']):
            with torch.no_grad():p.copy_(v.to(device))
            opt.load_state_dict(state)
        best=resume['best'];best_value=resume['best_value'];gradient_checked=resume['gradient_checked']
        evaluations+=resume.get('evaluations',0);elapsed_prior=resume.get('elapsed',0.)
    weights=masks*(confidence+fallback*(1-confidence))
    energies=(target.square()*masks).sum((1,2,3))/(3*masks.sum((1,2,3))).clamp_min(1)
    energies=energies.clamp_min(1e-4)
    edge_weights=[]
    for i,s in enumerate(stencils,1):
        edge_weights.append(confidence[i]*_apply_stencil(confidence[i-1],s)*masks[i]*
                            _apply_stencil(masks[i-1],s))

    def mean_square(value,weight):
        return (value.square()*weight).sum()/(3*weight.sum()).clamp_min(1)

    def effect_at(i):
        nonlocal evaluations
        result=decode(z[:,i:i+1]+std*parameters[i]).float()[0]-base[i][0].to(device)
        evaluations+=1
        if not bool(torch.isfinite(result).all()):raise FloatingPointError('Nonfinite decoded edit effect')
        return result

    with torch.no_grad():effects=[effect_at(i).cpu() for i in range(frames)]

    def global_score():
        fit=preserve=temporal=0.
        for i,e in enumerate(effects):
            fit+=float(mean_square(e-target[i],weights[i])/energies[i])*(anchor_weight if i==0 else 1.)
            preserve+=float(mean_square(e,1-masks[i])/energies[i])
        for i,s in enumerate(stencils,1):
            temporal+=float(mean_square(effects[i]-_apply_stencil(effects[i-1],s),edge_weights[i-1])/energies[i])
        reg=sum(float(p.detach().square().mean()) for p in parameters)
        return dict(proposal_fit=fit/frames,keep_error=preserve/frames,
                    temporal_error=temporal/max(1,frames-1),
                    objective=(fit+keep*preserve+joint*temporal+ridge*reg)/frames)

    def update_best(score):
        nonlocal best,best_value
        if score['objective']<best_value:
            best_value=score['objective'];best=[p.detach().cpu().clone() for p in parameters]

    update_best(global_score())

    def state(sweep):
        return dict(schema='edit-video-calibration-v1',sweep=sweep,
            parameters=[p.detach().cpu().clone() for p in parameters],
            optimizers=[_cpu_tree(o.state_dict()) for o in optimizers],best=_cpu_tree(best),
            best_value=best_value,history=copy.deepcopy(history),gradient_checked=gradient_checked,
            evaluations=evaluations,elapsed=elapsed_prior+time.monotonic()-started)

    for sweep in range(start,iterations):
        active_joint=0. if sweep<init_sweeps else joint
        for i,(p,opt) in enumerate(zip(parameters,optimizers)):
            opt.zero_grad(set_to_none=True)
            effect=effect_at(i)
            if not effect.requires_grad:raise RuntimeError('Decoder detached the editable latent')
            fit=mean_square(effect-target[i].to(device),weights[i].to(device))/energies[i].to(device)
            loss=fit*(anchor_weight if i==0 else 1.)+keep*mean_square(effect,(1-masks[i]).to(device))/energies[i].to(device)+ridge*p.square().mean()
            if active_joint and i>0:
                incoming=_apply_stencil(effects[i-1],stencils[i-1]).to(device)
                loss=loss+active_joint*mean_square(effect-incoming,edge_weights[i-1].to(device))/energies[i].to(device)
            if active_joint and i+1<frames:
                outgoing=_apply_stencil(effect,stencils[i])
                loss=loss+active_joint*mean_square(outgoing-effects[i+1].to(device),edge_weights[i].to(device))/energies[i+1].to(device)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite video calibration objective')
            if not gradient_checked and float(target[i].abs().sum())>1e-8:
                g=torch.autograd.grad(fit,p,retain_graph=True,allow_unused=True)[0]
                gradient_checked=g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum())>1e-12
            loss.backward()
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):raise FloatingPointError('Invalid decoder-input gradient')
            opt.step()
            with torch.no_grad():
                p.clamp_(-trust,trust);effects[i]=effect_at(i).cpu()
        score=global_score();update_best(score)
        row=dict(sweep=sweep+1,**score,temporal_edges=len(stencils),joint_active=bool(active_joint))
        history.append(row)
        if progress and ((sweep+1)%int(cfg.get('log_every',1))==0 or sweep+1==iterations):
            progress(dict(row,DI_throughput=(evaluations-(resume or {}).get('evaluations',0))/max(1e-6,time.monotonic()-started),
                throughput_unit='decoder forward frames/s/npu including calibration backward and IO'))
        if checkpoint and ((sweep+1)%int(cfg.get('checkpoint_every',1))==0 or sweep+1==iterations):checkpoint(state(sweep+1))
    final_state=state(iterations)
    with torch.no_grad():
        for p,b in zip(parameters,best):p.copy_(b.to(device))
        result=torch.stack([effect_at(i).cpu() for i in range(frames)])
    return dict(effect=result,delta=torch.cat([std.cpu()*b for b in best],1),history=history,
        resume_state=final_state,gradient_checked=gradient_checked,evaluations=evaluations,
        elapsed=elapsed_prior+time.monotonic()-started,best_objective=best_value,
        metadata=dict(temporal_edges=len(stencils),solver='block_coordinate_shared_effect_graph',
            initialization='dense_legal_proposal_fit',iterations=iterations))
