"""Trainable edit states enter VACE's control branch at every noise level.

The RGB and RAE arms use the same source/proposal context, LoRA configuration,
and bridge depth/width. Their input representations have different channel
counts; report the actual trainable parameter counts rather than claiming
exactly equal capacity. No source-flow confidence hard gate is applied here.
"""
import importlib

import torch
from torch import nn
from torch.nn import functional as F

from utils.vace_edit_runtime import LoRALinear, insert_lora, native_forward, native_modules


def _rgb(value, shape, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(name+' has an incompatible RGB canvas or frame count')
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(name+' must be finite floating-point RGB')
    if bool((value.abs()>1.00001).any()):
        raise ValueError(name+' must be normalized to [-1,1]')


def pack_edit_context(vae, source, reference, mask, anchor):
    """Official 96 channels with source inactive and edited-reference reactive.

    source/reference: [3,T,H,W] in [-1,1], mask: [1,T,H,W] in [0,1],
    anchor: [3,1,H,W] in [-1,1]. White mask means regenerate. The legal
    single-frame anchor is image-mode encoded into a separate temporal prefix.
    The source, never the proposal, supplies unedited/inactive pixels.
    """
    if not isinstance(source,torch.Tensor) or source.ndim!=4 or source.shape[0]!=3:
        raise ValueError('Source must be [3,T,H,W]')
    _,frames,height,width=source.shape
    if frames<1 or (frames-1)%4 or height%16 or width%16:
        raise ValueError('Native video requires 4n+1 frames and pad16 spatial dimensions')
    _rgb(source,source.shape,'source')
    _rgb(reference,source.shape,'reference')
    _rgb(anchor,(3,1,height,width),'anchor')
    if (not isinstance(mask,torch.Tensor) or mask.shape!=(1,frames,height,width) or
            not bool(torch.isfinite(mask).all()) or bool(((mask<0)|(mask>1)).any())):
        raise ValueError('Mask must be finite [1,T,H,W] values in [0,1]')
    if any(v.device!=source.device for v in (reference,mask,anchor)):
        raise ValueError('Source, reference, mask and anchor must share a device')
    binary=(mask>.5).float()
    encoded=vae.encode([source*(1-binary),reference*binary,anchor.to(source)])
    if len(encoded)!=3:
        raise ValueError('VAE must return inactive, reactive and anchor encodings')
    inactive,reactive,prefix=encoded
    expected=(16,(frames-1)//4+1,height//8,width//8)
    if inactive.shape!=expected or reactive.shape!=expected or prefix.shape!=(16,1,height//8,width//8):
        raise ValueError('VAE encodings disagree with the native causal/prefix grid')
    packed=binary[0].reshape(frames,height//8,8,width//8,8).permute(2,4,0,1,3).reshape(
        64,frames,height//8,width//8)
    packed=F.interpolate(packed[None],size=inactive.shape[1:],mode='nearest-exact')[0]
    context=torch.cat((inactive,reactive,packed.to(inactive)),0)
    first=torch.cat((prefix,torch.zeros_like(prefix),prefix.new_zeros(64,1,*prefix.shape[-2:])),0)
    return torch.cat((first.to(context),context),dim=1)


@torch.no_grad()
def refresh_reactive_context(vae, condition, reference, mask):
    """Replace only the video reactive encoding from the live edited reference.

    Preserve the independent anchor prefix, inactive source encoding and mask
    channels bit-for-bit. This is the same masked RGB encoding as
    pack_edit_context, not an unmasked clean-latent substitution. Return a
    candidate: the caller commits it together with the state/velocity update.
    """
    if reference.ndim != 4 or reference.shape[0] != 3:
        raise ValueError('Live reference must be [3,T,H,W]')
    _, frames, height, width = reference.shape
    if frames < 1 or (frames-1) % 4 or height % 16 or width % 16:
        raise ValueError('Live reference requires native 4n+1 times and pad16')
    _rgb(reference, reference.shape, 'live reference')
    expected = (96, 1+(frames-1)//4+1, height//8, width//8)
    if tuple(condition.shape) != expected or not bool(torch.isfinite(condition).all()):
        raise ValueError('Live context has an incompatible native/prefix lattice')
    if (mask.shape != (1, frames, height, width) or
            not bool(torch.isfinite(mask).all()) or bool(((mask < 0) | (mask > 1)).any())):
        raise ValueError('Live context needs finite source support in [0,1]')
    if condition.device != reference.device or mask.device != reference.device:
        raise ValueError('Live context, reference and mask must share a device')
    encoded = vae.encode([reference*(mask > .5).to(reference)])
    if len(encoded) != 1 or encoded[0].shape != condition[16:32, 1:].shape:
        raise ValueError('Reactive encoding changed the native video lattice')
    reactive = encoded[0].to(condition)
    if not bool(torch.isfinite(reactive).all()):
        raise FloatingPointError('Nonfinite reactive context in storage dtype')
    result = condition.clone()
    result[16:32, 1:].copy_(reactive)
    return result


def prepend_reference_clean(video_clean,condition,reference_count=1):
    """Build the training clean latent; future video target stays supervision.

    Reference slots are taken from the conditioning anchor encoding, not from
    target video frames. The runner must exclude these prefix slots from its
    video denoising objective and remove them BEFORE causal VAE RGB decoding.
    """
    if type(reference_count) is not int or reference_count<1:
        raise ValueError('At least one reference prefix is required')
    if (video_clean.ndim!=4 or video_clean.shape[0]!=16 or condition.ndim!=4 or
            condition.shape[0]!=96 or condition.shape[1]!=video_clean.shape[1]+reference_count or
            condition.shape[-2:]!=video_clean.shape[-2:]):
        raise ValueError('Video target and reference-conditioned context grids disagree')
    return torch.cat((condition[:16,:reference_count].to(video_clean),video_clean),dim=1)


class StateConditionedVACE(nn.Module):
    """Frozen native backbone with LoRA and a zero-start control-state bridge."""
    def __init__(self,backbone,cfg,state_channels):
        super().__init__()
        if any(isinstance(m,LoRALinear) for m in backbone.modules()):
            raise ValueError('Use a fresh native backbone; do not insert LoRA twice')
        self.backbone=backbone
        self.cfg=dict(cfg)
        self.state_channels=int(state_channels)
        width=int(cfg.get('bridge_width',128))
        if self.state_channels<1 or width<1:
            raise ValueError('Positive state channels and bridge width required')
        backbone.requires_grad_(False)
        self.lora_modules=insert_lora(backbone,int(cfg.get('lora_rank',16)),float(cfg.get('lora_alpha',16)))
        self.state_norm=nn.LayerNorm(self.state_channels).to(
            device=backbone.patch_embedding.weight.device,dtype=torch.float32)
        self.bridge=nn.Sequential(nn.Conv3d(self.state_channels,width,1),nn.SiLU(),
                                  nn.Conv3d(width,backbone.dim,1)).to(
                                      device=backbone.patch_embedding.weight.device,dtype=torch.float32)
        nn.init.zeros_(self.bridge[-1].weight)
        nn.init.zeros_(self.bridge[-1].bias)

    def forward(self,x,t,conditions,text,states,use_state=True):
        if not len(x) or len(x)!=len(conditions) or len(x)!=len(text):
            raise ValueError('Matching nonempty video, condition and text batches required')
        tokens=None
        if use_state:
            # Cached RAE states are FP16. Under CPU BF16 autocast, stack cannot
            # promote an FP16 list safely; convert each item before stacking.
            state_options=dict(device=x[0].device,dtype=self.bridge[0].weight.dtype)
            state=(states.to(**state_options) if isinstance(states,torch.Tensor) else
                   torch.stack([value.to(**state_options) for value in states]))
            if state.ndim!=5 or state.shape[:2]!=(len(x),self.state_channels):
                raise ValueError('State must be [B,C,T,Hpatch,Wpatch]')
            patch=tuple(self.backbone.patch_size)
            expected=tuple(x[0].shape[1+i]//patch[i] for i in range(3))
            if tuple(state.shape[2:])!=expected or any(tuple(v.shape)!=tuple(x[0].shape) for v in x):
                raise ValueError('Offline state/prefix grid differs from native patch coordinates')
            # RAE and raw RGB have different physical scales. Normalize each
            # token's feature axis identically in both arms, without hard clips.
            state=self.state_norm(state.permute(0,2,3,4,1)).permute(0,4,1,2,3)
            tokens=self.bridge(state).flatten(2).transpose(1,2)
        velocity,_,_=native_forward(self.backbone,x,t,conditions,text,
            use_checkpoint=self.training,context_scale=self.cfg.get('context_scale',1.),state_tokens=tokens)
        return torch.stack(velocity)

    def deltas(self):
        return {name:p.detach().float().cpu().clone() for name,p in self.named_parameters() if p.requires_grad}

    def restore(self,state):
        parameters={name:p for name,p in self.named_parameters() if p.requires_grad}
        if set(state)!=set(parameters):
            raise ValueError('State bridge/LoRA keys differ from this representation or configuration')
        with torch.no_grad():
            for name,p in parameters.items():
                if state[name].shape!=p.shape:
                    raise ValueError('State bridge/LoRA parameter shape changed: '+name)
                p.copy_(state[name])


@torch.no_grad()
def sample_state_native(model,condition,text,negative,state,cfg,seed,use_state=True,
                        velocity_callback=None):
    """Real native UniPC sampling through the trained wrapper in BOTH CFG arms.

    Conditioning retains the legal reference prefix. Return all latent slots;
    call decode_native(...,reference_count=1) to decode only video slots.
    An optional source-only feedback callback may correct velocity and update
    the mutable state/context for the next pair of CFG forwards. A context
    bound to the callback is a per-call copy. No target is consumed.
    """
    prefix,_=native_modules()
    cls=importlib.import_module(prefix+'.utils.fm_solvers_unipc').FlowUniPCMultistepScheduler
    device=condition.device
    generator=torch.Generator(device=device).manual_seed(seed)
    x=torch.randn((16,*condition.shape[1:]),device=device,generator=generator,dtype=torch.float32)
    scheduler=cls(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False)
    scheduler.set_timesteps(cfg['sample_steps'],device=device,shift=cfg['shift'])
    if velocity_callback is not None and hasattr(velocity_callback,'bind_condition'):
        # Own the mutable context for this one sampling trajectory; never
        # corrupt a cached caller tensor or a later same-seed comparison.
        condition=condition.clone()
        velocity_callback.bind_condition(condition)
    if velocity_callback is not None and hasattr(velocity_callback,'set_schedule'):
        velocity_callback.set_schedule(scheduler.sigmas[:len(scheduler.timesteps)])
    training=model.training
    model.eval()
    try:
        for i,t in enumerate(scheduler.timesteps):
            with torch.autocast(device.type,dtype=torch.bfloat16):
                c=model([x],t.reshape(1),[condition],[text],[state],use_state=use_state)[0]
                u=model([x],t.reshape(1),[condition],[negative],[state],use_state=use_state)[0]
            velocity=u+cfg['guidance']*(c-u)
            if not bool(torch.isfinite(velocity).all()):
                raise FloatingPointError('Nonfinite state-conditioned native velocity')
            if velocity_callback is not None:
                corrected=velocity_callback(step_index=i,sigma=float(scheduler.sigmas[i]),
                    latent=x,velocity=velocity)
                if corrected.shape!=velocity.shape or not bool(torch.isfinite(corrected).all()):
                    raise FloatingPointError('Invalid state feedback velocity')
                velocity=corrected
                if (hasattr(velocity_callback,'consume_history_reset') and
                        velocity_callback.consume_history_reset()):
                    # The bridge/native condition changed. Start a first-order step
                    # without reusing outputs under the previous condition;
                    # preserve the actual current timestep and sigma schedule.
                    scheduler.model_outputs=[None]*scheduler.config.solver_order
                    scheduler.timestep_list=[None]*scheduler.config.solver_order
                    scheduler.lower_order_nums=0
                    scheduler.last_sample=None
            x=scheduler.step(velocity[None],t,x[None],return_dict=False,generator=generator)[0][0]
            if (i+1)%5==0:print(f'[VACE state sample] {i+1}/{cfg["sample_steps"]}',flush=True)
    finally:
        model.train(training)
    return x


def state_runtime_preflight(device,cfg):
    """Exercise the new bridge on a real tiny native VACE before large loading.

    Validates both representation channel counts, checkpointed flow backward,
    unchanged frozen parameters, and a nonzero post-update state influence.
    This checks device operations, not full14B memory or natural-video quality.
    """
    from utils.native_wan_runtime import compact_storage
    device=torch.device(device)
    prefix,_=native_modules()
    cls=importlib.import_module(prefix+'.modules.vace_model').VaceWanModel
    records=[]
    for channels in (514,1538):
        native=cls(dim=64,ffn_dim=96,freq_dim=16,text_dim=32,num_heads=4,num_layers=2,
            in_dim=16,out_dim=16,text_len=8,vace_layers=[0],vace_in_dim=96).to(device)
        nn.init.normal_(native.head.head.weight,std=.1)
        # A released VACE checkpoint has trained control projections, whereas
        # the constructor deliberately starts them at zero.
        for block in native.vace_blocks:
            for name in ('before_proj','after_proj'):
                if hasattr(block,name):nn.init.normal_(getattr(block,name).weight,std=.1)
        if device.type!='cpu':compact_storage(native)
        native.eval().requires_grad_(False)
        x=[torch.randn(16,3,4,4,device=device)]
        time=torch.tensor([400.],device=device)
        condition=[torch.randn(96,3,4,4,device=device)]
        text=[torch.randn(5,32,device=device)]
        state=[torch.randn(channels,3,2,2,device=device)]
        amp=lambda:torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type!='cpu')
        with torch.no_grad(),amp():
            expected=native_forward(native,x,time,condition,text)[0][0]
        settings=dict(cfg,lora_rank=2,lora_alpha=2,bridge_width=8)
        model=StateConditionedVACE(native,settings,channels).train()
        frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        bridge_gradient=0.
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            with amp():
                value=model(x,time,condition,text,state)
                if step==0:
                    rms=float((value[0].float()-expected.float()).square().mean().sqrt())
                    if rms>.002 or not bool(torch.isfinite(value).all()):
                        raise RuntimeError('Zero bridge does not preserve the native forward')
                loss=(value.float()-.25).square().mean()
            loss.backward()
            bridge_gradient=float(model.bridge[-1].weight.grad.float().abs().sum())
            if not bridge_gradient>0 or not all(p.grad is None for n,p in model.named_parameters() if n in frozen):
                raise RuntimeError('State bridge/frozen native gradient contract failed')
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step()
        model.eval()
        alternate=[state[0].clone()]
        alternate[0][0]+=3
        with torch.no_grad(),amp():
            normal=model(x,time,condition,text,state)
            changed=model(x,time,condition,text,alternate)
            effect=float((normal.float()-changed.float()).abs().max())
        if not effect>0 or not bool(torch.isfinite(changed).all()):
            raise RuntimeError('Trained bridge state does not influence native velocity')
        if any(not torch.equal(p.detach(),frozen[n]) for n,p in model.named_parameters() if n in frozen):
            raise RuntimeError('Frozen native weights changed during bridge training')
        records.append(dict(state_channels=channels,zero_bridge_rms=rms,
            bridge_gradient_l1=bridge_gradient,trained_state_velocity_max_delta=effect,
            reference_slots=1,video_slots=2,passed=True))
        del model,native,optimizer
    return dict(passed=True,device=str(device),representations=records,
                limitation='Tiny device runtime only; full14B memory/quality remain unverified')
