"""Native VACE weights/interfaces with Ascend attention and checkpointed LoRA.

No dependency on the old C256 diffusion adapter. Only trainable deltas are saved.
"""
import importlib
import json
from pathlib import Path
import sys
import types
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from utils.native_wan_runtime import attention, rope_params, rope_apply, native_block_forward


def native_modules():
    root = Path(__file__).resolve().parents[1]/'Wan2.1'/'wan'
    prefix = '_vggae_edit_wan'
    for suffix in ('', '.modules', '.utils'):
        name=prefix+suffix
        if name not in sys.modules:
            module=types.ModuleType(name)
            module.__path__=[str(root.joinpath(*suffix.strip('.').split('.'))) if suffix else str(root)]
            sys.modules[name]=module
    attn=importlib.import_module(prefix+'.modules.attention')
    attn.flash_attention=attention
    model=importlib.import_module(prefix+'.modules.model')
    def cpu_rope(*a, **kw):
        with torch.device('cpu'):
            return rope_params(*a, **kw)
    model.rope_params=cpu_rope
    model.rope_apply=rope_apply
    model.WanAttentionBlock.forward=native_block_forward
    return prefix, model


def validate_assets(root):
    root=Path(root)
    cfg=json.loads((root/'config.json').read_text())
    expected=dict(model_type='vace',in_dim=16,out_dim=16,vace_in_dim=96,dim=5120,num_layers=40)
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Expected official native VACE-14B configuration: '+str(cfg))
    index=json.loads((root/'diffusion_pytorch_model.safetensors.index.json').read_text())
    files=sorted(set(index['weight_map'].values()))+['Wan2.1_VAE.pth','models_t5_umt5-xxl-enc-bf16.pth']
    for name in files:
        if Path(name).name != name or not (root/name).is_file() or not (root/name).stat().st_size:
            raise FileNotFoundError(str(root/name))
    if not (root/'google'/'umt5-xxl').is_dir():
        raise FileNotFoundError('Missing native local tokenizer google/umt5-xxl')
    from utils.file_signature import sampled_file_signature
    return dict(config=cfg, files={n:sampled_file_signature(str(root/n)) for n in files})


def load_backbone(root, device):
    """Meta init + tensor-at-a-time safetensors to avoid 8 copies of FP32 CPU weights."""
    from safetensors import safe_open
    prefix, _=native_modules()
    cls=importlib.import_module(prefix+'.modules.vace_model').VaceWanModel
    root=Path(root)
    cfg=json.loads((root/'config.json').read_text())
    with torch.device('meta'):
        model=cls(**{k:v for k,v in cfg.items() if not k.startswith('_')})
    expected=model.state_dict()
    index=json.loads((root/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    if set(expected)!=set(index):
        raise ValueError('Native model and checkpoint keys disagree')
    for shard in sorted(set(index.values())):
        with safe_open(str(root/shard),framework='pt',device='cpu') as f:
            for name in f.keys():
                if name not in expected or tuple(f.get_slice(name).get_shape()) != tuple(expected[name].shape):
                    raise ValueError('Checkpoint shape mismatch: '+name)
                fp32=(name.split('.')[0] in ('time_embedding','time_projection','head') or
                      '.norm' in name or name.endswith('modulation'))
                value=f.get_tensor(name).to(device=device,dtype=torch.float32 if fp32 else torch.bfloat16)
                model.load_state_dict({name:value},strict=False,assign=True)
        print('[VACE load] '+shard,flush=True)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError('Unmaterialized native parameters')
    model.eval().requires_grad_(False)
    return model


def load_vae(root, device):
    prefix,_=native_modules()
    cls=importlib.import_module(prefix+'.modules.vae').WanVAE
    return cls(vae_pth=str(Path(root)/'Wan2.1_VAE.pth'),device=device,dtype=torch.bfloat16)


def load_text(root, device):
    prefix,_=native_modules()
    cls=importlib.import_module(prefix+'.modules.t5').T5EncoderModel
    return cls(text_len=512,dtype=torch.bfloat16,device=device,
        checkpoint_path=str(Path(root)/'models_t5_umt5-xxl-enc-bf16.pth'),
        tokenizer_path=str(Path(root)/'google'/'umt5-xxl'))


def pack_context(vae, frames, mask):
    """Official [VAE(inactive), VAE(reactive), 8x8 packed mask] = 96 channels.

    Even the all-zero inactive pixels must go through VAE.encode: encoded zero
    RGB is not an all-zero normalized latent. White means regenerate.
    """
    if mask.shape != (1,*frames.shape[1:]):
        raise ValueError('Expected mask [1,T,H,W] aligned to source')
    t,h,w=frames.shape[1:]
    if (t-1)%4 or h%16 or w%16:
        raise ValueError('Native VACE requires 4n+1 frames and dimensions divisible by 16')
    mask=(mask>.5).float()
    inactive=vae.encode([frames*(1-mask)])[0]
    reactive=vae.encode([frames*mask])[0]
    packed=mask[0].reshape(t,h//8,8,w//8,8).permute(2,4,0,1,3).reshape(64,t,h//8,w//8)
    packed=F.interpolate(packed[None],size=inactive.shape[1:],mode='nearest-exact')[0]
    return torch.cat((inactive,reactive,packed),0)


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base=base
        self.scale=alpha/rank
        self.a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a,a=5**.5)
        base.requires_grad_(False)

    def forward(self,x):
        return self.base(x)+F.linear(F.linear(x,self.a),self.b)*self.scale


def insert_lora(model,rank=16,alpha=16):
    changed=[]
    for name,module in list(model.named_modules()):
        if (isinstance(module,nn.Linear) and ('.self_attn.' in name or '.cross_attn.' in name)
                and name.split('.')[-1] in ('q','k','v','o')):
            parent,key=name.rsplit('.',1)
            setattr(model.get_submodule(parent),key,LoRALinear(module,rank,alpha))
            changed.append(name)
    if not changed:
        raise ValueError('No native VACE attention projections found')
    return changed


def native_forward(model,x,t,vace_context,context,readout_block=None,use_checkpoint=False,context_scale=1.):
    """Same computation as upstream VaceWanModel.forward, exposing one hidden grid."""
    _, mm=native_modules()
    device=model.patch_embedding.weight.device
    if model.freqs.device!=device:
        model.freqs=model.freqs.to(device)
    embedded=[model.patch_embedding(u[None]) for u in x]
    grids=torch.tensor([list(u.shape[2:]) for u in embedded],dtype=torch.long)
    seq_lens=torch.tensor([u.shape[2]*u.shape[3]*u.shape[4] for u in embedded],dtype=torch.long)
    seq_len=int(seq_lens.max())
    hidden=torch.cat([F.pad(u.flatten(2).transpose(1,2),(0,0,0,seq_len-int(n))) for u,n in zip(embedded,seq_lens)])
    with torch.autocast(device.type,enabled=False):
        e=model.time_embedding(mm.sinusoidal_embedding_1d(model.freq_dim,t).float())
        e0=model.time_projection(e).unflatten(1,(6,model.dim))
    text=model.text_embedding(torch.stack([F.pad(u,(0,0,0,model.text_len-u.shape[0])) for u in context]))
    kw=dict(e=e0,seq_lens=seq_lens,grid_sizes=grids,freqs=model.freqs,context=text,context_lens=None)
    c=torch.cat([model.vace_patch_embedding(u[None]).flatten(2).transpose(1,2) for u in vace_context])
    hints=[]
    def call(block,*args,**kwargs):
        if use_checkpoint and torch.is_grad_enabled():
            return checkpoint(block,*args,**kwargs,use_reentrant=False)
        return block(*args,**kwargs)
    for block in model.vace_blocks:
        c,hint=call(block,c,x=hidden,**kw)
        hints.append(hint)
    readout=None
    for i,block in enumerate(model.blocks):
        hidden=call(block,hidden,hints=hints,context_scale=context_scale,**kw)
        if i==readout_block:
            readout=hidden
    value=model.head(hidden,e)
    return [u.float() for u in model.unpatchify(value,grids)],readout,grids


class EditingModel(nn.Module):
    def __init__(self,backbone,cfg,with_rae=False):
        super().__init__()
        self.backbone,self.cfg=backbone,cfg
        insert_lora(backbone,cfg['lora_rank'],cfg['lora_alpha'])
        self.readout=(nn.Sequential(nn.LayerNorm(backbone.dim),nn.Linear(backbone.dim,256))
                      if with_rae else None)
        self.last_readout_hidden = None

    def forward(self,x,t,condition,text,detach_readout=False):
        velocity,h,grids=native_forward(self.backbone,x,t,condition,text,
            self.cfg['rae_block'] if self.readout is not None else None,self.training,self.cfg['context_scale'])
        z=None
        if self.readout is not None:
            self.last_readout_hidden = h if self.training else None
            h=h.detach() if detach_readout else h
            t1,h1,w1=grids[0].tolist()
            z=self.readout(h).transpose(1,2).reshape(len(x),256,t1,h1,w1)
            if self.cfg.get('rae_alignment', 'legacy') == 'legacy':
                z=F.interpolate(z.float(),size=(self.cfg['rae_frames'],18,18),mode='trilinear',align_corners=False)
            else:
                from utils.editing_alignment import ALIGNMENT
                if self.cfg['rae_alignment'] != ALIGNMENT:
                    raise ValueError('Unknown RAE coordinate contract')
                # Keep native coordinates. Loss maps them using this sample's
                # actual pad16 rectangle and teacher letterbox, not aspect stretch.
                z=z.float()
            z=z.permute(0,2,3,4,1)
        return torch.stack(velocity),z

    def deltas(self):
        return {n:p.detach().float().cpu() for n,p in self.named_parameters() if p.requires_grad}

    def restore(self,state):
        params={n:p for n,p in self.named_parameters() if p.requires_grad}
        if set(params)!=set(state):
            raise ValueError('Adapter/readout keys differ; use the matching arm/config')
        with torch.no_grad():
            for n,p in params.items(): p.copy_(state[n])


def flow_pair(clean,noise,sigma):
    s=sigma.reshape(-1,1,1,1,1)
    return (1-s)*clean+s*noise,noise-clean


def rae_supervision(pred,target,decoder,sigma,cfg,frame,valid_rect=None,spatial=None,support=None):
    if cfg.get('rae_alignment', 'legacy') != 'legacy':
        from utils.editing_alignment import aligned_rae_loss
        if spatial is None or valid_rect is None:
            raise ValueError('Aligned RAE needs native and teacher image coordinates')
        return aligned_rae_loss(pred,target,decoder,sigma,cfg,frame,valid_rect,spatial,support)
    # Target-derived scale is detached, and the decoder weights stay frozen.
    scale=target.detach().flatten(0,-2).std(0,unbiased=False).clamp_min(.1)
    pixel_mask=None
    latent_error=((pred-target)/scale).square().mean(-1)
    if valid_rect is not None:
        top,left,height,width=valid_rect
        pixel_mask=pred.new_zeros(1,1,518,518)
        pixel_mask[:,:,top:top+height,left:left+width]=1
        latent_mask=F.interpolate(pixel_mask,size=pred.shape[2:4],mode='area')[:,0,None]
        latent=(latent_error*latent_mask).sum((1,2,3))/(latent_mask.sum()*pred.shape[1]).clamp_min(1)
    else:
        latent=latent_error.flatten(1).mean(1)
    with torch.no_grad(): truth=decoder.decode(target[:,frame:frame+1]).float()
    rgb_error=(decoder.decode(pred[:,frame:frame+1]).float()-truth).abs().mean(-1)
    if pixel_mask is not None:
        rgb_mask=F.interpolate(pixel_mask,size=rgb_error.shape[-2:],mode='area')
        rgb=(rgb_error*rgb_mask).sum((1,2,3))/rgb_mask.sum().clamp_min(1)
    else:
        rgb=rgb_error.flatten(1).mean(1)
    valid=(sigma<=cfg['rae_max_sigma']).float()
    return (valid*(cfg['rae_latent_weight']*latent+cfg['rae_rgb_weight']*rgb)).mean(),dict(
        rae_latent=float(latent.detach().mean()),rae_rgb=float(rgb.detach().mean()),rae_active=float(valid.mean()))


@torch.no_grad()
def sample_native(model,condition,text,negative,cfg,seed):
    prefix,_=native_modules()
    cls=importlib.import_module(prefix+'.utils.fm_solvers_unipc').FlowUniPCMultistepScheduler
    device=condition.device
    generator=torch.Generator(device=device).manual_seed(seed)
    x=torch.randn((16,*condition.shape[1:]),device=device,generator=generator,dtype=torch.float32)
    scheduler=cls(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False)
    scheduler.set_timesteps(cfg['sample_steps'],device=device,shift=cfg['shift'])
    backbone=model.backbone if isinstance(model,EditingModel) else model
    for i,t in enumerate(scheduler.timesteps):
        with torch.autocast(device.type,dtype=torch.bfloat16):
            c=native_forward(backbone,[x],t.reshape(1),[condition],[text],context_scale=cfg['context_scale'])[0][0]
            u=native_forward(backbone,[x],t.reshape(1),[condition],[negative],context_scale=cfg['context_scale'])[0][0]
        x=scheduler.step((u+cfg['guidance']*(c-u))[None],t,x[None],return_dict=False,generator=generator)[0][0]
        if (i+1)%5==0: print(f'[VACE sample] {i+1}/{cfg["sample_steps"]}',flush=True)
    return x


def runtime_preflight(device):
    """Real small native VACE on the selected accelerator, before large loading.

    Verifies attention backward, checkpoint recomputation and native equivalence.
    Does not claim the full model fits; the two-update phase measures that.
    """
    from utils.native_wan_runtime import compact_storage
    prefix,_=native_modules()
    cls=importlib.import_module(prefix+'.modules.vace_model').VaceWanModel
    m=cls(dim=64,ffn_dim=96,freq_dim=16,text_dim=32,num_heads=4,num_layers=2,
        in_dim=16,out_dim=16,text_len=8,vace_layers=[0],vace_in_dim=96).to(device)
    nn.init.normal_(m.head.head.weight,std=.1)
    m.eval().requires_grad_(False)
    if device.type!='cpu': compact_storage(m)
    x=[torch.randn(16,2,4,4,device=device)]; t=torch.tensor([300.],device=device)
    cond=[torch.randn(96,2,4,4,device=device)]; text=[torch.randn(5,32,device=device)]
    autocast=lambda:torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type!='cpu')
    with torch.no_grad(),autocast():
        reference=m(x,t,cond,text,seq_len=8)[0]
        actual=native_forward(m,x,t,cond,text)[0][0]
    rms=float((reference-actual).square().mean().sqrt())
    if not torch.isfinite(actual).all() or rms>.002: raise RuntimeError('Native VACE equivalence failed')
    cfg=dict(lora_rank=2,lora_alpha=2,context_scale=1.)
    wrapped=EditingModel(m,cfg).to(device).train()
    with autocast():
        prediction,_=wrapped(x,t,cond,text)
        prediction.square().mean().backward()
    trainable=[p for p in wrapped.parameters() if p.requires_grad]
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable):
        raise RuntimeError('Native attention/checkpoint/LoRA backward failed')
    grad=sum(float(p.grad.abs().sum()) for p in trainable)
    if grad<=0: raise RuntimeError('LoRA has no gradient')
    return dict(device=str(device),native_forward_rms=rms,lora_gradient_l1=grad,
                result='passed small native forward/backward; full 14B memory not measured here')
