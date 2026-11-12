"""Final-output calibration interventions and bridge-free operator controls.

No edited future target is accepted here. The magnitude control uses another
source/reference-only arm, and is explicitly a diagnostic, not a deployable gain.
"""
import copy
import time
import torch
from torch.nn import functional as F

CALIBRATION_ARMS = ('cal_on_online_on', 'cal_on_online_off',
    'cal_off_online_on', 'cal_off_online_off', 'cal_off_magnitude',
    'cal_off_fixed_bridge_on', 'cal_off_fixed_bridge_off',
    'cal_on_fixed_bridge_on', 'cal_on_fixed_bridge_off')
OPERATOR_ARMS = ('op_sft', 'op_rgb', 'op_rgb_smooth', 'op_vae', 'op_rae')
ARMS = CALIBRATION_ARMS + OPERATOR_ARMS


def arm_spec(name):
    if name not in ARMS:
        raise ValueError('Unknown attribution arm: '+name)
    operator = name in OPERATOR_ARMS
    return dict(operator=operator, family='sft' if operator else 'rae',
        feedback=name != 'op_sft', calibrated=name.startswith('cal_on'),
        online=name in ('cal_on_online_on', 'cal_off_online_on', 'cal_off_fixed_bridge_on', 'cal_on_fixed_bridge_on'),
        fixed_bridge='fixed_bridge' in name, magnitude=name == 'cal_off_magnitude')


def supported_rms(value, mask):
    value, mask = value.float(), mask.to(value).float()
    return ((value.square()*mask).sum()/(mask.sum()*value.shape[1]).clamp_min(1)).sqrt()


def smooth_effect(effect, sigma):
    """Predeclared spatial Gaussian control; no time smoothing or target tuning."""
    if sigma <= 0:
        raise ValueError('Positive smoothing sigma required')
    radius = max(1, int(3*sigma+.5))
    axis = torch.arange(-radius, radius+1, device=effect.device).float()
    kernel = torch.exp(-axis.square()/(2*sigma*sigma)); kernel /= kernel.sum()
    kernel = (kernel[:, None]*kernel[None, :])[None, None].expand(effect.shape[1], 1, -1, -1)
    return F.conv2d(F.pad(effect, (radius,)*4, mode='replicate'), kernel, groups=effect.shape[1])


class MagnitudeProjector:
    """Match pre-clipping correction RMS to the calibrated reuse arm.

    Both projectors see the SAME current prediction. All observed data are from
    source and shared proposal. Clipping can change the achieved dose; log both.
    This changes feedback only; the bridge retains uncalibrated states.
    """
    def __init__(self, base, calibrated, mask):
        self.base, self.calibrated, self.mask = base, calibrated, mask.cpu()

    def state_dict(self):
        return dict(base=self.base.state_dict(), calibrated=self.calibrated.state_dict())

    def load_state_dict(self, value):
        self.base.load_state_dict(value['base']); self.calibrated.load_state_dict(value['calibrated'])

    def project(self, current, progress=None):
        result = self.base.project(current, progress)
        target = self.calibrated.project(current, progress)
        # This diagnostic performs an additional decoder readout. Retain its
        # diagnostics rather than charging only the uncalibrated branch.
        result['diagnostics']['magnitude_reference_readout'] = copy.deepcopy(target['diagnostics'])
        if result['diagnostics'].get('fallback') or target['diagnostics'].get('fallback'):
            raise RuntimeError('Magnitude diagnostic requires two valid readouts')
        correction = result['reference']-current
        # The hard anchor is identical for all arms and is not scaled.
        mask = self.mask.clone(); mask[0] = 0
        desired = supported_rms(target['reference']-current, mask)
        before = supported_rms(correction, mask)
        factor = float(desired/before.clamp_min(1e-12)) if float(before)>1e-12 else 1.
        adjusted = (current+correction*factor).clamp(0, 1)
        adjusted[0] = result['reference'][0]
        result['reference'] = adjusted
        result['diagnostics']['magnitude_control'] = dict(scale=factor,
            before=float(before), requested=float(desired), achieved=float(supported_rms(adjusted-current, mask)),
            clipped_fraction=float(((current+correction*factor)<0).logical_or((current+correction*factor)>1).float().mean()),
            target_access=False, reference='calibrated_reuse_at_same_current_prediction')
        return result


def fit_video_vae(vae, source, proposal, valid, cfg, device):
    """Full video encode/decode; no image-mode or framewise VAE approximation.

    Run BEFORE loading the 14B backbone. The entire causal decoder is activation
    checkpointed, preserving its temporal graph and native normalization.
    """
    from torch.utils.checkpoint import checkpoint
    started = time.monotonic()
    source, proposal = source.to(device).float(), proposal.to(device).float()
    if source.shape != proposal.shape or source.ndim != 4 or (len(source)-1)%4:
        raise ValueError('Full matching 4n+1 video required')
    def video(x): return (x*2-1).permute(1,0,2,3)
    def decode(z):
        return (vae.decode([z])[0].permute(1,0,2,3).float()+1)/2
    with torch.no_grad():
        z, encoded = vae.encode([video(source), video(proposal)])
        base = decode(z)
    if base.shape != source.shape or z.shape[1] != 1+(len(source)-1)//4:
        raise ValueError('Native VAE time lattice changed')
    scale = z.float().flatten(1).std(1,unbiased=False).clamp_min(.05)[:,None,None,None]
    q0 = ((encoded-z)/scale).detach(); q = torch.nn.Parameter(q0.clone())
    optimizer = torch.optim.Adam([q], lr=float(cfg['lr']))
    weight=valid.to(device).float(); wanted=proposal-source
    def loss(qvalue, grad=False):
        value = checkpoint(decode, z+scale*qvalue, use_reentrant=False) if grad else decode(z+scale*qvalue)
        effect=value-base
        fit=((effect-wanted).square()*weight).sum()/(3*weight.sum()).clamp_min(1)
        return fit,effect
    with torch.no_grad(): initial,_=loss(q); best=float(initial); bestq=q.detach().clone()
    if not torch.isfinite(initial):raise FloatingPointError('Nonfinite initial VAE response')
    gradients=[]
    for step in range(int(cfg['steps'])):
        with torch.enable_grad():
            objective,_=loss(q,True)
            objective=objective+float(cfg.get('latent_weight',1e-4))*(q-q0).square().mean()
            gradient=torch.autograd.grad(objective,q)[0]
        if not torch.isfinite(gradient).all(): raise FloatingPointError('VAE state gradient is nonfinite')
        gradients.append(float(gradient.norm())); q.grad=gradient
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            q.copy_(q0+(q-q0).clamp(-float(cfg['trust']),float(cfg['trust'])))
            score,_=loss(q)
            if not torch.isfinite(score): raise FloatingPointError('VAE calibration is nonfinite')
            if float(score)<best: best=float(score); bestq=q.detach().clone()
        print(f'[full-video VAE calibration] step={step+1} mse={float(score):.7g}',flush=True)
    with torch.no_grad(): _,effect=loss(bestq)
    return effect.cpu(), dict(before=float(initial), after=best, gradient_norms=gradients,
        latent_shape=list(z.shape), frames=len(source), native_temporal_compression=4,
        reconstruction_mse=float((base-source).square().mean()),
        elapsed_seconds=time.monotonic()-started, optimized_weights=False)


class NativeEffectFeedback:
    """Same native encoding-difference writeback, no learned state bridge.

    All operator arms share this gate, schedule and hard anchor. Their sole
    difference is the cached source-conditioned visible effect. No online solve
    is performed in this initial operator comparison.
    """
    def __init__(self, vae, reference, effect, cfg, observer=None):
        self.vae,self.ref,self.effect,self.cfg=vae,reference,effect.cpu().float(),cfg
        self.observer=observer; self.selected_steps=set(); self.events=[]; self.applied=set()
        self.last_reference=None; self.last_projection=None; self.state=None; self.reset=False
        if self.effect.shape != reference['source'].shape: raise ValueError('Native effect lattice mismatch')

    def bind_condition(self, condition): pass
    def set_schedule(self, sigmas):
        values=torch.as_tensor(sigmas).float().tolist()
        usable=[i for i,s in enumerate(values) if 0<s<=self.cfg['writeback_max_sigma']]
        self.selected_steps={min(usable,key=lambda i:abs(values[i]-s)) for s in self.cfg['writeback_sigmas']} if usable else set()
        self.applied=set(); self.events=[]; self.reset=False

    def consume_history_reset(self):
        value,self.reset=self.reset,False
        return value

    @torch.no_grad()
    def __call__(self, *, step_index, sigma, latent, velocity):
        from utils.vace_edit_runtime import decode_native
        if step_index not in self.selected_steps or step_index in self.applied:return velocity
        self.applied.add(step_index); started=time.monotonic()
        event=dict(step_index=int(step_index),sigma=float(sigma),applied=False,fallback=False)
        corrected=velocity
        try:
            clean=latent.float()-float(sigma)*velocity.float()
            decoded=decode_native(self.vae,clean,1).float()
            current=((decoded.permute(1,0,2,3).cpu()+1)/2).clamp(0,1)
            if current.shape != self.effect.shape:raise ValueError('Native video shape mismatch')
            gate=self.ref['masks'].float()*self.ref['confidence'].float()
            proposal=(current+gate*(self.ref['source']+self.effect-current)).clamp(0,1)
            y,x,h,w=self.ref['spatial']['valid_rect']
            proposal[0,:,y:y+h,x:x+w]=self.ref['anchor'][:,y:y+h,x:x+w]
            encoded=self.vae.encode([(proposal*2-1).permute(1,0,2,3).to(decoded),decoded])
            change=self.cfg['writeback_alpha']*(encoded[0].float()-encoded[1].float())
            if change.shape!=clean[:,1:].shape:raise ValueError('Native latent lattice mismatch')
            rms=change.square().mean((-2,-1),keepdim=True).sqrt()
            cap=self.cfg['writeback_max_clean_rms']
            change*= (cap/rms.clamp_min(1e-12)).clamp_max(1)
            corrected=velocity.float().clone();corrected[:,1:]-=change/float(sigma)
            if not torch.isfinite(corrected).all():raise FloatingPointError('Nonfinite operator writeback')
            self.last_reference=dict(self.ref,reference=proposal)
            self.last_projection=dict(raw_effect=self.effect,gate=gate)
            self.reset=True
            event.update(applied=True,clean_delta_rms=float(change.square().mean().sqrt()),
                pixel_correction_rms=float(supported_rms(proposal-current,gate)),
                capped_fraction=float((rms>cap).float().mean()),reference_prefix_preserved=True)
        except Exception as exc:
            corrected=velocity;event.update(fallback=True,error=repr(exc))
        elapsed=time.monotonic()-started
        event.update(elapsed_seconds=elapsed,DI_throughput=len(self.effect)/max(elapsed,1e-6) if event['applied'] else 0,
            throughput_unit='feedback video frames/s/rank')
        self.events.append(event)
        if self.observer:self.observer(event)
        return corrected
