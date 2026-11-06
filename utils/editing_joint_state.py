"""Joint, source-conditioned edit-state correction for actual sampler feedback.

Only a source video, a legal anchor/reference, source correspondences and the
current generator prediction enter this module. There is no future target API.
The RGB control solves the same visible-response problem on pixel variables.
RAE correction retains one frame's decoder graph at a time and differentiates
only its latent input. Unknown surfaces remain generator proposals; source-only
graph paths can recover edit evidence where the dense proposal is unreliable.
"""
import copy
import time

import torch

from utils.editing_video_transport import _apply_stencil, _stencil, propagate_reference


def build_source_edges(source, cfg=None):
    """Return directed adjacent/anchor edges, each with its OWN source confidence.

    An edge ``src -> dst`` contains a backward pull flow on the dst canvas:
    dst[p] corresponds to src[p + flow[p]]. Confidence is forward/backward and
    source-photometric reliability, not confidence propagated along an edit.
    Long-range flow remains an approximation, not semantic reidentification.
    """
    cfg = dict(cfg or {})
    source = torch.as_tensor(source).detach().float().cpu()
    if source.ndim != 4 or source.shape[1] != 3 or len(source) < 1:
        raise ValueError('source must be [T,3,H,W]')
    if not bool(torch.isfinite(source).all()):
        raise ValueError('Nonfinite source')
    count, _, height, width = source.shape
    pairs = set()
    for i in range(count-1):
        pairs.update(((i, i+1), (i+1, i)))
    if cfg.get('graph_anchor_edges', True):
        for i in range(1, count):
            pairs.update(((0, i), (i, 0)))
    options = dict(cfg, flow_direct_anchor=False, flow_min_confidence=0.)
    support = torch.ones(1, height, width)
    edges = []
    for src, dst in sorted(pairs):
        pair = source[[src, dst]]
        value = propagate_reference(pair, pair[0], support, options)
        edges.append(dict(src=src, dst=dst,
            flow=value['flows_backward'][0].contiguous(),
            confidence=value['confidence'][1].contiguous(),
            kind='adjacent' if abs(src-dst) == 1 else 'anchor',
            provenance='source_rgb_forward_backward_and_photometric_only'))
    return edges


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite '+name)


def _weighted_mse(value, weight):
    # Confidence is a precision, not just a sampling distribution: uniformly
    # reducing it must reduce the constraint, rather than cancel in its divisor.
    support = (weight > 0).to(weight.dtype).sum()
    return (value.square()*weight).sum()/(value.shape[-3]*support).clamp_min(1.)


class JointEditProjector:
    """Warm-started nonlinear response correction on a shared source graph.

    ``source/proposal/current_rgb`` are [T,3,H,W] in [0,1], at the common RAE
    canvas/times. ``source_z`` is [T,G,G,C] or [1,T,G,G,C]. ``decode`` accepts
    [1,1,G,G,C] and returns [1,3,H,W]. Set decode/source_z to None for RGB.
    ``initial_delta`` has the source_z shape; ``scale`` is a positive channel
    scale from TRAINING statistics (otherwise source-only scale is reported).

    Returned reference/effect are for sampler feedback. raw_effect is the
    solver's decoded response before reliability mixing and hard anchor. Delta
    is [T,G,G,C], suitable for alignment into the state bridge, or None for RGB.
    No model parameter is updated. The caller owns temporal/native-grid mapping.
    """
    def __init__(self, decode, source_z, source, anchor, masks, confidence,
                 flows_backward=None, cfg=None, initial_delta=None,
                 initial_effect=None, valid=None, proposal=None, edges=None,
                 scale=None):
        self.cfg = dict(cfg or {})
        self.decode = decode
        self.mode = 'rae' if decode is not None else 'rgb'
        self.source = torch.as_tensor(source).detach().float().cpu()
        self.anchor = torch.as_tensor(anchor).detach().float().cpu()
        if self.source.ndim != 4 or self.source.shape[1] != 3 or len(self.source) < 1:
            raise ValueError('source must be [T,3,H,W]')
        self.frames, _, self.height, self.width = self.source.shape
        if self.anchor.shape == (1, *self.source.shape[1:]):
            self.anchor = self.anchor[0]
        if self.anchor.shape != self.source.shape[1:]:
            raise ValueError('anchor must match one RGB source frame')
        self.masks = torch.as_tensor(masks).detach().float().cpu()
        self.confidence = torch.as_tensor(confidence).detach().float().cpu()
        shape = (self.frames, 1, self.height, self.width)
        if self.masks.shape != shape or self.confidence.shape != shape:
            raise ValueError('masks/confidence must be [T,1,H,W]')
        if valid is None:
            self.valid = torch.ones_like(self.masks)
        else:
            self.valid = torch.as_tensor(valid).detach().float().cpu()
            if self.valid.shape == shape[1:]:
                self.valid = self.valid[None]
            if self.valid.shape == (1, *shape[1:]):
                self.valid = self.valid.expand(shape).clone()
            if self.valid.shape != shape:
                raise ValueError('valid must be [T,1,H,W] or [1,1,H,W]')
        for name in ('source', 'anchor', 'masks', 'confidence', 'valid'):
            value = getattr(self, name)
            _finite(value, name)
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(name+' must be in [0,1]')
        self.masks *= self.valid
        self.confidence *= self.valid
        if proposal is None:
            proposal = self.source if initial_effect is None else self.source+torch.as_tensor(initial_effect)
        self.proposal = torch.as_tensor(proposal).detach().float().cpu()
        if self.proposal.shape != self.source.shape:
            raise ValueError('proposal must match source')
        _finite(self.proposal, 'proposal')
        self.evidence = (self.proposal-self.source)*self.valid
        self.evidence[0] = (self.anchor-self.source[0])*self.valid[0]

        self.steps = int(self.cfg.get('iterations',
            int(self.cfg.get('sweeps', 2))*int(self.cfg.get('steps', 2))))
        self.lr = float(self.cfg.get('lr', .05))
        self.trust = float(self.cfg.get('trust', .5))
        self.rgb_trust = float(self.cfg.get('rgb_trust', .25))
        self.native_weight = float(self.cfg.get('native_weight', .5))
        self.observation_weight = float(self.cfg.get('observation_weight', 2.))
        self.anchor_weight = float(self.cfg.get('anchor_weight', 8.))
        self.joint_weight = float(self.cfg.get('joint_weight', self.cfg.get('temporal_weight', 1.)))
        self.keep_weight = float(self.cfg.get('keep_weight', .5))
        self.ridge = float(self.cfg.get('latent_weight', self.cfg.get('ridge', 1e-4)))
        self.reachable_mix = float(self.cfg.get('reachable_mix', .65))
        self.reach_min = float(self.cfg.get('reach_min', .1))
        self.protect_mix = float(self.cfg.get('protect_mix', 0.))
        self.hard_anchor = bool(self.cfg.get('hard_anchor', True))
        if (self.steps < 0 or self.lr <= 0 or min(self.trust, self.rgb_trust) <= 0 or
                min(self.native_weight, self.observation_weight, self.joint_weight,
                    self.keep_weight, self.ridge) < 0 or self.anchor_weight <= 0 or
                not 0 <= self.reachable_mix <= 1 or not 0 <= self.reach_min <= 1 or
                not 0 <= self.protect_mix <= 1):
            raise ValueError('Invalid joint correction configuration')
        self.edges = self._edges(edges, flows_backward)
        if self.joint_weight and self.frames > 1 and not self.edges:
            raise ValueError('Joint correction requires source edges; use joint_weight=0 explicitly')
        self.reachable = self._reachability()
        # Anchor paths can supply evidence outside a damaged proposal support.
        self.support = torch.maximum(self.masks, self.reachable)*self.valid
        self.observed = self.confidence*self.masks
        self.mix = torch.maximum(self.observed, self.reachable*self.reachable_mix)*self.valid
        self.mix[0] = torch.maximum(self.mix[0], self.masks[0])
        for edge in self.edges:
            a, b, stencil = edge['src'], edge['dst'], edge['stencil']
            edge['weight'] = (edge['confidence']*self.support[b]*
                _apply_stencil(self.support[a], stencil)*self.valid[b]*
                _apply_stencil(self.valid[a], stencil)).clamp(0, 1)

        self.device = torch.device(self.cfg.get('device', 'cpu'))
        self.z = None
        self.scale_provenance = 'pixel_units'
        if self.mode == 'rae':
            if source_z is None:
                raise ValueError('RAE mode requires source_z')
            self.z = torch.as_tensor(source_z).detach().float()
            if self.z.ndim == 5 and self.z.shape[0] == 1:
                self.z = self.z[0]
            if self.z.ndim != 4 or len(self.z) != self.frames:
                raise ValueError('source_z must be [T,G,G,C] or [1,T,G,G,C]')
            _finite(self.z, 'source_z')
            self.device = self.z.device
            channels = self.z.shape[-1]
            if scale is None:
                self.scale = self.z.flatten(0, 2).std(0, unbiased=False).clamp_min(.05)
                self.scale_provenance = 'source_window_std_fallback_not_training_stats'
            else:
                self.scale = torch.as_tensor(scale).detach().float().to(self.device).reshape(-1)
                if self.scale.numel() != channels:
                    raise ValueError('scale must contain one positive value per RAE channel')
                self.scale_provenance = 'caller_supplied_training_scale'
            _finite(self.scale, 'scale')
            if bool((self.scale <= 0).any()):
                raise ValueError('scale must be positive')
            self.scale = self.scale.reshape(1, 1, 1, channels)
            initial = torch.zeros_like(self.z) if initial_delta is None else torch.as_tensor(initial_delta).detach().float().to(self.device)
            if initial.ndim == 5 and initial.shape[0] == 1:
                initial = initial[0]
            if initial.shape != self.z.shape:
                raise ValueError('initial_delta must match source_z')
            _finite(initial, 'initial_delta')
            self.initial = (initial/self.scale).detach()
        else:
            if source_z is not None:
                raise ValueError('RGB mode does not accept latent states')
            if initial_delta is not None:
                raise ValueError('RGB mode uses initial_effect, not initial_delta')
            initial = self.evidence if initial_effect is None else torch.as_tensor(initial_effect).detach().float().cpu()
            if initial.shape != self.source.shape:
                raise ValueError('initial_effect must match source')
            _finite(initial, 'initial_effect')
            self.initial = initial.to(self.device)
            self.scale = None
        self.parameters = [torch.nn.Parameter(v[None].clone()) for v in self.initial]
        # Optimizer moments are kept across sampler feedback calls. They are
        # numerical solver state, not learned model/decoder parameters.
        self.optimizers = [torch.optim.Adam([p], lr=self.lr) for p in self.parameters]
        self.base = []
        self.decoder_evaluations = 0
        self.response_evaluations = 0
        if self.mode == 'rae':
            with torch.no_grad():
                for i in range(self.frames):
                    image = self.decode(self.z[i:i+1][None]).float()
                    self.decoder_evaluations += 1
                    if image.shape != (1, 3, self.height, self.width):
                        raise ValueError('Decoder output differs from common RGB canvas')
                    _finite(image, 'decoded source')
                    self.base.append(image[0].detach().cpu())
        self.calls = 0

    def _edges(self, edges, flows_backward):
        if edges is not None and flows_backward is not None:
            raise ValueError('Supply explicit edges or adjacent backward flows, not both')
        if edges is None:
            edges = []
            if flows_backward is not None:
                flows = torch.as_tensor(flows_backward).detach().float().cpu()
                if flows.shape != (self.frames-1, self.height, self.width, 2):
                    raise ValueError('flows_backward must connect consecutive source states')
                for i, flow in enumerate(flows, 1):
                    edges.append(dict(src=i-1, dst=i, flow=flow,
                        confidence=self.confidence[i],
                        provenance='caller_adjacent_flow_with_supplied_confidence'))
        result = []
        for edge in edges:
            a, b = int(edge['src']), int(edge['dst'])
            if a == b or min(a, b) < 0 or max(a, b) >= self.frames:
                raise ValueError('Invalid source edge endpoints')
            flow = torch.as_tensor(edge['flow']).detach().float().cpu()
            weight = torch.as_tensor(edge['confidence']).detach().float().cpu()
            if weight.shape == (self.height, self.width):
                weight = weight[None]
            if flow.shape != (self.height, self.width, 2) or weight.shape != (1, self.height, self.width):
                raise ValueError('Source edge canvas differs')
            _finite(flow, 'flow'); _finite(weight, 'edge confidence')
            if bool(((weight < 0) | (weight > 1)).any()):
                raise ValueError('Edge confidence must be in [0,1]')
            result.append(dict(src=a, dst=b, stencil=_stencil(flow), confidence=weight,
                provenance=edge.get('provenance', 'caller_source_only_edges')))
        return result

    def _reachability(self):
        reachable = torch.zeros_like(self.masks)
        reachable[0] = self.masks[0]*self.valid[0]
        for _ in range(max(1, self.frames-1)):
            updated = reachable.clone()
            for edge in self.edges:
                a, b = edge['src'], edge['dst']
                candidate = (_apply_stencil(reachable[a], edge['stencil'])*
                             edge['confidence']*self.valid[b])
                updated[b] = torch.maximum(updated[b], candidate)
            if torch.equal(updated, reachable):
                break
            reachable = updated
        return reachable*(reachable >= self.reach_min)

    def _effect(self, frame):
        self.response_evaluations += 1
        parameter = self.parameters[frame]
        if self.mode == 'rgb':
            return parameter[0]
        value = self.decode((self.z[frame:frame+1]+self.scale*parameter)[None]).float()
        self.decoder_evaluations += 1
        if value.shape != (1, 3, self.height, self.width):
            raise ValueError('Decoder output changed shape during correction')
        return value[0]-self.base[frame].to(self.device)

    def _score(self, effects, current_effect, energies):
        native = observation = anchor = keep = temporal = 0.
        for i, effect in enumerate(effects):
            native += float(_weighted_mse(effect-current_effect[i], self.valid[i])/energies[i])
            observation += float(_weighted_mse(effect-self.evidence[i], self.observed[i])/energies[i])
            keep += float(_weighted_mse(effect, self.valid[i]*(1-self.support[i]))/energies[i])
            if i == 0:
                anchor = float(_weighted_mse(effect-self.evidence[i], self.valid[i])/energies[i])
        for edge in self.edges:
            a, b = edge['src'], edge['dst']
            temporal += float(_weighted_mse(effects[b]-_apply_stencil(effects[a], edge['stencil']),
                                           edge['weight'])/energies[b])
        regularizer = sum(float((p.detach()[0]-self.initial[i]).square().mean())
                          for i, p in enumerate(self.parameters))
        objective = (self.native_weight*native+self.observation_weight*observation+
                     self.anchor_weight*anchor+self.keep_weight*keep+
                     self.joint_weight*temporal+self.ridge*regularizer)/self.frames
        return dict(objective=objective, native_fit=native/self.frames,
            observed_fit=observation/self.frames, anchor_fit=anchor,
            keep_error=keep/self.frames, temporal_error=temporal/max(1, len(self.edges)))

    def project(self, current_rgb, progress=None):
        """Correct current generator prediction; numerical failure returns it.

        Invalid caller shapes/values raise. A finite-but-unreachable edit or
        decoder numerical failure is recorded and falls back to the incoming
        GENERATOR prediction, never a silent copy of the source video.
        """
        started = time.monotonic()
        current = torch.as_tensor(current_rgb).detach().float().cpu()
        if current.shape != self.source.shape:
            raise ValueError('current_rgb must match source times and RGB canvas')
        _finite(current, 'current generator prediction')
        if bool(((current < 0) | (current > 1)).any()):
            raise ValueError('current_rgb must be in [0,1]')
        self.calls += 1
        before_evaluations = self.decoder_evaluations
        before_responses = self.response_evaluations
        current_effect = (current-self.source)*self.valid
        # No edit evidence must not turn this controller into an AE restoration
        # operator. The incoming generator prediction and sampling remain intact.
        no_edit = float(self.evidence.abs().max()) <= float(self.cfg.get('no_edit_epsilon', 1e-7))
        diagnostics = dict(mode=self.mode, call=self.calls, scale_provenance=self.scale_provenance,
            graph_edges=len(self.edges), iterations=self.steps, fallback=False,
            no_edit=no_edit, gradient_checked=False, history=[],
            observed_fraction=float((self.observed > .1).float().mean()),
            reachable_fraction=float((self.reachable > 0).float().mean()),
            recovered_fraction=float(((self.observed <= .1) & (self.reachable > 0)).float().mean()),
            unknown_fraction=float(((self.support > .1) & (self.mix == 0)).float().mean()),
            mix_mean=float(self.mix.mean()), protect_mix=self.protect_mix,
            hard_anchor=self.hard_anchor,
            limitation='Source flow reliability is not semantic edit correctness; no unseen-surface guarantee.')
        checkpoint = [p.detach().clone() for p in self.parameters]
        optimizer_checkpoint = [copy.deepcopy(o.state_dict()) for o in self.optimizers]
        if no_edit:
            diagnostics.update(elapsed_seconds=time.monotonic()-started, decoder_evaluations=0,
                               response_evaluations=0, DI_throughput=0.,
                               throughput_unit='response forward frames/s/rank including state backward')
            return dict(reference=current.clone(), effect=current-self.source,
                raw_effect=torch.zeros_like(current), delta=self._delta(),
                mix=torch.zeros_like(self.mix), reachable=self.reachable.clone(),
                diagnostics=diagnostics)
        try:
            with torch.no_grad():
                effects = [self._effect(i).detach().cpu() for i in range(self.frames)]
            for effect in effects:
                _finite(effect, 'initial decoded response')
            energy = (self.evidence.square()*self.support).sum((1, 2, 3))/(3*self.support.sum((1, 2, 3))).clamp_min(1.)
            # A missing proposal must still get useful graph/native gradients.
            anchor_energy = _weighted_mse(self.evidence[0], self.valid[0]).clamp_min(1e-4)
            energies = torch.maximum(energy, anchor_energy*.25).clamp_min(1e-4)
            initial_score = self._score(effects, current_effect, energies)
            best_score = initial_score['objective']
            best = checkpoint
            best_optimizers = optimizer_checkpoint
            diagnostics['before'] = initial_score
            for sweep in range(self.steps):
                for i, (parameter, optimizer) in enumerate(zip(self.parameters, self.optimizers)):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.enable_grad():
                        effect = self._effect(i)
                        if not effect.requires_grad:
                            raise RuntimeError('Decoder detached the editable state')
                        valid = self.valid[i].to(self.device)
                        energy_i = energies[i].to(self.device)
                        loss = self.native_weight*_weighted_mse(effect-current_effect[i].to(self.device), valid)/energy_i
                        loss = loss+self.observation_weight*_weighted_mse(effect-self.evidence[i].to(self.device), self.observed[i].to(self.device))/energy_i
                        if i == 0:
                            loss = loss+self.anchor_weight*_weighted_mse(effect-self.evidence[0].to(self.device), valid)/energy_i
                        loss = loss+self.keep_weight*_weighted_mse(effect, (self.valid[i]*(1-self.support[i])).to(self.device))/energy_i
                        for edge in self.edges:
                            a, b = edge['src'], edge['dst']
                            if i == b:
                                residual = effect-_apply_stencil(effects[a], edge['stencil']).to(self.device)
                            elif i == a:
                                residual = _apply_stencil(effect, edge['stencil'])-effects[b].to(self.device)
                            else:
                                continue
                            loss = loss+self.joint_weight*_weighted_mse(residual, edge['weight'].to(self.device))/energies[b].to(self.device)
                        response_loss = loss
                        loss = loss+self.ridge*(parameter[0]-self.initial[i]).square().mean()
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError('Nonfinite joint correction objective')
                        if not diagnostics['gradient_checked']:
                            response_gradient = torch.autograd.grad(response_loss, parameter, retain_graph=True, allow_unused=True)[0]
                            if response_gradient is not None and bool(torch.isfinite(response_gradient).all()):
                                diagnostics['gradient_checked'] = float(response_gradient.abs().sum()) > 1e-12
                        grad = torch.autograd.grad(loss, parameter)[0]
                    if not bool(torch.isfinite(grad).all()):
                        raise FloatingPointError('Nonfinite decoder-input correction gradient')
                    parameter.grad = grad
                    torch.nn.utils.clip_grad_norm_([parameter], float(self.cfg.get('gradient_clip', 1.)))
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        trust = self.trust if self.mode == 'rae' else self.rgb_trust
                        center = self.initial[i:i+1]
                        parameter.copy_(torch.maximum(torch.minimum(parameter, center+trust), center-trust))
                        effects[i] = self._effect(i).detach().cpu()
                    _finite(effects[i], 'updated decoded response')
                    if progress is not None:
                        elapsed = max(time.monotonic()-started, 1e-6)
                        progress(dict(phase='joint_state_frame', mode=self.mode, call=self.calls,
                            sweep=sweep+1, frame=i+1, frames=self.frames,
                            DI_throughput=(self.response_evaluations-before_responses)/elapsed,
                            throughput_unit='response forward frames/s/rank including state backward'))
                score = self._score(effects, current_effect, energies)
                if not torch.isfinite(torch.tensor(score['objective'])):
                    raise FloatingPointError('Nonfinite global correction score')
                if score['objective'] < best_score:
                    best_score = score['objective']
                    best = [p.detach().clone() for p in self.parameters]
                    best_optimizers = [copy.deepcopy(o.state_dict()) for o in self.optimizers]
                row = dict(sweep=sweep+1, **score)
                diagnostics['history'].append(row)
                if progress is not None:
                    elapsed = max(time.monotonic()-started, 1e-6)
                    progress(dict(phase='joint_state_correction', mode=self.mode, call=self.calls,
                        **row, DI_throughput=(self.response_evaluations-before_responses)/elapsed,
                        throughput_unit='response forward frames/s/rank including state backward'))
            with torch.no_grad():
                for p, value, optimizer, optimizer_state in zip(self.parameters, best, self.optimizers, best_optimizers):
                    p.copy_(value); optimizer.load_state_dict(optimizer_state)
                effects = [self._effect(i).detach().cpu() for i in range(self.frames)]
            raw = torch.stack(effects)
            _finite(raw, 'final decoded response')
            diagnostics['after'] = self._score(effects, current_effect, energies)
            # Reliable graph paths can carry corrected effects beyond missing
            # proposal observations. Truly unobserved/unreachable pixels retain
            # the current native output, not source or an invented hard target.
            mixed = self.mix*raw+(1-self.mix)*current_effect
            protection = self.protect_mix*self.valid*(1-self.support)
            mixed = mixed*(1-protection)
            reference = current+self.valid*(self.source+mixed-current)
            reference = reference.clamp(0, 1)
            if self.hard_anchor:
                reference[0] = current[0]+self.valid[0]*(self.anchor-current[0])
            diagnostics['raw_anchor_relative_mse'] = float(_weighted_mse(raw[0]-self.evidence[0], self.valid[0])/energies[0])
            diagnostics['feedback_l1'] = float((reference-current).abs().mean())
        except Exception as exc:
            with torch.no_grad():
                for parameter, value, optimizer, optimizer_state in zip(self.parameters, checkpoint, self.optimizers, optimizer_checkpoint):
                    parameter.copy_(value); optimizer.load_state_dict(optimizer_state)
            diagnostics.update(fallback=True, error=repr(exc), feedback_l1=0.)
            reference = current.clone()
            raw = current_effect.clone()
        elapsed = time.monotonic()-started
        evaluations = self.decoder_evaluations-before_evaluations
        diagnostics.update(elapsed_seconds=elapsed, decoder_evaluations=evaluations,
            response_evaluations=self.response_evaluations-before_responses,
            DI_throughput=(self.response_evaluations-before_responses)/max(elapsed, 1e-6),
            throughput_unit='response forward frames/s/rank including state backward')
        return dict(reference=reference, effect=reference-self.source, raw_effect=raw,
            delta=self._delta(), mix=self.mix.clone(), reachable=self.reachable.clone(),
            diagnostics=diagnostics)

    def _delta(self):
        if self.mode == 'rgb':
            return None
        return torch.cat([self.scale*p.detach() for p in self.parameters], 0).cpu()

    def state_dict(self):
        """Small solver snapshot for rollback if downstream VAE writeback fails."""
        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu(v) for v in value]
            if isinstance(value, tuple):
                return tuple(cpu(v) for v in value)
            return copy.deepcopy(value)
        return dict(schema='joint-edit-solver-v1', mode=self.mode,
            parameters=[p.detach().cpu().clone() for p in self.parameters],
            optimizers=[cpu(o.state_dict()) for o in self.optimizers], calls=self.calls)

    def load_state_dict(self, value):
        if (value.get('schema') != 'joint-edit-solver-v1' or value.get('mode') != self.mode or
                len(value.get('parameters', [])) != self.frames or
                len(value.get('optimizers', [])) != self.frames):
            raise ValueError('Incompatible joint solver snapshot')
        for parameter, saved in zip(self.parameters, value['parameters']):
            if saved.shape != parameter.shape or not bool(torch.isfinite(saved).all()):
                raise ValueError('Invalid saved joint solver parameter')
        with torch.no_grad():
            for parameter, saved, optimizer, optimizer_state in zip(
                    self.parameters, value['parameters'], self.optimizers, value['optimizers']):
                parameter.copy_(saved.to(parameter.device))
                optimizer.load_state_dict(optimizer_state)
        self.calls = int(value.get('calls', 0))
