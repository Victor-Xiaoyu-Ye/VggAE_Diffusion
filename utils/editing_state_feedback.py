"""Source-only decoder-calibrated feedback into the actual VACE trajectory.

The frozen RAE decoder operates on nine source-time states. Only the *change*
to the current VACE prediction is mapped through the two image canvases, so
resampling itself never restores the source or introduces a writeback residual.
"""
import json
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from utils.editing_data import rae_canvas
from utils.editing_video_transport import warp_flow
from utils.vace_edit_runtime import decode_native
from utils.vace_state_data import align_rae_state, build_rgb_state


class SceneDecoderOnly(nn.Module):
    """Exact SceneRAE decode path, with no encoder/compressor on the device."""
    def __init__(self, teacher, checkpoint_decode=False):
        super().__init__()
        self.channels = teacher.channels
        self.geo_expand = teacher.geo_projection.expand
        self.tex_expand = teacher.tex_projection.expand
        self.decoder = teacher.decoder
        self.checkpoint_decode = bool(checkpoint_decode)
        self.eval().requires_grad_(False)

    def _decode(self, z):
        geo, texture = z.split(self.channels//2, dim=-1)
        return self.decoder(self.geo_expand(geo), self.tex_expand(texture))

    def forward(self, z):
        if z.ndim != 5 or z.shape[1] != 1:
            raise ValueError('Joint solver decoder requires one source-time state per call')
        if self.checkpoint_decode and torch.is_grad_enabled() and z.requires_grad:
            from torch.utils.checkpoint import checkpoint
            value = checkpoint(self._decode, z, use_reentrant=False)
        else:
            value = self._decode(z)
        return value[:, 0].permute(0, 3, 1, 2).float()


def load_decoder_only(path, device, checkpoint_decode=True):
    """Strict CPU load, retain only expand+decoder, then move the small path."""
    from utils.editing_train import load_teacher
    teacher = load_teacher(path, torch.device('cpu'))
    result = SceneDecoderOnly(teacher, checkpoint_decode=checkpoint_decode)
    del teacher
    return result.to(device).eval().requires_grad_(False)


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Nonfinite '+name)


def map_square_correction(value, square_rect, spatial):
    """Map only a residual, with exactly zero native padding."""
    top, left, height, width = map(int, spatial['valid_rect'])
    y, x, rh, rw = map(int, square_rect)
    canvas_h, canvas_w = map(int, spatial['canvas'])
    cropped = value[..., y:y+rh, x:x+rw]
    if cropped.shape[-2:] != (rh, rw) or min(rh, rw, height, width) <= 0:
        raise ValueError('Invalid square/native valid rectangle')
    resized = F.interpolate(cropped.float(), size=(height, width), mode='bilinear', align_corners=False)
    result = value.new_zeros((*value.shape[:-2], canvas_h, canvas_w))
    result[..., top:top+height, left:left+width] = resized
    return result


def expand_source_correction(correction, inputs, total_frames):
    """Place measured times exactly; warp intervening corrections once.

    Pairwise confidence is required for unobserved frames; cumulative proposal
    confidence is intentionally not multiplied again. Missing reliability keeps
    the current native prediction, including newly disoccluded pixels.
    """
    observed = [int(v) for v in inputs['observed_indices']]
    if len(observed) != len(correction) or observed != sorted(set(observed)):
        raise ValueError('Observed source times must be unique and ordered')
    if not observed or min(observed) < 0 or max(observed) >= total_frames:
        raise ValueError('Observed source times outside generated window')
    result = correction.new_zeros((total_frames, *correction.shape[1:]))
    result[observed] = correction
    flows, confidence = inputs.get('full_flows_backward'), inputs.get('full_edge_confidence')
    if flows is None or confidence is None:
        return result
    flows, confidence = torch.as_tensor(flows).float().cpu(), torch.as_tensor(confidence).float().cpu()
    side_h, side_w = correction.shape[-2:]
    if flows.shape != (total_frames-1, side_h, side_w, 2):
        raise ValueError('Full source flows must connect consecutive native times')
    if confidence.shape != (total_frames-1, 1, side_h, side_w):
        raise ValueError('Need pairwise confidence for every full source edge')
    _finite(flows, 'source flows'); _finite(confidence, 'source edge confidence')
    if bool(((confidence < 0) | (confidence > 1)).any()):
        raise ValueError('Source edge confidence outside [0,1]')
    known = set(observed)
    for i in range(1, total_frames):
        if i not in known:
            result[i] = warp_flow(result[i-1], flows[i-1])*confidence[i-1]
    return result


def align_feedback_gate(mix, inputs, reference):
    """Source-time reliability -> native patch16, with zero reference prefix.

    This is a conservative state-update gate, not a claim that nonlinear latent
    blending equals RGB blending. In particular, zero confidence cannot replace
    a previously cached edit state through the next model forward.
    """
    mix = torch.as_tensor(mix).detach().float().cpu()
    if mix.shape != (len(inputs['observed_indices']), 1, *inputs['source'].shape[-2:]):
        raise ValueError('Bridge reliability and source-time canvases differ')
    _finite(mix, 'bridge reliability')
    if bool(((mix < 0) | (mix > 1)).any()):
        raise ValueError('Bridge reliability outside [0,1]')
    aligned = map_square_correction(mix, inputs['valid_rect'], reference['spatial'])
    mapping = {int(v): i for i, v in enumerate(inputs['observed_indices'])}
    times = list(range(0, len(reference['source']), 4))
    if any(t not in mapping for t in times):
        raise ValueError('Missing source-time reliability for a native temporal anchor')
    selected = aligned[[mapping[t] for t in times]]
    selected = torch.cat((torch.zeros_like(selected[:1]), selected), 0)
    return F.avg_pool2d(selected, 16, stride=16).permute(1, 0, 2, 3).contiguous()


def _device_memory(device):
    if device.type == 'cpu':
        return dict(device='cpu', accelerator_memory_available=False)
    backend = getattr(torch, device.type)
    result = dict(device=str(device), accelerator_memory_available=True)
    for key, name in (('allocated_gib', 'memory_allocated'),
                      ('reserved_gib', 'memory_reserved'),
                      ('peak_allocated_gib', 'max_memory_allocated'),
                      ('peak_reserved_gib', 'max_memory_reserved')):
        if hasattr(backend, name):
            try:
                result[key] = float(getattr(backend, name)(device))/1024**3
            except Exception as exc:
                result[key+'_unavailable'] = repr(exc)
    return result


def solver_evidence(diagnostics):
    """Report optimization separately from later VAE/writeback success.

    Derive the objective comparison from finite measured values, not a solver's
    claimed success flag. A downstream failure can still roll this candidate
    back; consumers must additionally check applied and not fallback.
    """
    before = diagnostics.get('before', {}).get('objective')
    after = diagnostics.get('after', {}).get('objective')
    valid = (isinstance(before, (int, float)) and isinstance(after, (int, float))
             and math.isfinite(before) and math.isfinite(after))
    improved = bool(valid and after < before and not diagnostics.get('fallback'))
    return dict(solver_objective_observed=valid, solver_objective_improved=improved,
        solver_objective_before=before if valid else None,
        solver_objective_after=after if valid else None,
        solver_relative_decrease=(before-after)/max(abs(before), 1e-12) if valid else None,
        solver_accepted_steps=diagnostics.get('accepted_steps'),
        solver_rejected_steps=diagnostics.get('rejected_steps'),
        solver_trial_evaluations=diagnostics.get('trial_evaluations'),
        solver_backtracks=diagnostics.get('backtracks'),
        solver_state_delta_rms=diagnostics.get('actual_state_delta_rms'),
        solver_step_sizes=diagnostics.get('step_sizes'))


class StateFeedback:
    """Transactional callback: solver -> reference -> bridge + clean residual.

    ``reference`` is the decoded Stage65 source/shared-proposal payload. Its
    target video is neither accessed nor accepted as an inference condition.
    ``observer(event)`` may persist ``last_reference`` and ``last_projection``.
    """
    def __init__(self, vae, projector, inputs64, reference, state, cfg, arm, observer=None):
        self.vae, self.projector, self.inputs, self.reference = vae, projector, inputs64, reference
        self.state, self.arm, self.observer = state, str(arm), observer
        self.cfg = dict(cfg)
        self.cfg.update(cfg.get('feedback', {}))
        self.alpha = float(self.cfg.get('alpha', self.cfg.get('writeback_alpha', .35)))
        self.max_rms = float(self.cfg.get('max_clean_delta_rms', self.cfg.get('writeback_max_clean_rms', .25)))
        self.max_sigma = float(self.cfg.get('max_sigma', self.cfg.get('writeback_max_sigma', .75)))
        self.requested = tuple(float(v) for v in self.cfg.get('sigmas', self.cfg.get('writeback_sigmas', (.6, .35, .15))))
        self.update_bridge_state = bool(self.cfg.get('update_bridge_state', True))
        self.reset_history = bool(self.cfg.get('reset_solver_history', True))
        if not 0 <= self.alpha <= 1 or self.max_rms <= 0 or not 0 < self.max_sigma < 1:
            raise ValueError('Invalid feedback alpha, clean RMS cap or maximum sigma')
        if not self.requested or len(self.requested) > 3 or any(not 0 < v <= self.max_sigma for v in self.requested):
            raise ValueError('Feedback needs at most three positive bounded sigma requests')
        self.events, self.selected_steps, self.applied = [], set(), set()
        self.last_reference, self.last_projection = None, None
        self._reset_requested = False

    def set_schedule(self, sigmas):
        schedule = torch.as_tensor(sigmas).detach().float().cpu().flatten().tolist()
        usable = [i for i, s in enumerate(schedule) if 0 < s <= self.max_sigma]
        self.selected_steps = {min(usable, key=lambda i: abs(schedule[i]-s)) for s in self.requested} if usable else set()
        self.schedule = schedule
        self.events, self.applied = [], set()
        self._reset_requested = False

    def consume_history_reset(self):
        reset, self._reset_requested = self._reset_requested, False
        return reset

    def _event(self, event, started):
        elapsed = time.monotonic()-started
        event.update(elapsed_seconds=elapsed,
            DI_throughput=(len(self.reference['source'])/max(elapsed, 1e-6) if event.get('applied') else 0.),
            throughput_unit='feedback video frames/s/rank including solver and VAE',
            memory=_device_memory(self.state.device))
        self.events.append(event)
        if self.observer is not None:
            self.observer(event)

    @torch.no_grad()
    def __call__(self, *, step_index, sigma, latent, velocity):
        if step_index not in self.selected_steps or step_index in self.applied:
            return velocity
        self.applied.add(step_index)
        started = time.monotonic()
        event = dict(phase='state_feedback', arm=self.arm, step_index=int(step_index),
            sigma=float(sigma), alpha=self.alpha, applied=False, fallback=False,
            state_delta_rms=0., clean_delta_rms=0., prefix_delta_rms=0.)
        if self.alpha == 0:
            event['reason'] = 'alpha_zero_exact_noop'
            self._event(event, started)
            return velocity
        snapshot = None
        try:
            if not 0 < sigma <= self.max_sigma or latent.shape != velocity.shape or latent.ndim != 4:
                raise ValueError('Invalid feedback sigma or latent/velocity shape')
            _finite(latent, 'sampling latent'); _finite(velocity, 'sampling velocity')
            if hasattr(self.projector, 'state_dict'):
                snapshot = self.projector.state_dict()
            clean = latent.float()-float(sigma)*velocity.float()
            decoded = decode_native(self.vae, clean, reference_count=1).float()
            source = self.reference['source']
            if decoded.shape != (3, len(source), *source.shape[-2:]):
                raise ValueError('Decoded native prediction and shared source canvas differ')
            _finite(decoded, 'native clean RGB')
            current = ((decoded.detach().cpu().permute(1, 0, 2, 3)+1)/2).clamp(0, 1)
            side_h, side_w = self.inputs['source'].shape[-2:]
            if side_h != side_w:
                raise ValueError('RAE solver canvas must be square')
            square, square_rect = rae_canvas(current, self.reference['spatial']['valid_rect'], size=side_h)
            if tuple(square_rect) != tuple(self.inputs['valid_rect']):
                raise ValueError('Prepared source and live prediction use different RAE letterboxes')
            ids = list(map(int, self.inputs['observed_indices']))
            current9 = square[ids].contiguous()
            def solver_progress(row):
                # Console heartbeat only. The outer observer persists complete
                # feedback transactions rather than every decoder frame.
                print('[state feedback solver] '+json.dumps(dict(row,
                    arm=self.arm, step_index=int(step_index), sigma=float(sigma))), flush=True)
            projected = self.projector.project(current9, progress=solver_progress)
            diagnostics = projected.get('diagnostics', {})
            event['solver'] = diagnostics
            event.update(solver_evidence(diagnostics))
            if diagnostics.get('fallback'):
                raise RuntimeError('Joint solver fallback: '+str(diagnostics.get('error', 'unspecified')))
            corrected9 = torch.as_tensor(projected['reference']).detach().float().cpu()
            if corrected9.shape != current9.shape:
                raise ValueError('Solver reference shape differs from live source-time prediction')
            _finite(corrected9, 'solver reference')
            correction = expand_source_correction(corrected9-current9, self.inputs, len(source))
            correction = map_square_correction(correction, square_rect, self.reference['spatial'])
            # An independent anchor is shared by all arms. Reinsert its native
            # pixels only when the solver actually requested an edit.
            if diagnostics.get('hard_anchor') and not diagnostics.get('no_edit'):
                top, left, height, width = map(int, self.reference['spatial']['valid_rect'])
                correction[0, :, top:top+height, left:left+width] = (
                    torch.as_tensor(self.reference['anchor']).float()[:, top:top+height, left:left+width]-
                    current[0, :, top:top+height, left:left+width])
            if diagnostics.get('no_edit') or not bool(correction.abs().max() > 0):
                event['reason'] = 'zero_pixel_correction_exact_noop'
                self.last_projection = projected
                self._event(event, started)
                return velocity
            updated = dict(self.reference, reference=(current+correction).clamp(0, 1))
            if not self.update_bridge_state:
                new_state = self.state.clone()
            elif self.arm.startswith('rae'):
                source_z = torch.as_tensor(self.inputs['source_z']).detach().float().cpu()
                if source_z.ndim == 5:
                    source_z = source_z[0]
                delta = torch.as_tensor(projected['delta']).detach().float().cpu()
                if delta.shape != source_z.shape:
                    raise ValueError('Solver RAE delta and cached source states differ')
                new_state = align_rae_state(source_z, source_z+delta,
                    self.inputs['anchor_z'], self.inputs, updated)
            elif self.arm.startswith('rgb'):
                new_state = build_rgb_state(updated)
            else:
                raise ValueError('Feedback requires an RGB or RAE state arm')
            if new_state.shape != self.state.shape:
                raise ValueError('Updated bridge state shape differs')
            _finite(new_state, 'updated bridge state')
            new_state = new_state.to(self.state)
            _finite(new_state, 'updated bridge storage dtype')
            if self.update_bridge_state and self.arm.startswith('rae'):
                gate = align_feedback_gate(projected['mix'], self.inputs, self.reference).to(self.state)
                channels = (new_state.shape[0]-2)//2
                if new_state.shape[0] != 2*channels+2 or gate.shape[1:] != new_state.shape[1:]:
                    raise ValueError('Reliability gate and RAE bridge lattice differ')
                candidate = new_state
                new_state = self.state.clone()
                new_state[channels:2*channels] += gate*(
                    candidate[channels:2*channels]-self.state[channels:2*channels])
                event.update(bridge_gate_mean=float(gate.mean()),
                    bridge_gate='RAE edit channels only; conservative reliability blend, not exact RGB equivalence')
            # Prefix is the fixed independent anchor, never a generated frame.
            new_state[:, 0].copy_(self.state[:, 0])
            proposed = (decoded+2*correction.permute(1, 0, 2, 3).to(decoded)).clamp(-1, 1)
            encodings = self.vae.encode([proposed, decoded])
            target, original = (value.float() for value in encodings)
            if target.shape != clean[:, 1:].shape or original.shape != target.shape:
                raise ValueError('VAE feedback encoding changed the native video lattice')
            change = self.alpha*(target-original)
            _finite(change, 'VAE clean correction')
            rms = change.square().mean((-2, -1), keepdim=True).sqrt()
            change *= (self.max_rms/rms.clamp_min(1e-12)).clamp_max(1.)
            corrected = velocity.float().clone()
            corrected[:, 1:] -= change/float(sigma)
            _finite(corrected, 'corrected sampling velocity')
            event.update(applied=True, clean_delta_rms=float(change.square().mean().sqrt()),
                state_delta_rms=float((new_state.float()-self.state.float()).square().mean().sqrt()),
                pixel_correction_l1=float(correction.abs().mean()),
                actual_sigma=float(sigma), reference_prefix_preserved=True,
                solver_history_reset=self.reset_history, update_bridge_state=self.update_bridge_state)
            # Commit only after every decode, projection, alignment, encoding
            # and finite check succeeded. Any earlier exception is a no-op.
            self.state.copy_(new_state)
            self.last_reference, self.last_projection = updated, projected
            self._reset_requested = self.reset_history
        except Exception as exc:
            if snapshot is not None:
                self.projector.load_state_dict(snapshot)
            event.update(applied=False, fallback=True, solver_rolled_back=snapshot is not None,
                error=repr(exc), reason='feedback_failed_preserved_native')
            corrected = velocity
        self._event(event, started)
        return corrected
