"""Frozen-method controls for attribution, not new effect-tuned architecture."""
import copy

import torch
from torch import nn

from utils.vace_edit_runtime import insert_lora, native_forward
from utils.editing_state_feedback import StateFeedback


MAIN_ARMS = ('native', 'native_reference', 'sft', 'rgb_base', 'rae_base', 'rgb_full', 'rae_full')
ABLATION_ARMS = ('rae_reinject', 'rae_no_temporal', 'rae_no_graph', 'rae_shuffle_graph',
                 'rae_no_reliability', 'rae_bridge_only', 'rae_velocity_only',
                 'rae_zero_edit', 'rae_no_state', 'native_reference_60')
SECOND_SEED_ARMS = ('native', 'rgb_full', 'rae_full')


def arm_spec(arm):
    if arm not in MAIN_ARMS+ABLATION_ARMS:
        raise ValueError('Unknown declared paper arm: '+arm)
    family = arm.split('_')[0]
    feedback = family in ('rgb', 'rae') and arm not in ('rgb_base', 'rae_base', 'rae_zero_edit', 'rae_no_state')
    return dict(arm=arm, family=family, feedback=feedback,
                context='native_context' if arm == 'native' else 'context',
                use_state=arm != 'rae_no_state', velocity=arm != 'rae_bridge_only',
                bridge=arm != 'rae_velocity_only', sample_steps=60 if arm == 'native_reference_60' else None)


class ReferenceSFT(nn.Module):
    """Same native reference condition and LoRA, with NO state modules."""
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone, self.cfg = backbone, dict(cfg)
        backbone.requires_grad_(False)
        self.lora_modules = insert_lora(backbone, int(cfg['lora_rank']), float(cfg['lora_alpha']))

    def forward(self, x, t, conditions, text, states=None, use_state=True):
        velocities, _, _ = native_forward(self.backbone, x, t, conditions, text,
            use_checkpoint=self.training, context_scale=self.cfg.get('context_scale', 1.), state_tokens=None)
        return torch.stack(velocities)

    def deltas(self):
        return {n:p.detach().float().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def restore(self, values):
        parameters = {n:p for n,p in self.named_parameters() if p.requires_grad}
        if set(parameters) != set(values):
            raise ValueError('SFT LoRA keys differ; cannot use a state-conditioned checkpoint')
        with torch.no_grad():
            for name, parameter in parameters.items():
                if parameter.shape != values[name].shape:
                    raise ValueError('SFT checkpoint shape mismatch: '+name)
                parameter.copy_(values[name])


def intervened_state(state, arm):
    value = state.clone()
    if arm == 'rae_zero_edit':
        channels = (value.shape[0]-2)//2
        if value.shape[0] != 2*channels+2:
            raise ValueError('Need source/edit/confidence/support state channels')
        # Keep source, support/confidence and the independent anchor prefix.
        value[channels:2*channels, 1:] = 0
    return value


def intervened_inputs(inputs, solver, arm):
    """Change only named online-solver components; cached RGB reference is fixed."""
    result, options = dict(inputs), dict(solver)
    edges = [dict(e) for e in inputs['edges']]
    if arm == 'rae_reinject':
        options['iterations'] = 0
    if arm in ('rae_no_temporal', 'rae_no_graph'):
        options['joint_weight'] = 0.
    if arm == 'rae_no_graph':
        edges = []
    if arm == 'rae_shuffle_graph':
        for edge in edges:
            flow = torch.as_tensor(edge['flow']).clone().float()
            h, w = flow.shape[:2]
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid = torch.stack((xx, yy), -1).to(flow)
            # Permute the pull coordinates, not their displacement vectors.
            # A constant zero flow must become an incorrect correspondence.
            edge['flow'] = torch.roll(grid+flow, shifts=max(1, w//3), dims=1)-grid
    if arm == 'rae_no_reliability':
        result['confidence'] = torch.ones_like(inputs['confidence'])
        result['full_edge_confidence'] = torch.ones_like(inputs['full_edge_confidence'])
        for edge in edges:
            edge['confidence'] = torch.ones_like(edge['confidence'])
    result['edges'] = edges
    return result, options


class PaperFeedback(StateFeedback):
    """Path ablation retains identical solving/VAE work, suppresses one commit.

    bridge-only discards the proposed velocity correction. The event separates
    proposed and actual velocity changes, and the sampler still resets history
    after changing its state input. No mutation of the production callback.
    """
    def __init__(self, *args, apply_velocity=True, observer=None, **kwargs):
        super().__init__(*args, observer=None, **kwargs)
        self.paper_observer, self.apply_velocity = observer, bool(apply_velocity)

    def _event(self, event, started):
        event['velocity_write_enabled'] = self.apply_velocity
        event['proposed_clean_delta_rms'] = event.get('clean_delta_rms', 0.)
        if not self.apply_velocity:
            event['clean_delta_rms'] = 0.
        super()._event(event, started)
        if self.paper_observer is not None:
            self.paper_observer(event)

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        return result if self.apply_velocity else kwargs['velocity']
