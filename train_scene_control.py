"""Matched controlled 4->5 generation with frozen stage53 representations.

A is observed-prefix completion. B additionally receives supplied future
camera/point trajectories. C also transports observed texture along those
trajectories. Neither B nor C is autonomous prediction. This entry reuses the
tested prefix optimizer, sampler, EMA, snapshots, dual IO and resume loop.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from audit_scene_flow_weights import ensure_disjoint, receipt_spec, stage_asset
from train_scene_motion import build_plan as prefix_source_plan
from train_scene_pipeline import barrier, distributed
from train_scene_prefix import train
from utils.scene_control import CONTROL_ARMS, CONTROL_SCHEMA, bind_control, create_control_model, load_warm
from utils.scene_quality_data import load_street_selection
from utils.scene_run import ArtifactStore, StageBudget, digest


class ControlRuntime:
    """Explicit extension points; no monkeypatch or mutable model hooks."""
    def __init__(self, arm, control_lr_multiplier=10.):
        if arm not in CONTROL_ARMS:
            raise ValueError('Unknown controlled-generation arm: '+str(arm))
        self.arm = arm
        self.control_lr_multiplier = float(control_lr_multiplier)
        if not math.isfinite(self.control_lr_multiplier) or self.control_lr_multiplier<=0:
            raise ValueError('Control learning-rate multiplier must be positive and finite')

    def make_model(self, model_args):
        return create_control_model(model_args, self.arm)

    def load_warm(self, model, state):
        return load_warm(model, state)

    def checkpoint_metadata(self):
        return dict(model_family='SceneFlow' if self.arm=='baseline' else 'ControlledSceneFlow',
                    control_schema=CONTROL_SCHEMA, experiment_arm=self.arm,
                    control_lr_multiplier=self.control_lr_multiplier)

    def optimizer_groups(self, model):
        backbone, adapter = [], []
        for name, param in model.named_parameters():
            (adapter if name.startswith('control.') else backbone).append(param)
        groups = [dict(params=backbone,lr_scale=1.,name='backbone')]
        if adapter:
            groups.append(dict(params=adapter,lr_scale=self.control_lr_multiplier,name='control'))
        return groups

    @torch.no_grad()
    def training_metrics(self, model, net, noisy):
        if self.arm=='baseline':
            return {}
        features = net.controls.float()
        observed_and_noisy = torch.cat((net.known_prefix,noisy),1).float().flatten(1,2)
        injected = model.control(features.flatten(1,2))
        base = model.input(observed_and_noisy)
        rms = lambda value: float(value.float().square().mean().sqrt())
        return dict(control_camera_rms=rms(features[...,:8]),
            control_motion_rms=rms(features[...,8:16]),
            control_reference_rms=rms(features[...,16:-1]),
            control_track_support_fraction=float((features[...,10]>0).float().mean()),
            control_reference_support_fraction=float((features[...,-1]>0).float().mean()),
            control_reference_support_mean=float(features[...,-1].mean()),
            control_injection_rms=rms(injected),
            control_to_input_rms_ratio=rms(injected)/max(rms(base),1e-8),
            control_lr_multiplier=self.control_lr_multiplier)

    def bind(self, model, batch, sample, training=False):
        # A still reads the same prepared sample bank as B/C, but it never
        # passes the controls to the model. A/B/C thus share sample exposure.
        if self.arm != 'baseline':
            controls = sample.get('controls')
            if not isinstance(controls, dict):
                raise ValueError('Prepared motion controls are missing; do not silently run an uncontrolled arm')
            schema = controls.get('schema', CONTROL_SCHEMA)
            if schema != CONTROL_SCHEMA:
                raise ValueError('Unsupported motion-control schema')
        return bind_control(model, batch, sample, self.arm, training=training)

    @staticmethod
    def seed_training(seed, rank):
        # Matched sample/noise schedules regardless of extra adapter init RNG.
        random.seed(seed+rank)
        np.random.seed(seed+rank)
        torch.manual_seed(seed+rank)

    def annotate(self, row, sample):
        controlled = self.arm != 'baseline'
        labels = sample.get('controls', {})
        rays = torch.as_tensor(labels.get('rays', []))
        weight = torch.as_tensor(labels.get('track_weight', []))
        row.update(experiment_arm=self.arm,
            source_eval_index=sample.get('source_eval_index'),
            task='supplied-future-motion conditioned video' if controlled else 'observed-prefix completion',
            condition_policy='future camera and point tracks; observed-only appearance' if controlled else
                             'observed prefix only',
            camera_present=bool(controlled and rays.numel() and (rays[..., -1]>0).any()),
            future_control_supplied=controlled,
            control_provenance='offline full-window teacher pseudo-labels; not predicted at inference' if controlled else None,
            semantic_instance_masks=False,
            reference_source='normalized texture from observed prefix only' if self.arm=='reference' else None,
            track_valid_fraction=float((weight>0).float().mean()) if weight.numel() else 0.,
            appearance_leakage_policy='No future RGB, texture or dense future depth enters conditions.',
            annotation_note='Point query IDs are correspondences, not semantic object identities.')
        if 'supervision' in row:
            row['supervision']='GT future latent is the denoising target; B/C separately receive explicit future motion controls.'


def build_plan(source, codec, controls, cfg, selection, world):
    """Pin both the original latent/weight contract and the new control bank."""
    base, original = prefix_source_plan(source, codec, cfg, selection, world)
    ready = controls.json('data/cache_complete.json', True)
    prepared = controls.json('contract.json', True)
    if ready['identity'] != prepared['identity']:
        raise ValueError('Prepared controls manifest and contract differ')
    if ready.get('source_prefix') != base['source_prefix']:
        raise ValueError('Controls were prepared from a different source prefix run')
    if ready.get('source_cache_manifest_sha') != digest(original):
        raise ValueError('Controls do not identify the original matched latent cache')
    if ready.get('mean') != base['mean'] or ready.get('std') != base['std']:
        raise ValueError('Control preparation must preserve original latent normalization')
    if not ready.get('shards') or not ready.get('samples'):
        raise ValueError('No valid prepared windows; cannot start training on an empty bank')
    evaluation = receipt_spec(controls, 'data/eval.pt')
    if evaluation['identity'] != ready['identity']:
        raise ValueError('Control evaluation and training caches differ')
    base.pop('identity', None)
    base.update(schema='scene-control-reference-v1',
        control_identity=ready['identity'], control_manifest_sha=digest(ready),
        control_contract_sha=digest(prepared), source_evaluation=base['evaluation'],
        evaluation=evaluation, matched_windows=ready['samples'],
        condition=('observed prefix only' if cfg['arm']=='baseline' else
                   'supplied future camera and sparse point tracks; no future appearance'),
        reference=('observed texture soft transport' if cfg['arm']=='reference' else 'none'),
        comparison='All arms use identical prepared windows/source/noise. Compare equal-update snapshots.',
        control_status='teacher pseudo-controls; no semantic masks and no autonomous motion planner',
        feature_target='unchanged absolute C256; original AE and statistics frozen')
    return dict(base, identity=digest(base)), ready


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--cohort', required=True)
    p.add_argument('--selection', required=True)
    p.add_argument('--encoder')  # preparation owns teacher; kept for wrapper compatibility
    p.add_argument('--source-local', required=True)
    p.add_argument('--source-root', action='append', default=[])
    p.add_argument('--controls-local', required=True)
    p.add_argument('--controls-root', action='append', default=[])
    p.add_argument('--output', required=True)
    p.add_argument('--root', action='append', default=[])
    p.add_argument('--arm', choices=CONTROL_ARMS, default=os.environ.get('SCENE_CONTROL_ARM','reference'))
    p.add_argument('--stage', choices=['preflight','train'], default='train')
    p.add_argument('--startup-seconds', type=float, default=0.)
    return p


def main():
    a = parser().parse_args()
    ensure_disjoint(a)
    # The prepared controls are immutable and shared by every training arm.
    controls_check = argparse.Namespace(output=a.output, root=a.root,
        source_local=a.controls_local, source_root=a.controls_root)
    ensure_disjoint(controls_check)
    cfg = json.loads(Path(a.config).read_text(encoding='utf8'))
    cfg['arm'] = a.arm
    if cfg['history']!=4 or cfg['views']!=9:
        raise ValueError('Stage56 preserves four observed and five generated frames')
    if cfg['condition_policy']!='future_camera_tracks_with_observed_only_appearance':
        raise ValueError('Unsupported control task policy')
    codec_namespace = cfg['codec_namespace']
    if not codec_namespace.replace('_','').replace('-','').isalnum():
        raise ValueError('Invalid codec namespace')
    selection = load_street_selection(a.cohort, a.selection)
    device, rank, world = distributed()
    if world != 24:
        raise ValueError('Production stage56 uses three nodes x eight ranks')
    node = int(os.environ.get('GROUP_RANK',0))
    local_rank = int(os.environ.get('LOCAL_RANK',0))
    ControlRuntime.seed_training(cfg['seed'], rank)
    store = ArtifactStore(a.output,a.root)
    staging = Path(a.output)/'_inputs'/f'node{node}'
    source = ArtifactStore(staging/'prefix',[a.source_local,*a.source_root])
    controls = ArtifactStore(staging/'controls',[a.controls_local,*a.controls_root])
    codec_roots = [str(Path(a.source_local).parent/codec_namespace/'production')]
    codec_roots += [r.rstrip('/').rsplit('/',1)[0]+'/'+codec_namespace+'/production' for r in a.source_root]
    codec = ArtifactStore(staging/'codec',codec_roots)
    progress = store.json('progress.json') or {}
    budget = StageBudget(cfg['hours']*3600 if cfg['hours'] else math.inf,
                         progress.get('total_spent',0.)+max(0.,a.startup_seconds))
    budget.install_signals()
    plan, manifest = build_plan(source,codec,controls,cfg,selection,world)
    if rank==0:
        old = store.json('contract.json')
        if old and old['identity']!=plan['identity']:
            raise ValueError('Changed source/config/controls: use a fresh namespace')
        store.write_json('contract.json',plan)
    barrier()
    complete = store.json('complete.json')
    baseline = store.json(f"eval/{cfg['source_label']}/summary.json")
    artifacts_done = bool(complete and complete.get('evaluation_status')=='completed' and
                          baseline and baseline.get('status')=='completed')
    if not artifacts_done and not (a.stage=='preflight' and store.json('preflight_complete.json')):
        ae_path = stage_asset(codec,plan['ae'],local_rank,rank)
        eval_path = stage_asset(controls,plan['evaluation'],local_rank,rank)
        weights_path = stage_asset(source,plan['source_weights'],local_rank,rank)
        train(a,cfg,plan,source,store,manifest,selection,ae_path,weights_path,eval_path,
              budget,device,rank,world,data_store=controls,
              runtime=ControlRuntime(a.arm,cfg.get('control_lr_multiplier',10.)))
    barrier()
    import train_scene_pipeline as pipeline
    if pipeline.WRITER is not None:
        pipeline.WRITER.close()
        pipeline.WRITER = None
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__=='__main__':
    main()
