"""Matched controlled 4->5 generation with frozen stage53 representations.

A is observed-prefix completion. B additionally receives supplied future
camera/point trajectories. C also transports observed texture along those
trajectories. Neither B nor C is autonomous prediction. This entry reuses the
tested prefix optimizer, sampler, EMA, snapshots, dual IO and resume loop.
An optional camera arm isolates supplied camera rays from point controls.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from audit_scene_flow_weights import broadcast, ensure_disjoint, receipt_spec, stage_asset
from train_scene_motion import build_plan as prefix_source_plan
from train_scene_pipeline import barrier, distributed
from train_scene_prefix import train
from utils.scene_control import CONTROL_ARMS, CONTROL_SCHEMA, bind_control, create_control_model, load_warm
from utils.scene_control_inference import CONTROL_INPUT_POLICY, inference_control_view
from utils.scene_control_readiness import SCHEMA as READINESS_SCHEMA, scan_prepared_payload, summarize
from utils.scene_quality_data import load_street_selection
from utils.scene_run import ArtifactStore, StageBudget, digest


def _input_control_view(controls, policy, arm):
    if arm == 'camera':
        # Do not require or inspect teacher track/motion metadata for a
        # camera-only baseline, including malformed or absent point labels.
        return inference_control_view(dict(schema=controls.get('schema',CONTROL_SCHEMA),rays=controls['rays']))
    return inference_control_view(controls) if policy == CONTROL_INPUT_POLICY else controls


class ControlRuntime:
    """Explicit extension points; no monkeypatch or mutable model hooks."""
    def __init__(self, arm, control_lr_multiplier=10., control_input_policy=None):
        if arm not in CONTROL_ARMS:
            raise ValueError('Unknown controlled-generation arm: '+str(arm))
        self.arm = arm
        if control_input_policy not in (None, CONTROL_INPUT_POLICY):
            raise ValueError('Unsupported control input policy: '+str(control_input_policy))
        self.control_input_policy = control_input_policy
        self.control_lr_multiplier = float(control_lr_multiplier)
        if not math.isfinite(self.control_lr_multiplier) or self.control_lr_multiplier<=0:
            raise ValueError('Control learning-rate multiplier must be positive and finite')

    def make_model(self, model_args):
        return create_control_model(model_args, self.arm)

    def load_warm(self, model, state):
        return load_warm(model, state)

    def checkpoint_metadata(self):
        result = dict(model_family='SceneFlow' if self.arm=='baseline' else 'ControlledSceneFlow',
                    control_schema=CONTROL_SCHEMA, experiment_arm=self.arm,
                    control_lr_multiplier=self.control_lr_multiplier)
        if self.control_input_policy is not None:
            result['control_input_policy'] = self.control_input_policy
        return result

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
        result = dict(control_camera_rms=rms(features[...,:8]),
            control_motion_rms=rms(features[...,8:16]),
            control_reference_rms=rms(features[...,16:-1]),
            control_track_support_fraction=float((features[...,10]>0).float().mean()),
            control_reference_support_fraction=float((features[...,-1]>0).float().mean()),
            control_reference_support_mean=float(features[...,-1].mean()),
            control_injection_rms=rms(injected),
            control_to_input_rms_ratio=rms(injected)/max(rms(base),1e-8),
            control_lr_multiplier=self.control_lr_multiplier)
        if self.control_input_policy is not None:
            result.update(control_input_policy=self.control_input_policy,
                control_dynamic_mean=float(features[...,12].mean()),
                control_static_mean=float(features[...,13].mean()),
                control_unknown_mean=float(features[...,14].mean()),
                control_weight_semantics='binary_supplied_visibility_not_calibrated_confidence')
        return result

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
            if self.control_input_policy == CONTROL_INPUT_POLICY or self.arm == 'camera':
                # Once per bind, never per denoising step. Immutable source
                # controls and historical configurations retain their values.
                sample = dict(sample, controls=_input_control_view(controls,self.control_input_policy,self.arm))
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
        if controlled and (self.control_input_policy == CONTROL_INPUT_POLICY or self.arm == 'camera'):
            labels = _input_control_view(labels,self.control_input_policy,self.arm)
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
        if self.control_input_policy is not None:
            row['control_input_policy'] = self.control_input_policy
            if controlled:
                row.update(condition_policy='supplied camera rays, point coordinates and binary visibility; observed-only appearance',
                    control_provenance='supplied inputs; a prepared bank may contain offline teacher pseudo-tracks, not autonomous predictions',
                    control_weight_semantics='binary_supplied_visibility_not_calibrated_confidence',
                    appearance_leakage_policy='No future RGB, texture, depth or future-depth-derived dynamic/static labels enters conditions.',
                    motion_state_policy='dynamic/static zero; unknown one everywhere')
        if 'supervision' in row:
            row['supervision']='GT future latent is the denoising target; B/C separately receive explicit future motion controls.'
        if self.arm == 'camera':
            row.update(task='supplied-future-camera conditioned video',
                condition_policy='supplied camera rays only; no point tracks, motion evidence or appearance transport',
                control_provenance='supplied camera rays with observed-prefix scale; no autonomous camera planner',
                motion_state_policy='dynamic/static zero; unknown one everywhere',track_valid_fraction=0.)
            if 'supervision' in row:
                row['supervision']='GT future latent is the denoising target; supplied camera rays are the only future control.'


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
    if cfg['arm'] == 'camera':
        base.update(condition='supplied future camera rays only; no point/motion/reference condition',
            control_status='supplied camera condition; no autonomous camera planner')
    return dict(base, identity=digest(base)), ready


def resolve_control_readiness(controls, manifest, history, *, train_shard_limit=4, control_input_policy=None, camera_only=False):
    """Use a complete preparation report, or explicitly bounded legacy evidence.

    This is an observation of immutable inputs, never part of run identity.
    A bounded scan with no support cannot establish absence across the bank.
    """
    if train_shard_limit < 1:
        raise ValueError('Readiness scan needs at least one training shard')
    if control_input_policy not in (None, CONTROL_INPUT_POLICY):
        raise ValueError('Unsupported control input policy')
    embedded = manifest.get('control_readiness')
    if (control_input_policy is None and not camera_only and isinstance(embedded, dict) and embedded.get('schema') == READINESS_SCHEMA
            and all(isinstance(embedded.get('splits', {}).get(s), dict) for s in ('train', 'eval'))
            and embedded['splits']['train'].get('windows') == manifest['samples']
            and embedded['splits']['eval'].get('windows') == manifest.get('eval_cases')):
        return dict(source='complete_preparation_manifest', summary=copy.deepcopy(embedded),
                    coverage={s:dict(complete=True, scope='all_prepared_windows') for s in ('train', 'eval')})
    shards = manifest['shards']
    count = min(train_shard_limit, len(shards))
    indexes = ([0] if count == 1 else
               [i*(len(shards)-1)//(count-1) for i in range(count)]) if count else []
    records, failures = [], []
    jobs = [('eval', 'data/eval.pt')] + [('train', shards[i]['file']) for i in indexes]
    for split, name in jobs:
        try:
            payload = controls.load(name, manifest['identity'], required=True)
            if control_input_policy == CONTROL_INPUT_POLICY or camera_only:
                payload = dict(payload, samples=[dict(s, controls=_input_control_view(s['controls'],
                    control_input_policy,'camera' if camera_only else 'reference'))
                    for s in payload.get('samples', [])])
            records.extend(scan_prepared_payload(payload, split=split, history=history))
            del payload
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            error = dict(split=split, file=name, error=type(exc).__name__+': '+str(exc))
            failures.append(error)
            records.append(error)
    summary = summarize(records)
    expected = dict(train=manifest['samples'], eval=manifest.get('eval_cases'))
    coverage = {}
    for split in ('train', 'eval'):
        report = summary['splits'][split]
        all_files = split == 'eval' or len(indexes) == len(shards)
        counts_match = expected[split] is None or report['windows'] == expected[split]
        complete = all_files and counts_match and not any(e['split'] == split for e in failures)
        coverage[split] = dict(complete=complete,
            scope='all_prepared_windows' if complete else 'bounded_or_incomplete_scan',
            expected_windows=expected[split], scanned_windows=report['windows'])
        if not complete:
            # Do not promote evidence from four shards to an all-bank claim.
            for name in ('motion_status', 'reference_status', 'camera_status'):
                if report[name] == 'INVALID_ZERO_SUPPORT':
                    report[name] = 'UNVERIFIED_ZERO_IN_BOUNDED_SCAN'
    coverage['train'].update(scanned_shards=len(indexes), total_shards=len(shards),
                             shard_indices=indexes, shard_limit=train_shard_limit)
    result = dict(source='legacy_prepared_tensor_scan', summary=summary, coverage=coverage,
                read_errors=failures, embedded_report_used=False,
                note='No teacher inference or threshold changes. Nonzero support is not a quality certificate.')
    if control_input_policy is not None:
        result.update(source='inference_control_view_tensor_scan',control_input_policy=control_input_policy,
            note='Readiness of the actual v2 view; positive supplied weights become binary visibility. This does not validate teacher tracks.')
    if camera_only:
        result.update(source='camera_only_tensor_scan',camera_only=True,
            note='Only supplied finite camera rays are assessed; missing track/motion labels are irrelevant.')
    return result


def control_readiness_decision(arm, evidence):
    """Require evidence of the advertised training condition, not image quality."""
    if arm not in CONTROL_ARMS:
        raise ValueError('Unknown controlled-generation arm: '+str(arm))
    if arm == 'baseline':
        return dict(action='continue', status='baseline_no_controls_required', reasons=[], warnings=[])
    required = ['camera'] if arm == 'camera' else ['motion'] + (['reference'] if arm == 'reference' else [])
    splits = evidence['summary']['splits']
    def has_signal(report, condition):
        if condition == 'camera':
            return report.get('assessed_windows',0)>0 and report.get('future_camera_present_tokens',0)>0
        if condition == 'reference':
            return report.get('reference_eligible_pairs', 0) > 0
        return (report.get('assessed_windows', 0) > report.get('zero_motion_windows', 0)
                and report.get('future_track_positive', 0) > 0
                and report.get('future_motion_supported_tokens', 0) > 0)
    reasons = [dict(condition=name, split='train',
                    status=splits['train'].get(name+'_status', 'UNAVAILABLE_MISSING_COUNTS'),
                    coverage=evidence['coverage']['train'])
               for name in required if not has_signal(splits['train'], name)]
    warnings = [dict(condition=name, split='eval',
                     status=splits['eval'].get(name+'_status', 'UNAVAILABLE_MISSING_COUNTS'),
                     note='Evaluation cannot establish the effect of this condition on these windows.')
                for name in required if not has_signal(splits['eval'], name)]
    return dict(action='skip' if reasons else 'continue',
        status='skipped_no_training_condition_evidence' if reasons else 'nonzero_training_condition_evidence',
        required_conditions=required, reasons=reasons, warnings=warnings,
        camera_status=splits['train'].get('camera_status'),
        note=('Camera-only requires supplied camera rays; no point/motion/reference support is required.' if arm=='camera' else
              'Camera presence alone does not validate a motion/reference arm. '
              'Individual unsupported windows remain legal; image quality is not gated.'))


def control_startup(controls, manifest, store, plan, cfg, device, rank, *, stage='train'):
    """One leader publishes the decision; every rank takes the same branch.

    Completed historical runs are only inspected. Readiness is deliberately
    outside build_plan so adding this diagnostic does not break old resumes.
    """
    outcome = None
    if rank == 0:
        try:
            old = store.json('contract.json')
            if old and old['identity'] != plan['identity']:
                raise ValueError('Changed source/config/controls: use a fresh namespace')
            complete = store.json('complete.json')
            baseline = store.json(f"eval/{cfg['source_label']}/summary.json")
            historical_done = bool(complete and complete.get('evaluation_status') == 'completed'
                                   and baseline and baseline.get('status') == 'completed')
            if historical_done:
                report = dict(identity=plan['identity'], arm=cfg['arm'], action='historical_complete',
                    status='historical_completed_run_preserved',
                    note='Existing results were not changed or newly certified by readiness checks.')
            else:
                evidence = (dict(source='not_required_for_baseline') if cfg['arm'] == 'baseline'
                            else resolve_control_readiness(controls, manifest, cfg['history'],
                                control_input_policy=cfg.get('control_input_policy'),camera_only=cfg['arm']=='camera'))
                decision = control_readiness_decision(cfg['arm'], evidence)
                report = dict(identity=plan['identity'], control_identity=manifest['identity'],
                    arm=cfg['arm'], readiness=evidence, **decision)
                store.write_json('control_readiness.json', report)
                if decision['action'] == 'skip':
                    store.write_json('skipped_invalid_controls.json', report)
                else:
                    if store.json('skipped_invalid_controls.json'):
                        store.write_json('skipped_invalid_controls.json',
                            dict(report, status='superseded_by_current_continue_decision'))
                    if not old:
                        store.write_json('contract.json', plan)
                    if stage == 'preflight' and store.json('preflight_complete.json'):
                        report['action'] = 'preflight_complete'
            print('[control readiness] '+json.dumps({key:report[key] for key in
                ('arm', 'action', 'status', 'reasons', 'warnings', 'camera_status') if key in report}), flush=True)
            outcome = dict(ok=True, report=report)
        except Exception as exc:
            # A leader-side read/publication failure must not strand the other
            # ranks in a different training collective.
            outcome = dict(ok=False, error=type(exc).__name__+': '+str(exc))
    outcome = broadcast(outcome, device)
    if not outcome['ok']:
        raise RuntimeError('Control readiness startup failed on leader: '+outcome['error'])
    return outcome['report']


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
    runtime = ControlRuntime(a.arm,cfg.get('control_lr_multiplier',10.),cfg.get('control_input_policy'))
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
    startup = control_startup(controls,manifest,store,plan,cfg,device,rank,stage=a.stage)
    if startup['action'] == 'continue':
        ae_path = stage_asset(codec,plan['ae'],local_rank,rank)
        eval_path = stage_asset(controls,plan['evaluation'],local_rank,rank)
        weights_path = stage_asset(source,plan['source_weights'],local_rank,rank)
        train(a,cfg,plan,source,store,manifest,selection,ae_path,weights_path,eval_path,
              budget,device,rank,world,data_store=controls,
              runtime=runtime)
    barrier()
    import train_scene_pipeline as pipeline
    if pipeline.WRITER is not None:
        pipeline.WRITER.close()
        pipeline.WRITER = None
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__=='__main__':
    main()
