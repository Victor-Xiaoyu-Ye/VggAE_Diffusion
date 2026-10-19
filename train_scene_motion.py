"""Matched fixed-window motion-weighting pilot using completed stage53 assets.

No new annotations, RGB cache, encoder training or camera controls. Each arm
starts from the same online weights and uses the same sample/noise RNG schedule.
Stop on supervised validation plateau, never declare visual quality from loss.
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
from train_scene_pipeline import barrier, distributed
from train_scene_prefix import train
from utils.scene_quality_data import load_street_selection
from utils.scene_run import ArtifactStore, StageBudget, digest


def build_plan(source, codec, cfg, selection, world):
    previous=source.json('contract.json',True)
    complete=source.json('complete.json',True)
    manifest=source.json('data/cache_complete.json',True)
    if previous['config']['history']!=cfg['history'] or previous['config']['views']!=cfg['views']:
        raise ValueError('Source prefix and requested prediction window differ')
    if complete['identity']!=previous['identity'] or manifest['identity']!=previous['identity']:
        raise ValueError('Source completion/cache identities differ')
    if previous['street_ids_sha']!=digest(sorted(selection['allowed_ids'])):
        raise ValueError('Source street IDs differ; do not silently change the paired data')
    weights=receipt_spec(source,cfg['source_weights'])
    evaluation=receipt_spec(source,'data/eval.pt')
    ae=receipt_spec(codec,previous['ae']['file'])
    if weights['identity']!=previous['identity'] or evaluation['identity']!=previous['identity']:
        raise ValueError('Source weights or evaluation belong to another prefix run')
    if any(ae[k]!=previous['ae'][k] for k in ('identity','sha256','size')):
        raise ValueError('Frozen codec differs from the source latent cache')
    plan=dict(schema='scene-motion-pilot-v1',config=cfg,world=world,
        source_prefix=previous['identity'],source_weights=weights,ae=ae,
        evaluation=evaluation,model=previous['model'],mean=previous['mean'],std=previous['std'],
        cache_identity=manifest['identity'],cache_manifest_sha=digest(manifest),
        street_ids_sha=previous['street_ids_sha'],warm_start='online only; fresh AdamW and FP32 EMA',
        condition='four observed RGB-derived latents only; no text/camera/GT future',
        comparison='Compare equal-update snapshots; separate convergence times are not matched exposure',
        stopping='Supervised global and high-change x0 errors plateau; not visual quality certification')
    return dict(plan,identity=digest(plan)),manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--cohort',required=True)
    p.add_argument('--selection',required=True); p.add_argument('--encoder')  # launcher compatibility; unused
    p.add_argument('--source-local',required=True); p.add_argument('--source-root',action='append',default=[])
    p.add_argument('--output',required=True); p.add_argument('--root',action='append',default=[])
    p.add_argument('--stage',choices=['preflight','train'],default='train')
    p.add_argument('--startup-seconds',type=float,default=0.)
    a=p.parse_args(); ensure_disjoint(a)
    cfg=json.loads(Path(a.config).read_text(encoding='utf8'))
    arm=os.environ.get('SCENE_MOTION_ARM','motion')
    if arm not in ('control','motion'):
        raise ValueError('SCENE_MOTION_ARM must be control or motion')
    cfg['arm']=arm
    if arm=='control':
        cfg['motion_objective']=dict(cfg['motion_objective'],strength=0.)
    if cfg['history']!=4 or cfg['views']!=9 or cfg['condition_policy']!='observed_prefix_only_no_text_no_camera':
        raise ValueError('This comparison preserves stage53 conditions and targets')
    codec_namespace=cfg['codec_namespace']
    if not codec_namespace.replace('_','').replace('-','').isalnum():
        raise ValueError('Invalid codec namespace')
    selection=load_street_selection(a.cohort,a.selection)
    device,rank,world=distributed()
    if world!=24:
        raise ValueError('The production comparison uses three nodes x eight ranks')
    local_rank=int(os.environ.get('LOCAL_RANK',0)); node=int(os.environ.get('GROUP_RANK',0))
    random.seed(cfg['seed']+rank); np.random.seed(cfg['seed']+rank); torch.manual_seed(cfg['seed']+rank)
    store=ArtifactStore(a.output,a.root)
    source=ArtifactStore(Path(a.output)/'_inputs'/f'node{node}'/'prefix',[a.source_local,*a.source_root])
    codec_roots=[str(Path(a.source_local).parent/codec_namespace/'production')]
    codec_roots += [r.rstrip('/').rsplit('/',1)[0]+'/'+codec_namespace+'/production' for r in a.source_root]
    codec=ArtifactStore(Path(a.output)/'_inputs'/f'node{node}'/'codec',codec_roots)
    progress=store.json('progress.json') or {}
    seconds=cfg['hours']*3600 if cfg['hours'] else math.inf
    budget=StageBudget(seconds,progress.get('total_spent',0.)+max(0.,a.startup_seconds))
    budget.install_signals()
    plan,manifest=build_plan(source,codec,cfg,selection,world)
    if rank==0:
        old=store.json('contract.json')
        if old and old['identity']!=plan['identity']:
            raise ValueError('Changed experiment/source/config: use a fresh namespace')
        store.write_json('contract.json',plan)
    barrier()
    complete=store.json('complete.json')
    evaluation_done=bool(complete and complete.get('evaluation_status')=='completed')
    baseline=store.json(f"eval/{cfg['source_label']}/summary.json")
    artifacts_done=evaluation_done and baseline and baseline.get('status')=='completed'
    if not artifacts_done and not (a.stage=='preflight' and store.json('preflight_complete.json')):
        ae_path=stage_asset(codec,plan['ae'],local_rank,rank)
        eval_path=stage_asset(source,plan['evaluation'],local_rank,rank)
        weights_path=stage_asset(source,plan['source_weights'],local_rank,rank)
        train(a,cfg,plan,source,store,manifest,selection,ae_path,weights_path,eval_path,
              budget,device,rank,world,data_store=source)
    barrier()
    import train_scene_pipeline as pipeline
    if pipeline.WRITER is not None:
        pipeline.WRITER.close(); pipeline.WRITER=None
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__=='__main__':
    main()
