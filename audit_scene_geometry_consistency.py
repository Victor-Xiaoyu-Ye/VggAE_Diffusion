"""Stage58: fixed-representation context/cache and geometry diagnostics.

One independent process on NPU0 per node; no optimizer or HCCL. Reuses the
sealed Stage53 evaluation bank and frozen AE. Numerical findings never act as
a quality gate. Component execution failures are recorded and work continues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

import audit_scene_end_to_end as previous
from utils.native_audit_io import recover_case
from utils.scene_geometry_consistency import geometry_probe, representation_probe
from utils.scene_run import ArtifactStore, digest

PROJECT = Path(__file__).resolve().parent


def encoder_signature(path):
    """Exactly the existing full teacher's bounded checkpoint identity scheme."""
    path=Path(path);size=path.stat().st_size;checksum=hashlib.sha256()
    with path.open('rb') as handle:
        for offset in sorted(set((0,max(0,size//2-524288),max(0,size-1048576)))):
            handle.seek(offset);checksum.update(str(offset).encode()+handle.read(1048576))
    return dict(checkpoint_size=size,checkpoint_sample_sha256=checksum.hexdigest(),
                checkpoint_sample_scheme='offset-prefixed first/middle/last 1MiB')


def check_expected_encoder(actual, cfg):
    if any(actual.get(k)!=v for k,v in cfg.get('teacher_expected',{}).items()):
        raise ValueError('Encoder checkpoint differs from pinned existing StreamVGGT source')


def jobs_for(cfg):
    cases = cfg['cases']
    if not cases or len(set(cases)) != len(cases) or any(type(i) is not int or i < 0 for i in cases):
        raise ValueError('cases must preserve distinct original nonnegative evaluation indices')
    jobs = [dict(key=f'representation/case{i:03d}', kind='representation', case=i) for i in cases]
    for mode in ('real', 'repeated_first'):
        jobs += [dict(key=f'geometry/{mode}/case{i:03d}', kind='geometry', input=mode, case=i) for i in cases]
    return jobs


def make_plan(cfg, source, codec, nodes):
    # Reuse all existing source/AE/cohort receipt checks, then replace ONLY the
    # planned diagnostic jobs. No global monkey-patch of the stage57 module.
    compatible = dict(cfg)
    compatible.setdefault('teacher_cases', cfg['cases'])
    compatible.setdefault('teacher_precisions', ['bfloat16', 'float32'])
    compatible.setdefault('teacher_inputs', ['real', 'repeated_first'])
    fields = previous.make_plan(compatible, source, codec, nodes)
    fields.pop('identity')
    fields.update(schema='scene-geometry-consistency-audit-v1', config=cfg, jobs=jobs_for(cfg),
        encoder_input_policy='sealed_source_eval_raw_uint8_to_float32_div255',
        representation_policy='Production feature extraction and frozen SceneRAE; independent anchor exactly as source cache.',
        geometry_policy='Existing full-head StreamVGGT camera/depth plus independently qualified LK; no neural track scores used.',
        interpretation='Context sensitivity, cached mismatch, geometry residual and perceptual quality remain separate.',
        publication_scope='Diagnostic only; no training or output-target replacement.')
    return dict(fields, identity=digest(fields))


def _phase_callback(store, plan, job):
    start = time.monotonic()
    records = []
    files = []
    def save(name, payload, details):
        relative = job['key'] + '/intermediate/' + name + '.pt'
        store.save(relative, payload, plan['identity'])
        files.extend([relative, relative + '.json'])
        records.append(dict(phase=name, **details))
        elapsed = max(time.monotonic() - start, 1e-6)
        processed = sum(r.get('input_frames', 0) for r in records)
        status = dict(status='running', phases=records, input_RGB_frame_evaluations=processed,
            elapsed_seconds=elapsed, DI_throughput=processed/elapsed,
            DI_unit='RGB_forward_frame_evaluations_per_second_including_completed_phase_compute_and_artifact_IO; excludes_model_setup')
        store.write_json(job['key'] + '/progress.json', status)
        print('[geometry audit] ' + json.dumps(dict(job=job['key'], phase=name,
              processed_frames=processed, DI_throughput=processed/elapsed), ensure_ascii=False), flush=True)
    return save, files


def _save_difference_images(store, key, maps):
    from PIL import Image, ImageDraw
    files = []
    for comparison in ('cached_vs_reencoded_full', 'full_vs_independent_prefix_raw',
                       'full_vs_changed_future_raw_prefix'):
        rows = maps[comparison]
        frames, tokens = rows['geo'].shape
        side = math.isqrt(tokens)
        if side * side != tokens:
            raise ValueError('Expected square spatial latent grid for difference maps')
        image = Image.new('RGB', (max(640, frames*76), 210), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((4, 4), comparison, fill='black')
        for row, branch in enumerate(('geo','texture')):
            values = rows[branch].float().numpy()
            bound = float(np.quantile(values, .99))
            draw.text((4, 22 + row*92), f'{branch}: mean abs standardized error; shared p99={bound:.6g}', fill='black')
            for frame, value in enumerate(values):
                normalized = np.clip(value.reshape(side, side)/max(bound, 1e-12), 0, 1)
                rgb = np.stack((normalized, normalized**2, np.zeros_like(normalized)), -1)
                tile = Image.fromarray(np.rint(rgb*255).astype(np.uint8)).resize((72, 64), Image.Resampling.NEAREST)
                image.paste(tile, (frame*76, 38 + row*92))
                draw.text((frame*76+2, 104 + row*92), f'f{frame}', fill='black')
        name = key + '/' + comparison + '.png'
        path = store.local/name; path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path); files.append(name)
    return files


def representation_action(store, plan, job, sample, encoder, rae, device):
    from train_scene_pipeline import amp, features, video
    from train_scene_prefix import labelled_preview
    frames = previous.sealed_rgb(sample)
    save_phase, files = _phase_callback(store, plan, job)
    levels = list(rae.config.levels)

    @torch.no_grad()
    def encode(rgb):
        image = rgb[None].to(device)
        tokens = features(encoder, image, device)
        with amp(device):
            z = rae.encode(tokens, image).flatten(2, 3)[0].float().cpu()
        # Only four common-prefix frames of the four production fusion levels;
        # exact token comparison uses these CPU arrays, never pooled surrogates.
        selected = {int(level):tokens[level][0, :4].detach().cpu() for level in levels}
        return dict(latent=z, tokens=selected)

    @torch.no_grad()
    def decode(latent):
        with amp(device):
            return rae.decode(latent.to(device).reshape(1,-1,rae.grid,rae.grid,rae.channels))[0].float().cpu()

    report, artifacts = representation_probe(frames, sample, plan['mean'], plan['std'], encode, decode,
                                             on_phase=save_phase)
    report.update(case=job['case'], id=sample['id'], frame_ids=sample['frame_ids'],
        selected_token_layers=levels, original_cache_dtype=str(sample['target'].dtype),
        sealed_RGB_sha256=hashlib.sha256(torch.as_tensor(sample['raw']).contiguous().numpy().tobytes()).hexdigest(),
        stage53_source_weights_receipt=plan['weights'], source_cache_unchanged=True)
    name = job['key'] + '/latents.pt'
    store.save(name, {k:v for k,v in artifacts.items() if k != 'reconstructions'}, plan['identity'])
    files += [name, name+'.json', job['key']+'/progress.json']
    metrics = job['key'] + '/core_metrics.json'
    store.write_json(metrics, report); files.append(metrics)
    report['media_errors'] = []
    try:
        files += _save_difference_images(store, job['key'], artifacts['difference_maps'])
    except Exception as exc:
        report['media_errors'].append(dict(artifact='difference_maps',error=repr(exc)))
    r = artifacts['reconstructions']; raw = frames.permute(0,2,3,1)
    previews = [
        ('full_reconstruction', torch.cat((raw,r['cached'],r['reencoded_full']),2),
         ['SEALED RAW','ORIGINAL CACHED AE','REENCODED AE']),
        ('common_prefix', torch.cat((raw[:4],r['reencoded_full'][:4],r['independent_prefix'],r['changed_future_prefix']),2),
         ['OBSERVED RAW','FULL9 ENCODE PREFIX','INDEPENDENT PREFIX4','CHANGED FUTURE PREFIX']),
    ]
    for label, rgb, titles in previews:
        try:
            relative=job['key']+'/'+label+'.mp4'
            video(store.local/relative, labelled_preview(rgb,titles,4),fps=8)
            files.append(relative)
        except Exception as exc:
            report['media_errors'].append(dict(artifact=label,error=repr(exc)))
    report['DI_scope_detail'] = ('Final DI numerator counts 23 RGB input-frame evaluations '
        '(full9+prefix4+intervened9+anchor1); denominator includes component setup, encoding, '
        'reconstruction, metrics and intermediate IO, but excludes the final report/seal publication. '
        'Unique source frames=9; this is not generation or training throughput.')
    return report, files, report['encoder_input_frames']


def geometry_action(store, plan, job, sample, teacher):
    from utils.scene_lk_probe import probe_lk
    frames = previous.sealed_rgb(sample)
    repeated = job['input'] == 'repeated_first'
    if repeated:
        frames = frames[:1].expand_as(frames).clone()
    save_phase, files = _phase_callback(store, plan, job)
    outputs, lk, lk_reports = {}, {}, {}
    for name, rgb in [('full',frames),('prefix',frames[:4])]:
        outputs[name] = teacher.infer(rgb)
        save_phase('teacher_'+name, outputs[name], dict(input_frames=len(rgb),
            precision=teacher.metadata['precision'], neural_tracks_used=False))
        lk[name], lk_reports[name] = probe_lk(rgb,outputs[name])
        save_phase('lk_'+name, {k:lk[name][k] for k in ('tracks','visibility','track_confidence','query_points')},
                   dict(input_frames=len(rgb),status=lk_reports[name]['status']))
    report, compact = geometry_probe(outputs['full'],outputs['prefix'],lk['full'],repeated=repeated)
    report.update(case=job['case'], id=sample['id'], frame_ids=sample['frame_ids'],
        teacher_metadata=teacher.metadata, lk_reports=lk_reports,
        input_frames_identical=bool(torch.equal(frames,frames[:1].expand_as(frames))),
        neural_tracks_used=False, unique_sealed_RGB_frames=9,
        input_RGB_forward_frame_evaluations=26,
        forward_scope='13 full/prefix neural RGB-frame evaluations plus 13 LK image-frame evaluations; LK pair calls and decoder frames are not additional numerator items.')
    # Let the shared runner distinguish an actual LK execution failure from a
    # legitimate zero-support observation without adding a quality threshold.
    report['lk_comparison'] = dict(status='completed' if all(
        r['status']=='completed' and not r['errors'] for r in lk_reports.values()) else 'failed')
    artifact=job['key']+'/geometry.pt'
    store.save(artifact,dict(full=lk['full'],prefix=lk['prefix'],measurements=compact,
                            repeated_input=repeated),plan['identity'])
    files += [artifact,artifact+'.json',job['key']+'/progress.json']
    metrics=job['key']+'/core_metrics.json'
    store.write_json(metrics,report);files.append(metrics)
    report['media_errors']=[]
    try:
        files.append(previous.track_preview(store,job['key']+'/lk',frames,lk['full']))
    except Exception as exc:
        report['media_errors'].append(dict(artifact='LK support preview',error=repr(exc)))
    report['DI_scope_detail']='Numerator=26 full/prefix teacher+LK input-frame evaluations; wall time includes setup, all diagnostics and intermediate IO but excludes final report/seal publication; not training throughput.'
    return report, files, 26


def work(args, cfg, plan, store, source, codec, node, nodes):
    samples=previous.checked_load(source,plan['evaluation'])['samples']
    if max(cfg['cases']) >= len(samples):
        raise ValueError('Selected original evaluation index exceeds sealed bank')
    name=args.device
    if name.startswith('npu'):
        import torch_npu  # noqa:F401
        name='npu:0'
        torch.npu.set_device(name)
    device=torch.device(name)
    from utils.device import configure_backend_compatibility
    configure_backend_compatibility(device.type)
    assigned=[j for i,j in enumerate(plan['jobs']) if i%nodes==node]
    encoder=rae=teacher=None
    memory=[]
    for job in assigned:
        sample=samples[job['case']]
        def action():
            nonlocal encoder,rae,teacher
            check_expected_encoder(plan['encoder'],cfg)
            if job['kind']=='representation':
                teacher=None;previous.release_device(device)
                if encoder is None:
                    from train_scene_pipeline import make_encoder
                    encoder=make_encoder(args.encoder,device)
                if rae is None:
                    from models.scene_rae import SceneRAE
                    checkpoint=previous.checked_load(codec,plan['ae'])
                    rae=SceneRAE(checkpoint['spec']['source_config'],checkpoint['spec']['channels'])
                    rae.load_state_dict(checkpoint['model'],strict=True)
                    rae=rae.eval().requires_grad_(False).to(device)
                result=representation_action(store,plan,job,sample,encoder,rae,device)
                result[0]['encoder_checkpoint_identity']=plan['encoder']
                return result
            encoder=rae=None;previous.release_device(device)
            if teacher is None:
                from utils.scene_trackmask_teacher import VGGTTrackTeacher
                teacher=VGGTTrackTeacher(args.encoder,device=device,backend='streamvggt',
                    precision=cfg.get('geometry_precision','bfloat16'),query_stride=cfg.get('query_stride',21),
                    query_margin=cfg.get('query_margin',7),query_chunk_size=cfg.get('query_chunk_size',0))
                if any(teacher.metadata.get(k)!=v for k,v in cfg.get('teacher_expected',{}).items()):
                    teacher=None
                    raise ValueError('Teacher checkpoint differs from pinned source; label a separate comparison')
            return geometry_action(store,plan,job,sample,teacher)
        previous.run_component(store,plan['identity'],job,action,node)
        if device.type in ('npu','cuda'):
            try:
                backend=getattr(torch,device.type)
                memory.append(dict(job=job['key'],allocated_GiB=backend.memory_allocated(device)/1024**3,
                    reserved_GiB=backend.memory_reserved(device)/1024**3,
                    process_peak_allocated_GiB=backend.max_memory_allocated(device)/1024**3))
            except Exception as exc:
                memory.append(dict(job=job['key'],unavailable=repr(exc)))
            store.write_json(f'nodes/node{node}/memory.json',memory)
        previous.release_device(device)
    store.write_json(f'nodes/node{node}/work_complete.json',dict(identity=plan['identity'],
        jobs=[j['key'] for j in assigned],status='all_assigned_components_attempted',memory=memory))
    return 0


def merge(store, plan, nodes):
    reports={}; node_errors=[]
    for node in range(nodes):
        try:
            contract=store.json(f'nodes/node{node}/contract.json',True)
            if contract['identity']!=plan['identity']:
                raise ValueError('Worker source/config/code identity differs')
        except Exception as exc:
            node_errors.append(dict(node=node,error=repr(exc)))
    for job in plan['jobs']:
        try:
            report=store.json(job['key']+'/report.json',True)
            if not isinstance(report,dict) or report.get('identity')!=plan['identity']:
                raise ValueError('Component report identity differs')
            if report.get('execution_status')=='completed' and recover_case(
                    store.local,job['key']+'/done.json',plan['identity'],store.roots) is None:
                raise ValueError('Completed component has no valid artifact seal')
            reports[job['key']]=report
        except Exception as exc:
            reports[job['key']]=dict(execution_status='failed',error=repr(exc))
    failed=[j['key'] for j in plan['jobs'] if reports[j['key']].get('execution_status')!='completed']
    summary=dict(schema='scene-geometry-consistency-summary-v1',identity=plan['identity'],
        execution_status='completed' if not failed and not node_errors else 'incomplete_or_failed_components',
        failed_or_missing=failed,node_contract_errors=node_errors,components=reports,
        quality_status='requires_review_of_context_cache_and_geometry_evidence',
        quality_gate_applied=False, numerical_findings_do_not_block_subsequent_training=True,
        interpretation='Execution success is not geometry accuracy or generation quality; cached targets were not replaced.')
    store.write_json('summary.json',summary)
    return 0 if summary['execution_status']=='completed' else 2


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['work','merge'],required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--local','--output',dest='output',required=True)
    p.add_argument('--root',action='append',default=[])
    p.add_argument('--source-local',required=True)
    p.add_argument('--source-root',action='append',default=[])
    p.add_argument('--encoder',required=True);p.add_argument('--device',default='npu')
    p.add_argument('--node-rank',type=int);p.add_argument('--nodes',type=int)
    return p


def main():
    args=parser().parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    node=args.node_rank if args.node_rank is not None else int(os.environ.get('NODE_RANK',os.environ.get('GROUP_RANK',0)))
    nodes=args.nodes if args.nodes is not None else int(os.environ.get('NNODES',1))
    if not 0<=node<nodes:
        raise ValueError('Invalid node rank/count')
    local=Path(args.output).resolve();source_local=Path(args.source_local).resolve()
    if local==source_local or local in source_local.parents or source_local in local.parents or set(args.root)&set(args.source_root):
        raise ValueError('Diagnostic output must be separate from read-only source')
    store=ArtifactStore(local,args.root);store.status_name=f'publication_status-node{node}.json'
    source=ArtifactStore(local/'_inputs'/f'node{node}'/'source',[args.source_local,*args.source_root])
    namespace=cfg['codec_namespace']
    if not namespace.replace('_','').replace('-','').isalnum():
        raise ValueError('Invalid codec namespace')
    codec=ArtifactStore(local/'_inputs'/f'node{node}'/'codec',[str(source_local.parent/namespace/'production')]+
        [r.rstrip('/').rsplit('/',1)[0]+'/'+namespace+'/production' for r in args.source_root])
    plan=make_plan(cfg,source,codec,nodes)
    fields={k:v for k,v in plan.items() if k!='identity'}
    fields['encoder']=encoder_signature(args.encoder)
    plan=dict(fields,identity=digest(fields))
    if args.stage=='work':
        previous.pin_plan(store,plan,node)
        return work(args,cfg,plan,store,source,codec,node,nodes)
    return merge(store,plan,nodes)


if __name__=='__main__':
    raise SystemExit(main())
