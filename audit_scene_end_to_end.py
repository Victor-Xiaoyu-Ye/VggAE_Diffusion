"""Stage57: sealed, independent teacher/data/codec/generator diagnostics.

One process per node, no HCCL. No optimizer, new weight download, threshold
change or modification to source assets. Component failures remain reports;
merge fails for missing/failed work but zero support is a functional finding.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

import torch

from audit_scene_flow_weights import heartbeat, receipt_spec
from utils.native_audit_io import recover_case, seal
from utils.scene_run import ArtifactStore, digest

PROJECT = Path(__file__).resolve().parent


def jobs_for(cfg):
    cases = cfg['cases']
    if not cases or len(set(cases)) != len(cases) or any(type(i) is not int or i < 0 for i in cases):
        raise ValueError('cases must be distinct original nonnegative evaluation indices')
    if not set(cfg['teacher_cases']) <= set(cases):
        raise ValueError('Teacher cases must be in the sealed evaluation selection')
    if cfg['teacher_precisions'] != ['bfloat16', 'float32']:
        raise ValueError('Keep the paired BF16/FP32 diagnostic')
    if cfg['teacher_inputs'] != ['real', 'repeated_first']:
        raise ValueError('Keep real/repeated-frame controls')
    jobs = []
    if cfg.get('replay', True):
        jobs += [dict(key=f'replay/case{i:03d}', kind='replay', case=i) for i in cases]
    for precision in cfg['teacher_precisions']:
        for kind in cfg['teacher_inputs']:
            jobs += [dict(key=f'teacher/{precision}/{kind}/case{i:03d}', kind='teacher',
                          case=i, precision=precision, input=kind) for i in cfg['teacher_cases']]
    jobs += [dict(key=f'generation/case{i:03d}', kind='generation', case=i) for i in cases]
    return jobs


def code_signature():
    # Full repository Python source hashes pin imported model/IO/data behavior,
    # not just the new entry point. Tests and runtime outputs are excluded.
    paths = [p for folder in ('models', 'utils', 'data', 'streamvggt', 'scripts', 'third_party/streamvggt_trackmask')
             for p in (PROJECT/folder).rglob('*.py')]
    paths += list(PROJECT.glob('*.py')) + [PROJECT/'scripts/audit_scene_control_teacher.py']
    return digest({p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(set(paths))})


def runtime_versions():
    import importlib.metadata
    versions={'python':platform.python_version(),'torch':str(torch.__version__)}
    for distribution in ('torch-npu','numpy','Pillow','opencv-python','opencv-python-headless','imageio','imageio-ffmpeg'):
        try:
            versions[distribution]=importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution]=None
    return versions


def make_plan(cfg, source, codec, nodes):
    prior = source.json('contract.json', True)
    manifest = source.json('data/cache_complete.json', True)
    done = source.json('complete.json', True)
    if any(x['identity'] != prior['identity'] for x in (manifest, done)):
        raise ValueError('Source cache/completion contract differs')
    if prior['config']['history'] != 4 or prior['config']['views'] != 9:
        raise ValueError('Audit expects the existing observed4/future5 task')
    evaluation = receipt_spec(source, 'data/eval.pt')
    weights = receipt_spec(source, cfg['source_weights'])
    ae = receipt_spec(codec, prior['ae']['file'])
    if any(x['identity'] != prior['identity'] for x in (evaluation, weights)):
        raise ValueError('Evaluation/weights differ from the source identity')
    if any(ae[k] != prior['ae'][k] for k in ('identity', 'size', 'sha256')):
        raise ValueError('Frozen AE differs from the cached representation')
    cohort = PROJECT/cfg['cohort']
    cohort_sha = hashlib.sha256(cohort.read_bytes()).hexdigest()
    if prior.get('cohort_sha256') != cohort_sha:
        raise ValueError('Cohort differs from the frozen source provenance')
    fields = dict(schema=cfg['schema'], config=cfg, nodes=nodes, jobs=jobs_for(cfg),
        source_identity=prior['identity'], source_contract_sha=digest(prior),
        evaluation=evaluation, weights=weights, ae=ae, model=prior['model'],
        mean=prior['mean'], std=prior['std'], code_sha256=code_signature(), cohort_sha256=cohort_sha,
        runtime_versions=runtime_versions(),
        interpretation='Execution, functional condition support and visual quality are separate.',
        publication_scope='Inference diagnosis only; no claim of improved generation.')
    return dict(fields, identity=digest(fields))


def checked_load(store, spec):
    payload = store.load(spec['file'], spec['identity'], True)
    got = receipt_spec(store, spec['file'])
    if got != spec:
        raise ValueError('Input receipt changed after contract pinning')
    return payload


def pin_plan(store, plan, node):
    # Every node owns a separate contract file: no simultaneous publication.
    name = f'nodes/node{node}/contract.json'
    old = store.json(name)
    if old and old['identity'] != plan['identity']:
        raise ValueError('Changed code/config/source; use a fresh SCENE_AUDIT_NAMESPACE')
    store.write_json(name, plan)


def run_component(store, identity, job, action, node):
    """Resume only content-verified successes; failures retry next invocation."""
    key, started = job['key'], time.monotonic()
    recovery_error=None
    try:
        completed = recover_case(store.local, key+'/done.json', identity, store.roots)
        if completed:
            # Republish on resume to repair a previously degraded second replica.
            store.publish([*completed['files'], key+'/done.json'])
            return store.json(key+'/report.json', True)
    except Exception as exc:
        recovery_error=repr(exc)
        # A damaged previously sealed case cannot kill other independent jobs.
        # Recompute it using the pinned source, then produce a fresh content seal.
    try:
        with heartbeat('stage57 '+key, node):
            report, files, frames = action()
        nested_failed=(report.get('lk_comparison',{}).get('status')=='failed' or bool(report.get('media_errors')))
        report = dict(report, execution_status='failed' if nested_failed else 'completed')
    except Exception as exc:
        report, files, frames = dict(execution_status='failed', error=repr(exc),
            traceback=traceback.format_exc(limit=15)), [], 0
    seconds = max(time.monotonic()-started, 1e-6)
    report.update(job=job, node=node, identity=identity, elapsed_seconds=seconds,
        processed_frames=frames, DI_throughput=frames/seconds,
        DI_unit='diagnostic_input_RGB_frames_per_second_including_instrumentation_and_artifact_IO')
    if recovery_error:
        report['recovery_error_before_retry']=recovery_error
    name = key+'/report.json'
    store.write_json(name, report)
    if report['execution_status'] == 'completed':
        seal(store.local, key+'/done.json', identity, [*files, name], dict(job=job, node=node))
        store.publish([*files, name, key+'/done.json'])
    print('[end-to-end audit] '+json.dumps({k:report[k] for k in
        ('job','execution_status','elapsed_seconds','DI_throughput','DI_unit')}, ensure_ascii=False), flush=True)
    return report


def sealed_rgb(sample):
    raw = torch.as_tensor(sample['raw'])
    if raw.ndim != 4 or raw.shape[:2] != (9,3) or raw.dtype != torch.uint8:
        raise ValueError('Expected original sealed uint8 RGB[9,3,H,W], not a preview')
    return raw.float()/255


def reference_probe(sample, controls, mean, std):
    """Score observed-only transport; future target appears ONLY in metrics."""
    from utils.scene_control import transport_prefix_texture
    from utils.scene_prefix import prepare_prefix
    mu=torch.as_tensor(mean).float().flatten()
    sigma=torch.as_tensor(std).float().flatten().clamp_min(1e-4)
    batch=prepare_prefix(sample,mu,sigma,hist=4,device=torch.device('cpu'))
    observed=torch.cat((batch.anchor,batch.known),1)
    half=observed.shape[-1]//2
    def warp(xy,weight):
        return transport_prefix_texture(observed,controls['prefix_tracks'][None],
            controls['prefix_track_weight'][None],xy[None],weight[None])
    destinations=controls['tracks'][4:]
    weight=controls['track_weight'][4:]
    ref=warp(destinations,weight)
    # Same source, wrong destination: does correspondence beat generic copying?
    wrong=warp(destinations.roll(1,1),weight.roll(1,1))
    mask=ref[...,-1]>0
    truth=batch.target[...,half:]
    copy=observed[:,-1:,:,half:].expand_as(truth)
    def errors(value,selected):
        if not bool(selected.any()):
            return None
        delta=(value-truth)[selected]
        return dict(l1=float(delta.abs().mean()),mse=float(delta.square().mean()))
    joint=mask & (wrong[...,-1]>0)
    dynamic=controls['motion'][4:,:,4][None]>0
    return dict(status='supported' if bool(mask.any()) else 'no_reference_support',
        units='original_training_standardized_texture_channels',
        condition='observed4 appearance only; supplied future point coordinates',
        future_gt_used_for='scoring only; never transport source',
        supported_tokens=int(mask.sum()),total_tokens=mask.numel(),
        reference=errors(ref[...,:-1],mask),copy_last=errors(copy,mask),
        same_support_wrong_destination_comparison=dict(tokens=int(joint.sum()),
            correct=errors(ref[...,:-1],joint),wrong=errors(wrong[...,:-1],joint),copy=errors(copy,joint)),
        dynamic_pseudo_region=dict(tokens=int((mask&dynamic).sum()),
            reference=errors(ref[...,:-1],mask&dynamic),copy=errors(copy,mask&dynamic)),
        limitation='Sparse/pseudo-supported subset only; not a calibrated dynamic mask or full-video quality metric.')


def track_preview(store, key, frames, outputs):
    import cv2
    import numpy as np
    from train_scene_pipeline import video
    from train_scene_prefix import labelled_preview
    data=np.ascontiguousarray((frames.permute(0,2,3,1).float().clamp(0,1).numpy()*255).round().astype(np.uint8))
    xy=torch.as_tensor(outputs['tracks']).float().cpu().numpy()
    good=((torch.as_tensor(outputs['visibility'])>=.5)&
          (torch.as_tensor(outputs['track_confidence'])>=.2)).cpu().numpy()
    # Accepted correspondences are green; rejected query positions red. This
    # overlay does NOT label moving objects and never supplies training RGB.
    for t,frame in enumerate(data):
        for q in range(0,len(xy[t]),4):
            x,y=xy[t,q]
            if np.isfinite([x,y]).all() and 0<=x<frame.shape[1] and 0<=y<frame.shape[0]:
                cv2.circle(frame,(round(float(x)),round(float(y))),2,
                    (0,255,0) if good[t,q] else (255,0,0),-1)
    rgb=labelled_preview(torch.from_numpy(data).float()/255,
        ['TRACK SUPPORT green=accepted red=rejected; NOT object mask'],4)
    name=key+'/tracks.mp4'
    video(store.local/name,rgb,8)
    return name


def replay_action(sample, row, localize):
    from utils.scene_control_data import replay_rgb
    # Deliberately bypass only the replay equality assertion to MEASURE the
    # mismatch; replayed frames never replace source targets or teacher inputs.
    original = sealed_rgb(sample)
    replayed, provenance = replay_rgb({k:v for k,v in sample.items() if k!='raw'}, row,
                                      localize, size=original.shape[-1])
    delta = ((replayed*255).round()-(original*255).round()).abs()
    return dict(functional_status='exact_match' if not bool(delta.any()) else 'rgb_replay_mismatch',
        provenance=provenance, original_id=sample['id'], frame_ids=sample['frame_ids'],
        maximum_uint8_error=float(delta.max()), mean_uint8_error=float(delta.mean()),
        changed_channel_fraction=float((delta>0).float().mean()),
        per_frame_mean_uint8_error=delta.flatten(1).mean(1).tolist(),
        policy='Mismatch measured, not accepted for training; teacher uses sealed RGB.'), [], len(original)


def teacher_action(store, plan, job, sample, teacher):
    from utils.scene_teacher_probe import probe_teacher
    from utils.scene_control_data import build_controls, window_key
    from utils.scene_control_readiness import assess_controls
    from utils.scene_control_inference import inference_control_view
    frames = sealed_rgb(sample)
    if job['input']=='repeated_first':
        frames=frames[:1].expand_as(frames).clone()
    results = {name:probe_teacher(teacher, rgb, job['case'], job['input']+'/'+name)
               for name,rgb in (('full',frames),('prefix',frames[:4]))}
    # Save partial hook evidence even if a later head failed.
    report = dict(probes={name:r['report'] for name,r in results.items()},
                  input_source='original_sealed_eval_raw_not_preview', tracking_kind='StreamVGGT')
    outputs={name:r['outputs'] for name,r in results.items()}
    name=job['key']+'/probe.json'
    store.write_json(name, report)
    if any(v is None for v in outputs.values()):
        raise RuntimeError('Teacher forward failed; see saved probe.json for layer/logit evidence')
    controls = build_controls(outputs['full'],outputs['prefix'],history=4,grid=plan['model']['grid'])
    report['readiness']=assess_controls(controls,split='eval',window_key=window_key(sample),teacher=outputs)
    portable=inference_control_view(controls)
    report['inference_control_view_v2']=dict(readiness=assess_controls(portable,split='eval'),
        policy=portable['metadata'])
    if job['input']=='real':
        report['reference_probe']=reference_probe(sample,controls,plan['mean'],plan['std'])
        report['inference_control_view_v2']['reference_probe']=reference_probe(sample,portable,plan['mean'],plan['std'])
    report['motion_diagnostics']=controls['metadata']['teacher_motion_diagnostics']
    report['functional_status']='scored_not_quality_certified'
    files=[name]
    artifact=job['key']+'/teacher.pt'
    store.save(artifact,dict(**outputs,controls=controls,inference_controls_v2=portable,
        window_key=window_key(sample)),plan['identity'])
    files += [artifact,artifact+'.json']
    previews=[(job['key'],outputs['full'])]
    if plan['config'].get('lk_comparison', True):
        try:
            from utils.scene_lk_probe import probe_lk
            lk_full,full_report=probe_lk(frames,outputs['full'])
            lk_prefix,prefix_report=probe_lk(frames[:4],outputs['prefix'])
            lk_controls=build_controls(lk_full,lk_prefix,history=4,grid=plan['model']['grid'])
            lk_portable=inference_control_view(lk_controls)
            lk_ok=all(r.get('status')=='completed' and not r.get('errors') for r in (full_report,prefix_report))
            report['lk_comparison']=dict(status='completed' if lk_ok else 'failed',full=full_report,prefix=prefix_report,
                readiness=assess_controls(lk_controls,split='eval',window_key=window_key(sample)),
                motion_diagnostics=lk_controls['metadata']['teacher_motion_diagnostics'],
                policy='Diagnostic alternative only; unchanged neural camera/depth, no automatic training replacement.')
            report['lk_comparison']['inference_control_view_v2']=dict(
                readiness=assess_controls(lk_portable,split='eval'),policy=lk_portable['metadata'])
            if job['input']=='real':
                report['lk_comparison']['reference_probe']=reference_probe(sample,lk_controls,plan['mean'],plan['std'])
                report['lk_comparison']['inference_control_view_v2']['reference_probe']=reference_probe(sample,lk_portable,plan['mean'],plan['std'])
            lk_name=job['key']+'/lk.pt'
            store.save(lk_name,dict(full=lk_full,prefix=lk_prefix,controls=lk_controls,
                                  inference_controls_v2=lk_portable),plan['identity'])
            files += [lk_name,lk_name+'.json']
            previews.append((job['key']+'/lk',lk_full))
        except Exception as exc:
            report['lk_comparison']=dict(status='failed',error=repr(exc),traceback=traceback.format_exc(limit=8))
    metrics=job['key']+'/core_metrics.json'
    store.write_json(metrics,report)
    files.append(metrics)
    report['media_errors']=[]
    for key,tracked in previews:
        try:
            files.append(track_preview(store,key,frames,tracked))
        except Exception as exc:
            report['media_errors'].append(dict(key=key,error=repr(exc)))
    return report, files, 13


def generation_action(store, plan, job, sample, donor, model, rae, device):
    from utils.scene_generation_audit import audit_generation_case
    from train_scene_pipeline import video
    from train_scene_prefix import labelled_preview
    report,artifacts=audit_generation_case(model,rae,sample,plan['mean'],plan['std'],device,
        seed=plan['config']['seed'],steps=plan['config']['sample_steps'],matched_other_sample=donor,
        model_label='stage53_online31069')
    metrics=job['key']+'/core_metrics.json'
    store.write_json(metrics,report)
    latent=job['key']+'/generated_latent.pt'
    store.save(latent,dict(latent=artifacts['generated_future_latent'],id=sample['id'],
        frame_ids=sample['frame_ids'][4:],seed=plan['config']['seed']),plan['identity'])
    files=[metrics,latent,latent+'.json']
    report['media_errors']=[]
    for name in ('preview','GT_FUTURE_GEO_ORACLE','GT_FUTURE_TEXTURE_ORACLE'):
        try:
            rgb=artifacts[name]
            if name=='preview':
                rgb=labelled_preview(rgb,report['preview_columns'],4)
            else:
                rgb=labelled_preview(rgb,[name+' (FUTURE GT; DIAGNOSTIC ONLY)'],4)
            relative=job['key']+'/'+name+'.mp4'
            video(store.local/relative,rgb,fps=8)
            files.append(relative)
        except Exception as exc:
            report['media_errors'].append(dict(artifact=name,error=repr(exc)))
    return report, files, 9


def release_device(device):
    gc.collect()
    if device.type in ('cuda','npu'):
        getattr(torch,device.type).empty_cache()


def work(a, cfg, plan, store, source, codec, node, nodes):
    payload=checked_load(source,plan['evaluation'])
    samples=payload['samples']
    if max(cfg['cases']) >= len(samples):
        raise ValueError('Selected original index exceeds source evaluation bank')
    assigned=[job for i,job in enumerate(plan['jobs']) if i%nodes==node]
    device_name=a.device
    if device_name.startswith('npu'):
        import torch_npu  # noqa: F401; register device before torch.device
        device_name='npu:0' if device_name=='npu' else device_name
        torch.npu.set_device(device_name)
    device=torch.device(device_name)
    from utils.device import configure_backend_compatibility
    configure_backend_compatibility(device.type)
    cohort=json.loads((PROJECT/cfg['cohort']).read_text(encoding='utf8'))
    from data.scene_rgb import SceneRGB
    reader=SceneRGB(cohort)
    rows={r['id']:r for r in cohort['records']}
    model=rae=teacher=None
    current_precision=None
    teacher_error=generation_error=None
    memory=[]
    for job in assigned:
        sample=samples[job['case']]
        def action():
            nonlocal model,rae,teacher,current_precision,teacher_error,generation_error
            if job['kind']=='replay':
                return replay_action(sample,rows[sample['id']],reader.local)
            if job['kind']=='teacher':
                if current_precision!=job['precision']:
                    teacher=None; release_device(device)
                    current_precision=job['precision']; teacher_error=None
                    try:
                        from utils.scene_trackmask_teacher import VGGTTrackTeacher
                        teacher=VGGTTrackTeacher(a.encoder,device=device,backend='streamvggt',
                            precision=current_precision,query_stride=cfg['query_stride'],
                            query_margin=cfg['query_margin'],query_chunk_size=cfg['query_chunk_size'])
                        if any(teacher.metadata.get(k)!=v for k,v in cfg['teacher_expected'].items()):
                            raise ValueError('Teacher differs from the actual failed-run checkpoint; use a separately labelled comparison')
                    except Exception as exc:
                        teacher_error=repr(exc)
                if teacher_error:
                    raise RuntimeError('Teacher setup unavailable: '+teacher_error)
                return teacher_action(store,plan,job,sample,teacher)
            teacher=None; release_device(device)
            if model is None and generation_error is None:
                try:
                    from models.scene_flow import SceneFlow
                    from models.scene_rae import SceneRAE
                    warm=checked_load(source,plan['weights'])
                    if warm.get('model_args')!=plan['model'] or warm.get('step')!=31069:
                        raise ValueError('Source checkpoint internal architecture/step mismatch')
                    model=SceneFlow(**plan['model'])
                    model.load_state_dict(warm['model'],strict=True)
                    model=model.eval().requires_grad_(False).to(device)
                    del warm
                    ae=checked_load(codec,plan['ae'])
                    rae=SceneRAE(ae['spec']['source_config'],ae['spec']['channels'])
                    rae.load_state_dict(ae['model'],strict=True)
                    rae=rae.eval().requires_grad_(False).to(device)
                    del ae
                except Exception as exc:
                    generation_error=repr(exc); model=rae=None; release_device(device)
            if generation_error:
                raise RuntimeError('Frozen generation setup unavailable: '+generation_error)
            donors=[s for s in samples if s['id']!=sample['id'] and s['target'].shape==sample['target'].shape]
            if not donors:
                raise ValueError('No cross-scene same-codec donor in source eval bank')
            return generation_action(store,plan,job,sample,donors[0],model,rae,device)
        run_component(store,plan['identity'],job,action,node)
        if device.type in ('npu','cuda'):
            try:
                backend=getattr(torch,device.type)
                memory.append(dict(job=job['key'],allocated_GiB=backend.memory_allocated(device)/1024**3,
                    reserved_GiB=backend.memory_reserved(device)/1024**3,
                    process_peak_allocated_GiB=backend.max_memory_allocated(device)/1024**3))
            except Exception as exc:
                memory.append(dict(job=job['key'],unavailable=repr(exc)))
            store.write_json(f'nodes/node{node}/memory.json',memory)
        release_device(device)
    store.write_json(f'nodes/node{node}/work_complete.json',dict(identity=plan['identity'],
        jobs=[j['key'] for j in assigned],status='all_assigned_components_attempted',memory=memory))
    return 0


def summarize_jobs(plan, reports):
    failed=[j['key'] for j in plan['jobs'] if reports.get(j['key'],{}).get('execution_status')!='completed']
    lk_failed=[key for key,r in reports.items() if r.get('lk_comparison',{}).get('status')=='failed']
    return dict(schema='scene-end-to-end-summary-v1',identity=plan['identity'],
        execution_status='completed' if not failed and not lk_failed else 'incomplete_or_failed_components',
        failed_or_missing=failed,failed_lk_comparisons=lk_failed,
        teacher_support=[dict(key=key,readiness=r.get('readiness'),reference_probe=r.get('reference_probe'),
                              inference_control_view_v2=r.get('inference_control_view_v2'),
                              motion_diagnostics=r.get('motion_diagnostics'),lk=r.get('lk_comparison'))
                         for key,r in reports.items() if key.startswith('teacher/')],
        data_replay=[dict(key=key,status=r.get('functional_status'),
                         mean_uint8_error=r.get('mean_uint8_error'))
                     for key,r in reports.items() if key.startswith('replay/')],
        generation=[dict(key=key,generation=r.get('generation'),gt_oracles=r.get('gt_oracles'),
                         decoder_error_structure=r.get('decoder_error_structure'),
                         decoder_branches=r.get('decoder_branches'))
                    for key,r in reports.items() if key.startswith('generation/')],
        quality_status='requires_paired_visual_and_independent_geometry_review',
        training_authorized_by_audit=False,
        next_decision='Use real correspondence support, oracle gains and branch perturbations to choose the next experiment; do not infer success from exit0.')


def merge(store, plan, nodes):
    node_errors=[]
    for node in range(nodes):
        try:
            other=store.json(f'nodes/node{node}/contract.json',True)
            if other['identity']!=plan['identity']:
                raise ValueError('Nodes ran different source/config/code contracts')
        except Exception as exc:
            node_errors.append(dict(node=node,error=repr(exc)))
    reports={}
    for job in plan['jobs']:
        try:
            report=store.json(job['key']+'/report.json')
        except Exception as exc:
            reports[job['key']]=dict(execution_status='failed',error='Report unavailable: '+repr(exc))
            continue
        if report:
            if not isinstance(report,dict) or report.get('identity')!=plan['identity']:
                reports[job['key']]=dict(execution_status='failed',error='Component report identity differs')
                continue
            if report.get('execution_status')=='completed':
                try:
                    recovered=recover_case(store.local,job['key']+'/done.json',plan['identity'],store.roots)
                    if recovered is None:
                        raise ValueError('Completion seal missing')
                except Exception as exc:
                    report=dict(report,execution_status='failed',error='Artifact validation: '+repr(exc))
            reports[job['key']]=report
    summary=summarize_jobs(plan,reports)
    summary['node_contract_errors']=node_errors
    if node_errors:
        summary['execution_status']='incomplete_or_failed_components'
    store.write_json('summary.json',summary)
    return 0 if summary['execution_status']=='completed' else 2


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['work','merge'],required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True); p.add_argument('--root',action='append',default=[])
    p.add_argument('--source-local',required=True); p.add_argument('--source-root',action='append',default=[])
    p.add_argument('--encoder',required=True); p.add_argument('--device',default='npu')
    return p


def main():
    a=parser().parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding='utf8'))
    node=int(os.environ.get('NODE_RANK',os.environ.get('GROUP_RANK',0)))
    nodes=int(os.environ.get('NNODES',1))
    if not 0<=node<nodes:
        raise ValueError('Invalid NODE_RANK/NNODES')
    local=Path(a.output).resolve(); source_local=Path(a.source_local).resolve()
    if local==source_local or local in source_local.parents or source_local in local.parents or set(a.root)&set(a.source_root):
        raise ValueError('Audit output must be separate from read-only source')
    store=ArtifactStore(a.output,a.root)
    store.status_name=f'publication_status-node{node}.json'
    source=ArtifactStore(local/'_inputs'/f'node{node}'/'source',[a.source_local,*a.source_root])
    ns=cfg['codec_namespace']
    if not ns.replace('_','').replace('-','').isalnum():
        raise ValueError('Invalid codec namespace')
    codec=ArtifactStore(local/'_inputs'/f'node{node}'/'codec',
        [str(source_local.parent/ns/'production')]+[r.rstrip('/').rsplit('/',1)[0]+'/'+ns+'/production' for r in a.source_root])
    plan=make_plan(cfg,source,codec,nodes)
    if a.stage=='work':
        pin_plan(store,plan,node)
        return work(a,cfg,plan,store,source,codec,node,nodes)
    return merge(store,plan,nodes)


if __name__=='__main__':
    raise SystemExit(main())
