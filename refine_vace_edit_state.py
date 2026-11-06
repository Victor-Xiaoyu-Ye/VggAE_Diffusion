"""Stage66: closed-loop decoder-calibrated editing on frozen Stage65 weights.

Five matched outputs: native, trained RGB/RAE without feedback, and RGB/RAE
with joint state projection driven by the CURRENT clean video prediction.
Historical artifacts are read-only; no future edited labels enter this runner.
"""
import argparse
import gc
from html import escape
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from utils.scene_run import ArtifactStore
from utils.editing_data import digest, file_hash, rae_canvas
from utils.editing_train import (distributed, gather, barrier, progress,
    load_teacher, load_encoder, write_video)
from utils.native_audit_io import seal, recover_case
from utils.editing_publication import reconcile_case
from utils.vace_edit_runtime import (load_backbone, load_vae, validate_assets,
    sample_native, decode_native)
from utils.vace_state_runtime import StateConditionedVACE, sample_state_native
from utils.vace_state_data import prepare_rae_state, build_rgb_state
from train_vace_edit_state import decoded_reference, publish_lossless
from edit_reference_video import (crop_video, crop_float, output_metrics,
    publish_case, recovered_case, tensor_hash, failure)

ARMS = ('native', 'rgb_base', 'rae_base', 'rgb_joint', 'rae_joint')


def write_panels(path, panels, labels, fps):
    """Arbitrary-width comparison with equal timestamps, no spatial stretching."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
    if len(panels) != len(labels) or len({tuple(v.shape) for v in panels}) != 1:
        raise ValueError('Comparison panels must have identical video grids')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for row in zip(*panels):
        cells = []
        for value, label in zip(row, labels):
            canvas = Image.fromarray(value.permute(1,2,0).cpu().numpy().astype('uint8'))
            ImageDraw.Draw(canvas).text((6,6),label,fill='white',stroke_width=1,stroke_fill='black')
            cells.append(np.asarray(canvas))
        combined = np.concatenate(cells,axis=1)
        frames.append(np.pad(combined,((0,combined.shape[0]%2),(0,combined.shape[1]%2),(0,0))))
    imageio.mimwrite(str(path),frames,fps=fps,macro_block_size=1)


def code_signature():
    root = Path(__file__).resolve().parent
    names = ('refine_vace_edit_state.py', 'utils/editing_joint_state.py',
        'utils/editing_state_feedback.py', 'utils/vace_state_runtime.py',
        'utils/vace_state_data.py', 'utils/vace_edit_runtime.py')
    return {name: file_hash(root/name) for name in names}


def input_stores(a):
    return (ArtifactStore(a.states, a.state_root),
            ArtifactStore(a.transport, a.transport_root))


def old_arm(store, arm):
    return ArtifactStore(store.local/'arms'/arm,
                         [r.rstrip('/')+'/arms/'+arm for r in store.roots])


def manifest(a, cfg, out):
    """Snapshot old identities, NOT today's Stage65 source-code signature.

    Also callable from baseline render if teacher preparation failed. Missing
    selected weights fail their own arm without suppressing the native arm.
    """
    states, transport = input_stores(a)
    old = states.json('manifest.json', True)
    legacy = transport.json('manifest.json', True)
    if old['transport'] != legacy['identity'] or old['teacher'] != legacy['teacher']:
        raise ValueError('Stage64/65 cached source identities disagree')
    records = sorted([dict(r) for r in old['records'] if r['partition'] == 'dev'],
                     key=lambda r:r['id'])
    if cfg['cases']:
        records = records[:int(cfg['cases'])]
    if not records or cfg['frames'] != old['config']['frames']:
        raise ValueError('Need nonempty development rows with matching source times')
    selected = {}
    for arm in ('rgb', 'rae'):
        receipt = old_arm(states, arm).json('selected.pt.json')
        selected[arm] = receipt['sha256'] if receipt else None
    result = dict(schema='edit-state-feedback-v1', config=cfg, code=code_signature(),
        stage65=old['identity'], stage64=legacy['identity'],
        stage65_code=old['code'], base_config=old['config'], teacher=old['teacher'],
        assets=old.get('assets'), selected=selected, records=records)
    result['identity'] = digest(result)
    previous = out.json('manifest.json')
    if previous and previous['identity'] != result['identity']:
        raise ValueError('Feedback inputs/code/config changed; choose a fresh FEEDBACK_NAMESPACE')
    return result


def saved_manifest(a, cfg, out, rank, world):
    m = manifest(a, cfg, out)
    if any(value != m['identity'] for value in gather(m['identity'], world)):
        raise ValueError('Nodes disagree on Stage66 inputs')
    if rank == 0:
        out.write_json('manifest.json', m)
    barrier()
    return m


def sampling_config(m, cfg):
    result = dict(m['base_config'])
    for key in ('seed', 'frames', 'fps', 'sample_steps', 'shift', 'guidance', 'context_scale'):
        result[key] = cfg[key]
    return result


def _rgb(value):
    return value.float()/255 if value.dtype == torch.uint8 else value.float()


def _release(device):
    gc.collect()
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k:_cpu_snapshot(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_snapshot(v) for v in value)
    return value


def prepare(a, cfg, out, device, rank, world):
    from utils.editing_joint_state import build_source_edges
    from utils.editing_video_transport import propagate_reference
    m = saved_manifest(a, cfg, out, rank, world)
    states, transport = input_stores(a)
    # One checkpoint download per node, not eight concurrent identical copies.
    staged = []
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        for arm in ('rgb', 'rae'):
            try:
                old_arm(states, arm).load('selected.pt', digest([m['stage65'], arm]), True)
                staged.append(dict(arm=arm, status='completed'))
            except Exception as exc:
                staged.append(dict(arm=arm, status='failed', error=repr(exc)))
    all_staged = sum(gather(staged, world), [])
    if rank == 0:
        out.write_json('checkpoint_staging.json', dict(records=all_staged))
    barrier()
    if file_hash(a.rae) != m['teacher']['rae'] or file_hash(a.encoder) != m['teacher']['encoder']:
        raise ValueError('Teacher differs from the production source cache')
    training_scale = transport.json('prepared.json', True)['source_scale']
    if training_scale is None:
        raise ValueError('Missing source-only training latent normalization')
    reports = []; rae = encoder = None; started = time.monotonic()
    for row in m['records'][rank::world]:
        try:
            name = 'prepared/'+row['id']+'.pt'
            value = out.load(name, m['identity'])
            if value is None:
                old_input = transport.load('inputs/'+row['id']+'.pt', m['stage64'], True)
                state = states.load('states/'+row['id']+'.pt', m['stage65'], True)
                if state['provenance'] not in ('model_output', 'user_edit'):
                    raise ValueError('Only model/user development anchors are legal')
                reference = decoded_reference(state['reference'])
                # Explicit allowlist excludes every supervision/target field.
                names = ('source', 'proposal', 'source_z', 'anchor_z', 'anchor',
                    'support', 'masks', 'confidence', 'observed_indices', 'valid_rect', 'source_spatial')
                inputs = {key:old_input[key] for key in names}
                inputs['source_z'] = inputs['source_z'].float()
                square, rect = rae_canvas(reference['source'], reference['spatial']['valid_rect'],
                                         size=inputs['source'].shape[-1])
                if list(rect) != list(inputs['valid_rect']):
                    raise ValueError('Source letterbox coordinate mismatch')
                options = dict(m['base_config']['transport'])
                edges = build_source_edges(square[inputs['observed_indices']], options)
                # Full-rate PAIR confidence, never the accumulated chain score.
                full_flow = []; full_confidence = []
                for index in range(len(square)-1):
                    pair = square[index:index+2]
                    edge = propagate_reference(pair, pair[0], torch.ones_like(pair[0, :1]),
                        dict(options, flow_direct_anchor=False))
                    full_flow.append(edge['flows_backward'][0].half())
                    full_confidence.append(edge['confidence'][1].half())
                    if index % 4 == 0:
                        progress(out, f'prepare_r{rank}', dict(phase='source_correspondence',
                            id=row['id'], edge=index+1, total=len(square)-1))
                if rae is None:
                    rae = load_teacher(a.rae, device).float()
                    encoder = load_encoder(a.encoder, device)
                # Stage65 only saved an aligned bridge map; rebuild the original
                # decoder-grid state once, before VACE14B is resident.
                calibrated = prepare_rae_state(inputs, reference, rae, encoder,
                                                m['base_config'], device=device)
                inputs.update(initial_delta=(calibrated['edited_z'].float()-inputs['source_z']).half(),
                    scale=torch.tensor(training_scale), edges=[{k:(v.half() if isinstance(v, torch.Tensor) else v)
                        for k,v in edge.items()} for edge in edges],
                    full_flows_backward=torch.stack(full_flow),
                    full_edge_confidence=torch.stack(full_confidence))
                value = dict(inputs=inputs, initialization=calibrated['calibration'],
                    provenance=state['provenance'], teacher=m['teacher'],
                    protocol='source-only correspondence plus legal reference, no future labels')
                out.save(name, value, m['identity'])
            reports.append(dict(id=row['id'], status='completed', file=name,
                initialization=value['initialization']))
            progress(out, f'prepare_r{rank}', dict(phase='prepared', id=row['id'], completed=len(reports),
                DI_throughput=len(reports)*cfg['frames']/max(time.monotonic()-started, 1e-6),
                throughput_unit='source frames/s/rank including correspondence, encoding, calibration and IO'))
        except Exception as exc:
            reports.append(failure(out, 'prepare', row, exc))
            _release(device)
    reports = sum(gather(reports, world), [])
    if rank == 0:
        out.write_json('phases/prepare.json', dict(requested=len(m['records']), cases=reports))
    barrier()
    if any(r['status'] != 'completed' for r in reports):
        raise RuntimeError('Some feedback inputs failed; independent baseline arms remain runnable')


def _same_sampling(m, cfg):
    return all(m['base_config'].get(k) == cfg[k]
               for k in ('seed', 'frames', 'fps', 'sample_steps', 'shift', 'guidance', 'context_scale'))


def _historical_baseline(states, m, arm, key, expected_input, seed, cfg):
    if not cfg.get('reuse_baselines', True) or not _same_sampling(m, cfg):
        return None
    original_arm = 'native' if arm == 'native' else arm.split('_')[0]
    checkpoint = 'native' if arm == 'native' else m['selected'][original_arm]
    try:
        value = states.load('render/'+original_arm+'/'+key+'/result.pt', m['stage65'])
    except Exception as exc:
        print('[feedback baseline reuse miss] '+json.dumps(dict(id=key, arm=arm, error=repr(exc))), flush=True)
        return None
    if value is None:
        return None
    expected = digest([expected_input, original_arm, checkpoint])
    if (value.get('generation_identity') != expected or value.get('input_identity') != expected_input
            or value.get('checkpoint') != checkpoint or value.get('seed') != seed):
        return None
    return value


def render(a, cfg, out, device, rank, world):
    from utils.editing_joint_state import JointEditProjector
    from utils.editing_state_feedback import StateFeedback, load_decoder_only
    m = saved_manifest(a, cfg, out, rank, world)
    states, _ = input_stores(a)
    arm = a.arm; family = 'native' if arm == 'native' else arm.split('_')[0]
    joint = arm.endswith('_joint')
    sampler_cfg = sampling_config(m, cfg)
    checkpoint = 'native' if family == 'native' else m['selected'][family]
    model = vae = decoder = None; reports = []; started = time.monotonic()
    for row in m['records'][rank::world]:
        key = row['id']; folder = 'render/'+arm+'/'+key
        feedback = projector = latent = features = source = valid = inputs = prepared = None
        try:
            if checkpoint is None:
                raise FileNotFoundError('Selected '+family+' checkpoint unavailable')
            state = states.load('states/'+key+'.pt', m['stage65'], True)
            native = states.load('native/'+key+'.pt', m['stage65'], True)
            if state['provenance'] not in ('model_output', 'user_edit'):
                raise ValueError('Illegal development anchor')
            ref = decoded_reference(state['reference'])
            seed = cfg['seed']+int(digest(key)[:8], 16)%1000000
            old_input = digest([m['stage65'], key, seed, tensor_hash(state['reference']['source']),
                tensor_hash(state['reference']['reference']), tensor_hash(ref['masks']), tensor_hash(ref['anchor'])])
            input_identity = digest([m['identity'], key, old_input])
            generation = digest([input_identity, arm, checkpoint])
            payload, record = recovered_case(out, folder, m['identity'])
            if payload is not None and payload.get('generation_identity') != generation:
                payload = record = None
            if record:
                reports.append(publish_lossless(out, folder, m['identity'], payload, cfg['fps']))
                continue
            if payload is None:
                historical = None if joint else _historical_baseline(states, m, arm, key, old_input, seed, cfg)
                events = []; diagnostics = []; feedback = None; reused = historical is not None
                if historical is not None:
                    generated = historical['generated'].float().cpu()
                else:
                    prepared = out.load('prepared/'+key+'.pt', m['identity'], True) if joint else None
                    if model is None:
                        if m['assets'] is not None and validate_assets(a.model) != m['assets']:
                            raise ValueError('Native assets differ from Stage65')
                        core = load_backbone(a.model, device)
                        if family == 'native':
                            model = core
                        else:
                            model = StateConditionedVACE(core, sampler_cfg,
                                m['base_config']['state_channels'][family]).to(device)
                            selected = old_arm(states, family).load('selected.pt', digest([m['stage65'], family]), True)
                            model.restore(selected['model'])
                        model.eval().requires_grad_(False)
                    if vae is None:
                        vae = load_vae(a.model, device)
                    features = None
                    if family != 'native':
                        features = (build_rgb_state(ref) if family == 'rgb' else state['rae_state'].float()).to(device)
                    if joint:
                        inputs = prepared['inputs']
                        if family == 'rae' and decoder is None:
                            if file_hash(a.rae) != m['teacher']['rae']:
                                raise ValueError('Decoder differs from cached source states')
                            decoder = load_decoder_only(a.rae, device, checkpoint_decode=cfg.get('decoder_checkpoint', True))
                        source = _rgb(inputs['source']).to(device)
                        valid = source.new_zeros(len(source), 1, *source.shape[-2:])
                        y,x,h,w = inputs['valid_rect']; valid[:,:,y:y+h,x:x+w] = 1
                        projector = JointEditProjector(
                            decode=decoder if family == 'rae' else None,
                            source_z=inputs['source_z'].to(device) if family == 'rae' else None,
                            source=source, anchor=_rgb(inputs['anchor']).to(device),
                            masks=inputs['masks'].to(device).float(), confidence=inputs['confidence'].to(device).float(),
                            flows_backward=None, cfg=dict(cfg['solver'],device=str(device)),
                            initial_delta=inputs['initial_delta'].to(device) if family == 'rae' else None,
                            initial_effect=(_rgb(inputs['proposal'])-_rgb(inputs['source'])).to(device),
                            proposal=_rgb(inputs['proposal']).to(device), edges=inputs['edges'],
                            scale=inputs['scale'].to(device), valid=valid)
                        def observer(event):
                            events.append(dict(event))
                            step = int(event.get('step', event.get('step_index', len(events))))
                            if event.get('applied') and feedback.last_reference is not None:
                                snapshot = _cpu_snapshot(dict(reference=feedback.last_reference,
                                    state=feedback.state, projection=feedback.last_projection,
                                    event=event, generation_identity=generation))
                                # Persist every completed correction before subsequent denoising.
                                out.save(folder+f'/feedback/step_{step:03d}.pt', snapshot, m['identity'])
                                preview=folder+f'/feedback/step_{step:03d}.mp4'
                                rect=ref['spatial']['valid_rect']
                                write_video(out.local/preview, crop_video(ref['source'],rect),
                                    crop_video(ref['reference'],rect),
                                    crop_video(snapshot['reference']['reference'],rect),cfg['fps'],
                                    labels=['SOURCE','SHARED REFERENCE','LIVE CORRECTION'])
                                receipt=folder+f'/feedback/step_{step:03d}.json'
                                seal(out.local,receipt,m['identity'],[preview],dict(id=key,arm=arm,event=event))
                                out.publish([preview,receipt])
                            progress(out, f'feedback_{family}_r{rank}', dict(event,id=key,arm=arm))
                        feedback = StateFeedback(vae, projector, inputs, ref, features, cfg, family, observer=observer)
                    with torch.no_grad():
                        if family == 'native':
                            latent = sample_native(model, native['native_context'].to(device), state['text'].to(device),
                                state['negative'].to(device), sampler_cfg, seed)
                        else:
                            latent = sample_state_native(model, native['context'].to(device), state['text'].to(device),
                                state['negative'].to(device), features, sampler_cfg, seed, velocity_callback=feedback)
                        generated = ((decode_native(vae, latent, 1).permute(1,0,2,3).float()+1)/2).clamp(0,1).cpu()
                    if feedback is not None:
                        diagnostics = feedback.events
                if not bool(torch.isfinite(generated).all()) or generated.shape != ref['source'].shape:
                    raise ValueError('Generated video has invalid shape/values')
                payload = dict(generated=generated, spatial=ref['spatial'], input_identity=input_identity,
                    generation_identity=generation, checkpoint=checkpoint, seed=seed, reference_count=1,
                    historical_reused=reused, events=diagnostics, source_input_identity=old_input)
            rect = ref['spatial']['valid_rect']
            metrics = dict(id=key, status='completed', arm=arm, input_identity=input_identity,
                generation_identity=generation, checkpoint=checkpoint, seed=seed, provenance=state['provenance'],
                historical_reused=payload['historical_reused'], feedback_events=payload['events'],
                **output_metrics(crop_float(ref['source'],rect), crop_float(payload['generated'],rect), crop_float(ref['masks'],rect)))
            publish_case(out, folder, m['identity'], payload, metrics,
                [ref['source'], ref['reference'], payload['generated']], ['SOURCE','SHARED REFERENCE',arm.upper()], rect,cfg['fps'])
            reports.append(publish_lossless(out, folder, m['identity'], payload, cfg['fps']))
            progress(out, f'render_{arm}_r{rank}', dict(phase='render', arm=arm, completed=len(reports),
                DI_throughput=len(reports)*cfg['frames']/max(time.monotonic()-started,1e-6),
                throughput_unit='completed output frames/s/rank including feedback, reuse and IO'))
        except Exception as exc:
            reports.append(dict(failure(out,'render_'+arm,row,exc), arm=arm))
        finally:
            feedback = projector = latent = features = source = valid = inputs = prepared = None
            _release(device)
    reports = sum(gather(reports, world), [])
    if rank == 0:
        out.write_json('phases/render_'+arm+'.json', dict(requested=len(m['records']), cases=reports))
    barrier()
    if any(r['status'] != 'completed' for r in reports):
        raise RuntimeError('Some '+arm+' cases failed; successful cases and other arms retained')


def summarize(a, cfg, out):
    m = manifest(a, cfg, out); failures = []; sync = []; means = {}; feedback_health = {}
    page = ['<!doctype html><meta charset="utf-8"><title>Stage66 feedback comparison</title>',
        '<style>body{font:16px sans-serif;max-width:1600px;margin:25px auto}video{max-width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>Native / trained RGB / trained RAE / RGB feedback / RAE feedback</h1>',
        '<p>Same source, legal anchor, original proposal and seed. Base and feedback arms share each representation\'s selected weights. Internal development set; no SOTA claim.</p>']
    for arm in ARMS:
        report = out.json('phases/render_'+arm+'.json') or {}
        rows = report.get('cases',[])
        if (len(rows) != len(m['records']) or
                {r['id'] for r in rows} != {r['id'] for r in m['records']}):
            failures.append(dict(arm=arm,error='Missing arm or case coverage',actual=len(rows)))
        completed = [r for r in rows if r['status']=='completed']
        failures.extend(r for r in rows if r['status']!='completed')
        means[arm] = {k:sum(r[k] for r in completed)/len(completed) for k in
            ('source_change','edit_region_change','keep_change','output_frame_delta')} if completed else None
        if arm.endswith('_joint'):
            events=[e for r in completed for e in r.get('feedback_events',[])]
            feedback_health[arm]=dict(requested_events=len(events),
                applied=sum(bool(e.get('applied')) for e in events),
                fallback=sum(bool(e.get('fallback')) for e in events),
                cases_without_applied_feedback=[r['id'] for r in completed
                    if not any(e.get('applied') for e in r.get('feedback_events',[]))])
        for r in completed:
            try:
                roots = reconcile_case(out, r['receipt'], m['identity'])
                sync.append(dict(receipt=r['receipt'],ok=all(v['ok'] for v in roots)))
                for event in r.get('feedback_events',[]):
                    if not event.get('applied'):continue
                    receipt=f"render/{arm}/{r['id']}/feedback/step_{int(event['step_index']):03d}.json"
                    roots=reconcile_case(out,receipt,m['identity'])
                    sync.append(dict(receipt=receipt,ok=all(v['ok'] for v in roots)))
            except Exception as exc:
                sync.append(dict(receipt=r['receipt'],ok=False,error=repr(exc)))
    for row in m['records']:
        key = row['id']; folder='comparisons/'+key
        try:
            payloads = [out.load('render/'+arm+'/'+key+'/result.pt',m['identity'],True) for arm in ARMS]
            if len({p['input_identity'] for p in payloads}) != 1:
                raise ValueError('Unmatched comparison inputs')
            for arm,payload in zip(ARMS,payloads):
                checkpoint='native' if arm=='native' else m['selected'][arm.split('_')[0]]
                if (payload['checkpoint'] != checkpoint or payload['generation_identity'] !=
                        digest([payload['input_identity'],arm,checkpoint])):
                    raise ValueError('Stale comparison checkpoint/generation for '+arm)
            generations=[p['generation_identity'] for p in payloads]
            record=recover_case(out.local,folder+'/receipt.json',m['identity'],out.roots)
            if record and record['metadata'].get('generations')!=generations:
                record=None
            if record is None:
                rect=payloads[0]['spatial']['valid_rect']
                frames=[crop_video(p['generated'],rect) for p in payloads]
                movie=folder+'/native_base_feedback.mp4'
                write_panels(out.local/movie,frames,['NATIVE','RGB BASE','RAE BASE','RGB FEEDBACK','RAE FEEDBACK'],cfg['fps'])
                # Also publish compact causal comparisons, not only a tiny five-column grid.
                files=[movie]
                for family,indices in [('rgb',(1,3,4)),('rae',(2,4,3))]:
                    name=folder+'/'+family+'_base_feedback.mp4'
                    write_video(out.local/name,*[frames[i] for i in indices],cfg['fps'],
                        labels=[ARMS[i].upper() for i in indices]);files.append(name)
                record=seal(out.local,folder+'/receipt.json',m['identity'],files,dict(id=key,generations=generations))
            out.publish([*record['files'],folder+'/receipt.json'])
            roots=reconcile_case(out,folder+'/receipt.json',m['identity'])
            sync.append(dict(receipt=folder+'/receipt.json',ok=all(v['ok'] for v in roots)))
            page.extend(['<h2>'+escape(key+' '+row.get('instruction',''))+'</h2>',
                '<video controls preload="none" src="'+folder+'/native_base_feedback.mp4"></video>'])
        except Exception as exc:
            failures.append(dict(id=key,phase='comparison',error=repr(exc)))
    bad=[r for r in sync if not r['ok']]
    summary=dict(status='completed' if not failures and not bad else 'incomplete',cases=len(m['records']),
        means=means,feedback_health=feedback_health,failures=failures,media_failures=bad,
        protocol='Frozen Stage65 weights; current prediction -> joint source-correspondence projection -> state and velocity write',
        limitations=['Internal dev; all failed cases retained. No SOTA or unique RAE benefit is assumed.',
            'Source flow is approximate correspondence, not semantic segmentation or certified 3D geometry.',
            'Intermediate solver states are saved; interrupted sampling restarts the case from the same seed, not from an invalid partial UniPC history.',
            'Keep/edit pixel changes do not score semantic correctness or realism.'])
    out.write_json('summary.json',summary);out.write_json('media_sync.json',dict(checked=len(sync),failed=bad))
    page.append('<pre>'+escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
    (out.local/'report.html').write_text('\n'.join(page),encoding='utf8');out.publish(['report.html'])
    if failures or bad:
        raise RuntimeError('Incomplete Stage66; inspect summary.json; successful independent arms retained')


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('prepare','render','summarize'),required=True)
    p.add_argument('--arm',choices=ARMS,default='native')
    for name in ('config','output','states','transport'):
        p.add_argument('--'+name,required=True)
    for name in ('rae','encoder','model'):
        p.add_argument('--'+name,default='')
    for name in ('root','state-root','transport-root'):
        p.add_argument('--'+name,action='append',default=[])
    return p


def main():
    a=build_parser().parse_args();cfg=json.loads(Path(a.config).read_text())
    out=ArtifactStore(a.output,a.root)
    if a.phase=='summarize':
        if int(os.environ.get('NODE_RANK','0'))==0:summarize(a,cfg,out)
        return
    device,rank,world=distributed()
    try:
        globals()[a.phase](a,cfg,out,device,rank,world)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
