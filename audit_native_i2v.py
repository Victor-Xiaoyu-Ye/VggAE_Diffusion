#!/usr/bin/env python3
"""Native Wan I2V -> frozen R7 replay -> existing EMA6000 generation. No training."""
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from PIL import Image

from scripts.window_run_io import atomic
from utils.native_audit_io import (sha256, seal, recover_case, recover_json,
                                  validate_wan_checkpoint, aligned_indices)
from utils.training import atomic_torch_save
from utils.window_training import load_artifact, digest


def status(out, phase, **extra):
    value = dict(phase=phase, unix_time=time.time(), **extra)
    atomic(Path(out)/'status.json', value)
    print('[native I2V audit] ' + json.dumps(value, ensure_ascii=False), flush=True)


def write_video(path, video, fps):
    """THWC uint8; atomically publish MP4 without macroblock resizing."""
    import imageio.v2 as imageio
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.pending.mp4')
    try:
        imageio.mimsave(str(temporary), list(video.cpu().numpy()), fps=fps,
                       codec='libx264', pixelformat='yuv420p', macro_block_size=1)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def uint8(video):
    return video.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8).cpu()


def prepare(a):
    from data.latent_shard_dataset import read_shard_manifest
    from diagnose_window_diffusion import select_samples
    from utils.window_training import validate_statistics, validate_batch, collate
    from utils.file_signature import sampled_file_signature
    from utils.window_codec import validate_runtime
    from utils.r7_representation import validate_contract
    out = Path(a.output_dir)
    old = recover_json('run_contract.json', out, a.read_root, required=a.resume)
    if old is not None and not a.resume:
        raise FileExistsError('existing contract requires RESUME=1')
    wan = validate_wan_checkpoint(a.wan_ckpt)
    saved = load_artifact(a.checkpoint)
    if (saved.get('schema') != 'r7-window-trainer-v1' or saved.get('step') != 6000
            or not saved['args']['no_text'] or saved['args'].get('memorize_clips', 0)):
        raise ValueError('requires reviewed uncaptioned non-memory EMA6000 checkpoint')
    identity = saved['contract']['identity']; stats = saved['statistics']
    cfg = validate_statistics(stats)
    if validate_runtime(stats['representation']) != 'legacy' or cfg['temporal_factor'] != 2:
        raise ValueError('requires t2v2 legacy codec')
    if saved['args']['sample_steps'] != 64 or saved['args']['sample_method'] != 'euler':
        raise ValueError('reference must use production Euler64')
    if a.seeds != [int(s) for s in saved['args']['sample_seeds'].split(',')]:
        raise ValueError('seeds must match reviewed run')
    if digest(stats) != identity['statistics']:
        raise ValueError('training statistics fingerprint mismatch')
    if digest(load_artifact(a.eval_stats)) != identity['eval_statistics']:
        raise ValueError('evaluation statistics fingerprint mismatch')
    if read_shard_manifest(a.eval_manifest) != identity['eval_manifest']:
        raise ValueError('evaluation manifest differs from source training run')
    if sha256(a.ae_reference) != identity.get('ae_reference_sha256'):
        raise ValueError('reviewed AE reference differs from checkpoint')
    ref = json.loads(Path(a.ae_reference).read_text())
    if stats['representation']['signatures'] != ref['ae_signature']:
        raise ValueError('AE signatures differ from reviewed reference')
    for key, path in (('r7', a.r7_ckpt), ('streamvggt', a.encoder_ckpt)):
        if sampled_file_signature(path) != ref['ae_signature'][key]:
            raise ValueError(key + ' checkpoint signature differs')
    artifact = load_artifact(a.r7_ckpt)
    validate_contract(stats['representation'], artifact['representation_contract'])
    del artifact
    original = select_samples(a.eval_manifest, saved['args']['eval_clips'], saved['args']['seed'])
    if [s['video_id'] for s in original] != [r['video_id'] for r in ref['clips']]:
        raise ValueError('original evaluation order differs from reference')
    if not 1 <= a.clips <= len(original):
        raise ValueError('invalid clip count')
    selected = original[:a.clips]
    inputs = Path(a.work_dir); inputs.mkdir(parents=True, exist_ok=True)
    records = []
    for index, sample in enumerate(selected):
        validate_batch(collate([sample]), 4)
        if sample.get('rgb') is None or sample['rgb'].shape != (9, 3, 518, 518) or sample['rgb'].dtype != torch.uint8:
            raise ValueError('requires exact cached RAW uint8 TCHW 9x518 clip')
        # Do not include future captions or other future annotations in native inputs.
        item = {key: sample[key] for key in ('rgb', 'cond', 'target', 'video_id')}
        item.update(index=index, window_id=sample.get('window_id'), frame_indices=sample.get('frame_indices'))
        path = inputs/f'clip{index:03d}.pt'
        atomic_torch_save(item, str(path))
        records.append(dict(index=index, video_id=sample['video_id'], window_id=sample.get('window_id'),
            frame_indices=sample.get('frame_indices'), rgb_sha256=hashlib.sha256(sample['rgb'].numpy().tobytes()).hexdigest(),
            cond_digest=digest(sample['cond']), target_digest=digest(sample['target'])))
    code_root = Path(__file__).parent
    code_paths = [Path(__file__), code_root/'diagnose_window_diffusion.py']
    for folder in ('utils', 'models', 'data', 'streamvggt', 'Wan2.1/wan'):
        code_paths += sorted((code_root/folder).rglob('*.py'))
    contract = dict(schema='native-i2v-audit-v1', records=records, seeds=a.seeds,
        checkpoint=sampled_file_signature(a.checkpoint), source_step=6000, weights='ema',
        representation=stats['representation'], statistics=digest(stats), ae_reference=sha256(a.ae_reference),
        wan_checkpoint=wan, native=dict(frames=81, fps=16, max_area=480*832, solver='unipc', steps=40,
            shift=3., guidance=5., prompt=a.prompt, negative_prompt='native config default',
            future_information=False, seed_rule='seed+1009*original_eval_index; NPU RNG'),
        torch_version=torch.__version__,
        comparison=dict(frames=9, fps=8, duration_seconds=1., size=[518,518],
            native_indices=aligned_indices(81), resize='bilinear antialias align_corners=False',
            native_window_storage='uint8; replay and reference use same quantized RGB',
            raw_reference='original real future; not a uniquely correct native continuation',
            r7_seed_rule='seed+1009*original_eval_index; CPU RNG exactly as original'),
        code={p.relative_to(code_root).as_posix(): sha256(p) for p in code_paths})
    if old is not None and old != contract:
        raise ValueError('resume contract changed; choose a fresh namespace')
    atomic(out/'run_contract.json', contract)
    status(out, 'prepared', clips=len(records), cases=len(records)*len(a.seeds), no_training=True)


def native(a, contract):
    from utils.native_wan_runtime import attention_gate, load_pipeline
    out = Path(a.output_dir); chash = digest(contract)
    pending = []
    for rec in contract['records']:
        for seed in a.seeds:
            stem = f'clip{rec["index"]:03d}_seed{seed}'
            if not recover_case(out, f'native/{stem}.json', chash, a.read_root):
                pending.append((rec, seed, stem))
    if not pending:
        status(out, 'native_completed', reused=True); return
    device = torch.device('npu:0')
    gate = attention_gate(device)
    atomic(out/'attention_gate.json', dict(status='passed', rows=gate))
    status(out, 'loading_native', remaining=len(pending))
    pipeline, runtime = load_pipeline(a.wan_ckpt)
    atomic(out/'native_runtime.json', runtime)
    total_time = 0.
    for done, (rec, seed, stem) in enumerate(pending, 1):
        item = load_artifact(str(Path(a.work_dir)/f'clip{rec["index"]:03d}.pt'))
        image = Image.fromarray(item['rgb'][0].permute(1, 2, 0).numpy())
        calls = 0; started = time.monotonic()
        def progress(module, inputs, result):
            nonlocal calls
            calls += 1
            if calls % 2 == 0:
                status(out, 'native_denoising', case=stem, step=calls//2, steps=40,
                       elapsed_seconds=time.monotonic()-started)
        hook = pipeline.model.register_forward_hook(progress)
        try:
            with torch.no_grad():
                video = pipeline.generate(a.prompt, image, max_area=480*832, frame_num=81,
                    shift=3., sample_solver='unipc', sampling_steps=40, guide_scale=5.,
                    seed=seed+1009*rec['index'], offload_model=True)
            torch.npu.synchronize()
        finally:
            hook.remove()
        if calls != 80:
            raise RuntimeError(f'expected 40 conditional/unconditional pairs, got {calls} forwards')
        if video.ndim != 4 or video.shape[:2] != (3,81) or not torch.isfinite(video).all():
            raise RuntimeError('invalid native generated video')
        seconds = time.monotonic()-started; total_time += seconds
        # Native result remains intact for viewing; metrics never read the lossy MP4.
        full = uint8((video.permute(1,2,3,0)+1)*.5)
        native_path = f'native/{stem}.mp4'
        write_video(out/native_path, full, 16)
        chosen = ((video[:, contract['comparison']['native_indices']].permute(1,0,2,3).float()+1)*.5).cpu()
        window = F.interpolate(chosen, (518,518), mode='bilinear', align_corners=False, antialias=True)
        window = uint8(window.permute(0,2,3,1))
        window_path = f'native/{stem}_window.pt'
        atomic_torch_save(dict(rgb=window, record=rec, seed=seed, contract_hash=chash), str(out/window_path))
        anchor_path = f'native/{stem}_input.png'
        image.save(out/anchor_path)
        metadata = dict(record=rec, seed=seed, native_shape=list(full.shape), generation_seconds=seconds,
            DI_throughput=81/seconds, throughput_units='native generated frames/s; includes conditioning and VAE',
            denoiser_forwards=calls, peak_allocated_bytes=torch.npu.max_memory_allocated())
        seal(out, f'native/{stem}.json', chash, [native_path, window_path, anchor_path], metadata)
        status(out, 'native_case_completed', case=stem, completed_this_attempt=done,
               DI_throughput=done*81/total_time, throughput_units='generated frames/s, model generation time')
        del video, full, chosen, window
        gc.collect(); torch.npu.empty_cache()
    status(out, 'native_completed', cases=len(contract['records'])*len(a.seeds))


def replay(a, contract):
    from models.r7_window_dit import R7WindowDiT
    from streamvggt.models.streamvggt import StreamVGGT
    from utils.encoder_loader import load_encoder_checkpoint
    from utils.r7_representation import load_r7_modules, encode_dual
    from utils.window_codec import configure_codec
    from utils.window_flow import WindowFlow
    from utils.window_training import normalize, inverse
    from utils.video_preview import save_video_preview
    out = Path(a.output_dir); chash = digest(contract); device = torch.device('npu:0')
    saved = load_artifact(a.checkpoint); stats = saved['statistics']; args = saved['args']
    artifact = load_artifact(a.r7_ckpt)
    config, compressor, texture, codec, decoder, _ = load_r7_modules(artifact)
    configure_codec(codec, 'legacy')
    for m in (compressor, texture, codec, decoder):
        m.to(device).eval().requires_grad_(False)
    encoder = StreamVGGT(img_size=config.target_size, patch_size=14, embed_dim=1024)
    load_encoder_checkpoint(encoder, a.encoder_ckpt)
    encoder.to(device=device, dtype=torch.float16).eval().requires_grad_(False)
    model = R7WindowDiT(**saved['model_args']).to(device).eval().requires_grad_(False)
    state = model.state_dict()
    if set(saved['ema']) != {k for k,v in state.items() if v.is_floating_point()}:
        raise ValueError('incomplete source EMA')
    state.update(saved['model']); state.update(saved['ema']); model.load_state_dict(state, strict=True)
    flow = WindowFlow(args['prediction'], args['loss_floor'], args['time_shift'], args.get('time_distribution','logit_normal_0_1'))
    if flow.contract() != saved['contract']['flow']:
        raise ValueError('source flow contract differs')
    dtype = {'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[args['dtype']]
    st = {key: {n: stats[key][n].to(device).float() for n in ('mean','std')} for key in ('cond','target')}
    reference = json.loads(Path(a.ae_reference).read_text())
    ref = {r['video_id']:r['ae_psnr_full_vs_raw'] for r in reference['clips']}
    del saved, artifact, state
    def decode(z):
        g, t = codec.decode(z)
        rgb = decoder(g,t)[..., :3].float()
        if not torch.isfinite(rgb).all():
            raise RuntimeError('nonfinite R7 replay')
        return rgb.clamp(0,1)[0]
    rows=[]; processed=0; started=time.monotonic()
    with torch.no_grad():
        for rec in contract['records']:
            item = load_artifact(str(Path(a.work_dir)/f'clip{rec["index"]:03d}.pt'))
            c, y = item['cond'][None].to(device).float(), item['target'][None].to(device).float()
            raw = item['rgb'].permute(0,2,3,1).to(device).float()/255
            ae = decode(torch.cat((c,y),1).reshape(1,5,18,18,192))
            ae_psnr = float(-10*(ae-raw).square().mean().clamp_min(1e-12).log10())
            if abs(ae_psnr-ref[rec['video_id']]) > reference['max_clip_psnr_delta']:
                raise ValueError(f'RAW AE replay differs for {rec["video_id"]}: {ae_psnr}')
            for seed in a.seeds:
                stem=f'clip{rec["index"]:03d}_seed{seed}'
                previous = recover_case(out, f'comparison/{stem}.json', chash, a.read_root)
                if previous:
                    rows.append(previous['metadata']); continue
                if not recover_case(out, f'native/{stem}.json', chash, a.read_root):
                    raise FileNotFoundError('native case incomplete: '+stem)
                data = load_artifact(str(out/f'native/{stem}_window.pt'))
                if data['contract_hash'] != chash or data['record'] != rec or data['seed'] != seed:
                    raise ValueError('native window identity mismatch')
                native_rgb = data['rgb'].to(device).float()/255
                frames=native_rgb.permute(0,3,1,2)[None]
                geo, tex = encode_dual(encoder, compressor, texture, frames, torch.float16)
                g0, t0 = encode_dual(encoder, compressor, texture, frames[:,:1], torch.float16)
                full = codec.encode(geo,tex); anchor=codec.encode(g0,t0)
                native_replay=decode(torch.cat((anchor,full[:,1:]),1))
                noise=torch.randn(y.shape,generator=torch.Generator().manual_seed(seed+1009*rec['index'])).to(device)
                pred=flow.sample(model,normalize(c,st['cond']),noise,64,dtype,method='euler')
                generated=decode(torch.cat((c,inverse(pred,st['target'])),1).reshape(1,5,18,18,192))
                error=(native_replay-native_rgb).square()
                row=dict(record=rec,seed=seed,raw_ae_psnr=ae_psnr,
                    native_roundtrip_psnr=float(-10*error.mean().clamp_min(1e-12).log10()),
                    native_roundtrip_l1=float((native_replay-native_rgb).abs().mean()),
                    r7_generated_raw_l1=float((generated[1:]-raw[1:]).abs().mean()),
                    r7_generated_ae_l1=float((generated[1:]-ae[1:]).abs().mean()),
                    native_anchor_l1=float((native_rgb[:1]-raw[:1]).abs().mean()),
                    first_frame='native replay uses native frame0; R7 diffusion uses original cached anchor',
                    frame_indices=contract['comparison']['native_indices'])
                videos=dict(raw=raw,raw_ae=ae,native=native_rgb,native_r7_replay=native_replay,r7_diffusion=generated)
                files=[]
                for label, rgb in videos.items():
                    name=f'comparison/{stem}_{label}.mp4';write_video(out/name,uint8(rgb),8);files.append(name)
                # Three frames are sufficient for a compact, lossless labelled contact sheet.
                previews=save_video_preview(str(out/'comparison'),stem,
                    {k:v[[0,4,8]] for k,v in videos.items()},save_frames=False,save_mp4=False,metadata=row)
                files += [Path(previews['grid']).relative_to(out).as_posix(),
                          Path(previews['manifest']).relative_to(out).as_posix()]
                # Preserve exact replay inputs and outputs, independent of MP4 compression.
                tensor_path=f'comparison/{stem}_windows.pt'
                atomic_torch_save({k:uint8(v) for k,v in videos.items()},str(out/tensor_path));files.append(tensor_path)
                latent_path=f'comparison/{stem}_latents.pt'
                atomic_torch_save(dict(native_r7=torch.cat((anchor,full[:,1:]),1).cpu(),
                    r7_generated_normalized=pred.cpu(), original_cond=c.cpu(),
                    original_target=y.cpu(), noise=noise.cpu(), contract_hash=chash),str(out/latent_path))
                files.append(latent_path)
                seal(out,f'comparison/{stem}.json',chash,files,row);rows.append(row)
                processed += 1
                elapsed=time.monotonic()-started
                status(out,'comparison_case_completed',case=stem,completed=len(rows),
                    DI_throughput=processed*9/elapsed,throughput_units='new comparison windows source frames/s; includes IO and all arms')
    atomic(out/'summary.json',dict(status='completed',no_training=True,contract_hash=chash,
        cases=len(rows),rows=rows,quality_passed=None,
        note='Completion verifies artifact/AE contracts, not perceptual quality. Inspect videos.'))
    status(out,'completed',cases=len(rows),quality_passed=None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','native','replay'))
    for name in ('output_dir','work_dir','wan_ckpt','checkpoint','r7_ckpt','encoder_ckpt','eval_manifest','eval_stats','ae_reference'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--clips',type=int,default=32)
    p.add_argument('--seeds',type=int,nargs='+',default=[101,211])
    p.add_argument('--prompt',default='A realistic continuous video of the scene.')
    p.add_argument('--read_root',action='append',default=[])
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();Path(a.output_dir).mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    phase_started = time.time()
    try:
        if a.phase=='prepare':
            prepare(a);return
        from utils.device import get_device_name, configure_backend_compatibility
        if get_device_name() != 'npu':
            raise RuntimeError('this native pipeline requires Ascend NPU')
        torch.npu.set_device(0);configure_backend_compatibility('npu')
        contract=json.loads((Path(a.output_dir)/'run_contract.json').read_text())
        if contract['seeds'] != a.seeds or contract['native']['prompt'] != a.prompt or len(contract['records']) != a.clips:
            raise ValueError('phase arguments differ from prepared contract')
        if a.phase=='native':native(a,contract)
        else:replay(a,contract)
    except BaseException as exc:
        status(a.output_dir,'failed',stage=a.phase,error=repr(exc));raise
    finally:
        atomic(Path(a.output_dir)/f'{a.phase}_timing.json', dict(
            started_unix=phase_started, finished_unix=time.time(),
            elapsed_seconds=time.time()-phase_started,
            scope='entire phase for this attempt, including model loading and IO'))


if __name__=='__main__':main()
