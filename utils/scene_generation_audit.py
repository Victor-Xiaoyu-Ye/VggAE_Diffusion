"""Read-only fixed-window generation/decoder/condition diagnostics.

Caller owns checkpoint identity, IO and publication. Future GT enters only the
explicitly labelled supervised denoising and decoder-counterfactual probes.
Nothing here trains, chooses a checkpoint or certifies geometry/visual quality.
"""
from __future__ import annotations

import math
import torch

from utils.scene_prefix import PrefixFlowNet, prepare_prefix
from utils.scene_prefix_eval import _frame_ids, _raw_rgb, _motion, stable_case_seed
from utils.scene_motion_objective import x0_error_metrics
from utils.scene_control import ControlledPrefixFlowNet, control_features
from utils.window_flow import WindowFlow


def tensor_summary(value):
    value = value.detach().float()
    finite = torch.isfinite(value)
    valid = value[finite]
    return dict(shape=list(value.shape), dtype=str(value.dtype), finite_count=int(finite.sum()),
        count=value.numel(), all_finite=bool(finite.all()),
        rms=float(valid.square().mean().sqrt()) if valid.numel() else None,
        mean=float(valid.mean()) if valid.numel() else None,
        std=float(valid.std(unbiased=False)) if valid.numel() else None,
        min=float(valid.min()) if valid.numel() else None,
        max=float(valid.max()) if valid.numel() else None)


def difference(value, reference):
    delta = value.float() - reference.float()
    finite = bool(torch.isfinite(delta).all())
    return dict(all_finite=finite,
        l1=float(delta.abs().mean()) if finite else None,
        rms=float(delta.square().mean().sqrt()) if finite else None,
        max_abs=float(delta.abs().max()) if finite else None)


def _rms_matched_gaussian(error, generator):
    """Independent isotropic Gaussian direction at the error's empirical RMS.

    Matching is global over one normalized latent branch (frames/tokens/channels),
    not per channel or frame. A zero error yields zero noise, without division by
    zero. The supplied generator keeps this probe isolated from sampler RNG.
    """
    error=error.detach().float()
    if not error.numel() or not torch.isfinite(error).all():
        raise ValueError('RMS matching requires a finite nonempty error tensor')
    gaussian=torch.randn(error.shape,device=error.device,dtype=torch.float32,generator=generator)
    rms=error.square().mean().sqrt()
    if rms==0:
        return torch.zeros_like(error)
    gaussian_rms=gaussian.square().mean().sqrt()
    if not torch.isfinite(gaussian_rms) or gaussian_rms==0:
        raise RuntimeError('Cannot normalize a degenerate Gaussian probe')
    return gaussian*(rms/gaussian_rms)


@torch.no_grad()
def audit_generation_case(model, rae, sample, mean, std, device, *, seed=101,
                          steps=32, forward_dtype=None, branch_audit=True,
                          controls=True, control_arm='reference', matched_other_sample=None,
                          noise_levels=(.25, .5, .75, .95), perturb_scales=(.1, .5),
                          model_label='caller_supplied_checkpoint'):
    """Return (JSON-safe report, CPU tensors for outer artifact publication).

    ``sample`` uses production eval layout: target[9,N,C], raw[9,3,H,W], optional
    independent anchor[1,N,C] and stage56 controls. Models must already be loaded
    on ``device``. Source mean/std must come from the original frozen cache.
    Optional other sample must belong to the SAME codec/statistics contract;
    caller verifies identity. Otherwise shuffle is spatial-within-frame, labelled.

    Artifacts: RGB float32[T,H,W,3] raw/ae/generated and future-GT oracle hybrids;
    generated_future_latent[5,N,C] denormalized, and a four-column preview.
    Forward defaults to BF16 on accelerators, FP32 on CPU. Euler state/time grid
    are explicitly FP32. No global RNG, weights, gradients or mode flags remain
    changed. Zero-init controlled adapters are expected to have no sensitivity.
    """
    device = torch.device(device)
    core = getattr(model, 'module', model)
    target = sample['target']
    if not torch.is_tensor(target) or target.ndim != 3 or target.shape[0] != 9:
        raise ValueError('Audit requires an unbatched nine-frame target (observed4/future5)')
    total, tokens, channels = target.shape
    history = 4
    if channels % 2 or tokens != rae.grid**2 or channels != rae.channels:
        raise ValueError('Expected SceneRAE equal geo/texture halves and matching grid')
    if steps < 1 or not noise_levels or any(not math.isfinite(u) or not 0 < u <= 1 for u in noise_levels):
        raise ValueError('Invalid sampler steps or denoising noise levels')
    if any(not math.isfinite(s) or s <= 0 for s in perturb_scales):
        raise ValueError('Perturbation scales must be finite and positive')
    dtype = forward_dtype or (torch.float32 if device.type == 'cpu' else torch.bfloat16)
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Audit supports explicit FP32 or BF16 forwards')
    if control_arm not in ('camera', 'motion', 'reference'):
        raise ValueError('Control arm must be camera, motion or reference')
    mu = torch.as_tensor(mean, device=device, dtype=torch.float32).reshape(-1)
    sigma = torch.as_tensor(std, device=device, dtype=torch.float32).reshape(-1)
    if mu.shape != (channels,) or sigma.shape != mu.shape or not torch.isfinite(mu).all() or not torch.isfinite(sigma).all():
        raise ValueError('Invalid source statistics')
    sigma = sigma.clamp_min(1e-4)
    batch = prepare_prefix(sample, mu, sigma, hist=history, device=device)
    observed = torch.cat((batch.anchor, batch.known), 1)
    truth = target.to(device=device, dtype=torch.float32).unsqueeze(0)
    if not torch.isfinite(truth).all():
        raise ValueError('Nonfinite cached target')
    # The independent anchor is preserved for every decoder counterfactual.
    truth = torch.cat((observed * sigma + mu, truth[:, history:]), 1)
    raw = _raw_rgb(sample, total, device)
    noise_seed = stable_case_seed(sample, seed)
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    noise = torch.randn(batch.target.shape, device=device, dtype=torch.float32, generator=generator)
    flow = WindowFlow()
    ordinary = PrefixFlowNet(core, batch.known, model_frames=8)
    is_controlled = hasattr(core, 'control')
    maps = control_features(batch, sample, control_arm) if is_controlled and controls and sample.get('controls') is not None else None
    net = ControlledPrefixFlowNet(core, batch.known, maps, model_frames=8) if maps is not None else ordinary
    states = [(module, module.training) for owner in (core, rae) for module in owner.modules()]
    report = dict(schema='scene-generation-chain-audit-v1', model_label=model_label,
        id=str(sample.get('id', '')), frame_ids=_frame_ids(sample), history_frames=4, future_frames=5,
        noise_seed=noise_seed, seed=int(seed), forward_dtype=str(dtype),
        metric_note='Paired RGB/latent sensitivity only; not FVD, semantic motion or geometry correctness.',
        future_gt_policy='Only supervised-noise probes, clean AE and explicitly labelled counterfactuals read future targets.',
        tensors=dict(cached_latent=tensor_summary(truth), normalized_future=tensor_summary(batch.target),
            observed=tensor_summary(observed), noise=tensor_summary(noise)),
        normalization=dict(mean=tensor_summary(mu), effective_std=tensor_summary(sigma),
            independent_anchor_vs_cached_first=difference(batch.anchor,
                (target[:1].to(device=device,dtype=torch.float32).unsqueeze(0)-mu)/sigma)),
        sampler=dict(method='euler', steps=steps, state_dtype='torch.float32',
            grid_dtype='torch.float32', grid=torch.linspace(1, 0, steps+1, dtype=torch.float32).tolist(),
            trajectory=[]), controls=dict(status='not_applicable_source_model' if not is_controlled else 'no_supplied_maps'))
    artifacts = {}

    def forward(bound, z, u):
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            value = bound(z, u, batch.anchor).float()
        if not torch.isfinite(value).all():
            raise RuntimeError('Nonfinite denoiser output')
        return value

    def decode(z):
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            rgb = rae.decode(z.reshape(1, total, rae.grid, rae.grid, channels))[0].float()
        if rgb.shape != raw.shape or not torch.isfinite(rgb).all():
            raise RuntimeError('Decoder returned nonfinite or mismatched RGB')
        return rgb

    def sample_from(bound, trace=False):
        x = noise.clone()
        grid = torch.linspace(1., 0., steps+1, device=device, dtype=torch.float32)
        for i, (left, right) in enumerate(zip(grid[:-1], grid[1:])):
            u = left.expand(1)
            prediction = forward(bound, x, u)
            if trace:
                report['sampler']['trajectory'].append(dict(step=i, u=float(left),
                    state=tensor_summary(x), predicted_x0=tensor_summary(prediction)))
            x = x + (right-left) * flow.velocity(prediction, x, u)
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise RuntimeError('Euler state must remain finite FP32')
        return x

    core.eval(); rae.eval()
    try:
        generated = sample_from(net, trace=True)
        full = torch.cat((observed, generated), 1)*sigma + mu
        decoded, clean = decode(full), decode(truth)
        future = decoded[history:]
        report['tensors'].update(generated_normalized=tensor_summary(generated),
            generated_latent=tensor_summary(full[:, history:]), generated_rgb=tensor_summary(future),
            ae_rgb=tensor_summary(clean[history:]))
        report['generation'] = dict(rgb_vs_raw=difference(future, raw[history:]),
            ae_vs_raw=difference(clean[history:], raw[history:]), gen_vs_ae=difference(future, clean[history:]),
            normalized_x0_error=difference(generated, batch.target),
            generated_motion=_motion(future, raw[history-1]), gt_motion=_motion(raw[history:],raw[history-1]))
        report['generation']['per_future_frame']=[dict(frame_index=history+i,
            source_frame_id=_frame_ids(sample)[history+i] if len(_frame_ids(sample))==total else None,
            generated_vs_raw=difference(future[i],raw[history+i]),
            ae_vs_raw=difference(clean[history+i],raw[history+i]),
            gen_vs_ae=difference(future[i],clean[history+i])) for i in range(total-history)]
        display = torch.cat((raw[:history], future), 0)
        copy = torch.cat((raw[:history], raw[history-1:history].expand_as(future)), 0)
        artifacts.update(raw_rgb=raw.cpu(), ae_rgb=clean.cpu(), generated_rgb=display.cpu(),
            generated_future_latent=full[0, history:].cpu(), preview=torch.cat((raw, clean, display, copy),2).cpu())
        report['preview_columns'] = ['RAW','AE_GT_RECONSTRUCTION','GENERATED','COPY']
        report['denoising_probes'] = []
        previous = batch.known[:,-1:]
        for value in noise_levels:
            u = torch.tensor([value], device=device, dtype=torch.float32)
            noisy = (1-value)*batch.target + value*noise
            predicted = forward(net, noisy, u)
            probe_rgb = decode(torch.cat((observed,predicted),1)*sigma+mu)[history:]
            report['denoising_probes'].append(dict(u=float(value),
                kind='GT_CORRUPTED_FUTURE_SUPERVISED_PROBE_NOT_GENERATION',
                noisy=tensor_summary(noisy), predicted=tensor_summary(predicted),
                latent_error=x0_error_metrics(predicted,batch.target,previous,log_activity=True,activity_floor=1e-4),
                rgb_vs_raw=difference(probe_rgb,raw[history:]), rgb_vs_ae=difference(probe_rgb,clean[history:])))

        if branch_audit:
            half=channels//2
            perm=torch.randperm(tokens, device=device, generator=generator)
            donor = None
            if matched_other_sample is not None:
                donor = torch.as_tensor(matched_other_sample['target'], device=device,dtype=torch.float32).unsqueeze(0)
                if donor.shape != truth.shape or not torch.isfinite(donor).all():
                    raise ValueError('Cross-scene donor must have the same finite target layout')
            report['decoder_branches'] = dict(
                note='Future-only counterfactual interventions, NOT generated outputs. Zero/mean can be out of distribution; sensitivity alone is not geometry quality.',
                shuffle='cross_scene' if donor is not None else 'within_frame_spatial_permutation',
                donor_id=str(matched_other_sample.get('id','')) if donor is not None else None,
                donor_contract='Caller must verify same codec/statistics.' if donor is not None else None,
                rows=[])
            for base_name, z, reference_rgb in (('clean_gt',truth,clean),('sampled',full,decoded)):
                for branch,start,end in (('geo',0,half),('texture',half,channels)):
                    modes=['zero_raw','mean_train','shuffle']+[f'std_noise_{s:g}' for s in perturb_scales]
                    perturb=torch.randn(z[:,history:,:,start:end].shape,device=device,generator=generator)
                    for mode in modes:
                        altered=z.clone(); target_branch=altered[:,history:,:,start:end]
                        if mode=='zero_raw': target_branch.zero_()
                        elif mode=='mean_train': target_branch.copy_(mu[start:end].expand_as(target_branch))
                        elif mode=='shuffle': target_branch.copy_(donor[:,history:,:,start:end] if donor is not None else z[:,history:,perm,start:end])
                        else: target_branch.add_(perturb*sigma[start:end]*float(mode.removeprefix('std_noise_')))
                        rgb=decode(altered)[history:]
                        report['decoder_branches']['rows'].append(dict(base=base_name,branch=branch,intervention=mode,
                            channel_range=[start,end], normalized_perturbation=tensor_summary(
                                (altered[:,history:,:,start:end]-z[:,history:,:,start:end])/sigma[start:end]),
                            rgb_change=difference(rgb,reference_rgb[history:]), rgb_vs_raw=difference(rgb,raw[history:])))
            report['gt_oracles'] = []
            for branch,start,end in (('geo',0,half),('texture',half,channels)):
                hybrid=full.clone(); hybrid[:,history:,:,start:end]=truth[:,history:,:,start:end]
                rgb=decode(hybrid)
                name='GT_FUTURE_'+branch.upper()+'_ORACLE'
                artifacts[name]=torch.cat((raw[:history],rgb[history:]),0).cpu()
                report['gt_oracles'].append(dict(name=name,uses_future_gt=True,
                    rgb_vs_raw=difference(rgb[history:],raw[history:]), rgb_vs_gen=difference(rgb[history:],future)))

            # Equal-magnitude decoder probes isolate error structure from total
            # normalized latent MSE. Both are GT-future oracle interventions;
            # they are not independent video generation or geometry metrics.
            structure_seed=(noise_seed+0x5DEECE66D)%(2**63-1)
            structure_generator=torch.Generator(device=device).manual_seed(structure_seed)
            report['decoder_error_structure']=dict(
                kind='GT_FUTURE_BRANCH_ERROR_ORACLE_NOT_GENERATION',uses_future_gt=True,
                base='clean_gt',other_branch='unchanged_future_gt',
                note='Compare actual generated error with an independent isotropic Gaussian direction at equal normalized branch MSE. RGB differences describe decoder sensitivity; neither probe establishes geometric correctness.',
                matching_space='source_standardized_latent_branch',
                matching_scope='global_over_future_frames_tokens_and_branch_channels',
                gaussian_seed=structure_seed,rows=[])
            for branch,start,end in (('geo',0,half),('texture',half,channels)):
                actual_error=generated[...,start:end]-batch.target[...,start:end]
                gaussian_error=_rms_matched_gaussian(actual_error,structure_generator)
                real_mse=float(actual_error.square().mean())
                for name,error in (('actual_generated_error',actual_error),('rms_matched_gaussian',gaussian_error)):
                    altered=truth.clone()
                    altered[:,history:,:,start:end].add_(error*sigma[start:end])
                    rgb=decode(altered)[history:]
                    realized_error=(altered[:,history:,:,start:end]-truth[:,history:,:,start:end])/sigma[start:end]
                    report['decoder_error_structure']['rows'].append(dict(
                        branch=branch,intervention=name,uses_future_gt=True,channel_range=[start,end],
                        normalized_error=tensor_summary(error),normalized_error_mse=float(error.square().mean()),
                        requested_actual_error_mse=real_mse,
                        realized_normalized_error=tensor_summary(realized_error),
                        realized_normalized_error_mse=float(realized_error.square().mean()),
                        error_vs_actual_generated_error=difference(error,actual_error),
                        rgb_vs_ae=difference(rgb,clean[history:]),rgb_vs_raw=difference(rgb,raw[history:])))

        if maps is not None:
            last=core.control[-1]
            zero_init=bool((last.weight==0).all() and (last.bias is None or (last.bias==0).all()))
            shuffled=maps.roll(1,dims=2).flip(1)
            report['controls'] = dict(status='evaluated_supplied_controls', arm=control_arm,
                zero_init_adapter=zero_init,
                interpretation='No condition effect is expected for zero-init adapter.' if zero_init else 'Sensitivity does not establish control accuracy.',
                camera=tensor_summary(maps[...,:8]), motion=tensor_summary(maps[...,8:16]),
                reference=tensor_summary(maps[...,16:-1]),
                track_support_fraction=float((maps[...,10]>0).float().mean()),
                reference_support_fraction=float((maps[...,-1]>0).float().mean()),
                shuffle_kind='temporal reversal plus one-token spatial roll of supplied maps; deliberately wrong controls',
                shuffle_input_change=difference(shuffled,maps), variants=[])
            with torch.autocast(device.type,dtype=dtype,enabled=dtype!=torch.float32):
                injection=core.control(maps.flatten(1,2)).float()
            report['controls']['injection']=tensor_summary(injection)
            zero_maps=torch.zeros_like(maps)
            if control_arm=='camera':
                # Unknown-state context is fixed for every camera-only input;
                # remove the camera signal without changing that constant.
                zero_maps[...,14]=1.
                report['controls']['zero_condition']='camera_rays_zero_with_unknown_one'
            for name, alternative in (('zero',zero_maps),('shuffle',shuffled)):
                alternative_net=ControlledPrefixFlowNet(core,batch.known,alternative,model_frames=8)
                sampled=sample_from(alternative_net)
                rgb=decode(torch.cat((observed,sampled),1)*sigma+mu)[history:]
                report['controls']['variants'].append(dict(condition=name,
                    latent_change=difference(sampled,generated),rgb_change=difference(rgb,future),
                    rgb_vs_raw=difference(rgb,raw[history:])))
        report['artifacts']={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in artifacts.items()}
        return report,artifacts
    finally:
        for module,training in states:
            module.training=training
