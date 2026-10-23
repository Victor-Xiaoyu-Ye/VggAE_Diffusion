"""Paired denoising probes and explicitly GT-assisted latent branch diagnostics.

These are diagnostics, not additional generation conditions or training losses.
The probe deliberately denoises corrupted ground-truth future targets; it is
not free sampling. Branch oracles deliberately use future ground truth and must
never be counted as generated samples or geometry-quality measurements.
"""
import math

import torch

from utils.scene_prefix import PrefixFlowNet, prepare_prefix
from utils.scene_prefix_eval import _frame_ids, _raw_rgb, stable_case_seed
from utils.scene_motion_objective import _activity, x0_error_metrics


def _layout(rae, sample, history):
    target = sample['target']
    if not torch.is_tensor(target) or target.ndim != 3:
        raise ValueError('sample target must be unbatched [T,N,C]')
    total, tokens, channels = target.shape
    if isinstance(history, bool) or not isinstance(history, int) or not 1 <= history < total:
        raise ValueError('history must leave observed and future frames')
    if tokens != rae.grid ** 2 or channels != rae.channels:
        raise ValueError('sample latent layout must match the frozen codec')
    if not torch.isfinite(target).all():
        raise ValueError('nonfinite target latent')
    return total, tokens, channels


def _masked_mean(error, mask):
    return float(error[mask].mean() if mask.any() else error.mean())


@torch.no_grad()
def probe_case(model, rae, sample, mean, std, device, history=4, seed=303,
               times=(.25, .5, .75, .95), *, net_factory=None):
    """Return JSON-safe, unweighted x0 errors at fixed corruption levels.

    The same case-seeded Gaussian tensor is reused across all ``times`` and
    checkpoints. The high-change mask is based only on the clean normalized
    latent differences, including the last observed -> first future transition.
    Its log1p transform and 1e-4 floor match ``x0_error_metrics`` exactly.
    It is NOT a semantic foreground, optical-flow or independently moving mask.
    Empty masks fall back to the global error, including per-frame empty masks.
    """
    device = torch.device(device)
    total, _, _ = _layout(rae, sample, history)
    times = tuple(float(value) for value in times)
    if not times or any(not math.isfinite(u) or not 0 < u <= 1 for u in times):
        raise ValueError('probe times must be nonempty finite values in (0,1]')
    batch = prepare_prefix(sample, mean, std, hist=history, device=device)
    previous = batch.known[:, -1:] if history > 1 else batch.anchor
    activity_floor = 1e-4
    score = _activity(batch.target, previous, log_activity=True)
    threshold = score.mean(dim=(1, 2), keepdim=True)
    mask = (score >= threshold) & (score > activity_floor)
    noise_seed = stable_case_seed(sample, seed)
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    noise = torch.randn(batch.target.shape, device=device, dtype=torch.float32,
                        generator=generator)
    core = getattr(model, 'module', model)
    net = (PrefixFlowNet(core, batch.known, model_frames=total-1) if net_factory is None
           else net_factory(core, batch, sample, training=False))
    states = [(module, module.training) for module in core.modules()]
    dtype = torch.float32 if device.type == 'cpu' else torch.bfloat16
    rows, errors = [], []
    core.eval()
    try:
        for value in times:
            u = torch.tensor([value], device=device, dtype=torch.float32)
            noisy = (1-value)*batch.target + value*noise
            with torch.autocast(device.type, dtype=dtype, enabled=device.type != 'cpu'):
                prediction = net(noisy, u, batch.anchor)
            error = (prediction.float()-batch.target).square().mean(dim=-1)
            if not torch.isfinite(error).all():
                raise RuntimeError('nonfinite denoising probe error')
            common = x0_error_metrics(prediction, batch.target, previous,
                                      log_activity=True, activity_floor=activity_floor)
            errors.append(error)
            rows.append(dict(u=value, probe_global_mse=common['x0_mse'],
                probe_change_mse=common['changed_region_x0_mse'],
                per_future_frame=[dict(frame_index=history+i,
                    global_mse=float(error[:, i].mean()),
                    change_mse=_masked_mean(error[:, i], mask[:, i]))
                    for i in range(total-history)]))
    finally:
        for module, training in states:
            module.training = training
    average = torch.stack(errors).mean(0)
    return dict(schema='scene-motion-denoising-probe-v1', id=str(sample.get('id', '')),
        frame_ids=_frame_ids(sample), seed=int(seed), noise_seed=noise_seed,
        history_frames=history, future_frames=total-history, times=list(times),
        kind='GT-corrupted future denoising validation; NOT free generation',
        supervision='Future GT enters the noisy supervised target, never the observed conditions.',
        metric_note='Unweighted normalized x0 MSE; no training loss/time/motion weights.',
        mask_definition='log1p(RMS adjacent clean normalized latent change) >= case mean and > 1e-4; '
                        'includes last observed boundary; NOT semantic foreground or optical flow.',
        activity_transform='log1p', activity_floor=activity_floor,
        change_threshold=float(threshold), change_fraction=float(mask.float().mean()),
        empty_change_fallback=not bool(mask.any()),
        probe_global_mse=float(average.mean()), probe_change_mse=_masked_mean(average, mask),
        per_time=rows, per_future_frame=[dict(frame_index=history+i,
            global_mse=float(average[:, i].mean()),
            change_mse=_masked_mean(average[:, i], mask[:, i]),
            change_fraction=float(mask[:, i].float().mean()),
            empty_change_fallback=not bool(mask[:, i].any()))
            for i in range(total-history)])


@torch.no_grad()
def oracle_branches(rae, sample, generated_z, device, history=4):
    """Return ``(metadata, preview[T,H,5*W,3])`` using already-sampled latents.

    ``generated_z`` must be actual denormalized future-only [T-history,N,C].
    The codec contract is equal contiguous geo/texture halves, as in SceneRAE.
    The two hybrid columns replace one FUTURE latent branch with cached GT.
    They are GT-assisted counterfactual reconstructions, not generated videos.
    Even a temporal decoder receives the actual observed anchor/prefix in every
    branch; no future GT enters the ordinary GEN decode.
    """
    device = torch.device(device)
    total, tokens, channels = _layout(rae, sample, history)
    if channels < 2 or channels % 2:
        raise ValueError('branch oracle requires equal contiguous geo/texture channel halves')
    if not torch.is_tensor(generated_z) or generated_z.shape != (total-history, tokens, channels):
        raise ValueError('generated_z must be denormalized future-only [T-history,N,C]')
    generated = generated_z.to(device=device, dtype=torch.float32).unsqueeze(0)
    if not torch.isfinite(generated).all():
        raise ValueError('nonfinite generated latent')
    batch = prepare_prefix(sample, torch.zeros(channels), torch.ones(channels),
                           hist=history, device=device)
    observed = torch.cat((batch.anchor, batch.known), dim=1)
    half = channels//2
    futures = {
        'AE': batch.target,
        'GEN': generated,
        'GT_GEO+PRED_TEX': torch.cat((batch.target[..., :half], generated[..., half:]), dim=-1),
        'PRED_GEO+GT_TEX': torch.cat((generated[..., :half], batch.target[..., half:]), dim=-1),
    }
    states = [(module, module.training) for module in rae.modules()]
    dtype = torch.float32 if device.type == 'cpu' else torch.bfloat16
    decoded = {}
    rae.eval()
    try:
        for label, future in futures.items():
            full = torch.cat((observed, future), dim=1)
            with torch.autocast(device.type, dtype=dtype, enabled=device.type != 'cpu'):
                rgb = rae.decode(full.reshape(1, total, rae.grid, rae.grid, channels))[0].float()
            if not torch.isfinite(rgb).all():
                raise RuntimeError('nonfinite oracle decoded RGB')
            decoded[label] = rgb
    finally:
        for module, training in states:
            module.training = training
    raw = _raw_rgb(sample, total, device)
    if any(rgb.shape != raw.shape for rgb in decoded.values()):
        raise ValueError('oracle decoded RGB must match cached raw RGB shape')
    displays = {'RAW': raw, 'AE': decoded['AE']}
    for label in ('GEN', 'GT_GEO+PRED_TEX', 'PRED_GEO+GT_TEX'):
        displays[label] = torch.cat((raw[:history], decoded[label][history:]), dim=0)
    columns = list(displays)
    error = generated-batch.target
    branch_errors = {'geo': error[..., :half].square().mean(dim=(0, 2, 3)),
                     'texture': error[..., half:].square().mean(dim=(0, 2, 3))}
    frame_rows = []
    transitions = []
    for index in range(history, total):
        frame_rows.append(dict(frame_index=index,
            raw_l1={label: float((decoded[label][index]-raw[index]).abs().mean())
                    for label in decoded},
            gen_vs_ae_l1=float((decoded['GEN'][index]-decoded['AE'][index]).abs().mean()),
            latent_branch_mse={label: float(values[index-history])
                               for label, values in branch_errors.items()}))
        transitions.append(dict(from_frame=index-1, to_frame=index,
            is_observed_boundary=index == history,
            rgb_difference={label: float((rgb[index]-rgb[index-1]).abs().mean())
                            for label, rgb in displays.items()}))
    metadata = dict(schema='scene-latent-branch-oracle-v1', id=str(sample.get('id', '')),
        frame_ids=_frame_ids(sample), history_frames=history, future_frames=total-history,
        columns=columns, gt_assisted_columns=['AE', 'GT_GEO+PRED_TEX', 'PRED_GEO+GT_TEX'],
        kind='Diagnostic GT-assisted branch swaps; NOT extra free-generation samples.',
        metric_note='Branch names denote learned channel provenance, not GT depth/geometry labels. '
                    'RGB difference is not flow or geometric consistency. Hybrid quality is not sample quality.',
        observed_display='GEN and hybrids show raw observed prefix; AE shows decoded observed prefix.',
        transition_note='Differences follow displayed columns; GEN/hybrid boundary uses last raw observed '
                        'frame, AE boundary uses reconstructed observed frame.',
        latent_branch_ranges=dict(geo=[0, half], texture=[half, channels]),
        latent_branch_mse={label: float(values.mean()) for label, values in branch_errors.items()},
        per_future_frame=frame_rows, per_transition=transitions)
    return metadata, torch.cat(list(displays.values()), dim=2).detach().cpu()
