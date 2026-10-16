"""Pure prefix-completion sampling and small, explicitly non-geometric metrics.

The sampler reads only observed latent values. Full future latents/RGB are used
after sampling for the separately labelled AE oracle and paired diagnostics.
These pixel/motion/sharpness proxies are not geometry, perceptual quality or
FVD scores. The coordinator owns EMA swaps, distributed jobs and RNG recovery.
"""
import hashlib
import json
import math

import torch
from torch.nn import functional as F

from utils.scene_prefix import PrefixFlowNet, prepare_prefix
from utils.window_flow import WindowFlow


def _frame_ids(sample):
    values = sample.get('frame_ids', [])
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    return [value.item() if hasattr(value, 'item') else value for value in values]


def stable_case_seed(sample, seed=101):
    """Stable noise identity, independent of rank, world size and checkpoint."""
    identity = dict(id=str(sample.get('id', '')), frame_ids=_frame_ids(sample), seed=int(seed))
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(payload.encode('utf8')).digest()[:8], 'big') & ((1 << 63) - 1)


def _raw_rgb(sample, total, device):
    raw = torch.as_tensor(sample['raw'], device=device)
    if raw.ndim != 4 or raw.shape[0] != total or raw.shape[1] != 3:
        raise ValueError('raw must have layout [T,3,H,W] matching the latent window')
    if raw.dtype == torch.uint8:
        raw = raw.float() / 255.
    elif raw.is_floating_point():
        raw = raw.float()
    else:
        raise ValueError('raw must be uint8 or floating point RGB in [0,1]')
    if not torch.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1:
        raise ValueError('raw RGB must be finite and in [0,1]')
    return raw.permute(0, 2, 3, 1).contiguous()


def _motion(rgb, last_observed):
    steps = torch.cat((last_observed.unsqueeze(0), rgb), dim=0).diff(dim=0).abs()
    return dict(all_steps=float(steps.mean()), seam=float(steps[0].mean()),
                internal=float(steps[1:].mean()) if len(steps) > 1 else None)


def _sharpness(rgb):
    """Mean absolute grayscale Laplacian; noise can also make this larger."""
    gray = (rgb * rgb.new_tensor([.299, .587, .114])).sum(-1).unsqueeze(1)
    kernel = gray.new_tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])[None, None]
    laplace = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='replicate'), kernel)
    return float(laplace.abs().mean())


@torch.no_grad()
def sample_case(model, rae, sample, mean, std, device, history=4, steps=32, seed=101):
    """Return ``(metrics, preview_rgb, generated_future_latents)`` on CPU.

    ``preview_rgb`` is float32 [T,H,4*W,3], with RAW/AE/GENERATED/COPY columns.
    GENERATED displays the actual observed RGB prefix, then decoded predictions.
    Returned latents are denormalized future-only [T-history,N,C]. Models are
    temporarily in eval mode; this helper neither modifies weights nor swaps EMA.
    """
    device = torch.device(device)
    target = sample['target']
    if not torch.is_tensor(target) or target.ndim != 3:
        raise ValueError('sample target must be unbatched [T,N,C]')
    total, tokens, channels = target.shape
    if isinstance(history, bool) or not isinstance(history, int) or not 1 <= history < total:
        raise ValueError('history must leave at least one observed and one future frame')
    if tokens != rae.grid ** 2 or channels != rae.channels:
        raise ValueError('sample latent layout does not match the frozen SceneRAE')
    mean = torch.as_tensor(mean, dtype=torch.float32, device=device).reshape(-1)
    std = torch.as_tensor(std, dtype=torch.float32, device=device).reshape(-1).clamp_min(1e-4)

    # No value from target[history:] enters this condition assembly. The zeros
    # exist only to use the shared prefix contract with its original slot count.
    observed = target[:history].to(device=device, dtype=torch.float32)
    condition_target = torch.cat((observed, observed.new_zeros(total-history, tokens, channels)))
    condition = {'target': condition_target}
    if 'anchor' in sample:
        condition['anchor'] = sample['anchor']
    part = prepare_prefix(condition, mean, std, hist=history, device=device)
    core = getattr(model, 'module', model)
    net = PrefixFlowNet(core, part.known, model_frames=total-1)
    noise_seed = stable_case_seed(sample, seed)
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    noise = torch.randn((1, total-history, tokens, channels), device=device,
                        dtype=torch.float32, generator=generator)
    dtype = torch.float32 if device.type == 'cpu' else torch.bfloat16
    model_training, rae_training = core.training, rae.training
    core.eval()
    rae.eval()
    try:
        normalized = WindowFlow().sample(net, part.anchor, noise, steps=steps, dtype=dtype)
        generated_z = normalized * std + mean
        # Even a decoder with temporal attention sees only known observations
        # and generated future latents. The full-GT oracle is a separate call.
        observed_z = torch.cat((part.anchor, part.known), dim=1) * std + mean
        full_z = torch.cat((observed_z, generated_z), dim=1)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type != 'cpu'):
            decoded = rae.decode(full_z.reshape(1, total, rae.grid, rae.grid, channels))[0].float()
            oracle = rae.decode(target.to(device=device, dtype=torch.float32).reshape(
                1, total, rae.grid, rae.grid, channels))[0].float()
        raw = _raw_rgb(sample, total, device)
        if decoded.shape != raw.shape or oracle.shape != raw.shape:
            raise ValueError('decoded RGB shape must match cached raw RGB')
        if not torch.isfinite(decoded).all():
            raise RuntimeError('nonfinite generated RGB')
        future = decoded[history:]
        truth = raw[history:]
        last = raw[history-1]
        copied = last.unsqueeze(0).expand_as(future)
        generated_column = torch.cat((raw[:history], future), dim=0)
        copy_column = torch.cat((raw[:history], copied), dim=0)
        motion, gt_motion = _motion(future, last), _motion(truth, last)
        nominal_fps = float(sample.get('nominal_fps', 8.))
        if not math.isfinite(nominal_fps) or nominal_fps <= 0:
            raise ValueError('nominal_fps must be positive and finite')
        decoder = getattr(rae, 'decoder', None)
        temporal_blocks = getattr(decoder, 'num_temporal_blocks', None)
        row = dict(id=sample.get('id', ''), dataset=sample.get('dataset', ''),
            seed=int(seed), noise_seed=noise_seed, frame_ids=_frame_ids(sample),
            history_frames=history, future_frames=total-history, steps=int(steps),
            nominal_fps=nominal_fps, nominal_future_seconds=(total-history)/nominal_fps,
            source_fps=sample.get('source_fps', sample.get('fps')),
            time_status=sample.get('time_status', 'Source timestamps were not supplied.'),
            timing_note='Nominal playback timing; not a substitute for source timestamps.',
            columns=['RAW', 'AE', 'GENERATED', 'COPY'],
            observed_display='GENERATED/COPY prefix is observed raw RGB, not predicted RGB.',
            metric_scope=f'future frames [{history}:{total}]; copy uses raw[{history-1}]',
            metric_note='Paired RGB/motion/Laplacian proxies only; not geometry or FVD. '
                        'Alternative futures can differ from the single truth; noise can increase sharpness.',
            raw_l1=float((future-truth).abs().mean()),
            ae_l1=float((oracle[history:]-truth).abs().mean()),
            copy_l1=float((copied-truth).abs().mean()),
            generated_motion_l1=motion['all_steps'], gt_motion_l1=gt_motion['all_steps'],
            copy_motion_l1=0., generated_seam_l1=motion['seam'], gt_seam_l1=gt_motion['seam'],
            generated_future_motion_l1=motion['internal'], gt_future_motion_l1=gt_motion['internal'],
            generated_laplacian_abs=_sharpness(future), gt_laplacian_abs=_sharpness(truth),
            ae_laplacian_abs=_sharpness(oracle[history:]), copy_laplacian_abs=_sharpness(copied),
            normalized_latent_mean=float(normalized.mean()),
            normalized_latent_std=float(normalized.std(unbiased=False)),
            decoder_temporal_blocks=temporal_blocks, has_caption=False, camera_present=False)
        preview = torch.cat((raw, oracle, generated_column, copy_column), dim=2)
        return row, preview.detach().cpu(), generated_z[0].detach().cpu()
    finally:
        core.train(model_training)
        rae.train(rae_training)
