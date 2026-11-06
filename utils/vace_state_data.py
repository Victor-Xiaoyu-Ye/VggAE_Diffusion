"""Source/reference-only states for the trained VACE editing interface.

RGB and RAE arms receive the SAME legal RGB flow proposal. The RAE state is a
representation of that complete proposal, not Stage64's small bypass residual.
All routines intentionally have no edited future target argument.
"""
import time

import torch
from torch.nn import functional as F

from utils.editing_alignment import sample_axis
from utils.editing_data import rae_canvas


ALIGNMENT = 'native_patch16_causal_anchors_with_independent_reference_v1'


def _rgb(value):
    value = torch.as_tensor(value).detach().cpu()
    if value.dtype == torch.uint8:
        value = value.float()/255
    else:
        value = value.float()
    if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
        raise ValueError('RGB inputs must be finite in [0,1] or uint8')
    return value


def _valid(spatial, device='cpu'):
    height, width = map(int, spatial['canvas'])
    y, x, h, w = map(int, spatial['valid_rect'])
    if min(y, x) < 0 or min(h, w) <= 0 or y+h > height or x+w > width:
        raise ValueError('Invalid native valid rectangle')
    result = torch.zeros(1, 1, height, width, device=device)
    result[:, :, y:y+h, x:x+w] = 1
    return result


def _to_native(value, square_rect, spatial):
    y, x, h, w = map(int, square_rect)
    top, left, hh, ww = map(int, spatial['valid_rect'])
    result = value.new_zeros(len(value), value.shape[1], *spatial['canvas'])
    result[:, :, top:top+hh, left:left+ww] = F.interpolate(
        value[:, :, y:y+h, x:x+w], (hh, ww), mode='bilinear', align_corners=False)
    return result


def build_reference(inputs64, source, anchor, cfg):
    """Expand one cached nine-state RGB proposal to the native 17-frame canvas.

    Even frames use exactly the cached proposal-minus-source edit. Odd frames
    warp the preceding edit with adjacent SOURCE flow. No separately computed
    full-rate proposal is substituted underneath a learned residual.
    """
    from utils.editing_video_transport import propagate_reference, warp_flow
    source, anchor = _rgb(source), _rgb(anchor)
    spatial = inputs64['source_spatial']
    if source.ndim != 4 or source.shape[1] != 3 or anchor.shape != source.shape[1:]:
        raise ValueError('Expected source [T,3,H,W] and matching anchor [3,H,W]')
    if list(source.shape[-2:]) != list(spatial['canvas']):
        raise ValueError('Native source canvas differs from cached metadata')
    indices = list(map(int, inputs64['observed_indices']))
    if indices != list(range(0, len(source), 2)) or indices[-1] != len(source)-1:
        raise ValueError('Expected every second source frame, including endpoints')
    old_source, proposal = _rgb(inputs64['source']), _rgb(inputs64['proposal'])
    if old_source.shape != proposal.shape or len(proposal) != len(indices):
        raise ValueError('Cached source/proposal shape differs')
    if old_source.shape[-2] != old_source.shape[-1]:
        raise ValueError('Expected square cached teacher canvas')
    size = old_source.shape[-1]
    square, rect = rae_canvas(source, spatial['valid_rect'], size=size)
    ar, ar_rect = rae_canvas(anchor[None], spatial['valid_rect'], size=size)
    if list(rect) != list(inputs64['valid_rect']) or ar_rect != rect:
        raise ValueError('Cached and native letterbox coordinates differ')
    support = torch.as_tensor(inputs64['support']).float().cpu()
    full = propagate_reference(square, ar[0], support, cfg.get('transport', {}))
    flows = full['flows_backward']
    if tuple(flows.shape) != (len(source)-1, size, size, 2):
        raise ValueError('Need adjacent source-only flow for every native time')
    effect = proposal-old_source
    masks9 = torch.as_tensor(inputs64['masks']).float().cpu().clamp(0, 1)
    conf9 = torch.as_tensor(inputs64['confidence']).float().cpu().clamp(0, 1)
    expected = (len(indices), 1, size, size)
    if tuple(masks9.shape) != expected or tuple(conf9.shape) != expected:
        raise ValueError('Cached confidence/support is not aligned')
    effects = torch.zeros_like(square)
    masks = square.new_zeros(len(source), 1, size, size)
    confidence = masks.clone()
    for j, frame in enumerate(indices):
        effects[frame], masks[frame], confidence[frame] = effect[j], masks9[j], conf9[j]
        if frame+1 < len(source):
            effects[frame+1] = warp_flow(effect[j:j+1], flows[frame])[0]
            masks[frame+1] = warp_flow(masks9[j:j+1], flows[frame])[0]
            confidence[frame+1] = (warp_flow(conf9[j:j+1], flows[frame])[0]
                                   *full['confidence'][frame+1]).clamp(0, 1)
    # Only differences cross the resize; original native source detail remains.
    reference = (source+_to_native(effects, rect, spatial)).clamp(0, 1)
    reference[0] = anchor
    valid = _valid(spatial)
    dilation = int(cfg.get('support_dilation', 9))
    if dilation < 1 or dilation % 2 != 1:
        raise ValueError('support_dilation must be positive and odd')
    masks = F.max_pool2d(masks, dilation, stride=1, padding=dilation//2)
    masks = _to_native(masks, rect, spatial).clamp(0, 1)*valid
    confidence = _to_native(confidence, rect, spatial).clamp(0, 1)*valid
    return dict(source=source, reference=reference, anchor=anchor,
        masks=masks, confidence=confidence, spatial=spatial,
        metadata=dict(proposal='unchanged_stage64_rgb_flow_no_learned_residual',
            expansion='cached_even_effect_adjacent_source_warp_odd',
            anchor_exact=bool(torch.equal(reference[0], anchor)),
            flow=full['metadata'], future_target_conditioned=False))


def calibrate_edit_states(decode, source_z, edited_z, source, proposal, masks,
                          valid_rect, cfg):
    """Fit a decodable edit DIFFERENCE to the shared legal RGB proposal.

    Each frame retains its best finite state, one decoder backward graph at a
    time. Only latent variables are optimized; autograd.grad does not populate
    decoder parameter gradients. The RGB proposal is NEVER replaced by decoding.
    """
    started = time.monotonic()
    if source_z.ndim != 4 or source_z.shape != edited_z.shape:
        raise ValueError('States must be matching [T,G,G,C]')
    source_z = source_z.detach().float()
    edited_z = edited_z.detach().float().to(source_z.device)
    if not bool(torch.isfinite(source_z).all() and torch.isfinite(edited_z).all()):
        raise ValueError('Nonfinite encoder state')
    source, proposal = _rgb(source), _rgb(proposal)
    if source.shape != proposal.shape or len(source) != len(source_z):
        raise ValueError('RGB proposal/state times differ')
    masks = torch.as_tensor(masks).detach().float().cpu().clamp(0, 1)
    if masks.shape != (len(source), 1, *source.shape[-2:]):
        raise ValueError('Calibration support must match RGB canvas')
    valid = torch.zeros_like(masks[:1])
    y, x, h, w = map(int, valid_rect)
    if min(y, x) < 0 or min(h, w) <= 0 or y+h > source.shape[-2] or x+w > source.shape[-1]:
        raise ValueError('Invalid calibration rectangle')
    valid[:, :, y:y+h, x:x+w] = 1
    options = cfg.get('calibration', cfg)
    steps = int(options.get('steps', 4))
    lr = float(options.get('lr', .03))
    trust = float(options.get('trust', .25))
    ridge = float(options.get('latent_weight', 1e-4))
    keep = float(options.get('keep_weight', 1.))
    if steps < 0 or lr <= 0 or trust <= 0 or ridge < 0 or not 0 < keep <= 1:
        raise ValueError('Invalid calibration settings')
    scale = source_z.flatten(0, 2).std(0, unbiased=False).clamp_min(.05)[None, None, None, :]
    output, reports = [], []
    for frame in range(len(source_z)):
        z = source_z[frame:frame+1][None]
        initial = edited_z[frame:frame+1][None]
        best, before, after, updates, grad_norm = initial.detach().clone(), None, None, 0, 0.
        failure = None
        try:
            truth = (proposal[frame:frame+1]-source[frame:frame+1]).to(z.device)
            weight = (valid*(keep+(1-keep)*masks[frame:frame+1])).to(z.device)
            with torch.no_grad():
                base = decode(z).detach().float()
            if base.shape != truth.shape:
                raise ValueError('Decoder output and proposal canvas differ')
            q0 = ((initial-z)/scale).detach()
            q = torch.nn.Parameter(q0.clone())
            optimizer = torch.optim.Adam([q], lr=lr)

            def fit(current):
                rgb = decode(z+scale*current).float()-base
                value = ((rgb-truth).square()*weight).sum()/(3*weight.sum()).clamp_min(1)
                if not bool(torch.isfinite(value)):
                    raise FloatingPointError('Nonfinite decoded calibration loss')
                return value

            with torch.no_grad():
                before = after = float(fit(q))
            for _ in range(steps):
                with torch.enable_grad():
                    loss = fit(q)+ridge*(q-q0).square().mean()
                    grad = torch.autograd.grad(loss, q)[0]
                if not bool(torch.isfinite(grad).all()):
                    raise FloatingPointError('Nonfinite calibration gradient')
                grad_norm = max(grad_norm, float(grad.norm()))
                q.grad = grad
                torch.nn.utils.clip_grad_norm_([q], 1.)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    q.copy_(torch.maximum(torch.minimum(q, q0+trust), q0-trust))
                    score = float(fit(q))
                    updates += 1
                    if score < after:
                        after = score
                        best = (z+scale*q).detach().clone()
        except Exception as exc:
            failure = repr(exc)
        output.append(best[0, 0].cpu())
        reports.append(dict(frame=frame, before=before, after=after, updates=updates,
                            gradient_norm=grad_norm, fallback=failure is not None, error=failure))
    valid_reports = [r for r in reports if r['before'] is not None and r['after'] is not None]
    return torch.stack(output), dict(requested_steps=steps, frames=reports,
        before=(sum(r['before'] for r in valid_reports)/len(valid_reports) if valid_reports else None),
        after=(sum(r['after'] for r in valid_reports)/len(valid_reports) if valid_reports else None),
        fallback_count=sum(r['fallback'] for r in reports),
        elapsed_seconds=time.monotonic()-started, shared_rgb_reference_unchanged=True)


def _time_indices(frames):
    if frames < 1 or (frames-1) % 4:
        raise ValueError('Native causal VAE needs 4n+1 source frames')
    return list(range(0, frames, 4))


def _auxiliary(reference):
    source, spatial = reference['source'], reference['spatial']
    valid = _valid(spatial)
    ids = _time_indices(len(source))
    values = []
    for name in ('confidence', 'masks'):
        field = torch.as_tensor(reference[name]).float().cpu()
        if field.shape != (len(source), 1, *source.shape[-2:]):
            raise ValueError('Reference confidence/support differs from RGB')
        chosen = torch.cat((field[:1], field[ids]), 0)*valid
        values.append(F.avg_pool2d(chosen, 16, stride=16))
    return torch.cat(values, 1), F.avg_pool2d(valid, 16, stride=16)


def align_rae_state(source_z, edited_z, anchor_z, inputs64, reference):
    """RAE tokens -> native patch centres with a SEPARATE reference prefix.

    Returns [2*C+2, 1+(T-1)/4+1, H/16, W/16]. Native video temporal anchors
    index actual frames 0,4,...; prefix uses independent single-frame anchor_z.
    Pixel boundaries explicitly invert teacher letterboxing and native padding.
    """
    source_z = torch.as_tensor(source_z).detach().float().cpu()
    edited_z = torch.as_tensor(edited_z).detach().float().cpu()
    anchor_z = torch.as_tensor(anchor_z).detach().float().cpu()
    if source_z.ndim != 4 or source_z.shape != edited_z.shape or anchor_z.shape != source_z[:1].shape:
        raise ValueError('Need matching [T,G,G,C] and independent [1,G,G,C] anchor')
    spatial = reference['spatial']
    canvas_h, canvas_w = map(int, spatial['canvas'])
    if canvas_h % 16 or canvas_w % 16:
        raise ValueError('Native canvas must be divisible by 16')
    ids = _time_indices(len(reference['source']))
    mapping = {int(t): i for i, t in enumerate(inputs64['observed_indices'])}
    if any(t not in mapping for t in ids):
        raise ValueError('Cached source states miss native temporal anchors')
    indices = [mapping[t] for t in ids]
    src = torch.cat((source_z[:1], source_z[indices]), 0)
    edit = torch.cat((anchor_z, edited_z[indices]), 0)
    top, left, height, width = map(int, spatial['valid_rect'])
    y, x, rh, rw = map(int, inputs64['valid_rect'])
    side_h, side_w = inputs64['source'].shape[-2:]
    yy = y+((torch.arange(canvas_h//16).float()+.5)*16-top)*rh/height
    xx = x+((torch.arange(canvas_w//16).float()+.5)*16-left)*rw/width
    yy = yy*source_z.shape[1]/side_h-.5
    xx = xx*source_z.shape[2]/side_w-.5
    value = torch.cat((src, edit-src), -1)
    value = sample_axis(sample_axis(value, yy, 1), xx, 2).permute(0, 3, 1, 2)
    auxiliary, valid = _auxiliary(reference)
    value = torch.cat((value*valid, auxiliary), 1)
    return value.permute(1, 0, 2, 3).contiguous()


def build_rgb_state(reference):
    """Pixel-patch control: 768 source + 768 edit + confidence/support.

    This receives the exact same reference as the RAE arm. It has an honest,
    slightly larger bridge input projection, no padded/fake feature dimensions.
    """
    source, proposal, anchor = (_rgb(reference[k]) for k in ('source', 'reference', 'anchor'))
    if proposal.shape != source.shape or anchor.shape != source.shape[1:]:
        raise ValueError('Reference source/proposal/anchor mismatch')
    h, w = source.shape[-2:]
    if h % 16 or w % 16:
        raise ValueError('RGB control needs the native pad16 canvas')
    valid = _valid(reference['spatial'])
    ids = _time_indices(len(source))
    src = torch.cat((source[:1], source[ids]), 0)*valid
    edits = torch.cat(((anchor-source[0])[None], (proposal-source)[ids]), 0)*valid
    values = torch.cat((F.pixel_unshuffle(src, 16), F.pixel_unshuffle(edits, 16)), 1)
    auxiliary, _ = _auxiliary(reference)
    return torch.cat((values, auxiliary), 1).permute(1, 0, 2, 3).contiguous()


def prepare_rae_state(inputs64, reference, rae, encoder, cfg, device=None):
    """Encode the complete shared proposal, calibrate, then align its state."""
    from audit_edit_transport import encode_window, rae_decoder
    if device is None:
        parameter = next(iter(rae.parameters()), None)
        device = parameter.device if parameter is not None else torch.device('cpu')
    device = torch.device(device)
    rae.eval().requires_grad_(False)
    source_z = torch.as_tensor(inputs64['source_z']).detach().float()
    proposal = _rgb(inputs64['proposal'])
    source = _rgb(inputs64['source'])
    # Never encode paired future targets. This is the complete SOURCE-derived
    # RGB proposal shared with the RGB arm, not the Stage64 learned residual.
    encoded = encode_window(rae, encoder, proposal[None].to(device))[0].detach().float()
    if encoded.shape != source_z.shape or not bool(torch.isfinite(encoded).all()):
        raise ValueError('Full reference encoder state shape/value mismatch')
    edited, diagnostics = calibrate_edit_states(rae_decoder(rae), source_z.to(device),
        encoded.to(device), source, proposal, inputs64['masks'], inputs64['valid_rect'], cfg)
    state = align_rae_state(source_z, edited, inputs64['anchor_z'], inputs64, reference)
    if not bool(torch.isfinite(state).all()):
        raise ValueError('Nonfinite aligned RAE state')
    cached = (state.half(), source_z.cpu().half(), edited.half())
    if any(not bool(torch.isfinite(value).all()) for value in cached):
        raise ValueError('RAE state overflows the FP16 cache representation')
    return dict(state=cached[0], source_z=cached[1], edited_z=cached[2],
        calibration=diagnostics, alignment=ALIGNMENT,
        state_channels=state.shape[0], native_video_frame_indices=_time_indices(len(reference['source'])),
        prefix='independent edited anchor; source slot uses observed source frame0',
        input_protocol='source_and_shared_rgb_reference_only', future_target_conditioned=False,
        cached_bytes=sum(v.numel()*v.element_size() for v in cached))
