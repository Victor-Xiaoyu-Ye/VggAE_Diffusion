"""Explicit video/image coordinates and training-only edit-support diagnostics.

Difference masks are supervision proxies, never semantic labels or inference
conditions. All routines use ordinary differentiable tensor operations on NPU.
"""
import torch
from torch.nn import functional as F


ALIGNMENT = 'letterbox_pixel_centres_v1'


def sample_axis(value, positions, dim):
    """Linear interpolation at explicit index coordinates; border extension.

    Avoid 5D grid_sample, whose accelerator backward support is less portable.
    Indices/weights do not carry gradients, but the sampled features do.
    """
    positions = positions.to(device=value.device, dtype=torch.float32)
    low = positions.floor().long()
    high = low + 1
    shape = [1] * value.ndim
    shape[dim] = len(positions)
    weight = (positions - low.float()).reshape(shape)
    a = value.index_select(dim, low.clamp(0, value.shape[dim]-1))
    b = value.index_select(dim, high.clamp(0, value.shape[dim]-1))
    return a + (b-a)*weight


def align_readout(pred, spatial, teacher_rect, target_shape, frames, stride=4):
    """B,T,H,W,C native hidden grid -> teacher canvas, in pixel coordinates.

    Native temporal slots are indexed by causal-VAE anchors 0,4,... . This
    defines indexing, not an assertion that a causal latent is a single frame.
    Teacher frame indices equal those used by prepare_vace_edit.cache_teacher.
    Both native pad16 and teacher letterbox offsets are explicitly inverted.
    """
    if pred.ndim != 5 or len(target_shape) != 3:
        raise ValueError('Expected B,T,H,W,C and target T,H,W')
    if pred.shape[1] != (frames-1)//stride+1 or (frames-1) % stride:
        raise ValueError('Native time grid does not match the declared causal VAE anchors')
    nt, nh, nw = target_shape
    top, left, height, width = spatial['valid_rect']
    canvas_h, canvas_w = spatial['canvas']
    y, x, rh, rw = teacher_rect
    if min(height, width, rh, rw) <= 0:
        raise ValueError('Empty image rectangle')
    device = pred.device
    ids = torch.linspace(0, frames-1, nt, device=device).round()
    yy = (torch.arange(nh, device=device).float()+.5)*518/nh
    xx = (torch.arange(nw, device=device).float()+.5)*518/nw
    # Coordinate zero is the top/left image boundary, not a pixel centre.
    yy = (top+(yy-y)*height/rh)*pred.shape[2]/canvas_h-.5
    xx = (left+(xx-x)*width/rw)*pred.shape[3]/canvas_w-.5
    value = sample_axis(pred.float(), ids/stride, 1)
    value = sample_axis(value, yy, 2)
    return sample_axis(value, xx, 3)


def teacher_valid_mask(rect, shape, device):
    top, left, height, width = [int(v) for v in rect]
    pixels = torch.zeros(1, 1, 518, 518, device=device)
    pixels[:, :, top:top+height, left:left+width] = 1
    return F.interpolate(pixels, size=shape, mode='area')


def change_support(source, target, threshold=16/255, dilation=5):
    """T,1,H,W soft support from paired training/evaluation RGB only.

    Detect local changes after a small blur; dilate to include uncertain
    boundaries. Camera shifts, bad specialist targets and compression can
    still create incorrect support. Do not call this an object/motion mask.
    """
    if source.shape != target.shape or source.ndim != 4 or source.shape[1] != 3:
        raise ValueError('Expected paired T,3,H,W uint8-range RGB')
    if dilation < 1 or dilation % 2 != 1:
        raise ValueError('Support dilation must be a positive odd size')
    delta = (source.float()-target.float()).abs().amax(1, keepdim=True)/255
    delta = F.avg_pool2d(delta, 3, stride=1, padding=1)
    support = (delta > threshold).float()
    return F.max_pool2d(support, dilation, stride=1, padding=dilation//2)


def support_on_teacher(source, target, spatial, rect, count, cfg, pixel_support=None):
    """CPU preparation; caller transfers only small masks needed by its loss."""
    ids = torch.linspace(0, len(source)-1, count).round().long()
    top, left, h, w = spatial['valid_rect']
    support = (change_support(source[ids, :, top:top+h, left:left+w],
                              target[ids, :, top:top+h, left:left+w],
                              cfg.get('edit_support_threshold', 16/255),
                              cfg.get('edit_support_dilation', 5)) if pixel_support is None
               else pixel_support[ids, :, top:top+h, left:left+w])
    y, x, rh, rw = rect
    resized = F.interpolate(support, size=(rh, rw), mode='nearest')
    return F.pad(resized, (x, 518-rw-x, y, 518-rh-y))[:, 0]


def weighted_mean(error, weight):
    """One mean per batch; zero-area regions are finite and differentiable."""
    weight = torch.broadcast_to(weight, error.shape)
    return (error*weight).flatten(1).sum(1)/weight.flatten(1).sum(1).clamp_min(1e-6)


def region_mean(error, valid, support, fraction=.5):
    if support is None:
        return weighted_mean(error, valid)
    edit, keep = valid*support, valid*(1-support)
    edit_ok = torch.broadcast_to(edit, error.shape).flatten(1).sum(1) > 1e-6
    keep_ok = torch.broadcast_to(keep, error.shape).flatten(1).sum(1) > 1e-6
    a, b = float(fraction)*edit_ok.float(), (1-float(fraction))*keep_ok.float()
    return (a*weighted_mean(error, edit)+b*weighted_mean(error, keep))/(a+b).clamp_min(1e-6)


def balanced_flow_loss(pred, target, support, spatial, cfg):
    """Ordinary VAE-space region baseline, with the SAME proxy as RAE arm.

    The support of a causal latent is the union of RGB frames in its group:
    frame0, frames1..4, frames5..8, ... . It is a coarse influence proxy, not an
    exact inverse of the VAE receptive field. Never used by the sampler.
    """
    t = pred.shape[2]
    if len(support) != 4*(t-1)+1:
        raise ValueError('Flow mask and causal VAE time dimensions differ')
    masks = torch.stack([support[0]]+[support[4*i-3:4*i+1].amax(0) for i in range(1,t)])
    masks = F.interpolate(masks.float(), size=pred.shape[-2:], mode='area')[:, 0][None].to(pred.device)
    ch,cw = spatial['canvas']; y,x,h,w = spatial['valid_rect']
    valid = torch.zeros(1,1,ch,cw,device=pred.device); valid[:,:,y:y+h,x:x+w]=1
    valid = F.interpolate(valid,size=pred.shape[-2:],mode='area')[:,0,None]
    error = (pred.float()-target.float()).square().mean(1)
    loss = region_mean(error,valid,masks,cfg.get('flow_edit_fraction',.5)).mean()
    return loss,dict(flow_edit=float(weighted_mean(error,valid*masks).detach().mean()),
                     flow_keep=float(weighted_mean(error,valid*(1-masks)).detach().mean()),
                     flow_edit_fraction=float(weighted_mean(masks,valid).mean()))


def aligned_rae_loss(pred, target, decoder, sigma, cfg, frame, rect, spatial, support=None):
    pred = align_readout(pred, spatial, rect, target.shape[1:4], cfg['frames'])
    valid = teacher_valid_mask(rect, target.shape[2:4], pred.device)[:, 0, None]
    # Exclude padding from normalization as well as the final loss.
    weights = valid[..., None].expand_as(target)
    denom = weights.sum((0, 1, 2, 3)).clamp_min(1)
    mean = (target.detach()*weights).sum((0, 1, 2, 3))/denom
    scale = (((target.detach()-mean).square()*weights).sum((0, 1, 2, 3))/denom).sqrt().clamp_min(.1)
    latent_error = ((pred-target)/scale).square().mean(-1)
    latent_support = None
    if support is not None:
        support = support.to(pred.device).float()
        latent_support = F.interpolate(support[:, None], size=target.shape[2:4], mode='area')[:, 0][None]
    latent = region_mean(latent_error, valid, latent_support, cfg.get('rae_edit_fraction', .5))
    # The black RGB letterbox does NOT encode into zero RAE tokens. For this
    # training-only decoder loss, use detached teacher padding context on both
    # sides, so artificial padding cannot create a spurious boundary objective.
    # No target tokens ever enter the main velocity prediction or inference.
    z = pred[:, frame:frame+1]*valid[:, :1, ..., None] + target[:, frame:frame+1].detach()*(1-valid[:, :1, ..., None])
    with torch.no_grad(): truth = decoder.decode(target[:, frame:frame+1]).float()
    rgb_error = (decoder.decode(z).float()-truth).abs().mean(-1)
    rgb_valid = teacher_valid_mask(rect, rgb_error.shape[-2:], pred.device)
    rgb_support = None if support is None else F.interpolate(support[frame:frame+1, None], size=rgb_error.shape[-2:], mode='area')
    rgb = region_mean(rgb_error, rgb_valid, rgb_support, cfg.get('rae_edit_fraction', .5))
    active = (sigma <= cfg['rae_max_sigma']).float()
    addition = (active*(cfg['rae_latent_weight']*latent+cfg['rae_rgb_weight']*rgb)).mean()
    stats = dict(rae_latent=float(latent.detach().mean()), rae_rgb=float(rgb.detach().mean()),
                 rae_active=float(active.mean()), rae_aligned=True)
    if latent_support is not None:
        stats['edit_support_fraction'] = float(weighted_mean(latent_support, valid).mean())
    return addition, stats


def hidden_gradient_diagnostics(flow, addition, hidden):
    """Diagnostics at a shared activation, not extra parameter backward calls."""
    a = torch.autograd.grad(flow, hidden, retain_graph=True, allow_unused=True)[0]
    b = torch.autograd.grad(addition, hidden, retain_graph=True, allow_unused=True)[0]
    if a is None or b is None:
        return dict(aux_probe_active=0., aux_flow_cosine=0., aux_flow_grad_ratio=0.)
    a, b = a.detach().float(), b.detach().float()
    na, nb = a.norm(), b.norm()
    active = float(na > 1e-12 and nb > 1e-12)
    return dict(aux_probe_active=active,
                aux_flow_cosine=float((a*b).sum()/(na*nb).clamp_min(1e-12)) if active else 0.,
                aux_flow_grad_ratio=float(nb/na.clamp_min(1e-12)) if active else 0.)


def region_pixel_report(source, target, generated, cfg, external_mask=None):
    """Post-generation diagnostics. Target data is NEVER used as conditioning."""
    mask = (change_support(source, target, cfg.get('edit_support_threshold', 16/255),
                           cfg.get('edit_support_dilation', 5)) if external_mask is None else external_mask.float())
    a, b, c = [v.float()/255 for v in (source, target, generated)]
    def mean(error, region):
        if float(region.sum()) < 1e-6: return None
        return float((error.mean(1, keepdim=True)*region).sum()/region.sum())
    return dict(edit_region_target_l1=mean((c-b).abs(), mask),
                keep_region_source_l1=mean((c-a).abs(), 1-mask),
                keep_region_target_l1=mean((c-b).abs(), 1-mask),
                edit_region_output_change=mean((c-a).abs(), mask),
                edit_region_target_change=mean((b-a).abs(), mask),
                edit_support_fraction=float(mask.mean()),
                copy_source_target_psnr=float(-10*(a-b).square().mean().clamp_min(1e-10).log10()),
                keep_temporal_delta_error=mean(((c[1:]-c[:-1])-(a[1:]-a[:-1])).abs(),
                                               (1-mask[1:])*(1-mask[:-1])),
                region_metric_caveat=('User-provided source-video region; not geometry/semantic accuracy.' if external_mask is not None
                    else 'Paired-RGB difference proxy only, NOT semantic object annotation, instruction adherence or geometry accuracy.'))
