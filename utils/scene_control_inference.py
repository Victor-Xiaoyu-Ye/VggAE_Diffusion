"""Camera/point-control view constructible without future RGB or depth.

The caller supplies relative camera rays, matching normalized point coordinates
and visibility. Existing positive track weights are treated as *binary supplied
visibility*, not calibrated confidence. This does not repair a failed tracker.
Legacy teacher motion labels and all diagnostic metadata are deliberately unread.
"""
from collections.abc import Mapping
import math

import torch

from utils.scene_control import CONTROL_SCHEMA, _xy_weights


CONTROL_INPUT_POLICY = 'supplied_camera_tracks_visibility_only_v2'


def inference_control_view(controls):
    """Return a fresh schema-compatible control dictionary; never mutate input.

    Accept unbatched [T,N,8] or batched [B,T,N,8] rays and corresponding full and
    independent-prefix tracks. Camera ray construction belongs to the caller:
    use supplied cameras and an observed-prefix-only scale. Missing coordinates
    or visibility provide no point support. Absent prefixes default to four
    observed slots, matching this experiment's 4->5 contract.

    Motion channels keep the historical layout. dx/dy are displacement from
    query frame zero; support is clipped bilinear mass. The legacy confidence
    slot means binary presence of supplied visible points. Dynamic/static stay
    zero and unknown stays one everywhere, even at supported points.
    """
    if not isinstance(controls, Mapping):
        raise ValueError('Expected supplied control dictionary')
    if controls.get('schema', CONTROL_SCHEMA) != CONTROL_SCHEMA:
        raise ValueError('Unsupported motion-control schema')
    rays=torch.as_tensor(controls['rays'],dtype=torch.float32).detach().clone()
    unbatched=rays.ndim==3
    if unbatched:
        rays=rays.unsqueeze(0)
    if rays.ndim!=4 or rays.shape[-1]!=8 or not torch.isfinite(rays).all():
        raise ValueError('Camera rays must be finite [T,N,8] or [B,T,N,8]')
    b,t,n,_=rays.shape; grid=math.isqrt(n)
    if b<1 or t<2 or grid<1 or grid*grid!=n:
        raise ValueError('Control rays require a square grid and a nonempty future')

    def read(key, ndim):
        value=controls.get(key)
        if value is None:
            return None
        value=torch.as_tensor(value,device=rays.device,dtype=torch.float32).detach().clone()
        if unbatched and value.ndim==ndim-1:
            value=value.unsqueeze(0)
        if value.ndim!=ndim or value.shape[0]!=b:
            raise ValueError(f'{key} has an incompatible batch layout')
        return value

    xy=read('tracks',4); pxy=read('prefix_tracks',4)
    weight=read('track_weight',3); pweight=read('prefix_track_weight',3)
    present=[x for x in (xy,pxy,weight,pweight) if x is not None]
    counts=[x.shape[2] for x in present]
    q=counts[0] if counts else 0
    if any(v!=q for v in counts):
        raise ValueError('Full/prefix tracks and visibility must preserve query IDs')
    h=pxy.shape[1] if pxy is not None else pweight.shape[1] if pweight is not None else min(4,t-1)
    if not 0<h<t:
        raise ValueError('Observed prefix must be shorter than the supplied window')

    def normalize_points(coords, weights, frames):
        supplied=coords is not None
        if coords is None:
            coords=rays.new_zeros((b,frames,q,2))
        if coords.shape!=(b,frames,q,2):
            raise ValueError('Coordinates must match their full/prefix frame and query layout')
        valid=torch.isfinite(coords).all(-1)&(coords>=0).all(-1)&(coords<=1).all(-1)
        if weights is None:
            weights=rays.new_zeros((b,frames,q))
        if weights.shape!=(b,frames,q):
            raise ValueError('Visibility must match point coordinates')
        visible=(torch.isfinite(weights)&(weights>0)&valid&supplied).float()
        # A finite out-of-image sentinel preserves absence if the view is
        # passed through this function again; zero would invent an origin.
        safe=torch.where(valid[...,None],coords,torch.full_like(coords,-1.))
        return safe,visible,valid

    xy,weight,valid=normalize_points(xy,weight,t)
    pxy,pweight,_=normalize_points(pxy,pweight,h)
    motion=rays.new_zeros((b,t,n,8)); motion[...,6]=1.
    if q:
        numerator=rays.new_zeros((b,t,n,2)); denominator=rays.new_zeros((b,t,n))
        # Reuse the transport's grid-center bilinear and border semantics.
        entries,_=_xy_weights(xy,grid)
        displacement=xy-xy[:,:1]
        for index,bilinear in entries:
            mass=bilinear*weight*valid[:,:1]
            denominator.scatter_add_(2,index,mass)
            numerator.scatter_add_(2,index[...,None].expand(b,t,q,2),displacement*mass[...,None])
        motion[...,:2]=numerator/denominator.clamp_min(1e-8)[...,None]
        motion[...,2]=denominator.clamp(0,1)
        motion[...,3]=(denominator>0).float()
    result=dict(schema=CONTROL_SCHEMA,rays=rays,motion=motion,tracks=xy,track_weight=weight,
        prefix_tracks=pxy,prefix_track_weight=pweight,metadata=dict(
            control_input_policy=CONTROL_INPUT_POLICY,history=h,frames=t,grid=grid,
            coordinate_system='normalized_RGB_xy',
            camera_policy='caller_supplied_camera_rays_with_independent_observed_prefix_scale',
            visibility_policy='positive_finite_in_bounds_input_weight_becomes_binary_visibility',
            confidence_channel='binary_visibility_presence_not_calibrated_confidence',
            motion_labels='dynamic_static_zero_unknown_one; no_future_depth_derived_labels',
            appearance_source='independent_observed_prefix_texture_only',
            future_depth_used=False,teacher_diagnostics_used=False,
            note='This view does not validate or restore failed teacher tracks.'))
    if unbatched:
        for key in ('rays','motion','tracks','track_weight','prefix_tracks','prefix_track_weight'):
            result[key]=result[key].squeeze(0)
    return result
