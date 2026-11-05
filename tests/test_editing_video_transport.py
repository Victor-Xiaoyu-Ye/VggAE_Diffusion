"""Natural-video transport contracts, graph gradients and exact sweep recovery.

OpenCV fixtures isolate pixel correspondence contracts; an actual DIS test also
checks a textured known translation. The actual decoder solver/autograd is used.
Neither synthetic motion check establishes natural-video correspondence quality.
"""
import copy
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from utils.editing_video_transport import propagate_reference,calibrate_video,warp_flow


def known_flow_backend(reject=False):
    calls=[]
    class DIS:
        def calc(self,a,b,initial):
            calls.append((a.copy(),b.copy()))
            flow=np.zeros((*a.shape,2),dtype=np.float32)
            flow[...,0]=2*(int(b[0,0])-int(a[0,0]))
            if reject:flow[...,1]=1000.
            return flow
    module=types.ModuleType('cv2')
    module.COLOR_RGB2GRAY=1;module.INTER_AREA=2;module.INTER_LINEAR=3
    module.DISOPTICAL_FLOW_PRESET_MEDIUM=2;module.error=RuntimeError
    module.cvtColor=lambda x,kind:x[:,:,0].copy()
    def resize(x,size,interpolation):
        assert x.shape[:2]==(size[1],size[0]),'This exact-pixel fixture does not resize'
        return x.copy()
    module.resize=resize;module.DISOpticalFlow_create=lambda preset:DIS()
    return module,calls


def moving_source(frames=4):
    generator=torch.Generator().manual_seed(51)
    first=torch.rand(3,32,48,generator=generator)*.4+.25
    values=[]
    for i in range(frames):
        image=torch.zeros_like(first)
        if i:image[:,:,2*i:]=first[:,:,:-2*i]
        else:image.copy_(first)
        image[:,0,0]=i/255
        values.append(image)
    source=torch.stack(values)
    support=torch.zeros(1,32,48);support[:,8:24,12:28]=1
    return source,source[0]+support*torch.tensor([.1,-.05,.08])[:,None,None],support


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Conv2d(3,3,1,bias=False)
        with torch.no_grad():self.linear.weight.copy_(torch.eye(3)[:,:,None,None])
        self.requires_grad_(False)
    def forward(self,z):return self.linear(z[:,0].permute(0,3,1,2))


def inverse_fixture():
    z=torch.zeros(1,3,6,8,3)
    source=torch.full((3,3,6,8),.4)
    target=source+torch.tensor([.06,.09,.12])[:,None,None,None]
    mask=torch.ones(3,1,6,8);conf=mask.clone();flow=torch.zeros(2,6,8,2)
    cfg=dict(iterations=16,lr=.4,trust=6.,ridge=1e-5,keep_weight=1.,
             joint_weight=.25,initialization_sweeps=2,checkpoint_every=1,anchor_weight=4.)
    return TinyDecoder(),z,source,target,conf,mask,cfg,flow


class VideoTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_pixel_warp_direction_border_and_backward_gradient(self):
        image=torch.arange(5*7).reshape(1,5,7).float().requires_grad_()
        flow=torch.zeros(5,7,2);flow[...,0]=-2
        result=warp_flow(image,flow)
        torch.testing.assert_close(result[:,:,2:],image[:,:,:-2])
        self.assertEqual(float(result[:,:,:2].abs().sum()),0.)
        result.sum().backward()
        expected=torch.ones_like(image);expected[:,:,-2:]=0
        torch.testing.assert_close(image.grad,expected)
        # Fractional displacement has bilinear, not nearest-token, semantics.
        half=flow*.25
        torch.testing.assert_close(warp_flow(image.detach(),half)[:,:,1:],
            .5*(image.detach()[:,:,1:]+image.detach()[:,:,:-1]))

    def test_source_only_translation_moves_effect_and_support(self):
        source,anchor,support=moving_source()
        cv,calls=known_flow_backend()
        with patch.dict(sys.modules,{'cv2':cv}):
            value=propagate_reference(source,anchor,support,dict(flow_max_side=512))
        self.assertEqual(tuple(value['flows_backward'].shape),(3,32,48,2))
        torch.testing.assert_close(value['flows_backward'][...,0],torch.full((3,32,48),-2.))
        for i in range(4):
            expected=torch.zeros_like(support);expected[:,8:24,12+2*i:28+2*i]=1
            torch.testing.assert_close(value['masks'][i],expected,atol=1e-6,rtol=0)
            expected_effect=expected*torch.tensor([.1,-.05,.08])[:,None,None]
            torch.testing.assert_close(value['proposal'][i]-source[i],expected_effect,atol=1e-6,rtol=0)
        self.assertEqual(value['metadata']['correspondence_source'],'source_rgb_only')
        # All estimator inputs are source frames; changing the anchor cannot
        # alter computed flow or confidence.
        with patch.dict(sys.modules,{'cv2':cv}):
            changed=propagate_reference(source,source[0],support,dict(flow_max_side=512))
        torch.testing.assert_close(value['flows_backward'],changed['flows_backward'],rtol=0,atol=0)
        torch.testing.assert_close(value['confidence'],changed['confidence'],rtol=0,atol=0)
        torch.testing.assert_close(changed['proposal'],source,rtol=0,atol=0)
        self.assertGreater(len(calls),6) # includes direct-anchor reconnection.

    def test_unreliable_out_of_frame_correspondence_keeps_source(self):
        source,anchor,support=moving_source(2);cv,_=known_flow_backend(reject=True)
        with patch.dict(sys.modules,{'cv2':cv}):
            value=propagate_reference(source,anchor,support,{})
        self.assertEqual(float(value['confidence'][1].sum()),0.)
        torch.testing.assert_close(value['proposal'][1],source[1],rtol=0,atol=0)
        torch.testing.assert_close(value['proposal'][0],anchor,rtol=0,atol=0)

    def test_actual_dis_textured_motion_and_propagated_edit(self):
        try:import cv2
        except ImportError:self.skipTest('OpenCV unavailable in this validation environment')
        cv2.setNumThreads(2)
        generator=torch.Generator().manual_seed(541)
        first=torch.rand(3,96,128,generator=generator)*.6+.2
        frames=[]
        for t in range(4):
            image=torch.zeros_like(first)
            if t:image[:,t:,3*t:]=first[:,:-t,:-3*t]
            else:image.copy_(first)
            frames.append(image)
        source=torch.stack(frames)
        support=torch.zeros(1,96,128);support[:,24:64,32:80]=1
        color=torch.tensor([.12,-.04,.07])[:,None,None]
        result=propagate_reference(source,source[0]+support*color,support,
            dict(flow_backend='dis',flow_max_side=384))
        self.assertEqual(result['metadata']['flow_backend'],'dis')
        self.assertEqual(result['metadata']['flow_fallback_count'],0)
        central=result['flows_backward'][:,20:-20,24:-24]
        epe=(central-torch.tensor([-3.,-1.])).square().sum(-1).sqrt()
        self.assertLess(float(epe.mean()),.5)
        for t in range(1,4):
            expected_mask=torch.zeros_like(support)
            expected_mask[:,24+t:64+t,32+3*t:80+3*t]=1
            actual_effect=result['proposal'][t]-source[t]
            self.assertLess(float((actual_effect-expected_mask*color).abs().mean()),.002)
            interior=result['confidence'][t,:,28+t:60+t,36+3*t:76+3*t]
            self.assertGreater(float(interior.mean()),.75)

    def test_zero_edit_seventeen_to_nine_and_native_canvas_preserves_details(self):
        try:import cv2
        except ImportError:self.skipTest('OpenCV unavailable in this validation environment')
        from utils.editing_data import rae_canvas
        from edit_reference_video import from_square
        cv2.setNumThreads(2)
        source=torch.rand(17,3,24,40,generator=torch.Generator().manual_seed(68))
        spatial=dict(canvas=[24,40],valid_rect=[1,2,22,36])
        square,srect=rae_canvas(source,spatial['valid_rect'],size=32)
        support=torch.ones(1,32,32)
        full=propagate_reference(square,square[0],support,dict(flow_max_side=64))
        indices=list(range(0,17,2))
        coarse=propagate_reference(square[indices],square[0],support,dict(flow_max_side=64))
        self.assertEqual(tuple(full['flows_backward'].shape),(16,32,32,2))
        self.assertEqual(tuple(coarse['flows_backward'].shape),(8,32,32,2))
        torch.testing.assert_close(full['proposal'],square,rtol=0,atol=0)
        torch.testing.assert_close(coarse['proposal'],square[indices],rtol=0,atol=0)
        # Only an edit difference may go through resize. Source content must
        # not be replaced by a roundtrip-resized image, even with perfect masks.
        restored=source+from_square(full['proposal']-square,srect,spatial)
        torch.testing.assert_close(restored,source,rtol=0,atol=0)

    def test_calibration_real_input_gradient_and_joint_improvement(self):
        model,z,source,proposal,conf,mask,cfg,flow=inverse_fixture()
        result=calibrate_video(model,z,source,proposal,conf,mask,cfg,flows_backward=flow)
        initial_error=float((proposal-source).square().mean())
        actual_error=float((result['effect']-(proposal-source)).square().mean())
        self.assertLess(actual_error,initial_error*.15)
        self.assertTrue(result['gradient_checked'])
        self.assertEqual(result['metadata']['temporal_edges'],2)
        self.assertTrue(result['history'][-1]['joint_active'])
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        # Strengthening graph coupling changes the actual edit states and
        # reduces disagreement, not just a logged auxiliary metric.
        independent=calibrate_video(model,z,source,proposal,conf,mask,
            dict(cfg,joint_weight=0.,iterations=40),flows_backward=flow)
        coupled=calibrate_video(model,z,source,proposal,conf,mask,
            dict(cfg,joint_weight=8.,iterations=40),flows_backward=flow)
        def jump(v):return float((v['effect'][1:]-v['effect'][:-1]).square().mean())
        self.assertLess(jump(coupled),jump(independent)*.5)

    def test_exact_resume_after_sweep_with_adam_and_best_graph(self):
        model,z,source,proposal,conf,mask,cfg,flow=inverse_fixture()
        full=calibrate_video(model,z,source,proposal,conf,mask,cfg,flows_backward=flow)
        saved={}
        def interrupt(state):
            if state['sweep']==5:
                saved.update(copy.deepcopy(state));raise InterruptedError('simulated interruption')
        with self.assertRaises(InterruptedError):
            calibrate_video(model,z,source,proposal,conf,mask,cfg,
                flows_backward=flow,checkpoint=interrupt)
        resumed=calibrate_video(model,z,source,proposal,conf,mask,cfg,
            flows_backward=flow,resume=saved)
        torch.testing.assert_close(full['delta'],resumed['delta'],rtol=0,atol=0)
        torch.testing.assert_close(full['effect'],resumed['effect'],rtol=0,atol=0)
        self.assertEqual(full['history'],resumed['history'])
        self.assertTrue(all(t.device.type=='cpu' for o in saved['optimizers']
            for s in o['state'].values() for t in s.values() if isinstance(t,torch.Tensor)))

    def test_no_edit_identity_and_explicit_missing_graph(self):
        model,z,source,_,conf,mask,cfg,flow=inverse_fixture()
        identity=calibrate_video(model,z,source,source,conf,mask,cfg,flows_backward=flow)
        self.assertEqual(float(identity['effect'].abs().sum()),0.)
        self.assertEqual(float(identity['delta'].abs().sum()),0.)
        with self.assertRaisesRegex(ValueError,'requires source flow'):
            calibrate_video(model,z,source,source,conf,mask,cfg)

    def test_detached_decoder_fails_before_claiming_calibration(self):
        model,z,source,proposal,conf,mask,cfg,flow=inverse_fixture()
        with self.assertRaisesRegex(RuntimeError,'detached'):
            calibrate_video(lambda v:model(v).detach(),z,source,proposal,conf,mask,cfg,
                flows_backward=flow)


if __name__=='__main__':unittest.main()
