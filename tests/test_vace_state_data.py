"""Source-only state construction, temporal-prefix and spatial contracts."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from utils.vace_state_data import (build_reference, calibrate_edit_states,
    align_rae_state, build_rgb_state, prepare_rae_state)


def fake_flow(source, anchor, support, cfg):
    return dict(proposal=torch.full_like(source,.99),
        confidence=torch.ones(len(source),1,*source.shape[-2:]),
        masks=torch.ones(len(source),1,*source.shape[-2:]),
        flows_backward=torch.zeros(len(source)-1,*source.shape[-2:],2),
        metadata=dict(correspondence_source='source_rgb_only', fixture=True))


def fixture():
    source=torch.full((17,3,16,32),.2)
    source[:,0]+=torch.arange(17)[:,None,None]*.005
    anchor=source[0].clone();anchor[2,:,8:24]+=.2
    from utils.editing_data import rae_canvas
    square,rect=rae_canvas(source[::2],[0,0,16,32],size=32)
    ar,_=rae_canvas(anchor[None],[0,0,16,32],size=32)
    proposal=square.clone()
    for i in range(9):proposal[i,2,8:24,8:24]+=.1+i*.01
    proposal[0]=ar[0]
    masks=torch.zeros(9,1,32,32);masks[:,:,8:24,8:24]=1
    z=torch.arange(9,dtype=torch.float32)[:,None,None,None].expand(9,4,4,3).clone()
    inputs=dict(source=square,proposal=proposal,source_z=z,
        anchor_z=torch.full((1,4,4,3),20.),anchor=ar[0],
        masks=masks,confidence=torch.ones_like(masks),support=masks[0],
        observed_indices=list(range(0,17,2)),valid_rect=rect,
        source_spatial=dict(canvas=[16,32],valid_rect=[0,0,16,32]))
    return inputs,source,anchor


class TinyRAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain=nn.Parameter(torch.tensor(.65),requires_grad=False)

    def decode(self,z):
        b,t,h,w,c=z.shape
        rgb=F.interpolate(z[...,:3].permute(0,1,4,2,3).flatten(0,1),
            (32,32),mode='bilinear',align_corners=False)
        return (self.gain*rgb).permute(0,2,3,1).reshape(b,t,32,32,3)


def tiny_encode(rae,encoder,frames):
    b,t,c,h,w=frames.shape
    return F.adaptive_avg_pool2d(frames.flatten(0,1),(4,4)).permute(0,2,3,1).reshape(b,t,4,4,c)


class NoTargetDict(dict):
    def __getitem__(self,key):
        if key in ('target','supervision','future_target'):
            raise AssertionError('Edited future target must not be read')
        return super().__getitem__(key)


class StateDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def reference(self):
        inputs,source,anchor=fixture()
        with patch('utils.editing_video_transport.propagate_reference',fake_flow):
            reference=build_reference(inputs,source,anchor,dict(support_dilation=1))
        return inputs,reference

    def test_reference_uses_cached_proposal_not_full_rate_flow_proposal(self):
        inputs,source,anchor=fixture();before=copy.deepcopy(inputs)
        with patch('utils.editing_video_transport.propagate_reference',fake_flow):
            result=build_reference(inputs,source,anchor,dict(support_dilation=1))
        self.assertTrue(torch.equal(result['reference'][0],anchor))
        # Fake full-flow proposal is .99; that output must never be used.
        self.assertLess(float(result['reference'].max()),.6)
        for index,t in enumerate(range(2,17,2),1):
            expected=source[t].clone();expected[2,:,8:24]+=.1+index*.01
            torch.testing.assert_close(result['reference'][t],expected,rtol=0,atol=1e-7)
        # Zero adjacent flow transports each preceding coarse edit onto the
        # next true source frame; source motion/detail is never copied forward.
        for t in range(1,17,2):
            expected=source[t]+(result['reference'][t-1]-source[t-1])
            torch.testing.assert_close(result['reference'][t],expected,rtol=0,atol=1e-7)
        for key in ('source','proposal','source_z','anchor_z'):
            torch.testing.assert_close(inputs[key],before[key],rtol=0,atol=0)

    def test_odd_frame_uses_spatial_source_flow(self):
        inputs,source,anchor=fixture()
        def translated(*args):
            value=fake_flow(*args)
            value['flows_backward'][...,0]=1
            return value
        with patch('utils.editing_video_transport.propagate_reference',translated):
            result=build_reference(inputs,source,anchor,dict(support_dilation=1))
        # Flow pulls input x+1; the rightmost mask/effect boundary shifts left.
        self.assertAlmostEqual(float(result['reference'][3,2,5,22]-source[3,2,5,22]),.11,places=6)
        self.assertAlmostEqual(float(result['reference'][3,2,5,23]-source[3,2,5,23]),0.,places=6)

    def test_rgb_prefix_and_video_indices_are_separate(self):
        _,ref=self.reference()
        state=build_rgb_state(ref)
        self.assertEqual(state.shape,(1538,6,1,2))
        source_patches=state[:768].permute(1,0,2,3)
        edit_patches=state[768:1536].permute(1,0,2,3)
        source=F.pixel_shuffle(source_patches,16)
        edit=F.pixel_shuffle(edit_patches,16)
        torch.testing.assert_close(source[0],ref['source'][0])
        torch.testing.assert_close(edit[0],ref['anchor']-ref['source'][0])
        for k,t in enumerate([0,4,8,12,16],1):
            torch.testing.assert_close(source[k],ref['source'][t])
            torch.testing.assert_close(edit[k],ref['reference'][t]-ref['source'][t])

    def test_rae_prefix_does_not_mix_with_video_time(self):
        inputs,ref=self.reference()
        source=inputs['source_z'];edited=source+3
        state=align_rae_state(source,edited,inputs['anchor_z'],inputs,ref)
        self.assertEqual(state.shape,(8,6,1,2))
        torch.testing.assert_close(state[:3,0],torch.zeros(3,1,2))
        torch.testing.assert_close(state[3:6,0],torch.full((3,1,2),20.))
        for k,v in enumerate([0,2,4,6,8],1):
            torch.testing.assert_close(state[:3,k],torch.full((3,1,2),float(v)))
            torch.testing.assert_close(state[3:6,k],torch.full((3,1,2),3.))

    def test_rae_alignment_inverts_letterbox_and_respects_native_padding(self):
        inputs,ref=self.reference()
        # Teacher rectangle [8,0,16,32] is a letterbox of native16x32. A token
        # field encoding square y pixel-centres should sample y=16, not8.
        yy=(torch.arange(4)+.5)*8
        field=yy[None,:,None,None].expand(9,4,4,3).clone()
        result=align_rae_state(field,field+1,field[:1]+2,inputs,ref)
        torch.testing.assert_close(result[:3,1],torch.full((3,1,2),16.))
        # Add a whole padding patch and verify it contributes exactly zero.
        expanded=copy.deepcopy(ref)
        expanded['spatial']=dict(canvas=[32,32],valid_rect=[16,0,16,32])
        for name in ('source','reference','masks','confidence'):
            expanded[name]=F.pad(expanded[name],(0,0,16,0))
        expanded['anchor']=F.pad(expanded['anchor'],(0,0,16,0))
        padded=align_rae_state(field,field+1,field[:1]+2,inputs,expanded)
        self.assertEqual(float(padded[:,:,0].abs().max()),0.)
        torch.testing.assert_close(padded[:3,1,1],torch.full((3,2),16.))
        rgb=build_rgb_state(expanded)
        self.assertEqual(float(rgb[:,:,0].abs().max()),0.)

    def test_calibration_updates_latents_only_and_never_worsens_fit(self):
        torch.manual_seed(8)
        teacher=TinyRAE();original=copy.deepcopy(teacher.state_dict())
        source_z=torch.full((3,4,4,3),.2)
        edit_z=source_z+.08
        source=torch.full((3,3,32,32),.2)
        proposal=source+.15
        masks=torch.ones(3,1,32,32)
        from audit_edit_transport import rae_decoder
        edited,report=calibrate_edit_states(rae_decoder(teacher),source_z,edit_z,
            source,proposal,masks,[0,0,32,32],dict(calibration=dict(steps=8,lr=.2,trust=5.)))
        self.assertLess(report['after'],report['before'])
        self.assertGreater(float((edited-edit_z).abs().max()),0.)
        self.assertTrue(all(r['updates']==8 and r['gradient_norm']>0 for r in report['frames']))
        self.assertEqual(report['fallback_count'],0)
        self.assertIsNone(teacher.gain.grad)
        torch.testing.assert_close(teacher.state_dict()['gain'],original['gain'],rtol=0,atol=0)

    def test_portrait_alignment_does_not_stretch_teacher_padding(self):
        inputs,ref=self.reference()
        inputs=copy.deepcopy(inputs);ref=copy.deepcopy(ref)
        inputs['valid_rect']=[0,8,32,16]
        ref['spatial']=dict(canvas=[32,16],valid_rect=[0,0,32,16])
        for name in ('source','reference','masks','confidence'):
            ref[name]=ref[name].transpose(-1,-2)
        ref['anchor']=ref['anchor'].transpose(-1,-2)
        xx=(torch.arange(4)+.5)*8
        field=xx[None,None,:,None].expand(9,4,4,3).clone()
        value=align_rae_state(field,field+1,field[:1]+2,inputs,ref)
        self.assertEqual(value.shape,(8,6,2,1))
        torch.testing.assert_close(value[:3,1],torch.full((3,2,1),16.))

    def test_calibration_failure_keeps_finite_encoder_state(self):
        source_z=torch.zeros(2,2,2,3);edit_z=source_z+.1
        source=torch.zeros(2,3,8,8);proposal=source+.2
        def invalid(z):return torch.full((1,3,8,8),float('nan'))
        edited,report=calibrate_edit_states(invalid,source_z,edit_z,source,proposal,
            torch.ones(2,1,8,8),[0,0,8,8],dict(calibration=dict(steps=2)))
        torch.testing.assert_close(edited,edit_z,rtol=0,atol=0)
        self.assertEqual(report['fallback_count'],2)
        self.assertTrue(all(r['error'] for r in report['frames']))

    def test_complete_proposal_encoded_not_future_target_or_tiny_residual(self):
        inputs,ref=self.reference()
        inputs=NoTargetDict(inputs,target=torch.full((17,3,32,32),float('nan')))
        # Give the source states their actual tiny-encoder representation.
        teacher=TinyRAE()
        inputs['source_z']=tiny_encode(teacher,None,inputs['source'][None])[0]
        inputs['anchor_z']=tiny_encode(teacher,None,inputs['anchor'][None,None])[0]
        seen=[]
        def record(rae,encoder,frames):
            seen.append(frames.detach().cpu().clone())
            return tiny_encode(rae,encoder,frames)
        before=ref['reference'].clone()
        with patch('audit_edit_transport.encode_window',record):
            state=prepare_rae_state(inputs,ref,teacher,None,dict(calibration=dict(steps=0)),device='cpu')
        self.assertEqual(len(seen),1)
        torch.testing.assert_close(seen[0][0],inputs['proposal'])
        self.assertEqual(state['state'].shape,(8,6,1,2))
        self.assertGreater(float((state['edited_z']-state['source_z']).abs().sum()),0.)
        torch.testing.assert_close(ref['reference'],before,rtol=0,atol=0)
        self.assertFalse(state['future_target_conditioned'])
        self.assertEqual(state['input_protocol'],'source_and_shared_rgb_reference_only')


if __name__=='__main__':unittest.main()
