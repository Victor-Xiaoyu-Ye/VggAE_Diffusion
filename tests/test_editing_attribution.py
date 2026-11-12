"""Exercise actual source-only operators and native tiny VACE integration."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from utils.editing_attribution import (ARMS, arm_spec, supported_rms, smooth_effect,
    fit_video_vae, NativeEffectFeedback, MagnitudeProjector)
from audit_edit_attribution import assignments, validate_config, projector
from tests.test_editing_state_feedback import Codec, fixture
from tests.test_vace_edit_pilot import cpu_native_import
from tests.test_vace_state_runtime import backbone, config
from utils.vace_state_runtime import StateConditionedVACE, sample_state_native

ROOT=Path(__file__).resolve().parents[1]


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_config_and_interventions(self):
        cfg=json.loads((ROOT/'configs/editing_attribution_v1.json').read_text())
        validate_config(cfg)
        for name in ARMS:
            spec=arm_spec(name)
            if name.startswith('op_'):self.assertEqual(spec['family'],'sft')
            if 'fixed_bridge' in name:self.assertTrue(spec['fixed_bridge'])
        self.assertTrue(arm_spec('cal_on_fixed_bridge_on')['online'])
        self.assertFalse(arm_spec('cal_off_fixed_bridge_off')['online'])
        with self.assertRaises(ValueError):arm_spec('rae_full')
        cfg['frames']=1
        with self.assertRaises(ValueError):validate_config(cfg)

    def test_smoke_visits_all_nodes_and_full_has_no_duplicates(self):
        self.assertEqual([r for r in range(24) if assignments(list(range(3)),r,24)],[0,8,16])
        found=sum([assignments(list(range(32)),r,24) for r in range(24)],[])
        self.assertEqual(sorted(found),list(range(32)))

    def test_smoothing_is_spatial_only(self):
        x=torch.arange(17).float()[:,None,None,None].expand(17,3,16,16)/17
        self.assertTrue(torch.allclose(x,smooth_effect(x,2),atol=1e-6))
        x=torch.zeros(17,3,16,16);x[8,:,8,8]=1
        result=smooth_effect(x,2)
        self.assertEqual(float(result[:8].abs().sum()),0)
        self.assertLess(float(result[8,:,8,8].max()),1)

    def test_full_video_vae_fit_and_gradient(self):
        class RecordingCodec(Codec):
            def encode(self, videos):
                self.encoded_shapes=[list(v.shape) for v in videos]
                return super().encode(videos)
        codec=RecordingCodec()
        torch.manual_seed(4)
        source=torch.rand(17,3,16,32)*.3+.3;proposal=source+.12
        effect,report=fit_video_vae(codec,source,proposal,torch.ones(17,1,16,32),
            dict(steps=2,lr=.03,trust=.25),torch.device('cpu'))
        self.assertEqual(effect.shape,source.shape)
        self.assertEqual(codec.encoded_shapes,[[3,17,16,32]]*2)
        self.assertEqual(report['latent_shape'][1],5)
        self.assertGreater(max(report['gradient_norms']),0)
        self.assertLessEqual(report['after'],report['before'])
        self.assertTrue(torch.isfinite(effect).all())

    def test_native_operator_prefix_and_live_write(self):
        inputs,reference,state,_=fixture()
        codec=Codec();cfg=json.loads((ROOT/'configs/editing_attribution_v1.json').read_text())['feedback']
        cb=NativeEffectFeedback(codec,reference,reference['reference']-reference['source'],cfg)
        cb.set_schedule(torch.tensor([1.,.6,.35,.15,0.]))
        latent=torch.randn(16,6,4,6)*.1;velocity=torch.randn_like(latent)*.05
        result=cb(step_index=1,sigma=.6,latent=latent,velocity=velocity)
        self.assertTrue(cb.events[-1]['applied'],cb.events)
        self.assertTrue(torch.equal(result[:,:1],velocity[:,:1]))
        self.assertGreater(float((result[:,1:]-velocity[:,1:]).abs().max()),0)
        self.assertTrue(cb.consume_history_reset());self.assertFalse(cb.consume_history_reset())
        duplicate=cb(step_index=1,sigma=.6,latent=latent,velocity=velocity)
        self.assertTrue(torch.equal(duplicate,velocity))

    def test_native_causal_vae_checkpointed_full_video_gradient(self):
        # Exercise the actual Wan causal cache/recomputation implementation,
        # at reduced width; the simple Codec test alone cannot validate this.
        with cpu_native_import():
            import importlib
            from utils.vace_edit_runtime import native_modules
            WanVAE_=importlib.import_module(native_modules()[0]+'.modules.vae').WanVAE_
            model=WanVAE_(dim=8,z_dim=4,num_res_blocks=1).eval().requires_grad_(False)
            class TinyNative:
                def encode(self,videos):
                    return [model.encode(v[None],[0.,1.])[0] for v in videos]
                def decode(self,latents):
                    return [model.decode(z[None],[0.,1.])[0].clamp(-1,1) for z in latents]
            source=torch.rand(17,3,16,16)*.2+.4
            effect,report=fit_video_vae(TinyNative(),source,source+.05,torch.ones(17,1,16,16),
                dict(steps=1,lr=.01,trust=.1),torch.device('cpu'))
            self.assertEqual(effect.shape,source.shape)
            self.assertTrue(torch.isfinite(effect).all())
            self.assertGreater(report['gradient_norms'][0],0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_native_operator_failure_is_exact_noop(self):
        _,reference,_,_=fixture();cfg=json.loads((ROOT/'configs/editing_attribution_v1.json').read_text())['feedback']
        cb=NativeEffectFeedback(Codec(fail_encode=True),reference,reference['reference']-reference['source'],cfg)
        cb.set_schedule([1.,.6,.15]);x=torch.randn(16,6,4,6);v=torch.randn_like(x)
        self.assertTrue(torch.equal(cb(step_index=1,sigma=.6,latent=x,velocity=v),v))
        self.assertTrue(cb.events[-1]['fallback']);self.assertFalse(cb.consume_history_reset())

    def test_same_tiny_generator_effect_changes_final_video(self):
        torch.manual_seed(17)
        with cpu_native_import():
            net=StateConditionedVACE(backbone(),config(),1538).eval()
            _,reference,_,_=fixture()
            cb=NativeEffectFeedback(Codec(),reference,reference['reference']-reference['source'],
                json.loads((ROOT/'configs/editing_attribution_v1.json').read_text())['feedback'])
            condition=torch.zeros(96,6,4,6);text=torch.randn(4,12)
            settings=dict(config(),sample_steps=6,shift=3,guidance=1.)
            plain=sample_state_native(net,condition,text,text,None,settings,42,use_state=False)
            edited=sample_state_native(net,condition,text,text,None,settings,42,use_state=False,velocity_callback=cb)
            self.assertTrue(cb.events)
            self.assertTrue(all(e['applied'] for e in cb.events),cb.events)
            self.assertGreater(float((plain-edited).abs().max()),0)
            self.assertEqual(float(condition.abs().sum()),0)

    def test_magnitude_control_retains_anchor_and_state(self):
        class Fixed:
            def __init__(self,d):self.d=d
            def state_dict(self):return {}
            def load_state_dict(self,s):pass
            def project(self,current,progress=None):
                reference=current+self.d;reference[0]=.6
                return dict(reference=reference,delta=torch.tensor(self.d),diagnostics={})
        current=torch.full((3,3,8,8),.3)
        wrapped=MagnitudeProjector(Fixed(.05),Fixed(.1),torch.ones(3,1,8,8))
        result=wrapped.project(current)
        self.assertTrue(torch.allclose(result['reference'][1:],current[1:]+.1))
        self.assertTrue(torch.all(result['reference'][0]==.6))
        self.assertAlmostEqual(float(result['delta']),.05)
        self.assertFalse(result['diagnostics']['magnitude_control']['target_access'])

    def test_online_toggle_changes_solver_not_cached_states(self):
        inputs,reference,state,_=fixture('rae_joint')
        inputs.update(source=(torch.ones(9,3,32,32)*.3*255).byte(),anchor=(torch.ones(3,32,32)*.4*255).byte(),
            proposal=(torch.ones(9,3,32,32)*.4*255).byte(),masks=torch.ones(9,1,32,32),
            confidence=torch.ones(9,1,32,32),scale=torch.ones(256),edges=[])
        cal=dict(source_z=inputs['source_z'],raw=inputs['source_z']+.01,calibrated=inputs['source_z']+.02)
        original=copy.deepcopy(cal)
        def decode(z):
            return torch.nn.functional.interpolate(z[0,:,:,:,:3].permute(0,3,1,2),size=(32,32),mode='bilinear')
        options=dict(sweeps=2,steps=2,joint_weight=0.)
        off=projector(inputs,cal,decode,options,'cpu',False,False)
        on=projector(inputs,cal,decode,options,'cpu',True,True)
        self.assertEqual(off.steps,0);self.assertEqual(on.steps,4)
        self.assertTrue(torch.allclose(off._delta(),torch.full_like(cal['raw'],.01),atol=1e-6))
        self.assertTrue(torch.allclose(on._delta(),torch.full_like(cal['raw'],.02),atol=1e-6))
        for k in cal:self.assertTrue(torch.equal(cal[k],original[k]))


if __name__=='__main__':unittest.main()
