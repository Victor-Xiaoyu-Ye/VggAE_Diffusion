"""Native-flow writes: sign, no-op, prefix isolation and actual sampler effect."""
import importlib
import unittest

import torch
from torch.nn import functional as F

from tests.test_vace_edit_pilot import cpu_native_import,tiny_native
from tests.test_vace_reference_conditioning import ObservableCodec,inputs
from utils.vace_edit_runtime import (build_reference_writeback,native_modules,
    pack_context,sample_native)


class BiasedCodec:
    """Known encoding bias exposes accidental direct overwrite/roundtrip drift."""
    def __init__(self):
        self.encode_calls=[]
        self.decode_calls=[]

    def encode(self,values):
        self.encode_calls.extend(v.clone() for v in values)
        return [v[torch.arange(16)%3,::4]+7 for v in values]

    def decode(self,values):
        self.decode_calls.extend(v.clone() for v in values)
        return [F.interpolate(v[:3][None],size=(4*(v.shape[1]-1)+1,*v.shape[-2:]),
                              mode='nearest')[0]+.2 for v in values]


def callback(codec,reference=None,mask=None,reference_count=1,**overrides):
    cfg=dict(writeback_sigmas=[.5],writeback_alpha=.5,
             writeback_max_clean_rms=10.,writeback_max_sigma=.75)
    cfg.update(overrides)
    if reference is None:reference=torch.full((3,5,2,2),.5)
    value=build_reference_writeback(codec,reference,cfg,reference_count,mask)
    value.set_schedule([1.,.5,0.])
    return value


class VaceWritebackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_encoding_difference_has_correct_velocity_sign_and_preserves_prefix(self):
        codec=BiasedCodec();write=callback(codec)
        latent=torch.zeros(16,3,2,2);latent[:,:1]=1000
        velocity=torch.zeros_like(latent)
        result=write(step_index=1,sigma=.5,latent=latent,velocity=velocity)
        # decoded=.2, reference=.5, E difference=.3 (the +7 must cancel).
        # alpha=.5 gives clean delta .15 and velocity correction -.15/.5=-.3.
        torch.testing.assert_close(result[:,1:],torch.full_like(result[:,1:],-.3))
        torch.testing.assert_close(result[:,:1],velocity[:,:1],rtol=0,atol=0)
        torch.testing.assert_close(latent-.5*result-(latent-.5*velocity),
            torch.cat((torch.zeros(16,1,2,2),torch.full((16,2,2,2),.15)),dim=1))
        self.assertEqual(tuple(codec.decode_calls[0].shape),(16,2,2,2))
        self.assertFalse(write.events[0]['zero_effect'])

    def test_reference_equals_clean_and_empty_mask_are_exact_noops_without_encode(self):
        for reference,mask in [(torch.full((3,5,2,2),.2),None),
                               (torch.ones(3,5,2,2),torch.zeros(1,5,2,2))]:
            codec=BiasedCodec();write=callback(codec,reference,mask)
            latent=torch.zeros(16,3,2,2);velocity=torch.zeros_like(latent)
            output=write(step_index=1,sigma=.5,latent=latent,velocity=velocity)
            self.assertIs(output,velocity)
            self.assertEqual(codec.encode_calls,[])
            self.assertTrue(write.events[0]['zero_effect'])

    def test_mask_only_changes_its_observed_reference_pixels(self):
        codec=BiasedCodec();mask=torch.zeros(1,5,2,2);mask[:,:,:,0]=1
        write=callback(codec,mask=mask)
        result=write(step_index=1,sigma=.5,latent=torch.zeros(16,3,2,2),
                     velocity=torch.zeros(16,3,2,2))
        proposal,current=codec.encode_calls
        torch.testing.assert_close(proposal[:,:,:,1],current[:,:,:,1],rtol=0,atol=0)
        torch.testing.assert_close(result[:,1:,:,-1],torch.zeros(16,2,2),rtol=0,atol=0)
        self.assertLess(float(result[:,1:,:,0].mean()),0.)

    def test_clean_delta_cap_is_per_channel_time_and_reported(self):
        codec=BiasedCodec();write=callback(codec,writeback_max_clean_rms=.025)
        output=write(step_index=1,sigma=.5,latent=torch.zeros(16,3,2,2),
                     velocity=torch.zeros(16,3,2,2))
        correction=-.5*output[:,1:]
        self.assertLessEqual(float(correction.square().mean((-2,-1)).sqrt().max()),.02500001)
        self.assertAlmostEqual(write.events[0]['clipped_fraction'],1.)
        self.assertAlmostEqual(write.events[0]['max_channel_clean_rms'],.025,places=6)

    def test_schedule_uses_actual_positions_once_and_reuse_resets(self):
        codec=BiasedCodec();write=callback(codec,writeback_sigmas=[.31,.3,.15])
        write.set_schedule([1.,.89,.48,.29,.14,0.])
        self.assertEqual(write.schedule,{3:.29,4:.14})
        latent=velocity=torch.zeros(16,3,2,2)
        self.assertIs(write(step_index=0,sigma=1.,latent=latent,velocity=velocity),velocity)
        write(step_index=3,sigma=.29,latent=latent,velocity=velocity)
        count=len(codec.decode_calls)
        self.assertIs(write(step_index=3,sigma=.29,latent=latent,velocity=velocity),velocity)
        self.assertEqual(len(codec.decode_calls),count)
        write.set_schedule([1.,.29,0.])
        self.assertEqual(write.events,[])
        write(step_index=1,sigma=.29,latent=latent,velocity=velocity)
        self.assertEqual(len(write.events),1)
        write.set_schedule([1.,.9,0.])
        self.assertEqual(write.schedule,{})

    def test_invalid_protocol_fails_explicitly(self):
        codec=BiasedCodec();ref=torch.zeros(3,5,2,2)
        for cfg in [dict(writeback_alpha=-1),dict(writeback_alpha=float('nan')),
                    dict(writeback_max_clean_rms=0),dict(writeback_max_sigma=1),
                    dict(writeback_sigmas=[0])]:
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):
                build_reference_writeback(codec,ref,cfg)
        for mask in [torch.zeros(5,2,2),torch.ones(1,5,2,2)*1.1,
                     torch.ones(1,5,2,2)*float('nan')]:
            with self.assertRaises(ValueError):build_reference_writeback(codec,ref,{},mask=mask)
        write=build_reference_writeback(codec,ref,{},reference_count=1)
        with self.assertRaises(RuntimeError):
            write(step_index=1,sigma=.3,latent=torch.zeros(16,3,2,2),velocity=torch.zeros(16,3,2,2))
        write.set_schedule([1.,.3,0.])
        with self.assertRaises(ValueError):
            write(step_index=1,sigma=.2,latent=torch.zeros(16,3,2,2),velocity=torch.zeros(16,3,2,2))

    def test_real_sampler_noop_equivalence_actual_sigmas_and_active_output_change(self):
        with cpu_native_import():
            torch.manual_seed(19);model=tiny_native().eval().requires_grad_(False)
            frames,mask=inputs();codec=ObservableCodec()
            condition=pack_context(codec,frames,mask,[frames[:,:1]])
            text,negative=torch.randn(5,12),torch.randn(3,12)
            cfg=dict(sample_steps=4,shift=3.,guidance=2.,context_scale=1.)
            expected=sample_native(model,condition,text,negative,cfg,seed=27)
            calls=[]
            def noop(**kw):
                calls.append((kw['step_index'],kw['sigma']))
                return kw['velocity']
            actual=sample_native(model,condition,text,negative,cfg,seed=27,velocity_callback=noop)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            prefix,_=native_modules()
            cls=importlib.import_module(prefix+'.utils.fm_solvers_unipc').FlowUniPCMultistepScheduler
            scheduler=cls(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False)
            scheduler.set_timesteps(4,device='cpu',shift=3.)
            self.assertEqual(calls,[(i,float(s)) for i,s in enumerate(scheduler.sigmas[:-1])])
            write=build_reference_writeback(codec,torch.full_like(frames,.1),
                dict(writeback_alpha=.5,writeback_sigmas=[.3,.15],writeback_max_clean_rms=.2),
                reference_count=1,mask=torch.ones(1,*frames.shape[1:]))
            changed=sample_native(model,condition,text,negative,cfg,seed=27,velocity_callback=write)
            self.assertGreater(len(write.events),0)
            self.assertTrue(any(not row['zero_effect'] for row in write.events))
            self.assertTrue(bool(torch.isfinite(changed).all()))
            self.assertGreater(float((changed-expected).abs().max()),1e-4)


if __name__=='__main__':unittest.main()
