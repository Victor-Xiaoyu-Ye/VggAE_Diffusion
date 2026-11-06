"""Real tiny VACE contracts for trained state injection, prefix and sampling."""
import copy
import unittest

import torch
from torch import nn

from tests.test_vace_edit_pilot import cpu_native_import,tiny_native
from tests.test_vace_reference_conditioning import ObservableCodec,inputs
from utils.vace_edit_runtime import decode_native,native_forward,pack_context,sample_native
from utils.vace_state_runtime import (StateConditionedVACE,pack_edit_context,
    prepend_reference_clean,sample_state_native,state_runtime_preflight)


def backbone():
    model=tiny_native()
    # Released control weights are trained, unlike the constructor's zeros.
    for block in model.vace_blocks:
        for name in ('before_proj','after_proj'):
            if hasattr(block,name):
                nn.init.normal_(getattr(block,name).weight,std=.08)
    return model


def config():
    return dict(lora_rank=2,lora_alpha=2,bridge_width=8,context_scale=1.,
                sample_steps=3,shift=3.,guidance=2.)


def batch():
    return dict(x=[torch.randn(16,3,4,6)],t=torch.tensor([400.]),
                conditions=[torch.randn(96,3,4,6)],text=[torch.randn(5,12)],
                states=[torch.randn(7,3,2,3)])


class StateRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.importer=cpu_native_import();self.importer.__enter__()
        self.addCleanup(self.importer.__exit__,None,None,None)
        torch.manual_seed(37)

    def test_reference_active_source_inactive_and_prefix_are_exact(self):
        source,mask=inputs();reference=-source;anchor=torch.full_like(source[:,:1],.6)
        codec=ObservableCodec()
        condition=pack_edit_context(codec,source,reference,mask,anchor)
        expected=pack_context(ObservableCodec(),source,mask,[anchor])
        binary=(mask>.5).float()
        expected[16:32,1:]=ObservableCodec.encode_one(reference*binary)
        torch.testing.assert_close(condition,expected,rtol=0,atol=0)
        torch.testing.assert_close(codec.encoded_inputs[0],source*(1-binary),rtol=0,atol=0)
        torch.testing.assert_close(codec.encoded_inputs[1],reference*binary,rtol=0,atol=0)
        torch.testing.assert_close(codec.encoded_inputs[2],anchor,rtol=0,atol=0)
        outside=reference*binary+(source*.17)*(1-binary)
        same=pack_edit_context(ObservableCodec(),source,outside,mask,anchor)
        torch.testing.assert_close(same,condition,rtol=0,atol=0)
        changed=pack_edit_context(ObservableCodec(),source,source,mask,anchor)
        torch.testing.assert_close(changed[:16],condition[:16],rtol=0,atol=0)
        self.assertGreater(float((changed[16:32,1:]-condition[16:32,1:]).abs().sum()),0)

    def test_clean_prefix_comes_from_condition_and_is_trimmed_before_decode(self):
        source,mask=inputs();anchor=-source[:,:1]
        condition=pack_edit_context(ObservableCodec(),source,source,mask,anchor)
        target=ObservableCodec.encode_one(source*.3)
        clean=prepend_reference_clean(target,condition)
        torch.testing.assert_close(clean[:,:1],ObservableCodec.encode_one(anchor),rtol=0,atol=0)
        torch.testing.assert_close(clean[:,1:],target,rtol=0,atol=0)
        changed=prepend_reference_clean(target+7,condition)
        torch.testing.assert_close(changed[:,:1],clean[:,:1],rtol=0,atol=0)
        codec=ObservableCodec();decoded=decode_native(codec,clean,1)
        self.assertEqual(tuple(decoded.shape),tuple(source.shape))
        torch.testing.assert_close(codec.decoded_inputs[0],target,rtol=0,atol=0)

    def test_zero_bridge_matches_old_native_with_and_without_state(self):
        native=backbone().eval();values=batch()
        with torch.no_grad():
            expected=native_forward(native,values['x'],values['t'],values['conditions'],values['text'])[0][0]
            model=StateConditionedVACE(native,config(),7).eval()
            enabled=model(**values)[0];disabled=model(**values,use_state=False)[0]
        torch.testing.assert_close(enabled,expected,rtol=0,atol=0)
        torch.testing.assert_close(disabled,expected,rtol=0,atol=0)
        self.assertEqual(float(model.bridge[-1].weight.abs().sum()),0.)

    def test_real_flow_updates_bridge_and_lora_frozen_base_has_no_grad(self):
        model=StateConditionedVACE(backbone(),config(),7).train();values=batch()
        target=torch.randn_like(values['x'][0])[None]
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        before=model.deltas()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss=(model(**values)-target).square().mean();loss.backward()
            self.assertGreater(float(model.bridge[-1].weight.grad.abs().sum()),0.)
            self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if n in frozen))
            optimizer.step()
        self.assertGreater(float(model.bridge[0].weight.grad.abs().sum()),0.)
        after=model.deltas()
        self.assertTrue(any(not torch.equal(before[n],after[n]) for n in before if n.startswith('bridge.')))
        self.assertTrue(any(not torch.equal(before[n],after[n]) for n in before if n.startswith('backbone.')))
        for n,p in model.named_parameters():
            if n in frozen:torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
        model.eval()
        with torch.no_grad():
            output=model(**values)
            alternate=values['states'][0].clone();alternate[0]+=2
            other=dict(values,states=[alternate])
            perturbed=model(**other);disabled=model(**values,use_state=False)
        self.assertGreater(float((output-perturbed).abs().max()),1e-5)
        self.assertGreater(float((output-disabled).abs().max()),1e-5)

    def test_checkpoint_recompute_matches_plain_state_gradients(self):
        model=StateConditionedVACE(backbone(),config(),7)
        nn.init.normal_(model.bridge[-1].weight,std=.03)
        other=copy.deepcopy(model).eval();model.train();values=batch()
        a=model(**values);b=other(**values)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.square().mean().backward();b.square().mean().backward()
        left=dict(model.named_parameters());right=dict(other.named_parameters())
        for name,p in left.items():
            if p.requires_grad:torch.testing.assert_close(p.grad,right[name].grad,rtol=1e-5,atol=1e-7)

    def test_sampler_calls_wrapper_with_state_in_both_cfg_branches(self):
        model=StateConditionedVACE(backbone(),config(),7).train();values=batch()
        nn.init.normal_(model.bridge[-1].weight,std=.2)
        condition,text,state=values['conditions'][0],values['text'][0],values['states'][0]
        negative=torch.randn(3,12);calls=[]
        def observe(module,args,kwargs):
            calls.append((args[1].clone(),args[3][0].clone(),args[4][0].clone(),kwargs['use_state']))
        handle=model.register_forward_pre_hook(observe,with_kwargs=True)
        output=sample_state_native(model,condition,text,negative,state,config(),83)
        handle.remove()
        self.assertTrue(model.training)
        self.assertEqual(len(calls),2*config()['sample_steps'])
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertEqual(tuple(output.shape),(16,3,4,6))
        for i,(_,used_text,used_state,enabled) in enumerate(calls):
            torch.testing.assert_close(used_state,state,rtol=0,atol=0)
            torch.testing.assert_close(used_text,text if i%2==0 else negative,rtol=0,atol=0)
            self.assertTrue(enabled)
        changed_state=state.clone();changed_state[0]+=3
        alternate=sample_state_native(model,condition,text,negative,changed_state,config(),83)
        self.assertGreater(float((output-alternate).abs().max()),1e-5)
        disabled=sample_state_native(model,condition,text,negative,state,config(),83,use_state=False)
        native=sample_native(model.backbone,condition,text,negative,config(),83)
        torch.testing.assert_close(disabled,native,rtol=0,atol=0)

    def test_trainable_delta_roundtrip_and_state_shape_are_strict(self):
        model=StateConditionedVACE(backbone(),config(),7).eval();values=batch()
        initial=model.deltas()
        nn.init.normal_(model.bridge[-1].weight,std=.1)
        learned=model.deltas();model.restore(initial)
        self.assertEqual(float(model.bridge[-1].weight.abs().sum()),0.)
        model.restore(learned)
        for n,p in model.named_parameters():
            if p.requires_grad:torch.testing.assert_close(p,learned[n],rtol=0,atol=0)
        for state in (torch.zeros(1,7,2,2,3),torch.zeros(1,6,3,2,3),torch.zeros(1,7,3,3,2)):
            with self.assertRaises(ValueError):model(**dict(values,states=state))
        with self.assertRaises(ValueError):model.restore({})
        with self.assertRaises(ValueError):StateConditionedVACE(model.backbone,config(),7)
        with self.assertRaises(ValueError):native_forward(model.backbone,values['x'],values['t'],values['conditions'],values['text'],state_tokens=torch.zeros(1,4,32))

    def test_fp16_cached_state_samples_under_bf16_autocast(self):
        model=StateConditionedVACE(backbone(),config(),7).eval();values=batch()
        nn.init.normal_(model.bridge[-1].weight,std=.2)
        half=values['states'][0].half()
        condition=values['conditions'][0].bfloat16()
        text=values['text'][0].bfloat16();negative=torch.randn(3,12).bfloat16()
        actual=sample_state_native(model,condition,text,negative,half,config(),83)
        expected=sample_state_native(model,condition,text,negative,half.float(),config(),83)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
            listed=model(**dict(values,states=[half]))
            batched=model(**dict(values,states=half[None]))
        torch.testing.assert_close(listed,batched,rtol=0,atol=0)

    def test_context_rejects_future_reference_shapes_and_invalid_values(self):
        source,mask=inputs();reference=-source;anchor=source[:,:1]
        for bad in (source,torch.zeros(3,1,16,16),torch.full_like(anchor,float('nan')),torch.full_like(anchor,2.)):
            with self.assertRaises(ValueError):pack_edit_context(ObservableCodec(),source,reference,mask,bad)
        with self.assertRaises(ValueError):pack_edit_context(ObservableCodec(),source,reference,mask*2,anchor)
        condition=pack_edit_context(ObservableCodec(),source,reference,mask,anchor)
        with self.assertRaises(ValueError):prepend_reference_clean(torch.zeros(16,1,4,6),condition)

    def test_callable_device_preflight_uses_both_real_representation_widths(self):
        report=state_runtime_preflight(torch.device('cpu'),config())
        self.assertTrue(report['passed'])
        self.assertEqual([r['state_channels'] for r in report['representations']],[514,1538])
        for row in report['representations']:
            self.assertGreater(row['bridge_gradient_l1'],0)
            self.assertGreater(row['trained_state_velocity_max_delta'],0)
            self.assertEqual(row['zero_bridge_rms'],0)


if __name__=='__main__':unittest.main()
