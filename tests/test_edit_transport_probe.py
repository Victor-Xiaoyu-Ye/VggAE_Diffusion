"""Real reverse gradients, withheld-observation isolation, exact resume and IO."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from utils.edit_transport_probe import (transform_grid,warp,observation_mask,rgb_completion,effect_metrics,solve_effect)
from utils.scene_run import ArtifactStore
from tests.test_vace_edit_pilot import cpu_native_import

REPO=Path(__file__).resolve().parents[1]


def config():
    value=json.loads((REPO/'configs/edit_transport_probe_v1.json').read_text())
    value.update(cases=1,target_indices=[8],strengths=[1.],observed_fractions=[.25],noise_stds=[0.],
        iterations=2,checkpoint_every=1,log_every=1,rgb_fill_iterations=3)
    return value


class FakeRAE(nn.Module):
    def encode(self,tokens,frames):
        b,t,c,h,w=frames.shape
        z=F.adaptive_avg_pool2d(frames.flatten(0,1),(18,18)).permute(0,2,3,1)
        return z[:,:,:,torch.arange(256)%3].reshape(b,t,18,18,256)
    def decode(self,z):
        b,t,h,w,c=z.shape
        rgb=F.interpolate(z[...,:3].reshape(b*t,h,w,3).permute(0,3,1,2),(518,518),mode='bilinear',align_corners=False)
        return rgb.permute(0,2,3,1).reshape(b,t,518,518,3)


class FakeEncoder(nn.Module):
    def forward(self,frames):return [frames],0


class FakeNative(nn.Module):
    def encode(self,x,scale):return F.adaptive_avg_pool3d(x,(1,18,18))
    def decode(self,z,scale):return F.interpolate(z,size=(1,528,528),mode='trilinear',align_corners=False)


def fixture(root,cases=1):
    cfg=config();cfg['cases']=cases
    data=ArtifactStore(root/'data',[]);bank=dict(identity='stage60-fixed',records=[])
    for i in range(cases):
        source=torch.randint(20,220,(17,3,32,48),generator=torch.Generator().manual_seed(i),dtype=torch.uint8)
        name=f'native/{i}.pt';data.save(name,dict(source=source,target='NEVER READ THIS',
            timing=dict(spatial=dict(canvas=[32,48],valid_rect=[0,0,32,48]))),bank['identity'])
        bank['records'].append(dict(id=str(i),file=name,split='dev'))
    # Reserved test must never be selected or loaded.
    bank['records'].append(dict(id='reserved',file='missing-test.pt',split='test'))
    data.write_json('bank.json',bank)
    for name in ('rae.pt','encoder.pt','Wan2.1_VAE.pth'):(root/name).write_bytes(name.encode())
    args=SimpleNamespace(rae=str(root/'rae.pt'),encoder=str(root/'encoder.pt'),model=str(root),phase='rae')
    return cfg,data,args


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_known_pixel_mapping_and_sparse_observation_seed(self):
        image=torch.arange(8*12).reshape(1,1,8,12).float()
        torch.testing.assert_close(warp(image,transform_grid(8,12)),image)
        shifted=warp(image,transform_grid(8,12,shift=(2.,0.)))
        torch.testing.assert_close(shifted[:,:,:,2:],image[:,:,:,:-2])
        mask=observation_mask(torch.ones(1,1,32,32),.25,8,42,torch.ones(1,1,32,32))
        self.assertGreater(float(mask.sum()),0);self.assertLess(float(mask.sum()),32*32)

    def test_anchor_dose_uses_actual_decoder_effect_not_linear_rescaling(self):
        from audit_edit_transport import anchor_effects
        base=torch.ones(1,3,4,4)*1.2;delta=torch.ones_like(base)*.7
        effects=anchor_effects(torch.sigmoid,base,delta,[.5,1.])
        torch.testing.assert_close(effects[.5],torch.sigmoid(base+.5*delta)-torch.sigmoid(base))
        self.assertFalse(torch.allclose(effects[.5],effects[1.]*.5))

    def test_nonfinite_and_detached_decoders_fail_visibly_per_cell(self):
        import audit_edit_transport as audit
        cfg=config();base=torch.zeros(1,3,8,8);truth=torch.ones_like(base)*.1;mask=torch.ones(1,1,8,8)
        with tempfile.TemporaryDirectory(dir=REPO.parent) as temp,contextlib.redirect_stdout(io.StringIO()):
            out=ArtifactStore(temp,[])
            row=audit.cell(out,'id','nonfinite',dict(method='bad'),base,truth,mask,mask,prediction=base*float('nan'))
            self.assertEqual(row['status'],'failed')
            row=audit.cell(out,'id','detached',dict(method='bad'),base,truth,mask,mask,
                decode=lambda x:x.detach().sigmoid(),base=base,scale=torch.ones_like(base),cfg=cfg)
            self.assertEqual(row['status'],'failed')

    def test_solver_exact_interruption_resume_and_frozen_parameters(self):
        model=nn.Conv2d(3,3,1,bias=False).requires_grad_(False)
        with torch.no_grad():model.weight.copy_(torch.eye(3)[:,:,None,None])
        decode=lambda z:model(z).sigmoid()
        cfg=config();cfg.update(iterations=12,log_every=2,checkpoint_every=4,lr=.1)
        base=torch.zeros(1,3,8,8);target=torch.ones_like(base)*.1;mask=torch.ones(1,1,8,8)
        full=solve_effect(decode,base,target,mask,torch.ones_like(base),cfg)
        with tempfile.TemporaryDirectory(dir=REPO.parent) as temp:
            store=ArtifactStore(temp,[])
            def checkpoint(state):
                store.save('optimizer.pt',state,'id')
                if state['step']==4:raise InterruptedError('simulated job stop')
            with self.assertRaises(InterruptedError):solve_effect(decode,base,target,mask,torch.ones_like(base),cfg,checkpoint=checkpoint)
            resumed=solve_effect(decode,base,target,mask,torch.ones_like(base),cfg,resume=store.load('optimizer.pt','id',True))
        torch.testing.assert_close(full['delta'],resumed['delta'],rtol=0,atol=0)
        torch.testing.assert_close(full['effect'],resumed['effect'],rtol=0,atol=0)
        self.assertTrue(full['gradient_checked']);self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertLess(float((full['effect']-target).square().mean()),float(target.square().mean()))

    def test_withheld_targets_cannot_change_solver_state(self):
        import audit_edit_transport as audit
        base=torch.zeros(1,3,8,8);mask=torch.zeros(1,1,8,8);mask[:,:,:4]=1
        observed=torch.ones_like(base)*.1*mask
        targets=[observed.clone(),observed+(1-mask)*.9]
        cfg=config();cfg['iterations']=4
        with tempfile.TemporaryDirectory(dir=REPO.parent) as temp,contextlib.redirect_stdout(io.StringIO()):
            out=ArtifactStore(temp,[]);states=[];rows=[]
            for i,target in enumerate(targets):
                rows.append(audit.observed_cell(out,'id',str(i),dict(method='calibrated'),base.sigmoid(),target,
                    observed,torch.ones_like(mask),mask,torch.sigmoid,base,torch.ones_like(base),cfg,0,False))
                states.append(out.load(f'cells/{i}/result.pt',__import__('utils.editing_data',fromlist=['digest']).digest(['id',str(i)]),True)['delta'])
            torch.testing.assert_close(states[0],states[1],rtol=0,atol=0)
            self.assertNotEqual(rows[0]['heldout_edit_l1'],rows[1]['heldout_edit_l1'])
        # RGB completion has the same information restriction.
        a=rgb_completion(observed,mask,base,4)
        b=rgb_completion(observed+(1-mask)*999,mask,base,4)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_real_tiny_wan_image_decoder_input_gradient_and_padding(self):
        import importlib
        from utils.vace_edit_runtime import native_modules
        from audit_edit_transport import native_image_codec
        with cpu_native_import():
            prefix,_=native_modules();cls=importlib.import_module(prefix+'.modules.vae').WanVAE_
            model=cls(dim=8,z_dim=4,dim_mult=[1,1,1,1],num_res_blocks=1,
                temperal_downsample=[False,True,True]).eval().requires_grad_(False)
            wrapper=SimpleNamespace(model=model,scale=[torch.zeros(4),torch.ones(4)])
            image=torch.rand(1,3,18,30)
            base,decode,scale=native_image_codec(wrapper,image)
            parameter=base.detach().requires_grad_();rgb=decode(parameter)
            self.assertEqual(tuple(rgb.shape),tuple(image.shape))
            rgb.square().mean().backward()
            self.assertIsNotNone(parameter.grad);self.assertGreater(float(parameter.grad.abs().sum()),0.)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_whole_probe_cache_resume_coverage_and_both_output_repairs(self):
        import audit_edit_transport as audit
        with tempfile.TemporaryDirectory(prefix='transport integration ',dir=REPO.parent) as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);cfg,data,args=fixture(root)
            cfg.update(strengths=[.5,1.],observed_fractions=[1.,.25],noise_stds=[0.,.005])
            out=ArtifactStore(root/'local',[str(root/'primary'),str(root/'mirror')])
            vae=SimpleNamespace(model=FakeNative(),scale=[torch.tensor(0.),torch.tensor(1.)])
            with patch.object(audit,'load_teacher',side_effect=lambda *_:FakeRAE()),\
                 patch.object(audit,'load_encoder',side_effect=lambda *_:FakeEncoder()),\
                 patch('utils.vace_edit_runtime.load_vae',return_value=vae):
                audit.prepare(args,cfg,data,out,torch.device('cpu'),0,1)
                audit.run_codec(args,cfg,out,torch.device('cpu'),0,1)
                args.phase='vae';audit.run_codec(args,cfg,out,torch.device('cpu'),0,1)
                audit.summarize(out)
                self.assertEqual(out.json('summary.json')['status'],'completed')
                self.assertEqual(out.json('summary.json')['actual'],dict(rae=20,rgb=24,vae=8))
                self.assertNotIn('reserved',json.dumps(out.json('source_bank.json')))
                receipt=next(r['receipt'] for r in out.json('phase_summary/rae.json')['cases']
                    if r['scenario']=='h1_dense_codec_effect' and r['strength']==max(cfg['strengths']))
                media=next(n for n in out.json(receipt)['files'] if n.endswith('.png'))
                (root/'primary'/media).unlink();(out.local/media).unlink()
                audit.summarize(out)
                self.assertTrue((root/'primary'/media).is_file());self.assertTrue((out.local/media).is_file())
                # A killed collective cannot hide completed cells: finalization
                # recovers per-rank ledgers, while keeping status incomplete.
                saved=out.json('phase_summary/vae.json')
                for folder in (out.local,root/'primary',root/'mirror'):(folder/'phase_summary/vae.json').unlink()
                with self.assertRaises(RuntimeError):audit.summarize(out)
                self.assertEqual(out.json('summary.json')['actual']['vae'],8)
                self.assertEqual(out.json('summary.json')['status'],'incomplete')
                out.write_json('phase_summary/vae.json',saved)
                # Missing source preview in BOTH replicas is reproducible from
                # cached source states; no encoder/target access is needed.
                for folder in (out.local,root/'primary',root/'mirror'):(folder/'source_previews/0.mp4').unlink()
                with patch.object(audit,'encode_window',side_effect=AssertionError('source states already cached')):
                    audit.prepare(args,cfg,data,out,torch.device('cpu'),0,1)
                self.assertTrue((root/'primary'/'source_previews/0.mp4').is_file())
                # Completed cells recover payloads and never call the decoder again.
                with patch.object(audit,'solve_effect',side_effect=AssertionError('must resume sealed cell')):
                    args.phase='rae';audit.run_codec(args,cfg,out,torch.device('cpu'),0,1)


if __name__=='__main__':unittest.main()
