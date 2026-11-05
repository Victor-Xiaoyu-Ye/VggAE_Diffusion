"""Stage64 integration with real training, frozen decoder gradients and dual IO.

Only production encoders/rendering are replaced by small explicit fixtures.
The learned module, optimizer, checkpoint/resume, split/data preparation,
reference writeback, receipts and recovery execute their actual implementation.
"""
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

import edit_reference_video as previous
import train_edit_transport as runner
from utils.scene_run import ArtifactStore
from utils.native_audit_io import sha256
from tests.test_edit_reference_pipeline import TinyCodec, SourceOnlySample, fake_write_video


REPO=Path(__file__).resolve().parents[1]
SCRATCH=Path('D:/workspace/VggAE_DataAudit/stage64_pipeline_tests') if Path('D:/workspace').exists() else None


def tiny_canvas(frames,valid_rect,size=32):
    y,x,h,w=valid_rect
    frames=frames[:,:,y:y+h,x:x+w]
    scale=size/max(h,w);hh,ww=round(h*scale),round(w*scale)
    out=F.interpolate(frames.float(),(hh,ww),mode='bilinear',align_corners=False)
    yy,xx=(size-hh)//2,(size-ww)//2
    return F.pad(out,(xx,size-ww-xx,yy,size-hh-yy)),[yy,xx,hh,ww]


def tiny_encode(rae,encoder,frames):
    b,t,c,h,w=frames.shape
    pooled=F.adaptive_avg_pool2d(frames.flatten(0,1),(2,2))
    return pooled.permute(0,2,3,1).reshape(b,t,2,2,c)


class FrozenDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain=nn.Parameter(torch.tensor(.7),requires_grad=False)

    def decode(self,z):
        b,t,h,w,c=z.shape
        rgb=F.interpolate(z[...,:3].permute(0,1,4,2,3).flatten(0,1),(32,32),mode='bilinear',align_corners=False)
        return (rgb*self.gain).permute(0,2,3,1).reshape(b,t,32,32,3)


def uncertain_flow(source,anchor,support,cfg):
    confidence=torch.full((len(source),1,*source.shape[-2:]),.3)
    confidence[0]=1
    return dict(proposal=(source+(anchor-source[0])*support).clamp(0,1),
        confidence=confidence,masks=support[None].expand(len(source),-1,-1,-1).clone(),
        flows_backward=torch.zeros(len(source)-1,*source.shape[-2:],2),
        metadata=dict(flow_backend='fixture_uncertain_source_flow',correspondence_source='source_rgb_only'))


def fixture(root,dev_count=2):
    cfg=json.loads((REPO/'configs/edit_transport_learned_v1.json').read_text())
    cfg.update(cases=dev_count,sample_steps=3,support_dilation=1)
    cfg['learned'].update(width=12,heads=3,depth=1,support_dilation=1,dropout=0.)
    cfg['training'].update(max_steps=4,min_steps=10,lr=.01,warmup_steps=1,
        save_every=1,eval_every=2,log_every=1,monitor_fraction=.25,monitor_cases=1,
        identity_fraction=0.,patience=4,ema_decay=.9)
    bank=ArtifactStore(root/'native',[])
    refs=ArtifactStore(root/'stage63',[])
    records=[];dev=[]
    yy,xx=torch.meshgrid(torch.arange(16),torch.arange(32),indexing='ij')
    for index in range(4+dev_count):
        split='train' if index<4 else 'dev'
        key=f'{split}{index if split=="train" else index-4}'
        source=(30+index*12+yy*2+xx//2)[None,None].expand(17,3,-1,-1).byte().clone()
        source[:,0]+=torch.arange(17,dtype=torch.uint8)[:,None,None]
        target=source.clone()
        target[:,2,:,8:24]+=60+torch.arange(17,dtype=torch.uint8)[:,None,None]*2
        spatial=dict(canvas=[16,32],valid_rect=[0,0,16,32])
        row=dict(id=key,split=split,group='group-'+key,file=f'native/{key}.pt',instruction='Make the centre blue')
        payload=dict(source=source,target=target,text=torch.ones(4,12)*(index+1),negative=torch.zeros(3,12),
            timing=dict(spatial=spatial,output_fps=8),row=row)
        if split=='dev':
            payload=SourceOnlySample(payload)
            dev.append(row)
            refs.save('anchors/'+key+'/result.pt',dict(source=source.float()/255,anchor=target[0].float()/255,
                support=torch.ones(1,16,32),spatial=spatial,text=payload['text'],negative=payload['negative'],
                provenance='model_output'),'stage63-identity')
        bank.save(row['file'],payload,'native-identity')
        records.append(row)
    reserved=dict(id='reserved',split='test',group='group-reserved')
    bank.write_json('bank.json',dict(identity='native-identity',records=records,
        manifest=dict(records=records+[reserved])))
    refs.write_json('manifest.json',dict(identity='stage63-identity',bank='native-identity',cases=dev))
    for name in ('rae.pt','encoder.pt'):(root/name).write_bytes(name.encode())
    args=SimpleNamespace(data=str(bank.local),data_root=[],references=str(refs.local),reference_root=[],
        rae=str(root/'rae.pt'),encoder=str(root/'encoder.pt'),model='tiny-native',stop_after=0)
    return cfg,args,ArtifactStore(root/'output',[str(root/'primary'),str(root/'mirror')])


class LearnedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if SCRATCH:SCRATCH.mkdir(parents=True,exist_ok=True)

    def temporary(self):return tempfile.TemporaryDirectory(dir=SCRATCH)

    def mocks(self,stack,calls,fail_call=None):
        teachers=[]
        def teacher(*_):
            model=FrozenDecoder().eval();teachers.append(model);return model
        stack.enter_context(patch.object(runner,'load_teacher',side_effect=teacher))
        stack.enter_context(patch.object(runner,'load_encoder',return_value=nn.Identity()))
        stack.enter_context(patch.object(runner,'code_signature',return_value={'fixture':'fixed'}))
        stack.enter_context(patch.object(runner,'rae_canvas',side_effect=tiny_canvas))
        stack.enter_context(patch('utils.edit_transport_data.rae_canvas',side_effect=tiny_canvas))
        stack.enter_context(patch('audit_edit_transport.encode_window',side_effect=tiny_encode))
        stack.enter_context(patch('utils.editing_video_transport.propagate_reference',side_effect=uncertain_flow))
        stack.enter_context(patch.object(previous,'write_video',side_effect=fake_write_video))
        stack.enter_context(patch.object(runner,'write_video',side_effect=fake_write_video))
        stack.enter_context(patch.object(runner.StageBudget,'install_signals',return_value=None))
        stack.enter_context(patch('utils.vace_edit_runtime.load_backbone',return_value=nn.Identity()))
        stack.enter_context(patch('utils.vace_edit_runtime.load_vae',side_effect=lambda *_:TinyCodec()))
        def sample(model,condition,text,negative,cfg,seed,velocity_callback=None):
            index=len(calls)
            calls.append(dict(condition=condition.clone(),text=text.clone(),seed=seed,
                callback=velocity_callback is not None))
            if fail_call==index:raise RuntimeError('fixture native case failure')
            latent=torch.zeros(16,*condition.shape[1:])
            if velocity_callback is not None:
                velocity_callback.set_schedule([1.,.3,0.])
                velocity=velocity_callback(step_index=1,sigma=.3,latent=latent,
                    velocity=torch.zeros_like(latent))
                latent=latent-.3*velocity
            return latent
        stack.enter_context(patch('utils.vace_edit_runtime.sample_native',side_effect=sample))
        return teachers

    def execute(self,args,cfg,out,render=True):
        device=torch.device('cpu')
        runner.prepare(args,cfg,out,device,0,1)
        runner.train(args,cfg,out,device,0,1)
        runner.predict(args,cfg,out,device,0,1)
        if render:
            runner.render(args,cfg,out,device,0,1)
            runner.summarize(out,cfg)

    def test_all_phases_train_parameters_preserve_anchor_and_match_three_arms(self):
        with self.temporary() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);cfg,args,out=fixture(root);calls=[]
            teachers=self.mocks(stack,calls)
            self.execute(args,cfg,out)
            prepared=out.json('prepared.json',True);identity=prepared['identity']
            self.assertFalse(any(r['status']!='completed' for r in prepared['records']))
            values=[]
            for row in prepared['records']:
                inputs=out.load(row['inputs'],identity,True)
                self.assertNotIn('target',inputs)
                if row['partition']=='train':values.append(inputs['source_z'].double().reshape(-1,3))
                if row['partition']=='monitor':self.assertIn('heldout_train_groups',inputs['protocol'])
                if row['partition']=='dev':self.assertNotIn('supervision',row)
            expected=torch.cat(values).std(0,unbiased=False).clamp_min(.05).float()
            torch.testing.assert_close(torch.tensor(prepared['source_scale']),expected,rtol=1e-6,atol=1e-7)
            state=out.load('checkpoint_latest.pt',identity,True)
            self.assertEqual(state['step'],4)
            self.assertGreater(float(state['model']['output.weight'].abs().sum()),0.)
            self.assertTrue(state['optimizer']['state'])
            for teacher in teachers:
                self.assertFalse(teacher.gain.requires_grad)
                self.assertIsNone(teacher.gain.grad)
                self.assertAlmostEqual(float(teacher.gain),.7,places=6)
            self.assertEqual(len(calls),6)
            for key,group in [('dev0',calls[:3]),('dev1',calls[3:])]:
                self.assertEqual([g['callback'] for g in group],[False,True,True])
                self.assertEqual(len({g['seed'] for g in group}),1)
                for g in group[1:]:
                    torch.testing.assert_close(g['condition'],group[0]['condition'],rtol=0,atol=0)
                prediction=out.load('predictions/'+key+'/result.pt',identity,True)
                torch.testing.assert_close(prediction['rgb'][0],prediction['anchor'],rtol=0,atol=0)
                torch.testing.assert_close(prediction['learned'][0],prediction['anchor'],rtol=0,atol=0)
                for variant in ('rgb_writeback','learned_rae_writeback'):
                    payload=out.load('render/'+variant+'/'+key+'/result.pt',identity,True)
                    self.assertEqual(len(payload['events']),1)
                    self.assertFalse(payload['events'][0]['zero_effect'])
                    self.assertGreater(payload['events'][0]['clean_delta_rms'],0)
            self.assertEqual(out.json('summary.json')['status'],'completed')
            self.assertEqual(out.json('media_sync.json')['failed'],[])

    def test_single_cpu_checkpoint_resume_is_exact(self):
        with self.temporary() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);cfg,args,out=fixture(root);self.mocks(stack,[])
            runner.prepare(args,cfg,out,torch.device('cpu'),0,1)
            runner.train(args,cfg,out,torch.device('cpu'),0,1)
            identity=out.json('prepared.json')['identity']
            full=out.load('checkpoint_latest.pt',identity,True)
            resumed=ArtifactStore(root/'resumed',[str(root/'resume_primary'),str(root/'resume_mirror')])
            for name in ('manifest.json','prepared.json'):resumed.write_json(name,out.json(name,True))
            for row in out.json('prepared.json')['records']:
                for field in ('inputs','supervision'):
                    if field in row:resumed.save(row[field],out.load(row[field],identity,True),identity)
            args.stop_after=2
            runner.train(args,cfg,resumed,torch.device('cpu'),0,1)
            self.assertEqual(resumed.load('checkpoint_latest.pt',identity,True)['step'],2)
            args.stop_after=0
            runner.train(args,cfg,resumed,torch.device('cpu'),0,1)
            final=resumed.load('checkpoint_latest.pt',identity,True)
            for name in full['model']:
                torch.testing.assert_close(full['model'][name],final['model'][name],rtol=0,atol=0)
                torch.testing.assert_close(full['ema'][name],final['ema'][name],rtol=0,atol=0)
            self.assertEqual(final['step'],4)
            self.assertEqual(final['best_score'],full['best_score'])

    def test_sampling_failure_keeps_later_arms_and_cases(self):
        with self.temporary() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            cfg,args,out=fixture(Path(temp));calls=[];self.mocks(stack,calls,fail_call=0)
            self.execute(args,cfg,out,render=False)
            runner.render(args,cfg,out,torch.device('cpu'),0,1)
            rows=out.json('phases/render.json')['cases']
            self.assertEqual(len(rows),6)
            self.assertEqual(sum(r['status']=='completed' for r in rows),5)
            self.assertTrue(all(r['status']=='completed' for r in rows if r['id']=='dev1'))
            with self.assertRaisesRegex(RuntimeError,'Incomplete Stage64'):runner.summarize(out,cfg)
            self.assertEqual(out.json('summary.json')['status'],'incomplete')
            self.assertTrue((out.local/'comparisons/dev1/native_rgb_learned.mp4').exists())

    def test_media_recovery_repairs_both_roots_without_models(self):
        with self.temporary() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            cfg,args,out=fixture(Path(temp));calls=[];self.mocks(stack,calls)
            self.execute(args,cfg,out)
            names=['predictions/dev0/comparison.mp4','render/native_reference/dev0/comparison.mp4',
                'render/rgb_writeback/dev0/comparison.mp4','render/learned_rae_writeback/dev0/comparison.mp4',
                'comparisons/dev0/native_rgb_learned.mp4']
            for name in names:
                (out.local/name).unlink()
                (Path(out.roots[0])/name).unlink()
            with patch.object(runner,'load_teacher',side_effect=AssertionError('Unexpected teacher load')), \
                 patch('utils.vace_edit_runtime.load_backbone',side_effect=AssertionError('Unexpected backbone load')), \
                 patch('utils.vace_edit_runtime.sample_native',side_effect=AssertionError('Unexpected sampling')):
                runner.predict(args,cfg,out,torch.device('cpu'),0,1)
                runner.render(args,cfg,out,torch.device('cpu'),0,1)
                runner.summarize(out,cfg)
            self.assertEqual(len(calls),6)
            for name in names:
                for mirror in out.roots:self.assertEqual(sha256(out.local/name),sha256(Path(mirror)/name))
            self.assertEqual(out.json('summary.json')['status'],'completed')

    def test_changed_selection_refreshes_only_learned_render_and_comparison(self):
        with self.temporary() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            cfg,args,out=fixture(Path(temp));calls=[];self.mocks(stack,calls)
            self.execute(args,cfg,out)
            identity=out.json('prepared.json')['identity']
            preserved={name:sha256(out.local/name) for name in
                ['render/native_reference/dev0/comparison.mp4','render/rgb_writeback/dev0/comparison.mp4']}
            old=out.load('render/learned_rae_writeback/dev0/result.pt',identity,True)['generation_identity']
            checkpoint=out.load('selected.pt',identity,True)
            checkpoint['model']['output.bias']=checkpoint['model']['output.bias']+.3
            checkpoint['step']+=1
            out.save('selected.pt',checkpoint,identity)
            runner.predict(args,cfg,out,torch.device('cpu'),0,1)
            runner.render(args,cfg,out,torch.device('cpu'),0,1)
            runner.summarize(out,cfg)
            # Two cases refresh only the learned arm. Native and RGB remain
            # valid because their sources, anchors, masks and references match.
            self.assertEqual(len(calls),8)
            for name,value in preserved.items():self.assertEqual(sha256(out.local/name),value)
            actual=out.load('render/learned_rae_writeback/dev0/result.pt',identity,True)
            self.assertNotEqual(actual['generation_identity'],old)
            prediction=out.load('predictions/dev0/result.pt',identity,True)
            self.assertEqual(prediction['selected_hash'],out.json('selected.pt.json')['sha256'])
            comparison=out.json('comparisons/dev0/receipt.json')
            self.assertEqual(comparison['metadata']['generations'][2],actual['generation_identity'])

    def test_failed_learned_prediction_retries_and_preserves_native_rgb(self):
        with self.temporary() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            cfg,args,out=fixture(Path(temp));calls=[];self.mocks(stack,calls)
            runner.prepare(args,cfg,out,torch.device('cpu'),0,1)
            runner.train(args,cfg,out,torch.device('cpu'),0,1)
            with patch.object(runner,'load_teacher',side_effect=RuntimeError('fixture decoder unavailable')):
                runner.predict(args,cfg,out,torch.device('cpu'),0,1)
            self.assertTrue(all(not r['learned_available'] for r in out.json('phases/predict.json')['cases']))
            runner.render(args,cfg,out,torch.device('cpu'),0,1)
            self.assertEqual(len(calls),4)
            runner.predict(args,cfg,out,torch.device('cpu'),0,1)
            self.assertTrue(all(r['learned_available'] for r in out.json('phases/predict.json')['cases']))
            runner.render(args,cfg,out,torch.device('cpu'),0,1)
            runner.summarize(out,cfg)
            self.assertEqual(len(calls),6)
            self.assertEqual(out.json('summary.json')['status'],'completed')


if __name__=='__main__':unittest.main()
