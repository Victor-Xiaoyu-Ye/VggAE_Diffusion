"""Stage63 phase integration: real dual artifacts with observable tiny codecs.

No production weights/NPU are loaded. Source-only inference, matched render
inputs, native writeback plumbing and resumed media repair are exercised.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import edit_reference_video as runner
from utils.scene_run import ArtifactStore
from utils.native_audit_io import sha256

REPO=Path(__file__).resolve().parents[1]


class SourceOnlySample(dict):
    """Torch-saveable record that fails immediately if inference reads target."""
    def __getitem__(self,key):
        if key=='target':raise AssertionError('Inference accessed dataset target')
        return super().__getitem__(key)

    def get(self,key,default=None):
        if key=='target':raise AssertionError('Inference accessed dataset target')
        return super().get(key,default)


class TinyCodec:
    def encode(self,values):
        result=[]
        for value in values:
            selected=value[:,::4].mean(0,keepdim=True)
            pooled=F.avg_pool2d(selected.permute(1,0,2,3),8).permute(1,0,2,3)
            result.append(pooled.expand(16,-1,-1,-1)+.25)
        return result

    def decode(self,values):
        return [F.interpolate((v[:3]-.25)[None],
            size=(4*(v.shape[1]-1)+1,v.shape[2]*8,v.shape[3]*8),
            mode='nearest')[0] for v in values]


class FakeRAE(nn.Module):
    def decode(self,z):
        value=z[...,:3].permute(0,1,4,2,3).flatten(0,1)
        image=F.interpolate(value,(518,518),mode='bilinear',align_corners=False)
        return image.permute(0,2,3,1).reshape(1,z.shape[1],518,518,3)


def fake_propagate(source,anchor,support,cfg):
    effect=(anchor-source[0])*support
    return dict(proposal=(source+effect[None]).clamp(0,1),
        confidence=torch.ones(len(source),1,*source.shape[-2:]),
        masks=support[None].expand(len(source),-1,-1,-1).clone(),
        flows_backward=torch.zeros(len(source)-1,*source.shape[-2:],2),
        metadata=dict(backend='fixture_source_correspondence',used_target=False))


def fake_calibrate(decode,z,source,proposal,confidence,masks,cfg,checkpoint=None,
                   resume=None,progress=None,flows_backward=None):
    if flows_backward is None:raise AssertionError('Joint temporal edges were omitted')
    if checkpoint:checkpoint(dict(fixture=True,sweep=1))
    if progress:progress(dict(sweep=1,DI_throughput=1.,throughput_unit='fixture frames/s'))
    return dict(effect=(proposal-source)*.8,delta=torch.ones_like(z)*.01,
                history=[dict(sweep=1,objective=.1)])


def fake_write_video(path,source,reference,output,fps,labels=None):
    # Preserve meaningful aligned panel shape/data checks without a codec/ffmpeg
    # dependency. ArtifactStore still computes and verifies real content hashes.
    assert source.shape==reference.shape==output.shape
    assert source.dtype==reference.dtype==output.dtype==torch.uint8
    assert len(source)==17 and fps==8 and len(labels)==3
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    header=json.dumps(dict(shape=list(source.shape),fps=fps,labels=labels)).encode()
    path.write_bytes(header+b'\n'+source.numpy().tobytes()+reference.numpy().tobytes()+output.numpy().tobytes())


def fixture(root,count=1,references=False):
    cfg=json.loads((REPO/'configs/edit_reference_video_v1.json').read_text())
    cfg.update(cases=count,sample_steps=3,anchor_steps=3,support_dilation=1)
    data=ArtifactStore(root/'bank',[])
    bank=dict(identity='fixed-source-bank',records=[])
    for i in range(count):
        key=f'case{i}'
        source=(torch.arange(17)[:,None,None,None]+40+i*6).expand(17,3,16,32).byte().clone()
        item=SourceOnlySample(source=source,target='POISON: never a condition',
            text=torch.full((4,12),float(i+1)),negative=torch.zeros(3,12),
            timing=dict(spatial=dict(canvas=[16,32],valid_rect=[0,0,16,32])))
        name=f'native/{key}.pt';data.save(name,item,bank['identity'])
        bank['records'].append(dict(id=key,file=name,split='dev',instruction='Make the surface blue'))
    bank['records'].append(dict(id='reserved',file='absent.pt',split='test'))
    data.write_json('bank.json',bank)
    for name in ('rae.pt','stream.pt'):(root/name).write_bytes(name.encode())
    args=SimpleNamespace(model='existing-native',adapter='',adapter_config='',controls='',references='',
                         rae=str(root/'rae.pt'),encoder=str(root/'stream.pt'))
    if references:
        from PIL import Image
        cases={}
        for i in range(count):
            name=f'user-reference-{i}.png'
            Image.fromarray(np.full((16,32,3),120+i,dtype=np.uint8)).save(root/name)
            cases[f'case{i}']=dict(image=name,provenance='user_edit')
        ref=root/'references.json';ref.write_text(json.dumps(dict(schema='edit-references-v1',cases=cases)))
        args.references=str(ref)
    out=ArtifactStore(root/'local',[str(root/'primary'),str(root/'mirror')])
    return cfg,data,out,args


class ReferencePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def mocks(self,stack,calls):
        stack.enter_context(patch.object(runner,'write_video',side_effect=fake_write_video))
        stack.enter_context(patch.object(runner,'code_signature',return_value={'fixture':'immutable'}))
        stack.enter_context(patch('utils.vace_edit_runtime.validate_assets',return_value={'fixture':'native'}))
        stack.enter_context(patch.object(runner,'load_backbone',return_value=nn.Identity()))
        stack.enter_context(patch.object(runner,'load_vae',side_effect=lambda *_:TinyCodec()))
        stack.enter_context(patch.object(runner,'load_teacher',side_effect=lambda *_:FakeRAE()))
        stack.enter_context(patch.object(runner,'load_encoder',return_value=nn.Identity()))
        stack.enter_context(patch('audit_edit_transport.encode_window',side_effect=lambda rae,enc,frames:
            torch.zeros(1,frames.shape[1],18,18,256)))
        stack.enter_context(patch('utils.editing_video_transport.propagate_reference',side_effect=fake_propagate))
        stack.enter_context(patch('utils.editing_video_transport.calibrate_video',side_effect=fake_calibrate))

        def sample(model,condition,text,negative,cfg,seed,velocity_callback=None):
            calls.append(dict(context=condition.clone(),seed=seed,text=text.clone(),
                              callback=velocity_callback is not None))
            latent=torch.zeros(16,*condition.shape[1:])
            if velocity_callback is not None:
                velocity_callback.set_schedule([1.,.3,0.])
                corrected=velocity_callback(step_index=1,sigma=.3,latent=latent,
                                            velocity=torch.zeros_like(latent))
                latent=latent-.3*corrected
            return latent
        stack.enter_context(patch.object(runner,'sample_native',side_effect=sample))

    def execute(self,a,cfg,data,out):
        device=torch.device('cpu')
        runner.anchor_phase(a,cfg,data,out,device,0,1)
        runner.transport_phase(a,cfg,out,device,0,1)
        runner.render_phase(a,cfg,out,device,0,1)
        runner.summarize(out,cfg)

    def test_auto_anchor_recovers_fixed_stage61_raw750_and_uses_raw_weights(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            previous=ArtifactStore(root/'primary/vace_edit_aligned_v2',[])
            name='region_sft/checkpoints/step_000750.pt'
            previous.save(name,dict(step=750,arm='region_sft',config={},
                model={'weight':torch.tensor(.75)},ema={'weight':torch.tensor(.9)}),'stage61-fixed')
            out=ArtifactStore(root/'local',[(root/'primary/edit_reference_video_v1').as_posix()])
            a=SimpleNamespace(adapter='auto',data_root=[],model='fixture',adapter_config='')
            before=sha256(previous.local/name)
            with patch.dict('os.environ',{'LOCAL_RANK':'0','LOCAL_WORLD_SIZE':'8'}):
                result=runner.choose_anchor_adapter(a,out,0,1)
            self.assertEqual(result['mode'],'stage61_region_sft_raw750')
            self.assertEqual(sha256(a.adapter),before)
            self.assertEqual(sha256(previous.local/name),before)
            self.assertTrue(Path(a.adapter).is_relative_to(out.local/'inputs'))

            class RecordingAdapter(nn.Module):
                def restore(self,value):self.restored=value
            adapter=RecordingAdapter()
            with patch.object(runner,'load_backbone',return_value=nn.Identity()), \
                 patch('utils.vace_edit_runtime.EditingModel',return_value=adapter):
                actual=runner.backbone(a,torch.device('cpu'),adapter=True)
            self.assertIs(actual,adapter)
            self.assertEqual(float(adapter.restored['weight']),.75)

    def test_auto_anchor_missing_or_any_node_unavailable_falls_back_to_native(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp)
            out=ArtifactStore(root/'local',[(root/'primary/edit_reference_video_v1').as_posix()])
            a=SimpleNamespace(adapter='auto',data_root=[])
            with patch.dict('os.environ',{'LOCAL_RANK':'0','LOCAL_WORLD_SIZE':'8'}):
                result=runner.choose_anchor_adapter(a,out,0,1)
            self.assertEqual(a.adapter,'')
            self.assertEqual(result['mode'],'native_fallback')
            self.assertTrue(result['fallback_errors'])

            previous=ArtifactStore(root/'primary/vace_edit_aligned_v2',[])
            previous.save('region_sft/checkpoints/step_000750.pt',
                dict(step=750,arm='region_sft',config={},model={}), 'stage61-fixed')
            a.adapter='auto'
            def gathered(entry,world):
                self.assertEqual(world,24)
                rows=[None]*24;rows[0]=entry;rows[8]={'error':'remote node unavailable'};rows[16]=entry
                return rows
            with patch.dict('os.environ',{'LOCAL_RANK':'0','LOCAL_WORLD_SIZE':'8'}), \
                 patch.object(runner,'gather',side_effect=gathered):
                result=runner.choose_anchor_adapter(a,out,0,24)
            self.assertEqual(a.adapter,'')
            self.assertEqual(result['mode'],'native_fallback')
            self.assertIn('remote node unavailable',result['fallback_errors'])

    def test_all_phases_target_is_forbidden_and_render_conditions_match(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);cfg,data,out,a=fixture(root)
            calls=[];self.mocks(stack,calls)
            self.execute(a,cfg,data,out)
            self.assertEqual(len(calls),3)
            baseline,full=calls[-2:]
            torch.testing.assert_close(baseline['context'],full['context'],rtol=0,atol=0)
            torch.testing.assert_close(baseline['text'],full['text'],rtol=0,atol=0)
            self.assertEqual(baseline['seed'],full['seed'])
            self.assertFalse(baseline['callback']);self.assertTrue(full['callback'])
            report=out.json('summary.json',True)
            self.assertEqual(report['status'],'completed')
            self.assertEqual(report['completed'],dict(anchors=1,transport=1,render=2))
            self.assertEqual(report['comparisons'][0]['writeback_count'],1)
            self.assertGreater(report['comparisons'][0]['output_difference'],0)
            self.assertTrue((out.local/'transport/case0/optimizer.pt').is_file())
            self.assertEqual(out.json('media_sync.json')['status'],'synced')
            self.assertTrue((out.local/'report.html').is_file())
            for name in ('anchors/case0/comparison.mp4','transport/case0/comparison.mp4',
                         'render/native_reference/case0/comparison.mp4',
                         'render/rae_writeback/case0/comparison.mp4','comparisons/case0/source_native_rae.mp4'):
                for remote in out.roots:
                    self.assertEqual(sha256(out.local/name),sha256(Path(remote)/name))
            self.assertNotIn('reserved',[r['id'] for r in out.json('manifest.json')['cases']])

    def test_completed_resume_repairs_missing_primary_without_models(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);cfg,data,out,a=fixture(root)
            calls=[];self.mocks(stack,calls);self.execute(a,cfg,data,out)
            names=['anchors/case0/comparison.mp4','transport/case0/comparison.mp4',
                   'render/native_reference/case0/comparison.mp4','render/rae_writeback/case0/comparison.mp4',
                   'comparisons/case0/source_native_rae.mp4']
            for name in names:
                (Path(out.roots[0])/name).unlink()
                (out.local/name).unlink()
            # Recovery must use intact mirror media and not sample/encode again.
            with patch.object(runner,'sample_native',side_effect=AssertionError('Unexpected resampling')), \
                 patch.object(runner,'load_backbone',side_effect=AssertionError('Unexpected model load')), \
                 patch.object(runner,'load_teacher',side_effect=AssertionError('Unexpected RAE load')):
                self.execute(a,cfg,data,out)
            self.assertEqual(len(calls),3)
            for name in names:
                self.assertEqual(sha256(out.local/name),sha256(Path(out.roots[0])/name))
            self.assertEqual(out.json('summary.json')['status'],'completed')

    def test_user_reference_anchor_needs_no_backbone_and_declares_protocol(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);cfg,data,out,a=fixture(root,references=True)
            calls=[];self.mocks(stack,calls)
            with patch.object(runner,'load_backbone',side_effect=AssertionError('User anchor should not load model')):
                runner.anchor_phase(a,cfg,data,out,torch.device('cpu'),0,1)
            self.assertEqual(calls,[])
            manifest=out.json('manifest.json');payload=out.load('anchors/case0/result.pt',manifest['identity'],True)
            self.assertEqual(payload['provenance'],'user_edit');self.assertTrue(payload['external'])
            self.assertGreater(payload['diagnostics']['anchor_change'],.1)
            runner.transport_phase(a,cfg,out,torch.device('cpu'),0,1)
            runner.render_phase(a,cfg,out,torch.device('cpu'),0,1)
            runner.summarize(out,cfg)
            self.assertEqual(len(calls),2)
            self.assertEqual(out.json('summary.json')['comparisons'][0]['provenance'],'user_edit')

    def test_fallback_baseline_is_resampled_when_shared_transport_recovers(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);cfg,data,out,a=fixture(root,references=True)
            calls=[];self.mocks(stack,calls)
            runner.anchor_phase(a,cfg,data,out,torch.device('cpu'),0,1)
            runner.render_phase(a,cfg,out,torch.device('cpu'),0,1)
            identity=out.json('manifest.json')['identity']
            before=out.load('render/native_reference/case0/result.pt',identity,True)
            self.assertEqual(before['mask_origin'],'native_resolution_flow_fallback')
            self.assertEqual(len(calls),1)
            runner.transport_phase(a,cfg,out,torch.device('cpu'),0,1)
            runner.render_phase(a,cfg,out,torch.device('cpu'),0,1)
            self.assertEqual(len(calls),3,'Old fallback baseline was incorrectly reused')
            after=out.load('render/native_reference/case0/result.pt',identity,True)
            full=out.load('render/rae_writeback/case0/result.pt',identity,True)
            self.assertEqual(after['mask_origin'],'shared_source_flow')
            self.assertEqual(after['input_identity'],full['input_identity'])
            torch.testing.assert_close(calls[-2]['context'],calls[-1]['context'],rtol=0,atol=0)
            runner.summarize(out,cfg)
            self.assertEqual(out.json('summary.json')['status'],'completed')

    def test_case_failure_does_not_block_other_cases_or_native_fallback(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);cfg,data,out,a=fixture(root,count=2,references=True)
            calls=[];self.mocks(stack,calls)
            runner.anchor_phase(a,cfg,data,out,torch.device('cpu'),0,1)
            iteration=[0]
            def failing_calibrate(*args,**kwargs):
                iteration[0]+=1
                if iteration[0]==1:raise RuntimeError('fixture failure on first transport case')
                return fake_calibrate(*args,**kwargs)
            with patch('utils.editing_video_transport.calibrate_video',side_effect=failing_calibrate):
                runner.transport_phase(a,cfg,out,torch.device('cpu'),0,1)
            runner.render_phase(a,cfg,out,torch.device('cpu'),0,1)
            with self.assertRaisesRegex(RuntimeError,'Incomplete'):
                runner.summarize(out,cfg)
            report=out.json('summary.json')
            self.assertEqual(report['status'],'incomplete')
            self.assertEqual(report['completed'],dict(anchors=2,transport=1,render=3))
            self.assertEqual([r['id'] for r in report['comparisons']],['case1'])
            render=out.json('phases/render.json')['cases']
            successful={(r['id'],r['variant']) for r in render if r['status']=='completed'}
            self.assertIn(('case0','native_reference'),successful)
            self.assertIn(('case1','rae_writeback'),successful)


if __name__=='__main__':unittest.main()
