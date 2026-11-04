"""Coordinate oracles, native-model gradient paths and legal control protocols."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from utils.editing_alignment import (ALIGNMENT,align_readout,region_mean,balanced_flow_loss,
    change_support,support_on_teacher,hidden_gradient_diagnostics,region_pixel_report)
from utils.editing_controls import load_controls,source_box_mask
from utils.vace_edit_runtime import EditingModel,rae_supervision
from utils.scene_run import ArtifactStore
from tests.test_vace_edit_pilot import cpu_native_import,tiny_native,config

REPO=Path(__file__).resolve().parents[1]


def aligned_config():
    cfg=config();cfg.update(frames=5,rae_frames=3,rae_alignment=ALIGNMENT,gradient_probe_every=1,
        rae_max_sigma=1.,accum=1,steps=3)
    return cfg


class Decoder(nn.Module):
    def __init__(self):super().__init__();self.project=nn.Linear(256,3)
    def decode(self,z):return self.project(z)


def training_fixture(root):
    store=ArtifactStore(root/'data',[])
    bank=dict(identity='native-fixture',records=[]);teacher=dict(identity='teacher-fixture',records=[])
    torch.manual_seed(7)
    for i in range(4):
        ident=str(i);file=f'native/{i}.pt'
        source=torch.randint(0,200,(5,3,32,32),dtype=torch.uint8);target=source.clone()
        target[:,:,8:16,8:16]=255
        store.save(file,dict(clean=torch.randn(16,2,4,4),context=torch.randn(96,2,4,4),
            text=torch.randn(5,12),empty=torch.randn(1,12),source=source,target=target,
            timing=dict(spatial=dict(canvas=[32,32],valid_rect=[0,0,32,32]))),bank['identity'])
        bank['records'].append(dict(id=ident,file=file,split='train'))
        file=f'rae/{i}.pt';store.save(file,dict(z=torch.randn(3,18,18,256),valid_rect=[0,0,518,518]),teacher['identity'])
        teacher['records'].append(dict(id=ident,file=file))
    return store,bank,teacher


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_pixel_coordinate_ramps_portrait_landscape_padding_and_causal_time(self):
        # Native values represent normalized source-image coordinates and time
        # measured in RGB-frame units. Their output should have the SAME meaning.
        for canvas,rect,tr in [([336,592],[0,0,336,592],[112,0,294,518]),
                              ([592,336],[0,0,592,336],[0,112,518,294]),
                              ([352,608],[7,5,336,592],[112,0,294,518]),
                              ([576,576],[0,0,576,576],[0,0,518,518])]:
            ch,cw=canvas;y,x,h,w=rect;gh,gw=ch//16,cw//16
            yy=((torch.arange(gh)+.5)*16-y)/h;xx=((torch.arange(gw)+.5)*16-x)/w
            grid=torch.stack(torch.meshgrid(torch.arange(5).float()*4,yy,xx,indexing='ij'),-1)[None]
            actual=align_readout(grid,dict(canvas=canvas,valid_rect=rect),tr,(9,18,18),17)[0]
            ty,tx,th,tw=tr
            q=(torch.arange(18)+.5)*518/18
            ey=((q-ty)/th).clamp(float(yy[0]),float(yy[-1]))
            ex=((q-tx)/tw).clamp(float(xx[0]),float(xx[-1]))
            expected=torch.stack(torch.meshgrid(torch.arange(9).float()*2,ey,ex,indexing='ij'),-1)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        with self.assertRaises(ValueError):align_readout(grid,dict(canvas=canvas,valid_rect=rect),tr,(9,18,18),16)

    def test_region_means_and_causal_flow_mask_exclude_padding(self):
        error=torch.tensor([[9.,1.,1.,1.]],requires_grad=True);valid=torch.ones_like(error)
        mask=torch.tensor([[1.,0.,0.,0.]])
        self.assertEqual(float(region_mean(error,valid,mask)),5.)
        region_mean(error,valid,mask).sum().backward()
        torch.testing.assert_close(error.grad,torch.tensor([[.5,1/6,1/6,1/6]]))
        for support in (torch.zeros_like(mask),torch.ones_like(mask)):
            self.assertEqual(float(region_mean(error,valid,support)),3.)
        self.assertEqual(float(region_mean(error,valid*0,mask)),0.)
        support=torch.zeros(5,1,4,4);support[4,0,0,0]=1
        pred=torch.zeros(1,16,2,4,4);pred[:,:,1,0,0]=3
        pred[:,:,:,-1]=1000 # native pad row must not enter either region
        loss,stats=balanced_flow_loss(pred,torch.zeros_like(pred),support,
            dict(canvas=[4,4],valid_rect=[0,0,3,4]),{})
        self.assertAlmostEqual(float(loss),4.5);self.assertEqual(stats['flow_edit'],9.)

    def test_support_and_source_only_boxes_are_distinct(self):
        source=torch.zeros(5,3,32,32,dtype=torch.uint8);target=source.clone();target[:,:,8:16,8:16]=255
        mask=change_support(source,target,dilation=1)
        self.assertGreater(float(mask.sum()),0)
        spatial=dict(canvas=[32,32],valid_rect=[0,0,32,32])
        tmask=support_on_teacher(source,target,spatial,[0,0,518,518],3,{},pixel_support=mask)
        self.assertEqual(tuple(tmask.shape),(3,518,518))
        case=dict(keyframes=[[0,0,0,.25,.25],[4,.5,.5,.75,.75]])
        box=source_box_mask(case,dict(canvas=[32,32],valid_rect=[4,4,24,24]),5)
        self.assertEqual(float(box[0,0,4:10,4:10].sum()),36.)
        self.assertEqual(float(box[0,2,10:16,10:16].sum()),36.)
        self.assertEqual(float(box[:,:,:4].sum()),0.)
        self.assertEqual(float(box[:,:,:,28:].sum()),0.)
        report=region_pixel_report(source,target,source,{})
        self.assertEqual(report['keep_region_source_l1'],0.)
        self.assertGreater(report['edit_region_target_l1'],0.) # source copy is not editing
        controls=load_controls(REPO/'configs/editing_source_regions_dev_v1.json')
        self.assertEqual(len(controls['cases']),4)

    def test_actual_native_aligned_aux_and_probe_gradients(self):
        with cpu_native_import():
            torch.manual_seed(31);cfg=aligned_config()
            model=EditingModel(tiny_native().requires_grad_(False),cfg,True).train()
            decoder=Decoder().requires_grad_(False)
            x=[torch.randn(16,2,4,4)];c=[torch.randn(96,2,4,4)];text=[torch.randn(5,12)]
            pred,z=model(x,torch.tensor([300.]),c,text)
            self.assertEqual(tuple(z.shape),(1,2,2,2,256))
            target=torch.randn(1,3,18,18,256);before=target.clone()
            addition,_=rae_supervision(z,target,decoder,torch.tensor([.3]),cfg,1,[112,0,294,518],
                spatial=dict(canvas=[32,32],valid_rect=[7,0,18,32]),support=torch.ones(3,518,518))
            flow=pred.square().mean()
            stats=hidden_gradient_diagnostics(flow,addition,model.last_readout_hidden)
            self.assertEqual(stats['aux_probe_active'],1.)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            (flow+addition).backward()
            grads={n:p.grad for n,p in model.named_parameters() if p.requires_grad}
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads.values()))
            self.assertTrue(any(float(g.abs().sum())>0 for n,g in grads.items() if n.endswith('.b')))
            self.assertTrue(all(p.grad is None for p in decoder.parameters()))
            torch.testing.assert_close(target,before,rtol=0,atol=0)

    def test_region_sampling_uses_source_only_and_distinct_receipts(self):
        import train_vace_edit as trainer
        class VAE:
            def encode(self,frames):
                return [torch.nn.functional.adaptive_avg_pool3d(x[None],(2,2,2))[0].mean(0,keepdim=True).expand(16,-1,-1,-1) for x in frames]
            def decode(self,z):return [torch.zeros(3,5,16,16)]
        with tempfile.TemporaryDirectory(prefix='region protocol ',dir=REPO.parent) as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);spec=dict(schema='source-box-controls-v1',annotation_source='manual_source_video',
                cases={'a':dict(keyframes=[[0,.25,.25,.75,.75],[4,.25,.25,.75,.75]])})
            controls=root/'controls.json';controls.write_text(json.dumps(spec))
            cfg=aligned_config();cfg.update(steps=3,dev_pairs=1,eval_cases=1,region_metrics=True)
            source=torch.randint(0,255,(5,3,16,16),dtype=torch.uint8);conditions=[]
            def sample(model,condition,*_):conditions.append(condition.clone());return torch.zeros(16,2,2,2)
            def movie(path,*_,**__):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'test-video')
            for run in (0,1):
                value=dict(context=torch.ones(96,2,2,2)*9,source=source,target=torch.ones_like(source)*run*255,
                    text=torch.zeros(1,12),negative=torch.zeros(1,12),row=dict(instruction='change colour'),
                    timing=dict(output_fps=8,spatial=dict(canvas=[16,16],valid_rect=[0,0,16,16])))
                bank=dict(identity='bank',records=[dict(id='a',file='a.pt',split='dev',prompt_kind='instruction')])
                data=SimpleNamespace(load=lambda *_:value);out=ArtifactStore(root/str(run),[])
                with patch.object(trainer,'sample_native',side_effect=sample),patch.object(trainer,'load_vae',return_value=VAE()),\
                     patch.object(trainer,'write_video',side_effect=movie):
                    trainer.evaluate(SimpleNamespace(arm='sft',controls=str(controls),model='fixture'),cfg,bank,data,out,
                        SimpleNamespace(eval=lambda:None),torch.device('cpu'),0,1,3,'ema','id')
                self.assertTrue((out.local/'samples/s000003/ema/a/receipt.json').is_file(),
                    out.json('samples/s000003/ema/a/failure.json'))
                self.assertTrue((out.local/'samples/s000003/ema_source_region/a/receipt.json').is_file())
                self.assertEqual(out.json('evaluation/s000003_ema_source_region.json')['protocol'],'text_plus_source_box_diagnostic')
            torch.testing.assert_close(conditions[0],conditions[2])
            torch.testing.assert_close(conditions[1],conditions[3]) # arbitrary target replacement cannot alter condition
            self.assertFalse(torch.equal(conditions[0],conditions[1]))

    def test_detached_aux_matches_actual_sft_optimizer_updates(self):
        import train_vace_edit as trainer
        with tempfile.TemporaryDirectory(prefix='aligned-edit ',dir=REPO.parent) as temp,cpu_native_import(),\
             patch.object(trainer,'evaluate',return_value=[]),contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);data,bank,teacher=training_fixture(root);states=[];cfg=aligned_config()
            for arm in ('sft','rae_detached','region_sft','rae_balanced'):
                torch.manual_seed(31)
                model=EditingModel(tiny_native().requires_grad_(False),cfg,arm.startswith('rae'))
                decoder=Decoder().requires_grad_(False) if arm.startswith('rae') else None
                trainer.run_training(SimpleNamespace(arm=arm,stop_after=0),cfg,bank,data,ArtifactStore(root/arm,[]),
                    arm,torch.device('cpu'),0,1,model,teacher if decoder else None,decoder)
                states.append(model.deltas())
            for name,value in states[0].items():
                torch.testing.assert_close(value,states[1][name],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(value,states[3][name]) for name,value in states[2].items()))


if __name__=='__main__':unittest.main()
