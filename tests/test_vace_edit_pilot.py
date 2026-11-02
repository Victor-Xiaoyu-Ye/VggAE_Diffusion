"""CPU contracts: real tiny VACE equivalence/gradients, pairing and exact resume.

CPU import shims only map vendored NPU autocast decorators; no accelerator claim.
"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from utils.editing_data import make_manifest,unpack_archive,paired_indices,spatial_plan,apply_spatial,rae_canvas
from utils.vace_edit_runtime import (native_modules,native_forward,EditingModel,
    flow_pair,pack_context,rae_supervision,load_backbone,runtime_preflight)
from utils.editing_train import sample_row,update_ema
from utils.scene_run import ArtifactStore


@contextlib.contextmanager
def cpu_native_import():
    original=torch.amp.autocast
    def autocast(device_type='cpu',*args,**kwargs):
        return original('cpu' if device_type=='npu' else device_type,*args,**kwargs)
    npu=types.ModuleType('torch.npu'); amp=types.ModuleType('torch.npu.amp')
    amp.autocast=lambda **kw:original('cpu',**kw)
    npu.amp=amp
    npu.is_available=lambda:False
    previous={k:sys.modules.get(k) for k in ('torch.npu','torch.npu.amp')}
    sys.modules.update({'torch.npu':npu,'torch.npu.amp':amp})
    try:
        with patch.object(torch,'npu',npu,create=True),patch.object(torch.amp,'autocast',autocast):
            yield
    finally:
        for key,value in previous.items():
            if value is None: sys.modules.pop(key,None)
            else: sys.modules[key]=value


def tiny_native():
    import importlib
    prefix,_=native_modules()
    cls=importlib.import_module(prefix+'.modules.vace_model').VaceWanModel
    model=cls(dim=32,ffn_dim=48,freq_dim=16,text_dim=12,num_heads=2,num_layers=2,
        in_dim=16,out_dim=16,text_len=8,vace_layers=[0],vace_in_dim=96)
    # Real checkpoint heads are nonzero; test gradients through a representative head.
    nn.init.normal_(model.head.head.weight,std=.1)
    return model


def config():
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/vace_edit_pilot_v1.json').read_text())
    cfg.update(steps=3,accum=2,log_every=1,save_every=2,eval_every=100,rae_frames=2,rae_block=0,
        lora_rank=2,lora_alpha=2,rae_warmup=0,lr=.001)
    return cfg


class VaceEditingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def setUp(self):
        self.cpu=cpu_native_import()
        self.cpu.__enter__()
        self.addCleanup(self.cpu.__exit__,None,None,None)

    def test_real_tiny_native_forward_and_checkpoint_backward(self):
        torch.manual_seed(31)
        m=tiny_native().eval()
        x=[torch.randn(16,2,4,4)]; t=torch.tensor([500.])
        c=[torch.randn(96,2,4,4)]; text=[torch.randn(5,12)]
        expected=m(x,t,c,text,seq_len=8)[0]
        actual=native_forward(m,x,t,c,text)[0][0]
        torch.testing.assert_close(actual,expected)
        m.requires_grad_(False)
        edited=EditingModel(m,config(),with_rae=True)
        with torch.no_grad():
            a,z=edited(x,t,c,text)
        torch.testing.assert_close(a[0],expected) # LoRA initial delta exactly zero.
        edited.train()
        a,z=edited(x,t,c,text)
        loss=a.square().mean()+z.square().mean()
        loss.backward()
        grads={n:p.grad for n,p in edited.named_parameters() if p.requires_grad}
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads.values()))
        self.assertTrue(any(float(g.abs().sum())>0 for n,g in grads.items() if n.endswith('.b')))
        self.assertTrue(all(p.grad is None for n,p in edited.named_parameters() if not p.requires_grad))

    def test_meta_shard_loader_preserves_keys_precision_and_preflight(self):
        from safetensors.torch import save_file
        import importlib
        m=tiny_native()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fields=('dim','ffn_dim','freq_dim','text_dim','num_heads','num_layers','in_dim','out_dim',
                    'text_len','vace_layers','vace_in_dim','model_type','patch_size')
            (root/'config.json').write_text(json.dumps({k:getattr(m,k) for k in fields}))
            state=m.state_dict(); keys=list(state)
            mapping={}
            for i,chunk in enumerate((keys[::2],keys[1::2])):
                name=f'part-{i}.safetensors'
                save_file({k:state[k].contiguous() for k in chunk},str(root/name))
                mapping.update({k:name for k in chunk})
            (root/'diffusion_pytorch_model.safetensors.index.json').write_text(json.dumps(dict(weight_map=mapping)))
            with contextlib.redirect_stdout(io.StringIO()): loaded=load_backbone(root,torch.device('cpu'))
            self.assertTrue(all(not p.is_meta and not p.requires_grad for p in loaded.parameters()))
            self.assertEqual(loaded.blocks[0].self_attn.q.weight.dtype,torch.bfloat16)
            self.assertEqual(loaded.time_embedding[0].weight.dtype,torch.float32)
            self.assertEqual(loaded.blocks[0].self_attn.norm_q.weight.dtype,torch.float32)
            for key,value in loaded.state_dict().items():
                torch.testing.assert_close(value,state[key].to(value.dtype),rtol=0,atol=0)
        self.assertGreater(runtime_preflight(torch.device('cpu'))['lora_gradient_l1'],0)

    def test_aspect_ratio_fov_area_and_no_upscale(self):
        for h,w in ((592,336),(1080,1920),(512,512),(100,300),(80,1600)):
            plan=spatial_plan(h,w,256*448,768)
            rh,rw=plan['resized']; ch,cw=plan['canvas']
            self.assertLessEqual(ch*cw,256*448)
            self.assertEqual(ch%16+cw%16,0)
            self.assertLessEqual(rh,h); self.assertLessEqual(rw,w)
            self.assertLess(abs(rh/rw-h/w),2/min(rw,w))
            x=torch.ones(2,3,h,w)*133
            resized=apply_spatial(x,plan)
            top,left,hh,ww=plan['valid_rect']
            self.assertTrue(bool((resized[:,:,top:top+hh,left:left+ww]==133).all()))
            square,rect=rae_canvas(resized.float(),plan['valid_rect'])
            self.assertEqual(tuple(square.shape),(2,3,518,518))
            self.assertAlmostEqual(rect[2]/rect[3],h/w,delta=2/min(rect[2:]))

    def test_mask_encoding_keeps_vae_zero_offset_and_packing(self):
        class VAE:
            def encode(self,frames):
                return [torch.nn.functional.adaptive_avg_pool3d(x[None],(2,2,2))[0].mean(0,keepdim=True).expand(16,-1,-1,-1)+7 for x in frames]
        frames=torch.ones(3,5,16,16)*.25
        white=pack_context(VAE(),frames,torch.ones(1,5,16,16))
        self.assertEqual(tuple(white.shape),(96,2,2,2))
        self.assertEqual(float(white[:16].mean()),7.)
        self.assertEqual(float(white[16:32].mean()),7.25)
        self.assertEqual(float(white[32:].mean()),1.)
        black=pack_context(VAE(),frames,torch.zeros(1,5,16,16))
        torch.testing.assert_close(black[:16],white[16:32])
        self.assertEqual(float(black[32:].sum()),0.)

    def test_flow_endpoints_and_inverse(self):
        clean=torch.randn(2,16,2,4,4); noise=torch.randn_like(clean)
        sigma=torch.tensor([0.,1.]); x,v=flow_pair(clean,noise,sigma)
        torch.testing.assert_close(x[0],clean[0]); torch.testing.assert_close(x[1],noise[1])
        torch.testing.assert_close(x-sigma[:,None,None,None,None]*v,clean)

    def test_cache_integration_source_condition_and_reserved_test(self):
        import prepare_vace_edit as preparation
        from utils.editing_data import digest
        cfg=config(); cfg.update(frames=5,height=16,width=16,max_side=16)
        class VAE:
            def encode(self,values):
                return [torch.nn.functional.adaptive_avg_pool3d(v[None],(2,2,2))[0].mean(0,keepdim=True).expand(16,-1,-1,-1)+7 for v in values]
        class Text:
            def __call__(self,texts,device): return [torch.ones(2,12) for _ in texts]
        with tempfile.TemporaryDirectory() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp); data=root/'data';data.mkdir()
            records=[dict(id=s,split=s,instruction='Make blue.',prompt_kind='edit_instruction') for s in ('train','dev','test')]
            (data/'manifest.json').write_text(json.dumps(dict(records=records,benchmark=[],identity='fixture')))
            source=torch.full((5,3,16,16),32,dtype=torch.uint8);target=torch.full_like(source,224)
            timing=dict(spatial=spatial_plan(16,16,256,16),original_fps=[8,8],output_fps=8)
            stack.enter_context(patch('utils.editing_train.distributed',return_value=(torch.device('cpu'),0,1)))
            stack.enter_context(patch('utils.vace_edit_runtime.runtime_preflight',return_value={'passed':True}))
            stack.enter_context(patch('utils.vace_edit_runtime.validate_assets',return_value={'fixture':True}))
            stack.enter_context(patch('utils.vace_edit_runtime.load_text',return_value=Text()))
            stack.enter_context(patch('utils.vace_edit_runtime.load_vae',return_value=VAE()))
            stack.enter_context(patch.object(preparation,'load_pair',return_value=(source,target,timing)))
            args=types.SimpleNamespace(output=str(root/'cache'),root=[str(root/'mirror')],data=str(data),model='fixture',benchmark='unused')
            preparation.cache(args,cfg)
            store=ArtifactStore(root/'cache',[]); bank=store.json('bank.json',True)
            self.assertEqual({r['id'] for r in bank['records']},{'train','dev'})
            item=store.load('native/train.pt',bank['identity'],True)
            self.assertAlmostEqual(float(item['context'][16:32].float().mean()),32/127.5-1+7,delta=.02)
            self.assertAlmostEqual(float(item['clean'].float().mean()),224/127.5-1+7,delta=.02)
            self.assertFalse((root/'cache/native/test.pt').exists())
            self.assertIn('[16, 16]',store.json('coverage.json')['source_sizes'])

    def test_rae_aux_gradient_reaches_lora_and_detached_control(self):
        class Decoder(nn.Module):
            def __init__(self): super().__init__(); self.map=nn.Linear(256,3)
            def decode(self,z): return self.map(z)
        decoder=Decoder().requires_grad_(False)
        torch.manual_seed(11)
        model=EditingModel(tiny_native().requires_grad_(False),config(),True)
        x=[torch.randn(16,2,4,4)]; c=[torch.randn(96,2,4,4)]; text=[torch.randn(5,12)]
        for detach in (False,True):
            model.zero_grad(set_to_none=True)
            _,z=model(x,torch.tensor([300.]),c,text,detach_readout=detach)
            target=torch.randn_like(z)
            loss,_=rae_supervision(z,target,decoder,torch.tensor([.3]),config(),0)
            loss.backward()
            amount=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters()
                       if n.startswith('backbone.') and p.grad is not None)
            self.assertEqual(amount>0,not detach)
            self.assertTrue(all(p.grad is None for p in decoder.parameters()))

    def test_pairing_group_split_archive_safety_and_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'unpacked'/'4').mkdir(parents=True)
            rows=[]
            for i in range(8):
                # Two edited variants with identical source bytes stay in one split.
                (root/'unpacked'/'4'/f'{i}_org.mp4').write_bytes(str(i//2).encode())
                (root/'unpacked'/'4'/f'{i}.mp4').write_bytes(f'edited{i}'.encode())
                rows.append(dict(source_video_path=f'local_style_transfer/4/{i}_org.mp4',
                    target_video_path=f'local_style_transfer/4/{i}.mp4',instruction='Make the shirt blue.'))
            (root/'updated_data_local_style_transfer.json').write_text(json.dumps(rows))
            manifest=make_manifest(root,dict(seed=42,train_pairs=10,dev_pairs=1,test_pairs=1))
            groups={}
            for row in manifest['records']: groups.setdefault(row['group'],set()).add(row['split'])
            self.assertTrue(all(len(s)==1 for s in groups.values()))
            self.assertEqual(manifest['counts'],dict(train=4,dev=2,test=2))
            tar=root/'bad.tar.gz'
            with tarfile.open(tar,'w:gz') as f:
                entry=tarfile.TarInfo('../outside.mp4'); entry.size=1; f.addfile(entry,io.BytesIO(b'x'))
            with self.assertRaises(ValueError): unpack_archive(tar,root/'extracted')
        self.assertEqual(paired_indices(48,32,24,16,17,16)[1],list(range(17)))
        with self.assertRaises(ValueError): paired_indices(10,10,16,16,17,16)
        with self.assertRaises(ValueError): paired_indices(48,16,24,16,17,16)

    def test_train_resume_and_dual_read_with_actual_optimizer(self):
        import train_vace_edit as trainer
        class Toy(nn.Module):
            def __init__(self): super().__init__(); self.weight=nn.Parameter(torch.tensor(.2))
            def forward(self,x,t,c,text,detach_readout=False): return torch.stack(x)*self.weight,None
            def deltas(self): return {'weight':self.weight.detach().cpu().clone()}
            def restore(self,state):
                with torch.no_grad(): self.weight.copy_(state['weight'])
        cfg=config()
        with tempfile.TemporaryDirectory() as tmp,patch.object(trainer,'evaluate',return_value=[]),contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp); data=ArtifactStore(root/'data',[])
            bank=dict(identity='test-bank',records=[])
            for i in range(4):
                row=dict(id=str(i),file=f'{i}.pt',split='train'); bank['records'].append(row)
                data.save(row['file'],dict(clean=torch.ones(16,2,4,4)*(i+1),context=torch.zeros(96,2,4,4),text=torch.zeros(2,12),empty=torch.zeros(1,12)),'test-bank')
            direct=ArtifactStore(root/'direct',[])
            resumed=ArtifactStore(root/'resumed',[str(root/'primary'),str(root/'mirror')])
            def run(out,limit):
                a=types.SimpleNamespace(arm='sft',stop_after=limit)
                trainer.run_training(a,cfg,bank,data,out,'test-run',torch.device('cpu'),0,1,Toy())
                return out.load('checkpoint_latest.pt','test-run',True)
            full=run(direct,0); run(resumed,2)
            # Simulate losing both local checkpoint and the primary payload.
            (resumed.local/'checkpoint_latest.pt').unlink()
            (root/'primary'/'checkpoint_latest.pt').unlink()
            result=run(resumed,0)
            self.assertEqual(result['step'],3)
            torch.testing.assert_close(result['model']['weight'],full['model']['weight'],rtol=0,atol=0)
            torch.testing.assert_close(result['ema']['weight'],full['ema']['weight'],rtol=0,atol=0)
            self.assertEqual(result['ema']['weight'].dtype,torch.float32)


if __name__=='__main__': unittest.main()
