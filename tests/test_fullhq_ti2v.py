import csv
import hashlib
import json
from pathlib import Path
import tempfile
import os
import sys
import types
import unittest
from unittest.mock import patch

import torch
from scripts.prepare_fullhq import prepare
from utils.fullhq_text import ShardedCaptionBank,TextCFG
from utils.window_training import use_ema
from utils.training import EMA
from models.r7_window_dit import R7WindowDiT


class FullHQTests(unittest.TestCase):
    def test_all_valid_rows_with_heldout_exclusion_and_failure_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);source=p/'metadata.csv'
            with source.open('w',newline='') as f:
                writer=csv.DictWriter(f,['id','video path','fps','num frames']);writer.writeheader()
                for vid,fps in [('street',30),('drone',30),('eval',30),('test',30),('bad',0)]:
                    writer.writerow({'id':vid,'video path':f'videos/group_0000/{vid}.mp4','fps':fps,'num frames':90})
            protocol=p/'protocol.json';protocol.write_text(json.dumps(dict(metadata_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                splits={'eval_street_128':['eval'],'test_other_128':['test']})))
            report=prepare(source,protocol,p/'out')
            self.assertEqual(report['train_videos'],2);self.assertEqual(report['invalid'][0]['video_id'],'bad')
            with (p/'out/train.csv').open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual({r['id'] for r in rows},{'street','drone'})
            source.write_text(source.read_text()+'\n')
            with self.assertRaises(ValueError):prepare(source,protocol,p/'out')

    def bank(self,p):
        files={}
        for name,value in [('empty_prompt.pt',torch.zeros(1,4096)),('a.pt',{'a':torch.ones(2,4096)}),('b.pt',{'b':torch.ones(3,4096)*2})]:
            torch.save(value,p/name);files[name]=hashlib.sha256((p/name).read_bytes()).hexdigest()
        (p/'index.json').write_text(json.dumps(dict(index={'a':'a.pt','b':'b.pt','missing':None},files=files,text_len=256)))
        (p/'_SUCCESS').write_text(json.dumps(dict(schema='fullhq-text-v1',index_sha256=hashlib.sha256((p/'index.json').read_bytes()).hexdigest())))

    def test_text_loads_only_needed_shards_and_explicit_missing_caption(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);self.bank(p)
            from utils.moxing_io import read_bytes
            with patch('utils.fullhq_text.read_bytes',wraps=read_bytes) as read:
                bank=ShardedCaptionBank(str(p),max_shards=1)
                self.assertFalse(any(str(c.args[0]).endswith(('a.pt','b.pt')) for c in read.call_args_list))
                values,valid=bank.batch(['a','missing'],'cpu')
                self.assertEqual(valid.sum(1).tolist(),[2,1]);self.assertEqual(float(values[1].sum()),0.)
                bank.get('b');self.assertEqual(list(bank.cache),['b.pt'])
                (p/'a.pt').write_bytes(b'corrupt')
                with self.assertRaises(ValueError):bank.get('a')
                with self.assertRaises(ValueError):bank.batch(['unregistered'],'cpu')

    def test_text_cfg_preserves_image_and_pads_empty_independently(self):
        calls=[];anchor=torch.randn(1,1,4,6);x=torch.zeros(1,4,4,6)
        def model(x,u,c,t,valid):
            calls.append((c.clone(),t.shape,valid.clone()));return torch.ones_like(x)*t.mean()
        got=TextCFG(model,torch.zeros(1,4096),3)(x,torch.ones(1),anchor,torch.ones(1,5,4096),torch.ones(1,5,dtype=torch.bool))
        torch.testing.assert_close(got,torch.ones_like(x)*3)
        for c,_,_ in calls:torch.testing.assert_close(c,anchor)
        self.assertEqual(calls[1][1],torch.Size([1,1,4096]))

    def test_cpu_ema_backup_restores_even_on_failure(self):
        model=torch.nn.Linear(3,2);ema=EMA(model);before={k:v.clone() for k,v in model.state_dict().items()}
        for value in ema.shadow.values():value.zero_()
        with self.assertRaises(RuntimeError):
            with use_ema(model,ema,cpu_backup=True):
                self.assertEqual(float(model.weight.detach().abs().sum()),0.);raise RuntimeError('interrupted eval')
        for name,value in model.state_dict().items():torch.testing.assert_close(value,before[name])

    def test_large_model_parameter_budget(self):
        with torch.device('meta'):model=R7WindowDiT(width=1536,depth=24,heads=24,text_dim=4096)
        n=sum(p.numel() for p in model.parameters())
        self.assertEqual(n,1652968512)
        self.assertLess(n*20/2**30,31.)

    def test_strict_video_rejects_missing_and_out_of_range_frames(self):
        # EXR bindings are unrelated to the RGB decoder exercised here.
        with patch.dict(sys.modules,{'OpenEXR':types.ModuleType('OpenEXR'),'Imath':types.ModuleType('Imath')}):
            from utils.video_io import read_video_frames,VideoDecodeError
        import numpy as np
        class Capture:
            def isOpened(self):return True
            def get(self,key):return 3
            def set(self,key,value):self.position=int(value)
            def read(self):return (False,None) if self.position==1 else (True,np.zeros((4,4,3),dtype=np.uint8))
            def release(self):pass
        with patch('utils.video_io.cv2.VideoCapture',side_effect=lambda path:Capture()):
            for indices in ([0,1],[0,9]):
                with self.assertRaises(VideoDecodeError):
                    read_video_frames('fixture',num_frames=2,target_size=4,frame_indices=indices,strict_frames=True)
            frames=read_video_frames('fixture',num_frames=2,target_size=4,frame_indices=[0,1])
            self.assertEqual(frames.shape[0],2)

    def test_text_worker_resume_merge_and_caption_payload(self):
        from scripts.precompute_fullhq_text import main
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);wan=p/'wan';wan.mkdir();(wan/'models_t5_umt5-xxl-enc-bf16.pth').write_bytes(b'fixture')
            tokenizer=wan/'google/umt5-xxl';tokenizer.mkdir(parents=True);(tokenizer/'tokenizer.json').write_text('{}')
            source=p/'rows.csv'
            with source.open('w',newline='') as f:
                w=csv.DictWriter(f,['id','video path']);w.writeheader()
                for vid in ('a','b','c'):
                    w.writerow({'id':vid,'video path':f'videos/group_0000/{vid}.mp4'})
                    path=p/f'annotations/group_0000/{vid}/caption.json';path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(dict(SceneDescription='' if vid=='c' else 'street '+vid)))
            class Encoder:
                def __init__(self,**kw):pass
                def __call__(self,texts,device):return [torch.ones(max(1,len(t)),4096)*len(t) for t in texts]
            fake=types.ModuleType('precompute_wan_text_embeddings')
            fake._load_wan_module=lambda name:types.SimpleNamespace(T5EncoderModel=Encoder)
            argv=['text','--csv',str(source),'--annotation_root',str(p/'annotations'),'--output',str(p/'bank'),
                  '--staging',str(p/'stage'),'--wan_ckpt',str(wan),'--nodes','1','--shard_size','2','--max_missing_fraction','.5']
            with patch.dict(sys.modules,{'precompute_wan_text_embeddings':fake}),patch.dict(os.environ,{'NODE_RANK':'0'}), \
                 patch('utils.device.get_device',return_value=torch.device('cpu')),patch('utils.device.get_device_name',return_value='npu'), \
                 patch.object(torch,'npu',types.SimpleNamespace(set_device=lambda n:None),create=True):
                with patch.object(sys,'argv',argv):main()
                fake._load_wan_module=lambda name:(_ for _ in ()).throw(AssertionError('resume must not reload T5'))
                with patch.object(sys,'argv',argv):main()
                with patch.object(sys,'argv',argv+['--merge']):main()
            bank=ShardedCaptionBank(str(p/'bank'))
            self.assertEqual(bank.caption('a'),'street a');self.assertEqual(bank.caption('c'),'')
            self.assertEqual(len(bank.values),3)


if __name__=='__main__':unittest.main()
