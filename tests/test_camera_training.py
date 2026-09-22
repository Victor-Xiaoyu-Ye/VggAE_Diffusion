import tempfile
import unittest
from pathlib import Path
import torch
from models.r7_window_dit import R7WindowDiT
from utils.camera_training import CameraBank, BoundCameraModel, initialize_from_checkpoint, CameraFilteredDataset
from utils.fullhq_text import TextCFG


class CameraTrainingTests(unittest.TestCase):
    def test_exact_join_null_wrong_and_shared_cfg_control(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bank.pt'
            features={k:torch.ones(3,14)*v for k,v in [('train:0,1,2',1),('a:0,1,2',2),('b:0,1,2',3)]}
            torch.save(dict(schema='r7-camera-pilot-bank-v1',features=features,
                train_keys=['train:0,1,2'],eval_keys=['a:0,1,2','b:0,1,2']),path)
            bank=CameraBank(path,3);batch=dict(video_id=['a'],frame_indices=[[0,1,2]])
            camera,present=bank.batch(batch,'cpu','wrong')
            self.assertEqual(float(camera[0,0,0]),3);self.assertEqual(float(camera[0,0,10]),2)
            self.assertFalse(bool(bank.batch(batch,'cpu','null')[1][0]))
            with self.assertRaises(KeyError):bank.batch(dict(video_id=['a'],frame_indices=[[0,2,3]]),'cpu')
            calls=[]
            def model(x,u,c,t,v,**kw):
                calls.append(kw);return x
            bound=BoundCameraModel(model,camera,present)
            TextCFG(bound,torch.zeros(1,4096),3)(torch.ones(1,2,4,6),torch.ones(1),torch.ones(1),
                torch.ones(1,3,4096),torch.ones(1,3,dtype=torch.bool))
            self.assertEqual(len(calls),2)
            for call in calls:
                self.assertIs(call['camera'],camera);self.assertIs(call['camera_present'],present)

    def test_initialization_loads_only_matching_base_and_rejects_changed_statistics(self):
        args=dict(channels=6,grid=2,future=2,temporal_factor=1,width=24,depth=1,heads=3,text_dim=0)
        base=R7WindowDiT(**args);model=R7WindowDiT(**args,camera_dim=14)
        statistics=dict(sample=torch.ones(2));runtime={'norm':'legacy'}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'init.pt'
            torch.save(dict(schema='r7-window-trainer-v1',step=6,model_args=args,statistics=statistics,
                contract=dict(identity=dict(representation=dict(window_codec_runtime=runtime))),ema=base.state_dict()),path)
            result=initialize_from_checkpoint(model,path,6,'ema',dict(args,camera_dim=14),statistics,runtime,large=True)
            self.assertEqual(result['step'],6)
            for key,value in base.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key])
            self.assertEqual(float(model.camera_conditioner.projection[-1].weight.detach().abs().sum()),0)
            with self.assertRaises(ValueError):initialize_from_checkpoint(model,path,7,'ema',args,statistics,runtime)
            with self.assertRaises(ValueError):initialize_from_checkpoint(model,path,6,'ema',args,dict(sample=torch.zeros(2)),runtime)

    def test_filter_is_exact_and_same_for_both_arms(self):
        samples=[dict(video_id='x',frame_indices=[0,1]),dict(video_id='x',frame_indices=[2,3])]
        self.assertEqual(list(CameraFilteredDataset(samples,{'x:2,3'})),samples[1:])


if __name__=='__main__':unittest.main()
