"""CPU integration: actual codec/flow, synthetic frozen features and RGB only.

These tests do not claim to validate StreamVGGT weights, LPIPS or Ascend.
"""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from data.scene_rgb import SceneRGB, camera_rays, temporal_stream
from models.scene_flow import SceneFlow
from models.scene_rae import SceneRAE, relation_loss
from scripts.audit_wai_rgb import summarize
from utils.r7_representation import R7Config, build_modules, merged_representation_state
from utils.scene_run import ArtifactStore, StageBudget
from utils.window_flow import WindowFlow
from utils.scene_text import SceneCaptionBank
import train_scene_pipeline as pipeline


def ddp_worker(rank,root):
    os.environ['RANK']=str(rank)
    torch.set_num_threads(1)
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method=(Path(root)/'rendezvous').as_uri(),rank=rank,world_size=2)
    try:
        torch.manual_seed(42+rank)
        model=SceneFlow(channels=4,grid=2,width=24,depth=1,head_width=48,head_depth=1,heads=4,text_dim=8)
        wrapped=pipeline.wrap(model,torch.device('cpu'),2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        assert pipeline.any_rank(rank==1,torch.device('cpu'))
        for length in (1,3):
            optimizer.zero_grad()
            for micro in range(2):
                import contextlib
                ctx=wrapped.no_sync() if micro==0 else contextlib.nullcontext()
                with ctx:
                    out=wrapped(torch.randn(1,length,4,4),torch.rand(1),torch.randn(1,1,4,4),
                        torch.randn(1,2,8),torch.ones(1,2,dtype=torch.bool),ref_present=torch.tensor([length>1]))
                    (out-1).square().mean().backward()
            optimizer.step()
        sums=pipeline.gather(float(sum(p.detach().sum() for p in model.parameters())),2)
        assert abs(sums[0]-sums[1])<1e-5
        store=ArtifactStore(Path(root)/'shared',[str(Path(root)/'a'),str(Path(root)/'b')])
        store.save(f'rank{rank}.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict()),'ddp')
        pipeline.barrier()
        for r in range(2):
            assert store.load(f'rank{r}.pt','ddp',True)['optimizer']
    finally:
        dist.destroy_process_group()


class FrozenFixture(nn.Module):
    def forward(self, x):
        b,t,c,h,w=x.shape
        z=F.adaptive_avg_pool2d(x.flatten(0,1),(2,2)).permute(0,2,3,1).reshape(b,t,4,3)
        return [torch.cat((z,z.square(),z.sin(),z.cos()),-1)*(1+i*.1) for i in range(4)],0


class PerceptualFixture(nn.Module):
    def forward(self,a,b):
        return (a-b).square().mean((1,2,3),keepdim=True)


class ScenePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mmap_filename_contract_and_optimizer_recovery(self):
        # Emulate torch_npu's stricter Linux mmap argument check on CPU.
        # Patch only scene_run's os binding so Windows pathlib stays native.
        native_load=torch.load
        def npu_load(filename, **kwargs):
            if kwargs.get('mmap') and not isinstance(filename,str):
                raise TypeError('f must be a string filename in order to use mmap argument')
            self.assertTrue(kwargs['mmap'])
            if os.name=='nt':
                kwargs['mmap']=False  # This is not a real NPU mmap test.
            return native_load(filename,**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            replicas=[str(root/'primary'),str(root/'mirror')]
            store=ArtifactStore(root/'writer',replicas)
            model=nn.Linear(3,2)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            x=torch.randn(2,3)
            def update(net,opt):
                opt.zero_grad()
                net(x).square().mean().backward()
                opt.step()
            update(model,optimizer)
            store.save('ae/checkpoint_latest.pt',dict(step=1,model=model.state_dict(),
                       optimizer=optimizer.state_dict()),'fixture')
            # A restarted worker must also recover the published file.
            reader=ArtifactStore(root/'reader',replicas)
            linux_os=SimpleNamespace(name='posix',getpid=os.getpid,replace=os.replace)
            with patch('utils.scene_run.os',linux_os), patch('utils.scene_run.torch.load',side_effect=npu_load):
                saved=reader.load('ae/checkpoint_latest.pt','fixture',required=True)
            restored=nn.Linear(3,2)
            restored.load_state_dict(saved['model'])
            restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=.001)
            restored_optimizer.load_state_dict(saved['optimizer'])
            self.assertEqual(saved['step'],1)
            update(model,optimizer)
            update(restored,restored_optimizer)
            for expected,actual in zip(model.parameters(),restored.parameters()):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            self.assertTrue(all(state['step'].item()==2 for state in restored_optimizer.state.values()))

    def test_direct_relation_gradient(self):
        z=torch.randn(1,2,3,3,4,requires_grad=True)
        loss=relation_loss(z,torch.randn(1,2,3,3,12))
        loss.backward()
        self.assertGreater(z.grad.norm().item(),0)

    def test_same_weights_image_video_and_null_reference(self):
        model=SceneFlow(channels=4,grid=2,width=24,depth=1,head_width=48,
                        head_depth=1,heads=4,text_dim=8).eval()
        for p in model.parameters():
            nn.init.normal_(p,std=.1)
        anchor=torch.randn(2,1,4,4)
        text=torch.randn(2,3,8); valid=torch.ones(2,3,dtype=torch.bool)
        for t in (1,3):
            noise=torch.randn(2,t,4,4);u=torch.ones(2)*.5
            first=model(noise,u,anchor,text,valid,ref_present=torch.zeros(2,dtype=torch.bool))
            second=model(noise,u,anchor*999,text,valid,ref_present=torch.zeros(2,dtype=torch.bool))
            torch.testing.assert_close(first,second)
            first.square().mean().backward()
            self.assertTrue(torch.isfinite(model.input.weight.grad).all())

    def test_camera_convention_crop_and_invalid_absence(self):
        meta=dict(camera_convention='opencv',camera_model='PINHOLE',shared_intrinsics=True,
                  h=100,w=200,fl_x=100,fl_y=100,cx=100,cy=50)
        poses=[dict(transform_matrix=np.eye(4).tolist()) for _ in range(2)]
        poses[1]['transform_matrix'][0][3]=2
        rays=camera_rays(poses,meta,[0,1],1)
        torch.testing.assert_close(rays[0,0,:3],torch.tensor([0.,0.,1.]))
        torch.testing.assert_close(rays[1,0,3:6],torch.tensor([0.,-1.,0.]))
        self.assertTrue(rays[...,7].all())
        scaled=copy.deepcopy(poses)
        scaled[1]['transform_matrix'][0][3]=200
        torch.testing.assert_close(rays,camera_rays(scaled,meta,[0,1],1))
        scaled[1]['transform_matrix'][0][0]=-1
        self.assertFalse(camera_rays(scaled,meta,[0,1],1).any())
        poses[1]['transform_matrix'][0][0]=4
        self.assertFalse(camera_rays(poses,meta,[0,1],1).any())

    def test_metadata_counts_empty_directories_and_bad_flags(self):
        report=summarize(dict(frames=[dict(image='images/1.jpg'),
            dict(image='images/2.jpg'),dict(image='images/3.jpg',is_bad=True)]),
            {'images/1.jpg':12,'images/2.jpg':0,'images/3.jpg':12})
        self.assertEqual((report['usable_frames'],report['missing_rgb'],report['flagged_bad_frames']),(1,1,1))

    def test_stereo_camera_is_not_video_time_and_gaps_are_not_joined(self):
        import random
        frames=[dict(image=f'images/frame_{side}_{i:04d}.png')
                for i in (1,2,3,8,9) for side in ('left','right')]
        result=temporal_stream(frames,random.Random(42))
        self.assertEqual(len(result),3)
        self.assertEqual(len({f['image'].split('_')[1] for f in result}),1)
        self.assertEqual([int(f['image'][-8:-4]) for f in result],[1,2,3])

    def test_dual_read_survives_corrupt_primary_and_rejects_contract_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            writer=ArtifactStore(root/'writer',[str(root/'a'),str(root/'b')])
            writer.save('model.pt',dict(value=torch.tensor([1.,2.])),identity='test')
            (root/'a/model.pt').write_bytes(b'broken')
            reader=ArtifactStore(root/'reader',[str(root/'a'),str(root/'b')])
            out=reader.load('model.pt',identity='test',required=True)
            torch.testing.assert_close(out['value'],torch.tensor([1.,2.]))
            with self.assertRaises(ValueError):
                reader.load('model.pt',identity='other',required=True)

    def test_two_rank_mixed_tasks_and_concurrent_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(ddp_worker,args=(tmp,),nprocs=2,join=True)

    def test_newest_receipt_overrides_stale_local_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            replicas=[str(root/'a'),str(root/'b')]
            writer=ArtifactStore(root/'writer',replicas)
            reader=ArtifactStore(root/'reader',replicas)
            writer.save('latest.pt',dict(step=1),'state')
            self.assertEqual(reader.load('latest.pt','state')['step'],1)
            writer.save('latest.pt',dict(step=2),'state')
            self.assertEqual(reader.load('latest.pt','state')['step'],2)
            self.assertEqual(reader.json('latest.pt.json')['sha256'],writer.json('latest.pt.json')['sha256'])

    def test_caption_subset_is_reused_without_reading_large_source_again(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source';source.mkdir()
            torch.save(torch.zeros(1,4096),source/'empty_prompt.pt')
            torch.save(dict(embeddings={'keep':torch.ones(2,4096),'drop':torch.zeros(2,4096)},
                            captions={'keep':'a street','drop':'unused'}),source/'part.pt')
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.pt')}
            (source/'index.json').write_text(json.dumps(dict(index={'keep':'part.pt','drop':'part.pt'},files=files,text_len=256)))
            (source/'_SUCCESS').write_text(json.dumps(dict(schema='fullhq-text-v1',
                index_sha256=hashlib.sha256((source/'index.json').read_bytes()).hexdigest())))
            bank=SceneCaptionBank(str(source),['keep'],root/'cache')
            torch.testing.assert_close(bank.get('keep'),torch.ones(2,4096))
            (source/'part.pt').unlink()
            another=SceneCaptionBank(str(source),['keep'],root/'cache')
            self.assertEqual(another.caption('keep'),'a street')
            self.assertNotIn('drop',another.cache['part.pt'])

    def test_cohort_has_one_parent_and_preserves_historical_splits(self):
        cohort=json.loads(Path('configs/scene_cohort_v1.json').read_text(encoding='utf8'))
        rows=cohort['records']
        self.assertEqual(len(rows),len({r['id'] for r in rows}))
        self.assertTrue(cohort['historical_split_preserved'])
        self.assertFalse(cohort['media_decode_verified'])
        self.assertTrue(all(r['split'] in ('train','val','test') for r in rows))
        for source in ('dl3dv','mvssynth','scannetppv2','spatialvid','omniworld'):
            self.assertEqual({r['split'] for r in rows if r['dataset']==source},{'train','val','test'})
        history=json.loads(Path('configs/spatialvid_domain_v2.json').read_text())
        train={r['id'].split(':',1)[1] for r in rows if r['dataset']=='spatialvid' and r['split']=='train'}
        # Existing protocol contains all protected eval/test IDs. Inspect its
        # named split collections, not arbitrary metadata strings.
        protected=set()
        for name,ids in history['splits'].items():
            if 'eval' in name or 'test' in name:
                protected.update(ids)
        self.assertEqual(len(protected),512)
        self.assertFalse(train & protected)
        self.assertEqual({r['split'] for r in rows if r['dataset']=='pointodyssey'},{'train','val'})
        spring_train={r['id'].split(':',1)[1] for r in rows if r['dataset']=='spring' and r['split']=='train'}
        self.assertFalse(spring_train & {'0003','0019','0028','0029','0031','0034','0035','0040','0042','0046'})

    def test_pipeline_rgb_ae_cache_image_video_and_resume(self):
        # Full orchestration with small real modules. Only costly pretrained
        # teacher/perceptual networks and MP4 rendering are replaced by fixtures.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            records=[]
            for scene,split in [('s0','train'),('s1','train'),('s2','val')]:
                folder=root/scene/'images';folder.mkdir(parents=True)
                frames=[]
                for i in range(5):
                    rgb=np.zeros((28,28,3),np.uint8)
                    rgb[:,:]=[i*30,80,120];rgb[5:16,4+i:15+i]=[200,150,50]
                    Image.fromarray(rgb).save(folder/f'{i:03d}.png')
                    pose=np.eye(4);pose[0,3]=i*.1
                    frames.append(dict(image=f'images/{i:03d}.png',transform_matrix=pose.tolist()))
                meta=dict(frames=frames,camera_convention='opencv',camera_model='PINHOLE',
                          shared_intrinsics=True,h=28,w=28,fl_x=28,fl_y=28,cx=14,cy=14)
                (root/scene/'scene_meta.json').write_text(json.dumps(meta))
                records.append(dict(id='dl3dv:'+scene,dataset='dl3dv',kind='wai',split=split,root=str(root/scene)))
            cohort=dict(seed='test',records=records)
            cfg=json.loads(Path('configs/scene_15day_v1.json').read_text())
            cfg.update(size=28,grid=2,views=3,channels=16,workers=0,ae_steps=2,image_steps=2,
                video_steps=2,ae_accum=1,flow_accum=1,eval_per_domain=1,log_every=1,shard_samples=2,
                cache_windows=1,flow=dict(width=24,depth=1,head_width=48,head_depth=1,heads=4,text_dim=4096))
            rc=R7Config(target_size=28,input_grid=2,latent_grid=2,levels=(0,1,2,3),token_dim=12,
                geo_dim=8,tex_dim=8,tex_base_ch=8,decoder_base_dim=16,decoder_num_resblocks=1,
                decoder_temporal_blocks=1,temporal_depth=1,geo_latent_dim=4,tex_latent_dim=4,seq_len=3)
            modules=build_modules(rc)
            torch.save(dict(args=asdict(rc),model=merged_representation_state(*modules)),root/'r7.pt')
            store=ArtifactStore(root/'out',[str(root/'replica1'),str(root/'replica2')])
            args=SimpleNamespace(r7=str(root/'r7.pt'),encoder='fixture',smoke=True,text='',stage='ae')
            def fake_video(path,frames,fps=8):
                path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
                self.assertTrue(torch.isfinite(frames).all())
                path.write_bytes(b'fixture-video')
            with patch.object(pipeline,'make_encoder',lambda p,d:FrozenFixture()), \
                 patch('utils.scene_losses.get_lpips',lambda d:PerceptualFixture()), \
                 patch.object(pipeline,'video',fake_video):
                pipeline.train_ae(args,cfg,cohort,store,'fixture',torch.device('cpu'),0,1)
                self.assertIsNotNone(store.json('ae/complete.json'))
                pipeline.cache(args,cfg,cohort,store,'fixture',torch.device('cpu'),0,1)
                for phase in ('image','video'):
                    args.stage=phase
                    pipeline.train_flow(args,cfg,cohort,store,'fixture',torch.device('cpu'),0,1)
                    result=store.json(phase+'/complete.json',True)
                    self.assertEqual(result['step'],2)
                    self.assertTrue(store.json(phase+'/resume_rehearsal.json')['optimizer_restored'])
                self.assertEqual(store.json('ae/complete.json')['quality_policy'],'report_only')
                manifest=store.json('cache/complete.json')
                self.assertEqual(len(manifest['mean']),16)
                self.assertTrue(all(v>0 for v in manifest['std']))
            if pipeline.WRITER is not None:
                pipeline.WRITER.close();pipeline.WRITER=None


if __name__=='__main__':
    unittest.main()
