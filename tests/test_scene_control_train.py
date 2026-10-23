"""Real tiny controlled-flow optimizer, sampling, dual recovery and DDP tests."""
from contextlib import ExitStack
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

import train_scene_prefix as trainer
from train_scene_control import ControlRuntime, build_plan
from tests.test_scene_prefix_train import (fixture, new_store, seed_all, TinyRAE,
    fake_preview, record_metrics)
from tests import test_scene_prefix_train as prefix_test_helpers
from tests.test_scene_motion_plan import fixture as plan_fixture
from utils.scene_run import ArtifactStore, StageBudget, digest


def add_controls(sample):
    points=torch.tensor([[.25,.25],[.75,.75]])
    tracks=points[None].expand(9,2,2).clone()
    tracks[4:,:,0]-=torch.arange(1,6)[:,None]*.02
    rays=torch.zeros(9,4,8); rays[...,2]=1.; rays[...,7]=1.
    motion=torch.zeros(9,4,8); motion[...,2]=1.; motion[4:,:,0]=-.1
    return dict(sample,controls=dict(schema='scene_control_v1',rays=rays,motion=motion,
        tracks=tracks,track_weight=torch.ones(9,2),prefix_tracks=tracks[:4].clone(),
        prefix_track_weight=torch.ones(4,2)))


def controlled_fixture(root):
    data=fixture(root)
    data['samples']=[add_controls(sample) for sample in data['samples']]
    payload=data['source'].load('eval.pt','fixture-validation',True)
    data['source'].save('eval.pt',dict(samples=[dict(add_controls(s),source_eval_index=17)
        for s in payload['samples']]),'fixture-validation')
    return data


def execute(data,store,manifest,arm,*,stage='train',steps=3,rank=0,world=1,bank=None,smoke=False):
    cfg=dict(data['cfg'],max_steps=steps,source_label='source_fixture',
             eval_every_steps=1,convergence=dict(min_steps=100,patience=4,relative_tol=.002))
    if smoke:
        cfg['convergence']=None
    with ExitStack() as stack:
        stack.enter_context(patch.object(trainer,'SceneRAE',TinyRAE))
        stack.enter_context(patch.object(trainer,'preview',fake_preview))
        stack.enter_context(patch.object(trainer,'metrics',record_metrics))
        source=data['source']
        trainer.train(SimpleNamespace(stage=stage),cfg,data['plan'],source,store,manifest,data['selection'],
            source.local/'rae.pt',source.local/'weights.pt',source.local/'eval.pt',
            StageBudget(10000),torch.device('cpu'),rank,world,data_store=bank,
            runtime=ControlRuntime(arm))
    return store.load('train/checkpoint_latest.pt',data['plan']['identity'],True)


def controlled_ddp_worker(rank,folder):
    root=Path(folder)
    os.environ.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2',LOCAL_WORLD_SIZE='2')
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=(root/'rendezvous').as_uri(),rank=rank,world_size=2,
                            timeout=timedelta(seconds=120))
    try:
        data=torch.load(str(root/'fixture.pt'),weights_only=False)
        data['source']=ArtifactStore(root/'source',[])
        store=ArtifactStore(root/'ddp',[str(root/'ddp-primary'),str(root/'ddp-mirror')])
        seed_all(42+rank)
        state=execute(data,store,data['manifest'],'reference',stage='preflight',rank=rank,world=2)
        torch.save(state['model'],root/f'rank{rank}.pt')
        dist.barrier()
    finally:
        dist.destroy_process_group()


class ControlTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_nested_equal(self,a,b):
        # Reuse the exact tensor/RNG comparison, not the inherited test suite.
        return prefix_test_helpers.PrefixTrainingTests.assert_nested_equal(self,a,b)

    def test_all_arms_preserve_source_baseline_and_share_noise_schedule(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp); data=controlled_fixture(root)
            baselines=[]; random_states=[]; bins=[]
            for arm in ('baseline','motion','reference'):
                store,manifest,_,_=new_store(root,arm,data,malformed=False)
                seed_all(42)
                saved=execute(data,store,manifest,arm,steps=1)
                baselines.append(store.load('eval/source_fixture/online/case000_seed101.pt',
                    data['plan']['identity'],True)['latent'])
                row=store.json('eval/step0000001/ema/case000_seed101.json',True)
                self.assertEqual(row['future_control_supplied'],arm!='baseline')
                self.assertFalse(row['semantic_instance_masks'])
                self.assertEqual(row['source_eval_index'],17)
                self.assertEqual(row['denoising_probe']['experiment_arm'],arm)
                random_states.append(saved['rng'])
                records=[json.loads(line) for line in (store.local/'metrics-r000.jsonl').read_text().splitlines()]
                logs=[r for r in records if r['stage']=='train']
                self.assertTrue(all(r['DI_throughput']>0 for r in logs))
                bins.append([r['noise_bin_counts'] for r in logs])
                if arm!='baseline':
                    self.assertGreater(saved['model']['control.2.weight'].abs().sum().item(),0.)
            for i in (1,2):
                torch.testing.assert_close(baselines[0],baselines[i],rtol=0,atol=0)
                self.assertEqual(bins[0],bins[i])
                self.assert_nested_equal(random_states[0],random_states[i])

    def test_reference_resume_from_mirror_matches_continuous(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp); data=controlled_fixture(root)
            store,manifest,_,_=new_store(root,'resumed',data,malformed=False)
            other,other_manifest,_,_=new_store(root,'continuous',data,malformed=False)
            seed_all(42)
            two=execute(data,store,manifest,'reference',stage='preflight')
            self.assertEqual(two['step'],2)
            latest='train/checkpoint_latest.pt'
            (store.local/latest).unlink()
            (Path(store.roots[0])/latest).write_bytes(b'bad-primary')
            seed_all(999)
            resumed=execute(data,store,manifest,'reference')
            seed_all(42)
            continuous=execute(data,other,other_manifest,'reference')
            for key in ('model','optimizer','ema','ema_metadata','cursors','rng','convergence_state'):
                self.assert_nested_equal(resumed[key],continuous[key])
            self.assertEqual(resumed['step'],3)

    def test_source_preview_repair_does_not_use_trained_adapters(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp); data=controlled_fixture(root)
            store,manifest,_,_=new_store(root,'repair',data,malformed=False)
            seed_all(42)
            saved=execute(data,store,manifest,'reference',steps=2,smoke=True)
            path='eval/source_fixture/online/case000_seed101.pt'
            before=store.load(path,data['plan']['identity'],True)['latent'].clone()
            for destination in [store.local,*map(Path,store.roots)]:
                (destination/'eval/source_fixture/summary.json').unlink()
                (destination/'eval/source_fixture/online/case000_seed101.done.json').unlink()
            after=execute(data,store,manifest,'reference',steps=2,smoke=True)
            torch.testing.assert_close(before,store.load(path,data['plan']['identity'],True)['latent'],rtol=0,atol=0)
            self.assert_nested_equal(saved['model'],after['model'])
            self.assertEqual(saved['cursors'],after['cursors'])
            # A partial summary must also be repaired without plateau tracking.
            summary=store.json('eval/source_fixture/summary.json',True)
            store.write_json('eval/source_fixture/summary.json',dict(summary,status='partial'))
            after=execute(data,store,manifest,'reference',steps=2,smoke=True)
            self.assertEqual(store.json('eval/source_fixture/summary.json')['status'],'completed')
            self.assertEqual(saved['cursors'],after['cursors'])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(),'Gloo unavailable')
    def test_two_rank_ddp_with_reference_adapter_and_empty_local_eval_jobs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp); data=controlled_fixture(root)
            _,manifest,_,_=new_store(root,'ddp',data,malformed=False)
            torch.save(dict({k:v for k,v in data.items() if k!='source'},manifest=manifest),root/'fixture.pt')
            torch.multiprocessing.spawn(controlled_ddp_worker,args=(str(root),),nprocs=2,join=True)
            a=torch.load(str(root/'rank0.pt'),weights_only=False)
            b=torch.load(str(root/'rank1.pt'),weights_only=False)
            self.assert_nested_equal(a,b)
            self.assertGreater(a['control.2.weight'].abs().sum().item(),0.)


class ControlPlanTests(unittest.TestCase):
    def test_control_data_and_source_receipts_are_pinned_for_every_arm(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp)
            source,codec,cfg,selection,previous,original=plan_fixture(root)
            controls=ArtifactStore(root/'controls',[])
            controls.write_json('contract.json',dict(identity='controls-identity'))
            manifest=dict(identity='controls-identity',source_prefix=previous['identity'],
                source_cache_manifest_sha=digest(original),mean=previous['mean'],std=previous['std'],
                shards=[dict(file='data/shards/one.pt',sources=['spatialvid'])],samples=1)
            controls.write_json('data/cache_complete.json',manifest)
            controls.save('data/eval.pt',dict(samples=[]),'controls-identity')
            for arm in ('baseline','motion','reference'):
                plan,got=build_plan(source,codec,controls,dict(cfg,arm=arm),selection,24)
                self.assertEqual(got,manifest)
                self.assertEqual(plan['control_identity'],'controls-identity')
                self.assertEqual(plan['mean'],previous['mean'])
                self.assertEqual(plan['matched_windows'],1)
            for key,wrong in [('source_prefix','another-run'),('source_cache_manifest_sha','bad'),('std',[1.]*4)]:
                controls.write_json('data/cache_complete.json',dict(manifest,**{key:wrong}))
                with self.assertRaises(ValueError):
                    build_plan(source,codec,controls,dict(cfg,arm='reference'),selection,24)


if __name__=='__main__':
    unittest.main()
