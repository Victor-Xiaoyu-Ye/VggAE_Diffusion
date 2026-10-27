"""V2 control construction and actual runtime tests, without teacher inference."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from utils.scene_control import control_features, create_control_model
from utils.scene_control_inference import CONTROL_INPUT_POLICY, inference_control_view
from utils.scene_prefix import prepare_prefix
from train_scene_control import ControlRuntime, resolve_control_readiness, control_readiness_decision


def fixture():
    xy=torch.tensor([[[.25,.25],[.75,.75]]]).expand(9,2,2).clone()
    xy[4:,:,0]-=torch.arange(1,6)[:,None]*.03
    rays=torch.zeros(9,4,8);rays[...,2]=1.;rays[...,7]=1.
    return dict(schema='scene_control_v1',rays=rays,motion=torch.full((9,4,8),77.),
        tracks=xy,track_weight=torch.full((9,2),.2),prefix_tracks=xy[:4].clone(),
        prefix_track_weight=torch.full((4,2),.1),metadata=dict(teacher_motion_diagnostics={'future_depth':777}))


class InferenceControlViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_view_equal(self,a,b):
        self.assertEqual(set(a),set(b))
        for key in a:
            if torch.is_tensor(a[key]):
                torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
            else:
                self.assertEqual(a[key],b[key])

    def test_ignores_future_labels_diagnostics_and_positive_confidence_amplitude(self):
        controls=fixture(); saved=copy.deepcopy(controls)
        expected=inference_control_view(controls)
        changed=copy.deepcopy(controls)
        changed['motion'].fill_(float('nan'))
        changed['track_weight'].fill_(1e-10);changed['prefix_track_weight'].fill_(.99)
        changed['metadata']={'teacher_motion_diagnostics':{'depth':torch.full((9,5,5),float('nan'))}}
        changed['future_depth']=object()
        self.assert_view_equal(expected,inference_control_view(changed))
        self.assert_view_equal(controls,saved)
        self.assertTrue((expected['motion'][...,4:6]==0).all())
        self.assertTrue((expected['motion'][...,6]==1).all())
        self.assertEqual(set(expected['track_weight'].unique().tolist()),{1.})
        self.assertFalse(expected['metadata']['future_depth_used'])

    def test_camera_and_tracks_change_only_supplied_inputs(self):
        controls=fixture();base=inference_control_view(controls)
        moved=copy.deepcopy(controls);moved['tracks'][4:,0,1]+=.1
        other=inference_control_view(moved)
        self.assertFalse(torch.equal(base['motion'],other['motion']))
        camera=copy.deepcopy(controls);camera['rays'][...,3]=.5
        other=inference_control_view(camera)
        self.assertFalse(torch.equal(base['rays'],other['rays']))
        torch.testing.assert_close(base['motion'],other['motion'],atol=0,rtol=0)

    def test_subpixel_splat_binary_visibility_and_empty_tracks(self):
        controls=fixture()
        controls['tracks']=torch.tensor([.5,.5]).reshape(1,1,2).expand(9,1,2)
        controls['prefix_tracks']=controls['tracks'][:4]
        controls['track_weight']=torch.ones(9,1)*.123
        controls['prefix_track_weight']=torch.ones(4,1)*.456
        view=inference_control_view(controls)
        torch.testing.assert_close(view['motion'][...,2],torch.full((9,4),.25))
        self.assertTrue((view['motion'][...,3]==1).all())
        for drop in ('tracks','track_weight','all'):
            missing=copy.deepcopy(controls)
            for name in (('tracks','track_weight','prefix_tracks','prefix_track_weight') if drop=='all' else (drop,)):
                missing.pop(name)
            empty=inference_control_view(missing)
            self.assertTrue((empty['motion'][...,:6]==0).all())
            self.assertTrue((empty['motion'][...,6]==1).all())
            self.assertEqual(empty['track_weight'].count_nonzero(),0)

    def test_invalid_positions_and_visibility_cannot_create_support(self):
        controls=fixture()
        controls['tracks'][:,:,0]=float('nan')
        controls['prefix_track_weight'].fill_(float('inf'))
        view=inference_control_view(controls)
        self.assertTrue(torch.isfinite(view['tracks']).all())
        self.assertEqual(view['track_weight'].count_nonzero(),0)
        self.assertEqual(view['prefix_track_weight'].count_nonzero(),0)
        self.assertEqual(view['motion'][...,:6].count_nonzero(),0)
        self.assert_view_equal(view,inference_control_view(view))

    def test_finite_extreme_coordinates_cannot_overflow_displacement(self):
        controls=fixture()
        controls['tracks'][0]=3e38
        controls['tracks'][1:]=-3e38
        controls['prefix_tracks'][:]=-3e38
        view=inference_control_view(controls)
        self.assertTrue(torch.isfinite(view['motion']).all())
        self.assertTrue((view['tracks']==-1).all())
        self.assertEqual(view['motion'][...,:6].count_nonzero(),0)
        self.assertTrue((view['motion'][...,6]==1).all())
        self.assertEqual(view['track_weight'].count_nonzero(),0)

    def test_batched_and_unbatched_are_identical_and_view_is_idempotent(self):
        controls=fixture();view=inference_control_view(controls)
        self.assert_view_equal(view,inference_control_view(view))
        batched={key:value.unsqueeze(0) if torch.is_tensor(value) else value for key,value in controls.items()}
        result=inference_control_view(batched)
        for key,value in view.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value,result[key][0],atol=0,rtol=0)

    def test_runtime_legacy_unchanged_new_view_once_per_bind_and_metrics_annotations(self):
        args=dict(channels=4,grid=2,width=24,depth=1,head_width=48,head_depth=1,
                  heads=4,text_dim=8,checkpoint_blocks=False)
        model=create_control_model(args,'reference')
        sample=dict(target=torch.ones(9,4,4),controls=fixture())
        batch=prepare_prefix(sample,torch.zeros(4),torch.ones(4))
        legacy=ControlRuntime('reference')
        torch.testing.assert_close(legacy.bind(model,batch,sample).controls,
            control_features(batch,sample,'reference'),atol=0,rtol=0)
        self.assertNotIn('control_input_policy',legacy.checkpoint_metadata())
        runtime=ControlRuntime('reference',control_input_policy=CONTROL_INPUT_POLICY)
        import train_scene_control as entry
        with patch.object(entry,'inference_control_view',wraps=inference_control_view) as transform:
            net=runtime.bind(model,batch,sample)
            for _ in range(2):
                net(torch.zeros_like(batch.target),torch.tensor([.5]),batch.anchor)
            self.assertEqual(transform.call_count,1)
        self.assertEqual(sample['controls']['motion'][0,0,0],77.)
        self.assertTrue((net.controls[...,12:14]==0).all())
        self.assertTrue((net.controls[...,14]==1).all())
        self.assertEqual(runtime.checkpoint_metadata()['control_input_policy'],CONTROL_INPUT_POLICY)
        metrics=runtime.training_metrics(model,net,torch.zeros_like(batch.target))
        self.assertEqual(metrics['control_dynamic_mean'],0)
        self.assertEqual(metrics['control_static_mean'],0)
        self.assertEqual(metrics['control_unknown_mean'],1)
        row={};runtime.annotate(row,sample)
        self.assertEqual(row['control_input_policy'],CONTROL_INPUT_POLICY)
        self.assertIn('binary',row['control_weight_semantics'])
        with self.assertRaisesRegex(ValueError,'Unsupported control input'):
            ControlRuntime('reference',control_input_policy='typo')

    def test_readiness_uses_actual_view_not_stored_teacher_motion(self):
        from utils.scene_control_readiness import assess_controls,summarize
        from utils.scene_run import ArtifactStore
        controls=fixture(); controls['motion'].zero_()
        embedded=summarize([assess_controls(controls,split=s) for s in ('train','eval')])
        manifest=dict(identity='test',samples=1,eval_cases=1,
            shards=[dict(file='train.pt')],control_readiness=embedded)
        with tempfile.TemporaryDirectory() as tmp:
            store=ArtifactStore(tmp,[])
            for name in ('train.pt','data/eval.pt'):
                store.save(name,dict(samples=[dict(id='case',controls=controls)]),'test')
            legacy=resolve_control_readiness(store,manifest,4)
            self.assertEqual(legacy['summary']['splits']['train']['motion_status'],'INVALID_ZERO_SUPPORT')
            new=resolve_control_readiness(store,manifest,4,control_input_policy=CONTROL_INPUT_POLICY)
            self.assertEqual(new['summary']['splits']['train']['motion_status'],'NONZERO_SUPPORT')
            self.assertEqual(new['control_input_policy'],CONTROL_INPUT_POLICY)

    def test_camera_arm_same_parameters_and_no_track_information(self):
        args=dict(channels=4,grid=2,width=24,depth=1,head_width=48,head_depth=1,
                  heads=4,text_dim=8,checkpoint_blocks=False)
        shapes=[]
        for arm in ('camera','motion','reference'):
            model=create_control_model(args,arm)
            shapes.append({key:value.shape for key,value in model.state_dict().items()})
        self.assertEqual(shapes[0],shapes[1]);self.assertEqual(shapes[0],shapes[2])
        model=create_control_model(args,'camera').eval()
        with torch.no_grad():
            model.control[-1].weight.normal_(0,.1)
            model.output[-1].weight.normal_(0,.1)
        runtime=ControlRuntime('camera',control_input_policy=CONTROL_INPUT_POLICY)
        item=dict(target=torch.ones(9,4,4),controls=fixture())
        batch=prepare_prefix(item,torch.zeros(4),torch.ones(4))
        first=runtime.bind(model,batch,item)
        changed=copy.deepcopy(item)
        changed['controls']['tracks']=object()  # Camera path must not read it.
        changed['controls']['prefix_track_weight']=object()
        changed['controls']['motion'].fill_(float('nan'))
        second=runtime.bind(model,batch,changed)
        torch.testing.assert_close(first.controls,second.controls,atol=0,rtol=0)
        expected=first.controls.clone();expected[...,:8]=0;expected[...,14]=0
        self.assertEqual(expected.count_nonzero(),0)
        self.assertTrue((first.controls[...,14]==1).all())
        noise=torch.randn_like(batch.target);u=torch.tensor([.5])
        torch.testing.assert_close(first(noise,u,batch.anchor),second(noise,u,batch.anchor),atol=0,rtol=0)
        before=model.control[-1].weight.detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        first(noise,u,batch.anchor).square().mean().backward();optimizer.step()
        self.assertFalse(torch.equal(before,model.control[-1].weight))
        changed['controls']['rays'][...,3]=.5
        self.assertFalse(torch.equal(first.controls,runtime.bind(model,batch,changed).controls))
        row={'supervision':'old'};runtime.annotate(row,changed)
        self.assertIn('camera rays only',row['condition_policy'])
        self.assertEqual(row['track_valid_fraction'],0)

    def test_camera_readiness_ignores_missing_tracks_but_requires_camera(self):
        from utils.scene_run import ArtifactStore
        controls=dict(schema='scene_control_v1',rays=fixture()['rays'])
        manifest=dict(identity='test',samples=1,eval_cases=1,shards=[dict(file='train.pt')])
        with tempfile.TemporaryDirectory() as tmp:
            store=ArtifactStore(tmp,[])
            for name in ('train.pt','data/eval.pt'):
                store.save(name,dict(samples=[dict(id='case',controls=controls)]),'test')
            evidence=resolve_control_readiness(store,manifest,4,
                control_input_policy=CONTROL_INPUT_POLICY,camera_only=True)
            decision=control_readiness_decision('camera',evidence)
            self.assertEqual(decision['required_conditions'],['camera'])
            self.assertEqual(decision['action'],'continue')
            self.assertEqual(control_readiness_decision('motion',evidence)['action'],'skip')
            absent=copy.deepcopy(evidence)
            absent['summary']['splits']['train']['future_camera_present_tokens']=0
            self.assertEqual(control_readiness_decision('camera',absent)['action'],'skip')

    def test_new_config_only_adds_explicit_policy_and_notes(self):
        root=Path(__file__).resolve().parents[1]
        old=json.loads((root/'configs/scene_control_reference_v1.json').read_text())
        new=json.loads((root/'configs/scene_control_reference_v2.json').read_text())
        self.assertNotIn('control_input_policy',old)
        self.assertEqual(new.pop('control_input_policy'),CONTROL_INPUT_POLICY)
        notes=new.pop('notes');old_notes=old.pop('notes')
        self.assertEqual(notes[:len(old_notes)],old_notes)
        self.assertEqual(old,new)

    def test_actual_tiny_training_with_v2_policy(self):
        import train_scene_prefix as trainer
        from tests.test_scene_control_train import controlled_fixture
        from tests.test_scene_prefix_train import new_store,TinyRAE,fake_preview,record_metrics,seed_all
        from utils.scene_run import StageBudget
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp);data=controlled_fixture(root)
            store,manifest,_,_=new_store(root,'v2',data,malformed=False)
            cfg=dict(data['cfg'],max_steps=1,source_label='source_fixture',eval_every_steps=1,
                control_input_policy=CONTROL_INPUT_POLICY)
            runtime=ControlRuntime('reference',control_input_policy=CONTROL_INPUT_POLICY)
            seed_all(42)
            with ExitStack() as stack:
                stack.enter_context(patch.object(trainer,'SceneRAE',TinyRAE))
                stack.enter_context(patch.object(trainer,'preview',fake_preview))
                stack.enter_context(patch.object(trainer,'metrics',record_metrics))
                source=data['source']
                trainer.train(SimpleNamespace(stage='train'),cfg,data['plan'],source,store,manifest,
                    data['selection'],source.local/'rae.pt',source.local/'weights.pt',source.local/'eval.pt',
                    StageBudget(10000),torch.device('cpu'),0,1,runtime=runtime)
            saved=store.load('train/checkpoint_latest.pt',data['plan']['identity'],True)
            self.assertGreater(saved['model']['control.2.weight'].abs().sum(),0)
            row=store.json('eval/step0000001/ema/case000_seed101.json',True)
            self.assertEqual(row['control_input_policy'],CONTROL_INPUT_POLICY)
            records=[json.loads(line) for line in (store.local/'metrics-r000.jsonl').read_text().splitlines()]
            train=[r for r in records if r['stage']=='train'][0]
            self.assertEqual(train['control_input_policy'],CONTROL_INPUT_POLICY)
            self.assertGreater(train['DI_throughput'],0)


if __name__=='__main__':
    unittest.main()
