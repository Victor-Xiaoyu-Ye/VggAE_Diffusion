"""CPU integration contracts; not real NPU/OBS/checkpoint certification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import audit_scene_end_to_end as audit
from utils.scene_run import ArtifactStore


class EndToEndAuditTest(unittest.TestCase):
    def config(self):
        return json.loads((audit.PROJECT/'configs/scene_end_to_end_audit_v1.json').read_text())

    def test_fixed_jobs_cover_original_failed_cases_once_across_three_nodes(self):
        jobs=audit.jobs_for(self.config())
        self.assertEqual(len(jobs),24)
        self.assertEqual(len({j['key'] for j in jobs}),24)
        owners=[set(j['key'] for j in jobs[r::3]) for r in range(3)]
        self.assertEqual(len(set.union(*owners)),24)
        self.assertFalse(owners[0]&owners[1])
        self.assertTrue({7,24,41,48}<={j['case'] for j in jobs if j['kind']=='generation'})

    def test_invalid_cases_not_silently_reindexed(self):
        for cases in ([0,0],[-1],[True],[]):
            cfg=self.config(); cfg['cases']=cases
            with self.assertRaises(ValueError): audit.jobs_for(cfg)

    def test_sealed_rgb_rejects_preview_or_floating_reinterpretation(self):
        raw=torch.randint(0,256,(9,3,4,4),dtype=torch.uint8)
        torch.testing.assert_close(audit.sealed_rgb({'raw':raw}),raw.float()/255)
        with self.assertRaises(ValueError): audit.sealed_rgb({'raw':raw.float()})

    def test_failed_component_does_not_prevent_next_then_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ArtifactStore(Path(tmp)/'local',[str(Path(tmp)/'a'),str(Path(tmp)/'b')])
            fail=dict(key='teacher/failed',kind='teacher',case=0)
            good=dict(key='generation/case000',kind='generation',case=0)
            def broken(): raise RuntimeError('synthetic failed head')
            r=audit.run_component(store,'identity',fail,broken,0)
            self.assertEqual(r['execution_status'],'failed')
            self.assertFalse((store.local/'teacher/failed/done.json').exists())
            completed=lambda: ({'functional_status':'zero_support'},[],13)
            r=audit.run_component(store,'identity',good,completed,0)
            self.assertEqual(r['execution_status'],'completed')
            r=audit.run_component(store,'identity',fail,completed,0)
            self.assertEqual(r['functional_status'],'zero_support')
            self.assertEqual(r['DI_unit'].split('_')[0],'diagnostic')

    def test_success_resume_repairs_primary_from_mirror_without_rerunning(self):
        with tempfile.TemporaryDirectory() as tmp:
            local=Path(tmp)/'local'; first=Path(tmp)/'primary'; second=Path(tmp)/'mirror'
            store=ArtifactStore(local,[str(first),str(second)])
            job=dict(key='generation/case000',kind='generation',case=0)
            def action():
                name=job['key']+'/data.bin'
                (local/name).parent.mkdir(parents=True,exist_ok=True)
                (local/name).write_bytes(b'valid bytes')
                return {'ok':True},[name],9
            audit.run_component(store,'identity',job,action,0)
            for root in (local,first): (root/job['key']/'data.bin').write_bytes(b'corrupt')
            with patch.object(audit,'heartbeat',side_effect=AssertionError('must not rerun')):
                audit.run_component(store,'identity',job,action,0)
            self.assertEqual((first/job['key']/'data.bin').read_bytes(),b'valid bytes')

    def test_identity_change_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ArtifactStore(tmp,[])
            audit.pin_plan(store,{'identity':'first'},0)
            with self.assertRaises(ValueError): audit.pin_plan(store,{'identity':'different'},0)

    def test_unrecoverable_old_case_recomputed_without_stopping_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ArtifactStore(tmp,[])
            job=dict(key='teacher/case000',kind='teacher',case=0)
            with patch.object(audit,'recover_case',side_effect=RuntimeError('all copies corrupt')):
                result=audit.run_component(store,'i',job,lambda: ({'ok':True},[],9),0)
            self.assertEqual(result['execution_status'],'completed')
            self.assertIn('corrupt',result['recovery_error_before_retry'])

    def test_zero_support_is_completed_execution_not_quality_success(self):
        job=dict(key='teacher/test',kind='teacher',case=0)
        plan=dict(identity='i',jobs=[job])
        report={'execution_status':'completed','readiness':{'future_track_positive':0}}
        s=audit.summarize_jobs(plan,{job['key']:report})
        self.assertEqual(s['execution_status'],'completed')
        self.assertFalse(s['training_authorized_by_audit'])
        self.assertIn('requires',s['quality_status'])
        report['lk_comparison']={'status':'failed'}
        self.assertEqual(audit.summarize_jobs(plan,{job['key']:report})['execution_status'],
                         'incomplete_or_failed_components')

    def test_missing_result_is_not_success(self):
        plan=dict(identity='i',jobs=[{'key':'missing'}])
        self.assertEqual(audit.summarize_jobs(plan,{})['failed_or_missing'],['missing'])

    def test_failed_lk_is_not_sealed_and_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ArtifactStore(tmp,[]); job=dict(key='teacher/test',kind='teacher',case=0)
            action=lambda: ({'lk_comparison':{'status':'failed','error':'cv2'}},[],13)
            result=audit.run_component(store,'i',job,action,0)
            self.assertEqual(result['execution_status'],'failed')
            self.assertFalse((store.local/'teacher/test/done.json').exists())
            result=audit.run_component(store,'i',job,lambda: ({'lk_comparison':{'status':'completed'}},[],13),0)
            self.assertEqual(result['execution_status'],'completed')

    def test_merge_writes_partial_summary_if_worker_contract_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ArtifactStore(tmp,[])
            plan=dict(identity='i',jobs=[dict(key='generation/test',kind='generation',case=0)])
            self.assertEqual(audit.merge(store,plan,3),2)
            summary=store.json('summary.json',True)
            self.assertEqual(len(summary['node_contract_errors']),3)
            self.assertEqual(summary['failed_or_missing'],['generation/test'])

    def test_malformed_report_still_leaves_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ArtifactStore(tmp,[])
            store.write_json('nodes/node0/contract.json',{'identity':'i'})
            store.write_json('generation/test/report.json',{'broken':True})
            plan=dict(identity='i',jobs=[dict(key='generation/test',kind='generation',case=0)])
            self.assertEqual(audit.merge(store,plan,1),2)
            self.assertEqual(store.json('summary.json')['failed_or_missing'],['generation/test'])

    def test_track_preview_accepts_planar_raw_storage(self):
        try:
            import cv2  # noqa:F401
        except ImportError:
            self.skipTest('OpenCV runtime not installed')
        with tempfile.TemporaryDirectory() as tmp:
            frames=torch.zeros(9,3,42,42)
            output=dict(tracks=torch.full((9,4,2),21.),visibility=torch.ones(9,4),
                        track_confidence=torch.ones(9,4))
            store=ArtifactStore(tmp,[])
            def save(path,rgb,fps):
                path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'mock-video')
                self.assertTrue(bool(rgb[:,32:].sum()))
            with patch('train_scene_pipeline.video',side_effect=save):
                name=audit.track_preview(store,'teacher/tiny',frames,output)
            self.assertTrue((Path(tmp)/name).exists())

    def transport_data(self):
        xy=torch.tensor([[.25,.25],[.75,.25],[.25,.75],[.75,.75]])
        target=torch.arange(16).reshape(1,4,4).float().expand(9,-1,-1).clone()
        sample=dict(target=target,id='toy',frame_ids=list(range(9)))
        controls=dict(tracks=xy[None].expand(9,-1,-1).clone(),track_weight=torch.ones(9,4),
            prefix_tracks=xy[None].expand(4,-1,-1).clone(),prefix_track_weight=torch.ones(4,4),
            motion=torch.zeros(9,4,8))
        return sample,controls

    def test_reference_requires_supported_pairs_and_scores_same_region(self):
        sample,controls=self.transport_data()
        result=audit.reference_probe(sample,controls,[0]*4,[1]*4)
        self.assertEqual(result['supported_tokens'],20)
        self.assertEqual(result['reference']['mse'],0)
        self.assertEqual(result['copy_last']['mse'],0)
        controls['prefix_track_weight'].zero_()
        result=audit.reference_probe(sample,controls,[0]*4,[1]*4)
        self.assertEqual(result['status'],'no_reference_support')
        self.assertIsNone(result['reference'])

    def test_reference_does_not_read_future_appearance(self):
        sample,controls=self.transport_data()
        from utils.scene_control import transport_prefix_texture as real_transport
        calls=[]
        def wrapped(observed,*args,**kwargs):
            calls.append(observed.clone())
            return real_transport(observed,*args,**kwargs)
        with patch('utils.scene_control.transport_prefix_texture',side_effect=wrapped):
            audit.reference_probe(sample,controls,[0]*4,[1]*4)
            altered=copy.deepcopy(sample); altered['target'][4:]+=999
            audit.reference_probe(altered,controls,[0]*4,[1]*4)
        torch.testing.assert_close(calls[0],calls[2])
        torch.testing.assert_close(calls[1],calls[3])

    def test_replay_mismatch_quantified_without_replacing_targets(self):
        sample=dict(raw=torch.zeros(9,3,2,2,dtype=torch.uint8),id='x',frame_ids=list(range(9)))
        def replay(s,row,localize,size):
            self.assertNotIn('raw',s)
            return torch.ones(9,3,2,2)/255,{'test':True}
        with patch('utils.scene_control_data.replay_rgb',side_effect=replay):
            report,files,frames=audit.replay_action(sample,{},lambda x:x)
        self.assertEqual(report['functional_status'],'rgb_replay_mismatch')
        self.assertEqual(report['mean_uint8_error'],1)
        self.assertFalse(bool(sample['raw'].any()))

    def test_generation_metrics_and_latent_survive_ffmpeg_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ArtifactStore(tmp,[])
            job=dict(key='generation/tiny',kind='generation',case=0)
            item=dict(id='tiny',frame_ids=list(range(9)))
            artifacts={key:torch.zeros(9,2,2,3) for key in
                ('preview','GT_FUTURE_GEO_ORACLE','GT_FUTURE_TEXTURE_ORACLE')}
            artifacts['generated_future_latent']=torch.zeros(5,4,4)
            plan=dict(identity='i',mean=[0]*4,std=[1]*4,config=dict(seed=101,sample_steps=2))
            with patch('utils.scene_generation_audit.audit_generation_case',return_value=(
                    {'generation':{'sentinel':123},'preview_columns':['a']},artifacts)), \
                 patch('train_scene_prefix.labelled_preview',side_effect=lambda x,*args:x), \
                 patch('train_scene_pipeline.video',side_effect=RuntimeError('ffmpeg failure')):
                report,files,frames=audit.generation_action(store,plan,job,item,item,None,None,'cpu')
            self.assertEqual(len(report['media_errors']),3)
            self.assertEqual(store.json(job['key']+'/core_metrics.json')['generation']['sentinel'],123)
            self.assertIsNotNone(store.load(job['key']+'/generated_latent.pt','i',True))

    def test_teacher_orchestration_preserves_nested_lk_error_and_portable_probe(self):
        from test_scene_control_data import teacher
        sample=dict(id='spatialvid:tiny',dataset='spatialvid',frame_ids=list(range(9)),
                    raw=torch.zeros(9,3,28,28,dtype=torch.uint8),target=torch.zeros(9,4,4))
        plan=dict(identity='i',model={'grid':2},mean=[0]*4,std=[1]*4,config={'lk_comparison':True})
        job=dict(key='teacher/tiny',kind='teacher',case=0,input='real')
        def probe(_teacher,frames,*args):
            return dict(report={'status':'completed'},outputs=teacher(len(frames)))
        def lk(frames,original):
            return original,dict(status='completed_with_tracking_errors',errors=[{'error':'synthetic cv2'}])
        with tempfile.TemporaryDirectory() as tmp:
            store=ArtifactStore(tmp,[])
            def preview(store,key,*args):
                name=key+'/mock.mp4'; path=store.local/name
                path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'mock')
                return name
            with patch('utils.scene_teacher_probe.probe_teacher',side_effect=probe), \
                 patch('utils.scene_lk_probe.probe_lk',side_effect=lk), \
                 patch.object(audit,'track_preview',side_effect=preview):
                report,files,frames=audit.teacher_action(store,plan,job,sample,None)
            self.assertEqual(report['lk_comparison']['status'],'failed')
            self.assertIn('inference_control_view_v2',report)
            self.assertEqual(report['inference_control_view_v2']['reference_probe']['reference']['mse'],0)
            self.assertIn(job['key']+'/core_metrics.json',files)


if __name__=='__main__': unittest.main()
