"""Synthetic CPU diagnostic contracts; no real encoder/NPU certification."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import audit_scene_geometry_consistency as audit
from utils.scene_geometry_consistency import (
    difference, future_intervention, geometry_probe, latent_difference,
    observed_camera_gauge, representation_probe, single_scale_depth,
)
from utils.scene_run import ArtifactStore


def encoding_functions(leaky=False):
    def encode(rgb):
        pooled=F.adaptive_avg_pool2d(rgb, (2,2)).permute(0,2,3,1).reshape(len(rgb),4,3)
        context=pooled.mean(0,keepdim=True).expand_as(pooled) if leaky else pooled.cumsum(0)/torch.arange(1,len(rgb)+1)[:,None,None]
        geometry=pooled[...,:2]+.25*context[...,:2]
        texture=pooled[...,1:]
        z=torch.cat((geometry,texture),-1)
        return dict(latent=z,tokens={4:geometry.clone(),23:pooled.clone()})
    def decode(z):
        return F.interpolate(z[...,:3].reshape(len(z),2,2,3).permute(0,3,1,2),size=(4,4),mode='nearest').permute(0,2,3,1)
    return encode,decode


def representation_fixture(leaky=False):
    g=torch.Generator().manual_seed(58)
    raw=torch.randint(0,256,(9,3,4,4),generator=g,dtype=torch.uint8)
    frames=raw.float()/255
    encode,decode=encoding_functions(leaky)
    full=encode(frames)['latent'];anchor=encode(frames[:1])['latent']
    cached=torch.cat((anchor,full[1:]),0).bfloat16()
    sample=dict(id='case041',raw=raw,target=cached,anchor=anchor.bfloat16(),frame_ids=list(range(9)))
    return frames,sample,encode,decode


def geometry_fixture(moving_camera=False):
    t,h,w=3,64,64
    yy,xx=torch.meshgrid(torch.arange(12.,48.,6.),torch.arange(12.,48.,6.),indexing='ij')
    q=torch.stack((xx,yy),-1).reshape(-1,2)
    intrinsic=torch.tensor([[50.,0.,32.],[0.,50.,32.],[0.,0.,1.]])[None].repeat(t,1,1)
    extrinsic=torch.eye(3,4)[None].repeat(t,1,1)
    tracks=q[None].repeat(t,1,1)
    if moving_camera:
        extrinsic[:,0,3]=torch.arange(t)*.1
        tracks[:,:,0]+=torch.arange(t)[:,None]*2.5  # fx * translation / z
    full=dict(depth=torch.full((t,h,w),2.),depth_conf=torch.ones(t,h,w),
        extrinsics=extrinsic,intrinsics=intrinsic,query_points=q,tracks=tracks,
        visibility=torch.ones(t,len(q)),track_confidence=torch.ones(t,len(q)),
        image_size_hw=[h,w],reference_frame=0)
    prefix={k:(v[:2].clone() if torch.is_tensor(v) and k!='query_points' else v) for k,v in full.items()}
    return full,prefix


class GeometryConsistencyTests(unittest.TestCase):
    def test_future_intervention_keeps_prefix_and_does_not_mutate_input(self):
        frames,_,_,_=representation_fixture()
        before=frames.clone();changed=future_intervention(frames)
        torch.testing.assert_close(frames,before)
        self.assertTrue(torch.equal(changed[:4],frames[:4]))
        self.assertFalse(torch.equal(changed[4:],frames[4:]))

    def test_causal_common_prefix_and_separate_bfloat_cache_rounding(self):
        frames,sample,encode,decode=representation_fixture()
        before=sample['target'].clone();phases=[]
        report,artifacts=representation_probe(frames,sample,[0]*4,[1]*4,encode,decode,
                                             on_phase=lambda n,p,d:phases.append((n,p,d)))
        self.assertEqual(report['encoder_input_frames'],23)
        self.assertEqual([p[0] for p in phases],['full','prefix','future_intervened','anchor','representation_comparisons'])
        for name in ('full_vs_independent_prefix_raw','full_vs_changed_future_raw_prefix'):
            self.assertEqual(report['comparisons'][name]['joint']['raw']['rms'],0.)
        self.assertEqual(report['comparisons']['cached_vs_reencoded_bfloat16_storage']['joint']['raw']['rms'],0.)
        self.assertGreater(report['comparisons']['cached_vs_reencoded_full']['joint']['raw']['rms'],0.)
        torch.testing.assert_close(sample['target'],before)
        self.assertEqual(tuple(artifacts['cached_target'].shape),(9,4,4))
        self.assertTrue(report['observed_prefix_exactly_preserved'])
        json.dumps(report,allow_nan=False)

    def test_future_leak_measured_without_silent_cache_replacement_or_gate(self):
        frames,sample,encode,decode=representation_fixture(leaky=True)
        sample['target'][4:]+=1
        before=sample['target'].clone()
        report,artifacts=representation_probe(frames,sample,[0]*4,[1]*4,encode,decode)
        comparison=report['comparisons']['full_vs_changed_future_raw_prefix']
        self.assertGreater(comparison['geo']['raw']['rms'],0)
        self.assertEqual(comparison['texture']['raw']['rms'],0)
        self.assertGreater(report['common_prefix_tokens']['future_intervened']['4']['rms'],0)
        self.assertGreater(report['comparisons']['cached_vs_reencoded_full']['joint']['raw']['rms'],.1)
        torch.testing.assert_close(artifacts['cached_target'],before.float())
        self.assertEqual(report['quality_status'],'measured_not_certified')

    def test_comparison_evidence_survives_a_decoder_failure(self):
        frames,sample,encode,_=representation_fixture()
        saved=[]
        def broken_decode(z):raise RuntimeError('synthetic decoder failure')
        with self.assertRaisesRegex(RuntimeError,'decoder failure'):
            representation_probe(frames,sample,[0]*4,[1]*4,encode,broken_decode,
                on_phase=lambda name,payload,details:saved.append((name,payload)))
        self.assertEqual(saved[-1][0],'representation_comparisons')
        self.assertIn('cached_vs_reencoded_full',saved[-1][1]['comparisons'])
        self.assertEqual(tuple(saved[0][1]['latent'].shape),(9,4,4))

    def test_one_global_scale_removes_scale_but_not_spatial_shape_error(self):
        reference=np.arange(1.,25.).reshape(2,3,4)
        result,residual=single_scale_depth(reference,reference*2)
        self.assertEqual(result['single_candidate_to_reference_scale'],.5)
        self.assertEqual(result['aligned_rms'],0.)
        self.assertTrue(np.all(residual==0))
        candidate=reference*2;candidate[:,0]*=1.5
        result,_=single_scale_depth(reference,candidate)
        self.assertGreater(result['aligned_rms'],0.)
        self.assertLess(result['aligned_rms'],result['raw_rms'])
        empty,_=single_scale_depth(np.ones((2,2)),np.zeros((2,2)))
        self.assertEqual(empty['status'],'no_positive_depth_support')

    def test_camera_motion_does_not_make_static_planar_points_dynamic(self):
        full,prefix=geometry_fixture(moving_camera=True)
        report,compact=geometry_probe(full,prefix,full)
        self.assertGreater(report['per_frame'][2]['observed_track_displacement_px']['mean'],4.9)
        self.assertLess(report['per_frame'][2]['rigid_residual_on_lk_support_px']['maximum'],1e-4)
        self.assertEqual(report['motion_diagnostics']['dynamic_tracks'],0)
        self.assertGreater(report['motion_diagnostics']['static_tracks'],0)
        self.assertFalse(report['calibration_applied'])
        torch.testing.assert_close(compact['camera_observed_gauge'][:,0,3],torch.tensor([0.,-.05,-.1],dtype=torch.float64))

    def test_fixed_prefix_scale_does_not_read_future_depth(self):
        full,prefix=geometry_fixture()
        gauge,scale=observed_camera_gauge(full,prefix)
        altered=dict(full);altered['depth']=full['depth'].clone();altered['depth'][2:]*=100
        second,newscale=observed_camera_gauge(altered,prefix)
        self.assertEqual(scale,newscale)
        np.testing.assert_array_equal(gauge,second)

    def test_repeated_depth_scale_report_is_diagnostic_and_not_applied(self):
        full,prefix=geometry_fixture()
        full['depth'][1:]*=2
        prefix['depth']=full['depth'][:2].clone()
        before=full['depth'].clone()
        report,_=geometry_probe(full,prefix,full,repeated=True)
        fit=report['static_single_scale_depth']
        self.assertEqual(fit['single_candidate_to_reference_scale'],.5)
        self.assertEqual(fit['aligned_rms'],0.)
        self.assertGreater(fit['raw_rms'],0.)
        torch.testing.assert_close(full['depth'],before)
        json.dumps(report,allow_nan=False)

    def test_unknown_support_is_not_reported_as_zero_geometry_error(self):
        full,prefix=geometry_fixture()
        tracked=dict(full);tracked['visibility']=torch.zeros_like(full['visibility'])
        report,_=geometry_probe(full,prefix,tracked)
        self.assertIsNone(report['per_frame'][1]['rigid_residual_on_lk_support_px']['mean'])
        self.assertEqual(report['motion_diagnostics']['static_tracks'],0)
        self.assertEqual(report['motion_diagnostics']['unknown_tracks'],len(full['query_points']))

    def test_jobs_include_original_case041_and_are_disjoint_per_node(self):
        cfg={'cases':[0,4,7,11,24,36,41,48]}
        jobs=audit.jobs_for(cfg)
        self.assertEqual(len(jobs),24)
        self.assertEqual(sum(j['case']==41 for j in jobs),3)
        sets=[set(j['key'] for j in jobs[n::3]) for n in range(3)]
        self.assertEqual(len(set.union(*sets)),24)
        self.assertFalse(sets[0]&sets[1])
        with self.assertRaises(ValueError):audit.jobs_for({'cases':[0,0]})

    def test_cli_supports_old_output_and_explicit_new_rank(self):
        common=['--config','c.json','--source-local','source','--encoder','encoder','--stage','work']
        a=audit.parser().parse_args(common+['--local','out','--node-rank','2','--nodes','3'])
        self.assertEqual((a.output,a.node_rank,a.nodes),('out',2,3))
        self.assertEqual(audit.parser().parse_args(common+['--output','out']).output,'out')

    def test_checkpoint_signature_also_pins_production_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt';path.write_bytes(b'synthetic checkpoint contents')
            signature=audit.encoder_signature(path)
            cfg={'teacher_expected':{k:signature[k] for k in ('checkpoint_size','checkpoint_sample_sha256')}}
            audit.check_expected_encoder(signature,cfg)
            path.write_bytes(b'changed checkpoint contents')
            with self.assertRaisesRegex(ValueError,'Encoder checkpoint differs'):
                audit.check_expected_encoder(audit.encoder_signature(path),cfg)

    def test_phase_save_has_receipt_and_merge_does_not_quality_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ArtifactStore(Path(tmp)/'local',[str(Path(tmp)/'primary'),str(Path(tmp)/'mirror')])
            job=dict(key='representation/case041',kind='representation',case=41)
            plan=dict(identity='i',jobs=[job])
            store.write_json('nodes/node0/contract.json',dict(identity='i'))
            callback,files=audit._phase_callback(store,plan,job)
            callback('full',{'latent':torch.ones(9,4,4)},{'input_frames':9})
            self.assertIsNotNone(store.load('representation/case041/intermediate/full.pt','i',True))
            report=audit.previous.run_component(store,'i',job,
                lambda: ({'quality_status':'large_measured_mismatch'},files,9),0)
            self.assertEqual(report['execution_status'],'completed')
            self.assertEqual(audit.merge(store,plan,1),0)
            self.assertFalse(store.json('summary.json')['quality_gate_applied'])
            # A missing second job is execution incompleteness, and still leaves
            # an inspectable summary rather than discarding successful evidence.
            plan['jobs'].append(dict(key='geometry/missing',kind='geometry',case=0))
            self.assertEqual(audit.merge(store,plan,1),2)
            self.assertEqual(store.json('summary.json')['failed_or_missing'],['geometry/missing'])

    def test_geometry_component_uses_independent_lk_and_seals_intermediates(self):
        class Teacher:
            metadata={'precision':'float32','backend':'synthetic'}
            def infer(self,frames):
                base,_=geometry_fixture()
                for name in ('depth','depth_conf','extrinsics','intrinsics','tracks','visibility','track_confidence'):
                    base[name]=base[name][:1].repeat(len(frames),*[1]*(base[name].ndim-1))
                base['visibility'].zero_();base['track_confidence'].zero_()
                return base
        def lk(frames,original):
            output=dict(original)
            output['visibility']=torch.ones_like(original['visibility'])
            output['track_confidence']=torch.ones_like(original['track_confidence'])
            return output,dict(status='completed',errors=[],backend='synthetic_lk')
        def preview(store,key,*args):
            path=store.local/(key+'/mock.mp4');path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'synthetic preview')
            return key+'/mock.mp4'
        with tempfile.TemporaryDirectory() as tmp:
            store=ArtifactStore(tmp,[])
            job=dict(key='geometry/real/case041',kind='geometry',input='real',case=41)
            sample=dict(id='toy',frame_ids=list(range(9)),raw=torch.zeros(9,3,64,64,dtype=torch.uint8))
            plan={'identity':'i'}
            with patch('utils.scene_lk_probe.probe_lk',side_effect=lk),patch.object(audit.previous,'track_preview',side_effect=preview):
                report=audit.previous.run_component(store,'i',job,
                    lambda:audit.geometry_action(store,plan,job,sample,Teacher()),0)
            self.assertEqual(report['execution_status'],'completed')
            self.assertFalse(report['neural_tracks_used'])
            self.assertEqual(report['processed_frames'],26)
            self.assertGreater(report['motion_diagnostics']['static_tracks'],0)
            self.assertIsNotNone(store.load(job['key']+'/geometry.pt','i',True))
            self.assertIsNotNone(store.load(job['key']+'/intermediate/teacher_prefix.pt','i',True))
            self.assertTrue((store.local/job['key']/'done.json').exists())

    def test_numerical_shape_mismatch_is_explicit(self):
        with self.assertRaises(ValueError):difference(torch.ones(2,3),torch.ones(3,2))
        with self.assertRaises(ValueError):latent_difference(torch.ones(2,3,5),torch.ones(2,3,5),[0]*5,[1]*5)


if __name__=='__main__':
    unittest.main()
