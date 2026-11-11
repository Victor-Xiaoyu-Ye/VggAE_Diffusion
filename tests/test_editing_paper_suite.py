"""Source/label firewall, actual tiny VACE/SFT/solver paths and media recovery.

Unavailable production codecs/backbones use the explicit Stage65 fixtures.
No test result is an Ascend performance or production quality claim.
"""
import contextlib
import copy
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

import paper_editing_suite as run
from utils.editing_paper_data import candidate_rows, freeze_cohort, load_source, load_evaluation_labels, TrainingCacheView
from utils.editing_paper_controls import intervened_inputs, intervened_state, MAIN_ARMS, ABLATION_ARMS
from utils.editing_paper_metrics import aggregate, evaluate_videos, optional_lpips
from utils.editing_paper_report import metric_tables, summarize
from utils.editing_data import file_hash, rae_canvas
from utils.scene_run import ArtifactStore
import tests.test_edit_state_feedback_pipeline as feedback_fixture
from tests.test_vace_edit_state_pipeline import fixed_backbone, TinyCodec, TinyRAE
from audit_edit_transport import rae_decoder

REPO=Path(__file__).resolve().parents[1]
SCRATCH=REPO.parent/'VggAE_DataAudit'/'paper_suite_validation'


class PaperContractTests(unittest.TestCase):
    def test_only_reviewed_exact_hash_recovery_preserves_original_identity(self):
        migration=json.loads((REPO/'configs/editing_paper_resume_compat.json').read_text())
        current=run.code_signature()
        for name,sha in migration['after'].items():self.assertEqual(current[name],sha)
        SCRATCH.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            out=ArtifactStore(directory,[]);cfg={'cases':123}
            prior=dict(current,**migration['before'])
            manifest=dict(code=prior,config=cfg,identity='original-frozen-contract')
            out.write_json('manifest.json',manifest)
            self.assertEqual(run.checked(None,cfg,out),manifest)
            self.assertEqual(out.json('manifest.json'),manifest)
            self.assertTrue(list((out.local/'recovery').glob('compat_*.json')))
            with self.assertRaisesRegex(ValueError,'code/config changed'):run.checked(None,{'cases':122},out)
            for name in ('paper_editing_suite.py','utils/editing_joint_state.py','utils/vace_edit_runtime.py'):
                changed=dict(current,**{name:'unreviewed-edit'})
                with self.subTest(name=name),patch.object(run,'code_signature',return_value=changed),self.assertRaisesRegex(ValueError,'code/config changed'):
                    run.checked(None,cfg,out)

    def test_repair_restores_only_data_and_does_not_reselect_or_train(self):
        with patch.object(run,'checked'),patch('prepare_vace_edit.stage_data') as staged,\
             patch.object(run,'index',side_effect=AssertionError('Do not reselect')),\
             patch.object(run,'sft',side_effect=AssertionError('Do not retrain')):
            a=SimpleNamespace(source='remote-pilot',data='local-pilot')
            run.repair_inputs(a,{},None);staged.assert_called_once_with(a.source,a.data)
            staged.reset_mock();run.repair_inputs(a,{'skip_staging':True},None);staged.assert_not_called()

    def test_real_cli_phase_dispatch_including_text(self):
        import sys
        SCRATCH.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            config=Path(directory)/'config.json';config.write_text('{}')
            for phase,function in [('text','texts')]+[(p,p) for p in ('screen','sources','anchors','prepare','cache','sft','render','evaluate')]:
                with self.subTest(phase=phase),patch.object(run,function) as called,\
                     patch.object(run,'distributed',return_value=(torch.device('cpu'),0,1)),\
                     patch.object(sys,'argv',['paper_editing_suite.py','--phase',phase,'--config',str(config),'--output',directory]):
                    run.main();called.assert_called_once()

    def test_human_scores_require_real_votes_and_hide_neither_from_win_rate(self):
        from scripts.score_editing_paper_human import score
        key=dict(trials=[dict(trial='a',id='x',group='g',a='rgb_full',b='rae_full')])
        vote=dict(rater='r1',ratings=[dict(trial='a',ratings=dict(instruction='B',temporal_quality='neither'))])
        result=score(key,[vote],bootstrap=20)
        self.assertEqual(result['results']['rgb_full']['instruction']['rae_preference'],1.)
        self.assertNotIn('rae_preference',result['results']['rgb_full']['temporal_quality'])
        self.assertEqual(score(key,[])['status'],'no_ratings')
        with self.assertRaisesRegex(ValueError,'Duplicate'):score(key,[vote,vote])

    def test_reserved_byte_groups_and_input_only_visual_screen(self):
        row=lambda i,split,group:dict(id=i,split=split,group=group,source=i+'.mp4',target='POISON',instruction='edit')
        bank=dict(records=[],manifest=dict(records=[row('train','train','a'),row('dev','dev','b'),
            row('leak','test','a'),row('test1','test','c'),row('test2','test','c'),row('test3','test','d')]))
        rows,_=candidate_rows(bank);self.assertEqual(len(rows),2)
        self.assertTrue(all('target' not in r for r in rows))
        fingerprint=lambda h:dict(phash=[h]*3,texture_std=[12.]*3)
        reports=[dict(id=r['id'],status='completed',file=r['id']+'.pt',fingerprint=fingerprint('f'*16 if r['group']=='d' else '0'*16)) for r in rows]
        exposed=[dict(id='seen',status='completed',group='x',fingerprint=fingerprint('0'*16))]
        selected=freeze_cohort(rows,reports,exposed,128)
        self.assertEqual([r['group'] for r in selected['records']],['d'])
        self.assertEqual(selected['requested'],128);self.assertEqual(selected['selected'],1)
        self.assertIn('not a scene-disjoint guarantee',selected['caveat'])
        self.assertEqual(candidate_rows(bank,smoke=True)[0][0]['id'],'dev')

    def test_state_graph_controls_are_specific_and_nonmutating(self):
        state=torch.randn(514,6,2,2);original=state.clone()
        zero=intervened_state(state,'rae_zero_edit')
        torch.testing.assert_close(zero[:256],state[:256]);torch.testing.assert_close(zero[:,0],state[:,0])
        torch.testing.assert_close(zero[-2:],state[-2:]);self.assertEqual(float(zero[256:512,1:].abs().sum()),0.)
        torch.testing.assert_close(state,original)
        inputs=dict(edges=[dict(flow=torch.zeros(8,8,2),confidence=torch.ones(1,8,8)*.2)],
                    confidence=torch.ones(2,1,8,8)*.2,full_edge_confidence=torch.ones(1,1,8,8)*.2)
        changed,options=intervened_inputs(inputs,dict(joint_weight=1),'rae_shuffle_graph')
        self.assertGreater(float(changed['edges'][0]['flow'].abs().sum()),0)
        self.assertEqual(float(inputs['edges'][0]['flow'].abs().sum()),0)
        value,options=intervened_inputs(inputs,dict(joint_weight=1),'rae_no_temporal')
        self.assertEqual(len(value['edges']),1);self.assertEqual(options['joint_weight'],0)
        value,options=intervened_inputs(inputs,dict(joint_weight=1),'rae_no_graph');self.assertFalse(value['edges'])
        self.assertEqual(intervened_inputs(inputs,{},'rae_reinject')[1]['iterations'],0)

    def test_paired_statistics_match_seeds_not_frame_counts(self):
        records=[dict(status='completed',group='g',id='x',arm=a,seed_base=s,metrics=dict(target_l1=v))
                 for a,s,v in [('rae_full',42,2.),('rae_full',43,100.),('rgb_full',42,3.)]]
        report=aggregate(records,['rgb_full','rae_full'],bootstrap=50)
        self.assertEqual(report['paired']['rgb_full']['target_l1']['rae_minus_control'],-1)
        self.assertEqual(report['paired']['rgb_full']['target_l1']['paired_source_groups'],1)
        tables=metric_tables(records,dict(seed=42,main_arms=['rgb_full','rae_full'],ablation_arms=[],second_seed_arms=['rae_full','rgb_full']),dict(ablation_ids=['x']))
        self.assertEqual(tables['main']['means']['rae_full']['target_l1']['mean'],2)

    def test_independent_metrics_expose_copy_tradeoff(self):
        torch.manual_seed(42);source=torch.rand(3,3,16,32)*.3;target=source.clone();target[:,2,4:12,8:24]+=.4
        perfect=evaluate_videos(source,target,target);copy_result=evaluate_videos(source,target,source)
        self.assertAlmostEqual(perfect['target_ssim'],1.,places=5)
        self.assertEqual(perfect['dataset_keep_source_l1'],0.)
        self.assertGreater(copy_result['dataset_edit_l1'],0.)
        self.assertIsNone(perfect['target_lpips'])
        self.assertEqual(optional_lpips('cpu',False)[1]['status'],'disabled')

    def test_source_cache_translation_cannot_reuse_checkpoints(self):
        from unittest.mock import Mock
        out,old=Mock(),Mock();view=TrainingCacheView(out,old,'old','new')
        view.load('native/x.pt','new',True);old.load.assert_called_once_with('native/x.pt','old',True)
        view.load('arms/sft/selected.pt','new',True);out.load.assert_called_once()
        with self.assertRaises(ValueError):view.load('native/x.pt','old')


class PaperPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);SCRATCH.mkdir(parents=True,exist_ok=True)

    def test_full_source_to_sft_all_controls_evaluation_and_dual_recovery(self):
        # Reuse the independently exercised old Stage65 fixture, not its tests.
        feedback_fixture.FeedbackPipelineTests.setUpClass()
        harness=feedback_fixture.FeedbackPipelineTests(methodName='test_missing_historical_payload_regenerates_in_new_output')
        harness.setUp();self.addCleanup(harness.doCleanups)
        _,a,_,old=harness.old_run(sample_steps=3)
        root=harness.root
        a.bank=str(root/'stage65/bank');a.bank_root=[];a.data=str(root/'raw');a.source='fixture-no-network'
        raw=Path(a.data);raw.mkdir();bank_store=ArtifactStore(a.bank,[]);bank=bank_store.json('bank.json',True)
        bank['config']=dict(height=16,width=32,max_side=32,frames=17,fps=8,negative_prompt='bad',benchmark_fps=8)
        import imageio.v2 as imageio
        rows=[]
        for k,part in enumerate(('train','monitor','dev','test')):
            name=part+'-0';rng=np.random.default_rng(42+k)
            source=rng.integers(20,170,(17,16,32,3),dtype=np.uint8)
            target=source.copy();target[:,4:12,8:24,2]=220
            for label,frames in (('source',source),('target',target)):
                imageio.mimwrite(str(raw/(name+'_'+label+'.mp4')),list(frames),fps=8,macro_block_size=1)
            row=dict(id=name,group=file_hash(raw/(name+'_source.mp4')),source=name+'_source.mp4',
                target=name+'_target.mp4',split='train' if part=='monitor' else part,instruction='Make the center blue')
            rows.append(row)
            for cached in bank['records']:
                if cached['id']==name:cached.update({**row,'file':cached['file']})
        bank['manifest']=dict(records=rows);bank_store.write_json('bank.json',bank)
        cfg=json.loads((REPO/'configs/editing_paper_evidence_v1.json').read_text())
        cfg.update(cases=1,skip_staging=True,sample_steps=3,lpips=False)
        cfg['feedback']['solver'].update(sweeps=1,steps=1)
        cfg['anchor']['support_dilation']=1
        out=ArtifactStore(root/'paper',[str(root/'paper-primary'),str(root/'paper-mirror')])
        stack=harness.stack
        stack.enter_context(patch.object(run,'validate_assets',return_value={'fixture':'native'}))
        stack.enter_context(patch.object(run,'load_backbone',side_effect=fixed_backbone))
        stack.enter_context(patch.object(run,'load_vae',side_effect=lambda *_:TinyCodec()))
        stack.enter_context(patch.object(run,'load_teacher',side_effect=lambda *_:TinyRAE()))
        stack.enter_context(patch.object(run,'load_encoder',return_value=nn.Identity()))
        stack.enter_context(patch.object(run,'load_text',return_value=lambda texts,device:[torch.linspace(-.3,.5,48).reshape(4,12) for _ in texts]))
        stack.enter_context(patch('utils.edit_transport_data.rae_canvas',side_effect=lambda frames,rect:rae_canvas(frames,rect,size=32)))
        stack.enter_context(patch('utils.editing_state_feedback.load_decoder_only',side_effect=lambda *_,**__:rae_decoder(TinyRAE())))
        device=torch.device('cpu');run.index(a,cfg,out)
        self.assertTrue(all('target' not in r for r in out.json('manifest.json')['candidates']))
        # Deleting target before conditioning is stronger than a source signature check.
        target_path=raw/'test-0_target.mp4';saved=target_path.read_bytes();target_path.unlink()
        for phase in (run.screen,run.sources):phase(a,cfg,out,device,0,1)
        run.select(a,cfg,out)
        self.assertEqual(out.json('cohort.json')['selected'],1)
        for phase in (run.texts,run.anchors,run.prepare,run.cache):
            phase(a,cfg,out,device,0,1)
            phase_name='text' if phase==run.texts else phase.__name__
            report=out.json('phases/'+phase_name+'.json')
            self.assertEqual(report['completed'],1,report)
        run.sft(a,cfg,out,device,0,1);run.weights(a,cfg,out)
        sft=run.arm_store(out,'sft')
        self.assertEqual(sft.json('training_summary.json')['updates'],2)
        self.assertIn('no state bridge',sft.json('run.json')['architecture'])
        contract=out.json('sft_contract.json')
        trained=sft.load('selected.pt',run.digest([contract['identity'],'sft']),True)
        self.assertTrue(all(n.startswith('backbone.') and n.endswith(('.a','.b')) for n in trained['model']),list(trained['model']))
        m=out.json('manifest.json');outputs={}
        for arm in MAIN_ARMS+ABLATION_ARMS:
            a.arm=arm;run.render(a,cfg,out,device,0,1)
            report=out.json('phases/render_'+arm+'.json')
            self.assertEqual(report['completed'],report['requested'],report)
            outputs[arm]=out.load(f'render/{arm}/seed42/test-0/result.pt',m['identity'],True)
        self.assertEqual(len({p['input_identity'] for p in outputs.values()}),1)
        for arm in ('rae_full','rgb_full','rae_bridge_only','rae_velocity_only'):
            applied=[e for e in outputs[arm]['events'] if e['applied']]
            self.assertTrue(applied,outputs[arm]['events'])
            if arm=='rae_bridge_only':self.assertTrue(all(e['clean_delta_rms']==0 for e in applied))
            if arm=='rae_velocity_only':self.assertTrue(all(e['state_delta_rms']==0 for e in applied))
        self.assertGreater(float((outputs['rae_full']['generated']-outputs['rae_base']['generated']).abs().sum()),0.)
        self.assertGreater(float((outputs['rae_base']['generated']-outputs['rae_no_state']['generated']).abs().sum()),0.)
        # A repair reruns only the missing seed, retaining the other seed's
        # receipt and payload bytes, without repeating SFT or completed samples.
        phase_name='phases/render_rae_full.json';prior=out.json(phase_name)
        retained=next(r for r in prior['records'] if r['seed_base']==42)
        completed_file=out.local/'render/rae_full/seed42/test-0/result.pt'
        completed_sha=file_hash(completed_file)
        missing_folder='render/rae_full/seed43/test-0'
        for root_path in [out.local,*map(Path,out.roots)]:
            for name in ('result.pt','result.pt.json','receipt.json','generated_frames.npz','comparison.mp4'):
                (root_path/missing_folder/name).unlink(missing_ok=True)
        prior['records']=[retained,dict(id='test-0',seed_base=43,status='failed',error='interrupted')]
        prior['completed']=1;out.write_json(phase_name,prior)
        a.arm='rae_full';a.missing_only=True
        recover=run.recovered_case
        def missing_case_only(store,folder,identity):
            self.assertEqual(folder,missing_folder)
            return recover(store,folder,identity)
        with patch.object(run,'recovered_case',side_effect=missing_case_only) as recovered:
            run.render(a,cfg,out,device,0,1)
            self.assertEqual(recovered.call_count,1)
        self.assertEqual(file_hash(completed_file),completed_sha)
        repaired=out.json(phase_name)
        self.assertEqual(repaired['completed'],2,repaired)
        self.assertEqual(next(r for r in repaired['records'] if r['seed_base']==42),retained)
        self.assertEqual(sft.json('training_summary.json')['updates'],2)
        with patch.object(run,'load_backbone',side_effect=AssertionError('All cases complete')),\
             patch.object(run,'recovered_case',side_effect=AssertionError('Do not load completed tensors')):
            run.render(a,cfg,out,device,0,1)
        a.missing_only=False
        target_path.write_bytes(saved)
        run.evaluate(a,cfg,out,device,0,1)
        report=out.json('phases/evaluate.json');self.assertEqual(report['completed'],report['requested'],report)
        summarize(a,cfg,out);self.assertEqual(out.json('summary.json')['status'],'completed')
        public=(out.local/'human/index.html').read_text(encoding='utf8')
        self.assertNotIn('rae_full',public);self.assertEqual(out.json('human_private/key.json')['trials'][0]['id'],'test-0')
        movie='render/rae_full/seed42/test-0/comparison.mp4';npz='render/rae_full/seed42/test-0/generated_frames.npz'
        (out.local/movie).unlink();(Path(out.roots[0])/movie).unlink();(Path(out.roots[1])/npz).unlink()
        with patch.object(run,'load_backbone',side_effect=AssertionError('Do not resample complete cases')):
            a.arm='rae_full';run.render(a,cfg,out,device,0,1);summarize(a,cfg,out)
        for name in (movie,npz):
            self.assertEqual(len({file_hash(Path(r)/name) for r in [out.local,*out.roots]}),1)
        # An absent arm is reported, while measured tables and successful media survive.
        out.write_json('phases/render_native_reference_60.json',dict(records=[],requested=1,completed=0))
        with self.assertRaisesRegex(RuntimeError,'incomplete'):summarize(a,cfg,out)
        self.assertEqual(out.json('summary.json')['coverage']['native_reference_60']['completed'],0)


if __name__=='__main__':unittest.main()
