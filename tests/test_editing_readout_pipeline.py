"""New suite executes real tiny VACE, callbacks, videos, recovery and reporting.

Only production model/codec weights and evaluation-label loading are fixtures.
Both old imports and new extension/dose outputs pass the real artifact paths.
"""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import unittest

import imageio.v2 as imageio
import torch

import audit_edit_attribution as old
import audit_edit_readout_strength as run
from utils.scene_run import ArtifactStore
from utils.editing_data import file_hash
from tests import test_editing_attribution_pipeline as fixture_module
from tests.test_editing_state_feedback import Codec
from tests.test_vace_edit_state_pipeline import fixed_backbone

ROOT=Path(__file__).resolve().parents[1]


class ReadoutPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        fixture_module.AttributionPipelineTests.setUp(self)
        self.prior=self.out;self.old_cfg=self.cfg
        self.cfg=json.loads((ROOT/'configs/editing_readout_strength_v1.json').read_text())
        self.cfg.update({k:copy.deepcopy(self.old_cfg[k]) for k in run.PROTOCOL_KEYS})
        self.cfg.update(lpips=False,expected_sources=2,expected_subset=1)
        self.a.attribution=str(self.prior.local);self.a.attribution_root=self.prior.roots
        self.a.cohort='all';self.a.phase='prepare'
        rows=self.p.json('cohort.json')['records']
        rows.append(dict(id='extension',group='new-group',source_file='sources/extension.pt'))
        self.p.write_json('cohort.json',dict(identity='cohort-fixture',ablation_ids=['case'],records=rows))
        for kind in ('prepared','sources','native','text'):
            self.p.save(kind+'/extension.pt',self.p.load(kind+'/case.pt',self.pm['identity'],True),self.pm['identity'])
        old.prepare_vae(self.a,self.old_cfg,self.prior,self.device,0,1)
        for arm in old.OPERATOR_ARMS:
            self.a.arm=arm;old.render(self.a,self.old_cfg,self.prior,self.device,0,1)
            self.assertEqual(self.prior.json('phases/render_'+arm+'.json')['completed'],1)
        self.out=ArtifactStore(self.root/'new',[str(self.root/'new-primary'),str(self.root/'new-mirror')])
        self.stack.enter_context(patch.object(run,'validate_assets',return_value={'fixture':True}))
        self.stack.enter_context(patch.object(run,'load_backbone',side_effect=fixed_backbone))
        self.stack.enter_context(patch.object(run,'load_vae',side_effect=lambda *_:Codec()))
        run.index(self.a,self.cfg,self.out)
        self.new_m=self.out.json('manifest.json');self.new_identity=self.new_m['identity']

    def prepare_all(self):
        self.a.cohort='subset'
        for phase in ('prepare','prepare_vae'):
            self.a.phase=phase;run.prepare(self.a,self.cfg,self.out,self.device,0,1)
        # A tiny prepared RAE fixture for the new source. VAE extension is REAL full-video fit.
        cal=self.prior.load('calibration/case.pt',self.identity,True)
        self.out.save('calibration/extension.pt',cal,self.new_identity)
        self.a.cohort='extension';self.a.phase='prepare_vae'
        run.prepare(self.a,self.cfg,self.out,self.device,0,1)

    def test_import_all_operators_new_curves_extension_media_and_resume(self):
        self.prepare_all()
        old_hash=file_hash(self.prior.local/'render/op_rae/seed42/case/result.pt')
        for cohort,arms in [('subset',run.specifications(self.cfg)),('extension',old.OPERATOR_ARMS)]:
            self.a.cohort=cohort
            for arm in arms:
                self.a.arm=arm
                if cohort=='subset' and arm in old.OPERATOR_ARMS:
                    with patch.object(run,'load_backbone',side_effect=AssertionError('Must import original output')):
                        run.render(self.a,self.cfg,self.out,self.device,0,1)
                    metric=self.out.json(f'render/{arm}/seed42/case/metrics.json')
                    self.assertTrue(metric['imported'])
                    self.assertEqual(self.out.json(f'progress/render_subset_{arm}_r0.json')['DI_throughput'],0.)
                else:run.render(self.a,self.cfg,self.out,self.device,0,1)
            with patch('utils.editing_paper_data.load_evaluation_labels',return_value=(self.ref['source']*255).round().byte()):
                run.evaluate(self.a,self.cfg,self.out,self.device,0,1)
            if cohort=='subset':
                run.summarize(self.a,self.cfg,self.out)
                self.assertTrue(self.out.json('summary_subset.json')['complete'])
        self.a.cohort='all';run.summarize(self.a,self.cfg,self.out)
        summary=self.out.json('summary.json')
        self.assertTrue(summary['complete'],summary['failures']);self.assertEqual(summary['evaluated'],18)
        self.assertEqual(summary['imported'],5)
        imported=self.out.load('render/op_rae/seed42/case/result.pt',self.new_identity,True)
        prior=self.prior.load('render/op_rae/seed42/case/result.pt',self.identity,True)
        self.assertTrue(torch.equal(imported['generated'],prior['generated']))
        self.assertEqual(imported['generation_identity'],prior['generation_identity'])
        self.assertEqual(file_hash(self.prior.local/'render/op_rae/seed42/case/result.pt'),old_hash)
        movie=self.out.local/'render/op_rae_a015/seed42/case/comparison.mp4'
        with imageio.get_reader(str(movie)) as reader:
            self.assertEqual(reader.count_frames(),17);self.assertEqual(reader.get_data(0).shape,(24,120,3))
        # A missing primary replica is repaired from local/mirror without resampling.
        relative='render/op_rae_a015/seed42/case/comparison.mp4'
        (Path(self.out.roots[0])/relative).unlink()
        result=self.out.local/'render/op_rae_a015/seed42/case/result.pt';keep=file_hash(result)
        self.a.cohort='subset';self.a.arm='op_rae_a015'
        with patch.object(run,'load_backbone',side_effect=AssertionError('No repeat sampling')):
            run.render(self.a,self.cfg,self.out,self.device,0,1)
        self.assertEqual(file_hash(result),keep)
        self.a.cohort='all';run.summarize(self.a,self.cfg,self.out)
        self.assertEqual(file_hash(Path(self.out.roots[0])/relative),file_hash(self.out.local/relative))
        self.assertEqual(self.out.json('progress/render_subset_op_rae_a015_r0.json')['DI_throughput'],0.)
        # Fresh local disk and corrupt primary payload: recover from mirror, then repair primary.
        relative_result='render/op_rae_a015/seed42/case/result.pt'
        (Path(self.out.roots[0])/relative_result).write_bytes(b'broken primary')
        fresh=ArtifactStore(self.root/'fresh-node',self.out.roots)
        self.a.cohort='subset'
        with patch.object(run,'load_backbone',side_effect=AssertionError('Dual read must avoid generation')):
            run.render(self.a,self.cfg,fresh,self.device,0,1)
        self.assertEqual(file_hash(fresh.local/relative_result),keep)
        self.assertEqual(file_hash(Path(self.out.roots[0])/relative_result),keep)

    def test_missing_case_fails_phase_but_other_arms_run_and_are_retained(self):
        self.a.cohort='extension';self.a.arm='op_rae'
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):run.render(self.a,self.cfg,self.out,self.device,0,1)
        self.a.arm='op_rgb';run.render(self.a,self.cfg,self.out,self.device,0,1)
        self.assertEqual(self.out.json('phases/render_extension_op_rgb.json')['completed'],1)
        self.a.cohort='all'
        with self.assertRaisesRegex(RuntimeError,'incomplete'):run.summarize(self.a,self.cfg,self.out)
        self.assertTrue(self.out.json('summary.json')['failures'])

    def test_import_rejects_changed_generation_and_never_silently_resamples(self):
        folder='render/op_rgb/seed42/case';payload=self.prior.load(folder+'/result.pt',self.identity,True)
        payload['generation_identity']='wrong';self.prior.save(folder+'/result.pt',payload,self.identity)
        self.a.cohort='subset';self.a.arm='op_rgb'
        with patch.object(run,'load_backbone',side_effect=AssertionError('Do not hide mismatch')),self.assertRaisesRegex(RuntimeError,'Incomplete'):
            run.render(self.a,self.cfg,self.out,self.device,0,1)
        self.assertIn('identity',self.out.json('phases/render_subset_op_rgb.json')['records'][0]['error'])

    def test_smoke_never_imports_full_results_and_contract_detects_changes(self):
        smoke=dict(self.cfg,smoke=True)
        self.assertIsNone(run.import_output(self.a,smoke,self.out,self.new_m,{'id':'case'},'',None,{}))
        changed=copy.deepcopy(self.cfg);changed['rgb_smoothing_sigma']=3
        with self.assertRaisesRegex(ValueError,'contract changed'):run.checked(self.a,changed,self.out)
        with self.assertRaisesRegex(ValueError,'protocol changed'):run.check_parent(self.a,changed,self.out)

    def test_new_rae_calibration_and_temporal_expansion_then_cache_resume(self):
        self.a.cohort='extension';self.a.phase='prepare'
        original=self.prior.load('calibration/case.pt',self.identity,True)
        encoded=original['raw'][None]
        with patch.object(run,'load_teacher',return_value=torch.nn.Identity()),\
                patch.object(run,'load_encoder',return_value=torch.nn.Identity()),\
                patch('audit_edit_transport.encode_window',return_value=encoded),\
                patch('audit_edit_transport.rae_decoder',return_value=fixture_module.decoder):
            run.prepare(self.a,self.cfg,self.out,self.device,0,1)
        payload=self.out.load('calibration/extension.pt',self.new_identity,True)
        self.assertEqual(payload['operator_rae_effect'].shape,self.ref['source'].shape)
        self.assertEqual(payload['diagnostics']['fallback_count'],0)
        self.assertTrue(payload['source_reference_only'])
        with patch.object(run,'load_teacher',side_effect=AssertionError('Retain calibrated state')):
            run.prepare(self.a,self.cfg,self.out,self.device,0,1)
        self.assertEqual(self.out.json('progress/prepare_extension_r0.json')['DI_throughput'],0.)


if __name__=='__main__':unittest.main()
