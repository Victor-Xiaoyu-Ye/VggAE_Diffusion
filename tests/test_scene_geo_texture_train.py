"""Actual tiny optimizer/resume and shared-source preparation for Stage58."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import train_scene_geo_texture as entry
import train_scene_prefix as trainer
from tests.test_scene_motion_plan import fixture as plan_fixture
from tests.test_scene_prefix_train import (fixture, new_store, seed_all, TinyRAE,
    fake_preview, record_metrics)
from tests import test_scene_prefix_train as prefix_helpers
from models.scene_flow import SceneFlow
from utils.scene_geo_condition import GeoConditionRuntime
from utils.scene_geo_data import activity_score, choose_windows
from utils.scene_run import ArtifactStore, StageBudget


def execute(data, output, bank, manifest, arm, stage='train', steps=3, preview_fn=fake_preview,
            rank=0, world=1):
    cfg = dict(data['cfg'], max_steps=steps, source_label='source_fixture', convergence=None,
        save_every_steps=2, eval_every_steps=2, geo_condition_cases=[0], label_previews=True)
    runtime = GeoConditionRuntime(arm, data['plan']['mean'], data['plan']['std'])
    with ExitStack() as stack:
        stack.enter_context(patch.object(trainer, 'SceneRAE', TinyRAE))
        stack.enter_context(patch.object(trainer, 'preview', preview_fn))
        stack.enter_context(patch.object(trainer, 'metrics', record_metrics))
        source = data['source']
        trainer.train(SimpleNamespace(stage=stage), cfg, data['plan'], source, output,
            manifest, data['selection'], source.local/'rae.pt', source.local/'weights.pt',
            source.local/'eval.pt', StageBudget(10000), torch.device('cpu'), rank, world,
            data_store=bank, runtime=runtime)
    return output.load('train/checkpoint_latest.pt', data['plan']['identity'], True)


def two_eval(data):
    sample = data['source'].load('eval.pt', 'fixture-validation', True)['samples'][0]
    other = dict(data['samples'][1], id='spatialvid:other-heldout')
    data['source'].save('eval.pt', dict(samples=[sample, other]), 'fixture-validation')


def geo_fixture(root):
    data = fixture(root)
    data['plan']['model'] = dict(data['plan']['model'], channels=256)
    data['plan']['mean'] *= 64
    data['plan']['std'] *= 64
    model = SceneFlow(**data['plan']['model'])
    with torch.no_grad():
        model.output[-1].weight.normal_(0, .025)
    data['source'].save('weights.pt', dict(model=model.state_dict(),
        model_args=data['plan']['model'], step=71609), 'historical-source')
    codec = TinyRAE({'latent_grid': 2},256)
    data['source'].save('rae.pt',dict(model=codec.state_dict(),
        spec=dict(source_config={'latent_grid':2}, channels=256)), 'historical-ae')
    data['samples'] = [dict(s, target=s['target'].repeat(1,1,64), anchor=s['anchor'].repeat(1,1,64))
                       for s in data['samples']]
    data['source'].save('eval.pt',dict(samples=[dict(data['samples'][0],id='spatialvid:heldout')]),'fixture-validation')
    return data


class GeoTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_optimizer_exact_resume_and_condition_artifacts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root = Path(tmp); data = geo_fixture(root); two_eval(data)
            bank, manifest, _, _ = new_store(root, 'bank', data, malformed=False)
            continuous = ArtifactStore(root/'continuous', [])
            resumed = ArtifactStore(root/'resumed', [str(root/'primary'), str(root/'mirror')])
            seed_all(42); a = execute(data, continuous, bank, manifest, 'oracle_geo')
            seed_all(42); first = execute(data, resumed, bank, manifest, 'oracle_geo', stage='preflight')
            self.assertEqual(first['step'], 2)
            self.assertTrue(resumed.json('train/full_step0000002.pt.json'))
            b = execute(data, resumed, bank, manifest, 'oracle_geo')
            prefix_helpers.PrefixTrainingTests.assert_nested_equal(self, a['model'], b['model'])
            prefix_helpers.PrefixTrainingTests.assert_nested_equal(self, a['optimizer'], b['optimizer'])
            prefix_helpers.PrefixTrainingTests.assert_nested_equal(self, a['ema'], b['ema'])
            self.assertEqual(a['cursors'], b['cursors'])
            row = resumed.json('eval/step0000003/ema/case000_seed101.json', True)
            self.assertEqual(row['condition_diagnostics']['status'], 'completed')
            self.assertIn('future_texture_mse', row)
            self.assertGreater(row['condition_diagnostics']['variants']['zero']['rgb_change_from_correct_l1'], 0)
            self.assertNotEqual(row['condition_diagnostics']['donor_case_id'], row['id'])
            self.assertTrue(resumed.json('eval/step0000003/ema/case000_seed101_GEO_SHUFFLE.pt.json'))
            self.assertEqual(resumed.json('complete.json')['evaluation_status'], 'completed')
            lines = (resumed.local/'metrics-r000.jsonl').read_text().splitlines()
            train_rows = [json.loads(s) for s in lines if json.loads(s)['stage'] == 'train']
            self.assertTrue(all('texture_x0_mse' in r and 'DI_throughput' in r for r in train_rows))

    def assert_nested_equal(self, a, b):
        return prefix_helpers.PrefixTrainingTests.assert_nested_equal(self, a, b)

    def test_failed_intervention_media_repaired_without_extra_updates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root = Path(tmp); data = geo_fixture(root); two_eval(data)
            bank, manifest, _, _ = new_store(root, 'bank', data, malformed=False)
            out = ArtifactStore(root/'out', [])
            def fail_one(store, name, frames):
                return [] if 'GEO_ZERO' in name else fake_preview(store, name, frames)
            seed_all(42); a = execute(data, out, bank, manifest, 'oracle_geo', preview_fn=fail_one)
            self.assertEqual(out.json('complete.json')['evaluation_status'], 'partial')
            b = execute(data, out, bank, manifest, 'oracle_geo')
            self.assertEqual(a['cursors'], b['cursors'])
            self.assert_nested_equal(a['model'], b['model'])
            self.assertEqual(out.json('complete.json')['evaluation_status'], 'completed')

    def test_resume_wrong_arm_rejected_and_source_warm_is_matched(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root = Path(tmp); data = geo_fixture(root); two_eval(data)
            bank, manifest, _, _ = new_store(root, 'bank', data, malformed=False)
            outputs = []
            for arm in ('baseline', 'oracle_geo'):
                out = ArtifactStore(root/arm, [])
                seed_all(42); execute(data, out, bank, manifest, arm, stage='preflight')
                outputs.append(out.load('eval/source_fixture/online/case000_seed101.pt',
                    data['plan']['identity'], True)['latent'])
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                execute(data, ArtifactStore(root/'baseline', []), bank, manifest, 'oracle_geo')


class GeoPreparationTests(unittest.TestCase):
    def test_selection_deduplicates_and_is_order_independent(self):
        records = [dict(key=str(i), id='spatialvid:'+str(i), frame_ids=list(range(9)),
                        file='shard', index=i, activity=float(i)) for i in range(12)]
        a = choose_windows(records, 6, 42)
        b = choose_windows(list(reversed(records))+records[:2], 6, 42)
        self.assertEqual(a, b)
        self.assertEqual({r['index'] for r in a if r['selection_stratum']=='high_latent_change'}, {9,10,11})

    @patch.object(entry, 'code_signature', return_value='cpu-fixture-code')
    def test_prepare_resume_mirror_repair_split_filter_and_shared_plan(self, _):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root = Path(tmp)
            origin, codec, cfg, selection, previous, manifest = plan_fixture(root)
            cfg = dict(cfg, seed=42, convergence=None, max_steps=3, accum=1,
                dataset=dict(train_windows=3, candidate_windows=8, eval_cases=1, high_change_fraction=.5))
            def item(ident, offset, bad=False):
                z = torch.arange(9*4*4).reshape(9,4,4).float()*.01*(offset+1)
                if bad: z[-1] = float('nan')
                return dict(id=ident, dataset='spatialvid', target=z, anchor=z[:1].clone(),
                    frame_ids=list(range(offset, offset+9)), raw=torch.zeros(9,3,2,2,dtype=torch.uint8))
            samples = [item('spatialvid:a',0), item('spatialvid:a',10), item('spatialvid:b',20),
                       item('spatialvid:b',30,True), item('spatialvid:val',40)]
            origin.save('data/shards/one.pt', dict(samples=samples), previous['identity'])
            manifest['shards'][0]['sources'] = ['spatialvid']*len(samples)
            origin.write_json('data/cache_complete.json', manifest)
            origin.save('data/eval.pt', dict(samples=[item('spatialvid:val',0)]), previous['identity'])
            source = ArtifactStore(root/'private-source', [str(origin.local)])
            store = ArtifactStore(root/'subset', [str(root/'subset-primary'), str(root/'subset-mirror')])
            original_hash = hashlib.sha256((origin.local/'data/shards/one.pt').read_bytes()).hexdigest()
            ready = entry.prepare(source, codec, store, cfg, selection)
            self.assertEqual(ready['samples'], 3)
            self.assertEqual(ready['mean'], previous['mean'])
            self.assertFalse(any(r['id']=='spatialvid:val' for r in ready['selection']))
            self.assertEqual(original_hash, hashlib.sha256((origin.local/'data/shards/one.pt').read_bytes()).hexdigest())
            self.assertFalse((source.local/'data/shards/one.pt').exists())
            shard = ready['shards'][0]['file']
            (store.local/shard).write_bytes(b'broken')
            (root/'subset-primary'/shard).write_bytes(b'broken-primary')
            again = entry.prepare(source, codec, store, cfg, selection)
            self.assertEqual(ready, again)
            for arm in ('baseline','oracle_geo'):
                plan, got = entry.training_plan(source, codec, store, dict(cfg, arm=arm), selection,24)
                self.assertEqual(got, ready)
                self.assertEqual(plan['matched_windows'], 3)
                self.assertEqual(plan['data_identity'], ready['identity'])
            with self.assertRaisesRegex(ValueError, 'changed'):
                entry.prepare(source, codec, store, dict(cfg, seed=99), selection)


if __name__ == '__main__':
    unittest.main()
