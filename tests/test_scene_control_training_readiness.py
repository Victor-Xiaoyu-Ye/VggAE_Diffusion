"""Condition-presence decisions and collective startup, not quality gates."""
from datetime import timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

import train_scene_control as entry
from tests.test_scene_control_readiness import controls as make_controls, sample
from train_scene_control import (control_readiness_decision, control_startup,
                                 resolve_control_readiness)
from utils.scene_control_readiness import assess_controls, summarize
from utils.scene_run import ArtifactStore


def manifest_with_report(train, evaluation=None):
    evaluation = train if evaluation is None else evaluation
    records = [assess_controls(value, split='train') for value in train]
    records += [assess_controls(value, split='eval') for value in evaluation]
    return dict(identity='controls-id', samples=len(train), eval_cases=len(evaluation),
                shards=[dict(file='data/prepared/job00000.pt')],
                control_readiness=summarize(records))


def config(arm):
    return dict(arm=arm, history=2, source_label='source_fixture')


def ddp_readiness_worker(rank, folder, fail_write):
    root = Path(folder)
    dist.init_process_group('gloo', init_method=(root/'rendezvous').as_uri(),
        rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        store = ArtifactStore(root/'output', [])
        controls = ArtifactStore(root/'controls', [])
        manifest = manifest_with_report([make_controls(False)])
        result = None
        try:
            if fail_write and rank == 0:
                with patch.object(store, 'write_json', side_effect=RuntimeError('test publication failure')):
                    result = control_startup(controls, manifest, store, dict(identity='run-id'),
                        config('reference'), torch.device('cpu'), rank)
            else:
                result = control_startup(controls, manifest, store, dict(identity='run-id'),
                    config('reference'), torch.device('cpu'), rank)
        except RuntimeError as exc:
            result = dict(error=str(exc))
        (root/f'rank{rank}.json').write_text(json.dumps(result), encoding='utf8')
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TrainingReadinessTests(unittest.TestCase):
    def evidence(self, manifest):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(tmp, [])
            with patch.object(store, 'load', side_effect=AssertionError('Manifest needs no tensor download')):
                return resolve_control_readiness(store, manifest, 2)

    def test_complete_empty_conditions_skip_controlled_arms_but_not_baseline(self):
        evidence = self.evidence(manifest_with_report([make_controls(False)]))
        for arm in ('motion', 'reference'):
            decision = control_readiness_decision(arm, evidence)
            self.assertEqual(decision['action'], 'skip')
            self.assertEqual(decision['reasons'][0]['status'], 'INVALID_ZERO_SUPPORT')
            self.assertEqual(decision['camera_status'], 'NONZERO_SUPPORT')
        self.assertEqual(control_readiness_decision('baseline', evidence)['action'], 'continue')

    def test_one_empty_window_does_not_block_nonzero_training_signal(self):
        evidence = self.evidence(manifest_with_report([make_controls(False), make_controls()]))
        self.assertEqual(control_readiness_decision('reference', evidence)['action'], 'continue')
        self.assertEqual(evidence['summary']['splits']['train']['zero_motion_windows'], 1)

    def test_missing_reference_does_not_block_motion_arm(self):
        value = make_controls()
        value['prefix_track_weight'].zero_()
        evidence = self.evidence(manifest_with_report([value]))
        self.assertEqual(control_readiness_decision('motion', evidence)['action'], 'continue')
        decision = control_readiness_decision('reference', evidence)
        self.assertEqual(decision['action'], 'skip')
        self.assertEqual([r['condition'] for r in decision['reasons']], ['reference'])

    def test_eval_absence_is_disclosed_without_blocking_valid_training(self):
        evidence = self.evidence(manifest_with_report([make_controls()], [make_controls(False)]))
        decision = control_readiness_decision('reference', evidence)
        self.assertEqual(decision['action'], 'continue')
        self.assertEqual(len(decision['warnings']), 2)
        self.assertTrue(all(w['split'] == 'eval' for w in decision['warnings']))

    def test_legacy_bounded_zero_is_not_claimed_as_complete_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(tmp, [])
            shards = [dict(file=f'data/prepared/job{i:05d}.pt') for i in range(6)]
            for i, shard in enumerate(shards):
                store.save(shard['file'], dict(samples=[sample(make_controls(i == 2))]), 'controls-id')
            store.save('data/eval.pt', dict(samples=[sample(make_controls())]), 'controls-id')
            manifest = dict(identity='controls-id', samples=6, eval_cases=1, shards=shards)
            bounded = resolve_control_readiness(store, manifest, 2)
            self.assertEqual(bounded['coverage']['train']['shard_indices'], [0, 1, 3, 5])
            self.assertFalse(bounded['coverage']['train']['complete'])
            decision = control_readiness_decision('motion', bounded)
            self.assertEqual(decision['action'], 'skip')
            self.assertEqual(decision['reasons'][0]['status'], 'UNVERIFIED_ZERO_IN_BOUNDED_SCAN')
            complete = resolve_control_readiness(store, manifest, 2, train_shard_limit=6)
            self.assertTrue(complete['coverage']['train']['complete'])
            self.assertEqual(control_readiness_decision('reference', complete)['action'], 'continue')

    def test_invalid_controlled_skip_is_published_and_resume_identity_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root/'run', [str(root/'mirror')])
            bank = ArtifactStore(root/'controls', [])
            plan = dict(identity='original-run-id', unchanged=True)
            store.write_json('contract.json', plan)
            before = (store.local/'contract.json').read_bytes()
            report = control_startup(bank, manifest_with_report([make_controls(False)]),
                store, plan, config('reference'), torch.device('cpu'), 0)
            self.assertEqual(report['action'], 'skip')
            self.assertEqual((store.local/'contract.json').read_bytes(), before)
            self.assertEqual(store.json('skipped_invalid_controls.json')['identity'], plan['identity'])
            self.assertTrue((root/'mirror/skipped_invalid_controls.json').is_file())
            self.assertIsNone(store.json('complete.json'))

    def test_baseline_continues_without_scanning_or_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(Path(tmp)/'run', [])
            bank = ArtifactStore(Path(tmp)/'controls', [])
            with patch.object(bank, 'load', side_effect=AssertionError('Baseline must not need controls')):
                report = control_startup(bank, dict(identity='controls-id'), store,
                    dict(identity='run-id'), config('baseline'), torch.device('cpu'), 0)
            self.assertEqual(report['action'], 'continue')
            self.assertIsNone(store.json('skipped_invalid_controls.json'))
            self.assertEqual(store.json('contract.json')['identity'], 'run-id')

    def test_main_skips_invalid_arm_before_staging_but_runs_baseline(self):
        for arm in ('reference', 'baseline'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg = dict(config(arm), history=4, views=9, seed=42, hours=0,
                    codec_namespace='codec', condition_policy='future_camera_tracks_with_observed_only_appearance')
                path = root/'config.json'
                path.write_text(json.dumps(cfg), encoding='utf8')
                args = SimpleNamespace(config=str(path), cohort='unused', selection='unused',
                    source_local=str(root/'source'), source_root=[], controls_local=str(root/'controls'),
                    controls_root=[], output=str(root/'output'), root=[], startup_seconds=0,
                    arm=arm, stage='train')
                prepared = manifest_with_report([make_controls(False)])
                plan = dict(identity='run-id', ae={}, evaluation={}, source_weights={})
                with patch.object(entry, 'parser') as parser, \
                     patch.object(entry, 'distributed', return_value=(torch.device('cpu'), 0, 24)), \
                     patch.object(entry, 'load_street_selection', return_value={}), \
                     patch.object(entry, 'build_plan', return_value=(plan, prepared)), \
                     patch.object(entry.StageBudget, 'install_signals'), \
                     patch.object(entry, 'stage_asset', return_value=root/'unused.pt') as stage, \
                     patch.object(entry, 'train') as train:
                    parser.return_value.parse_args.return_value = args
                    entry.main()
                self.assertEqual(train.call_count, int(arm == 'baseline'))
                self.assertEqual(stage.call_count, 3 if arm == 'baseline' else 0)
                self.assertEqual((root/'output/skipped_invalid_controls.json').is_file(), arm != 'baseline')

    def test_completed_historical_run_is_read_only_and_not_recertified(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(Path(tmp)/'run', [])
            bank = ArtifactStore(Path(tmp)/'controls', [])
            plan = dict(identity='historical-id')
            store.write_json('contract.json', plan)
            store.write_json('complete.json', dict(evaluation_status='completed'))
            store.write_json('eval/source_fixture/summary.json', dict(status='completed'))
            before = {p.relative_to(store.local):p.read_bytes() for p in store.local.rglob('*') if p.is_file()}
            with patch.object(bank, 'load', side_effect=AssertionError('Historical run must remain read-only')):
                report = control_startup(bank, {}, store, plan,
                    config('reference'), torch.device('cpu'), 0)
            self.assertEqual(report['action'], 'historical_complete')
            after = {p.relative_to(store.local):p.read_bytes() for p in store.local.rglob('*') if p.is_file()}
            self.assertEqual(before, after)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
    def test_two_ranks_take_same_skip_and_same_leader_failure_branch(self):
        for fail_write in (False, True):
            with self.subTest(fail_write=fail_write), tempfile.TemporaryDirectory() as tmp:
                torch.multiprocessing.spawn(ddp_readiness_worker, args=(tmp, fail_write), nprocs=2, join=True)
                a, b = [json.loads((Path(tmp)/f'rank{i}.json').read_text(encoding='utf8')) for i in range(2)]
                self.assertEqual(a, b)
                if fail_write:
                    self.assertIn('test publication failure', a['error'])
                else:
                    self.assertEqual(a['action'], 'skip')


if __name__ == '__main__':
    unittest.main()
