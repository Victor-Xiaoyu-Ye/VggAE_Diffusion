"""Stage66 integration with real tiny VACE/UniPC and explicit fixture codecs.

Only heavyweight assets/encoders/decoders are replaced. Source flow, RAE
calibration, bridge alignment, joint optimization, feedback, native sampling,
ArtifactStore receipts and media IO use the production implementation. A
nonzero fixture bridge is not evidence of learned or production image quality.
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

import refine_vace_edit_state as runner
import train_vace_edit_state as stage65
from audit_edit_transport import rae_decoder
from tests.test_vace_edit_state_pipeline import (
    fixture, mock_assets, fixed_backbone, TinyCodec, TinyRAE)
from utils.native_audit_io import sha256
from utils.scene_run import ArtifactStore
from utils.vace_state_runtime import StateConditionedVACE

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO.parent/'VggAE_DataAudit'/'stage66_validation'


class FeedbackPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        SCRATCH.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock_assets())
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(runner, 'validate_assets', return_value={'fixture':'native'}))
        self.stack.enter_context(patch.object(runner, 'load_backbone', side_effect=fixed_backbone))
        self.stack.enter_context(patch.object(runner, 'load_vae', side_effect=lambda *_:TinyCodec()))
        self.stack.enter_context(patch.object(runner, 'load_teacher', side_effect=lambda *_:TinyRAE()))
        self.stack.enter_context(patch.object(runner, 'load_encoder', return_value=nn.Identity()))
        self.stack.enter_context(patch('utils.editing_state_feedback.load_decoder_only',
            side_effect=lambda *_, **__:rae_decoder(TinyRAE())))
        self.device = torch.device('cpu')

    def old_run(self, config_name='edit_state_feedback_v1.json', sample_steps=None):
        old_root = self.root/'stage65'
        old_root.mkdir()
        base, args, old = fixture(old_root)
        if sample_steps is not None:
            base['sample_steps'] = int(sample_steps)
        stage65.states(args, base, old, self.device, 0, 1)
        stage65.cache(args, base, old, self.device, 0, 1)
        m = stage65.checked_manifest(old, base)
        records = old.json('phases/cache.json')['records']
        for family in ('rgb', 'rae'):
            with torch.random.fork_rng():
                torch.manual_seed(42)
                model = StateConditionedVACE(fixed_backbone(), base, base['state_channels'][family])
            # Real bridge/control operations with an explicit fixture update;
            # zero-initialized control would not exercise state synchronization.
            with torch.no_grad():
                model.bridge[2].weight.fill_(.02)
                model.bridge[2].bias.fill_(.01)
            stage65.arm_store(old, family).save('selected.pt',
                dict(model=model.deltas(), step=1, fixture='nonzero bridge; not trained'),
                stage65.digest([m['identity'], family]))
            args.arm = family
            stage65.render_records(args, base, old, m, records, self.device, 0, 1, model=model)
        args.arm = 'native'
        stage65.render(args, base, old, self.device, 0, 1)
        ArtifactStore(args.transport, args.transport_root).write_json('prepared.json',
            dict(source_scale=[1.]*256))
        cfg = json.loads((REPO/'configs'/config_name).read_text())
        cfg.update({key:base[key] for key in
            ('seed', 'frames', 'fps', 'sample_steps', 'shift', 'guidance', 'context_scale')})
        cfg.update(cases=1)
        cfg['solver'].update(sweeps=1, steps=1)
        a = SimpleNamespace(states=str(old.local), state_root=old.roots,
            transport=args.transport, transport_root=args.transport_root,
            rae=args.rae, encoder=args.encoder, model=args.model, arm='native')
        out = ArtifactStore(self.root/'feedback',
            [str(self.root/'feedback-primary'), str(self.root/'feedback-mirror')])
        return cfg, a, out, old

    def render(self, a, cfg, out, arm):
        a.arm = arm
        try:
            runner.render(a, cfg, out, self.device, 0, 1)
        except RuntimeError as exc:
            raise AssertionError(json.dumps(out.json('phases/render_'+arm+'.json'), indent=2)) from exc

    def assert_mirrors(self, out, names):
        for name in names:
            self.assertEqual(len({sha256(Path(root)/name) for root in [out.local, *out.roots]}), 1, name)

    def test_full_feedback_pairing_media_and_recovery_without_resampling(self):
        cfg, a, out, old = self.old_run()
        runner.prepare(a, cfg, out, self.device, 0, 1)
        m = out.json('manifest.json')
        prepared = out.load('prepared/dev-0.pt', m['identity'], True)
        self.assertEqual(prepared['provenance'], 'model_output')
        self.assertNotIn('target', prepared['inputs'])
        self.assertEqual(tuple(prepared['inputs']['scale'].shape), (256,))
        self.assertTrue(prepared['inputs']['edges'])
        payloads = {}
        for arm in runner.ARMS:
            if arm.endswith('_joint'):
                self.render(a, cfg, out, arm)
            else:
                with patch.object(runner, 'load_backbone', side_effect=AssertionError('Historical baseline must be reused')):
                    self.render(a, cfg, out, arm)
            folder = 'render/'+arm+'/dev-0'
            payloads[arm] = out.load(folder+'/result.pt', m['identity'], True)
            self.assertEqual(payloads[arm]['historical_reused'], not arm.endswith('_joint'))
            if not arm.endswith('_joint'):
                historical_arm = 'native' if arm == 'native' else arm.split('_')[0]
                historical = old.load('render/'+historical_arm+'/dev-0/result.pt', m['stage65'], True)
                torch.testing.assert_close(payloads[arm]['generated'], historical['generated'], rtol=0, atol=0)
            with np.load(out.local/folder/'generated_frames.npz') as archive:
                np.testing.assert_array_equal(archive['frames'],
                    runner.crop_video(payloads[arm]['generated'], [0,0,16,32]).numpy())
            self.assert_mirrors(out, [folder+'/comparison.mp4', folder+'/generated_frames.npz'])
        self.assertEqual(len({p['input_identity'] for p in payloads.values()}), 1)
        for family in ('rgb', 'rae'):
            joint = payloads[family+'_joint']
            events = joint['events']
            self.assertTrue(events, family)
            self.assertTrue(all(not event['fallback'] for event in events), events)
            applied = [event for event in events if event['applied']]
            self.assertTrue(applied, events)
            self.assertTrue(any(event['clean_delta_rms'] > 0 for event in applied), events)
            self.assertTrue(any(event['state_delta_rms'] > 0 for event in applied), events)
            self.assertGreater(float((joint['generated']-payloads[family+'_base']['generated']).abs().sum()), 0)
            saved = list((out.local/'render'/str(family+'_joint')/'dev-0'/'feedback').glob('step_*.pt'))
            self.assertEqual(len(saved), len(applied))
        runner.summarize(a, cfg, out)
        self.assertEqual(out.json('summary.json')['status'], 'completed')
        comparisons = ['comparisons/dev-0/'+name for name in
            ('native_base_feedback.mp4', 'rgb_base_feedback.mp4', 'rae_base_feedback.mp4')]
        self.assert_mirrors(out, comparisons)
        # Different missing copies must be recovered through independent roots;
        # neither final native sampling nor the feedback solver may run again.
        movie = 'render/rae_joint/dev-0/comparison.mp4'
        npz = 'render/rae_joint/dev-0/generated_frames.npz'
        (out.local/movie).unlink()
        (Path(out.roots[0])/movie).unlink()
        (Path(out.roots[1])/npz).unlink()
        (out.local/comparisons[0]).unlink()
        (Path(out.roots[1])/comparisons[0]).unlink()
        feedback_preview = next((out.local/'render/rae_joint/dev-0/feedback').glob('step_*.mp4'))
        preview_name = feedback_preview.relative_to(out.local).as_posix()
        feedback_preview.unlink()
        (Path(out.roots[0])/preview_name).unlink()
        with patch.object(runner, 'load_backbone', side_effect=AssertionError('Resume must not resample')):
            self.render(a, cfg, out, 'rae_joint')
            runner.summarize(a, cfg, out)
        self.assert_mirrors(out, [movie, npz, preview_name, *comparisons])
        self.assertEqual(out.json('summary.json')['status'], 'completed')

    def test_missing_historical_payload_regenerates_in_new_output(self):
        cfg, a, out, old = self.old_run()
        name = 'render/native/dev-0/result.pt'
        expected = old.load(name, old.json('manifest.json')['identity'], True)['generated'].clone()
        for root in [old.local, *old.roots]:
            self.assertTrue((Path(root)/(name+'.json')).is_file())
            (Path(root)/name).unlink()
        with patch.object(runner, 'load_backbone', side_effect=fixed_backbone) as load:
            self.render(a, cfg, out, 'native')
        self.assertEqual(load.call_count, 1)
        payload = out.load(name, out.json('manifest.json')['identity'], True)
        self.assertFalse(payload['historical_reused'])
        torch.testing.assert_close(payload['generated'], expected, rtol=0, atol=0)
        self.assert_mirrors(out, ['render/native/dev-0/comparison.mp4',
            'render/native/dev-0/generated_frames.npz'])

    def test_v2_adaptive_solver_reaches_real_sampling_and_reports_new_states(self):
        """A successful writeback alone must not stand in for solver progress."""
        cfg, a, out, old = self.old_run('edit_state_feedback_bt_v2.json', sample_steps=5)
        self.assertEqual(cfg['solver']['optimizer'], 'adaptive_backtracking')
        runner.prepare(a, cfg, out, self.device, 0, 1)
        manifest = out.json('manifest.json')
        self.assertEqual(manifest['config']['solver']['optimizer'], 'adaptive_backtracking')
        outputs = {}
        for arm in runner.ARMS:
            self.render(a, cfg, out, arm)
            outputs[arm] = out.load('render/'+arm+'/dev-0/result.pt', manifest['identity'], True)
        runner.summarize(a, cfg, out)
        summary = out.json('summary.json')
        self.assertEqual(summary['status'], 'completed')
        for family in ('rgb', 'rae'):
            arm = family+'_joint'
            events = outputs[arm]['events']
            self.assertTrue(events)
            self.assertTrue(all(e['applied'] and not e['fallback'] for e in events), events)
            for event in events:
                solver = event['solver']
                self.assertEqual(solver['optimizer'], 'adaptive_backtracking')
                self.assertEqual(event['solver_objective_improved'],
                    solver['after']['objective'] < solver['before']['objective'])
                self.assertEqual(event['solver_state_delta_rms'], solver['actual_state_delta_rms'])
                self.assertLessEqual(solver['after']['objective'], solver['before']['objective'])
                self.assertTrue(event['reference_prefix_preserved'])
                self.assertEqual(event['prefix_delta_rms'], 0.)
                self.assertGreater(event['clean_delta_rms'], 0.)
                # All retries have a finite budget independent of objective
                # difficulty; a rejected block cannot silently loop forever.
                blocks = solver['accepted_steps']+solver['rejected_steps']
                self.assertLessEqual(solver['trial_evaluations'],
                    blocks*cfg['solver']['backtrack_max_trials'])
            committed = [e for e in events if e['applied'] and not e['fallback'] and
                         e['solver']['after']['objective'] < e['solver']['before']['objective']]
            self.assertTrue(committed, family+' fixture failed to improve its numerical objective')
            changed = [e for e in committed if e['solver']['actual_state_delta_rms'] > 0]
            self.assertTrue(changed)
            health = summary['feedback_health'][arm]
            self.assertEqual(health['objective_observed_events'], len(events))
            self.assertEqual(health['solver_improved_events'], len(committed))
            self.assertEqual(health['committed_improved_events'], len(committed))
            self.assertEqual(health['committed_new_state_events'], len(changed))
            self.assertEqual(health['applied_without_objective_improvement'], len(events)-len(committed))
            self.assertEqual(health['accepted_block_steps'], sum(e['solver']['accepted_steps'] for e in events))
            self.assertEqual(health['trial_evaluations'], sum(e['solver']['trial_evaluations'] for e in events))
            self.assertEqual(health['cases_without_committed_improvement'], [])
            self.assertGreater(float((outputs[arm]['generated']-outputs[family+'_base']['generated']).abs().sum()), 0.)
            self.assert_mirrors(out, ['render/'+arm+'/dev-0/comparison.mp4',
                                     'render/'+arm+'/dev-0/generated_frames.npz'])
        # Native and no-feedback baselines retain their own Stage65 identities.
        for arm in ('native', 'rgb_base', 'rae_base'):
            self.assertTrue(outputs[arm]['historical_reused'])

    def test_prepare_failure_preserves_independent_baselines(self):
        cfg, a, out, _ = self.old_run()
        with patch.object(runner, 'load_teacher', side_effect=RuntimeError('fixture teacher unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'independent baseline'):
                runner.prepare(a, cfg, out, self.device, 0, 1)
        report = out.json('phases/prepare.json')
        self.assertEqual(report['cases'][0]['status'], 'failed')
        self.assertFalse((out.local/'prepared/dev-0.pt').exists())
        for arm in ('native', 'rgb_base', 'rae_base'):
            with patch.object(runner, 'load_backbone', side_effect=AssertionError('Baseline is independent of teacher')):
                self.render(a, cfg, out, arm)
            self.assertEqual(out.json('phases/render_'+arm+'.json')['cases'][0]['status'], 'completed')
        with self.assertRaises(RuntimeError):
            runner.summarize(a, cfg, out)
        self.assertEqual(out.json('summary.json')['status'], 'incomplete')

    def test_real_adaptive_solver_downstream_failure_restores_moments_and_lr(self):
        from utils.editing_joint_state import JointEditProjector
        from utils.editing_state_feedback import StateFeedback
        cfg, a, out, old = self.old_run('edit_state_feedback_bt_v2.json')
        runner.prepare(a, cfg, out, self.device, 0, 1)
        manifest = out.json('manifest.json')
        inputs = out.load('prepared/dev-0.pt', manifest['identity'], True)['inputs']
        old_state = old.load('states/dev-0.pt', manifest['stage65'], True)
        reference = stage65.decoded_reference(old_state['reference'])
        source = runner._rgb(inputs['source'])
        valid = source.new_zeros(len(source), 1, *source.shape[-2:])
        y, x, h, w = inputs['valid_rect']; valid[:, :, y:y+h, x:x+w] = 1
        projector = JointEditProjector(rae_decoder(TinyRAE()), inputs['source_z'].float(),
            source, runner._rgb(inputs['anchor']), inputs['masks'].float(), inputs['confidence'].float(),
            cfg=cfg['solver'], initial_delta=inputs['initial_delta'].float(), valid=valid,
            proposal=runner._rgb(inputs['proposal']), edges=inputs['edges'], scale=inputs['scale'])
        primed = projector.project((source-.1).clamp(0, 1))
        self.assertFalse(primed['diagnostics']['fallback'])
        self.assertGreater(primed['diagnostics']['accepted_steps'], 0)
        self.assertTrue(any(optimizer.state for optimizer in projector.optimizers))
        before = projector.state_dict()
        features = old_state['rae_state'].float().clone()
        state_before = features.clone()

        class FailingCodec(TinyCodec):
            def encode(self, values):
                raise RuntimeError('injected downstream VAE encoder failure')

        feedback = StateFeedback(FailingCodec(), projector, inputs, reference, features, cfg, 'rae_joint')
        feedback.set_schedule([.6])
        velocity = torch.zeros(16, 6, 2, 4)
        result = feedback(step_index=0, sigma=.6, latent=velocity.clone(), velocity=velocity)
        self.assertIs(result, velocity)
        self.assertTrue(feedback.events[-1]['fallback'])
        self.assertIn('downstream VAE encoder failure', feedback.events[-1]['error'])
        self.assertGreater(feedback.events[-1]['solver']['accepted_steps'], 0)
        torch.testing.assert_close(features, state_before, rtol=0, atol=0)

        def exact(left, right):
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    exact(left[key], right[key])
            elif isinstance(left, (tuple, list)):
                self.assertEqual(len(left), len(right))
                for item, other in zip(left, right):
                    exact(item, other)
            else:
                self.assertEqual(left, right)

        exact(before, projector.state_dict())
        health = runner.feedback_health([dict(id='dev-0', feedback_events=feedback.events)])
        self.assertEqual(health['committed_improved_events'], 0)
        self.assertEqual(health['committed_new_state_events'], 0)
        self.assertFalse(feedback.consume_history_reset())

    def test_configuration_change_requires_fresh_namespace(self):
        cfg, a, out, _ = self.old_run()
        self.render(a, cfg, out, 'native')
        changed = copy.deepcopy(cfg)
        changed['solver']['steps'] += 1
        with self.assertRaisesRegex(ValueError, 'fresh FEEDBACK_NAMESPACE'):
            runner.manifest(a, changed, out)


if __name__ == '__main__':
    unittest.main()
