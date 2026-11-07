"""Actual nonlinear decoder inverse/graph tests, not production quality claims."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from utils.editing_joint_state import JointEditProjector, build_source_edges


class SquareDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Intentionally requires grad: solver must use input-only autograd and
        # must never populate gradients or update even such decoder parameters.
        self.factor = nn.Parameter(torch.tensor(1.))

    def forward(self, value):
        return value[:, 0].permute(0, 3, 1, 2).square()*self.factor


def edges(frames, height, width, confidence=1.):
    return [dict(src=i, dst=i+1, flow=torch.zeros(height, width, 2),
                 confidence=torch.full((1, height, width), confidence))
            for i in range(frames-1)]


def fixture(frames=3, size=6):
    z = torch.linspace(.2, .65, frames)[:, None, None, None].expand(frames, size, size, 3).clone()
    decoder = SquareDecoder()
    with torch.no_grad():
        source = torch.stack([decoder(z[i:i+1][None])[0] for i in range(frames)])
    effect = torch.full_like(source, .05)
    mask = torch.ones(frames, 1, size, size)
    anchor = source[0]+effect[0]
    return decoder, z, source, effect, mask, anchor


class JointStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_nonlinear_response_is_calibrated_on_destination_and_freezes_decoder(self):
        decoder, z, source, effect, mask, anchor = fixture()
        initial = torch.full_like(z, .1)
        with torch.no_grad():
            naive = torch.stack([decoder((z[i:i+1]+initial[i:i+1])[None])[0]-source[i]
                                 for i in range(len(z))])
        projector = JointEditProjector(decoder, z, source, anchor, mask, mask,
            cfg=dict(iterations=65, lr=.025, trust=.5, native_weight=0.,
                     observation_weight=1., anchor_weight=4., joint_weight=2.,
                     keep_weight=0., ridge=0.),
            initial_delta=initial, proposal=source+effect,
            edges=edges(len(z), 6, 6), scale=torch.ones(3))
        result = projector.project(source+effect)
        self.assertFalse(result['diagnostics']['fallback'])
        self.assertTrue(result['diagnostics']['gradient_checked'])
        error = float((result['raw_effect']-effect).square().mean())
        naive_error = float((naive-effect).square().mean())
        self.assertLess(error, naive_error*.05)
        self.assertGreater(float((result['delta'][0]-result['delta'][-1]).abs().mean()), .04)
        self.assertIsNone(decoder.factor.grad)
        self.assertEqual(float(decoder.factor), 1.)
        self.assertEqual(result['diagnostics']['scale_provenance'], 'caller_supplied_training_scale')
        self.assertEqual(tuple(result['delta'].shape), tuple(z.shape))

    def test_missing_proposal_recovers_from_source_graph_instead_of_copying_source(self):
        _, _, source, effect, mask, anchor = fixture()
        observation = torch.zeros_like(mask); observation[0] = 1
        proposal = source.clone(); proposal[0] = anchor
        projector = JointEditProjector(None, None, source, anchor, mask, observation,
            cfg=dict(iterations=80, lr=.03, native_weight=.15, observation_weight=1.,
                     joint_weight=8., anchor_weight=10., ridge=0., reachable_mix=.8),
            initial_effect=proposal-source, proposal=proposal,
            edges=edges(3, 6, 6))
        native = source.clone()
        result = projector.project(native)
        self.assertFalse(result['diagnostics']['fallback'])
        self.assertGreater(float(result['reachable'][2].mean()), .99)
        self.assertGreater(float(result['effect'][2].mean()), .025)
        self.assertGreater(result['diagnostics']['recovered_fraction'], .6)
        self.assertIsNone(result['delta'])
        # Source-only graph conveys an edit despite absent proposal evidence.
        self.assertFalse(torch.equal(result['reference'][2], source[2]))

    def test_unknown_region_keeps_native_not_source_and_reads_current_each_call(self):
        _, _, source, effect, mask, anchor = fixture()
        confidence = torch.zeros_like(mask); confidence[0] = 1
        proposal = source.clone(); proposal[0] = anchor
        projector = JointEditProjector(None, None, source, anchor, mask, confidence,
            cfg=dict(iterations=3), initial_effect=proposal-source, proposal=proposal,
            edges=edges(3, 6, 6, confidence=0.))
        native_a = (source+.07).clamp(0, 1)
        result_a = projector.project(native_a)
        torch.testing.assert_close(result_a['reference'][1:], native_a[1:], rtol=0, atol=0)
        self.assertEqual(float(result_a['mix'][1:].sum()), 0.)
        native_b = (source+.12).clamp(0, 1)
        result_b = projector.project(native_b)
        torch.testing.assert_close(result_b['reference'][1:], native_b[1:], rtol=0, atol=0)
        self.assertFalse(torch.equal(result_b['reference'][1:], result_a['reference'][1:]))
        self.assertGreater(result_b['diagnostics']['unknown_fraction'], .6)

    def test_outgoing_edge_backprop_changes_source_frame_even_when_no_incoming(self):
        _, _, source, _, mask, anchor = fixture(frames=2)
        proposal = source.clone(); proposal[0] = anchor; proposal[1] += .2
        common = dict(iterations=60, lr=.02, native_weight=0., observation_weight=1.,
                      anchor_weight=.01, hard_anchor=False, ridge=0.)
        graph = edges(2, 6, 6)
        independent = JointEditProjector(None, None, source, anchor, mask, mask,
            cfg=dict(common, joint_weight=0.), initial_effect=proposal-source,
            proposal=proposal, edges=graph).project(proposal)
        coupled = JointEditProjector(None, None, source, anchor, mask, mask,
            cfg=dict(common, joint_weight=8.), initial_effect=proposal-source,
            proposal=proposal, edges=graph).project(proposal)
        jump = lambda out:float((out['raw_effect'][1]-out['raw_effect'][0]).square().mean())
        self.assertLess(jump(coupled), jump(independent)*.1)
        self.assertGreater(float(coupled['raw_effect'][0].mean()),
                           float(independent['raw_effect'][0].mean())+.005)

    def test_trust_region_anchor_and_no_edit_identity_are_distinct(self):
        decoder, z, source, effect, mask, anchor = fixture()
        initial = torch.zeros_like(z)
        projector = JointEditProjector(decoder, z, source, anchor, mask, mask,
            cfg=dict(iterations=3, lr=1., trust=.02), initial_delta=initial,
            proposal=source+effect, edges=edges(3, 6, 6), scale=torch.full((3,), .5))
        result = projector.project(source+.02)
        self.assertLessEqual(float(result['delta'].abs().max()), .010001)
        torch.testing.assert_close(result['reference'][0], anchor, rtol=0, atol=0)
        # The decoded anchor can remain imperfect despite the shared hard-anchor
        # output constraint; its fit is recorded separately and never hidden.
        self.assertGreater(result['diagnostics']['raw_anchor_relative_mse'], .1)
        identity = JointEditProjector(decoder, z, source, source[0], mask, mask,
            cfg=dict(iterations=3), initial_delta=torch.zeros_like(z),
            proposal=source, edges=edges(3, 6, 6), scale=torch.ones(3))
        native = source+.04
        unchanged = identity.project(native)
        torch.testing.assert_close(unchanged['reference'], native, rtol=0, atol=0)
        self.assertTrue(unchanged['diagnostics']['no_edit'])
        self.assertEqual(unchanged['diagnostics']['decoder_evaluations'], 0)

    def test_nonfinite_decoder_falls_back_to_native_and_rolls_back_solver(self):
        decoder, z, source, effect, mask, anchor = fixture()
        holder = {'broken':False}
        def decode(value):
            out = decoder(value)
            return out*float('nan') if holder['broken'] else out
        projector = JointEditProjector(decode, z, source, anchor, mask, mask,
            cfg=dict(iterations=3), initial_delta=torch.zeros_like(z),
            proposal=source+effect, edges=edges(3, 6, 6), scale=torch.ones(3))
        projector.project(source+effect)
        before = projector.state_dict()
        holder['broken'] = True
        native = source+.03
        result = projector.project(native)
        self.assertTrue(result['diagnostics']['fallback'])
        torch.testing.assert_close(result['reference'], native, rtol=0, atol=0)
        after = projector.state_dict()
        for a, b in zip(before['parameters'], after['parameters']):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertIn('Nonfinite', result['diagnostics']['error'])

    def test_snapshot_replay_is_exact_and_detached_decoder_is_reported(self):
        decoder, z, source, effect, mask, anchor = fixture()
        projector = JointEditProjector(decoder, z, source, anchor, mask, mask,
            cfg=dict(sweeps=2, steps=2), initial_delta=torch.zeros_like(z),
            proposal=source+effect, edges=edges(3, 6, 6), scale=torch.ones(3))
        self.assertEqual(projector.steps, 4)
        projector.project(source+effect)
        saved = projector.state_dict()
        events = []
        expected = projector.project(source+.025, progress=events.append)
        projector.load_state_dict(saved)
        replay = projector.project(source+.025)
        torch.testing.assert_close(expected['delta'], replay['delta'], rtol=0, atol=0)
        torch.testing.assert_close(expected['reference'], replay['reference'], rtol=0, atol=0)
        self.assertEqual(sum(e['phase']=='joint_state_frame' for e in events), 12)
        detached = JointEditProjector(lambda v:decoder(v).detach(), z, source, anchor, mask, mask,
            cfg=dict(iterations=1), proposal=source+effect,
            edges=edges(3, 6, 6), scale=torch.ones(3)).project(source)
        self.assertTrue(detached['diagnostics']['fallback'])
        self.assertFalse(detached['diagnostics']['gradient_checked'])
        self.assertIn('detached', detached['diagnostics']['error'])

    def test_edge_builder_uses_pair_local_source_confidence_and_both_directions(self):
        source = torch.stack([torch.full((3, 6, 8), i*.2) for i in range(4)])
        calls = []
        def fake(pair, anchor, support, cfg):
            calls.append(pair.clone())
            torch.testing.assert_close(anchor, pair[0])
            self.assertFalse(cfg['flow_direct_anchor'])
            self.assertEqual(cfg['flow_min_confidence'], 0.)
            return dict(flows_backward=torch.zeros(1,6,8,2),
                        confidence=torch.stack([torch.ones(1,6,8), torch.full((1,6,8),.75)]))
        with patch('utils.editing_joint_state.propagate_reference', fake):
            result = build_source_edges(source)
        pairs = {(v['src'], v['dst']) for v in result}
        self.assertEqual(pairs, {(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(0,2),(2,0),(0,3),(3,0)})
        self.assertEqual(len(calls), len(pairs))
        self.assertTrue(all(torch.equal(v['confidence'], torch.full((1,6,8),.75)) for v in result))

    def test_invalid_alignment_or_precision_fails_before_mutation(self):
        decoder, z, source, effect, mask, anchor = fixture()
        with self.assertRaisesRegex(ValueError, 'scale must be positive'):
            JointEditProjector(decoder, z, source, anchor, mask, mask,
                proposal=source+effect, edges=edges(3,6,6), scale=torch.zeros(3))
        with self.assertRaisesRegex(ValueError, 'requires source edges'):
            JointEditProjector(None, None, source, anchor, mask, mask,
                proposal=source+effect)
        projector = JointEditProjector(None,None,source,anchor,mask,mask,
            proposal=source+effect, edges=edges(3,6,6))
        before = copy.deepcopy(projector.state_dict())
        with self.assertRaisesRegex(ValueError, 'Nonfinite current'):
            projector.project(source*float('nan'))
        self.assertEqual(projector.calls, before['calls'])


class AdaptiveJointStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def rgb(self, *, initial=.0495, iterations=4, **options):
        source = torch.full((1, 3, 3, 3), .2)
        mask = torch.ones(1, 1, 3, 3)
        cfg = dict(optimizer='adaptive_backtracking', iterations=iterations,
                   lr=.03, max_lr=.03, joint_weight=0., ridge=0.,
                   native_weight=.5, observation_weight=1., anchor_weight=8.)
        cfg.update(options)
        p = JointEditProjector(None, None, source, source[0]+.05, mask, mask,
            initial_effect=torch.full_like(source, initial), proposal=source+.05, cfg=cfg)
        return p, source+.05

    def assert_snapshot_equal(self, first, second, *, ignore=()):
        def check(a, b):
            if isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            elif isinstance(a, dict):
                self.assertEqual(set(a), set(b))
                for k in a:
                    check(a[k], b[k])
            elif isinstance(a, (tuple, list)):
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b):
                    check(x, y)
            else:
                self.assertEqual(a, b)
        check({k:v for k,v in first.items() if k not in ignore},
              {k:v for k,v in second.items() if k not in ignore})

    def test_block_delta_matches_global_with_bidirectional_flow_and_all_terms(self):
        torch.manual_seed(19)
        _, _, source, effect, mask, anchor = fixture(size=5)
        mask[:, :, :, :2] = 0
        valid = torch.ones_like(mask); valid[:, :, 0] = 0
        confidence = torch.rand_like(mask)
        graph = edges(3, 5, 5, confidence=.7)
        graph += [dict(src=e['dst'], dst=e['src'], flow=e['flow'].clone(),
                       confidence=e['confidence'].clone()) for e in graph]
        for j, e in enumerate(graph):
            e['flow'][..., 0] = .25 if j % 2 else -.3
            e['flow'][..., 1] = .15
        p = JointEditProjector(None, None, source, anchor, mask, confidence,
            cfg=dict(optimizer='adaptive_backtracking', ridge=.3, keep_weight=.8,
                     native_weight=.4, observation_weight=1.7, anchor_weight=3., joint_weight=2.3),
            initial_effect=effect+torch.randn_like(effect)*.02, proposal=source+effect,
            valid=valid, edges=graph)
        effects = [v[0].detach().clone() for v in p.parameters]
        current = torch.randn_like(source)*.03
        energies = torch.tensor([.003, .005, .011])
        for i, param in enumerate(p.parameters):
            old = param.detach().clone()
            before = p._score(effects, current, energies)['objective']
            local_before = p._block_score(i, effects[i], param, effects, current, energies)
            with torch.no_grad():
                param.add_(torch.randn_like(param)*.01)
            candidate = param[0].detach().clone()
            local_after = p._block_score(i, candidate, param, effects, current, energies)
            updated = list(effects); updated[i] = candidate
            after = p._score(updated, current, energies)['objective']
            self.assertAlmostEqual(after-before, (local_after-local_before)/p.frames, places=11)
            with torch.no_grad():
                param.copy_(old)

    def test_near_optimum_rgb_fixed_four_sweeps_reject_but_adaptive_descends(self):
        adaptive, current = self.rgb()
        fixed, _ = self.rgb(optimizer='fixed_adam')
        old = fixed.state_dict()
        legacy = fixed.project(current)
        self.assertEqual(legacy['diagnostics']['actual_state_delta_rms'], 0.)
        self.assertFalse(legacy['diagnostics']['objective_improved'])
        self.assertTrue(all(h['objective'] > legacy['diagnostics']['before']['objective']
                            for h in legacy['diagnostics']['history']))
        self.assert_snapshot_equal(old, fixed.state_dict(), ignore=('calls',))
        result = adaptive.project(current)
        d = result['diagnostics']
        self.assertFalse(d['fallback'])
        self.assertTrue(d['objective_improved'])
        self.assertGreater(d['actual_state_delta_rms'], 0.)
        self.assertGreater(d['accepted_steps'], 0)
        self.assertGreater(d['backtracks'], 0)
        self.assertEqual(d['accepted_steps']+d['rejected_steps'], 4)
        self.assertLess(d['after']['objective'], d['before']['objective']*.25)
        history = [d['before']['objective']]+[v['objective'] for v in d['history']]
        self.assertTrue(all(a >= b for a,b in zip(history, history[1:])))

    def test_nonlinear_decoder_adaptive_trials_are_no_grad_and_input_only(self):
        decoder, z, source, effect, mask, anchor = fixture()
        grad_modes = []
        def decode(value):
            grad_modes.append(torch.is_grad_enabled())
            return decoder(value)
        p = JointEditProjector(decode, z, source, anchor, mask, mask,
            cfg=dict(optimizer='adaptive_backtracking', iterations=4, lr=.03,
                     native_weight=.5, observation_weight=1., joint_weight=2.),
            initial_delta=torch.full_like(z, .1), proposal=source+effect,
            edges=edges(3,6,6), scale=torch.ones(3))
        original = decoder.factor.detach().clone()
        result = p.project(source+effect)
        d = result['diagnostics']
        self.assertFalse(d['fallback'])
        self.assertTrue(d['objective_improved'])
        self.assertGreater(d['accepted_steps'], 0)
        self.assertLessEqual(d['trial_evaluations'], 12*8)
        self.assertEqual(sum(grad_modes), 12)
        self.assertIsNone(decoder.factor.grad)
        torch.testing.assert_close(decoder.factor.detach(), original, rtol=0, atol=0)

    def test_stale_adam_momentum_restarts_projected_non_descent_direction(self):
        p, current = self.rgb(initial=.06, iterations=1)
        parameter = p.parameters[0]
        p.optimizers[0].state[parameter] = dict(step=torch.tensor(100.),
            exp_avg=torch.full_like(parameter, -10.), exp_avg_sq=torch.ones_like(parameter))
        result = p.project(current)
        d = result['diagnostics']
        self.assertFalse(d['fallback'])
        self.assertEqual(d['direction_restarts'], 1)
        self.assertEqual(d['accepted_steps'], 1)
        self.assertTrue(d['objective_improved'])
        self.assertLess(float(p.parameters[0].mean()), .06)
        self.assertEqual(float(p.optimizers[0].state[parameter]['step']), 1.)

    def test_projected_box_constraint_is_respected(self):
        p, current = self.rgb(initial=0., iterations=4, rgb_trust=.001)
        result = p.project(current)
        self.assertFalse(result['diagnostics']['fallback'])
        self.assertTrue(result['diagnostics']['objective_improved'])
        self.assertLessEqual(float(p.parameters[0].abs().max()), .00100001)
        self.assertEqual(result['diagnostics']['accepted_steps'], 1)
        self.assertGreaterEqual(result['diagnostics']['rejected_steps'], 3)

    def test_nonfinite_candidate_backtracks_without_hiding_setup_errors(self):
        decoder, z, source, effect, mask, anchor = fixture(frames=1, size=3)
        def decode(value):
            output = decoder(value)
            # Baseline and differentiable start are valid; oversized candidate
            # steps cross a decoder numerical boundary, then smaller ones work.
            return output*float('nan') if float(value.detach().max()) > .21 else output
        p = JointEditProjector(decode, z, source, anchor, mask, mask,
            cfg=dict(optimizer='adaptive_backtracking', iterations=1, lr=.03, joint_weight=0.),
            proposal=source+effect, initial_delta=torch.zeros_like(z), scale=torch.ones(3))
        result = p.project(source+effect)
        d = result['diagnostics']
        self.assertFalse(d['fallback'])
        self.assertGreater(d['nonfinite_trials'], 0)
        self.assertGreater(d['backtracks'], 0)
        self.assertEqual(d['accepted_steps'], 1)
        self.assertTrue(d['objective_improved'])
        # Explicit numerical exceptions also backtrack; other error types do
        # not. This wrapper's invalid region is the same as the NaN fixture.
        def throws_numeric(value):
            if float(value.detach().max()) > .21:
                raise FloatingPointError('Synthetic nonlinear decoder overflow')
            return decoder(value)
        numeric = JointEditProjector(throws_numeric, z, source, anchor, mask, mask,
            cfg=dict(optimizer='adaptive_backtracking', iterations=1, lr=.03, joint_weight=0.),
            proposal=source+effect, initial_delta=torch.zeros_like(z), scale=torch.ones(3))
        retried = numeric.project(source+effect)
        self.assertFalse(retried['diagnostics']['fallback'])
        self.assertGreater(retried['diagnostics']['nonfinite_trials'], 0)
        self.assertTrue(retried['diagnostics']['objective_improved'])
        def wrong_shape(value):
            out = decoder(value)
            return out[:, :, :2] if float(value.detach().max()) > .201 else out
        broken = JointEditProjector(wrong_shape, z, source, anchor, mask, mask,
            cfg=dict(optimizer='adaptive_backtracking', iterations=1, lr=.03, joint_weight=0.),
            proposal=source+effect, initial_delta=torch.zeros_like(z), scale=torch.ones(3))
        original = broken.state_dict()
        bad = broken.project(source+effect)
        self.assertTrue(bad['diagnostics']['fallback'])
        self.assertIn('changed shape', bad['diagnostics']['error'])
        self.assertEqual(bad['diagnostics']['backtracks'], 0)
        self.assert_snapshot_equal(original, broken.state_dict(), ignore=('calls',))

    def test_all_rejected_restores_parameters_moments_but_keeps_smaller_lr(self):
        p, current = self.rgb(initial=.049999, iterations=1, backtrack_max_trials=1)
        before = p.state_dict()
        result = p.project(current)
        after = p.state_dict()
        d = result['diagnostics']
        self.assertFalse(d['fallback'])
        self.assertFalse(d['objective_improved'])
        self.assertEqual(d['accepted_steps'], 0)
        self.assertEqual(d['rejected_steps'], 1)
        self.assertEqual(d['actual_state_delta_rms'], 0.)
        self.assertEqual(d['trial_evaluations'], 1)
        self.assertLess(after['step_sizes'][0], before['step_sizes'][0])
        self.assert_snapshot_equal(before, after, ignore=('calls','step_sizes'))

    def test_adaptive_snapshot_replay_and_downstream_failure_rollback(self):
        p, current = self.rgb()
        p.project(current)
        before = p.state_dict()
        expected = p.project(current+.001)
        expected_state = p.state_dict()
        p.load_state_dict(before)  # Caller rolls back after its VAE write fails.
        self.assert_snapshot_equal(before, p.state_dict())
        replay = p.project(current+.001)
        self.assert_snapshot_equal(expected_state, p.state_dict())
        torch.testing.assert_close(expected['reference'], replay['reference'], rtol=0, atol=0)
        self.assertEqual(expected['diagnostics']['step_sizes'], replay['diagnostics']['step_sizes'])
        prior = p.state_dict()
        def fail_after_update(event):
            raise RuntimeError('Synthetic downstream bookkeeping failure')
        bad = p.project(current+.01, progress=fail_after_update)
        self.assertTrue(bad['diagnostics']['fallback'])
        self.assertFalse(bad['diagnostics']['objective_improved'])
        self.assertEqual(bad['diagnostics']['actual_state_delta_rms'], 0.)
        self.assert_snapshot_equal(prior, p.state_dict(), ignore=('calls',))

    def test_default_fixed_path_matches_explicit_fixed_exactly(self):
        explicit, current = self.rgb(optimizer='fixed_adam')
        source = explicit.source
        cfg = dict(explicit.cfg); cfg.pop('optimizer')
        implicit = JointEditProjector(None, None, source, explicit.anchor,
            explicit.masks, explicit.confidence, initial_effect=explicit.initial,
            proposal=explicit.proposal, cfg=cfg)
        for value in (current, current+.006):
            first = implicit.project(value)
            second = explicit.project(value)
            torch.testing.assert_close(first['reference'], second['reference'], rtol=0, atol=0)
            self.assert_snapshot_equal(implicit.state_dict(), explicit.state_dict())


if __name__ == '__main__':
    unittest.main()
