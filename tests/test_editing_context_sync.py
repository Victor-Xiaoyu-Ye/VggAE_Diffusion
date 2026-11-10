"""Live native-context feedback, with real tiny VACE/UniPC and fixture codecs."""
import copy
import unittest

import torch
from torch import nn

from tests.test_editing_state_feedback import Codec, fixture
from tests.test_vace_edit_pilot import cpu_native_import
from tests.test_vace_state_runtime import backbone, config
from utils.editing_state_feedback import StateFeedback
from utils.vace_state_runtime import (pack_edit_context, refresh_reactive_context,
    StateConditionedVACE, sample_state_native)


def packed(codec, reference):
    rgb = lambda value: (value*2-1).permute(1, 0, 2, 3)
    return pack_edit_context(codec, rgb(reference['source']), rgb(reference['reference']),
        reference['masks'].permute(1, 0, 2, 3), rgb(reference['anchor'][None]))


class ContextSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(125)

    def test_refresh_matches_official_masked_repacking_and_preserves_other_channels(self):
        _, ref, _, _ = fixture()
        ref['masks'][..., :16] = .2
        ref['masks'][..., 16:] = .7
        codec = Codec()
        old = packed(codec, ref)
        original = old.clone()
        changed = ref['reference'].clone()
        changed[:, 0] += .15
        new = refresh_reactive_context(codec, old, (changed*2-1).permute(1,0,2,3),
            ref['masks'].permute(1,0,2,3))
        expected = packed(codec, dict(ref, reference=changed))
        torch.testing.assert_close(new, expected, rtol=0, atol=0)
        torch.testing.assert_close(old, original, rtol=0, atol=0)
        for actual, before in ((new[:, :1], old[:, :1]), (new[:16], old[:16]),
                               (new[32:], old[32:])):
            torch.testing.assert_close(actual, before, rtol=0, atol=0)
        self.assertGreater(float((new[16:32, 1:]-old[16:32, 1:]).abs().max()), 0.)

    def callback(self, arm='rae_sync', codec=None, **cfg):
        inputs, reference, state, projector = fixture(arm)
        codec = codec or Codec()
        context = packed(Codec(), reference)
        callback = StateFeedback(codec, projector, inputs, reference, state,
            dict(update_native_context=True, **cfg), arm)
        callback.bind_condition(context)
        callback.set_schedule([1., .6, .35, .15, 0.])
        return callback

    def test_both_representations_commit_context_state_and_velocity(self):
        for arm in ('rae_sync', 'rgb_sync'):
            callback = self.callback(arm)
            state_before, context_before = callback.state.clone(), callback.condition.clone()
            velocity = torch.zeros(16,6,4,6)
            output = callback(step_index=1, sigma=.6, latent=velocity.clone(), velocity=velocity)
            event = callback.events[-1]
            self.assertTrue(event['applied'], event)
            self.assertFalse(event['fallback'], event)
            self.assertTrue(event['context_updated'], event)
            self.assertGreater(event['reactive_context_delta_rms'], 0.)
            self.assertTrue(all(event[k] for k in ('context_prefix_preserved',
                'context_inactive_preserved','context_mask_preserved')))
            self.assertGreater(float((callback.state-state_before).abs().sum()), 0.)
            self.assertGreater(float((output-velocity).abs().sum()), 0.)
            torch.testing.assert_close(callback.condition[:,0], context_before[:,0], rtol=0, atol=0)
            expected = refresh_reactive_context(Codec(), context_before,
                (callback.last_reference['reference']*2-1).permute(1,0,2,3),
                callback.reference['masks'].permute(1,0,2,3))
            torch.testing.assert_close(callback.condition, expected, rtol=0, atol=0)
            self.assertTrue(callback.consume_history_reset())

    def test_context_encoding_failure_rolls_back_solver_and_both_mutable_inputs(self):
        class FailingContextCodec(Codec):
            def encode(self, values):
                if len(values) == 1:
                    raise RuntimeError('Injected reactive encoding failure after clean residual')
                return super().encode(values)
        callback = self.callback(codec=FailingContextCodec())
        before, state_before = callback.condition.clone(), callback.state.clone()
        velocity = torch.zeros(16,6,4,6)
        output = callback(step_index=1, sigma=.6, latent=velocity.clone(), velocity=velocity)
        self.assertIs(output, velocity)
        torch.testing.assert_close(callback.condition, before, rtol=0, atol=0)
        torch.testing.assert_close(callback.state, state_before, rtol=0, atol=0)
        self.assertEqual(callback.projector.calls, 0)
        self.assertTrue(callback.events[-1]['fallback'])
        self.assertFalse(callback.events[-1]['context_updated'])
        self.assertFalse(callback.consume_history_reset())

    def test_nonfinite_reactive_encoding_never_commits(self):
        class NonfiniteCodec(Codec):
            def encode(self, values):
                result = super().encode(values)
                if len(values) == 1:
                    result[0].fill_(float('nan'))
                return result
        callback = self.callback(codec=NonfiniteCodec())
        before, state_before = callback.condition.clone(), callback.state.clone()
        velocity = torch.zeros(16,6,4,6)
        self.assertIs(callback(step_index=1, sigma=.6, latent=velocity, velocity=velocity), velocity)
        torch.testing.assert_close(callback.condition, before, rtol=0, atol=0)
        torch.testing.assert_close(callback.state, state_before, rtol=0, atol=0)
        self.assertTrue(callback.events[-1]['fallback'])

    def test_zero_alpha_and_no_edit_preserve_context_exactly(self):
        for no_edit in (False, True):
            callback = self.callback(**({} if no_edit else {'alpha': 0.}))
            callback.projector.no_edit = no_edit
            before = callback.condition.clone()
            velocity = torch.randn(16,6,4,6)
            self.assertIs(callback(step_index=1, sigma=.6, latent=velocity, velocity=velocity), velocity)
            torch.testing.assert_close(callback.condition, before, rtol=0, atol=0)
            self.assertFalse(callback.events[-1]['context_updated'])
            self.assertFalse(callback.consume_history_reset())

    def test_unbound_context_is_visible_failure_not_silent_fixed_condition(self):
        callback = self.callback()
        callback.condition = None
        velocity = torch.zeros(16,6,4,6)
        self.assertIs(callback(step_index=1, sigma=.6, latent=velocity, velocity=velocity), velocity)
        self.assertTrue(callback.events[-1]['fallback'])
        self.assertIn('binding', callback.events[-1]['error'])

    def test_tiny_unipc_reads_updated_context_in_both_cfg_branches_without_mutating_cache(self):
        with cpu_native_import():
            cfg = dict(config(), sample_steps=5)
            inputs, ref, state, projector = fixture()
            model = StateConditionedVACE(backbone(), cfg, state.shape[0]).eval()
            nn.init.normal_(model.bridge[-1].weight, std=.05)
            condition = packed(Codec(), ref)
            cached = condition.clone()
            text, negative = torch.randn(5,12), torch.randn(3,12)
            old_state = state.clone()
            old = StateFeedback(Codec(), copy.deepcopy(projector), inputs, ref, old_state, {}, 'rgb_joint')
            fixed = sample_state_native(model, condition, text, negative, old_state, cfg, 83, velocity_callback=old)
            live_state = state.clone()
            live = StateFeedback(Codec(), copy.deepcopy(projector), inputs, ref, live_state,
                {'update_native_context': True}, 'rgb_sync')
            seen = []
            handle = model.register_forward_pre_hook(
                lambda _m, args, _kwargs: seen.append(args[2][0].clone()), with_kwargs=True)
            try:
                synced = sample_state_native(model, condition, text, negative, live_state,
                    cfg, 83, velocity_callback=live)
            finally:
                handle.remove()
            self.assertTrue(any(e['context_updated'] for e in live.events), live.events)
            self.assertFalse(any(e['fallback'] for e in live.events), live.events)
            self.assertGreater(float((seen[-1]-seen[0]).abs().mean()), 0.)
            for index in range(0,len(seen),2):
                torch.testing.assert_close(seen[index], seen[index+1], rtol=0, atol=0)
            torch.testing.assert_close(condition, cached, rtol=0, atol=0)
            torch.testing.assert_close(seen[-1][:16], cached[:16], rtol=0, atol=0)
            torch.testing.assert_close(seen[-1][32:], cached[32:], rtol=0, atol=0)
            self.assertGreater(float((synced-fixed).abs().mean()), 1e-6)


if __name__ == '__main__':
    unittest.main()
