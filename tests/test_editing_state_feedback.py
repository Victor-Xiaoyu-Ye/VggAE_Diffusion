"""Real tiny VACE/UniPC plus explicit RGB-codec fixtures for live feedback."""
import copy
import json
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from tests.test_vace_edit_pilot import cpu_native_import
from tests.test_vace_state_runtime import backbone, config
from utils.editing_data import rae_canvas
from utils.editing_state_feedback import (SceneDecoderOnly, StateFeedback,
    align_feedback_gate, expand_source_correction, map_square_correction)
from utils.vace_state_data import align_rae_state, build_rgb_state
from utils.vace_state_runtime import StateConditionedVACE, sample_state_native


class Codec:
    def __init__(self, fail_encode=False):
        self.fail_encode, self.decoded = fail_encode, []

    def encode(self, values):
        if self.fail_encode:
            raise RuntimeError('Injected encoder failure')
        return [F.avg_pool2d(v[:, ::4].permute(1, 0, 2, 3), 8).permute(1, 0, 2, 3)
                .repeat(6, 1, 1, 1)[:16]+.125 for v in values]

    def decode(self, values):
        self.decoded.extend(v.clone() for v in values)
        return [F.interpolate(v[:3][None].float(),
            size=(4*(v.shape[1]-1)+1, v.shape[2]*8, v.shape[3]*8),
            mode='trilinear', align_corners=False)[0].tanh() for v in values]


class Projector:
    """Observable current-dependent stand-in for the separately tested solver."""
    def __init__(self, source, delta, fail=False, fallback=False, no_edit=False):
        self.source, self.delta = source, delta
        self.fail, self.fallback, self.no_edit = fail, fallback, no_edit
        self.calls, self.inputs = 0, []

    def state_dict(self):
        return dict(calls=self.calls)

    def load_state_dict(self, value):
        self.calls = value['calls']

    def project(self, current, progress=None):
        self.calls += 1; self.inputs.append(current.clone())
        if self.fail:
            raise RuntimeError('Injected solver failure')
        reference = current if self.no_edit else (.6*current+.4*(self.source+.1)).clamp(0, 1)
        return dict(reference=reference, delta=self.delta+current.mean()*.01,
            raw_effect=reference-self.source, effect=reference-self.source,
            mix=torch.ones(len(current), 1, *current.shape[-2:]),
            diagnostics=dict(fallback=self.fallback, no_edit=self.no_edit, hard_anchor=False))


def fixture(arm='rgb_joint', **projector_kwargs):
    source = torch.full((17, 3, 32, 48), .35)
    reference = dict(source=source, reference=source+.05, anchor=source[0]+.05,
        masks=torch.ones(17, 1, 32, 48), confidence=torch.ones(17, 1, 32, 48),
        spatial=dict(canvas=[32, 48], valid_rect=[4, 4, 24, 40]))
    square, rect = rae_canvas(source, reference['spatial']['valid_rect'], size=32)
    inputs = dict(source=square[::2], observed_indices=list(range(0, 17, 2)),
        valid_rect=rect, source_z=torch.randn(9, 4, 4, 256)*.1,
        anchor_z=torch.randn(1, 4, 4, 256)*.1,
        full_flows_backward=torch.zeros(16, 32, 32, 2),
        full_edge_confidence=torch.ones(16, 1, 32, 32))
    delta = torch.full_like(inputs['source_z'], .03)
    state = (build_rgb_state(reference) if arm.startswith('rgb') else
        align_rae_state(inputs['source_z'], inputs['source_z']+delta,
            inputs['anchor_z'], inputs, reference))
    projector = Projector(square[::2], delta, **projector_kwargs)
    return inputs, reference, state, projector


class FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(76)

    def callback(self, arm='rgb_joint', codec=None, cfg=None, **kwargs):
        inputs, reference, state, projector = fixture(arm, **kwargs)
        callback = StateFeedback(codec or Codec(), projector, inputs, reference, state,
            cfg or {}, arm)
        callback.set_schedule([1., .6, .35, .15, 0.])
        return callback

    def test_sigma_selection_is_actual_unique_and_bounded(self):
        callback = self.callback()
        callback.set_schedule([1., .96, .5, .18, 0.])
        self.assertEqual(callback.selected_steps, {2, 3})

    def test_non_square_padding_and_odd_visibility_preserve_native(self):
        inputs, reference, _, _ = fixture()
        correction = torch.ones(9, 3, 32, 32)*.12
        inputs['full_edge_confidence'][0] = 0
        full = expand_source_correction(correction, inputs, 17)
        self.assertEqual(float(full[1].abs().max()), 0.)
        torch.testing.assert_close(full[3], correction[1], rtol=0, atol=0)
        native = map_square_correction(full, inputs['valid_rect'], reference['spatial'])
        self.assertEqual(tuple(native.shape), (17, 3, 32, 48))
        self.assertEqual(float(native[:, :, :4].abs().max()), 0.)
        self.assertEqual(float(native[:, :, :, :4].abs().max()), 0.)
        torch.testing.assert_close(native[2, :, 4:28, 4:44], torch.full((3, 24, 40), .12))
        # A target->previous +1 displacement moves the impulse left, not right.
        impulse = torch.zeros_like(correction); impulse[0, :, 12, 12] = 1
        inputs['full_edge_confidence'].fill_(1)
        inputs['full_flows_backward'][0, :, :, 0] = 1
        moved = expand_source_correction(impulse, inputs, 17)
        self.assertEqual(float(moved[1, 0, 12, 11]), 1.)
        self.assertEqual(float(moved[1, 0, 12, 12]), 0.)

    def test_real_writeback_sign_prefix_rms_and_both_state_types(self):
        for arm in ('rgb_joint', 'rae_joint'):
            callback = self.callback(arm, cfg={'max_clean_delta_rms': .002})
            before = callback.state.clone()
            latent, velocity = torch.zeros(16, 6, 4, 6), torch.zeros(16, 6, 4, 6)
            output = callback(step_index=1, sigma=.6, latent=latent, velocity=velocity)
            self.assertFalse(callback.events[-1]['fallback'])
            self.assertTrue(callback.events[-1]['applied'])
            self.assertGreater(float((output-velocity).abs().max()), 0.)
            torch.testing.assert_close(output[:, 0], velocity[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(callback.state[:, 0], before[:, 0], rtol=0, atol=0)
            self.assertGreater(float((callback.state[:, 1:]-before[:, 1:]).abs().sum()), 0.)
            clean_change = -.6*(output-velocity)
            self.assertLessEqual(float(clean_change.square().mean((-1, -2)).sqrt().max()), .0020001)
            # Native current is .5, corrected source edit is .45: negative edit
            # must decrease clean latent (and thus increase velocity).
            self.assertLess(float(clean_change[:, 1:].mean()), 0.)
            self.assertTrue(callback.consume_history_reset())
            self.assertFalse(callback.consume_history_reset())
            self.assertEqual(tuple(callback.vae.decoded[0].shape), (16, 5, 4, 6))

    def test_zero_alpha_and_no_edit_are_exact_noops(self):
        for cfg, kwargs in (({'alpha': 0.}, {'fail': True}), ({}, {'no_edit': True})):
            callback = self.callback(cfg=cfg, **kwargs)
            before = callback.state.clone()
            latent, velocity = torch.randn(16, 6, 4, 6), torch.randn(16, 6, 4, 6)
            output = callback(step_index=1, sigma=.6, latent=latent, velocity=velocity)
            self.assertIs(output, velocity)
            torch.testing.assert_close(callback.state, before, rtol=0, atol=0)
            self.assertFalse(callback.consume_history_reset())
            if cfg:
                self.assertEqual(callback.projector.calls, 0)

    def test_solver_and_encoder_failures_rollback_state_and_solver(self):
        for kwargs, codec in (({'fail': True}, Codec()), ({'fallback': True}, Codec()), ({}, Codec(True))):
            callback = self.callback(codec=codec, **kwargs)
            before = callback.state.clone()
            latent, velocity = torch.zeros(16, 6, 4, 6), torch.randn(16, 6, 4, 6)
            output = callback(step_index=1, sigma=.6, latent=latent, velocity=velocity)
            self.assertIs(output, velocity)
            torch.testing.assert_close(callback.state, before, rtol=0, atol=0)
            self.assertEqual(callback.projector.calls, 0)
            self.assertTrue(callback.events[-1]['fallback'])
            self.assertFalse(callback.consume_history_reset())

    def test_correction_depends_on_current_prediction(self):
        a, b = self.callback(), self.callback()
        velocity = torch.zeros(16, 6, 4, 6)
        a(step_index=1, sigma=.6, latent=velocity.clone(), velocity=velocity)
        b(step_index=1, sigma=.6, latent=velocity+.2, velocity=velocity)
        self.assertGreater(float((a.projector.inputs[0]-b.projector.inputs[0]).abs().mean()), .05)
        self.assertGreater(float((a.last_reference['reference']-b.last_reference['reference']).abs().mean()), .01)

    def test_formal_config_overrides_and_velocity_only_ablation(self):
        callback = self.callback(cfg=dict(writeback_alpha=.2, writeback_max_clean_rms=.001,
            writeback_max_sigma=.7, writeback_sigmas=[.35], update_bridge_state=False))
        self.assertEqual(callback.selected_steps, {2})
        before = callback.state.clone()
        velocity = torch.zeros(16, 6, 4, 6)
        result = callback(step_index=2, sigma=.35, latent=velocity.clone(), velocity=velocity)
        self.assertTrue(callback.events[-1]['applied'])
        torch.testing.assert_close(callback.state, before, rtol=0, atol=0)
        self.assertEqual(callback.events[-1]['state_delta_rms'], 0.)
        self.assertLessEqual(float((.35*result).square().mean((-1, -2)).sqrt().max()), .0010001)
        callback = self.callback(cfg=dict(writeback_alpha=0.))
        self.assertIs(callback(step_index=1, sigma=.6, latent=velocity, velocity=velocity), velocity)
        self.assertEqual(callback.projector.calls, 0)

    def test_unreliable_rae_bridge_preserves_old_state_and_static_channels(self):
        callback = self.callback('rae_joint')
        project = callback.projector.project
        def masked(current, progress=None):
            value = project(current, progress=progress)
            value['mix'][2].zero_()  # native frame4 -> bridge temporal slot2
            return value
        callback.projector.project = masked
        before = callback.state.clone()
        velocity = torch.zeros(16, 6, 4, 6)
        output = callback(step_index=1, sigma=.6, latent=velocity.clone(), velocity=velocity)
        self.assertTrue(callback.events[-1]['applied'])
        self.assertGreater(float(output.abs().sum()), 0.)
        torch.testing.assert_close(callback.state[:, 2], before[:, 2], rtol=0, atol=0)
        torch.testing.assert_close(callback.state[:256], before[:256], rtol=0, atol=0)
        torch.testing.assert_close(callback.state[-2:], before[-2:], rtol=0, atol=0)
        self.assertGreater(float((callback.state[256:512, 3]-before[256:512, 3]).abs().sum()), 0.)
        # Every event including memory and solver diagnostics is JSON-safe.
        json.dumps(callback.events)
        self.assertEqual(callback.events[-1]['memory']['device'], 'cpu')
        zeros = torch.zeros(9, 1, 32, 32)
        gate = align_feedback_gate(zeros, callback.inputs, callback.reference)
        self.assertEqual(float(gate.abs().max()), 0.)

    def test_tiny_native_unipc_final_output_and_next_cfg_state_change(self):
        with cpu_native_import():
            cfg = dict(config(), sample_steps=5)
            inputs, reference, state, projector = fixture()
            model = StateConditionedVACE(backbone(), cfg, state.shape[0]).eval()
            nn.init.normal_(model.bridge[-1].weight, std=.05)
            condition, text, negative = torch.randn(96, 6, 4, 6), torch.randn(5, 12), torch.randn(3, 12)
            plain = sample_state_native(model, condition, text, negative, state.clone(), cfg, 83)
            zero_state = state.clone()
            zero = StateFeedback(Codec(), projector, inputs, reference, zero_state, {'alpha': 0.}, 'rgb_joint')
            noop = sample_state_native(model, condition, text, negative, zero_state, cfg, 83, velocity_callback=zero)
            torch.testing.assert_close(noop, plain, rtol=0, atol=0)
            live_state = state.clone(); observed = []
            callback = StateFeedback(Codec(), projector, inputs, reference, live_state, {}, 'rgb_joint')
            handle = model.register_forward_pre_hook(
                lambda _m, args, _kwargs: observed.append(args[4][0].clone()), with_kwargs=True)
            live = sample_state_native(model, condition, text, negative, live_state, cfg, 83, velocity_callback=callback)
            handle.remove()
            self.assertTrue(any(event['applied'] for event in callback.events))
            self.assertFalse(any(event['fallback'] for event in callback.events))
            self.assertGreater(float((live-plain).abs().mean()), 1e-5)
            self.assertGreater(float((observed[-1]-observed[0]).abs().mean()), 1e-5)
            for i in range(0, len(observed), 2):
                torch.testing.assert_close(observed[i], observed[i+1], rtol=0, atol=0)
            torch.testing.assert_close(observed[-1][:, 0], state[:, 0], rtol=0, atol=0)

    def test_decoder_only_matches_exact_teacher_path_and_input_gradients(self):
        class RGBDecoder(nn.Module):
            def forward(self, a, b):
                return torch.cat((a, b), -1)[..., :3].sigmoid()
        teacher = SimpleNamespace(channels=4,
            geo_projection=SimpleNamespace(expand=nn.Linear(2, 2)),
            tex_projection=SimpleNamespace(expand=nn.Linear(2, 2)), decoder=RGBDecoder())
        model = SceneDecoderOnly(teacher, checkpoint_decode=True)
        z = torch.randn(1, 1, 3, 3, 4, requires_grad=True)
        geo, tex = z.split(2, dim=-1)
        expected = teacher.decoder(teacher.geo_projection.expand(geo), teacher.tex_projection.expand(tex))
        value = model(z)
        torch.testing.assert_close(value, expected[:, 0].permute(0, 3, 1, 2), rtol=0, atol=0)
        value.sum().backward()
        self.assertGreater(float(z.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
