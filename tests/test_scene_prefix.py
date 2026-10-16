"""Prefix boundary and gradient tests; no claim of real encoder/NPU validation."""
import unittest

import torch
from torch import nn

from models.scene_flow import SceneFlow
from utils.scene_prefix import PrefixFlowNet, conservative_lr, prepare_prefix
from utils.window_flow import WindowFlow


class SpyFlow(nn.Module):
    def __init__(self, text_dim=7):
        super().__init__()
        self.args = {'text_dim': text_dim}
        self.scale = nn.Parameter(torch.tensor(.7))
        self.calls = []
        self.last_output = None

    def forward(self, noisy, u, anchor, text, text_valid, ref_present=None, rays=None):
        self.calls.append(dict(noisy=noisy.detach().clone(), u=u.detach().clone(),
                               anchor=anchor.detach().clone(), text=text.detach().clone(),
                               text_valid=text_valid.detach().clone(),
                               ref_present=ref_present.detach().clone(), rays=rays))
        # Slot-dependent term exposes accidentally rebased/truncated positions;
        # global mixing lets gradients reach known inputs through future outputs.
        pos = torch.arange(noisy.shape[1], device=noisy.device)[None, :, None, None]
        output = noisy*self.scale + noisy.mean(1, keepdim=True)*.1 + pos*.01
        if output.requires_grad:
            output.retain_grad()
        self.last_output = output
        return output


class DDPShapeWrapper(nn.Module):
    """Only checks forwarding through .module wrappers, not distributed execution."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ScenePrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(53)
        self.z = torch.randn(2, 9, 4, 4)
        self.mean = torch.tensor([.1, -.2, .3, -.4])
        self.std = torch.tensor([1., 2., .5, 3.])

    def test_prepare_original_normalization_and_explicit_anchor(self):
        anchor = self.z[:, :1] + .1
        part = prepare_prefix({'target': self.z, 'anchor': anchor}, self.mean, self.std)
        expected = (self.z-self.mean)/self.std
        self.assertEqual(part.history_frames, 4)
        torch.testing.assert_close(part.anchor, (anchor-self.mean)/self.std, rtol=0, atol=0)
        torch.testing.assert_close(part.known, expected[:, 1:4], rtol=0, atol=0)
        torch.testing.assert_close(part.target, expected[:, 4:9], rtol=0, atol=0)
        single = prepare_prefix({'target': self.z[0], 'anchor': anchor[0]}, self.mean, self.std)
        torch.testing.assert_close(single.known, part.known[:1], rtol=0, atol=0)

    def test_future_values_never_enter_conditions_or_pure_noise_sampling(self):
        changed = self.z.clone()
        changed[:, 4:] = changed[:, 4:]*100 + 77
        a = prepare_prefix(self.z, self.mean, self.std)
        b = prepare_prefix(changed, self.mean, self.std)
        torch.testing.assert_close(a.anchor, b.anchor, rtol=0, atol=0)
        torch.testing.assert_close(a.known, b.known, rtol=0, atol=0)
        self.assertFalse(torch.equal(a.target, b.target))
        spy = SpyFlow()
        noise = torch.randn_like(a.target)
        outputs = []
        for part in (a, b):
            net = PrefixFlowNet(spy, part.known)
            outputs.append(WindowFlow().sample(net, part.anchor, noise, steps=3,
                text=torch.randn(2, 3, 7), text_valid=torch.zeros(2, 3, dtype=torch.bool)))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(spy.calls[0]['noisy'][:, 3:], noise, rtol=0, atol=0)
        self.assertTrue(torch.equal(spy.calls[0]['u'], torch.ones(2)))
        for call in spy.calls:
            self.assertEqual(tuple(call['noisy'].shape), (2, 8, 4, 4))
            torch.testing.assert_close(call['noisy'][:, :3], a.known, rtol=0, atol=0)
            self.assertEqual(tuple(call['text'].shape), (2, 1, 7))
            self.assertEqual(call['text'].count_nonzero().item(), 0)
            self.assertTrue(call['text_valid'].all())
            self.assertTrue(call['ref_present'].all())
            self.assertIsNone(call['rays'])

    def test_history_one_exact_parity_with_actual_scene_flow(self):
        model = SceneFlow(channels=4, grid=2, width=24, depth=1, head_width=48,
                          head_depth=1, heads=4, text_dim=8).eval()
        # The default output head is zero; make this a nontrivial parity test.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.uniform_(-.1, .1)
        part = prepare_prefix(self.z, self.mean, self.std, hist=1)
        noise, u = torch.randn_like(part.target), torch.tensor([.2, .8])
        net = PrefixFlowNet(DDPShapeWrapper(model), part.known)
        direct = model(noise, u, part.anchor, torch.zeros(2, 1, 8),
                       torch.ones(2, 1, dtype=torch.bool),
                       ref_present=torch.ones(2, dtype=torch.bool), rays=None)
        got = net(noise, u, part.anchor)
        self.assertGreater(direct.abs().sum().item(), 0.)
        torch.testing.assert_close(got, direct, rtol=0, atol=0)
        self.assertEqual(net.text_dim, 8)

    def test_only_future_outputs_receive_loss_but_model_keeps_gradients(self):
        part = prepare_prefix(self.z, self.mean, self.std)
        known = part.known.clone().requires_grad_()
        noisy = torch.randn_like(part.target, requires_grad=True)
        spy = SpyFlow()
        output = PrefixFlowNet(spy, known)(noisy, torch.ones(2), part.anchor)
        self.assertEqual(tuple(output.shape), (2, 5, 4, 4))
        output.square().mean().backward()
        self.assertEqual(spy.last_output.grad[:, :3].count_nonzero().item(), 0)
        self.assertGreater(spy.last_output.grad[:, 3:].abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(spy.scale.grad))
        self.assertGreater(spy.scale.grad.abs().item(), 0.)
        self.assertGreater(noisy.grad.abs().sum().item(), 0.)
        self.assertGreater(known.grad.abs().sum().item(), 0.)

    def test_reject_layouts_that_rebase_original_positions(self):
        part = prepare_prefix(self.z, self.mean, self.std)
        net = PrefixFlowNet(SpyFlow(), part.known)
        with self.assertRaisesRegex(ValueError, 'original model slot'):
            net(torch.randn(2, 4, 4, 4), torch.ones(2), part.anchor)
        with self.assertRaises(ValueError):
            prepare_prefix(self.z, self.mean, self.std, hist=9)
        with self.assertRaises(ValueError):
            prepare_prefix({'target': self.z, 'anchor': self.z[:, :2]}, self.mean, self.std)

    def test_step_lr_boundaries_and_resume_index(self):
        schedule = [conservative_lr(i, 1000) for i in range(1000)]
        self.assertAlmostEqual(schedule[0], 1e-5/200)
        self.assertAlmostEqual(schedule[199], 1e-5)
        self.assertAlmostEqual(schedule[200], 1e-5)
        self.assertAlmostEqual(schedule[-1], 2e-6)
        self.assertTrue(all(a <= b for a, b in zip(schedule[:199], schedule[1:200])))
        self.assertTrue(all(a >= b for a, b in zip(schedule[200:-1], schedule[201:])))
        self.assertEqual(conservative_lr(713, 1000), schedule[713])
        self.assertEqual(conservative_lr(1000, 1000), 2e-6)
        self.assertEqual(conservative_lr(0, 1), 1e-5)
        self.assertEqual(conservative_lr(0, 10, warmup_steps=0), 1e-5)
        self.assertEqual(conservative_lr(9, 10, warmup_steps=0), 2e-6)
        self.assertEqual(conservative_lr(1, 2), 2e-6)
        for kwargs in ({'step': -1, 'max_steps': 10}, {'step': 0, 'max_steps': 0},
                       {'step': 0, 'max_steps': 10, 'min_lr': 1.},
                       {'step': 0, 'max_steps': 10, 'peak_lr': float('nan')}):
            with self.assertRaises(ValueError):
                conservative_lr(**kwargs)


if __name__ == '__main__':
    unittest.main()
