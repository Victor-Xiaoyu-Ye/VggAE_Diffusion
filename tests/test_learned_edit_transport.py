import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models.edit_transport import EditTransport, decoded_reference_frame


class ToyDecoder(nn.Module):
    def __init__(self, channels=4, size=8):
        super().__init__()
        self.proj = nn.Linear(channels, 3, bias=False)
        self.size = size
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[:, :3].copy_(torch.eye(3))
        self.requires_grad_(False)

    def forward(self, z):
        image = self.proj(z[:, 0]).permute(0, 3, 1, 2)
        return F.interpolate(image, (self.size, self.size), mode='bilinear',
                             align_corners=False)


class LearnedEditTransportTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.cfg = dict(width=16, depth=2, heads=4, max_frames=5,
                        delta_limit=.75, support_dilation=3,
                        trusted_confidence=.95)
        self.z = torch.randn(2, 3, 2, 2, 4) * .1
        self.anchor = self.z[:, :1] + .2
        self.source = torch.full((2, 3, 3, 8, 8), .3)
        self.proposal = self.source + .15
        self.confidence = torch.zeros(2, 3, 1, 8, 8)
        self.masks = torch.ones_like(self.confidence)
        self.scale = torch.ones(4)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def predict(self, model, **changes):
        values = dict(source_z=self.z, anchor_z=self.anchor, source=self.source,
                      proposal=self.proposal, confidence=self.confidence,
                      masks=self.masks, scale=self.scale)
        values.update(changes)
        return model(**values)

    def render(self, decoder, delta, frame=1, **changes):
        values = dict(decode=decoder, source_z=self.z, delta=delta,
                      scale=self.scale, source=self.source, proposal=self.proposal,
                      confidence=self.confidence, masks=self.masks,
                      frame=frame, cfg=self.cfg)
        values.update(changes)
        return decoded_reference_frame(**values)

    def test_zero_initialized_network_is_exact_rgb_bypass(self):
        model = EditTransport(self.cfg, input_channels=4)
        delta = self.predict(model)
        self.assertEqual(tuple(delta.shape), tuple(self.z.shape))
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        decoder = ToyDecoder()
        for frame in range(3):
            reference, info = self.render(decoder, delta, frame)
            self.assertTrue(torch.equal(reference, self.proposal[:, frame]))
            self.assertEqual(float(info['raw_residual'].abs().max()), 0.)

    def test_observed_anchor_is_exact_even_for_wrong_latent_edit(self):
        delta = torch.randn_like(self.z) * 50
        def forbidden_decoder(_):
            raise AssertionError('Observed anchor must not pass through decoder')
        reference, info = self.render(forbidden_decoder, delta, frame=0)
        self.assertTrue(torch.equal(reference, self.proposal[:, 0]))
        self.assertEqual(float(info['gate'].sum()), 0.)

    def test_frozen_decoder_passes_gradient_and_training_improves_future(self):
        model = EditTransport(self.cfg, input_channels=4)
        decoder = ToyDecoder()
        before = {k: v.clone() for k, v in decoder.state_dict().items()}
        target = self.proposal[:, 1].clone()
        target[:, 0] += .10
        target[:, 1] -= .06
        optimizer = torch.optim.Adam(model.parameters(), lr=.004)
        losses = []
        for _ in range(18):
            optimizer.zero_grad()
            delta = self.predict(model)
            reference, _ = self.render(decoder, delta)
            loss = F.mse_loss(reference, target)
            loss.backward()
            self.assertIsNotNone(model.output.weight.grad)
            self.assertTrue(torch.isfinite(model.output.weight.grad).all())
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0] * .2)
        self.assertGreater(float(model.output.weight.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))
        for key, value in decoder.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        reference, _ = self.render(decoder, self.predict(model), frame=0)
        self.assertTrue(torch.equal(reference, self.proposal[:, 0]))

    def test_trusted_pixels_protected_uncertain_support_writable(self):
        delta = torch.ones_like(self.z) * .1
        confidence = self.confidence.clone()
        confidence[:, :, :, :, :4] = .99
        reference, info = self.render(ToyDecoder(), delta, confidence=confidence)
        self.assertTrue(torch.equal(reference[:, :, :, :4], self.proposal[:, 1, :, :, :4]))
        self.assertGreater(float((reference[:, :, :, 4:] - self.proposal[:, 1, :, :, 4:]).mean()), .09)
        self.assertTrue(torch.equal(info['editable_support'], torch.ones_like(info['editable_support'])))
        # This support remains nonzero in low-confidence areas for native writeback.
        self.assertGreater(float(info['editable_support'][:, :, :, 4:].sum()), 0.)

    def test_mask_dilation_and_bounds(self):
        masks = torch.zeros_like(self.masks)
        masks[:, :, :, 3, 3] = 2.
        confidence = torch.full_like(self.confidence, -.2)
        reference, info = self.render(ToyDecoder(), torch.ones_like(self.z) * .1,
                                      masks=masks, confidence=confidence)
        self.assertGreaterEqual(float(info['gate'].min()), 0.)
        self.assertLessEqual(float(info['gate'].max()), 1.)
        self.assertEqual(int((info['editable_support'][0] > 0).sum()), 9)
        self.assertTrue(torch.equal(reference[:, :, 0, :], self.proposal[:, 1, :, 0, :]))
        self.assertGreater(float((reference[:, :, 2:5, 2:5] - self.proposal[:, 1, :, 2:5, 2:5]).mean()), 0.)

    def test_no_across_batch_leakage_and_reference_matters(self):
        model = EditTransport(self.cfg, input_channels=4).eval()
        # A trained/nonzero head is necessary to inspect actual conditioning.
        with torch.no_grad():
            model.output.weight.normal_(std=.04)
        expected = self.predict(model)
        alternate = self.anchor.clone()
        alternate[1] += torch.tensor([1., -2., .3, -.2])
        changed = self.predict(model, anchor_z=alternate)
        self.assertTrue(torch.equal(expected[0], changed[0]))
        self.assertGreater(float((expected[1, 1:] - changed[1, 1:]).abs().max()), 1e-5)
        singleton = model(self.z[:1], self.anchor[:1], self.source[:1], self.proposal[:1],
                          self.confidence[:1], self.masks[:1], self.scale)
        torch.testing.assert_close(expected[:1], singleton, rtol=1e-5, atol=1e-6)
        self.assertLessEqual(float(expected.abs().max()), self.cfg['delta_limit'])
        self.assertTrue(torch.equal(expected[:, 0], torch.zeros_like(expected[:, 0])))

    def test_checkpointed_blocks_preserve_outputs_and_gradients(self):
        plain = EditTransport(self.cfg, input_channels=4)
        checkpointed = EditTransport(dict(self.cfg, activation_checkpoint=True), input_channels=4)
        with torch.no_grad():
            plain.output.weight.normal_(std=.02)
        checkpointed.load_state_dict(plain.state_dict())
        a, b = self.predict(plain), self.predict(checkpointed)
        torch.testing.assert_close(a, b)
        a.square().sum().backward()
        b.square().sum().backward()
        for p, q in zip(plain.parameters(), checkpointed.parameters()):
            if p.grad is None:
                self.assertIsNone(q.grad)
            else:
                torch.testing.assert_close(p.grad, q.grad)

    def test_invalid_shapes_fail_explicitly(self):
        model = EditTransport(self.cfg, input_channels=4)
        with self.assertRaisesRegex(ValueError, 'one edited first-frame'):
            self.predict(model, anchor_z=self.z)
        with self.assertRaisesRegex(ValueError, 'Confidence'):
            self.predict(model, masks=self.masks[:, :1])
        with self.assertRaisesRegex(ValueError, 'Frame index'):
            self.render(ToyDecoder(), torch.zeros_like(self.z), frame=3)


if __name__ == '__main__':
    unittest.main()
