"""CPU camera interface contracts; these tests do not establish video quality."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.r7_window_dit import PairedCameraConditioner, R7WindowDiT


class CameraDiTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, camera_dim=14, **kw):
        return R7WindowDiT(channels=6, grid=2, future=4, temporal_factor=2,
                           width=24, depth=2, heads=3, text_dim=0,
                           checkpoint_blocks=False, camera_dim=camera_dim, **kw)

    def inputs(self):
        return torch.randn(2, 4, 4, 6), torch.tensor([.25, .75]), torch.randn(2, 1, 4, 6)

    def test_default_state_and_rng_unchanged_zero_init_exact(self):
        torch.manual_seed(713)
        base = self.model(0)
        state_after_base = torch.get_rng_state().clone()
        torch.manual_seed(713)
        controlled = self.model(14)
        self.assertTrue(torch.equal(state_after_base, torch.get_rng_state()))
        self.assertFalse(any('camera' in key for key in base.state_dict()))
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, controlled.state_dict()[key], rtol=0, atol=0)
        # Nonzero base head/gates prevents a trivial all-zero output comparison.
        with torch.no_grad():
            base.head.weight.normal_(std=.1)
            for block in base.blocks:
                block.mod[-1].bias.normal_(std=.1)
        missing = controlled.load_state_dict(base.state_dict(), strict=False)
        self.assertFalse(missing.unexpected_keys)
        self.assertEqual(set(missing.missing_keys),
                         {'camera_conditioner.'+key for key in controlled.camera_conditioner.state_dict()})
        x, u, c = self.inputs()
        camera, present = torch.randn(2, 9, 14), torch.tensor([True, False])
        expected = base(x, u, c)
        self.assertGreater(float(expected.detach().abs().max()), .01)
        torch.testing.assert_close(controlled(x, u, c, camera=camera, camera_present=present),
                                   expected, rtol=0, atol=0)
        torch.testing.assert_close(controlled(x, u, c), expected, rtol=0, atol=0)

    def test_motion_changes_output_and_absent_features_do_not_leak(self):
        model = self.model()
        with torch.no_grad():
            model.head.weight.normal_(std=.1)
            model.camera_conditioner.projection[-1].weight.normal_(std=.1)
        x, u, c = self.inputs()
        camera = torch.zeros(2, 9, 14)
        present = torch.tensor([True, False])
        result = model(x, u, c, camera=camera, camera_present=present)
        changed = camera.clone()
        changed[:, 1:, :3] = 2.
        moved = model(x, u, c, camera=changed, camera_present=present)
        self.assertGreater(float((moved[0]-result[0]).detach().abs().max()), 1e-5)
        torch.testing.assert_close(moved[1], result[1], rtol=0, atol=0)
        # Presence1+identity/stay differs from omitted control after adapter learns.
        absent = model(x, u, c)
        self.assertGreater(float((absent[0]-result[0]).detach().abs().max()), 1e-5)
        camera[1].fill_(float('nan'))
        torch.testing.assert_close(model(x, u, c, camera=camera, camera_present=present),
                                   result, rtol=0, atol=0)
        # Every RGB frame, including both members of all four pairs, must have
        # an active path to the output after the adapter starts learning.
        camera = torch.randn(2, 9, 14, requires_grad=True)
        model(x, u, c, camera=camera, camera_present=torch.ones(2, dtype=torch.bool)).square().sum().backward()
        self.assertTrue((camera.grad.abs().sum(-1) > 0).all())

    def test_all_nine_poses_are_packed_in_order(self):
        conditioner = PairedCameraConditioner(4, 2, 24)
        camera = torch.arange(9*14, dtype=torch.float32).reshape(1, 9, 14)
        captured = []
        handle = conditioner.projection[0].register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
        conditioner(camera, torch.tensor([True]), torch.zeros(1))
        handle.remove()
        self.assertEqual(captured[0].shape, (1, 4, 43))
        for slot in range(4):
            torch.testing.assert_close(captured[0][0, slot, :14], camera[0, 0])
            torch.testing.assert_close(captured[0][0, slot, 14:28], camera[0, 1+slot*2])
            torch.testing.assert_close(captured[0][0, slot, 28:42], camera[0, 2+slot*2])
            self.assertEqual(captured[0][0, slot, -1], 1.)

    def test_gradients_participate_even_when_all_camera_absent(self):
        model = self.model()
        model.checkpoint_blocks = True
        with torch.no_grad():
            model.head.weight.normal_(std=.1)
        x, u, c = self.inputs()
        for absent in (True, False):
            model.zero_grad(set_to_none=True)
            kw = {} if absent else dict(camera=torch.randn(2, 9, 14),
                                        camera_present=torch.tensor([True, False]))
            with torch.autocast('cpu', dtype=torch.bfloat16):
                result = model(x, u, c, **kw)
            result.square().mean().backward()
            for name, param in model.camera_conditioner.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertGreater(float(model.camera_conditioner.projection[-1].bias.grad.abs().sum()), 0.)

    def test_invalid_camera_contracts_fail(self):
        model = self.model()
        x, u, c = self.inputs()
        camera = torch.randn(2, 9, 14)
        present = torch.tensor([True, False])
        bad = [dict(camera=camera), dict(camera_present=present),
               dict(camera=camera[:, :-1], camera_present=present),
               dict(camera=camera[:, :, :-1], camera_present=present),
               dict(camera=camera, camera_present=present[:, None]),
               dict(camera=camera, camera_present=present.float())]
        for kw in bad:
            with self.subTest(keys=list(kw)):
                with self.assertRaises(ValueError):
                    model(x, u, c, **kw)
        camera[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            model(x, u, c, camera=camera, camera_present=present)
        with self.assertRaisesRegex(ValueError, 'disabled'):
            self.model(0)(x, u, c, camera=torch.zeros_like(camera), camera_present=present)
        with self.assertRaises(ValueError):
            self.model(12)

    def test_existing_positional_return_aux_stays_compatible(self):
        model = self.model(aux_layer=1)
        x, u, c = self.inputs()
        result, auxiliary = model(x, u, c, None, None, True,
                                  torch.zeros(2, 9, 14), torch.ones(2, dtype=torch.bool))
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(auxiliary.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
