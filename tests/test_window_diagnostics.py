import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_window_diffusion import ARMS, condition
from utils.window_flow import WindowFlow


class DiagnosticTests(unittest.TestCase):
    def test_paired_noise_and_condition_effect(self):
        anchor = torch.ones(1, 1, 4, 2)
        donor = anchor*3
        noise = torch.randn(1, 2, 4, 2)
        original = noise.clone()
        def oracle(x, u, c, text, valid):
            return c.expand_as(x)
        results = {}
        for arm, method, steps, mode in ARMS:
            results[arm] = WindowFlow().sample(oracle, condition(anchor, donor, mode), noise,
                                               steps=steps, method=method)
        for arm in ('euler64', 'euler128', 'heun64'):
            torch.testing.assert_close(results[arm], anchor.expand_as(noise))
        torch.testing.assert_close(results['zero'], torch.zeros_like(noise))
        torch.testing.assert_close(results['shuffle'], donor.expand_as(noise))
        self.assertTrue(torch.equal(noise, original))
        self.assertTrue(torch.equal(anchor, torch.ones_like(anchor)))
        with self.assertRaises(ValueError): condition(anchor, donor, 'invalid')


if __name__ == '__main__': unittest.main()
