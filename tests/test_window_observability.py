import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.window_flow import WindowFlow
from utils.window_observability import NoiseMeter


class ObservabilityTests(unittest.TestCase):
    def test_meter_preserves_rng_gradients_and_objective(self):
        for mode in ('x0', 'velocity'):
            flow = WindowFlow(mode)
            u = torch.tensor([.01,.1,.3,.5,.7,.9,.99])
            y, noise = torch.randn(7,2,3,4), torch.randn(7,2,3,4)
            out = torch.randn_like(y).requires_grad_()
            expected = float(flow.loss(out,y,noise,u).detach())
            rng = torch.get_rng_state().clone()
            meter = NoiseMeter('cpu'); meter.update(flow,out,y,noise,u)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertIsNone(out.grad)
            metrics = meter.flush()
            actual = sum(metrics[f'train/noise_bin{i}/objective']*metrics[f'train/noise_bin{i}/count'] for i in range(5))/len(u)
            self.assertAlmostEqual(actual,expected,delta=max(1e-5,expected*1e-6))
            self.assertEqual(float(meter.sums.sum()),0)

    def test_shift_is_training_distribution_only(self):
        torch.manual_seed(123)
        a = WindowFlow(time_shift=1).times(100000,'cpu')
        torch.manual_seed(123)
        b = WindowFlow(time_shift=3).times(100000,'cpu')
        torch.testing.assert_close(b,3*a/(1+2*a))
        self.assertAlmostEqual(float((b>.9).float().mean()),.13597,delta=.005)
        self.assertAlmostEqual(float((a>.9).float().mean()),.01400,delta=.002)
        noise = torch.randn(1,2,3,4); anchor = torch.randn(1,1,3,4)
        def oracle(x,u,c,text,valid): return c.expand_as(x)
        torch.testing.assert_close(WindowFlow(time_shift=1).sample(oracle,anchor,noise),
                                   WindowFlow(time_shift=3).sample(oracle,anchor,noise))

    def test_latent_moments_merge_batches(self):
        x = torch.arange(48).reshape(2,2,3,4).float()/10
        meter = NoiseMeter('cpu')
        meter.observe_latents('target',x[:1]); meter.observe_latents('target',x[1:])
        result = meter.flush()
        std = x.permute(1,3,0,2).flatten(2).std(-1,unbiased=False)
        self.assertAlmostEqual(result['train/normalized_target/std_median'],float(std.median()),places=5)
        self.assertAlmostEqual(result['train/normalized_target/mean_abs_average'],float(x.mean((0,2)).abs().mean()),places=5)


if __name__ == '__main__': unittest.main()
