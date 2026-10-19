"""Numerical contracts for motion emphasis, common probes and plateau resume."""
import copy
import json
import unittest

import torch

from utils.scene_motion_objective import motion_weighted_x0_loss, x0_error_metrics, PlateauTracker
from utils.window_flow import WindowFlow


class MotionObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(54)
        self.clean = torch.randn(2, 5, 6, 4)
        self.last = torch.randn(2, 1, 6, 4)
        self.output = torch.randn_like(self.clean)
        self.u = torch.tensor([.3, .9])

    def test_zero_strength_matches_original_loss_and_output_gradient_exactly(self):
        for dtype in (torch.float32, torch.bfloat16):
            a = self.output.detach().to(dtype).clone().requires_grad_()
            b = a.detach().clone().requires_grad_()
            original = WindowFlow().loss(a, self.clean, torch.zeros_like(a), self.u)
            candidate, metrics = motion_weighted_x0_loss(b, self.clean, self.last, self.u, strength=0)
            torch.testing.assert_close(candidate, original, rtol=0, atol=0)
            original.backward(); candidate.backward()
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            self.assertEqual(metrics['weight_min'], 1.)
            self.assertEqual(metrics['weight_max'], 1.)

    def test_weights_are_per_sample_mean_one_positive_capped_and_detached(self):
        clean = torch.zeros_like(self.clean).requires_grad_()
        last = self.last.clone().requires_grad_()
        with torch.no_grad():
            clean[0, 2:, 1] = 10000
            clean[1, :, 2] = .001
        prediction = self.output.clone().requires_grad_()
        loss, metrics = motion_weighted_x0_loss(prediction, clean, last, self.u, strength=20, cap=3)
        loss.backward()
        self.assertIsNone(clean.grad)
        self.assertIsNone(last.grad)
        self.assertGreater(prediction.grad.abs().sum().item(), 0)
        self.assertGreaterEqual(metrics['weight_min'], 1/3-1e-6)
        self.assertLessEqual(metrics['weight_max'], 3+1e-6)
        self.assertAlmostEqual(metrics['weight_mean'], 1., places=6)
        self.assertLess(metrics['weight_sample_mean_max_error'], 2e-7)

    def test_zero_change_is_uniform_and_empty_changed_region_reports_global(self):
        last = torch.randn(2, 1, 6, 4)
        clean = last.expand(-1, 5, -1, -1).clone()
        got, metrics = motion_weighted_x0_loss(self.output, clean, last, self.u)
        expected = WindowFlow().loss(self.output, clean, torch.zeros_like(clean), self.u)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        self.assertEqual(metrics['weight_min'], 1.)
        self.assertEqual(metrics['weight_max'], 1.)
        self.assertEqual(metrics['changed_count'], 0)
        self.assertTrue(metrics['changed_region_fallback_global'])
        self.assertEqual(metrics['x0_mse'], metrics['changed_region_x0_mse'])

    def test_loss_matches_changed_target_and_never_rewards_a_static_copy(self):
        last = torch.zeros(1, 1, 4, 2)
        clean = torch.zeros(1, 5, 4, 2)
        clean[:, :, 0] = torch.arange(1, 6).reshape(1, 5, 1)
        exact, _ = motion_weighted_x0_loss(clean, clean, last, torch.tensor([.5]))
        copied, metrics = motion_weighted_x0_loss(torch.zeros_like(clean), clean, last, torch.tensor([.5]))
        self.assertEqual(exact.item(), 0.)
        self.assertGreater(copied.item(), 0.)
        self.assertGreater(metrics['changed_region_x0_mse'], metrics['rest_region_x0_mse'])
        # Both preserve the same temporal deltas; RGB/latent absolute errors
        # still matter because this is weighted x0, not a smoothness objective.
        offset, _ = motion_weighted_x0_loss(clean+2, clean, last, torch.tensor([.5]))
        self.assertGreater(offset.item(), 0.)

    def test_noise_denominator_is_applied_once_including_floor(self):
        clean = torch.zeros(1, 5, 6, 4)
        last = torch.zeros(1, 1, 6, 4)
        output = torch.ones_like(clean)*2
        for u in (.25, .01):
            loss, _ = motion_weighted_x0_loss(output, clean, last, torch.tensor([u]))
            self.assertAlmostEqual(loss.item(), (2/max(u,.05))**2, places=3)

    def test_first_change_uses_last_observed_and_common_probe_does_not_depend_on_strength(self):
        clean = torch.zeros(1, 5, 3, 2)
        last = torch.zeros(1, 1, 3, 2); last[:, :, 1] = 2
        output = torch.ones_like(clean)
        common = x0_error_metrics(output, clean, last)
        self.assertEqual(common['changed_count'], 1)
        for strength in (0., 1., 2.):
            _, metrics = motion_weighted_x0_loss(output, clean, last, torch.tensor([.5]), strength=strength)
            for key, value in common.items():
                self.assertEqual(metrics[key], value)

    def test_no_rng_consumption_and_metrics_are_json_serializable(self):
        before = torch.get_rng_state().clone()
        _, metrics = motion_weighted_x0_loss(self.output, self.clean, self.last, self.u)
        x0_error_metrics(self.output, self.clean, self.last)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        json.dumps(metrics, allow_nan=False)

    def test_invalid_layout_or_configuration_is_rejected(self):
        for changes in ({'strength':-1}, {'cap':.5}, {'loss_floor':0}, {'activity_floor':0}):
            with self.assertRaises(ValueError):
                motion_weighted_x0_loss(self.output, self.clean, self.last, self.u, **changes)
        with self.assertRaises(ValueError):
            motion_weighted_x0_loss(self.output, self.clean, self.last[:, 0], self.u)


class PlateauTrackerTests(unittest.TestCase):
    def test_either_metric_resets_patience_and_minimum_steps_is_obeyed(self):
        tracker = PlateauTracker(min_steps=4000, patience=2, relative_tol=.002)
        values = dict(x0_mse=1., changed_region_x0_mse=2.)
        self.assertFalse(tracker.update(0, values)['converged'])
        tracker.update(1000, values)
        self.assertFalse(tracker.update(2000, values)['converged'])
        improved = dict(values, changed_region_x0_mse=1.9)
        row = tracker.update(3000, improved)
        self.assertEqual(row['stale'], 0)
        self.assertEqual(row['improved'], ['changed_region_x0_mse'])
        self.assertFalse(tracker.update(4000, improved)['converged'])
        self.assertTrue(tracker.update(5000, improved)['converged'])
        row = tracker.update(6000, dict(improved, x0_mse=.9))
        self.assertFalse(row['converged'])
        self.assertEqual(row['stale'], 0)

    def test_duplicate_missing_nonfinite_and_older_probes_do_not_advance_state(self):
        tracker = PlateauTracker()
        values = dict(x0_mse=1., changed_region_x0_mse=2.)
        tracker.update(1000, values)
        expected = copy.deepcopy(tracker.state_dict())
        for step, invalid in ((1000, values), (0, values), (2000, {}),
                (2000, dict(values, x0_mse=float('nan'))),
                (2000, dict(values, changed_region_x0_mse=float('inf')))):
            self.assertFalse(tracker.update(step, invalid)['updated'])
            self.assertEqual(tracker.state_dict(), expected)
        self.assertTrue(tracker.update(2000, values)['updated'])

    def test_minor_improvements_accumulate_and_resume_is_exact(self):
        one = PlateauTracker(min_steps=4000, patience=4, relative_tol=.002)
        one.update(0, dict(x0_mse=1., changed_region_x0_mse=2.))
        one.update(1000, dict(x0_mse=.999, changed_region_x0_mse=2.))
        self.assertEqual(one.best['x0_mse'], .999)
        self.assertEqual(one.stale, 1)
        two = PlateauTracker(min_steps=4000, patience=4, relative_tol=.002)
        two.load_state_dict(json.loads(json.dumps(one.state_dict())))
        for step, global_error in ((2000,.997), (3000,.997), (4000,.997), (5000,.997), (6000,.997)):
            metrics = dict(x0_mse=global_error, changed_region_x0_mse=2.)
            self.assertEqual(one.update(step, metrics), two.update(step, metrics))
            self.assertEqual(one.state_dict(), two.state_dict())
        self.assertTrue(one.converged)
        self.assertTrue(one.state_dict()['converged'])
        recovered = PlateauTracker(min_steps=4000, patience=4, relative_tol=.002)
        recovered.load_state_dict(json.loads(json.dumps(one.state_dict())))
        self.assertTrue(recovered.converged)
        self.assertEqual(recovered.state_dict(), one.state_dict())
        wrong = PlateauTracker(patience=3)
        with self.assertRaises(ValueError):
            wrong.load_state_dict(one.state_dict())


if __name__ == '__main__':
    unittest.main()
