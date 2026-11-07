"""A real velocity write must not be mislabeled as successful joint solving."""
import json
import unittest

from refine_vace_edit_state import feedback_health
from utils.editing_state_feedback import solver_evidence


def event(before=1., after=1., applied=True, fallback=False, delta=0., **extra):
    solver = dict(before=dict(objective=before), after=dict(objective=after),
        objective_improved=True, actual_state_delta_rms=delta,
        accepted_steps=2 if after < before else 0, rejected_steps=3,
        trial_evaluations=7, backtracks=4)
    solver.update(extra)
    return dict(applied=applied, fallback=fallback, solver=solver)


class FeedbackReportingTests(unittest.TestCase):
    def test_applied_unchanged_objective_is_not_an_optimization_success(self):
        value = event()
        stats = feedback_health([dict(id='unchanged', feedback_events=[value]*3)])
        self.assertEqual(stats['applied'], 3)
        self.assertEqual(stats['solver_improved_events'], 0)
        self.assertEqual(stats['committed_improved_events'], 0)
        self.assertEqual(stats['committed_new_state_events'], 0)
        self.assertEqual(stats['applied_without_objective_improvement'], 3)
        self.assertEqual(stats['cases_without_committed_improvement'], ['unchanged'])
        self.assertEqual(stats['cases_without_applied_feedback'], [])

    def test_vae_failure_does_not_count_candidate_optimization_as_committed(self):
        failed = event(after=.8, applied=False, fallback=True, delta=.1)
        succeeded = event(after=.7, delta=.2)
        rows = [dict(id='rollback', feedback_events=[failed]),
                dict(id='success', feedback_events=[succeeded])]
        stats = feedback_health(rows)
        self.assertEqual(stats['solver_improved_events'], 2)
        self.assertEqual(stats['committed_improved_events'], 1)
        self.assertEqual(stats['committed_new_state_events'], 1)
        self.assertEqual(stats['accepted_block_steps'], 2)
        self.assertEqual(stats['trial_evaluations'], 14)
        self.assertEqual(stats['cases_without_committed_improvement'], ['rollback'])

    def test_absent_nonfinite_or_solver_fallback_is_not_improvement(self):
        for diag in ({}, dict(before=dict(objective=float('nan')), after=dict(objective=0.)),
                     dict(before=dict(objective=1.), after=dict(objective=.1), fallback=True)):
            value = solver_evidence(diag)
            self.assertFalse(value['solver_objective_improved'])
            json.dumps(value, allow_nan=False)

    def test_legacy_counts_are_unknown_and_objective_readout_remains_usable(self):
        value = dict(applied=True, solver=dict(before=dict(objective=2.), after=dict(objective=1.)))
        stats = feedback_health([dict(id='legacy', feedback_events=[value])])
        self.assertEqual(stats['committed_improved_events'], 1)
        self.assertEqual(stats['committed_new_state_events'], 0)
        self.assertIsNone(stats['trial_evaluations'])
        self.assertIsNone(stats['accepted_block_steps'])


if __name__ == '__main__':
    unittest.main()
