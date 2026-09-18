import unittest
from utils.cache_failure_budget import failure_budget_exceeded


class CacheFailureBudgetTests(unittest.TestCase):
    def test_clustered_missing_group_does_not_reject_valid_full_dataset(self):
        # A 600/11000 prefix breaches 5%, but 600/60809 overall does not.
        self.assertFalse(failure_budget_exceeded(600,60809,.05))
        self.assertFalse(failure_budget_exceeded(3040,60809,.05))
        self.assertTrue(failure_budget_exceeded(3041,60809,.05))

    def test_resume_accumulates_failures_and_final_limit_is_strict(self):
        self.assertFalse(failure_budget_exceeded(50,1000,.05))
        self.assertTrue(failure_budget_exceeded(40+11,1000,.05))
        self.assertFalse(failure_budget_exceeded(0,1000,0))
        self.assertTrue(failure_budget_exceeded(1,1000,0))

    def test_invalid_counts_and_limits(self):
        for args in [(-1,100,.05),(101,100,.05),(1,100,1),(1,100,float('nan'))]:
            with self.assertRaises(ValueError):failure_budget_exceeded(*args)


if __name__=='__main__':unittest.main()
