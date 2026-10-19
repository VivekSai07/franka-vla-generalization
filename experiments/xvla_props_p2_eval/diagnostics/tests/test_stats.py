import unittest

import numpy as np

from experiments.xvla_props_p2_eval.diagnostics import stats


class ConcentrationTests(unittest.TestCase):
    def test_statistic_is_the_sum_of_squared_layout_counts(self):
        self.assertEqual(stats.concentration({'a': 3, 'b': 0, 'c': 1}), 10)

    def test_failures_packed_into_few_layouts_are_concentrated(self):
        failed = {f'L{i}': [False, False] for i in range(10)}
        for k in ('L0', 'L1', 'L2'):
            failed[k] = [True, True]
        self.assertLess(stats.concentration_p(failed), 0.05)

    def test_failures_spread_one_per_layout_are_not(self):
        failed = {f'L{i}': [False, False] for i in range(10)}
        for k in ('L0', 'L1', 'L2', 'L3', 'L4', 'L5'):
            failed[k] = [True, False]
        self.assertGreater(stats.concentration_p(failed), 0.5)

    def test_seed_is_deterministic(self):
        failed = {f'L{i}': [i < 4, i < 2] for i in range(10)}
        self.assertEqual(stats.concentration_p(failed, n=2000, seed=3), stats.concentration_p(failed, n=2000, seed=3))

    def test_no_failures_is_not_significant(self):
        self.assertEqual(stats.concentration_p({'a': [False, False], 'b': [False, False]}), 1.0)


class MannWhitneyTests(unittest.TestCase):
    def test_separated_groups(self):
        self.assertLess(stats.mann_whitney_p([1, 2, 3], [10, 11, 12, 13]), 0.1)

    def test_identical_groups(self):
        self.assertGreater(stats.mann_whitney_p([5, 6, 7], [5, 6, 7]), 0.9)

    def test_small_groups_are_exact(self):
        # 3 against 4 values: 35 equally likely rank sets; only the two extreme sets reach the observed gap.
        self.assertAlmostEqual(stats.mann_whitney_p([1, 2, 3], [10, 11, 12, 13]), 2 / 35, places=12)

    def test_exact_p_is_symmetric_and_seed_free(self):
        a, b = [1.0, 2.0, 9.0], [3.0, 4.0, 5.0, 6.0]
        self.assertEqual(stats.mann_whitney_p(a, b, seed=1), stats.mann_whitney_p(a, b, seed=2))
        self.assertAlmostEqual(stats.mann_whitney_p(a, b), stats.mann_whitney_p(b, a), places=12)

    def test_ties_use_average_ranks(self):
        self.assertAlmostEqual(stats.mann_whitney_p([1, 1, 1], [1, 1, 1, 1]), 1.0, places=12)

    def test_large_groups_are_monte_carlo_but_deterministic_and_nearly_symmetric(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(0.5, 1, 14), rng.normal(0, 1, 30)
        self.assertEqual(stats.mann_whitney_p(a, b, seed=1), stats.mann_whitney_p(a, b, seed=1))
        self.assertLess(abs(stats.mann_whitney_p(a, b, seed=1) - stats.mann_whitney_p(b, a, seed=2)), 0.03)

    def test_empty_group_is_refused(self):
        with self.assertRaises(ValueError):
            stats.mann_whitney_p([], [1, 2])


if __name__ == '__main__':
    unittest.main()
