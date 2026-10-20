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


class Gate1NullTests(unittest.TestCase):
    def test_same_distribution_gives_a_null_well_below_one(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=(20, 4)), rng.normal(size=(20, 4))
        out = stats.gate1_null(a, b, n=300, seed=0)
        self.assertLess(out['forward'], 0.9)
        self.assertGreater(out['forward'], 0.3)

    def test_the_null_depends_only_on_the_pooled_values_and_group_sizes(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(size=(15, 4)), rng.normal(size=(25, 4))
        ab, ba = stats.gate1_null(a, b, n=400, seed=0), stats.gate1_null(b, a, n=400, seed=0)
        # exchanging the groups exchanges the directions; the pool is the same, so the means agree to sampling error
        self.assertAlmostEqual(ab['forward'], ba['backward'], delta=0.06)
        self.assertAlmostEqual(ab['backward'], ba['forward'], delta=0.06)
        self.assertEqual(ab, stats.gate1_null(a, b, n=400, seed=0), 'deterministic for a seed')

    def test_a_smaller_group_is_judged_by_a_noisier_and_narrower_envelope(self):
        rng = np.random.default_rng(3)
        pool = rng.normal(size=(200, 4))
        wide = stats.gate1_null(pool[:100], pool[100:], n=200, seed=0)
        small = stats.gate1_null(pool[:100], pool[100:110], n=200, seed=0)
        self.assertGreater(wide['forward'], small['forward'], 'an envelope from 10 samples covers less than one from 100')

    def test_a_clearly_different_pair_still_gets_the_same_null_as_a_matched_pair(self):
        # The null re-splits the pool at random, so it ignores how the two groups were actually split. If it used the
        # observed split, two far-apart groups would read 0.0, not the matched-pair value.
        rng = np.random.default_rng(4)
        base = rng.normal(size=(30, 4))
        matched = stats.gate1_null(base[:15], base[15:], n=300, seed=0)
        apart = stats.gate1_null(base[:15], base[15:] + 50.0, n=300, seed=0)
        self.assertGreater(apart['forward'], 0.2, 'observed split leaked into the null')
        self.assertGreater(apart['backward'], 0.2, 'observed split leaked into the null')
        self.assertLessEqual(matched['forward'], 1.0)

    def test_envelope_bounds_are_p1_and_p99_exactly(self):
        # Deterministic: 101 values 0..100 in every column give p1 = 1 and p99 = 99, never the min 0 or the max 100.
        x = np.tile(np.arange(101.0)[:, None], (1, 4))
        env = stats.percentile_envelope(x)
        np.testing.assert_allclose(env['lo'], [1.0]*4)
        np.testing.assert_allclose(env['hi'], [99.0]*4)

    def test_gate1_null_uses_that_envelope(self):
        # With a stubbed envelope that contains everything the null must read 1.0; with one that contains nothing, 0.0.
        x = np.random.default_rng(5).normal(size=(30, 4))
        everything = lambda s: dict(lo=np.full(4, -1e9), hi=np.full(4, 1e9))
        nothing = lambda s: dict(lo=np.full(4, 1e9), hi=np.full(4, 2e9))
        self.assertEqual(stats.gate1_null(x[:15], x[15:], n=20, seed=0, envelope=everything), dict(forward=1.0, backward=1.0))
        self.assertEqual(stats.gate1_null(x[:15], x[15:], n=20, seed=0, envelope=nothing), dict(forward=0.0, backward=0.0))

    def test_identical_tiny_samples_are_fully_inside_nothing_is_claimed(self):
        a = np.zeros((10, 4)); out = stats.gate1_null(a, a, n=50, seed=0)
        self.assertEqual((out['forward'], out['backward']), (1.0, 1.0))

    def test_inside_fraction_helper(self):
        env = dict(lo=np.zeros(4), hi=np.ones(4))
        x = np.array([[0.5]*4, [2.0, 0.5, 0.5, 0.5], [0, 0, 0, 0], [1, 1, 1, 1]])
        self.assertEqual(stats.inside_fraction(x, env), 0.75)


if __name__ == '__main__':
    unittest.main()
