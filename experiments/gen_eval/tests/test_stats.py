import unittest

import numpy as np

from experiments.gen_eval import stats


def rows(spec):
    """spec: {object: [(nominal, perturbed), ...]}."""
    return [dict(layout_id=f"{o}{i}", object=o, nominal=n, perturbed=p) for o, pairs in spec.items() for i, (n, p) in enumerate(pairs)]


class McNemar(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(stats.mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(stats.mcnemar(0, 5), 2 / 32)
        self.assertAlmostEqual(stats.mcnemar(1, 9), 2 * (1 + 10) / 1024)
        self.assertEqual(stats.mcnemar(3, 3), 1.0)
        self.assertEqual(stats.mcnemar(2, 7), stats.mcnemar(7, 2))

    def test_pair_counts(self):
        r = rows({"can": [(1, 1), (1, 0), (1, 0), (0, 1), (0, 0)]})
        self.assertEqual(stats.pair_counts(r), (1, 2, 1, 1))


class PairedDrop(unittest.TestCase):
    def test_a_positive_drop_is_a_loss(self):
        d = stats.paired_drop(rows({"can": [(1, 0)] * 10}))
        self.assertEqual((d["estimate"], d["lo"], d["hi"]), (1.0, 1.0, 1.0))
        d = stats.paired_drop(rows({"can": [(0, 1)] * 10}))
        self.assertEqual(d["estimate"], -1.0)

    def test_the_estimate_is_the_mean_over_layouts(self):
        r = rows({"can": [(1, 1)] * 6 + [(1, 0)] * 2, "milk": [(1, 0)] * 2})
        self.assertAlmostEqual(stats.paired_drop(r)["estimate"], 4 / 10)

    def test_resampling_stays_inside_each_object(self):
        # one object loses every pair, the other none: stratified resampling cannot change the mean, an unstratified one would
        r = rows({"can": [(1, 0)] * 10, "milk": [(1, 1)] * 10})
        d = stats.paired_drop(r)
        self.assertEqual((d["lo"], d["hi"]), (0.5, 0.5))

    def test_deterministic_and_seeded(self):
        r = rows({"can": [(1, 0)] * 17 + [(1, 1)] * 23, "milk": [(1, 0)] * 7 + [(0, 0)] * 23})
        self.assertEqual(stats.paired_drop(r), stats.paired_drop(r))
        self.assertEqual((stats.BOOTSTRAP_N, stats.BOOTSTRAP_SEED), (10_000, 0))
        seen = []
        real = np.random.default_rng
        np.random.default_rng = lambda seed=None: seen.append(seed) or real(seed)
        try:
            stats.paired_drop(r)
            stats.paired_drop(r, seed=7)
        finally:
            np.random.default_rng = real
        self.assertEqual(seen, [0, 7])

    def test_interval_brackets_a_noisy_estimate_and_is_a_95_percent_interval(self):
        r = rows({"can": [(1, 0)] * 5 + [(1, 1)] * 45})
        d = stats.paired_drop(r)
        self.assertAlmostEqual(d["estimate"], 0.1)
        self.assertTrue(d["lo"] < 0.1 < d["hi"])
        rng = np.random.default_rng(0)
        draws = rng.integers(0, 50, size=(10_000, 50))
        values = np.array([1.0] * 5 + [0.0] * 45)[draws].mean(axis=1)
        self.assertAlmostEqual(d["lo"], float(np.percentile(values, 2.5)))
        self.assertAlmostEqual(d["hi"], float(np.percentile(values, 97.5)))

    def test_empty(self):
        self.assertEqual(stats.paired_drop([]), {"estimate": None, "lo": None, "hi": None})
        self.assertIsNone(stats.held(stats.paired_drop([])))


class Held(unittest.TestCase):
    """The pre-registered rule: held iff the drop's 95% interval upper bound is at most 10 points."""

    def test_the_threshold_is_ten_points_and_inclusive(self):
        self.assertEqual(stats.HELD_MAX_DROP, 0.10)
        self.assertTrue(stats.held({"estimate": 0.0, "lo": -0.1, "hi": 0.10}))
        self.assertFalse(stats.held({"estimate": 0.0, "lo": -0.1, "hi": 0.1001}))
        self.assertTrue(stats.held({"estimate": -0.3, "lo": -0.5, "hi": -0.1}))

    def test_a_point_estimate_under_ten_points_is_not_enough(self):
        self.assertFalse(stats.held({"estimate": 0.05, "lo": -0.02, "hi": 0.14}))

    def test_a_clean_run_is_held_and_a_collapse_is_not(self):
        self.assertTrue(stats.held(stats.paired_drop(rows({"can": [(1, 1)] * 50}))))
        self.assertFalse(stats.held(stats.paired_drop(rows({"can": [(1, 0)] * 30 + [(1, 1)] * 20}))))


class Holm(unittest.TestCase):
    def test_step_down(self):
        out = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5, "e": 0.9})
        self.assertAlmostEqual(out["a"], 0.05)
        self.assertAlmostEqual(out["c"], 0.12)
        self.assertAlmostEqual(out["b"], 0.12)
        self.assertAlmostEqual(out["d"], 1.0)
        self.assertEqual(out["e"], 1.0)

    def test_monotone_and_capped(self):
        out = stats.holm({"a": 0.2, "b": 0.3})
        self.assertAlmostEqual(out["a"], 0.4)
        self.assertAlmostEqual(out["b"], 0.4)


class Selection(unittest.TestCase):
    def test_most_successes_and_ties_go_to_the_later_checkpoint(self):
        self.assertEqual(stats.select({50760: 10, 88830: 12, 126900: 12}), 126900)
        self.assertEqual(stats.tied({50760: 10, 88830: 12, 126900: 12}), [88830, 126900])
        self.assertEqual(stats.select({1: 5}), 1)
        with self.assertRaises(ValueError):
            stats.select({})


if __name__ == "__main__":
    unittest.main()
