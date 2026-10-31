import unittest
from math import comb

import numpy as np

from experiments.vla_inputs_eval import stats


def rows_of(spec):
    """spec: {object: [(new, control), ...]} -> rows."""
    rows = []
    for obj, pairs in spec.items():
        for i, (new, control) in enumerate(pairs):
            rows.append(dict(layout_id=f"{obj}_{i}", object=obj, new=new, control=control))
    return rows


class McNemar(unittest.TestCase):
    def test_the_recorded_p2_result(self):
        # P2 report tier, checkpoint 101,520 against 126,900: 15 and 8 discordant trials
        self.assertAlmostEqual(stats.mcnemar(15, 8), 0.2100, places=3)

    def test_matches_a_direct_binomial_sum(self):
        for b, c in ((0, 5), (3, 9), (12, 12), (1, 0)):
            n = b + c
            want = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n)
            self.assertAlmostEqual(stats.mcnemar(b, c), want, places=12)

    def test_no_discordant_pairs_is_one(self):
        self.assertEqual(stats.mcnemar(0, 0), 1.0)

    def test_symmetric_and_two_sided(self):
        self.assertEqual(stats.mcnemar(3, 9), stats.mcnemar(9, 3))
        self.assertAlmostEqual(stats.mcnemar(0, 6), 2 / 64, places=12)  # one-sided would be 1/64
        self.assertEqual(stats.mcnemar(5, 5), 1.0)


class PairedEffect(unittest.TestCase):
    def test_every_layout_better_gives_one_one_one(self):
        rows = rows_of({"a": [(1, 0)] * 10, "b": [(1, 0)] * 10})
        self.assertEqual(stats.paired_effect(rows), {"estimate": 1.0, "lo": 1.0, "hi": 1.0})

    def test_estimate_is_the_mean_paired_difference(self):
        rows = rows_of({"a": [(1, 0), (0, 1), (1, 1), (0, 0)], "b": [(1, 0), (1, 0)]})
        self.assertAlmostEqual(stats.paired_effect(rows, n=200)["estimate"], 2 / 6)

    def test_estimate_does_not_depend_on_the_resample(self):
        rows = rows_of({"a": [(1, 0)] * 3 + [(0, 0)] * 9, "b": [(0, 1)] * 2 + [(0, 0)] * 8})
        for seed in range(5):
            self.assertAlmostEqual(stats.paired_effect(rows, n=1, seed=seed)["estimate"], (3 - 2) / 22)

    def test_resampling_keeps_per_object_counts(self):
        rows = rows_of({"a": [(1, 0)] * 7, "b": [(0, 1)] * 3, "c": [(1, 1)] * 5})
        rng = np.random.default_rng(1)
        for _ in range(20):
            sample = stats.bootstrap_sample(rows, rng)
            self.assertEqual({o: sum(r["object"] == o for r in sample) for o in "abc"}, {"a": 7, "b": 3, "c": 5})

    def test_stratified_resampling_does_not_wobble_a_fixed_mix(self):
        # object a always +1 (10 layouts), object b always 0 (90): stratified, the mean difference is exactly 0.1
        rows = rows_of({"a": [(1, 0)] * 10, "b": [(1, 1)] * 90})
        result = stats.paired_effect(rows, n=500)
        self.assertEqual((result["lo"], result["hi"]), (0.1, 0.1))

    def test_interval_brackets_the_estimate_and_is_deterministic(self):
        rng = np.random.default_rng(5)
        rows = rows_of({o: [(int(rng.integers(2)), int(rng.integers(2))) for _ in range(40)] for o in "abc"})
        first = stats.paired_effect(rows, n=2000, seed=0)
        self.assertEqual(first, stats.paired_effect(rows, n=2000, seed=0))
        self.assertLess(first["lo"], first["estimate"] + 1e-12)
        self.assertGreater(first["hi"], first["estimate"] - 1e-12)
        self.assertGreater(first["hi"] - first["lo"], 0.05)
        self.assertNotEqual(first, stats.paired_effect(rows, n=2000, seed=1))


class Holm(unittest.TestCase):
    def test_hand_computed_three_value_fixture(self):
        got = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(got["a"], 0.03)   # 0.01 * 3
        self.assertAlmostEqual(got["c"], 0.06)   # 0.03 * 2
        self.assertAlmostEqual(got["b"], 0.06)   # 0.04 * 1, raised to the running maximum

    def test_capped_at_one_and_monotone(self):
        got = stats.holm({"a": 0.5, "b": 0.9, "c": 0.7})
        self.assertTrue(all(v <= 1.0 for v in got.values()))
        self.assertEqual(got["a"], 1.0)

    def test_the_family_enters_unrun_arms_as_one(self):
        got = stats.family_holm({"A1": 0.004})
        self.assertEqual(sorted(got), ["A1", "A2", "A3", "A4"])
        self.assertAlmostEqual(got["A1"], 0.016)  # 0.004 x 4
        self.assertEqual({k: v for k, v in got.items() if k != "A1"}, {"A2": 1.0, "A3": 1.0, "A4": 1.0})

    def test_an_arm_outside_the_family_is_refused(self):
        with self.assertRaises(ValueError):
            stats.family_holm({"A5": 0.01})


class Select(unittest.TestCase):
    def test_most_successes_wins(self):
        self.assertEqual(stats.select({50760: 10, 76140: 14, 101520: 12}), 76140)

    def test_a_tie_goes_to_the_later_checkpoint_whatever_the_order(self):
        self.assertEqual(stats.select({101520: 19, 126900: 19}), 126900)
        self.assertEqual(stats.select({126900: 19, 101520: 19}), 126900)

    def test_ties_are_listed(self):
        self.assertEqual(stats.tied({101520: 19, 126900: 19, 88830: 17}), [101520, 126900])
        self.assertEqual(stats.tied({1: 3, 2: 4}), [2])

    def test_empty_is_refused(self):
        with self.assertRaises(ValueError):
            stats.select({})


if __name__ == "__main__":
    unittest.main()
