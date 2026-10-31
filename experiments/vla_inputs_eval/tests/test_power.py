import unittest

from experiments.vla_inputs_eval import power


class AgainstTheRecordedFigures(unittest.TestCase):
    def test_the_audit_enumeration_at_384_layouts(self):
        # audit R4 (exact enumeration): alpha .05 and .0125, N = 384
        for (d, delta, alpha), want in {(0.19, 0.06, 0.05): 0.743, (0.30, 0.06, 0.05): 0.540, (0.40, 0.06, 0.05): 0.429,
                                        (0.19, 0.10, 0.05): 0.996, (0.30, 0.10, 0.05): 0.943, (0.40, 0.10, 0.05): 0.861,
                                        (0.19, 0.06, 0.0125): 0.540, (0.30, 0.06, 0.0125): 0.330, (0.40, 0.06, 0.0125): 0.238,
                                        (0.19, 0.10, 0.0125): 0.980, (0.30, 0.10, 0.0125): 0.851, (0.40, 0.10, 0.0125): 0.708}.items():
            self.assertAlmostEqual(power.exact_power(384, d, delta, alpha), want, delta=0.0015, msg=(d, delta, alpha))

    def test_the_size_decision_table_at_576_layouts(self):
        for (d, delta, alpha), want in {(0.30, 0.10, 0.0125): 0.970, (0.40, 0.10, 0.0125): 0.897, (0.30, 0.10, 0.05): 0.992,
                                        (0.40, 0.10, 0.05): 0.964}.items():
            self.assertAlmostEqual(power.exact_power(576, d, delta, alpha), want, delta=0.0015, msg=(d, delta, alpha))


class Properties(unittest.TestCase):
    def test_with_no_true_difference_the_rejection_rate_does_not_exceed_alpha(self):
        for alpha in (0.05, 0.0125):
            self.assertLessEqual(power.exact_power(200, 0.3, 0.0, alpha), alpha)

    def test_power_grows_with_layouts_and_with_the_effect(self):
        self.assertLess(power.exact_power(100, 0.3, 0.1, 0.05), power.exact_power(300, 0.3, 0.1, 0.05))
        self.assertLess(power.exact_power(300, 0.3, 0.06, 0.05), power.exact_power(300, 0.3, 0.1, 0.05))

    def test_a_smaller_alpha_costs_power(self):
        self.assertLess(power.exact_power(300, 0.3, 0.1, 0.0125), power.exact_power(300, 0.3, 0.1, 0.05))

    def test_an_effect_larger_than_the_discordance_is_refused(self):
        with self.assertRaises(ValueError):
            power.exact_power(100, 0.1, 0.2, 0.05)

    def test_a_worse_arm_almost_never_wins_the_test(self):
        self.assertLess(power.exact_power(300, 0.3, -0.1, 0.05), 0.001)  # rejection needs the NEW arm ahead


class MonteCarloAgrees(unittest.TestCase):
    def test_the_simulation_matches_the_enumeration_within_four_standard_errors(self):
        exact = power.exact_power(150, 0.3, 0.1, 0.05)
        mc, se = power.monte_carlo_power(150, 0.3, 0.1, 0.05, reps=4000)
        self.assertLess(abs(mc - exact), 4 * se + 1e-9)
        self.assertGreater(se, 0)


if __name__ == "__main__":
    unittest.main()
