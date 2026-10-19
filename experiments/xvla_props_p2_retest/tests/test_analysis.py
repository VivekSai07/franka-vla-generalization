import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from experiments.xvla_props_p2_retest import analysis


def row(pair, arm, ok, obj='milk', cause=None, side='left', count=1):
    return dict(layout_id=f'retest_{obj}_{pair:03d}_{arm}', object=obj, task_success=ok, blue_slot=side, distractor_count=count,
                cause='success' if ok else (cause or 'never_lifted'))


def pair_rows(spec, obj='milk', start=0, **kw):
    """spec: list of (trained_ok, held_ok)"""
    out = []
    for i, (t, h) in enumerate(spec, start=start):
        out += [row(i, 'trained', t, obj, **kw), row(i, 'held', h, obj, **kw)]
    return out


class PairingTests(unittest.TestCase):
    def test_pairs_join_the_two_arms_by_pair_id_in_any_row_order(self):
        rows = pair_rows([(True, False), (False, True)])
        rows = rows[::-1]
        pairs, dropped = analysis.pairs_from_rows(rows)
        self.assertEqual(dropped, [])
        self.assertEqual([(p['pair_id'], p['trained'], p['held'], p['d']) for p in pairs],
                         [('retest_milk_000', True, False, 1), ('retest_milk_001', False, True, -1)])

    def test_a_missing_arm_is_an_error_not_a_smaller_n(self):
        rows = pair_rows([(True, True), (True, False)])[:-1]
        with self.assertRaises(ValueError):
            analysis.pairs_from_rows(rows)

    def test_an_excluded_arm_drops_the_whole_pair_and_says_so(self):
        rows = pair_rows([(True, True), (True, False), (False, False)])
        pairs, dropped = analysis.pairs_from_rows(rows, excluded=['retest_milk_001_held'])
        self.assertEqual([p['pair_id'] for p in pairs], ['retest_milk_000', 'retest_milk_002'])
        self.assertEqual(dropped, ['retest_milk_001'])

    def test_a_duplicated_arm_row_is_an_error_not_a_silent_overwrite(self):
        rows = pair_rows([(True, False)]) + [row(0, 'held', True)]       # the held-out arm of pair 0 appears twice, differently
        with self.assertRaises(ValueError):
            analysis.pairs_from_rows(rows)

    def test_unknown_arm_suffix_is_an_error(self):
        with self.assertRaises(ValueError):
            analysis.pairs_from_rows([row(0, 'other', True)])


class PenaltyTests(unittest.TestCase):
    def pairs(self, spec_by_object):
        rows = []
        for obj, spec in spec_by_object.items():
            rows += pair_rows(spec, obj)
        return analysis.pairs_from_rows(rows)[0]

    def test_point_estimate_is_the_mean_of_trained_minus_held_over_all_pairs(self):
        # milk: d = +1, +1, 0, -1 (sum 1); bread: d = +1, 0 (sum 1): 2 / 6
        p = self.pairs({'milk': [(True, False), (True, False), (True, True), (False, True)], 'bread': [(True, False), (False, False)]})
        self.assertAlmostEqual(analysis.penalty(p, n=200)['estimate'], 2/6)

    def test_all_pairs_identical_gives_a_degenerate_interval(self):
        p = self.pairs({'milk': [(True, False)]*10, 'bread': [(True, False)]*10})
        out = analysis.penalty(p, n=500)
        self.assertEqual((out['estimate'], out['lo'], out['hi']), (1.0, 1.0, 1.0))

    def test_resampling_is_stratified_by_object(self):
        # milk d = +1 always, bread d = -1 always, equal n: any stratified resample gives exactly 0; an unstratified one does not
        p = self.pairs({'milk': [(True, False)]*10, 'bread': [(False, True)]*10})
        out = analysis.penalty(p, n=2000)
        self.assertEqual((out['estimate'], out['lo'], out['hi']), (0.0, 0.0, 0.0))

    def test_resampling_is_over_pairs_so_arms_stay_together(self):
        # every pair has d = 0 (both succeed or both fail); independent resampling of the two arms would give d != 0
        p = self.pairs({'milk': [(True, True)]*5 + [(False, False)]*5, 'bread': [(True, True)]*5 + [(False, False)]*5})
        out = analysis.penalty(p, n=2000)
        self.assertEqual((out['estimate'], out['lo'], out['hi']), (0.0, 0.0, 0.0))

    def test_interval_brackets_a_noisy_estimate_and_is_deterministic(self):
        spec = [(True, False)]*6 + [(False, True)]*2 + [(True, True)]*6 + [(False, False)]*6
        p = self.pairs({'milk': spec, 'bread': spec})
        a, b = analysis.penalty(p, n=3000, seed=0), analysis.penalty(p, n=3000, seed=0)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a['estimate'], 4/20)
        self.assertLess(a['lo'], a['estimate'])
        self.assertGreater(a['hi'], a['estimate'])
        self.assertGreater(a['lo'], -0.3)
        self.assertLess(a['hi'], 0.7)

    def test_interval_is_the_2p5_to_97p5_percentile_of_the_resampled_means(self):
        # one object, 4 pairs with d = 1, 1, 1, 0: resampled mean = k/4 with k ~ Binomial(4, 3/4) summing d; the exact distribution is known
        p = self.pairs({'milk': [(True, False)]*3 + [(True, True)]})
        out = analysis.penalty(p, n=20000, seed=3)
        # P(mean=0) = 0.25^4 = 0.0039 < 2.5%, so the 2.5th percentile is 0.25 (P(mean<=0.25) = 0.0039+4*.75*.25^3 = 0.0508)
        self.assertEqual(out['lo'], 0.25)
        self.assertEqual(out['hi'], 1.0)

    def test_per_object_estimates_are_reported_with_their_own_n(self):
        p = self.pairs({'milk': [(True, False)]*4, 'bread': [(True, True)]*2})
        out = analysis.penalty(p, n=100)
        self.assertEqual(out['per_object']['milk'], dict(n=4, estimate=1.0))
        self.assertEqual(out['per_object']['bread'], dict(n=2, estimate=0.0))
        self.assertEqual(out['n_pairs'], 6)


class ReadingTests(unittest.TestCase):
    def test_the_four_outcomes(self):
        self.assertEqual(analysis.reading(0.02, 0.08), 'exists_and_small')
        self.assertEqual(analysis.reading(0.05, 0.20), 'exists')
        self.assertEqual(analysis.reading(-0.04, 0.06), 'small')
        self.assertEqual(analysis.reading(-0.10, 0.15), 'inconclusive')

    def test_both_boundaries_are_strict(self):
        self.assertEqual(analysis.reading(0.0, 0.2), 'inconclusive', 'lower bound exactly 0 does not exclude 0')
        self.assertEqual(analysis.reading(0.01, 0.10), 'exists', 'upper bound exactly 10 points is not below 10 points')
        self.assertEqual(analysis.reading(0.0, 0.0999), 'small')


class TablesTests(unittest.TestCase):
    def setUp(self):
        rows = (pair_rows([(True, False), (True, True)], 'milk', side='left', count=1) +
                pair_rows([(False, True), (False, False)], 'milk', start=2, side='right', count=2) +
                pair_rows([(True, False)], 'bread', start=0, side='left', count=2))
        self.pairs, _ = analysis.pairs_from_rows(rows)

    def test_pair_table_counts_the_four_outcomes_per_object_and_overall(self):
        t = analysis.pair_table(self.pairs)
        self.assertEqual(t['milk'], dict(both=1, only_trained=1, only_held=1, neither=1))
        self.assertEqual(t['bread'], dict(both=0, only_trained=1, only_held=0, neither=0))
        self.assertEqual(t['all'], dict(both=1, only_trained=2, only_held=1, neither=1))

    def test_penalty_by_side_and_by_distractor_count(self):
        by = analysis.penalty_by(self.pairs)
        self.assertEqual(by['blue_slot']['left'], dict(n=3, estimate=2/3))        # d = 1, 0, 1
        self.assertEqual(by['blue_slot']['right'], dict(n=2, estimate=-1/2))      # d = -1, 0
        self.assertEqual(by['distractor_count']['1'], dict(n=2, estimate=1/2))    # d = 1, 0
        self.assertEqual(by['distractor_count']['2'], dict(n=3, estimate=0/3))    # d = -1, 0, 1

    def test_failure_causes_per_arm(self):
        rows = [row(0, 'trained', False, cause='timeout_carrying'), row(0, 'held', False, cause='wrong_bin'),
                row(1, 'trained', True), row(1, 'held', False, cause='never_lifted')]
        pairs, _ = analysis.pairs_from_rows(rows)
        c = analysis.causes(pairs)
        self.assertEqual(c['trained'], {'timeout_carrying': 1, 'success': 1})
        self.assertEqual(c['held'], {'wrong_bin': 1, 'never_lifted': 1})


if __name__ == '__main__':
    unittest.main()
