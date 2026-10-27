"""Pure tests of the classifier, statistics, selection rules and the pre-registered criteria. No simulator."""
import json
import unittest

import numpy as np

from experiments.xvla_props_p3_eval import analysis as a

BIN = {'can': 'orange', 'milk': 'orange', 'bottle': 'orange', 'bread': 'blue', 'lemon': 'blue', 'cereal': 'blue'}
OTHER = {'orange': 'blue', 'blue': 'orange'}


def row(layout_id, obj, success=True, placed='correct', slot='left', cause=None):
    """A trial row. `placed`: 'correct', 'wrong' or None (not resting in a bin)."""
    bin_ = BIN[obj]
    placed_bin = {'correct': bin_, 'wrong': OTHER[bin_], None: None}[placed]
    if cause is None:
        cause = 'success' if success else ('never_lifted' if placed_bin is None else
                                           'wrong_bin' if placed == 'wrong' else 'unstable_in_target')
    return dict(layout_id=layout_id, object=obj, bin=bin_, blue_slot=slot, task_success=success, placed_bin=placed_bin,
                cause=cause, kind='held_out' if obj in a.HELD_OUT else 'trained')


def rows_for(obj, n, correct, stable=None, start=0, slot='left'):
    """n layouts of one object: `stable` of them rest in a bin (default all), `correct` of those in the right one."""
    stable = n if stable is None else stable
    out = []
    for i in range(n):
        placed = None if i >= stable else ('correct' if i < correct else 'wrong')
        out.append(row(f'{obj}_{start + i}', obj, success=placed == 'correct', placed=placed, slot=slot))
    return out


def trace(z, placed_bin=None, gripper_open=True):
    return dict(props_z=np.asarray(z, float), placed_bin=placed_bin, final_gripper_open=gripper_open)


LIFTED = [0.0] * 5 + [0.05] * 40
FLAT = [0.0] * 50


class LiftTests(unittest.TestCase):
    def test_lift_needs_the_height_and_the_sustained_ticks(self):
        self.assertEqual(a.lift_tick(LIFTED), 5)
        self.assertIsNone(a.lift_tick(FLAT))
        self.assertIsNone(a.lift_tick([]))
        bump = [0.0] * 5 + [0.05] * (a.SUSTAIN_TICKS - 1) + [0.0] * 20
        self.assertIsNone(a.lift_tick(bump), 'a bump shorter than the sustain window is not a lift')
        self.assertEqual(a.lift_tick([0.0] * 3 + [0.05] * a.SUSTAIN_TICKS), 3)
        self.assertIsNone(a.lift_tick([0.0] + [a.LIFT_M] * 20), 'exactly the threshold is not above it')
        self.assertEqual(a.lift_tick([0.0] + [a.LIFT_M + 1e-6] * 20), 1)


class ClassifyTests(unittest.TestCase):
    def test_success_wins(self):
        self.assertEqual(a.classify(trace(FLAT), 'blue', True), 'success')

    def test_never_lifted(self):
        self.assertEqual(a.classify(trace(FLAT), 'blue', False), 'never_lifted')
        self.assertEqual(a.classify(trace(FLAT, placed_bin='blue'), 'blue', False), 'never_lifted')

    def test_a_lifted_object_resting_in_a_bin_is_judged_by_which_bin_before_the_gripper(self):
        # FAILURES #53: the most specific condition first, whatever the gripper does.
        for gripper in (True, False):
            self.assertEqual(a.classify(trace(LIFTED, 'orange', gripper), 'blue', False), 'wrong_bin')
            self.assertEqual(a.classify(trace(LIFTED, 'blue', gripper), 'blue', False), 'unstable_in_target')

    def test_not_in_a_bin_is_carrying_or_dropped_by_the_gripper_state(self):
        self.assertEqual(a.classify(trace(LIFTED, None, False), 'blue', False), 'timeout_carrying')
        self.assertEqual(a.classify(trace(LIFTED, None, True), 'blue', False), 'dropped_outside')

    def test_every_cause_is_produced_and_there_is_no_wrong_object(self):
        produced = {a.classify(trace(FLAT), 'blue', True), a.classify(trace(FLAT), 'blue', False),
                    a.classify(trace(LIFTED, 'orange'), 'blue', False), a.classify(trace(LIFTED, 'blue'), 'blue', False),
                    a.classify(trace(LIFTED, None, False), 'blue', False), a.classify(trace(LIFTED, None, True), 'blue', False)}
        self.assertEqual(produced, set(a.CAUSES))
        self.assertNotIn('wrong_object', a.CAUSES)


class WilsonTests(unittest.TestCase):
    def test_known_values(self):
        lo, hi = a.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        lo, hi = a.wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=6)
        self.assertAlmostEqual(hi, 0.2775, places=3)
        lo, hi = a.wilson(30, 30)
        self.assertAlmostEqual(lo, 0.8865, places=3)
        self.assertAlmostEqual(hi, 1.0, places=6)

    def test_empty_is_none_and_the_interval_is_ordered_and_json_safe(self):
        self.assertEqual(a.wilson(0, 0), (None, None))
        json.dumps(a.bin_choice([]), allow_nan=False)
        json.dumps(a.readouts([], []), allow_nan=False)
        for n in (1, 7, 80):
            for k in range(n + 1):
                lo, hi = a.wilson(k, n)
                self.assertLessEqual(lo, k / n + 1e-12)
                self.assertGreaterEqual(hi, k / n - 1e-12)
                self.assertTrue(0 <= lo <= hi <= 1)
        self.assertEqual(a.wilson(0, 10)[0], 0.0)
        self.assertEqual(a.wilson(10, 10)[1], 1.0)


class BinChoiceTests(unittest.TestCase):
    def test_counts_are_over_stable_placements_only(self):
        rows = rows_for('can', 10, correct=6, stable=8)      # 6 right, 2 wrong, 2 never rest in a bin
        bc = a.bin_choice(rows)
        self.assertEqual((bc['layouts'], bc['stable'], bc['correct'], bc['wrong']), (10, 8, 6, 2))
        self.assertAlmostEqual(bc['wrong_rate'], 0.25)
        self.assertAlmostEqual(bc['accuracy'], 0.75)
        self.assertEqual(bc['accuracy_interval'], list(a.wilson(6, 8)))
        self.assertEqual(bc['wrong_interval'], list(a.wilson(2, 8)))

    def test_a_correct_bin_without_success_still_counts_as_a_stable_correct_placement(self):
        bc = a.bin_choice([row('x', 'can', success=False, placed='correct', cause='unstable_in_target')])
        self.assertEqual((bc['stable'], bc['correct']), (1, 1))

    def test_no_stable_placement_gives_none_not_a_division_error(self):
        bc = a.bin_choice(rows_for('can', 4, correct=0, stable=0))
        self.assertEqual((bc['stable'], bc['wrong_rate'], bc['accuracy']), (0, None, None))
        self.assertEqual(a.bin_choice([])['layouts'], 0)


def trained_rows(stable, wrong):
    """80 trained layouts: `stable` rest in a bin, `wrong` of those in the wrong one (spread over the objects)."""
    rows, objs = [], ('can', 'milk', 'bread', 'lemon')
    for i in range(80):
        obj = objs[i % 4]
        placed = None if i >= stable else ('wrong' if i < wrong else 'correct')
        rows.append(row(f'trained_{i}', obj, success=placed == 'correct', placed=placed))
    return rows


def held_rows(bottle, cereal):
    """`bottle`, `cereal`: (layouts, stable, correct)."""
    return rows_for('bottle', bottle[0], bottle[2], bottle[1]) + rows_for('cereal', cereal[0], cereal[2], cereal[1])


class CriteriaTests(unittest.TestCase):
    def verdict(self, rows, excluded=()):
        return a.criteria(rows, excluded)

    def test_trained_gate_boundaries(self):
        passing = trained_rows(stable=80, wrong=4)           # exactly 5.0%
        self.assertEqual(a.criteria(passing, [])['1_trained_wrong_bin_at_most_5pct']['verdict'], 'pass')
        failing = trained_rows(stable=80, wrong=5)           # 6.25%
        self.assertEqual(a.criteria(failing, [])['1_trained_wrong_bin_at_most_5pct']['verdict'], 'fail')
        self.assertEqual(a.criteria(trained_rows(stable=60, wrong=3), [])['1_trained_wrong_bin_at_most_5pct']['verdict'], 'pass')
        self.assertEqual(a.criteria(trained_rows(stable=60, wrong=4), [])['1_trained_wrong_bin_at_most_5pct']['verdict'], 'fail')

    def test_trained_floor_makes_it_not_interpretable_even_when_perfect(self):
        out = a.criteria(trained_rows(stable=59, wrong=0), [])['1_trained_wrong_bin_at_most_5pct']
        self.assertEqual(out['verdict'], 'not_interpretable')
        out = a.criteria(trained_rows(stable=60, wrong=0), [])['1_trained_wrong_bin_at_most_5pct']
        self.assertEqual(out['verdict'], 'pass')
        self.assertEqual((out['threshold'], out['floor']), (0.05, 60))
        self.assertEqual(a.criteria(trained_rows(stable=59, wrong=30), [])['1_trained_wrong_bin_at_most_5pct']['verdict'],
                         'not_interpretable', 'below the floor is neither pass nor fail, whatever the rate')

    def test_the_gate_uses_bin_choice_not_success(self):
        rows = trained_rows(stable=80, wrong=0)
        for r in rows[:40]:     # half the placements are right but unstable: not a success, still a correct bin
            r.update(task_success=False, cause='unstable_in_target')
        self.assertEqual(a.criteria(rows, [])['1_trained_wrong_bin_at_most_5pct']['verdict'], 'pass')

    def test_held_out_floor_and_evidence_per_object(self):
        crit = a.criteria(trained_rows(80, 0) + held_rows((80, 80, 80), (80, 80, 80)), [])
        held = crit['2_held_out_generalisation_exploratory']
        self.assertEqual(held['verdict'], 'evidence')
        self.assertEqual({o: v['status'] for o, v in held['per_object'].items()}, {'bottle': 'evidence', 'cereal': 'evidence'})
        # 29 stable placements: below the floor, whatever the accuracy.
        low = a.criteria(held_rows((80, 29, 29), (80, 80, 80)), [])['2_held_out_generalisation_exploratory']
        self.assertEqual(low['per_object']['bottle']['status'], 'not_interpretable')
        self.assertEqual(low['verdict'], 'not_interpretable', 'one object below the floor makes the result unreadable')
        self.assertEqual(a.criteria(held_rows((80, 30, 30), (80, 80, 80)), [])['2_held_out_generalisation_exploratory']
                         ['per_object']['bottle']['status'], 'evidence')

    def test_the_evidence_boundary_is_the_wilson_lower_bound_strictly_above_one_half(self):
        n = 40
        above = next(k for k in range(n + 1) if a.wilson(k, n)[0] > 0.5)
        self.assertGreater(above, 0)
        below = above - 1
        self.assertLessEqual(a.wilson(below, n)[0], 0.5)
        status = lambda k: a.criteria(held_rows((n, n, k), (80, 80, 80)), [])['2_held_out_generalisation_exploratory']['per_object']['bottle']['status']
        self.assertEqual(status(above), 'evidence')
        self.assertEqual(status(below), 'not_shown')

    def test_overall_held_out_verdict_needs_both_objects(self):
        n = 40
        weak = next(k for k in range(n + 1) if a.wilson(k, n)[0] > 0.5) - 1
        verdict = lambda b, c: a.criteria(held_rows(b, c), [])['2_held_out_generalisation_exploratory']['verdict']
        self.assertEqual(verdict((80, 80, 80), (n, n, weak)), 'not_shown')
        self.assertEqual(verdict((n, n, weak), (80, 80, 80)), 'not_shown')
        self.assertEqual(verdict((n, n, weak), (n, n, weak)), 'not_shown')
        self.assertEqual(verdict((80, 80, 80), (80, 20, 20)), 'not_interpretable')
        self.assertEqual(verdict((n, n, weak), (80, 20, 20)), 'not_interpretable', 'an unreadable object outranks a weak one')

    def test_trained_and_held_out_rows_are_kept_apart(self):
        rows = trained_rows(80, 0) + held_rows((80, 80, 0), (80, 80, 0))   # held-out all wrong, trained perfect
        crit = a.criteria(rows, [])
        self.assertEqual(crit['1_trained_wrong_bin_at_most_5pct']['verdict'], 'pass')
        self.assertEqual(crit['1_trained_wrong_bin_at_most_5pct']['choice']['layouts'], 80)
        self.assertEqual(crit['2_held_out_generalisation_exploratory']['verdict'], 'not_shown')

    def test_excluded_layouts_are_dropped_before_the_floors(self):
        rows = trained_rows(80, 0)
        self.assertEqual(a.criteria(rows, [])['layouts'], 80)
        excluded = [r['layout_id'] for r in rows[:21]]
        crit = a.criteria(rows, excluded)
        self.assertEqual(crit['layouts'], 59)
        self.assertEqual(crit['1_trained_wrong_bin_at_most_5pct']['verdict'], 'not_interpretable')

    def test_no_rows_is_not_interpretable_and_serialises(self):
        # The branch real data may reach (a stage where nothing comes to rest in a bin): FAILURES-class crash at collect.
        crit = a.criteria([], [])
        json.dumps(crit, allow_nan=False)
        self.assertEqual(crit['1_trained_wrong_bin_at_most_5pct']['verdict'], 'not_interpretable')
        self.assertEqual(crit['2_held_out_generalisation_exploratory']['verdict'], 'not_interpretable')
        self.assertEqual(crit['layouts'], 0)

    def test_milk_is_not_special_cased_in_the_gate(self):
        # The gate pools the four trained objects; milk's motor failures are not stable placements, so they cannot
        # trip it. A milk that is never lifted does not enter bin choice at all.
        rows = [row(f'm{i}', 'milk', success=False, placed=None) for i in range(20)] + [
            r for r in trained_rows(60, 0) if r['object'] != 'milk']
        self.assertEqual(a.criteria(rows, [])['1_trained_wrong_bin_at_most_5pct']['choice']['wrong'], 0)


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.rows = (rows_for('can', 6, 6, slot='left') + rows_for('milk', 4, 2, slot='right', start=100)
                     + rows_for('bottle', 5, 4, start=200, slot='right') + rows_for('cereal', 5, 5, start=300))

    def test_overall_by_kind_slot_and_object(self):
        out = a.readouts(self.rows, [])
        self.assertEqual(out['overall']['n'], 20)
        self.assertEqual(out['overall']['count'], 6 + 2 + 4 + 5)
        self.assertEqual(out['by_kind']['trained']['n'], 10)
        self.assertEqual(out['by_kind']['held_out']['count'], 9)
        self.assertEqual(out['by_blue_slot']['right']['n'], 9)
        self.assertEqual(set(out['by_object']), {'can', 'milk', 'bottle', 'cereal'})
        self.assertEqual(out['by_object']['milk']['success']['count'], 2, 'milk is read on its own')
        self.assertEqual(out['by_object']['bottle']['bin_choice']['wrong'], 1)
        self.assertEqual(out['by_object']['milk']['by_blue_slot']['right']['success']['n'], 4)
        self.assertEqual(out['by_object']['milk']['by_blue_slot']['left']['success']['n'], 0)
        self.assertEqual(out['bin_choice']['held_out']['stable'], 10)
        self.assertEqual(out['causes']['wrong_bin'], 3)
        self.assertEqual(out['causes']['success'], 17 - 0)  # 6 + 2 + 4 + 5 = 17 successes

    def test_infeasible_layouts_are_dropped_and_counted(self):
        out = a.readouts(self.rows, ['can_0', 'cereal_300', 'not_a_layout'])
        self.assertEqual(out['dropped_infeasible_trials'], 2)
        self.assertEqual(out['overall']['n'], 18)
        self.assertEqual(out['by_object']['can']['success']['n'], 5)
        self.assertEqual(out['by_object']['cereal']['bin_choice']['layouts'], 4)
        self.assertEqual(out['excluded_layouts'], ['can_0', 'cereal_300', 'not_a_layout'])

    def test_a_success_without_a_correct_stable_placement_is_visible(self):
        self.assertEqual(a.readouts(self.rows, [])['success_without_a_correct_stable_placement'], 0)
        broken = self.rows + [row('odd', 'can', success=True, placed=None, cause='success')]
        self.assertEqual(a.readouts(broken, [])['success_without_a_correct_stable_placement'], 1)

    def test_an_absent_object_is_not_reported_and_nothing_divides_by_zero(self):
        out = a.readouts(rows_for('can', 3, 3), [])
        self.assertEqual(set(out['by_object']), {'can'})
        self.assertEqual(out['by_kind']['held_out'], dict(n=0, count=0, rate=None, interval=None))
        self.assertEqual(a.readouts([], [])['overall']['rate'], None)


class SelectionTests(unittest.TestCase):
    def test_selection_counts_only_trained_successes_on_feasible_layouts(self):
        rows = rows_for('can', 5, 5) + rows_for('bottle', 9, 9, start=50) + rows_for('milk', 4, 2, start=100)
        self.assertEqual(a.selection_count(rows, []), 5 + 2)
        self.assertEqual(a.selection_count(rows, ['can_0', 'bottle_50', 'milk_100']), 4 + 1)
        self.assertEqual(a.selection_count([], []), 0)

    def test_screen_survivors_are_everything_within_two_of_the_best(self):
        counts = {50760: 5, 76140: 12, 88830: 18, 101520: 17, 114210: 16, 126900: 15}
        self.assertEqual(a.screen_survivors(counts, 2), [88830, 101520, 114210])
        self.assertEqual(a.screen_survivors({1: 10, 2: 8, 3: 7}, 2), [1, 2])
        self.assertEqual(a.screen_survivors({1: 10}, 2), [1])
        self.assertEqual(a.screen_survivors({1: 10, 2: 10, 3: 10}, 0), [1, 2, 3])

    def test_report_winner_is_the_best_and_a_tie_goes_to_the_later_checkpoint(self):
        self.assertEqual(a.report_winner({88830: 50, 101520: 60, 126900: 55}), 101520)
        self.assertEqual(a.report_winner({88830: 60, 101520: 60, 126900: 55}), 101520)
        self.assertEqual(a.report_winner({88830: 60, 126900: 60}), 126900)
        self.assertEqual(a.report_winner({126900: 1}), 126900)


class ConstantsTests(unittest.TestCase):
    def test_pre_registered_thresholds(self):
        self.assertEqual((a.WRONG_BIN_MAX, a.TRAINED_FLOOR, a.HELD_OUT_FLOOR, a.EVIDENCE_LOWER_BOUND), (0.05, 60, 30, 0.5))
        self.assertEqual((a.LIFT_M, a.SUSTAIN_TICKS), (0.02, 15))
        self.assertEqual(a.HELD_OUT, ('bottle', 'cereal'))
        self.assertEqual(set(a.OBJECTS), set(BIN))


if __name__ == '__main__':
    unittest.main()
