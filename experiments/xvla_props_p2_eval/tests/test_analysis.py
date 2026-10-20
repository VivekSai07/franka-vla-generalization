import json
import unittest
from pathlib import Path

import numpy as np

from experiments.xvla_props_p2_eval import analysis as a

NAMES = ['can', 'milk', 'lemon']
SEALED = Path('/vla/outputs/props_p2_data/props_p2_20261013/episodes')


def zs(ticks=200, lifts=None):
    """props_z for NAMES; `lifts` maps a name to (start_tick, rise_m) held to the end."""
    z = np.zeros((ticks, len(NAMES)))
    for name, (tick, rise) in (lifts or {}).items():
        z[tick:, NAMES.index(name)] = rise
    return z


def trace(z, placed=None, open_=True):
    return dict(props_z=z, names=NAMES, placed_bin=placed, final_gripper_open=open_)


class LiftTests(unittest.TestCase):
    def test_threshold_and_sustain(self):
        flat = np.zeros(100)
        self.assertIsNone(a.lift_tick(flat))
        bump = flat.copy(); bump[10:14] = 0.2
        self.assertIsNone(a.lift_tick(bump), 'a 4-tick bump is not a lift')
        small = flat.copy(); small[10:] = 0.019
        self.assertIsNone(a.lift_tick(small), 'below 2 cm')
        up = flat.copy(); up[10:] = 0.021
        self.assertEqual(a.lift_tick(up), 10)
        self.assertIsNone(a.lift_tick([]))

    def test_first_lifted_is_the_earliest_and_needs_matching_shapes(self):
        self.assertEqual(a.first_lifted(zs(lifts={'milk': (50, .1), 'can': (80, .2)}), NAMES), 'milk')
        self.assertEqual(a.first_lifted(zs(lifts={'can': (50, .1), 'milk': (50, .1)}), NAMES), 'can')
        self.assertIsNone(a.first_lifted(zs(), NAMES))
        with self.assertRaises(ValueError):
            a.first_lifted(np.zeros((5, 2)), NAMES)


class ClassifyTests(unittest.TestCase):
    def test_success_wins_even_if_a_distractor_moved(self):
        self.assertEqual(a.classify(trace(zs(lifts={'milk': (10, .1), 'can': (40, .2)})), 'can', 'blue', True), 'success')

    def test_never_lifted(self):
        self.assertEqual(a.classify(trace(zs()), 'can', 'blue', False), 'never_lifted')

    def test_wrong_object_is_the_first_lifted_being_a_distractor(self):
        self.assertEqual(a.classify(trace(zs(lifts={'milk': (30, .1)})), 'can', 'blue', False), 'wrong_object')
        self.assertEqual(a.classify(trace(zs(lifts={'milk': (30, .1), 'can': (90, .2)}), placed='blue'), 'can', 'blue', False),
                         'wrong_object', 'the first object lifted decides, even if the target is lifted later')

    def test_target_first_then_the_p1_chain(self):
        lift = zs(lifts={'can': (30, .2)})
        self.assertEqual(a.classify(trace(lift, placed='orange'), 'can', 'blue', False), 'wrong_bin')
        self.assertEqual(a.classify(trace(lift, placed='blue', open_=False), 'can', 'blue', False), 'unstable_in_target')
        self.assertEqual(a.classify(trace(lift, placed=None, open_=False), 'can', 'blue', False), 'timeout_carrying')
        self.assertEqual(a.classify(trace(lift, placed=None, open_=True), 'can', 'blue', False), 'dropped_outside')
        later = zs(lifts={'can': (30, .2), 'milk': (120, .1)})
        self.assertEqual(a.classify(trace(later, placed='blue'), 'can', 'blue', False), 'unstable_in_target',
                         'a distractor disturbed after the target was lifted first is not wrong_object')

    def test_every_cause_is_reachable_and_named(self):
        seen = {a.classify(trace(zs(lifts=l), placed=p, open_=o), 'can', 'blue', s)
                for l, p, o, s in [({'can': (1, .2)}, 'blue', True, True), ({}, None, True, False),
                                   ({'milk': (1, .2)}, None, True, False), ({'can': (1, .2)}, 'orange', True, False),
                                   ({'can': (1, .2)}, 'blue', True, False), ({'can': (1, .2)}, None, False, False),
                                   ({'can': (1, .2)}, None, True, False)]}
        self.assertEqual(seen, set(a.CAUSES))

    @unittest.skipUnless(SEALED.is_dir(), 'sealed P2 expert traces not mounted')
    def test_validated_on_the_540_real_expert_traces_before_any_rollout(self):
        # Positive control: the target must register as lifted and no distractor ever may. Without the control a
        # classifier that never reports wrong_object would look correct on successes.
        objects = ('can', 'bread', 'lemon', 'milk', 'bottle', 'cereal')
        n = target_first = distractor_lifted = 0
        swapped_wrong = 0
        for path in sorted(SEALED.glob('episode_*.npz')):
            z = np.load(path)
            names = [o for o in objects if o in [str(z['object'])] + [str(d) for d in z['distractors']]]
            pose = z['props_pose'][:, :, 2]
            target = str(z['object'])
            first = a.first_lifted(pose, names)
            n += 1
            target_first += first == target
            distractor_lifted += any(a.lift_tick(pose[:, i]) is not None for i, nm in enumerate(names) if nm != target)
            other = [nm for nm in names if nm != target][0]
            swapped_wrong += a.classify(dict(props_z=pose, names=names, placed_bin=None, final_gripper_open=True),
                                        other, 'blue', False) == 'wrong_object'
        self.assertEqual((n, target_first, distractor_lifted), (540, 540, 0))
        self.assertEqual(swapped_wrong, 540, 'naming a distractor as the target must flag the target as wrong_object')


class StatisticsAndRulesTests(unittest.TestCase):
    def rows(self, spec, kind='trained', count=1, slot='left'):
        return [dict(layout_id=f'L{i}', ordinal=i, task_success=s, cause='success' if s else c, kind=kind,
                     distractor_count=count, blue_slot=slot, wrong_object=(c == 'wrong_object'))
                for i, (s, c) in enumerate(spec)]

    def test_readouts_drop_excluded_layouts_and_split_every_factor(self):
        rows = (self.rows([(True, ''), (False, 'wrong_object'), (False, 'never_lifted')], 'trained', 1, 'left')
                + [dict(r, layout_id=f'H{i}') for i, r in enumerate(self.rows([(True, ''), (True, '')], 'held_out', 2, 'right'))])
        out = a.readouts(rows, ['L2'])
        self.assertEqual(out['dropped_infeasible_trials'], 1)
        self.assertEqual((out['overall']['n'], out['overall']['count']), (4, 3))
        self.assertEqual(out['by_kind']['held_out']['rate'], 1.0)
        self.assertEqual(out['by_distractor_count']['2']['n'], 2)
        self.assertEqual(out['by_distractor_count']['0']['n'], 0)
        self.assertEqual(out['by_blue_slot']['right']['rate'], 1.0)
        self.assertEqual(out['wrong_object']['overall']['count'], 1)
        self.assertEqual(out['causes']['wrong_object'], 1)
        self.assertEqual(sum(out['causes'].values()), 4)

    def test_blue_slot_split_reveals_a_position_shortcut(self):
        rows = self.rows([(True, '')] * 5, slot='left') + [dict(r, layout_id=f'R{i}') for i, r in
                                                          enumerate(self.rows([(False, 'wrong_bin')] * 5, slot='right'))]
        out = a.readouts(rows, [])
        self.assertEqual((out['by_blue_slot']['left']['rate'], out['by_blue_slot']['right']['rate']), (1.0, 0.0))

    def test_screen_survivors_keep_everything_within_two_of_the_best(self):
        self.assertEqual(a.screen_survivors({1: 18, 2: 16, 3: 15, 4: 17}), [1, 2, 4])
        self.assertEqual(a.screen_survivors({1: 10, 2: 10}), [1, 2], 'a tie is not ranked')
        self.assertEqual(a.screen_survivors({5: 3, 6: 20}), [6])

    def test_report_winner_is_the_highest_and_ties_go_to_the_later_checkpoint(self):
        self.assertEqual(a.report_winner({88830: 90, 101520: 91, 126900: 85}), 101520)
        self.assertEqual(a.report_winner({88830: 90, 126900: 90}), 126900)
        self.assertEqual(a.report_winner({126900: 90, 88830: 90, 101520: 90}), 126900, 'never array order')

    def test_mcnemar_exact(self):
        self.assertEqual(a.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(a.mcnemar_exact(10, 0), 2 / 1024)
        self.assertAlmostEqual(a.mcnemar_exact(5, 5), 1.0)

    def test_clustered_interval_uses_layouts_as_the_unit(self):
        rows = [dict(layout_id='A', v=1)] * 8 + [dict(layout_id='B', v=0)] * 8
        lo, hi = a.clustered_interval(rows, 'v')
        self.assertEqual((lo, hi), (0.0, 1.0), 'two layouts cannot give a tight interval however many trials')


class CriteriaTests(unittest.TestCase):
    def make(self, trained_fail, held_fail, p1_succ, wrong=0):
        """120 trials, 80 trained then 40 held-out; failures are the *last* ones of each kind, the first `wrong` of
        the P2 failures being wrong-object."""
        kinds = ['trained'] * 80 + ['held_out'] * 40
        fail = [i >= 80 - trained_fail for i in range(80)] + [i >= 40 - held_fail for i in range(40)]
        wrongs = [i for i, f in enumerate(fail) if f][:wrong]

        def rows(failed, with_wrong):
            return [dict(layout_id=f'L{i}', ordinal=i, task_success=not f, kind=k, distractor_count=1,
                         blue_slot='left', wrong_object=with_wrong and i in wrongs,
                         cause='success' if not f else ('wrong_object' if with_wrong and i in wrongs else 'never_lifted'))
                    for i, (f, k) in enumerate(zip(failed, kinds))]
        p1_failed = [i % 120 >= p1_succ * 120 // 120 for i in range(120)]
        return rows(fail, True), rows(p1_failed, False)

    def test_all_four_pass(self):
        p2, p1 = self.make(10, 6, 60)
        out = a.criteria(p2, p1, [])
        self.assertTrue(out['all_passed'], out)
        self.assertEqual(out['trials'], 120)

    def test_each_criterion_can_fail_independently(self):
        out = a.criteria(*self.make(40, 20, 40), [])
        self.assertFalse(out['1_overall_at_least_70pct']['passed'])
        self.assertFalse(out['all_passed'])
        p2, p1 = self.make(10, 6, 60, wrong=10)
        out = a.criteria(p2, p1, [])
        self.assertFalse(out['2_wrong_object_at_most_5pct']['passed'])
        self.assertTrue(out['1_overall_at_least_70pct']['passed'])
        out = a.criteria(*self.make(2, 25, 60), [])
        self.assertFalse(out['3_kind_gap_within_10_points']['passed'])
        self.assertGreater(out['3_kind_gap_within_10_points']['value'], 0.10)
        out = a.criteria(*self.make(10, 6, 112), [])
        self.assertFalse(out['4_beats_p1_non_overlapping']['passed'], 'a better rate with overlapping intervals is not enough')
        self.assertFalse(out['all_passed'])

    def test_pairing_requires_shared_trials_and_excluded_layouts_are_dropped(self):
        p2, p1 = self.make(10, 6, 60)
        out = a.criteria(p2, p1, ['L0', 'L1'])
        self.assertEqual(out['trials'], 118)
        with self.assertRaises(ValueError):
            a.criteria(p2, [], [])


if __name__ == '__main__':
    unittest.main()
