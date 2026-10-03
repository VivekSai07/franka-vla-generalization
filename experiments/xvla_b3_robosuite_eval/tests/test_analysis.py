import unittest

import numpy as np

from experiments.xvla_b3_robosuite_eval import analysis

Z0 = 0.021


def make_trace(colours, ticks, lifts=None, gripper=None):
    """cube_xyz (ticks, n, 3) with z = Z0 + lift; `lifts` maps colour -> per-tick z offset (m)."""
    xyz = np.zeros((ticks, len(colours), 3))
    xyz[:, :, 2] = Z0
    for colour, offset in (lifts or {}).items():
        xyz[:, colours.index(colour), 2] += np.asarray(offset)
    action = np.zeros((ticks, 8))
    action[:, 7] = 255 if gripper is None else gripper
    return dict(cube_xyz=xyz, colours=np.asarray(colours), applied_action=action)


def ramp(ticks, start, stop, height):
    out = np.zeros(ticks)
    out[start:stop] = height
    return out


COLOURS = ['blue', 'red', 'yellow']


class LiftTests(unittest.TestCase):
    def test_a_bumped_distractor_is_not_the_first_lift(self):
        """Review focus 3: a distractor rises 2 cm (bumped), then the pick cube rises 15 cm."""
        trace = make_trace(COLOURS, 200, lifts={'blue': ramp(200, 20, 60, 0.02), 'red': ramp(200, 80, 200, 0.15)})
        self.assertEqual(analysis.first_lifted(trace), 'red')

    def test_a_short_high_bump_does_not_count_either(self):
        trace = make_trace(COLOURS, 200, lifts={'blue': ramp(200, 20, 30, 0.08), 'red': ramp(200, 80, 200, 0.15)})
        self.assertEqual(analysis.first_lifted(trace), 'red')

    def test_nothing_lifted(self):
        self.assertIsNone(analysis.first_lifted(make_trace(COLOURS, 100)))

    def test_the_earlier_sustained_lift_wins(self):
        trace = make_trace(COLOURS, 200, lifts={'blue': ramp(200, 100, 200, 0.1), 'red': ramp(200, 50, 200, 0.1)})
        self.assertEqual(analysis.first_lifted(trace), 'red')


class ClassifyTests(unittest.TestCase):
    def cause(self, trace, success=False):
        return analysis.classify(trace, 'red', success)

    def test_success_on_a_real_carry_classifies_as_success_with_the_pick_lifted_first(self):
        trace = make_trace(COLOURS, 300, lifts={'red': ramp(300, 60, 250, 0.15)})
        self.assertEqual(self.cause(trace, True), 'success')
        self.assertEqual(analysis.first_lifted(trace), 'red')

    def test_wrong_cube(self):
        trace = make_trace(COLOURS, 300, lifts={'blue': ramp(300, 60, 200, 0.1)})
        self.assertEqual(self.cause(trace), 'wrong_cube')

    def test_never_lifted(self):
        trace = make_trace(COLOURS, 300, lifts={'red': ramp(300, 60, 90, 0.01)})
        self.assertEqual(self.cause(trace), 'never_lifted')

    def test_lifted_not_carried(self):
        trace = make_trace(COLOURS, 300, lifts={'red': ramp(300, 60, 300, 0.1)}, gripper=0)
        self.assertEqual(self.cause(trace), 'lifted_not_carried')

    def test_released_off_target(self):
        gripper = np.zeros(300)
        gripper[200:] = 255
        trace = make_trace(COLOURS, 300, lifts={'red': ramp(300, 60, 200, 0.1)}, gripper=gripper)
        self.assertEqual(self.cause(trace), 'released_off_target')

    def test_the_gripper_being_open_before_the_lift_is_not_a_release(self):
        gripper = np.zeros(300)
        gripper[:40] = 255
        trace = make_trace(COLOURS, 300, lifts={'red': ramp(300, 60, 300, 0.1)}, gripper=gripper)
        self.assertEqual(self.cause(trace), 'lifted_not_carried')

    def test_a_wrong_cube_lift_wins_over_a_later_pick_lift(self):
        trace = make_trace(COLOURS, 300, lifts={'blue': ramp(300, 40, 120, 0.1), 'red': ramp(300, 150, 300, 0.1)})
        self.assertEqual(self.cause(trace), 'wrong_cube')


def row(layout, condition, success, kind='trained', edge='interior', cause=None, seed_index=0):
    cause = cause or ('success' if success else 'lifted_not_carried')
    return dict(layout_id=layout, condition=condition, task_success=success, kind=kind, edge=edge,
                cause=cause, seed_index=seed_index)


TABLE = [
    row('L1', 'control', True), row('L1', 'distractor', True),
    row('L2', 'control', True, 'held_out'), row('L2', 'distractor', False, 'held_out', cause='wrong_cube'),
    row('L3', 'control', False, edge='inner', cause='never_lifted'),
    row('L3', 'distractor', False, edge='inner'),
    row('L4', 'control', True, 'held_out', 'outer'), row('L4', 'distractor', True, 'held_out', 'outer'),
]


class ReadoutTests(unittest.TestCase):
    def test_exact_numbers_on_a_hand_built_table(self):
        out = analysis.readouts(TABLE, set())
        self.assertEqual((out['overall']['n'], out['overall']['count']), (8, 5))
        self.assertEqual(out['overall']['layouts'], 4)
        self.assertEqual((out['by_kind']['trained']['count'], out['by_kind']['trained']['n']), (2, 4))
        self.assertEqual((out['by_kind']['held_out']['count'], out['by_kind']['held_out']['n']), (3, 4))
        self.assertEqual((out['by_condition']['control']['count'], out['by_condition']['control']['n']), (3, 4))
        self.assertEqual((out['by_condition']['distractor']['count'], out['by_condition']['distractor']['n']), (2, 4))
        contrast = out['distractor_minus_control']
        self.assertEqual(contrast['layouts'], 4)
        self.assertAlmostEqual(contrast['mean_difference'], -0.25)
        self.assertEqual(contrast['excluded_unpaired_layouts'], [])
        edge = out['by_edge']
        self.assertEqual((edge['interior']['count'], edge['interior']['n']), (3, 4))
        self.assertEqual((edge['inner']['count'], edge['inner']['n']), (0, 2))
        self.assertEqual((edge['outer']['count'], edge['outer']['n']), (2, 2))
        self.assertEqual((edge['edge_any']['count'], edge['edge_any']['n']), (2, 4))
        self.assertEqual((out['wrong_cube']['overall']['count'], out['wrong_cube']['overall']['n']), (1, 8))
        self.assertEqual(out['wrong_cube']['by_kind']['held_out']['count'], 1)
        self.assertEqual(out['wrong_cube']['by_kind']['trained']['count'], 0)
        self.assertEqual(out['causes'], {'success': 5, 'wrong_cube': 1, 'never_lifted': 1,
                                         'lifted_not_carried': 1, 'released_off_target': 0})
        self.assertEqual(out['dropped_infeasible_trials'], 0)

    def test_excluded_layouts_are_dropped_everywhere_and_counted(self):
        """Review focus 4: expert-infeasible layouts are never silently counted."""
        extra = [row('L9', 'control', False, cause='never_lifted'), row('L9', 'distractor', False, cause='never_lifted')]
        out = analysis.readouts(TABLE + extra, {'L9'})
        self.assertEqual(out['dropped_infeasible_trials'], 2)
        self.assertEqual(out['excluded_layouts'], ['L9'])
        self.assertEqual(out['overall']['n'], 8)
        self.assertEqual(out['causes']['never_lifted'], 1)
        self.assertEqual(out['distractor_minus_control']['layouts'], 4)
        self.assertEqual(sum(out['causes'].values()), 8)
        with_it = analysis.readouts(TABLE + extra, set())
        self.assertEqual(with_it['overall']['n'], 10)
        self.assertEqual(with_it['dropped_infeasible_trials'], 0)

    def test_the_contrast_is_paired_by_layout_and_drops_unpaired_layouts_with_a_count(self):
        """Review focus 5."""
        unpaired = [row('L6', 'control', False)]
        out = analysis.readouts(TABLE + unpaired, set())
        contrast = out['distractor_minus_control']
        self.assertEqual(contrast['layouts'], 4)
        self.assertEqual(contrast['excluded_unpaired_layouts'], ['L6'])
        self.assertAlmostEqual(contrast['mean_difference'], -0.25)
        # An unpaired comparison would give 2/4 - 3/5 = -0.1 here, and the table is not that.
        control = out['by_condition']['control']
        self.assertEqual((control['count'], control['n']), (3, 5))

    def test_a_layout_with_two_seeds_contributes_one_difference(self):
        rows = [row('A', 'control', True, seed_index=0), row('A', 'control', False, seed_index=1),
                row('A', 'distractor', False, seed_index=0), row('A', 'distractor', False, seed_index=1),
                row('B', 'control', True, seed_index=0), row('B', 'distractor', True, seed_index=0)]
        out = analysis.readouts(rows, set())['distractor_minus_control']
        self.assertEqual(out['layouts'], 2)
        self.assertAlmostEqual(out['mean_difference'], (-0.5 + 0.0) / 2)

    def test_empty_groups_are_reported_not_invented(self):
        out = analysis.readouts([row('L1', 'control', True)], set())
        self.assertEqual(out['by_kind']['held_out']['n'], 0)
        self.assertIsNone(out['by_kind']['held_out']['rate'])
        self.assertEqual(out['distractor_minus_control']['layouts'], 0)
        self.assertIsNone(out['distractor_minus_control']['mean_difference'])


class IntervalTests(unittest.TestCase):
    def rows(self, layouts):
        return [dict(layout_id=f'L{i}', task_success=(i % 2 == 0)) for i in range(layouts) for _ in range(2)]

    def test_more_layouts_give_a_narrower_interval_at_the_same_rate(self):
        small = analysis.clustered_interval(self.rows(4), 'task_success')
        large = analysis.clustered_interval(self.rows(40), 'task_success')
        self.assertAlmostEqual(sum(r['task_success'] for r in self.rows(4)) / 8, 0.5)
        self.assertAlmostEqual(sum(r['task_success'] for r in self.rows(40)) / 80, 0.5)
        self.assertLess(large[1] - large[0], small[1] - small[0])
        self.assertLessEqual(large[0], 0.5)
        self.assertGreaterEqual(large[1], 0.5)

    def test_resamples_layouts_not_trials(self):
        # Every layout is all-success or all-fail: the bootstrap spread comes from layouts only.
        rows = self.rows(4)
        lo, hi = analysis.clustered_interval(rows, 'task_success')
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_deterministic(self):
        rows = self.rows(10)
        self.assertEqual(analysis.clustered_interval(rows, 'task_success'),
                         analysis.clustered_interval(rows, 'task_success'))

    def test_no_simulator_imports(self):
        import ast
        from pathlib import Path
        tree = ast.parse((Path(analysis.__file__)).read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                self.assertNotIn(name.split('.')[0], {'mujoco', 'robosuite', 'torch', 'av'})


if __name__ == '__main__':
    unittest.main()
