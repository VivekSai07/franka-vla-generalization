import unittest

import numpy as np

from experiments.xvla_props_p1_eval import analysis


def row(ordinal, layout_id, kind, blue_slot, bin_colour, task_success, cause):
    return dict(ordinal=ordinal, layout_id=layout_id, kind=kind, blue_slot=blue_slot, bin=bin_colour,
                object='can', instruction=f'put the can in the {bin_colour} bin',
                task_success=task_success, cause=cause)


class LiftedTests(unittest.TestCase):
    def test_never_lifted(self):
        self.assertFalse(analysis.lifted_and_sustained(np.zeros(100)))

    def test_lifted_but_not_sustained(self):
        z = np.zeros(100)
        z[50] = 0.10  # a single-tick bump, not held
        self.assertFalse(analysis.lifted_and_sustained(z))

    def test_lifted_and_sustained(self):
        z = np.zeros(100)
        z[50:80] = 0.10
        self.assertTrue(analysis.lifted_and_sustained(z))

    def test_empty_series(self):
        self.assertFalse(analysis.lifted_and_sustained(np.array([])))


class ClassifyTests(unittest.TestCase):
    def test_success(self):
        trace = dict(prop_z=np.full(50, 0.1), placed_bin='blue', final_gripper_open=True)
        self.assertEqual(analysis.classify(trace, 'blue', True), 'success')

    def test_never_lifted(self):
        trace = dict(prop_z=np.zeros(50), placed_bin=None, final_gripper_open=True)
        self.assertEqual(analysis.classify(trace, 'blue', False), 'never_lifted')

    def test_wrong_bin(self):
        z = np.zeros(900)
        z[400:700] = 0.10
        trace = dict(prop_z=z, placed_bin='orange', final_gripper_open=True)
        self.assertEqual(analysis.classify(trace, 'blue', False), 'wrong_bin')

    def test_timeout_carrying(self):
        z = np.zeros(900)
        z[400:] = 0.10
        trace = dict(prop_z=z, placed_bin=None, final_gripper_open=False)
        self.assertEqual(analysis.classify(trace, 'blue', False), 'timeout_carrying')

    def test_dropped_outside(self):
        z = np.zeros(900)
        z[400:700] = 0.10
        trace = dict(prop_z=z, placed_bin=None, final_gripper_open=True)
        self.assertEqual(analysis.classify(trace, 'blue', False), 'dropped_outside')

    def test_unstable_in_target(self):
        """Resting in the correct bin but never sustaining 30 ticks (e.g. slow finger release) is
        its own cause, not dropped_outside/timeout_carrying — found by inspection in screen_015
        (step_126900, bottle->orange): placed_bin == bin_colour, final_gripper_open True, the
        success streak just never completed. The unfixed classifier reported `dropped_outside`."""
        z = np.zeros(900)
        z[400:700] = 0.10
        trace = dict(prop_z=z, placed_bin='orange', final_gripper_open=True)
        self.assertEqual(analysis.classify(trace, 'orange', False), 'unstable_in_target')

    def test_unstable_in_target_even_if_gripper_still_closed(self):
        z = np.zeros(900)
        z[400:700] = 0.10
        trace = dict(prop_z=z, placed_bin='blue', final_gripper_open=False)
        self.assertEqual(analysis.classify(trace, 'blue', False), 'unstable_in_target')


class ReadoutsTests(unittest.TestCase):
    def test_overall_by_kind_and_wrong_bin(self):
        rows = [
            row(1, 'screen_000', 'trained', 'left', 'blue', True, 'success'),
            row(2, 'screen_001', 'trained', 'right', 'blue', False, 'wrong_bin'),
            row(3, 'report_000', 'held_out', 'left', 'orange', True, 'success'),
            row(4, 'report_001', 'held_out', 'right', 'orange', False, 'never_lifted'),
        ]
        out = analysis.readouts(rows, excluded_layouts=())
        self.assertEqual(out['overall']['n'], 4)
        self.assertEqual(out['overall']['count'], 2)
        self.assertEqual(out['by_kind']['trained']['n'], 2)
        self.assertEqual(out['by_kind']['held_out']['n'], 2)
        self.assertEqual(out['wrong_bin']['overall']['count'], 1)
        self.assertEqual(out['causes']['wrong_bin'], 1)
        self.assertEqual(out['causes']['never_lifted'], 1)
        self.assertEqual(out['dropped_infeasible_trials'], 0)

    def test_excluded_layouts_are_dropped_and_counted(self):
        rows = [row(1, 'screen_000', 'trained', 'left', 'blue', True, 'success'),
                row(2, 'screen_001', 'trained', 'right', 'blue', False, 'never_lifted')]
        out = analysis.readouts(rows, excluded_layouts=('screen_001',))
        self.assertEqual(out['overall']['n'], 1)
        self.assertEqual(out['dropped_infeasible_trials'], 1)
        self.assertEqual(out['excluded_layouts'], ['screen_001'])


class ArrangementShortcutTests(unittest.TestCase):
    """Review Focus 5: the split must be by `blue_slot`, not by the instructed/target bin colour."""

    def _position_only_rows(self):
        """A synthetic policy that always reaches left: succeeds iff blue_slot == 'left', regardless
        of the target colour. Half the rows name blue, half name orange, evenly within each slot."""
        rows = []
        n = 0
        for slot in ('left', 'right'):
            for bin_colour in ('blue', 'orange'):
                for i in range(10):
                    n += 1
                    success = (slot == 'left')
                    rows.append(row(n, f'{slot}_{bin_colour}_{i}', 'trained', slot, bin_colour,
                                    success, 'success' if success else 'wrong_bin'))
        return rows

    def test_by_blue_slot_reveals_the_shortcut(self):
        rows = self._position_only_rows()
        out = analysis.readouts(rows, excluded_layouts=())
        left, right = out['by_blue_slot']['left']['rate'], out['by_blue_slot']['right']['rate']
        self.assertAlmostEqual(left, 1.0)
        self.assertAlmostEqual(right, 0.0)
        self.assertGreater(abs(left - right), 0.5, 'arrangement gap must be large for a position shortcut')

    def test_grouping_by_bin_colour_instead_would_hide_the_shortcut(self):
        """The trap Review Focus 5 names: success is uncorrelated with the instructed colour here."""
        rows = self._position_only_rows()
        by_bin = {colour: sum(r['task_success'] for r in rows if r['bin'] == colour) /
                          len([r for r in rows if r['bin'] == colour]) for colour in ('blue', 'orange')}
        self.assertAlmostEqual(by_bin['blue'], 0.5)
        self.assertAlmostEqual(by_bin['orange'], 0.5)
        self.assertEqual(set(analysis.readouts(rows, ())['by_blue_slot']), {'left', 'right'})
        self.assertNotIn('by_bin', analysis.readouts(rows, ()))


if __name__ == '__main__':
    unittest.main()
