import unittest

from experiments.xvla_props_p2_eval.diagnostics import taxonomy as t


def m(**kw):
    base = dict(closed=True, target_in_env=True, distractor_in_env=[False], held=False, lifted=False)
    base.update(kw)
    return base


def row(cause, success=False):
    return dict(cause=cause, task_success=success)


class ClassifyTests(unittest.TestCase):
    def test_success(self):
        self.assertEqual(t.classify(row('success', True), m(lifted=True)), 'success')

    def test_wrong_object_wins_over_every_grasp_measure(self):
        self.assertEqual(t.classify(row('wrong_object'), m(target_in_env=False, distractor_in_env=[True])), 'wrong_object')

    def test_target_lifted_then_failed(self):
        for cause in ('timeout_carrying', 'dropped_outside', 'unstable_in_target', 'wrong_bin'):
            self.assertEqual(t.classify(row(cause), m(lifted=True, held=True)), 'lifted_then_failed', cause)

    def test_never_lifted_chain(self):
        nl = row('never_lifted')
        self.assertEqual(t.classify(nl, m(closed=False)), 'never_closed')
        self.assertEqual(t.classify(nl, m(target_in_env=False, distractor_in_env=[True])), 'closed_on_distractor')
        self.assertEqual(t.classify(nl, m(target_in_env=False, distractor_in_env=[False])), 'closed_off_target')
        self.assertEqual(t.classify(nl, m(target_in_env=True, held=False)), 'closed_not_held')
        self.assertEqual(t.classify(nl, m(target_in_env=True, held=True)), 'held_not_lifted')

    def test_target_in_envelope_is_never_called_a_distractor_grasp(self):
        self.assertEqual(t.classify(row('never_lifted'), m(target_in_env=True, distractor_in_env=[True])), 'closed_not_held')

    def test_two_distractors_any_inside(self):
        self.assertEqual(t.classify(row('never_lifted'), m(target_in_env=False, distractor_in_env=[False, True])),
                         'closed_on_distractor')
        self.assertEqual(t.classify(row('never_lifted'), m(target_in_env=False, distractor_in_env=[False, False])),
                         'closed_off_target')

    def test_unknown_cause_is_refused(self):
        with self.assertRaises(ValueError):
            t.classify(row('mystery'), m())

    def test_classes_are_the_spec_order(self):
        self.assertEqual(t.CLASSES, ('wrong_object', 'lifted_then_failed', 'never_closed', 'closed_on_distractor',
                                     'closed_off_target', 'closed_not_held', 'held_not_lifted'))


if __name__ == '__main__':
    unittest.main()
