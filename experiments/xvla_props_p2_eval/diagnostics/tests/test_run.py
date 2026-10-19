"""Robosuite image only: per-trial measures, Gates 0 and 1, and the analysis tables, on real fixture trials."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import mujoco
    MUJOCO = mujoco.__version__ == '3.4.0'
except ImportError:  # admission image: no simulator
    MUJOCO = False

EPISODES = Path('/vla/outputs/props_p2_data/props_p2_20261013/episodes')
if MUJOCO:
    from experiments.xvla_props_p2_eval import protocol
    from experiments.xvla_props_p2_eval.diagnostics import reference, run
    AVAILABLE = EPISODES.is_dir() and (protocol.EVAL_ROOT/'report_p2_step_101520').is_dir()
else:
    AVAILABLE = False


def pick(predicate):
    root = protocol.EVAL_ROOT/'report_p2_step_101520'
    rows = json.loads((root/'result.json').read_text())['episodes']
    row = next(r for r in rows if predicate(r))
    return protocol.trial_folder(root, row), row


@unittest.skipUnless(AVAILABLE, 'needs the robosuite image and the P2 evaluation outputs')
class MeasureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = sorted(EPISODES.glob('episode_*.npz'))
        cls.ref = reference.build_reference(files[::9])  # a stride across all cells, so every object has an envelope
        assert len(cls.ref) == 6, sorted(cls.ref)
        calm = lambda r: r['task_success'] and max(r['distractor_displacement_m'].values()) < 1e-3
        cls.cases = {'success': pick(calm), 'never_lifted': pick(lambda r: r['cause'] == 'never_lifted'),
                     'wrong_object': pick(lambda r: r['cause'] == 'wrong_object')}

    def measure(self, key):
        folder, row = self.cases[key]
        return run.measure_trial(folder, self.ref)

    def test_success_is_lifted_closed_and_held(self):
        m = self.measure('success')
        self.assertTrue(m['gate0'])
        self.assertEqual(m['class'], 'success')
        self.assertTrue(m['lifted'] and m['closed'] and m['held'])
        self.assertIsNotNone(m['grasp_closure'])
        self.assertLessEqual(m['grasp_closure'], m['lift_tick'])

    def test_never_lifted_lands_in_a_grasp_class_with_a_closest_approach(self):
        m = self.measure('never_lifted')
        self.assertTrue(m['gate0'])
        self.assertFalse(m['lifted'])
        self.assertIn(m['class'], ('never_closed', 'closed_on_distractor', 'closed_off_target', 'closed_not_held', 'held_not_lifted'))
        self.assertGreater(m['closest_approach_m'], 0.0)

    def test_wrong_object_keeps_the_evaluations_label_and_names_the_lifted_distractor(self):
        m = self.measure('wrong_object')
        self.assertTrue(m['gate0'])
        self.assertEqual(m['class'], 'wrong_object')
        self.assertEqual(m['first_lifted'], self.cases['wrong_object'][1]['first_lifted'])

    def test_measure_is_deterministic(self):
        self.assertEqual(self.measure('never_lifted'), self.measure('never_lifted'))


class GateTests(unittest.TestCase):
    def test_gate1_fraction_and_threshold(self):
        rows = [dict(class_='success', in_env=True)] * 8 + [dict(class_='success', in_env=False)] * 2
        self.assertAlmostEqual(run.gate1_fraction([dict(task_success=True, grasp_inside=r['in_env']) for r in rows]), 0.8)
        self.assertTrue(run.gate1_passes(0.8))
        self.assertFalse(run.gate1_passes(0.79))

    def test_gate1_ignores_failures_and_unclosed_successes(self):
        rows = [dict(task_success=True, grasp_inside=True), dict(task_success=False, grasp_inside=False),
                dict(task_success=True, grasp_inside=None)]
        self.assertEqual(run.gate1_fraction(rows), 0.5)

    def test_gate0_summary_requires_every_trial(self):
        self.assertTrue(run.gate0_ok([dict(gate0=True), dict(gate0=True)]))
        self.assertFalse(run.gate0_ok([dict(gate0=True), dict(gate0=False)]))
        self.assertFalse(run.gate0_ok([]))


if __name__ == '__main__':
    unittest.main()
