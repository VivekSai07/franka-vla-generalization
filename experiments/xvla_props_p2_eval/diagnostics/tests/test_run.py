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
from experiments.xvla_props_p2_eval import protocol
from experiments.xvla_props_p2_eval.diagnostics import run  # imports no simulator at module level

if MUJOCO:
    from experiments.xvla_props_p2_eval.diagnostics import reference
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


def synthetic(ticks=100, closed=(20, 60), lift_at=40, target_off=0.0, distractor_off=0.2, touch=(25, 60)):
    """A hand-built replay: identity hand at the origin, target on the hand axis, one distractor beside it."""
    import numpy as np
    rep = dict(names=['can', 'milk'], gate0=True, first_mismatch=None)
    rep['hand_pos'] = np.zeros((ticks, 3))
    rep['hand_rot'] = np.tile(np.eye(3).ravel(), (ticks, 1))
    rep['left_pos'] = np.tile([-0.04, 0, 0.05], (ticks, 1))
    rep['right_pos'] = np.tile([0.04, 0, 0.05], (ticks, 1))
    props = np.zeros((ticks, 2, 3))
    props[:, 0] = [target_off, 0, 0.15]
    props[:, 1] = [distractor_off, 0, 0.15]
    if lift_at is not None:
        props[lift_at:, 0, 2] += 0.05
    rep['props_xyz'] = props
    rep['props_yaw_deg'] = np.zeros((ticks, 2))
    command = np.full(ticks, 255.0)
    if closed is not None:
        command[closed[0]:closed[1]] = 0.0
    rep['command'] = command
    for key in ('left_touch', 'right_touch', 'touched'):
        rep[key] = np.zeros((ticks, 2), bool)
    if touch is not None:
        rep['left_touch'][touch[0]:touch[1], 0] = rep['right_touch'][touch[0]:touch[1], 0] = True
    return rep


REF = {'can': dict(lo=[-0.02, -0.02, 0.10, -20], hi=[0.02, 0.02, 0.20, 20], n=10),
       'milk': dict(lo=[-0.02, -0.02, 0.10, -20], hi=[0.02, 0.02, 0.20, 20], n=10)}


def fake_row(cause='never_lifted', success=False):
    return dict(cause=cause, task_success=success)


class ReduceTests(unittest.TestCase):
    """Every field of the reduction on synthetic replays with known answers (no simulator needed)."""

    def test_a_clean_grasp(self):
        m = run.measures_from_replay(fake_row('success', True), synthetic(), REF)
        self.assertEqual((m['first_closure'], m['grasp_closure'], m['lift_tick']), (20, 20, 40))
        self.assertTrue(m['closed'] and m['lifted'] and m['target_in_env'] and m['grasp_inside'] and m['held'])
        self.assertEqual(m['distractor_in_env'], [False])
        self.assertEqual(m['class'], 'success')
        self.assertAlmostEqual(m['closest_approach_m'], 0.10, places=6)

    def test_target_off_the_envelope_is_an_approach_error(self):
        m = run.measures_from_replay(fake_row(), synthetic(target_off=0.1, lift_at=None), REF)
        self.assertFalse(m['target_in_env'])
        self.assertEqual(m['class'], 'closed_off_target')

    def test_gripper_closed_on_the_distractor(self):
        m = run.measures_from_replay(fake_row(), synthetic(target_off=0.1, distractor_off=0.0, lift_at=None), REF)
        self.assertEqual(m['distractor_in_env'], [True])
        self.assertEqual(m['class'], 'closed_on_distractor')

    def test_never_closed(self):
        m = run.measures_from_replay(fake_row(), synthetic(closed=None, lift_at=None, touch=None), REF)
        self.assertFalse(m['closed'])
        self.assertIsNone(m['grasp_closure'])
        self.assertEqual(m['class'], 'never_closed')

    def test_chatter_is_not_a_closure(self):
        m = run.measures_from_replay(fake_row(), synthetic(closed=(30, 33), lift_at=None, touch=None), REF)
        self.assertFalse(m['closed'])

    def test_closed_in_envelope_but_not_held(self):
        m = run.measures_from_replay(fake_row(), synthetic(lift_at=None, touch=(25, 30)), REF)
        self.assertTrue(m['target_in_env'] and not m['held'])
        self.assertEqual(m['class'], 'closed_not_held')

    def test_held_but_not_lifted(self):
        m = run.measures_from_replay(fake_row(), synthetic(lift_at=None), REF)
        self.assertTrue(m['held'])
        self.assertEqual(m['class'], 'held_not_lifted')

    def test_held_is_measured_after_the_closure_not_before(self):
        m = run.measures_from_replay(fake_row(), synthetic(lift_at=None, touch=(2, 12)), REF)
        self.assertFalse(m['held'], 'contact before the first closure is not a hold')

    def test_grasp_closure_follows_the_run_that_lifts(self):
        rep = synthetic(closed=(20, 25), lift_at=None)
        rep['command'][50:90] = 0.0
        rep['props_xyz'][70:, 0, 2] += 0.05
        m = run.measures_from_replay(fake_row('timeout_carrying'), rep, REF)
        self.assertEqual((m['first_closure'], m['grasp_closure']), (20, 50))
        self.assertEqual(m['class'], 'lifted_then_failed')

    def test_a_distractor_is_judged_against_its_own_objects_envelope(self):
        # The milk is gripped 0.03 m off the hand axis: outside the can's envelope, inside the milk's own.
        ref = {'can': REF['can'], 'milk': dict(lo=[-0.05, -0.02, 0.10, -20], hi=[0.05, 0.02, 0.20, 20], n=10)}
        rep = synthetic(target_off=0.1, distractor_off=0.03, lift_at=None)
        m = run.measures_from_replay(fake_row(), rep, ref)
        self.assertEqual(m['distractor_in_env'], [True])
        self.assertEqual(m['class'], 'closed_on_distractor')
        self.assertEqual(run.measures_from_replay(fake_row(), rep, REF)['distractor_in_env'], [False])

    def test_touched_distractors_are_listed(self):
        rep = synthetic()
        rep['touched'][30:40, 1] = True
        self.assertEqual(run.measures_from_replay(fake_row('success', True), rep, REF)['touched_distractors'], ['milk'])

    def test_missing_object_envelope_is_refused(self):
        with self.assertRaises(ValueError):
            run.measures_from_replay(fake_row(), synthetic(), {'can': REF['can']})


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
