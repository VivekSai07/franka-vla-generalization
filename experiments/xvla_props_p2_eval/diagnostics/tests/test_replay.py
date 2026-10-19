"""Robosuite image only (MuJoCo 3.4.0): replay of recorded trials. Needs the evaluation outputs and the sealed P2 plan."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
    MUJOCO = mujoco.__version__ == '3.4.0'
except ImportError:  # admission image: no simulator
    MUJOCO = False

if MUJOCO:
    # A missing or broken replay module must fail loudly here, never turn into a skip.
    from experiments.xvla_props_p2_eval import protocol
    from experiments.xvla_props_p2_eval.diagnostics import replay
    EVAL = protocol.EVAL_ROOT/'report_p2_step_101520'
    AVAILABLE = protocol.PLAN_JSON.exists() and EVAL.is_dir()
else:
    AVAILABLE = False


def pick(root, predicate):
    rows = json.loads((Path(root)/'result.json').read_text())['episodes']
    row = next(r for r in rows if predicate(r))
    return protocol.trial_folder(root, row), row


@unittest.skipUnless(AVAILABLE, 'needs the robosuite image and the P2 evaluation outputs')
class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        calm = lambda r: r['task_success'] and max(r['distractor_displacement_m'].values()) < 1e-3
        cls.success_folder, cls.success_row = pick(EVAL, calm)
        cls.success = replay.replay_trial(cls.success_folder)
        cls.failure_folder, cls.failure_row = pick(EVAL, lambda r: r['cause'] == 'never_lifted')
        cls.failure = replay.replay_trial(cls.failure_folder)

    def test_gate0_reproduces_a_success_and_a_failure_exactly(self):
        for out in (self.success, self.failure):
            self.assertTrue(out['gate0'], out['first_mismatch'])
            self.assertIsNone(out['first_mismatch'])

    def test_shapes_follow_the_recorded_trace(self):
        trace = np.load(self.success_folder/'trace.npz')
        ticks, props = len(trace['applied_action']), len(self.success_row['names'])
        self.assertEqual(self.success['hand_pos'].shape, (ticks, 3))
        self.assertEqual(self.success['hand_rot'].shape, (ticks, 9))
        self.assertEqual(self.success['props_xyz'].shape, (ticks, props, 3))
        self.assertEqual(self.success['left_touch'].shape, (ticks, props))
        self.assertEqual(self.success['command'].shape, (ticks,))
        self.assertEqual(self.success['names'], self.success_row['names'])

    def test_positive_control_the_target_is_gripped_on_a_success(self):
        both = self.success['left_touch'][:, 0] & self.success['right_touch'][:, 0]
        self.assertGreaterEqual(int(both.sum()), 30)

    def test_negative_control_an_unmoved_distractor_is_never_touched(self):
        self.assertFalse(self.success['touched'][:, 1:].any())

    def test_positive_control_a_moved_distractor_registers_contact(self):
        folder, row = pick(protocol.EVAL_ROOT/'confirm_p2_step_101520',
                           lambda r: max(r['distractor_displacement_m'].values()) > 0.05)
        out = replay.replay_trial(folder)
        self.assertTrue(out['gate0'])
        moved = max(row['distractor_displacement_m'], key=row['distractor_displacement_m'].get)
        column = row['names'].index(moved)
        self.assertTrue(out['touched'][:, column].any(), moved)

    def test_negative_control_a_corrupted_action_fails_gate0(self):
        tmp = Path(tempfile.mkdtemp())
        shutil.copy(self.success_folder/'episode.json', tmp/'episode.json')
        trace = dict(np.load(self.success_folder/'trace.npz'))
        trace['applied_action'] = trace['applied_action'].copy()
        trace['applied_action'][100, 0] += 0.01
        np.savez(tmp/'trace.npz', **trace)
        out = replay.replay_trial(tmp)
        self.assertFalse(out['gate0'])
        self.assertGreater(out['first_mismatch'], 100)


if __name__ == '__main__':
    unittest.main()
