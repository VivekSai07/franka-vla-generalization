import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import mujoco
import numpy as np
import torch

from experiments.xvla_b3_robosuite_eval import protocol, rollout, scenes


class StubPolicy:
    """Fixed in-range action: arm held at home, gripper open."""

    def __init__(self):
        from src.robots.franka_panda.config import Q_HOME
        self.action = torch.tensor([*Q_HOME[:7], 255.0], dtype=torch.float32).unsqueeze(0)
        self.resets = 0

    def reset(self):
        self.resets += 1

    def select_action(self, batch):
        return self.action


class RolloutTests(unittest.TestCase):
    def setUp(self):
        if not protocol.BANK_NPZ.exists():
            self.skipTest('B3 parent not mounted')

    def run_trial(self, trial, ticks, folder):
        model, data = scenes.restore(trial['bank_key'])
        with mujoco.Renderer(model, height=512, width=512) as renderer, patch.object(rollout, 'MAX_TICKS', ticks):
            return rollout.rollout(StubPolicy(), lambda x: x, lambda x: x, model, data, renderer, trial, folder)

    def test_fifty_ticks_with_a_stub_policy(self):
        trial = next(t for t in protocol.report_trials() if t['kind'] == 'held_out' and t['condition'] == 'distractor')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = self.run_trial(trial, 50, folder)
            episode = json.loads((folder/'episode.json').read_text())
            for key in ('pick', 'place', 'kind', 'edge', 'condition', 'layout_id', 'seed', 'colours',
                        'first_lifted', 'cause', 'outcome', 'task_success', 'ticks', 'instruction'):
                self.assertIn(key, episode)
            self.assertEqual(episode['outcome'], 'timeout')
            self.assertFalse(episode['task_success'])
            self.assertEqual(episode['ticks'], 50)
            self.assertEqual(episode['cause'], 'never_lifted')
            self.assertIsNone(episode['first_lifted'])
            self.assertEqual(episode['kind'], 'held_out')
            self.assertEqual(result['instruction'], trial['instruction'])
            trace = np.load(folder/'trace.npz')
            self.assertEqual(trace['cube_xyz'].shape, (50, len(trial['colours']), 3))
            self.assertEqual(sorted(str(c) for c in trace['colours']), trial['colours'])
            self.assertEqual(trace['applied_action'].shape, (50, 8))
            with av.open(str(folder/'cameras.mp4')) as video:
                frames = sum(1 for _ in video.decode(video.streams.video[0]))
            self.assertEqual(frames, 51)

    def test_the_instruction_context_is_restored_after_a_rollout(self):
        from experiments.xvla_single_task import contract as single_task
        before = single_task.TASK
        trial = protocol.screen_trials()[0]
        with tempfile.TemporaryDirectory() as tmp:
            self.run_trial(trial, 3, Path(tmp))
        self.assertEqual(single_task.TASK, before)

    def test_a_trial_that_disagrees_with_the_scene_is_rejected(self):
        trial = dict(protocol.screen_trials()[0])
        model, data = scenes.restore(trial['bank_key'])
        trial['colours'] = ['blue', 'red']
        with tempfile.TemporaryDirectory() as tmp, mujoco.Renderer(model, height=512, width=512) as renderer:
            with self.assertRaises(ValueError):
                rollout.rollout(StubPolicy(), lambda x: x, lambda x: x, model, data, renderer, trial, Path(tmp))

    def test_seeding_is_deterministic(self):
        rollout.seed_trial(7)
        a = (torch.rand(2), np.random.rand(2))
        rollout.seed_trial(7)
        b = (torch.rand(2), np.random.rand(2))
        self.assertTrue(torch.equal(a[0], b[0]))
        np.testing.assert_array_equal(a[1], b[1])


if __name__ == '__main__':
    unittest.main()
