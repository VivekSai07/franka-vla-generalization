import unittest
from unittest import mock

import numpy as np

from experiments.vla_inputs_signals import replay

INDICES = (0, 271, 539)


def _noop(tick, model, data):
    pass


class ExactReplay(unittest.TestCase):
    def test_three_episodes_replay_exactly(self):
        for index in INDICES:
            sealed = replay.load_episode(index)
            self.assertEqual(sealed["state"].dtype, np.float32)
            self.assertTrue(np.array_equal(sealed["state"], sealed["qpos"][:, :9].astype(np.float32)))
            summary = replay.frames(index, _noop)
            self.assertEqual(summary, {"frames": 452, "index": index})

    def test_on_frame_sees_the_observation_instant(self):
        sealed = replay.load_episode(271)
        seen = []

        def on_frame(tick, model, data):
            seen.append((tick, data.qpos.copy()))

        replay.frames(271, on_frame)
        self.assertEqual([t for t, _ in seen], list(range(452)))
        for tick, qpos in seen:
            self.assertTrue(np.array_equal(qpos, sealed["qpos"][tick]), tick)

    def test_corrupted_action_is_caught(self):
        sealed = replay.load_episode(0)
        action = sealed["action"].copy()
        action[50:, 0] += np.float32(0.05)  # joint 1 command, well inside the actuator range
        with mock.patch.object(replay, "load_episode", return_value={**sealed, "action": action}):
            with self.assertRaises(replay.ReplayMismatch) as caught:
                replay.frames(0, _noop)
        self.assertEqual(caught.exception.index, 0)
        self.assertGreaterEqual(caught.exception.tick, 50)
        self.assertIn(caught.exception.field, ("qpos", "props_pose"))

    def test_final_state_checked(self):
        sealed = replay.load_episode(0)
        with mock.patch.object(replay, "load_episode", return_value={**sealed, "action": sealed["action"][:-1]}):
            with self.assertRaises(replay.ReplayMismatch) as caught:
                replay.frames(0, _noop)
        self.assertEqual(caught.exception.field, "final_qpos")

    def test_corrupted_sealed_trace_is_caught(self):
        sealed = replay.load_episode(0)
        qpos = sealed["qpos"].copy()
        qpos[200, 3] = np.nextafter(qpos[200, 3], np.inf)  # one ulp
        with mock.patch.object(replay, "load_episode", return_value={**sealed, "qpos": qpos}):
            with self.assertRaises(replay.ReplayMismatch) as caught:
                replay.frames(0, _noop)
        self.assertEqual((caught.exception.tick, caught.exception.field), (200, "qpos"))

    def test_corrupted_sealed_prop_pose_is_caught(self):
        sealed = replay.load_episode(0)
        props = sealed["props_pose"].copy()
        props[120, 0, 2] = np.nextafter(props[120, 0, 2], np.inf)
        with mock.patch.object(replay, "load_episode", return_value={**sealed, "props_pose": props}):
            with self.assertRaises(replay.ReplayMismatch) as caught:
                replay.frames(0, _noop)
        self.assertEqual((caught.exception.tick, caught.exception.field), (120, "props_pose"))

    def test_sealed_episode_identity_matches_the_plan(self):
        sealed = replay.load_episode(271)
        self.assertEqual(sealed["episode"]["object"], "can")
        self.assertEqual(sealed["episode"]["episode_index"], 271)
        with mock.patch.object(replay, "_plan_episode", return_value={**sealed["episode"], "object": "milk"}):
            with self.assertRaises(ValueError):
                replay.load_episode(271)


if __name__ == "__main__":
    unittest.main()
