import json
import subprocess
import sys
import unittest

import numpy as np

from experiments.props_scene import constants as c
from experiments.props_scene import validate as v

ATTEMPT = {"object": "can", "bin": "orange", "blue_slot": "left", "pose": [0.28, -0.06, -40.0], "seed": 7,
           "home_pose_jitter_deg": [0.5, -0.4, 0.3, 0.0, 0.8, -0.9, 0.2]}


class ReplayTests(unittest.TestCase):
    def test_replay_reproduces_a_success(self):
        result, actions, traces, check, digest, start_error = v.execute(ATTEMPT, render=False)
        self.assertTrue(result["success"], result["reason"])
        self.assertTrue(check["ok"], check)
        self.assertLess(check["max_policy_state_error"], 1e-3)
        self.assertLess(check["max_prop_pose_error"], 1e-3)
        self.assertEqual(check["stable_ticks"], result["stable_ticks"])
        self.assertLess(start_error, 0.002)
        # the recorded float32 action is exactly the command the simulator received
        for action, trace in zip(actions, traces):
            np.testing.assert_array_equal(trace["ctrl"][:7], action[:7].astype(np.float64))
            self.assertEqual(float(trace["ctrl"][-1]), float(action[7]))

    def test_replay_reproduces_a_failure_too(self):
        attempt = {**ATTEMPT, "object": "cereal", "bin": "blue", "pose": [0.30, 0.02, 80.0]}
        result, actions, traces, check, _, _ = v.execute(attempt, render=False)
        self.assertFalse(result["success"])             # the written definition cannot be met by a standing box
        self.assertTrue(check["ok"], check)

    def test_replay_notices_a_changed_action(self):
        result, actions, traces, _, _, _ = v.execute(ATTEMPT, render=False)
        tampered = [a.copy() for a in actions]
        for k in range(200, len(tampered)):
            tampered[k][1] += 0.15
        check = v.replay(result, tampered, traces)
        self.assertFalse(check["ok"])
        self.assertGreater(check["max_policy_state_error"], 1e-3)

    def test_trace_digest_depends_on_every_array(self):
        result, actions, traces, _, digest, _ = v.execute(ATTEMPT, render=False)
        self.assertEqual(digest, v.trace_digest(actions, traces))
        changed = [dict(t) for t in traces]
        changed[100]["props_pose"] = changed[100]["props_pose"] + 1e-9
        self.assertNotEqual(digest, v.trace_digest(actions, changed))
        other = [a.copy() for a in actions]
        other[5][0] += 1e-6
        self.assertNotEqual(digest, v.trace_digest(other, traces))


class DeterminismTests(unittest.TestCase):
    def test_two_processes_give_byte_equal_traces_including_the_images(self):
        command = [sys.executable, "-m", "experiments.props_scene.validate", json.dumps(ATTEMPT)]
        digests = []
        for _ in range(2):
            out = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip().splitlines()
            digests.append(out[-1])
        self.assertEqual(len(digests[0]), 64)
        self.assertEqual(digests[0], digests[1])
        # and it is the same trace this process produces
        self.assertEqual(digests[0], v.execute(ATTEMPT, render=True)[4])


if __name__ == "__main__":
    unittest.main()
