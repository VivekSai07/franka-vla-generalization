import unittest

import numpy as np


class R6DriftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiments.robosuite_scene.expert import run_episode
        cls.frames, cls.traces = [], []

        def sink(frame, trace):
            cls.frames.append({"action": frame["action"].copy()})
            cls.traces.append({k: np.copy(trace[k]) for k in ("qpos", "ctrl", "sim_time", "substeps", "released_stack")})

        cls.result = run_episode(0, sink)

    def test_recorded_action_replay_matches(self):
        from experiments.robosuite_scene.validate import replay_recorded
        out = replay_recorded(self.result, self.frames, self.traces)
        print("recorded", out)
        self.assertLess(out["max_policy_state_error"], 1e-3)
        self.assertGreaterEqual(out["stable_ticks"], 30)

    def test_reference_replay_matches(self):
        from experiments.robosuite_scene.validate import replay_reference
        out = replay_reference(self.result, self.frames, self.traces)
        print("reference", out)
        self.assertLessEqual(out["max_full_qpos_error"], 1e-5)

    def test_urdf_agrees_with_robosuite_panda(self):
        from experiments.robosuite_scene.validate import kinematic_mismatch
        mismatch = kinematic_mismatch(self.result, self.traces)
        print("kinematic_mismatch_m", mismatch)
        self.assertLess(mismatch, 1e-3)


if __name__ == "__main__":
    unittest.main()
