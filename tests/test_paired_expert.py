"""Expert wiring tests with synthetic clock/IK/images: no real trajectories run."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np


class PairedExpertTests(unittest.TestCase):
    def test_configuration_matches_original_physical_contract_without_stepping(self):
        from experiments.paired_demonstrations.expert_v1 import Expert, configuration
        expert = Expert()
        config = configuration(expert.model, expert.robot)
        self.assertEqual(config["timing"]["substeps_per_control"], 20)
        self.assertEqual(config["camera_names"], ["front_cam", "gripper_cam"])
        self.assertEqual(config["expert_command_contract"]["version"], "bounded_joint_integration_v1")
        self.assertEqual(config["expert_command_contract"]["max_velocity"], 2.)
        self.assertEqual(expert.model.nq, 37)

    def test_synthetic_clock_preserves_observe_command_step_and_gripper_phases(self):
        from experiments.paired_demonstrations import expert_v1 as module
        from experiments.paired_demonstrations.plan import make_plan
        from data_collection.benchmark import capture_state
        from src.robots.franka_panda.config import Q_HOME
        expert = module.Expert()
        expert.ik = SimpleNamespace(compute_velocity=lambda *a, **k: (np.zeros(9), 0, True))
        data = mujoco.MjData(expert.model)
        data.time = .2
        data.qpos[:len(Q_HOME)] = Q_HOME
        initial = capture_state(expert.model, data)
        frames, traces = [], []

        class Renderer:
            def __init__(self, *a, **k): pass
            def update_scene(self, *a, **k): pass
            def render(self): return np.zeros((512, 512, 3), np.uint8)
            def close(self): pass

        class Grasp:
            def __init__(self, model, data, target): self.data = data
            def command(self, value): self.data.ctrl[-1] = value
            def is_grasped(self): return True

        def step(model, data, dt, arm_dof):
            for _ in range(20): data.time += 1 / 600
            return 20

        global_rng = np.random.get_state()
        with patch.object(module.mujoco, "Renderer", Renderer), \
             patch.object(module, "GraspController", Grasp), \
             patch.object(module, "step_control", step), \
             patch.object(module, "stack_metrics", return_value={"released_stack": True}):
            result = expert.run(initial, make_plan()["episodes"][0],
                                lambda frame, trace: (frames.append(frame), traces.append(trace)))
        self.assertTrue(result["success"])
        self.assertEqual(len(frames), 528)
        transitions = np.flatnonzero(np.diff([f["action"][7] for f in frames])) + 1
        np.testing.assert_array_equal(transitions, [107, 347])
        np.testing.assert_array_equal(np.random.get_state()[1], global_rng[1])
        self.assertEqual(np.random.get_state()[2:], global_rng[2:])
        for frame, trace in zip(frames, traces):
            self.assertEqual(trace["substeps"], 20)
            self.assertEqual(trace["image_time"], trace["sim_time"])
            np.testing.assert_array_equal(frame["action"], trace["ctrl"].astype(np.float32))
            np.testing.assert_array_equal(frame["observation.state"], trace["qpos"][:9].astype(np.float32))
        self.assertEqual(traces[0]["phase_index"], 0)
        self.assertEqual(traces[-1]["phase_index"], 8)
        # Exercise all three replay paths on the SAME synthetic (not physical)
        # clock, then prove a corrupted requested IK velocity fails comparison.
        from experiments.paired_demonstrations import validate as validator
        trace = {key: np.asarray([t[key] for t in traces]) for key in traces[0]}
        trace.update(final_qpos=result["final_qpos"], final_state=result["final_state"])
        entry = {**make_plan()["episodes"][0], "frames": 528, "stable_ticks": 528}
        actions = np.stack([f["action"] for f in frames])
        with patch.object(module.mujoco, "Renderer", Renderer), \
             patch.object(module, "GraspController", Grasp), \
             patch.object(module, "step_control", step), \
             patch.object(module, "stack_metrics", return_value={"released_stack": True}), \
             patch.object(validator, "step_control", step), \
             patch.object(validator, "stack_metrics", return_value={"released_stack": True}):
            checks = validator.replay_episode(expert, initial, entry, trace, actions)
            self.assertEqual([r["mode"] for r in checks], ["scripted_expert", "reference", "recorded_actions"])
            self.assertTrue(all(r["passed"] for r in checks))
            # Raw rerender hashes are independent of lossy video decoding.
            original_hash = trace["image_sha256"][0, 0]
            trace["image_sha256"][0, 0] = "f" * 64
            with self.assertRaises(AssertionError): validator.replay_episode(expert, initial, entry, trace, actions)
            trace["image_sha256"][0, 0] = original_hash
            trace["requested_joint_velocity"][0, 0] = .1
            with self.assertRaises(AssertionError): validator.replay_episode(expert, initial, entry, trace, actions)

    def test_reset_placement_uses_local_rng_and_never_steps_an_expert(self):
        import random
        from experiments.paired_demonstrations.expert_v1 import Expert, initialize_layout
        from experiments.paired_demonstrations.plan import make_plan
        expert = Expert()
        old_state = random.getstate()
        layout = make_plan()["layouts"][0]
        first = initialize_layout(expert.model, layout)
        second = initialize_layout(expert.model, layout)
        np.testing.assert_array_equal(first["initial_state"], second["initial_state"])
        np.testing.assert_array_equal(first["qpos"], second["qpos"])
        self.assertEqual(random.getstate(), old_state)
        self.assertAlmostEqual(first["initial_state"][0], .2)


if __name__ == "__main__": unittest.main()