"""Regression tests for exact cadence and current observation geometry."""
import unittest

import mujoco
import numpy as np

from data_collection.benchmark import step_control, capture_state


def moving_model():
    return mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.002" gravity="0 0 0"/>'
        '<worldbody><body name="moving"><freejoint/>'
        '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )


class TimingTests(unittest.TestCase):
    def test_nonintegral_period_is_rejected_instead_of_overshooting(self):
        model = moving_model()
        with self.assertRaises(ValueError):
            step_control(model, mujoco.MjData(model), 1 / 30, 0)

    def test_300_ticks_are_ten_seconds_and_every_tick_has_20_substeps(self):
        from data_collection.benchmark import configure_timing
        model = moving_model()
        contract = configure_timing(model, 30)
        self.assertEqual(contract["substeps_per_control"], 20)
        data = mujoco.MjData(model)
        for tick in range(300):
            self.assertEqual(step_control(model, data, 1 / 30, 0), 20)
            self.assertAlmostEqual(data.time, (tick + 1) / 30, places=10)

    def test_observation_refresh_preserves_integration_state(self):
        from data_collection.benchmark import synchronize_observation
        model = moving_model()
        data = mujoco.MjData(model)
        data.qvel[0] = 1.
        mujoco.mj_step(model, data)
        self.assertNotEqual(data.xpos[1, 0], data.qpos[0])
        before = capture_state(model, data)
        synchronize_observation(model, data)
        np.testing.assert_array_equal(capture_state(model, data), before)
        self.assertAlmostEqual(data.xpos[1, 0], data.qpos[0], places=12)

    def test_step_returns_synchronized_geometry(self):
        model = moving_model()
        model.opt.timestep = 1 / 600
        data = mujoco.MjData(model)
        data.qvel[0] = 1.
        step_control(model, data, 1 / 30, 0)
        self.assertAlmostEqual(data.xpos[1, 0], data.qpos[0], places=12)

    def test_action_diagnostics_separate_joint_and_gripper_clipping(self):
        from validation.rollout_diagnostics import bound_action
        ranges = np.array([[-1., 1.]] * 7)
        _, metrics = bound_action(np.array([2.] + [0.] * 6 + [-3.]), ranges)
        self.assertEqual(metrics["joint_clip_mask"], [True] + [False] * 6)
        self.assertTrue(metrics["gripper_clipped"])
        action, metrics = bound_action(np.array([0.] * 7 + [128.]), ranges)
        self.assertEqual(action[7], 255)
        self.assertFalse(metrics["gripper_clipped"])
        self.assertTrue(metrics["gripper_thresholded"])

    def test_old_or_modified_timing_manifest_is_rejected(self):
        from data_collection.benchmark import configure_timing, restore_timing
        model = moving_model()
        with self.assertRaises(ValueError):
            restore_timing(model, {"fps": 30, "physics_dt": .002})
        contract = configure_timing(model, 30)
        manifest = {"fps": 30, "physics_dt": 1 / 600, "timing": contract}
        self.assertEqual(restore_timing(model, manifest), contract)
        manifest["timing"] = {**contract, "substeps_per_control": 17}
        with self.assertRaises(ValueError):
            restore_timing(model, manifest)

    def test_reference_commands_replay_under_the_fixed_clock(self):
        from data_collection.benchmark import configure_timing, restore_state, synchronize_observation
        model = moving_model()
        configure_timing(model, 30)
        first, second = mujoco.MjData(model), mujoco.MjData(model)
        first.qvel[0] = .5
        synchronize_observation(model, first)
        restore_state(model, second, capture_state(model, first))
        for _ in range(60):
            step_control(model, first, 1 / 30, 0)
            step_control(model, second, 1 / 30, 0)
            np.testing.assert_array_equal(capture_state(model, first), capture_state(model, second))
            np.testing.assert_array_equal(first.xpos, second.xpos)


if __name__ == "__main__":
    unittest.main()