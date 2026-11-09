"""Forked-expert tests; robosuite image. The sealed P2 data must be mounted."""
import unittest

import numpy as np

from experiments.gen_scene import compose, gates, levels
from experiments.gen_scene.conditions import NOMINAL, Condition
from experiments.gen_scene.expert import GenExpert
from experiments.gen_scene.metrics import placement as relative_placement
from experiments.props_scene import constants as c
from experiments.props_scene.metrics import placement as absolute_placement
from helpers import layouts, make_layout


def run(layout, condition, render=False):
    model, resolved = compose.model_for(layout, condition)
    expert = GenExpert(model, resolved["primitive_names"])
    scene = compose.reset(model, resolved["poses"], layout["home_pose_jitter_deg"])
    traces = []
    episode = dict(object=layout["object"], bin=layout["bin"], seed=layout["seed"], home_pose_jitter_deg=layout["home_pose_jitter_deg"])
    result = expert.run(scene["initial_state"], episode, lambda frame, trace: traces.append(trace), render=render)
    return result, traces, model, resolved


def target_column(layout, resolved):
    """The target's column in `props_pose`: the trace lists the present props in the sealed order (`constants.OBJECTS`)."""
    present = [o for o in c.OBJECTS if o in resolved["props"]]
    return present.index(layout["object"])


def target_rise(traces, column):
    z = np.array([t["props_pose"][column][2] for t in traces])
    return float(z.max() - z[0])


def hand_targets(traces):
    """The z of the hand target at every tick, with the phase it belongs to."""
    return np.array([t["target_pos"][2] for t in traces]), np.array([t["phase_index"] for t in traces])


class SealedEpisodes(unittest.TestCase):
    """G0 spec D9 (ii): under nominal the fork is the sealed expert."""

    def test_qpos_ctrl_and_props_traces_are_byte_equal(self):
        for index in gates.GATE_EPISODES:
            out = gates.expert(index)
            self.assertEqual(out["trace_equal"], {"qpos": True, "ctrl": True, "props_pose": True}, index)
            self.assertTrue(out["final_qpos_equal"] and out["success"] and out["initial_state_equal_sealed"], index)
            self.assertEqual(out["ticks"], out["sealed_ticks"])
            self.assertEqual(out["ticks"], 452)


class TableHeight(unittest.TestCase):
    """The grasp, carry and release heights follow the table; the sealed constants alone would not."""

    def test_the_expert_succeeds_and_carries_the_same_distance_at_every_height(self):
        layout = make_layout(1, "milk", ("can",))
        base, base_traces, _, base_resolved = run(layout, NOMINAL)
        self.assertTrue(base["success"])
        base_rise = target_rise(base_traces, target_column(layout, base_resolved))
        self.assertGreater(base_rise, 0.15)
        for level in ("table_up5", "table_down5", "table_up10"):
            condition = levels.spatial(level, layout, "x", 0)
            result, traces, model, resolved = run(layout, condition)
            self.assertTrue(result["success"], (level, result["reason"]))
            self.assertEqual(result["frames"], base["frames"], level)
            self.assertLess(abs(target_rise(traces, target_column(layout, resolved)) - base_rise), 0.005, level)

    def test_every_hand_target_from_approach_to_open_moves_by_the_offset(self):
        layout = make_layout(1, "milk", ("can",))
        _, base_traces, _, _ = run(layout, NOMINAL)
        base_z, phases = hand_targets(base_traces)
        for level, offset in (("table_up5", 0.05), ("table_down5", -0.05), ("table_up10", 0.10)):
            _, traces, _, _ = run(layout, levels.spatial(level, layout, "x", 0))
            z, _ = hand_targets(traces)
            holding = phases <= 6      # approach, descend, close, lift, move, lower, open (retract is relative, home is the robot's)
            self.assertEqual(len(z), len(base_z))
            self.assertLess(float(np.abs(z[holding] - base_z[holding] - offset).max()), 0.003, level)

    def test_release_is_over_the_wall_not_over_the_floor(self):
        layout = make_layout(2, "can", ("bread", "lemon"))
        for level in ("table_up10", "table_down5"):
            condition = levels.spatial(level, layout, "x", 0)
            result, _, model, resolved = run(layout, condition)
            self.assertTrue(result["success"])
            self.assertTrue(result["final_placement"]["resting_in_bin"], level)
            self.assertAlmostEqual(result["final_placement"]["bottom_z_m"], condition.table_offset_m, delta=0.02)

    def test_placement_is_relative_to_the_bin(self):
        layout = make_layout(2, "can", ("bread", "lemon"))
        result, _, model, resolved = run(layout, levels.spatial("table_up10", layout, "x", 0))
        import mujoco
        from data_collection.benchmark import restore_state
        data = mujoco.MjData(model)
        restore_state(model, data, result["final_state"])
        mujoco.mj_forward(model, data)
        self.assertTrue(relative_placement(model, data, "can", "blue")["resting_in_bin"])
        self.assertFalse(absolute_placement(model, data, "can", "blue")["resting_in_bin"])   # props_scene's test misses it at +10 cm


class MoreObjects(unittest.TestCase):
    def test_primitives_ride_in_the_trace_and_the_expert_still_succeeds(self):
        layout = layouts(24)[3]
        condition = levels.distractors("five", layout, layout["identity"])
        result, traces, model, resolved = run(layout, condition)
        self.assertEqual(traces[0]["props_pose"].shape[0], len(resolved["names"]))
        self.assertTrue(result["success"], result["reason"])

    def test_the_target_is_a_prop_of_the_six(self):
        layout = make_layout(0)
        model, resolved = compose.model_for(layout, NOMINAL)
        with self.assertRaises(ValueError):
            GenExpert(model).run(np.zeros(1), dict(object="ball_pink", bin="blue", seed=0, home_pose_jitter_deg=[0] * 7), lambda *a: None)


class FromMovedPose(unittest.TestCase):
    def test_start_override_is_where_the_expert_grasps(self):
        layout = make_layout(3, "can", ("bread",))
        condition = levels.spatial("outer_start", layout, layout["identity"], 0)
        result, traces, model, resolved = run(layout, condition)
        self.assertEqual(tuple(resolved["start"]), tuple(condition.start))
        first = traces[0]["props_pose"][target_column(layout, resolved)]
        self.assertLess(abs(first[0] - condition.start[0]), 2e-3)
        self.assertLess(abs(first[1] - condition.start[1]), 2e-3)

    def test_the_expert_aims_at_the_bin_where_it_is(self):
        layout = make_layout(3, "can", ("bread",))
        for shift in (0.04, -0.04):
            condition = Condition(axis="spatial", level="outer_start", start=(0.44, 0.0, 0.0), bin_shift_m=shift)
            result, _, model, _ = run(layout, condition)
            self.assertTrue(result["success"], (shift, result["reason"]))
            self.assertGreater(result["final_placement"]["xy_margin_m"], 0.095)   # inside and near the middle of the bin
            self.assertAlmostEqual(model.body("bin_blue").pos[1], 0.33 + shift, places=12)


if __name__ == "__main__":
    unittest.main()
