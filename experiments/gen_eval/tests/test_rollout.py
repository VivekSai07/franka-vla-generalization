"""The forked rollout; robosuite image. A scripted adapter replays an expert's recorded actions, so there is no policy and no GPU.

Under the nominal condition the fork must be M0's rollout bit for bit (the CPU half of gate D9 iii); each disturbance must fire
exactly when its definition says, and only then.
"""
import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from experiments.gen_eval import protocol, rollout, scenes
from experiments.gen_eval.plan import _with_conditions
from experiments.gen_scene import compose, levels
from experiments.gen_scene.conditions import NOMINAL, Condition
from experiments.gen_scene.expert import GenExpert
from experiments.vla_inputs_eval import adapters
from experiments.vla_inputs_eval import protocol as m0_protocol
from experiments.vla_inputs_eval import rollout as m0_rollout
from experiments.vla_inputs_eval import scenes as m0_scenes
from helpers import layouts, make_layout

TRACE_KEYS = ("state", "raw_action", "applied_action", "props_xyz", "xy_margin_m", "z_m", "lowest_z_m", "gripper_open",
              "finger_contact", "wrist_wrench", "finger_force", "target_box")
_ACTIONS = {}


class ReplayAdapter:
    """Returns a recorded action per tick, then holds the last one."""
    name = "replay"

    def __init__(self, actions):
        self.actions, self.tick = actions, 0

    def reset(self):
        self.tick = 0

    def act(self, obs):
        adapters.check_observation(obs)
        action = self.actions[min(self.tick, len(self.actions) - 1)]
        self.tick += 1
        return np.array(action, dtype=np.float32)


class Constant:
    """The same action every tick; NaN from call `nan_after` + 1 on."""
    name = "constant"

    def __init__(self, action, nan_after=None):
        self.action, self.nan_after, self.calls = np.asarray(action, dtype=np.float32), nan_after, 0

    def reset(self):
        self.calls = 0

    def act(self, obs):
        self.calls += 1
        if self.nan_after is not None and self.calls > self.nan_after:
            return np.full(8, np.nan, dtype=np.float32)
        return self.action.copy()


def expert_actions(layout, condition=NOMINAL):
    key = (layout["layout_id"], layout["object"], tuple(layout["start"]), condition.digest())
    if key not in _ACTIONS:
        model, resolved = compose.model_for(layout, condition)
        expert = GenExpert(model, resolved["primitive_names"])
        scene = compose.reset(model, resolved["poses"], layout["home_pose_jitter_deg"])
        frames = []
        episode = dict(object=layout["object"], bin=layout["bin"], seed=layout["seed"], home_pose_jitter_deg=layout["home_pose_jitter_deg"])
        result = expert.run(scene["initial_state"], episode, lambda frame, trace: frames.append(frame["action"].copy()), render=False)
        assert result["success"], result["reason"]
        _ACTIONS[key] = np.stack(frames)
    return _ACTIONS[key]


def trial_for(layout, axis="nominal", ordinal=1):
    return protocol.trial_of(layout if "axes" in layout else dict(layout, axes={}), axis, ordinal)


def run_trial(layout, trial, adapter, folder):
    condition = Condition.from_dict(trial["condition"])
    model, data, resolved = scenes.restore(layout, condition)
    with mujoco.Renderer(model, height=512, width=512) as renderer:
        return rollout.run(adapter, adapters.state_p2, model, data, resolved, renderer, trial, folder, condition)


def planned(count=8):
    """Layouts with their conditions attached, as the plan does (layout n gets level (n + offset) % levels)."""
    return _with_conditions(layouts(count))


def trace_of(folder):
    with np.load(Path(folder) / "trace.npz") as z:
        return {k: z[k] for k in z.files}


def hold(model, data):
    """The action that keeps the arm where it is with the gripper open."""
    return np.concatenate([data.qpos[:7], [255.0]]).astype(np.float32)


class M0Parity(unittest.TestCase):
    """The fork under the nominal condition is M0's rollout: every array and every outcome field."""

    def test_a_full_trial_is_identical_to_m0s_in_every_trace_array(self):
        layout = make_layout(1, "milk", ("can",))
        actions = expert_actions(layout)
        with tempfile.TemporaryDirectory() as tmp:
            mine, theirs = Path(tmp) / "mine", Path(tmp) / "m0"
            mine.mkdir()
            theirs.mkdir()
            row = run_trial(layout, trial_for(layout), ReplayAdapter(actions), mine)
            m0_trial = m0_protocol.trial_of(layout, 1)
            model, data, names = m0_scenes.restore(layout)
            with mujoco.Renderer(model, height=512, width=512) as renderer:
                m0_row = m0_rollout.run(ReplayAdapter(actions), adapters.state_p2, model, data, names, renderer, m0_trial, theirs)
            a, b = trace_of(mine), trace_of(theirs)
            self.assertEqual(sorted(a), sorted(TRACE_KEYS))
            self.assertEqual(sorted(a), sorted(b))
            for key in TRACE_KEYS:
                self.assertEqual((a[key].dtype, a[key].shape), (b[key].dtype, b[key].shape), key)
                self.assertEqual(a[key].tobytes(), b[key].tobytes(), key)
            self.assertTrue(row["task_success"])
            self.assertGreater(len(a["raw_action"]), 300)
            for key, value in m0_row.items():
                if key != "wall_seconds":
                    self.assertEqual(row[key], value, key)
            self.assertEqual(json.loads((mine / "episode.json").read_text())["ticks"], len(a["raw_action"]))

    def test_the_new_fields_of_a_nominal_trial(self):
        layout = make_layout(1, "milk", ("can",))
        with tempfile.TemporaryDirectory() as tmp:
            row = run_trial(layout, trial_for(layout), ReplayAdapter(expert_actions(layout)), Path(tmp))
        self.assertEqual((row["grasp_lift"], row["disturbance_fired"], row["disturbance_tick"], row["recovered"]), (True, False, None, False))
        self.assertGreater(row["max_lift_m"], 0.15)
        self.assertEqual((row["axis"], row["level"], row["max_ticks"]), ("nominal", "nominal", 900))


class Ticks(unittest.TestCase):
    def test_the_tick_cap_is_the_trials(self):
        layout = make_layout(1, "milk", ("can",))
        with tempfile.TemporaryDirectory() as tmp:
            for cap in (7, 12):
                folder = Path(tmp) / f"cap{cap}"
                folder.mkdir()
                model, data, resolved = scenes.restore(layout, NOMINAL)
                with mujoco.Renderer(model, height=512, width=512) as renderer:
                    row = rollout.run(Constant(hold(model, data)), adapters.state_p2, model, data, resolved, renderer,
                                      dict(trial_for(layout), max_ticks=cap), folder)
                self.assertEqual((row["ticks"], row["outcome"], row["task_success"]), (cap, "timeout", False))

    def test_a_nonfinite_action_is_an_outcome_not_a_missing_trial(self):
        layout = make_layout(1, "milk", ("can",))
        with tempfile.TemporaryDirectory() as tmp:
            model, data, resolved = scenes.restore(layout, NOMINAL)
            with mujoco.Renderer(model, height=512, width=512) as renderer:
                row = rollout.run(Constant(hold(model, data), nan_after=3), adapters.state_p2, model, data, resolved, renderer,
                                  trial_for(layout), Path(tmp))
            trace = trace_of(tmp)
        self.assertEqual((row["outcome"], row["task_success"], row["cause"]), ("nonfinite_action", False, "never_lifted"))
        self.assertEqual((row["ticks"], len(trace["applied_action"])), (4, 3))      # the bad action is recorded, never applied

    def test_the_target_must_come_first(self):
        layout = make_layout(1, "milk", ("can",))
        model, data, resolved = scenes.restore(layout, NOMINAL)
        bad = dict(resolved, names=list(reversed(resolved["names"])))
        with tempfile.TemporaryDirectory() as tmp, mujoco.Renderer(model, height=512, width=512) as renderer:
            with self.assertRaises(ValueError):
                rollout.run(Constant(hold(model, data)), adapters.state_p2, model, data, bad, renderer, trial_for(layout), Path(tmp))


class TableOffset(unittest.TestCase):
    def test_a_raised_table_trial_succeeds_and_names_the_bin_it_rests_in(self):
        layout = make_layout(2, "can", ("bread", "lemon"))
        condition = levels.spatial("table_up10", layout, "x", 0)
        layout = dict(layout, axes=dict(spatial=dict(level="table_up10", condition=condition.to_dict())))
        with tempfile.TemporaryDirectory() as tmp:
            row = run_trial(layout, trial_for(layout, "spatial"), ReplayAdapter(expert_actions(layout, condition)), Path(tmp))
        self.assertTrue(row["task_success"])
        self.assertEqual((row["placed_bin"], row["cause"]), ("blue", "success"))     # props_scene's absolute test would say None here
        self.assertTrue(row["grasp_lift"])


class Move(unittest.TestCase):
    HOME_XY = np.array([0.30701957, 0.0])      # the grasp centre at Q_HOME, measured 2026-11-09

    @classmethod
    def move_layouts(cls):
        return [l for l in planned(24) if l["axes"]["disturbance"]["level"] == "move"]

    @classmethod
    def setUpClass(cls):
        cls.layout = next(l for l in cls.move_layouts() if np.hypot(*(np.asarray(l["start"][:2]) - cls.HOME_XY)) > 0.07)
        cls.condition = Condition.from_dict(cls.layout["axes"]["disturbance"]["condition"])
        cls.actions = expert_actions(cls.layout)
        with tempfile.TemporaryDirectory() as tmp:
            cls.row = run_trial(cls.layout, trial_for(cls.layout, "disturbance"), ReplayAdapter(cls.actions), Path(tmp))
            cls.trace = trace_of(tmp)

    def test_a_layout_that_starts_under_the_hand_fires_at_tick_zero(self):
        """The rule is literal: the first tick within 6 cm. 12 of 144 benchmark layouts start within 6 cm of the home grasp centre."""
        layout = next(l for l in self.move_layouts() if np.hypot(*(np.asarray(l["start"][:2]) - self.HOME_XY)) <= 0.05)
        with tempfile.TemporaryDirectory() as tmp:
            row = run_trial(layout, trial_for(layout, "disturbance"), ReplayAdapter(expert_actions(layout)), Path(tmp))
            first = trace_of(tmp)["props_xyz"][0, 0, :2]
        self.assertEqual((row["disturbance_fired"], row["disturbance_tick"]), (True, 0))
        move_to = layout["axes"]["disturbance"]["condition"]["move_to"]
        self.assertLess(float(np.abs(first - np.asarray(move_to)).max()), 1e-9)

    def first_tick_within_six_cm(self):
        """Independent of the rollout: forward kinematics of the recorded arm state, finger midpoint against the layout's start."""
        model, _ = compose.model_for(self.layout, NOMINAL)
        data = mujoco.MjData(model)
        left, right = model.body("left_finger").id, model.body("right_finger").id
        start = np.asarray(self.layout["start"][:2])
        for tick, state in enumerate(self.trace["state"]):
            data.qpos[:9] = state
            mujoco.mj_kinematics(model, data)
            mid = (data.xpos[left] + data.xpos[right])[:2] / 2
            if np.hypot(*(mid - start)) <= 0.06:
                return tick
        return None

    def test_fires_at_the_first_tick_within_six_cm_and_moves_the_target_exactly_eight_cm(self):
        self.assertTrue(self.row["disturbance_fired"])
        fired = self.row["disturbance_tick"]
        self.assertEqual(fired, self.first_tick_within_six_cm())
        target = self.trace["props_xyz"][:, 0, :]
        start, move_to = np.asarray(self.layout["start"][:2]), np.asarray(self.condition.move_to)
        self.assertAlmostEqual(float(np.hypot(*(move_to - start))), 0.08, places=12)
        self.assertLess(float(np.abs(target[fired, :2] - move_to).max()), 1e-9)
        if fired > 0:
            self.assertLess(float(np.abs(target[fired - 1, :2] - start).max()), 2e-3)
        self.assertEqual(self.row["max_ticks"], 1200)

    def test_it_fires_once_the_target_stays_where_it_was_put_until_the_hand_touches_it(self):
        fired = self.row["disturbance_tick"]
        target = self.trace["props_xyz"][:, 0, :2]
        step = np.abs(np.diff(target[fired:fired + 40], axis=0)).max(axis=1)
        self.assertLess(float(step.max()), 5e-3)

    def test_the_open_loop_expert_then_misses_its_grasp_and_the_trial_is_not_recovered(self):
        self.assertFalse(self.row["task_success"])
        self.assertFalse(self.row["recovered"])

    def test_nothing_fires_when_a_close_command_comes_first(self):
        self.assertGreater(self.first_tick_within_six_cm(), 0)          # the hand is not already there at tick 0
        closed = self.actions.copy()
        closed[:, 7] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            row = run_trial(self.layout, trial_for(self.layout, "disturbance"), ReplayAdapter(closed), Path(tmp))
        self.assertFalse(row["disturbance_fired"])
        self.assertIsNone(row["disturbance_tick"])

    def test_the_same_layout_under_nominal_conditions_never_fires_and_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = run_trial(self.layout, trial_for(self.layout), ReplayAdapter(self.actions), Path(tmp))
        self.assertEqual((row["disturbance_fired"], row["task_success"]), (False, True))


class Drop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = next(l for l in planned(8) if l["axes"]["disturbance"]["level"] == "drop")
        cls.actions = expert_actions(cls.layout)
        with tempfile.TemporaryDirectory() as tmp:
            cls.row = run_trial(cls.layout, trial_for(cls.layout, "disturbance"), ReplayAdapter(cls.actions), Path(tmp))
            cls.trace = trace_of(tmp)

    def test_fires_at_the_first_tick_five_cm_above_the_start_and_only_then(self):
        z = self.trace["props_xyz"][:, 0, 2]
        lifted = np.flatnonzero(z - z[0] >= 0.05)
        self.assertEqual(self.row["disturbance_tick"], int(lifted[0]))
        self.assertTrue(self.row["disturbance_fired"])
        self.assertGreater(int(lifted[0]), 30)          # not during the approach: the target had to be carried up first

    def test_the_gripper_is_forced_open_for_exactly_ten_ticks_and_the_policy_keeps_the_rest(self):
        t = self.row["disturbance_tick"]
        applied, raw = self.trace["applied_action"], self.trace["raw_action"]
        expected_raw = np.stack([self.actions[min(i, len(self.actions) - 1)] for i in range(len(raw))])
        np.testing.assert_array_equal(raw, expected_raw)                           # the policy's output is recorded untouched
        self.assertTrue(np.all(raw[t:t + 10, 7] == 0.0))                           # the expert was holding: the override is real
        self.assertTrue(np.all(applied[t:t + 10, 7] == 255.0))
        np.testing.assert_allclose(applied[t:t + 10, :7], raw[t:t + 10, :7], rtol=0, atol=1e-6)    # the arm commands are the policy's
        self.assertTrue(np.all(applied[t + 10:t + 14, 7] == raw[t + 10:t + 14, 7]))          # control is back after ten ticks
        self.assertTrue(np.all(applied[t - 3:t, 7] == raw[t - 3:t, 7]))                       # and nothing changed before

    def test_the_lift_is_recorded_and_an_open_loop_policy_does_not_recover(self):
        self.assertTrue(self.row["grasp_lift"])
        self.assertGreaterEqual(self.row["max_lift_m"], 0.05)
        self.assertEqual(self.row["recovered"], bool(self.row["task_success"] and self.row["disturbance_fired"]))
        self.assertEqual(self.row["max_ticks"], 1200)


class EveryAxisRuns(unittest.TestCase):
    """A trial under each kind of condition runs end to end and records its condition."""

    def test_lighting_background_and_distractor_conditions_run_with_the_experts_actions(self):
        layout = planned(8)[3]
        actions = expert_actions(layout)
        for axis in ("lighting", "background", "distractors"):
            with tempfile.TemporaryDirectory() as tmp:
                row = run_trial(layout, trial_for(layout, axis), ReplayAdapter(actions), Path(tmp))
                self.assertEqual((row["axis"], row["level"], row["condition"]),
                                 (axis, layout["axes"][axis]["level"], layout["axes"][axis]["condition"]))
                self.assertEqual(trace_of(tmp)["props_xyz"].shape[1], len(row["names"]))
                self.assertEqual(row["names"][0], layout["object"])


if __name__ == "__main__":
    unittest.main()
