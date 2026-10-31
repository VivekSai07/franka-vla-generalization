import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import robosuite  # noqa: F401
except ImportError:  # the admission image has no robosuite
    raise unittest.SkipTest("needs the robosuite image")
import mujoco

from experiments.vla_inputs_eval import adapters, rollout, scenes
from experiments.vla_inputs_signals import contract as s0

P2_ROOT = Path(__file__).resolve().parents[3] / "outputs/props_p2_data/props_p2_20261013"
EPISODE = 271  # a can episode of the sealed P2 plan


def sealed_episode(index=EPISODE):
    return json.loads((P2_ROOT / "plan.json").read_text())["episodes"][index]


def trial_of(episode, **extra):
    return dict(layout_id=episode["layout_id"], object=episode["object"], bin=episode["bin"], seed=episode["seed"],
                instruction=episode["instruction"], **extra)


class ReplayAdapter:
    """Plays the sealed expert actions of a P2 episode back as the policy."""
    name = "replay"

    def __init__(self, actions):
        self.actions, self.tick = actions, 0

    def reset(self):
        self.tick = 0

    def act(self, obs):
        action = self.actions[self.tick]
        self.tick += 1
        return action


class ConstantAdapter:
    name = "constant"

    def __init__(self, value=None, bad_at=None, shape=(8,)):
        self.value, self.bad_at, self.shape, self.tick = value, bad_at, shape, 0

    def reset(self):
        self.tick = 0

    def act(self, obs):
        if self.shape != (8,):
            return np.zeros(self.shape, dtype=np.float32)
        action = np.zeros(self.shape, dtype=np.float32) if self.value is None else self.value.copy()
        if self.value is None:
            action[:7] = obs["state"][:7]
            action[7] = 255.0
        if self.bad_at == self.tick:
            action[0] = np.nan
        self.tick += 1
        return action


def run_trial(adapter, episode, folder, **kwargs):
    model, data, names = scenes.restore(episode)
    with mujoco.Renderer(model, height=512, width=512) as renderer:
        return rollout.run(adapter, adapters.state_p2, model, data, names, renderer, trial_of(episode), Path(folder), **kwargs)


def sealed_actions(index=EPISODE):
    from experiments.vla_inputs_signals import replay
    return replay.load_episode(index)["action"]


class ExpertReplayThroughTheRealLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.row = run_trial(ReplayAdapter(sealed_actions()), sealed_episode(), cls.folder)
        with np.load(cls.folder / "trace.npz") as stored:
            cls.trace = {k: stored[k] for k in stored.files}
        sidecar = s0.SIDECAR_ROOT / "p2_signals_20261030/episodes" / f"episode_{EPISODE:03d}.npz"
        with np.load(sidecar) as stored:
            cls.sidecar = {k: stored[k] for k in stored.files}
        with np.load(P2_ROOT / "episodes" / f"episode_{EPISODE:03d}.npz") as stored:
            cls.sealed = {k: stored[k] for k in stored.files}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_experts_own_actions_succeed(self):
        self.assertEqual(self.row["outcome"], "success", self.row["cause"])
        self.assertEqual(self.row["cause"], "success")
        self.assertEqual(self.row["names"][0], "can")

    def test_every_output_file_is_written(self):
        for name in ("cameras.mp4", "trace.npz", "episode.json"):
            self.assertGreater((self.folder / name).stat().st_size, 0, name)
        self.assertEqual(json.loads((self.folder / "episode.json").read_text())["ticks"], self.row["ticks"])

    def test_trace_keys_and_lengths(self):
        ticks = self.row["ticks"]
        self.assertEqual(set(self.trace), set(rollout.TRACE_KEYS) | set(rollout.SIGNAL_KEYS))
        self.assertEqual(self.trace["wrist_wrench"].shape, (ticks, 6))
        self.assertEqual(self.trace["finger_force"].shape, (ticks, 2))
        self.assertEqual(self.trace["target_box"].shape, (ticks, 2, 5))
        self.assertEqual(self.trace["state"].shape, (ticks, 9))
        self.assertEqual(self.trace["raw_action"].shape, (ticks, 8))

    def test_the_rollout_trajectory_is_the_sealed_one(self):
        ticks = self.row["ticks"]
        self.assertTrue(np.array_equal(self.trace["state"], self.sealed["qpos"][:ticks, :9]))

    def test_the_raw_action_is_recorded_exactly_as_the_adapter_returned_it(self):
        ticks = self.row["ticks"]
        self.assertTrue(np.array_equal(self.trace["raw_action"], sealed_actions()[:ticks]))

    def test_the_signal_tap_equals_the_sealed_sidecar_byte_for_byte(self):
        """Train/rollout parity: the evaluator's signals on the expert's trajectory are the S0 sidecar's."""
        ticks = self.row["ticks"]
        for key in rollout.SIGNAL_KEYS:
            self.assertTrue(np.array_equal(self.trace[key], self.sidecar[key][:ticks]), key)
        self.assertGreater(float(self.trace["finger_force"].max()), 1.0)  # the grasp really was recorded


class FailureModes(unittest.TestCase):
    def test_a_nonfinite_action_is_an_outcome_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = run_trial(ConstantAdapter(bad_at=3), sealed_episode(), tmp)
            self.assertEqual((row["outcome"], row["ticks"], row["task_success"]), ("nonfinite_action", 4, False))
            with np.load(Path(tmp) / "trace.npz") as stored:
                self.assertEqual(stored["raw_action"].shape[0], 4)
                self.assertEqual(stored["applied_action"].shape[0], 3)

    def test_holding_still_times_out_as_never_lifted(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = run_trial(ConstantAdapter(), sealed_episode(), tmp, max_ticks=30)
            self.assertEqual((row["outcome"], row["ticks"], row["cause"]), ("timeout", 30, "never_lifted"))
            self.assertIsNone(row["first_lifted"])

    def test_a_wrong_shape_action_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_trial(ConstantAdapter(shape=(7,)), sealed_episode(), tmp, max_ticks=3)

    def test_the_target_must_come_first(self):
        episode = sealed_episode()
        model, data, names = scenes.restore(episode)
        with tempfile.TemporaryDirectory() as tmp, mujoco.Renderer(model, height=512, width=512) as renderer:
            with self.assertRaises(ValueError):
                rollout.run(ConstantAdapter(), adapters.state_p2, model, data, names[::-1], renderer, trial_of(episode),
                            Path(tmp), max_ticks=2)

    def test_the_adapter_is_reset_once_before_the_first_act_and_the_trial_is_seeded(self):
        import random
        import torch
        events = []

        class Spy(ConstantAdapter):
            def reset(self):
                events.append("reset")
                super().reset()

            def act(self, obs):
                if self.tick == 0:
                    events.append(("act", random.random(), float(np.random.rand()), float(torch.rand(1))))
                return super().act(obs)

        episode = sealed_episode()
        with tempfile.TemporaryDirectory() as tmp:
            run_trial(Spy(), episode, tmp, max_ticks=2)
        random.seed(episode["seed"])
        np.random.seed(episode["seed"])
        torch.manual_seed(episode["seed"])
        self.assertEqual(events, ["reset", ("act", random.random(), float(np.random.rand()), float(torch.rand(1)))])

    def test_the_state_builder_is_what_the_adapter_sees_and_what_is_traced(self):
        seen = []

        class Spy(ConstantAdapter):
            def act(self, obs):
                seen.append(obs["state"])
                return super().act(obs)

        def doubled(model, data, target):
            return data.qpos[:9] * 2.0

        episode = sealed_episode()
        model, data, names = scenes.restore(episode)
        with tempfile.TemporaryDirectory() as tmp, mujoco.Renderer(model, height=512, width=512) as renderer:
            rollout.run(Spy(), doubled, model, data, names, renderer, trial_of(episode), Path(tmp), max_ticks=2)
            with np.load(Path(tmp) / "trace.npz") as stored:
                self.assertTrue(np.array_equal(stored["state"][0], seen[0]))
        self.assertTrue(np.array_equal(seen[0], scenes.restore(episode)[1].qpos[:9] * 2.0))


if __name__ == "__main__":
    unittest.main()
