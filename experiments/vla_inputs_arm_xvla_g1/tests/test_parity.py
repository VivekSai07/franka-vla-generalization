"""Train/rollout parity of G1's 17-value state. Robosuite image, CPU (OSMesa renders every tick); no policy.

G0's own rollout loop (`gen_eval.rollout.run`, the loop every G1 trial runs) is driven by the recorded actions of sealed G1 episodes,
in the scene rebuilt from the plan with `gen_scene.compose`. The state it hands the policy at every tick must equal, byte for byte,
the row the training loader builds for that frame, and the normalised, padded proprio tensor must equal the one the training
processors build. The bin the rollout is told to fill is the wrong one, so the episode never ends early and all 452 ticks run.
Episodes: 397 (no distractors, middle table height, P2's start region: the closest the plan comes to the nominal scene), 2155 (dark,
three distractors) and 458 (bright, raised table); the last two have clipped frames, so the clip is part of what is proven.
"""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

EPISODES = (397, 2155, 458)
FRAMES = 452
TICKS = (0, 120, 260, 451)


def modules():
    from experiments.gen_g1_data import plan as plan_module
    from experiments.vla_inputs_arm_xvla_g1 import adapter
    from experiments.xvla_g1_rand import contract, forces
    return plan_module, adapter, contract, forces


def mounted():
    return all(path.exists() for path in (modules()[3].SEALED / "seal.json", modules()[3].SEALED / "episodes"))


def sealed_rows(index):
    """The sealed dataset's (action, state) rows of one episode, float32 (the files the trainer reads)."""
    import pyarrow.parquet as pq
    root = modules()[3].SEALED / "dataset"
    rows = [r for path in sorted((root / "meta/episodes").rglob("*.parquet")) for r in pq.read_table(path).to_pylist()
            if r["episode_index"] == index]
    if len(rows) != 1:
        raise ValueError(f"episode {index} is not in the sealed dataset exactly once")
    path = root / f"data/chunk-{rows[0]['data/chunk_index']:03d}/file-{rows[0]['data/file_index']:03d}.parquet"
    table = pq.read_table(path, columns=["episode_index", "frame_index", "action", "observation.state"]).to_pandas()
    table = table[table["episode_index"] == index].sort_values("frame_index")
    if table["frame_index"].tolist() != list(range(rows[0]["length"])):
        raise ValueError(f"episode {index}: frame_index is not 0..length-1")
    return np.stack(table["action"].to_numpy()).astype(np.float32), np.stack(table["observation.state"].to_numpy()).astype(np.float32)


class ReplayPolicy:
    """Plays the recorded actions and keeps the observations it was given, exactly as the rollout built them."""
    name = "replay"

    def __init__(self, actions):
        self.actions, self.observations, self.calls = actions, [], 0

    def reset(self):
        self.calls = 0

    def act(self, obs):
        self.observations.append({"state": obs["state"].copy(), "instruction": obs["instruction"]})
        action = self.actions[self.calls]
        self.calls += 1
        return action.copy()


def rollout_states(index):
    """(trace arrays, sealed npz arrays, sealed dataset rows) of one sealed episode driven through G0's rollout loop."""
    import mujoco
    from data_collection.benchmark import capture_state
    from experiments.gen_eval import rollout, scenes
    plan_module, adapter, contract, forces = modules()
    plan = json.loads((forces.SEALED / "plan.json").read_text())["episodes"][index]
    with np.load(forces.SEALED / "episodes" / f"episode_{index:04d}.npz", allow_pickle=False) as npz:
        sealed = {key: npz[key] for key in ("qpos", "initial_state", "ctrl")}
        # a retried attempt drew its own home pose; the sealed episode records the one it was collected with
        plan = dict(plan, home_pose_jitter_deg=[float(x) for x in npz["home_pose_jitter_deg"]])
    action, state = sealed_rows(index)
    condition = plan_module.condition_of(plan)
    model, data, resolved = scenes.restore(plan, condition)
    if not np.array_equal(capture_state(model, data), sealed["initial_state"]):
        raise ValueError(f"episode {index}: the rebuilt scene is not the sealed initial state")
    wrong_bin = "orange" if plan["bin"] == "blue" else "blue"
    trial = dict(ordinal=1, layout_id=plan["layout_id"], tier="parity", seed=plan["seed"], seed_index=0, object=plan["object"],
                 bin=wrong_bin, instruction=plan["instruction"], max_ticks=FRAMES, condition=condition.to_dict(), axis="nominal",
                 level="nominal", kind="trained")
    policy = ReplayPolicy(action)
    with tempfile.TemporaryDirectory() as folder, mujoco.Renderer(model, height=512, width=512) as renderer:
        result = rollout.run(policy, adapter.state_g1, model, data, resolved, renderer, trial, Path(folder), condition)
        with np.load(Path(folder) / "trace.npz") as trace:
            arrays = {key: trace[key] for key in trace.files}
    return dict(result=result, trace=arrays, observations=policy.observations, sealed=sealed, action=action, state=state, plan=plan)


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not mounted():
            raise unittest.SkipTest("the sealed G1 dataset is not mounted")
        cls.plan_module, cls.adapter, cls.c, cls.f = modules()
        cls.cache = {index: rollout_states(index) for index in EPISODES}

    def test_every_tick_runs_and_the_rollout_reproduces_the_sealed_trajectory(self):
        for index, run in self.cache.items():
            self.assertEqual(run["result"]["ticks"], FRAMES, index)
            self.assertEqual(run["result"]["outcome"], "timeout", index)
            np.testing.assert_array_equal(run["trace"]["state"][:, :9], run["sealed"]["qpos"][:, :9], err_msg=str(index))

    def test_the_state_the_policy_is_given_equals_the_training_row_byte_for_byte_at_every_frame(self):
        c = self.c
        for index, run in self.cache.items():
            given = np.stack([o["state"] for o in run["observations"]])
            self.assertEqual(given.shape, (FRAMES, 17))
            for tick in range(FRAMES):
                row = c.joined_state(torch.from_numpy(run["state"][tick]), index, tick).numpy()
                live = np.asarray(given[tick], dtype=np.float32)
                self.assertEqual(live.tobytes(), row.tobytes(), (index, tick))

    def test_the_trace_records_what_the_policy_was_given(self):
        for index, run in self.cache.items():
            np.testing.assert_array_equal(run["trace"]["state"], np.stack([o["state"] for o in run["observations"]]))

    def test_the_nine_joint_values_are_the_sealed_state_and_the_rest_are_g1s_standardised_signals(self):
        raw = self.f.load_raw(verify=False)["values"]
        k = self.f.constants()
        for index, run in self.cache.items():
            live = run["trace"]["state"]
            np.testing.assert_array_equal(live[:, :9].astype(np.float32), run["state"])
            expected = ((np.clip(raw[index], k["low"], k["high"]) - k["mean"]) / k["std"]).astype(np.float32)
            self.assertEqual(live[:, 9:].astype(np.float32).tobytes(), expected.tobytes(), index)

    def test_the_rollouts_signals_are_the_sealed_ones_before_the_clip(self):
        raw = self.f.load_raw(verify=False)["values"]
        for index, run in self.cache.items():
            np.testing.assert_array_equal(np.concatenate([run["trace"]["wrist_wrench"], run["trace"]["finger_force"]], axis=1), raw[index])

    def test_some_frames_are_clipped_so_the_clip_is_part_of_what_is_proven(self):
        raw = self.f.load_raw(verify=False)["values"]
        k = self.f.constants()
        for index in (2155, 458):
            self.assertGreater(int(((raw[index] < k["low"]) | (raw[index] > k["high"])).any(axis=1).sum()), 0, index)

    def test_a2s_constants_would_not_have_matched(self):
        from experiments.xvla_p2_force import forces as a2
        raw = self.f.load_raw(verify=False)["values"]
        index = EPISODES[1]
        mine = self.cache[index]["trace"]["state"][:, 9:].astype(np.float32)
        theirs = a2.standardise(raw[index], a2.constants())
        self.assertNotEqual(mine.tobytes(), theirs.tobytes())


class NominalRawParityTests(unittest.TestCase):
    """P2's nominal scene, exact S0 replay of sealed P2 episodes: the builder's extras are G1's standardisation of the sidecar's rows."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.replay = importlib.import_module("experiments.vla_inputs_signals.replay")
            cls.a2 = importlib.import_module("experiments.xvla_p2_force.forces")
        except Exception as error:  # pragma: no cover
            raise unittest.SkipTest(f"P2 replay unavailable: {error}")
        if not (cls.a2.SIDECAR / "seal.json").exists():
            raise unittest.SkipTest("the S0 sidecar is not mounted")
        cls.plan_module, cls.adapter, cls.c, cls.f = modules()

    def test_the_builder_gives_g1s_standardisation_of_the_sealed_nominal_signals(self):
        raw = self.a2.load_raw(verify=False)["values"]
        k = self.f.constants()
        for index in (0, 18):
            target = self.replay.load_episode(index)["episode"]["object"]
            seen = {}

            def on_frame(tick, model, data):
                seen[tick] = self.adapter.state_g1(model, data, target)

            self.replay.frames(index, on_frame)
            live = np.stack([seen[t] for t in range(FRAMES)])
            expected = ((np.clip(raw[index], k["low"], k["high"]) - k["mean"]) / k["std"]).astype(np.float32)
            self.assertEqual(live[:, 9:].astype(np.float32).tobytes(), expected.tobytes(), index)


class PolicyInputParityTests(unittest.TestCase):
    """The normalised, padded tensor the policy receives, not only the raw concatenation."""

    @classmethod
    def setUpClass(cls):
        if not mounted():
            raise unittest.SkipTest("the sealed G1 dataset is not mounted")
        cls.plan_module, cls.adapter, cls.c, cls.f = modules()
        if not (cls.c.CACHE / "tokenizer").exists():
            raise unittest.SkipTest("base bundle not mounted")
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
        config = cls.c.make_config("cpu")
        cls.policy = XVLAPolicy.__new__(XVLAPolicy)
        torch.nn.Module.__init__(cls.policy)
        cls.policy.config, cls.policy.model = config, SimpleNamespace(dim_proprio=20, dim_action=20)
        cls.pre, _ = make_xvla_pre_post_processors(config, {"action": {"mean": torch.zeros(8), "std": torch.ones(8)}})
        cls.index = EPISODES[1]
        cls.rolled = rollout_states(cls.index)

    def rollout_proprio(self, tick):
        """What `XVLAG1Adapter.act` hands the processors for the state the rollout built, then what the policy builds from it."""
        from experiments.xvla_props_p2_eval import protocol
        c = self.c
        instruction = c.planned_instructions()[self.index]
        frame = {"observation.state": torch.tensor(self.rolled["observations"][tick]["state"], dtype=torch.float32), "task": instruction,
                 **{key: torch.zeros(3, 512, 512) for key in c.CAMERAS}}
        with protocol.eval_instruction(instruction):
            batch = self.pre(c.observation(frame))
        return self.policy._build_model_inputs(batch)["proprio"]

    def test_the_normalised_and_padded_proprio_is_identical_in_training_and_rollout(self):
        c = self.c
        rows = []
        for tick in TICKS:
            rows.append({"episode_index": torch.tensor(self.index), "task": c.planned_instructions()[self.index],
                         "observation.state": c.joined_state(torch.from_numpy(self.rolled["state"][tick]), self.index, tick),
                         **{key: torch.zeros(3, 512, 512) for key in c.CAMERAS},
                         "action": torch.zeros(30, 8), "action_is_pad": torch.zeros(30, dtype=torch.bool)})
        trained = self.policy._build_model_inputs(self.pre(c.collate_rows(rows)))["proprio"]
        self.assertEqual(tuple(trained.shape), (4, 20))
        for k, tick in enumerate(TICKS):
            live = self.rollout_proprio(tick)
            self.assertEqual(live.numpy().tobytes(), trained[k:k + 1].numpy().tobytes(), tick)
            self.assertEqual(torch.count_nonzero(live[:, 17:]).item(), 0)
            self.assertGreater(torch.count_nonzero(live[:, 9:17]).item(), 0)

    def test_a_nine_value_state_is_not_accepted_by_the_rollout_check(self):
        c = self.c
        frame = {"observation.state": torch.zeros(9), "task": c.planned_instructions()[self.index],
                 **{key: torch.zeros(3, 512, 512) for key in c.CAMERAS}}
        from experiments.xvla_props_p2_eval import protocol
        with protocol.eval_instruction(frame["task"]), self.assertRaises(ValueError):
            c.observation(frame)


if __name__ == "__main__":
    unittest.main()
