"""The live identity gate (spec D9 iii): comparison, the verdict file, and every way it must refuse or fail. Pure numpy and json;
admission image.

The rollouts themselves need the GPU and run only through `scripts/gen_eval.sh identity_gate`; here the runner is a stand-in that
copies or alters A2's recorded trial, so what is tested is that the gate recognises identity and refuses to call anything else
identical. Adapted from `vla_inputs_m3b_analysis/tests/test_identity_gate.py`.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.gen_eval import identity_gate
from experiments.vla_inputs_eval import protocol as m0_protocol

KEYS = ("state", "raw_action", "applied_action", "props_xyz", "xy_margin_m", "z_m", "lowest_z_m", "gripper_open", "finger_contact",
        "wrist_wrench", "finger_force", "target_box")
CORE = identity_gate.M0_CORE_SHA256
NEW_FIELDS = dict(axis="nominal", level="nominal", condition={"axis": "nominal"}, max_ticks=900, max_lift_m=0.2, grasp_lift=True,
                  disturbance_fired=False, disturbance_tick=None, recovered=False)


class StubAdapter:
    name = "stub"

    def reset(self):
        pass

    def act(self, obs):
        raise AssertionError("the stub runner never asks the adapter to act")


def layout(i):
    return dict(layout_id=f"L{i:03d}", tier="comparison", seed=1000 + i, object="milk", bin="blue", blue_slot="left",
                cell=["milk", "blue", "left", 1], instruction="put the milk in the blue bin", distractor_count=1,
                distractors=[dict(object="can", pose=[0.3, 0.1, 10.0])], start=[0.35, -0.1, 20.0], home_pose_jitter_deg=[0.0] * 7,
                held_out=False)


def arrays(ticks=40, seed=0):
    rng = np.random.default_rng(seed)
    out = {key: rng.random((ticks, 3)).astype(np.float64) for key in KEYS}
    out["gripper_open"] = rng.random(ticks) > 0.5
    out["target_box"] = rng.random((ticks, 2, 5))
    out["raw_action"] = rng.random((ticks, 8)).astype(np.float32)
    return out


def episode_of(trial, **kw):
    return dict(trial, outcome="success", task_success=True, ticks=40, placed_bin="blue", final_gripper_open=True, cause="success",
                first_lifted="milk", wrong_object=False, distractor_displacement_m={"can": 0.0}, names=["milk", "can"], wall_seconds=12.5, **kw)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.campaign, self.out = base / "campaign", base / "scratch"
        self.out.mkdir()
        n = 300
        self.plan = dict(layouts={f"L{i:03d}": layout(i) for i in range(n)}, tiers={"comparison": [f"L{i:03d}" for i in range(n)]})
        self.feas = dict(feasible=[f"L{i:03d}" for i in range(n)])
        self.trials = m0_protocol.stage_trials(self.plan, self.feas, "comparison")
        self.recorded_dir = m0_protocol.stage_dir(self.campaign, "A2", "comparison", 88830)
        self.recorded = {}
        for trial in self.trials:
            if trial["ordinal"] in (1, 291):
                folder = m0_protocol.trial_folder(self.recorded_dir, trial)
                folder.mkdir(parents=True)
                self.recorded[trial["ordinal"]] = arrays(seed=trial["ordinal"])
                np.savez_compressed(folder / "trace.npz", **self.recorded[trial["ordinal"]])
                (folder / "episode.json").write_text(json.dumps(episode_of(trial)))
                (folder / "cameras.mp4").write_bytes(b"video-%d" % trial["ordinal"])
        (self.campaign / "arms/A2").mkdir(parents=True, exist_ok=True)
        (self.campaign / "arms/A2/prepared.json").write_text(json.dumps(dict(
            core=dict(sha256=CORE), arm=dict(name="A2", adapter=identity_gate.IDENTIFIER), phase_checkpoint=101520,
            checkpoints={"88830": dict(path="/ckpt/step_088830", model_sha256="x")})))

    def snapshot(self, root):
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(Path(root).rglob("*")) if p.is_file()}

    def copy_runner(self, change=None, wall=99.0, drop_field=None):
        """A runner that 'rolls out' by copying A2's recorded trial, then optionally changing it; the episode gains G0's new fields."""
        ran, seen = [], []

        def runner(adapter, state_builder, layout_record, trial, folder):
            ran.append(trial["ordinal"])
            seen.append(trial)
            source = m0_protocol.trial_folder(self.recorded_dir, next(t for t in self.trials if t["ordinal"] == trial["ordinal"]))
            with np.load(source / "trace.npz") as trace:
                data = {k: trace[k].copy() for k in trace.files}
            if change:
                change(trial["ordinal"], data)
            np.savez_compressed(folder / "trace.npz", **data)
            episode = json.loads((source / "episode.json").read_text())
            episode.update(NEW_FIELDS)
            episode["wall_seconds"] = wall
            if drop_field:
                episode.pop(drop_field)
            (folder / "episode.json").write_text(json.dumps(episode))
            (folder / "cameras.mp4").write_bytes(b"another-encoding")
            return episode

        runner.ran, runner.seen = ran, seen
        return runner

    def run_gate(self, runner=None, ordinals=(1, 291), **kwargs):
        return identity_gate.run(self.campaign, self.out, 88830, ordinals, verify=False, factory=lambda prepared, checkpoint: StubAdapter(),
                                 runner=runner or self.copy_runner(), manifests=lambda campaign: (self.plan, self.feas), **kwargs)


class Pinned(unittest.TestCase):
    def test_the_gate_is_pinned_before_any_run(self):
        self.assertEqual(identity_gate.IDENTIFIER, "experiments.vla_inputs_arm_xvla_force.adapter:XVLAForceAdapter")
        self.assertEqual((identity_gate.ORDINALS, identity_gate.CHECKPOINT, identity_gate.ARM), ((1, 291), 88830, "A2"))
        self.assertEqual(identity_gate.M0_CORE_SHA256, "c4f6c303f2ae37509530323efe7d190686b1b5258910863f041241d30bc5b39d")
        self.assertEqual(identity_gate.IGNORED_EPISODE_FIELDS, ("wall_seconds",))

    def test_the_recorded_trials_are_the_two_the_spec_names(self):
        """Spec D9 iii: A2's recorded M0 comparison trials 1 and 291."""
        self.assertEqual(identity_gate.ORDINALS, (1, 291))
        self.assertEqual(identity_gate.ARM, "A2")


class WorkerRuntime(unittest.TestCase):
    """The gate must run under the very runtime M0's worker sets (`evaluate.main`): the M3b gate's first live attempt did not, and its
    tick-0 action already differed from the recorded one by 3e-4 (FAILURES #70)."""

    def fake_torch(self, calls):
        import sys
        import types
        torch = types.ModuleType("torch")
        torch.use_deterministic_algorithms = lambda flag: calls.append(("deterministic", flag))
        previous = sys.modules.get("torch")
        sys.modules["torch"] = torch
        self.addCleanup(lambda: sys.modules.__setitem__("torch", previous) if previous is not None else sys.modules.pop("torch", None))

    def test_configure_runtime_sets_what_the_worker_sets(self):
        import os
        calls = []
        self.fake_torch(calls)
        saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "CUBLAS_WORKSPACE_CONFIG")}
        self.addCleanup(lambda: [os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v) for k, v in saved.items()])
        for key in saved:
            os.environ.pop(key, None)
        identity_gate.configure_runtime()
        self.assertEqual(calls, [("deterministic", True)])
        self.assertEqual({k: os.environ[k] for k in saved}, dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                                                                 CUBLAS_WORKSPACE_CONFIG=":4096:8"))

    def test_the_command_line_configures_the_runtime_before_anything_runs(self):
        from unittest.mock import patch
        order = []
        with patch.object(identity_gate, "configure_runtime", side_effect=lambda: order.append("configure")), \
                patch.object(identity_gate, "run", side_effect=lambda *a, **k: order.append("run") or dict(identical=True)):
            self.assertEqual(identity_gate.main(["--campaign", "/c", "--out", "/o"]), 0)
        self.assertEqual(order, ["configure", "run"])

    def test_the_worker_source_still_sets_the_same_runtime_in_both_evaluators(self):
        for rel in ("vla_inputs_eval/evaluate.py", "gen_eval/evaluate.py"):
            source = (Path(identity_gate.__file__).parents[1] / rel).read_text()
            self.assertIn("torch.use_deterministic_algorithms(True)", source, rel)
            self.assertIn('CUBLAS_WORKSPACE_CONFIG=":4096:8"', source, rel)


class CompareTrial(Fixture):
    def folders(self, change=None, episode_change=None):
        source = m0_protocol.trial_folder(self.recorded_dir, self.trials[0])
        mine = Path(tempfile.mkdtemp(dir=self.tmp.name)) / "mine"
        shutil.copytree(source, mine)
        if change:
            with np.load(mine / "trace.npz") as trace:
                data = {k: trace[k].copy() for k in trace.files}
            change(data)
            np.savez_compressed(mine / "trace.npz", **data)
        episode = json.loads((mine / "episode.json").read_text())
        episode.update(NEW_FIELDS)
        if episode_change:
            episode_change(episode)
        (mine / "episode.json").write_text(json.dumps(episode))
        return source, mine

    def test_identical_trials_are_identical_and_the_wall_clock_and_new_fields_are_ignored(self):
        source, mine = self.folders(episode_change=lambda e: e.update(wall_seconds=1.0))
        got = identity_gate.compare_trial(source, mine)
        self.assertTrue(got["identical"])
        self.assertEqual(got["differing_arrays"], [])
        self.assertEqual(sorted(got["arrays"]), sorted(KEYS))

    def test_one_ulp_in_one_array_is_a_difference_and_the_first_differing_tick_is_named(self):
        def change(data):
            data["raw_action"][17, 3] = np.nextafter(data["raw_action"][17, 3], np.float32(2.0))

        got = identity_gate.compare_trial(*self.folders(change))
        self.assertFalse(got["identical"])
        self.assertEqual(got["differing_arrays"], ["raw_action"])
        self.assertEqual(got["first_difference_tick"], 17)

    def test_a_different_dtype_shape_or_key_set_is_a_difference(self):
        for name, change in (("dtype", lambda d: d.update(state=d["state"].astype(np.float32))),
                             ("shape", lambda d: d.update(state=d["state"][:-1])),
                             ("same bytes, other dtype", lambda d: d.update(state=d["state"].view(np.int64))),
                             ("missing", lambda d: d.pop("finger_force")),
                             ("extra", lambda d: d.update(extra=np.zeros(3)))):
            with self.subTest(name):
                got = identity_gate.compare_trial(*self.folders(change))
                self.assertFalse(got["identical"])
                self.assertTrue(got["differing_arrays"] or got["key_mismatch"])

    def test_nan_in_the_same_place_is_identical_and_in_another_place_is_not(self):
        with np.load(m0_protocol.trial_folder(self.recorded_dir, self.trials[0]) / "trace.npz") as trace:
            base = {k: trace[k].copy() for k in trace.files}
        episode = (m0_protocol.trial_folder(self.recorded_dir, self.trials[0]) / "episode.json").read_text()

        def make(name, nan_at):
            data = {k: v.copy() for k, v in base.items()}
            if nan_at is not None:
                data["z_m"][nan_at] = np.nan
            folder = Path(self.tmp.name) / name
            folder.mkdir()
            np.savez_compressed(folder / "trace.npz", **data)
            (folder / "episode.json").write_text(episode)
            return folder

        self.assertTrue(identity_gate.compare_trial(make("a", (3, 0)), make("b", (3, 0)))["identical"])
        self.assertFalse(identity_gate.compare_trial(make("c", (3, 0)), make("d", (3, 1)))["identical"])
        self.assertFalse(identity_gate.compare_trial(make("e", (3, 0)), make("f", None))["identical"])

    def test_a_different_outcome_or_tick_count_in_episode_json_is_a_difference(self):
        for change in (lambda e: e.update(task_success=False), lambda e: e.update(ticks=41), lambda e: e.update(cause="timeout_carrying"),
                       lambda e: e.update(placed_bin="orange"), lambda e: e.update(distractor_displacement_m={"can": 0.5})):
            got = identity_gate.compare_trial(*self.folders(episode_change=change))
            self.assertFalse(got["identical"])
            self.assertTrue(got["differing_episode_fields"])

    def test_a_field_M0_recorded_that_the_rerun_lacks_is_a_difference(self):
        got = identity_gate.compare_trial(*self.folders(episode_change=lambda e: e.pop("first_lifted")))
        self.assertFalse(got["identical"])
        self.assertEqual(got["episode_fields_missing_from_rerun"], ["first_lifted"])

    def test_the_video_is_recorded_but_does_not_decide(self):
        source, mine = self.folders()
        (mine / "cameras.mp4").write_bytes(b"other")
        got = identity_gate.compare_trial(source, mine)
        self.assertTrue(got["identical"])
        self.assertFalse(got["video_identical"])


class Run(Fixture):
    def test_identical_reruns_pass_and_write_the_verdict_beside_the_scratch_trials_only(self):
        before = self.snapshot(self.campaign)
        runner = self.copy_runner()
        got = self.run_gate(runner)
        self.assertTrue(got["identical"])
        self.assertEqual(runner.ran, [1, 291])
        written = json.loads((self.out / "identity_gate/result.json").read_text())
        self.assertTrue(written["identical"])
        self.assertEqual((written["checkpoint"], written["adapter"], written["arm"]), (88830, identity_gate.IDENTIFIER, "A2"))
        self.assertEqual(self.snapshot(self.campaign), before)  # the real campaign is read, never written
        self.assertTrue(any(p.name == "trace.npz" for p in (self.out / "identity_gate").rglob("*")))

    def test_the_trial_it_runs_is_nominal_with_m0s_tick_cap_and_m0s_ordinal_and_layout(self):
        runner = self.copy_runner()
        self.run_gate(runner)
        from experiments.gen_scene.conditions import NOMINAL
        for trial in runner.seen:
            self.assertEqual((trial["axis"], trial["level"], trial["max_ticks"]), ("nominal", "nominal", 900))
            self.assertEqual(trial["condition"], NOMINAL.to_dict())
            m0 = next(t for t in self.trials if t["ordinal"] == trial["ordinal"])
            for key in ("layout_id", "seed", "object", "bin", "blue_slot", "start", "distractors", "home_pose_jitter_deg", "instruction"):
                self.assertEqual(trial[key], m0[key], key)

    def test_any_difference_in_any_trial_fails_the_gate(self):
        def change(ordinal, data):
            if ordinal == 291:
                data["wrist_wrench"][5, 2] += 1e-9

        got = self.run_gate(self.copy_runner(change))
        self.assertFalse(got["identical"])
        self.assertEqual({t["ordinal"]: t["identical"] for t in got["trials"]}, {1: True, 291: False})
        self.assertFalse(json.loads((self.out / "identity_gate/result.json").read_text())["identical"])

    def test_the_command_line_exits_nonzero_on_a_difference(self):
        self.assertEqual(identity_gate.exit_code(dict(identical=True)), 0)
        self.assertEqual(identity_gate.exit_code(dict(identical=False)), 1)

    def test_it_refuses_an_unknown_ordinal_an_incomplete_recorded_trial_and_an_existing_output(self):
        with self.assertRaisesRegex(ValueError, "ordinal"):
            self.run_gate(ordinals=(1, 5000))
        (m0_protocol.trial_folder(self.recorded_dir, self.trials[290]) / "trace.npz").unlink()
        with self.assertRaisesRegex(RuntimeError, "recorded"):
            self.run_gate()

    def test_a_verdict_is_never_overwritten(self):
        self.run_gate(ordinals=(1,))
        with self.assertRaises(FileExistsError):
            self.run_gate(ordinals=(291,))
        self.assertEqual(json.loads((self.out / "identity_gate/result.json").read_text())["ordinals"], [1])

    def test_it_refuses_a_campaign_that_is_its_own_output(self):
        with self.assertRaisesRegex(ValueError, "scratch"):
            identity_gate.run(self.campaign, self.campaign, 88830, (1, 291), verify=False, factory=lambda p, c: StubAdapter(),
                              runner=self.copy_runner(), manifests=lambda c: (self.plan, self.feas))

    def test_it_refuses_another_core_another_adapter_and_a_checkpoint_a2_was_not_prepared_with(self):
        prepared = self.campaign / "arms/A2/prepared.json"
        record = json.loads(prepared.read_text())
        record["core"]["sha256"] = "0" * 64
        prepared.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "core"):
            self.run_gate()
        record["core"]["sha256"] = CORE
        record["arm"]["adapter"] = "experiments.vla_inputs_arm_xvla.adapter:XVLAAdapter"
        prepared.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "adapter"):
            self.run_gate()
        record["arm"]["adapter"] = identity_gate.IDENTIFIER
        prepared.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            identity_gate.run(self.campaign, self.out, 101520, (1, 291), verify=False, factory=lambda p, c: StubAdapter(),
                              runner=self.copy_runner(), manifests=lambda c: (self.plan, self.feas))

    def test_verification_against_the_repository_runs_unless_switched_off(self):
        calls = []

        class Evaluate:
            @staticmethod
            def verify_core(prepared, repo):
                calls.append("core")

            @staticmethod
            def verify_manifests(root, prepared):
                calls.append("manifests")

            @staticmethod
            def verify_arm(prepared, repo, deep=False):
                calls.append("arm")

            @staticmethod
            def verify_checkpoint(prepared, checkpoint):
                calls.append("checkpoint")

        identity_gate.run(self.campaign, self.out, 88830, (1, 291), verify=True, factory=lambda p, c: StubAdapter(), runner=self.copy_runner(),
                          manifests=lambda c: (self.plan, self.feas), evaluate_module=Evaluate)
        self.assertEqual(sorted(calls), ["arm", "checkpoint", "core", "manifests"])


if __name__ == "__main__":
    unittest.main()
