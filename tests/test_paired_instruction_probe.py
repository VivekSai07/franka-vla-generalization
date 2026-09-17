import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from data_collection.benchmark import capture_state, configure_timing
from data_collection.pilot_integrity import scene_digest
from data_collection.pilot_plan import make_plan
from training.smolvla_pilot import file_hash


class PairedInstructionProbeTests(unittest.TestCase):
    def helpers(self):
        try:
            return importlib.import_module("validation.paired_instruction_probe")
        except ModuleNotFoundError as exc:
            if exc.name == "validation.paired_instruction_probe":
                self.fail("Paired instruction probe has not been implemented")
            raise

    def test_standalone_cli_help_works_outside_repository_without_policy_load(self):
        script = Path(__file__).resolve().parents[1] / "validation/paired_instruction_probe.py"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(script), "--help"], cwd=directory,
                                    capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--baseline", result.stdout)
        self.assertIn("--output", result.stdout)

    def fixture(self):
        plan = make_plan("pilot", 20260915)
        def artifact(slot, index):
            return {**copy.deepcopy(slot), "file": f"scene_{index}.npz",
                    "sha256": f"{index:064x}", "scene_hash": f"{index + 1000:064x}",
                    "seed": slot["retry_seeds"][0]}
        tuning = [artifact(s, i) for i, s in enumerate(plan["scene_slots"][:120])]
        episodes = [{**artifact(s, i), "episode_index": i, "task_success": True}
                    for i, s in enumerate(plan["slots"])]
        manifest = {"episodes": episodes,
                    "splits": {"train": list(range(300)), "validation": list(range(300, 420))}}
        return manifest, tuning

    def test_selection_is_exact_36_plus_12_with_unchanged_roles_seeds_identity(self):
        manifest, tuning = self.fixture()
        before = copy.deepcopy((manifest, tuning))
        trials = self.helpers().select_trials(manifest, tuning)
        self.assertEqual(len(trials), 48)
        self.assertEqual([t["trial_id"] for t in trials], list(range(48)))
        for pair_index in range(12):
            original, reversed_, seedcontrol = trials[pair_index * 3:pair_index * 3 + 3]
            ordinal = pair_index * 10
            self.assertEqual([t["condition"] for t in (original, reversed_, seedcontrol)],
                             ["original", "reversed", "seedcontrol"])
            self.assertEqual([t["baseline_ordinal"] for t in (original, reversed_, seedcontrol)],
                             [ordinal] * 3)
            self.assertEqual([t["ordinal"] for t in (original, reversed_, seedcontrol)],
                             [ordinal, ordinal, ordinal + 1])
            self.assertEqual([t["rollout_seed"] for t in (original, reversed_, seedcontrol)],
                             [20260916 + ordinal, 20260916 + ordinal, 20260917 + ordinal])
            self.assertEqual(original["entry"], tuning[ordinal])
            self.assertEqual(seedcontrol["entry"], tuning[ordinal])
            want = copy.deepcopy(tuning[ordinal])
            want["pick"], want["place"] = want["place"], want["pick"]
            self.assertEqual(reversed_["entry"], want)
        trains = trials[36:]
        self.assertEqual([t["source_episode_index"] for t in trains], list(range(0, 300, 25)))
        self.assertEqual({t["condition"] for t in trains}, {"train"})
        for t in trains:
            self.assertIsNone(t["baseline_ordinal"])
            self.assertEqual(t["entry"]["split"], "train")
            self.assertEqual(len(t["entry"]["colors"]), 2)
            self.assertEqual(t["rollout_seed"], 20260916 + t["source_episode_index"])
        self.assertEqual((manifest, tuning), before)
        trials[0]["entry"]["colors"].reverse()
        self.assertEqual((manifest, tuning), before)
        self.assertEqual(trials[1]["entry"]["colors"], ["red", "green"])

    def test_selection_ignores_outcomes_and_train_input_order(self):
        manifest, tuning = self.fixture()
        h = self.helpers()
        expected = h.select_trials(manifest, tuning)
        # Demonstration outcome metadata must not enter selected trial entries.
        for e in manifest["episodes"]:
            e["task_success"] = False
            e["stable_ticks"] = -100
        manifest["episodes"].reverse()
        self.assertEqual(h.select_trials(manifest, tuning), expected)

    def test_missing_two_cube_pair_never_falls_back_to_validation_or_final(self):
        h = self.helpers()
        for kind in ("train_pair", "tuning_pair", "overlap", "wrong_split", "duplicate", "final", "short"):
            manifest, tuning = self.fixture()
            if kind == "train_pair":
                for e in manifest["episodes"][:25]:
                    if len(e["colors"]) == 2:
                        e["colors"].append("blue")
            elif kind == "tuning_pair":
                for e in tuning[:10]:
                    if len(e["colors"]) == 2:
                        e["colors"].append("blue")
            elif kind == "overlap":
                manifest["splits"]["train"].append(300)
            elif kind == "wrong_split":
                manifest["episodes"][0]["split"] = "validation"
            elif kind == "duplicate":
                manifest["episodes"][1]["episode_index"] = 0
            elif kind == "final":
                tuning[0]["split"] = "final"
            else:
                tuning.pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                h.select_trials(manifest, tuning)

    def trace_fixture(self):
        return {"raw_action": np.zeros((3, 8), np.float32),
                "applied_action": np.zeros((3, 8)), "state": np.zeros((3, 9)),
                "post_state": np.zeros((3, 9)), "sim_time": np.arange(3.) / 30,
                "post_sim_time": np.arange(1., 4.) / 30,
                "initial_integration_state": np.zeros(12), "terminal_integration_state": np.ones(12),
                "terminal_qpos": np.ones(9), "terminal_qvel": np.zeros(9),
                "terminal_state": np.ones(9), "terminal_sim_time": np.array(.1),
                "released_stack": np.zeros(3, dtype=bool)}

    def test_original_comparison_checks_every_array_and_stops_on_mismatch(self):
        h = self.helpers()
        base = self.trace_fixture()
        self.assertTrue(h.compare_traces(base, copy.deepcopy(base))["exact"])
        for key in base:
            bad = copy.deepcopy(base)
            if bad[key].dtype == bool:
                bad[key].flat[-1] = True
            else:
                bad[key].flat[-1] += 2e-6
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                h.compare_traces(base, bad)
        for mutation in (lambda b: b.pop("state"),
                         lambda b: b.update(state=b["state"][:-1]),
                         lambda b: b.update(state=b["state"].astype(np.float32))):
            bad = copy.deepcopy(base)
            mutation(bad)
            with self.assertRaises(ValueError):
                h.compare_traces(base, bad)

    def test_original_comparison_absolute_tolerances_and_nonfinite_pattern(self):
        h = self.helpers()
        base = self.trace_fixture()
        within = copy.deepcopy(base)
        within["raw_action"][0, 0] = 5e-8
        within["post_state"][0, 0] = 5e-9
        self.assertFalse(h.compare_traces(base, within)["exact"])
        within["post_state"][0, 0] = 2e-8
        with self.assertRaisesRegex(ValueError, "post_state"):
            h.compare_traces(base, within)
        base["raw_action"][-1, 0] = np.nan
        self.assertTrue(h.compare_traces(base, copy.deepcopy(base))["exact"])
        bad = copy.deepcopy(base)
        bad["raw_action"][-1, 0] = np.inf
        with self.assertRaises(ValueError):
            h.compare_traces(base, bad)

    def test_summaries_keep_null_failures_in_paired_denominators(self):
        trials = self.helpers().select_trials(*self.fixture())
        rows = []
        for t in trials:
            # Only one pair switches. Ten pairs are both-null, one one-null.
            choice = None
            if t["baseline_ordinal"] == 0:
                choice = "green" if t["condition"] == "reversed" else "red"
            elif t["baseline_ordinal"] == 10 and t["condition"] == "reversed":
                choice = "blue"
            rows.append({**t, "task_success": t["trial_id"] == 0,
                         "unassisted_success": False, "outcome": "timeout" if choice else "nonfinite_action",
                         "diagnostics": {"first_bilateral_choice": choice,
                             "first_sustained_choice": None, "first_lifted_choice": None,
                             "wrong_pairs_stable_30_ticks": []}})
        result = self.helpers().summarize(trials, rows)
        self.assertEqual(result["finished_trials"], 48)
        for condition in ("original", "reversed", "seedcontrol", "train"):
            self.assertEqual(result["conditions"][condition]["trials"], 12)
        self.assertEqual(result["conditions"]["original"]["success_rate"], 1 / 12)
        switch = result["paired"]["reversed"]["first_bilateral_choice"]
        self.assertEqual(switch["pairs"], 12)
        self.assertEqual(switch["switches"], 1)
        self.assertEqual(switch["switch_rate"], 1 / 12)
        self.assertEqual(switch["both_null"], 10)
        self.assertEqual(switch["one_null"], 1)
        self.assertEqual(result["paired"]["seedcontrol"]["first_bilateral_choice"]["switches"], 0)
        partial = self.helpers().summarize(trials, rows[:1])
        self.assertEqual(partial["paired"]["reversed"]["first_bilateral_choice"]["missing_pairs"], 12)
        self.assertEqual(partial["paired"]["reversed"]["first_bilateral_choice"]["switch_rate"], 0)

    def small_model(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'
            '<body name="target_cube_red" pos=".3 -.1 .02"><freejoint/><geom size=".02"/></body>'
            '<body name="target_cube_green" pos=".5 .1 .02"><freejoint/><geom size=".02"/></body>'
            '</worldbody></mujoco>')
        return model, mujoco.MjData(model)

    def test_train_derivation_handles_initial_qpos_trajectory_or_integration_only(self):
        h = self.helpers()
        model, data = self.small_model()
        for schema in ("initial_qpos", "trajectory", "integration_only"):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "replay"
                source.mkdir()
                output = root / "new"
                output.mkdir()
                path = source / "episode.npz"
                arrays = {"initial_state": capture_state(model, data)}
                if schema == "initial_qpos":
                    arrays["initial_qpos"] = data.qpos.copy()
                elif schema == "trajectory":
                    arrays["qpos"] = np.stack([data.qpos, data.qpos + .1])
                np.savez(path, **arrays)
                entry = {"episode_index": 0, "split": "train", "file": path.name,
                         "sha256": file_hash(path), "scene_hash": scene_digest(data.qpos),
                         "pick": "red", "place": "green", "colors": ["red", "green"]}
                original = copy.deepcopy(entry)
                derived, provenance = h.derive_train_state(model, source, entry, output)
                self.assertEqual(entry, original)
                self.assertEqual(file_hash(path), original["sha256"])
                self.assertEqual(provenance["source_path"], str(path))
                self.assertEqual(provenance["source_sha256"], original["sha256"])
                self.assertEqual(provenance["derived_sha256"], file_hash(output / derived["file"]))
                with np.load(output / derived["file"], allow_pickle=False) as state:
                    self.assertEqual(set(state.files), {"initial_state", "qpos"})
                    np.testing.assert_array_equal(state["qpos"], data.qpos)
                    np.testing.assert_array_equal(state["initial_state"], arrays["initial_state"])
                with self.assertRaises(FileExistsError):
                    h.derive_train_state(model, source, entry, output)

    def test_train_derivation_rejects_hash_qpos_scene_and_source_escape(self):
        h = self.helpers()
        model, data = self.small_model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out"
            output.mkdir()
            path = root / "episode.npz"
            for kind in ("hash", "qpos", "scene", "path", "split", "state"):
                arrays = {"initial_state": capture_state(model, data), "initial_qpos": data.qpos.copy()}
                if kind == "qpos":
                    arrays["initial_qpos"][0] += .01
                if kind == "state":
                    arrays["initial_state"] = arrays["initial_state"][:-1]
                np.savez(path, **arrays)
                entry = {"episode_index": 0, "split": "train", "file": path.name,
                         "sha256": file_hash(path), "scene_hash": scene_digest(data.qpos),
                         "pick": "red", "place": "green", "colors": ["red", "green"]}
                if kind == "hash":
                    entry["sha256"] = "bad"
                if kind == "scene":
                    entry["scene_hash"] = "bad"
                if kind == "path":
                    entry["file"] = "../episode.npz"
                if kind == "split":
                    entry["split"] = "validation"
                with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                    h.derive_train_state(model, root, entry, output)
                self.assertEqual(list(output.iterdir()), [])

    def test_contact_diagnostics_require_consecutive_ticks_and_report_ties(self):
        h = self.helpers()
        contacts = np.zeros((35, 2, 2), dtype=bool)
        contacts[0, 0, 0] = True  # Unilateral is not a first-object choice.
        contacts[1:10, 0, :] = True  # Nine ticks must not qualify as sustained.
        contacts[11:21, 1, :] = True
        lift = np.zeros((35, 2))
        lift[11:21, 1] = .06
        released = np.zeros((35, 2), dtype=bool)
        released[:30, 1] = True  # green->red, not instructed red->green
        result = h.contact_summary(["red", "green"], contacts, lift, released, "red", "green")
        self.assertEqual(result["first_bilateral_choice"], "red")
        self.assertEqual(result["first_sustained_choice"], "green")
        self.assertEqual(result["first_lifted_choice"], "green")
        self.assertIsNone(result["per_cube"]["red"]["first_sustained_tick"])
        self.assertEqual(result["per_cube"]["green"]["first_sustained_tick"], 20)
        self.assertEqual(result["per_cube"]["red"]["first_contact_tick"], 0)
        self.assertEqual(result["wrong_pairs_stable_30_ticks"], ["green->red"])
        self.assertEqual(result["per_pair"]["green->red"]["first_stable_tick"], 29)
        contacts[1, 1, :] = True
        tied = h.contact_summary(["red", "green"], contacts, lift, released, "red", "green")
        self.assertIsNone(tied["first_bilateral_choice"])
        self.assertEqual(tied["first_bilateral_colors"], ["red", "green"])

    def implemented(self, name):
        h = self.helpers()
        self.assertTrue(hasattr(h, name), f"Missing diagnostic behavior: {name}")
        return getattr(h, name)

    def saved_trace(self, directory, nonfinite=False):
        from data_collection.benchmark import apply_action, step_control, synchronize_observation
        from src.robots.franka_panda.config import SCENE_PATH, Q_HOME, ACTIVE_JOINTS, ARM_DOF
        from validation.gates import fresh_stack_metrics
        from validation.rollout_diagnostics import bound_action
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        timing = configure_timing(model, 30)
        data, scratch = mujoco.MjData(model), mujoco.MjData(model)
        data.qpos[:9] = Q_HOME
        synchronize_observation(model, data)
        initial = capture_state(model, data)
        trace = {k: [] for k in ("state", "sim_time", "raw_action", "applied_action", "post_state",
                                  "post_sim_time", "released_stack")}
        for _ in range(3):
            raw = np.array(Q_HOME[:7] + [255.], dtype=np.float32)
            action, _ = bound_action(raw, model.actuator_ctrlrange[ACTIVE_JOINTS])
            trace["state"].append(data.qpos[:9].copy())
            trace["sim_time"].append(float(data.time))
            trace["raw_action"].append(raw)
            apply_action(model, data, action, ACTIVE_JOINTS)
            step_control(model, data, 1 / 30, ARM_DOF)
            trace["applied_action"].append(action)
            trace["post_state"].append(data.qpos[:9].copy())
            trace["post_sim_time"].append(float(data.time))
            trace["released_stack"].append(fresh_stack_metrics(model, data, scratch, "red", "green")["released_stack"])
        if nonfinite:
            trace["state"].append(data.qpos[:9].copy())
            trace["sim_time"].append(float(data.time))
            trace["raw_action"].append(np.full(8, np.nan, dtype=np.float32))
        arrays = {k: np.asarray(v) for k, v in trace.items()}
        arrays.update(initial_integration_state=initial, terminal_integration_state=capture_state(model, data),
                      terminal_qpos=data.qpos.copy(), terminal_qvel=data.qvel.copy(),
                      terminal_state=data.qpos[:9].copy(), terminal_sim_time=np.array(data.time))
        path = directory / "trace.npz"
        np.savez(path, **arrays)
        episode = {"pick": "red", "place": "green", "colors": ["red", "green"],
                   "completed_control_ticks": 3, "action_attempts": 4 if nonfinite else 3,
                   "actual_chunk_predictions": 1, "task_success": False,
                   "outcome": "nonfinite_action" if nonfinite else "timeout"}
        return model, {"timing": timing, "control_dt_requested": 1 / 30}, episode, arrays

    def test_cpu_saved_action_replay_matches_states_and_handles_nonfinite_attempt(self):
        replay = self.implemented("replay_trial")
        for nonfinite in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                model, manifest, episode, _ = self.saved_trace(folder, nonfinite)
                diagnostic = replay(model, manifest, episode, folder)
                self.assertTrue(diagnostic["replay_exact"])
                self.assertEqual(diagnostic["completed_control_ticks"], 3)
                self.assertEqual(set(diagnostic["per_pair"]), {"red->green", "green->red"})
                with np.load(folder / "contacts.npz", allow_pickle=False) as saved:
                    self.assertEqual(saved["finger_contacts"].shape, (3, 2, 2))
                    self.assertEqual(saved["lift_m"].shape, (3, 2))
                    self.assertEqual(saved["released_stack"].shape, (3, 2))
                with self.assertRaises(FileExistsError):
                    replay(model, manifest, episode, folder)

    def test_cpu_replay_rejects_action_state_time_terminal_and_success_corruption(self):
        replay = self.implemented("replay_trial")
        for key in ("applied_action", "state", "post_state", "sim_time", "post_sim_time",
                    "terminal_integration_state", "terminal_qpos", "terminal_qvel", "terminal_sim_time", "success"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                model, manifest, episode, arrays = self.saved_trace(folder)
                if key == "success":
                    episode["task_success"] = True
                else:
                    arrays[key].flat[-1] += .01
                    np.savez(folder / "trace.npz", **arrays)
                with self.assertRaises(ValueError):
                    replay(model, manifest, episode, folder)
                self.assertFalse((folder / "diagnostics.json").exists())

    def baseline_fixture(self, directory):
        from validation.evaluate_pilot import aggregate
        manifest, tuning = self.fixture()
        root = directory / "baseline"
        root.mkdir()
        rows = []
        for ordinal, entry in enumerate(tuning):
            folder = root / f"trial_{ordinal:03d}"
            folder.mkdir()
            row = {**copy.deepcopy(entry), "ordinal": ordinal, "rollout_seed": 20260916 + ordinal,
                   "instruction": f"pick up the {entry['pick']} cube and place it on top of the {entry['place']} cube",
                   "task_success": False, "unassisted_success": False, "outcome": "timeout",
                   "completed_control_ticks": 1200, "action_attempts": 1200, "ticks": 1200,
                   "sim_seconds": 40., "actual_chunk_predictions": 120,
                   "chunk_size": 50, "n_action_steps": 10, "success_streak": 0}
            (folder / "episode.json").write_text(json.dumps(row))
            (folder / "cameras.mp4").write_bytes(b"fake video; no GPU policy")
            np.savez(folder / "trace.npz", **self.trace_fixture())
            rows.append(row)
        report = {"status": "complete", "purpose": "pilot_tuning_only", "episodes": rows,
                  "post_run_fingerprints_unchanged": True, "fingerprints": {"data": "bound"},
                  "dataset_root": str(directory / "data"), "checkpoint": str(directory / "training/checkpoints/020000"),
                  "scene_bank": str(directory / "bank"), "max_control_ticks": 1200, "max_sim_seconds": 40,
                  "seed": 20260916, "training_updates": 0, "retries": False, "resume": False,
                  "final_scenes_evaluated": False, **aggregate(rows)}
        (root / "result.json").write_text(json.dumps(report))
        return root, report, manifest, tuning

    def test_baseline_gate_requires_all_120_bound_episodes_and_artifacts(self):
        verify = self.implemented("verify_baseline")
        with tempfile.TemporaryDirectory() as directory:
            root, report, _, tuning = self.baseline_fixture(Path(directory))
            verify(root, report, tuning, {"data": "bound"})
            for key, bad in (("status", "running"), ("finished_trials", 119),
                             ("post_run_fingerprints_unchanged", False), ("fingerprints", {}),
                             ("seed", 7), ("max_control_ticks", 600), ("final_scenes_evaluated", True)):
                changed = copy.deepcopy(report)
                changed[key] = bad
                with self.subTest(key=key), self.assertRaises(ValueError):
                    verify(root, changed, tuning, {"data": "bound"})
            changed = copy.deepcopy(report)
            changed["episodes"][0]["rollout_seed"] += 1
            with self.assertRaises(ValueError):
                verify(root, changed, tuning, {"data": "bound"})
            (root / "trial_119/cameras.mp4").unlink()
            with self.assertRaises(ValueError):
                verify(root, report, tuning, {"data": "bound"})

    def test_file_inventory_detects_baseline_mutation_addition_and_removal(self):
        inventory = self.implemented("file_inventory")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "trace.npz"
            path.write_bytes(b"original")
            before = inventory(root)
            self.assertEqual(before, {"trace.npz": file_hash(path)})
            path.write_bytes(b"mutation")
            self.assertNotEqual(inventory(root), before)
            path.write_bytes(b"original")
            (root / "new").write_bytes(b"extra")
            self.assertNotEqual(inventory(root), before)
            path.unlink()
            self.assertNotEqual(inventory(root), before)

    def test_run_complete_requires_48_replays_12_originals_and_unchanged_postflight(self):
        self.exercise_run("complete")

    def test_run_fails_on_original_mismatch_without_retry_or_later_trial(self):
        self.exercise_run("mismatch")

    def test_run_fails_on_postflight_or_baseline_changes(self):
        self.exercise_run("postflight")
        self.exercise_run("baseline_change")

    def test_run_rejects_stale_preflight_before_loading_policy(self):
        self.exercise_run("stale")

    def test_run_fails_on_plan_or_derived_state_changes(self):
        self.exercise_run("plan_change")
        self.exercise_run("initial_change")

    def test_run_infrastructure_exception_is_failed_without_retry(self):
        self.exercise_run("infrastructure")

    def exercise_run(self, mode):
        from contextlib import ExitStack
        from types import SimpleNamespace
        main = self.implemented("main")
        h = self.helpers()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root, report, manifest, tuning = self.baseline_fixture(Path(directory))
            manifest.update(timing={}, control_dt_requested=1 / 30)
            output = Path(directory) / "new"
            calls = []
            parameter = torch.nn.Parameter(torch.ones(1))
            policy = SimpleNamespace(config=SimpleNamespace(), evaluating=False,
                                     parameters=lambda: [parameter])
            policy.eval = lambda: setattr(policy, "evaluating", True)
            def load(path):
                self.assertNotEqual(mode, "stale", "A stale preflight must not load a policy")
                self.assertTrue((output / "plan.json").is_file())
                self.assertEqual(len(json.loads((output / "plan.json").read_text())["trials"]), 48)
                return policy
            def derive(model, source, entry, target):
                path = target / f"train_{entry['episode_index']:06d}.npz"
                np.savez(path, state=np.zeros(1))
                return {**entry, "file": path.name, "sha256": file_hash(path)}, {
                    "source_path": str(source / entry["file"]), "source_sha256": entry["sha256"],
                    "derived_path": str(path), "derived_sha256": file_hash(path)}
            def trial(policy, pre, post, model, manifest, entry, bank, folder, ordinal):
                calls.append((entry, ordinal))
                self.assertFalse(parameter.requires_grad)
                self.assertTrue(policy.evaluating)
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertFalse(torch.backends.cudnn.benchmark)
                self.assertEqual(torch.get_num_threads(), 8)
                if mode == "infrastructure":
                    raise ValueError("Policy infrastructure failure")
                i = len(calls) - 1
                if i < 36:
                    want_ordinal = (i // 3) * 10 + int(i % 3 == 2)
                    self.assertEqual(ordinal, want_ordinal)
                    self.assertEqual(entry["colors"], tuning[(i // 3) * 10]["colors"])
                    self.assertEqual(entry["pick"], tuning[(i // 3) * 10]["place" if i % 3 == 1 else "pick"])
                else:
                    self.assertEqual(entry["split"], "train")
                    self.assertEqual(ordinal, (i - 36) * 25)
                row = {**report["episodes"][ordinal if i < 36 else 0], **entry, "ordinal": ordinal,
                       "rollout_seed": 20260916 + ordinal}
                row["instruction"] = f"pick up the {entry['pick']} cube and place it on top of the {entry['place']} cube"
                arrays = self.trace_fixture()
                if mode == "mismatch":
                    arrays["raw_action"][-1, 0] += .01
                np.savez(folder / "trace.npz", **arrays)
                (folder / "episode.json").write_text(json.dumps(row))
                (folder / "cameras.mp4").write_bytes(b"fake")
                if mode == "baseline_change" and i == 47:
                    (root / "trial_119/cameras.mp4").write_bytes(b"changed")
                if mode == "plan_change" and i == 47:
                    (output / "plan.json").write_text("{}")
                if mode == "initial_change" and i == 47:
                    (output / "initial_states/new.npz").write_bytes(b"extra")
                return row
            def replay(model, manifest, row, folder):
                result = {**dict.fromkeys(("first_bilateral_choice", "first_sustained_choice", "first_lifted_choice")),
                          "wrong_pairs_stable_30_ticks": [], "replay_exact": True}
                (folder / "diagnostics.json").write_text(json.dumps(result))
                return result
            stack.enter_context(patch.object(h.baseline, "preflight", side_effect=[
                (manifest, tuning, None, {"data": "stale" if mode == "stale" else "bound"}),
                (manifest, tuning, None, {"data": "changed" if mode == "postflight" else "bound"})]))
            stack.enter_context(patch.object(h.baseline, "load_fp32_checkpoint", side_effect=load))
            stack.enter_context(patch.object(h.baseline, "run_trial", side_effect=trial))
            stack.enter_context(patch.object(h, "derive_train_state", side_effect=derive))
            stack.enter_context(patch.object(h, "replay_trial", side_effect=replay))
            stack.enter_context(patch("lerobot.policies.factory.make_pre_post_processors", return_value=(None, None)))
            argv = ["--baseline", str(root), "--output", str(output)]
            if mode == "complete":
                main(argv)
            else:
                with self.assertRaises(ValueError):
                    main(argv)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["status"], "complete" if mode == "complete" else "failed")
            self.assertEqual(len(calls), 0 if mode == "stale" else
                             1 if mode in ("mismatch", "infrastructure") else 48)
            if mode == "complete":
                self.assertEqual(result["finished_trials"], 48)
                self.assertEqual(result["original_comparisons"], 12)
                self.assertEqual(result["replayed_trials"], 48)
                self.assertTrue(result["post_run_fingerprints_unchanged"])
            saved = (output / "result.json").read_bytes()
            with self.assertRaises(FileExistsError):
                main(argv)
            self.assertEqual((output / "result.json").read_bytes(), saved)

    def test_main_refuses_output_inside_baseline(self):
        main = self.implemented("main")
        with tempfile.TemporaryDirectory() as directory:
            root, _, _, _ = self.baseline_fixture(Path(directory))
            with self.assertRaises(ValueError):
                main(["--baseline", str(root), "--output", str(root / "new")])
            self.assertFalse((root / "new").exists())


if __name__ == "__main__":
    unittest.main()