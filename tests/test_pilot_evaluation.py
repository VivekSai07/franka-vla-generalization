import copy
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data_collection.pilot_plan import make_plan, plan_digest


class PilotEvaluationTests(unittest.TestCase):
    def helpers(self):
        try:
            return importlib.import_module("validation.evaluate_pilot")
        except ModuleNotFoundError as exc:
            if exc.name == "validation.evaluate_pilot":
                self.fail("Standalone pilot evaluator has not been implemented")
            raise

    def bank(self):
        plan = make_plan("pilot", 20260915)
        bank = {"status": "complete", "plan_digest": plan_digest(plan),
                "scenes": [{**slot, "seed": slot["retry_seeds"][0],
                            "file": slot["slot_id"].replace("/", "_") + ".npz",
                            "sha256": f"{i:064x}", "scene_hash": f"{i + 1000:064x}"}
                           for i, slot in enumerate(plan["scene_slots"])]}
        return plan, bank

    def test_01_fixed_120_tuning_selection_excludes_final(self):
        plan, bank = self.bank()
        selected = self.helpers().select_tuning(plan, bank)
        self.assertEqual(len(selected), 120)
        self.assertEqual([s["slot_id"] for s in selected],
                         [f"pilot/tuning/{i:04d}" for i in range(120)])
        self.assertEqual({s["split"] for s in selected}, {"tuning"})
        self.assertEqual(len({s["scene_hash"] for s in selected}), 120)

    def test_bad_counts_duplicates_order_and_plan_rejected(self):
        plan, bank = self.bank()
        mutations = [lambda b: b["scenes"].pop(),
                     lambda b: b["scenes"].append(b["scenes"][0]),
                     lambda b: b["scenes"].__setitem__(1, b["scenes"][0]),
                     lambda b: b["scenes"].reverse(),
                     lambda b: b["scenes"][120].update(scene_hash=b["scenes"][0]["scene_hash"]),
                     lambda b: b["scenes"][1].update(file=b["scenes"][0]["file"]),
                     lambda b: b["scenes"][0].update(seed=1),
                     lambda b: b.update(status="incomplete"),
                     lambda b: b.update(plan_digest="wrong")]
        for mutate in mutations:
            bad = copy.deepcopy(bank)
            mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.helpers().select_tuning(plan, bad)
        with self.assertRaises(ValueError):
            self.helpers().select_tuning(plan, bank, {bank["scenes"][0]["scene_hash"]})
        preparation, _ = self.bank()
        preparation["mode"] = "preparation"
        with self.assertRaises(ValueError):
            self.helpers().select_tuning(preparation, bank)

    def test_success_requires_30_consecutive_strict_released_stack(self):
        helper = self.helpers()
        stable = 0
        for _ in range(29):
            stable, success = helper.advance_streak(stable, {"released_stack": True})
            self.assertFalse(success)
        stable, success = helper.advance_streak(stable, {"released_stack": False})
        self.assertEqual((stable, success), (0, False))
        for _ in range(30):
            stable, success = helper.advance_streak(stable, {"released_stack": True})
        self.assertEqual((stable, success), (30, True))

    def test_aggregates_include_nonfinite_and_timeout_failures(self):
        rows = [{"pick": "red", "place": "green", "colors": ["red", "green"],
                 "task_success": s, "unassisted_success": u, "outcome": outcome}
                for s, u, outcome in [(True, True, "success"), (True, False, "success"),
                                      (False, False, "nonfinite_action"), (False, False, "timeout")]]
        result = self.helpers().aggregate(rows)
        self.assertEqual(result["finished_trials"], 4)
        self.assertEqual(result["planned_trials"], 120)
        for counts in (result["overall"], result["per_pair"]["red->green"], result["per_cube_count"]["2"]):
            self.assertEqual(counts["trials"], 4)
            self.assertEqual(counts["successes"], 2)
            self.assertEqual(counts["success_rate"], .5)
            self.assertEqual(counts["unassisted_success_rate"], .25)
        self.assertEqual(self.helpers().aggregate([])["overall"]["success_rate"], None)

    def implemented(self, name):
        helper = self.helpers()
        self.assertTrue(hasattr(helper, name), f"Missing evaluator behavior: {name}")
        return getattr(helper, name)

    def test_checkpoint_requires_selected_complete_20k_and_model_hash(self):
        verify = self.implemented("verify_checkpoint")
        helper = self.helpers()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoints/020000"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"test weights")
            marker = {"status": "complete", "update": 20000, "reload_fp32_weights_exact": True,
                      "reload_max_action_error": 0., "model_sha256": helper.file_hash(checkpoint / "model.safetensors")}
            report = {"status": "complete", "purpose": "smolvla_balanced_pilot", "training_steps": 20000,
                      "completed_training_steps": 20000, "best_checkpoint_update": 20000,
                      "dataset_root": str(root), "dataset_fingerprint": "data", "timing": {"fps": 30},
                      "training_source_hashes": helper.training_source_hashes(),
                      "final_data_binding_passed": True, "checkpoints": [marker]}
            def save():
                (root / "result.json").write_text(json.dumps(report))
                (checkpoint / "checkpoint_result.json").write_text(json.dumps(marker))
            save()
            verify(checkpoint, root, {"fps": 30}, "data")
            for key, value in [("status", "training"), ("best_checkpoint_update", 18000),
                               ("completed_training_steps", 19999), ("final_data_binding_passed", False),
                               ("dataset_fingerprint", "changed"), ("training_source_hashes", {})]:
                old = report[key]
                report[key] = value
                save()
                with self.subTest(key=key), self.assertRaises(ValueError):
                    verify(checkpoint, root, {"fps": 30}, "data")
                report[key] = old
            save()
            (checkpoint / "model.safetensors").write_bytes(b"changed weights")
            with self.assertRaises(ValueError):
                verify(checkpoint, root, {"fps": 30}, "data")

    def test_scene_restore_checks_hash_full_state_qpos_and_geometry(self):
        restore = self.implemented("restore_scene")
        import mujoco
        from data_collection.benchmark import capture_state
        from data_collection.pilot_integrity import scene_digest
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'
            '<body name="target_cube_red" pos=".3 -.1 .02"><freejoint/><geom size=".02"/></body>'
            '<body name="target_cube_green" pos=".5 .1 .02"><freejoint/><geom size=".02"/></body>'
            '</worldbody></mujoco>')
        data = mujoco.MjData(model)
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory)
            path = bank / "scene.npz"
            state = capture_state(model, data)
            np.savez(path, initial_state=state, qpos=data.qpos.copy())
            entry = {"file": path.name, "sha256": self.helpers().file_hash(path),
                     "scene_hash": scene_digest(data.qpos), "colors": ["red", "green"]}
            restored = restore(model, bank, entry)
            np.testing.assert_array_equal(capture_state(model, restored), state)
            for state_value, qpos in [(state[:-1], data.qpos), (state, data.qpos + .1)]:
                np.savez(path, initial_state=state_value, qpos=qpos)
                entry["sha256"] = self.helpers().file_hash(path)
                with self.assertRaises((ValueError, AssertionError)):
                    restore(model, bank, entry)
            np.savez(path, initial_state=state, qpos=data.qpos)
            entry["sha256"] = "wrong"
            with self.assertRaises(ValueError):
                restore(model, bank, entry)
            entry["file"] = "../scene.npz"
            with self.assertRaises(ValueError):
                restore(model, bank, entry)

    def test_main_fails_closed_and_never_overwrites_output(self):
        main = self.implemented("main")
        helper = self.helpers()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            argv = ["--root", directory, "--checkpoint", directory,
                    "--scene-bank", directory, "--output", str(output)]
            with patch.object(helper, "run", side_effect=ValueError("stale provenance")):
                with self.assertRaises(ValueError):
                    main(argv)
            result = output / "result.json"
            self.assertEqual(json.loads(result.read_text())["status"], "failed")
            previous = result.read_bytes()
            with self.assertRaises(FileExistsError):
                main(argv)
            self.assertEqual(result.read_bytes(), previous)

    def test_trial_nonfinite_is_local_failure_and_restores_chunk_wrapper(self):
        self.exercise_trial("nonfinite_action", 1, 0)

    def test_trial_success_streak_chunk_counts_trace_and_clipping(self):
        self.exercise_trial("success", 30, 30, clipped=True)

    def test_trial_unclipped_success_is_intervention_free(self):
        self.exercise_trial("success", 30, 30)

    def test_nonfinite_after_ten_controls_counts_failed_attempt_chunk(self):
        self.exercise_trial("late_nonfinite", 11, 10)

    def test_trial_timeout_runs_exactly_1200_ticks(self):
        self.exercise_trial("timeout", 1200, 1200)

    def test_trial_infrastructure_and_wrong_chunk_cadence_raise(self):
        with self.assertRaisesRegex(RuntimeError, "infrastructure"):
            self.exercise_trial("infrastructure", 1, 0)
        with self.assertRaisesRegex(ValueError, "chunk"):
            self.exercise_trial("wrong_cadence", 30, 30)

    def exercise_trial(self, outcome, attempts, ticks, clipped=False):
        trial = self.implemented("run_trial")
        helper = self.helpers()
        from contextlib import ExitStack
        from collections import deque
        # No MuJoCo steps, renderer or real policy: test the actual evaluator loop
        # against explicit boundary doubles, with real action bounds and MP4 encoder.
        class Policy:
            config = SimpleNamespace(chunk_size=50, n_action_steps=10)
            def reset(self):
                self.queue = deque()
                self.chunks = 0
            def _get_action_chunk(self):
                self.chunks += 1
                invalid = outcome == "nonfinite_action" or (outcome == "late_nonfinite" and self.chunks == 2)
                value = float("nan") if invalid else (2. if clipped else 0.)
                return torch.full((10, 8), value)
            def select_action(self, frame):
                if outcome == "infrastructure":
                    raise RuntimeError("infrastructure")
                if not self.queue or outcome == "wrong_cadence":
                    self.queue.extend(self._get_action_chunk())
                return self.queue.popleft()[None]
        class Renderer:
            def __init__(self, *args, **kwargs):
                pass
            def update_scene(self, *args, **kwargs):
                pass
            def render(self):
                return np.zeros((512, 512, 3), dtype=np.uint8)
            def close(self):
                pass
        data = SimpleNamespace(qpos=np.zeros(9), qvel=np.zeros(9), time=.2, xpos=np.zeros((1, 3)))
        model = SimpleNamespace(actuator_ctrlrange=np.tile([-1., 1.], (9, 1)))
        entry = {"pick": "red", "place": "green", "colors": ["red", "green"], "slot_id": "pilot/tuning/0000"}
        manifest = {"camera_names": ["front_cam", "gripper_cam"], "control_dt_requested": 1/30}
        policy = Policy()
        original = policy._get_action_chunk
        observations = []
        def pre(frame):
            observations.append(frame)
            return frame
        def step(*args):
            data.time += 1/30
            data.qpos += .001
            return 20
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for name, replacement in {"restore_scene": lambda *a: data,
                    "capture_state": lambda *a: np.r_[data.time, data.qpos],
                    "synchronize_observation": lambda *a: None, "apply_action": lambda *a: None,
                    "step_control": step,
                    "fresh_stack_metrics": lambda *a: {"released_stack": outcome in ("success", "wrong_cadence"),
                                                         "xy_m": .01, "z_m": .04, "finger_contact": False},
                    "target_diagnostics": lambda *a: {"pick_lift_m": .04, "finger_midpoint_to_pick_m": .1,
                                                      "bilateral_target_contact": False,
                                                      "pick_position": np.zeros(3), "place_position": np.zeros(3)}}.items():
                stack.enter_context(patch.object(helper, name, replacement))
            stack.enter_context(patch.object(helper.mujoco, "Renderer", Renderer))
            stack.enter_context(patch.object(helper.mujoco, "MjData", return_value=data))
            stack.enter_context(patch.object(helper, "GraspController", return_value=SimpleNamespace(target_id=0)))
            result = trial(policy, pre, lambda x: x, model, manifest, entry, Path(directory), Path(directory), 0)
            self.assertEqual(policy._get_action_chunk, original)
            self.assertEqual(result["outcome"], "nonfinite_action" if outcome == "late_nonfinite" else outcome)
            self.assertEqual(result["ticks"], attempts)
            self.assertEqual(result["completed_control_ticks"], ticks)
            self.assertEqual(result["action_attempts"], attempts)
            self.assertEqual(result["actual_chunk_predictions"], (attempts + 9)//10)
            self.assertEqual(result["task_success"], outcome == "success")
            self.assertEqual(result["unassisted_success"], outcome == "success" and not clipped)
            self.assertEqual(result["joint_clipped_ticks"], ticks if clipped else 0)
            self.assertEqual(result["instruction"], "pick up the red cube and place it on top of the green cube")
            self.assertEqual(set(observations[0]), {"observation.state", "observation.images.image",
                                                  "observation.images.image2", "task"})
            with np.load(Path(directory) / "trace.npz", allow_pickle=False) as trace:
                self.assertEqual(trace["raw_action"].shape, (attempts, 8))
                self.assertEqual(trace["applied_action"].shape, (ticks, 8))
                self.assertEqual(len(trace["post_state"]), ticks)
                self.assertAlmostEqual(trace["terminal_sim_time"].item() - .2, ticks/30)
                np.testing.assert_array_equal(trace["terminal_state"], data.qpos)
                self.assertEqual(trace["terminal_released_stack"].item(), outcome == "success")
            self.assertGreater((Path(directory) / "cameras.mp4").stat().st_size, 0)

    def test_run_finishes_all_120_including_nonfinite_and_binds_post_run(self):
        self.exercise_run(120, changed=False)

    def test_run_never_completes_for_short_bank_or_changed_post_run(self):
        self.exercise_run(119, changed=False)
        self.exercise_run(120, changed=True)

    def exercise_run(self, count, changed):
        helper = self.helpers()
        from contextlib import ExitStack
        selected = [{"slot_id": f"pilot/tuning/{i:04d}", "pick": "red", "place": "green",
                     "colors": ["red", "green"]} for i in range(count)]
        policy = SimpleNamespace(config=SimpleNamespace(), eval=lambda: None, parameters=lambda: [])
        snapshots = [({}, selected, None, {"data": "before"}),
                     ({}, selected, None, {"data": "after" if changed else "before"})]
        snapshots = [({"timing": {}}, s, m, f) for _, s, m, f in snapshots]
        def trial(*args):
            entry, ordinal = args[5], args[8]
            return {**entry, "ordinal": ordinal, "task_success": ordinal > 0,
                    "unassisted_success": ordinal > 0, "outcome": "success" if ordinal else "nonfinite_action"}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "evaluation"
            stack.enter_context(patch.object(helper, "preflight", side_effect=snapshots))
            stack.enter_context(patch.object(helper, "load_fp32_checkpoint", return_value=policy))
            stack.enter_context(patch.object(helper, "run_trial", side_effect=trial))
            stack.enter_context(patch("lerobot.policies.factory.make_pre_post_processors", return_value=(None, None)))
            argv = ["--root", directory, "--checkpoint", directory, "--scene-bank", directory, "--output", str(output)]
            if count != 120 or changed:
                with self.assertRaises(ValueError):
                    helper.main(argv)
            else:
                helper.main(argv)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["status"], "failed" if count != 120 or changed else "complete")
            self.assertEqual(result["finished_trials"], count)
            self.assertEqual(result["overall"]["trials"], count)
            self.assertEqual(result["overall"]["successes"], count - 1)
            self.assertEqual([r["ordinal"] for r in result["episodes"]], list(range(count)))


if __name__ == "__main__":
    unittest.main()