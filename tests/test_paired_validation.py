"""Corruption tests for paired-only gates; fixtures are not training data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


def trace_fixture():
    from src.controllers.joint_target_limits import bounded_joint_target
    from src.robots.franka_panda.config import Q_HOME
    n = 4
    bounds = np.array([[-4., 4.]] * 7)
    requested = np.ones((n, 7)) * .25
    previous = np.asarray(Q_HOME[:7]).copy()
    commands, velocities, saturations = [], [], []
    for row in requested:
        previous, velocity, saturation = bounded_joint_target(previous, row, 1 / 30, bounds)
        commands.append(np.r_[previous, 255.])
        velocities.append(velocity)
        saturations.append(saturation)
    qpos = np.zeros((n, 37))
    qpos[:, :9] = Q_HOME
    trace = {"qpos": qpos, "ctrl": np.array(commands), "sim_time": .2 + np.arange(n) / 30,
             "image_time": .2 + np.arange(n) / 30, "substeps": np.full(n, 20),
             "requested_joint_velocity": requested, "applied_joint_velocity": np.array(velocities),
             "command_saturation": np.array(saturations), "released_stack": np.ones(n, bool),
             "body_xpos": np.zeros((n, 20, 3)), "camera_xpos": np.zeros((n, 2, 3)),
             "camera_xmat": np.zeros((n, 2, 9)), "phase_index": np.zeros(n, int),
             "phase_time": np.arange(n) / 30, "target_pos": np.zeros((n, 3)),
             "image_sha256": np.full((n, 2), "a" * 64)}
    return trace, trace["ctrl"].astype(np.float32), qpos[:, :9].astype(np.float32), np.arange(n) / 30, bounds


class PairedValidationTests(unittest.TestCase):
    def test_failed_revalidation_invalidates_previous_pass_without_legacy_reports(self):
        from experiments.paired_demonstrations import validate as validator
        from experiments.paired_demonstrations.plan import make_plan, digest
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "outputs/paired_demonstrations/incomplete"
            root.mkdir(parents=True)
            plan = make_plan()
            (root / "manifest.json").write_text(json.dumps({"status": "recording", "plan": plan,
                                                           "plan_digest": digest(plan)}))
            (root / "attempts.jsonl").write_text("")
            (root / "paired_validation.json").write_text('{"status": "passed", "training_ready": true}')
            with patch.object(validator, "ROOT", workspace), self.assertRaises(ValueError):
                validator.validate(root)
            result = json.loads((root / "paired_validation.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["training_ready"])
            self.assertFalse((root / "dataset/replay/replay_result.json").exists())
            self.assertFalse((root / "validation_result.json").exists())

    def test_command_trace_reconstructs_sent_controls_and_measured_rows(self):
        from experiments.paired_demonstrations.validate import validate_trace
        validate_trace(*trace_fixture())

    def test_corrupted_numeric_control_timing_state_and_padding_are_rejected(self):
        from experiments.paired_demonstrations.validate import validate_trace
        for mutation in ("nan", "bound", "label", "velocity", "saturation", "time", "image_time", "steps", "state", "pad"):
            trace, actions, states, times, bounds = trace_fixture()
            if mutation == "nan": trace["qpos"][0, 0] = np.nan
            if mutation == "bound": actions[0, 0] = 8.
            if mutation == "label": actions[0, 7] = 0.
            if mutation == "velocity": trace["requested_joint_velocity"][0, 1] += .01
            if mutation == "saturation": trace["command_saturation"][0, 0] = True
            if mutation == "time": times[1] += .001
            if mutation == "image_time": trace["image_time"][1] += 1 / 30
            if mutation == "steps": trace["substeps"][1] = 19
            if mutation == "state": states[0, 0] += .01
            if mutation == "pad": actions = np.pad(actions, ((0, 0), (0, 24)))
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, AssertionError)):
                validate_trace(trace, actions, states, times, bounds)

    def test_schedule_requires_all36_exact_episodes_and_one_successful_attempt_each(self):
        from experiments.paired_demonstrations.validate import validate_schedule
        from experiments.paired_demonstrations.plan import make_plan, digest
        plan = make_plan()
        entries = [{**e, "frames": 528, "success": True} for e in plan["episodes"]]
        manifest = {"status": "complete", "plan": plan, "plan_digest": digest(plan), "episodes": entries}
        validate_schedule(manifest, entries)
        for mutation in ("incomplete", "missing", "task", "split", "extra_attempt", "failure", "frame_count"):
            bad, ledger = copy.deepcopy(manifest), copy.deepcopy(entries)
            if mutation == "incomplete": bad["status"] = "recording"
            if mutation == "missing": bad["episodes"].pop()
            if mutation == "task": bad["episodes"][0]["task_index"] = 1
            if mutation == "split": bad["episodes"][0]["split"] = "heldout"
            if mutation == "extra_attempt": ledger.append(ledger[0])
            if mutation == "failure": ledger[1]["success"] = False
            if mutation == "frame_count": bad["episodes"][0]["frames"] = 527
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): validate_schedule(bad, ledger)

    def test_same_reset_rejects_any_full_state_difference_not_only_geometry(self):
        from experiments.paired_demonstrations.validate import same_reset
        frozen = {"initial_state": np.arange(100, dtype=np.float64), "qpos": np.arange(37, dtype=np.float64)}
        trace = {"initial_state": frozen["initial_state"].copy(), "qpos": np.array([frozen["qpos"]])}
        same_reset(frozen, trace)
        trace["initial_state"][45] += 1e-12
        with self.assertRaises(AssertionError): same_reset(frozen, trace)

    def test_old_or_incomplete_reports_never_grant_paired_training_readiness(self):
        from experiments.paired_demonstrations.validate import require_paired_pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "paired_validation.json").write_text(json.dumps({"status": "passed"}))
            with self.assertRaises((ValueError, KeyError, FileNotFoundError)): require_paired_pass(root)

    def test_replay_completeness_requires_every_mode_for_every_episode(self):
        from experiments.paired_demonstrations.validate import check_replay_coverage
        results = [{"episode": i, "mode": mode, "passed": True}
                   for i in range(36) for mode in ("scripted_expert", "reference", "recorded_actions")]
        check_replay_coverage(results)
        for bad in (results[:-1], results + [results[0]], [{**r, "passed": False} for r in results]):
            with self.assertRaises(ValueError): check_replay_coverage(bad)


if __name__ == "__main__": unittest.main()