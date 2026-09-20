"""Lifecycle tests: fake expert/media only; no simulator steps or renderer."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.fixed_layout.contract import make_plan, MIN_FREE_BYTES, MAX_SECONDS


class RecordTests(unittest.TestCase):
    def setUp(self):
        try:
            self.r = importlib.import_module("experiments.fixed_layout.record")
        except ModuleNotFoundError as exc:
            self.fail(f"Task 2 recorder missing: {exc}")

    def test_schedule_freezes_24_seeds_copies_reset_and_gates_first_three(self):
        plan, initial, events, calls = make_plan(), np.arange(242, dtype=np.float64), [], []
        def execute(ep, reset):
            self.assertEqual(len(plan["episodes"]), 24)
            np.testing.assert_array_equal(reset, np.arange(242))
            if ep["episode_index"] == 3:
                self.assertEqual([e["episode_index"] for e in events if e["event"] == "admitted"], [0, 1, 2])
                self.assertEqual(sum(e["event"] == "preparation_pass" for e in events), 1)
            calls.append(copy.deepcopy(ep))
            reset[:] = -1
            ep["seed"] = -1  # Callee may not mutate the frozen schedule.
            return {"success": True, "frames": 528}
        result = self.r.run_schedule(plan, initial, execute, lambda ep, out: None, events.append)
        self.assertEqual([e["seed"] for e in calls], [e["seed"] for e in make_plan()["episodes"]])
        self.assertEqual(len(result), 24)
        self.assertEqual(plan, make_plan())

    def test_failed_expert_stops_before_third_slot(self):
        calls, events = [], []
        def execute(ep, initial):
            calls.append(ep["episode_index"])
            return {"success": len(calls) == 1, "frames": 528}
        with self.assertRaises(RuntimeError):
            self.r.run_schedule(make_plan(), np.zeros(242), execute, lambda *a: None, events.append)
        self.assertEqual(calls, [0, 1])
        self.assertFalse(any(e.get("training_ready") for e in events))
        self.assertEqual(events[-1]["event"], "failed")

    def test_validation_exception_stops_before_next_expert(self):
        calls, events = [], []
        def execute(ep, initial):
            calls.append(ep["episode_index"])
            return {"success": True, "frames": 528}
        def reject(*args):
            raise ValueError("media or contact gate")
        with self.assertRaisesRegex(ValueError, "media or contact"):
            self.r.run_schedule(make_plan(), np.zeros(242), execute, reject, events.append)
        self.assertEqual(calls, [0])
        self.assertEqual(events[-1]["event"], "failed")

    def test_plan_and_initial_mutation_block_next_execution(self):
        for kind in ("plan", "initial"):
            plan, initial, calls = make_plan(), np.zeros(242), []
            def execute(ep, state):
                calls.append(ep["episode_index"])
                if kind == "plan":
                    plan["episodes"][1]["seed"] += 1
                else:
                    initial[0] = 1
                return {"success": True, "frames": 528}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.r.run_schedule(plan, initial, execute, lambda *a: None, lambda e: None)
            self.assertEqual(calls, [0])

    def test_disk_gate_creates_no_output_or_expert_and_no_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "absent"
            with patch.object(self.r, "OUTPUT", output), patch.object(self.r, "disk_usage") as usage, \
                    patch.object(self.r, "Expert", side_effect=AssertionError("must not construct")):
                usage.return_value.free = MIN_FREE_BYTES - 1
                with self.assertRaisesRegex(ValueError, "30 GiB"):
                    self.r.collect()
                self.assertFalse(output.exists())
                with self.assertRaises(TypeError):
                    self.r.collect(reserve_bytes=0)

    def test_preexisting_root_rejected_before_expert(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.r, "OUTPUT", Path(tmp)), \
                patch.object(self.r, "Expert", side_effect=AssertionError("must not construct")):
            with self.assertRaises(FileExistsError):
                self.r.collect()

    def test_budget_enforces_duration_and_reserve_at_each_check(self):
        with patch.object(self.r, "monotonic", return_value=10) as clock, \
                patch.object(self.r, "disk_usage") as usage:
            usage.return_value.free = MIN_FREE_BYTES
            budget = self.r.Budget(Path("/tmp"))
            budget.check()
            clock.return_value = 10 + MAX_SECONDS
            with self.assertRaises(TimeoutError):
                budget.check()
            clock.return_value = 10
            usage.return_value.free -= 1
            with self.assertRaises(ValueError):
                budget.check()

    def sample(self):
        frame = {"observation.state": np.zeros(9, np.float32), "action": np.zeros(8, np.float32),
                 "task": make_plan()["episodes"][0]["instruction"],
                 "observation.images.image": np.zeros((512, 512, 3), np.uint8),
                 "observation.images.image2": np.full((512, 512, 3), 7, np.uint8)}
        trace = {"qpos": np.arange(37, dtype=np.float64), "ctrl": np.zeros(8), "substeps": 20,
                 "camera_xmat": np.ones((3, 9)), "sim_time": .2}
        return frame, trace

    def test_sink_persists_original_arrays_before_media_exception(self):
        frame, trace = self.sample()
        with tempfile.TemporaryDirectory() as tmp:
            class Media:
                def add_frame(self, frame):
                    raise RuntimeError("encode failed")
            sink = self.r.TickSink(Path(tmp), Media(), lambda: None)
            with self.assertRaisesRegex(RuntimeError, "encode failed"):
                sink(frame, trace)
            with np.load(Path(tmp) / "ticks/000000.npz", allow_pickle=False) as raw:
                for key in ("action", "observation.state", "observation.images.image", "observation.images.image2"):
                    np.testing.assert_array_equal(raw[key], frame[key])
                for key in trace:
                    np.testing.assert_array_equal(raw["trace/" + key], trace[key])
            self.assertEqual(sink.count, 1)
            sink.interrupted(RuntimeError("encode failed"))
            terminal = json.loads((Path(tmp) / "interrupted.json").read_text())
            self.assertEqual(terminal["last_known_tick"], 0)
            self.assertEqual(terminal["state_scope"], "last_emitted_pre_step_observation")
            self.assertIsNone(terminal["terminal_integration_state"])

    def test_sink_deadline_preserves_delivered_tick_and_blocks_media(self):
        frame, trace = self.sample()
        with tempfile.TemporaryDirectory() as tmp:
            class Media:
                def add_frame(self, frame):
                    raise AssertionError("media must not run")
            def expired():
                raise TimeoutError("deadline")
            sink = self.r.TickSink(Path(tmp), Media(), expired)
            with self.assertRaises(TimeoutError):
                sink(frame, trace)
            self.assertTrue((Path(tmp) / "ticks/000000.npz").is_file())

    def test_attempt_exception_preserves_original_even_if_finalization_fails(self):
        frame, trace = self.sample()
        error = RuntimeError("original expert failure")
        class Expert:
            def run(self, initial, episode, sink):
                sink(frame, trace)
                raise error
        class Media:
            def add_frame(self, frame):
                pass
            def finalize(self):
                raise OSError("footer failed")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as raised:
                self.r.record_attempt(Path(tmp), Expert(), Media(), make_plan()["episodes"][0],
                                      np.zeros(242), lambda: None)
            self.assertIs(raised.exception, error)
            self.assertTrue((Path(tmp) / "ticks/000000.npz").is_file())
            failure = json.loads((Path(tmp) / "interrupted.json").read_text())
            self.assertIn("original expert failure", failure["error"])
            self.assertIn("footer failed", (Path(tmp) / "finalization-error.json").read_text())

    def test_media_failure_after_expert_return_keeps_known_terminal(self):
        frame, trace = self.sample()
        class Expert:
            def run(self, initial, episode, sink):
                for _ in range(528):
                    sink(frame, trace)
                return {"success": True, "frames": 528, "final_qpos": np.arange(37, dtype=np.float64),
                        "final_state": np.arange(242, dtype=np.float64)}
        class Media:
            def add_frame(self, frame):
                pass
            def save_episode(self):
                raise OSError("media save failed")
            def finalize(self):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(OSError, "media save failed"):
                self.r.record_attempt(Path(tmp), Expert(), Media(), make_plan()["episodes"][0], np.zeros(242), lambda: None)
            self.assertEqual(len(list((Path(tmp) / "ticks").glob("*.npz"))), 528)
            with np.load(Path(tmp) / "terminal.npz") as terminal:
                np.testing.assert_array_equal(terminal["final_state"], np.arange(242))
            marker = json.loads((Path(tmp) / "interrupted.json").read_text())
            self.assertEqual(marker["state_scope"], "returned_terminal_state")
            self.assertEqual(marker["terminal_integration_state"], "terminal.npz:final_state")
            self.assertNotIn("expert_return_value", marker["unavailable"])

    def test_sink_original_exception_survives_expert_cleanup_exception(self):
        frame, trace = self.sample()
        original = ValueError("first media failure")
        class Expert:
            def run(self, initial, episode, sink):
                try:
                    sink(frame, trace)
                finally:
                    raise OSError("expert cleanup failure")
        class Media:
            def add_frame(self, frame):
                raise original
            def finalize(self):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as raised:
                self.r.record_attempt(Path(tmp), Expert(), Media(), make_plan()["episodes"][0], np.zeros(242), lambda: None)
            self.assertIs(raised.exception, original)
            self.assertTrue((Path(tmp) / "ticks/000000.npz").is_file())

    def test_cli_cannot_override_budget_reset_or_output(self):
        import sys
        for args in (["--collect", "--reserve-bytes", "0"], ["--collect", "--resume"],
                     ["--collect", "--root", "/tmp/new"], ["--collect", "--seed", "1"],
                     ["--collect", "--preflight"]):
            with self.subTest(args=args), patch.object(sys, "argv", ["record", *args]), \
                    patch.object(self.r, "collect", side_effect=AssertionError("must not execute")), self.assertRaises(SystemExit) as error:
                self.r.main()
            self.assertEqual(error.exception.code, 2)

    def test_snapshot_checks_reject_plan_reset_source_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.r.write_json(root / "plan.json", make_plan())
            self.r.write_json(root / "provenance.json", {"files": {"helper": "abc"}})
            self.r.write_npz(root / "reset.npz", {"initial_state": np.arange(242, dtype=np.float64)})
            hashes = self.r.snapshot_hashes(root)
            with patch.object(self.r, "provenance", return_value={"files": {"helper": "abc"}}):
                self.r.check_snapshots(root, hashes)
                for filename in ("plan.json", "reset.npz", "provenance.json"):
                    p = root / filename
                    original = p.read_bytes()
                    p.write_bytes(original + b" ")
                    with self.subTest(filename=filename), self.assertRaises(ValueError):
                        self.r.check_snapshots(root, hashes)
                    p.write_bytes(original)
            with patch.object(self.r, "provenance", return_value={"files": {"helper": "changed"}}), \
                    self.assertRaises(ValueError):
                self.r.check_snapshots(root, hashes)

    def test_exclusive_writes_do_not_overwrite_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            self.r.write_json(path, {"failed": True})
            with self.assertRaises(FileExistsError):
                self.r.write_json(path, {"failed": False})
            self.assertEqual(json.loads(path.read_text()), {"failed": True})

    def test_snapshot_manifest_cannot_add_paths_outside_frozen_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            self.r.write_json(root / "plan.json", make_plan())
            self.r.write_json(root / "provenance.json", {"files": {}})
            self.r.write_npz(root / "reset.npz", {"initial_state": np.zeros(242)})
            outside = Path(tmp) / "unrelated-fixture.txt"
            outside.write_text("synthetic nonsecret fixture")
            hashes = {**self.r.snapshot_hashes(root), "../unrelated-fixture.txt": self.r.file_hash(outside)}
            with patch.object(self.r, "provenance", return_value={"files": {}}), self.assertRaises(ValueError):
                self.r.check_snapshots(root, hashes)


if __name__ == "__main__":
    unittest.main()