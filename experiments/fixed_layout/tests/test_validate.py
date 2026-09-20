"""Synthetic arrays and real temporary parquet/media; simulator execution is fake."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.fixed_layout.contract import make_plan, TASK
from src.robots.franka_panda.config import Q_HOME


def rows_fixture(root, index=0, n=528):
    actions = np.tile(np.r_[Q_HOME[:7], 255.].astype(np.float32), (n, 1))
    states = np.tile(np.asarray(Q_HOME[:9], np.float32), (n, 1))
    table = pa.table({
        "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 8)),
        "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 9)),
        "timestamp": pa.array(np.arange(n, dtype=np.float32) / 30),
        "frame_index": pa.array(np.arange(n)), "episode_index": pa.array(np.full(n, index)),
        "index": pa.array(np.arange(n) + index * 528), "task_index": pa.array(np.zeros(n, np.int64))})
    path = root / "rows.parquet"
    pq.write_table(table, path)
    return path, actions, states


def grasp_fixture(n=80):
    actions = np.zeros((n, 8), dtype=np.float32)
    actions[-30:, -1] = 255
    post = {"left_contact": np.zeros(n, bool), "right_contact": np.zeros(n, bool),
            "pick_xyz": np.tile([.28, -.38, .021], (n, 1)),
            "place_xyz": np.tile([.267, .351, .021], (n, 1)),
            "hand_xyz": np.zeros((n, 3))}
    post["left_contact"][30:50] = True
    post["right_contact"][30:50] = True
    post["pick_xyz"][35:50, 2] = .081
    post["pick_xyz"][-30:] = [.267, .351, .061]
    return actions, post


class SyntheticSimulator:
    """Numeric test double only. No MuJoCo model, renderer, expert or native step."""
    def __init__(self, velocity=.001):
        self.velocity = velocity
        ids = {"target_cube_red": 1, "target_cube_green": 2, "hand": 3, "left_finger": 4, "right_finger": 5}
        self.model = SimpleNamespace(nq=37, nbody=6, ncam=3, geom_bodyid=np.arange(6),
                                     body=lambda name: SimpleNamespace(id=ids[name]))
        self.bounds = np.tile([-10., 10.], (7, 1))
        self.run_calls, self.steps = 0, 0
        self.config = {"synthetic_fixture": True}
        self.times, self.phases, self.phase_times, self.commands = [], [], [], []
        durations, grippers = [2., 1.5, 1., 2., 3., 2., 1., 2., 3.], [255, 255, 0, 0, 0, 0, 255, 255, 255]
        time = sum([1 / 600] * 120)
        phase, start, target = 0, time, np.asarray(Q_HOME[:7]).copy()
        for i in range(528):
            self.times.append(time)
            self.phases.append(phase)
            self.phase_times.append(time - start)
            target = target.copy(); target[0] += self.velocity / 30
            self.commands.append(np.r_[target, grippers[phase]])
            if time - start > durations[phase]:
                phase += 1; start = time
            for _ in range(20):
                time += 1 / 600
        self.times.append(time)
        self.commands = np.asarray(self.commands)
        self.qposes = np.zeros((529, 37))
        for i in range(529):
            self.qposes[i, :7] = Q_HOME[:7] if i == 0 else self.commands[i - 1, :7]
            opened = i == 0 or self.commands[i - 1, 7] > 127
            self.qposes[i, 7:9] = .04 if opened else 0.
            self.qposes[i, 9:12] = [.28, -.38, .021 if i < 160 else .081]
            self.qposes[i, 12:15] = [.267, .351, .021]
            if i >= 350:
                self.qposes[i, 9:12] = [.267, .351, .061]

    def data(self, model):
        return SimpleNamespace(qpos=np.zeros(37), ctrl=np.zeros(8), time=0., tick=0)

    def synchronize(self, model, data):
        data.xpos = np.zeros((6, 3))
        data.xpos[1], data.xpos[2] = data.qpos[9:12], data.qpos[12:15]
        data.xpos[3] = data.xpos[1] + [0, 0, .1]
        data.cam_xpos = np.zeros((3, 3))
        data.cam_xmat = np.tile(np.eye(3).reshape(1, 9), (3, 1))
        data.contact = [SimpleNamespace(geom1=1, geom2=j) for j, finger in ((4, 7), (5, 8)) if data.qpos[finger] < .02]
        data.ncon = len(data.contact)

    def restore(self, model, data, state):
        data.time, data.qpos[:], data.ctrl[:] = state[0], state[1:38], state[38:46]
        data.tick = int(np.argmin(np.abs(np.asarray(self.times) - data.time)))
        self.synchronize(model, data)

    def step(self, model, data, dt, arm_dof):
        self.steps += 1
        data.tick += 1
        data.time, data.qpos[:] = self.times[data.tick], self.qposes[data.tick]
        data.qpos[:7] = data.ctrl[:7]
        self.synchronize(model, data)
        return 20

    def run(self, *args):
        self.run_calls += 1
        raise AssertionError("Validator must never regenerate an expert")

    def patches(self, validator):
        stack = ExitStack()
        stack.enter_context(patch.object(validator, "mujoco", SimpleNamespace(MjData=self.data)))
        stack.enter_context(patch.object(validator, "restore_state", self.restore))
        stack.enter_context(patch.object(validator, "synchronize_observation", self.synchronize))
        stack.enter_context(patch.object(validator, "step_control", self.step))
        return stack

    def fixture(self, root, validator, episode_index=0, images=None):
        """Write actual arrays/files; no mocked row, trace, contact or admission gate."""
        from experiments.fixed_layout.record import write_json, write_npz
        from experiments.paired_demonstrations.media import pixels_digest
        root.mkdir()
        (root / "ticks").mkdir()
        (root / "replays").mkdir()
        rgb = np.zeros((512, 512, 3), np.uint8)
        images = images or {"observation.images.image": rgb, "observation.images.image2": rgb}
        hashes = np.asarray([pixels_digest(image) for image in images.values()])
        trace_rows = []
        for i in range(528):
            data = self.data(self.model)
            data.qpos[:], data.time = self.qposes[i], self.times[i]
            self.synchronize(self.model, data)
            observed = validator._observation(self.model, data)
            trace = {key: observed[key] for key in ("qpos", "sim_time", "body_xpos", "camera_xpos", "camera_xmat")}
            trace.update(ctrl=self.commands[i], substeps=np.asarray(20), requested_joint_velocity=np.r_[self.velocity, np.zeros(6)],
                applied_joint_velocity=np.r_[self.velocity, np.zeros(6)], command_saturation=np.zeros(7, bool),
                image_time=np.asarray(self.times[i]), image_sha256=hashes,
                phase_index=np.asarray(self.phases[i]), phase_time=np.asarray(self.phase_times[i]), target_pos=np.zeros(3))
            post = self.data(self.model)
            post.qpos[:], post.ctrl[:], post.time = self.qposes[i + 1], self.commands[i], self.times[i + 1]
            self.synchronize(self.model, post)
            trace["released_stack"] = np.asarray(validator.stack_metrics(self.model, post, "red", "green")["released_stack"])
            trace_rows.append(trace)
            raw = {"trace/" + k: v for k, v in trace.items()}
            raw.update({"action": self.commands[i].astype(np.float32), "observation.state": self.qposes[i, :9].astype(np.float32),
                "task": np.asarray(TASK), **images})
            write_npz(root / "ticks" / f"{i:06d}.npz", raw)
        trace = {k: np.asarray([row[k] for row in trace_rows]) for k in trace_rows[0]}
        initial = self.data(self.model)
        initial.qpos[:], initial.time = self.qposes[0], self.times[0]
        self.synchronize(self.model, initial)
        reset = {"initial_state": np.r_[initial.time, initial.qpos, initial.ctrl], "qpos": initial.qpos.copy(),
                 "initial_ctrl": initial.ctrl.copy(), "body_xpos": initial.xpos.copy(),
                 "camera_xpos": initial.cam_xpos.copy(), "camera_xmat": initial.cam_xmat.copy()}
        write_npz(root / "terminal.npz", {"final_qpos": post.qpos, "final_state": np.r_[post.time, post.qpos, post.ctrl]})
        write_json(root / "slot.json", make_plan()["episodes"][episode_index])
        stable = sum(bool(v) for v in trace["released_stack"])
        write_json(root / "outcome.json", {"success": True, "frames": 528, "stable_ticks": stable,
                                            "final_metrics": validator.stack_metrics(self.model, post, "red", "green")})
        return reset, trace


class ValidateTests(unittest.TestCase):
    def setUp(self):
        try:
            self.v = importlib.import_module("experiments.fixed_layout.validate")
        except ModuleNotFoundError as exc:
            self.fail(f"Task 2 validator missing: {exc}")

    def test_persisted_rows_float32_indices_and_finite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, actions, states = rows_fixture(Path(tmp))
            actual = self.v.read_rows(path, 0)
            np.testing.assert_array_equal(actual[0], actions)
            np.testing.assert_array_equal(actual[1], states)
            baseline = pq.read_table(path)
            for key, values in (("frame_index", np.zeros(528, np.int64)),
                                ("task_index", np.ones(528, np.int64)),
                                ("episode_index", np.ones(528, np.int64)),
                                ("index", np.arange(528) + 1),
                                ("timestamp", np.full(528, np.nan, np.float32))):
                changed = baseline.set_column(baseline.schema.get_field_index(key), key, pa.array(values))
                pq.write_table(changed, path)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.v.read_rows(path, 0)

    def test_rows_reject_float64_padding_oracle_fields_and_wrong_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, actions, states = rows_fixture(root)
            original = pq.read_table(path)
            variants = [original.set_column(0, "action", pa.array(actions.tolist(), type=pa.list_(pa.float64(), 8))),
                        original.append_column("cube_pose", pa.array(np.zeros(528))), original.slice(0, 527)]
            for changed in variants:
                pq.write_table(changed, path)
                with self.assertRaises(ValueError):
                    self.v.read_rows(path, 0)

    def test_real_metadata_fixtures_enforce_24_by_528_one_task_and_20_4_split(self):
        import pandas as pd
        from experiments.paired_demonstrations.media import FEATURES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = root / "dataset/meta"
            (meta / "episodes").mkdir(parents=True)
            features = json.loads(json.dumps(FEATURES))
            for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
                features[key] = {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
            info = {"total_episodes": 24, "total_frames": 12672, "total_tasks": 1, "fps": 30,
                    "splits": {"train": "0:20", "validation": "20:24"}, "features": features}
            (meta / "info.json").write_text(json.dumps(info))
            pd.DataFrame({"task_index": [0]}, index=[TASK]).to_parquet(meta / "tasks.parquet")
            episodes = [{"episode_index": i, "length": 528, "tasks": [TASK],
                         "dataset_from_index": i * 528, "dataset_to_index": (i + 1) * 528} for i in range(24)]
            pq.write_table(pa.Table.from_pylist(episodes), meta / "episodes/episodes.parquet")
            self.assertEqual(len(self.v._metadata(root, 24, final=True)[1]), 24)
            for key, value in (("total_frames", 12671), ("total_episodes", 23), ("total_tasks", 2),
                               ("splits", {"train": "0:24"}), ("fps", 29)):
                altered = {**info, key: value}
                (meta / "info.json").write_text(json.dumps(altered))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.v._metadata(root, 24, final=True)
            (meta / "info.json").write_text(json.dumps(info))
            bad = copy.deepcopy(episodes); bad[20]["tasks"] = ["reverse roles"]
            pq.write_table(pa.Table.from_pylist(bad), meta / "episodes/episodes.parquet")
            with self.assertRaises(ValueError):
                self.v._metadata(root, 24, final=True)

    def test_operational_grasp_lift_and_final_strict_stack_gates(self):
        actions, post = grasp_fixture()
        result = self.v.contact_gates(actions, post, .021, "red", "green")
        self.assertEqual(result["closed_ticks"], 50)
        self.assertEqual(result["bilateral_ticks"], 20)
        self.assertEqual(result["stable_ticks"], 30)
        self.assertAlmostEqual(result["contact_lift_m"], .060)
        self.assertEqual(result["hand_distance_scope"], "descriptive_only_hand_body_origin")

    def test_never_close_one_tick_contact_lift_without_contact_and_wrong_role_fail(self):
        a, p = grasp_fixture()
        variants = []
        never = a.copy(); never[:, -1] = 255
        variants.append((never, p, "red", "green"))
        once = never.copy(); once[35, -1] = 0
        variants.append((once, p, "red", "green"))
        transient = copy.deepcopy(p); transient["right_contact"][:] = False; transient["right_contact"][35] = True
        variants.append((a, transient, "red", "green"))
        no_lift = copy.deepcopy(p); no_lift["pick_xyz"][30:50, 2] = .021
        variants.append((a, no_lift, "red", "green"))
        nan = copy.deepcopy(p); nan["pick_xyz"][2, 0] = np.nan
        variants.append((a, nan, "red", "green"))
        touching = copy.deepcopy(p); touching["left_contact"][-1] = True
        variants.append((a, touching, "red", "green"))
        variants.append((a, p, "green", "red"))
        for args in variants:
            with self.subTest(case=len(variants)), self.assertRaises(ValueError):
                self.v.contact_gates(args[0], args[1], .021, args[2], args[3])

    def test_replay_coverage_exactly_two_unique_modes_per_episode(self):
        results = [{"episode": i, "mode": m, "passed": True} for i in range(24)
                   for m in ("reference", "persisted_float32")]
        self.v.check_replay_coverage(results)
        for bad in (results[:-1], results + results[:1], results[:-1] + [results[0]],
                    [{**r, "mode": "scripted_expert"} for r in results],
                    [{**r, "passed": False} for r in results]):
            with self.assertRaises(ValueError):
                self.v.check_replay_coverage(bad)

    def test_replay_coverage_rejects_bool_spoofed_episode_ids(self):
        results = [{"episode": i, "mode": m, "passed": True} for i in range(24) for m in ("reference", "persisted_float32")]
        results[0]["episode"] = False
        with self.assertRaises(ValueError):
            self.v.check_replay_coverage(results)

    def test_duplicates_report_hashes_rms_units_and_block_admission(self):
        a = np.zeros((528, 8), np.float32)
        b = a.copy(); b[:, 0] = .1
        s = np.zeros((528, 9), np.float32)
        report = self.v.duplicate_report([(a, s), (b, s), (a.copy(), s.copy())])
        self.assertEqual(report["exact_duplicates"], [[0, 2]])
        self.assertEqual(len(report["sequences"]), 3)
        self.assertEqual(report["units"], {"arm": "rad", "gripper": "command_0_255"})
        self.assertAlmostEqual(report["pairs"][0]["arm_rms_rad"], .1 / np.sqrt(7), places=7)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.v.require_no_duplicates(report)

    def test_fingerprint_binds_all_files_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "validation.json").write_text('{"status":"passed"}')
            (root / "trace.npz").write_bytes(b"original")
            first = self.v.fingerprint(root)
            (root / "trace.npz").write_bytes(b"changed")
            self.assertNotEqual(first, self.v.fingerprint(root))
            (root / "escape").symlink_to("/etc/hostname")
            with self.assertRaises(ValueError):
                self.v.fingerprint(root)

    def test_saved_pass_alone_and_partial_roots_never_admit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "validation.json").write_text(json.dumps({"status": "passed", "training_ready": True}))
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.v.require_fixed_layout_pass(root)

    def test_decode_checks_deadline_inside_real_video_and_intervals(self):
        import av
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.mp4"
            with av.open(str(path), "w") as video:
                stream = video.add_stream("libsvtav1", rate=30)
                stream.width, stream.height, stream.pix_fmt = 512, 512, "yuv420p"
                stream.options = {"preset": "12", "svtav1-params": "lp=1"}
                for i in range(4):
                    frame = av.VideoFrame.from_ndarray(np.full((512, 512, 3), i, np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        video.mux(packet)
                for packet in stream.encode():
                    video.mux(packet)
            self.assertEqual(self.v.validate_video(path, [(0., 4 / 30, 4)], lambda: None), 4)
            calls = []
            def expired():
                calls.append(1)
                if len(calls) == 3:
                    raise TimeoutError("decode deadline")
            with self.assertRaises(TimeoutError):
                self.v.validate_video(path, [(0., 4 / 30, 4)], expired)
            for intervals in ([(0., 3 / 30, 4)], [(1 / 30, 5 / 30, 4)], [(0., 4 / 30, 4)] * 2):
                with self.assertRaises(ValueError):
                    self.v.validate_video(path, intervals, lambda: None)

    def test_replay_budget_rejects_rerun_and_checks_deadline_before_step(self):
        expert = SimpleNamespace(model=object(), run=lambda *a: self.fail("expert regeneration forbidden"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference").mkdir()
            with self.assertRaises(FileExistsError):
                self.v.replay_controls(root, expert, {}, {}, np.zeros((528, 8), np.float32),
                                       "reference", lambda: None)
        with tempfile.TemporaryDirectory() as tmp:
            def expired():
                raise TimeoutError("replay deadline")
            with patch.object(self.v, "step_control", side_effect=AssertionError("no step")), \
                    self.assertRaises(TimeoutError):
                self.v.replay_controls(Path(tmp), expert, {}, {}, np.zeros((528, 8), np.float32),
                                       "reference", expired)

    def test_media_provenance_includes_new_imported_helpers_and_runtime(self):
        from experiments.fixed_layout.source import provenance
        p = provenance()
        for name in ("experiments/paired_demonstrations/validate.py", "experiments/paired_demonstrations/record.py"):
            self.assertIn(name, p["files"])
        for name in ("av", "pyarrow", "pandas", "datasets", "Pillow", "imageio-ffmpeg"):
            self.assertIn(name, p["media_runtime"])
        self.assertIn("ffmpeg_libraries", p["media_runtime"])

    def test_full_synthetic_trace_and_two_control_replays_without_expert(self):
        sim = SyntheticSimulator()
        with tempfile.TemporaryDirectory() as tmp, sim.patches(self.v):
            root = Path(tmp) / "attempt"
            reset, trace = sim.fixture(root, self.v)
            a, s, t = sim.commands.astype(np.float32), sim.qposes[:-1, :9].astype(np.float32), np.arange(528, dtype=np.float32) / 30
            actual, terminal, gates = self.v.load_attempt(root, make_plan()["episodes"][0], a, s, t, sim, reset, lambda: None)
            self.assertGreaterEqual(gates["stable_ticks"], 30)
            for mode in ("reference", "persisted_float32"):
                result = self.v.replay_controls(root / "replays", sim, reset, actual, a, mode, lambda: None)
                self.assertTrue(result["passed"])
                self.assertLessEqual(result["max_policy_state_error"], .001)
            self.assertEqual(sim.steps, 1056)
            self.assertEqual(sim.run_calls, 0)
            with self.assertRaises(FileExistsError):
                self.v.replay_controls(root / "replays", sim, reset, actual, a, "reference", lambda: None)
            self.assertEqual(sim.steps, 1056)
            initial_path = root / "replays/persisted_float32/000000.npz"
            initial_arrays = self.v._npz(initial_path)
            changed = copy.deepcopy(initial_arrays)
            changed["pre/qpos"][30] += .01
            np.savez_compressed(initial_path, **changed)
            with self.assertRaisesRegex(ValueError, "reset"):
                self.v.audit_replay(root / "replays/persisted_float32", sim, reset, actual, a, "persisted_float32", lambda: None)
            np.savez_compressed(initial_path, **initial_arrays)
            raw_path = root / "ticks/000100.npz"
            arrays = self.v._npz(raw_path)
            arrays["action"][0] += .01
            np.savez_compressed(raw_path, **arrays)
            with self.assertRaisesRegex(ValueError, "original observations"):
                self.v.load_attempt(root, make_plan()["episodes"][0], a, s, t, sim, reset, lambda: None)
            replay_path = root / "replays/reference/000100.npz"
            arrays = self.v._npz(replay_path)
            arrays["post/qpos"][0] += .01
            np.savez_compressed(replay_path, **arrays)
            with self.assertRaises(ValueError):
                self.v.audit_replay(root / "replays/reference", sim, reset, actual, a, "reference", lambda: None)

    def test_deadline_interrupts_inside_replay_and_preserves_emitted_steps(self):
        sim = SyntheticSimulator()
        with tempfile.TemporaryDirectory() as tmp, sim.patches(self.v):
            root = Path(tmp)
            reset = {"initial_state": np.r_[sim.times[0], sim.qposes[0], np.zeros(8)]}
            def check():
                if sim.steps == 3:
                    raise TimeoutError("third replay tick")
            with self.assertRaises(TimeoutError):
                self.v.replay_controls(root, sim, reset, {"ctrl": sim.commands}, sim.commands.astype(np.float32), "reference", check)
            self.assertEqual(sim.steps, 3)
            self.assertEqual(len(list((root / "reference").glob("*.npz"))), 3)
            self.assertTrue((root / "reference/failure.json").exists())

    def test_stale_seal_blocks_changed_report_data_and_missing_media_before_admission(self):
        from experiments.fixed_layout import record
        with tempfile.TemporaryDirectory() as tmp, patch.object(record, "disk_usage") as disk:
            disk.return_value.free = 40 * 1024**3
            root = Path(tmp)
            report = {"status": "passed", "training_ready": True}
            (root / "validation.json").write_text(json.dumps(report))
            (root / "media.mp4").write_bytes(b"synthetic media placeholder")
            (root / "seal.json").write_text(json.dumps({"fingerprint": self.v.fingerprint(root)}))
            for filename in ("media.mp4", "validation.json"):
                path = root / filename
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "stale"):
                    self.v.require_fixed_layout_pass(root)
                path.write_bytes(original)

    def test_saved_duplicate_pass_is_explicitly_blocked_before_training_admission(self):
        from experiments.fixed_layout import record
        with tempfile.TemporaryDirectory() as tmp, patch.object(record, "disk_usage") as disk:
            disk.return_value.free = 40 * 1024**3
            root = Path(tmp)
            (root / "validation.json").write_text(json.dumps({"status": "passed", "training_ready": True,
                                                             "duplicates": {"exact_duplicates": [[0, 1]]}}))
            (root / "seal.json").write_text(json.dumps({"fingerprint": self.v.fingerprint(root)}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.v.require_fixed_layout_pass(root)

    def test_current_source_or_runtime_drift_blocks_sealed_admission(self):
        from experiments.fixed_layout import record
        with tempfile.TemporaryDirectory() as tmp, patch.object(record, "disk_usage") as disk, \
                patch.object(self.v, "provenance", return_value={"files": {"helper.py": "new"}}), \
                patch.object(self.v, "Expert", side_effect=AssertionError("must reject before model construction")):
            disk.return_value.free = 40 * 1024**3
            root = Path(tmp)
            (root / "validation.json").write_text(json.dumps({"status": "passed", "training_ready": True,
                "duplicates": {"exact_duplicates": []}, "provenance": {"files": {"helper.py": "old"}}}))
            (root / "seal.json").write_text(json.dumps({"fingerprint": self.v.fingerprint(root)}))
            with self.assertRaisesRegex(ValueError, "Source/runtime"):
                self.v.require_fixed_layout_pass(root)


if __name__ == "__main__":
    unittest.main()