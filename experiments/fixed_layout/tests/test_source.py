"""Source binding on real read-only data and temporary corrupted copies, no steps."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from data_collection.benchmark import capture_state, restore_state
from experiments.pose_snapshot_v2.expert import Expert

ROOT = Path(__file__).resolve().parents[3]
NPZ = ROOT / "datasets/franka_pilot_20260915/replay/episode_000007.npz"
SHA = "ca5c19eda5d089a4a33a838a3fc3e55b44895f77aa40e8d92d9490953290221b"


class SourceTests(unittest.TestCase):
    def setUp(self):
        try:
            self.s = importlib.import_module("experiments.fixed_layout.source")
        except ModuleNotFoundError as exc:
            self.fail(f"Task 1 source binding is not implemented: {exc}")
        self.expert = Expert()  # Constructor configures the simulator; no run/settle/render.
        self.manifest = json.loads(NPZ.with_name("manifest.json").read_text())
        with np.load(NPZ, allow_pickle=False) as trace:
            self.arrays = {key: trace[key].copy() for key in trace.files}

    def fixture(self, directory, *, arrays=None, manifest=None):
        path = Path(directory) / "episode_000007.npz"
        np.savez_compressed(path, **(self.arrays if arrays is None else arrays))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        meta = copy.deepcopy(self.manifest if manifest is None else manifest)
        meta["episodes"][7]["sha256"] = sha
        meta_path = path.with_name("manifest.json")
        meta_path.write_text(json.dumps(meta))
        return dict(source_path=path, manifest_path=meta_path, source_sha256=sha)

    def test_real_reset_bit_exact_fresh_copies_without_steps_render_or_expert(self):
        forbidden = {"mj_step", "mj_step1", "mj_step2", "mj_Euler", "mj_RungeKutta"}
        calls = []

        def guard(frame, event, arg):
            if event == "c_call" and getattr(arg, "__name__", "") in forbidden:
                raise AssertionError("Physics stepping forbidden")
            if event == "call":
                filename, name = frame.f_code.co_filename, frame.f_code.co_name
                if (name == "run" and filename.endswith("expert.py")) or (
                        "mujoco" in filename and "render" in filename):
                    raise AssertionError("Expert execution/render forbidden")
            if event == "c_call" and getattr(arg, "__name__", "") == "mj_forward":
                calls.append("forward")

        old_profile = sys.getprofile()
        try:
            sys.setprofile(guard)
            reset = self.s.load_reset(self.expert)
            again = self.s.load_reset(self.expert)
        finally:
            sys.setprofile(old_profile)
        self.assertTrue(calls, "Guard must observe native MuJoCo calls")
        self.assertEqual(set(reset), {"initial_state", "qpos", "initial_ctrl", "body_xpos",
                                      "camera_xpos", "camera_xmat"})
        self.assertEqual(reset["initial_state"].dtype, np.float64)
        self.assertEqual(reset["initial_state"].tobytes(), self.arrays["initial_state"].tobytes())
        for key in ("qpos", "body_xpos", "camera_xpos"):
            np.testing.assert_array_equal(reset[key], self.arrays[key][0])
        data = mujoco.MjData(self.expert.model)
        restore_state(self.expert.model, data, reset["initial_state"])
        self.assertEqual(capture_state(self.expert.model, data).tobytes(), reset["initial_state"].tobytes())
        np.testing.assert_array_equal(reset["initial_ctrl"], data.ctrl)
        np.testing.assert_array_equal(reset["camera_xmat"], data.cam_xmat)
        self.assertFalse(np.array_equal(reset["initial_ctrl"], self.arrays["ctrl"][0]))
        for key in reset:
            self.assertTrue(np.isfinite(reset[key]).all())
            self.assertFalse(np.shares_memory(reset[key], again[key]))

    def test_hash_is_checked_before_npz_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            args["source_path"].write_bytes(b"not a zip file")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.s._load_reset(self.expert, **args)

    def test_manifest_identity_roles_frames_and_scene_must_match(self):
        for key, value in (("episode_index", 8), ("episode_index", 7.0),
                           ("pick", "green"), ("place", "red"),
                           ("instruction", "another task"), ("frames", 527),
                           ("colors", ["red", "green", "blue"]), ("split", "validation"),
                           ("file", "../episode_000007.npz"), ("scene_hash", "0" * 64)):
            meta = copy.deepcopy(self.manifest)
            meta["episodes"][7][key] = value
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.s._load_reset(self.expert, **self.fixture(directory, manifest=meta))

    def test_manifest_sha_cannot_replace_fixed_expected_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            meta = json.loads(args["manifest_path"].read_text())
            meta["episodes"][7]["sha256"] = "0" * 64
            args["manifest_path"].write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                self.s._load_reset(self.expert, **args)

    def test_corrupt_qpos_body_camera_time_and_declared_positions_rejected(self):
        for key in ("qpos", "body_xpos", "camera_xpos", "sim_time"):
            arrays = copy.deepcopy(self.arrays)
            arrays[key].flat[0] += .001
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.s._load_reset(self.expert, **self.fixture(directory, arrays=arrays))
        meta = copy.deepcopy(self.manifest)
        meta["episodes"][7]["initial_objects"]["red"][0] += .01
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            self.s._load_reset(self.expert, **self.fixture(directory, manifest=meta))

    def test_nonfinite_wrong_shape_and_float32_precision_loss_rejected(self):
        variants = []
        for key in ("initial_state", "qpos", "body_xpos", "camera_xpos", "ctrl"):
            arrays = copy.deepcopy(self.arrays)
            arrays[key].flat[0] = np.nan
            variants.append(arrays)
        arrays = copy.deepcopy(self.arrays)
        arrays["qpos"] = arrays["qpos"][:-1]
        variants.append(arrays)
        arrays = copy.deepcopy(self.arrays)
        arrays["initial_state"] = arrays["initial_state"].astype(np.float32)
        variants.append(arrays)
        for i, arrays in enumerate(variants):
            with self.subTest(i=i), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.s._load_reset(self.expert, **self.fixture(directory, arrays=arrays))

    def test_optional_control_and_rotation_evidence_is_checked_when_present(self):
        arrays = copy.deepcopy(self.arrays)
        arrays["initial_ctrl"] = np.ones(self.expert.model.nu)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            self.s._load_reset(self.expert, **self.fixture(directory, arrays=arrays))
        arrays.pop("initial_ctrl")
        arrays["camera_xmat"] = np.zeros((528, self.expert.model.ncam, 9))
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            self.s._load_reset(self.expert, **self.fixture(directory, arrays=arrays))

    def test_calibration_physics_runtime_and_asset_drift_rejected(self):
        for key in ("camera_pos", "camera_quat", "camera_fovy", "timing", "versions", "source_hashes"):
            meta = copy.deepcopy(self.manifest)
            if key == "source_hashes":
                path = next(p for p in meta[key] if p.startswith("src/"))
                meta[key][path] = "0" * 64
            elif key == "versions":
                meta[key]["mujoco"] = "0.0"
            else:
                meta[key] = []
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.s._load_reset(self.expert, **self.fixture(directory, manifest=meta))
        self.expert.model.cam_quat[0, 0] += .01
        with self.assertRaises(ValueError):
            self.s.load_reset(self.expert)

    def test_live_physics_and_non_camera_geometry_cannot_hide_behind_asset_hashes(self):
        # Changing mass/gravity leaves reset qpos and positions intact. Source
        # file hashes alone must not certify a caller-mutated runtime model.
        for mutate in (lambda m: m.opt.gravity.__setitem__(2, -1.),
                       lambda m: m.body_mass.__setitem__(12, .5),
                       lambda m: m.light_diffuse.__setitem__((0, 0), .1)):
            expert = Expert()
            mutate(expert.model)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.s.load_reset(expert)

    def test_any_restored_integration_bit_change_is_rejected(self):
        def altered_restore(model, data, state):
            restore_state(model, data, state)
            data.qacc_warmstart[0] = np.nextafter(data.qacc_warmstart[0], np.inf)

        with patch.object(self.s, "restore_state", altered_restore), self.assertRaises(ValueError):
            self.s.load_reset(self.expert)

    def test_provenance_covers_helpers_assets_new_python_spec_and_source(self):
        report = self.s.provenance()
        json.dumps(report, allow_nan=False)
        files = report["files"]
        for path in ("experiments/pose_snapshot_v2/expert.py",
                     "experiments/pose_snapshot_v2/planner.py",
                     "experiments/paired_demonstrations/expert_v1.py",
                     "experiments/paired_demonstrations/media.py",
                     "experiments/paired_demonstrations/plan.py",
                     "data_collection/collect_libero_dataset.py", "data_collection/benchmark.py",
                     "src/robots/franka_panda/robot.py",
                     "docs/superpowers/specs/2026-09-18-fixed-layout-baseline-design.md"):
            self.assertEqual(files[path], hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
        for path in (ROOT / "experiments/fixed_layout").rglob("*.py"):
            self.assertIn(str(path.relative_to(ROOT)), files)
        self.assertEqual(files[str(NPZ.relative_to(ROOT))], SHA)
        self.assertIn("datasets/franka_pilot_20260915/replay/manifest.json", files)
        self.assertEqual(report["runtime"]["mujoco"], mujoco.__version__)

    def test_preflight_is_json_safe_honest_and_does_not_create_output(self):
        before = self.s.OUTPUT.exists()
        report = self.s.preflight()
        json.dumps(report, allow_nan=False)
        self.assertTrue(report["reset_valid"])
        self.assertEqual(report["source_sha256"], SHA)
        self.assertIs(report["storage_eligible"], report["free_bytes"] >= 30 * 1024**3)
        self.assertEqual(report["initial_state"], self.arrays["initial_state"].tolist())
        self.assertEqual(report["checks"]["camera_xmat"], "derived_from_source_bound_model_no_saved_rotation_trace")
        self.assertEqual(report["checks"]["initial_ctrl"], "restored_from_full_integration_state_not_first_command")
        self.assertFalse(report["collection_ready"])
        self.assertEqual(self.s.OUTPUT.exists(), before)

    def test_preflight_storage_and_invalid_source_are_separate(self):
        # Only this module's disk query is replaced; no global simulator patches.
        for free in (30 * 1024**3 - 1, 30 * 1024**3):
            with patch.object(self.s, "disk_usage", return_value=type("Usage", (), {"free": free})()):
                report = self.s.preflight()
            self.assertTrue(report["reset_valid"])
            self.assertEqual(report["storage_eligible"], free == 30 * 1024**3)
            self.assertFalse(report["collection_ready"])
        with patch.object(self.s, "load_reset", side_effect=ValueError("SHA256 mismatch")):
            report = self.s.preflight()
        self.assertFalse(report["reset_valid"])
        self.assertIn("SHA256", report["error"])
        self.assertIn("free_bytes", report)


if __name__ == "__main__":
    unittest.main()