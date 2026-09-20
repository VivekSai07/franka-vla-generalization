"""CPU fixtures test orchestration/media, not expert feasibility or trajectories."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np


class PairedRecordingTests(unittest.TestCase):
    def write_legacy_gap(self, workspace, version, count):
        info = workspace / f"datasets/franka_libero_stacking_v{version}/meta/info.json"
        info.parent.mkdir(parents=True, exist_ok=True)
        info.write_text(json.dumps({"total_episodes": count}))
        return info

    def legacy_inventory_fixture(self, workspace):
        from experiments.paired_demonstrations.record import prior_inventory
        from src.robots.franka_panda.config import SCENE_PATH
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        for version, count in ((1, 2), (2, 1)):
            self.write_legacy_gap(workspace, version, count)
        self.write_paired_history(workspace / "outputs/paired_demonstrations/prior", model, count=1)
        return model, prior_inventory(workspace, model)

    def test_approved_legacy_gaps_are_visible_unchecked_and_hash_bound(self):
        from experiments.paired_demonstrations.record import require_inventory, preflight
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _, inventory = self.legacy_inventory_fixture(workspace)
            try:
                require_inventory(inventory)
            except ValueError as exc:
                self.fail(f"The user-approved exact three-episode exception was blocked: {exc}")
            expected = []
            for version, count in ((1, 2), (2, 1)):
                dataset = f"datasets/franka_libero_stacking_v{version}"
                info = dataset + "/meta/info.json"
                expected.append({"dataset": dataset, "episode_count": count,
                    "missing_message": f"{dataset}: {count} episodes without full initial geometry",
                    "reason": "userapproved20260917", "geometry_checked": False,
                    "excluded_from_new_train_eval": True,
                    "files": {info: hashlib.sha256((workspace / info).read_bytes()).hexdigest()}})
            self.assertEqual(inventory["historical_geometry_exceptions"], expected)
            self.assertEqual(inventory["missing"], [entry["missing_message"] for entry in expected])
            self.assertEqual(len(inventory["geometry_hashes"]), 1)
            result = preflight(workspace)
            self.assertEqual(result["collection_status"], "not_started")
            self.assertEqual(result["historical_geometry_exceptions"], expected)
            self.assertEqual(result["missing_geometry"], inventory["missing"])
            self.assertFalse(result["training_ready"])
            self.assertEqual(result["expert_trajectories_executed"], 0)
            self.assertEqual([p.name for p in (workspace / "outputs/paired_demonstrations").iterdir()], ["prior"])

    def test_exception_never_allows_additional_gaps_or_changed_counts(self):
        from experiments.paired_demonstrations.record import prior_inventory, require_inventory, preflight, collect
        for change in ("v1_count", "v2_count", "other_root", "missing_metadata", "missing_bank", "missing_paired"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                model, _ = self.legacy_inventory_fixture(workspace)
                if change == "v1_count": self.write_legacy_gap(workspace, 1, 3)
                if change == "v2_count": self.write_legacy_gap(workspace, 2, 2)
                if change == "other_root": self.write_legacy_gap(workspace, "1_copy", 2)
                if change == "missing_metadata": (workspace / "datasets/franka_libero_stacking_v1/meta/info.json").unlink()
                if change == "missing_bank": (workspace / "outputs/forgotten_scenes").mkdir()
                if change == "missing_paired": (workspace / "outputs/paired_demonstrations/forgotten").mkdir()
                with self.assertRaises(ValueError): require_inventory(prior_inventory(workspace, model))
                self.assertEqual(preflight(workspace)["collection_status"], "blocked")
                output = workspace / "outputs/paired_demonstrations/new"
                with self.assertRaises(ValueError): collect(output, workspace, reserve_bytes=0)
                self.assertFalse(output.exists())

    def test_exception_evidence_cannot_be_omitted_forged_or_marked_checked(self):
        from experiments.paired_demonstrations.record import require_inventory
        with tempfile.TemporaryDirectory() as tmp:
            _, inventory = self.legacy_inventory_fixture(Path(tmp))
            self.assertIn("historical_geometry_exceptions", inventory)
            for change in ("omit", "extra", "reason", "checked", "included", "hash", "no_binding", "no_geometry", "duplicate_missing"):
                bad = copy.deepcopy(inventory)
                if change == "omit": bad.pop("historical_geometry_exceptions")
                if change == "extra": bad["historical_geometry_exceptions"].append(copy.deepcopy(bad["historical_geometry_exceptions"][0]))
                if change == "reason": bad["historical_geometry_exceptions"][0]["reason"] = "automatic"
                if change == "checked": bad["historical_geometry_exceptions"][0]["geometry_checked"] = True
                if change == "included": bad["historical_geometry_exceptions"][0]["excluded_from_new_train_eval"] = False
                if change == "hash": bad["historical_geometry_exceptions"][0]["files"] = {"fake": "0" * 64}
                if change == "no_binding": bad["files"].pop("datasets/franka_libero_stacking_v1/meta/info.json")
                if change == "no_geometry": bad["geometry_hashes"] = []
                if change == "duplicate_missing": bad["missing"].append(bad["missing"][0])
                with self.subTest(change=change), self.assertRaises(ValueError): require_inventory(bad)

    def test_approved_gap_does_not_excuse_replay_integrity_errors(self):
        from experiments.paired_demonstrations.record import prior_inventory, require_inventory
        for corruption in ("hash", "count", "unindexed", "nonfinite", "path"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                model, _ = self.legacy_inventory_fixture(workspace)
                replay = workspace / "datasets/franka_libero_stacking_v1/replay"
                replay.mkdir()
                path = replay / "episode.npz"
                qpos = mujoco.MjData(model).qpos.copy()
                if corruption == "nonfinite": qpos[9] = np.nan
                np.savez(path, qpos=qpos)
                entry = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                if corruption == "hash": entry["sha256"] = "0" * 64
                if corruption == "path": entry["file"] = "../episode.npz"
                if corruption == "unindexed": np.savez(replay / "orphan.npz", qpos=qpos)
                (replay / "manifest.json").write_text(json.dumps({"episodes": [entry]}))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    require_inventory(prior_inventory(workspace, model))

    def test_validation_and_training_gate_recheck_exception_evidence_and_live_inventory(self):
        from experiments.paired_demonstrations import validate as validator
        from experiments.paired_demonstrations.record import provenance
        from experiments.paired_demonstrations.plan import VERSION, digest, make_plan
        from experiments.paired_demonstrations.media import ENCODING
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            model, inventory = self.legacy_inventory_fixture(workspace)
            self.assertIn("historical_geometry_exceptions", inventory)
            root = workspace / "outputs/paired_demonstrations/current"
            root.mkdir()
            plan = make_plan()
            entries = [{**e, "success": True, "frames": 528} for e in plan["episodes"]]
            manifest = {"schema": VERSION, "provenance": provenance(), "encoding": ENCODING,
                "prior_workspace": str(workspace), "inventory_digest": digest(inventory),
                "historical_geometry_exceptions": inventory["historical_geometry_exceptions"],
                "status": "complete", "plan": plan, "plan_digest": digest(plan), "episodes": entries}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "plan.json").write_text(json.dumps(plan))
            (root / "attempts.jsonl").write_text("\n".join(json.dumps(e) for e in entries))
            (root / "prior_inventory.json").write_text(json.dumps(inventory))
            result = {"status": "passed", "training_ready": True, "schema": VERSION,
                "episode_count": 36, "provenance": provenance(), "fingerprint": validator.fingerprint(root),
                "alignment_returncode": 0, "alignment_output": "PASS: recorded actions",
                "historical_geometry_exceptions": inventory["historical_geometry_exceptions"],
                "results": [{"episode": i, "mode": mode, "passed": True}
                            for i in range(36) for mode in validator.MODES]}
            report = root / "paired_validation.json"
            report.write_text(json.dumps(result))
            self.assertEqual(validator.check_sources_and_inventory(root, manifest, model), inventory)
            self.assertEqual(validator.require_paired_pass(root), result)
            # Full validator reaches the real inventory check, then rejects missing
            # new reset states without rendering/replaying. Failure keeps the caveat.
            from experiments.paired_demonstrations.expert_v1 import Expert
            manifest["configuration"] = Expert().config
            manifest["states"] = []
            (root / "manifest.json").write_text(json.dumps(manifest))
            with patch.object(validator, "ROOT", workspace), self.assertRaisesRegex(ValueError, "Missing preregistered layouts"):
                validator.validate(root)
            failed = json.loads(report.read_text())
            self.assertFalse(failed["training_ready"])
            self.assertEqual(failed["historical_geometry_exceptions"], inventory["historical_geometry_exceptions"])
            result["fingerprint"] = validator.fingerprint(root)
            report.write_text(json.dumps(result))
            self.assertEqual(validator.require_paired_pass(root), result)
            bad = copy.deepcopy(manifest)
            bad["historical_geometry_exceptions"] = []
            with self.assertRaises(ValueError): validator.check_sources_and_inventory(root, bad, model)
            for change in ("omitted_report", "forged_report", "new_gap", "changed_hash"):
                with self.subTest(change=change):
                    bad = copy.deepcopy(result)
                    if change == "omitted_report": bad.pop("historical_geometry_exceptions")
                    if change == "forged_report": bad["historical_geometry_exceptions"][0]["geometry_checked"] = True
                    if change == "new_gap": self.write_legacy_gap(workspace, 3, 1)
                    if change == "changed_hash":
                        path = workspace / "datasets/franka_libero_stacking_v1/meta/info.json"
                        path.write_text(path.read_text() + "\n")
                    report.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError): validator.require_paired_pass(root)
                    if change == "new_gap":
                        with self.assertRaises(ValueError): validator.check_sources_and_inventory(root, manifest, model)
                        extra = workspace / "datasets/franka_libero_stacking_v3"
                        (extra / "meta/info.json").unlink()
                        (extra / "meta").rmdir()
                        extra.rmdir()

    def write_paired_history(self, root, model, count=18, status="failed", offset=0):
        from experiments.paired_demonstrations.plan import make_plan
        states = root / "states"
        states.mkdir(parents=True)
        entries, qposes = [], []
        for layout in make_plan()["layouts"][:count]:
            qpos = mujoco.MjData(model).qpos.copy()
            qpos[9] += (offset + layout["ordinal"] + 1) * .001
            path = states / f"layout_{layout['ordinal']:02d}.npz"
            np.savez(path, qpos=qpos)
            entries.append({**layout, "file": str(path.relative_to(root)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            qposes.append(qpos)
        (root / "manifest.json").write_text(json.dumps(
            {"status": status, "states": entries, "episodes": []}))
        return qposes

    def test_inventory_includes_previous_paired_and_partial_reset_sets(self):
        from experiments.paired_demonstrations.record import prior_inventory, require_inventory, cube_digest, reject_duplicate
        from src.robots.franka_panda.config import SCENE_PATH
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "datasets").mkdir()
            expected_files, expected_geometry = {}, set()
            for name, count, status, offset in (("completed", 18, "complete", 0),
                                                ("partial18reset", 18, "failed", 18),
                                                ("partial3reset", 3, "preparing", 36)):
                root = workspace / "outputs/paired_demonstrations" / name
                qposes = self.write_paired_history(root, model, count, status, offset)
                expected_geometry.update(cube_digest(model, qpos) for qpos in qposes)
                for path in root.rglob("*"):
                    if path.is_file():
                        expected_files[str(path.relative_to(workspace))] = hashlib.sha256(path.read_bytes()).hexdigest()
            inventory = prior_inventory(workspace, model)
            self.assertEqual(inventory["files"], expected_files)
            self.assertEqual(inventory["geometry_hashes"], sorted(expected_geometry))
            self.assertFalse(inventory["final_states_restored"])
            require_inventory(inventory)
            with self.assertRaises(ValueError):
                reject_duplicate(cube_digest(model, qposes[0]), set(inventory["geometry_hashes"]))

    def test_previous_paired_missing_manifest_blocks_inventory(self):
        from experiments.paired_demonstrations.record import prior_inventory, require_inventory
        from src.robots.franka_panda.config import SCENE_PATH
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "datasets").mkdir()
            self.write_paired_history(workspace / "outputs/paired_demonstrations/valid", model)
            root = workspace / "outputs/paired_demonstrations/forgotten"
            self.write_paired_history(root, model, count=1)
            (root / "manifest.json").unlink()
            inventory = prior_inventory(workspace, model)
            self.assertTrue(inventory["missing"])
            with self.assertRaises(ValueError): require_inventory(inventory)

    def test_previous_paired_state_corruption_cannot_disappear_from_inventory(self):
        from experiments.paired_demonstrations.record import prior_inventory
        from src.robots.franka_panda.config import SCENE_PATH
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        for corruption in ("hash", "unindexed", "missing", "wrong_directory"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                (workspace / "datasets").mkdir()
                root = workspace / "outputs/paired_demonstrations/partial18reset"
                self.write_paired_history(root, model)
                path = root / "states/layout_00.npz"
                if corruption == "hash":
                    np.savez(path, qpos=np.zeros(model.nq))
                elif corruption == "unindexed":
                    np.savez(root / "states/orphan.npz", qpos=mujoco.MjData(model).qpos)
                elif corruption == "missing":
                    path.unlink()
                else:
                    path.rename(root / path.name)
                    manifest = json.loads((root / "manifest.json").read_text())
                    manifest["states"][0]["file"] = path.name
                    (root / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    prior_inventory(workspace, model)

    def test_inventory_excludes_only_explicit_current_root_on_validation_recheck(self):
        from experiments.paired_demonstrations.record import prior_inventory, provenance
        from experiments.paired_demonstrations.plan import VERSION, digest
        from experiments.paired_demonstrations.media import ENCODING
        from experiments.paired_demonstrations.validate import check_sources_and_inventory
        from src.robots.franka_panda.config import SCENE_PATH
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "datasets").mkdir()
            prior = workspace / "outputs/paired_demonstrations/prior"
            self.write_paired_history(prior, model)
            current = workspace / "outputs/paired_demonstrations/current"
            before = prior_inventory(workspace, model, current_root=current)
            self.write_paired_history(current, model, offset=18)
            self.assertEqual(prior_inventory(workspace, model, current_root=current), before)
            self.assertNotEqual(prior_inventory(workspace, model), before)
            (current / "prior_inventory.json").write_text(json.dumps(before))
            manifest = {"schema": VERSION, "provenance": provenance(), "encoding": ENCODING,
                        "historical_geometry_exceptions": [],
                        "prior_workspace": str(workspace), "inventory_digest": digest(before)}
            self.assertEqual(check_sources_and_inventory(current, manifest, model), before)
            for invalid in (current.parent, workspace / "datasets", prior / "states"):
                with self.assertRaises(ValueError): prior_inventory(workspace, model, current_root=invalid)
            (prior / "manifest.json").write_text((prior / "manifest.json").read_text() + "\n")
            with self.assertRaises(ValueError): check_sources_and_inventory(current, manifest, model)

    def test_preflight_reports_missing_coverage_without_creating_outputs(self):
        from experiments.paired_demonstrations.record import preflight
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "datasets/legacy/meta").mkdir(parents=True)
            (root / "datasets/legacy/meta/info.json").write_text('{"total_episodes": 1}')
            (root / "outputs").mkdir()
            result = preflight(root)
            self.assertEqual(result["configuration_status"], "passed")
            self.assertEqual(result["collection_status"], "blocked")
            self.assertEqual(result["encoding"], {"vcodec": "libsvtav1", "pix_fmt": "yuv420p",
                                                 "crf": 30, "g": 2, "encoder_threads": 1})
            self.assertEqual(list((root / "outputs").iterdir()), [])

    def test_collect_saves_all_states_before_first_expert_and_preserves_first_failure(self):
        from experiments.paired_demonstrations.record import collect
        from experiments.paired_demonstrations import expert_v1
        from data_collection.benchmark import capture_state
        expert = expert_v1.Expert()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self.write_legacy_gap(workspace, 1, 2)
            self.write_legacy_gap(workspace, 2, 1)
            bank = workspace / "outputs/old_scenes"
            bank.mkdir(parents=True)
            path = bank / "initial.npz"
            data = mujoco.MjData(expert.model)
            np.savez(path, qpos=data.qpos.copy())
            (bank / "manifest.json").write_text(json.dumps({"scenes": [
                {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}))
            root = workspace / "outputs/paired_demonstrations/synthetic"
            def initialize(model, layout):
                data = mujoco.MjData(model)
                data.qpos[9] += (layout["ordinal"] + 1) * .001
                return {"initial_state": capture_state(model, data), "qpos": data.qpos.copy()}
            def fail(initial, episode, sink):
                self.assertEqual(len(list((root / "states").glob("*.npz"))), 18)
                frozen = json.loads((root / "manifest.json").read_text())
                self.assertEqual(frozen["status"], "recording")
                self.assertEqual(len(frozen["states"]), 18)
                raise ValueError("synthetic first failure")
            with patch.object(expert_v1, "initialize_layout", initialize), \
                 patch.object(expert_v1.Expert, "run", side_effect=fail):
                with self.assertRaisesRegex(ValueError, "synthetic first failure"):
                    collect(root, workspace, reserve_bytes=0)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            inventory = json.loads((root / "prior_inventory.json").read_text())
            self.assertEqual(manifest["historical_geometry_exceptions"], inventory["historical_geometry_exceptions"])
            self.assertEqual(len(manifest["historical_geometry_exceptions"]), 2)
            self.assertEqual(manifest["episodes"], [])
            self.assertEqual(len((root / "attempts.jsonl").read_text().splitlines()), 1)
            self.assertTrue((root / "transcripts/episode_000000.npz").exists())
            self.assertFalse((root / "dataset/replay/manifest.json").exists())
            with self.assertRaises(FileExistsError): collect(root, workspace, reserve_bytes=0)

    def test_cube_digest_excludes_robot_but_includes_all_cube_quaternions(self):
        from experiments.paired_demonstrations.record import cube_digest
        from src.robots.franka_panda.config import SCENE_PATH
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        data = mujoco.MjData(model)
        first = cube_digest(model, data.qpos)
        data.qpos[0] += .1
        self.assertEqual(first, cube_digest(model, data.qpos))
        body = model.body("target_cube_yellow").id
        address = model.jnt_qposadr[model.body_jntadr[body]]
        data.qpos[address + 4] += .01
        self.assertNotEqual(first, cube_digest(model, data.qpos))
        data.qpos[address] = np.nan
        with self.assertRaises(ValueError): cube_digest(model, data.qpos)

    def test_legacy_dataset_without_initial_geometry_is_blocked(self):
        from experiments.paired_demonstrations.record import prior_inventory, require_inventory
        from src.robots.franka_panda.config import SCENE_PATH
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = root / "datasets/old/meta/info.json"
            info.parent.mkdir(parents=True)
            info.write_text(json.dumps({"total_episodes": 2}))
            (root / "outputs").mkdir()
            inventory = prior_inventory(root, mujoco.MjModel.from_xml_path(SCENE_PATH))
            self.assertEqual(len(inventory["missing"]), 1)
            with self.assertRaises(ValueError): require_inventory(inventory)

    def test_inventory_hashes_bank_geometry_without_restoring_it(self):
        from experiments.paired_demonstrations.record import prior_inventory, require_inventory, cube_digest
        from src.robots.franka_panda.config import SCENE_PATH
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "datasets").mkdir()
            bank = root / "outputs/frozen/final_scenes"
            bank.mkdir(parents=True)
            path = bank / "final.npz"
            qpos = mujoco.MjData(model).qpos.copy()
            np.savez(path, qpos=qpos)
            (bank / "manifest.json").write_text(json.dumps({"scenes": [
                {"file": "final.npz", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}))
            inventory = prior_inventory(root, model)
            require_inventory(inventory)
            self.assertEqual(inventory["geometry_hashes"], [cube_digest(model, qpos)])
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError): prior_inventory(root, model)

    def test_unindexed_or_missing_bank_manifests_cannot_disappear_from_audit(self):
        from experiments.paired_demonstrations.record import prior_inventory, require_inventory
        from src.robots.franka_panda.config import SCENE_PATH
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "datasets").mkdir()
            bank = root / "outputs/forgotten_scenes"
            bank.mkdir(parents=True)
            np.savez(bank / "orphan.npz", qpos=mujoco.MjData(model).qpos.copy())
            inventory = prior_inventory(root, model)
            self.assertTrue(inventory["missing"])
            with self.assertRaises(ValueError): require_inventory(inventory)
            (bank / "manifest.json").write_text('{"scenes": []}')
            with self.assertRaises(ValueError): prior_inventory(root, model)

    def test_output_must_be_new_and_not_nested_in_existing_artifacts(self):
        from experiments.paired_demonstrations.record import check_output
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "outputs/paired_demonstrations").mkdir(parents=True)
            good = workspace / "outputs/paired_demonstrations/new"
            check_output(good, workspace)
            good.mkdir()
            with self.assertRaises(FileExistsError): check_output(good, workspace)
            for path in (workspace / "datasets/old/new", workspace / "src/new", workspace.parent / "elsewhere"):
                with self.assertRaises(ValueError): check_output(path, workspace)

    def test_both_directions_receive_complete_identical_independent_reset_copies(self):
        from experiments.paired_demonstrations.record import run_schedule
        from experiments.paired_demonstrations.plan import make_plan
        plan = make_plan()
        states = {l["layout_id"]: np.array([l["ordinal"], 17., 91.]) for l in plan["layouts"]}
        observed, ledger = [], []
        def execute(episode, state):
            observed.append((episode["layout_id"], state.copy()))
            state[:] = -1
            return {"success": True}
        result = run_schedule(plan, states, execute, ledger.append)
        self.assertEqual(len(result), 36)
        self.assertEqual(len(ledger), 36)
        for i in range(0, 36, 2):
            np.testing.assert_array_equal(observed[i][1], observed[i + 1][1])
            self.assertEqual(observed[i][1][1], 17.)

    def test_first_task_failure_is_logged_and_stops_without_retry(self):
        from experiments.paired_demonstrations.record import run_schedule
        from experiments.paired_demonstrations.plan import make_plan
        plan = make_plan()
        states = {l["layout_id"]: np.zeros(3) for l in plan["layouts"]}
        ledger = []
        def fail(episode, state):
            return {"success": False, "reason": "grasp"}
        with self.assertRaises(RuntimeError): run_schedule(plan, states, fail, ledger.append)
        self.assertEqual(len(ledger), 1)
        self.assertFalse(ledger[0]["success"])
        self.assertEqual(ledger[0]["episode_index"], 0)

    def test_integrity_exception_is_logged_and_stops_without_retry(self):
        from experiments.paired_demonstrations.record import run_schedule
        from experiments.paired_demonstrations.plan import make_plan
        plan = make_plan()
        states = {l["layout_id"]: np.zeros(3) for l in plan["layouts"]}
        ledger = []
        def fail(episode, state):
            raise ValueError("invalid command")
        with self.assertRaises(ValueError): run_schedule(plan, states, fail, ledger.append)
        self.assertEqual(len(ledger), 1)
        self.assertFalse(ledger[0]["success"])

    def test_duplicate_cube_geometry_is_rejected_not_resampled(self):
        from experiments.paired_demonstrations.record import reject_duplicate
        occupied = {"old"}
        reject_duplicate("fresh", occupied)
        self.assertEqual(occupied, {"old", "fresh"})
        for duplicate in ("old", "fresh"):
            with self.assertRaises(ValueError): reject_duplicate(duplicate, occupied)


class PairedMediaTests(unittest.TestCase):
    def test_pinned_lerobot_av1_lossy_roundtrip_intervals_and_task_rows(self):
        import av
        from experiments.paired_demonstrations.media import PairedDataset, FEATURES, pixels_digest
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "synthetic"
            ds = PairedDataset.create(root.name, fps=30, features=FEATURES, root=root,
                                      vcodec="libsvtav1", encoder_threads=1)
            expected = []
            for episode in range(2):
                for tick in range(3):
                    image = np.random.RandomState(episode * 3 + tick).randint(0, 256, (512, 512, 3), dtype=np.uint8)
                    expected.append(pixels_digest(image))
                    ds.add_frame({"observation.images.image": image, "observation.images.image2": image,
                                  "observation.state": np.zeros(9, np.float32), "action": np.zeros(8, np.float32),
                                  "task": f"synthetic {episode}"})
                ds.save_episode()
            ds.finalize()
            loaded = LeRobotDataset(root.name, root=root)
            loaded._ensure_hf_dataset_loaded()
            self.assertEqual(loaded.meta.total_episodes, 2)
            self.assertEqual(loaded.meta.total_frames, 6)
            for key in ("observation.images.image", "observation.images.image2"):
                info = loaded.features[key]["info"]
                self.assertEqual((info["video.codec"], info["video.pix_fmt"]), ("av1", "yuv420p"))
                from experiments.paired_demonstrations.validate import validate_camera_video
                intervals = {}
                for episode in loaded.meta.episodes:
                    prefix = f"videos/{key}"
                    path = root / loaded.meta.info["video_path"].format(
                        video_key=key, chunk_index=episode[f"{prefix}/chunk_index"],
                        file_index=episode[f"{prefix}/file_index"])
                    intervals.setdefault(path, []).append((episode[f"{prefix}/from_timestamp"],
                                                           episode[f"{prefix}/to_timestamp"], 3))
                actual = []
                count = 0
                for path in sorted((root / "videos" / key).rglob("*.mp4")):
                    count += validate_camera_video(path, intervals[path])
                    with av.open(str(path)) as video:
                        stream = video.streams.video[0]
                        self.assertEqual((stream.codec.canonical_name, stream.pix_fmt), ("av1", "yuv420p"))
                        actual.extend(pixels_digest(frame.to_ndarray(format="rgb24"))
                                      for frame in video.decode(stream))
                    for bad in (intervals[path][:-1], intervals[path] + [intervals[path][0]],
                                [(0., .1, 2)], [(1 / 30, 4 / 30, 3)]):
                        with self.assertRaises(ValueError): validate_camera_video(path, bad)
                self.assertEqual(count, 6)
                self.assertEqual(len(actual), 6)
                self.assertTrue(all(a != b for a, b in zip(actual, expected)))
            self.assertEqual([int(loaded.hf_dataset[i]["task_index"]) for i in range(6)], [0, 0, 0, 1, 1, 1])
            # Exercise the installed policy-facing decoder too, not just PyAV.
            for key in ("observation.images.image", "observation.images.image2"):
                decoded = loaded[0][key].numpy()
                self.assertEqual(decoded.shape, (3, 512, 512))
                self.assertTrue(np.isfinite(decoded).all())
                self.assertTrue(((decoded >= 0) & (decoded <= 1)).all())

    def test_corrupt_video_fails_full_decode(self):
        from experiments.paired_demonstrations.media import decode_video
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.mp4"
            path.write_bytes(b"broken")
            with self.assertRaises(Exception): decode_video(path)

    def test_full_decode_rejects_wrong_codec_pixel_format_size_fps_and_time(self):
        import av
        from fractions import Fraction
        from experiments.paired_demonstrations.media import decode_video
        cases = (("libx264", "yuv420p", 512, 30, 0),
                 ("libsvtav1", "yuv420p10le", 512, 30, 0),
                 ("libsvtav1", "yuv420p", 256, 30, 0),
                 ("libsvtav1", "yuv420p", 512, 25, 0),
                 ("libsvtav1", "yuv420p", 512, 30, 1))
        for codec, pixel_format, size, fps, offset in cases:
            with self.subTest(case=(codec, pixel_format, size, fps, offset)), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "invalid.mp4"
                options = {"preset": "12", "svtav1-params": "lp=1"} if codec == "libsvtav1" else {"threads": "1"}
                with av.open(str(path), "w") as video:
                    stream = video.add_stream(codec, rate=fps, options=options)
                    stream.width = stream.height = size
                    stream.pix_fmt = pixel_format
                    for tick in range(3):
                        frame = av.VideoFrame.from_ndarray(np.zeros((size, size, 3), np.uint8), format="rgb24")
                        frame.pts, frame.time_base = tick + offset, Fraction(1, fps)
                        video.mux(stream.encode(frame))
                    video.mux(stream.encode())
                with self.assertRaises(ValueError): decode_video(path)

    def test_dataset_rejects_non_av1_and_nonserial_encoding(self):
        from experiments.paired_demonstrations.media import PairedDataset, FEATURES
        for codec, parallel, batch in (("h264", False, 1), ("libsvtav1", True, 1),
                                       ("libsvtav1", False, 2)):
            with self.subTest(codec=codec, parallel=parallel, batch=batch), tempfile.TemporaryDirectory() as tmp:
                ds = PairedDataset.create("synthetic", fps=30, features=FEATURES, root=Path(tmp) / "synthetic",
                                          vcodec=codec, encoder_threads=1, batch_encoding_size=batch)
                image = np.zeros((512, 512, 3), np.uint8)
                ds.add_frame({"observation.images.image": image, "observation.images.image2": image,
                              "observation.state": np.zeros(9, np.float32), "action": np.zeros(8, np.float32),
                              "task": "synthetic"})
                try:
                    with self.assertRaises(ValueError): ds.save_episode(parallel_encoding=parallel)
                finally:
                    ds.finalize()


if __name__ == "__main__": unittest.main()