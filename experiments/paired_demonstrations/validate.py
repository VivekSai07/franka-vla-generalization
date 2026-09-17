"""Paired-only validation. Never writes or grants a legacy pilot/replay PASS."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

from data_collection.benchmark import (apply_action, restore_state, stack_metrics, step_control,
                                      synchronize_observation, write_json)
from data_collection.pilot_integrity import validate_expert_commands, validate_scene_geometry
from src.controllers.joint_target_limits import bounded_joint_target
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF, Q_HOME
from experiments.paired_demonstrations.plan import VERSION, complete_starts, digest, validate_plan
from experiments.paired_demonstrations.record import (ROOT, cube_digest, file_hash, safe_child,
    provenance, prior_inventory, require_inventory, reject_duplicate)
from experiments.paired_demonstrations.media import CAMERAS, FEATURES, ENCODING, decode_video

MODES = ("scripted_expert", "reference", "recorded_actions")


def validate_schedule(manifest, ledger):
    plan = manifest["plan"]
    validate_plan(plan)
    if manifest["status"] != "complete" or manifest["plan_digest"] != digest(plan):
        raise ValueError("Paired recording is incomplete or plan binding changed")
    if len(manifest["episodes"]) != 36 or len(ledger) != 36:
        raise ValueError("Exactly 36 episodes and 36 first-attempt successes required")
    for expected, entry, attempt in zip(plan["episodes"], manifest["episodes"], ledger):
        for value in (entry, attempt):
            if digest({k: value.get(k) for k in expected}) != digest(expected):
                raise ValueError("Episode/attempt task, seed, split or order differs from plan")
            if value.get("success") is not True or value.get("frames") != 528:
                raise ValueError("Failed or wrong-length episode cannot complete this batch")
        if entry != attempt:
            raise ValueError("Manifest and attempt transcript differ")


def same_reset(frozen, trace):
    np.testing.assert_array_equal(trace["initial_state"], frozen["initial_state"])
    np.testing.assert_array_equal(trace["qpos"][0], frozen["qpos"])


def validate_trace(trace, actions, states, timestamps, bounds):
    n = len(actions)
    if actions.shape != (n, 8) or actions.dtype != np.float32 or states.shape != (n, 9) or states.dtype != np.float32:
        raise ValueError("Canonical float32 unpadded 8-action/9-state schema required")
    shapes = {"qpos": (n, 37), "ctrl": (n, 8), "sim_time": (n,), "substeps": (n,),
              "requested_joint_velocity": (n, 7), "applied_joint_velocity": (n, 7),
              "command_saturation": (n, 7), "released_stack": (n,), "image_time": (n,),
              "image_sha256": (n, 2), "phase_index": (n,), "phase_time": (n,), "target_pos": (n, 3)}
    for key, shape in shapes.items():
        if trace[key].shape != shape:
            raise ValueError(f"Malformed transcript: {key}")
    for key, value in trace.items():
        if key != "image_sha256" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite transcript: {key}")
    for key in ("body_xpos", "camera_xpos", "camera_xmat"):
        if trace[key].ndim != 3 or len(trace[key]) != n:
            raise ValueError("Malformed observed geometry")
    if not np.isfinite(states).all() or not np.isfinite(timestamps).all() or timestamps.shape != (n,):
        raise ValueError("Nonfinite/malformed persisted observation")
    validate_expert_commands(actions, bounds)
    np.testing.assert_array_equal(states, trace["qpos"][:, :9].astype(np.float32))
    np.testing.assert_array_equal(actions, trace["ctrl"].astype(np.float32))
    np.testing.assert_array_equal(trace["substeps"], np.full(n, 20))
    np.testing.assert_array_equal(trace["image_time"], trace["sim_time"])
    np.testing.assert_allclose(trace["sim_time"] - trace["sim_time"][0], np.arange(n) / 30, rtol=0, atol=1e-9)
    np.testing.assert_allclose(timestamps, np.arange(n) / 30, rtol=0, atol=2e-6)
    previous = np.asarray(Q_HOME[:7]).copy()
    for i in range(n):
        target, velocity, saturation = bounded_joint_target(
            previous, trace["requested_joint_velocity"][i], 1 / 30, bounds)
        np.testing.assert_allclose(target, trace["ctrl"][i, ACTIVE_JOINTS], rtol=0, atol=1e-12)
        np.testing.assert_allclose(velocity, trace["applied_joint_velocity"][i], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(saturation, trace["command_saturation"][i])
        previous = target


def check_replay_coverage(results):
    expected = Counter((i, mode) for i in range(36) for mode in MODES)
    if (Counter((r["episode"], r["mode"]) for r in results) != expected
            or any(r.get("passed") is not True for r in results)):
        raise ValueError("All 36 episodes must pass each paired replay mode exactly once")


def fingerprint(root):
    root = Path(root)
    files = [root / name for name in ("manifest.json", "plan.json", "prior_inventory.json", "attempts.jsonl")]
    for directory in ("states", "transcripts", "dataset/meta", "dataset/data", "dataset/videos"):
        files.extend(p for p in (root / directory).rglob("*") if p.is_file())
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(files)})


def check_sources_and_inventory(root, manifest, model):
    if manifest.get("schema") != VERSION or manifest["provenance"] != provenance():
        raise ValueError("Paired sources, runtime or schema changed")
    if manifest["encoding"] != ENCODING:
        raise ValueError("AV1/yuv420p encoding contract changed")
    saved = json.loads((root / "prior_inventory.json").read_text())
    require_inventory(saved)
    if digest(saved) != manifest["inventory_digest"]:
        raise ValueError("Prior inventory binding changed")
    if digest(manifest.get("historical_geometry_exceptions")) != digest(saved["historical_geometry_exceptions"]):
        raise ValueError("Manifest historical geometry exceptions changed")
    actual = prior_inventory(Path(manifest["prior_workspace"]), model, current_root=root)
    require_inventory(actual)
    if actual != saved:
        raise ValueError("Historical geometry coverage/files changed")
    return saved


def load_states(root, manifest, model, inventory):
    plan = manifest["plan"]
    if len(manifest["states"]) != 18:
        raise ValueError("Missing preregistered layouts")
    occupied, frozen = set(inventory["geometry_hashes"]), {}
    files = set()
    for expected, entry in zip(plan["layouts"], manifest["states"]):
        if digest({k: entry.get(k) for k in expected}) != digest(expected):
            raise ValueError("Frozen state schedule differs from plan")
        path = safe_child(root, entry["file"])
        if path.parent != root / "states" or file_hash(path) != entry["sha256"]:
            raise ValueError("State artifact path/hash mismatch")
        files.add(path)
        with np.load(path, allow_pickle=False) as loaded:
            arrays = dict(loaded)
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Nonfinite frozen reset")
        data = mujoco.MjData(model)
        restore_state(model, data, arrays["initial_state"])
        for key, actual in (("qpos", data.qpos), ("initial_ctrl", data.ctrl), ("body_xpos", data.xpos),
                            ("camera_xpos", data.cam_xpos), ("camera_xmat", data.cam_xmat)):
            np.testing.assert_array_equal(arrays[key], actual)
        validate_scene_geometry(model, data, entry["colors"])
        geometry = cube_digest(model, data.qpos)
        if geometry != entry["cube_geometry_sha256"]:
            raise ValueError("Cube geometry binding mismatch")
        reject_duplicate(geometry, occupied)
        frozen[entry["layout_id"]] = arrays
    if files != set((root / "states").glob("*.npz")):
        raise ValueError("Unexpected or duplicated state artifacts")
    return frozen


def replay_episode(expert, initial, entry, trace, actions):
    """True seeded expert reference PLUS separate float64/float32 control replays."""
    count = 0

    def compare(frame, actual):
        nonlocal count
        if count >= entry["frames"]:
            raise ValueError("Regenerated expert emitted extra frames")
        for key, value in actual.items():
            # Same runtime/seed/full reset must recreate the true expert exactly,
            # including IK requests, jitter targets, phases and observed RGB.
            np.testing.assert_array_equal(value, trace[key][count], err_msg=f"expert {key} tick {count}")
        np.testing.assert_array_equal(frame["action"], actions[count])
        count += 1

    outcome = expert.run(initial, entry, compare)
    if count != entry["frames"] or outcome["success"] is not True or outcome["stable_ticks"] != entry["stable_ticks"]:
        raise ValueError("True scripted expert failed to reproduce the episode")
    np.testing.assert_array_equal(outcome["final_qpos"], trace["final_qpos"])
    np.testing.assert_array_equal(outcome["final_state"], trace["final_state"])
    results = [{"episode": entry["episode_index"], "mode": "scripted_expert", "passed": True,
                "frames": count, "stable_ticks": outcome["stable_ticks"]}]
    for mode in ("reference", "recorded_actions"):
        model = expert.model
        data = mujoco.MjData(model)
        restore_state(model, data, initial)
        max_policy, max_full, stable = 0., 0., 0
        for tick in range(len(actions)):
            synchronize_observation(model, data)
            max_policy = max(max_policy, float(np.max(np.abs(data.qpos[:9] - trace["qpos"][tick, :9]))))
            max_full = max(max_full, float(np.max(np.abs(data.qpos - trace["qpos"][tick]))))
            if abs(data.time - trace["sim_time"][tick]) > 1e-9:
                raise ValueError("Replay clock drift")
            if mode == "reference":
                np.testing.assert_allclose(data.xpos, trace["body_xpos"][tick], rtol=0, atol=1e-9)
                np.testing.assert_allclose(data.cam_xpos, trace["camera_xpos"][tick], rtol=0, atol=1e-9)
                np.testing.assert_allclose(data.cam_xmat, trace["camera_xmat"][tick], rtol=0, atol=1e-9)
                data.ctrl[:] = trace["ctrl"][tick]
            else:
                apply_action(model, data, actions[tick], ACTIVE_JOINTS)
            if step_control(model, data, 1 / 30, ARM_DOF) != trace["substeps"][tick]:
                raise ValueError("Replay substeps changed")
            metrics = stack_metrics(model, data, entry["pick"], entry["place"])
            stable = stable + 1 if metrics["released_stack"] else 0
            if mode == "reference" and metrics["released_stack"] != bool(trace["released_stack"][tick]):
                raise ValueError("Reference success transcript mismatch")
        max_policy = max(max_policy, float(np.max(np.abs(data.qpos[:9] - trace["final_qpos"][:9]))))
        max_full = max(max_full, float(np.max(np.abs(data.qpos - trace["final_qpos"]))))
        error, tolerance = (max_full, 1e-5) if mode == "reference" else (max_policy, 1e-3)
        if error > tolerance or stable < 30:
            raise ValueError(f"{mode} drift or released-stack failure: error={error}, stable={stable}")
        results.append({"episode": entry["episode_index"], "mode": mode, "passed": True,
                        "max_policy_state_error": max_policy, "max_full_qpos_error": max_full,
                        "tolerance": tolerance, "stable_ticks": stable, "final_metrics": metrics})
    return results


def validate_camera_video(path, intervals):
    """Full decode and episode coverage; raw hash equality belongs to expert replay only."""
    from validation.validate_pilot import validate_video_intervals
    count = decode_video(path)
    validate_video_intervals(intervals, count, 30)
    return count


def validate(root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from experiments.paired_demonstrations.expert_v1 import Expert
    root = Path(root).resolve()
    if root.parent != ROOT / "outputs/paired_demonstrations" or not root.is_dir():
        raise ValueError("Validator accepts only a new paired root, never old dataset roots")
    report_path = root / "paired_validation.json"
    write_json(report_path, {"status": "in_progress", "training_ready": False})
    exceptions, inventory_rechecked = [], False
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        exceptions = manifest.get("historical_geometry_exceptions", [])
        ledger = [json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()]
        validate_schedule(manifest, ledger)
        if json.loads((root / "plan.json").read_text()) != manifest["plan"]:
            raise ValueError("Preregistered plan differs from manifest")
        expert = Expert()
        if manifest["configuration"] != expert.config:
            raise ValueError("Physical/camera/governor configuration changed")
        inventory = check_sources_and_inventory(root, manifest, expert.model)
        exceptions, inventory_rechecked = inventory["historical_geometry_exceptions"], True
        before = fingerprint(root)
        frozen = load_states(root, manifest, expert.model, inventory)
        dsroot = root / "dataset"
        ds = LeRobotDataset(dsroot.name, root=dsroot)
        ds._ensure_hf_dataset_loaded()
        if (ds.meta.total_episodes != 36 or ds.meta.total_frames != 19008
                or len(ds.hf_dataset) != 19008 or ds.meta.fps != 30):
            raise ValueError("Paired dataset requires exactly 36 x 528 rows at 30 Hz")
        if set(ds.meta.video_keys) != set(CAMERAS):
            raise ValueError("Wrong camera streams")
        for key, wanted in FEATURES.items():
            for field in ("dtype", "shape", "names"):
                if digest(ds.features[key][field]) != digest(wanted[field]):
                    raise ValueError(f"Wrong canonical schema: {key}/{field}")
        expected_features = set(FEATURES) | {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        if set(ds.features) != expected_features:
            raise ValueError("Unexpected model inputs/padding/oracle fields")
        if len(ds.meta.tasks) != 12:
            raise ValueError("Exactly twelve ordered tasks required")
        offset, results, intervals, trace_files = 0, [], {}, set()
        state_entries = {e["layout_id"]: e for e in manifest["states"]}
        for entry in manifest["episodes"]:
            index = entry["episode_index"]
            ep = ds.meta.episodes[index]
            start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            if start != offset or end - start != 528 or ep["tasks"] != [entry["instruction"]]:
                raise ValueError("Episode coverage or task mismatch")
            offset = end
            rows = [ds.hf_dataset[i] for i in range(start, end)]
            for tick, row in enumerate(rows):
                task = int(row["task_index"])
                if (int(row["index"]) != start + tick or int(row["episode_index"]) != index
                        or int(row["frame_index"]) != tick or task != entry["task_index"]
                        or ds.meta.tasks.iloc[task].name != entry["instruction"]):
                    raise ValueError("Persisted row/task ID or boundary mismatch")
            actions = np.stack([r["action"].numpy() for r in rows])
            states = np.stack([r["observation.state"].numpy() for r in rows])
            times = np.array([r["timestamp"].item() for r in rows])
            # Complete-only interface has exactly 479 valid starts, all 50 indices
            # in this episode. No temporal padding or 24 dummy dimensions on disk.
            starts = complete_starts(end - start)
            if len(starts) != 479 or any(s + 50 > len(actions) for s in starts):
                raise ValueError("Complete-only chunk boundary failure")
            path = safe_child(root, entry["file"])
            if path.parent != root / "transcripts" or file_hash(path) != entry["sha256"]:
                raise ValueError("Transcript path/hash mismatch")
            trace_files.add(path)
            with np.load(path, allow_pickle=False) as loaded:
                trace = dict(loaded)
            frozen_entry = state_entries[entry["layout_id"]]
            for key, state_key in (("state_file", "file"), ("state_sha256", "sha256"),
                                   ("cube_geometry_sha256", "cube_geometry_sha256")):
                if entry[key] != frozen_entry[state_key]:
                    raise ValueError("Direction state binding mismatch")
            same_reset(frozen[entry["layout_id"]], trace)
            validate_trace(trace, actions, states, times, expert.bounds)
            for key in CAMERAS:
                prefix = f"videos/{key}"
                path = safe_child(dsroot, ds.meta.info["video_path"].format(
                    video_key=key, chunk_index=ep[f"{prefix}/chunk_index"], file_index=ep[f"{prefix}/file_index"]))
                intervals.setdefault(path, []).append((ep[f"{prefix}/from_timestamp"], ep[f"{prefix}/to_timestamp"], 528))
            results.extend(replay_episode(expert, frozen[entry["layout_id"]]["initial_state"], entry, trace, actions))
            print(f"Paired replay {index + 1}/36: all three modes passed", flush=True)
        if trace_files != set((root / "transcripts").glob("*.npz")):
            raise ValueError("Missing, duplicated or extra transcripts")
        videos = set((dsroot / "videos").rglob("*.mp4"))
        if videos != set(intervals):
            raise ValueError("Unreferenced or missing video files")
        video_frames = 0
        for path in sorted(videos):
            video_frames += validate_camera_video(path, intervals[path])
        if video_frames != 38016:
            raise ValueError("Expected 38,016 decoded camera frames")
        check_replay_coverage(results)
        alignment = subprocess.run([sys.executable, str(ROOT / "data_collection/verify_action_alignment.py"),
                                    "--root", str(dsroot), "--repo-id", dsroot.name],
                                   capture_output=True, text=True)
        if alignment.returncode or "PASS: recorded actions" not in alignment.stdout:
            raise ValueError("Unchanged alignment gate failed: " + alignment.stdout + alignment.stderr)
        check_sources_and_inventory(root, manifest, expert.model)
        after = fingerprint(root)
        if after != before:
            raise ValueError("Data/state artifacts changed during validation")
        report = {"status": "passed", "training_ready": True, "schema": VERSION,
                  "historical_geometry_exceptions": exceptions,
                  "historical_inventory_rechecked": inventory_rechecked,
                  "fingerprint": after, "provenance": provenance(), "episode_count": 36,
                  "frame_count": 19008, "decoded_camera_frames": video_frames,
                  "layout_counts": {"train": 12, "heldout": 6}, "episode_counts": {"train": 24, "heldout": 12},
                  "complete_starts_per_episode": 479, "temporal_padding": False,
                  "stock_model_dummy_action_dimensions": 24, "results": results,
                  "alignment_output": alignment.stdout, "alignment_returncode": alignment.returncode,
                  "legacy_pilot_gates_claimed": False}
        write_json(report_path, report)
        return report
    except Exception as exc:
        write_json(report_path, {"status": "failed", "training_ready": False,
                                "historical_geometry_exceptions": exceptions,
                                "historical_inventory_rechecked": inventory_rechecked,
                                "error": str(exc), "error_type": type(exc).__name__})
        raise


def require_paired_pass(root):
    from experiments.paired_demonstrations.expert_v1 import Expert
    root = Path(root).resolve()
    result = json.loads((root / "paired_validation.json").read_text())
    if (result.get("status") != "passed" or result.get("training_ready") is not True
            or result.get("schema") != VERSION or result.get("episode_count") != 36
            or result.get("provenance") != provenance() or result.get("fingerprint") != fingerprint(root)
            or result.get("alignment_returncode") != 0 or "PASS: recorded actions" not in result.get("alignment_output", "")):
        raise ValueError("Paired validation absent, incomplete, failed or stale")
    check_replay_coverage(result["results"])
    manifest = json.loads((root / "manifest.json").read_text())
    validate_schedule(manifest, [json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()])
    inventory = check_sources_and_inventory(root, manifest, Expert().model)
    if digest(result.get("historical_geometry_exceptions")) != digest(inventory["historical_geometry_exceptions"]):
        raise ValueError("Validation historical geometry exceptions changed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(validate(parser.parse_args().root), indent=2))