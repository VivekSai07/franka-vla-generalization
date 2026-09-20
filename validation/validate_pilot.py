"""Fail-closed plan, scene-bank, media, replay and alignment validation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import av
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_collection.benchmark import require_complete, runtime_versions, source_provenance, restore_timing, restore_state, write_json
from data_collection.pilot_plan import validate_plan, plan_digest
from data_collection.pilot_integrity import validate_accepted_slots, validate_expert_commands, scene_digest, validate_scene_geometry
from validation.gates import dataset_fingerprint


def decode_video(path, fps, dimensions):
    count, previous = 0, None
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        if float(stream.average_rate) != fps or (stream.width, stream.height) != dimensions:
            raise ValueError("Video FPS or dimensions mismatch")
        for frame in video.decode(stream):
            if (frame.width, frame.height) != dimensions or frame.pts is None:
                raise ValueError("Invalid decoded video frame")
            timestamp = float(frame.pts * frame.time_base)
            if previous is not None and abs(timestamp - previous - 1 / fps) > 1e-5:
                raise ValueError("Video timestamps are discontinuous")
            frame.to_ndarray(format="rgb24")  # Decode actual pixels, not just packet headers.
            previous, count = timestamp, count + 1
    if count == 0:
        raise ValueError("Empty video")
    return count


def bank_fingerprint(bank):
    digest = hashlib.sha256()
    for path in sorted(Path(bank).rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(bank)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_video_intervals(intervals, decoded_frames, fps):
    """Every persisted episode interval must cover its own frames exactly once."""
    cursor = 0.
    total = 0
    for start, end, frames in sorted(intervals):
        if abs(start - cursor) > 1e-5 or abs((end - start) * fps - frames) > 1e-3:
            raise ValueError("Duplicate, overlapping, missing or wrong-length episode video interval")
        cursor, total = end, total + frames
    if total != decoded_frames or abs(cursor * fps - decoded_frames) > 1e-3:
        raise ValueError("Video mapping does not cover decoded media")


def check_final_binding(root, replay_fingerprint):
    result = json.loads((Path(root) / "validation_result.json").read_text())
    if (result.get("status") != "passed" or result.get("dataset_fingerprint") != replay_fingerprint
            or result.get("versions") != runtime_versions()
            or result.get("source_hashes") != source_provenance(ROOT)
            or result.get("bank_fingerprint") != bank_fingerprint(Path(result["scene_bank"]))):
        raise ValueError("Pilot final validation absent, stale or failed")
    return result


def validate_bank(bank, plan, model, occupied):
    manifest = json.loads((bank / "manifest.json").read_text())
    if (manifest["status"] != "complete" or manifest["plan_digest"] != plan_digest(plan)
            or manifest["source_hashes"] != source_provenance(ROOT)
            or manifest["versions"] != runtime_versions()
            or len(manifest["scenes"]) != len(plan["scene_slots"])):
        raise ValueError("Scene bank is incomplete or has different provenance/plan")
    expected_timing = restore_timing(model, {"fps": 30, "physics_dt": manifest["timing"]["physics_dt"], "timing": manifest["timing"]})
    for slot, entry in zip(plan["scene_slots"], manifest["scenes"]):
        for key, value in slot.items():
            if entry.get(key) != value:
                raise ValueError("Scene bank slot differs from plan")
        path = bank / entry["file"]
        if path.parent != bank or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Scene artifact missing or hash mismatch")
        with np.load(path, allow_pickle=False) as state:
            data = mujoco.MjData(model)
            restore_state(model, data, state["initial_state"])
            np.testing.assert_array_equal(data.qpos, state["qpos"])
            actual = scene_digest(data.qpos)
            if actual != entry["scene_hash"] or actual in occupied:
                raise ValueError("Duplicate scene across banks/demo splits or wrong geometry hash")
            occupied.add(actual)
            validate_scene_geometry(model, data, slot["colors"])
    return dict(Counter(s["split"] for s in manifest["scenes"]))


def validate(root, bank):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from data_collection.replay_benchmark import replay
    root, bank = Path(root).resolve(), Path(bank).resolve()
    report_path = root / "validation_result.json"
    write_json(report_path, {"status": "in_progress"})
    try:
        manifest = require_complete(root)
        plan = manifest["plan"]
        validate_plan(plan)
        if plan_digest(plan) != manifest["plan_digest"]:
            raise ValueError("Recording plan hash mismatch")
        if manifest["versions"] != runtime_versions() or manifest["source_hashes"] != source_provenance(ROOT):
            raise ValueError("Recording runtime/source changed")
        validate_accepted_slots(plan, manifest["episodes"])
        splits = {key: [i for i, slot in enumerate(plan["slots"]) if slot["split"] == key] for key in plan["splits"]}
        if splits != manifest["splits"]:
            raise ValueError("Split assignment mismatch")
        ds = LeRobotDataset(root.name, root=root)
        ds._ensure_hf_dataset_loaded()
        expected_features = {"action": [8], "observation.state": [9],
                             "observation.images.image": [512, 512, 3], "observation.images.image2": [512, 512, 3]}
        for key, shape in expected_features.items():
            if list(ds.features[key]["shape"]) != shape:
                raise ValueError("Incorrect data schema")
        if (ds.meta.total_episodes != len(plan["slots"]) or ds.meta.fps != 30
                or len(ds.hf_dataset) != ds.meta.total_frames):
            raise ValueError("Dataset count or FPS mismatch")
        model = mujoco.MjModel.from_xml_path(str(ROOT / manifest["scene_path"]))
        restore_timing(model, manifest)
        from src.robots.franka_panda.config import ACTIVE_JOINTS
        from src.robots.franka_panda.config import Q_HOME
        from src.robots.franka_panda.robot import FrankaPanda
        from src.controllers.joint_target_limits import bounded_joint_target
        robot = FrankaPanda()
        expected_bounds = np.column_stack((
            np.maximum(model.actuator_ctrlrange[ACTIVE_JOINTS, 0], robot.model.lowerPositionLimit[ACTIVE_JOINTS]),
            np.minimum(model.actuator_ctrlrange[ACTIVE_JOINTS, 1], robot.model.upperPositionLimit[ACTIVE_JOINTS])))
        expected_contract = {"version": "bounded_joint_integration_v1", "joint_bounds": expected_bounds.tolist(),
                             "max_velocity": 2., "position_margin_rad": 1e-6, "saturation_recorded": True,
                             "initial_target": np.asarray(Q_HOME[:7]).tolist(),
                             "initialization_exception": "reset_controls_during_settling_then_home_target_baseline"}
        if manifest.get("expert_command_contract") != expected_contract:
            raise ValueError("Expert command contract missing or different from verified controller")
        occupied = set()
        video_intervals = {}
        offset = 0
        for entry in manifest["episodes"]:
            ep = ds.meta.episodes[entry["episode_index"]]
            start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            if start != offset or end - start != entry["frames"]:
                raise ValueError("Invalid episode index boundaries")
            offset = end
            actions = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(start, end)])
            instruction = f"pick up the {entry['pick']} cube and place it on top of the {entry['place']} cube"
            if entry.get("instruction") != instruction or ep["tasks"] != [instruction]:
                raise ValueError("Episode instruction does not match planned task")
            for i in range(start, end):
                row = ds.hf_dataset[i]
                task_index = int(row["task_index"])
                if int(row["episode_index"]) != entry["episode_index"] or int(row["frame_index"]) != i - start:
                    raise ValueError("Row episode/frame association mismatch")
                if ds.meta.tasks.iloc[task_index].name != instruction:
                    raise ValueError("Persisted frame instruction mismatch")
            for key in ds.meta.video_keys:
                prefix = f"videos/{key}"
                video_path = root / ds.meta.info["video_path"].format(
                    video_key=key, chunk_index=ep[f"{prefix}/chunk_index"], file_index=ep[f"{prefix}/file_index"])
                if root not in video_path.resolve().parents:
                    raise ValueError("Video path escapes dataset")
                video_intervals.setdefault(video_path, []).append((
                    ep[f"{prefix}/from_timestamp"], ep[f"{prefix}/to_timestamp"], end - start))
            validate_expert_commands(actions, model.actuator_ctrlrange[ACTIVE_JOINTS])
            path = root / "replay" / entry["file"]
            if path.parent != root / "replay":
                raise ValueError("Replay file escapes dataset")
            with np.load(path, allow_pickle=False) as trace:
                if any(not np.isfinite(trace[key]).all() for key in trace.files):
                    raise ValueError("Nonfinite replay trace")
                if "expert_command_contract" in manifest:
                    contract = manifest["expert_command_contract"]
                    bounds = np.asarray(contract["joint_bounds"])
                    validate_expert_commands(actions, bounds)
                    applied = trace["applied_joint_velocity"]
                    requested = trace["requested_joint_velocity"]
                    saturation = trace["command_saturation"]
                    if applied.shape != (end - start, 7) or requested.shape != applied.shape or saturation.shape != applied.shape:
                        raise ValueError("Malformed expert command trace")
                    if np.any(np.abs(applied) > contract["max_velocity"] + 1e-10):
                        raise ValueError("Applied expert velocity exceeds cap")
                    np.testing.assert_array_equal(saturation, np.abs(applied - requested) > 1e-12)
                    np.testing.assert_allclose(np.diff(trace["ctrl"][:, ACTIVE_JOINTS], axis=0),
                                               applied[1:] / 30, rtol=0, atol=1e-12)
                    if int(np.any(saturation, axis=1).sum()) != entry["saturated_ticks"]:
                        raise ValueError("Expert saturation summary mismatch")
                    np.testing.assert_array_equal(np.sum(saturation, axis=0), entry["saturated_ticks_per_joint"])
                    previous = np.asarray(contract["initial_target"])
                    for tick in range(end - start):
                        target, velocity, limited = bounded_joint_target(previous, requested[tick], 1/30, expected_bounds)
                        np.testing.assert_allclose(target, trace["ctrl"][tick, ACTIVE_JOINTS], rtol=0, atol=1e-12)
                        np.testing.assert_allclose(velocity, applied[tick], rtol=0, atol=1e-12)
                        np.testing.assert_array_equal(limited, saturation[tick])
                        previous = target
                actual = scene_digest(trace["qpos"][0])
                if actual != entry["scene_hash"] or actual in occupied:
                    raise ValueError("Duplicate or incorrect demonstration geometry")
                occupied.add(actual)
        if offset != len(ds.hf_dataset):
            raise ValueError("Frame coverage mismatch")
        bank_counts = validate_bank(bank, plan, model, occupied)
        # Check retry ledger, including rejected scenes, against immutable slot seeds.
        ledger = [json.loads(line) for line in (root / "replay/slots.jsonl").read_text().splitlines()]
        position = 0
        for slot, accepted in zip(plan["slots"], manifest["episodes"]):
            for retry in range(10):
                if position >= len(ledger):
                    raise ValueError("Missing slot attempt record")
                attempt = ledger[position]; position += 1
                if (attempt["slot_id"] != slot["slot_id"] or attempt["retry"] != retry
                        or attempt["seed"] != slot["retry_seeds"][retry] or attempt["split"] != slot["split"]):
                    raise ValueError("Slot attempt violated seed/identity schedule")
                if attempt["accepted"]:
                    if attempt["seed"] != accepted["seed"] or attempt["scene_hash"] != accepted["scene_hash"]:
                        raise ValueError("Accepted attempt does not match episode")
                    break
            else:
                raise ValueError("Exhausted slot cannot be complete")
        if position != len(ledger):
            raise ValueError("Unexpected extra attempt records")
        video_counts = {}
        for key in ds.meta.video_keys:
            files = sorted((root / "videos" / key).rglob("*.mp4"))
            count = 0
            for path in files:
                decoded = decode_video(path, 30, (512, 512))
                if path not in video_intervals:
                    raise ValueError("Unreferenced video file")
                validate_video_intervals(video_intervals.pop(path), decoded, 30)
                count += decoded
            if count != ds.meta.total_frames:
                raise ValueError("Decoded video frame count mismatch")
            video_counts[key] = count
        if set(video_counts) != {"observation.images.image", "observation.images.image2"}:
            raise ValueError("Missing camera video streams")
        if video_intervals:
            raise ValueError("Episode references missing video media")
        replay(root)
        command = [sys.executable, str(ROOT / "data_collection/verify_action_alignment.py"),
                   "--root", str(root), "--repo-id", root.name]
        alignment = subprocess.run(command, capture_output=True, text=True)
        print(alignment.stdout, flush=True)
        if alignment.returncode or "PASS: recorded actions" not in alignment.stdout:
            raise ValueError("Alignment gate failed: " + alignment.stdout + alignment.stderr)
        report = {"status": "passed", "plan_digest": plan_digest(plan),
                  "dataset_fingerprint": dataset_fingerprint(root), "versions": runtime_versions(),
                  "source_hashes": source_provenance(ROOT), "scene_bank": str(bank),
                  "bank_fingerprint": bank_fingerprint(bank), "bank_counts": bank_counts,
                  "episode_count": ds.meta.total_episodes, "frame_count": ds.meta.total_frames,
                  "split_counts": {key: len(value) for key, value in splits.items()},
                  "attempts": len(ledger), "video_frames": video_counts,
                  "alignment_output": alignment.stdout,
                  "coverage": dict(Counter(f"{e['split']}:{e['pick']}->{e['place']}:{len(e['colors'])-2}" for e in manifest["episodes"])),
                  "layout_filtering_caveat": "Accepted layouts are conditioned on expert success; scene banks are not."}
        write_json(report_path, report)
        print(f"PASS: validated {ds.meta.total_episodes} episodes and independent scene banks", flush=True)
        return report
    except Exception as exc:
        write_json(report_path, {"status": "failed", "error": str(exc), "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene-bank", type=Path, required=True)
    args = parser.parse_args()
    validate(args.root, args.scene_bank)