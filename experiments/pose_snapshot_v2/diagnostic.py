"""One fixed-input diagnostic, not a collector. No retry or alternative layout.

Default is preflight only. --execute requires passing saved CPU test evidence;
the persistent exclusive marker is acquired before the one actual Expert.run.
Saved-control replays do not invoke an expert. All output files are exclusive.
"""
import argparse
import ast
import difflib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time
import traceback

import mujoco
import numpy as np
import pinocchio as pin
from PIL import Image, ImageDraw

from data_collection.benchmark import (
    capture_state, restore_state, synchronize_observation, step_control, stack_metrics,
)
from experiments.paired_demonstrations import expert_v1
from experiments.paired_demonstrations.record import file_hash, provenance
from experiments.pose_snapshot_v2.expert import Expert, EXPERT_VERSION
from src.controllers.grasp_controller import GraspController
from src.controllers.joint_target_limits import bounded_joint_target
from src.robots.franka_panda.config import Q_HOME, ACTIVE_JOINTS, ARM_DOF

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "outputs/paired_demonstrations/paired_20260917"
OUT = ROOT / "outputs/diagnostics/pose_snapshot_v2_20260917"
IMAGE = "sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc"
TRACE_HASH = "4d66200d0e132307a12947cc4afdb44465959c3a5065686870137e6089a24a26"
STATE_HASH = "c487a146d5cd373e6fddcf2d00262e1102224015d62309a98f115aafcf35c296"


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, default=serial, allow_nan=False)
        stream.write("\n")


def save_npz(path, arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def check_integrity():
    before = json.loads((OUT / "hashes_before.json").read_text())
    after = {"source": {p: file_hash(ROOT / p) for p in before["source"]},
             "paired_artifacts": {str(p.relative_to(ROOT)): file_hash(p)
                 for p in sorted(ART.rglob("*")) if p.is_file()}}
    for key in ("source", "paired_artifacts"):
        if before[key] != after[key]:
            raise ValueError(f"Protected {key} changed; stop")
    return after


def loop_proof():
    old_source = textwrap.dedent(inspect.getsource(expert_v1.Expert.run))
    new_source = textwrap.dedent(inspect.getsource(Expert.run))
    old, new = ast.parse(old_source), ast.parse(new_source)
    replacements = 0
    for node in ast.walk(new):
        if isinstance(node, ast.Name) and node.id == "SnapshotTaskSpaceTrajectory":
            node.id = "TaskSpaceTrajectory"
            replacements += 1
    if replacements != 1 or ast.dump(new) != ast.dump(old):
        raise ValueError("Expert loop changed beyond the single constructor injection")
    diff = "".join(difflib.unified_diff(old_source.splitlines(True), new_source.splitlines(True),
        fromfile="experiments/paired_demonstrations/expert_v1.py:Expert.run",
        tofile="experiments/pose_snapshot_v2/expert.py:Expert.run"))
    return {"normalized_ast_equal": True, "constructor_replacements": replacements,
            "old_expert_sha256": file_hash(Path(expert_v1.__file__)),
            "new_sources": {str(p.relative_to(ROOT)): file_hash(p)
                            for p in sorted(Path(__file__).parent.glob("*.py"))}}, diff


def preflight():
    check_integrity()
    if os.environ.get("DIAGNOSTIC_IMAGE_ID") != IMAGE:
        raise ValueError("Exact pinned image identity required")
    if os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("GPU visibility must be disabled")
    if list(Path("/dev").glob("nvidia*")):
        raise ValueError("Unexpected GPU device")
    attempts = [json.loads(line) for line in (ART / "attempts.jsonl").read_text().splitlines()]
    if len(attempts) != 5:
        raise ValueError("Expected preserved four successes and exact fifth failure")
    entry = attempts[4]
    expected = {"episode_index": 4, "episode_id": "train/red-blue/0/forward",
        "layout_id": "train/red-blue/0", "seed": 488840035, "pick": "red", "place": "blue",
        "state_file": "states/layout_02.npz", "file": "transcripts/episode_000004.npz",
        "sha256": TRACE_HASH, "state_sha256": STATE_HASH, "success": False, "frames": 528}
    if any(entry.get(k) != v for k, v in expected.items()):
        raise ValueError("Only the exact authorized failed episode is permitted")
    if file_hash(ART / entry["file"]) != TRACE_HASH or file_hash(ART / entry["state_file"]) != STATE_HASH:
        raise ValueError("Saved episode or frozen state hash mismatch")
    manifest = json.loads((ART / "manifest.json").read_text())
    if provenance() != manifest["provenance"]:
        raise ValueError("Original source/runtime provenance changed")
    expert = Expert()
    if expert.config != dict(manifest["configuration"], expert_version=EXPERT_VERSION):
        raise ValueError("Configuration differs beyond explicit expert version")
    initial = load_npz(ART / entry["state_file"])
    trace = load_npz(ART / entry["file"])
    np.testing.assert_array_equal(initial["initial_state"], trace["initial_state"])
    np.testing.assert_array_equal(initial["qpos"], trace["qpos"][0])
    data = mujoco.MjData(expert.model)
    restore_state(expert.model, data, initial["initial_state"])
    np.testing.assert_array_equal(capture_state(expert.model, data), initial["initial_state"])
    np.testing.assert_array_equal(data.qpos, initial["qpos"])
    np.testing.assert_array_equal(data.ctrl, initial["initial_ctrl"])
    np.testing.assert_array_equal(data.xpos, initial["body_xpos"])
    loop_proof()
    return expert, entry, initial, trace


class TraceSink:
    """Persist each tick before optional media work; never inspect live robot FK."""

    def __init__(self, instruction):
        self.instruction = instruction
        self.rows, self.samples = [], {}
        self.previous = None
        (OUT / "ticks").mkdir(exist_ok=False)

    def __call__(self, frame, trace):
        tick = len(self.rows)
        if tick >= 528:
            raise RuntimeError("528-tick diagnostic cap exceeded; no continuation")
        if frame["task"] != self.instruction:
            raise ValueError("Instruction changed")
        row = {key: np.asarray(value).copy() for key, value in trace.items()}
        row.update(action=frame["action"].copy(), observation_state=frame["observation.state"].copy())
        save_npz(OUT / "ticks" / f"tick_{tick:06d}.npz", row)
        self.rows.append(row)
        sample = {key: frame[key].copy() for key in expert_v1.CAMERAS}
        phase = int(trace["phase_index"])
        if self.previous is None or phase != self.previous[1]:
            if self.previous is not None:
                self.samples[self.previous[0]] = self.previous[2]
            self.samples[tick] = sample
        if tick % 30 == 0:
            self.samples[tick] = sample
        self.previous = (tick, phase, sample)

    def finish(self, initial, result):
        arrays = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]} if self.rows else {}
        arrays["initial_state"] = initial.copy()
        if result is not None:
            arrays.update(final_qpos=result["final_qpos"], final_state=result["final_state"])
        save_npz(OUT / "corrected_trace.npz", arrays)
        if self.previous is not None:
            self.samples[self.previous[0]] = self.previous[2]
        for key, camera in expert_v1.CAMERAS.items():
            count = len(self.samples)
            if not count:
                continue
            sheet = Image.new("RGB", (4 * 512, ((count + 3) // 4) * 540), "white")
            draw = ImageDraw.Draw(sheet)
            for i, (tick, images) in enumerate(sorted(self.samples.items())):
                x, y = (i % 4) * 512, (i // 4) * 540
                sheet.paste(Image.fromarray(images[key]), (x, y))
                draw.text((x + 4, y + 515), f"{camera} tick {tick} phase {int(self.rows[tick]['phase_index'])}", fill="black")
            with (OUT / f"{camera}_contact_sheet.png").open("xb") as stream:
                sheet.save(stream, format="PNG")
        save_json("camera_samples.json", {"ticks": sorted(self.samples), "size": [512, 512],
            "source": "unaltered pre-step RGB from the single actual expert invocation",
            "selection": "every 30 ticks plus phase starts/ends and last available tick"})
        return arrays


def replay_and_measure(model, bounds, initial, trace, result):
    """Replay float64 saved controls once; quantify without invoking an expert."""
    data = mujoco.MjData(model)
    restore_state(model, data, initial)
    grasp = GraspController(model, data, target="target_cube_red")
    red, blue, hand = [model.body(name).id for name in ("target_cube_red", "target_cube_blue", "hand")]
    states = expert_v1.generate_task_states(.105, robot_speed=1.)
    # Separate FK model is used only in offline measurement, not the expert loop.
    robot = expert_v1.FrankaPanda()
    rotation = robot.forward_kinematics(np.asarray(Q_HOME)).rotation.copy()
    target = np.asarray(Q_HOME[:7]).copy()
    rows, stable, peak_stable = [], 0, 0
    n = len(trace["qpos"])
    if not 0 < n <= 528:
        raise ValueError("Invalid trace length")
    for key, value in trace.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite trace field: {key}")
    for tick in range(n):
        synchronize_observation(model, data)
        np.testing.assert_array_equal(data.qpos, trace["qpos"][tick])
        np.testing.assert_array_equal(data.xpos, trace["body_xpos"][tick])
        np.testing.assert_array_equal(data.cam_xpos, trace["camera_xpos"][tick])
        np.testing.assert_array_equal(data.cam_xmat, trace["camera_xmat"][tick])
        if data.time != trace["sim_time"][tick] or data.time != trace["image_time"][tick]:
            raise ValueError("Observation time mismatch")
        ctrl = trace["ctrl"][tick]
        target, applied, saturated = bounded_joint_target(
            target, trace["requested_joint_velocity"][tick], 1 / 30, bounds)
        np.testing.assert_array_equal(target, ctrl[:7])
        np.testing.assert_array_equal(applied, trace["applied_joint_velocity"][tick])
        np.testing.assert_array_equal(saturated, trace["command_saturation"][tick])
        if np.any(ctrl[:7] < bounds[:, 0]) or np.any(ctrl[:7] > bounds[:, 1]):
            raise ValueError("Out-of-bounds command")
        if np.max(np.abs(trace["requested_joint_velocity"][tick])) > 2.:
            raise ValueError("Requested velocity cap exceeded")
        if "action" in trace:
            np.testing.assert_array_equal(trace["action"][tick], ctrl.astype(np.float32))
            np.testing.assert_array_equal(trace["observation_state"][tick], data.qpos[:9].astype(np.float32))
        delta = data.xpos[red] - data.xpos[blue]
        row = {"tick": tick, "phase_index": int(trace["phase_index"][tick]),
            "bilateral_contact": grasp.is_grasped(), "red_z_m": float(data.xpos[red, 2]),
            "xy_m": float(np.linalg.norm(delta[:2])), "z_m": float(delta[2]),
            "hand_endpoint_error_m": float(np.linalg.norm(data.xpos[hand] - trace["target_pos"][tick])),
            "hand_rotation_error_rad": float(np.linalg.norm(pin.log3(
                data.xmat[hand].reshape(3, 3).T @ rotation))),
            "saturated_joints": np.flatnonzero(saturated).tolist()}
        data.ctrl[:] = ctrl
        substeps = int(trace["substeps"][tick])
        if substeps == 0:
            if tick != n - 1 or result["reason"] != "grasp" or states[row["phase_index"]]["name"] != "Verify Lift":
                raise ValueError("Only original failed-lift terminal tick may have zero substeps")
        elif substeps != 20 or step_control(model, data, 1 / 30, ARM_DOF) != 20:
            raise ValueError("Expected exactly 20 physics substeps")
        metrics = stack_metrics(model, data, "red", "blue")
        released = False if substeps == 0 else metrics["released_stack"]
        if released != bool(trace["released_stack"][tick]):
            raise ValueError("Stack scoring mismatch")
        if substeps:
            stable = stable + 1 if released else 0
            peak_stable = max(peak_stable, stable)
        rows.append(row)
    np.testing.assert_array_equal(data.qpos, trace["final_qpos"])
    np.testing.assert_array_equal(capture_state(model, data), trace["final_state"])
    final_metrics = stack_metrics(model, data, "red", "blue")
    if result["frames"] != n or result["stable_ticks"] != stable or result["final_metrics"] != final_metrics:
        raise ValueError("Final result differs from independent saved-control replay")
    success = n == 528 and int(trace["phase_index"][-1]) == 8 and stable >= 30
    if result["success"] != success:
        raise ValueError("Success criterion mismatch")
    phases = []
    for phase in sorted(set(int(v) for v in trace["phase_index"])):
        subset = [row for row in rows if row["phase_index"] == phase]
        phases.append({"name": states[phase]["name"], "first_tick": subset[0]["tick"],
            "last_tick": subset[-1]["tick"], "ticks": len(subset),
            "bilateral_contact_ticks": sum(r["bilateral_contact"] for r in subset),
            "saturation_ticks": sum(bool(r["saturated_joints"]) for r in subset),
            "last_tick_metrics": subset[-1]})
    contact = [r["tick"] for r in rows if r["bilateral_contact"]]
    saturation = [r["tick"] for r in rows if r["saturated_joints"]]
    summary = {"frames": n, "success": success, "reason": result["reason"],
        "stable_ticks": stable, "peak_stable_ticks": peak_stable, "final_metrics": final_metrics,
        "max_lift_m": max(r["red_z_m"] for r in rows) - rows[0]["red_z_m"],
        "first_bilateral_contact_tick": contact[0] if contact else None,
        "bilateral_contact_ticks": len(contact), "saturation_ticks": len(saturation),
        "first_saturation_tick": saturation[0] if saturation else None,
        "saturation_per_joint": np.sum(trace["command_saturation"], axis=0),
        "controller_clip_ticks": int(np.count_nonzero(np.any(np.abs(trace["requested_joint_velocity"]) == 2., axis=1))),
        "substeps_counts": {str(v): int(np.count_nonzero(trace["substeps"] == v)) for v in np.unique(trace["substeps"])},
        "saved_control_replay_qpos_max_error": 0., "final_integration_state_exact": True,
        "finite_bounds_governor_alignment_scoring_passed": True, "phases": phases}
    return summary, rows


def execute():
    expert, entry, initial, original_trace = preflight()
    tests = json.loads((OUT / "tests_passed.json").read_text())
    if not (tests["existing"]["passed"] and tests["existing"]["count"] == 129
            and tests["new"]["passed"] and tests["new"]["count"] >= 8):
        raise ValueError("Passing existing 129 and new CPU tests required")
    if tests["source_proof"] != loop_proof()[0]:
        raise ValueError("Implementation changed since passing CPU tests")
    # Claim the entire execution before any replay; an interrupted attempt cannot resume.
    save_json("execution_started.json", {"episode": entry, "expert_invocation_budget": 1,
        "image": IMAGE, "version": EXPERT_VERSION, "start_unix": time.time()})
    proof, diff = loop_proof()
    save_json("loop_equivalence.json", proof)
    with (OUT / "expert_run.diff").open("x") as stream:
        stream.write(diff)
    save_json("configuration.json", expert.config)
    result, sink, exit_code = None, None, 2
    try:
        original, rows = replay_and_measure(expert.model, expert.bounds, initial["initial_state"], original_trace, entry)
        save_json("original_summary.json", original)
        save_json("original_measurements.json", rows)
        sink = TraceSink(entry["instruction"])
        save_json("expert_invocation_started.json", {"count": 1, "seed": entry["seed"]})
        started = time.monotonic()
        # The only actual expert invocation in this command. Never loop or retry.
        result = expert.run(initial["initial_state"].copy(), entry, sink)
        save_json("corrected_result.json", {**result, "wall_seconds": time.monotonic() - started})
        corrected_trace = sink.finish(initial["initial_state"], result)
        sink = None
        corrected, rows = replay_and_measure(expert.model, expert.bounds, initial["initial_state"], corrected_trace, result)
        save_json("corrected_summary.json", corrected)
        save_json("corrected_measurements.json", rows)
        save_json("comparison.json", {"original": original, "corrected": corrected,
            "scope": "endpoint copies only; one actual corrected expert invocation",
            "task_fixed": corrected["success"], "training_ready": False})
        exit_code = 0 if result["success"] else 1
        print(json.dumps({"corrected_success": result["success"], "reason": result["reason"],
                          "frames": result["frames"], "exit_code": exit_code}), flush=True)
    except Exception as exc:
        save_json("execution_error.json", {"type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(), "no_retry": True})
        if sink is not None and not (OUT / "corrected_trace.npz").exists():
            sink.finish(initial["initial_state"], result)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 2
    finally:
        after = check_integrity()
        save_json("hashes_after.json", after)
        save_json("execution_exit.json", {"exit_code": exit_code,
            "actual_expert_invocations": int((OUT / "expert_invocation_started.json").exists()),
            "source_files_unchanged": len(after["source"]),
            "paired_artifacts_unchanged": len(after["paired_artifacts"]), "no_retry": True})
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the exact one-invocation budget")
    args = parser.parse_args()
    if args.execute:
        return execute()
    _, entry, _, _ = preflight()
    print(json.dumps({"preflight": "passed", "episode_id": entry["episode_id"],
                      "seed": entry["seed"], "actual_expert_invocations": 0}))
    return 0


if __name__ == "__main__":
    sys.exit(main())