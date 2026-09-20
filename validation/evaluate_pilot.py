"""Offline selected-checkpoint evaluation on the fixed tuning bank; never final."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import sys
import time

import av
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_collection.pilot_plan import plan_digest, validate_plan
from data_collection.pilot_integrity import scene_digest, validate_scene_geometry
from data_collection.benchmark import (apply_action, capture_state, require_complete, restore_state,
    restore_timing, runtime_versions, source_provenance, step_control, synchronize_observation, write_json)
from training.smolvla_pilot import file_hash, training_source_hashes
from validation.gates import fresh_stack_metrics, require_replay_pass
from validation.precision import load_fp32_checkpoint
from validation.rollout_diagnostics import bound_action, target_diagnostics
from validation.validate_pilot import bank_fingerprint
from src.controllers.grasp_controller import GraspController
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF

SEED = 20260916


def select_tuning(plan, bank, occupied=()):
    validate_plan(plan)
    scenes = bank["scenes"]
    if (plan["mode"] != "pilot" or bank["status"] != "complete"
            or bank["plan_digest"] != plan_digest(plan) or len(scenes) != 360
            or Counter(s["split"] for s in scenes) != {"tuning": 120, "final": 240}):
        raise ValueError("Expected complete canonical pilot tuning/final bank")
    seen, files = set(occupied), set()
    for slot, entry in zip(plan["scene_slots"], scenes):
        if (any(entry.get(k) != v for k, v in slot.items())
                or entry["seed"] != slot["retry_seeds"][0]
                or not entry["scene_hash"] or entry["scene_hash"] in seen
                or entry["file"] in files):
            raise ValueError("Changed slot, seed, duplicate scene or artifact")
        seen.add(entry["scene_hash"])
        files.add(entry["file"])
    return [entry for entry in scenes if entry["split"] == "tuning"]


def advance_streak(stable, metrics):
    stable = stable + 1 if metrics["released_stack"] else 0
    return stable, stable >= 30


def aggregate(episodes):
    def counts(rows):
        n = len(rows)
        successes = sum(r["task_success"] for r in rows)
        unassisted = sum(r["unassisted_success"] for r in rows)
        return {"trials": n, "successes": successes, "unassisted_successes": unassisted,
                "success_rate": successes / n if n else None,
                "unassisted_success_rate": unassisted / n if n else None}
    result = {"planned_trials": 120, "finished_trials": len(episodes), "overall": counts(episodes)}
    for name, key in (("per_pair", lambda r: f"{r['pick']}->{r['place']}"),
                      ("per_cube_count", lambda r: str(len(r["colors"])))):
        result[name] = {value: counts([r for r in episodes if key(r) == value])
                        for value in sorted({key(r) for r in episodes})}
    return result


def verify_checkpoint(checkpoint, root, timing, dataset_hash):
    report = json.loads((checkpoint.parent.parent / "result.json").read_text())
    marker = json.loads((checkpoint / "checkpoint_result.json").read_text())
    if (checkpoint.name != "020000" or checkpoint.parent.name != "checkpoints"
            or report.get("status") != "complete" or report.get("purpose") != "smolvla_balanced_pilot"
            or report.get("training_steps") != 20000 or report.get("completed_training_steps") != 20000
            or report.get("best_checkpoint_update") != 20000 or report.get("final_data_binding_passed") is not True
            or Path(report["dataset_root"]).resolve() != root.resolve()
            or report["dataset_fingerprint"] != dataset_hash or report["timing"] != timing
            or report["training_source_hashes"] != training_source_hashes()
            or marker.get("status") != "complete" or marker.get("update") != 20000
            or marker.get("reload_fp32_weights_exact") is not True
            or not 0 <= marker.get("reload_max_action_error", float("inf")) <= 1e-5
            or [m for m in report["checkpoints"] if m["update"] == 20000] != [marker]
            or marker["model_sha256"] != file_hash(checkpoint / "model.safetensors")):
        raise ValueError("Selected complete 020000 checkpoint/data/source/model binding failed")


def restore_scene(model, bank, entry):
    path = (bank / entry["file"]).resolve()
    if path.parent != bank.resolve() or file_hash(path) != entry["sha256"]:
        raise ValueError("Scene artifact path/hash mismatch")
    data = mujoco.MjData(model)
    with np.load(path, allow_pickle=False) as state:
        restore_state(model, data, state["initial_state"])
        np.testing.assert_array_equal(capture_state(model, data), state["initial_state"])
        np.testing.assert_array_equal(data.qpos, state["qpos"])
    if scene_digest(data.qpos) != entry["scene_hash"]:
        raise ValueError("Scene qpos hash mismatch")
    validate_scene_geometry(model, data, entry["colors"])
    return data


def preflight(root, checkpoint, bank):
    manifest = require_complete(root)
    gate = require_replay_pass(root, ROOT)  # Fresh data/source/runtime AND final-validation binding.
    final = json.loads((root / "validation_result.json").read_text())
    bank_manifest = json.loads((bank / "manifest.json").read_text())
    versions, sources, bank_hash = runtime_versions(), source_provenance(ROOT), bank_fingerprint(bank)
    if (manifest.get("validation_required") is not True or final["status"] != "passed"
            or final["dataset_fingerprint"] != gate["dataset_fingerprint"]
            or Path(final["scene_bank"]).resolve() != bank or final["bank_fingerprint"] != bank_hash
            or final["bank_counts"] != {"tuning": 120, "final": 240}
            or final["source_hashes"] != sources or final["versions"] != versions
            or bank_manifest["source_hashes"] != sources or bank_manifest["versions"] != versions
            or manifest["splits"] != {"train": list(range(300)), "validation": list(range(300, 420))}
            or len(manifest["episodes"]) != 420 or manifest["fps"] != 30
            or manifest["camera_names"] != ["front_cam", "gripper_cam"] or manifest["camera_size"] != [512, 512]
            or manifest["control_dt_requested"] != 1/30 or bank_manifest["timing"] != manifest["timing"]):
        raise ValueError("Pilot bank/data/runtime/camera/timing binding mismatch")
    selected = select_tuning(manifest["plan"], bank_manifest, {e["scene_hash"] for e in manifest["episodes"]})
    if manifest["plan_digest"] != plan_digest(manifest["plan"]) or final["plan_digest"] != manifest["plan_digest"]:
        raise ValueError("Dataset/final-validation plan mismatch")
    scene_path = (ROOT / manifest["scene_path"]).resolve()
    if ROOT not in scene_path.parents:
        raise ValueError("Scene model escapes source root")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    timing = restore_timing(model, manifest)
    if timing["substeps_per_control"] != 20:
        raise ValueError("Expected 20 physics substeps")
    verify_checkpoint(checkpoint, root, timing, gate["dataset_fingerprint"])
    config = json.loads((checkpoint / "config.json").read_text())
    initial = json.loads((checkpoint.parent.parent / "initial_config/config.json").read_text())
    expected = {"observation.state": [9], "observation.images.image": [3, 512, 512],
                "observation.images.image2": [3, 512, 512]}
    if (config != initial or config["chunk_size"] != 50 or config["n_action_steps"] != 10
            or config["n_obs_steps"] != 1 or config["type"] != "smolvla"
            or {k: v["shape"] for k, v in config["input_features"].items()} != expected
            or {k: v["shape"] for k, v in config["output_features"].items()} != {"action": [8]}
            or not Path(config["vlm_model_name"]).is_dir()):
        raise ValueError("Saved policy schema/config/local VLM mismatch")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for step in json.loads((checkpoint / name).read_text())["steps"]:
            if "state_file" in step:
                path = (checkpoint / step["state_file"]).resolve()
                if path.parent != checkpoint or not path.is_file():
                    raise ValueError("Missing or nonlocal saved processor state")
            if step["registry_name"] == "tokenizer_processor" and step["config"]["tokenizer_name"] != config["vlm_model_name"]:
                raise ValueError("Processor tokenizer differs from local training assets")
    for entry in selected:
        restore_scene(model, bank, entry)  # Validate all tuning initializations before inference; never restore final.
    sources.update(training_source_hashes())
    sources.update({p: file_hash(ROOT / p) for p in ("validation/evaluate_pilot.py", "validation/rollout_diagnostics.py")})
    fingerprints = {"dataset": gate["dataset_fingerprint"], "bank": bank_hash, "sources": sources, "versions": versions,
                    "checkpoint": {p.name: file_hash(p) for p in sorted(checkpoint.iterdir()) if p.is_file()},
                    "training_result": file_hash(checkpoint.parent.parent / "result.json"),
                    "initial_config": file_hash(checkpoint.parent.parent / "initial_config/config.json"),
                    "final_validation": file_hash(root / "validation_result.json"),
                    "local_vlm_assets": bank_fingerprint(Path(config["vlm_model_name"]))}
    return manifest, selected, model, fingerprints


def run_trial(policy, pre, post, model, manifest, entry, bank, output, ordinal):
    if (policy.config.chunk_size, policy.config.n_action_steps) != (50, 10):
        raise ValueError("Expected chunk 50/action steps 10")
    data, scratch = restore_scene(model, bank, entry), mujoco.MjData(model)
    policy.reset()
    random.seed(SEED + ordinal)
    np.random.seed(SEED + ordinal)
    torch.manual_seed(SEED + ordinal)
    grasp = GraspController(model, data, target=f"target_cube_{entry['pick']}")
    initial_z, initial_time = float(data.xpos[grasp.target_id, 2]), float(data.time)
    initial_state = capture_state(model, data)
    instruction = f"pick up the {entry['pick']} cube and place it on top of the {entry['place']} cube"
    trace = {k: [] for k in ("state", "sim_time", "raw_action", "applied_action", "post_state", "post_sim_time")}
    stable = calls = 0
    success, outcome = False, "timeout"
    original = policy._get_action_chunk
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    started = time.monotonic()
    with ExitStack() as resources:
        renderer = mujoco.Renderer(model, height=512, width=512)
        resources.callback(renderer.close)
        record = resources.enter_context(av.open(str(output / "cameras.mp4"), "w"))
        stream = record.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 1024, 512, "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}
        def images_and_video():
            images = []
            for camera in manifest["camera_names"]:
                renderer.update_scene(data, camera=camera)
                images.append(renderer.render().copy())
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.concatenate(images, axis=1), format="rgb24")):
                record.mux(packet)
            return images
        policy._get_action_chunk = counted
        try:
            for _ in range(1200):
                synchronize_observation(model, data)
                state = data.qpos[:9].copy()
                frame = {"observation.state": torch.tensor(state, dtype=torch.float32), "task": instruction}
                for key, image in zip(("observation.images.image", "observation.images.image2"), images_and_video()):
                    frame[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
                with torch.inference_mode():
                    action = post(policy.select_action(pre(frame))).squeeze(0).cpu().numpy()
                if action.shape != (8,):
                    raise ValueError("Policy action shape is not eight physical commands")
                for key, value in (("state", state), ("sim_time", float(data.time)), ("raw_action", action.copy())):
                    trace[key].append(value)
                if not np.isfinite(action).all():
                    outcome = "nonfinite_action"
                    break
                bounded, clips = bound_action(action, model.actuator_ctrlrange[list(ACTIVE_JOINTS)])
                apply_action(model, data, bounded, ACTIVE_JOINTS)
                if step_control(model, data, manifest["control_dt_requested"], ARM_DOF) != 20:
                    raise ValueError("Physics substep count changed")
                metrics = fresh_stack_metrics(model, data, scratch, entry["pick"], entry["place"])
                diagnostics = target_diagnostics(model, data, entry["pick"], entry["place"], grasp, initial_z)
                for key, value in {"applied_action": bounded.copy(), "post_state": data.qpos[:9].copy(),
                        "post_sim_time": float(data.time), **clips, **metrics, **diagnostics}.items():
                    trace.setdefault(key, []).append(value)
                if not np.isfinite(capture_state(model, data)).all():
                    raise ValueError("Nonfinite simulator state")
                stable, success = advance_streak(stable, metrics)
                if success:
                    outcome = "success"
                    break
            images_and_video()  # Include terminal scene, not only pre-command observations.
        finally:
            policy._get_action_chunk = original
            for packet in stream.encode():
                record.mux(packet)
    attempts, ticks = len(trace["raw_action"]), len(trace["applied_action"])
    if calls != (attempts + 9)//10:
        raise ValueError("Actual chunk predictions differ from ceil(action_attempts/10)")
    if not np.isclose(data.time - initial_time, ticks/30, rtol=0, atol=1e-8):
        raise ValueError("Simulator duration differs from completed control ticks/30")
    masks = np.asarray(trace.get("joint_clip_mask", []), dtype=bool).reshape(-1, 7)
    grip = np.asarray(trace.get("gripper_clipped", []), dtype=bool)
    interventions = int(np.sum(np.any(masks, axis=1) | grip))
    terminal = {**fresh_stack_metrics(model, data, scratch, entry["pick"], entry["place"]),
                **target_diagnostics(model, data, entry["pick"], entry["place"], grasp, initial_z)}
    arrays = {k: np.asarray(v) for k, v in trace.items()}
    arrays["applied_action"] = arrays["applied_action"].reshape(-1, 8)
    np.savez_compressed(output / "trace.npz", **arrays, initial_integration_state=initial_state,
                        terminal_integration_state=capture_state(model, data), terminal_state=data.qpos[:9].copy(),
                        terminal_qpos=data.qpos.copy(), terminal_qvel=data.qvel.copy(), terminal_sim_time=float(data.time),
                        **{f"terminal_{k}": np.asarray(v) for k, v in terminal.items()})
    result = {**entry, "ordinal": ordinal, "rollout_seed": SEED + ordinal, "instruction": instruction,
              "outcome": outcome, "ticks": attempts, "action_attempts": attempts,
              "completed_control_ticks": ticks, "sim_seconds": ticks/30,
              "actual_chunk_predictions": calls, "chunk_size": 50, "n_action_steps": 10,
              "task_success": success, "unassisted_success": success and interventions == 0,
              "intervention_steps": interventions, "joint_clipped_ticks": int(np.any(masks, axis=1).sum()),
              "joint_clipped_ticks_per_dimension": masks.sum(axis=0).tolist(), "gripper_clipped_ticks": int(grip.sum()),
              "gripper_thresholded_ticks": int(np.sum(trace.get("gripper_thresholded", []))),
              "max_joint_clip_rad": max(trace.get("max_joint_clip_rad", [0.])),
              "gripper_adapter": "threshold_at_127_not_counted_as_clipping", "success_streak": stable,
              "final_metrics": {k: np.asarray(v).tolist() for k, v in terminal.items()},
              "wall_seconds": time.monotonic() - started}
    write_json(output / "episode.json", result)
    return result


def run(args, report):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    from lerobot.policies.factory import make_pre_post_processors
    manifest, selected, model, before = preflight(args.root, args.checkpoint, args.scene_bank)
    report.update(status="running", fingerprints=before, timing=manifest["timing"])
    write_json(args.output / "result.json", report)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    policy = load_fp32_checkpoint(args.checkpoint)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))
    for ordinal, entry in enumerate(selected):
        directory = args.output / f"trial_{ordinal:03d}"
        directory.mkdir(exist_ok=False)
        print(f"TRIAL {ordinal + 1}/120 {entry['slot_id']}", flush=True)
        report["episodes"].append(run_trial(policy, pre, post, model, manifest, entry, args.scene_bank, directory, ordinal))
        report.update(aggregate(report["episodes"]))
        write_json(args.output / "result.json", report)
        print("RESULT", json.dumps({"trial": ordinal + 1, "outcome": report["episodes"][-1]["outcome"],
                                   "overall": report["overall"]}), flush=True)
    _, _, _, after = preflight(args.root, args.checkpoint, args.scene_bank)
    if len(report["episodes"]) != 120 or before != after:
        raise ValueError("Incomplete trial count or changed post-run fingerprints")
    report.update(status="complete", post_run_fingerprints_unchanged=True)
    write_json(args.output / "result.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "checkpoint", "scene-bank", "output"):
        parser.add_argument("--" + name, type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "preflight", "purpose": "pilot_tuning_only", "episodes": [],
              "dataset_root": str(args.root), "checkpoint": str(args.checkpoint), "scene_bank": str(args.scene_bank),
              "max_control_ticks": 1200, "max_sim_seconds": 40, "seed": SEED, "training_updates": 0,
              "retries": False, "resume": False, "final_scenes_evaluated": False, **aggregate([])}
    write_json(args.output / "result.json", report)
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(args.output / "result.json", report)
        raise


if __name__ == "__main__":
    main()