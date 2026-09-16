"""Replay exact commands and persisted LeRobot actions; fail on drift/provenance."""
import argparse
import hashlib
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_collection.benchmark import (
    apply_action, require_complete, restore_state, source_provenance,
    stack_metrics, step_control, write_json, restore_timing, synchronize_observation,
)
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from validation.gates import dataset_fingerprint, fresh_stack_metrics


def replay(root):
    root = Path(root).resolve()
    # Invalidate any old PASS before a check that can fail.
    write_json(root / "replay/replay_result.json", {"passed": False, "status": "in_progress"})
    manifest = require_complete(root)
    if manifest["source_hashes"] != source_provenance(ROOT):
        raise ValueError("Source/assets changed since collection")
    if manifest["versions"]["mujoco"] != mujoco.__version__:
        raise ValueError("MuJoCo version changed since recording")
    ds = LeRobotDataset(repo_id=root.name, root=root)
    ds._ensure_hf_dataset_loaded()
    if ds.meta.fps != manifest["fps"]:
        raise ValueError("LeRobot FPS metadata disagrees with recorded timing")
    assert ds.meta.total_episodes == len(manifest["episodes"]) == manifest["requested_episodes"]
    results = []
    for entry in manifest["episodes"]:
        path = root / "replay" / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Replay sidecar hash mismatch")
        with np.load(path, allow_pickle=False) as loaded:
            trace = dict(loaded)
        ep = ds.meta.episodes[entry["episode_index"]]
        start, end = ep["dataset_from_index"], ep["dataset_to_index"]
        actions = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(start, end)])
        states = np.stack([ds.hf_dataset[i]["observation.state"].numpy() for i in range(start, end)])
        timestamps = np.array([ds.hf_dataset[i]["timestamp"].item() for i in range(start, end)])
        n = len(actions)
        assert n == entry["frames"] == len(trace["qpos"]) == len(trace["ctrl"])
        assert n == len(trace["sim_time"]) == len(trace["substeps"])
        # Both simulator and persisted LeRobot timestamps must describe the same clock.
        np.testing.assert_allclose(trace["sim_time"] - trace["sim_time"][0],
                       np.arange(n) / manifest["fps"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(timestamps, trace["sim_time"] - trace["sim_time"][0], rtol=0, atol=2e-6)
        assert np.isfinite(actions).all() and np.isfinite(states).all()
        np.testing.assert_allclose(states, trace["qpos"][:, :9], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(actions[:, :7], trace["ctrl"][:, list(ACTIVE_JOINTS)].astype(np.float32))
        np.testing.assert_array_equal(actions[:, 7], trace["ctrl"][:, -1])
        for mode in ("reference", "recorded_actions"):
            model = mujoco.MjModel.from_xml_path(str(ROOT / manifest["scene_path"]))
            timing = restore_timing(model, manifest)
            data = mujoco.MjData(model)
            scratch = mujoco.MjData(model)
            restore_state(model, data, trace["initial_state"])
            max_state_error, max_full_error, stable = 0., 0., 0
            for i in range(n):
                synchronize_observation(model, data)
                if mode == "reference":
                    np.testing.assert_allclose(data.xpos, trace["body_xpos"][i], rtol=0, atol=1e-9)
                    np.testing.assert_allclose(data.cam_xpos, trace["camera_xpos"][i], rtol=0, atol=1e-9)
                max_state_error = max(max_state_error, float(np.max(np.abs(data.qpos[:9] - trace["qpos"][i, :9]))))
                max_full_error = max(max_full_error, float(np.max(np.abs(data.qpos - trace["qpos"][i]))))
                assert abs(data.time - trace["sim_time"][i]) < 1e-9
                if mode == "reference":
                    data.ctrl[:] = trace["ctrl"][i]
                else:
                    apply_action(model, data, actions[i], ACTIVE_JOINTS)
                steps = step_control(model, data, manifest["control_dt_requested"], ARM_DOF)
                assert steps == trace["substeps"][i]
                assert steps == timing["substeps_per_control"]
                metrics = fresh_stack_metrics(model, data, scratch, entry["pick"], entry["place"])
                stable = stable + 1 if metrics["released_stack"] else 0
            if mode == "reference":
                np.testing.assert_allclose(data.qpos, trace["final_qpos"], rtol=0, atol=1e-5)
            max_state_error = max(max_state_error, float(np.max(np.abs(data.qpos[:9] - trace["final_qpos"][:9]))))
            max_full_error = max(max_full_error, float(np.max(np.abs(data.qpos - trace["final_qpos"]))))
            error = max_full_error if mode == "reference" else max_state_error
            tolerance = 1e-5 if mode == "reference" else 1e-3
            passed = error <= tolerance and stable >= manifest["fps"]
            result = {"episode": entry["episode_index"], "mode": mode,
                      "max_policy_state_error": max_state_error, "max_full_qpos_error": max_full_error,
                      "stable_ticks": stable, "final_metrics": metrics, "passed": passed}
            result["max_timestamp_error_s"] = float(np.max(np.abs(timestamps - (trace["sim_time"] - trace["sim_time"][0]))))
            result["timing_version"] = timing["version"]
            print(result, flush=True)
            results.append(result)
            if not passed:
                write_json(root / "replay/replay_result.json", {"passed": False, "results": results})
                raise AssertionError("Replay drift or released-stack gate failed")
    write_json(root / "replay/replay_result.json", {"passed": True, "results": results,
                                                  "dataset_fingerprint": dataset_fingerprint(root)})
    print(f"PASS: all {ds.meta.total_episodes} episodes replayed with both reference and recorded commands")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    replay(parser.parse_args().root)