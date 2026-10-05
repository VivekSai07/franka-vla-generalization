"""B3 admission of one episode, validation of the whole dataset, fingerprint and seal.

Imports no robosuite at module level; `admit` takes its replay function as a parameter so it is
testable without a simulator.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.b3_robosuite_data import contract
from experiments.b3_robosuite_data import plan as p
from experiments.paired_demonstrations.plan import instruction

GATES = ("expert_success", "frames", "stack", "distractors_undisturbed", "replay", "labels")
REPLAY_LIMIT = 1e-3
STABLE_TICKS = 30


def _cube_qpos_address(model, colour):
    joint = model.joint(int(model.body(f"target_cube_{colour}").jntadr[0]))
    return int(joint.qposadr[0])


def distractor_disturbance(model, initial_qpos, final_qpos, distractors):
    """Horizontal distance (m) each distractor moved between reset and the end of the episode."""
    moved = {}
    for colour in distractors:
        adr = _cube_qpos_address(model, colour)
        moved[colour] = float(np.hypot(*(np.asarray(final_qpos[adr:adr + 2]) - np.asarray(initial_qpos[adr:adr + 2]))))
    return moved


def _default_replay(result, frames, traces, pick, place):
    from experiments.robosuite_scene.validate import replay_recorded
    return replay_recorded(result, frames, traces, pick=pick, place=place)


def _labels_ok(episode, frames):
    if episode["instruction"] != instruction(episode["pick"], episode["place"]):
        return False
    for frame in frames:
        state, action = np.asarray(frame["observation.state"]), np.asarray(frame["action"])
        if (frame["task"] != episode["instruction"] or state.dtype != np.float32 or state.shape != (9,)
                or action.dtype != np.float32 or action.shape != (8,)):
            return False
    return True


def admit(episode, result, frames, traces, replay=_default_replay):
    """Record every gate; `reason` is the first failing gate, or "admitted".

    The replay gate re-runs the physics, so it is skipped (and recorded as failed) when the expert
    or the frame count already failed: there is nothing consistent to replay.
    """
    gates, details = {}, {}
    gates["expert_success"] = bool(result["success"]) and int(result["stable_ticks"]) >= STABLE_TICKS
    gates["frames"] = len(frames) == p.EXPECTED_FRAMES == int(result["frames"]) == len(traces)
    xy, z = result.get("final_xy_m"), result.get("final_z_m")
    gates["stack"] = xy is not None and z is not None and bool(xy < 0.02 and 0.035 < z < 0.045)
    moved = distractor_disturbance(result["model"], result["reset"]["qpos"], result["final_qpos"],
                                   episode["distractors"])
    details["distractor_moved_m"] = moved
    gates["distractors_undisturbed"] = all(m <= p.DISTRACTOR_DISTURBANCE_M for m in moved.values())
    if gates["expert_success"] and gates["frames"]:
        try:
            error = float(replay(result, frames, traces, episode["pick"], episode["place"])["max_policy_state_error"])
            details["replay_max_policy_state_error"] = error
            gates["replay"] = error < REPLAY_LIMIT
        except ValueError as exc:
            details["replay_error"] = str(exc)
            gates["replay"] = False
    else:
        gates["replay"] = False
        details["replay_error"] = "skipped: expert or frame gate failed"
    gates["labels"] = _labels_ok(episode, frames)
    reason = next((name for name in GATES if not gates[name]), "admitted")
    return {"admitted": reason == "admitted", "gates": gates, "reason": reason, "details": details}


# --- fingerprint and seal ------------------------------------------------------------------------

def _canonical_json_digest(path):
    value = json.loads(path.read_text())
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_digest(path):
    if path.suffix == ".json":
        return _canonical_json_digest(path)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def fingerprint(root):
    """sha256 over {relative path: digest} of every file but root/seal.json; JSON by canonical form."""
    root, files = Path(root), {}
    if root.is_symlink():
        raise ValueError("Symlink root")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in sealed root: {path.relative_to(root)}")
        if path.is_file() and path != root / "seal.json":
            files[str(path.relative_to(root))] = _file_digest(path)
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


# --- dataset validation --------------------------------------------------------------------------

def read_dataset_meta(root):
    """LeRobot v3 metadata: counts, fps, the task table, and each episode's instruction and length."""
    import pandas as pd
    import pyarrow.parquet as pq
    meta = Path(root) / "dataset/meta"
    info = json.loads((meta / "info.json").read_text())
    episodes = []
    for path in sorted((meta / "episodes").rglob("*.parquet")):
        episodes.extend(pq.read_table(path).to_pylist())
    return {"total_episodes": info["total_episodes"], "total_frames": info["total_frames"], "fps": info["fps"],
            "task_table": sorted(pd.read_parquet(meta / "tasks.parquet").index),
            "tasks": [list(e["tasks"]) for e in episodes], "lengths": [e["length"] for e in episodes],
            "indices": [e["episode_index"] for e in episodes]}


def _slots(plan, journal):
    admitted = [j for j in journal if j.get("admitted")]
    if [j["slot"] for j in admitted] != list(range(len(plan["episodes"]))):
        raise ValueError("Journal does not admit every slot exactly once, in plan order")
    return [p.resample_layout(plan["episodes"][j["slot"]], int(j.get("r", 0))) for j in admitted]


def validate_dataset(root, plan, journal, reader=read_dataset_meta):
    episodes = _slots(plan, journal)
    meta = reader(root)
    n = len(episodes)
    if (meta["total_episodes"] != n or len(meta["tasks"]) != n
            or meta["total_frames"] != p.EXPECTED_FRAMES * n or meta["fps"] != 30
            or any(length != p.EXPECTED_FRAMES for length in meta["lengths"])):
        raise ValueError("Dataset counts differ from the plan")
    if sorted(meta["task_table"]) != sorted(plan["instructions"]):
        raise ValueError("Dataset task table differs from the 10 trained instructions")
    for i, (episode, tasks) in enumerate(zip(episodes, meta["tasks"])):
        if tasks != [episode["instruction"]]:
            raise ValueError(f"Episode {i} instruction differs from its plan slot")
    train = [e for e in episodes if e["split"] == "train"]
    validation = [e for e in episodes if e["split"] == "validation"]
    cell = lambda e: (e["pick"], e["place"], e["side"])
    if (len(train) != p.TRAIN_COUNT or len(validation) != p.VALIDATION_COUNT
            or Counter(map(cell, train)) != {c: p.TRAIN_PER_CELL for c in p.CELLS}
            or Counter(map(cell, validation)) != {c: p.VALIDATION_PER_CELL for c in p.CELLS}):
        raise ValueError("Cell balance differs from the plan")
    distractors = Counter(len(e["distractors"]) for e in train)
    edges = Counter(e["edge"] for e in train)
    if distractors != {0: 100, 1: 200, 2: 200}:
        raise ValueError(f"Distractor balance differs from the plan: {dict(distractors)}")
    if edges != {"interior": 400, "inner": 50, "outer": 50}:
        raise ValueError(f"Edge share differs from the plan: {dict(edges)}")
    if any((e["pick"], e["place"]) in p.HELD_OUT_PAIRS for e in episodes):
        raise ValueError("A held-out pair was collected")
    return {"episodes": n, "train": len(train), "validation": len(validation),
            "per_cell_train": p.TRAIN_PER_CELL, "per_cell_validation": p.VALIDATION_PER_CELL,
            "distractors_train": {str(k): v for k, v in sorted(distractors.items())},
            "edge_train": dict(sorted(edges.items())),
            "substituted_slots": sum(int(j.get("r", 0)) > 0 for j in journal if j.get("admitted")),
            "held_out_pairs_collected": 0}


def seal(root, plan, provenance):
    root = Path(root)
    substitutions = root / "substitutions.json"
    sealed = {"fingerprint": fingerprint(root), "provenance_sha256": p.digest(provenance),
              "plan_sha256": p.digest(plan),
              "substitutions_sha256": p.digest(json.loads(substitutions.read_text()) if substitutions.exists() else {}),
              "image": contract.IMAGE, "min_clearance_m": plan["min_clearance_m"],
              "episodes": len(plan["episodes"])}
    (root / "seal.json").write_text(json.dumps(sealed, indent=2, allow_nan=False) + "\n")
    return sealed
