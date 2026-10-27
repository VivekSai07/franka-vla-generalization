"""P3 admission of one episode, validation of the merged dataset, fingerprint and seal.

Imports no robosuite at module level; `admit` takes its replay function as a parameter so it is
testable without a simulator. P1's gates are kept unchanged. The task of every frame is the one fixed
prompt: the expert's own text must never appear in a dataset.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.props_p3_data import contract
from experiments.props_p3_data import plan as p

GATES = ("expert_success", "frames", "replay", "labels", "schema")
STABLE_TICKS = 30
CAMERA_KEYS = ("observation.images.image", "observation.images.image2")


def _default_replay(result, frames, traces):
    from experiments.props_scene.validate import replay
    return replay(result, [f["action"] for f in frames], traces)


def _schema_ok(frames, images):
    for frame in frames:
        state, action = np.asarray(frame["observation.state"]), np.asarray(frame["action"])
        if state.dtype != np.float32 or state.shape != (9,) or action.dtype != np.float32 or action.shape != (8,):
            return False
        for key in CAMERA_KEYS if images else ():
            image = frame.get(key)
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (512, 512, 3):
                return False
    return True


def _labels_ok(episode, frames):
    """The fixed prompt in the episode and in every frame, and the container the object's category names."""
    if episode["instruction"] != p.PROMPT or episode["bin"] != p.container_of(episode["object"]):
        return False
    return all(frame["task"] == p.PROMPT for frame in frames)


def admit(episode, result, frames, traces, replay=_default_replay, images=True):
    """Record every gate; `reason` is the first failing gate, or "admitted". `images=False` for unrendered gate runs."""
    gates, details = {}, {}
    gates["expert_success"] = bool(result["success"]) and int(result["stable_ticks"]) >= STABLE_TICKS
    gates["frames"] = len(frames) == p.EXPECTED_EPISODE_FRAMES == int(result["frames"]) == len(traces)
    if gates["expert_success"] and gates["frames"]:
        try:
            check = replay(result, frames, traces)
            details["replay"] = {k: v for k, v in check.items() if k != "ok"}
            gates["replay"] = bool(check["ok"])
        except ValueError as exc:
            details["replay_error"] = str(exc)
            gates["replay"] = False
    else:
        gates["replay"] = False
        details["replay_error"] = "skipped: expert or frame gate failed"
    gates["labels"] = _labels_ok(episode, frames)
    gates["schema"] = _schema_ok(frames, images)
    reason = next((name for name in GATES if not gates[name]), "admitted")
    return {"admitted": reason == "admitted", "gates": gates, "reason": reason, "details": details}


# --- fingerprint and seal ------------------------------------------------------------------------

def _file_digest(path):
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
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


# --- dataset validation ---------------------------------------------------------------------------

def read_dataset_meta(root):
    """LeRobot v3 metadata: counts, fps, the task table, and each episode's task and length."""
    import pandas as pd
    import pyarrow.parquet as pq
    meta = Path(root) / "dataset/meta"
    info = json.loads((meta / "info.json").read_text())
    episodes = []
    for path in sorted((meta / "episodes").rglob("*.parquet")):
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda e: e["episode_index"])
    return {"total_episodes": info["total_episodes"], "total_frames": info["total_frames"], "fps": info["fps"],
            "task_table": sorted(pd.read_parquet(meta / "tasks.parquet").index),
            "tasks": [list(e["tasks"]) for e in episodes], "lengths": [e["length"] for e in episodes],
            "indices": [e["episode_index"] for e in episodes]}


def admitted_episodes(plan, journal):
    """The admitted layout of every slot, in plan order; raises unless each slot is admitted exactly once."""
    admitted = [j for j in journal if j.get("admitted")]
    if [j["slot"] for j in admitted] != list(range(len(plan["episodes"]))):
        raise ValueError("Journal does not admit every slot exactly once, in plan order")
    return [p.resample_start(plan["episodes"][j["slot"]], int(j.get("r", 0))) for j in admitted]


def validate_dataset(root, plan, journal, reader=read_dataset_meta):
    episodes = admitted_episodes(plan, journal)
    meta = reader(root)
    n = len(episodes)
    # The held-out objects and the container mapping come first: they are the premise of the experiment.
    if any(e["object"] in p.HELD_OUT_OBJECTS for e in episodes):
        raise ValueError("A held-out object was collected")
    if any(e["bin"] != p.container_of(e["object"]) for e in episodes):
        raise ValueError("An episode's bin is not its object's container")
    if (meta["total_episodes"] != n or len(meta["tasks"]) != n or meta["indices"] != list(range(n))
            or meta["total_frames"] != p.EXPECTED_EPISODE_FRAMES * n or meta["fps"] != 30
            or any(length != p.EXPECTED_EPISODE_FRAMES for length in meta["lengths"])):
        raise ValueError("Dataset counts differ from the plan")
    if list(meta["task_table"]) != [p.PROMPT]:
        raise ValueError("Dataset task table is not exactly the fixed prompt")
    for i, tasks in enumerate(meta["tasks"]):
        if tasks != [p.PROMPT]:
            raise ValueError(f"Episode {i} task is not the fixed prompt")
    train = [e for e in episodes if e["split"] == "train"]
    validation = [e for e in episodes if e["split"] == "validation"]
    if (len(train) != p.TRAIN_COUNT or len(validation) != p.VALIDATION_COUNT
            or Counter(tuple(e["cell"]) for e in train) != {c: p.TRAIN_PER_CELL for c in p.TRAINED_CELLS}
            or Counter(tuple(e["cell"]) for e in validation) != {c: p.VALIDATION_PER_CELL for c in p.TRAINED_CELLS}):
        raise ValueError("Cell balance differs from the plan")
    return {"episodes": n, "train": len(train), "validation": len(validation),
            "per_cell_train": p.TRAIN_PER_CELL, "per_cell_validation": p.VALIDATION_PER_CELL,
            "objects": dict(Counter(e["object"] for e in episodes)),
            "substituted_slots": sum(int(j.get("r", 0)) > 0 for j in journal if j.get("admitted")),
            "held_out_objects_collected": 0, "prompt": p.PROMPT}


def seal(root, plan, provenance):
    root = Path(root)
    substitutions = root / "substitutions.json"
    sealed = {"fingerprint": fingerprint(root), "provenance_sha256": p.digest(provenance),
              "plan_sha256": p.digest(plan),
              "substitutions_sha256": p.digest(json.loads(substitutions.read_text()) if substitutions.exists() else {}),
              "image": contract.IMAGE, "episodes": len(plan["episodes"])}
    (root / "seal.json").write_text(json.dumps(sealed, indent=2, allow_nan=False) + "\n")
    return sealed
