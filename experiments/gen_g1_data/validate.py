"""G1 admission of one episode, validation of the merged dataset, the rejection report, fingerprint and seal.

Imports no robosuite at module level; `admit` takes its replay and contact functions as parameters so it is testable without a
simulator. P2's gates are kept (`distractors` now also covers pool A primitives) and `signals` is added: the attempt's exact replay,
which yields S0's signal arrays, must succeed and the arrays must be complete, finite and correctly segmented.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.gen_g1_data import contract
from experiments.gen_g1_data import plan as p

GATES = ("expert_success", "distractors", "frames", "replay", "labels", "schema", "signals")
STABLE_TICKS = 30
CAMERA_KEYS = ("observation.images.image", "observation.images.image2")


def _default_replay(episode, result, frames, traces):
    from experiments.gen_g1_data.episode import replay_signals
    arrays, details = replay_signals(episode, result, [f["action"] for f in frames], traces)
    return {"ok": True, "arrays": arrays, **details}


def _default_contacts(result, frames, traces):
    from experiments.gen_g1_data.episode import distractor_contacts
    return distractor_contacts(result, [f["action"] for f in frames], traces)


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
    if episode["instruction"] != p.instruction(episode["object"], episode["bin"]):
        return False
    return all(frame["task"] == episode["instruction"] for frame in frames)


def distractors_ok(episode, result, report):
    """Every distractor present and touched by nothing but the table, at any physics substep."""
    expected = [d["object"] for d in episode["distractors"]] + [f"{s}_{c}" for s, c, *_ in p.condition_of(episode).primitives]
    if sorted(result["distractors"]) != sorted(expected):
        return False
    return not report["touched"]


def signals_ok(arrays, episode, frames):
    """The S0 sidecar checks (shapes, dtypes, finite, monotone segments) on this episode's arrays."""
    from experiments.vla_inputs_signals.sidecar import check_arrays
    try:
        check_arrays(arrays, int(episode["episode_index"]), frames)
    except ValueError:
        return False
    return True


def admit(episode, result, frames, traces, replay=_default_replay, contacts=_default_contacts, images=True):
    """Record every gate; `reason` is the first failing gate, or "admitted". `images=False` for unrendered gate runs.
    On success `result["signals"]` holds the sidecar arrays (the committer writes them)."""
    gates, details = {}, {"distractor_motion": result["distractor_motion"], "expert_reason": result["reason"]}
    gates["expert_success"] = bool(result["success"]) and int(result["stable_ticks"]) >= STABLE_TICKS
    try:
        report = contacts(result, frames, traces)
        details["distractor_touched"] = report["touched"]
        gates["distractors"] = distractors_ok(episode, result, report)
    except ValueError as exc:
        details["distractor_error"] = str(exc)
        gates["distractors"] = False
    gates["frames"] = len(frames) == p.EXPECTED_EPISODE_FRAMES == int(result["frames"]) == len(traces)
    arrays = None
    if gates["expert_success"] and gates["frames"]:
        try:
            check = replay(episode, result, frames, traces)
            arrays = check.pop("arrays", None)
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
    gates["signals"] = bool(gates["replay"] and arrays is not None and signals_ok(arrays, episode, len(frames)))
    if gates["signals"]:
        result["signals"] = arrays
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
    """LeRobot v3 metadata: counts, fps, the task table, and each episode's instruction and length."""
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
    """The admitted scene of every slot, in plan order; raises unless each slot is admitted exactly once."""
    admitted = [j for j in journal if j.get("admitted")]
    if [j["slot"] for j in admitted] != list(range(len(plan["episodes"]))):
        raise ValueError("Journal does not admit every slot exactly once, in plan order")
    return [p.resample_start(plan["episodes"][j["slot"]], int(j.get("r", 0))) for j in admitted]


def _read_npz(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def check_recorded_files(root, episodes, frames=p.EXPECTED_EPISODE_FRAMES):
    """Every admitted slot has its trace and its signal arrays, and both say what the plan says: the same scene, the same signals."""
    root = Path(root)
    names = [f"episode_{e['episode_index']:04d}.npz" for e in episodes]
    for folder in ("episodes", "signals"):
        if sorted(q.name for q in (root / folder).iterdir()) != names:
            raise ValueError(f"{folder}/ does not hold exactly the admitted episodes")
    from experiments.vla_inputs_signals.sidecar import check_arrays
    for episode, name in zip(episodes, names):
        trace = _read_npz(root / "episodes" / name)
        identity = (str(trace["object"]), str(trace["bin"]), str(trace["blue_slot"]), [str(d) for d in trace["distractors"]])
        if (identity != (episode["object"], episode["bin"], episode["blue_slot"], [d["object"] for d in episode["distractors"]])
                or not np.array_equal(trace["start"], np.asarray(episode["start"]))
                or str(trace["condition_sha256"]) != episode["condition_sha256"]
                or json.loads(str(trace["condition"])) != episode["condition"]
                or str(trace["layout_id"]) != episode["layout_id"]
                or not np.array_equal(trace["home_pose_jitter_deg"], np.asarray(json.loads(str(trace["draw"]))["home_pose_jitter_deg"]))):
            raise ValueError(f"{name}: the recorded trace is not the plan's scene")
        arrays = _read_npz(root / "signals" / name)
        check_arrays(arrays, int(episode["episode_index"]), frames)
        if str(arrays["object"]) != episode["object"] or str(arrays["split"]) != episode["split"]:
            raise ValueError(f"{name}: the signals are not this episode's")
        if len(trace["qpos"]) != frames or len(arrays["wrist_wrench"]) != frames:
            raise ValueError(f"{name}: a trace or signal array is not {frames} ticks long")


def validate_dataset(root, plan, journal, reader=read_dataset_meta, files=True):
    episodes = admitted_episodes(plan, journal)
    meta = reader(root)
    n = len(episodes)
    if (meta["total_episodes"] != n or len(meta["tasks"]) != n or meta["indices"] != list(range(n))
            or meta["total_frames"] != p.EXPECTED_EPISODE_FRAMES * n or meta["fps"] != 30
            or any(length != p.EXPECTED_EPISODE_FRAMES for length in meta["lengths"])):
        raise ValueError("Dataset counts differ from the plan")
    if sorted(meta["task_table"]) != sorted({e["instruction"] for e in plan["episodes"]}):
        raise ValueError("Dataset task table differs from the trained instructions")
    for i, (episode, tasks) in enumerate(zip(episodes, meta["tasks"])):
        if tasks != [episode["instruction"]]:
            raise ValueError(f"Episode {i} instruction differs from its plan slot")
    train = [e for e in episodes if e["split"] == "train"]
    validation = [e for e in episodes if e["split"] == "validation"]
    if (len(train) != p.TRAIN_COUNT or len(validation) != p.VALIDATION_COUNT
            or Counter(tuple(e["cell"]) for e in train) != {c: p.TRAIN_PER_CELL for c in p.TRAINED_CELLS}
            or Counter(tuple(e["cell"]) for e in validation) != {c: p.VALIDATION_PER_CELL for c in p.TRAINED_CELLS}
            or Counter(e["distractor_count"] for e in train) != p.count_deck(p.TRAIN_COUNT)
            or Counter(e["distractor_count"] for e in validation) != p.count_deck(p.VALIDATION_COUNT)
            or Counter(e["quintile"] for e in train) != {q: p.TRAIN_COUNT // p.QUINTILES for q in range(p.QUINTILES)}
            or Counter(e["quintile"] for e in validation) != {q: p.VALIDATION_COUNT // p.QUINTILES for q in range(p.QUINTILES)}):
        raise ValueError("Cell, distractor-count or height-stratum balance differs from the plan")
    if any((e["object"], e["bin"]) in {tuple(x) for x in p.HELD_OUT_PAIRS} for e in episodes):
        raise ValueError("A held-out pair was collected")
    for e in episodes:
        c = p.condition_of(e)
        c.check()
        if c.pool != "train":
            raise ValueError(f"{e['layout_id']}: a condition outside the train pool")
    if files:
        check_recorded_files(root, episodes)
    return {"episodes": n, "train": len(train), "validation": len(validation),
            "distractor_counts": {str(k): v for k, v in sorted(Counter(e["distractor_count"] for e in episodes).items())},
            "height_strata": {str(k): v for k, v in sorted(Counter(e["quintile"] for e in episodes).items())},
            "start_regions": dict(Counter(e["region"] for e in episodes)),
            "substituted_slots": sum(int(j.get("r", 0)) > 0 for j in journal if j.get("admitted")),
            "held_out_pairs_collected": 0, "signals_checked": bool(files)}


def rejection_report(plan, journal):
    """First-attempt and whole-run rejection rates by object, height stratum, distractor count and start region, and by reason.
    Rates are rejected attempts over all attempts of that slot's stratum, so a skew in what the admitted data under-represents shows."""
    slots = plan["episodes"]
    by = {"object": lambda e: e["object"], "height_stratum": lambda e: str(e["quintile"]),
          "distractor_count": lambda e: str(e["distractor_count"]), "start_region": lambda e: e["region"]}
    report = {"attempts": len(journal), "rejected": sum(not j["admitted"] for j in journal),
              "reasons": dict(Counter(j["reason"] for j in journal if not j["admitted"]))}
    for name, key in by.items():
        attempts, rejected = Counter(), Counter()
        for j in journal:
            k = key(slots[j["slot"]])
            attempts[k] += 1
            rejected[k] += 0 if j["admitted"] else 1
        report[name] = {k: {"attempts": attempts[k], "rejected": rejected[k], "rate": rejected[k] / attempts[k]}
                        for k in sorted(attempts)}
    return report


def seal(root, plan, provenance):
    root = Path(root)
    substitutions = root / "substitutions.json"
    sealed = {"fingerprint": fingerprint(root), "provenance_sha256": p.digest(provenance), "plan_sha256": p.digest(plan),
              "substitutions_sha256": p.digest(json.loads(substitutions.read_text()) if substitutions.exists() else {}),
              "image": contract.IMAGE, "episodes": len(plan["episodes"])}
    (root / "seal.json").write_text(json.dumps(sealed, indent=2, allow_nan=False) + "\n")
    return sealed
