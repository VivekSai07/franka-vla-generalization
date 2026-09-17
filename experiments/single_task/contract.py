"""Fixed single-task contracts; no model construction or simulation stepping."""
import json
import math
from pathlib import Path
import sys

from training import smolvla_pilot as h
from data_collection.benchmark import source_provenance

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/franka_pilot_20260915"
OUTPUT = ROOT / "outputs/train/smolvla_single_task_20260917"
SCRATCH = OUTPUT.with_name(OUTPUT.name + "_preflight")
TRAIN, VAL = tuple(range(9)), tuple(range(300, 304))
SEED = 20260917
REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
TASK = "pick up the red cube and place it on top of the green cube"
OBS = {"observation.state": (9,), "observation.images.image": (3, 512, 512),
       "observation.images.image2": (3, 512, 512)}


def select_entries(manifest):
    train, val = h.checked_splits(manifest, len(manifest["episodes"]))
    entries = {e["episode_index"]: e for e in manifest["episodes"]}
    if (len(entries) != 420 or set(entries) != set(range(420))
            or not set(TRAIN) <= set(train) or not set(VAL) <= set(val)
            or manifest["fps"] != 30 or manifest["camera_size"] != [512, 512]
            or manifest["camera_names"] != ["front_cam", "gripper_cam"]
            or manifest["control_dt_requested"] != 1/30 or not manifest["validation_required"]):
        raise ValueError("Single-task manifest/split/schema mismatch")
    selected = [entries[i] for i in TRAIN + VAL]
    for e in selected:
        if (e["split"] != ("train" if e["episode_index"] in TRAIN else "validation")
                or (e["pick"], e["place"]) != ("red", "green")
                or len(e["colors"]) != 2 or set(e["colors"]) != {"red", "green"}
                or e["instruction"] != TASK or e["frames"] != 528
                or any(len(e[k]) != 64 or any(c not in "0123456789abcdef" for c in e[k])
                       for k in ("sha256", "scene_hash"))):
            raise ValueError("Selected episode identity/count/hash mismatch")
    return selected


def indices(episodes):
    for i in TRAIN + VAL:
        e = episodes[i]
        if (e["episode_index"] != i or e["dataset_to_index"]-e["dataset_from_index"] != 528):
            raise ValueError("Episode metadata identity/frame count mismatch")
    rows = h.frame_indices(episodes, TRAIN)
    starts = h.valid_chunk_starts(episodes, TRAIN, 50)
    val_rows = h.frame_indices(episodes, VAL)
    if (len(rows) != 4752 or len(set(rows)) != 4752 or len(starts) != 4311
            or len(set(val_rows)) != 2112 or set(rows) & set(val_rows)):
        raise ValueError("Train/validation row leakage or wrong count")
    probes = {name: [int(episodes[i]["dataset_from_index"])+offset for i in ids
                     for offset in (0, 30, 60, 107, 347)]
              for name, ids in (("training", TRAIN), ("validation", VAL))}
    return rows, starts, probes


class PolicyRows:
    """Whitelist observations; never send episode identity or future targets to probes."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        frame = self.dataset[index]
        if (int(frame["episode_index"]) not in TRAIN + VAL or frame["task"] != TASK
                or {k for k in frame if k.startswith("observation.")} != set(OBS)):
            raise ValueError("Unapproved episode/task/privileged observation")
        return {k: frame[k] for k in (*OBS, "task", "action", "action_is_pad")}


def learning_rate(update):
    if type(update) is not int or not 1 <= update <= 5000:
        raise ValueError("Update must be in [1,5000]")
    return (1e-4*update/500 if update <= 500 else
            1e-5 + .5*(1e-4-1e-5)*(1+math.cos(math.pi*(update-500)/4500)))


def claim_output(output, allowed):
    output, allowed = Path(output), Path(allowed)
    if output.absolute() != allowed.absolute() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Output must be the exclusive approved path, without symlinks")
    output.mkdir(parents=True, exist_ok=False)


def inventory(directory):
    paths = sorted(Path(directory).rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("Immutable artifacts cannot be symlinks")
    return {str(p.relative_to(directory)): h.file_hash(p) for p in paths if p.is_file()}


def require_absolute_reload(marker):
    if not 0 <= marker.get("reload_max_action_error", float("inf")) <= 1e-5:
        raise ValueError("Checkpoint absolute reload action error must be <= 1e-5")


def sources():
    result = {**source_provenance(ROOT), **h.training_source_hashes()}
    paths = [ROOT / p for p in ("validation/evaluate_pilot.py", "validation/rollout_diagnostics.py",
                              "validation/paired_instruction_probe.py")]
    paths += sorted(Path(__file__).parent.rglob("*.py"))
    result.update({str(p.relative_to(ROOT)): h.file_hash(p) for p in paths})
    return result


def trials(manifest):
    return [(e, e["episode_index"]+1, SEED+e["episode_index"]) for e in select_entries(manifest)]


def summarize(rows):
    ids = [r["episode_index"] for r in rows]
    if len(set(ids)) != len(ids) or not set(ids) <= set(TRAIN+VAL):
        raise ValueError("Unapproved or duplicate evaluation result")
    result = {name: dict(trials=sum(i in group for i in ids),
                        successes=sum(bool(r["task_success"]) for r in rows if r["episode_index"] in group), goal=goal)
              for name, group, goal in (("seen", TRAIN, 8), ("held_out", VAL, 3))}
    return {**result, "planned_trials": 13, "finished_trials": len(rows),
            "provisional_capability": len(rows) == 13 and all(r["successes"] >= r["goal"] for r in result.values()),
            "statistical_generalization": False}


def save_failure(output, report, exc):
    failure = dict(status="failed", error_type=type(exc).__name__, error=str(exc))
    try:
        with (Path(output)/"failure.json").open("x") as stream:
            json.dump(failure, stream, allow_nan=False)
    except BaseException as secondary:
        print(f"Failure record write failed: {secondary}; original: {exc}", file=sys.stderr, flush=True)
    try:
        h.write_json(Path(output)/"result.json", {**report, **failure})
    except BaseException as secondary:
        print(f"Partial report write failed: {secondary}; original: {exc}", file=sys.stderr, flush=True)