"""Bounded 36 paired tuning + 12 training-scene diagnostics; never final scenes."""
import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_collection.benchmark import (COLORS, apply_action, capture_state, restore_state,
    stack_metrics, step_control, synchronize_observation, write_json)
from data_collection.pilot_integrity import scene_digest, validate_scene_geometry
from src.controllers.grasp_controller import GraspController
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF
from validation import evaluate_pilot as baseline
from validation.rollout_diagnostics import bound_action


CONDITIONS = ("original", "reversed", "seedcontrol", "train")
CHOICES = ("first_bilateral_choice", "first_sustained_choice", "first_lifted_choice")
ACTION_ATOL = 1e-7
STATE_ATOL = 1e-8
GRASP_TICKS = 10
LIFT_M = .05
REQUIRED_ARRAYS = ("raw_action", "applied_action", "state", "sim_time", "post_state",
                   "post_sim_time", "initial_integration_state", "terminal_integration_state",
                   "terminal_state", "terminal_qpos", "terminal_qvel", "terminal_sim_time")


def select_trials(manifest, tuning):
    """Outcome-independent selection; callers must first pass baseline preflight."""
    if len(tuning) != 120 or any(e["split"] != "tuning" for e in tuning):
        raise ValueError("Expected 120 tuning entries, never final")
    episodes = manifest["episodes"]
    train_ids, val_ids = manifest["splits"]["train"], manifest["splits"]["validation"]
    by_id = {e["episode_index"]: e for e in episodes}
    if (len(by_id) != len(episodes) or len(set(train_ids)) != len(train_ids)
            or set(train_ids) & set(val_ids)
            or any(i not in by_id or by_id[i]["split"] != "train" for i in train_ids)):
        raise ValueError("Invalid or overlapping train IDs/splits")
    pairs = set(itertools.permutations(COLORS, 2))

    def first_per_pair(indexed):
        chosen = {}
        for index, entry in indexed:
            pair = (entry["pick"], entry["place"])
            if len(entry["colors"]) == 2 and set(entry["colors"]) == set(pair):
                chosen.setdefault(pair, (index, entry))
        if set(chosen) != pairs:
            raise ValueError("Missing two-cube ordered pair; no fallback allowed")
        return sorted(chosen.values(), key=lambda item: item[0])

    trials = []
    for ordinal, entry in first_per_pair(enumerate(tuning)):
        for condition in CONDITIONS[:3]:
            copied = copy.deepcopy(entry)
            if condition == "reversed":
                copied["pick"], copied["place"] = copied["place"], copied["pick"]
            seed_ordinal = ordinal + int(condition == "seedcontrol")
            trials.append({"trial_id": len(trials), "condition": condition,
                           "baseline_ordinal": ordinal, "ordinal": seed_ordinal,
                           "rollout_seed": baseline.SEED + seed_ordinal, "entry": copied})
    for index, entry in first_per_pair((i, by_id[i]) for i in sorted(train_ids)):
        # Keep initialization identity, never demonstration outcome measurements.
        identity = {k: copy.deepcopy(entry[k]) for k in
                    ("slot_id", "split", "episode_index", "pick", "place", "colors",
                     "seed", "file", "sha256", "scene_hash")}
        trials.append({"trial_id": len(trials), "condition": "train", "baseline_ordinal": None,
                       "source_episode_index": index, "ordinal": index,
                       "rollout_seed": baseline.SEED + index, "entry": identity})
    return trials


def check_array(name, expected, actual, atol=STATE_ATOL):
    """Exact preferred, zero relative tolerance; reject shape/dtype/nonfinite drift."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError(f"{name}: shape/dtype mismatch")
    floating = np.issubdtype(expected.dtype, np.floating)
    if floating and name != "raw_action" and not (np.isfinite(expected).all() and np.isfinite(actual).all()):
        raise ValueError(f"{name}: nonfinite state/diagnostic")
    if np.array_equal(expected, actual) or (floating and np.array_equal(expected, actual, equal_nan=True)):
        return {"exact": True, "max_abs_error": 0.}
    if not floating or not np.allclose(expected, actual, rtol=0, atol=atol, equal_nan=True):
        raise ValueError(f"{name}: baseline/replay mismatch (atol={atol}, rtol=0)")
    finite = np.isfinite(expected) & np.isfinite(actual)
    error = float(np.max(np.abs(expected[finite].astype(np.float64) - actual[finite]), initial=0.))
    return {"exact": False, "max_abs_error": error}


def compare_traces(expected, actual):
    """Compare every NPZ array, including late/terminal integration and diagnostics."""
    if set(expected) != set(actual) or not set(REQUIRED_ARRAYS) <= set(expected):
        raise ValueError("Trace array keys missing or changed")
    results = {}
    for name in expected:
        results[name] = check_array(name, expected[name], actual[name],
                                   ACTION_ATOL if name in ("raw_action", "applied_action") else STATE_ATOL)
    return {"exact": all(r["exact"] for r in results.values()), "arrays": results,
            "action_atol": ACTION_ATOL, "state_atol": STATE_ATOL, "rtol": 0}


def derive_train_state(model, source, entry, output):
    """Adapt a hash-bound demo reset to restore_scene's schema in NEW output only."""
    source, output = Path(source).resolve(), Path(output).resolve()
    path = (source / entry["file"]).resolve()
    if (entry["split"] != "train" or path.parent != source
            or baseline.file_hash(path) != entry["sha256"]):
        raise ValueError("Train source path/hash/split mismatch")
    data = mujoco.MjData(model)
    with np.load(path, allow_pickle=False) as saved:
        initial = saved["initial_state"].copy()
        restore_state(model, data, initial)
        np.testing.assert_array_equal(capture_state(model, data), initial)
        qpos = data.qpos.copy()
        if "initial_qpos" in saved:
            np.testing.assert_array_equal(qpos, saved["initial_qpos"])
        elif "qpos" in saved:
            recorded = saved["qpos"]
            np.testing.assert_array_equal(qpos, recorded[0] if recorded.ndim == 2 else recorded)
    if scene_digest(qpos) != entry["scene_hash"]:
        raise ValueError("Train scene hash mismatch")
    validate_scene_geometry(model, data, entry["colors"])
    derived = output / f"train_{entry['episode_index']:06d}.npz"
    with derived.open("xb") as stream:
        np.savez_compressed(stream, initial_state=initial, qpos=qpos)
    digest = baseline.file_hash(derived)
    adapted = {**copy.deepcopy(entry), "file": derived.name, "sha256": digest}
    baseline.restore_scene(model, output, adapted)
    return adapted, {"source_path": str(path), "source_sha256": entry["sha256"],
                     "derived_path": str(derived), "derived_sha256": digest}


def streak_summary(mask, required):
    current = longest = 0
    first = None
    for tick, value in enumerate(mask):
        current = current + 1 if value else 0
        longest = max(longest, current)
        if current >= required and first is None:
            first = tick
    return longest, first


def contact_summary(colors, contacts, lift, released, pick, place):
    """Post-control tick events; tied first choices are null, not color-order wins."""
    per_cube = {}
    for index, color in enumerate(colors):
        left, right = contacts[:, index, 0], contacts[:, index, 1]
        any_touch, bilateral = left | right, left & right
        def first(mask):
            hits = np.flatnonzero(mask)
            return int(hits[0]) if len(hits) else None
        longest, sustained = streak_summary(bilateral, GRASP_TICKS)
        _, lifted = streak_summary(bilateral & (lift[:, index] > LIFT_M), GRASP_TICKS)
        per_cube[color] = {"left_contact_ticks": int(left.sum()), "right_contact_ticks": int(right.sum()),
            "contact_ticks": int(any_touch.sum()), "bilateral_ticks": int(bilateral.sum()),
            "first_contact_tick": first(any_touch), "first_bilateral_tick": first(bilateral),
            "first_sustained_tick": sustained, "first_lifted_tick": lifted,
            "longest_bilateral_streak": longest, "max_lift_m": float(np.max(lift[:, index], initial=0.))}
    result = {"per_cube": per_cube, "per_pair": {}}
    for label in ("bilateral", "sustained", "lifted"):
        times = {c: per_cube[c][f"first_{label}_tick"] for c in colors}
        valid = [t for t in times.values() if t is not None]
        tick = min(valid) if valid else None
        tied = [c for c in colors if times[c] == tick] if tick is not None else []
        result.update({f"first_{label}_choice": tied[0] if len(tied) == 1 else None,
                       f"first_{label}_colors": tied, f"first_{label}_tick": tick})
    for index, pair in enumerate(itertools.permutations(colors, 2)):
        longest, first = streak_summary(released[:, index], 30)
        result["per_pair"]["->".join(pair)] = {"longest_streak": longest, "first_stable_tick": first}
    result["wrong_pairs_stable_30_ticks"] = [pair for pair, info in result["per_pair"].items()
        if info["first_stable_tick"] is not None and pair != f"{pick}->{place}"]
    return result


def summarize(trials, rows):
    result = {"planned_trials": len(trials), "finished_trials": len(rows), "conditions": {}, "paired": {}}
    by_id = {r["trial_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate diagnostic trial result")
    for condition in CONDITIONS:
        group = [r for r in rows if r["condition"] == condition]
        n = len(group)
        successes = sum(r["task_success"] for r in group)
        unclipped = sum(r["unassisted_success"] for r in group)
        result["conditions"][condition] = {"trials": n, "successes": successes,
            "success_rate": successes / n if n else None, "unassisted_successes": unclipped,
            "unassisted_success_rate": unclipped / n if n else None,
            "wrong_stack_trials": sum(bool(r["diagnostics"]["wrong_pairs_stable_30_ticks"]) for r in group),
            "outcomes": dict(baseline.Counter(r["outcome"] for r in group)),
            "choices": {key: {"requested": sum(r["diagnostics"][key] == r["entry"]["pick"] for r in group),
                "null": sum(r["diagnostics"][key] is None for r in group),
                "requested_rate": sum(r["diagnostics"][key] == r["entry"]["pick"] for r in group) / n if n else None}
                for key in CHOICES}}
    originals = [t for t in trials if t["condition"] == "original"]
    for condition in ("reversed", "seedcontrol"):
        peers = {t["baseline_ordinal"]: t for t in trials if t["condition"] == condition}
        result["paired"][condition] = {}
        for key in CHOICES:
            counts = dict(pairs=len(originals), switches=0, both_null=0, one_null=0, missing_pairs=0)
            for original in originals:
                a = by_id.get(original["trial_id"])
                b = by_id.get(peers[original["baseline_ordinal"]]["trial_id"])
                if a is None or b is None:
                    counts["missing_pairs"] += 1
                    continue
                x, y = a["diagnostics"][key], b["diagnostics"][key]
                counts["switches"] += int(x is not None and y is not None and x != y)
                counts["both_null"] += int(x is None and y is None)
                counts["one_null"] += int((x is None) != (y is None))
            counts["switch_rate"] = counts["switches"] / len(originals) if originals else None
            result["paired"][condition][key] = counts
    return result


def replay_trial(model, manifest, episode, directory):
    """CPU saved-action replay only; no policy calls or additional rollout budget."""
    directory = Path(directory)
    if any((directory / name).exists() for name in ("contacts.npz", "diagnostics.json")):
        raise FileExistsError("Replay output already exists")
    with np.load(directory / "trace.npz", allow_pickle=False) as saved:
        trace = {k: saved[k] for k in saved.files}
    if not set(REQUIRED_ARRAYS) <= set(trace):
        raise ValueError("Incomplete replay trace")
    n, attempts = episode["completed_control_ticks"], episode["action_attempts"]
    nonfinite = episode["outcome"] == "nonfinite_action"
    if (not 0 <= n <= 1200 or attempts != n + int(nonfinite)
            or episode["actual_chunk_predictions"] != (attempts + 9) // 10
            or trace["raw_action"].shape != (attempts, 8)
            or trace["applied_action"].shape != (n, 8)
            or trace["state"].shape != (attempts, 9)
            or any(len(trace[k]) != n for k in ("post_state", "post_sim_time"))
            or trace["sim_time"].shape != (attempts,)
            or not np.isfinite(trace["raw_action"][:n]).all()
            or (nonfinite and np.isfinite(trace["raw_action"][-1]).all())):
        raise ValueError("Replay action/state counts or nonfinite outcome mismatch")
    data = mujoco.MjData(model)
    restore_state(model, data, trace["initial_integration_state"])
    colors = episode["colors"]
    controllers = [GraspController(model, data, target=f"target_cube_{c}") for c in colors]
    initial_z = np.array([data.xpos[g.target_id, 2] for g in controllers])
    pairs = list(itertools.permutations(colors, 2))
    requested = pairs.index((episode["pick"], episode["place"]))
    contacts = np.zeros((n, len(colors), 2), dtype=bool)
    lift = np.zeros((n, len(colors)))
    released = np.zeros((n, len(pairs)), dtype=bool)
    checks = []
    def check(name, expected, actual, atol=STATE_ATOL):
        checks.append(check_array(name, expected, actual, atol))
    check("initial_integration_state", trace["initial_integration_state"], capture_state(model, data))
    initial_time = float(data.time)
    for tick, action in enumerate(trace["applied_action"]):
        synchronize_observation(model, data)
        check("state", trace["state"][tick], data.qpos[:9])
        check("sim_time", trace["sim_time"][tick], np.asarray(data.time))
        bounded, _ = bound_action(trace["raw_action"][tick], model.actuator_ctrlrange[list(ACTIVE_JOINTS)])
        check("applied_action", bounded, action, ACTION_ATOL)
        apply_action(model, data, action, ACTIVE_JOINTS)
        if step_control(model, data, manifest["control_dt_requested"], ARM_DOF) != 20:
            raise ValueError("Replay physics substeps changed")
        check("post_state", trace["post_state"][tick], data.qpos[:9])
        check("post_sim_time", trace["post_sim_time"][tick], np.asarray(data.time))
        # step_control already forwards geometry without changing integrator state.
        for contact in data.contact[:data.ncon]:
            bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
            for index, grasp in enumerate(controllers):
                for side, finger in enumerate((grasp.left_id, grasp.right_id)):
                    contacts[tick, index, side] |= bodies == {finger, grasp.target_id}
        lift[tick] = [data.xpos[g.target_id, 2] - z for g, z in zip(controllers, initial_z)]
        released[tick] = [stack_metrics(model, data, *pair)["released_stack"] for pair in pairs]
        check("released_stack", trace["released_stack"][tick], released[tick, requested])
    if nonfinite:
        check("state", trace["state"][-1], data.qpos[:9])
        check("sim_time", trace["sim_time"][-1], np.asarray(data.time))
    for name, actual in (("terminal_integration_state", capture_state(model, data)),
                         ("terminal_state", data.qpos[:9]), ("terminal_qpos", data.qpos),
                         ("terminal_qvel", data.qvel), ("terminal_sim_time", np.asarray(data.time))):
        check(name, trace[name], actual)
    if not np.isclose(data.time - initial_time, n / 30, rtol=0, atol=STATE_ATOL):
        raise ValueError("Replay duration changed")
    result = contact_summary(colors, contacts, lift, released, episode["pick"], episode["place"])
    correct = result["per_pair"][f"{episode['pick']}->{episode['place']}"]["first_stable_tick"] is not None
    if correct != episode["task_success"]:
        raise ValueError("Replay requested-pair success mismatch")
    result.update(completed_control_ticks=n, replay_exact=all(c["exact"] for c in checks),
                  max_replay_error=max((c["max_abs_error"] for c in checks), default=0.))
    with (directory / "contacts.npz").open("xb") as stream:
        np.savez_compressed(stream, colors=np.asarray(colors), pairs=np.asarray(pairs),
                            finger_contacts=contacts, lift_m=lift, released_stack=released,
                            post_sim_time=trace["post_sim_time"])
    write_json(directory / "diagnostics.json", result)
    return result


def file_inventory(directory):
    """Hash every baseline artifact, including all videos, without loading models."""
    directory = Path(directory).resolve()
    inventory = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Artifact inventory must not contain symlinks")
        if path.is_file():
            inventory[str(path.relative_to(directory))] = baseline.file_hash(path)
    return inventory


def verify_baseline(directory, report, tuning, fingerprints):
    expected = {"status": "complete", "purpose": "pilot_tuning_only", "finished_trials": 120,
                "planned_trials": 120, "post_run_fingerprints_unchanged": True,
                "max_control_ticks": 1200, "max_sim_seconds": 40, "seed": baseline.SEED,
                "training_updates": 0, "retries": False, "resume": False, "final_scenes_evaluated": False}
    if (any(report.get(k) != v for k, v in expected.items()) or len(tuning) != 120
            or len(report["episodes"]) != 120 or fingerprints != report["fingerprints"]
            or any(report.get(k) != v for k, v in baseline.aggregate(report["episodes"]).items())):
        raise ValueError("Completed 120-trial baseline/fingerprints gate failed")
    for ordinal, (entry, row) in enumerate(zip(tuning, report["episodes"])):
        folder = Path(directory) / f"trial_{ordinal:03d}"
        instruction = f"pick up the {entry['pick']} cube and place it on top of the {entry['place']} cube"
        if (entry["split"] != "tuning" or any(row.get(k) != v for k, v in entry.items())
                or row["ordinal"] != ordinal or row["rollout_seed"] != baseline.SEED + ordinal
                or row["instruction"] != instruction or (row["chunk_size"], row["n_action_steps"]) != (50, 10)
                or not 0 <= row["completed_control_ticks"] <= 1200
                or row["actual_chunk_predictions"] != (row["action_attempts"] + 9) // 10
                or any(not (folder / name).is_file() or (folder / name).stat().st_size == 0
                       for name in ("episode.json", "trace.npz", "cameras.mp4"))
                or json.loads((folder / "episode.json").read_text()) != row):
            raise ValueError(f"Baseline episode/artifact binding failed: {ordinal}")


def run(args, report):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if os.environ.get("MUJOCO_GL") != "osmesa":
        raise ValueError("Use baseline MUJOCO_GL=osmesa environment before Python imports")
    before_files = file_inventory(args.baseline)
    report["baseline_files"] = before_files
    try:
        saved = json.loads((args.baseline / "result.json").read_text())
        root, checkpoint, bank = (Path(saved[k]).resolve() for k in ("dataset_root", "checkpoint", "scene_bank"))
        manifest, tuning, model, fingerprints = baseline.preflight(root, checkpoint, bank)
        verify_baseline(args.baseline, saved, tuning, fingerprints)
        report.update(fingerprints=fingerprints, timing=manifest["timing"],
                      baseline_result_sha256=before_files["result.json"])
        trials = select_trials(manifest, tuning)
        initial_dir = args.output / "initial_states"
        initial_dir.mkdir(exist_ok=False)
        for trial in trials:
            if trial["condition"] == "train":
                entry, provenance = derive_train_state(model, root / "replay", trial["entry"], initial_dir)
                trial.update(source_entry=trial["entry"], entry=entry, provenance=provenance, bank=str(initial_dir))
            else:
                trial["bank"] = str(bank)
        plan = {"trials": trials, "baseline": str(args.baseline), "fingerprints": fingerprints,
                "baseline_result_sha256": before_files["result.json"], "seed": baseline.SEED,
                "max_sim_seconds": 40, "max_control_ticks": 1200, "chunk_size": 50, "n_action_steps": 10,
                "action_atol": ACTION_ATOL, "state_atol": STATE_ATOL, "rtol": 0,
                "grasp_consecutive_ticks": GRASP_TICKS, "lift_threshold_m": LIFT_M,
                "stack_consecutive_ticks": 30, "first_choice": "bilateral; simultaneous ties are null",
                "sustained_choice": "10 consecutive bilateral post-control ticks; completion tick",
                "lifted_choice": "10 consecutive bilateral ticks each above initial Z + 0.05m",
                "paired_denominator": "all 12 layouts, null/missing are not switches",
                "train_seed_rule": "20260916 + source episode index", "resume": False, "retries": False}
        write_json(args.output / "plan.json", plan)
        plan_hash = baseline.file_hash(args.output / "plan.json")
        initial_hashes = file_inventory(initial_dir)
        source_hash = baseline.file_hash(Path(__file__))
        report.update(status="running", plan_sha256=plan_hash, probe_source_sha256=source_hash,
                      initial_state_files=initial_hashes, **summarize(trials, []))
        write_json(args.output / "result.json", report)
        from lerobot.policies.factory import make_pre_post_processors
        baseline.torch.set_num_threads(8)
        baseline.torch.backends.cudnn.benchmark = False
        baseline.torch.use_deterministic_algorithms(True)
        policy = baseline.load_fp32_checkpoint(checkpoint)
        policy.eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
        for trial in trials:
            folder = args.output / f"trial_{trial['trial_id']:03d}"
            folder.mkdir(exist_ok=False)
            report["active_trial"] = trial["trial_id"]
            write_json(args.output / "result.json", report)
            print(f"PROBE {trial['trial_id'] + 1}/48 {trial['condition']} seed={trial['rollout_seed']}", flush=True)
            episode = baseline.run_trial(policy, pre, post, model, manifest, trial["entry"],
                                         Path(trial["bank"]), folder, trial["ordinal"])
            row = {**trial, **episode}
            if trial["condition"] == "original":
                ordinal = trial["baseline_ordinal"]
                prior = saved["episodes"][ordinal]
                # Floating final diagnostics are compared in the full NPZ below.
                for key in ("ordinal", "rollout_seed", "instruction", "outcome", "action_attempts",
                            "completed_control_ticks", "actual_chunk_predictions", "task_success",
                            "unassisted_success", "success_streak", "chunk_size", "n_action_steps"):
                    if episode[key] != prior[key]:
                        raise ValueError(f"Original baseline metadata mismatch: {key}")
                with np.load(args.baseline / f"trial_{ordinal:03d}/trace.npz", allow_pickle=False) as expected, \
                        np.load(folder / "trace.npz", allow_pickle=False) as actual:
                    row["baseline_comparison"] = compare_traces(expected, actual)
                report["original_comparisons"] += 1
            row["diagnostics"] = replay_trial(model, manifest, episode, folder)
            report["replayed_trials"] += 1
            report["episodes"].append(row)
            report.update(summarize(trials, report["episodes"]))
            write_json(args.output / "result.json", report)
        _, _, _, after = baseline.preflight(root, checkpoint, bank)
        if (len(report["episodes"]) != 48 or report["replayed_trials"] != 48
                or report["original_comparisons"] != 12 or fingerprints != after
                or file_inventory(initial_dir) != initial_hashes
                or baseline.file_hash(args.output / "plan.json") != plan_hash
                or baseline.file_hash(Path(__file__)) != source_hash):
            raise ValueError("Incomplete diagnostic or changed post-run fingerprints")
    finally:
        unchanged = file_inventory(args.baseline) == before_files
        report["baseline_files_unchanged"] = unchanged
        if not unchanged:
            raise ValueError("Baseline files changed during diagnostic")
    report.update(status="complete", active_trial=None, post_run_fingerprints_unchanged=True)
    write_json(args.output / "result.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "output"):
        parser.add_argument("--" + name, type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args(argv)
    # Reject an output nested in immutable inputs BEFORE even making a directory.
    saved = json.loads((args.baseline / "result.json").read_text())
    protected = [args.baseline, Path(saved["dataset_root"]).resolve(), Path(saved["scene_bank"]).resolve(),
                 Path(saved["checkpoint"]).resolve().parent.parent, baseline.ROOT / ".cache"]
    if any(args.output == p or p in args.output.parents for p in protected):
        raise ValueError("Diagnostic output must not be inside baseline/data/bank/model inputs")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "preflight", "purpose": "paired_instruction_probe_48", "baseline": str(args.baseline),
              "episodes": [], "planned_trials": 48, "finished_trials": 0, "original_comparisons": 0,
              "replayed_trials": 0, "training_updates": 0, "resume": False, "retries": False,
              "final_scenes_evaluated": False, "post_run_fingerprints_unchanged": False}
    write_json(args.output / "result.json", report)
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(args.output / "result.json", report)
        raise


if __name__ == "__main__":
    main()