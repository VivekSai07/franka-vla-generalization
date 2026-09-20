"""Final-5k evaluation on thirteen original resets; baseline physics/score unchanged."""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from data_collection.benchmark import capture_state, restore_state, restore_timing
from data_collection.pilot_integrity import scene_digest, validate_scene_geometry
from validation import evaluate_pilot as baseline
from . import contract as c


def derive_initial(model, source, entry, output):
    """Explicit generic copy of paired derive_train_state, also permitting IDs300..303."""
    source, output = Path(source).resolve(), Path(output).resolve()
    path = (source/entry["file"]).resolve()
    i = entry["episode_index"]
    if (i not in c.TRAIN+c.VAL or entry["split"] != ("train" if i in c.TRAIN else "validation")
            or path.parent != source or source == output or source in output.parents
            or c.h.file_hash(path) != entry["sha256"]):
        raise ValueError("Original reset ID/split/path/SHA mismatch")
    data = mujoco.MjData(model)
    with np.load(path, allow_pickle=False) as saved:
        initial = saved["initial_state"].copy()
        if not np.isfinite(initial).all():
            raise ValueError("Nonfinite initial integration state")
        restore_state(model, data, initial)
        np.testing.assert_array_equal(capture_state(model, data), initial)
        qpos = data.qpos.copy()
        recorded = saved["initial_qpos"] if "initial_qpos" in saved else saved["qpos"]
        np.testing.assert_array_equal(qpos, recorded[0] if recorded.ndim == 2 else recorded)
    if scene_digest(qpos) != entry["scene_hash"]:
        raise ValueError("Original reset scene hash mismatch")
    validate_scene_geometry(model, data, entry["colors"])
    derived = output/f"initial_{i:06d}.npz"
    with derived.open("xb") as stream:
        np.savez_compressed(stream, initial_state=initial, qpos=qpos)
    identity = {k: entry[k] for k in ("episode_index", "slot_id", "split", "pick", "place", "colors", "seed", "scene_hash")}
    adapted = {**identity, "file": derived.name, "sha256": c.h.file_hash(derived)}
    baseline.restore_scene(model, output, adapted)
    return adapted, dict(source_path=str(path), source_sha256=entry["sha256"],
                         derived_path=str(derived), derived_sha256=adapted["sha256"])


def verify_checkpoint(output, gate, manifest):
    report = json.loads((output/"result.json").read_text())
    checkpoint = output/"checkpoints/005000"
    marker = json.loads((checkpoint/"checkpoint_result.json").read_text())
    expected = dict(status="complete", purpose="single_task", training_steps=5000,
                    completed_training_steps=5000, selected_checkpoint_update=5000,
                    final_data_binding_passed=True, dataset_root=str(c.DATA.resolve()),
                    dataset_fingerprint=gate["dataset_fingerprint"], timing=manifest["timing"],
                    source_hashes=c.sources(), train_episodes=list(c.TRAIN), validation_episodes=list(c.VAL),
                    training_frames=4752, valid_training_chunk_starts=4311)
    if (any(report.get(k) != v for k, v in expected.items())
            or marker.get("status") != "complete" or marker.get("update") != 5000
            or marker.get("reload_fp32_weights_exact") is not True
            or not 0 <= marker.get("reload_max_action_error", float("inf")) <= 1e-5
            or [m for m in report["checkpoints"] if m["update"] == 5000] != [marker]
            or marker["model_sha256"] != c.h.file_hash(checkpoint/"model.safetensors")
            or report["final_checkpoint_files"] != c.inventory(checkpoint)
            or report["asset_files"] != c.inventory(c.ROOT/".cache/benchmark_models")
            or report["training_stats_sha256"] != c.h.file_hash(output/"training_stats.json")):
        raise ValueError("Complete final-5000 training/checkpoint/source binding failed")
    config = json.loads((checkpoint/"config.json").read_text())
    if (config != json.loads((output/"initial_config/config.json").read_text())
            or (config["chunk_size"], config["n_action_steps"], config["n_obs_steps"], config["type"]) != (50, 10, 1, "smolvla")
            or {k: tuple(v["shape"]) for k, v in config["input_features"].items()} != c.OBS
            or {k: v["shape"] for k, v in config["output_features"].items()} != {"action": [8]}):
        raise ValueError("Saved policy schema/config mismatch")
    return checkpoint


def training_bytes(output):
    # Do not repeatedly hash growing evaluation videos just to exclude them.
    return {str(p.relative_to(output)): c.h.file_hash(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p.relative_to(output).parts[0] != "evaluation"}


def run(output):
    from lerobot.policies.factory import make_pre_post_processors
    directory = output/"evaluation"
    directory.mkdir(exist_ok=False)
    report = dict(status="preparing", episodes=[], max_control_ticks=1200, max_sim_seconds=40,
                  strict_score_ticks=30, seed=c.SEED, resume=False, retries=False, **c.summarize([]))
    before = training_bytes(output)
    try:
        manifest = c.h.require_complete(c.DATA)
        gate = c.h.require_replay_pass(c.DATA, c.ROOT)
        checkpoint = verify_checkpoint(output, gate, manifest)
        scene = (c.ROOT/manifest["scene_path"]).resolve()
        if c.ROOT not in scene.parents or baseline.SEED != 20260916:
            raise ValueError("Scene path or baseline seed changed")
        model = mujoco.MjModel.from_xml_path(str(scene))
        if restore_timing(model, manifest)["substeps_per_control"] != 20:
            raise ValueError("Expected baseline physics timing")
        initial = directory/"initial_states"
        initial.mkdir()
        planned = []
        for entry, ordinal, seed in c.trials(manifest):
            adapted, evidence = derive_initial(model, c.DATA/"replay", entry, initial)
            planned.append(dict(entry=adapted, provenance=evidence, ordinal=ordinal, rollout_seed=seed))
        resets = c.inventory(initial)
        report.update(trials=planned, training_files=before, status="running")
        c.h.write_json(directory/"result.json", report)
        torch.use_deterministic_algorithms(True)
        policy = c.h.load_fp32_checkpoint(checkpoint)
        policy.eval()
        for p in policy.parameters():
            p.requires_grad_(False)
        pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
        for trial in planned:
            folder = directory/f"episode_{trial['entry']['episode_index']:06d}"
            folder.mkdir(exist_ok=False)
            report["active_episode"] = trial["entry"]["episode_index"]
            c.h.write_json(directory/"result.json", report)
            row = baseline.run_trial(policy, pre, post, model, manifest, trial["entry"], initial, folder, trial["ordinal"])
            if row["ordinal"] != trial["ordinal"] or row["rollout_seed"] != trial["rollout_seed"] or row["outcome"] == "nonfinite_action":
                raise ValueError("Seed/ordinal mismatch or nonfinite policy action; no retries")
            report["episodes"].append(row)
            report.update(c.summarize(report["episodes"]))
            c.h.write_json(directory/"result.json", report)
            print("EVAL", json.dumps({k: report[k] for k in ("finished_trials", "seen", "held_out")}), flush=True)
        after = c.h.require_replay_pass(c.DATA, c.ROOT)
        verify_checkpoint(output, after, c.h.require_complete(c.DATA))
        if gate != after or c.inventory(initial) != resets or report["finished_trials"] != 13:
            raise ValueError("Evaluation completion/reset/data binding failed")
        report.update(status="complete", active_episode=None, training_bytes_unchanged=True)
    except BaseException as exc:
        c.save_failure(directory, report, exc)
        raise
    finally:
        if training_bytes(output) != before:
            exc = ValueError("Training bytes changed during evaluation")
            c.save_failure(directory, report, exc)
            raise exc
    c.h.write_json(directory/"result.json", report)