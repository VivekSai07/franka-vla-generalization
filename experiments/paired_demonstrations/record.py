"""Exclusive paired collection and read-only configuration/old-geometry preflight."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np

from experiments.paired_demonstrations.plan import COLORS, IMAGE, VERSION, digest, make_plan, validate_plan
from data_collection.benchmark import runtime_versions, source_provenance, write_json

ROOT = Path(__file__).resolve().parents[2]

# Exact approved missing messages, not a root-level or generic ignore policy.
LEGACY_GEOMETRY_ALLOWLIST = {
    "datasets/franka_libero_stacking_v1: 2 episodes without full initial geometry":
        ("datasets/franka_libero_stacking_v1", 2),
    "datasets/franka_libero_stacking_v2: 1 episodes without full initial geometry":
        ("datasets/franka_libero_stacking_v2", 1),
}


def historical_geometry_exceptions(files, missing):
    """Describe only observed approved gaps, using hashes already in the inventory."""
    exceptions = []
    for message, (dataset, count) in LEGACY_GEOMETRY_ALLOWLIST.items():
        if message not in missing:
            continue
        info = dataset + "/meta/info.json"
        sha256 = files.get(info)
        if (not isinstance(sha256, str) or len(sha256) != 64
                or any(char not in "0123456789abcdef" for char in sha256)):
            raise ValueError("Approved legacy gap lacks its inventoried metadata SHA256")
        exceptions.append({"dataset": dataset, "episode_count": count,
            "missing_message": message, "reason": "userapproved20260917",
            "geometry_checked": False, "excluded_from_new_train_eval": True,
            "files": {info: sha256}})
    return exceptions


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe_child(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if Path(relative).is_absolute() or root not in path.resolve().parents or path.is_symlink():
        raise ValueError("Artifact path escapes its exclusive root")
    return path


def cube_digest(model, qpos):
    qpos = np.asarray(qpos, dtype="<f8")
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("Invalid full initial qpos")
    cubes = []
    for color in COLORS:
        body = model.body(f"target_cube_{color}").id
        joint = model.body_jntadr[body]
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Cube must have a free joint")
        address = model.jnt_qposadr[joint]
        cubes.extend(qpos[address:address + 7])
    return hashlib.sha256(np.asarray(cubes, dtype="<f8").tobytes()).hexdigest()


def prior_inventory(workspace, model, *, current_root=None):
    """Read qpos only: never restore/run any old bank, including sealed final scenes.

    Include incomplete datasets and paired reset sets, excluding only current_root.
    Missing geometry stays visible; only the exact approved legacy gaps may pass.
    Never invent geometry from seeds or images, or excuse other integrity errors.
    """
    workspace = Path(workspace).resolve()
    if current_root is not None:
        current_root = Path(current_root).resolve()
        if current_root.parent != workspace / "outputs/paired_demonstrations":
            raise ValueError("Current root must be a direct child of outputs/paired_demonstrations")
    files, geometries, missing = {}, set(), []
    if not (workspace / "datasets").is_dir() or not (workspace / "outputs").is_dir():
        raise ValueError("Prior workspace must expose datasets and outputs")

    def bind(path):
        files[str(path.relative_to(workspace))] = file_hash(path)

    def scan(manifest_path, key, artifact_dir=None):
        bind(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        entries = manifest[key]
        indexed = set()
        for entry in entries:
            path = safe_child(manifest_path.parent, entry["file"])
            if artifact_dir is not None and path.parent != artifact_dir:
                raise ValueError("Historical reset artifact must be in its states directory")
            if path in indexed:
                raise ValueError("Duplicate historical artifact manifest entry")
            indexed.add(path)
            bind(path)
            if file_hash(path) != entry["sha256"]:
                raise ValueError(f"Prior artifact hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as trace:
                qpos = trace["qpos"]
                geometries.add(cube_digest(model, qpos[0] if qpos.ndim == 2 else qpos))
        if indexed != set((artifact_dir or manifest_path.parent).glob("*.npz")):
            raise ValueError("Unindexed or missing historical initial-state artifact")
        return len(entries)

    for dataset in sorted((workspace / "datasets").iterdir()):
        if not dataset.is_dir():
            continue
        info = dataset / "meta/info.json"
        manifest = dataset / "replay/manifest.json"
        if not info.exists():
            missing.append(str(dataset.relative_to(workspace)) + ": missing metadata/initial geometry")
            continue
        bind(info)
        count = json.loads(info.read_text())["total_episodes"]
        if not manifest.exists():
            missing.append(f"{dataset.relative_to(workspace)}: {count} episodes without full initial geometry")
        elif scan(manifest, "episodes") != count:
            missing.append(f"{dataset.relative_to(workspace)}: metadata/replay count mismatch")
    for bank in sorted((workspace / "outputs").glob("**/*scenes")):
        if current_root is not None and (bank == current_root or current_root in bank.parents):
            continue
        if bank.is_dir():
            manifest = bank / "manifest.json"
            if not manifest.exists():
                missing.append(f"{bank.relative_to(workspace)}: missing scene manifest")
            else:
                scan(manifest, "scenes")
    for paired in sorted((workspace / "outputs/paired_demonstrations").glob("*")):
        if not paired.is_dir() or paired == current_root:
            continue
        manifest = paired / "manifest.json"
        if not manifest.exists():
            missing.append(f"{paired.relative_to(workspace)}: missing paired reset manifest")
        else:
            scan(manifest, "states", paired / "states")
    return {"files": files, "geometry_hashes": sorted(geometries), "missing": missing,
            "historical_geometry_exceptions": historical_geometry_exceptions(files, missing),
            "scope": "all dataset directories, all *scenes manifests and prior paired reset sets under outputs; current root excluded",
            "final_states_restored": False}


def require_inventory(inventory):
    missing = inventory["missing"]
    unknown = [message for message in missing if message not in LEGACY_GEOMETRY_ALLOWLIST]
    if unknown or len(missing) != len(set(missing)):
        raise ValueError("Historical cube geometry coverage incomplete: " + "; ".join(missing))
    expected = historical_geometry_exceptions(inventory["files"], missing)
    if digest(inventory.get("historical_geometry_exceptions")) != digest(expected):
        raise ValueError("Historical geometry exception evidence missing or changed")
    if not inventory["files"] or not inventory["geometry_hashes"]:
        raise ValueError("Empty prior geometry inventory")


def reject_duplicate(geometry_hash, occupied):
    if geometry_hash in occupied:
        raise ValueError("Duplicate cube initialization; no resampling permitted")
    occupied.add(geometry_hash)


def check_output(output, workspace):
    output, workspace = Path(output).resolve(), Path(workspace).resolve()
    # One explicitly designated new namespace; cannot nest inside an old run.
    if output.parent != workspace / "outputs/paired_demonstrations":
        raise ValueError("New root must be a direct child of outputs/paired_demonstrations")
    if output.exists():
        raise FileExistsError(f"No overwrite/resume: {output}")
    return output


def run_schedule(plan, states, execute, save_attempt):
    validate_plan(plan)
    if set(states) != {layout["layout_id"] for layout in plan["layouts"]}:
        raise ValueError("All 18 reset states must exist before any expert execution")
    results = []
    for episode in plan["episodes"]:
        try:
            outcome = execute(episode, states[episode["layout_id"]].copy())
        except Exception as exc:
            save_attempt({**episode, "success": False, "reason": str(exc), "error_type": type(exc).__name__})
            raise
        record = {**episode, **outcome}
        save_attempt(record)
        if outcome.get("success") is not True:
            raise RuntimeError("First expert task failure; batch incomplete, no retry/resume")
        results.append(record)
    return results


def provenance():
    from experiments.paired_demonstrations.expert_v1 import BASE_SHA256
    sources = source_provenance(ROOT)
    if sources["data_collection/collect_libero_dataset.py"] != BASE_SHA256:
        raise ValueError("Versioned expert base collector changed; review a new adapter version")
    for path in sorted(Path(__file__).parent.glob("*.py")):
        sources[str(path.relative_to(ROOT))] = file_hash(path)
    spec = ROOT / "docs/superpowers/specs/2026-09-17-paired-demonstration-learning-design.md"
    sources[str(spec.relative_to(ROOT))] = file_hash(spec)
    return {"sources": sources, "versions": runtime_versions(), "python": platform.python_version(),
            "required_image": IMAGE, "renderer": os.environ.get("MUJOCO_GL", "unset")}


def preflight(workspace=ROOT):
    """No renderer, reset generation, IK trajectory or dataset creation."""
    from experiments.paired_demonstrations.expert_v1 import Expert
    from experiments.paired_demonstrations.media import ENCODING
    expert = Expert()
    if expert.model.nq != 37 or expert.model.nu != 8 or expert.robot.model.nq != 9:
        raise ValueError("Unexpected robot/cube action-state schema")
    inventory = prior_inventory(workspace, expert.model)
    inventory_error = None
    try:
        require_inventory(inventory)
    except ValueError as exc:
        inventory_error = str(exc)
    return {"configuration_status": "passed", "collection_status": "blocked" if inventory_error else "not_started",
            "inventory_error": inventory_error,
            "historical_geometry_exceptions": inventory["historical_geometry_exceptions"],
            "plan_digest": digest(make_plan()), "configuration": expert.config,
            "provenance": provenance(), "encoding": ENCODING,
            "historical_geometry_count": len(inventory["geometry_hashes"]),
            "historical_files": len(inventory["files"]), "missing_geometry": inventory["missing"],
            "expert_trajectories_executed": 0, "training_ready": False}


def collect(output, workspace=ROOT, *, reserve_bytes=30 * 1024**3):
    """Single-use orchestration. CLI never exposes a disk reserve override."""
    from experiments.paired_demonstrations.expert_v1 import Expert, initialize_layout
    from experiments.paired_demonstrations.media import PairedDataset, FEATURES, ENCODING
    output = check_output(output, workspace)
    expert = Expert()
    bound = provenance()
    inventory = prior_inventory(workspace, expert.model, current_root=output)
    require_inventory(inventory)  # Block unknown historical geometry before writing or settling.
    if shutil.disk_usage(workspace).free < reserve_bytes:
        raise ValueError("At least 30 GiB free space required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "states").mkdir()
    (output / "transcripts").mkdir()
    plan = make_plan()
    write_json(output / "plan.json", plan)
    write_json(output / "prior_inventory.json", inventory)
    manifest = {"schema": VERSION, "status": "preparing", "plan": plan, "plan_digest": digest(plan),
                "provenance": bound, "configuration": expert.config, "encoding": ENCODING,
                "prior_workspace": str(Path(workspace).resolve()), "inventory_digest": digest(inventory),
                "historical_geometry_exceptions": inventory["historical_geometry_exceptions"],
                "states": [], "episodes": [], "training_ready": False,
                "selection_caveat": "No resampling; completion is conditional on both directions succeeding on every planned layout."}
    write_json(output / "manifest.json", manifest)
    dataset = None
    started = time.monotonic()
    try:
        occupied = set(inventory["geometry_hashes"])
        states = {}
        for layout in plan["layouts"]:
            arrays = initialize_layout(expert.model, layout)
            geometry = cube_digest(expert.model, arrays["qpos"])
            reject_duplicate(geometry, occupied)
            path = output / "states" / f"layout_{layout['ordinal']:02d}.npz"
            np.savez_compressed(path, **arrays)
            manifest["states"].append({**layout, "file": str(path.relative_to(output)),
                                       "sha256": file_hash(path), "cube_geometry_sha256": geometry})
            states[layout["layout_id"]] = arrays["initial_state"].copy()
            write_json(output / "manifest.json", manifest)
        # The complete seed/state list exists before creating any expert frames.
        manifest["status"] = "recording"
        write_json(output / "manifest.json", manifest)
        dataset = PairedDataset.create(repo_id="dataset", root=output / "dataset", fps=30, features=FEATURES,
                           vcodec=ENCODING["vcodec"], encoder_threads=ENCODING["encoder_threads"])
        frozen_by_id = {entry["layout_id"]: entry for entry in manifest["states"]}

        def execute(episode, initial):
            if provenance() != bound:
                raise ValueError("Source/runtime changed before the next expert; stop immediately")
            frozen = frozen_by_id[episode["layout_id"]]
            state_path = safe_child(output, frozen["file"])
            if file_hash(state_path) != frozen["sha256"]:
                raise ValueError("Reset artifact changed before expert execution")
            with np.load(state_path, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(initial, loaded["initial_state"])
            ticks, outcome = [], None
            path = output / "transcripts" / f"episode_{episode['episode_index']:06d}.npz"
            wall_start = time.monotonic()

            def sink(frame, trace):
                ticks.append(trace)
                dataset.add_frame(frame)

            try:
                outcome = expert.run(initial, episode, sink)
            finally:
                arrays = {key: np.asarray([tick[key] for tick in ticks]) for key in ticks[0]} if ticks else {}
                arrays["initial_state"] = initial
                if outcome is not None:
                    arrays["final_qpos"] = outcome.pop("final_qpos")
                    arrays["final_state"] = outcome.pop("final_state")
                np.savez_compressed(path, **arrays)
            outcome.update(file=str(path.relative_to(output)), sha256=file_hash(path),
                           state_file=frozen["file"], state_sha256=frozen["sha256"],
                           cube_geometry_sha256=frozen["cube_geometry_sha256"],
                           expert_wall_seconds=time.monotonic() - wall_start)
            encoding_start = time.monotonic()
            if outcome["success"]:
                if outcome["frames"] != plan["expected_episode_frames"]:
                    raise ValueError("Expert duration/frame count differs from original 528-frame timing")
                dataset.save_episode()
            outcome["encoding_wall_seconds"] = time.monotonic() - encoding_start
            outcome["wall_seconds"] = time.monotonic() - wall_start
            return outcome

        def save_attempt(entry):
            with (output / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if entry["success"]:
                manifest["episodes"].append(entry)
            write_json(output / "manifest.json", manifest)
            print(f"PAIRED {entry['episode_index'] + 1}/36 {entry['episode_id']}: "
                  f"success={entry['success']} wall_seconds={entry.get('wall_seconds')}", flush=True)

        run_schedule(plan, states, execute, save_attempt)
        dataset.finalize()
        dataset = None
        if bound != provenance() or inventory != prior_inventory(workspace, expert.model, current_root=output):
            raise ValueError("Sources/runtime/prior geometry inventory changed during collection")
        manifest.update(status="complete", collection_wall_seconds=time.monotonic() - started)
        write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        if dataset is not None:
            # Close valid parquet footers without accepting/clearing a partial episode.
            dataset.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Configuration and historical geometry audit only")
    parser.add_argument("--root", type=Path, help="New direct child of outputs/paired_demonstrations")
    args = parser.parse_args()
    if args.preflight:
        if args.root is not None:
            parser.error("Preflight never creates a dataset; omit --root")
        result = preflight()
        print(json.dumps(result, indent=2))
        return 2 if result["collection_status"] == "blocked" else 0
    if args.root is None:
        parser.error("Collection requires an explicit fresh --root")
    collect(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())