"""Build, merge, report, seal and verify the S0 sidecar. Robosuite image (replay needs the P2 scene).

Layout: `build_shard` writes `<work>/shard_KK/` (episodes `K::shards`, `provenance.json`, `runtime.json`, `done.json`);
`merge` verifies every shard against the current source and hard-links the episodes into `<out>/episodes/` with
`index.json`; `report` writes `report.json` from the sidecar alone; `seal` writes `seal.json`; `verify` recomputes it.
Nothing is ever overwritten, and nothing here writes under the sealed P2 root.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np

from experiments.vla_inputs_signals import contract

SEGMENTS = ("before_close", "closing", "closed_and_lifted", "after_release")
FORCE_NAMES = ("fx", "fy", "fz", "tx", "ty", "tz", "finger_left", "finger_right")
LIFT_M = 0.02
GRIPPER_COLUMN = 7
KEYS = ("episode_index", "frame_index", "wrist_wrench", "finger_force", "target_box", "props_box", "prop_names",
        "segment", "phase_index", "object", "split")


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def runtime():
    return {"image": contract.IMAGE, "mujoco": importlib.metadata.version("mujoco"), "numpy": np.__version__,
            "robosuite": importlib.metadata.version("robosuite"), "python": platform.python_version()}


def _require_pinned_runtime():
    import mujoco
    if mujoco.__version__ != contract.MUJOCO_VERSION:
        raise SystemExit(f"MuJoCo {mujoco.__version__} is not the pinned {contract.MUJOCO_VERSION}; run in the robosuite image")


def segments_of(action, target_z):
    """0 before the first close, 1 closed until the target is `LIFT_M` above its start, 2 closed and lifted, 3 after the last close."""
    closed = np.nonzero(np.asarray(action)[:, GRIPPER_COLUMN] == 0.0)[0]
    if not len(closed):
        raise ValueError("the gripper never closes")
    first, last = int(closed[0]), int(closed[-1])
    lifted = np.nonzero(np.asarray(target_z)[first:last + 1] - target_z[0] > LIFT_M)[0]
    if not len(lifted):
        raise ValueError("the target is never lifted while closed")
    lift = first + int(lifted[0])
    tick = np.arange(len(action))
    return np.where(tick < first, 0, np.where(tick < lift, 1, np.where(tick <= last, 2, 3))).astype(np.int8)


def build_episode(index):
    """The sidecar arrays of one sealed episode, by exact replay; any replay difference raises `ReplayMismatch`."""
    from experiments.vla_inputs_signals import replay, signals
    sealed = replay.load_episode(index)
    names = replay.objects_of(index)
    target = sealed["episode"]["object"]
    column = names.index(target)
    wrench, force, boxes = [], [], []

    def on_frame(tick, model, data):
        wrench.append(signals.wrist_wrench(model, data))
        force.append(signals.finger_force(model, data))
        boxes.append(signals.props_box(model, data, names))

    result = replay.frames(index, on_frame)
    props_box = np.stack(boxes)
    return {
        "episode_index": np.array(index), "frame_index": np.arange(result["frames"]),
        "wrist_wrench": np.stack(wrench), "finger_force": np.stack(force),
        "target_box": props_box[:, column].copy(), "props_box": props_box, "prop_names": np.array(names),
        "segment": segments_of(sealed["action"], sealed["props_pose"][:, column, 2]),
        "phase_index": result["phase_index"], "object": np.array(target), "split": np.array(sealed["episode"]["split"]),
    }


def check_arrays(arrays, index, frames):
    """Raises ValueError unless `arrays` is a complete, finite, correctly identified episode of `frames` frames."""
    if set(arrays) != set(KEYS):
        raise ValueError(f"episode {index}: keys {sorted(arrays)} differ from {sorted(KEYS)}")
    n = len(arrays["prop_names"])
    shapes = {"frame_index": (frames,), "wrist_wrench": (frames, 6), "finger_force": (frames, 2),
              "target_box": (frames, 2, 5), "props_box": (frames, n, 2, 5), "segment": (frames,), "phase_index": (frames,)}
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"episode {index}: {key} has shape {arrays[key].shape}, expected {shape}")
    if int(arrays["episode_index"]) != index:
        raise ValueError(f"episode {index}: file says episode {int(arrays['episode_index'])}")
    if not np.array_equal(arrays["frame_index"], np.arange(frames)):
        raise ValueError(f"episode {index}: frame_index is not 0..{frames - 1}")
    for key in ("wrist_wrench", "finger_force", "target_box", "props_box"):
        if arrays[key].dtype != np.float64 or not np.isfinite(arrays[key]).all():
            raise ValueError(f"episode {index}: {key} is not finite float64")
    if str(arrays["object"]) not in [str(x) for x in arrays["prop_names"]]:
        raise ValueError(f"episode {index}: target {arrays['object']} is not among the props")
    segment = arrays["segment"]
    if segment.min() < 0 or segment.max() > 3 or (np.diff(segment) < 0).any():
        raise ValueError(f"episode {index}: segments are not monotone in 0..3")


def write_episode(path, arrays):
    """Write `path` atomically; `FileExistsError` if it exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _read_episode(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def shard_dir(work, k):
    return Path(work) / f"shard_{k:02d}"


def build_shard(work, k, shards, episodes=contract.EPISODES, frames=contract.FRAMES, builder=None):
    """Replay episodes `k::shards` into `<work>/shard_KK/`. The shard directory must not exist."""
    _require_pinned_runtime()
    contract.check_p2_frozen()
    builder = builder or build_episode
    shard = shard_dir(work, k)
    shard.mkdir(parents=True)
    done = {}
    for index in range(k, episodes, shards):
        arrays = builder(index)
        check_arrays(arrays, index, frames)
        name = f"episode_{index:03d}.npz"
        write_episode(shard / "episodes" / name, arrays)
        done[name] = sha256_file(shard / "episodes" / name)
        print(f"shard {k}: episode {index} done", flush=True)
    _write_json(shard / "provenance.json", contract.provenance())
    _write_json(shard / "runtime.json", runtime())
    _write_json(shard / "done.json", {"shard": k, "shards": shards, "frames": frames, "sha256": done})
    return done


def merge(work, out, shards=contract.SHARDS, episodes=contract.EPISODES, frames=contract.FRAMES):
    """Verify every shard against the current source and link the episodes into `out`."""
    out, provenance, index = Path(out), contract.provenance(), {}
    runtimes = []
    for k in range(shards):
        shard = shard_dir(work, k)
        done = json.loads((shard / "done.json").read_text())
        expected = [f"episode_{i:03d}.npz" for i in range(k, episodes, shards)]
        present = sorted(p.name for p in (shard / "episodes").glob("*.npz"))
        if (done["shard"], done["shards"], done["frames"]) != (k, shards, frames) or present != expected or sorted(done["sha256"]) != expected:
            raise ValueError(f"shard {k} is incomplete or is not shard {k} of {shards}")
        if json.loads((shard / "provenance.json").read_text()) != provenance:
            raise ValueError(f"shard {k} was built from different source than the current one")
        for name in expected:
            if sha256_file(shard / "episodes" / name) != done["sha256"][name]:
                raise ValueError(f"shard {k}: {name} changed since it was written")
        index.update(done["sha256"])
        runtimes.append(json.loads((shard / "runtime.json").read_text()))
    if any(r != runtimes[0] for r in runtimes):
        raise ValueError("shards were built under different runtimes")
    if sorted(index) != [f"episode_{i:03d}.npz" for i in range(episodes)]:
        raise ValueError("the shards do not cover every episode exactly once")
    out.mkdir(parents=True)
    (out / "episodes").mkdir()
    for name in sorted(index):
        os.link(shard_dir(work, int(name[8:11]) % shards) / "episodes" / name, out / "episodes" / name)
    _write_json(out / "index.json", index)
    _write_json(out / "provenance.json", provenance)
    _write_json(out / "runtime.json", runtimes[0])
    return index


def _validate(out, episodes, frames):
    out = Path(out)
    names = sorted(p.name for p in (out / "episodes").iterdir())
    if names != [f"episode_{i:03d}.npz" for i in range(episodes)]:
        raise ValueError("episodes are missing, duplicated or unexpected")
    index = json.loads((out / "index.json").read_text())
    for i, name in enumerate(names):
        if sha256_file(out / "episodes" / name) != index.get(name):
            raise ValueError(f"{name} differs from index.json")
        check_arrays(_read_episode(out / "episodes" / name), i, frames)
    return episodes * frames


def _stats(x):
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "min": x.min(0).tolist(), "max": x.max(0).tolist(),
            "p0.1": np.percentile(x, 0.1, axis=0).tolist(), "p99.9": np.percentile(x, 99.9, axis=0).tolist()}


def report(out):
    """Descriptive statistics from the demonstrations alone; the clip percentiles and z-scores use training rows only."""
    out = Path(out)
    episodes = [_read_episode(p) for p in sorted((out / "episodes").glob("*.npz"))]
    forces = lambda e: np.concatenate([e["wrist_wrench"], e["finger_force"]], axis=1)
    train = np.concatenate([forces(e) for e in episodes if str(e["split"]) == "train"])
    cells = {}
    for e in episodes:
        for s, name in enumerate(SEGMENTS):
            mask = e["segment"] == s
            if mask.any():
                cells.setdefault(str(e["object"]), {}).setdefault(name, []).append((e, mask))
    by_cell = {}
    for obj, segments in sorted(cells.items()):
        for name, parts in segments.items():
            wrench = np.concatenate([e["wrist_wrench"][m] for e, m in parts])
            finger = np.concatenate([e["finger_force"][m] for e, m in parts])
            steps = [np.abs(np.diff(forces(e)[m], axis=0)) for e, m in parts if m.sum() > 1]
            by_cell.setdefault(obj, {})[name] = {
                "frames": int(len(wrench)), "episodes": len(parts), "wrist_wrench": _stats(wrench), "finger_force": _stats(finger),
                "frame_to_frame_abs_mean": np.concatenate(steps).mean(0).tolist() if steps else None,
                "front_cam_in_frame_rate": float(np.concatenate([e["target_box"][m][:, 0, 4] for e, m in parts]).mean()),
                "gripper_cam_in_frame_rate": float(np.concatenate([e["target_box"][m][:, 1, 4] for e, m in parts]).mean()),
            }
    result = {
        "segments": list(SEGMENTS), "force_names": list(FORCE_NAMES), "episodes": len(episodes), "training_frames": int(len(train)),
        "clip_percentiles": {"p0.1": np.percentile(train, 0.1, axis=0).tolist(), "p99.9": np.percentile(train, 99.9, axis=0).tolist()},
        "max_abs_z": np.abs((train - train.mean(0)) / train.std(0)).max(0).tolist(), "by_object_segment": by_cell,
    }
    _write_json(out / "report.json", result)
    return result


def _file_digest(path):
    return digest(json.loads(path.read_text())) if path.suffix == ".json" else sha256_file(path)


def fingerprint(root):
    """sha256 over {relative path: digest} of every file but `seal.json`; JSON by canonical form."""
    root, files = Path(root), {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in sealed root: {path.relative_to(root)}")
        if path.is_file() and path != root / "seal.json":
            files[str(path.relative_to(root))] = _file_digest(path)
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def seal(out, episodes=contract.EPISODES, frames=contract.FRAMES):
    out = Path(out)
    if (out / "seal.json").exists():
        raise FileExistsError(out / "seal.json")
    if not (out / "report.json").exists():
        raise ValueError("report.json must exist before the seal")
    contract.check_p2_frozen()
    provenance = contract.provenance()
    if json.loads((out / "provenance.json").read_text()) != provenance:
        raise ValueError("the sidecar was built from different source than the current one")
    total = _validate(out, episodes, frames)
    sealed = {"fingerprint": fingerprint(out), "p2_seal": contract.P2_SEAL, "provenance_sha256": digest(provenance),
              "runtime_sha256": digest(json.loads((out / "runtime.json").read_text())), "image": contract.IMAGE,
              "episodes": episodes, "frames": total}
    _write_json(out / "seal.json", sealed)
    return sealed


def verify(out, episodes=contract.EPISODES, frames=contract.FRAMES):
    """Recompute everything the seal claims; raises ValueError on the first difference."""
    out = Path(out)
    sealed = json.loads((out / "seal.json").read_text())
    if fingerprint(out) != sealed["fingerprint"]:
        raise ValueError("the fingerprint differs from the seal")
    contract.check_p2_frozen()
    provenance = contract.provenance()
    if (sealed["p2_seal"] != contract.P2_SEAL or sealed["provenance_sha256"] != digest(provenance)
            or json.loads((out / "provenance.json").read_text()) != provenance or sealed["image"] != contract.IMAGE):
        raise ValueError("the seal is bound to a different P2 seal, source or image than the current ones")
    if (sealed["episodes"], sealed["frames"]) != (episodes, _validate(out, episodes, frames)):
        raise ValueError("episode or frame count differs from the seal")
    return sealed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=int, metavar="K")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--report", metavar="OUT")
    parser.add_argument("--seal", metavar="OUT")
    parser.add_argument("--verify", metavar="OUT")
    parser.add_argument("--shards", type=int, default=contract.SHARDS)
    parser.add_argument("--work")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.build is not None:
        build_shard(args.work, args.build, args.shards)
    elif args.merge:
        merge(args.work, args.out, args.shards)
    elif args.report:
        report(args.report)
    elif args.seal:
        print(json.dumps(seal(args.seal), indent=2))
    elif args.verify:
        print("S0 VERIFIED", json.dumps(verify(args.verify)), flush=True)
    else:
        parser.error("one of --build/--merge/--report/--seal/--verify")


if __name__ == "__main__":
    main()
