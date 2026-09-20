"""Independent persisted-data gates and exactly two saved-control replays.

Never calls an expert, renders an image, regenerates a trajectory or trains.
Only replay_controls advances physics; admission rechecks saved replay evidence.
"""
from collections import Counter
import hashlib
import json
from numbers import Real
from pathlib import Path

import av
import mujoco
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_collection.benchmark import apply_action, restore_state, stack_metrics, step_control, synchronize_observation
from data_collection.collect_libero_dataset import generate_task_states
from experiments.fixed_layout.contract import TASK, SOURCE_SHA256, MAX_SECONDS, make_plan, validate_plan
from experiments.fixed_layout.source import Expert, load_reset, provenance
from experiments.paired_demonstrations.media import CAMERAS, FEATURES, ENCODING, pixels_digest
from experiments.paired_demonstrations.validate import validate_trace
from validation.validate_pilot import validate_video_intervals
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF

MODES = ("reference", "persisted_float32")


def _json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def _equal(actual, expected, message, atol=None):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(message + " (shape)")
    ok = np.array_equal(actual, expected) if atol is None else np.allclose(actual, expected, rtol=0, atol=atol)
    if not ok:
        raise ValueError(message)


def _child(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    path = root / relative
    if relative.is_absolute() or root not in path.resolve().parents or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Artifact path escapes root or uses a symlink")
    return path


def fingerprint(root, check=lambda: None):
    """Bind ALL evidence, reports, raw RGB and metadata; only the final seal excluded."""
    from experiments.fixed_layout.record import file_hash
    root, files = Path(root), {}
    if root.is_symlink():
        raise ValueError("Symlink root")
    for path in sorted(root.rglob("*")):
        check()
        if path.is_symlink():
            raise ValueError("Symlink in dataset evidence")
        if path.is_file() and path != root / "seal.json":
            files[str(path.relative_to(root))] = file_hash(path, check)
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def read_rows(path, index):
    """Read parquet directly, never a cached HF dataset or writer's frame buffer."""
    table = pq.read_table(path)
    expected = {"action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"}
    if set(table.column_names) != expected or len(table) != 528:
        raise ValueError("Canonical persisted rows require 528 frames and no privileged fields")
    for key, size in (("action", 8), ("observation.state", 9)):
        dtype = table.schema.field(key).type
        if not pa.types.is_fixed_size_list(dtype) or dtype.list_size != size or dtype.value_type != pa.float32():
            raise ValueError("Persisted action/state must be unpadded float32")
    for key in expected:
        column = table[key]
        if column.null_count or (pa.types.is_fixed_size_list(column.type) and column.combine_chunks().values.null_count):
            raise ValueError("Null persisted row value")
    for key, wanted in (("frame_index", np.arange(528)), ("index", np.arange(528) + index * 528),
                        ("episode_index", np.full(528, index)), ("task_index", np.zeros(528, np.int64))):
        if table.schema.field(key).type != pa.int64():
            raise ValueError("Row indices must be int64")
        _equal(table[key].to_numpy(), wanted, "Persisted row/task indices differ")
    if table.schema.field("timestamp").type != pa.float32():
        raise ValueError("Persisted timestamps must be float32")
    actions = np.asarray(table["action"].to_pylist(), np.float32)
    states = np.asarray(table["observation.state"].to_pylist(), np.float32)
    times = table["timestamp"].to_numpy()
    if not all(np.isfinite(a).all() for a in (actions, states, times)):
        raise ValueError("Nonfinite persisted rows")
    _equal(times, np.arange(528) / 30, "Persisted timestamp discontinuity", 2e-6)
    return actions, states, times


def validate_video(path, intervals, check):
    check()
    for start, end, frames in intervals:
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
               or not np.isfinite(value) for value in (start, end)):
            raise ValueError("Video interval endpoints must be finite real scalars")
        if type(frames) is not int or frames <= 0 or start < 0 or end <= start:
            raise ValueError("Invalid video interval extent/frame count")
    count = 0
    with av.open(str(path)) as video:
        if len(video.streams) != 1 or len(video.streams.video) != 1:
            raise ValueError("Exactly one camera video stream, no audio")
        stream = video.streams.video[0]
        if (stream.codec.canonical_name != "av1" or stream.pix_fmt != "yuv420p"
                or float(stream.average_rate) != 30 or (stream.width, stream.height) != (512, 512)):
            raise ValueError("Expected 512 RGB AV1/yuv420p at 30 Hz")
        for tick, frame in enumerate(video.decode(stream)):
            check()
            if ((frame.width, frame.height) != (512, 512) or frame.pts is None
                    or abs(float(frame.pts * frame.time_base) - tick / 30) > 1e-5):
                raise ValueError("Decoded frame dimensions/timestamps mismatch")
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8:
                raise ValueError("Malformed decoded RGB")
            count += 1
            check()
    if count == 0:
        raise ValueError("Empty camera video")
    validate_video_intervals(intervals, count, 30)
    check()
    return count


def _longest(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def contact_gates(actions, post, initial_z, pick, place):
    if (pick, place) != ("red", "green"):
        raise ValueError("Wrong target/base roles")
    n = len(actions)
    if actions.shape != (n, 8) or not np.isfinite(actions).all() or not np.isfinite(initial_z):
        raise ValueError("Invalid contact-gate commands/reset")
    for key in ("pick_xyz", "place_xyz", "hand_xyz"):
        if post[key].shape != (n, 3) or not np.isfinite(post[key]).all():
            raise ValueError("Nonfinite or malformed actual post-step geometry")
    for key in ("left_contact", "right_contact"):
        if post[key].shape != (n,) or post[key].dtype != np.bool_:
            raise ValueError("Actual target finger contacts must be booleans")
    closed = actions[:, 7] <= 127
    bilateral = post["left_contact"] & post["right_contact"]
    delta = post["pick_xyz"] - post["place_xyz"]
    released = ((np.linalg.norm(delta[:, :2], axis=1) < .02) & (delta[:, 2] > .035)
                & (delta[:, 2] < .045) & (actions[:, 7] > 127)
                & ~post["left_contact"] & ~post["right_contact"])
    stable = 0
    for value in released:
        stable = stable + 1 if value else 0
    lift = float(np.max(post["pick_xyz"][bilateral, 2] - initial_z)) if bilateral.any() else 0.
    result = {"closed_ticks": _longest(closed), "bilateral_ticks": _longest(bilateral),
              "contact_lift_m": lift, "stable_ticks": stable,
              "minimum_hand_distance_m": float(np.min(np.linalg.norm(post["hand_xyz"] - post["pick_xyz"], axis=1))),
              "hand_distance_scope": "descriptive_only_hand_body_origin"}
    if result["closed_ticks"] < 30 or result["bilateral_ticks"] < 5 or lift <= .05 or stable < 30:
        raise ValueError("Closed/contact/lift/final released-stack admission failed: " + str(result))
    return result


def _observation(model, data):
    pick = model.body("target_cube_red").id
    place = model.body("target_cube_green").id
    left, right = model.body("left_finger").id, model.body("right_finger").id
    pairs = [{int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])}
             for c in data.contact[:data.ncon]]
    return {"qpos": data.qpos.copy(), "sim_time": np.asarray(data.time),
            "body_xpos": data.xpos.copy(), "camera_xpos": data.cam_xpos.copy(), "camera_xmat": data.cam_xmat.copy(),
            "pick_xyz": data.xpos[pick].copy(), "place_xyz": data.xpos[place].copy(),
            "hand_xyz": data.xpos[model.body("hand").id].copy(),
            "left_contact": np.asarray({left, pick} in pairs), "right_contact": np.asarray({right, pick} in pairs)}


def _at_qpos(model, qpos, time, ctrl):
    """Geometric contact reconstruction, NOT a physics step or a force measurement."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.time = float(time)
    data.ctrl[:] = ctrl
    synchronize_observation(model, data)
    return _observation(model, data)


def _stack(items):
    return {key: np.asarray([item[key] for item in items]) for key in items[0]}


def _original_post(expert, trace, terminal, check):
    """Post tick i is observed at tick i+1, or the returned terminal snapshot.

    The unchanged expert does not expose live MjData. Contacts are reconstructed
    from its actual saved qpos (not hand distance); reference replay independently
    measures post-step contacts and must match this original geometry.
    """
    post = []
    for i in range(528):
        check()
        last = i == 527
        qpos = terminal["final_qpos"] if last else trace["qpos"][i + 1]
        time = trace["sim_time"][i] + 1 / 30
        actual = _at_qpos(expert.model, qpos, time, trace["ctrl"][i])
        if not last:
            for key in ("body_xpos", "camera_xpos", "camera_xmat"):
                _equal(actual[key], trace[key][i + 1], "Original geometry/qpos mismatch", 1e-9)
        post.append(actual)
    return _stack(post)


def _validate_phases(trace, actions):
    phases = generate_task_states(.105, robot_speed=1.)
    phase, start = 0, trace["sim_time"][0]
    for tick, time in enumerate(trace["sim_time"]):
        if phase >= len(phases):
            raise ValueError("Extra frames after final expert phase")
        _equal(trace["phase_index"][tick], phase, "Expert phase order changed")
        _equal(trace["phase_time"][tick], time - start, "Expert phase clock changed", 1e-9)
        _equal(actions[tick, 7], phases[phase]["gripper"], "Expert gripper schedule changed")
        if time - start > phases[phase]["duration"]:
            phase += 1
            start = time
    if phase != len(phases):
        raise ValueError("Incomplete expert phases")


def load_attempt(path, episode, actions, states, times, expert, reset, check):
    path = Path(path)
    if _json(path / "slot.json") != episode or (path / "interrupted.json").exists():
        raise ValueError("Failed or changed attempt slot")
    outcome = _json(path / "outcome.json")
    if outcome.get("success") is not True or outcome.get("frames") != 528 or outcome.get("stable_ticks", 0) < 30:
        raise ValueError("Failed expert cannot be an imitation target")
    files = sorted((path / "ticks").iterdir())
    if [p.name for p in files] != [f"{i:06d}.npz" for i in range(528)]:
        raise ValueError("Missing/extra raw callback evidence")
    traces = []
    frame_keys = set(CAMERAS) | {"action", "observation.state", "task"}
    for i, p in enumerate(files):
        check()
        raw = _npz(p)
        if {key for key in raw if not key.startswith("trace/")} != frame_keys:
            raise ValueError("Unexpected original observation fields")
        if raw["task"].shape != () or raw["task"].item() != TASK:
            raise ValueError("Original instruction mismatch")
        for key, expected in (("action", actions[i]), ("observation.state", states[i])):
            if raw[key].dtype != np.float32:
                raise ValueError("Original policy rows must be float32")
            _equal(raw[key], expected, "Persisted rows differ from original observations/commands")
        trace = {key[6:]: value for key, value in raw.items() if key.startswith("trace/")}
        _equal(trace["image_sha256"], [pixels_digest(raw[key]) for key in CAMERAS], "Raw RGB hash mismatch")
        traces.append(trace)
    if any(set(t) != set(traces[0]) for t in traces):
        raise ValueError("Incomplete per-tick full trace")
    trace = _stack(traces)
    for key, shape in (("body_xpos", (528, expert.model.nbody, 3)),
                       ("camera_xpos", (528, expert.model.ncam, 3)),
                       ("camera_xmat", (528, expert.model.ncam, 9)), ("qpos", (528, expert.model.nq)),
                       ("ctrl", (528, 8))):
        if trace[key].shape != shape or trace[key].dtype != np.float64:
            raise ValueError("Trace geometry/control precision or shape changed")
    try:
        validate_trace(trace, actions, states, times, expert.bounds)
    except AssertionError as exc:
        raise ValueError("Trace/action/state/governor mismatch") from exc
    _validate_phases(trace, actions)
    for key in ("qpos", "body_xpos", "camera_xpos", "camera_xmat"):
        _equal(trace[key][0], reset[key], "Episode did not use exact source reset")
    terminal = _npz(path / "terminal.npz")
    if set(terminal) != {"final_qpos", "final_state"}:
        raise ValueError("Missing terminal evidence")
    for key, shape in (("final_qpos", reset["qpos"].shape), ("final_state", reset["initial_state"].shape)):
        if terminal[key].shape != shape or terminal[key].dtype != np.float64 or not np.isfinite(terminal[key]).all():
            raise ValueError("Malformed terminal state")
    data = mujoco.MjData(expert.model)
    restore_state(expert.model, data, terminal["final_state"])
    _equal(data.qpos, terminal["final_qpos"], "Terminal integration/qpos mismatch")
    _equal(data.time, trace["sim_time"][-1] + 1 / 30, "Terminal time mismatch", 1e-9)
    post = _original_post(expert, trace, terminal, check)
    gates = contact_gates(actions, post, reset["body_xpos"][expert.model.body("target_cube_red").id, 2], "red", "green")
    delta = post["pick_xyz"] - post["place_xyz"]
    released = ((np.linalg.norm(delta[:, :2], axis=1) < .02) & (delta[:, 2] > .035) & (delta[:, 2] < .045)
                & (actions[:, 7] > 127) & ~post["left_contact"] & ~post["right_contact"])
    _equal(released, trace["released_stack"], "Original released-stack trace is inconsistent")
    if gates["stable_ticks"] != outcome["stable_ticks"] or stack_metrics(expert.model, data, "red", "green") != outcome["final_metrics"]:
        raise ValueError("Expert outcome disagrees with actual terminal geometry")
    return trace, terminal, gates


def replay_controls(root, expert, reset, trace, actions, mode, check):
    """One control replay. An exclusive mode directory is the durable budget token."""
    from experiments.fixed_layout.record import write_json, write_npz
    if mode not in MODES:
        raise ValueError("Only the two approved saved-control replay modes exist")
    directory = Path(root) / mode
    directory.mkdir(exist_ok=False)
    check()
    data = mujoco.MjData(expert.model)
    restore_state(expert.model, data, reset["initial_state"].copy())
    try:
        for tick in range(528):
            check()
            synchronize_observation(expert.model, data)
            pre = _observation(expert.model, data)
            if mode == "reference":
                data.ctrl[:] = trace["ctrl"][tick]
            else:
                apply_action(expert.model, data, actions[tick], ACTIVE_JOINTS)
            check()
            substeps = step_control(expert.model, data, 1 / 30, ARM_DOF)
            post = _observation(expert.model, data)
            arrays = {"pre/" + k: v for k, v in pre.items()}
            arrays.update({"post/" + k: v for k, v in post.items()})
            arrays.update(ctrl=data.ctrl.copy(), substeps=np.asarray(substeps))
            write_npz(directory / f"{tick:06d}.npz", arrays)
            check()
        # Audit the actual saved replay evidence, not only accumulated counters.
        result = audit_replay(directory, expert, reset, trace, actions, mode, check)
        write_json(directory / "result.json", result)
        return result
    except BaseException as exc:
        try:
            write_json(directory / "failure.json", {"passed": False, "error": repr(exc)})
        except BaseException:
            pass  # Never replace the triggering failure with an evidence-write failure.
        raise


def audit_replay(directory, expert, reset, trace, actions, mode, check):
    directory = Path(directory)
    if (directory / "failure.json").exists() or sorted(p.name for p in directory.glob("*.npz")) != [f"{i:06d}.npz" for i in range(528)]:
        raise ValueError("Missing/duplicate/failed replay evidence")
    terminal = _npz(directory.parent.parent / "terminal.npz")
    post_rows, max_full, max_policy = [], 0., 0.
    previous_post = None
    for i in range(528):
        check()
        row = _npz(directory / f"{i:06d}.npz")
        expected_qpos = trace["qpos"][i + 1] if i < 527 else terminal["final_qpos"]
        wanted_ctrl = trace["ctrl"][i] if mode == "reference" else actions[i].astype(np.float64)
        _equal(row["ctrl"], wanted_ctrl, "Replay did not apply persisted controls")
        _equal(row["substeps"], 20, "Replay substep mismatch")
        for prefix, qpos, time in (("pre", trace["qpos"][i], trace["sim_time"][i]),
                                    ("post", expected_qpos, trace["sim_time"][i] + 1 / 30)):
            actual = {k[len(prefix) + 1:]: v for k, v in row.items() if k.startswith(prefix + "/")}
            if any(a.dtype.kind not in "bifu" or not np.isfinite(a).all() for a in actual.values()):
                raise ValueError("Nonfinite replay evidence")
            _equal(actual["sim_time"], time, "Replay clock drift", 1e-9)
            max_full = max(max_full, float(np.max(np.abs(actual["qpos"] - qpos))))
            max_policy = max(max_policy, float(np.max(np.abs(actual["qpos"][:9] - qpos[:9].astype(np.float32)))))
            geometry = _at_qpos(expert.model, actual["qpos"], actual["sim_time"], wanted_ctrl)
            if set(actual) != set(geometry):
                raise ValueError("Incomplete replay observation")
            for key in geometry:
                _equal(actual[key], geometry[key], "Replay geometry/contact evidence mismatch", 1e-9)
            if prefix == "pre":
                if previous_post is not None:
                    _equal(actual["qpos"], previous_post, "Discontinuous control replay")
                else:
                    _equal(actual["qpos"], reset["qpos"], "Replay reset mismatch")
                if mode == "reference":
                    for key in ("body_xpos", "camera_xpos", "camera_xmat"):
                        _equal(actual[key], trace[key][i], "Reference original geometry mismatch", 1e-9)
            else:
                post_rows.append(actual)
                previous_post = actual["qpos"]
    error, tolerance = (max_full, 1e-5) if mode == "reference" else (max_policy, 1e-3)
    if error > tolerance:
        raise ValueError(f"{mode} drift: {error} > {tolerance}")
    post = _stack(post_rows)
    gates = contact_gates(actions, post, reset["body_xpos"][expert.model.body("target_cube_red").id, 2], "red", "green")
    return {"mode": mode, "passed": True, "frames": 528, "max_full_qpos_error": max_full,
            "max_policy_state_error": max_policy, "tolerance": tolerance, "gates": gates}


def check_replay_coverage(results):
    if (any(type(r.get("episode")) is not int or type(r.get("mode")) is not str for r in results)
            or Counter((r["episode"], r["mode"]) for r in results) != Counter((i, m) for i in range(24) for m in MODES)
            or any(r.get("passed") is not True for r in results)):
        raise ValueError("Require exactly 24 x 2 unique successful control replays")


def duplicate_report(sequences):
    hashes, pairs, duplicates = [], [], []
    for i, (actions, states) in enumerate(sequences):
        hashes.append({"episode": i, "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                       "state_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
                       "sequence_sha256": hashlib.sha256(actions.tobytes() + states.tobytes()).hexdigest()})
        for j in range(i):
            previous, previous_states = sequences[j]
            delta = actions.astype(np.float64) - previous.astype(np.float64)
            pairs.append({"episodes": [j, i], "arm_rms_rad": float(np.sqrt(np.mean(delta[:, :7] ** 2))),
                          "gripper_rms_command": float(np.sqrt(np.mean(delta[:, 7] ** 2)))})
            if np.array_equal(actions, previous) and np.array_equal(states, previous_states):
                duplicates.append([j, i])
    return {"sequences": hashes, "pairs": pairs, "exact_duplicates": duplicates,
            "units": {"arm": "rad", "gripper": "command_0_255"}}


def require_no_duplicates(report):
    if report["exact_duplicates"]:
        raise ValueError("Exact duplicate executions block training admission/design; preserve, do not replace")


def _metadata(root, count, final=False):
    ds = Path(root) / "dataset"
    info = _json(ds / "meta/info.json")
    if (info["total_episodes"] != count or info["total_frames"] != 528 * count
            or info["total_tasks"] != 1 or info["fps"] != 30):
        raise ValueError("Dataset counts/task/fps mismatch")
    if final and info["splits"] != {"train": "0:20", "validation": "20:24"}:
        raise ValueError("Expected 20 train / 4 same-geometry validation split")
    expected = set(FEATURES) | {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    if set(info["features"]) != expected:
        raise ValueError("Privileged/missing policy feature")
    for key, feature in FEATURES.items():
        for field in ("dtype", "shape", "names"):
            wanted = list(feature[field]) if isinstance(feature[field], tuple) else feature[field]
            if info["features"][key][field] != wanted:
                raise ValueError("Dataset feature contract mismatch")
    tasks = pd.read_parquet(ds / "meta/tasks.parquet")
    if list(tasks.index) != [TASK] or list(tasks["task_index"]) != [0]:
        raise ValueError("Wrong instruction/task index")
    episodes = []
    for path in sorted((ds / "meta/episodes").rglob("*.parquet")):
        episodes.extend(pq.read_table(path).to_pylist())
    if len(episodes) != count:
        raise ValueError("Episode metadata coverage differs")
    for i, ep in enumerate(episodes):
        if (ep["episode_index"] != i or ep["length"] != 528 or ep["tasks"] != [TASK]
                or ep["dataset_from_index"] != 528 * i or ep["dataset_to_index"] != 528 * (i + 1)):
            raise ValueError("Episode metadata order/boundary/task mismatch")
    return info, episodes


def _media_binding(attempt, key, relative, interval, trace, check):
    from experiments.fixed_layout.record import file_hash
    binding = _json(Path(attempt) / "media" / f"{key}.json")
    camera = list(CAMERAS).index(key)
    expected_pixels = trace["image_sha256"][:, camera].tolist()
    frames = binding.get("frames", [])
    if (len(frames) != len(expected_pixels)
            or [f.get("frame_index") for f in frames] != list(range(len(expected_pixels)))
            or [f.get("raw_pixel_sha256") for f in frames] != expected_pixels
            or any(not isinstance(f.get("png_sha256"), str) or len(f["png_sha256"]) != 64
                   or any(c not in "0123456789abcdef" for c in f["png_sha256"]) for f in frames)):
        raise ValueError("Encoder input raw/camera/frame-order binding differs")
    video = _child(Path(attempt).parent.parent / "dataset", relative)
    expected = {"episode_index": int(Path(attempt).name), "video_key": key,
                "encoding": ENCODING, "frames": frames, "video_path": str(relative),
                "interval": list(interval[:2]), "video_sha256": file_hash(video, check)}
    if binding != expected:
        raise ValueError("Encoder media bytes/mapping/codec binding differs")


def _episode_inputs(root, episode, expert, reset, check, info, metadata):
    i, ds = episode["episode_index"], Path(root) / "dataset"
    ep = metadata[i]
    rows_path = _child(ds, info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]))
    actions, states, times = read_rows(rows_path, i)
    path = Path(root) / "attempts" / f"{i:06d}"
    trace, terminal, gates = load_attempt(path, episode, actions, states, times, expert, reset, check)
    videos, frames = [], 0
    for key in CAMERAS:
        prefix = "videos/" + key
        video = _child(ds, info["video_path"].format(video_key=key, chunk_index=ep[prefix + "/chunk_index"],
                                                   file_index=ep[prefix + "/file_index"]))
        matched = [row for row in metadata if (row[prefix + "/chunk_index"], row[prefix + "/file_index"])
                   == (ep[prefix + "/chunk_index"], ep[prefix + "/file_index"])]
        if len(matched) != 1 or matched[0]["episode_index"] != i:
            raise ValueError("Independent camera files required; shared/overlapping episode mapping")
        intervals = [(row[prefix + "/from_timestamp"], row[prefix + "/to_timestamp"], row["length"])
                     for row in matched]
        frames += validate_video(video, intervals, check)
        _media_binding(path, key, video.relative_to(ds), intervals[0], trace, check)
        videos.append(str(video.relative_to(ds)))
    return actions, states, trace, gates, {"rows": str(rows_path.relative_to(ds)), "videos": videos, "decoded_frames": frames}


def validate_episode(root, episode, expert, reset, check):
    from experiments.fixed_layout.record import write_json
    i = episode["episode_index"]
    info, metadata = _metadata(root, i + 1)
    actions, states, trace, gates, media = _episode_inputs(root, episode, expert, reset, check, info, metadata)
    path = Path(root) / "attempts" / f"{i:06d}"
    (path / "replays").mkdir(exist_ok=False)
    results = [replay_controls(path / "replays", expert, reset, trace, actions, mode, check) for mode in MODES]
    report = {"episode": i, "passed": True, "original_gates": gates, "media": media, "replays": results,
              "original_contacts": "geometric_reconstruction_from_saved_post_qpos_confirmed_by_reference_replay"}
    write_json(path / "validation.json", report)
    # Exact duplicate admission failure stops BEFORE any subsequent expert call.
    sequences = []
    for ep in metadata:
        p = _child(Path(root) / "dataset", info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]))
        a, s, _ = read_rows(p, ep["episode_index"])
        sequences.append((a, s))
    duplicates = duplicate_report(sequences)
    write_json(path / "duplicates.json", duplicates)
    require_no_duplicates(duplicates)
    return report


def _bindings(root, expert, reset, check):
    from experiments.fixed_layout.record import file_hash, check_snapshots
    root = Path(root)
    check()
    if (root / "failure.json").exists():
        raise ValueError("Collection failed; no admission")
    validate_plan(_json(root / "plan.json"))
    check_snapshots(root, _json(root / "snapshots.json"))
    if file_hash(root / "source.npz", check) != SOURCE_SHA256 or _json(root / "configuration.json") != expert.config:
        raise ValueError("Source snapshot/configuration changed")
    saved = _npz(root / "reset.npz")
    if set(saved) != set(reset):
        raise ValueError("Reset fields differ")
    for key in reset:
        if saved[key].dtype != reset[key].dtype:
            raise ValueError("Reset precision differs")
        _equal(saved[key], reset[key], "Frozen reset no longer matches pinned source")


def validate_dataset(root, expert, reset, check):
    """Final audit; never spends another replay token, never trusts saved PASS alone."""
    root = Path(root)
    _bindings(root, expert, reset, check)
    before = fingerprint(root, check)
    events = [_json_line(line) for line in (root / "journal.jsonl").read_text().splitlines()]
    expected = []
    for i in range(24):
        expected.extend([("started", i), ("admitted", i)])
        if i == 2:
            expected.append(("preparation_pass", None))
    if [(e["event"], e.get("episode_index")) for e in events] != expected:
        raise ValueError("Require 24 first-attempt successes, including first-three preparation gate")
    elapsed = [e["elapsed_seconds"] for e in events]
    if not all(np.isfinite(t) and 0 <= t < MAX_SECONDS for t in elapsed) or elapsed != sorted(elapsed):
        raise ValueError("Collection duration evidence invalid")
    for event, slot in zip((e for e in events if e["event"] == "started"), make_plan()["episodes"]):
        if {key: event.get(key) for key in slot} != slot:
            raise ValueError("Executed seeds/roles differ from immutable schedule")
    if sorted(p.name for p in (root / "attempts").iterdir()) != [f"{i:06d}" for i in range(24)]:
        raise ValueError("Exactly 24 attempt directories required")
    info, metadata = _metadata(root, 24, final=True)
    results, sequences, reports, rows, videos = [], [], [], set(), set()
    for episode in make_plan()["episodes"]:
        check()
        i = episode["episode_index"]
        actions, states, trace, gates, media = _episode_inputs(root, episode, expert, reset, check, info, metadata)
        path = root / "attempts" / f"{i:06d}"
        if sorted(p.name for p in (path / "replays").iterdir()) != sorted(MODES):
            raise ValueError("Missing/extra replay mode; no regenerated expert permitted")
        replay_results = []
        for mode in MODES:
            result = audit_replay(path / "replays" / mode, expert, reset, trace, actions, mode, check)
            if _json(path / "replays" / mode / "result.json") != result:
                raise ValueError("Replay result report changed")
            replay_results.append(result)
            results.append({"episode": i, **result})
        report = {"episode": i, "passed": True, "original_gates": gates, "media": media, "replays": replay_results,
                  "original_contacts": "geometric_reconstruction_from_saved_post_qpos_confirmed_by_reference_replay"}
        if report != _json(path / "validation.json"):
            raise ValueError("Per-episode validation report changed")
        sequences.append((actions, states))
        dup = duplicate_report(sequences)
        if _json(path / "duplicates.json") != dup:
            raise ValueError("Duplicate evidence changed")
        require_no_duplicates(dup)
        reports.append(report)
        rows.add(media["rows"])
        videos.update(media["videos"])
    ds = root / "dataset"
    if rows != {str(p.relative_to(ds)) for p in (ds / "data").rglob("*") if p.is_file()} or len(rows) != 24:
        raise ValueError("Missing/extra/overlapping persisted row files")
    if videos != {str(p.relative_to(ds)) for p in (ds / "videos").rglob("*") if p.is_file()} or len(videos) != 48:
        raise ValueError("Missing/extra/overlapping camera media")
    check_replay_coverage(results)
    _bindings(root, expert, reset, check)
    if before != fingerprint(root, check):
        raise ValueError("Dataset changed during independent validation")
    return {"status": "passed", "training_ready": True, "automatic_training": False,
            "episode_count": 24, "frame_count": 12672, "decoded_camera_frames": 25344,
            "train_ids": list(range(20)), "validation_ids": list(range(20, 24)),
            "validation_is_spatial_holdout": False, "replay_count": 48,
            "provenance": provenance(), "duplicates": duplicate_report(sequences), "episodes": reports}


def _json_line(line):
    return json.loads(line)


def require_fixed_layout_pass(root):
    """Read-only, fresh source/runtime/media/rows/replay-evidence admission; no stepping."""
    from experiments.fixed_layout.record import Budget
    root = Path(root)
    saved, seal = _json(root / "validation.json"), _json(root / "seal.json")
    budget = Budget(root)
    with budget:
        before = fingerprint(root, budget.check)
        if seal != {"fingerprint": before} or saved.get("status") != "passed" or saved.get("training_ready") is not True:
            raise ValueError("Fixed-layout PASS absent, incomplete or stale")
        require_no_duplicates(saved["duplicates"])
        if saved.get("provenance") != provenance():
            raise ValueError("Source/runtime changed since fixed-layout validation")
        expert = Expert()
        actual = validate_dataset(root, expert, load_reset(expert), budget.check)
        if saved != actual or fingerprint(root, budget.check) != before:
            raise ValueError("Fixed-layout report/data/source changed")
        require_no_duplicates(actual["duplicates"])
        return actual