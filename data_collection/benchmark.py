"""Model-independent replay metadata and control helpers (no policy inputs)."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mujoco
import numpy as np


STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
COLORS = ("red", "green", "blue", "yellow")
TIMING_VERSION = "fixed_substeps_synchronized_v2"


def configure_timing(model, fps):
    """Use a fixed 600 Hz physics clock; reject unsupported control rates."""
    if not np.isfinite(fps) or fps <= 0 or not np.isclose(600 / fps, round(600 / fps), rtol=0, atol=1e-10):
        raise ValueError("Control FPS must divide the 600 Hz physics clock")
    model.opt.timestep = 1 / 600
    return {"version": TIMING_VERSION, "physics_hz": 600, "fps": float(fps),
            "physics_dt": float(model.opt.timestep), "control_dt": 1 / fps,
            "substeps_per_control": int(round(600 / fps)),
            "observation_boundary": "forwarded_geometry_before_command",
            "settling_seconds": 0.2}


def restore_timing(model, manifest):
    """Never run an old cadence dataset under the new physics contract."""
    expected = configure_timing(model, manifest["fps"])
    if manifest.get("timing") != expected or not np.isclose(manifest["physics_dt"], expected["physics_dt"], rtol=0, atol=1e-14):
        raise ValueError("Missing or mismatched timing contract; use the archived runtime for old datasets")
    return expected


def synchronize_observation(model, data):
    """Refresh derived geometry/contact/bias without modifying integrator state."""
    state = capture_state(model, data)
    mujoco.mj_forward(model, data)
    mujoco.mj_setState(model, data, state, STATE_SPEC)


def validation_condition(index):
    """Six smoke conditions only; retrying an index preserves its condition."""
    pairs = (("red", "green"), ("green", "blue"), ("blue", "yellow"),
             ("yellow", "red"), ("red", "blue"), ("green", "red"))
    if not 0 <= index < len(pairs):
        raise ValueError("Validation schedule contains exactly six episodes")
    pick, place = pairs[index]
    extra = [color for color in COLORS if color not in (pick, place)][:index % 3]
    return pick, place, [pick, place] + extra


def capture_state(model, data):
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC), dtype=np.float64)
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def restore_state(model, data, state):
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (mujoco.mj_stateSize(model, STATE_SPEC),) or not np.isfinite(state).all():
        raise ValueError("Invalid MuJoCo integration state")
    mujoco.mj_resetData(model, data)
    mujoco.mj_setState(model, data, state, STATE_SPEC)
    mujoco.mj_forward(model, data)
    # Forward computes derived fields but can change solver warm-start state.
    mujoco.mj_setState(model, data, state, STATE_SPEC)


def validate_action(action):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError("Action must contain eight finite physical commands")
    return action


def apply_action(model, data, action, active_joints):
    action = validate_action(action)
    data.ctrl[list(active_joints)] = action[:7]
    data.ctrl[-1] = action[7]


def step_control(model, data, dt, arm_dof):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Control period must be positive and finite")
    ratio = dt / model.opt.timestep
    substeps = int(round(ratio))
    if substeps < 1 or not np.isclose(ratio, substeps, rtol=0, atol=1e-9):
        raise ValueError("Control period must be an integer number of physics steps")
    synchronize_observation(model, data)
    for _ in range(substeps):
        data.qfrc_applied[:arm_dof] = data.qfrc_bias[:arm_dof]
        mujoco.mj_step(model, data)
    synchronize_observation(model, data)
    return substeps


def stack_metrics(model, data, pick, place):
    pick_id = model.body(f"target_cube_{pick}").id
    place_id = model.body(f"target_cube_{place}").id
    delta = data.xpos[pick_id] - data.xpos[place_id]
    fingers = {model.body("left_finger").id, model.body("right_finger").id}
    touching = False
    for i in range(data.ncon):
        contact = data.contact[i]
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if pick_id in bodies and bodies.intersection(fingers):
            touching = True
    xy, z = float(np.linalg.norm(delta[:2])), float(delta[2])
    return {"xy_m": xy, "z_m": z, "finger_contact": touching,
            "released_stack": bool(xy < .02 and .035 < z < .045
                                   and data.ctrl[-1] > 127 and not touching)}


def source_provenance(root):
    root = Path(root)
    paths = [root / "data_collection/collect_libero_dataset.py", root / "data_collection/benchmark.py",
             root / "validation/gates.py", root / "data_collection/replay_benchmark.py"]
    paths += [root / "data_collection/pilot_plan.py", root / "data_collection/pilot_integrity.py",
              root / "data_collection/scene_bank.py", root / "validation/validate_pilot.py",
              root / "data_collection/verify_action_alignment.py"]
    paths += sorted(p for p in (root / "src").rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts
                    and p.suffix in (".py", ".xml", ".urdf", ".obj", ".stl", ".png", ".jpg"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def runtime_versions():
    return {name: importlib.metadata.version(name)
            for name in ("mujoco", "numpy", "pin", "lerobot", "torch", "torchvision", "torchcodec", "av")}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class BenchmarkRecorder:
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.directory = self.root / "replay"
        self.directory.mkdir(exist_ok=False)
        self.manifest = {**manifest, "schema_version": 1, "status": "incomplete", "episodes": []}
        write_json(self.directory / "manifest.json", self.manifest)

    def attempt(self, record):
        with (self.directory / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def save_episode(self, index, metadata, arrays):
        filename = f"episode_{index:06d}.npz"
        path = self.directory / filename
        if path.exists() or index != len(self.manifest["episodes"]):
            raise ValueError("Duplicate or out-of-order replay episode")
        np.savez_compressed(path, **arrays)
        entry = {**metadata, "episode_index": index, "file": filename,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.manifest["episodes"].append(entry)
        write_json(self.directory / "manifest.json", self.manifest)

    def finalize(self, count, splits):
        indices = [idx for values in splits.values() for idx in values]
        if (count != self.manifest["requested_episodes"] or count != len(self.manifest["episodes"])
                or sorted(indices) != list(range(count))):
            raise ValueError("Incomplete recording or invalid episode splits")
        info = json.loads((self.root / "meta/info.json").read_text())
        if info["total_episodes"] != count:
            raise ValueError("LeRobot/replay counts do not match")
        if "plan" in self.manifest:
            from data_collection.pilot_integrity import validate_accepted_slots
            validate_accepted_slots(self.manifest["plan"], self.manifest["episodes"])
            expected = {key: [i for i, s in enumerate(self.manifest["plan"]["slots"]) if s["split"] == key]
                        for key in self.manifest["plan"]["splits"]}
            if splits != expected:
                raise ValueError("Recorded splits differ from planned slots")
        if self.manifest.get("source_hashes") and self.manifest["source_hashes"] != source_provenance(Path(__file__).resolve().parents[1]):
            raise ValueError("Source changed during collection")
        self.manifest.update(status="complete", splits=splits)
        write_json(self.directory / "manifest.json", self.manifest)


def require_complete(root):
    manifest = json.loads((Path(root) / "replay/manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Dataset recording did not finalize")
    return manifest