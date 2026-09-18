"""Pinned pilot reset and read-only preflight. Never step, settle, render or run."""
import hashlib
import importlib.metadata
import io
import json
import platform
from pathlib import Path
from shutil import disk_usage

import mujoco
import numpy as np

from data_collection.benchmark import capture_state, restore_state, runtime_versions, source_provenance
from data_collection.pilot_integrity import scene_digest
from src.robots.franka_panda.config import SCENE_PATH
from experiments.pose_snapshot_v2.expert import Expert
from experiments.fixed_layout.contract import (
    ROOT, OUTPUT, MIN_FREE_BYTES, IMAGE, SOURCE, SOURCE_SHA256, SOURCE_SCENE_SHA256,
    TASK, EXPERT_VERSION, make_plan, validate_plan,
)

SPEC = "docs/superpowers/specs/2026-09-18-fixed-layout-baseline-design.md"
INITIAL_OBJECTS = {
    "red": [0.2820318976205449, -0.38303709520664875, 0.020994666209306827],
    "green": [0.2676253941656954, 0.3512266692018041, 0.020994666209306827],
}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _model_hash(model):
    """Hash MuJoCo's full compiled model in memory; no file or renderer needed."""
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def _check_manifest(expert, manifest, source_sha256):
    episodes = manifest.get("episodes", [])
    if (manifest.get("status") != "complete" or len(episodes) <= 7
            or sum(e.get("episode_index") == 7 for e in episodes) != 1):
        raise ValueError("Source manifest is incomplete or episode 7 is ambiguous")
    entry = episodes[7]
    expected = {"episode_index": 7, "file": "episode_000007.npz", "sha256": source_sha256,
                "scene_hash": SOURCE_SCENE_SHA256, "pick": "red", "place": "green",
                "colors": ["red", "green"], "instruction": TASK, "frames": 528,
                "split": "train", "initial_objects": INITIAL_OBJECTS}
    for key, value in expected.items():
        if type(entry.get(key)) is not type(value) or entry[key] != value:
            raise ValueError(f"Source manifest episode 7 mismatch: {key}")
    if expert.config.get("expert_version") != EXPERT_VERSION:
        raise ValueError("Expected unchanged pose-snapshot-v2 expert")
    for key in ("fps", "physics_dt", "control_dt_requested", "timing", "camera_names",
                "camera_size", "camera_fovy", "camera_pos", "camera_quat", "expert_command_contract"):
        if manifest.get(key) != expert.config[key]:
            raise ValueError(f"Source/runtime configuration mismatch: {key}")
    model = expert.model
    if model.opt.timestep != 1 / 600:
        raise ValueError("Runtime physics timestep mismatch")
    for key, actual in (("camera_pos", model.cam_pos), ("camera_quat", model.cam_quat),
                        ("camera_fovy", model.cam_fovy)):
        if not np.array_equal(actual, manifest[key]):
            raise ValueError(f"Live model calibration mismatch: {key}")
    current = source_provenance(ROOT)
    assets = {k: v for k, v in current.items() if k.startswith("src/")}
    historical = {k: v for k, v in manifest.get("source_hashes", {}).items() if k.startswith("src/")}
    if not assets or assets != historical:
        raise ValueError("Source robot/physics/camera assets changed")
    if manifest.get("versions") != runtime_versions():
        raise ValueError("Source runtime versions changed")
    if manifest.get("scene_path") != "src/robots/franka_panda/model/smolvla_scene.xml":
        raise ValueError("Source scene path mismatch")
    # File hashes cannot detect a caller's in-memory gravity/mass/light edits.
    # Compile the source-bound scene, apply only the declared timing, and compare
    # the complete model. Neither compilation nor serialization advances time.
    reference = mujoco.MjModel.from_xml_path(SCENE_PATH)
    reference.opt.timestep = 1 / 600
    if _model_hash(model) != _model_hash(reference):
        raise ValueError("Live model differs from source-bound physics/geometry/lights")


def _load_reset(expert, *, source_path, manifest_path, source_sha256) -> dict[str, np.ndarray]:
    """Private fixture seam; public load_reset never accepts a source/hash override."""
    # Hash and parse the SAME bytes, before any np.load or restoration.
    payload = Path(source_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != source_sha256:
        raise ValueError("Pinned source SHA256 mismatch")
    manifest = json.loads(Path(manifest_path).read_text())
    _check_manifest(expert, manifest, source_sha256)
    model = expert.model
    shapes = {"initial_state": (mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION),),
              "qpos": (528, model.nq), "ctrl": (528, model.nu), "sim_time": (528,),
              "body_xpos": (528, model.nbody, 3), "camera_xpos": (528, model.ncam, 3)}
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, shape in shapes.items():
        if key not in arrays or arrays[key].shape != shape or arrays[key].dtype != np.float64:
            raise ValueError(f"Missing, wrong shape or precision in source: {key}")
    if any(a.dtype.kind not in "bifu" or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Source arrays must be finite numeric values")
    initial = arrays["initial_state"]
    data = mujoco.MjData(model)
    restore_state(model, data, initial)
    if capture_state(model, data).tobytes() != initial.tobytes():
        raise ValueError("Initial integration state did not restore bit-exactly")
    if data.qpos.tobytes() != arrays["qpos"][0].tobytes():
        raise ValueError("Restored full qpos differs from source first row")
    if scene_digest(data.qpos) != SOURCE_SCENE_SHA256:
        raise ValueError("Source scene hash differs from approved geometry")
    if data.time != arrays["sim_time"][0]:
        raise ValueError("Source initial time mismatch")
    for key, actual in (("body_xpos", data.xpos), ("camera_xpos", data.cam_xpos)):
        if not np.array_equal(actual, arrays[key][0]):
            raise ValueError(f"Restored initial geometry differs from source: {key}")
    for color, position in INITIAL_OBJECTS.items():
        if not np.array_equal(data.xpos[model.body(f"target_cube_{color}").id], position):
            raise ValueError(f"Approved settled {color} position mismatch")
    # This old NPZ lacks these arrays. Do not substitute ctrl[0] (first COMMAND)
    # for reset controls, or pretend derived rotations were saved in the trace.
    for key, actual in (("initial_ctrl", data.ctrl), ("camera_xmat", data.cam_xmat)):
        if key in arrays:
            expected = arrays[key] if key == "initial_ctrl" else arrays[key][0]
            if not np.array_equal(actual, expected):
                raise ValueError(f"Optional source evidence mismatch: {key}")
    reset = {"initial_state": initial.copy(), "qpos": data.qpos.copy(),
             "initial_ctrl": data.ctrl.copy(), "body_xpos": data.xpos.copy(),
             "camera_xpos": data.cam_xpos.copy(), "camera_xmat": data.cam_xmat.copy()}
    if any(not np.isfinite(a).all() for a in reset.values()):
        raise ValueError("Nonfinite restored geometry")
    return reset


def load_reset(expert) -> dict[str, np.ndarray]:
    """Restore episode 7 on fresh MjData, without stepping or changing the expert."""
    path = ROOT / SOURCE
    return _load_reset(expert, source_path=path, manifest_path=path.with_name("manifest.json"),
                       source_sha256=SOURCE_SHA256)


def provenance() -> dict:
    """Explicit current source/assets/runtime inventory, not an old saved PASS."""
    files = source_provenance(ROOT)
    paths = [ROOT / SOURCE, (ROOT / SOURCE).with_name("manifest.json"), ROOT / SPEC]
    paths += [ROOT / name for name in (
        "experiments/paired_demonstrations/expert_v1.py",
        "experiments/paired_demonstrations/media.py", "experiments/paired_demonstrations/plan.py",
        "experiments/paired_demonstrations/validate.py", "experiments/paired_demonstrations/record.py",
        "experiments/pose_snapshot_v2/expert.py", "experiments/pose_snapshot_v2/planner.py")]
    paths += sorted(Path(__file__).parent.rglob("*.py"))
    files.update({str(path.relative_to(ROOT)): _hash(path) for path in paths})
    import av
    media = {}
    for name in ("av", "pyarrow", "pandas", "datasets", "Pillow", "imageio", "imageio-ffmpeg"):
        try:
            media[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            media[name] = "not_installed"
    media["ffmpeg_libraries"] = {key: list(value) for key, value in av.library_versions.items()}
    return {"files": dict(sorted(files.items())), "runtime": runtime_versions(),
            "media_runtime": media,
            "python": platform.python_version(), "platform": platform.platform(),
            "required_image": IMAGE}


def storage_path(output):
    """Probe the destination itself, or its nearest existing parent before mkdir."""
    path = Path(output).absolute()
    while not path.exists():
        if path.is_symlink():
            raise ValueError("Dangling output symlink")
        path = path.parent
    return path


def preflight() -> dict:
    """Read-only reset/source checks and separate storage eligibility; no output mkdir.

    Constructor only: the existing Expert configures MuJoCo/IK but does not run
    a trajectory or renderer. No learning model is constructed or loaded.
    """
    destination = storage_path(OUTPUT)
    free = disk_usage(destination).free
    report = {"reset_valid": False, "free_bytes": free, "min_free_bytes": MIN_FREE_BYTES,
              "storage_path": str(destination),
              "storage_eligible": free >= MIN_FREE_BYTES, "output": str(OUTPUT),
              "output_exists": OUTPUT.exists(), "collection_ready": False,
              "scope": "source_and_reset_only_no_steps_no_render_no_expert_execution",
              "pending": ["actual expert/contact/media/replay gates (software implementation is not data admission)",
                          "collection authorization and storage reserve"]}
    try:
        plan = make_plan()
        validate_plan(plan)
        before = provenance()
        reset = load_reset(Expert())
        if provenance() != before:
            raise ValueError("Source/runtime changed during preflight")
        report.update(reset_valid=True, source_sha256=before["files"][SOURCE],
                      scene_hash=SOURCE_SCENE_SHA256, provenance=before,
                      **{key: value.tolist() for key, value in reset.items()},
                      checks={"initial_state": "bit_exact_full_integration_state",
                              "qpos": "bit_exact_source_first_row",
                              "body_xpos": "exact_source_first_row",
                              "camera_xpos": "exact_source_first_row",
                              "camera_xmat": "derived_from_source_bound_model_no_saved_rotation_trace",
                              "initial_ctrl": "restored_from_full_integration_state_not_first_command"})
    except (ValueError, OSError, KeyError) as exc:
        report.update(error=f"{type(exc).__name__}: {exc}")
    return report