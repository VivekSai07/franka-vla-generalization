"""Canonical data-stage contract. Pure stdlib; no collection or model loading."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/fixed_layout/fixed_layout_20260918"
MIN_FREE_BYTES = 30 * 1024**3
MAX_SECONDS = 3 * 60 * 60
IMAGE = "sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc"
VERSION = "fixed_layout_baseline_v1"
MASTER_SEED = 20260918
SOURCE = "datasets/franka_pilot_20260915/replay/episode_000007.npz"
SOURCE_SHA256 = "ca5c19eda5d089a4a33a838a3fc3e55b44895f77aa40e8d92d9490953290221b"
SOURCE_SCENE_SHA256 = "88060815ffd36e578284aeda158cbaac38e0149db398f09c4cf22bad48150efd"
TASK = "pick up the red cube and place it on top of the green cube"
EXPERT_VERSION = "paired_scripted_expert_pose_snapshot_v2"


def make_plan() -> dict:
    """Return a fresh canonical plan; callers must validate it before use."""
    episodes = []
    for i in range(24):
        seed = int.from_bytes(hashlib.sha256(
            f"{VERSION}:{MASTER_SEED}:expert:{i}".encode("ascii")).digest()[:4], "big")
        episodes.append({"episode_index": i, "seed": seed,
                         "split": "train" if i < 20 else "validation",
                         "pick": "red", "place": "green", "colors": ["red", "green"],
                         "instruction": TASK, "task_index": 0, "source_episode_index": 7,
                         "scene_hash": SOURCE_SCENE_SHA256})
    return {
        "version": VERSION, "master_seed": MASTER_SEED,
        "seed_derivation": "sha256(namespace:master:expert:decimal_index) first 4 bytes big endian",
        "source": {"path": SOURCE, "sha256": SOURCE_SHA256,
                   "episode_index": 7, "scene_hash": SOURCE_SCENE_SHA256},
        "episodes": episodes, "train_ids": list(range(20)),
        "validation_ids": list(range(20, 24)), "preparation_ids": [0, 1, 2],
        "validation_is_spatial_holdout": False,
        "reset_policy": "identical_full_integration_state_no_resettle_no_randomization",
        "expert_version": EXPERT_VERSION, "waypoint_jitter_sigma_m": .005,
        "expert_attempts": 24, "replay_modes": ["reference", "persisted_float32"],
        "failure_policy": "first_failure_stop_no_retry_no_resume",
        "output": str(OUTPUT.relative_to(ROOT)), "min_free_bytes": MIN_FREE_BYTES,
        "max_seconds": MAX_SECONDS, "image": IMAGE,
        "expected_episode_frames": 528, "fps": 30, "physics_hz": 600,
        "substeps_per_control": 20, "state_dim": 9, "action_dim": 8,
        "state_dtype": "float32", "action_dtype": "float32",
        "action_semantics": "absolute_joint_commands_gripper_0_closed_255_open",
        "observation_alignment": "observe_command_record_pre_step",
        "camera_names": ["front_cam", "gripper_cam"], "camera_size": [512, 512],
        "camera_keys": ["observation.images.image", "observation.images.image2"],
        "encoding": {"vcodec": "libsvtav1", "pix_fmt": "yuv420p", "crf": 30,
                     "g": 2, "encoder_threads": 1},
        "gates": {"closed_consecutive_ticks": 30, "bilateral_consecutive_ticks": 5,
                  "lift_while_bilateral_gt_m": .05, "released_stack_consecutive_ticks": 30,
                  "stack_xy_lt_m": .02, "stack_z_gt_m": .035, "stack_z_lt_m": .045,
                  "gripper_open_gt": 127, "top_cube_finger_contact": False,
                  "reference_full_qpos_atol": 1e-5, "float32_policy_state_atol": 1e-3},
        "automatic_training": False,
    }


def _same(actual, expected):
    """Exact JSON types as well as values: False/0/0.0 and tuple/list differ."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (actual.keys() == expected.keys()
                and all(type(k) is str for k in actual)
                and all(_same(actual[k], value) for k, value in expected.items()))
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def validate_plan(plan) -> None:
    """Reject any mutation, unknown field or noncanonical type; never repair it."""
    if not _same(plan, make_plan()):
        raise ValueError("Plan differs from the approved fixed-layout contract")