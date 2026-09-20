import os
import sys
import time
import random
import argparse
import collections
import json

import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer
from pathlib import Path

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("ERROR: lerobot not installed.")
    sys.exit(1)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.controllers.ik_controller_m2 import IKController
from src.controllers.joint_target_limits import bounded_joint_target
from src.controllers.grasp_controller import GraspController
from src.planners.trajectory_planner import TaskSpaceTrajectory
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    SCENE_PATH,
    Q_HOME,
    ARM_DOF,
    ACTIVE_JOINTS
)
from data_collection.benchmark import (
    BenchmarkRecorder, capture_state, runtime_versions, source_provenance,
    stack_metrics, step_control, validation_condition, configure_timing, synchronize_observation,
)
from data_collection.pilot_plan import validate_plan, plan_digest, slot_seed
from data_collection.pilot_integrity import scene_digest, validate_expert_commands, validate_scene_geometry

# ---------------------------------------------------------------------------
# All 12 valid pick-place colour combinations.
# v1 used only "red → blue" for every episode, which prevented the VLM from
# grounding colour tokens to specific objects. v2 samples uniformly so every
# colour appears as both a pick target and a place target. See DATASET_DESIGN.md.
# ---------------------------------------------------------------------------
CUBE_COLORS = ["red", "green", "blue", "yellow"]
ALL_INSTRUCTIONS = [
    (pick, place, f"pick up the {pick} cube and place it on top of the {place} cube")
    for pick in CUBE_COLORS
    for place in CUBE_COLORS
    if pick != place
]

# Minimum XY clearance between spawned cubes (v1 used 0.12 m — too close).
MIN_CUBE_CLEARANCE = 0.15

# Success threshold: top-cube must be within 2 cm XY of bottom cube centre.
# v1 used 3 cm, which allowed visibly misaligned stacks into the dataset.
SUCCESS_XY_THRESHOLD = 0.02
SUCCESS_Z_MIN = 0.035
SUCCESS_Z_MAX = 0.045


def sample_episode_task():
    """Return (pick_color, place_color, instruction_string) sampled uniformly."""
    return random.choice(ALL_INSTRUCTIONS)


def get_cube_body_name(color: str) -> str:
    return f"target_cube_{color}"


def randomize_cubes(model, data, colors_present: list[str]):
    """
    Spawn only the cubes needed for this episode.
    Remaining cubes are moved off-table so they are invisible.

    v2 changes vs v1:
    - Spawn range widened: x=[0.20, 0.60], y=[-0.40, 0.40]
    - Min clearance raised from 0.12 m to MIN_CUBE_CLEARANCE (0.15 m)
    - Only colors_present are placed on the table; others are parked off-scene.
    """
    all_colors = ["red", "green", "blue", "yellow"]
    spawned_positions = []

    for color in all_colors:
        body_name = get_cube_body_name(color)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            continue
        jnt_id = model.body_jntadr[body_id]
        if jnt_id == -1:
            continue
        qpos_idx = model.jnt_qposadr[jnt_id]

        if color not in colors_present:
            # Park off-table so the camera never sees it.
            data.qpos[qpos_idx:qpos_idx + 3] = [2.0, 2.0, -1.0]
            data.qpos[qpos_idx + 3:qpos_idx + 7] = [1, 0, 0, 0]
            qvel_idx = model.jnt_dofadr[jnt_id]
            data.qvel[qvel_idx:qvel_idx + 6] = 0.0
            continue

        # Place on table with no overlap.
        for _ in range(1000):
            x = random.uniform(0.20, 0.60)
            y = random.uniform(-0.40, 0.40)
            if all((x - sx) ** 2 + (y - sy) ** 2 >= MIN_CUBE_CLEARANCE ** 2
                   for sx, sy in spawned_positions):
                spawned_positions.append((x, y))
                data.qpos[qpos_idx:qpos_idx + 3] = [x, y, 0.05]
                data.qpos[qpos_idx + 3:qpos_idx + 7] = [1, 0, 0, 0]
                qvel_idx = model.jnt_dofadr[jnt_id]
                data.qvel[qvel_idx:qvel_idx + 6] = 0.0
                break

        else:
            raise ValueError("Unable to place cubes within required clearance")


def sample_distractor_colors(pick_color: str, place_color: str) -> list[str]:
    """
    Return a scene colour list with a random number of distractors (0–2).
    v1 always included all 4 cubes; v2 varies this so the model cannot rely
    on object count as a cue.
    """
    distractors_available = [c for c in CUBE_COLORS if c not in (pick_color, place_color)]
    n_distractors = random.randint(0, len(distractors_available))
    distractors = random.sample(distractors_available, n_distractors)
    return [pick_color, place_color] + distractors


def generate_task_states(z_offset: float, robot_speed: float = 1.0):
    STACK_OFFSET = 0.04
    return [
        {"name": "Approach Pick",  "phase": "pick",  "offset": np.array([0, 0, z_offset + 0.15]), "gripper": 255, "duration": 2.0 / robot_speed, "jitter": True},
        {"name": "Descend to Pick","phase": "pick",  "offset": np.array([0, 0, z_offset]),         "gripper": 255, "duration": 1.5 / robot_speed, "jitter": True},
        {"name": "Grasping",       "phase": "pick",  "offset": np.array([0, 0, z_offset]),         "gripper": 0,   "duration": 1.0 / robot_speed, "jitter": False},
        {"name": "Verify Lift",    "phase": "pick",  "offset": np.array([0, 0, z_offset + 0.2]),   "gripper": 0,   "duration": 2.0 / robot_speed, "jitter": False},
        {"name": "Move to Place",  "phase": "place", "offset": np.array([0, 0, z_offset + 0.2]),   "gripper": 0,   "duration": 3.0 / robot_speed, "jitter": True},
        {"name": "Lower to Place", "phase": "place", "offset": np.array([0, 0, z_offset + STACK_OFFSET + 0.01]), "gripper": 0, "duration": 2.0 / robot_speed, "jitter": True},
        {"name": "Release",        "phase": "place", "offset": np.array([0, 0, z_offset + STACK_OFFSET + 0.01]), "gripper": 255, "duration": 1.0 / robot_speed, "jitter": False},
        {"name": "Retract",        "phase": "place", "offset": np.array([0, 0, z_offset + 0.2]),   "gripper": 255, "duration": 2.0 / robot_speed, "jitter": False},
        {"name": "Return to Home", "phase": "home",  "offset": np.zeros(3),                         "gripper": 255, "duration": 3.0 / robot_speed, "jitter": False},
    ]


def main():
    parser = argparse.ArgumentParser(description="Franka LeRobot Data Collection v2")
    parser.add_argument("--episodes",       type=int,   default=300,  help="Total episodes to collect")
    parser.add_argument("--fps",            type=int,   default=30,   help="Frames per second")
    parser.add_argument("--robot_speed",    type=float, default=1.0,  help="Physical robot speed multiplier")
    parser.add_argument("--playback_speed", type=float, default=1.0,  help="Sim playback speed (0 = max)")
    parser.add_argument("--headless",       action="store_true",      help="No viewer window")
    parser.add_argument("--root", type=Path, help="New output directory; never overwrite an existing root")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--benchmark", action="store_true", help="Six-episode seeded replayable validation batch")
    parser.add_argument("--max-attempts", type=int, default=30, help="Attempt bound for benchmark mode")
    parser.add_argument("--plan", type=Path, help="Immutable pilot/preparation slot manifest; overrides episode count")
    args = parser.parse_args()
    plan = None
    if args.plan:
        if args.benchmark or args.root is None or args.fps != 30 or args.robot_speed != 1.0:
            parser.error("--plan requires fresh --root, 30 fps, robot_speed 1 and no --benchmark")
        plan = json.loads(args.plan.read_text())
        validate_plan(plan)
        args.seed = plan["seed"]
        args.episodes = len(plan["slots"])
    if args.episodes <= 0 or args.fps <= 0 or args.robot_speed <= 0:
        parser.error("episodes, fps and robot_speed must be positive")
    if args.benchmark and (args.episodes != 6 or args.root is None or args.max_attempts < 6):
        parser.error("benchmark mode requires --episodes 6, --root, and max-attempts >=6")

    FPS = args.fps
    dt_control = 1.0 / FPS

    # --- Dataset versioning ---
    BASE_DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
    os.makedirs(BASE_DATASETS_DIR, exist_ok=True)
    version = 1
    while os.path.exists(os.path.join(BASE_DATASETS_DIR, f"franka_libero_stacking_v{version}")):
        version += 1
    DATASET_REPO_ID = f"franka_libero_stacking_v{version}"
    DATASET_DIR = os.path.join(BASE_DATASETS_DIR, DATASET_REPO_ID)
    if args.root is not None:
        DATASET_DIR = str(args.root.resolve())
        DATASET_REPO_ID = args.root.name
        if Path(DATASET_DIR).exists():
            raise FileExistsError(f"Refusing to overwrite {DATASET_DIR}")

    print(f"Initializing dataset: {DATASET_DIR}")
    print(f"Target: {args.episodes} episodes across {len(ALL_INSTRUCTIONS)} instruction variants")

    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        root=DATASET_DIR,
        fps=FPS,
        features={
            "action": {
                "dtype": "float32", "shape": (8,),
                "names": ["panda_joint1.pos", "panda_joint2.pos", "panda_joint3.pos",
                          "panda_joint4.pos", "panda_joint5.pos", "panda_joint6.pos",
                          "panda_joint7.pos", "gripper.pos"]
            },
            "observation.state": {
                "dtype": "float32", "shape": (9,),
                "names": ["panda_joint1.pos", "panda_joint2.pos", "panda_joint3.pos",
                          "panda_joint4.pos", "panda_joint5.pos", "panda_joint6.pos",
                          "panda_joint7.pos", "finger1.pos", "finger2.pos"]
            },
            "observation.images.image":  {"dtype": "video", "shape": (512, 512, 3), "names": ["height", "width", "channels"]},
            "observation.images.image2": {"dtype": "video", "shape": (512, 512, 3), "names": ["height", "width", "channels"]},
        }
    )

    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=10.0, kp_rot=5.0)

    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    timing = configure_timing(m, FPS)
    joint_bounds = np.column_stack((
        np.maximum(m.actuator_ctrlrange[ACTIVE_JOINTS, 0], robot.model.lowerPositionLimit[ACTIVE_JOINTS]),
        np.minimum(m.actuator_ctrlrange[ACTIVE_JOINTS, 1], robot.model.upperPositionLimit[ACTIVE_JOINTS])))
    d = mujoco.MjData(m)
    recorder = None
    if args.benchmark or plan:
        recorder = BenchmarkRecorder(DATASET_DIR, {
            "requested_episodes": args.episodes, "seed": args.seed,
            "fps": FPS, "control_dt_requested": dt_control, "physics_dt": m.opt.timestep,
            "timing": timing,
            "expert_command_contract": {"version": "bounded_joint_integration_v1",
                                        "joint_bounds": joint_bounds.tolist(), "max_velocity": 2.,
                                        "position_margin_rad": 1e-6, "saturation_recorded": True,
                                        "initial_target": np.asarray(Q_HOME[:7]).tolist(),
                                        "initialization_exception": "reset_controls_during_settling_then_home_target_baseline"},
            "versions": runtime_versions(), "source_hashes": source_provenance(PROJECT_ROOT),
            "scene_path": str(Path(SCENE_PATH).relative_to(PROJECT_ROOT)),
            "camera_names": ["front_cam", "gripper_cam"], "camera_size": [512, 512],
            "camera_fovy": m.cam_fovy.tolist(), "camera_pos": m.cam_pos.tolist(),
            "camera_quat": m.cam_quat.tolist(), "state_units": ["rad"] * 7 + ["m"] * 2,
            "action_units": ["rad"] * 7 + ["0_closed_255_open"],
            "role": "integration_only_not_benchmark_score",
            **({"plan": plan, "plan_digest": plan_digest(plan),
                "role": "pilot" if plan["mode"] == "pilot" else "preparation_only",
                "validation_required": True} if plan else {}),
        })

    # Collect at 512×512 — matches SmolVLM2's max_image_size.longest_edge=512.
    # The VLM upscales anything below 512, so 256×256 would be blurred.
    # Square aspect ratio avoids the padding bars that 480×640 (v1) produced.
    renderer = mujoco.Renderer(m, height=512, width=512)

    n_joints = min(len(Q_HOME), m.nq)
    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:n_joints] = Q_HOME[:n_joints]
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS = home_pose.translation.copy()

    GRIPPER_Z_OFFSET = 0.105

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.sync()

    # Track per-instruction counts for balanced coverage reporting.
    instruction_counts = {instr: 0 for _, _, instr in ALL_INSTRUCTIONS}
    episodes_collected = 0

    # ETA tracking — rolling average over the last 10 successful episodes.
    collection_start = time.time()
    episode_times: collections.deque = collections.deque(maxlen=10)
    attempts = 0
    slot_failures = 0
    accepted_scene_hashes = set()

    while episodes_collected < args.episodes:
        # --- Sample this episode's task ---
        slot = None
        if plan:
            if slot_failures >= plan["max_attempts_per_slot"]:
                raise RuntimeError("Slot failed ten attempts; dataset remains incomplete, do not resume automatically")
            slot = plan["slots"][episodes_collected]
            attempt_seed = slot_seed(slot, slot_failures)
            random.seed(attempt_seed); np.random.seed(attempt_seed)
            pick_color, place_color, colors_present = slot["pick"], slot["place"], slot["colors"]
            instruction = f"pick up the {pick_color} cube and place it on top of the {place_color} cube"
        elif recorder:
            if attempts >= args.max_attempts:
                raise RuntimeError("Benchmark attempt limit exceeded; dataset remains incomplete")
            attempt_seed = args.seed + attempts
            random.seed(attempt_seed)
            np.random.seed(attempt_seed)
            pick_color, place_color, colors_present = validation_condition(episodes_collected)
            instruction = f"pick up the {pick_color} cube and place it on top of the {place_color} cube"
        else:
            pick_color, place_color, instruction = sample_episode_task()
            colors_present = sample_distractor_colors(pick_color, place_color)
        attempts += 1
        episode_wall_start = time.time()

        print(f"\n[{episodes_collected + 1}/{args.episodes}] "
              f"Task: '{instruction}' | cubes on table: {colors_present}")

        # --- Reset physics ---
        mujoco.mj_resetData(m, d)
        randomize_cubes(m, d, colors_present)

        d.qpos[:n_joints] = Q_HOME[:n_joints]
        d.qvel[:] = 0.0
        q_target = np.zeros(m.nq)
        q_target[:n_joints] = Q_HOME[:n_joints]

        # Keep the previous 0.2 seconds of settling, not its old substep count.
        for _ in range(round(timing["settling_seconds"] / m.opt.timestep)):
            mujoco.mj_step(m, d)
        synchronize_observation(m, d)
        geometry_hash = scene_digest(d.qpos)
        if plan:
            validate_scene_geometry(m, d, colors_present)
            if geometry_hash in accepted_scene_hashes:
                raise ValueError("Duplicate initialized scene; collection halted")
        initial_state = capture_state(m, d) if recorder else None
        initial_objects = {c: d.xpos[m.body(get_cube_body_name(c)).id].tolist() for c in colors_present}
        trace_qpos, trace_ctrl, trace_time, trace_steps, trace_success = [], [], [], [], []
        trace_body_pos, trace_camera_pos = [], []
        trace_requested_dq, trace_applied_dq, trace_saturation = [], [], []
        stable_ticks = 0
        failure_reason = "placement"

        grasp_sys = GraspController(m, d, target=get_cube_body_name(pick_color))
        states = generate_task_states(GRIPPER_Z_OFFSET, robot_speed=args.robot_speed)

        current_state_idx = 0
        state_start_time = d.time
        current_trajectory = None
        episode_buffer = []
        episode_success = False
        _live_print_t = time.time()

        while current_state_idx < len(states):
            step_start = time.time()

            # --- Observations ---
            synchronize_observation(m, d)
            renderer.update_scene(d, camera="front_cam")
            img_agent = renderer.render().copy()
            renderer.update_scene(d, camera="gripper_cam")
            img_gripper = renderer.render().copy()

            q_current = d.qpos[:robot.model.nq].copy()
            if recorder:
                trace_qpos.append(d.qpos.copy())
                trace_time.append(float(d.time))
                trace_body_pos.append(d.xpos.copy())
                trace_camera_pos.append(d.cam_xpos.copy())
            obs_state = np.zeros(9, dtype=np.float32)
            obs_state[:7] = q_current[:7]
            obs_state[7] = q_current[7]
            obs_state[8] = q_current[8]

            # --- Controller: compute and send this tick's action BEFORE
            # recording it, so the saved action matches what was actually
            # sent (mirrors lerobot_record.py's observe -> act -> send -> save).
            state = states[current_state_idx]

            if current_trajectory is None:
                if state["phase"] == "pick":
                    body_id = m.body(get_cube_body_name(pick_color)).id
                    base_pos = d.xpos[body_id].copy()
                elif state["phase"] == "place":
                    body_id = m.body(get_cube_body_name(place_color)).id
                    base_pos = d.xpos[body_id].copy()
                else:
                    base_pos = HOME_POS - state["offset"]

                target_pos = base_pos + state["offset"]

                # Approach jitter: ±5 mm noise on jitter-enabled states.
                # Prevents the model from memorising a single canonical path.
                if state["jitter"]:
                    target_pos += np.random.normal(0, 0.005, 3)

                start_se3 = robot.forward_kinematics(q_current)
                end_se3 = pin.SE3(fixed_rotation, target_pos)
                current_trajectory = TaskSpaceTrajectory(
                    start_pose=start_se3, end_pose=end_se3,
                    duration=state["duration"], method="cubic"
                )

            t_state = d.time - state_start_time
            target_se3 = current_trajectory.get_pose(t_state)

            dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
            target, applied_dq, saturated = bounded_joint_target(
                q_target[ACTIVE_JOINTS], dq[ACTIVE_JOINTS], dt_control, joint_bounds)
            q_target[ACTIVE_JOINTS] = target
            d.ctrl[ACTIVE_JOINTS] = target
            if recorder:
                trace_requested_dq.append(dq[ACTIVE_JOINTS].copy())
                trace_applied_dq.append(applied_dq.copy())
                trace_saturation.append(saturated.copy())

            # --- Record the action that was just computed and sent above ---
            act_state = np.zeros(8, dtype=np.float32)
            act_state[:7] = q_target[:7]
            act_state[7] = float(state["gripper"])
            grasp_sys.command(state["gripper"])
            if plan:
                try:
                    validate_expert_commands(act_state[None], m.actuator_ctrlrange[ACTIVE_JOINTS])
                except ValueError:
                    recorder.attempt({"attempt": attempts, "slot_id": slot["slot_id"], "seed": attempt_seed,
                                      "accepted": False, "reason": "invalid_expert_command", "fatal": True})
                    raise

            episode_buffer.append({
                "observation.images.image":  img_agent,
                "observation.images.image2": img_gripper,
                "observation.state": obs_state,
                "action": act_state,
                "task": instruction,
            })

            is_holding = grasp_sys.is_grasped()

            # --- State transitions ---
            if t_state > state["duration"]:
                if state["name"] == "Verify Lift":
                    if not is_holding:
                        print(f"\r\033[K  [{d.time:.2f}s] Grasp failed — discarding episode.")
                        failure_reason = "grasp"
                        break
                    print(f"\r\033[K  [{d.time:.2f}s] Grasp verified.")

                current_state_idx += 1
                state_start_time = d.time
                current_trajectory = None

            # --- Physics step ---
            if recorder:
                trace_ctrl.append(d.ctrl.copy())
            substeps = step_control(m, d, dt_control, ARM_DOF)
            metrics = stack_metrics(m, d, pick_color, place_color)
            stable_ticks = stable_ticks + 1 if metrics["released_stack"] else 0
            if recorder:
                trace_steps.append(substeps)
                trace_success.append(metrics["released_stack"])
            if current_state_idx >= len(states):
                episode_success = stable_ticks >= FPS
                failure_reason = "not_released_and_stable_one_second"
                outcome = "SUCCESS" if episode_success else "FAILURE"
                print(f"\r\033[K  [{d.time:.2f}s] {outcome} "
                      f"(xy={metrics['xy_m']*100:.1f}cm, z={metrics['z_m']*100:.1f}cm, "
                      f"stable_ticks={stable_ticks})")

            _now = time.time()
            if _now - _live_print_t >= 0.4:
                _live_print_t = _now
                phase_frac = min(t_state / max(state["duration"], 0.001), 1.0)
                filled = int(20 * phase_frac)
                bar = "█" * filled + "░" * (20 - filled)
                if episode_times:
                    _avg = sum(episode_times) / len(episode_times)
                    _eta = int(_avg * (args.episodes - episodes_collected))
                    _h, _r = divmod(_eta, 3600)
                    _m, _s = divmod(_r, 60)
                    eta_str = f"{_h:02d}h{_m:02d}m{_s:02d}s"
                else:
                    eta_str = "--:--:--"
                print(
                    f"\r  [{bar}] {state['name']:<20s}  sim {d.time:5.1f}s | ETA {eta_str}   ",
                    end="", flush=True,
                )

            if viewer is not None and viewer.is_running():
                viewer.sync()

            if args.playback_speed > 0:
                elapsed = time.time() - step_start
                sleep_time = dt_control / args.playback_speed - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # --- Save or discard ---
        if recorder:
            if episode_success and stable_ticks < FPS:
                episode_success = False
                failure_reason = "not_released_and_stable_30_ticks"
            recorder.attempt({"attempt": attempts, "seed": attempt_seed,
                              "condition_index": episodes_collected,
                              "instruction": instruction, "colors": colors_present,
                              "accepted": episode_success,
                              "saturated_ticks": int(np.any(trace_saturation, axis=1).sum()) if trace_saturation else 0,
                              "reason": "success" if episode_success else failure_reason})
            if plan:
                # The slot ledger is separate so historical diagnostic records remain readable.
                with (Path(DATASET_DIR) / "replay/slots.jsonl").open("a") as stream:
                    stream.write(json.dumps({"slot_id": slot["slot_id"], "split": slot["split"],
                                             "seed": attempt_seed, "retry": slot_failures,
                                             "accepted": episode_success, "reason": "success" if episode_success else failure_reason,
                                             "scene_hash": geometry_hash, "initial_objects": initial_objects}) + "\n")
        if episode_success:
            print(f"  Saving {len(episode_buffer)} frames...")
            for frame in episode_buffer:
                dataset.add_frame(frame)
            dataset.save_episode()
            if recorder:
                recorder.save_episode(episodes_collected, {
                    "attempt": attempts, "seed": attempt_seed, "instruction": instruction,
                    "pick": pick_color, "place": place_color, "colors": colors_present,
                    "initial_objects": initial_objects, "frames": len(episode_buffer),
                    "stable_ticks": stable_ticks, "final_metrics": metrics,
                    "saturated_ticks": int(np.any(trace_saturation, axis=1).sum()),
                    "saturated_ticks_per_joint": np.sum(trace_saturation, axis=0).tolist(),
                    **({"slot_id": slot["slot_id"], "split": slot["split"], "scene_hash": geometry_hash} if plan else {}),
                }, {"initial_state": initial_state, "qpos": np.asarray(trace_qpos),
                    "ctrl": np.asarray(trace_ctrl), "sim_time": np.asarray(trace_time),
                    "body_xpos": np.asarray(trace_body_pos), "camera_xpos": np.asarray(trace_camera_pos),
                    "substeps": np.asarray(trace_steps), "released_stack": np.asarray(trace_success),
                    "requested_joint_velocity": np.asarray(trace_requested_dq),
                    "applied_joint_velocity": np.asarray(trace_applied_dq),
                    "command_saturation": np.asarray(trace_saturation),
                    "final_qpos": d.qpos.copy()})
            instruction_counts[instruction] += 1
            episodes_collected += 1
            slot_failures = 0
            accepted_scene_hashes.add(geometry_hash)

            episode_times.append(time.time() - episode_wall_start)
            avg_time = sum(episode_times) / len(episode_times)
            remaining = args.episodes - episodes_collected
            eta_sec = int(avg_time * remaining)
            eta_h, eta_rem = divmod(eta_sec, 3600)
            eta_m, eta_s = divmod(eta_rem, 60)
            elapsed_total = time.time() - collection_start
            print(f"  Progress: {episodes_collected}/{args.episodes} | "
                  f"avg {avg_time:.1f}s/ep | "
                  f"ETA {eta_h:02d}h{eta_m:02d}m{eta_s:02d}s | "
                  f"elapsed {elapsed_total / 3600:.2f}h")

            # Progress per instruction variant.
            if episodes_collected % 20 == 0:
                print("\nInstruction coverage so far:")
                for instr, count in sorted(instruction_counts.items(), key=lambda x: x[1]):
                    print(f"  {count:3d}  {instr}")
        else:
            slot_failures += 1
            print("  Discarding — will retry.")

    print(f"\nData collection complete: {episodes_collected} episodes saved.")
    print("\nFinal instruction coverage:")
    for instr, count in sorted(instruction_counts.items(), key=lambda x: x[1]):
        print(f"  {count:3d}  {instr}")

    dataset.finalize()
    if recorder:
        splits = ({key: [i for i, s in enumerate(plan["slots"]) if s["split"] == key]
                   for key in plan["splits"]} if plan else {"train": [0, 1, 2, 3], "validation": [4, 5]})
        recorder.finalize(episodes_collected, splits)
    print(f"Dataset saved at: {DATASET_DIR}")

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
