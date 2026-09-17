"""Versioned extraction of the bounded scripted collector, not a policy/relabeling.

Base: data_collection/collect_libero_dataset.py, SHA256 below. The task-state
generator, IK, trajectory, governor, gripper, timing and scoring are imported
unchanged. Only episode/reset orchestration and RNG ownership differ. The loop
is explicit source, not dynamically rewritten text. UI/ETA/retries are omitted.
"""
import random

import mujoco
import numpy as np
import pinocchio as pin

from data_collection.collect_libero_dataset import generate_task_states
from data_collection.benchmark import (capture_state, configure_timing, restore_state,
                                      synchronize_observation, step_control, stack_metrics)
from data_collection.pilot_integrity import validate_expert_commands, validate_scene_geometry
from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.controllers.joint_target_limits import bounded_joint_target
from src.planners.trajectory_planner import TaskSpaceTrajectory
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import SCENE_PATH, Q_HOME, ACTIVE_JOINTS, ARM_DOF
from experiments.paired_demonstrations.plan import COLORS, instruction
from experiments.paired_demonstrations.media import CAMERAS, pixels_digest

BASE_SHA256 = "35209e08f874689e643f01ac5ff3db6ce493ac81a0e17270511e4470448869cf"
EXPERT_VERSION = "paired_scripted_expert_v1"


def configuration(model, robot):
    timing = configure_timing(model, 30)
    bounds = np.column_stack((
        np.maximum(model.actuator_ctrlrange[ACTIVE_JOINTS, 0], robot.model.lowerPositionLimit[ACTIVE_JOINTS]),
        np.minimum(model.actuator_ctrlrange[ACTIVE_JOINTS, 1], robot.model.upperPositionLimit[ACTIVE_JOINTS])))
    return {"fps": 30, "physics_dt": float(model.opt.timestep), "control_dt_requested": 1 / 30,
            "timing": timing, "camera_names": list(CAMERAS.values()), "camera_size": [512, 512],
            "camera_fovy": model.cam_fovy.tolist(), "camera_pos": model.cam_pos.tolist(),
            "camera_quat": model.cam_quat.tolist(), "expert_version": EXPERT_VERSION,
            "expert_command_contract": {"version": "bounded_joint_integration_v1", "joint_bounds": bounds.tolist(),
                "max_velocity": 2., "position_margin_rad": 1e-6, "saturation_recorded": True,
                "initial_target": np.asarray(Q_HOME[:7]).tolist(),
                "initialization_exception": "reset_controls_during_settling_then_home_target_baseline"}}


def initialize_layout(model, layout):
    """Same spawn and settle algorithm, with an owned random.Random stream.

    Rejection *within* the original 1000-candidate clearance sampler is preserved;
    whole layouts are never resampled after geometry/feasibility failure.
    """
    rng = random.Random(layout["seed"])
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    spawned = []
    for color in COLORS:
        body = model.body(f"target_cube_{color}").id
        joint = model.body_jntadr[body]
        address, velocity = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        data.qpos[address + 3:address + 7] = [1, 0, 0, 0]
        data.qvel[velocity:velocity + 6] = 0
        if color not in layout["colors"]:
            data.qpos[address:address + 3] = [2., 2., -1.]
            continue
        for _ in range(1000):
            x, y = rng.uniform(.20, .60), rng.uniform(-.40, .40)
            if all((x - sx)**2 + (y - sy)**2 >= .15**2 for sx, sy in spawned):
                spawned.append((x, y))
                data.qpos[address:address + 3] = [x, y, .05]
                break
        else:
            raise ValueError("Original clearance sampler exhausted; no layout retry")
    data.qpos[:len(Q_HOME)] = Q_HOME
    data.qvel[:] = 0
    for _ in range(round(.2 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    synchronize_observation(model, data)
    validate_scene_geometry(model, data, layout["colors"])
    return {"initial_state": capture_state(model, data), "qpos": data.qpos.copy(),
            "initial_ctrl": data.ctrl.copy(), "body_xpos": data.xpos.copy(),
            "camera_xpos": data.cam_xpos.copy(), "camera_xmat": data.cam_xmat.copy()}


class Expert:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.robot = FrankaPanda()
        self.config = configuration(self.model, self.robot)
        self.bounds = np.asarray(self.config["expert_command_contract"]["joint_bounds"])
        self.ik = IKController(self.robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=10., kp_rot=5.)

    def run(self, initial_state, episode, sink):
        """Restore once, execute once. Sink receives copied pre-step frame/trace.

        The trace is delivered after this tick's physics solely so its substep
        count/outcome can accompany the pre-step observations and sent controls.
        A failed Verify Lift retains the final unstepped tick (substeps=0).
        """
        if episode["instruction"] != instruction(episode["pick"], episode["place"]):
            raise ValueError("Task label/roles mismatch before expert execution")
        model, robot = self.model, self.robot
        data = mujoco.MjData(model)
        restore_state(model, data, initial_state)
        rng = np.random.RandomState(episode["seed"])
        n_joints = min(len(Q_HOME), model.nq)
        q_target = np.zeros(model.nq)
        q_target[:n_joints] = Q_HOME[:n_joints]
        q_home_pin = pin.neutral(robot.model)
        q_home_pin[:n_joints] = Q_HOME[:n_joints]
        home_pose = robot.forward_kinematics(q_home_pin)
        fixed_rotation = home_pose.rotation.copy()
        home_pos = home_pose.translation.copy()
        grasp = GraspController(model, data, target=f"target_cube_{episode['pick']}")
        states = generate_task_states(.105, robot_speed=1.)
        phase, start_time, trajectory = 0, data.time, None
        stable, frames, success = 0, 0, False
        reason, metrics = "not_released_and_stable_one_second", {}
        renderer = mujoco.Renderer(model, height=512, width=512)
        try:
            while phase < len(states):
                synchronize_observation(model, data)
                images = {}
                for key, camera in CAMERAS.items():
                    renderer.update_scene(data, camera=camera)
                    images[key] = renderer.render().copy()
                q_current = data.qpos[:robot.model.nq].copy()
                trace = {"qpos": data.qpos.copy(), "sim_time": float(data.time),
                         "body_xpos": data.xpos.copy(), "camera_xpos": data.cam_xpos.copy(),
                         "camera_xmat": data.cam_xmat.copy(), "image_time": float(data.time),
                         "image_sha256": np.asarray([pixels_digest(images[key]) for key in CAMERAS]),
                         "phase_index": phase}
                state = states[phase]
                if trajectory is None:
                    if state["phase"] in ("pick", "place"):
                        color = episode[state["phase"]]
                        base_pos = data.xpos[model.body(f"target_cube_{color}").id].copy()
                    else:
                        base_pos = home_pos - state["offset"]
                    target_pos = base_pos + state["offset"]
                    if state["jitter"]:
                        target_pos += rng.normal(0, .005, 3)
                    trajectory = TaskSpaceTrajectory(
                        start_pose=robot.forward_kinematics(q_current),
                        end_pose=pin.SE3(fixed_rotation, target_pos), duration=state["duration"], method="cubic")
                t_state = data.time - start_time
                target_se3 = trajectory.get_pose(t_state)
                dq, err, done = self.ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
                target, applied, saturated = bounded_joint_target(
                    q_target[ACTIVE_JOINTS], dq[ACTIVE_JOINTS], 1 / 30, self.bounds)
                q_target[ACTIVE_JOINTS] = target
                data.ctrl[ACTIVE_JOINTS] = target
                action = np.zeros(8, dtype=np.float32)
                action[:7], action[7] = q_target[:7], float(state["gripper"])
                grasp.command(state["gripper"])
                validate_expert_commands(action[None], self.bounds)
                frame = {**images, "observation.state": q_current[:9].astype(np.float32),
                         "action": action, "task": episode["instruction"]}
                trace.update(ctrl=data.ctrl.copy(), requested_joint_velocity=dq[ACTIVE_JOINTS].copy(),
                             applied_joint_velocity=applied.copy(), command_saturation=saturated.copy(),
                             target_pos=target_pos.copy(), phase_time=t_state)
                holding = grasp.is_grasped()
                if t_state > state["duration"]:
                    if state["name"] == "Verify Lift" and not holding:
                        reason = "grasp"
                        trace.update(substeps=0, released_stack=False)
                        sink(frame, trace)
                        frames += 1
                        break
                    phase += 1
                    start_time, trajectory = data.time, None
                substeps = step_control(model, data, 1 / 30, ARM_DOF)
                metrics = stack_metrics(model, data, episode["pick"], episode["place"])
                stable = stable + 1 if metrics["released_stack"] else 0
                trace.update(substeps=substeps, released_stack=metrics["released_stack"])
                sink(frame, trace)
                frames += 1
                if phase >= len(states):
                    success = stable >= 30
            return {"success": bool(success), "reason": "success" if success else reason,
                    "frames": frames, "stable_ticks": stable, "final_metrics": metrics,
                    "final_qpos": data.qpos.copy(), "final_state": capture_state(model, data)}
        finally:
            renderer.close()