"""Versioned explicit run-method copy from pose_snapshot_v2/expert.py for the robosuite scene.

Same pattern as pose_snapshot_v2: inherit, copy only run, replace exactly one
thing -- the grasp height literal .105 becomes contract.GRASP_Z_OFFSET_M.
tests/test_expert_fork.py compares the normalized ASTs. The model is the
robosuite-composed one from scene.build_sim, so __init__ does not load SCENE_PATH.
"""
from experiments.paired_demonstrations.expert_v1 import (
    mujoco, np, pin, instruction, Q_HOME, GraspController,
    generate_task_states, synchronize_observation, CAMERAS, pixels_digest,
    bounded_joint_target, ACTIVE_JOINTS, validate_expert_commands, step_control,
    ARM_DOF, stack_metrics, capture_state, restore_state,
    configuration, FrankaPanda, IKController,
)
from experiments.pose_snapshot_v2.expert import Expert
from experiments.pose_snapshot_v2.planner import SnapshotTaskSpaceTrajectory
from experiments.robosuite_scene.contract import GRASP_Z_OFFSET_M
from experiments.robosuite_scene.scene import build_sim, overwrite_reset

EXPERT_VERSION = "robosuite_bringup_pose_snapshot_v2"


class RobosuiteExpert(Expert):
    def __init__(self):
        # Deliberately no super().__init__(): that loads the current scene from SCENE_PATH.
        self.env, self.model, _ = build_sim()
        self.robot = FrankaPanda()
        self.config = configuration(self.model, self.robot)
        self.config["expert_version"] = EXPERT_VERSION
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
        states = generate_task_states(GRASP_Z_OFFSET_M, robot_speed=1.)
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
                    trajectory = SnapshotTaskSpaceTrajectory(
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

def run_episode(seed, sink):
    """One red-on-green episode from the deterministic reset."""
    expert = RobosuiteExpert()
    reset = overwrite_reset(expert.model, mujoco.MjData(expert.model))
    episode = {"pick": "red", "place": "green", "seed": seed, "instruction": instruction("red", "green")}
    outcome = expert.run(reset["initial_state"], episode, sink)
    metrics = outcome["final_metrics"]
    return {"success": outcome["success"], "reason": outcome["reason"],
            "released_stack": metrics["released_stack"], "final_xy_m": metrics["xy_m"],
            "final_z_m": metrics["z_m"], "stable_ticks": outcome["stable_ticks"],
            "env": expert.env, "model": expert.model, "initial_state": reset["initial_state"],
            "final_state": outcome["final_state"], "final_qpos": outcome["final_qpos"]}
