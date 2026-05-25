import time
import random
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.planners.trajectory_planner import TaskSpaceTrajectory
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    SCENE_PATH,
    Q_HOME,
    ARM_DOF,
    ACTIVE_JOINTS
)

def randomize_cubes(model, data):
    """Randomly spawn the 4 cubes on the table without overlapping."""
    cubes = ["target_cube_red", "target_cube_green", "target_cube_blue", "target_cube_yellow"]
    spawned_positions = []
    
    for cube in cubes:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube)
        if body_id == -1: continue
        jnt_id = model.body_jntadr[body_id]
        if jnt_id == -1: continue
        qpos_idx = model.jnt_qposadr[jnt_id]
        
        while True:
            x = random.uniform(0.35, 0.65)
            y = random.uniform(-0.35, 0.35)
            conflict = False
            for (sx, sy) in spawned_positions:
                # Increased clearance from 6cm to 12cm to prevent gripper collision
                if (x - sx)**2 + (y - sy)**2 < 0.12**2:
                    conflict = True
                    break
            
            if not conflict:
                spawned_positions.append((x, y))
                data.qpos[qpos_idx:qpos_idx+3] = [x, y, 0.05]
                data.qpos[qpos_idx+3:qpos_idx+7] = [1, 0, 0, 0]
                qvel_idx = model.jnt_dofadr[jnt_id]
                data.qvel[qvel_idx:qvel_idx+6] = 0.0
                break

def generate_task_states(z_offset, robot_speed=1.0):
    STACK_OFFSET = 0.04
    # Realistic, standard robot durations scaled by robot_speed
    return [
        {"name": "Approach Pick", "target": "red", "offset": np.array([0, 0, z_offset + 0.15]), "gripper": 255, "duration": 2.0 / robot_speed},
        {"name": "Descend to Pick", "target": "red", "offset": np.array([0, 0, z_offset]), "gripper": 255, "duration": 1.5 / robot_speed},
        {"name": "Grasping", "target": "red", "offset": np.array([0, 0, z_offset]), "gripper": 0, "duration": 1.0 / robot_speed},
        {"name": "Verify Lift", "target": "red", "offset": np.array([0, 0, z_offset + 0.2]), "gripper": 0, "duration": 2.0 / robot_speed},
        {"name": "Move to Place", "target": "blue", "offset": np.array([0, 0, z_offset + 0.2]), "gripper": 0, "duration": 3.0 / robot_speed},
        {"name": "Lower to Place", "target": "blue", "offset": np.array([0, 0, z_offset + STACK_OFFSET + 0.015]), "gripper": 0, "duration": 2.0 / robot_speed},
        {"name": "Release", "target": "blue", "offset": np.array([0, 0, z_offset + STACK_OFFSET + 0.015]), "gripper": 255, "duration": 1.0 / robot_speed},
        {"name": "Retract", "target": "blue", "offset": np.array([0, 0, z_offset + 0.2]), "gripper": 255, "duration": 2.0 / robot_speed},
        {"name": "Return to Home", "target": "home", "offset": np.zeros(3), "gripper": 255, "duration": 3.0 / robot_speed},
    ]

import argparse

def main():
    parser = argparse.ArgumentParser(description="Franka Stacking Data Collection")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for data collection and control loop")
    parser.add_argument("--playback_speed", type=float, default=1.0, help="Simulation playback speed (1.0 = real-time, 5.0 = 5x faster, 0 = as fast as possible)")
    parser.add_argument("--robot_speed", type=float, default=1.0, help="Physical speed of the robot. Less than 1 = slower. 1.0 = standard realistic speed (20s episode).")
    args = parser.parse_args()
    
    FPS = args.fps
    dt_control = 1.0 / FPS
    
    print(f"Initializing High-Speed Stacking Pipeline at {FPS} FPS...")
    
    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=10.0, kp_rot=5.0)

    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)
    
    grasp_sys = GraspController(m, d, target="target_cube_red")

    randomize_cubes(m, d)

    n_joints = min(len(Q_HOME), m.nq)
    d.qpos[:n_joints] = Q_HOME[:n_joints]
    q_target = np.zeros(m.nq)
    q_target[:n_joints] = Q_HOME[:n_joints]

    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:n_joints] = Q_HOME[:n_joints]
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS = home_pose.translation.copy()

    for _ in range(100):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)

    GRIPPER_Z_OFFSET = 0.105 
    states = generate_task_states(GRIPPER_Z_OFFSET, robot_speed=args.robot_speed)
    
    current_state_idx = 0
    state_start_time = 0.0
    current_trajectory = None

    print(f"Fast Simulation Online. Control step: {dt_control:.4f}s")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            # --- SENSE ---
            q_current = d.qpos[:robot.model.nq].copy()

            if current_state_idx < len(states):
                state = states[current_state_idx]

                if current_trajectory is None:
                    if state["target"] == "red":
                        target_id = m.body("target_cube_red").id
                        base_pos = d.xpos[target_id].copy()
                    elif state["target"] == "blue":
                        target_id = m.body("target_cube_blue").id
                        base_pos = d.xpos[target_id].copy()
                    else:
                        base_pos = HOME_POS - state["offset"]
                        
                    target_pos = base_pos + state["offset"]

                    start_se3 = robot.forward_kinematics(q_current)
                    end_se3 = pin.SE3(fixed_rotation, target_pos)
                    current_trajectory = TaskSpaceTrajectory(
                        start_pose=start_se3, end_pose=end_se3, 
                        duration=state["duration"], method="cubic"
                    )

                t_state = d.time - state_start_time
                target_se3 = current_trajectory.get_pose(t_state)
                
                dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
                dq = np.clip(dq, -2.0, 2.0)
                
                # Apply control scaled by our macro dt_control
                for idx in ACTIVE_JOINTS:
                    q_target[idx] += dq[idx] * dt_control
                    d.ctrl[idx] = q_target[idx]

                grasp_sys.command(state["gripper"])
                is_holding = grasp_sys.is_grasped()

                if t_state > state["duration"]:
                    if state["name"] == "Verify Lift":
                        if not is_holding:
                            print(f"[{d.time:.2f}s] [WARNING] Grasp Failed or Dropped! Initiating Recovery...")
                            current_state_idx = 0
                            state_start_time = d.time
                            current_trajectory = None
                            continue 
                        else:
                            print(f"[{d.time:.2f}s] [OK] Grasp verified! Proceeding to place.")
                    
                    current_state_idx += 1
                    state_start_time = d.time
                    current_trajectory = None
                    if current_state_idx < len(states):
                        print(f"[{d.time:.2f}s] State → {states[current_state_idx]['name']}")
                    else:
                        print(f"[{d.time:.2f}s] [SUCCESS] Episode Completed Successfully!")
                        break
            else:
                for idx in ACTIVE_JOINTS:
                    d.ctrl[idx] = q_target[idx]

            # --- STEP PHYSICS INTERNALLY TO MATCH dt_control ---
            target_sim_time = d.time + dt_control
            while d.time < target_sim_time:
                d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
                mujoco.mj_step(m, d)

            # Sync viewer at exact desired real-world FPS pace
            viewer.sync()

            # Throttle playback speed so you can actually watch it happening!
            if args.playback_speed > 0:
                elapsed = time.time() - step_start
                expected_time_per_step = dt_control / args.playback_speed
                if elapsed < expected_time_per_step:
                    time.sleep(expected_time_per_step - elapsed)

if __name__ == "__main__":
    main()
