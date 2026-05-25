import os
import sys
import time
import random
import argparse

# Fix Windows OpenMP multi-runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer
from pathlib import Path

# Try importing lerobot (must be installed in the env)
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("ERROR: lerobot not installed! Please run 'pip install lerobot' in your environment.")
    sys.exit(1)

# Ensure the root of Smolvla is in path so we can import src
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

INSTRUCTION = "pick up the red cube and place it perfectly on top of the blue cube"

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
            x = random.uniform(0.25, 0.55)
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
    # Placement offset: changed from 0.002 to 0.01 as requested
    return [
        {"name": "Approach Pick", "target": "red", "offset": np.array([0, 0, z_offset + 0.15]), "gripper": 255, "duration": 2.0 / robot_speed},
        {"name": "Descend to Pick", "target": "red", "offset": np.array([0, 0, z_offset]), "gripper": 255, "duration": 1.5 / robot_speed},
        {"name": "Grasping", "target": "red", "offset": np.array([0, 0, z_offset]), "gripper": 0, "duration": 1.0 / robot_speed},
        {"name": "Verify Lift", "target": "red", "offset": np.array([0, 0, z_offset + 0.2]), "gripper": 0, "duration": 2.0 / robot_speed},
        {"name": "Move to Place", "target": "blue", "offset": np.array([0, 0, z_offset + 0.2]), "gripper": 0, "duration": 3.0 / robot_speed},
        {"name": "Lower to Place", "target": "blue", "offset": np.array([0, 0, z_offset + STACK_OFFSET + 0.01]), "gripper": 0, "duration": 2.0 / robot_speed},
        {"name": "Release", "target": "blue", "offset": np.array([0, 0, z_offset + STACK_OFFSET + 0.01]), "gripper": 255, "duration": 1.0 / robot_speed},
        {"name": "Retract", "target": "blue", "offset": np.array([0, 0, z_offset + 0.2]), "gripper": 255, "duration": 2.0 / robot_speed},
        {"name": "Return to Home", "target": "home", "offset": np.zeros(3), "gripper": 255, "duration": 3.0 / robot_speed},
    ]

def main():
    parser = argparse.ArgumentParser(description="Franka LeRobot Data Collection")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to collect")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for data collection")
    parser.add_argument("--robot_speed", type=float, default=1.0, help="Physical speed of the robot")
    parser.add_argument("--playback_speed", type=float, default=1.0, help="Simulation playback speed (1.0 = real-time, 5.0 = 5x faster, 0 = as fast as possible)")
    parser.add_argument("--headless", action="store_true", help="Run completely off-screen without a passive viewer")
    args = parser.parse_args()
    
    FPS = args.fps
    dt_control = 1.0 / FPS
    
    # --- DATASET VERSIONING SYSTEM ---
    BASE_DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
    os.makedirs(BASE_DATASETS_DIR, exist_ok=True)
    
    # Determine the next version number folder
    version = 1
    while os.path.exists(os.path.join(BASE_DATASETS_DIR, f"franka_stacking_v{version}")):
        version += 1
        
    DATASET_REPO_ID = f"franka_stacking_v{version}"
    DATASET_DIR = os.path.join(BASE_DATASETS_DIR, DATASET_REPO_ID)
    
    print(f"📁 Initializing NEW LeRobot Dataset at {DATASET_DIR}...")
    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        root=DATASET_DIR,  
        fps=FPS,
        features={
            "action": {
                "dtype": "float32", "shape": (8,), 
                "names": ["panda_joint1.pos", "panda_joint2.pos", "panda_joint3.pos", "panda_joint4.pos", "panda_joint5.pos", "panda_joint6.pos", "panda_joint7.pos", "gripper.pos"]
            },
            "observation.state": {
                "dtype": "float32", "shape": (9,), 
                "names": ["panda_joint1.pos", "panda_joint2.pos", "panda_joint3.pos", "panda_joint4.pos", "panda_joint5.pos", "panda_joint6.pos", "panda_joint7.pos", "finger1.pos", "finger2.pos"]
            },
            "observation.images.up": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
            "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
            "observation.images.front": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
        }
    )
    
    # --- SYSTEM INITIALIZATION ---
    print(f"Initializing High-Speed Stacking Pipeline at {FPS} FPS...")
    
    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=10.0, kp_rot=5.0)

    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)
    
    grasp_sys = GraspController(m, d, target="target_cube_red")
    
    # Offscreen renderer for datasets (Preserving high-quality 480x640 resolution)
    renderer = mujoco.Renderer(m, height=480, width=640)

    n_joints = min(len(Q_HOME), m.nq)
    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:n_joints] = Q_HOME[:n_joints]
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS = home_pose.translation.copy()
    
    GRIPPER_Z_OFFSET = 0.105 

    # Launch viewer optionally (helpful for debugging, but not necessary for logging)
    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.sync()

    # --- EPISODE LOOP ---
    episodes_collected = 0
    total_episodes = args.episodes
    
    while episodes_collected < total_episodes:
        print(f"\n🎬 --- Starting Episode {episodes_collected + 1}/{total_episodes} ---")
        
        # Fresh physics state
        mujoco.mj_resetData(m, d)
        randomize_cubes(m, d)
        
        d.qpos[:n_joints] = Q_HOME[:n_joints]
        d.qvel[:] = 0.0
        q_target = np.zeros(m.nq)
        q_target[:n_joints] = Q_HOME[:n_joints]
        
        for _ in range(100):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        
        states = generate_task_states(GRIPPER_Z_OFFSET, robot_speed=args.robot_speed)
        
        current_state_idx = 0
        state_start_time = d.time
        current_trajectory = None

        episode_buffer = []
        episode_success = False

        while current_state_idx < len(states):
            step_start = time.time()

            # ==========================================
            # 1. RENDER & LOG EXISTING STATE
            # ==========================================
            renderer.update_scene(d, camera="top_cam")
            img_top = renderer.render()
            
            renderer.update_scene(d, camera="front_cam")
            img_front = renderer.render()
            
            renderer.update_scene(d, camera="gripper_cam")
            img_gripper = renderer.render()
            
            # Pack observation and action (before taking the step)
            q_current = d.qpos[:robot.model.nq].copy()
            
            obs_state = np.zeros(9, dtype=np.float32)
            act_state = np.zeros(8, dtype=np.float32)
            
            # observation.state requires 7 arm joints + 2 finger joints
            obs_state[:7] = q_current[:7]
            obs_state[7] = q_current[7]  # finger1.pos
            obs_state[8] = q_current[8]  # finger2.pos
            
            # action requires 7 arm targets + 1 gripper command
            act_state[:7] = q_target[:7]
            act_state[7] = states[current_state_idx]["gripper"] if current_state_idx < len(states) else 0.0
            
            episode_buffer.append({
                "observation.images.up": img_top,
                "observation.images.wrist": img_gripper,
                "observation.images.front": img_front,
                "observation.state": obs_state,
                "action": act_state
            })
            
            # ==========================================
            # 2. RUN CONTROLLERS
            # ==========================================
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
            
            for idx in ACTIVE_JOINTS:
                q_target[idx] += dq[idx] * dt_control
                d.ctrl[idx] = q_target[idx]

            grasp_sys.command(state["gripper"])
            is_holding = grasp_sys.is_grasped()

            # State Transitions
            if t_state > state["duration"]:
                if state["name"] == "Verify Lift":
                    if not is_holding:
                        print(f"[{d.time:.2f}s] [WARNING] Grasp Failed or Dropped! Restarting Episode Loop...")
                        # If a single drop/failure occurs, we break entirely and try a fresh episode
                        # to ensure only perfect expert data enters the dataset!
                        break
                    else:
                        print(f"[{d.time:.2f}s] [OK] Grasp verified! Proceeding to place.")
                
                current_state_idx += 1
                state_start_time = d.time
                current_trajectory = None
                
                if current_state_idx < len(states):
                    print(f"[{d.time:.2f}s] State → {states[current_state_idx]['name']}")
                else:
                    # Final Placement Verification logic
                    red_id = m.body("target_cube_red").id
                    blue_id = m.body("target_cube_blue").id
                    pos_red = d.xpos[red_id]
                    pos_blue = d.xpos[blue_id]
                    
                    xy_dist = np.linalg.norm(pos_red[:2] - pos_blue[:2])
                    z_diff = pos_red[2] - pos_blue[2]
                    
                    if xy_dist < 0.03 and 0.035 < z_diff < 0.045:
                        print(f"[{d.time:.2f}s] [SUCCESS] Fast-Stacking Sequence Finished. Perfect placement verified!")
                        episode_success = True
                    else:
                        print(f"[{d.time:.2f}s] [FAILURE] Stack misaligned (xy_dist={xy_dist:.3f}, z_diff={z_diff:.3f}). Discarding episode.")
                        episode_success = False

            # ==========================================
            # 3. PHYSICS STEP
            # ==========================================
            target_sim_time = d.time + dt_control
            while d.time < target_sim_time:
                d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
                mujoco.mj_step(m, d)

            if viewer is not None and viewer.is_running():
                viewer.sync()
                
            if args.playback_speed > 0:
                elapsed = time.time() - step_start
                expected_time_per_step = dt_control / args.playback_speed
                if elapsed < expected_time_per_step:
                    time.sleep(expected_time_per_step - elapsed)

        # Check if episode was fully successful to save
        if episode_success:
            print(f"💾 Saving episode with {len(episode_buffer)} frames to dataset...")
            for frame in episode_buffer:
                dataset.add_frame(frame, task=INSTRUCTION)
            dataset.save_episode()
            episodes_collected += 1
        else:
            print("🗑️ Discarding failed episode buffer. (Will auto-retry next iteration)")

    # finalize Dataset
    print(f"\n✅ Data Collection Complete!")
    # In LeRobot 0.3.3, datasets are finalized incrementally on every `dataset.save_episode()`.
    # No extra consolidation step is needed!
    print(f"Dataset successfully stored at: {DATASET_DIR}")
    
    if viewer is not None:
        viewer.close()

if __name__ == "__main__":
    main()