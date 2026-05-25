import os
import sys
import time
import argparse
import numpy as np
import requests
import base64
import cv2

import mujoco
import mujoco.viewer
from pathlib import Path

# Ensure the root is in path so we can import src
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.controllers.grasp_controller import GraspController
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    SCENE_PATH,
    Q_HOME,
    ARM_DOF,
    ACTIVE_JOINTS
)

# --- SERVER CONFIGURATION ---
# In Docker Compose the server runs as a separate service named "server".
# Set SERVER_URL=http://server:8000/predict in the client container env,
# or leave unset to use localhost (works when both run on the host directly).
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000/predict")
SIM_FPS = 30
MAX_STEPS = 600  # 20 seconds timeout at 30 FPS

# --- DIRECTORY SETUP ---
BASE_VIDEO_DIR = os.path.join(CURRENT_DIR, "inference_videos")
os.makedirs(BASE_VIDEO_DIR, exist_ok=True)

def encode_image(img_arr):
    """ Encodes RGB image to Base64 JPEG for fast network transmission """
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')

def randomize_cubes(model, data):
    """Randomly spawn the 4 cubes on the table without overlapping."""
    import random
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

def main():
    parser = argparse.ArgumentParser(description="Franka SmolVLA Inference")
    parser.add_argument("--episodes", type=int, default=5, help="Number of inference episodes")
    parser.add_argument("--render", action="store_true", help="Open MuJoCo UI viewer (WARNING: Can severely slow down inference)")
    args = parser.parse_args()

    # 1. Check if the policy server is running
    print("🔍 Checking if SmolVLA server is running on port 8000...")
    try:
        requests.get(SERVER_URL.replace("/predict", "/docs"))
    except requests.exceptions.ConnectionError:
        print("❌ Server not found! Please start `policy_server.py` first.")
        sys.exit(1)

    # Automatically find the next run directory
    run_idx = 1
    while os.path.exists(os.path.join(BASE_VIDEO_DIR, f"test_{run_idx}")):
        run_idx += 1
    VIDEO_OUT_DIR = os.path.join(BASE_VIDEO_DIR, f"test_{run_idx}")
    os.makedirs(VIDEO_OUT_DIR, exist_ok=True)
    print(f"📁 Created new run directory: {VIDEO_OUT_DIR}")

    inference_log = []

    # 2. Initialize MuJoCo physics & Robot
    print("⚙️ Initializing MuJoCo Physics Engine in Headless Mode...")
    robot = FrankaPanda()
    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)
    
    # We will dynamically update the target in the loop, default to red
    grasp_sys = GraspController(m, d, target="target_cube_red")
    
    # Setup renderer for Neural Network cameras
    renderer = mujoco.Renderer(m, height=480, width=640)
    
    n_joints = min(len(Q_HOME), m.nq)
    dt_control = 1.0 / SIM_FPS

    # Optional UI Viewer (Disabled by default for performance)
    viewer = None
    if args.render:
        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.sync()

    start_msg = "🟢 Inference Starting! The SmolVLA model is now controlling the Franka arm."
    print(f"\n{start_msg}")
    inference_log.append(start_msg)

    # 3. Inference Loop
    for ep in range(args.episodes):
        ep_msg = f"\n--- Starting Evaluation Episode {ep + 1}/{args.episodes} ---"
        print(ep_msg)
        inference_log.append(ep_msg)

        # Reset Environment
        mujoco.mj_resetData(m, d)
        randomize_cubes(m, d)
        d.qpos[:n_joints] = Q_HOME[:n_joints]
        d.qvel[:] = 0.0
        
        # Settle physics
        for _ in range(100):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        
        # ✨ Dynamically pick task instruction to test Vision-Language grounding
        if np.random.rand() > 0.5:
            pick_color, base_color = "red", "blue"
            target_cube, base_cube = "target_cube_red", "target_cube_blue"
        else:
            pick_color, base_color = "blue", "red"
            target_cube, base_cube = "target_cube_blue", "target_cube_red"
            
        current_instruction = f"pick up the {pick_color} cube and place it perfectly on top of the {base_color} cube"
        print(f"📝 Instruction: {current_instruction}")
        
        # Update the grasp controller with the correct target for this episode
        grasp_sys.target_name = target_cube

        step_count = 0
        episode_video_frames = []
        task_succeeded = False
        success_timer = 0

        while step_count < MAX_STEPS:
            # --- A. RENDER CAMERAS (ONLY TWO NOW!) ---
            # Assuming 'front_cam' is your primary agentview for the LIBERO model
            renderer.update_scene(d, camera="front_cam")
            img_agentview = renderer.render()
            
            # Wrist camera
            renderer.update_scene(d, camera="gripper_cam")
            img_wrist = renderer.render()

            # Stitch for debugging/saving video (Agentview | Wrist)
            combined_frame = np.concatenate([img_agentview, img_wrist], axis=1)
            combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
            episode_video_frames.append(combined_frame_bgr)

            # --- B. EXTRACT STATE ---
            q_current = d.qpos[:robot.model.nq].copy()
            obs_state = np.zeros(9, dtype=np.float32)
            obs_state[:7] = q_current[:7]
            obs_state[7] = q_current[7]  # finger1
            obs_state[8] = q_current[8]  # finger2

            # --- C. REQUEST ACTION FROM POLICY SERVER ---
            req_data = {
                "agentview_image_b64": encode_image(img_agentview),
                "wrist_image_b64": encode_image(img_wrist),
                "state": obs_state.tolist(),
                "instruction": current_instruction
            }

            try:
                response = requests.post(SERVER_URL, json=req_data)
                response.raise_for_status()
                action = response.json()["action"]
            except Exception as e:
                print(f"❌ Network Error communicating with model server: {e}")
                break

            # --- D. APPLY PREDICTED ACTION ---
            # Action is 8-dim: [joint_1 ... joint_7, gripper_cmd]
            predicted_joints = action[:7]
            predicted_gripper = action[7]

            for i, idx in enumerate(ACTIVE_JOINTS):
                d.ctrl[idx] = predicted_joints[i]

            # In training, Grasping = 0, Release = 255
            # We threshold the predicted continuous float to discrete grasp command
            grasp_sys.command(255.0 if predicted_gripper > 127.0 else 0.0)

            # --- E. STEP PHYSICS ---
            target_sim_time = d.time + dt_control
            while d.time < target_sim_time:
                d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
                mujoco.mj_step(m, d)

            if viewer is not None and viewer.is_running():
                viewer.sync()

            # --- F. EVALUATE SUCCESS DYNAMICALLY ---
            pick_id = m.body(target_cube).id
            base_id = m.body(base_cube).id
            pos_pick = d.xpos[pick_id]
            pos_base = d.xpos[base_id]
            
            xy_dist = np.linalg.norm(pos_pick[:2] - pos_base[:2])
            z_diff = pos_pick[2] - pos_base[2]
            
            if xy_dist < 0.03 and 0.035 < z_diff < 0.045:
                success_timer += 1
                if success_timer >= 30: # Sustain for 1 second to confirm
                    succ_msg = f"✅ AI Successfully completed the stacking task in {step_count} steps!"
                    print(succ_msg)
                    inference_log.append(succ_msg)
                    task_succeeded = True
                    break
            else:
                success_timer = 0 # reset if it falls

            step_count += 1

        if not task_succeeded:
            fail_msg = "❌ AI failed to complete task within timeout."
            print(fail_msg)
            inference_log.append(fail_msg)

        # --- G. SAVE VIDEO ---
        if len(episode_video_frames) > 0:
            video_path = os.path.join(VIDEO_OUT_DIR, f"eval_episode_{ep+1}.mp4")
            print(f"🎬 Saving inference video to {video_path}...")
            height, width, _ = episode_video_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, SIM_FPS, (width, height))
            for frame in episode_video_frames:
                out.write(frame)
            out.release()
            print("✅ Video saved.")

    # Finish & Log
    log_file_path = os.path.join(VIDEO_OUT_DIR, "inference_summary.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(inference_log) + "\n")
    print(f"\n📝 Saved text summary to {log_file_path}")

    if viewer is not None:
        viewer.close()

if __name__ == "__main__":
    main()