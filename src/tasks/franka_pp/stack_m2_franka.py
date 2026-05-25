import time
import random
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
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
        if body_id == -1:
            continue
            
        jnt_id = model.body_jntadr[body_id]
        if jnt_id == -1:
            continue
            
        qpos_idx = model.jnt_qposadr[jnt_id]
        
        while True:
            # Table boundaries based on scene.xml (approximate)
            x = random.uniform(0.35, 0.55)
            y = random.uniform(-0.25, 0.25)
            
            conflict = False
            for (sx, sy) in spawned_positions:
                if (x - sx)**2 + (y - sy)**2 < 0.06**2: # 6cm clearance to be safe
                    conflict = True
                    break
            
            if not conflict:
                spawned_positions.append((x, y))
                # Set X, Y, Z (drop from 5cm)
                data.qpos[qpos_idx:qpos_idx+3] = [x, y, 0.05]
                # Reset orientation to identity and velocity to zero
                data.qpos[qpos_idx+3:qpos_idx+7] = [1, 0, 0, 0]
                qvel_idx = model.jnt_dofadr[jnt_id]
                data.qvel[qvel_idx:qvel_idx+6] = 0.0
                break

def get_body_position(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()

def main():
    print("🚀 Initializing Franka Mark-2 Systems for Stacking...")
    
    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=5.0, kp_rot=3.0)

    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)
    grasp_sys = GraspController(m, d, target="target_cube_red")

    # Randomize cube positions
    randomize_cubes(m, d)

    n_joints = min(len(Q_HOME), m.nq)
    d.qpos[:n_joints] = Q_HOME[:n_joints]
    
    q_target = np.zeros(m.nq)
    q_target[:n_joints] = Q_HOME[:n_joints]

    q_home_pin = np.zeros(robot.model.nq)
    q_home_pin[:n_joints] = Q_HOME[:n_joints]
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS = home_pose.translation.copy()

    # Step physics a few times to let cubes settle
    for _ in range(100):
        mujoco.mj_step(m, d)

    GRIPPER_Z_OFFSET = 0.105 
    # Box geom size=0.02 0.02 0.02. Full height is 0.04. 
    # So to stack perfectly, we need a 4cm offset in Z.
    STACK_OFFSET = 0.04 
    
    states = [
        {"name": "Approach Pick", "target": "red", "offset": np.array([0, 0, GRIPPER_Z_OFFSET + 0.15]), "gripper": 255, "duration": 2.0},
        {"name": "Descend to Pick", "target": "red", "offset": np.array([0, 0, GRIPPER_Z_OFFSET]), "gripper": 255, "duration": 1.5},
        {"name": "Grasping", "target": "red", "offset": np.array([0, 0, GRIPPER_Z_OFFSET]), "gripper": 0, "duration": 1.0},
        {"name": "Lift", "target": "red", "offset": np.array([0, 0, GRIPPER_Z_OFFSET + 0.2]), "gripper": 0, "duration": 2.0},
        {"name": "Move to Place", "target": "blue", "offset": np.array([0, 0, GRIPPER_Z_OFFSET + 0.2]), "gripper": 0, "duration": 3.0},
        {"name": "Lower to Place", "target": "blue", "offset": np.array([0, 0, GRIPPER_Z_OFFSET + STACK_OFFSET + 0.015]), "gripper": 0, "duration": 2.0},
        {"name": "Release", "target": "blue", "offset": np.array([0, 0, GRIPPER_Z_OFFSET + STACK_OFFSET + 0.015]), "gripper": 255, "duration": 1.0},
        {"name": "Retract", "target": "blue", "offset": np.array([0, 0, GRIPPER_Z_OFFSET + 0.2]), "gripper": 255, "duration": 2.0},
        {"name": "Return to Home", "target": "home", "offset": np.zeros(3), "gripper": 255, "duration": 3.0},
    ]

    current_state_idx = 0
    state_start_time = 0.0

    print("🟢 Simulation Online. Engaging physics...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()
            q_current = d.qpos[:robot.model.nq].copy()

            if current_state_idx < len(states):
                state = states[current_state_idx]

                # Dynamic Tracking
                if state["target"] == "red":
                    base_pos = get_body_position(m, d, "target_cube_red")
                    target_pos = base_pos + state["offset"]
                elif state["target"] == "blue":
                    base_pos = get_body_position(m, d, "target_cube_blue")
                    target_pos = base_pos + state["offset"]
                else:
                    target_pos = HOME_POS

                target_se3 = pin.SE3(fixed_rotation, target_pos)
                
                dq, err, done = ik.compute_velocity(
                    q_current, 
                    target_se3, 
                    q_posture=q_home_pin
                )

                max_speed = 1.0
                dq = np.clip(dq, -max_speed, max_speed)
                
                for idx in ACTIVE_JOINTS:
                    q_target[idx] += dq[idx] * m.opt.timestep
                    d.ctrl[idx] = q_target[idx]

                grasp_sys.command(state["gripper"])

                is_holding = grasp_sys.is_grasped()

                if state["name"] == "Grasping":
                    if is_holding and (d.time - state_start_time > state["duration"]):
                        print(f"[{d.time:.2f}s] ✅ Grasp secured! Lifting payload.")
                        current_state_idx += 1
                        state_start_time = d.time
                else:
                    if done or (d.time - state_start_time > state["duration"]):
                        current_state_idx += 1
                        state_start_time = d.time
                        if current_state_idx < len(states):
                            print(f"[{d.time:.2f}s] State → {states[current_state_idx]['name']}")

            else:
                for idx in ACTIVE_JOINTS:
                    d.ctrl[idx] = q_target[idx]

            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]

            mujoco.mj_step(m, d)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

if __name__ == "__main__":
    main()
