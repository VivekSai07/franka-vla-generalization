"""Reproduce a bounded first-approach violation; no dataset writes or policy."""
import json
import argparse
from pathlib import Path
import random
import sys

import mujoco
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_collection.collect_libero_dataset import randomize_cubes, generate_task_states
from data_collection.benchmark import configure_timing, synchronize_observation, step_control
from data_collection.pilot_plan import slot_seed
from src.robots.franka_panda.config import SCENE_PATH, Q_HOME, ACTIVE_JOINTS, ARM_DOF
from src.robots.franka_panda.robot import FrankaPanda
from src.controllers.ik_controller_m2 import IKController
from src.controllers.joint_target_limits import bounded_joint_target
from src.planners.trajectory_planner import TaskSpaceTrajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected", action="store_true")
    args = parser.parse_args()
    plan = json.loads((ROOT / "outputs/benchmark_preparation_20260915/preparation_plan.json").read_text())
    slot = plan["slots"][3]
    seed = slot_seed(slot, 0)
    random.seed(seed); np.random.seed(seed)
    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=10., kp_rot=5.)
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    configure_timing(model, 30)
    data = mujoco.MjData(model)
    randomize_cubes(model, data, slot["colors"])
    data.qpos[:len(Q_HOME)] = Q_HOME
    data.qvel[:] = 0
    for _ in range(120): mujoco.mj_step(model, data)
    synchronize_observation(model, data)
    home = pin.neutral(robot.model)
    home[:len(Q_HOME)] = Q_HOME
    rotation = robot.forward_kinematics(home).rotation.copy()
    phase = generate_task_states(.105)[0]
    target = data.xpos[model.body(f"target_cube_{slot['pick']}").id] + phase["offset"]
    target += np.random.normal(0, .005, 3)
    trajectory = TaskSpaceTrajectory(start_pose=robot.forward_kinematics(data.qpos[:robot.model.nq]),
                                    end_pose=pin.SE3(rotation, target), duration=phase["duration"], method="cubic")
    qtarget = np.zeros(model.nq)
    qtarget[:len(Q_HOME)] = Q_HOME
    started = data.time
    saturation_count = 0
    bounds = np.column_stack((np.maximum(model.actuator_ctrlrange[ACTIVE_JOINTS, 0], robot.model.lowerPositionLimit[ACTIVE_JOINTS]),
                              np.minimum(model.actuator_ctrlrange[ACTIVE_JOINTS, 1], robot.model.upperPositionLimit[ACTIVE_JOINTS])))
    for tick in range(60):
        synchronize_observation(model, data)
        current = data.qpos[:robot.model.nq].copy()
        dq, _, _ = ik.compute_velocity(current, trajectory.get_pose(data.time - started), q_posture=home)
        if args.corrected:
            target, applied, saturated = bounded_joint_target(qtarget[ACTIVE_JOINTS], dq[ACTIVE_JOINTS], 1/30, bounds)
            qtarget[ACTIVE_JOINTS] = target
            saturation_count += int(saturated.any())
        else:
            qtarget[ACTIVE_JOINTS] += np.clip(dq[ACTIVE_JOINTS], -2., 2.) / 30
        data.ctrl[ACTIVE_JOINTS] = qtarget[ACTIVE_JOINTS]
        data.ctrl[-1] = 255
        command = qtarget[:7].astype(np.float32)
        violations = np.flatnonzero((command < bounds[:, 0] - 1e-6) | (command > bounds[:, 1] + 1e-6))
        if len(violations):
            print(json.dumps({"slot": slot["slot_id"], "seed": seed, "tick": tick,
                              "sim_time": data.time, "elapsed": data.time - started,
                              "violations": [{"joint": int(i + 1), "command_rad": float(command[i]),
                                              "measured_rad": float(current[i]), "bounds_rad": bounds[i].tolist(),
                                              "dq_rad_s": float(dq[i])} for i in violations]}, indent=2))
            if args.corrected:
                raise AssertionError("Corrected expert still violated bounds")
            return
        step_control(model, data, 1 / 30, ARM_DOF)
    if args.corrected:
        print(json.dumps({"passed": True, "seed": seed, "ticks": 60, "saturated_ticks": saturation_count,
                          "scope": "first approach only; full task success requires preparation run"}))
    else:
        raise AssertionError("Expected first-approach violation was not reproduced")


if __name__ == "__main__": main()