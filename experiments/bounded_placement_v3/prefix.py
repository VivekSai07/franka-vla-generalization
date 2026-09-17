"""Explicit v2 prefix command extraction. No simulation, rewind or monkeypatch.

The caller performs exactly the original observation/command/20-substep loop
through Verify Lift, then hands that same live data to placement. The command
calculation can also be compared against saved observations without a trial.
"""
import numpy as np
import pinocchio as pin

from experiments.paired_demonstrations.expert_v1 import (
    Q_HOME, ACTIVE_JOINTS, GraspController, generate_task_states, instruction,
    bounded_joint_target, validate_expert_commands,
)
from experiments.pose_snapshot_v2.planner import SnapshotTaskSpaceTrajectory
from .solver import PlanningFailure


class Prefix:
    def __init__(self, expert, data, episode):
        if episode['instruction'] != instruction(episode['pick'], episode['place']):
            raise PlanningFailure('instruction_mismatch')
        self.expert, self.data, self.episode = expert, data, episode
        self.rng = np.random.RandomState(episode['seed'])
        n_joints = min(len(Q_HOME), expert.model.nq)
        self.q_target = np.zeros(expert.model.nq)
        self.q_target[:n_joints] = Q_HOME[:n_joints]
        self.q_home_pin = pin.neutral(expert.robot.model)
        self.q_home_pin[:n_joints] = Q_HOME[:n_joints]
        home_pose = expert.robot.forward_kinematics(self.q_home_pin)
        self.fixed_rotation = home_pose.rotation.copy()
        self.home_pos = home_pose.translation.copy()
        self.grasp = GraspController(expert.model, data, target=f"target_cube_{episode['pick']}")
        self.states = generate_task_states(.105, robot_speed=1.)
        self.phase, self.start_time, self.trajectory = 0, data.time, None
        self.frames = 0

    def command(self):
        if self.phase >= 4:
            raise PlanningFailure('prefix_already_complete')
        expert, data, episode = self.expert, self.data, self.episode
        q_current = data.qpos[:expert.robot.model.nq].copy()
        state = self.states[self.phase]
        if self.trajectory is None:
            color = episode[state['phase']]
            base_pos = data.xpos[expert.model.body(f'target_cube_{color}').id].copy()
            self.target_pos = base_pos+state['offset']
            if state['jitter']:
                self.target_pos += self.rng.normal(0, .005, 3)
            self.trajectory = SnapshotTaskSpaceTrajectory(
                start_pose=expert.robot.forward_kinematics(q_current),
                end_pose=pin.SE3(self.fixed_rotation, self.target_pos),
                duration=state['duration'], method='cubic')
        t_state = data.time-self.start_time
        target_se3 = self.trajectory.get_pose(t_state)
        dq, err, done = expert.ik.compute_velocity(q_current, target_se3, q_posture=self.q_home_pin)
        target, applied, saturated = bounded_joint_target(
            self.q_target[ACTIVE_JOINTS], dq[ACTIVE_JOINTS], 1/30, expert.bounds)
        self.q_target[ACTIVE_JOINTS] = target
        data.ctrl[ACTIVE_JOINTS] = target
        action = np.zeros(8, dtype=np.float32)
        action[:7], action[7] = self.q_target[:7], float(state['gripper'])
        self.grasp.command(state['gripper'])
        validate_expert_commands(action[None], expert.bounds)
        row = dict(ctrl=data.ctrl.copy(), action=action,
                   requested_joint_velocity=dq[ACTIVE_JOINTS].copy(), applied_joint_velocity=applied.copy(),
                   command_saturation=saturated.copy(), target_pos=self.target_pos.copy(),
                   phase_index=self.phase, phase_time=t_state, failure=None)
        holding = self.grasp.is_grasped()
        if t_state > state['duration']:
            if state['name'] == 'Verify Lift' and not holding:
                row['failure'] = 'grasp'
            else:
                self.phase += 1
                self.start_time, self.trajectory = data.time, None
        self.frames += 1
        return row


def placement_endpoints(prefix, data):
    """Freeze destination at lift completion and consume the original RNG draws."""
    base = data.xpos[prefix.expert.model.body(f"target_cube_{prefix.episode['place']}").id].copy()
    endpoints = []
    for state in prefix.states[4:7]:
        target = base+state['offset']
        if state['jitter']:
            target += prefix.rng.normal(0, .005, 3)
        endpoints.append(pin.SE3(prefix.fixed_rotation.copy(), target.copy()))
    return endpoints