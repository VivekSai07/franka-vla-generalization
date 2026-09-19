"""Broad-workspace expert: pose_snapshot_v2 with hover-Z jitter on pick/place approach.

Method diff vs. experiments/pose_snapshot_v2/expert.py:Expert.run:

* The generated `states` list is deep-copied, then `state['offset'][2]` is
  incremented by `episode['hover_z_offset']['pre_grasp']` when
  `state['name'] == 'Approach Pick'` and by `episode['hover_z_offset']['pre_place']`
  when `state['name'] == 'Move to Place'`. Nothing else in `run()` changes.
* Constructor sets `self.config['expert_version']` to
  `'paired_scripted_expert_broad_workspace_v1'`.
* Assertions on `episode['home_pose_jitter_deg']` (7 finite floats within +/-1.0)
  and `episode['hover_z_offset']` (dict keyed `pre_grasp`/`pre_place`, finite
  floats within +/-0.03 m) run before physics.

Home-pose jitter is materialized in the reset arm qpos by scenes.derive_scene
and enters `run` through `initial_state`; the expert only verifies the tag.
"""
import copy

from experiments.pose_snapshot_v2.expert import (
    Expert as LegacyExpert, mujoco, np, pin, instruction, Q_HOME, GraspController,
    generate_task_states, synchronize_observation, CAMERAS, pixels_digest,
    bounded_joint_target, ACTIVE_JOINTS, validate_expert_commands, step_control,
    ARM_DOF, stack_metrics, capture_state, restore_state,
)
from experiments.pose_snapshot_v2.planner import SnapshotTaskSpaceTrajectory

EXPERT_VERSION = 'paired_scripted_expert_broad_workspace_v1'
_PHASES = ('pre_grasp', 'pre_place')
_HOME_MAX_DEG = 1.0
_HOVER_MAX_M = .03
_HOVER_STATE_NAMES = {'Approach Pick': 'pre_grasp', 'Move to Place': 'pre_place'}


def _validate_home_jitter(home):
    if (type(home) is not list or len(home) != 7 or any(type(v) is not float for v in home)
            or not np.isfinite(home).all() or np.any(np.abs(home) > _HOME_MAX_DEG + 1e-12)):
        raise ValueError('home_pose_jitter_deg must be 7 finite floats within +/-1.0 degrees')


def _validate_hover_offset(offset):
    if type(offset) is not dict or set(offset) != set(_PHASES):
        raise ValueError('hover_z_offset must be a dict with exactly pre_grasp/pre_place keys')
    for phase in _PHASES:
        v = offset[phase]
        if type(v) is not float or not np.isfinite(v) or abs(v) > _HOVER_MAX_M + 1e-12:
            raise ValueError(f'hover_z_offset[{phase}] must be a finite float within +/-0.03 m')


def _apply_hover_offset(states, hover_offset):
    """Deep copy the generator output and shift z on the two named states only."""
    modified = copy.deepcopy(states)
    for state in modified:
        phase = _HOVER_STATE_NAMES.get(state['name'])
        if phase is not None:
            state['offset'] = state['offset'].astype(np.float64).copy()
            state['offset'][2] += hover_offset[phase]
    return modified


class Expert(LegacyExpert):
    def __init__(self):
        super().__init__()
        self.config['expert_version'] = EXPERT_VERSION

    def run(self, initial_state, episode, sink):
        """Restore once, execute once. Sink receives copied pre-step frame/trace."""
        if episode['instruction'] != instruction(episode['pick'], episode['place']):
            raise ValueError('Task label/roles mismatch before expert execution')
        _validate_home_jitter(episode.get('home_pose_jitter_deg'))
        _validate_hover_offset(episode.get('hover_z_offset'))
        model, robot = self.model, self.robot
        data = mujoco.MjData(model)
        restore_state(model, data, initial_state)
        rng = np.random.RandomState(episode['seed'])
        n_joints = min(len(Q_HOME), model.nq)
        q_target = np.zeros(model.nq)
        q_target[:n_joints] = Q_HOME[:n_joints]
        q_home_pin = pin.neutral(robot.model)
        q_home_pin[:n_joints] = Q_HOME[:n_joints]
        home_pose = robot.forward_kinematics(q_home_pin)
        fixed_rotation = home_pose.rotation.copy()
        home_pos = home_pose.translation.copy()
        grasp = GraspController(model, data, target=f"target_cube_{episode['pick']}")
        states = _apply_hover_offset(generate_task_states(.105, robot_speed=1.), episode['hover_z_offset'])
        phase, start_time, trajectory = 0, data.time, None
        stable, frames, success = 0, 0, False
        reason, metrics = 'not_released_and_stable_one_second', {}
        renderer = mujoco.Renderer(model, height=512, width=512)
        try:
            while phase < len(states):
                synchronize_observation(model, data)
                images = {}
                for key, camera in CAMERAS.items():
                    renderer.update_scene(data, camera=camera)
                    images[key] = renderer.render().copy()
                q_current = data.qpos[:robot.model.nq].copy()
                trace = {'qpos': data.qpos.copy(), 'sim_time': float(data.time),
                         'body_xpos': data.xpos.copy(), 'camera_xpos': data.cam_xpos.copy(),
                         'camera_xmat': data.cam_xmat.copy(), 'image_time': float(data.time),
                         'image_sha256': np.asarray([pixels_digest(images[key]) for key in CAMERAS]),
                         'phase_index': phase}
                state = states[phase]
                if trajectory is None:
                    if state['phase'] in ('pick', 'place'):
                        color = episode[state['phase']]
                        base_pos = data.xpos[model.body(f'target_cube_{color}').id].copy()
                    else:
                        base_pos = home_pos - state['offset']
                    target_pos = base_pos + state['offset']
                    if state['jitter']:
                        target_pos += rng.normal(0, .005, 3)
                    trajectory = SnapshotTaskSpaceTrajectory(
                        start_pose=robot.forward_kinematics(q_current),
                        end_pose=pin.SE3(fixed_rotation, target_pos), duration=state['duration'], method='cubic')
                t_state = data.time - start_time
                target_se3 = trajectory.get_pose(t_state)
                dq, err, done = self.ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
                target, applied, saturated = bounded_joint_target(
                    q_target[ACTIVE_JOINTS], dq[ACTIVE_JOINTS], 1 / 30, self.bounds)
                q_target[ACTIVE_JOINTS] = target
                data.ctrl[ACTIVE_JOINTS] = target
                action = np.zeros(8, dtype=np.float32)
                action[:7], action[7] = q_target[:7], float(state['gripper'])
                grasp.command(state['gripper'])
                validate_expert_commands(action[None], self.bounds)
                frame = {**images, 'observation.state': q_current[:9].astype(np.float32),
                         'action': action, 'task': episode['instruction']}
                trace.update(ctrl=data.ctrl.copy(), requested_joint_velocity=dq[ACTIVE_JOINTS].copy(),
                             applied_joint_velocity=applied.copy(), command_saturation=saturated.copy(),
                             target_pos=target_pos.copy(), phase_time=t_state)
                holding = grasp.is_grasped()
                if t_state > state['duration']:
                    if state['name'] == 'Verify Lift' and not holding:
                        reason = 'grasp'
                        trace.update(substeps=0, released_stack=False)
                        sink(frame, trace)
                        frames += 1
                        break
                    phase += 1
                    start_time, trajectory = data.time, None
                substeps = step_control(model, data, 1 / 30, ARM_DOF)
                metrics = stack_metrics(model, data, episode['pick'], episode['place'])
                stable = stable + 1 if metrics['released_stack'] else 0
                trace.update(substeps=substeps, released_stack=metrics['released_stack'])
                sink(frame, trace)
                frames += 1
                if phase >= len(states):
                    success = stable >= 30
            return {'success': bool(success), 'reason': 'success' if success else reason,
                    'frames': frames, 'stable_ticks': stable, 'final_metrics': metrics,
                    'final_qpos': data.qpos.copy(), 'final_state': capture_state(model, data)}
        finally:
            renderer.close()
