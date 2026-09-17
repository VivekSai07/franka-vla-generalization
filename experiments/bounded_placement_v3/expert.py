"""One bounded simulated expert invocation, with a physically unchanged v2 prefix.

Call only through independently reviewed diagnostic orchestration. No retries,
resampling, parameter tuning, fallback route or implicit release on failure.
"""
import mujoco
import numpy as np
import pinocchio as pin

from data_collection.benchmark import (
    capture_state, restore_state, synchronize_observation, step_control, stack_metrics,
)
from experiments.paired_demonstrations.expert_v1 import Expert as LegacyExpert
from experiments.paired_demonstrations.media import CAMERAS, pixels_digest
from .collision import CollisionChecker, body_pose
from .gates import ExecutionGuard, PlacementExecution, validate_release_geometry
from .path import plan_route, validate_target
from .prefix import Prefix, placement_endpoints
from .solver import PlanningFailure, finite

EXPERT_VERSION = 'paired_scripted_expert_bounded_placement_v3'
ROBOT_BODIES = [f'link{i}' for i in range(8)]+['hand', 'left_finger', 'right_finger']
GRIPPER_BODIES = ['hand', 'left_finger', 'right_finger']


class Expert(LegacyExpert):
    def __init__(self):
        super().__init__()
        self.config['expert_version'] = EXPERT_VERSION

    def run(self, initial_state, episode, sink, journal):
        """Restore once, prefix once, full route once. Always return terminal state.

        A plan failure can occur before any placement command is executed. The
        caller persists the mutable journal in finally, even if sink/rendering
        raises. Runtime errors are terminal; this method never retries them.
        """
        model, robot = self.model, self.robot
        data = mujoco.MjData(model)
        restore_state(model, data, initial_state)
        initial_time = float(data.time)
        prefix = Prefix(self, data, episode)
        frames, stable, post_ticks = 0, 0, 0
        phase_index, phase_start_tick = -1, 0
        checker, renderer, opened = None, None, False
        reason, success = 'not_started', False
        journal.update(placement={}, retreat={}, events=[], actual_substeps=0)

        def observation():
            synchronize_observation(model, data)
            finite(capture_state(model, data), data.qpos, data.qvel)
            images = {}
            for key, camera in CAMERAS.items():
                renderer.update_scene(data, camera=camera)
                images[key] = renderer.render().copy()
            trace = dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                         integration_state=capture_state(model, data), sim_time=float(data.time),
                         body_xpos=data.xpos.copy(), camera_xpos=data.cam_xpos.copy(),
                         camera_xmat=data.cam_xmat.copy(), image_time=float(data.time),
                         image_sha256=np.asarray([pixels_digest(images[k]) for k in CAMERAS]))
            trace['actual_contacts'] = [dict(geoms=[int(c.geom1), int(c.geom2)], distance_m=float(c.dist))
                                        for c in data.contact]
            return images, trace

        def emit(images, trace, command, steps=True):
            nonlocal frames, stable
            trace.update(command)
            frame = {**images, 'observation.state': trace['qpos'][:9].astype(np.float32),
                     'action': command['action'], 'task': episode['instruction']}
            substeps = step_control(model, data, 1/30, 7) if steps else 0
            if steps and substeps != 20:
                raise PlanningFailure('substep_count', actual=substeps)
            metrics = stack_metrics(model, data, episode['pick'], episode['place'])
            stable = stable+1 if opened and metrics['released_stack'] else 0
            trace.update(substeps=substeps, released_stack=bool(metrics['released_stack']),
                         post_state=capture_state(model, data))
            frames += 1
            journal['actual_substeps'] += substeps
            sink(frame, trace)

        def measurement(target_pose, final_lowering):
            pose = body_pose(model, data, 'hand')
            delta = (data.xpos[model.body(f"target_cube_{episode['pick']}").id]
                     -data.xpos[model.body(f"target_cube_{episode['place']}").id])
            contacts, unexpected = checker.contacts(data, final_lowering)
            row = dict(grasped=prefix.grasp.is_grasped(), xy=float(np.linalg.norm(delta[:2])),
                       z=float(delta[2]), hand_error=float(np.linalg.norm(pose.translation-target_pose.translation)),
                       rotation_error=float(np.linalg.norm(pin.log3(pose.rotation.T@target_pose.rotation))),
                       max_speed=float(np.max(np.abs(data.qvel[:7]))), unexpected=bool(unexpected))
            finite(list(row.values()))
            return row, contacts

        def placement_command(target, gripper, phase, target_pose, measure, contacts, gate_streak=0):
            nonlocal phase_index, phase_start_tick
            validate_target(target, self.bounds)
            delta = target-data.ctrl[:7]
            if np.max(np.abs(delta)) > 2/30+1e-14:
                raise PlanningFailure('execution_command_increment')
            if phase != phase_index:
                phase_index, phase_start_tick = phase, frames
            data.ctrl[:7], data.ctrl[-1] = target, gripper
            return dict(ctrl=data.ctrl.copy(), action=data.ctrl.astype(np.float32),
                        requested_joint_velocity=delta*30, applied_joint_velocity=delta*30,
                        command_saturation=np.zeros(7, dtype=bool), target_pos=target_pose.translation.copy(),
                        phase_index=phase, phase_time=(frames-phase_start_tick)/30,
                        phase_start_tick=phase_start_tick, elapsed_ticks=frames,
                        measurement=measure, contacts=contacts, readiness_streak=gate_streak,
                        intentional_release=opened, failure=None)

        try:
            renderer = mujoco.Renderer(model, height=512, width=512)
            while prefix.phase < 4:
                images, trace = observation()
                command = prefix.command()
                emit(images, trace, command, steps=command['failure'] is None)
                if command['failure']:
                    raise PlanningFailure(command['failure'])
            if frames != 197:
                raise PlanningFailure('prefix_tick_contract', actual=frames)
            # No rewind/reset. This is the state attained by the last prefix step.
            lift_state = capture_state(model, data)
            lift_pose = body_pose(model, data, 'hand').copy()
            journal.update(lift_state=lift_state, lift_pose=lift_pose.homogeneous.copy(), prefix_ticks=frames)
            endpoints = placement_endpoints(prefix, data)
            rng_state = prefix.rng.get_state()
            journal['rng_after_endpoints'] = dict(algorithm=rng_state[0], keys=rng_state[1],
                position=rng_state[2], has_gauss=rng_state[3], cached_gaussian=rng_state[4])
            checker = CollisionChecker(model, data, 'hand', f"target_cube_{episode['pick']}",
                f"target_cube_{episode['place']}", ROBOT_BODIES, GRIPPER_BODIES, range(7))
            journal.update(payload_hand_relative=checker.relative.homogeneous.copy(), filters=checker.filters)
            # Endpoints and payload evidence survive an early height contradiction.
            journal['placement']['endpoints'] = [p.homogeneous.copy() for p in endpoints]
            validate_release_geometry(endpoints[-1], checker.relative,
                                      data.xpos[model.body(f"target_cube_{episode['place']}").id])
            schedule = plan_route(robot, data.qpos[:robot.model.nq], data.ctrl[:7].copy(), endpoints,
                self.bounds, checker, [90, 60, 30], 1200-frames-60-90, journal['placement'])
            control = PlacementExecution(schedule, frames, data.ctrl[:7].copy())
            while not opened:
                images, trace = observation()
                measure, contacts = measurement(endpoints[-1], control.stage >= 1)
                journal['last_measurement'], journal['last_contacts'] = measure, contacts
                target, gripper, stage = control.command(data.qpos[:7], measure)
                opened = control.guard.opened
                if opened:
                    journal['release_tick'] = frames
                    stable = 0
                command = placement_command(target, gripper, 7 if opened else stage+4,
                                            endpoints[-1], measure, contacts, control.release.streak)
                emit(images, trace, command)
                post_ticks += int(opened)
            # Original 1-second open phase, inside the 3-second post-open window.
            guard = ExecutionGuard(frames)
            guard.opened = True
            while post_ticks < 30:
                images, trace = observation()
                measure, contacts = measurement(endpoints[-1], True)
                guard.check(data.qpos[:7], data.ctrl[:7], measure['grasped'], measure['unexpected'])
                emit(images, trace, placement_command(data.ctrl[:7].copy(), 255, 7, endpoints[-1], measure, contacts))
                guard.tick(); post_ticks += 1
            journal['placement_collision_samples'] = checker.journal
            checked = checker.checked
            checker = CollisionChecker(model, data, 'hand', f"target_cube_{episode['pick']}",
                f"target_cube_{episode['place']}", ROBOT_BODIES, [], range(7))
            checker.checked = checked
            # Fixed upward displacement from attained hand, not failed old target.
            retreat_pose = body_pose(model, data, 'hand').copy()
            retreat_pose.translation[2] += prefix.states[7]['offset'][2]-prefix.states[6]['offset'][2]
            retreat = plan_route(robot, data.qpos[:robot.model.nq], data.ctrl[:7].copy(), [retreat_pose],
                self.bounds, checker, [60], min(90-post_ticks, 1200-frames), journal['retreat'], carrying=False)
            index = 0
            while post_ticks < 90:
                images, trace = observation()
                measure, contacts = measurement(retreat_pose, True)
                guard.check(data.qpos[:7], data.ctrl[:7], measure['grasped'], measure['unexpected'])
                target = data.ctrl[:7].copy()
                if index < len(retreat) and guard.can_advance:
                    target = retreat[index][1].copy(); index += 1
                emit(images, trace, placement_command(target, 255, 8, retreat_pose, measure, contacts))
                guard.tick(); post_ticks += 1
            _, unexpected = checker.contacts(data, True)
            finite(capture_state(model, data))
            if unexpected:
                raise PlanningFailure('execution_collision', contacts=unexpected)
            if index != len(retreat) or np.max(np.abs(data.qpos[:7]-data.ctrl[:7])) > .05:
                raise PlanningFailure('retreat_not_complete', sent=index, required=len(retreat))
            success = stable >= 30
            reason = 'success' if success else 'not_released_and_stable_one_second'
        except Exception as exc:
            reason = getattr(exc, 'gate', type(exc).__name__)
            journal['failure'] = dict(first_failed_gate=reason, details=getattr(exc, 'details', {}), message=str(exc))
        finally:
            if checker is not None:
                journal['last_collision_samples'] = checker.journal
            journal.update(terminal_state=capture_state(model, data), terminal_ctrl=data.ctrl.copy(),
                           frames=frames, post_release_ticks=post_ticks, initial_time=initial_time,
                           final_time=float(data.time), intentional_release=opened)
            if renderer is not None:
                renderer.close()
        return dict(success=bool(success), reason=reason, frames=frames, stable_ticks=stable,
                    final_metrics=stack_metrics(model, data, episode['pick'], episode['place']),
                    final_qpos=data.qpos.copy(), final_state=capture_state(model, data),
                    final_ctrl=data.ctrl.copy(), actual_substeps=journal['actual_substeps'])