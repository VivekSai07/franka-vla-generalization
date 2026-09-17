"""CPU float64 saved-control replay; never invokes an expert run method."""
import json
from pathlib import Path
import av
import mujoco
import numpy as np
import pinocchio as pin

from data_collection.benchmark import (
    capture_state, restore_state, synchronize_observation, step_control, stack_metrics,
)
from experiments.paired_demonstrations.media import CAMERAS, pixels_digest
from .collision import CollisionChecker, body_pose
from .gates import PlacementExecution, ready
from .path import validate_target
from .prefix import Prefix
from .solver import PlanningFailure, finite


def read_ticks(out):
    rows = []
    for index, path in enumerate(sorted((Path(out)/'ticks').glob('tick_*.npz'))):
        if path.name != f'tick_{index:06d}.npz':
            raise PlanningFailure('recorded_tick_gap')
        with np.load(path, allow_pickle=False) as arrays:
            row = {key: arrays[key].copy() for key in arrays.files}
        row.update(json.loads(path.with_suffix('.json').read_text()))
        rows.append(row)
    return rows


def verify_reference(expert, initial, episode, rows, result, journal):
    model = expert.model
    data = mujoco.MjData(model)
    restore_state(model, data, initial)
    prefix = Prefix(expert, data, episode)
    previous_target = prefix.q_target[:7].copy()
    stable, substeps, streak, opened = 0, 0, 0, False
    initial_time, phase, phase_start = float(data.time), -1, 0
    control, checker, retreat_index, release_tick = None, None, 0, None
    if len(rows) > 1200 or len(rows) != result['frames']:
        raise PlanningFailure('replay_frame_count')
    for tick, row in enumerate(rows):
        synchronize_observation(model, data)
        for key in ('qpos', 'qvel', 'ctrl', 'integration_state', 'post_state', 'action', 'sim_time', 'phase_time'):
            finite(row[key])
        np.testing.assert_array_equal(data.qpos, row['qpos'])
        np.testing.assert_array_equal(data.qvel, row['qvel'])
        np.testing.assert_array_equal(capture_state(model, data), row['integration_state'])
        for actual, key in ((data.xpos, 'body_xpos'), (data.cam_xpos, 'camera_xpos'), (data.cam_xmat, 'camera_xmat')):
            np.testing.assert_array_equal(actual, row[key])
        if data.time != row['sim_time'] or data.time != row['image_time']:
            raise PlanningFailure('replay_clock')
        if abs((data.time-initial_time)-substeps/600) > 1e-9:
            raise PlanningFailure('replay_total_clock')
        ctrl = row['ctrl']
        validate_target(ctrl[:7], expert.bounds)
        if np.max(np.abs(ctrl[:7]-previous_target)) > 2/30+1e-14 or ctrl[-1] not in (0, 255):
            raise PlanningFailure('replay_command_bound')
        np.testing.assert_array_equal(row['action'], ctrl.astype(np.float32))
        np.testing.assert_array_equal(row['observation_state'], data.qpos[:9].astype(np.float32))
        current_phase = int(row['phase_index'])
        if tick < 197:
            expected = prefix.command()
            for key in ('ctrl', 'phase_index', 'phase_time'):
                np.testing.assert_array_equal(expected[key], row[key])
        else:
            if checker is None:
                from .expert import ROBOT_BODIES, GRIPPER_BODIES
                checker = CollisionChecker(model, data, 'hand', f"target_cube_{episode['pick']}",
                    f"target_cube_{episode['place']}", ROBOT_BODIES, GRIPPER_BODIES, range(7))
                schedule = [(r['stage'], np.asarray(r['q'])) for r in journal['placement']['commands']]
                for stage, minimum in enumerate((90, 60, 30)):
                    if sum(s == stage for s, _ in schedule) < minimum:
                        raise PlanningFailure('replay_phase_minimum')
                control = PlacementExecution(schedule, tick, previous_target)
            if current_phase < 4 or current_phase > 8 or current_phase < phase:
                raise PlanningFailure('replay_phase_order')
            if current_phase != phase:
                phase, phase_start = current_phase, tick
            if row['phase_start_tick'] != phase_start or row['phase_time'] != (tick-phase_start)/30:
                raise PlanningFailure('replay_phase_clock')
            endpoint = pin.SE3(np.asarray(journal['retreat' if current_phase == 8 else 'placement']['endpoints'][-1]))
            hand = body_pose(model, data, 'hand')
            metrics = stack_metrics(model, data, episode['pick'], episode['place'])
            if current_phase == 8:
                checker.gripper = set()
            contacts, unexpected = checker.contacts(data, opened or control.stage >= 1)
            if row['contacts'] != contacts:
                raise PlanningFailure('replay_contact_records')
            measure = dict(grasped=prefix.grasp.is_grasped(), xy=metrics['xy_m'], z=metrics['z_m'],
                           hand_error=float(np.linalg.norm(hand.translation-endpoint.translation)),
                           rotation_error=float(np.linalg.norm(pin.log3(hand.rotation.T@endpoint.rotation))),
                           max_speed=float(np.max(np.abs(data.qvel[:7]))),
                           unexpected=bool(unexpected))
            if row['measurement'] != measure:
                raise PlanningFailure('replay_measurement_records')
            if not opened:
                expected_target, expected_gripper, expected_stage = control.command(data.qpos[:7], measure)
                np.testing.assert_array_equal(ctrl[:7], expected_target)
                if ctrl[-1] != expected_gripper or current_phase != (7 if expected_gripper == 255 else expected_stage+4):
                    raise PlanningFailure('replay_schedule_advance')
                if int(row['readiness_streak']) != control.release.streak:
                    raise PlanningFailure('replay_readiness_streak')
            elif current_phase == 7:
                np.testing.assert_array_equal(ctrl[:7], previous_target)
            elif current_phase == 8:
                if tick-release_tick < 30:
                    raise PlanningFailure('replay_open_phase_minimum')
                retreat = journal['retreat']['commands']
                expected_target = previous_target.copy()
                if retreat_index < len(retreat) and np.max(np.abs(data.qpos[:7]-previous_target)) <= .05:
                    expected_target = np.asarray(retreat[retreat_index]['q'])
                    retreat_index += 1
                np.testing.assert_array_equal(ctrl[:7], expected_target)
            # Only additional final-arrival samples can count toward readiness.
            if current_phase in (6, 7) and not opened and int(row['readiness_streak']) > 0:
                streak = streak+1 if ready(**measure) else 0
            elif not opened:
                streak = 0
            if not opened and ctrl[-1] == 255:
                if streak < 10 or journal.get('release_tick') != tick or current_phase != 7:
                    raise PlanningFailure('replay_release_without_readiness')
                opened = True
                release_tick = tick
            if not opened and ctrl[-1] != 0:
                raise PlanningFailure('replay_unintentional_open')
            if unexpected:
                raise PlanningFailure('replay_unexpected_contact')
        data.ctrl[:] = ctrl
        count = int(row['substeps'])
        if count == 0:
            if tick != len(rows)-1 or result.get('reason') != 'grasp' or current_phase != 3:
                raise PlanningFailure('replay_zero_substeps')
        elif count != 20 or step_control(model, data, 1/30, 7) != 20:
            raise PlanningFailure('replay_substeps')
        substeps += count
        np.testing.assert_array_equal(capture_state(model, data), row['post_state'])
        metrics = stack_metrics(model, data, episode['pick'], episode['place'])
        if bool(row['released_stack']) != metrics['released_stack']:
            raise PlanningFailure('replay_stack_metric')
        stable = stable+1 if opened and metrics['released_stack'] else 0
        previous_target = ctrl[:7].copy()
    np.testing.assert_array_equal(data.qpos, result['final_qpos'])
    np.testing.assert_array_equal(capture_state(model, data), result['final_state'])
    np.testing.assert_array_equal(data.ctrl, result['final_ctrl'])
    if stable != result['stable_ticks'] or substeps != result['actual_substeps']:
        raise PlanningFailure('replay_final_count')
    if result['success'] and (not opened or stable < 30 or journal.get('post_release_ticks', 91) > 90):
        raise PlanningFailure('replay_false_success')
    return dict(qpos_max_error=0., float64_integration_states_exact=True, ticks=len(rows),
                actual_substeps=substeps, stable_ticks=stable, intentional_release=opened)


def verify_videos(out, rows):
    counts = {}
    for column, camera in enumerate(CAMERAS.values()):
        with av.open(str(Path(out)/f'{camera}.mkv')) as video:
            count = 0
            for tick, frame in enumerate(video.decode(video=0)):
                if tick >= len(rows) or abs(float(frame.pts*frame.time_base)-tick/30) > .000501:
                    raise PlanningFailure('video_tick_alignment')
                if pixels_digest(frame.to_ndarray(format='rgb24')) != rows[tick]['image_sha256'][column]:
                    raise PlanningFailure('video_pixels')
                count += 1
            if count != len(rows):
                raise PlanningFailure('video_frame_count')
            counts[camera] = count
    return counts