"""Copied Cartesian route, branch checks and vertex-preserving 30 Hz commands."""
import math
import numpy as np
import pinocchio as pin

from .solver import PlanningFailure, finite, pose_errors, solve_waypoint


def pose_waypoints(start, endpoints):
    previous, result = start.copy(), []
    finite(previous.homogeneous)
    for stage, endpoint in enumerate(endpoints):
        endpoint = endpoint.copy()
        finite(endpoint.homogeneous)
        distance = np.linalg.norm(endpoint.translation-previous.translation)
        angle = np.linalg.norm(pin.log3(previous.rotation.T@endpoint.rotation))
        n = max(1, math.ceil(distance/.005), math.ceil(angle/.02))
        if len(result)+n > 400:
            raise PlanningFailure('waypoint_limit', required=len(result)+n)
        rotation_log = pin.log3(previous.rotation.T@endpoint.rotation)
        for i in range(1, n+1):
            alpha = i/n
            pose = (endpoint.copy() if i == n else pin.SE3(
                previous.rotation@pin.exp3(alpha*rotation_log),
                (1-alpha)*previous.translation+alpha*endpoint.translation))
            result.append((stage, pose))
        previous = endpoint
    return result


def check_branch(previous, current):
    finite(previous, current)
    jump = float(np.max(np.abs(np.asarray(current)[:7]-np.asarray(previous)[:7])))
    if jump > .05:
        raise PlanningFailure('waypoint_branch_jump', max_joint_displacement_rad=jump)


def validate_target(q, bounds):
    q, bounds = np.asarray(q, dtype=np.float64), np.asarray(bounds, dtype=np.float64)
    finite(q, bounds)
    if q.shape != (7,) or bounds.shape != (7, 2) or np.any(bounds[:, 0]+1e-6 >= bounds[:, 1]-1e-6):
        raise PlanningFailure('command_shape_or_bounds')
    if np.any(q < bounds[:, 0]+1e-6) or np.any(q > bounds[:, 1]-1e-6):
        raise PlanningFailure('command_bounds', command=q.copy())


def time_parameterize(previous_target, vertices, minimum_ticks, bounds, available_ticks):
    """Every vertex is commanded exactly; never skip an intervening bend.

    Include measured start as the first vertex: previous targets remain the
    first command-increment baseline. Phase-minimum padding is distributed
    across that phase's segments, not across endpoints of unrelated segments.
    """
    previous = np.asarray(previous_target, dtype=np.float64).copy()
    validate_target(previous, bounds)
    if not vertices or available_ticks < 0:
        raise PlanningFailure('empty_path_or_deadline')
    segments = []
    for stage, value in vertices:
        target = np.asarray(value, dtype=np.float64).copy()
        validate_target(target, bounds)
        if not 0 <= stage < len(minimum_ticks) or (segments and stage < segments[-1][0]):
            raise PlanningFailure('path_stage_order')
        n = max(1, math.ceil(float(np.max(np.abs(target-previous)))/(2/30)))
        segments.append([stage, previous.copy(), target.copy(), n])
        previous = target
    for stage, minimum in enumerate(minimum_ticks):
        selected = [s for s in segments if s[0] == stage]
        if not selected or minimum < 0:
            raise PlanningFailure('missing_phase_or_minimum')
        extra = max(0, minimum-sum(s[3] for s in selected))
        for i, segment in enumerate(selected):
            segment[3] += extra//len(selected)+(i < extra % len(selected))
    required = sum(s[3] for s in segments)
    if required > available_ticks:
        raise PlanningFailure('path_deadline', required_ticks=required, available_ticks=available_ticks)
    schedule = []
    previous = np.asarray(previous_target, dtype=np.float64).copy()
    for stage, start, end, n in segments:
        for i in range(1, n+1):
            command = end.copy() if i == n else start+(end-start)*(i/n)
            validate_target(command, bounds)
            if np.max(np.abs(command-previous)) > 2/30+1e-14:
                raise PlanningFailure('command_increment')
            schedule.append((stage, command))
            previous = command
    return schedule


def plan_route(robot, measured_q, previous_target, endpoints, bounds, checker,
               minimum_ticks, available_ticks, journal, carrying=True):
    """Plan and check the whole route before returning any executable command."""
    q = np.asarray(measured_q, dtype=np.float64).copy()
    start_pose = robot.forward_kinematics(q).copy()
    vertices = [(0, q[:7].copy())]
    journal.update(start_q=q.copy(), previous_target=np.asarray(previous_target).copy(),
                   start_pose=start_pose.homogeneous.copy(),
                   endpoints=[p.homogeneous.copy() for p in endpoints], waypoints=[])
    # This is a command-space bridge, not a replacement geometric seed.
    # After opening, the stationary released cube may already rest on its base.
    # Keep the same penetration tolerance; it is no longer a carried payload.
    checker.segment(np.asarray(previous_target), q[:7], not carrying, carrying)
    for stage, target in pose_waypoints(start_pose, endpoints):
        record = dict(stage=stage, pose=target.homogeneous.copy(), candidates=[])
        journal['waypoints'].append(record)
        solved = solve_waypoint(robot, q, target, bounds, record['candidates'])
        _, _, position, rotation = pose_errors(robot, solved, target)
        record.update(position_error_m=position, orientation_error_rad=rotation)
        if position > .001 or rotation > .01:
            raise PlanningFailure('independent_fk_residual')
        check_branch(q, solved)
        checker.segment(q[:7], solved[:7], stage >= 1 or not carrying, carrying)
        q = solved.copy()
        vertices.append((stage, q[:7].copy()))
    schedule = time_parameterize(previous_target, vertices, minimum_ticks, bounds, available_ticks)
    journal.update(vertices=[dict(stage=s, q=v) for s, v in vertices],
                   commands=[dict(stage=s, q=v) for s, v in schedule], checked_samples=checker.checked)
    return schedule