"""Grasp definitions for the P2 failure diagnostics. Pure numpy; the definitions are fixed in
docs/superpowers/specs/2026-10-19-props-p2-diagnostics-design.md before any policy failure is examined.

Gripper command: 0 is closed, 255 is open; a closed run is at least `min_run` consecutive ticks below the threshold.
"""
import numpy as np

THRESHOLD = 127.5
MIN_RUN = 5
HELD_RUN = 10


def closed_runs(command, min_run=MIN_RUN, threshold=THRESHOLD):
    """[(first, last)] inclusive, for every run of at least `min_run` consecutive ticks below `threshold`."""
    closed = np.asarray(command, dtype=float) < threshold
    runs, start = [], None
    for tick, flag in enumerate(closed):
        if flag and start is None:
            start = tick
        if not flag and start is not None:
            if tick - start >= min_run:
                runs.append((start, tick - 1))
            start = None
    if start is not None and len(closed) - start >= min_run:
        runs.append((start, len(closed) - 1))
    return runs


def first_closure(command):
    runs = closed_runs(command)
    return runs[0][0] if runs else None


def grasp_closure(command, lift_tick):
    """First tick of the closed run containing `lift_tick`; otherwise the first closure; None if never closed."""
    runs = closed_runs(command)
    if not runs:
        return None
    if lift_tick is not None:
        for first, last in runs:
            if first <= lift_tick <= last:
                return first
    return runs[0][0]


def to_hand_frame(hand_pos, hand_rot, point):
    """`point` in the hand's own axes; `hand_rot` is the flat row-major hand-to-world rotation."""
    rotation = np.asarray(hand_rot, dtype=float).reshape(3, 3)
    return rotation.T @ (np.asarray(point, dtype=float) - np.asarray(hand_pos, dtype=float))


def yaw_error_deg(finger_axis_xy, object_yaw_deg):
    """Finger-axis yaw minus object yaw in degrees, modulo 90 (a grasp rotated a quarter turn is the same grasp)."""
    axis = np.degrees(np.arctan2(finger_axis_xy[1], finger_axis_xy[0]))
    return float((axis - object_yaw_deg + 45.0) % 90.0 - 45.0)


def envelope(samples):
    """p1 and p99 per column of an (N, 4) array: hand-frame x, y, z and the yaw error."""
    samples = np.asarray(samples, dtype=float)
    return dict(lo=np.percentile(samples, 1, axis=0), hi=np.percentile(samples, 99, axis=0), n=len(samples))


def inside(env, sample):
    sample = np.asarray(sample, dtype=float)
    return (sample >= env['lo']) & (sample <= env['hi'])


def held_ticks(left_contact, right_contact, start, min_run=HELD_RUN):
    """Both fingers touch the object on at least `min_run` consecutive ticks at or after `start`."""
    both = np.asarray(left_contact, dtype=bool) & np.asarray(right_contact, dtype=bool)
    run = 0
    for flag in both[start:]:
        run = run + 1 if flag else 0
        if run >= min_run:
            return True
    return False
