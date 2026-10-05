"""Finer failure modes, read off a saved trial; never replaces `analysis.classify`. Pure numpy.

Written from the 29 failures of the four screens (`screen_labels.json`, each checked on video)
and frozen before any report trial was read. The pre-fixed `cause` stays the reported cause.
Rules run in order on a failed trial and the first match wins.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.xvla_b3_robosuite_eval import analysis

MODES = ('cube_lost', 'wrong_cube_lifted', 'never_started', 'grasp_failed', 'stacked_on_wrong_cube',
         'stall_above_target', 'near_miss_toppled', 'released_away_from_target',
         'carry_aborted_near_pick', 'other')
LOST_DROP_M = 0.02        # the pick cube ends this far below where it started: off the table
CARRIED_M = 0.10          # lifted this high: it was being carried, not bumped
NEAR_TARGET_M = 0.05      # within this xy distance of the target cube
STALL_Z_M = 0.07          # above the target this far over the stacking height: not descending
STALL_TICKS = 300         # held closed above the target this long
STALL_TAIL = 100          # ... or still held closed there for the last this many ticks
ON_CUBE_XY_M = 0.03       # a cube resting on another: within this xy and 3-5 cm higher
START_RADIUS_M = 0.15     # the pick cube never left this circle around where it started
CLOSED = 128


def _lift(trace, colour):
    colours = [str(c) for c in np.asarray(trace['colours'])]
    z = np.asarray(trace['cube_xyz'])[:, colours.index(colour), 2]
    return z - z[0]


def _stacked_on_other(trace, pick, place):
    cube = np.asarray(trace['cube_xyz'])
    colours = [str(c) for c in np.asarray(trace['colours'])]
    p = colours.index(pick)
    for i, colour in enumerate(colours):
        if colour in (pick, place):
            continue
        dxy = float(np.linalg.norm(cube[-1, p, :2] - cube[-1, i, :2]))
        dz = float(cube[-1, p, 2] - cube[-1, i, 2])
        if dxy < ON_CUBE_XY_M and 0.03 < dz < 0.05:
            return colour
    return None


def failure_mode(trace, pick, place):
    """(mode, details) for a failed trial. `trace`: cube_xyz, colours, applied_action, xy_m, z_m."""
    cube = np.asarray(trace['cube_xyz'])
    colours = [str(c) for c in np.asarray(trace['colours'])]
    p = colours.index(pick)
    gripper = np.asarray(trace['applied_action'])[:, 7]
    xy, z = np.asarray(trace['xy_m']), np.asarray(trace['z_m'])
    closed = gripper < CLOSED
    lift = _lift(trace, pick)
    travel = float(np.linalg.norm(cube[-1, p, :2] - cube[0, p, :2]))
    details = dict(pick_travel_m=travel, closed_ticks=int(closed.sum()))
    if lift[-1] < -LOST_DROP_M:
        return 'cube_lost', details
    first = analysis.first_lifted(trace)
    if first is not None and first != pick:
        return 'wrong_cube_lifted', dict(details, lifted=first)
    lifted = [t for t in range(len(lift) - analysis.SUSTAIN_TICKS + 1)
              if (lift[t:t + analysis.SUSTAIN_TICKS] > analysis.LIFT_M).all()]
    if not lifted:
        return ('never_started' if not closed.any() else 'grasp_failed'), details
    resting_on = _stacked_on_other(trace, pick, place)
    if resting_on:
        return 'stacked_on_wrong_cube', dict(details, resting_on=resting_on)
    hover = closed & (xy < NEAR_TARGET_M) & (z > STALL_Z_M)
    tail = closed[-STALL_TAIL:].all() and xy[-1] < NEAR_TARGET_M and z[-1] > STALL_Z_M
    if hover.sum() >= STALL_TICKS or tail:
        return 'stall_above_target', dict(details, hover_ticks=int(hover.sum()), final_z_m=float(z[-1]))
    high = np.where(lift > CARRIED_M)[0]
    if len(high):
        opened = np.where(~closed[high[0]:])[0]
        if len(opened):
            tick = int(opened[0] + high[0])
            offset = (cube[tick, p, :2] - cube[tick, colours.index(place), :2]).tolist()
            release = dict(release_tick=tick, release_xy_m=float(xy[tick]), release_z_m=float(z[tick]),
                           release_offset_xy_m=offset)
            if xy[tick] < NEAR_TARGET_M:
                return 'near_miss_toppled', dict(details, **release)
            return 'released_away_from_target', dict(details, **release)
    if travel < START_RADIUS_M:
        return 'carry_aborted_near_pick', details
    return 'other', details


def review(directory):
    """Mode of every failed trial under a run directory: {trial folder: dict(cause, mode, ...)}."""
    out = {}
    for folder in sorted(Path(directory).glob('trial_*')):
        episode = json.loads((folder/'episode.json').read_text())
        if episode['task_success']:
            continue
        with np.load(folder/'trace.npz') as trace:
            mode, details = failure_mode({k: trace[k] for k in trace.files}, episode['pick'], episode['place'])
        out[folder.name] = dict(ordinal=episode['ordinal'], cause=episode['cause'], mode=mode, **details)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--dir', required=True, help='an evaluation run directory (read-only)')
    parser.add_argument('--out', required=True, help='where to write failure_modes.json')
    args = parser.parse_args(argv)
    result = review(args.dir)
    Path(args.out).write_text(json.dumps(dict(
        directory=str(args.dir), modes=dict(Counter(v['mode'] for v in result.values())),
        by_cause_and_mode=dict(Counter(f"{v['cause']}|{v['mode']}" for v in result.values())),
        trials=result), indent=2) + '\n')


if __name__ == '__main__':
    main()
