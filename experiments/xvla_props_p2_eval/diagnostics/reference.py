"""Expert grasp envelope for the P2 failure diagnostics, by forward kinematics over the sealed P2 episodes.
Robosuite image (MuJoCo 3.4.0).

Per sealed episode `qpos[t]` is the state before tick t's action and `ctrl[t, 7]` the gripper command applied at tick t,
the same convention the policy replay uses (`replay.replay_trial` reads the state before applying `applied_action[t]`).
The expert's grasp closure is the first closed run of `ctrl[:, 7]`; the hand-frame target position and the yaw error are
measured there with no physics stepping (set `qpos`, `mj_forward`).
"""
import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.xvla_props_p2_eval import scenes
from experiments.xvla_props_p2_eval.diagnostics import grasp

HAND = 'robot0_right_hand'


def _closure_state(path):
    """(model, data, target body id, closure tick) with the model posed at the expert's grasp closure."""
    z = np.load(path, allow_pickle=True)
    target = str(z['object'])
    names = [target, *[str(d) for d in z['distractors']]]
    model = scenes._model(names, str(z['blue_slot']))  # bounded cache: a compiled scene is ~390 MB
    tick = grasp.first_closure(z['ctrl'][:, 7])
    if tick is None:
        raise ValueError(f'{path}: the expert never closes the gripper')
    data = mujoco.MjData(model)
    data.qpos[:] = z['qpos'][tick]
    mujoco.mj_forward(model, data)
    return model, data, model.body(f'prop_{target}').id, tick


def _measure(model, data, body):
    """Hand-frame position of `body` and its yaw error against the finger axis, at the current pose."""
    hand = model.body(HAND).id
    pos = grasp.to_hand_frame(data.xpos[hand], data.xmat[hand], data.xpos[body])
    axis = data.xpos[model.body('right_finger').id] - data.xpos[model.body('left_finger').id]
    yaw = np.degrees(np.arctan2(data.xmat[body][3], data.xmat[body][0]))
    return np.array([*pos, grasp.yaw_error_deg(axis[:2], yaw)])


def expert_grasp_sample(path):
    """(x, y, z) of the target in the hand frame and the yaw error, at the expert's grasp closure."""
    model, data, body, _ = _closure_state(path)
    return _measure(model, data, body)


def expert_hold_check(path):
    """Evidence that the closure tick is the grasp: how far the target moves in the hand frame between 10 and 30 ticks
    after the closure (held: it should not), and how far it rises in the world over the following 30 ticks (carried)."""
    model, data, body, tick = _closure_state(path)
    z = np.load(path, allow_pickle=True)
    hand = model.body(HAND).id

    def at(t):
        data.qpos[:] = z['qpos'][t]
        mujoco.mj_forward(model, data)
        return grasp.to_hand_frame(data.xpos[hand], data.xmat[hand], data.xpos[body]), float(data.xpos[body][2])

    (early, _), (late, z_late), (_, z_end) = at(tick + 10), at(tick + 30), at(min(tick + 60, len(z['qpos']) - 1))
    return dict(drift_m=float(np.linalg.norm(late - early)), rise_m=z_end - z_late)


def build_reference(paths):
    """{object: {lo, hi, n}} (p1 and p99 of the 4-vector sample) over the episodes where that object is the target."""
    samples = {}
    for path in paths:
        name = str(np.load(path, allow_pickle=True)['object'])
        samples.setdefault(name, []).append(expert_grasp_sample(path))
    out = {}
    for name, rows in samples.items():
        env = grasp.envelope(np.array(rows))
        out[name] = dict(lo=env['lo'].tolist(), hi=env['hi'].tolist(), n=env['n'])
    return out


def write_reference(episodes_dir, out):
    reference = build_reference(sorted(Path(episodes_dir).glob('episode_*.npz')))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(reference, indent=2) + '\n')
    return reference
