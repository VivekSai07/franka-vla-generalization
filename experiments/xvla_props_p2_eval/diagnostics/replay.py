"""Replay a recorded P2 evaluation trial, unrendered, and record what the stored trace lacks: hand pose, finger
positions and every contact. Robosuite image (MuJoCo 3.4.0).

The trial's reserved layout is restored (`scenes.restore`) and its recorded `applied_action` is fed through the physics
of `data_collection.benchmark.step_control` unrolled substep by substep, as `props_p2_data.episode.distractor_contacts`
does. Gate 0: the replayed `qpos[:9]` and prop positions equal the recorded `state` and `props_xyz` exactly at every
tick; if they do not, the replay is not the recorded physics and nothing measured from it may be used.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

from data_collection.benchmark import apply_action, synchronize_observation
from experiments.xvla_props_p2_eval import scenes
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF

SUBSTEPS = 20  # 30 fps control over the compiled timestep; `step_control` raises if this changes
HAND = 'robot0_right_hand'


def replay_trial(folder):
    folder = Path(folder)
    row = json.loads((folder/'episode.json').read_text())
    trace = np.load(folder/'trace.npz')
    model, data, names = scenes.restore(row['layout_id'])
    if list(names) != list(row['names']):
        raise ValueError('Restored prop order differs from the recorded trial')
    if round(1/30/model.opt.timestep) != SUBSTEPS:
        raise ValueError('Physics substep count changed')
    prop_ids = [model.body(f'prop_{n}').id for n in names]
    hand, left, right = model.body(HAND).id, model.body('left_finger').id, model.body('right_finger').id
    allowed = {model.body('world').id, model.body('table').id}
    owner = {pid: i for i, pid in enumerate(prop_ids)}
    ticks, n = len(trace['applied_action']), len(names)
    out = dict(names=list(names), hand_pos=np.zeros((ticks, 3)), hand_rot=np.zeros((ticks, 9)),
               left_pos=np.zeros((ticks, 3)), right_pos=np.zeros((ticks, 3)), props_xyz=np.zeros((ticks, n, 3)),
               props_yaw_deg=np.zeros((ticks, n)), left_touch=np.zeros((ticks, n), bool),
               right_touch=np.zeros((ticks, n), bool), touched=np.zeros((ticks, n), bool),
               command=np.asarray(trace['applied_action'][:, 7], dtype=float), gate0=True, first_mismatch=None)
    for tick in range(ticks):
        synchronize_observation(model, data)
        positions = np.stack([data.xpos[i] for i in prop_ids])
        if out['gate0'] and not (np.array_equal(data.qpos[:9], trace['state'][tick])
                                 and np.array_equal(positions, trace['props_xyz'][tick])):
            out['gate0'], out['first_mismatch'] = False, tick
        out['hand_pos'][tick], out['hand_rot'][tick] = data.xpos[hand], data.xmat[hand]
        out['left_pos'][tick], out['right_pos'][tick] = data.xpos[left], data.xpos[right]
        out['props_xyz'][tick] = positions
        for i, pid in enumerate(prop_ids):
            rotation = data.xmat[pid]
            out['props_yaw_deg'][tick, i] = np.degrees(np.arctan2(rotation[3], rotation[0]))
        apply_action(model, data, trace['applied_action'][tick], ACTIVE_JOINTS)
        synchronize_observation(model, data)
        for _ in range(SUBSTEPS):
            data.qfrc_applied[:ARM_DOF] = data.qfrc_bias[:ARM_DOF]
            mujoco.mj_step(model, data)
            for c in range(data.ncon):
                a = int(model.geom_bodyid[data.contact[c].geom1])
                b = int(model.geom_bodyid[data.contact[c].geom2])
                for mine, other in ((a, b), (b, a)):
                    if mine not in owner:
                        continue
                    i = owner[mine]
                    if other == left:
                        out['left_touch'][tick, i] = True
                    elif other == right:
                        out['right_touch'][tick, i] = True
                    if other not in allowed:
                        out['touched'][tick, i] = True
        synchronize_observation(model, data)
    return out
