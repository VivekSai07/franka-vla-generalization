"""Stage 1: extract one feature row per report trial (both checkpoints) -> /out/trials.json."""
import glob, json, math, os
import numpy as np
import mujoco
from src.robots.franka_panda.config import SCENE_PATH
from experiments.local_position_probe.scenes import cube_qpos
from experiments.b2_distractor_data import plan as bp

RUN = '/vla/outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
CKPTS = (33400, 50000)
TABLE_Z = 0.020989

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model); mujoco.mj_forward(model, data)
base = data.xpos[model.body('link0').id].copy()
print('robot base (link0) xpos:', base)
BASE_XY = base[:2]

plan = bp.make_plan()
holdout = plan['evaluation_holdout']


def seg_dist(p, a, b):
    """distance from point p to segment ab, and projection parameter t."""
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab))), t


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


rows = []
for ckpt in CKPTS:
    root = f'{RUN}/evaluation_report_step_{ckpt:06d}'
    for ep_path in sorted(glob.glob(f'{root}/trial_*/episode.json')):
        d = json.load(open(ep_path))
        t = np.load(ep_path.replace('episode.json', 'trace.npz'))
        scene = np.load(f"{root}/{d['layout_id']}_{d['condition']}.npz")
        q0, q1 = scene['qpos'], t['terminal_qpos']
        xy, z, lift = t['xy_m'], t['z_m'], t['pick_lift_m']
        rel, fin = t['released_stack'], t['finger_contact']
        act = t['applied_action']
        pick, place = t['pick_position'], t['place_position']
        closed = act[:, 7] <= 127
        geom_ok = (xy < 0.02) & (z > 0.035) & (z < 0.045)
        n = len(xy)
        end = slice(max(0, n - 100), n)
        r = dict(ckpt=ckpt, ordinal=d['ordinal'], layout=d['layout_id'], condition=d['condition'],
                 cell=d['cell'], arrangement=d['arrangement'], pick=d['pick'], place=d['place'],
                 seed=d['seed'], success=bool(d['task_success']), ticks=n,
                 distractors=d['distractors'], n_distractors=len(d['distractors']),
                 wall_seconds=d['wall_seconds'], clipped_ticks=d['joint_clipped_ticks'],
                 max_clip_rad=d['max_joint_clip_rad'], stages=d['stages'])
        # terminal geometry
        r.update(term_xy=float(xy[-1]), term_z=float(z[-1]), term_lift=float(lift[-1]),
                 term_pick_z=float(pick[-1, 2]), term_finger=bool(fin[-1]),
                 term_open=bool(act[-1, 7] > 127), term_gripper_to_pick=float(t['finger_midpoint_to_pick_m'][-1]),
                 max_lift=float(lift.max()), min_xy_while_lifted=float(xy[lift > 0.05].min()) if (lift > 0.05).any() else None)
        r.update(geom_ok_ticks=int(geom_ok.sum()), geom_ok_run=longest_run(geom_ok),
                 released_run=longest_run(rel), released_ticks=int(rel.sum()),
                 grasp_attempts=int(((~closed[:-1]) & closed[1:]).sum() + int(closed[0])),
                 end_motion_rad=float(np.abs(np.diff(act[end, :7], axis=0)).mean()) if n > 2 else 0.0,
                 place_moved_m=float(np.linalg.norm(place[-1, :2] - place[0, :2])),
                 pick_moved_before_grasp_m=None)
        fc = d['stages'].get('first_closure_tick')
        if fc is not None:
            r['pick_moved_before_grasp_m'] = float(np.linalg.norm(pick[min(fc, n - 1), :2] - pick[0, :2]))
        # layout geometry
        pxy, qxy = pick[0, :2].astype(float), place[0, :2].astype(float)
        r.update(pick_xy=pxy.tolist(), place_xy=qxy.tolist(),
                 pick_reach=float(np.linalg.norm(pxy - BASE_XY)), place_reach=float(np.linalg.norm(qxy - BASE_XY)),
                 pick_place_dist=float(np.linalg.norm(pxy - qxy)),
                 home_jitter_norm_deg=float(np.linalg.norm(holdout[d['layout_id']]['home_pose_jitter_deg'])))
        # distractors: geometry + displacement
        dd = []
        for colour, dxy in d['distractor_xy_m'].items():
            s = cube_qpos(model, colour)
            disp = float(np.linalg.norm(q1[s:s + 2] - q0[s:s + 2]))
            dz = float(q1[s + 2] - q0[s + 2])
            p = np.asarray(dxy, float)
            seg, tproj = seg_dist(p, pxy, qxy)
            dd.append(dict(colour=colour, xy=list(map(float, dxy)), disp_m=disp, dz=dz,
                           to_pick=float(np.linalg.norm(p - pxy)), to_place=float(np.linalg.norm(p - qxy)),
                           to_path=seg, t_proj=tproj))
        r['dist'] = dd
        r['distractor_disturbed_2cm'] = any(x['disp_m'] > 0.02 for x in dd)
        # the four cubes' terminal displacement for the two targets
        rows.append(r)

json.dump(rows, open('/out/trials.json', 'w'))
print('trials extracted:', len(rows), {c: sum(1 for r in rows if r['ckpt'] == c) for c in CKPTS})
print('successes:', {c: sum(1 for r in rows if r['ckpt'] == c and r['success']) for c in CKPTS})
