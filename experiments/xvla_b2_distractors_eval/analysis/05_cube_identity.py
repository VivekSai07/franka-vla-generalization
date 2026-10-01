"""Stage 4: identity-aware behavioural classification of every failure (which cube did the policy handle?)."""
import glob, json, collections
import numpy as np
import mujoco
from src.robots.franka_panda.config import SCENE_PATH

RUN = '/vla/outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
R = json.load(open('/out/trials_classified.json'))
mjd = mujoco.MjData(model)


def tips(states):
    out = []
    for s in states:
        mjd.qpos[:9] = s; mujoco.mj_kinematics(model, mjd)
        out.append(0.5 * (mjd.xpos[model.body('left_finger').id] + mjd.xpos[model.body('right_finger').id]))
    return np.array(out)


def analyse(r):
    root = f"{RUN}/evaluation_report_step_{r['ckpt']:06d}"
    ep = glob.glob(f"{root}/trial_{r['ordinal']:03d}_*/episode.json")[0]
    t = np.load(ep.replace('episode.json', 'trace.npz'))
    pk, pl = t['pick_position'], t['place_position']          # pick = INSTRUCTED target, place = INSTRUCTED base
    n = len(pk)
    mv_pick = float(np.linalg.norm(pk[:, :3] - pk[0, :3], axis=1).max())
    mv_place = float(np.linalg.norm(pl[:, :3] - pl[0, :3], axis=1).max())
    tp = tips(t['state'][::5]); closed = t['applied_action'][::5, 7] <= 127
    dpk = np.linalg.norm(tp - pk[::5], axis=1); dpl = np.linalg.norm(tp - pl[::5], axis=1)
    near_pick = int((closed & (dpk < 0.06)).sum() * 5); near_place = int((closed & (dpl < 0.06)).sum() * 5)
    close_events = int(((~(t['applied_action'][:-1, 7] <= 127)) & (t['applied_action'][1:, 7] <= 127)).sum())
    pz_end, lz_end = float(pk[-1, 2]), float(pl[-1, 2])
    dxy = float(t['terminal_xy_m']); dz = float(t['terminal_z_m'])
    # did the WRONG cube end up on top of the instructed target?  (place cube above pick cube by ~cube height)
    reversed_stack = (abs(dz + 0.042) < 0.006 and dxy < 0.03)
    # final resting position relative to the instructed place cube
    speed_end = float(np.linalg.norm(np.diff(t['state'][-150:, :7], axis=0), axis=1).mean()) if n > 150 else 0.0
    r2 = dict(mv_pick=mv_pick, mv_place=mv_place, near_pick=near_pick, near_place=near_place, closes=close_events,
              reversed_stack=bool(reversed_stack), end_speed=speed_end, pk_z_end=pz_end, pl_z_end=lz_end)
    return r2


for r in R:
    if not r['success']: r.update(analyse(r))


def behaviour(r):
    if r['success']: return 'SUCCESS'
    if r['reversed_stack']: return 'B1 executed_REVERSED_instruction (wrong cube stacked, cleanly)'
    if r['mv_pick'] < 0.02 and r['mv_place'] >= 0.05 and r['near_place'] > 0:
        return 'B2 grabbed_WRONG_cube (instructed cube untouched)'
    if r['mv_pick'] < 0.02 and r['mv_place'] < 0.02:
        return 'B3 never_moved_either_cube (hover/dither near pick)' if r['near_pick'] > 0 or r['closes'] > 0 else 'B3 never_moved_either_cube'
    if r['mv_pick'] >= 0.05 and r['term_lift'] < 0.02 and r['term_xy'] < 0.03:
        return 'B6 stacked_but_xy/release_criterion_missed'
    if r['mv_pick'] >= 0.05 and r['term_lift'] >= 0.05:
        return 'B5 still_holding_cube_at_timeout'
    if r['mv_pick'] >= 0.05 and r['term_xy'] < 0.08:
        return 'B4 carried_then_set_down_NEXT_TO_target (2-8cm)'
    if r['mv_pick'] >= 0.05:
        return 'B7 cube_moved_but_ended_far_from_target (>8cm, dropped/pushed/knocked)'
    return 'B8 other'


for r in R: r['behaviour'] = behaviour(r)
order = sorted({r['behaviour'] for r in R if not r['success']})
print(f'{"behaviour":75s} {"33400":>6s} {"50000":>6s}')
for b in order:
    print(f'{b:75s} {sum(1 for r in R if r["behaviour"]==b and r["ckpt"]==33400):6d} {sum(1 for r in R if r["behaviour"]==b and r["ckpt"]==50000):6d}')

print('\n--- per-failure identity detail (sorted by layout) ---')
print(f'{"ck":>5s} {"ord":>3s} {"layout":>14s} {"cell":>4s} {"cond":>4s} {"pickMv":>6s} {"baseMv":>6s} {"nearPick":>8s} {"nearBase":>8s} {"closes":>6s} {"endSpd":>7s}  behaviour')
for r in sorted([x for x in R if not x['success']], key=lambda x: (x['layout'], x['ckpt'], x['ordinal'])):
    print(f'{r["ckpt"]:5d} {r["ordinal"]:3d} {r["layout"]:>14s} {r["cell"]:>4s} {r["condition"][:4]:>4s} {100*r["mv_pick"]:6.1f} {100*r["mv_place"]:6.1f} {r["near_pick"]:8d} {r["near_place"]:8d} {r["closes"]:6d} {r["end_speed"]:7.4f}  {r["behaviour"]}')
json.dump(R, open('/out/trials_behaviour.json', 'w'))
