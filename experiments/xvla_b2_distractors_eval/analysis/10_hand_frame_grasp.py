"""Stage 9: grasp pose error in the HAND frame, empty-close events, and time-to-complete decomposition."""
import glob, json, collections
import numpy as np
import mujoco
from src.robots.franka_panda.config import SCENE_PATH

RUN = '/vla/outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
mjd = mujoco.MjData(model)
R = json.load(open('/out/trials_behaviour.json'))
C1, C2 = 33400, 50000
HAND = model.body('hand').id


def trace(r):
    root = f"{RUN}/evaluation_report_step_{r['ckpt']:06d}"
    ep = glob.glob(f"{root}/trial_{r['ordinal']:03d}_*/episode.json")[0]
    return np.load(ep.replace('episode.json', 'trace.npz'))


def hand_frame(state, world_pt):
    mjd.qpos[:9] = state; mujoco.mj_kinematics(model, mjd)
    Rm = mjd.xmat[HAND].reshape(3, 3); return Rm.T @ (np.asarray(world_pt) - mjd.xpos[HAND])


rows = []
for r in R:
    t = trace(r)
    act = t['applied_action'][:, 7]; closed = act <= 127
    bil = np.asarray(t['bilateral_target_contact'], bool)
    pk = t['pick_position']; st = t['state']
    starts = np.flatnonzero((~closed[:-1]) & closed[1:]) + 1
    if closed[0]: starts = np.r_[0, starts]
    # an "empty close": a closing event after which bilateral contact never occurs within the next 40 ticks
    empty = 0; good = 0
    for s in starts:
        if bil[s:s + 40].any(): good += 1
        else: empty += 1
    fc = r['stages'].get('first_closure_tick')
    hf = hand_frame(st[fc], pk[fc]) if fc is not None else None
    fin_end = t['terminal_state'][7:9]
    rows.append(dict(r=r, empty=empty, good=good, hf=hf, fin_end=fin_end, n_close=len(starts)))

print('=' * 100); print('R. GRASP POSE ERROR AT FIRST CLOSURE (cube centre in the HAND frame, cm)'); print('=' * 100)
for ck in (C1, C2):
    s = np.array([x['hf'] for x in rows if x['r']['ckpt'] == ck and x['r']['success'] and x['hf'] is not None]) * 100
    print(f'ckpt {ck} SUCCESS (n={len(s)}): mean {np.round(s.mean(0),2)}  sd {np.round(s.std(0),2)}  [hand x,y,z cm]')
    f = [x for x in rows if x['r']['ckpt'] == ck and not x['r']['success'] and x['hf'] is not None]
    mu, sd = s.mean(0), s.std(0)
    z = [((x['hf'] * 100 - mu) / sd) for x in f]
    worst = sorted(zip(f, z), key=lambda p: -np.abs(p[1]).max())[:8]
    for x, zz in worst:
        print(f'   FAIL ord {x["r"]["ordinal"]:3d} {x["r"]["layout"]} hand-frame {np.round(x["hf"]*100,2)}  z-score vs successes {np.round(zz,1)}  ({x["r"]["behaviour"][:2]})')
    outl = sum(1 for zz in z if np.abs(zz).max() > 3)
    print(f'   failures with any hand-frame grasp coordinate >3 sd from the success distribution: {outl}/{len(f)}')

print('\n' + '=' * 100); print('S. EMPTY CLOSES (gripper closes with no bilateral contact within 40 ticks) and final finger joints'); print('=' * 100)
for ck in (C1, C2):
    for lab, sel in (('success', True), ('failure', False)):
        sub = [x for x in rows if x['r']['ckpt'] == ck and x['r']['success'] == sel]
        e = np.array([x['empty'] for x in sub]); g = np.array([x['good'] for x in sub])
        print(f'ckpt {ck} {lab:8s} n={len(sub):3d}: empty-closes mean {e.mean():.2f} (trials with >=1: {(e>=1).sum()}, >=3: {(e>=3).sum()}, max {e.max()})   contact-closes mean {g.mean():.2f}')
print('terminal finger joints of failures: closed-on-nothing (<0.005 each) vs holding a cube (~0.021)')
for ck in (C1, C2):
    f = [x for x in rows if x['r']['ckpt'] == ck and not x['r']['success']]
    closed_empty = [x for x in f if x['fin_end'].max() < 0.005]
    holding = [x for x in f if abs(x['fin_end'].mean() - 0.021) < 0.004]
    opn = [x for x in f if x['fin_end'].min() > 0.03]
    print(f'  ckpt {ck}: failures {len(f)}: empty-closed {len(closed_empty)} {[x["r"]["ordinal"] for x in closed_empty]} | holding cube {len(holding)} | open {len(opn)}')

print('\n' + '=' * 100); print('T. WHERE DOES THE TIME GO? (ticks to first carry-start vs total) for successes that were slow'); print('=' * 100)
for ck in (C1, C2):
    sl = [x for x in rows if x['r']['ckpt'] == ck and x['r']['success'] and x['r']['ticks'] > 600]
    for x in sl:
        s = x['r']['stages']
        print(f'  ck {ck} ord {x["r"]["ordinal"]} {x["r"]["layout"]}: ticks {x["r"]["ticks"]}, first lift>5cm @{s.get("first_contact_lift_gt_5cm_tick")}, close events {x["n_close"]} (empty {x["empty"]})')
print('median time to first >5cm lift among successes:', {ck: int(np.median([x['r']['stages']['first_contact_lift_gt_5cm_tick'] for x in rows if x['r']['ckpt']==ck and x['r']['success'] and x['r']['stages'].get('first_contact_lift_gt_5cm_tick')])) for ck in (C1, C2)})
print('\nfailures whose first >5cm lift never happened, or happened after tick 600 (slow/ineffective acquisition):')
for ck in (C1, C2):
    for x in rows:
        r = x['r']
        if r['ckpt'] == ck and not r['success']:
            lt = r['stages'].get('first_contact_lift_gt_5cm_tick')
            if lt is None or lt > 600:
                print(f'   ck {ck} ord {r["ordinal"]} {r["layout"]} first lift @{lt}, close events {x["n_close"]} (empty {x["empty"]}), behaviour {r["behaviour"][:2]}')
