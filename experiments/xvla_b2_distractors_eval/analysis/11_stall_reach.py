"""Stage 10: stall-after-grasp dwell, and reach / joint-limit margins per layout (the 'unreachable?' question)."""
import glob, json, collections
import numpy as np
import mujoco
from src.robots.franka_panda.config import SCENE_PATH
from experiments.b2_distractor_data import plan as bp

RUN = '/vla/outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
R = json.load(open('/out/trials_behaviour.json'))
C1, C2 = 33400, 50000
lo, hi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
print('joint limits lo', np.round(lo, 2), 'hi', np.round(hi, 2))


def trace(r):
    root = f"{RUN}/evaluation_report_step_{r['ckpt']:06d}"
    ep = glob.glob(f"{root}/trial_{r['ordinal']:03d}_*/episode.json")[0]
    return np.load(ep.replace('episode.json', 'trace.npz'))


rows = []
for r in R:
    t = trace(r); st = t['state'][:, :7]
    s = r['stages']; bc = s.get('first_bilateral_contact_tick'); lf = s.get('first_contact_lift_gt_5cm_tick')
    speed = np.r_[0, np.linalg.norm(np.diff(st, axis=0), axis=1)]
    margin = np.minimum(st - lo, hi - st)                      # per joint distance to the nearest limit
    jmin = margin.min(axis=0)
    # stall: ticks after bilateral contact with joint speed < 0.002 rad/tick while cube in hand (lift ~ 0) before first lift
    dwell = (lf - bc) if (bc is not None and lf is not None) else None
    stall_ticks = None
    if bc is not None:
        end = lf if lf is not None else len(speed)
        seg = speed[bc:end]
        stall_ticks = int((seg < 0.002).sum()) if len(seg) else 0
    rows.append(dict(r=r, dwell=dwell, stall=stall_ticks, jmin=jmin, worst_joint=int(jmin.argmin()), worst_margin=float(jmin.min())))

print('=' * 100); print('U. STALL AFTER GRASP: ticks between bilateral contact and first >5cm lift'); print('=' * 100)
for ck in (C1, C2):
    s = [x['dwell'] for x in rows if x['r']['ckpt'] == ck and x['r']['success'] and x['dwell'] is not None]
    print(f'ckpt {ck} successes: dwell median {int(np.median(s))} p95 {int(np.percentile(s,95))} max {max(s)} ticks (n={len(s)})')
    long_ = [x for x in rows if x['r']['ckpt'] == ck and x['dwell'] is not None and x['dwell'] > 150]
    for x in long_:
        print(f'   LONG DWELL ord {x["r"]["ordinal"]:3d} {x["r"]["layout"]} {x["r"]["cell"]} {"OK  " if x["r"]["success"] else "FAIL"} dwell {x["dwell"]} ticks, of which {x["stall"]} with joint speed <0.002 rad/tick')
    never = [x for x in rows if x['r']['ckpt'] == ck and not x['r']['success'] and x['r']['stages'].get('first_bilateral_contact_tick') is not None and x['r']['stages'].get('first_contact_lift_gt_5cm_tick') is None]
    print(f'   failures with bilateral contact but NEVER lifting: {len(never)}: {[x["r"]["ordinal"] for x in never]}')

print('\n' + '=' * 100); print('V. REACH / JOINT-LIMIT MARGIN: is unreachability a credible explanation?'); print('=' * 100)
names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']
print('min margin to any joint limit over the whole trial (rad), by outcome:')
for ck in (C1, C2):
    for lab, sel in (('success', True), ('failure', False)):
        sub = [x for x in rows if x['r']['ckpt'] == ck and x['r']['success'] == sel]
        m = np.array([x['worst_margin'] for x in sub])
        wj = collections.Counter(names[x['worst_joint']] for x in sub)
        print(f'  ckpt {ck} {lab:8s}: min-margin median {np.median(m):.3f} p5 {np.percentile(m,5):.3f} min {m.min():.3f}; limiting joint {dict(wj)}')
print('\nper-layout: limiting-joint margin (policy trajectory) for layouts in the persistent-failure set vs the layout population')
layout_margin = {}
for x in rows:
    layout_margin.setdefault(x['r']['layout'], []).append(x['worst_margin'])
pop = np.array([np.min(v) for v in layout_margin.values()])
for l in ['evaluation_054', 'evaluation_048', 'evaluation_059', 'evaluation_040', 'evaluation_037']:
    info = next(x['r'] for x in rows if x['r']['layout'] == l)
    print(f'  {l}: min joint margin over all 8 trials {np.min(layout_margin[l]):.3f} rad (pctile among 40 layouts {100*np.mean(pop<=np.min(layout_margin[l])):.0f}%), pick reach {info["pick_reach"]:.3f} m, place reach {info["place_reach"]:.3f} m')
allreach = np.array([next(x['r'] for x in rows if x['r']['layout'] == l)['pick_reach'] for l in layout_margin])
for l in ['evaluation_054', 'evaluation_048']:
    v = next(x['r'] for x in rows if x['r']['layout'] == l)['pick_reach']
    print(f'  {l} pick reach {v:.3f} m: pctile among the 40 report layouts {100*np.mean(allreach<=v):.0f}% (range {allreach.min():.3f}-{allreach.max():.3f})')
# which layouts have the smallest / largest reach, and did they fail?
order = sorted(layout_margin, key=lambda l: next(x['r'] for x in rows if x['r']['layout'] == l)['pick_reach'])
def fails(l): return sum(1 for x in rows if x['r']['layout'] == l and not x['r']['success'])
print('  5 NEAREST-to-base picks:', [(l[-3:], round(next(x['r'] for x in rows if x['r']['layout']==l)['pick_reach'],3), f'fail {fails(l)}/8') for l in order[:5]])
print('  5 FARTHEST picks       :', [(l[-3:], round(next(x['r'] for x in rows if x['r']['layout']==l)['pick_reach'],3), f'fail {fails(l)}/8') for l in order[-5:]])
