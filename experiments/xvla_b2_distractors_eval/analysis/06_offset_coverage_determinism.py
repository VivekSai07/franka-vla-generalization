"""Stage 5: placement-offset direction, training-coverage, and seed-determinism tests."""
import glob, json, collections
import numpy as np
from experiments.b2_distractor_data import plan as bp
from experiments.xvla_b2_distractors_eval import evaluate as ev

RUN = '/vla/outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
R = json.load(open('/out/trials_behaviour.json'))
rng = np.random.RandomState(99)
plan = bp.make_plan()
C1, C2 = 33400, 50000

print('=' * 100); print('G. PLACEMENT OFFSET DIRECTION for "set down next to target" failures (B4)'); print('=' * 100)
rows = []
for r in R:
    if r['success'] or not r['behaviour'].startswith('B4'): continue
    root = f"{RUN}/evaluation_report_step_{r['ckpt']:06d}"
    ep = glob.glob(f"{root}/trial_{r['ordinal']:03d}_*/episode.json")[0]
    t = np.load(ep.replace('episode.json', 'trace.npz'))
    pk, pl = t['terminal_pick_position'][:2], t['terminal_place_position'][:2]
    off = pk - pl                                   # where the picked cube ended up, relative to the target base cube
    radial = pl / np.linalg.norm(pl)                # unit vector base -> target cube
    tang = np.array([-radial[1], radial[0]])
    rows.append(dict(r=r, off=off, rad=float(off @ radial), tan=float(off @ tang), mag=float(np.linalg.norm(off))))
mags = np.array([x['mag'] for x in rows]); rad = np.array([x['rad'] for x in rows]); tan = np.array([x['tan'] for x in rows])
print(f'n={len(rows)}  offset magnitude: mean {100*mags.mean():.2f} cm  sd {100*mags.std():.2f}  min {100*mags.min():.1f}  max {100*mags.max():.1f}')
print(f'radial (+ = farther from robot base than the target): mean {100*rad.mean():+.2f} cm  sd {100*rad.std():.2f}   tangential mean {100*tan.mean():+.2f} sd {100*tan.std():.2f}')
print(f'  # with radial>0 (beyond target): {(rad>0).sum()}   radial<0 (short of target): {(rad<0).sum()}')
print('  by layout (mean offset magnitude, radial, tangential, n):')
by = collections.defaultdict(list)
for x in rows: by[x['r']['layout']].append(x)
for l, xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    print(f'    {l}  n={len(xs)}  mag {100*np.mean([x["mag"] for x in xs]):.2f}±{100*np.std([x["mag"] for x in xs]):.2f} cm  rad {100*np.mean([x["rad"] for x in xs]):+.2f}  tan {100*np.mean([x["tan"] for x in xs]):+.2f}  pick={xs[0]["r"]["pick"]} cell={xs[0]["r"]["cell"]}')
ov = np.var(mags); within = np.mean([np.var([x['mag'] for x in xs]) for xs in by.values() if len(xs) > 1])
print(f'  variance of magnitude: pooled {ov*1e4:.3f} cm^2, mean within-layout {within*1e4:.3f} cm^2  (within << pooled => layout-specific, as in FAILURES #25)')
print('  (success tolerance is xy < 2.0 cm; a cube wholly beside the target is ~ one cube-width = 4.2 cm)')

print('\n' + '=' * 100); print('H. TRAINING-DATA COVERAGE of the failing layouts'); print('=' * 100)
eps = plan['episodes'][:160]
layouts = {x['layout_id']: x for x in plan['layouts']}
def vec(cube_xy, pick):
    red, green = cube_xy[:2], cube_xy[2:]
    p, q = (red, green) if pick == 'red' else (green, red)
    return np.array([*p, *q]), 
train_v, train_cell = [], []
for e in eps:
    cxy = layouts[e['layout_id']]['cube_xy_m']
    train_v.append(vec(cxy, e['pick'])[0]); train_cell.append(e['cell'])
train_v = np.array(train_v); train_cell = np.array(train_cell)
picks = ev.eval_picks(); hold = plan['evaluation_holdout']
ev_v, ev_cell = {}, {}
for lid, h in hold.items():
    pick = picks[lid]
    ev_v[lid] = vec(h['cube_xy_m'], pick)[0]
    ev_cell[lid] = next(c for c, s in bp.CELLS.items() if s['pick'] == pick and s['arrangement'] == h['arrangement'])
report_ids = sorted({r['layout'] for r in R})
def nn(lid, same_cell):
    pool = train_v[train_cell == ev_cell[lid]] if same_cell else train_v
    d = np.linalg.norm(pool - ev_v[lid], axis=1)
    return float(d.min()), float(np.sort(d)[:3].mean())
per = {l: {ck: sum(1 for r in R if r['layout'] == l and r['ckpt'] == ck and not r['success']) for ck in (C1, C2)} for l in report_ids}
for same in (False, True):
    print(f'-- nearest-neighbour distance (4-D pick xy + place xy, metres) to training episodes, {"SAME-CELL only" if same else "all cells"}')
    for ck in (C1, C2):
        f = [nn(l, same) for l in report_ids if per[l][ck] > 0]; o = [nn(l, same) for l in report_ids if per[l][ck] == 0]
        for idx, name in ((0, '1-NN'), (1, 'mean 3-NN')):
            a = np.array([x[idx] for x in f]); b = np.array([x[idx] for x in o])
            allv = np.concatenate([a, b]); lab = np.array([1] * len(a) + [0] * len(b)); obs = a.mean() - b.mean()
            perm = [allv[p == 1].mean() - allv[p == 0].mean() for p in (rng.permutation(lab) for _ in range(5000))]
            pv = (np.sum(np.abs(perm) >= abs(obs)) + 1) / 5001
            print(f'   ckpt {ck} {name:9s} failing layouts {a.mean():.3f} (n={len(a)}) vs passing {b.mean():.3f} (n={len(b)})  diff {obs:+.3f}  perm p={pv:.3f}')
print('-- per persistent-failing layout: same-cell 1-NN distance vs the distribution over all 60 reserved layouts')
all_nn = np.array([nn(l, True)[0] for l in hold])
for l in ['evaluation_054', 'evaluation_048', 'evaluation_059', 'evaluation_040', 'evaluation_037']:
    d = nn(l, True)[0]
    print(f'   {l} cell {ev_cell[l]}: 1-NN {d:.3f}  (percentile among all 60 layouts: {100*np.mean(all_nn <= d):.0f}%)  pick_xy={np.round(ev_v[l][:2],3)} place_xy={np.round(ev_v[l][2:],3)}')

print('\n' + '=' * 100); print('I. DETERMINISM: is the same layout failing the SAME way regardless of seed / condition / checkpoint?'); print('=' * 100)
for l in ['evaluation_059', 'evaluation_054', 'evaluation_040', 'evaluation_048', 'evaluation_037']:
    print(f'-- {l}')
    for r in sorted([x for x in R if x['layout'] == l], key=lambda x: (x['ckpt'], x['ordinal'])):
        print(f'   ck {r["ckpt"]} ord {r["ordinal"]:3d} {r["condition"][:4]} seed {r["seed"]}  {"OK  " if r["success"] else "FAIL"}  xy={100*r["term_xy"]:5.1f}cm z={100*r["term_z"]:5.1f}cm ticks={r["ticks"]:4d}  {r.get("behaviour","")}')
