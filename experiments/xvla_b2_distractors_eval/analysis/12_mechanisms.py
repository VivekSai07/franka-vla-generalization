"""Stage 11: final first-cause mechanism table + local-region test for the grasp-offset layouts + failure CSV."""
import glob, json, collections, csv
import numpy as np
import mujoco
from src.robots.franka_panda.config import SCENE_PATH
from experiments.b2_distractor_data import plan as bp
from experiments.xvla_b2_distractors_eval import evaluate as ev

RUN = '/vla/outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
model = mujoco.MjModel.from_xml_path(SCENE_PATH); mjd = mujoco.MjData(model); HAND = model.body('hand').id
R = json.load(open('/out/trials_behaviour.json'))
C1, C2 = 33400, 50000
rng = np.random.RandomState(31)


def trace(r):
    root = f"{RUN}/evaluation_report_step_{r['ckpt']:06d}"
    ep = glob.glob(f"{root}/trial_{r['ordinal']:03d}_*/episode.json")[0]
    return np.load(ep.replace('episode.json', 'trace.npz'))


def hand_frame(state, pt):
    mjd.qpos[:9] = state; mujoco.mj_kinematics(model, mjd)
    return mjd.xmat[HAND].reshape(3, 3).T @ (np.asarray(pt) - mjd.xpos[HAND])


for r in R:
    t = trace(r); act = t['applied_action'][:, 7]; closed = act <= 127
    lift = t['pick_lift_m']; xy = t['xy_m']; z = t['z_m']
    opens = np.flatnonzero(closed[:-1] & ~closed[1:]) + 1
    carried = [i for i in opens if lift[max(0, i - 5):i + 1].max() > 0.04]
    r['rel_tick'] = int(carried[-1]) if carried else None
    r['rel_xy'] = float(xy[carried[-1]]) if carried else None
    fc = r['stages'].get('first_closure_tick')
    r['hf'] = (hand_frame(t['state'][fc], t['pick_position'][fc]) * 100).tolist() if fc is not None else None
    r['held_end'] = bool(t['terminal_state'][7:9].mean() > 0.015 and t['terminal_state'][7:9].mean() < 0.028)
    pl, pk = t['place_position'], t['pick_position']
    bl = np.flatnonzero(pl[:, 2] - pl[0, 2] > 0.05); pkl = np.flatnonzero(pk[:, 2] - pk[0, 2] > 0.05)
    r['base_first_lift'] = int(bl[0]) if len(bl) else None
    r['pick_first_lift'] = int(pkl[0]) if len(pkl) else None


def mechanism(r):
    s = r['stages']; bc = s.get('first_bilateral_contact_tick'); lf = s.get('first_contact_lift_gt_5cm_tick')
    if r['success']: return 'SUCCESS'
    handled_wrong = ((r['mv_place'] > 0.05 and r['mv_pick'] < 0.02) or r['reversed_stack']
                     or (r['base_first_lift'] is not None
                         and (r['pick_first_lift'] is None or r['base_first_lift'] < r['pick_first_lift'])))
    if handled_wrong: return 'M5 wrong-cube manipulation (instruction/identity error)'
    if lf is None:
        if bc is None or r['near_pick'] < 60 and r['closes'] >= 1 and r['mv_pick'] < 0.02:
            return 'M1 acquisition failure: never secured the cube (empty close / freeze / flailing)'
        return 'M2 grasped but never lifted (cube lost or re-grasp loop)'
    if r['rel_tick'] is None or r['rel_tick'] >= 1100 or r['term_lift'] > 0.05:
        return 'M3 too slow: hesitated, ran out of the 40 s budget (incl. near-misses)'
    if r['rel_xy'] is not None and r['rel_xy'] < 0.02:
        return 'M4b release looked fine (<2 cm) but cube slid / ended >=2 cm off'
    return 'M4a release offset >=2 cm: cube set down off-centre and toppled/slid'


for r in R: r['mech'] = mechanism(r)
labels = ['M1 acquisition failure: never secured the cube (empty close / freeze / flailing)',
          'M2 grasped but never lifted (cube lost or re-grasp loop)',
          'M3 too slow: hesitated, ran out of the 40 s budget (incl. near-misses)',
          'M4a release offset >=2 cm: cube set down off-centre and toppled/slid',
          'M4b release looked fine (<2 cm) but cube slid / ended >=2 cm off',
          'M5 wrong-cube manipulation (instruction/identity error)']
print('=' * 110); print('W. FIRST-CAUSE MECHANISM TABLE (mutually exclusive, all 41 failures)'); print('=' * 110)
print(f'{"mechanism":88s} {"33400":>6s} {"50000":>6s} {"total":>6s}')
for m in labels:
    a = sum(1 for r in R if r['mech'] == m and r['ckpt'] == C1); b = sum(1 for r in R if r['mech'] == m and r['ckpt'] == C2)
    print(f'{m:88s} {a:6d} {b:6d} {a+b:6d}')
print(f'{"TOTAL failures":88s} {sum(1 for r in R if not r["success"] and r["ckpt"]==C1):6d} {sum(1 for r in R if not r["success"] and r["ckpt"]==C2):6d}')
print('\nby condition (control/distractor):')
for m in labels:
    row = []
    for ck in (C1, C2):
        row.append(f'{sum(1 for r in R if r["mech"]==m and r["ckpt"]==ck and r["condition"]=="control"):2d}/{sum(1 for r in R if r["mech"]==m and r["ckpt"]==ck and r["condition"]=="distractor"):2d}')
    print(f'   {m[:70]:70s} 33400 {row[0]}   50000 {row[1]}')
print('\nwhich layouts produce which mechanism (layout: count):')
for m in labels:
    c = collections.Counter(r['layout'][-3:] for r in R if r['mech'] == m)
    print(f'   {m[:2]}: {dict(sorted(c.items(), key=lambda kv: -kv[1]))}')

print('\n' + '=' * 110); print('X. IS THE GRASP-OFFSET PROBLEM LOCAL TO ONE LAYOUT OR A WEAK REGION? (hand-frame x offset at first closure, cm)'); print('=' * 110)
lay = {}
for r in R:
    if r['hf'] is not None: lay.setdefault(r['layout'], []).append((r['ckpt'], r['success'], r['hf'], r['pick_xy']))
def lay_hx(l): return np.mean([h[0] for _, _, h, _ in lay[l]]), np.mean([h[1] for _, _, h, _ in lay[l]]), np.mean([h[2] for _, _, h, _ in lay[l]])
allx = np.array([lay_hx(l)[0] for l in lay]); ally = np.array([lay_hx(l)[1] for l in lay]); allz = np.array([lay_hx(l)[2] for l in lay])
print(f'layout-mean hand-frame offsets over 40 layouts: x {allx.mean():+.2f} sd {allx.std():.2f} | y {ally.mean():+.2f} sd {ally.std():.2f} | z {allz.mean():.2f} sd {allz.std():.2f}')
for target in ('evaluation_054', 'evaluation_048'):
    tx, ty, tz = lay_hx(target); p = np.array(next(iter(lay[target]))[3])
    print(f'-- {target}: pick_xy {np.round(p,3)}  hand-frame (x,y,z) = ({tx:+.2f},{ty:+.2f},{tz:.2f})  z-score of layout-mean x: {(tx-allx.mean())/allx.std():+.1f}, y: {(ty-ally.mean())/ally.std():+.1f}, z: {(tz-allz.mean())/allz.std():+.1f}')
    nbrs = sorted(((np.linalg.norm(np.array(lay[l][0][3]) - p), l) for l in lay if l != target))[:5]
    for d, l in nbrs:
        hx = lay_hx(l); f = sum(1 for r in R if r['layout'] == l and not r['success'])
        print(f'      neighbour {l[-3:]} {100*d:4.1f} cm away  hand (x,y,z)=({hx[0]:+.2f},{hx[1]:+.2f},{hx[2]:.2f})  fails {f}/8  pick={next(r for r in R if r["layout"]==l)["pick"]} cell={next(r for r in R if r["layout"]==l)["cell"]}')

print('\n-- pick-position coverage in the TRAINING set (same pick colour x arrangement cell) for acquisition-failing layouts')
plan = bp.make_plan(); layouts = {x['layout_id']: x for x in plan['layouts']}
tr = collections.defaultdict(list)
for e in plan['episodes'][:160]:
    c = layouts[e['layout_id']]['cube_xy_m']; p = c[:2] if e['pick'] == 'red' else c[2:]
    tr[e['cell']].append(p)
hold = plan['evaluation_holdout']; picks = ev.eval_picks()
def pick_nn(l):
    h = hold[l]; pk = picks[l]; cell = next(c for c, s in bp.CELLS.items() if s['pick'] == pk and s['arrangement'] == h['arrangement'])
    c = h['cube_xy_m']; p = np.array(c[:2] if pk == 'red' else c[2:])
    d = np.linalg.norm(np.array(tr[cell]) - p, axis=1)
    return float(d.min()), int((d < 0.05).sum()), cell
acq = sorted({r['layout'] for r in R if r['mech'].startswith('M1') or r['mech'].startswith('M2')})
rest = sorted({r['layout'] for r in R} - set(acq))
for name, group in (('acquisition-failing layouts', acq), ('all other layouts', rest)):
    v = np.array([pick_nn(l)[0] for l in group]); k = np.array([pick_nn(l)[1] for l in group])
    print(f'   {name:30s} n={len(group):2d}: nearest same-cell training pick {100*v.mean():.1f} cm (median {100*np.median(v):.1f}); # training picks within 5 cm: mean {k.mean():.2f}')
allv = np.array([pick_nn(l)[0] for l in lay]); obs = np.mean([pick_nn(l)[0] for l in acq]) - np.mean([pick_nn(l)[0] for l in rest])
labs = np.array([l in acq for l in lay]); vals = np.array([pick_nn(l)[0] for l in lay])
perm = [vals[p].mean() - vals[~p].mean() for p in (rng.permutation(labs) for _ in range(10000))]
print(f'   diff {100*obs:+.2f} cm, perm p = {(np.sum(np.abs(perm)>=abs(obs))+1)/10001:.3f}')
for l in ('evaluation_054', 'evaluation_048'):
    d, k, cell = pick_nn(l); print(f'   {l}: cell {cell}, nearest training pick {100*d:.1f} cm, {k} training picks within 5 cm')

rows = [r for r in R if not r['success']]
with open('/out/failures.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['ckpt', 'ordinal', 'layout', 'condition', 'cell', 'arrangement', 'pick', 'n_distractors', 'mechanism', 'release_xy_cm', 'final_xy_cm', 'final_z_cm', 'max_lift_cm', 'close_events', 'ticks'])
    for r in sorted(rows, key=lambda r: (r['ckpt'], r['ordinal'])):
        w.writerow([r['ckpt'], r['ordinal'], r['layout'], r['condition'], r['cell'], r['arrangement'], r['pick'], r['n_distractors'], r['mech'][:2],
                    '' if r['rel_xy'] is None else round(100 * r['rel_xy'], 2), round(100 * r['term_xy'], 2), round(100 * r['term_z'], 2), round(100 * r['max_lift'], 1), r['closes'], r['ticks']])
json.dump(R, open('/out/trials_final.json', 'w'))
print('\nwrote /out/failures.csv', len(rows), 'rows')
