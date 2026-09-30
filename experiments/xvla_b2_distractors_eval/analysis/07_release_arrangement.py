"""Stage 6: release-event analysis, near-miss timeouts, arrangement/pick effects at the LAYOUT level."""
import glob, json, collections
import numpy as np

RUN = '/vla/outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
R = json.load(open('/out/trials_behaviour.json'))
rng = np.random.RandomState(2024)
C1, C2 = 33400, 50000


def trace(r):
    root = f"{RUN}/evaluation_report_step_{r['ckpt']:06d}"
    ep = glob.glob(f"{root}/trial_{r['ordinal']:03d}_*/episode.json")[0]
    return np.load(ep.replace('episode.json', 'trace.npz'))


print('=' * 100); print('J. WHAT HAPPENS AT THE RELEASE? (B4 = cube ended beside target)'); print('=' * 100)
print('For each B4 failure: state of the carried cube at the LAST gripper-open event that follows a carry.')
print(f'{"ck":>5s} {"ord":>3s} {"layout":>14s} | at release: {"xy_cm":>6s} {"z_above_target_cm":>17s} {"lift_cm":>7s} | final xy_cm | released_then_ended_beside')
rel = []
for r in R:
    if r['success'] or not r['behaviour'].startswith('B4'): continue
    t = trace(r)
    act = t['applied_action'][:, 7]; closed = act <= 127
    lift = t['pick_lift_m']; xy = t['xy_m']; z = t['z_m']
    opens = np.flatnonzero(closed[:-1] & ~closed[1:]) + 1
    carried = [i for i in opens if lift[max(0, i - 5):i + 1].max() > 0.04]
    if not carried:
        print(f'{r["ckpt"]:5d} {r["ordinal"]:3d} {r["layout"]:>14s} | no open-after-carry event found'); continue
    i = carried[-1]
    # xy when the cube was last within reach of the target: min xy in the 30 ticks before release
    w = slice(max(0, i - 30), i + 1)
    rel.append(dict(r=r, xy_rel=float(xy[i]), z_rel=float(z[i]), lift_rel=float(lift[i]), xy_min_before=float(xy[w].min()),
                    xy_end=float(xy[-1]), z_min_before=float(z[w].min()), tick=int(i)))
    print(f'{r["ckpt"]:5d} {r["ordinal"]:3d} {r["layout"]:>14s} | {100*xy[i]:14.1f} {100*z[i]:17.1f} {100*lift[i]:7.1f} | {100*xy[-1]:8.1f}   tick {i}, min xy in prior 30 ticks {100*xy[w].min():.1f}cm')
xr = np.array([x['xy_rel'] for x in rel]); zr = np.array([x['z_rel'] for x in rel]); xe = np.array([x['xy_end'] for x in rel])
print(f'\nn={len(rel)}  xy at release: mean {100*xr.mean():.2f} cm (median {100*np.median(xr):.2f}); xy final: mean {100*xe.mean():.2f}')
print(f'  released with xy < 2cm (cube correctly over target): {(xr<0.02).sum()}   2-5cm: {((xr>=0.02)&(xr<0.05)).sum()}   >=5cm: {(xr>=0.05).sum()}')
print(f'  release height above target (z): mean {100*zr.mean():.1f} cm (correct resting height is 4.2 cm); released from >6 cm (dropped): {(zr>0.06).sum()}')
print(f'  => correct-xy-at-release but ended beside target (slid off/knocked on opening): {sum(1 for x in rel if x["xy_rel"]<0.02 and x["xy_end"]>0.03)}')

print('\n' + '=' * 100); print('K. NEAR-MISS TIMEOUTS (would more time have helped?)'); print('=' * 100)
for r in R:
    if r['success']: continue
    t = trace(r)
    rel_ = np.asarray(t['released_stack'], bool)
    run = 0
    for v in rel_[::-1]:
        if v: run += 1
        else: break
    if run > 0 or r['geom_ok_run'] >= 15:
        print(f'  ck {r["ckpt"]} ord {r["ordinal"]} {r["layout"]} {r["condition"][:4]}: stable-stack streak at the moment the 1200-tick limit hit = {run}/30 ticks; longest ever {r["geom_ok_run"]}')
slow = [(r['ckpt'], r['ordinal'], r['layout'], r['ticks']) for r in R if r['success'] and r['ticks'] > 700]
print(f'  successes that took >700 ticks (i.e. were slow but finished): {slow}')

print('\n' + '=' * 100); print('L. ARRANGEMENT / PICK EFFECTS TESTED AT THE LAYOUT LEVEL (layout is the independent unit)'); print('=' * 100)
layouts = sorted({r['layout'] for r in R})
info = {l: next(r for r in R if r['layout'] == l) for l in layouts}
for ck in (C1, C2):
    nfail = {l: sum(1 for r in R if r['layout'] == l and r['ckpt'] == ck and not r['success']) for l in layouts}
    for key, vals in (('arrangement', ('familiar', 'swapped')), ('pick', ('red', 'green'))):
        lab = np.array([info[l][key] == vals[0] for l in layouts]); f = np.array([nfail[l] for l in layouts], float)
        obs = f[lab].mean() - f[~lab].mean()
        perm = [f[p].mean() - f[~p].mean() for p in (rng.permutation(lab) for _ in range(20000))]
        pv = (np.sum(np.abs(perm) >= abs(obs)) + 1) / 20001
        print(f'ckpt {ck} {key:11s}: mean failures/layout {vals[0]} {f[lab].mean():.2f} (n={lab.sum()}) vs {vals[1]} {f[~lab].mean():.2f} (n={(~lab).sum()})  diff {obs:+.2f}  layout-level perm p={pv:.4f}')
    tbl = collections.Counter((info[l]['arrangement'], nfail[l] > 0) for l in layouts)
    print(f'   layouts with >=1 failure: familiar {tbl[("familiar",True)]}/{tbl[("familiar",True)]+tbl[("familiar",False)]}, swapped {tbl[("swapped",True)]}/{tbl[("swapped",True)]+tbl[("swapped",False)]}')
print('\n  pooled over BOTH checkpoints, per cell, layouts-with-failure:')
for cell in ('C1', 'C2', 'C3', 'C4'):
    ls = [l for l in layouts if info[l]['cell'] == cell]
    bad = [l for l in ls if any((not r['success']) for r in R if r['layout'] == l)]
    print(f'   {cell}: {len(bad)}/{len(ls)} layouts had a failure at some checkpoint; trial fail rate {sum(1 for r in R if r["cell"]==cell and not r["success"])}/{sum(1 for r in R if r["cell"]==cell)}')
print('\n  does the place-cube y-side (region A y<0 vs B y>0) or pick-cube y-side explain it instead of arrangement?')
for ck in (C1, C2):
    nfail = {l: sum(1 for r in R if r['layout'] == l and r['ckpt'] == ck and not r['success']) for l in layouts}
    for name, fn in (('pick in region B (y>0)', lambda i: i['pick_xy'][1] > 0), ('place in region B (y>0)', lambda i: i['place_xy'][1] > 0)):
        lab = np.array([fn(info[l]) for l in layouts]); f = np.array([nfail[l] for l in layouts], float)
        obs = f[lab].mean() - f[~lab].mean()
        perm = [f[p].mean() - f[~p].mean() for p in (rng.permutation(lab) for _ in range(20000))]
        pv = (np.sum(np.abs(perm) >= abs(obs)) + 1) / 20001
        print(f'   ckpt {ck} {name:26s}: failures/layout {f[lab].mean():.2f} (n={lab.sum()}) vs {f[~lab].mean():.2f} (n={(~lab).sum()})  diff {obs:+.2f}  perm p={pv:.4f}')
