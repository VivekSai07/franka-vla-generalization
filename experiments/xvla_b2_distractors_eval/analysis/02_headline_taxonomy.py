"""Stage 2: statistics + failure taxonomy over /out/trials.json."""
import json, math, collections
import numpy as np

R = json.load(open('/out/trials.json'))
rng = np.random.RandomState(12345)
C1, C2 = 33400, 50000


def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def sel(**kw):
    return [r for r in R if all(r[k] == v for k, v in kw.items())]


def rate(rows):
    return sum(r['success'] for r in rows), len(rows)


def cluster_boot(rows, stat, n_boot=4000):
    """Resample LAYOUTS (the independent unit) with replacement."""
    by = collections.defaultdict(list)
    for r in rows: by[r['layout']].append(r)
    L = sorted(by)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(L), len(L))
        vals.append(stat([r for i in pick for r in by[L[i]]]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def pct(k, n): return f'{k}/{n} = {100 * k / n:5.1f}%'


print('#' * 100); print('A. HEADLINE (report tier: 40 layouts x 2 conditions x 2 seeds)'); print('#' * 100)
for ck in (C1, C2):
    rows = sel(ckpt=ck)
    k, n = rate(rows)
    lo, hi = cluster_boot(rows, lambda x: sum(r['success'] for r in x) / len(x))
    wl, wh = wilson(k, n)
    print(f'ckpt {ck}: overall {pct(k, n)}  Wilson[{100*wl:.1f},{100*wh:.1f}]  layout-cluster[{100*lo:.1f},{100*hi:.1f}]')
    for cond in ('control', 'distractor'):
        sub = sel(ckpt=ck, condition=cond)
        k, n = rate(sub)
        lo, hi = cluster_boot(sub, lambda x: sum(r['success'] for r in x) / len(x))
        print(f'   {cond:10s} {pct(k, n)}  cluster[{100*lo:.1f},{100*hi:.1f}]')

print('\n-- within-layout paired contrast: control - distractor success rate (per layout, 2 seeds each) --')
for ck in (C1, C2):
    layouts = sorted({r['layout'] for r in sel(ckpt=ck)})
    diffs = []
    for l in layouts:
        c = [r['success'] for r in sel(ckpt=ck, layout=l, condition='control')]
        d_ = [r['success'] for r in sel(ckpt=ck, layout=l, condition='distractor')]
        diffs.append(np.mean(c) - np.mean(d_))
    diffs = np.array(diffs)
    boots = [rng.choice(diffs, len(diffs)).mean() for _ in range(10000)]
    # sign-flip permutation test on the layout-level differences
    obs = diffs.mean()
    perm = [(diffs * rng.choice([-1, 1], len(diffs))).mean() for _ in range(20000)]
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (len(perm) + 1)
    print(f'ckpt {ck}: mean(control-distractor) = {100*obs:+.2f} pts  95% CI[{100*np.percentile(boots,2.5):+.1f},{100*np.percentile(boots,97.5):+.1f}]  '
          f'perm p={p:.3f}  layouts better-on-control={int((diffs>0).sum())} worse={int((diffs<0).sum())} tied={int((diffs==0).sum())}')

print('\n-- paired checkpoint comparison, same trial ordinal => same layout, condition AND rollout seed --')
a = {r['ordinal']: r for r in sel(ckpt=C1)}
b = {r['ordinal']: r for r in sel(ckpt=C2)}
assert all(a[o]['seed'] == b[o]['seed'] and a[o]['layout'] == b[o]['layout'] and a[o]['condition'] == b[o]['condition'] for o in a)
both = sum(a[o]['success'] and b[o]['success'] for o in a)
only1 = sum(a[o]['success'] and not b[o]['success'] for o in a)
only2 = sum(b[o]['success'] and not a[o]['success'] for o in a)
neither = sum((not a[o]['success']) and (not b[o]['success']) for o in a)
print(f'both succeed {both} | only 33400 {only1} | only 50000 {only2} | both fail {neither}')
nd = only1 + only2
pv = sum(math.comb(nd, i) for i in range(0, min(only1, only2) + 1)) * 2 / 2 ** nd
print(f'McNemar exact two-sided p = {min(1.0, pv):.4f}  (discordant n={nd})')
for cond in ('control', 'distractor'):
    o1 = sum(a[o]['success'] and not b[o]['success'] for o in a if a[o]['condition'] == cond)
    o2 = sum(b[o]['success'] and not a[o]['success'] for o in a if a[o]['condition'] == cond)
    print(f'   {cond}: only33400={o1} only50000={o2}')

print('\n' + '#' * 100); print('B. FAILURE TAXONOMY (mutually exclusive, first-match, from terminal geometry + trace)'); print('#' * 100)


def classify(r):
    s = r['stages']
    if r['success']: return 'SUCCESS'
    if s.get('first_closure_tick') is None or s.get('first_bilateral_contact_tick') is None:
        return 'F1 never_grasped'
    if r['max_lift'] < 0.05:
        return 'F2 grasped_never_lifted'
    xy, z = r['term_xy'], r['term_z']
    if r['geom_ok_ticks'] > 0 and r['geom_ok_run'] >= 30 and not r['released_run'] >= 30:
        return 'F3 geometry_ok_but_release_criterion_unmet'
    if xy < 0.02 and 0.035 < z < 0.045:
        return 'F3 geometry_ok_but_release_criterion_unmet'
    if r['term_lift'] > 0.05:
        return 'F4 still_carrying_at_timeout'
    if xy < 0.02 and not (0.035 < z < 0.045):
        return 'F5 on_target_xy_but_bad_height'
    if xy < 0.08:
        return 'F6 placed_beside_target(2-8cm)'
    return 'F7 dropped_or_placed_far(>8cm)'


for r in R: r['cls'] = classify(r)
order = ['SUCCESS', 'F1 never_grasped', 'F2 grasped_never_lifted', 'F3 geometry_ok_but_release_criterion_unmet',
         'F4 still_carrying_at_timeout', 'F5 on_target_xy_but_bad_height', 'F6 placed_beside_target(2-8cm)',
         'F7 dropped_or_placed_far(>8cm)']
print(f'{"class":45s} {"33400":>8s} {"50000":>8s}   by condition (ctrl/dist) @33400   @50000')
for c in order:
    row = [sum(1 for r in sel(ckpt=ck) if r['cls'] == c) for ck in (C1, C2)]
    bc = [(sum(1 for r in sel(ckpt=ck, condition='control') if r['cls'] == c),
           sum(1 for r in sel(ckpt=ck, condition='distractor') if r['cls'] == c)) for ck in (C1, C2)]
    print(f'{c:45s} {row[0]:8d} {row[1]:8d}   {bc[0][0]:3d}/{bc[0][1]:3d}              {bc[1][0]:3d}/{bc[1][1]:3d}')

print('\n-- raw terminal geometry of ALL failures (sorted by xy) --')
fails = sorted([r for r in R if not r['success']], key=lambda r: r['term_xy'])
print(f'{"ck":>5s} {"ord":>3s} {"layout":>14s} {"cond":>4s} {"cell":>4s} {"xy_cm":>6s} {"z_cm":>6s} {"lift_cm":>7s} {"maxlift":>7s} {"fingr":>5s} {"open":>4s} {"geomOKticks":>11s} {"run":>4s} {"relTicks":>8s} {"grasps":>6s} {"class"}')
for r in fails:
    print(f'{r["ckpt"]:5d} {r["ordinal"]:3d} {r["layout"]:>14s} {r["condition"][:4]:>4s} {r["cell"]:>4s} {100*r["term_xy"]:6.1f} {100*r["term_z"]:6.1f} {100*r["term_lift"]:7.1f} {100*r["max_lift"]:7.1f} '
          f'{str(r["term_finger"])[0]:>5s} {str(r["term_open"])[0]:>4s} {r["geom_ok_ticks"]:11d} {r["geom_ok_run"]:4d} {r["released_ticks"]:8d} {r["grasp_attempts"]:6d} {r["cls"]}')

json.dump(R, open('/out/trials_classified.json', 'w'))
