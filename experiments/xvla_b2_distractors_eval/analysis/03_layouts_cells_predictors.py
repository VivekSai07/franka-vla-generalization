"""Stage 3: layout-level concentration, cell effects, predictors, time budget."""
import json, math, collections
import numpy as np

R = json.load(open('/out/trials_classified.json'))
rng = np.random.RandomState(7)
C1, C2 = 33400, 50000


def sel(rows=None, **kw):
    return [r for r in (rows or R) if all(r[k] == v for k, v in kw.items())]


print('#' * 100); print('C. LAYOUT SPECIFICITY'); print('#' * 100)
layouts = sorted({r['layout'] for r in R})
per = {}
for l in layouts:
    per[l] = {ck: sum(1 for r in sel(layout=l, ckpt=ck) if not r['success']) for ck in (C1, C2)}
print('layout failure counts (of 4 trials per layout per checkpoint); only layouts with any failure:')
print(f'{"layout":>15s} {"cell":>4s} {"arr":>9s} {"pick":>5s} {"nD":>2s}  fail@33400 fail@50000  classes')
for l in sorted(layouts, key=lambda l: -(per[l][C1] + per[l][C2])):
    if per[l][C1] + per[l][C2] == 0: continue
    rs = sel(layout=l)
    cls = collections.Counter(r['cls'].split()[0] for r in rs if not r['success'])
    print(f'{l:>15s} {rs[0]["cell"]:>4s} {rs[0]["arrangement"]:>9s} {rs[0]["pick"]:>5s} {rs[0]["n_distractors"]:2d}  {per[l][C1]:5d}/4    {per[l][C2]:5d}/4     {dict(cls)}')
n_fail_layouts = {ck: sum(1 for l in layouts if per[l][ck] > 0) for ck in (C1, C2)}
print('\nlayouts with >=1 failure:', n_fail_layouts, 'of', len(layouts))
for ck in (C1, C2):
    tot = sum(per[l][ck] for l in layouts)
    top = sorted((per[l][ck] for l in layouts), reverse=True)
    print(f'ckpt {ck}: {tot} failures; top-3 layouts hold {sum(top[:3])} ({100*sum(top[:3])/tot:.0f}%), top-5 hold {sum(top[:5])} ({100*sum(top[:5])/tot:.0f}%)')

# Permutation test of clustering: are failures more concentrated in few layouts than independent trials would produce?
for ck in (C1, C2):
    rows = sel(ckpt=ck)
    obs = sorted(((sum(not r['success'] for r in sel(rows, layout=l))) for l in layouts), reverse=True)
    obs_stat = sum(v * v for v in obs)
    labels = np.array([r['layout'] for r in rows]); fails = np.array([not r['success'] for r in rows])
    perm_stats = []
    for _ in range(5000):
        f = rng.permutation(fails)
        c = collections.Counter(labels[f])
        perm_stats.append(sum(v * v for v in c.values()))
    p = (np.sum(np.array(perm_stats) >= obs_stat) + 1) / 5001
    print(f'ckpt {ck}: failure concentration (sum of squared per-layout fail counts) = {obs_stat}, null mean {np.mean(perm_stats):.1f}, perm p = {p:.4f}')

# Repeatability: are the SAME layouts failing at both checkpoints?
a = np.array([per[l][C1] for l in layouts]); b = np.array([per[l][C2] for l in layouts])
print(f'\nper-layout failure count correlation between checkpoints: Pearson r = {np.corrcoef(a, b)[0, 1]:.3f}')
both = [l for l in layouts if per[l][C1] > 0 and per[l][C2] > 0]
print(f'layouts failing at BOTH checkpoints: {len(both)} -> {both}')
print(f'layouts failing only at 33400: {[l for l in layouts if per[l][C1] > 0 and per[l][C2] == 0]}')
print(f'layouts failing only at 50000: {[l for l in layouts if per[l][C1] == 0 and per[l][C2] > 0]}')
# seed/condition consistency within a layout at one checkpoint (is it deterministic per layout?)
print('\nwithin-layout outcome consistency (all 4 trials same outcome) per checkpoint:')
for ck in (C1, C2):
    cons = sum(1 for l in layouts if len({r['success'] for r in sel(layout=l, ckpt=ck)}) == 1)
    print(f'  ckpt {ck}: {cons}/{len(layouts)} layouts have identical outcome across control/distractor x 2 seeds')

print('\n' + '#' * 100); print('D. CELL / ARRANGEMENT / PICK EFFECTS'); print('#' * 100)
for key in ('cell', 'arrangement', 'pick', 'n_distractors', 'condition'):
    vals = sorted({r[key] for r in R})
    print(f'-- by {key}')
    for v in vals:
        row = []
        for ck in (C1, C2):
            rows = sel(ckpt=ck, **{key: v}); k = sum(r['success'] for r in rows)
            row.append(f'{k:3d}/{len(rows):3d} = {100*k/len(rows):5.1f}%')
        print(f'   {str(v):>10s}   33400: {row[0]}   50000: {row[1]}')
print('-- by cell x condition')
for cell in sorted({r['cell'] for r in R}):
    for cond in ('control', 'distractor'):
        row = []
        for ck in (C1, C2):
            rows = sel(ckpt=ck, cell=cell, condition=cond); k = sum(r['success'] for r in rows)
            row.append(f'{k:2d}/{len(rows):2d}')
        print(f'   {cell} {cond:10s} 33400: {row[0]}   50000: {row[1]}')
print('-- distractor colour (distractor condition only)')
for colours in (['blue'], ['yellow'], ['blue', 'yellow']):
    row = []
    for ck in (C1, C2):
        rows = [r for r in sel(ckpt=ck, condition='distractor') if sorted(r['distractors']) == sorted(colours)]
        k = sum(r['success'] for r in rows); row.append(f'{k:2d}/{len(rows):2d}')
    print(f'   {"+".join(colours):>12s} 33400: {row[0]}   50000: {row[1]}')

print('\n' + '#' * 100); print('E. GEOMETRY PREDICTORS (layout-level; failing-vs-passing layouts at 50000 and at 33400)'); print('#' * 100)
feat = ['pick_reach', 'place_reach', 'pick_place_dist', 'home_jitter_norm_deg']
for ck in (C1, C2):
    print(f'-- ckpt {ck}: mean feature, layouts with >=1 failure vs none')
    for f in feat:
        fl = [np.mean([r[f] for r in sel(layout=l, ckpt=ck)]) for l in layouts if per[l][ck] > 0]
        ok = [np.mean([r[f] for r in sel(layout=l, ckpt=ck)]) for l in layouts if per[l][ck] == 0]
        # Mann-Whitney U via permutation
        allv = np.array(fl + ok); labs = np.array([1] * len(fl) + [0] * len(ok))
        obs = np.mean(fl) - np.mean(ok)
        perm = [np.mean(allv[p == 1]) - np.mean(allv[p == 0]) for p in (rng.permutation(labs) for _ in range(5000))]
        pv = (np.sum(np.abs(perm) >= abs(obs)) + 1) / 5001
        print(f'   {f:22s} fail-layouts {np.mean(fl):.3f} (n={len(fl)})  ok-layouts {np.mean(ok):.3f} (n={len(ok)})  diff {obs:+.3f}  perm p={pv:.3f}')

print('-- distractor proximity (distractor-present trials): does a nearby distractor predict failure?')
for ck in (C1, C2):
    rows = sel(ckpt=ck, condition='distractor')
    for f in ('to_pick', 'to_place', 'to_path'):
        vals = [(min(x[f] for x in r['dist']), not r['success']) for r in rows]
        fv = [v for v, fl in vals if fl]; ov = [v for v, fl in vals if not fl]
        if fv: print(f'   ckpt {ck} min distractor {f:9s} failing: mean {np.mean(fv):.3f} (n={len(fv)})   passing: mean {np.mean(ov):.3f} (n={len(ov)})')
    disturbed = [r for r in rows if r['distractor_disturbed_2cm']]
    print(f'   ckpt {ck}: trials where a distractor moved >2 cm: {len(disturbed)}/{len(rows)}; failed among them: {sum(not r["success"] for r in disturbed)}')
    mx = max((x['disp_m'] for r in rows for x in r['dist']), default=0)
    print(f'   ckpt {ck}: max distractor displacement {100*mx:.1f} cm')

print('\n' + '#' * 100); print('F. TIME BUDGET (1200-tick = 40 s limit)'); print('#' * 100)
for ck in (C1, C2):
    s = [r['ticks'] for r in sel(ckpt=ck) if r['success']]
    print(f'ckpt {ck}: success ticks  min {min(s)}  median {int(np.median(s))}  p90 {int(np.percentile(s, 90))}  max {max(s)}')
    late = [r for r in sel(ckpt=ck) if r['success'] and r['ticks'] > 600]
    print(f'   successes needing >600 ticks (half the budget): {len(late)}; >900: {sum(1 for r in late if r["ticks"] > 900)}')
    fl = [r for r in sel(ckpt=ck) if not r['success']]
    print(f'   failures that ran to the 1200-tick limit: {sum(1 for r in fl if r["ticks"] == 1200)}/{len(fl)}')
    near = [r for r in fl if r['geom_ok_ticks'] > 0]
    print(f'   failures that had the stack geometry correct for >=1 tick: {len(near)} -> {[(r["ordinal"], r["geom_ok_ticks"], r["geom_ok_run"]) for r in near]}')
print('\nregrasp / dithering: gripper close events per trial (successes vs failures)')
for ck in (C1, C2):
    sg = [r['grasp_attempts'] for r in sel(ckpt=ck) if r['success']]
    fg = [r['grasp_attempts'] for r in sel(ckpt=ck) if not r['success']]
    print(f'   ckpt {ck}: success mean {np.mean(sg):.2f} max {max(sg)} | failure mean {np.mean(fg):.2f} max {max(fg)} | failures with >=5 grasp attempts: {sum(1 for g in fg if g >= 5)}/{len(fg)}')
