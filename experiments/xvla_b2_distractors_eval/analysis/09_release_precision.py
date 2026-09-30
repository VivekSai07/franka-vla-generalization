"""Stage 8: release-offset vs outcome for ALL trials, grasp quality at first closure, frozen-policy fraction, collateral motion."""
import glob, json, collections
import numpy as np

RUN = '/vla/outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
R = json.load(open('/out/trials_behaviour.json'))
C1, C2 = 33400, 50000
rng = np.random.RandomState(5)


def trace(r):
    root = f"{RUN}/evaluation_report_step_{r['ckpt']:06d}"
    ep = glob.glob(f"{root}/trial_{r['ordinal']:03d}_*/episode.json")[0]
    return np.load(ep.replace('episode.json', 'trace.npz'))


rows = []
for r in R:
    t = trace(r)
    act = t['applied_action'][:, 7]; closed = act <= 127
    lift = t['pick_lift_m']; xy = t['xy_m']; z = t['z_m']; fm = t['finger_midpoint_to_pick_m']
    opens = np.flatnonzero(closed[:-1] & ~closed[1:]) + 1
    carried = [i for i in opens if lift[max(0, i - 5):i + 1].max() > 0.04]
    fc = r['stages'].get('first_closure_tick')
    pl = t['place_position']
    rows.append(dict(r=r, rel_tick=(carried[-1] if carried else None),
                     rel_xy=(float(xy[carried[-1]]) if carried else None),
                     rel_z=(float(z[carried[-1]]) if carried else None),
                     close_dist=(float(fm[min(fc, len(fm) - 1)]) if fc is not None else None),
                     base_moved=float(np.linalg.norm(pl[:, :2] - pl[0, :2], axis=1).max()),
                     base_tilt_z=float(pl[:, 2].max() - pl[0, 2])))

print('=' * 100); print('M. OUTCOME vs LATERAL OFFSET AT THE (LAST) RELEASE-AFTER-CARRY, ALL 320 TRIALS'); print('=' * 100)
have = [x for x in rows if x['rel_xy'] is not None]
print(f'trials with a carry-then-release event: {len(have)}/{len(rows)}  (successes {sum(x["r"]["success"] for x in have)}, failures {sum(not x["r"]["success"] for x in have)})')
bins = [(0, .010), (.010, .015), (.015, .020), (.020, .025), (.025, .030), (.030, 1.0)]
print(f'{"xy at release":>16s} {"n":>4s} {"success":>8s} {"rate":>7s}    33400: k/n   50000: k/n')
for lo, hi in bins:
    sub = [x for x in have if lo <= x['rel_xy'] < hi]
    if not sub: continue
    k = sum(x['r']['success'] for x in sub)
    s1 = [x for x in sub if x['r']['ckpt'] == C1]; s2 = [x for x in sub if x['r']['ckpt'] == C2]
    print(f'{100*lo:5.1f}-{100*hi:5.1f} cm {len(sub):4d} {k:8d} {100*k/len(sub):6.1f}%    {sum(x["r"]["success"] for x in s1):3d}/{len(s1):3d}   {sum(x["r"]["success"] for x in s2):3d}/{len(s2):3d}')
print('\nrelease offset distribution, successes vs failures (cm): ')
for ck in (C1, C2):
    s = np.array([x['rel_xy'] for x in have if x['r']['ckpt'] == ck and x['r']['success']]) * 100
    f = np.array([x['rel_xy'] for x in have if x['r']['ckpt'] == ck and not x['r']['success']]) * 100
    print(f'  ckpt {ck}: success median {np.median(s):.2f} p90 {np.percentile(s,90):.2f} max {s.max():.2f} (n={len(s)}) | failure median {np.median(f):.2f} p10 {np.percentile(f,10):.2f} (n={len(f)})')
thr = [(t_, np.mean([x['r']['success'] for x in have if x['rel_xy'] < t_ / 100]), sum(1 for x in have if x['rel_xy'] < t_ / 100)) for t_ in (1.5, 2.0, 2.5)]
print('  P(success | release offset < T):', [(f'{t_}cm', f'{100*p:.1f}%', f'n={n}') for t_, p, n in thr])
print(f'  => cube edge is 4.2cm, so its centre of mass leaves the support at 2.1 cm lateral offset')
print('  per-checkpoint release-offset for ALL carried trials (mean, sd):')
for ck in (C1, C2):
    a = np.array([x['rel_xy'] for x in have if x['r']['ckpt'] == ck]) * 100
    print(f'    ckpt {ck}: mean {a.mean():.2f} cm sd {a.std():.2f}   share with offset >2.1cm: {100*np.mean(a>2.1):.1f}%')

print('\n' + '=' * 100); print('N. SLID OFF AFTER A GOOD RELEASE? (release xy < 2.0cm but did not succeed)'); print('=' * 100)
for x in have:
    r = x['r']
    if (not r['success']) and x['rel_xy'] < 0.02:
        t = trace(r)
        i = x['rel_tick']; xy = t['xy_m']; z = t['z_m']; fm = t['finger_midpoint_to_pick_m']
        print(f'  ck {r["ckpt"]} ord {r["ordinal"]} {r["layout"]} {r["cell"]}: release@tick {i} xy={100*xy[i]:.1f}cm z={100*z[i]:.1f}cm -> xy after 15/30/60 ticks: '
              f'{100*xy[min(i+15,len(xy)-1)]:.1f}/{100*xy[min(i+30,len(xy)-1)]:.1f}/{100*xy[min(i+60,len(xy)-1)]:.1f} cm, z {100*z[min(i+30,len(z)-1)]:.1f}cm; finger-to-cube at release {100*fm[i]:.1f}cm')

print('\n' + '=' * 100); print('O. GRASP QUALITY: fingertip-midpoint to cube at FIRST gripper closure'); print('=' * 100)
for ck in (C1, C2):
    s = np.array([x['close_dist'] for x in rows if x['r']['ckpt'] == ck and x['r']['success'] and x['close_dist'] is not None]) * 100
    f = [x for x in rows if x['r']['ckpt'] == ck and not x['r']['success'] and x['close_dist'] is not None]
    fa = np.array([x['close_dist'] for x in f]) * 100
    print(f'ckpt {ck}: successes  mean {s.mean():.2f} cm  p95 {np.percentile(s,95):.2f}  max {s.max():.2f} (n={len(s)})')
    print(f'          failures   mean {fa.mean():.2f} cm  (n={len(fa)});  closed >3.5cm from cube (in the air): {[ (x["r"]["ordinal"], round(100*x["close_dist"],1)) for x in f if x["close_dist"] > 0.035]}')
print('  (all trials, ANY checkpoint) share of trials that closed >3.5cm from the cube:', f'{sum(1 for x in rows if x["close_dist"] is not None and x["close_dist"]>0.035)}/{len(rows)}')

print('\n' + '=' * 100); print('P. FROZEN POLICY AT TIMEOUT (mean joint change per tick over last 150 ticks, rad)'); print('=' * 100)
for ck in (C1, C2):
    f = [x['r'] for x in rows if x['r']['ckpt'] == ck and not x['r']['success']]
    v = np.array([r['end_speed'] for r in f])
    print(f'ckpt {ck}: failures {len(f)}; end speed median {np.median(v):.4f}; frozen (<0.006 rad/tick, ~<0.2 rad/s): {(v<0.006).sum()}  active (>=0.006): {(v>=0.006).sum()}')
    by = collections.Counter((r['behaviour'].split()[0], 'frozen' if r['end_speed'] < 0.006 else 'active') for r in f)
    print('   by behaviour class:', dict(sorted(by.items())))

print('\n' + '=' * 100); print('Q. COLLATERAL: did the policy disturb the BASE (target) cube it was supposed to stack onto?'); print('=' * 100)
for ck in (C1, C2):
    sub = [x for x in rows if x['r']['ckpt'] == ck]
    for lab, sel in (('success', True), ('failure', False)):
        s = [x for x in sub if x['r']['success'] == sel]
        print(f'ckpt {ck} {lab:8s}: base cube moved >1cm in {sum(1 for x in s if x["base_moved"]>0.01):3d}/{len(s):3d}   >2cm in {sum(1 for x in s if x["base_moved"]>0.02):3d}/{len(s):3d}   >10cm in {sum(1 for x in s if x["base_moved"]>0.10):3d}/{len(s):3d}')
