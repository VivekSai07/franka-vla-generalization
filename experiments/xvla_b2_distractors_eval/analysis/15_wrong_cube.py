"""Wrong-cube check: was the INSTRUCTED BASE cube ever lifted (i.e. the policy picked the wrong cube)?"""
import glob, json
import numpy as np
RUN = '/vla/outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
R = json.load(open('/out/trials_final.json'))
out = []
for r in R:
    root = f"{RUN}/evaluation_report_step_{r['ckpt']:06d}"
    ep = glob.glob(f"{root}/trial_{r['ordinal']:03d}_*/episode.json")[0]
    t = np.load(ep.replace('episode.json', 'trace.npz'))
    pl, pk = t['place_position'], t['pick_position']
    base_lift = float(pl[:, 2].max() - pl[0, 2]); pick_lift = float(pk[:, 2].max() - pk[0, 2])
    base_xy = float(np.linalg.norm(pl[:, :2] - pl[0, :2], axis=1).max())
    tick = int(np.argmax(pl[:, 2] - pl[0, 2] > 0.05)) if base_lift > 0.05 else None
    out.append(dict(ckpt=r['ckpt'], ordinal=r['ordinal'], layout=r['layout'], cond=r['condition'], success=r['success'], mech=r['mech'][:3].strip(),
                    base_lift=base_lift, pick_lift=pick_lift, base_xy=base_xy, first_base_lift_tick=tick))
wrong = [o for o in out if o['base_lift'] > 0.05]
print('trials where the instructed BASE cube was lifted >5 cm:', len(wrong), 'of', len(out))
print('  of which successes:', sum(o['success'] for o in wrong))
for o in sorted(wrong, key=lambda o: (o['ckpt'], o['ordinal'])):
    print(f"  ck {o['ckpt']} ord {o['ordinal']:3d} {o['layout'][-3:]} {o['cond'][:4]} success={o['success']!s:5s} mech={o['mech']:4s} base_lift={100*o['base_lift']:.1f}cm base_xy_moved={100*o['base_xy']:.1f}cm pick_lift={100*o['pick_lift']:.1f}cm first_base_lift_tick={o['first_base_lift_tick']}")
moved = [o for o in out if o['base_xy'] > 0.05 and o['base_lift'] <= 0.05]
print('\ntrials where the base cube was pushed/dragged >5 cm without being lifted:', len(moved))
for o in sorted(moved, key=lambda o: (o['ckpt'], o['ordinal'])):
    print(f"  ck {o['ckpt']} ord {o['ordinal']:3d} {o['layout'][-3:]} success={o['success']!s:5s} mech={o['mech']:4s} base_xy_moved={100*o['base_xy']:.1f}cm")
print('\nbase cube moved >2cm in SUCCESSES:', sum(1 for o in out if o['success'] and o['base_xy'] > 0.02), ' in FAILURES:', sum(1 for o in out if not o['success'] and o['base_xy'] > 0.02))
