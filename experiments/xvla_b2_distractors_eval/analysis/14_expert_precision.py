import json
import numpy as np
P = '/vla/outputs/b2_distractor_data/b2_distractor_20260928'
v = json.load(open(f'{P}/validation.json'))
train = set(v['train_ids'])
fx, rel = [], []
for e in v['episodes']:
    i = e['episode'] if isinstance(e['episode'], int) else e['episode'].get('episode_index')
    d = e['diagnostics']
    fx.append(d['final_xy_m'])
    r = d['releases']
    if r: rel.append(np.hypot(r[-1]['delta_xyz'][0], r[-1]['delta_xyz'][1]))
fx = np.array(fx) * 100; rel = np.array(rel) * 100
print('EXPERT demonstrations (n=%d):' % len(fx))
print('  final xy error after settle (cm): mean %.2f  median %.2f  p90 %.2f  p99 %.2f  max %.2f' % (fx.mean(), np.median(fx), np.percentile(fx, 90), np.percentile(fx, 99), fx.max()))
print('  xy error AT RELEASE (cm):          mean %.2f  median %.2f  p90 %.2f  p99 %.2f  max %.2f' % (rel.mean(), np.median(rel), np.percentile(rel, 90), np.percentile(rel, 99), rel.max()))
print('  share of demos with release offset > 1.0 cm: %.1f%%, > 1.5 cm: %.1f%%, > 2.0 cm: %.1f%%' % tuple(100 * np.mean(rel > t) for t in (1.0, 1.5, 2.0)))
R = json.load(open('/out/trials_final.json'))
for ck in (33400, 50000):
    a = np.array([r['rel_xy'] for r in R if r['ckpt'] == ck and r['rel_xy'] is not None]) * 100
    print('POLICY ckpt %d release offset (cm, n=%d): mean %.2f median %.2f p90 %.2f p99 %.2f max %.2f | share >1.0: %.1f%%  >1.5: %.1f%%  >2.0: %.1f%%' % (
        ck, len(a), a.mean(), np.median(a), np.percentile(a, 90), np.percentile(a, 99), a.max(), *[100 * np.mean(a > t) for t in (1.0, 1.5, 2.0)]))
print('ratio policy/expert mean release offset:', {ck: round(float(np.mean([r['rel_xy'] for r in R if r['ckpt'] == ck and r['rel_xy'] is not None]) * 100 / rel.mean()), 2) for ck in (33400, 50000)})
