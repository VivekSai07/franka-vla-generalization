"""Hypothesis H_unreachable: can the SCRIPTED EXPERT solve the layouts the policy failed on?

Runs the collector's own expert (same hover jitter, same home-pose jitter, same derived reset
arrays) on reserved evaluation layouts. Writes results incrementally to /out/expert_check.json.
"""
import json, sys, time
import numpy as np
from experiments.b2_distractor_data import plan as bp
from experiments.b2_distractor_data.expert import Expert
from experiments.xvla_b2_distractors_eval import evaluate as ev

RUN = '/vla/outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928'
plan = bp.make_plan()
hold = plan['evaluation_holdout']
picks = ev.eval_picks()
R = json.load(open('/out/trials_classified.json'))

fail_layouts = sorted({r['layout'] for r in R if not r['success']})
persistent = ['evaluation_054', 'evaluation_048', 'evaluation_059', 'evaluation_040']
passing = sorted({r['layout'] for r in R} - set(fail_layouts))[:6]
jobs = [(l, 'control') for l in fail_layouts] + [(l, 'distractor') for l in persistent] + [(l, 'control') for l in passing]
# layouts that exist at 33400 only vs both, for labelling
print('jobs:', len(jobs), flush=True)

expert = Expert()
results = []
for layout, condition in jobs:
    t0 = time.time()
    scene = np.load(f'{RUN}/evaluation_report_step_033400/{layout}_{condition}.npz')
    h = hold[layout]
    pick = picks[layout]; place = 'green' if pick == 'red' else 'red'
    episode = dict(instruction=bp.instruction_for(pick, place), pick=pick, place=place,
                   home_pose_jitter_deg=[float(x) for x in h['home_pose_jitter_deg']],
                   hover_z_offset={k: float(v) for k, v in h['hover_z_offset'].items()},
                   seed=int(bp.seed(f'expertcheck:{layout}')))
    try:
        out = expert.run(scene['initial_state'], episode, lambda frame, trace: None)
        rec = dict(layout=layout, condition=condition, pick=pick, arrangement=h['arrangement'],
                   success=bool(out['success']), reason=out.get('reason'),
                   final=({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in out.get('metrics', {}).items()}
                          if isinstance(out.get('metrics'), dict) else None))
    except Exception as exc:  # an exception is itself evidence, record it
        rec = dict(layout=layout, condition=condition, pick=pick, arrangement=h['arrangement'],
                   success=False, reason=f'EXCEPTION {type(exc).__name__}: {exc}'[:300])
    rec['seconds'] = round(time.time() - t0, 1)
    results.append(rec)
    json.dump(results, open('/out/expert_check.json', 'w'), indent=1)
    print(json.dumps(rec), flush=True)
print('DONE', flush=True)
