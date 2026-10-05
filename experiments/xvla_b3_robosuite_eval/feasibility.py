"""The scripted expert on every banked reserved scene, before any policy number is read.

A layout the expert cannot solve is reported and excluded; the policy is never blamed for it.
Robosuite image, CPU, one process. Writes under `--out` only (never under the dataset).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from data_collection.benchmark import capture_state
from experiments.b3_robosuite_data.expert import B3Expert
from experiments.b3_robosuite_data.plan import seed
from experiments.paired_demonstrations.plan import instruction
from experiments.xvla_b3_robosuite_eval import protocol
from experiments.xvla_b3_robosuite_eval.scenes import present_colours, restore

MUJOCO_VERSION = '3.4.0'


def run_one(key):
    """Expert attempt on one banked scene: {success, reason, final_xy_m, final_z_m, frames, seconds}."""
    layout_id = key.split(':')[0]
    layout = json.loads(protocol.PLAN_JSON.read_text())['evaluation_holdout'][layout_id]
    model, data = restore(key)
    colours = tuple(present_colours(model))
    episode = dict(pick=layout['pick'], place=layout['place'],
                   instruction=instruction(layout['pick'], layout['place']),
                   seed=seed(f'evaluation:feasibility:{key}'),
                   home_pose_jitter_deg=layout['home_pose_jitter_deg'],
                   hover_z_offset=layout['hover_z_offset'])
    started = time.monotonic()
    outcome = B3Expert(colours).run(capture_state(model, data), episode, lambda frame, trace: None)
    metrics = outcome['final_metrics']
    return dict(success=bool(outcome['success']), reason=outcome['reason'],
                final_xy_m=metrics.get('xy_m'), final_z_m=metrics.get('z_m'),
                frames=outcome['frames'], seconds=round(time.monotonic()-started, 1))


def summarize(results):
    failed = sorted(k for k, v in results.items() if not v['success'])
    return dict(results=results, failed_keys=failed, failed_layouts=sorted({k.split(':')[0] for k in failed}),
                keys=len(results), succeeded=len(results)-len(failed))


def main(argv=None):
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if mujoco.__version__ != MUJOCO_VERSION:
        raise SystemExit(f'MuJoCo {mujoco.__version__} is not {MUJOCO_VERSION}; run in the robosuite image')
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'No overwrite: {out} exists')
    out.mkdir(parents=True)
    manifest = json.loads(protocol.BANK_JSON.read_text())
    results = {}
    for key in manifest:
        results[key] = run_one(key)
        (out/'feasibility.partial.json').write_text(json.dumps(summarize(results), indent=2)+'\n')
        print('FEAS', key, results[key]['success'], results[key]['reason'],
              None if results[key]['final_xy_m'] is None else round(results[key]['final_xy_m']*1000, 2),
              results[key]['seconds'], flush=True)
    summary = summarize(results)
    (out/'feasibility.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('FEAS_DONE', summary['succeeded'], 'of', summary['keys'], 'failed layouts', summary['failed_layouts'],
          flush=True)


if __name__ == '__main__':
    main()
