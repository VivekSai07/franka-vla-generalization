"""The scripted expert on every reserved props P1 layout, before any policy number is read.

A layout the expert cannot solve is reported and excluded; the policy is never blamed for it.
Robosuite image, CPU, one process, no rendering (feasibility only needs the physics outcome).
Writes under `--out` only (never under the dataset).
"""
import argparse
import json
import time
from pathlib import Path

from experiments.props_scene.expert import PropsExpert
from experiments.props_scene.scene import reset
from experiments.xvla_props_p1_eval import protocol

MUJOCO_VERSION = '3.4.0'
_EXPERTS = {}


def _expert(obj, blue_slot):
    key = (obj, blue_slot)
    if key not in _EXPERTS:
        _EXPERTS[key] = PropsExpert((obj,), blue_slot)
    return _EXPERTS[key]


def run_one(layout):
    """Expert attempt on one reserved layout: {success, reason, frames, stable_ticks, seconds}."""
    obj, bin_colour, blue_slot = layout['object'], layout['bin'], layout['blue_slot']
    x, y, yaw = layout['start']
    expert = _expert(obj, blue_slot)
    scene = reset(expert.model, {obj: (x, y, yaw)}, layout['home_pose_jitter_deg'])
    episode = {'object': obj, 'bin': bin_colour, 'seed': layout['seed'],
               'home_pose_jitter_deg': list(layout['home_pose_jitter_deg'])}
    started = time.monotonic()
    outcome = expert.run(scene['initial_state'], episode, lambda frame, trace: None, render=False)
    return dict(success=bool(outcome['success']), reason=outcome['reason'], frames=outcome['frames'],
                stable_ticks=outcome['stable_ticks'], seconds=round(time.monotonic()-started, 1))


def summarize(results):
    failed = sorted(k for k, v in results.items() if not v['success'])
    return dict(results=results, failed_keys=failed, failed_layouts=failed,  # one trial per layout: same set
                keys=len(results), succeeded=len(results)-len(failed))


def main(argv=None):
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if mujoco.__version__ != MUJOCO_VERSION:
        raise SystemExit(f'MuJoCo {mujoco.__version__} is not {MUJOCO_VERSION}; run in the robosuite image')
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'No overwrite: {out} exists')
    out.mkdir(parents=True)
    plan = protocol._plan()
    holdout = plan['evaluation_holdout']
    results = {}
    for key in sorted(holdout):
        results[key] = run_one(holdout[key])
        (out/'feasibility.partial.json').write_text(json.dumps(summarize(results), indent=2)+'\n')
        print('FEAS', key, results[key]['success'], results[key]['reason'], results[key]['seconds'], flush=True)
    summary = summarize(results)
    (out/'feasibility.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('FEAS_DONE', summary['succeeded'], 'of', summary['keys'], 'failed layouts', summary['failed_layouts'],
          flush=True)


if __name__ == '__main__':
    main()
