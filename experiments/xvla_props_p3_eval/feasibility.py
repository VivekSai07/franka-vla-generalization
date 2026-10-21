"""The scripted P3 expert on every reserved layout, before any policy number is read.

A layout the expert cannot solve is reported and excluded; the policy is never blamed for it (spec: dropped, not
replaced, and counted). Robosuite image, CPU, no rendering (feasibility needs only the physics outcome). Shardable:
`--shard k --shards n` writes `feasibility.shard_k.json` under `--out`; `--merge` combines them into
`feasibility.json`. Writes under `--out` only (never under the dataset).
"""
import argparse
import json
import time
from pathlib import Path

from experiments.xvla_props_p3_eval import protocol

MUJOCO_VERSION = '3.4.0'
KEYS = protocol.LAYOUTS


def run_one(layout):
    """Expert attempt on one reserved layout, built and run exactly as the collector did."""
    from experiments.props_p3_data import episode
    draw = {'seed': layout['seed'], 'home_pose_jitter_deg': list(layout['home_pose_jitter_deg'])}
    started = time.monotonic()
    outcome = episode.run_attempt(layout, draw, lambda frame, trace: None, render=False)
    return dict(success=bool(outcome['success']), reason=outcome['reason'], frames=outcome['frames'],
                stable_ticks=outcome['stable_ticks'], seconds=round(time.monotonic() - started, 1))


def summarize(results, layouts=None):
    """`layouts` maps key -> layout so the failures can be counted per object (the held-out ones matter most)."""
    failed = sorted(k for k, v in results.items() if not v['success'])
    summary = dict(results=results, failed_keys=failed, failed_layouts=failed,  # one expert attempt per layout
                   keys=len(results), succeeded=len(results) - len(failed))
    if layouts is not None:
        summary['failed_by_object'] = {o: sum(1 for k in failed if layouts[k]['object'] == o)
                                       for o in sorted({layouts[k]['object'] for k in results})}
    return summary


def keys_of_shard(shard, shards, keys):
    if type(shards) is not int or shards < 1 or type(shard) is not int or not 0 <= shard < shards:
        raise ValueError(f'shard {shard} of {shards} is not valid')
    return sorted(keys)[shard::shards]


def main(argv=None):
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--out', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args(argv)
    out = Path(args.out)
    holdout = protocol._plan()['evaluation_holdout']
    if args.merge:
        results = {}
        for path in sorted(out.glob('feasibility.shard_*.json')):
            results.update(json.loads(path.read_text())['results'])
        if set(results) != set(holdout) or len(holdout) != KEYS:
            raise SystemExit(f'Shards cover {len(results)} of {len(holdout)} layouts; every shard must finish first')
        summary = summarize(results, holdout)
        target = out/'feasibility.json'
        if target.exists():
            raise SystemExit(f'No overwrite: {target} exists')
        target.write_text(json.dumps(summary, indent=2) + '\n')
        print('FEAS_DONE', summary['succeeded'], 'of', summary['keys'], 'failed layouts', summary['failed_layouts'],
              summary['failed_by_object'], flush=True)
        return
    if mujoco.__version__ != MUJOCO_VERSION:
        raise SystemExit(f'MuJoCo {mujoco.__version__} is not {MUJOCO_VERSION}; run in the robosuite image')
    out.mkdir(parents=True, exist_ok=True)
    target = out/f'feasibility.shard_{args.shard}.json'
    if target.exists():
        raise SystemExit(f'No overwrite: {target} exists')
    results = {}
    for key in keys_of_shard(args.shard, args.shards, holdout):
        results[key] = run_one(holdout[key])
        (out/f'feasibility.shard_{args.shard}.partial.json').write_text(json.dumps(summarize(results)) + '\n')
        print('FEAS', key, results[key]['success'], results[key]['reason'], results[key]['seconds'], flush=True)
    target.write_text(json.dumps(summarize(results, holdout), indent=2) + '\n')
    print('FEAS_SHARD_DONE', args.shard, args.shards, flush=True)


if __name__ == '__main__':
    main()
