"""Per-trial measures, Gates 0 and 1, and the diagnostics CLI. Robosuite image (MuJoCo 3.4.0).

    run reference                              expert envelope -> outputs/xvla_props_p2_eval/diagnostics/reference.json
    run measure --shard K --shards N           replay every trial of RUNS; one measures file per trial
    run analyze                                Gate 0, Gate 1, then the tables of the spec

Definitions are the spec's (docs/superpowers/specs/2026-10-19-props-p2-diagnostics-design.md). The finger midpoint is the
mean of the `left_finger` and `right_finger` body positions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from experiments.xvla_props_p2_eval import analysis, protocol
from experiments.xvla_props_p2_eval.diagnostics import grasp, taxonomy

OUT = protocol.EVAL_ROOT/'diagnostics'
EPISODES = protocol.PARENT/'episodes'
RUNS = (('screen', 'p2', 101520), ('report', 'p2', 101520), ('control', 'p2', 101520), ('confirm', 'p2', 101520),
        ('report', 'p2', 126900), ('report', 'p1', 126900))
GATE1_MIN = 0.80


def run_name(mode, policy, candidate):
    return f'{mode}_{policy}_step_{candidate:06d}'


def gate0_ok(measures):
    return bool(measures) and all(m['gate0'] for m in measures)


def gate1_fraction(rows):
    """Fraction of successful trials whose grasp closure lies inside the target's expert envelope. A success without a
    grasp closure counts as outside (a success always closes the gripper first, so that would be an anomaly)."""
    wins = [r for r in rows if r['task_success']]
    return sum(1 for r in wins if r['grasp_inside']) / len(wins) if wins else float('nan')


def gate1_passes(fraction):
    return fraction >= GATE1_MIN


def _env(ref, name):
    if name not in ref:
        raise ValueError(f'No expert envelope for {name!r}')
    return dict(lo=np.asarray(ref[name]['lo'], dtype=float), hi=np.asarray(ref[name]['hi'], dtype=float))


def measure_trial(folder, ref):
    """Replay one recorded trial and reduce it to the compact measures of the spec."""
    from experiments.xvla_props_p2_eval.diagnostics import replay
    folder = Path(folder)
    row = json.loads((folder/'episode.json').read_text())
    rep = replay.replay_trial(folder)
    names, command = rep['names'], rep['command']
    z = rep['props_xyz'][:, :, 2]
    lift = analysis.lift_tick(z[:, 0])
    closure, gclosure = grasp.first_closure(command), grasp.grasp_closure(command, lift)

    def sample(tick, i):
        pos = grasp.to_hand_frame(rep['hand_pos'][tick], rep['hand_rot'][tick], rep['props_xyz'][tick, i])
        axis = rep['right_pos'][tick] - rep['left_pos'][tick]
        return np.array([*pos, grasp.yaw_error_deg(axis[:2], rep['props_yaw_deg'][tick, i])])

    closed = closure is not None
    m = dict(gate0=bool(rep['gate0']), first_mismatch=rep['first_mismatch'], lifted=lift is not None, lift_tick=lift,
             closed=closed, first_closure=closure, grasp_closure=gclosure, names=list(names),
             first_lifted=analysis.first_lifted(z, names))
    target_env = _env(ref, names[0])
    m['grasp_inside'] = bool(grasp.inside(target_env, sample(gclosure, 0)).all()) if gclosure is not None else None
    m['target_in_env'] = bool(grasp.inside(target_env, sample(closure, 0)).all()) if closed else False
    m['distractor_in_env'] = [bool(grasp.inside(_env(ref, names[i]), sample(closure, i)).all()) for i in range(1, len(names))] if closed else []
    m['held'] = bool(closed and grasp.held_ticks(rep['left_touch'][:, 0], rep['right_touch'][:, 0], closure))
    mid = (rep['left_pos'] + rep['right_pos'])/2
    m['closest_approach_m'] = float(np.linalg.norm(mid - rep['props_xyz'][:, 0], axis=1).min())
    if closed:
        at = sample(closure, 0)
        m['target_in_hand_at_closure'], m['yaw_error_deg'] = at[:3].tolist(), float(at[3])
    m['touched_distractors'] = [names[i] for i in range(1, len(names)) if rep['touched'][:, i].any()]
    m['class'] = taxonomy.classify(row, m)
    return m


def measures_path(mode, policy, candidate, trial_name):
    return OUT/'measures'/run_name(mode, policy, candidate)/f'{trial_name}.json'


def load_reference():
    return json.loads((OUT/'reference.json').read_text())


def do_reference():
    from experiments.xvla_props_p2_eval.diagnostics import reference
    ref = reference.write_reference(EPISODES, OUT/'reference.json')
    for name, v in sorted(ref.items()):
        print('REFERENCE', name, v['n'], [round(x, 3) for x in v['lo']], [round(x, 3) for x in v['hi']], flush=True)


def do_measure(shard, shards):
    ref = load_reference()
    jobs = []
    for mode, policy, candidate in RUNS:
        directory = protocol.output_dir(mode, policy, candidate)
        for row in json.loads((directory/'result.json').read_text())['episodes']:
            jobs.append((mode, policy, candidate, protocol.trial_folder(directory, row)))
    mine = jobs[shard::shards]
    for mode, policy, candidate, folder in mine:
        target = measures_path(mode, policy, candidate, folder.name)
        if target.exists():
            continue
        m = measure_trial(folder, ref)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(m, indent=2) + '\n')
        print('MEASURE', run_name(mode, policy, candidate), folder.name, m['class'], m['gate0'], flush=True)
    print('MEASURE_SHARD_DONE', shard, shards, len(mine), flush=True)


def load_measures():
    """{run name: [row merged with its measures]}; the evaluation row supplies cause, kind, layout and so on."""
    out = {}
    for mode, policy, candidate in RUNS:
        name = run_name(mode, policy, candidate)
        directory = protocol.output_dir(mode, policy, candidate)
        rows = []
        for row in json.loads((directory/'result.json').read_text())['episodes']:
            path = measures_path(mode, policy, candidate, protocol.trial_folder(directory, row).name)
            rows.append({**row, **json.loads(path.read_text())})
        out[name] = rows
    return out


def do_analyze():
    from experiments.xvla_props_p2_eval.diagnostics import tables
    data = load_measures()
    every = [r for rows in data.values() for r in rows]
    print(f'GATE0 {sum(r["gate0"] for r in every)}/{len(every)} trials reproduced exactly', flush=True)
    if not gate0_ok(every):
        print('GATE0 FAILED: no analysis on a replay that does not reproduce its own data', flush=True)
        return 1
    fraction = gate1_fraction(data[run_name('report', 'p2', 101520)] + data[run_name('screen', 'p2', 101520)]
                              + data[run_name('control', 'p2', 101520)] + data[run_name('confirm', 'p2', 101520)])
    print(f'GATE1 {fraction:.3f} of 101,520 successes have the grasp closure inside the expert envelope '
          f'(minimum {GATE1_MIN})', flush=True)
    if not gate1_passes(fraction):
        print('GATE1 FAILED: the envelope does not describe this policy\'s grasps; stop and ask the user', flush=True)
        return 2
    result = tables.build(data)
    (OUT/'analysis.json').write_text(json.dumps(result, indent=2, default=float) + '\n')
    print(tables.render(result), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('reference')
    m = sub.add_parser('measure')
    m.add_argument('--shard', type=int, default=0)
    m.add_argument('--shards', type=int, default=1)
    sub.add_parser('analyze')
    args = parser.parse_args(argv)
    if args.command == 'reference':
        do_reference()
    elif args.command == 'measure':
        do_measure(args.shard, args.shards)
    else:
        return do_analyze()
    return 0


if __name__ == '__main__':
    sys.exit(main())
