"""Per-trial measures, Gates 0 and 1, and the diagnostics CLI. Robosuite image (MuJoCo 3.4.0).

    run reference                              expert envelope -> outputs/xvla_props_p2_eval/diagnostics/reference.json
    run measure --shard K --shards N           replay every trial of RUNS; one raw measures file per trial
    run analyze                                Gate 0, Gate 1 (success envelopes, both directions), then the tables

Measures hold raw hand-frame samples; judging against an envelope (`judge`) happens at analysis time, so no replay is
repeated when an envelope changes. Envelope definition: the 2026-10-19 amendment of the spec.

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
        ('screen', 'p2', 126900), ('report', 'p2', 126900), ('report', 'p1', 126900))
MIN_SUCCESSES = 10   # per object, to define a success envelope
GATE1_MIN = 0.80


def run_name(mode, policy, candidate):
    return f'{mode}_{policy}_step_{candidate:06d}'


def gate0_ok(measures):
    return bool(measures) and all(m['gate0'] for m in measures)


def success_envelope(rows, minimum=MIN_SUCCESSES):
    """{object: {lo, hi, n}}: p1 to p99 per component of the grasp-closure sample over the successful trials in `rows`
    (merged evaluation rows and measures). An object with fewer than `minimum` successes stops the analysis."""
    samples = {}
    for r in rows:
        if r['task_success'] and r.get('grasp_sample') is not None:
            samples.setdefault(r['object'], []).append(r['grasp_sample'])
    out = {}
    for name in sorted({r['object'] for r in rows}):
        got = samples.get(name, [])
        if len(got) < minimum:
            raise ValueError(f'{name}: {len(got)} successes, fewer than the {minimum} needed for a success envelope')
        env = grasp.envelope(np.array(got))
        out[name] = dict(lo=env['lo'].tolist(), hi=env['hi'].tolist(), n=env['n'])
    return out


def gate1_fraction(rows, envelope):
    """Fraction of the successful trials in `rows` whose grasp closure lies inside `envelope` for their object (a
    success without a grasp sample counts as outside)."""
    wins = [r for r in rows if r['task_success']]
    if not wins:
        return float('nan')
    inside = 0
    for r in wins:
        if r.get('grasp_sample') is not None and grasp.inside(_env(envelope, r['object']), r['grasp_sample']).all():
            inside += 1
    return inside / len(wins)


def gate1_passes(forward, backward):
    """Both directions reach the minimum: 101,520 inside the 126,900-success envelope and the reverse."""
    return forward >= GATE1_MIN and backward >= GATE1_MIN


def _env(ref, name):
    if name not in ref:
        raise ValueError(f'No envelope for {name!r}')
    return dict(lo=np.asarray(ref[name]['lo'], dtype=float), hi=np.asarray(ref[name]['hi'], dtype=float))


def measures_from_replay(rep):
    """Reduce one replay to raw measures: ticks, hand-frame samples and held/contact facts. Pure; no envelope involved.
    A sample is [x, y, z, yaw error] of a prop in the hand frame at a tick (the pre-action state of that tick)."""
    names, command = rep['names'], rep['command']
    z = rep['props_xyz'][:, :, 2]
    lift = analysis.lift_tick(z[:, 0])
    closure, gclosure = grasp.first_closure(command), grasp.grasp_closure(command, lift)

    def sample(tick, i):
        pos = grasp.to_hand_frame(rep['hand_pos'][tick], rep['hand_rot'][tick], rep['props_xyz'][tick, i])
        axis = rep['right_pos'][tick] - rep['left_pos'][tick]
        return [*map(float, pos), grasp.yaw_error_deg(axis[:2], rep['props_yaw_deg'][tick, i])]

    closed = closure is not None
    mid = (rep['left_pos'] + rep['right_pos'])/2
    return dict(gate0=bool(rep['gate0']), first_mismatch=rep['first_mismatch'], names=list(names), lifted=lift is not None,
                lift_tick=lift, closed=closed, first_closure=closure, grasp_closure=gclosure,
                grasp_sample=sample(gclosure, 0) if gclosure is not None else None,
                closure_samples=[sample(closure, i) for i in range(len(names))] if closed else None,
                held=bool(closed and grasp.held_ticks(rep['left_touch'][:, 0], rep['right_touch'][:, 0], closure)),
                closest_approach_m=float(np.linalg.norm(mid - rep['props_xyz'][:, 0], axis=1).min()),
                first_lifted=analysis.first_lifted(z, names),
                touched_distractors=[names[i] for i in range(1, len(names)) if rep['touched'][:, i].any()])


def judge(row, m, ref):
    """Apply an envelope per object to raw measures and classify the trial (the spec's taxonomy). Pure."""
    names = m['names']
    envs = [_env(ref, name) for name in names]
    out = dict(m)
    closed = m['closed']
    out['grasp_inside'] = (bool(grasp.inside(envs[0], m['grasp_sample']).all()) if m['grasp_sample'] is not None else None)
    out['target_in_env'] = bool(grasp.inside(envs[0], m['closure_samples'][0]).all()) if closed else False
    out['distractor_in_env'] = ([bool(grasp.inside(envs[i], m['closure_samples'][i]).all()) for i in range(1, len(names))]
                                if closed else [])
    out['class'] = taxonomy.classify(row, out)
    return out


def measure_trial(folder):
    """Replay one recorded trial and reduce it to raw measures."""
    from experiments.xvla_props_p2_eval.diagnostics import replay
    return measures_from_replay(replay.replay_trial(Path(folder)))


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
        m = measure_trial(folder)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(m, indent=2) + '\n')
        print('MEASURE', run_name(mode, policy, candidate), folder.name, m['gate0'], flush=True)
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


W, V = run_name('report', 'p2', 101520), run_name('report', 'p2', 126900)
P101 = [run_name(m, 'p2', 101520) for m in ('screen', 'report', 'control', 'confirm')]
P126 = [run_name(m, 'p2', 126900) for m in ('screen', 'report')]


def gate1_null_by_object(rows_a, rows_b, n=2000, seed=0):
    """`stats.gate1_null` per object over the successes' grasp samples, weighted by the number of `rows_a` successes
    (forward) and `rows_b` successes (backward), i.e. what Gate 1 reads when the two policies do not differ."""
    from experiments.xvla_props_p2_eval.diagnostics import stats
    forward = backward = wa = wb = 0.0
    for name in sorted({r['object'] for r in rows_a}):
        a = [r['grasp_sample'] for r in rows_a if r['object'] == name and r['task_success'] and r.get('grasp_sample') is not None]
        b = [r['grasp_sample'] for r in rows_b if r['object'] == name and r['task_success'] and r.get('grasp_sample') is not None]
        out = stats.gate1_null(a, b, n=n, seed=seed)
        forward += out['forward']*len(a)
        wa += len(a)
        backward += out['backward']*len(b)
        wb += len(b)
    return dict(forward=forward/wa, backward=backward/wb)


def _write_descriptive(data, code, **gate1):
    """The analyses that need no envelope, written when the taxonomy cannot be produced; returns the exit `code`."""
    from experiments.xvla_props_p2_eval.diagnostics import tables
    result = tables.build_descriptive(data)
    result['gate1'] = gate1
    (OUT/'analysis_descriptive.json').write_text(json.dumps(result, indent=2, default=float) + '\n')
    print(tables.render_descriptive(result), flush=True)
    return code


def do_analyze():
    from experiments.xvla_props_p2_eval.diagnostics import tables
    data = load_measures()
    every = [r for rows in data.values() for r in rows]
    print(f'GATE0 {sum(r["gate0"] for r in every)}/{len(every)} trials reproduced exactly', flush=True)
    if not gate0_ok(every):
        print('GATE0 FAILED: no analysis on a replay that does not reproduce its own data', flush=True)
        return 1
    rows101 = [r for n in P101 for r in data[n]]
    rows126 = [r for n in P126 for r in data[n]]
    try:
        env_from_126, env_from_101 = success_envelope(rows126), success_envelope(rows101)
    except ValueError as exc:
        print(f'GATE1 STOPPED: {exc}; the analyses that need no envelope are still written', flush=True)
        return _write_descriptive(data, 2, stopped=str(exc))
    objects = {n for rows in data.values() for r in rows for n in r['names']}
    missing = sorted((objects - set(env_from_126)) | (objects - set(env_from_101)))
    if missing:       # an object that appears in any trial but has no envelope in either direction stops the analysis
        print(f'ENVELOPE MISSING for {missing} (too few successes as the target): Gate 1 and the failure taxonomy stop; '
              f'the analyses that need no envelope are still written', flush=True)
        return _write_descriptive(data, 3, missing_envelope=missing)
    forward, backward = gate1_fraction(rows101, env_from_126), gate1_fraction(rows126, env_from_101)
    print(f'GATE1 forward {forward:.3f} (101,520 successes inside the 126,900-success envelope), backward {backward:.3f} '
          f'(126,900 successes inside the 101,520-success envelope); minimum {GATE1_MIN} each', flush=True)
    null = gate1_null_by_object(rows101, rows126)
    print(f'GATE1 NULL (two policies drawn from one pooled distribution, same sample sizes, object-weighted): forward '
          f'{null["forward"]:.3f}, backward {null["backward"]:.3f}', flush=True)
    if not gate1_passes(forward, backward):
        print('GATE1 FAILED under the amended definition: no further redefinition (spec stop rule); the failure taxonomy '
              'is not produced. Writing the analyses that need no envelope.', flush=True)
        return _write_descriptive(data, 2, forward=forward, backward=backward, null=null, minimum=GATE1_MIN,
                                  envelope_from_126900=env_from_126, envelope_from_101520=env_from_101)
    judged = {}
    for name, rows in data.items():       # 101,520 and P1 are judged by 126,900's successes, 126,900 by 101,520's
        env = env_from_101 if name in P126 else env_from_126
        judged[name] = [judge(r, r, env) for r in rows]
    result = tables.build(judged)
    result['gate1'] = dict(forward=forward, backward=backward, envelope_from_126900=env_from_126,
                           envelope_from_101520=env_from_101)
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
