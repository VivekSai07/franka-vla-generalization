"""The morning summary of a pilot night: one line per pilot, so the judgement takes minutes.

`python -m experiments.xvla_p2_pilot.summary <night_root>` writes `summary.json` and `SUMMARY.md` into the night's root from what
the driver (`scripts/vla_inputs_pilot_night.sh`) left there: `night.json` (the variants and their arms), `events.jsonl`
(wall-clock and failures), `training/<variant>[_zeroeq]/result.json`, and the evaluator's records under the night root itself (the campaign root is the night root). Pure
Python: no simulator, no GPU. Deterministic: the same inputs give the same bytes.

The decision is the pre-registered one (design, "Decision rule"): **promote iff the success difference to the shared control, on
the paired pilot layouts, is at least +5 points**, otherwise drop. It is computed from the integer counts with exact
arithmetic, so a difference that is 4.99 points is a drop whatever floating point says. Mechanism counts (`never_lifted`,
`wrong_object`, ...) are reported beside Δ and never decide. A variant whose gate, training or evaluation did not complete gets
**no decision** and the reason; it is never read from partial data.

A pilot is a screen, never a reported result. Until the force pilot has been calibrated against M2's full result (plan Task 7)
every *drop* is provisional: a pilot may promote, and may drop only once the regime is calibrated.
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

PROMOTE_POINTS = 5
RULE = 'promote iff delta >= +5 points'
BUNDLE_CHECKPOINT = 101_520
CALIBRATION_NOTE = ('Calibration (plan Task 7) has not been done: every drop is provisional until the force pilot has been compared '
                    "with M2's full result. A pilot is a screen, not a result.")
EQUALITIES = ('metrics_equal', 'model_state_equal', 'optimizer_state_equal')


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an integer count')
    return value


def decide(new_successes, control_successes, layouts):
    """Δ in points over `layouts` paired layouts and the decision of the fixed rule, from exact integer arithmetic."""
    new, control, layouts = (_count(v, n) for v, n in ((new_successes, 'new_successes'), (control_successes, 'control_successes'),
                                                       (layouts, 'layouts')))
    if layouts <= 0 or not 0 <= new <= layouts or not 0 <= control <= layouts:
        raise ValueError('Counts must satisfy 0 <= successes <= layouts, layouts > 0')
    delta = Fraction(100*(new-control), layouts)
    return dict(delta_points=float(delta), decision='promote' if delta >= PROMOTE_POINTS else 'drop', rule=RULE,
                threshold_points=PROMOTE_POINTS, new_successes=new, control_successes=control, layouts=layouts)


def _read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _stage(root, arm, checkpoint):
    result = _read(Path(root)/'arms'/arm/'comparison'/f'step_{checkpoint:06d}/result.json')
    if result is None:
        return None
    if result.get('status') != 'complete':
        raise ValueError(f'{arm} comparison stage is not complete')
    episodes = result['episodes']
    return dict(arm=arm, checkpoint=checkpoint, successes=sum(1 for e in episodes if e['task_success']), layouts=len(episodes),
                causes=dict(result['readouts']['causes']))


def _events(root):
    path = Path(root)/'events.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _durations(events):
    """{(phase, variant): seconds} from matching start and end events; an unmatched start is not a duration."""
    started, out = {}, {}
    for event in events:
        key = (event.get('phase'), event.get('variant'))
        if event.get('event') == 'start':
            started[key] = event['t']
        elif event.get('event') == 'end' and key in started:
            out[key] = event['t']-started.pop(key)
    return out


def _problem(gate, training):
    if gate is None or gate.get('status') != 'complete' or not all(gate.get(k) is True for k in EQUALITIES):
        return 'zero-equivalence gate did not pass' + (f" ({gate.get('status')}: {gate.get('error', '')})" if gate else ' (no record)')
    if training is None or training.get('status') != 'complete':
        return 'training did not complete' + (f" ({training.get('status')}: {training.get('error', '')})" if training else ' (no record)')
    return None


def _variant(root, night, name, arm, control, durations):
    checkpoint = night['checkpoint']
    gate = _read(Path(root)/f'training/{name}_zeroeq/result.json')
    training = _read(Path(root)/f'training/{name}/result.json')
    compare = _read(Path(root)/f'arms/{arm}/compare_step_{checkpoint:06d}.json')
    clock = {phase: durations[(phase, name)] for phase in ('zero_equivalence', 'train', 'evaluate') if (phase, name) in durations}
    if 'train' not in clock and training and 'elapsed_seconds' in training:
        clock['train'] = training['elapsed_seconds']
    row = dict(name=name, arm=arm, status='no_decision', decision=None, reason=None, delta_points=None, wall_clock_s=clock,
               variant_sha256=(training or {}).get('variant', {}).get('sha256'))
    problem = _problem(gate, training)
    if problem:
        row['reason'] = problem
        return row
    if compare is None:
        row['reason'] = 'no comparison against the control (the evaluation did not complete)'
        return row
    if compare['checkpoint'] != checkpoint or compare['arm'] != arm:
        raise ValueError(f'{arm}: the comparison is of {compare["arm"]} step {compare["checkpoint"]}, not of this night\'s step {checkpoint}')
    if (compare['layouts'] != control['layouts'] or compare['control_successes'] != control['successes']
            or compare['new_successes']-compare['control_successes'] != compare['discordant']['new_only']-compare['discordant']['control_only']):
        raise ValueError(f'{arm}: the comparison disagrees with its own counts or with the control stage')
    row.update(decide(compare['new_successes'], compare['control_successes'], compare['layouts']))
    row.update(status='decided', discordant=compare['discordant'], mcnemar_exact_p=compare['mcnemar_exact_p'],
               effect_points={k: 100*v for k, v in compare['effect'].items()}, role=compare['role'])
    new, old = compare['readouts']['new']['causes'], compare['readouts']['control']['causes']
    row['mechanism'] = {cause: dict(variant=new[cause], control=old[cause], change=new[cause]-old[cause])
                        for cause in new if cause != 'success'}
    return row


def build(root):
    root = Path(root)
    night = _read(root/'night.json')
    if night is None:
        raise ValueError(f'{root} has no night.json; it is not a pilot night root')
    events = _events(root)
    durations = _durations(events)
    control = _stage(root, night['control_arm'], night['checkpoint'])
    if control is None:
        raise ValueError('The control has no completed comparison stage; there is nothing to compare against')
    control_record = _read(root/'training/control/result.json') or {}
    context = _stage(root, night['context_arm'], BUNDLE_CHECKPOINT)
    summary = dict(recipe_version=night['recipe_version'], updates=night['updates'], rule=RULE, threshold_points=PROMOTE_POINTS,
                   calibrated=False, calibration_note=CALIBRATION_NOTE,
                   control=dict(control, wall_clock_s=durations.get(('train', 'control'), control_record.get('elapsed_seconds'))),
                   context=None, variants=[_variant(root, night, name, arm, control, durations) for name, arm in night['variants']])
    if context is not None:
        summary['context'] = dict(context, continuation_alone_points=100*(control['successes']-context['successes'])/control['layouts'])
    failures = [f"{v['name']}: {v['reason']}" for v in summary['variants'] if v['status'] == 'no_decision']
    failures += [f"{e.get('phase')} {e.get('variant')}: {e.get('detail', '')}".strip() for e in events if e.get('event') == 'fail']
    summary['failures'] = failures
    times = [e['t'] for e in events if 't' in e]
    summary['total_wall_clock_s'] = max(times)-min(times) if times else None
    return summary


def _markdown(s):
    c = s['control']
    lines = ['# Pilot night summary', '',
             f"Recipe `{s['recipe_version']}`, {s['updates']} updates from checkpoint {BUNDLE_CHECKPOINT:,}. A pilot is a screen, never a "
             f"result. Rule: **{s['rule']}**.", s['calibration_note'], '',
             f"Control (`{c['arm']}`): {c['successes']}/{c['layouts']} on the pilot layouts."
             + (f" Untouched bundle checkpoint (`{s['context']['arm']}`): {s['context']['successes']}/{s['context']['layouts']}; "
                f"continuation alone {s['context']['continuation_alone_points']:+.2f} points." if s['context'] else ''), '',
             '| variant | arm | Δ points | decision | successes | discordant (new/control) | McNemar p | never_lifted (ctl→var) | wrong_object (ctl→var) |',
             '|---|---|---|---|---|---|---|---|---|']
    for v in s['variants']:
        if v['status'] == 'decided':
            m = v['mechanism']
            lines.append(f"| {v['name']} | {v['arm']} | {v['delta_points']:+.2f} | **{v['decision']}** | {v['new_successes']}/{v['layouts']} vs "
                         f"{v['control_successes']}/{v['layouts']} | {v['discordant']['new_only']}/{v['discordant']['control_only']} | "
                         f"{v['mcnemar_exact_p']:.4f} | {m['never_lifted']['control']}→{m['never_lifted']['variant']} | "
                         f"{m['wrong_object']['control']}→{m['wrong_object']['variant']} |")
        else:
            lines.append(f"| {v['name']} | {v['arm']} | | **no decision** | | | | {v['reason']} | |")
    lines += ['', 'Mechanism counts (all causes) are in `summary.json`; they are reported, never decisive. McNemar is the exact two-sided test on the '
                  'paired layouts; Δ is over the feasible pilot layouts.']
    lines += ['', 'Wall clock: ' + ', '.join(f"{v['name']} " + '/'.join(f'{k} {t/3600:.1f} h' for k, t in v['wall_clock_s'].items())
                                              for v in s['variants'] if v['wall_clock_s'])
              + (f"; night {s['total_wall_clock_s']/3600:.1f} h" if s['total_wall_clock_s'] else '')]
    if s['failures']:
        lines += ['', '**Failures:**'] + [f'- {f}' for f in s['failures']]
    return '\n'.join(lines)+'\n'


def write(root):
    root = Path(root)
    result = build(root)
    (root/'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    (root/'SUMMARY.md').write_text(_markdown(result))
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit('usage: python -m experiments.xvla_p2_pilot.summary <night_root>')
    result = write(argv[0])
    print(_markdown(result))


if __name__ == '__main__':
    main()
