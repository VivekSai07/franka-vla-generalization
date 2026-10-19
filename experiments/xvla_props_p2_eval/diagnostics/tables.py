"""The analysis tables of the P2 failure diagnostics (spec 2026-10-19). Pure: rows are evaluation rows merged with the
per-trial measures of `run.measures_from_replay`."""
import math
from collections import Counter, defaultdict

import numpy as np

from experiments.xvla_props_p2_eval.diagnostics import stats, taxonomy

ORDER = ('success', *taxonomy.CLASSES)
WINNER = 'report_p2_step_101520'
CONTRASTS = ('report_p2_step_126900', 'report_p1_step_126900')
WINNER_ALL = ('screen_p2_step_101520', 'report_p2_step_101520', 'control_p2_step_101520', 'confirm_p2_step_101520')


def min_distractor_distance(row):
    """Distance from the target's start to the nearest distractor's start (metres, table plane)."""
    if 'min_distractor_distance_m' in row:
        return row['min_distractor_distance_m']
    target = np.asarray(row['start'][:2], dtype=float)
    return min(float(np.linalg.norm(target - np.asarray(d['pose'][:2], dtype=float))) for d in row['distractors'])


def _count(rows):
    c = Counter(r['class'] for r in rows)
    out = {k: c.get(k, 0) for k in ORDER}
    out['n'] = len(rows)
    return out


def taxonomy_table(data):
    return {name: _count(rows) for name, rows in data.items()}


def never_lifted_concentration(rows, n=10_000, seed=0):
    """Layout concentration of the `never_lifted` trials among `rows` (one run), by permutation over trials with each
    layout's trial count kept."""
    by = defaultdict(list)
    for r in rows:
        by[r['layout_id']].append(r['cause'] == 'never_lifted')
    counts = {k: sum(v) for k, v in by.items()}
    return dict(failed_trials=sum(counts.values()), layouts_with_failures=sum(1 for v in counts.values() if v),
                layouts=len(by), statistic=stats.concentration(counts), p=stats.concentration_p(by, n=n, seed=seed))


def layouts_failing_at_every_run(runs, cause='never_lifted'):
    """Layouts in which every trial of every run in `runs` ({name: rows}) is `cause`."""
    per_run = []
    for rows in runs.values():
        by = defaultdict(list)
        for r in rows:
            by[r['layout_id']].append(r['cause'] == cause)
        per_run.append({k for k, v in by.items() if all(v)})
    return sorted(set.intersection(*per_run)) if per_run else []


def wrong_object_layout_test(rows):
    """Layout is the unit: positive if any of its trials is `wrong_object`; one two-sided permutation test on the
    layout's minimum target-distractor distance."""
    by = defaultdict(list)
    for r in rows:
        by[r['layout_id']].append(r)
    pos, neg = [], []
    for trials in by.values():
        value = min(min_distractor_distance(r) for r in trials)
        (pos if any(r['class'] == 'wrong_object' for r in trials) else neg).append(value)
    return dict(positive_layouts=len(pos), negative_layouts=len(neg),
                positive_median_m=float(np.median(pos)) if pos else math.nan,
                negative_median_m=float(np.median(neg)) if neg else math.nan,
                p=stats.mann_whitney_p(pos, neg) if pos and neg else math.nan)


def wrong_object_cases(rows):
    return [dict(layout=r['layout_id'], target=r['object'], lifted=r['first_lifted'], distractors=[d['object'] for d in r.get('distractors', [])],
                 min_distractor_distance_m=min_distractor_distance(r), closed_on_distractor=any(r.get('distractor_in_env', [])),
                 target_in_env=r.get('target_in_env')) for r in rows if r['class'] == 'wrong_object']


def milk_table(rows):
    """Milk trials by cell with the class counts and the per-trial grasp quantities."""
    out = {}
    for r in rows:
        if r['object'] != 'milk':
            continue
        key = f"milk->{r['bin']} ({r['kind']})"
        cell = out.setdefault(key, {**{k: 0 for k in ORDER}, 'n': 0, 'trials': []})
        cell['n'] += 1
        cell[r['class']] += 1
        cell['trials'].append(dict(layout=r['layout_id'], cls=r['class'], target_in_hand=r.get('target_in_hand_at_closure'),
                                   yaw_error_deg=r.get('yaw_error_deg'), closest_approach_m=r.get('closest_approach_m')))
    return out


def build(data):
    winner = data[WINNER]
    every_winner = [r for name in WINNER_ALL for r in data.get(name, [])]
    return dict(
        taxonomy=taxonomy_table(data),
        never_lifted_concentration={name: never_lifted_concentration(data[name]) for name in (WINNER, *CONTRASTS) if name in data},
        never_lifted_in_every_run=layouts_failing_at_every_run({k: data[k] for k in (WINNER, CONTRASTS[0]) if k in data}),
        never_lifted_closest_approach_m=[round(r['closest_approach_m'], 4) for r in every_winner if r['cause'] == 'never_lifted'],
        wrong_object_layout_test=wrong_object_layout_test(winner),
        wrong_object_cases={name: wrong_object_cases(data[name]) for name in (WINNER, *CONTRASTS) if name in data},
        milk=milk_table(every_winner),
        tests_run=['never_lifted_concentration (1 per run listed)', 'wrong_object_layout_test (1)', 'milk comparisons (reported, not tested)'],
    )


def _row(cells):
    return '| ' + ' | '.join(str(c) for c in cells) + ' |'


def render(result):
    names = list(result['taxonomy'])
    lines = ['### Taxonomy by run', '', _row(['class', *names]), _row(['---'] * (len(names) + 1))]
    for cls in (*ORDER, 'n'):
        lines.append(_row([cls, *(result['taxonomy'][n][cls] for n in names)]))
    lines += ['', '### Never-lifted layout concentration', '']
    for name, v in result['never_lifted_concentration'].items():
        lines.append(f"- {name}: {v['failed_trials']} never-lifted trials in {v['layouts_with_failures']} of {v['layouts']} layouts, "
                     f"statistic {v['statistic']}, permutation p = {v['p']:.4f}")
    lines.append(f"- layouts never solved in both 101,520 and 126,900 report runs: {result['never_lifted_in_every_run']}")
    w = result['wrong_object_layout_test']
    lines += ['', '### Wrong object (exploratory)', '',
              f"- layouts with a wrong-object trial: {w['positive_layouts']}, without: {w['negative_layouts']}; median nearest "
              f"target-distractor distance {w['positive_median_m']:.3f} m against {w['negative_median_m']:.3f} m; "
              f"exact/permutation p = {w['p']:.4f}"]
    for name, cases in result['wrong_object_cases'].items():
        lines.append(f'- {name}: {len(cases)} trials: ' + '; '.join(f"{c['layout']} lifted {c['lifted']} (target {c['target']}, gap {c['min_distractor_distance_m']:.3f} m)" for c in cases))
    lines += ['', '### Milk (101,520, all tiers)', '', _row(['cell', 'n', *ORDER[:0], *[c for c in ORDER]]), _row(['---'] * (len(ORDER) + 2))]
    for cell, v in result['milk'].items():
        lines.append(_row([cell, v['n'], *[v[c] for c in ORDER]]))
    return '\n'.join(lines)
