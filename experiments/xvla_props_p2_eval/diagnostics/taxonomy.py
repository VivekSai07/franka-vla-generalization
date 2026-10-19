"""First-cause classification of one trial for the P2 failure diagnostics (spec 2026-10-19, taxonomy table)."""

CLASSES = ('wrong_object', 'lifted_then_failed', 'never_closed', 'closed_on_distractor', 'closed_off_target',
           'closed_not_held', 'held_not_lifted')
LIFTED_THEN_FAILED = ('timeout_carrying', 'dropped_outside', 'unstable_in_target', 'wrong_bin')


def classify(row, m):
    """`row`: evaluation fields `cause`, `task_success`. `m`: `closed`, `target_in_env`, `distractor_in_env` (list),
    `held` and `lifted`, measured at the first closure (the grasp chain applies to `never_lifted` trials only)."""
    if row['task_success']:
        return 'success'
    cause = row['cause']
    if cause == 'wrong_object':
        return 'wrong_object'
    if cause in LIFTED_THEN_FAILED:
        return 'lifted_then_failed'
    if cause != 'never_lifted':
        raise ValueError(f'Unknown failure cause {cause!r}')
    if not m['closed']:
        return 'never_closed'
    if not m['target_in_env']:
        return 'closed_on_distractor' if any(m['distractor_in_env']) else 'closed_off_target'
    return 'held_not_lifted' if m['held'] else 'closed_not_held'
