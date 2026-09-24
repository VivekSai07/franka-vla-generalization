"""Broad-workspace dataset admission; per-episode reset SHAs and 400 replay coverage."""
from collections import Counter
from contextlib import contextmanager
import copy
import json
from pathlib import Path

import numpy as np

from experiments.fixed_layout import validate as generic
from experiments.fixed_layout.record import write_json, file_hash
from . import plan as p
from .scenes import make_bank, bank_manifest, provenance, exact, derive_scene


def metadata(root, count, final=False):
    """Local replacement for `generic._metadata`, which hardcodes a single task.

    The shared validator is bound by four contracts, so it is overridden here rather than edited:
    it asserts `total_tasks == 1`, `tasks.index == [TASK]` and `ep['tasks'] == [TASK]`. B1 has two
    instructions. Every other check is reproduced unchanged.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    if type(count) is not int or not 1 <= count <= p.ADMIT_COUNT:
        raise ValueError('Invalid admitted-demonstration count')
    ds = Path(root)/'dataset'
    info = generic._json(ds/'meta/info.json')
    # `total_tasks` grows as instructions first appear, so it is bounded during collection and
    # pinned exactly only at the end.
    seen_tasks = info['total_tasks']
    if (info['total_episodes'] != count or info['total_frames'] != p.EXPECTED_FRAMES*count
            or not 1 <= seen_tasks <= len(p.INSTRUCTIONS) or info['fps'] != 30):
        raise ValueError('Dataset counts/task/fps mismatch')
    if final and seen_tasks != len(p.INSTRUCTIONS):
        raise ValueError(f'Final dataset must contain {len(p.INSTRUCTIONS)} instructions, saw {seen_tasks}')
    expected = set(generic.FEATURES) | {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
    if set(info['features']) != expected:
        raise ValueError('Privileged/missing policy feature')
    for key, feature in generic.FEATURES.items():
        for field in ('dtype', 'shape', 'names'):
            wanted = list(feature[field]) if isinstance(feature[field], tuple) else feature[field]
            if info['features'][key][field] != wanted:
                raise ValueError('Dataset feature contract mismatch')

    # LeRobot numbers tasks by first appearance, which need not match the plan's sorted mapping.
    # The instruction *text* is what the policy consumes, so that is what is pinned: every task in
    # the table must be an approved instruction, the numbering must be a bijection, and each
    # episode must carry exactly the instruction its plan slot specifies.
    tasks = pd.read_parquet(ds/'meta/tasks.parquet')
    table = dict(zip(list(tasks.index), list(tasks['task_index'])))
    if not set(table) <= set(p.INSTRUCTIONS):
        raise ValueError(f'Unapproved instruction in dataset: {sorted(set(table) - set(p.INSTRUCTIONS))}')
    if len(set(table.values())) != len(table) or sorted(table.values()) != list(range(len(table))):
        raise ValueError(f'Task indices are not a contiguous bijection: {table}')
    if final and set(table) != set(p.INSTRUCTIONS):
        raise ValueError(f'Final dataset tasks {sorted(table)} != planned {sorted(p.INSTRUCTIONS)}')

    episodes = []
    for path in sorted((ds/'meta/episodes').rglob('*.parquet')):
        episodes.extend(pq.read_table(path).to_pylist())
    if len(episodes) != count:
        raise ValueError('Episode metadata coverage differs')
    planned = p.make_plan()['episodes']
    for i, ep in enumerate(episodes):
        if (ep['episode_index'] != i or ep['length'] != p.EXPECTED_FRAMES
                or ep['dataset_from_index'] != p.EXPECTED_FRAMES*i
                or ep['dataset_to_index'] != p.EXPECTED_FRAMES*(i+1)):
            raise ValueError('Episode metadata order/boundary mismatch')
        if list(ep['tasks']) != [planned[i]['instruction']]:
            raise ValueError(f'Episode {i} instruction differs from the plan')

    if final:
        require_final_split(info, count)
        require_cell_balance(planned)
    return info, episodes


def require_final_split(info, count):
    wanted = {'train': f'0:{p.TRAIN_LHS_COUNT + p.TRAIN_BASE_COUNT}',
              'validation': f'{p.TRAIN_LHS_COUNT + p.TRAIN_BASE_COUNT}:{p.ADMIT_COUNT}'}
    if count != p.ADMIT_COUNT or info.get('splits') != wanted:
        raise ValueError('Expected 160 training and 40 validation demonstrations')


def require_cell_balance(episodes):
    """Balance is the property that removes the position shortcut; assert it survived collection."""
    for split, n in (('train', p.TRAIN_LHS_COUNT + p.TRAIN_BASE_COUNT), ('validation', p.VALIDATION_COUNT)):
        counts = Counter(e['cell'] for e in episodes if e['split'] == split)
        if set(counts) != set(p.CELL_IDS) or set(counts.values()) != {n // len(p.CELL_IDS)}:
            raise ValueError(f'{split} cells unbalanced: {dict(sorted(counts.items()))}')


def check_replay_coverage(results):
    expected = Counter((i, m) for i in range(p.ADMIT_COUNT) for m in generic.MODES)
    if (any(type(r.get('episode')) is not int or type(r.get('mode')) is not str for r in results)
            or Counter((r['episode'], r['mode']) for r in results) != expected
            or any(r.get('passed') is not True for r in results)):
        raise ValueError(f'Require exactly {p.ADMIT_COUNT} x 2 unique saved-control replays')


def validate_journal(events, plan):
    p.validate_plan(plan)
    started = [e for e in events if e['event'] == 'started']
    admitted = [(e['event'], e.get('episode_index')) for e in events if e['event'] == 'admitted']
    # A retried slot journals several 'started' events, so require non-decreasing slot order
    # covering every slot rather than a strict one-to-one sequence.
    order = [e.get('episode_index') for e in started]
    if (order != sorted(order) or sorted(set(order)) != list(range(p.ADMIT_COUNT))
            or [e[1] for e in admitted] != list(range(p.ADMIT_COUNT))):
        raise ValueError('Journal must record ordered started/admitted for all admits')
    if sum(e['event'] == 'rejected' for e in events) > p.REJECTION_BUDGET:
        raise ValueError('Rejection budget exceeded in journal')
    times = [e['elapsed_seconds'] for e in events]
    if (any(type(t) not in (int, float) or not np.isfinite(t) or not 0 <= t < p.MAX_SECONDS for t in times)
            or times != sorted(times)):
        raise ValueError('Invalid twelve-hour elapsed evidence')
    if any(e.get('training_ready') is not False for e in events):
        raise ValueError('Premature training admission in journal')
    # Only the attempt that actually produced the admit carries the slot's immutable identity.
    pending = {}
    for event in events:
        if event['event'] == 'started':
            pending[event.get('episode_index')] = event
        elif event['event'] == 'admitted':
            index = event.get('episode_index')
            source = pending.get(index)
            if source is None:
                raise ValueError('Admitted demonstration without a matching started event')
            for key in ('episode_index', 'seed', 'layout_id', 'split', 'pick', 'place',
                        'colors', 'instruction', 'task_index', 'cell', 'arrangement'):
                if source.get(key) != plan['episodes'][index][key]:
                    raise ValueError(f'Journal slot immutable field changed: {key}')


def diagnostics(expert, trace, terminal, actions, reset, check):
    post = generic._original_post(expert, trace, terminal, check)
    delta = post['pick_xyz']-post['place_xyz']
    phases = []
    for phase in np.unique(trace['phase_index']):
        mask = trace['phase_index'] == phase
        phases.append(dict(phase_index=int(phase), ticks=int(mask.sum()), duration_seconds=float(mask.sum()/30),
            bilateral_ticks=int(np.sum((post['left_contact'] & post['right_contact'])[mask])),
            saturation_per_joint=trace['command_saturation'][mask].sum(axis=0).astype(int).tolist()))
    opened = np.flatnonzero((actions[1:, 7] > 127) & (actions[:-1, 7] <= 127))+1
    releases = [dict(tick=int(i), pick_xyz=post['pick_xyz'][i].tolist(),
                     place_xyz=post['place_xyz'][i].tolist(), delta_xyz=delta[i].tolist()) for i in opened]
    return dict(phases=phases, releases=releases, final_delta_xyz=delta[-1].tolist(),
                final_xy_m=float(np.linalg.norm(delta[-1, :2])),
                saturation_per_joint=trace['command_saturation'].sum(axis=0).astype(int).tolist(),
                initial_pick_xyz=reset['body_xpos'][expert.model.body('target_cube_red').id].tolist(),
                initial_place_xyz=reset['body_xpos'][expert.model.body('target_cube_green').id].tolist())


def episode_report(i, gates, media, replays, detail):
    return dict(episode=i, passed=True, original_gates=gates, media=media, replays=replays, diagnostics=detail,
        original_contacts='geometric_reconstruction_from_saved_post_qpos_confirmed_by_reference_replay')


def _matches_plan_slot(episode):
    plan = p.make_plan(); i = episode.get('episode_index')
    if type(i) is not int or not 0 <= i < p.ADMIT_COUNT:
        return False
    slot = plan['episodes'][i]
    for key in ('episode_index', 'seed', 'layout_id', 'split', 'pick', 'place',
                'colors', 'instruction', 'task_index', 'cell', 'arrangement',
                'hover_z_jitter_m', 'hover_z_offset'):
        if episode.get(key) != slot[key]:
            return False
    return True


def require_approved_draw(slot, executed):
    """An executed slot may differ from its plan slot only by an approved feasibility redraw."""
    from .record import redraw_home_pose_jitter, FEASIBILITY_REDRAWS
    for key in ('episode_index', 'seed', 'layout_id', 'split', 'pick', 'place',
                'colors', 'instruction', 'task_index', 'cell', 'arrangement',
                'hover_z_jitter_m', 'hover_z_offset'):
        if executed.get(key) != slot[key]:
            raise ValueError(f'Executed slot changed immutable field: {key}')
    index, k = slot['episode_index'], executed.get('feasibility_k')
    if type(k) is not int or not 0 <= k <= FEASIBILITY_REDRAWS:
        raise ValueError('Executed slot carries an out-of-range feasibility draw')
    approved = list(slot['home_pose_jitter_deg']) if k == 0 else redraw_home_pose_jitter(index, k-1)
    source = 'plan' if k == 0 else f'redraw:{k-1}'
    if list(executed.get('home_pose_jitter_deg', [])) != list(approved):
        raise ValueError('Executed home pose is not the approved draw for this slot')
    if executed.get('feasibility_source') != source:
        raise ValueError('Executed feasibility source does not match its draw index')


def read_rows_for_task(path, index, task_index):
    """`generic.read_rows` with the frame task index supplied by the caller.

    Byte-for-byte the shared implementation except that `task_index` is compared against
    `task_index` instead of zeros, because B1 has two instructions. The shared module is inside
    the sealed provenance of `broad_workspace_data` and must not be edited (FAILURES #28);
    `test_read_rows_matches_shared_on_single_task` pins this copy to the original.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    expected = {'action', 'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
    n = p.EXPECTED_FRAMES
    if set(table.column_names) != expected or len(table) != n:
        raise ValueError('Canonical persisted rows require 528 frames and no privileged fields')
    for key, size in (('action', 8), ('observation.state', 9)):
        dtype = table.schema.field(key).type
        if not pa.types.is_fixed_size_list(dtype) or dtype.list_size != size or dtype.value_type != pa.float32():
            raise ValueError('Persisted action/state must be unpadded float32')
    for key in expected:
        column = table[key]
        if column.null_count or (pa.types.is_fixed_size_list(column.type) and column.combine_chunks().values.null_count):
            raise ValueError('Null persisted row value')
    for key, wanted in (('frame_index', np.arange(n)), ('index', np.arange(n) + index*n),
                        ('episode_index', np.full(n, index)),
                        ('task_index', np.full(n, task_index, np.int64))):
        if table.schema.field(key).type != pa.int64():
            raise ValueError('Row indices must be int64')
        generic._equal(table[key].to_numpy(), wanted, 'Persisted row/task indices differ')
    if table.schema.field('timestamp').type != pa.float32():
        raise ValueError('Persisted timestamps must be float32')
    actions = np.asarray(table['action'].to_pylist(), np.float32)
    states = np.asarray(table['observation.state'].to_pylist(), np.float32)
    times = table['timestamp'].to_numpy()
    if not all(np.isfinite(a).all() for a in (actions, states, times)):
        raise ValueError('Nonfinite persisted rows')
    generic._equal(times, np.arange(n)/30, 'Persisted timestamp discontinuity', 2e-6)
    return actions, states, times


@contextmanager
def task_aware(instruction, task_index):
    """Widen the shared validator for exactly one episode's instruction, then restore it.

    Two sites in `fixed_layout.validate` assume a single task: `read_rows` expects every frame's
    index to be 0, and the raw-frame check compares against the module-level `TASK`.
    """
    original_rows, original_task = generic.read_rows, generic.TASK
    generic.read_rows = lambda path, index: read_rows_for_task(path, index, task_index)
    generic.TASK = instruction
    try:
        yield
    finally:
        generic.read_rows, generic.TASK = original_rows, original_task


def lerobot_task_index(root, instruction):
    """LeRobot numbers tasks by first appearance, so read the index it actually assigned."""
    import pandas as pd
    tasks = pd.read_parquet(Path(root)/'dataset'/'meta'/'tasks.parquet')
    table = dict(zip(list(tasks.index), list(tasks['task_index'])))
    if instruction not in table:
        raise ValueError(f'Instruction missing from the dataset task table: {instruction!r}')
    return int(table[instruction])


def validate_episode(root, episode, expert, reset, check):
    if not _matches_plan_slot(episode):
        raise ValueError('Unplanned demonstration slot; evaluation execution forbidden')
    i = episode['episode_index']; info, entries = metadata(root, i+1)
    instruction = episode['instruction']
    with task_aware(instruction, lerobot_task_index(root, instruction)):
        actions, states, trace, gates, media = generic._episode_inputs(root, episode, expert, reset, check, info, entries)
    path = Path(root) / 'attempts' / f'{i:06d}'; (path / 'replays').mkdir(exist_ok=False)
    results = [generic.replay_controls(path / 'replays', expert, reset, trace, actions, mode, check)
               for mode in generic.MODES]
    detail = diagnostics(expert, trace, generic._npz(path / 'terminal.npz'), actions, reset, check)
    report = episode_report(i, gates, media, results, detail); write_json(path / 'validation.json', report)
    sequences = []
    for ep in entries:
        row = generic._child(Path(root)/'dataset',
                             info['data_path'].format(chunk_index=ep['data/chunk_index'], file_index=ep['data/file_index']))
        a, s, _ = generic.read_rows(row, ep['episode_index']); sequences.append((a, s))
    dup = generic.duplicate_report(sequences); write_json(path / 'duplicates.json', dup); generic.require_no_duplicates(dup)
    check()
    return report


def bindings(root, expert, base_source, bank, check):
    from .record import check_snapshots
    root = Path(root); check()
    if (root / 'failure.json').exists():
        raise ValueError('Failed collection cannot admit training')
    plan = generic._json(root / 'plan.json'); p.validate_plan(plan)
    check_snapshots(root, generic._json(root / 'snapshots.json'), check)
    if file_hash(root / 'source.npz', check) != p.SOURCE_SHA256:
        raise ValueError('Source snapshot changed')
    if generic._json(root / 'configuration.json') != expert.config:
        raise ValueError('Configuration changed')
    actual = make_bank(expert.model, base_source, plan, check)
    if set(bank) != set(actual):
        raise ValueError('Incomplete live bank')
    if generic._json(root / 'bank.json') != bank_manifest(plan, actual):
        raise ValueError('Wrong frozen bank manifest')
    for key, expected in actual.items():
        check()
        for candidate in (bank[key], expected):
            if set(candidate) != set(expected) or any(not exact(candidate[k], expected[k]) for k in expected):
                raise ValueError(f'Live bank differs from freshly-derived: {key}')
    return plan


def fingerprint(root, check=lambda: None):
    return generic.fingerprint(root, check)


def _episode_reset_sha_matches(root, i, expected):
    path = Path(root) / 'resets' / f'episode_{i:03d}.npz'
    return path.is_file() and file_hash(path) == expected


def validate_inventory(root):
    root = Path(root)
    required = {'claim.json', 'plan.json', 'provenance.json', 'configuration.json', 'source.npz',
                'bank.json', 'snapshots.json', 'journal.jsonl', 'resets', 'attempts', 'dataset'}
    actual = {x.name for x in root.iterdir()}
    if not required <= actual or not actual <= required | {'validation.json', 'seal.json', 'rejected', 'amendment.json'}:
        raise ValueError('Missing/extra/failed final root evidence')
    claim = dict(version=p.VERSION, output=str(p.OUTPUT.relative_to(p.ROOT)), resume=False)
    if not p._same(generic._json(root / 'claim.json'), claim):
        raise ValueError('Exclusive output claim changed')


def validate_dataset(root, expert, base_source, bank, check):
    root = Path(root); validate_inventory(root)
    plan = bindings(root, expert, base_source, bank, check)
    before = generic.fingerprint(root, check)
    events = [json.loads(line) for line in (root / 'journal.jsonl').read_text().splitlines()]
    validate_journal(events, plan)
    if sorted(x.name for x in (root / 'attempts').iterdir()) != [f'{i:06d}' for i in range(p.ADMIT_COUNT)]:
        raise ValueError(f'Exactly {p.ADMIT_COUNT} attempt directories required')
    info, entries = metadata(root, p.ADMIT_COUNT, final=True)
    episode_resets = info.get('episode_resets', {})
    if set(episode_resets) != {str(i) for i in range(p.ADMIT_COUNT)}:
        raise ValueError('meta/info.json missing per-episode reset SHAs')
    for i in range(p.ADMIT_COUNT):
        check()
        if not _episode_reset_sha_matches(root, i, episode_resets[str(i)]):
            raise ValueError(f'Reset SHA mismatch for episode {i}')
    executed_layouts = {slot['layout_id'] for slot in plan['episodes']}
    if executed_layouts & set(plan['evaluation_layout_ids']):
        raise ValueError('Evaluation layout executed during data stage')
    reports, results, sequences, rows, videos = [], [], [], set(), set()
    layouts_by_id = {x['layout_id']: x for x in plan['layouts']}
    layouts_by_id['base'] = plan['base_layout']
    for slot in plan['episodes']:
        check(); i = slot['episode_index']
        executed = generic._json(root / 'attempts' / f'{i:06d}' / 'slot.json')
        require_approved_draw(slot, executed)
        reset = derive_scene(expert.model, base_source, layouts_by_id[executed['layout_id']],
                             dict(home_pose_jitter_deg=executed['home_pose_jitter_deg']))
        actions, states, trace, gates, media = generic._episode_inputs(root, executed, expert, reset, check, info, entries)
        path = root / 'attempts' / f'{i:06d}'
        wanted_files = {'slot.json', 'outcome.json', 'terminal.npz', 'ticks', 'media', 'replays',
                        'validation.json', 'duplicates.json', 'resources.json'}
        if {x.name for x in path.iterdir()} != wanted_files:
            raise ValueError('Missing/extra/failed attempt evidence')
        if {x.name for x in (path / 'media').iterdir()} != {f'{k}.json' for k in generic.CAMERAS}:
            raise ValueError('Missing/extra camera media bindings')
        if sorted(x.name for x in (path / 'replays').iterdir()) != sorted(generic.MODES):
            raise ValueError('Replay mode inventory differs')
        replays = []
        for mode in generic.MODES:
            directory = path / 'replays' / mode
            if {x.name for x in directory.iterdir()} != {'result.json'} | {f'{j:06d}.npz' for j in range(p.EXPECTED_FRAMES)}:
                raise ValueError('Missing/extra replay evidence')
            result = generic.audit_replay(directory, expert, reset, trace, actions, mode, check)
            if generic._json(directory / 'result.json') != result:
                raise ValueError('Replay report changed')
            replays.append(result); results.append(dict(episode=i, **result))
        detail = diagnostics(expert, trace, generic._npz(path / 'terminal.npz'), actions, reset, check)
        report = episode_report(i, gates, media, replays, detail)
        if generic._json(path / 'validation.json') != report:
            raise ValueError('Episode report changed')
        sequences.append((actions, states)); dup = generic.duplicate_report(sequences)
        if generic._json(path / 'duplicates.json') != dup:
            raise ValueError('Duplicate report changed')
        generic.require_no_duplicates(dup)
        resources = generic._json(path / 'resources.json')
        for k in ('wall_seconds', 'process_cpu_seconds', 'process_maxrss_kib'):
            v = resources.get(k)
            if type(v) not in (float, int) or not np.isfinite(v) or v < 0:
                raise ValueError('Invalid per-attempt CPU resources')
        if resources['wall_seconds'] >= p.MAX_SECONDS or resources['free_bytes'] < p.FLOOR_BYTES:
            raise ValueError('Resource budget violated')
        reports.append(report); rows.add(media['rows']); videos.update(media['videos'])
    ds = root / 'dataset'
    if rows != {str(x.relative_to(ds)) for x in (ds / 'data').rglob('*') if x.is_file()} or len(rows) != p.ADMIT_COUNT:
        raise ValueError('Missing/extra/overlapping row files')
    if videos != {str(x.relative_to(ds)) for x in (ds / 'videos').rglob('*') if x.is_file()} or len(videos) != 2*p.ADMIT_COUNT:
        raise ValueError('Missing/extra/overlapping camera media')
    check_replay_coverage(results); bindings(root, expert, base_source, bank, check)
    if before != generic.fingerprint(root, check):
        raise ValueError('Data changed during final audit')
    decoded = sum(r['media']['decoded_frames'] for r in reports)
    if decoded != p.DECODED_FRAMES:
        raise ValueError('Decoded camera frame total differs')
    if len(set(plan['policy_seeds'])) != 2:
        raise ValueError('Two policy seeds required')
    return dict(status='passed', training_ready=True, automatic_training=False,
                episode_count=p.ADMIT_COUNT, frame_count=p.FRAME_ROWS, decoded_camera_frames=decoded,
                train_ids=list(range(p.TRAIN_LHS_COUNT + p.TRAIN_BASE_COUNT)),
                validation_ids=list(range(p.TRAIN_LHS_COUNT + p.TRAIN_BASE_COUNT, p.ADMIT_COUNT)),
                statistics_episode_ids=list(range(p.TRAIN_LHS_COUNT + p.TRAIN_BASE_COUNT)),
                validation_is_spatial_holdout=False, replay_count=2*p.ADMIT_COUNT,
                frozen_bank_identities=p.UNIQUE_BANK_IDENTITIES,
                untouched_evaluation_layouts=plan['evaluation_layout_ids'],
                policy_seeds=plan['policy_seeds'], provenance=provenance(check),
                duplicates=generic.duplicate_report(sequences), episodes=reports,
                episode_resets={str(i): episode_resets[str(i)] for i in range(p.ADMIT_COUNT)})


def require_broad_workspace_pass(root):
    """Fresh complete media/rows/replay/reset/source audit, zero physics steps."""
    from .record import Budget
    from experiments.fixed_layout.source import Expert as BaseExpert, load_reset
    from .expert import Expert as BroadExpert
    root = Path(root); saved = generic._json(root / 'validation.json'); seal = generic._json(root / 'seal.json')
    with Budget(root) as budget:
        before = generic.fingerprint(root, budget.check)
        if (seal != {'fingerprint': before} or saved.get('status') != 'passed'
                or saved.get('training_ready') is not True):
            raise ValueError('Broad-workspace PASS absent or stale')
        generic.require_no_duplicates(saved['duplicates'])
        if saved.get('provenance') != provenance(budget.check):
            raise ValueError('Source/runtime changed')
        base_expert = BaseExpert(); base_src = load_reset(base_expert); expert = BroadExpert()
        bank = make_bank(base_expert.model, base_src, p.make_plan(), budget.check)
        actual = validate_dataset(root, expert, base_src, bank, budget.check)
        if saved != actual or generic.fingerprint(root, budget.check) != before:
            raise ValueError('Stale full admission')
        return actual
