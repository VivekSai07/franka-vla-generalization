"""Broad-workspace dataset admission; per-episode reset SHAs and 400 replay coverage."""
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from experiments.fixed_layout import validate as generic
from experiments.fixed_layout.record import write_json, file_hash
from . import plan as p
from .scenes import make_bank, bank_manifest, provenance, exact


def metadata(root, count, final=False):
    if type(count) is not int or not 1 <= count <= p.ADMIT_COUNT:
        raise ValueError('Invalid admitted-demonstration count')
    info, entries = generic._metadata(root, count, final=False)
    if final:
        wanted = {'train': f'0:{p.TRAIN_LHS_COUNT + p.TRAIN_BASE_COUNT}',
                  'validation': f'{p.TRAIN_LHS_COUNT + p.TRAIN_BASE_COUNT}:{p.ADMIT_COUNT}'}
        if count != p.ADMIT_COUNT or info.get('splits') != wanted:
            raise ValueError('Expected 160 training and 40 validation demonstrations')
    return info, entries


def check_replay_coverage(results):
    expected = Counter((i, m) for i in range(p.ADMIT_COUNT) for m in generic.MODES)
    if (any(type(r.get('episode')) is not int or type(r.get('mode')) is not str for r in results)
            or Counter((r['episode'], r['mode']) for r in results) != expected
            or any(r.get('passed') is not True for r in results)):
        raise ValueError(f'Require exactly {p.ADMIT_COUNT} x 2 unique saved-control replays')


def validate_journal(events, plan):
    p.validate_plan(plan)
    started = [(e['event'], e.get('episode_index')) for e in events if e['event'] == 'started']
    admitted = [(e['event'], e.get('episode_index')) for e in events if e['event'] == 'admitted']
    if [e[1] for e in started] != list(range(p.ADMIT_COUNT)) or [e[1] for e in admitted] != list(range(p.ADMIT_COUNT)):
        raise ValueError('Journal must record ordered started/admitted for all admits')
    if sum(e['event'] == 'rejected' for e in events) > p.REJECTION_BUDGET:
        raise ValueError('Rejection budget exceeded in journal')
    times = [e['elapsed_seconds'] for e in events]
    if (any(type(t) not in (int, float) or not np.isfinite(t) or not 0 <= t < p.MAX_SECONDS for t in times)
            or times != sorted(times)):
        raise ValueError('Invalid twelve-hour elapsed evidence')
    if any(e.get('training_ready') is not False for e in events):
        raise ValueError('Premature training admission in journal')
    for event, slot in zip((e for e in events if e['event'] == 'started'), plan['episodes']):
        for key in ('episode_index', 'seed', 'layout_id', 'split', 'pick', 'place',
                    'colors', 'instruction', 'task_index'):
            if event.get(key) != slot[key]:
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
                'colors', 'instruction', 'task_index', 'hover_z_jitter_m', 'hover_z_offset'):
        if episode.get(key) != slot[key]:
            return False
    return True


def validate_episode(root, episode, expert, reset, check):
    if not _matches_plan_slot(episode):
        raise ValueError('Unplanned demonstration slot; evaluation execution forbidden')
    i = episode['episode_index']; info, entries = metadata(root, i+1)
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
    if not required <= actual or not actual <= required | {'validation.json', 'seal.json'}:
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
    for slot in plan['episodes']:
        check(); i = slot['episode_index']; key = f'episode_{i:03d}'; reset = bank[key]
        actions, states, trace, gates, media = generic._episode_inputs(root, slot, expert, reset, check, info, entries)
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
