"""Own full admission; generic fixed-layout gates receive EACH planned reset.

Only validate_episode spends two saved-control replay tokens. Fresh admission
audits their durable evidence, never regenerates an expert or advances physics.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.fixed_layout import validate as generic
from experiments.fixed_layout.record import write_json, file_hash
from experiments.fixed_layout.source import Expert, load_reset
from . import plan as p
from .scenes import make_bank, bank_manifest, provenance, exact


def metadata(root, count, final=False):
    if type(count) is not int or not 1 <= count <= 60: raise ValueError('Invalid demonstration count')
    # This helper is count-parametric unless final=True (old hardcoded 20/4).
    info, entries = generic._metadata(root, count, final=False)
    if final and (count != 60 or info['splits'] != {'train': '0:48', 'validation': '48:60'}):
        raise ValueError('Exactly 48 training and 12 validation demonstrations required')
    return info, entries


def check_replay_coverage(results):
    if (any(type(r.get('episode')) is not int or type(r.get('mode')) is not str for r in results)
            or Counter((r['episode'], r['mode']) for r in results) != Counter((i, m) for i in range(60) for m in generic.MODES)
            or any(r.get('passed') is not True for r in results)):
        raise ValueError('Require exactly 60 x 2 unique saved-control replays, no evaluation experts')


def validate_journal(events, plan):
    p.validate_plan(plan); expected = []
    for i in range(60):
        expected.extend([('started', i), ('admitted', i)])
        if i == 11: expected.append(('preparation_pass', None))
    if [(e['event'], e.get('episode_index')) for e in events] != expected:
        raise ValueError('Require ordered 60 successes and first-12 preparation before attempt 13')
    times = [e['elapsed_seconds'] for e in events]
    if any(type(t) not in (int, float) or not np.isfinite(t) or not 0 <= t < p.MAX_SECONDS for t in times) or times != sorted(times):
        raise ValueError('Invalid six-hour elapsed evidence')
    for event, slot in zip((e for e in events if e['event'] == 'started'), plan['episodes']):
        if not p._same({key: event.get(key) for key in slot}, slot): raise ValueError('Journal seed/reset/split changed')
    if any(e.get('training_ready') is not False for e in events): raise ValueError('Premature training admission')
    prep = [e for e in events if e['event'] == 'preparation_pass'][0]
    if not p._same(prep.get('episode_ids'), list(range(12))): raise ValueError('Wrong preparation gate')


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
    releases = [dict(tick=int(i), pick_xyz=post['pick_xyz'][i].tolist(), place_xyz=post['place_xyz'][i].tolist(),
                    delta_xyz=delta[i].tolist()) for i in opened]
    return dict(phases=phases, releases=releases, final_delta_xyz=delta[-1].tolist(),
                final_xy_m=float(np.linalg.norm(delta[-1, :2])),
                saturation_per_joint=trace['command_saturation'].sum(axis=0).astype(int).tolist(),
                initial_pick_xyz=reset['body_xpos'][expert.model.body('target_cube_red').id].tolist(),
                initial_place_xyz=reset['body_xpos'][expert.model.body('target_cube_green').id].tolist())


def episode_report(i, gates, media, replays, detail):
    return dict(episode=i, passed=True, original_gates=gates, media=media, replays=replays, diagnostics=detail,
        original_contacts='geometric_reconstruction_from_saved_post_qpos_confirmed_by_reference_replay')


def validate_episode(root, episode, expert, reset, check):
    if not any(p._same(episode, slot) for slot in p.make_plan()['episodes']):
        raise ValueError('Unplanned demonstration slot; evaluation execution forbidden')
    i = episode['episode_index']; info, entries = metadata(root, i+1)
    actions, states, trace, gates, media = generic._episode_inputs(root, episode, expert, reset, check, info, entries)
    path = Path(root) / 'attempts' / f'{i:06d}'; (path / 'replays').mkdir(exist_ok=False)
    results = [generic.replay_controls(path / 'replays', expert, reset, trace, actions, mode, check) for mode in generic.MODES]
    detail = diagnostics(expert, trace, generic._npz(path / 'terminal.npz'), actions, reset, check)
    report = episode_report(i, gates, media, results, detail); write_json(path / 'validation.json', report)
    sequences = []
    for ep in entries:
        row = generic._child(Path(root)/'dataset', info['data_path'].format(chunk_index=ep['data/chunk_index'], file_index=ep['data/file_index']))
        a, s, _ = generic.read_rows(row, ep['episode_index']); sequences.append((a, s))
    dup = generic.duplicate_report(sequences); write_json(path / 'duplicates.json', dup); generic.require_no_duplicates(dup)
    check()
    return report


def bindings(root, expert, source, bank, check):
    from .record import check_snapshots
    root = Path(root); check()
    if (root / 'failure.json').exists(): raise ValueError('Failed collection cannot admit training')
    plan = generic._json(root / 'plan.json'); p.validate_plan(plan)
    check_snapshots(root, generic._json(root / 'snapshots.json'), check)
    if file_hash(root / 'source.npz', check) != p.SOURCE_SHA256 or generic._json(root / 'configuration.json') != expert.config:
        raise ValueError('Source/configuration changed')
    # Regenerate initializations, not trajectories, to reject a coherently rehashed wrong bank.
    actual = make_bank(expert.model, source, plan, check)
    if set(bank) != set(actual): raise ValueError('Incomplete live bank')
    if generic._json(root / 'bank.json') != bank_manifest(plan, actual): raise ValueError('Wrong frozen scene manifest')
    if sorted(x.name for x in (root / 'scenes').iterdir()) != sorted(f'{k}.npz' for k in actual):
        raise ValueError('Missing/extra frozen layouts')
    for name, expected in actual.items():
        check(); saved = generic._npz(root / 'scenes' / f'{name}.npz')
        for candidate in (bank[name], saved):
            if set(candidate) != set(expected) or any(not exact(candidate[k], expected[k]) for k in expected):
                raise ValueError('Wrong per-layout full reset, including untouched evaluation scenes')
    return plan


def validate_inventory(root):
    root = Path(root)
    required = {'claim.json', 'plan.json', 'provenance.json', 'configuration.json', 'source.npz',
                'bank.json', 'snapshots.json', 'journal.jsonl', 'scenes', 'attempts', 'dataset'}
    actual = {x.name for x in root.iterdir()}
    if not required <= actual or not actual <= required | {'validation.json', 'seal.json'}:
        raise ValueError('Missing/extra/failed final root evidence')
    claim = dict(version=p.VERSION, output=str(p.OUTPUT.relative_to(p.ROOT)), resume=False)
    if not p._same(generic._json(root / 'claim.json'), claim): raise ValueError('Exclusive output claim changed')


def validate_dataset(root, expert, source, bank, check):
    root = Path(root); validate_inventory(root)
    plan = bindings(root, expert, source, bank, check)
    before = generic.fingerprint(root, check)
    events = [json.loads(line) for line in (root / 'journal.jsonl').read_text().splitlines()]
    validate_journal(events, plan)
    if sorted(x.name for x in (root / 'attempts').iterdir()) != [f'{i:06d}' for i in range(60)]:
        raise ValueError('Exactly sixty attempts required; no evaluation execution')
    info, entries = metadata(root, 60, final=True)
    reports, results, sequences, rows, videos = [], [], [], set(), set()
    for slot in plan['episodes']:
        check(); i = slot['episode_index']; reset = bank[slot['layout_id']]
        actions, states, trace, gates, media = generic._episode_inputs(root, slot, expert, reset, check, info, entries)
        path = root / 'attempts' / f'{i:06d}'
        if {x.name for x in path.iterdir()} != {'slot.json', 'outcome.json', 'terminal.npz', 'ticks', 'media', 'replays', 'validation.json', 'duplicates.json', 'resources.json'}:
            raise ValueError('Missing/extra/failed attempt evidence')
        if {x.name for x in (path / 'media').iterdir()} != {f'{key}.json' for key in generic.CAMERAS}:
            raise ValueError('Missing/extra camera media bindings')
        if sorted(x.name for x in (path / 'replays').iterdir()) != sorted(generic.MODES): raise ValueError('Replay mode inventory differs')
        replays = []
        for mode in generic.MODES:
            directory = path / 'replays' / mode
            if {x.name for x in directory.iterdir()} != {'result.json'} | {f'{j:06d}.npz' for j in range(528)}:
                raise ValueError('Missing/extra replay evidence')
            result = generic.audit_replay(directory, expert, reset, trace, actions, mode, check)
            if generic._json(directory / 'result.json') != result: raise ValueError('Replay report changed')
            replays.append(result); results.append(dict(episode=i, **result))
        detail = diagnostics(expert, trace, generic._npz(path / 'terminal.npz'), actions, reset, check)
        report = episode_report(i, gates, media, replays, detail)
        if generic._json(path / 'validation.json') != report: raise ValueError('Episode report changed')
        sequences.append((actions, states)); dup = generic.duplicate_report(sequences)
        if generic._json(path / 'duplicates.json') != dup: raise ValueError('Duplicate report changed')
        generic.require_no_duplicates(dup)
        resources = generic._json(path / 'resources.json')
        for key in ('wall_seconds', 'process_cpu_seconds', 'process_maxrss_kib'):
            if type(resources.get(key)) not in (float, int) or not np.isfinite(resources[key]) or resources[key] < 0:
                raise ValueError('Invalid per-attempt CPU resources')
        if resources['wall_seconds'] >= p.MAX_SECONDS or resources['free_bytes'] < p.FLOOR_BYTES: raise ValueError('Resource budget violated')
        reports.append(report); rows.add(media['rows']); videos.update(media['videos'])
    ds = root / 'dataset'
    if rows != {str(x.relative_to(ds)) for x in (ds / 'data').rglob('*') if x.is_file()} or len(rows) != 60:
        raise ValueError('Missing/extra/overlapping row files')
    if videos != {str(x.relative_to(ds)) for x in (ds / 'videos').rglob('*') if x.is_file()} or len(videos) != 120:
        raise ValueError('Missing/extra/overlapping camera media')
    check_replay_coverage(results); bindings(root, expert, source, bank, check)
    if before != generic.fingerprint(root, check): raise ValueError('Data changed during final audit')
    decoded = sum(report['media']['decoded_frames'] for report in reports)
    if decoded != 63360: raise ValueError('Decoded camera frame total differs')
    return dict(status='passed', training_ready=True, automatic_training=False, episode_count=60,
                frame_count=31680, decoded_camera_frames=decoded, train_ids=list(range(48)),
                validation_ids=list(range(48, 60)), statistics_episode_ids=list(range(48)),
                validation_is_spatial_holdout=False, replay_count=120, frozen_layouts=77,
                untouched_evaluation_layouts=plan['evaluation_layout_ids'], policy_seeds=plan['policy_seeds'],
                provenance=provenance(check), duplicates=generic.duplicate_report(sequences), episodes=reports)


def require_local_varied_pass(root):
    """Fresh complete media/rows/replay/state/source audit, zero physics steps."""
    from .record import Budget
    root = Path(root); saved = generic._json(root / 'validation.json'); seal = generic._json(root / 'seal.json')
    with Budget(root) as budget:
        before = generic.fingerprint(root, budget.check)
        if seal != {'fingerprint': before} or saved.get('status') != 'passed' or saved.get('training_ready') is not True:
            raise ValueError('Local-varied PASS absent or stale')
        generic.require_no_duplicates(saved['duplicates'])
        if saved.get('provenance') != provenance(budget.check): raise ValueError('Source/runtime changed')
        expert = Expert(); source = load_reset(expert); bank = make_bank(expert.model, source, p.make_plan(), budget.check)
        actual = validate_dataset(root, expert, source, bank, budget.check)
        if saved != actual or generic.fingerprint(root, budget.check) != before: raise ValueError('Stale full admission')
        return actual