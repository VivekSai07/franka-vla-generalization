"""Fixed saved cases, exclusive evidence, independently reviewed execution only.

Importing does nothing. execute_reviewed() requires externally supplied review
of exact current code hashes plus matching CPU test and protected-file evidence.
There is no retry/resume or configurable case, root, seed or numerical parameter.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

from .solver import PlanningFailure

ROOT = Path(__file__).resolve().parents[2]
PAIRED = ROOT/'outputs/paired_demonstrations/paired_20260917'
OUT = ROOT/'outputs/diagnostics/bounded_placement_v3_20260917'
IMAGE = 'sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc'
CASE_SEEDS = {4: 488840035, 0: 1232395700, 1: 1762113100, 2: 1121959577, 3: 2349681391}
CASE_HASHES = {
    4: ('4d66200d0e132307a12947cc4afdb44465959c3a5065686870137e6089a24a26',
        'c487a146d5cd373e6fddcf2d00262e1102224015d62309a98f115aafcf35c296'),
    0: ('b9500d265393aa38f7945ee57d9d51217bc28b768c4f84b983b1ade4aac02b9a',
        '4902bd6cd14c3d77ac6bcdf4078ff84d617401d2d1589ae6ef45cb051bc85e97'),
    1: ('08c0e60ba523e2c8a327e2d51a40bde8db7706fa69e64866157092c7633399e1',
        '4902bd6cd14c3d77ac6bcdf4078ff84d617401d2d1589ae6ef45cb051bc85e97'),
    2: ('72c7b3d515a6f85dfbb8a5b37ea8e54eb1fd054164491f5e288fdc8c49e72153',
        'd1dc10274312570421fd9f80d5d56fc1e5405f4f4ca57259654be7365f514c97'),
    3: ('e4d0ad695796dbb3aa587ee88958992f9bd013e8efacbc3f7d85164b9dac5e4b',
        'd1dc10274312570421fd9f80d5d56fc1e5405f4f4ca57259654be7365f514c97'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def serial(value):
    """JSON-only copy: nonfinite markers are evidence, never valid solver inputs."""
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {'nonfinite': 'NaN' if np.isnan(value) else '+Infinity' if value > 0 else '-Infinity'}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(type(value).__name__)


def save_json(path, value):
    text = json.dumps(serial(value), indent=2, allow_nan=False)
    with Path(path).open('x') as stream:
        stream.write(text+'\n')


def saved_cases():
    entries = [json.loads(line) for line in (PAIRED/'attempts.jsonl').read_text().splitlines()]
    if len(entries) != 5 or [e['episode_index'] for e in entries] != list(range(5)):
        raise PlanningFailure('saved_case_inventory')
    cases = []
    for index, seed in CASE_SEEDS.items():
        entry = entries[index]
        if (entry['seed'] != seed or (entry['sha256'], entry['state_sha256']) != CASE_HASHES[index]
                or entry['file'] != f'transcripts/episode_{index:06d}.npz'
                or entry['state_file'] != f'states/layout_{index//2:02d}.npz'):
            raise PlanningFailure('saved_case_identity', index=index)
        if sha(PAIRED/entry['file']) != entry['sha256'] or sha(PAIRED/entry['state_file']) != entry['state_sha256']:
            raise PlanningFailure('saved_case_hash', index=index)
        cases.append(entry)
    return cases


def require_review(out, source_hashes):
    path = Path(out)/'independent_review.json'
    if not path.is_file():
        raise PlanningFailure('independent_review_required')
    review = json.loads(path.read_text())
    if (review.get('approved_for_actual_trials') is not True or review.get('source_hashes') != source_hashes
            or review.get('reviewer_role') != 'independent_top_agent'):
        raise PlanningFailure('independent_review_required')
    return review


def save_failure(out, error, plan, actual_expert_invocations):
    save_json(Path(out)/'failure.json', dict(
        first_failed_gate=getattr(error, 'gate', type(error).__name__),
        details=getattr(error, 'details', {}), message=str(error), plan=plan,
        actual_expert_invocations=actual_expert_invocations, success=False, no_retry=True))


def run_fixed_cases(invoke):
    results = []
    for entry in saved_cases():
        result = invoke(entry)
        results.append(result)
        if not result['success'] or not result['integrity_passed']:
            break
    return results


def preflight():
    from experiments.paired_demonstrations.record import provenance
    from .expert import Expert, EXPERT_VERSION
    from .verification import require_cpu_runtime
    require_cpu_runtime()
    cases = saved_cases()
    pairs = [('red', 'blue'), ('red', 'green'), ('green', 'red'), ('red', 'green'), ('green', 'red')]
    from experiments.paired_demonstrations.plan import instruction
    for entry, pair in zip(cases, pairs):
        if (entry['pick'], entry['place']) != pair or entry['instruction'] != instruction(*pair):
            raise PlanningFailure('saved_case_instruction')
    manifest = json.loads((PAIRED/'manifest.json').read_text())
    if provenance() != manifest['provenance']:
        raise PlanningFailure('original_runtime_or_source_provenance')
    expert = Expert()
    if expert.config != dict(manifest['configuration'], expert_version=EXPERT_VERSION):
        raise PlanningFailure('configuration_contract')
    return cases


def execute_reviewed():
    """Maximum five sequential expert calls; review author must not be implementer."""
    from .artifacts import TraceSink, save_npz
    from .expert import Expert
    from .replay import read_ticks, verify_reference, verify_videos
    from .verification import implementation_hashes, protected_hashes, require_test_evidence
    import mujoco
    from data_collection.benchmark import capture_state, restore_state

    sources = implementation_hashes()
    require_review(OUT, sources)  # Missing review cannot consume any invocation.
    require_test_evidence(OUT, sources)
    baseline = json.loads((OUT/'hashes_before.json').read_text())
    if baseline != protected_hashes():
        raise PlanningFailure('protected_hashes_changed_before_execution')
    # The single block marker also prevents restart after a preflight interruption.
    save_json(OUT/'execution_started.json', dict(case_order=[4, 0, 1, 2, 3], maximum_invocations=5,
                                               source_hashes=sources, image=IMAGE, no_retry=True))
    invoked, completed = 0, []

    def one_case(entry):
        nonlocal invoked
        if protected_hashes() != baseline or implementation_hashes() != sources:
            raise PlanningFailure('between_case_provenance')
        expert = Expert()
        with np.load(PAIRED/entry['state_file'], allow_pickle=False) as arrays:
            initial = {key: arrays[key].copy() for key in arrays.files}
        with np.load(PAIRED/entry['file'], allow_pickle=False) as arrays:
            np.testing.assert_array_equal(initial['initial_state'], arrays['initial_state'])
            np.testing.assert_array_equal(initial['qpos'], arrays['qpos'][0])
        data = mujoco.MjData(expert.model)
        restore_state(expert.model, data, initial['initial_state'])
        np.testing.assert_array_equal(capture_state(expert.model, data), initial['initial_state'])
        for key, value in ((data.qpos, 'qpos'), (data.ctrl, 'initial_ctrl'), (data.xpos, 'body_xpos')):
            np.testing.assert_array_equal(key, initial[value])
        out = OUT/f"episode_{entry['episode_index']:06d}"
        out.mkdir(exist_ok=False)
        save_json(out/'input.json', entry)
        save_npz(out/'initial_state.npz', initial)
        journal, result, sink, failure = {}, None, None, None
        try:
            reference = None
            if entry['episode_index'] == 4:
                with np.load(ROOT/'outputs/diagnostics/pose_snapshot_v2_20260917/corrected_trace.npz',
                             allow_pickle=False) as trace:
                    reference = {key: trace[key][:197].copy() for key in (
                        'qpos', 'ctrl', 'action', 'observation_state', 'phase_index', 'phase_time',
                        'target_pos', 'requested_joint_velocity', 'applied_joint_velocity',
                        'command_saturation', 'sim_time', 'image_sha256', 'substeps')}
            sink = TraceSink(out, entry['instruction'], reference)
            # Marker before the only expert call; even a crash consumes the case.
            save_json(out/'invocation_started.json', dict(seed=entry['seed'], ordinal=invoked+1))
            invoked += 1
            result = expert.run(initial['initial_state'].copy(), entry, sink, journal)
            if not result['success']:
                failure = PlanningFailure(result['reason'], **journal.get('failure', {}).get('details', {}))
            # Preserve native arrays before any JSON/reporting of a failed run.
            save_npz(out/'terminal_state.npz', {key: result[key] for key in ('final_state', 'final_qpos', 'final_ctrl')})
            save_json(out/'result.json', result)
            if failure is not None:
                save_failure(out, failure, journal, invoked)
            sink.finish(); sink = None
            rows = read_ticks(out)
            replay = verify_reference(expert, initial['initial_state'], entry, rows, result, journal)
            videos = verify_videos(out, rows)
            if protected_hashes() != baseline or implementation_hashes() != sources:
                raise PlanningFailure('after_case_provenance')
            save_json(out/'integrity.json', dict(reference=replay, videos=videos, protected_hashes_unchanged=True))
            return dict(episode_index=entry['episode_index'], success=result['success'], integrity_passed=True,
                        reason=result['reason'], final_metrics=result['final_metrics'])
        except Exception as exc:
            # A secondary reporting/replay rejection cannot replace the first
            # expert gate in either the episode or the block failure artifact.
            if failure is not None:
                journal['diagnostic_failure'] = dict(first_failed_gate=getattr(exc, 'gate', type(exc).__name__),
                    details=getattr(exc, 'details', {}), message=str(exc))
            if not (out/'failure.json').exists():
                save_failure(out, failure if failure is not None else exc, journal, invoked)
            if failure is not None:
                raise failure from exc
            raise
        finally:
            save_json(out/'plan_and_events.json', journal)
            if sink is not None:
                sink.finish()

    try:
        preflight()
        completed = run_fixed_cases(one_case)
        return completed
    except Exception as exc:
        save_failure(OUT, exc, {}, invoked)
        raise
    finally:
        after = protected_hashes()
        save_json(OUT/'hashes_after_execution.json', after)
        save_json(OUT/'execution_summary.json', dict(actual_expert_invocations=invoked, completed=completed,
            all_five_passed=len(completed) == 5 and all(r['success'] and r['integrity_passed'] for r in completed),
            protected_hashes_unchanged=baseline == after, no_retry=True, training_ready=False))