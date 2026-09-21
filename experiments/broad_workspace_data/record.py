"""Broad-workspace collector. Single-use; feasibility redraws; per-episode reset."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
from shutil import disk_usage, rmtree
import signal
import sys
from time import monotonic, process_time

import numpy as np
import pinocchio as pin
import mujoco

from experiments.fixed_layout.record import write_json, write_npz, file_hash, record_attempt, TickSink
from experiments.fixed_layout.media import BoundPairedDataset
from experiments.fixed_layout.source import (
    Expert as BaseExpert, load_reset, storage_path,
)
from experiments.paired_demonstrations.media import FEATURES, ENCODING
from src.robots.franka_panda.config import Q_HOME, ACTIVE_JOINTS
from . import plan as p
from .expert import Expert as BroadExpert
from .scenes import make_bank, bank_manifest, provenance, preflight, derive_scene, exact, scene_hash

OUTPUT = p.OUTPUT
FEASIBILITY_REDRAWS = 8
_IK_MAX_ITERS = 200
_IK_TOL_POS = 5e-3
_EXPERT_REJECT_MARKERS = (
    'Failed expert or noncanonical duration; raw attempt preserved',
    'First failed/wrong-duration expert; no retry within admit',
)


class ExpertRejection(RuntimeError):
    """Post-feasibility expert-execution failure; billed to the rejection budget, not a stop."""


def _reset_lerobot_pending_episode(dataset):
    """Discard the LeRobot pending-episode buffer + staging PNGs after a rejected attempt.

    LeRobot's ``add_frame`` appends to ``dataset.episode_buffer`` and writes PNG frames to
    ``dataset/images/<video_key>/episode-{N:06d}/`` keyed by the pending
    ``episode_index`` (== ``meta.total_episodes`` at buffer creation). ``save_episode``
    is what rotates that index; on rejection we never call it, so without this cleanup
    the next admitted attempt reuses the same episode_index and accumulates PNGs on
    top of the rejected ones, breaking the encoder's PNG/raw coverage check.
    """
    buffer = getattr(dataset, 'episode_buffer', None)
    if buffer is not None:
        pending = int(buffer.get('episode_index', getattr(dataset.meta, 'total_episodes', 0)))
    else:
        pending = int(getattr(dataset.meta, 'total_episodes', 0))
    for video_key in getattr(dataset.meta, 'video_keys', ()):
        stage = dataset.root / 'images' / video_key / f'episode-{pending:06d}'
        if stage.is_dir():
            rmtree(stage)
    dataset.episode_buffer = None


class Budget:
    def __init__(self, root):
        self.root, self.started = Path(root), monotonic()

    def check(self):
        if monotonic()-self.started >= p.MAX_SECONDS:
            raise TimeoutError('Broad-workspace twelve-hour deadline')
        if disk_usage(storage_path(self.root)).free < p.FLOOR_BYTES:
            raise ValueError('40 GiB floor required')

    def check_start(self):
        self.check()
        if disk_usage(storage_path(self.root)).free < p.START_BYTES:
            raise ValueError('60 GiB start reserve required')

    def __enter__(self):
        self.check()
        if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
            raise RuntimeError('Existing deadline cannot be replaced')
        self.old_handler = signal.getsignal(signal.SIGALRM)
        def expire(signum, frame): raise TimeoutError('Broad-workspace twelve-hour deadline')
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, max(.001, p.MAX_SECONDS-(monotonic()-self.started)))
        return self

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old_handler)


def claim_output(root):
    root = Path(root)
    if any(x.is_symlink() for x in (root, *root.parents)):
        raise ValueError('Output symlinks forbidden')
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise FileExistsError('No retry/resume/nonempty output')
    else:
        root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'claim.json', dict(version=p.VERSION, output=str(p.OUTPUT.relative_to(p.ROOT)), resume=False))
    if {x.name for x in root.iterdir()} != {'claim.json'}:
        raise FileExistsError('Concurrent writer during exclusive claim')


def frozen_names(plan):
    return {'plan.json', 'provenance.json', 'configuration.json', 'source.npz', 'bank.json'}


def check_snapshots(root, hashes, check=lambda: None):
    from experiments.fixed_layout.validate import _json
    root = Path(root); plan = _json(root / 'plan.json'); p.validate_plan(plan)
    if set(hashes) != frozen_names(plan):
        raise ValueError('Frozen snapshot inventory mismatch')
    for name, digest in hashes.items():
        path = root / name
        if any(x.is_symlink() for x in (path, *path.parents)) or file_hash(path, check) != digest:
            raise ValueError('Frozen source/plan/bank snapshot changed')
    if _json(root / 'provenance.json') != provenance(check):
        raise ValueError('Source/runtime changed')


def _redraw_home(episode_index, k, joint):
    identity = f'homepose:{episode_index}:redraw:{k}:{joint}'
    seeded = int.from_bytes(hashlib.sha256(
        f'{p.VERSION}:{p.MASTER}:{identity}'.encode('ascii')).digest()[:4], 'big')
    return float(np.random.RandomState(seeded).uniform(-p.HOME_JITTER_DEG_MAX, p.HOME_JITTER_DEG_MAX))


def redraw_home_pose_jitter(episode_index, k):
    """Deterministic seven-vector redraw for episode i, attempt k in [0, 7]."""
    if type(k) is not int or not 0 <= k < FEASIBILITY_REDRAWS:
        raise ValueError('Redraw index out of range')
    return [_redraw_home(episode_index, k, j) for j in range(7)]


def _within_joint_limits(model, arm_qpos):
    lower = model.jnt_range[:7, 0]
    upper = model.jnt_range[:7, 1]
    return bool(np.all(np.asarray(arm_qpos, np.float64) >= lower) and
                np.all(np.asarray(arm_qpos, np.float64) <= upper))


def _target_pose(base_pos, z_offset, fixed_rotation):
    return pin.SE3(fixed_rotation, base_pos + np.array([0., 0., z_offset]))


def _ik_converges(ik, robot, q_start, target_se3, q_posture):
    q = np.asarray(q_start, np.float64).copy()
    for _ in range(_IK_MAX_ITERS):
        dq, err_norm, done = ik.compute_velocity(q, target_se3, q_posture=q_posture)
        if done or float(err_norm) < _IK_TOL_POS:
            return True
        q[ACTIVE_JOINTS] = q[ACTIVE_JOINTS] + dq[ACTIVE_JOINTS] * (1./30.)
    return False


def feasible(expert, base_source, layout, jitter) -> bool:
    """Joint-limit + IK smoke check for pick/place targets from jittered home pose."""
    model, robot, ik = expert.model, expert.robot, expert.ik
    home_deg = np.asarray(jitter['home_pose_jitter_deg'], np.float64)
    arm_qpos = base_source['qpos'][:7] + np.deg2rad(home_deg)
    if not _within_joint_limits(model, arm_qpos):
        return False
    q_pin = pin.neutral(robot.model); q_pin[:7] = arm_qpos
    q_home_pin = pin.neutral(robot.model); q_home_pin[:min(len(Q_HOME), robot.model.nq)] = Q_HOME[:min(len(Q_HOME), robot.model.nq)]
    fixed_rotation = robot.forward_kinematics(q_home_pin).rotation.copy()
    xy = layout['cube_xy_m']
    z_source = base_source['qpos'][8]  # cube z index derived below; approximate via body_xpos.
    red_pos = np.array([xy[0], xy[1], base_source['body_xpos'][model.body('target_cube_red').id, 2]])
    green_pos = np.array([xy[2], xy[3], base_source['body_xpos'][model.body('target_cube_green').id, 2]])
    for base_pos, z_off in ((red_pos, .105 + .15), (green_pos, .105 + .2)):
        if not _ik_converges(ik, robot, q_pin, _target_pose(base_pos, z_off, fixed_rotation), q_home_pin):
            return False
    return True


def run_schedule(plan, base_source, model, execute, validate, journal, feasible_fn=None, before=lambda: None,
                 discard=None, completed=()):
    """Every plan slot must be admitted; a failed draw retries the SAME slot.

    Skipping a slot would make the admitted episode index drift away from the plan slot index,
    and the whole downstream contract (metadata[i], attempts/{i:06d}, resets/episode_{i:03d},
    train/validation split boundaries) is defined on that identity. Retrying instead keeps it.
    """
    p.validate_plan(plan)
    frozen = copy.deepcopy(plan)
    layouts = {x['layout_id']: x for x in frozen['layouts']}
    layouts['base'] = frozen['base_layout']
    admitted, rejected, attempts = [], [], 0
    if feasible_fn is None:
        feasible_fn = lambda layout, jitter: True
    if discard is None:
        discard = lambda index, k: None
    for slot in frozen['episodes']:
        i = slot['episode_index']
        if i in completed:  # already admitted by an earlier run of the same output
            admitted.append(i)
            continue
        try:
            before()
            layout = layouts[slot['layout_id']]
            draws = [dict(k=0, source='plan', home_pose_jitter_deg=list(slot['home_pose_jitter_deg']))]
            for k in range(1, FEASIBILITY_REDRAWS + 1):
                draws.append(dict(k=k, source=f'redraw:{k-1}',
                                  home_pose_jitter_deg=redraw_home_pose_jitter(i, k-1)))
            seated = False
            for candidate in draws:
                attempts += 1
                if attempts > p.ATTEMPT_LIMIT:
                    raise RuntimeError('Attempt budget exhausted before admission target')
                jitter = dict(home_pose_jitter_deg=candidate['home_pose_jitter_deg'])
                if not feasible_fn(layout, jitter):
                    continue
                attempt = dict(slot, home_pose_jitter_deg=candidate['home_pose_jitter_deg'],
                               feasibility_k=candidate['k'], feasibility_source=candidate['source'])
                arrays = derive_scene(model, base_source, layout,
                                      dict(home_pose_jitter_deg=candidate['home_pose_jitter_deg']))
                journal(dict(event='started', **attempt, training_ready=False))
                try:
                    outcome = execute(copy.deepcopy(attempt), arrays)
                    if (outcome.get('success') is not True or type(outcome.get('frames')) is not int
                            or outcome['frames'] != p.EXPECTED_FRAMES):
                        raise ExpertRejection(_EXPERT_REJECT_MARKERS[1])
                except ExpertRejection as exc:
                    rejected.append(i)
                    if len(rejected) > p.REJECTION_BUDGET:
                        raise RuntimeError('Rejection budget exhausted; no more attempts') from exc
                    try:
                        journal(dict(event='rejected', episode_index=i, layout_id=slot['layout_id'],
                                     tried=candidate['k']+1, reason='expert_execution',
                                     training_ready=False, error=repr(exc)))
                    except BaseException as secondary:
                        print(f'Journal failure: {secondary!r}', file=sys.stderr)
                    discard(i, candidate['k'])
                    continue
                validate(copy.deepcopy(attempt), outcome)
                before()
                journal(dict(event='admitted', episode_index=i, training_ready=False))
                admitted.append(i)
                seated = True
                break
            if not seated:
                raise RuntimeError(f'Slot {i} exhausted every feasible draw without an admitted demonstration')
        except BaseException as exc:
            try:
                journal(dict(event='failed', episode_index=i, training_ready=False, error=repr(exc)))
            except BaseException as secondary:
                print(f'Journal failure: {secondary!r}', file=sys.stderr)
            raise
    return admitted, rejected


def create_dataset(output, check):
    dataset = BoundPairedDataset.create(repo_id='broad_workspace_data', root=Path(output)/'dataset', fps=30,
        features=copy.deepcopy(FEATURES), vcodec=ENCODING['vcodec'],
        encoder_threads=ENCODING['encoder_threads'], metadata_buffer_size=1)
    dataset.meta.info['data_files_size_in_mb'] = 0
    dataset.meta.info['video_files_size_in_mb'] = 0
    dataset.fixed_layout_check = check
    return dataset


def open_dataset(output, check):
    """Reopen a finalized dataset for appending; __init__ drops the pinned encoder settings."""
    dataset = BoundPairedDataset(repo_id='broad_workspace_data', root=Path(output)/'dataset',
                                 video_backend='pyav')
    dataset.vcodec = ENCODING['vcodec']
    dataset._encoder_threads = ENCODING['encoder_threads']
    dataset.batch_encoding_size = 1
    dataset._streaming_encoder = None
    dataset.meta.info['data_files_size_in_mb'] = 0
    dataset.meta.info['video_files_size_in_mb'] = 0
    dataset.fixed_layout_check = check
    dataset._active_attempt_path = None
    return dataset


def _replace_json(path, value):
    """Amendment writer; every other evidence file stays write-once."""
    path = Path(path)
    temporary = path.with_name(path.name + '.amend')
    if temporary.exists():
        temporary.unlink()
    write_json(temporary, value)
    temporary.replace(path)


def resume_state(output):
    """Contiguous admitted prefix and journal clock left by the previous run."""
    output = Path(output)
    events = [json.loads(line) for line in (output/'journal.jsonl').read_text().splitlines()]
    admitted = [e['episode_index'] for e in events if e['event'] == 'admitted']
    if admitted != list(range(len(admitted))):
        raise ValueError('Previous journal did not admit a contiguous slot prefix')
    directories = sorted(x.name for x in (output/'attempts').iterdir())
    if directories != [f'{i:06d}' for i in admitted]:
        raise ValueError('Attempt directories do not match the admitted prefix')
    rejected = [e['episode_index'] for e in events if e['event'] == 'rejected']
    return admitted, rejected, max((e['elapsed_seconds'] for e in events), default=0.)


def amend_frozen_plan(output, plan, bank, bound, expert, check):
    """Rewrite the frozen plan/bank/provenance/snapshot set after a recorded layout substitution.

    Only layouts listed in INFEASIBLE_LAYOUT_RESAMPLES may move, no episode slot may change, and
    no already-collected slot may depend on a moved layout, so demonstrations recorded under the
    previous plan stay valid. The superseded provenance is preserved in amendment.json because
    substituting a layout necessarily edits the collector source.
    """
    output = Path(output)
    previous = json.loads((output/'plan.json').read_text())
    previous_bound = json.loads((output/'provenance.json').read_text())
    if previous == plan and previous_bound == bound:
        return json.loads((output/'snapshots.json').read_text())
    if previous['episodes'] != plan['episodes'] or previous['base_layout'] != plan['base_layout']:
        raise ValueError('Amendment may not change any episode slot')
    if json.loads((output/'configuration.json').read_text()) != expert.config:
        raise ValueError('Amendment may not change the expert configuration')
    old = {x['layout_id']: x for x in previous['layouts']}
    new = {x['layout_id']: x for x in plan['layouts']}
    moved = {k for k in old if old[k] != new.get(k)}
    if moved - set(p.INFEASIBLE_LAYOUT_RESAMPLES):
        raise ValueError(f'Unrecorded layout substitution: {sorted(moved - set(p.INFEASIBLE_LAYOUT_RESAMPLES))}')
    for directory in sorted((output/'attempts').iterdir()):
        if json.loads((directory/'slot.json').read_text())['layout_id'] in moved:
            raise ValueError(f'Attempt {directory.name} was collected under a substituted layout')
    history = json.loads((output/'amendment.json').read_text()) if (output/'amendment.json').exists() else []
    history.append(dict(substituted_layouts=sorted(moved),
                        resample_indices={k: p.INFEASIBLE_LAYOUT_RESAMPLES[k] for k in sorted(moved)},
                        previous_cube_xy_m={k: old[k]['cube_xy_m'] for k in sorted(moved)},
                        superseded_provenance=previous_bound,
                        admitted_before_amendment=len(list((output/'attempts').iterdir()))))
    _replace_json(output/'amendment.json', history)
    _replace_json(output/'plan.json', plan)
    _replace_json(output/'bank.json', bank_manifest(plan, bank))
    _replace_json(output/'provenance.json', bound)
    hashes = {name: file_hash(output/name, check) for name in sorted(frozen_names(plan))}
    _replace_json(output/'snapshots.json', hashes)
    return hashes


def collect(resume=False):
    """Main-only real collection after independent review; no arbitrary overrides."""
    from .validate import validate_episode, validate_dataset
    from experiments.fixed_layout.validate import fingerprint
    output = OUTPUT; budget = Budget(output)
    if any(x.is_symlink() for x in (output, *output.parents)):
        raise ValueError('Output symlinks forbidden')
    if resume:
        # The start reserve sizes a full 200-slot run; a resume only writes its remaining slots.
        budget.check()
        completed, previous_rejections, journal_offset = resume_state(output)
        if not completed:
            raise ValueError('Nothing to resume')
    else:
        budget.check_start()
        completed, previous_rejections, journal_offset = [], [], 0.
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError('No retry/resume')
    with budget:
        plan = p.make_plan(); bound = provenance(budget.check)
        base_expert = BaseExpert(); base_src = load_reset(base_expert)
        model = base_expert.model
        bank = make_bank(model, base_src, plan, budget.check)
        if provenance(budget.check) != bound:
            raise ValueError('Source changed during scene freeze')
        expert = BroadExpert()
        if not resume:
            budget.check_start(); claim_output(output)
        dataset = None; active = None; started = {}; accepted = []; episode_resets = {}
        rejections = list(previous_rejections)
        try:
            if resume:
                (output/'failure.json').unlink(missing_ok=True)
                hashes = amend_frozen_plan(output, plan, bank, bound, expert, budget.check)
                for index in completed:
                    episode_resets[index] = file_hash(output/'resets'/f'episode_{index:03d}.npz', budget.check)
            else:
                budget.check_start()
                write_json(output / 'plan.json', plan)
                write_json(output / 'provenance.json', bound)
                write_json(output / 'configuration.json', expert.config)
                write_json(output / 'bank.json', bank_manifest(plan, bank))
                with (output / 'source.npz').open('xb') as stream:
                    stream.write((p.ROOT / p.SOURCE).read_bytes()); stream.flush(); os.fsync(stream.fileno())
                (output / 'resets').mkdir()
                hashes = {name: file_hash(output / name, budget.check) for name in sorted(frozen_names(plan))}
                write_json(output / 'snapshots.json', hashes)
                (output / 'attempts').mkdir()

            def before():
                budget.check(); check_snapshots(output, hashes, budget.check)
                fresh = load_reset(base_expert)
                if expert.config != json.loads((output / 'configuration.json').read_text()) or any(
                        not exact(base_src[k], fresh[k]) for k in base_src):
                    raise ValueError('Live expert/source reset changed')
                budget.check()

            before()
            dataset = open_dataset(output, budget.check) if resume else create_dataset(output, budget.check)

            def journal(event):
                nonlocal active
                if event['event'] == 'started': active = event['episode_index']
                if event['event'] == 'admitted': accepted.append(event['episode_index'])
                if event['event'] == 'rejected': rejections.append(event['episode_index'])
                with (output / 'journal.jsonl').open('a') as stream:
                    stream.write(json.dumps({**event, 'elapsed_seconds': journal_offset + monotonic()-budget.started}, allow_nan=False)+'\n')
                    stream.flush(); os.fsync(stream.fileno())

            def feasible_fn(layout, jitter):
                return feasible(expert, base_src, layout, jitter)

            layouts_by_id = {x['layout_id']: x for x in plan['layouts']}
            layouts_by_id['base'] = plan['base_layout']

            def execute(slot, arrays):
                i = slot['episode_index']; path = output / 'attempts' / f'{i:06d}'
                path.mkdir(exist_ok=False); write_json(path / 'slot.json', slot)
                reset_path = output / 'resets' / f'episode_{i:03d}.npz'
                write_npz(reset_path, dict(initial_state=arrays['initial_state'], qpos=arrays['qpos']))
                episode_resets[i] = file_hash(reset_path, budget.check)
                started[i] = (monotonic(), process_time())
                try:
                    result = record_attempt(path, expert, dataset, slot, arrays['initial_state'].copy(), budget.check)
                except RuntimeError as exc:
                    if str(exc) in _EXPERT_REJECT_MARKERS:
                        _reset_lerobot_pending_episode(dataset)
                        raise ExpertRejection(str(exc)) from exc
                    raise
                dataset._writer_closed_for_reading = True
                return result

            def validate(slot, outcome):
                before()
                i = slot['episode_index']
                # A feasibility redraw or a retry changes the home pose, so the episode must be
                # checked against the reset it actually ran, not the plan-derived bank entry.
                layout = layouts_by_id[slot['layout_id']]
                reset = derive_scene(model, base_src, layout,
                                     dict(home_pose_jitter_deg=slot['home_pose_jitter_deg']))
                validate_episode(output, slot, expert, reset, budget.check)
                wall, cpu = started[i]
                usage = resource.getrusage(resource.RUSAGE_SELF)
                write_json(output / 'attempts' / f'{i:06d}' / 'resources.json', dict(
                    wall_seconds=monotonic()-wall, process_cpu_seconds=process_time()-cpu,
                    process_maxrss_kib=usage.ru_maxrss, process_user_seconds=usage.ru_utime,
                    process_system_seconds=usage.ru_stime, logical_cpus=os.cpu_count(),
                    affinity_cpus=sorted(os.sched_getaffinity(0)), free_bytes=disk_usage(output).free,
                    scope='CPU process cumulative high-water; no GPU attribution'))
                before()

            def discard(index, k):
                """Retire a rejected attempt so the same slot can be retried on the next draw."""
                _reset_lerobot_pending_episode(dataset)
                (output / 'rejected').mkdir(exist_ok=True)
                path = output / 'attempts' / f'{index:06d}'
                if path.is_dir():
                    path.rename(output / 'rejected' / f'{index:06d}_k{k}')
                reset_path = output / 'resets' / f'episode_{index:03d}.npz'
                if reset_path.exists():
                    reset_path.unlink()
                episode_resets.pop(index, None)
                started.pop(index, None)

            run_schedule(plan, base_src, model, execute, validate, journal, feasible_fn, before, discard,
                         completed=frozenset(completed))
            dataset.finalize(); dataset = None; before()
            info_path = output / 'dataset/meta/info.json'; info = json.loads(info_path.read_text())
            info['splits'] = {'train': f'0:{p.TRAIN_LHS_COUNT + p.TRAIN_BASE_COUNT}',
                              'validation': f'{p.TRAIN_LHS_COUNT + p.TRAIN_BASE_COUNT}:{p.ADMIT_COUNT}'}
            info['episode_resets'] = {str(i): episode_resets[i] for i in sorted(episode_resets)}
            # Retried slots run an approved redraw, so record what each episode actually used.
            executed = {i: json.loads((output / 'attempts' / f'{i:06d}' / 'slot.json').read_text())
                        for i in sorted(episode_resets)}
            info['home_pose_jitter_deg'] = {str(i): executed[i]['home_pose_jitter_deg'] for i in sorted(episode_resets)}
            info['hover_z_jitter_m'] = {str(i): executed[i]['hover_z_jitter_m'] for i in sorted(episode_resets)}
            temporary = info_path.with_suffix('.broad-workspace.json'); write_json(temporary, info); temporary.replace(info_path)
            report = validate_dataset(output, expert, base_src, bank, budget.check)
            before(); write_json(output / 'validation.json', report)
            write_json(output / 'seal.json', {'fingerprint': fingerprint(output, budget.check)})
            budget.check()
            return report
        except BaseException as exc:
            try:
                attempts = sorted(int(x.name) for x in (output / 'attempts').iterdir()) if (output / 'attempts').exists() else []
                wall, cpu = started.get(active, (monotonic(), process_time()))
                write_json(output / 'failure.json', dict(
                    training_ready=False, error=repr(exc), active_episode=active, admitted_ids=accepted,
                    rejected_ids=rejections, attempted_ids=attempts,
                    missing_ids=[i for i in range(p.ADMIT_COUNT) if i not in attempts],
                    elapsed_seconds=monotonic()-budget.started, active_wall_seconds=monotonic()-wall,
                    active_process_cpu_seconds=process_time()-cpu,
                    process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            except BaseException as secondary:
                print(f'Failure evidence unavailable: {secondary!r}', file=sys.stderr)
            raise
        finally:
            if dataset is not None:
                try: dataset.finalize()
                except BaseException as secondary:
                    print(f'Cleanup failed; primary preserved: {secondary!r}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--preflight', action='store_true')
    choice.add_argument('--collect', action='store_true')
    choice.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = preflight() if args.preflight else collect(resume=args.resume)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
