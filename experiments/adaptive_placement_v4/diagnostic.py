"""One saved-lift numerical plan. No expert run, renderer or physics stepping.

    python -m experiments.adaptive_placement_v4.diagnostic --tests
    python -m experiments.adaptive_placement_v4.diagnostic --run

The second command consumes the exclusive numerical slot, even on failure.
No physics entry point or authorization mechanism is implemented here.
"""
import argparse
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import pinocchio as pin

from data_collection.benchmark import capture_state, restore_state
from experiments.bounded_placement_v3.collision import CollisionChecker, body_pose
from experiments.bounded_placement_v3.diagnostic import ROOT, IMAGE, save_json, sha, saved_cases
from experiments.bounded_placement_v3.gates import validate_release_geometry
from experiments.bounded_placement_v3.path import pose_waypoints
from experiments.bounded_placement_v3.prefix import placement_endpoints
from experiments.bounded_placement_v3.solver import PlanningFailure, finite
from experiments.bounded_placement_v3.verification import implementation_hashes, require_cpu_runtime
from experiments.paired_demonstrations.expert_v1 import configuration, generate_task_states
from src.robots.franka_panda.config import SCENE_PATH, Q_HOME
from src.robots.franka_panda.robot import FrankaPanda
from .planner import plan_route

PACKAGE = ROOT/'experiments/adaptive_placement_v4'
OUT = ROOT/'outputs/diagnostics/adaptive_placement_v4_20260917'
V3 = ROOT/'outputs/diagnostics/bounded_placement_v3_20260917'
CASE = V3/'episode_000004'
ROBOT_BODIES = [f'link{i}' for i in range(8)]+['hand', 'left_finger', 'right_finger']
GRIPPER_BODIES = ['hand', 'left_finger', 'right_finger']


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(PACKAGE.rglob('*.py'))}


def check_preservation(baseline):
    changed = [p for p, digest in baseline.items() if not (ROOT/p).is_file() or sha(ROOT/p) != digest]
    if changed:
        raise PlanningFailure('protected_files_changed', paths=changed)
    return dict(files=len(baseline), unchanged=True)


def exact(actual, expected, field):
    if not np.array_equal(actual, expected):
        raise PlanningFailure('saved_lift_reconstruction_mismatch', field=field)


def reconstruct():
    """Reconstruct derived geometry/RNG only; never run prefix or solve a path."""
    bound = json.loads((V3/'tests_passed_reviewed.json').read_text())
    if implementation_hashes() != bound['source_hashes']:
        raise PlanningFailure('v3_hash_binding_changed')
    entry = json.loads((CASE/'input.json').read_text())
    if entry != saved_cases()[0] or entry['seed'] != 488840035:
        raise PlanningFailure('saved_episode_identity')
    saved = json.loads((CASE/'plan_and_events.json').read_text())
    failure = json.loads((CASE/'failure.json').read_text())
    with np.load(CASE/'terminal_state.npz', allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    finite(*arrays.values())
    if (saved['prefix_ticks'] != 197 or saved['frames'] != 197 or saved['intentional_release']
            or saved['failure']['first_failed_gate'] != 'waypoint_branch_jump'
            or failure['first_failed_gate'] != 'waypoint_branch_jump'):
        raise PlanningFailure('original_stop_not_lift_branch_rejection')
    exact(arrays['final_state'], saved['lift_state'], 'lift_state')
    exact(arrays['final_state'], saved['terminal_state'], 'terminal_state')
    exact(arrays['final_ctrl'], saved['terminal_ctrl'], 'terminal_ctrl')
    model, robot = mujoco.MjModel.from_xml_path(SCENE_PATH), FrankaPanda()
    config = configuration(model, robot)  # Configures timing only; no settling.
    bounds = np.asarray(config['expert_command_contract']['joint_bounds'])
    data = mujoco.MjData(model)
    restore_state(model, data, arrays['final_state'])
    exact(capture_state(model, data), arrays['final_state'], 'restored_state')
    exact(data.qpos, arrays['final_qpos'], 'measured_qpos')
    exact(data.ctrl, arrays['final_ctrl'], 'previous_ctrl')
    measured_q, previous_ctrl = data.qpos[:robot.model.nq].copy(), data.ctrl[:7].copy()
    exact(measured_q, saved['placement']['start_q'], 'start_q')
    exact(previous_ctrl, saved['placement']['previous_target'], 'previous_target')
    hand = body_pose(model, data, 'hand').copy()
    start = robot.forward_kinematics(measured_q).copy()
    exact(hand.homogeneous, saved['lift_pose'], 'hand_pose')
    exact(start.homogeneous, saved['placement']['start_pose'], 'pinocchio_start_pose')
    # Both frames must individually match their saved v3 values exactly. The
    # original models have a small cross-engine rotation difference (~3.49e-8
    # rad); do not add an invented cross-engine tolerance or change either frame.
    hand_frame_error = float(np.max(np.abs(hand.homogeneous-start.homogeneous)))
    checker = CollisionChecker(model, data, 'hand', f"target_cube_{entry['pick']}",
        f"target_cube_{entry['place']}", ROBOT_BODIES, GRIPPER_BODIES, range(7))
    exact(checker.relative.homogeneous, saved['payload_hand_relative'], 'payload_hand_relative')
    for key in ('contype', 'conaffinity', 'exclude_signature'):
        exact(checker.filters[key], saved['filters'][key], 'collision_'+key)

    # Reproduce ONLY the original RNG consumption, not the prefix motion.
    states = generate_task_states(.105, robot_speed=1.)
    rng = np.random.RandomState(entry['seed'])
    prefix_jitter = [rng.normal(0, .005, 3) for state in states[:4] if state['jitter']]
    q_home = pin.neutral(robot.model)
    q_home[:min(len(Q_HOME), model.nq)] = Q_HOME[:min(len(Q_HOME), model.nq)]
    fixed_rotation = robot.forward_kinematics(q_home).rotation.copy()
    adapter = SimpleNamespace(expert=SimpleNamespace(model=model), episode=entry, states=states,
                              fixed_rotation=fixed_rotation, rng=rng)
    endpoints = placement_endpoints(adapter, data)
    for i, (pose, expected) in enumerate(zip(endpoints, saved['placement']['endpoints'])):
        exact(pose.homogeneous, expected, f'endpoint_{i}')
    state = rng.get_state()
    rng_after = dict(algorithm=state[0], keys=state[1], position=state[2],
                     has_gauss=state[3], cached_gaussian=state[4])
    for key, value in rng_after.items():
        exact(value, saved['rng_after_endpoints'][key], 'rng_'+key)
    release_height = validate_release_geometry(endpoints[-1], checker.relative,
        data.xpos[model.body(f"target_cube_{entry['place']}").id])
    desired = pose_waypoints(start, endpoints)
    old_waypoints = saved['placement']['waypoints']
    for i, row in enumerate(old_waypoints):
        exact(desired[i][1].homogeneous, row['pose'], f'original_waypoint_{i}')
        exact(desired[i][0], row['stage'], f'original_waypoint_stage_{i}')
    last = old_waypoints[-1]
    previous_q = np.asarray(old_waypoints[-2]['candidates'][-1]['q'])
    solved_q = np.asarray(last['candidates'][-1]['q'])
    step = float(np.max(np.abs(solved_q[:7]-previous_q[:7])))
    exact(step, failure['details']['max_joint_displacement_rad'], 'original_rejected_step')
    original = dict(first_failed_gate='waypoint_branch_jump', rejected_waypoint_index=len(old_waypoints)-1,
                    joint_step_rad=step, limit_rad=.05, previous_q=previous_q, rejected=last,
                    attempted_waypoints=len(old_waypoints), collision_samples=len(saved['last_collision_samples']),
                    iterations=sum(len(r['candidates'])-1 for r in old_waypoints),
                    residual_evaluations=sum(len(r['candidates']) for r in old_waypoints))
    exact(capture_state(model, data), arrays['final_state'], 'reconstruction_no_state_mutation')
    evidence = dict(seed=entry['seed'], prefix_jitter=prefix_jitter, rng_after_endpoints=rng_after,
                    hand_frame_max_abs_error=hand_frame_error, predicted_release_height_m=release_height,
                    original_waypoints=len(desired), exact_saved_reconstruction=True,
                    input_hashes={str(p.relative_to(ROOT)): sha(p) for p in
                                  (CASE/'terminal_state.npz', CASE/'plan_and_events.json', CASE/'input.json',
                                   CASE/'failure.json', V3/'tests_passed_reviewed.json')})
    return SimpleNamespace(model=model, robot=robot, data=data, bounds=bounds, measured_q=measured_q,
        previous_ctrl=previous_ctrl, endpoints=endpoints, checker=checker, arrays=arrays,
        saved=saved, entry=entry, original=original, evidence=evidence, hand=hand)


def run_once(out, invoke, metadata):
    """Explicit injected numerical callable; exclusive marker is the no-retry lock."""
    out = Path(out)
    save_json(out/'numerical_attempt_started.json', dict(metadata=metadata, maximum_path_attempts=1, no_retry=True))
    journal = {}
    result = dict(numerical_pass=False, first_failed_gate=None, actual_expert_invocations=0,
                  physics_substeps=0, physics_authorized=False, independent_review_completed=False,
                  path_attempts=1, no_retry=True, training_ready=False)
    try:
        invoke(journal)
        if journal.get('success') is not True:
            raise PlanningFailure('numerical_result_missing')
        result['numerical_pass'] = True
    except Exception as exc:
        result.update(first_failed_gate=getattr(exc, 'gate', type(exc).__name__),
                      details=getattr(exc, 'details', {}), message=str(exc))
    finally:
        # Strict JSON retains NaN/Inf markers in failed candidate evidence.
        save_json(out/'plan.json', journal)
        result['budgets'] = journal.get('budgets', {})
        save_json(out/'result.json', result)
    return result


def test_evidence():
    require_cpu_runtime()
    baseline = json.loads((OUT/'hashes_before.json').read_text())
    check_preservation(baseline)
    before = sources()
    summary = dict(source_hashes=before, image=IMAGE, path_attempts=0, physics_substeps=0)
    # Freeze the originally inventoried test files, not concurrently added work.
    legacy = unittest.TestSuite()
    loader = unittest.TestLoader()
    for path in sorted(baseline):
        p = Path(path)
        if p.parent == Path('tests') and p.name.startswith('test_') and p.suffix == '.py':
            legacy.addTests(loader.discover(str(ROOT/'tests'), pattern=p.name))
    new = unittest.TestLoader().discover(str(PACKAGE/'tests'))
    for name, suite in (('legacy', legacy), ('new', new)):
        with (OUT/f'tests_{name}_verified.log').open('x') as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        summary[name] = dict(count=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                             skipped=len(result.skipped), passed=result.wasSuccessful() and not result.skipped)
        print(name, summary[name], flush=True)
    summary['preservation'] = check_preservation(baseline)
    summary['source_hashes_unchanged'] = before == sources()
    save_json(OUT/'tests_verified.json', summary)
    if (summary['legacy']['count'] != 168 or not summary['legacy']['passed'] or not summary['new']['passed']
            or not summary['source_hashes_unchanged']):
        raise PlanningFailure('cpu_test_or_binding_failure')
    return summary


def execute():
    require_cpu_runtime()
    tested = json.loads((OUT/'tests_verified.json').read_text())
    current = sources()
    if (tested['source_hashes'] != current or tested['image'] != IMAGE or tested['legacy']['count'] != 168
            or not tested['legacy']['passed'] or not tested['new']['passed']):
        raise PlanningFailure('current_tests_required')
    if (OUT/'numerical_attempt_started.json').exists():
        raise FileExistsError('The single numerical slot has already been consumed')
    baseline = json.loads((OUT/'hashes_before.json').read_text())
    check_preservation(baseline)
    ctx = reconstruct()
    save_json(OUT/'reconstruction.json', ctx.evidence)
    save_json(OUT/'original_rejection.json', ctx.original)
    with (OUT/'reconstructed_inputs.npz').open('xb') as stream:
        np.savez_compressed(stream, **ctx.arrays, measured_q=ctx.measured_q, previous_ctrl=ctx.previous_ctrl,
            hand_pose=ctx.hand.homogeneous, payload_hand_relative=ctx.checker.relative.homogeneous,
            endpoints=np.array([p.homogeneous for p in ctx.endpoints]), bounds=ctx.bounds)

    def numerical(journal):
        try:
            plan_route(ctx.robot, ctx.measured_q, ctx.previous_ctrl, ctx.endpoints, ctx.bounds,
                       ctx.checker, [90, 60, 30], 1200-197-60-90, journal)
        finally:
            journal['collision_samples'] = ctx.checker.journal
            journal['collision_filters'] = ctx.checker.filters
            journal['lift_state_unchanged'] = bool(np.array_equal(capture_state(ctx.model, ctx.data),
                                                                  ctx.arrays['final_state']))
            if not journal['lift_state_unchanged']:
                raise PlanningFailure('numerical_mutated_lift_state')

    result = run_once(OUT, numerical, dict(source_hashes=current, image=IMAGE, seed=ctx.entry['seed'],
                                          input_hashes=ctx.evidence['input_hashes']))
    preservation = check_preservation(baseline)
    preservation['source_hashes_unchanged'] = current == sources()
    save_json(OUT/'preservation_after.json', preservation)
    if not preservation['source_hashes_unchanged']:
        raise PlanningFailure('v4_sources_changed_during_attempt')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--tests', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.tests:
        test_evidence()
    else:
        raise SystemExit(0 if execute()['numerical_pass'] else 1)