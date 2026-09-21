"""Envelope F7 plan; 200 admits, 264 layouts, 80 held-out eval, 2 policy seeds."""
import hashlib

import numpy as np

from experiments.fixed_layout.contract import (
    ROOT, IMAGE, SOURCE, SOURCE_SHA256, SOURCE_SCENE_SHA256, TASK, _same,
)

OUTPUT = ROOT / 'outputs/broad_workspace_data/broad_workspace_20260920'
VERSION = 'broad_workspace_data_v1'
MASTER = 20260920
START_BYTES = 60 * 1024**3
FLOOR_BYTES = 40 * 1024**3
MAX_SECONDS = 12 * 60 * 60
SPEC = 'docs/superpowers/specs/2026-09-19-broad-workspace-design.md'
PLAN = 'docs/superpowers/plans/2026-09-19-broad-workspace.md'
EXPERT_VERSION = 'paired_scripted_expert_broad_workspace_v1'
BASE_EXPERT_VERSION = 'paired_scripted_expert_pose_snapshot_v2'

# Envelope F7: disjoint half-rectangles, non-overlap by construction.
RED_LOW = (.20, -.45)
RED_HIGH = (.35, -.20)
GREEN_LOW = (.20, .20)
GREEN_HIGH = (.35, .45)
AXIS_NAMES = ('red_x', 'red_y', 'green_x', 'green_y')
AXIS_LOW = (RED_LOW[0], RED_LOW[1], GREEN_LOW[0], GREEN_LOW[1])
AXIS_HIGH = (RED_HIGH[0], RED_HIGH[1], GREEN_HIGH[0], GREEN_HIGH[1])
BASE_RED_XY = (0.2820318976205449, -0.38303709520664875)
BASE_GREEN_XY = (0.2676253941656954, 0.3512266692018041)

HOME_JITTER_DEG_MAX = 1.0
HOVER_Z_JITTER_M_MAX = .03
PHASES = ('pre_grasp', 'pre_place')

TRAIN_LHS_COUNT = 144
TRAIN_BASE_COUNT = 16
VALIDATION_COUNT = 40
EVAL_HOLDOUT_COUNT = 80
ADMIT_COUNT = TRAIN_LHS_COUNT + TRAIN_BASE_COUNT + VALIDATION_COUNT
ATTEMPT_LIMIT = 300
REJECTION_BUDGET = 100
REPLAY_LIMIT = 2 * ADMIT_COUNT
UNIQUE_LAYOUTS = TRAIN_LHS_COUNT + VALIDATION_COUNT + EVAL_HOLDOUT_COUNT  # 264
UNIQUE_BANK_IDENTITIES = ADMIT_COUNT + EVAL_HOLDOUT_COUNT + 1  # 281
EXPECTED_FRAMES = 528
FRAME_ROWS = ADMIT_COUNT * EXPECTED_FRAMES  # 105_600
DECODED_FRAMES = 2 * FRAME_ROWS  # 211_200

# Layouts the scripted expert cannot stack from any approved home-pose draw or hover height.
# validation_020 placed the cube 2.0-2.1 cm off centre on all 16 measured attempts (admitted
# episodes land within 1.74 cm). validation_032 stacks fine but its grasp geometry makes the
# float32 action replay drift 1.51e-3 (tolerance 1e-3) identically on every redraw, so neither
# is recoverable by jitter; both are resampled inside the same F7 envelope at the recorded
# index. Extends the existing IK feasibility rejection sampling to expert completability.
INFEASIBLE_LAYOUT_RESAMPLES = {'validation_020': 2, 'validation_032': 0}


def seed(identity):
    return int.from_bytes(hashlib.sha256(f'{VERSION}:{MASTER}:{identity}'.encode('ascii')).digest()[:4], 'big')


def resample_layout_xy(layout_id, index):
    """Deterministic replacement draw inside envelope F7 for an uncompletable layout."""
    if type(index) is not int or index < 0:
        raise ValueError('Resample index must be a non-negative integer')
    stream = np.random.RandomState(seed(f'resample:{layout_id}:{index}'))
    return [float(AXIS_LOW[axis] + (AXIS_HIGH[axis]-AXIS_LOW[axis])*stream.uniform()) for axis in range(4)]


def _lhs(split_name, n, seeds):
    """4-D Latin Hypercube sample inside envelope F7; independent axes."""
    values = np.empty((n, 4), np.float64)
    for axis in range(4):
        streams = {}
        for kind in ('permutation', 'jitter'):
            identity = f'layout:{split_name}:{AXIS_NAMES[axis]}:{kind}'
            seeds[identity] = seed(identity)
            streams[kind] = np.random.RandomState(seeds[identity])
        low, high = AXIS_LOW[axis], AXIS_HIGH[axis]
        values[:, axis] = low + (high-low)*(streams['permutation'].permutation(n)+streams['jitter'].uniform(size=n))/n
    return values


def _draw_home_jitter(seeds, i, joint):
    identity = f'homepose:{i}:{joint}'
    seeds[identity] = seed(identity)
    return float(np.random.RandomState(seeds[identity]).uniform(-HOME_JITTER_DEG_MAX, HOME_JITTER_DEG_MAX))


def _draw_hover_jitter(seeds, i, phase):
    identity = f'hoverz:{i}:{phase}'
    seeds[identity] = seed(identity)
    return float(np.random.RandomState(seeds[identity]).uniform(-HOVER_Z_JITTER_M_MAX, HOVER_Z_JITTER_M_MAX))


def _draw_eval_home_jitter(seeds, layout_id, joint):
    identity = f'homepose:evaluation:{layout_id}:{joint}'
    seeds[identity] = seed(identity)
    return float(np.random.RandomState(seeds[identity]).uniform(-HOME_JITTER_DEG_MAX, HOME_JITTER_DEG_MAX))


def _draw_eval_hover_jitter(seeds, layout_id, phase):
    identity = f'hoverz:evaluation:{layout_id}:{phase}'
    seeds[identity] = seed(identity)
    return float(np.random.RandomState(seeds[identity]).uniform(-HOVER_Z_JITTER_M_MAX, HOVER_Z_JITTER_M_MAX))


def base_layout():
    return dict(layout_id='base', split='base', candidate_index=None,
                cube_xy_m=[float(BASE_RED_XY[0]), float(BASE_RED_XY[1]),
                           float(BASE_GREEN_XY[0]), float(BASE_GREEN_XY[1])])


def make_plan() -> dict:
    seeds = {}
    layouts = []
    split_map = (('training_varied', 'train', TRAIN_LHS_COUNT),
                 ('validation', 'validation', VALIDATION_COUNT),
                 ('evaluation', 'evaluation', EVAL_HOLDOUT_COUNT))
    for stream_name, split, n in split_map:
        xy = _lhs(stream_name, n, seeds)
        layouts.extend(dict(layout_id=f'{split}_{i:03d}', split=split, candidate_index=i,
                            cube_xy_m=xy[i].tolist()) for i in range(n))
    for layout in layouts:
        index = INFEASIBLE_LAYOUT_RESAMPLES.get(layout['layout_id'])
        if index is not None:
            layout['cube_xy_m'] = resample_layout_xy(layout['layout_id'], index)
    training_lhs = [x for x in layouts if x['split'] == 'train']
    validation_lhs = [x for x in layouts if x['split'] == 'validation']
    evaluation_lhs = [x for x in layouts if x['split'] == 'evaluation']

    episodes = []
    for i in range(TRAIN_LHS_COUNT):
        layout_id = training_lhs[i]['layout_id']
        episodes.append(_episode(seeds, i, layout_id, 'train'))
    for k in range(TRAIN_BASE_COUNT):
        i = TRAIN_LHS_COUNT + k
        episodes.append(_episode(seeds, i, 'base', 'train'))
    for j in range(VALIDATION_COUNT):
        i = TRAIN_LHS_COUNT + TRAIN_BASE_COUNT + j
        layout_id = validation_lhs[j]['layout_id']
        episodes.append(_episode(seeds, i, layout_id, 'validation'))

    evaluation_holdout = {}
    for layout in evaluation_lhs:
        lid = layout['layout_id']
        home = [_draw_eval_home_jitter(seeds, lid, j) for j in range(7)]
        hover = [_draw_eval_hover_jitter(seeds, lid, phase) for phase in PHASES]
        evaluation_holdout[lid] = dict(home_pose_jitter_deg=home, hover_z_jitter_m=hover,
                                       hover_z_offset=dict(zip(PHASES, hover)))
    evaluation_holdout = dict(sorted(evaluation_holdout.items()))

    policy = []
    for k in range(2):
        identity = f'policy:{k}'
        seeds[identity] = seed(identity)
        policy.append(seeds[identity])
    expert = [e['seed'] for e in episodes]
    if len(set(seeds.values())) != len(seeds):
        raise ValueError('Seed collision; stop without redraw')
    reserved = set(range(20260918, 20260938))
    if set(policy) & (set(expert) | reserved):
        raise ValueError('Policy seed overlaps expert/reserved; stop without redraw')

    return dict(
        version=VERSION, master_seed=MASTER, rng='numpy.random.RandomState', rng_seeds=seeds,
        seed_derivation='SHA256(namespace:master:identity) first four bytes big endian',
        source=dict(path=SOURCE, sha256=SOURCE_SHA256, episode_index=7, scene_hash=SOURCE_SCENE_SHA256),
        base_layout=base_layout(), layouts=layouts, episodes=episodes,
        train_ids=list(range(TRAIN_LHS_COUNT + TRAIN_BASE_COUNT)),
        validation_ids=list(range(TRAIN_LHS_COUNT + TRAIN_BASE_COUNT, ADMIT_COUNT)),
        evaluation_layout_ids=[x['layout_id'] for x in evaluation_lhs],
        evaluation_holdout=evaluation_holdout,
        policy_seeds=policy, expert_attempts=ADMIT_COUNT, attempt_limit=ATTEMPT_LIMIT,
        rejection_budget=REJECTION_BUDGET, feasibility_redraws=8,
        infeasible_layout_resamples=dict(INFEASIBLE_LAYOUT_RESAMPLES),
        replay_modes=['reference', 'persisted_float32'], replay_limit=REPLAY_LIMIT,
        unique_layouts=UNIQUE_LAYOUTS, unique_bank_identities=UNIQUE_BANK_IDENTITIES,
        reset_policy='per_episode_arm_qpos_and_cube_xy_no_settle',
        envelope=dict(name='F7', red=dict(low=list(RED_LOW), high=list(RED_HIGH)),
                      green=dict(low=list(GREEN_LOW), high=list(GREEN_HIGH))),
        expert_version=EXPERT_VERSION, base_expert_version=BASE_EXPERT_VERSION,
        home_pose_jitter_deg_max=float(HOME_JITTER_DEG_MAX),
        hover_z_jitter_m_max=float(HOVER_Z_JITTER_M_MAX),
        output=str(OUTPUT.relative_to(ROOT)), start_free_bytes=START_BYTES, floor_free_bytes=FLOOR_BYTES,
        max_seconds=MAX_SECONDS, image=IMAGE, expected_episode_frames=EXPECTED_FRAMES, fps=30,
        physics_hz=600, substeps_per_control=20, state_dim=9, action_dim=8, policy_dtype='float32',
        camera_size=[512, 512],
        action_semantics='absolute_joint_commands_gripper_0_closed_255_open',
        observation_alignment='observe_command_record_pre_step',
        failure_policy='first_failure_stop_no_retry_no_resume_admits_only_reject_budget',
        automatic_training=False, validation_is_spatial_holdout=False,
        statistics_episode_ids=list(range(TRAIN_LHS_COUNT + TRAIN_BASE_COUNT)),
    )


def _episode(seeds, i, layout_id, split):
    identity = f'expert:{i}'
    seeds[identity] = seed(identity)
    home = [_draw_home_jitter(seeds, i, j) for j in range(7)]
    hover = [_draw_hover_jitter(seeds, i, phase) for phase in PHASES]
    return dict(episode_index=i, seed=seeds[identity], layout_id=layout_id, split=split,
                pick='red', place='green', colors=['red', 'green'], instruction=TASK, task_index=0,
                source_episode_index=None,
                home_pose_jitter_deg=home, hover_z_jitter_m=hover,
                hover_z_offset=dict(zip(PHASES, hover)))


def validate_plan(plan):
    if not _same(plan, make_plan()):
        raise ValueError('Noncanonical broad-workspace plan; no count/seed/layout overrides')
