"""B1 language grounding: envelope F7, 200 admits, 4 cells (instruction x colour arrangement).

The two cube positions are drawn as **regions**, not colours. Region A is the negative-y
half-rectangle and region B the positive-y one — numerically identical to the broad-workspace
envelope. Which colour occupies which region is decided by the episode's cell, so colour is
decorrelated from position and the instruction can only be resolved by seeing colour.
"""
import hashlib

import numpy as np

from experiments.fixed_layout.contract import (
    ROOT, IMAGE, SOURCE, SOURCE_SHA256, SOURCE_SCENE_SHA256, TASK, _same,
)

OUTPUT = ROOT / 'outputs/b1_language_data/b1_language_20260925'
VERSION = 'b1_language_data_v1'
MASTER = 20260925
START_BYTES = 60 * 1024**3
FLOOR_BYTES = 40 * 1024**3
MAX_SECONDS = 12 * 60 * 60
SPEC = 'docs/superpowers/specs/2026-09-24-b1-language-grounding-design.md'
PLAN = 'docs/superpowers/plans/2026-09-25-b1-language-grounding.md'
EXPERT_VERSION = 'paired_scripted_expert_b1_language_v1'
BASE_EXPERT_VERSION = 'paired_scripted_expert_pose_snapshot_v2'

# Envelope F7: disjoint half-rectangles, non-overlap by construction. Named by REGION, not colour.
REGION_A_LOW = (.20, -.45)
REGION_A_HIGH = (.35, -.20)
REGION_B_LOW = (.20, .20)
REGION_B_HIGH = (.35, .45)
AXIS_NAMES = ('regionA_x', 'regionA_y', 'regionB_x', 'regionB_y')
AXIS_LOW = (REGION_A_LOW[0], REGION_A_LOW[1], REGION_B_LOW[0], REGION_B_LOW[1])
AXIS_HIGH = (REGION_A_HIGH[0], REGION_A_HIGH[1], REGION_B_HIGH[0], REGION_B_HIGH[1])
BASE_REGION_A_XY = (0.2820318976205449, -0.38303709520664875)
BASE_REGION_B_XY = (0.2676253941656954, 0.3512266692018041)

# Two instructions. `task_index` is the index into this SORTED tuple, so it never depends on
# collection order — LeRobot assigns its own indices by first appearance, and validate.py asserts
# the two agree.
TASK_RED_ON_GREEN = TASK
TASK_GREEN_ON_RED = 'pick up the green cube and place it on top of the red cube'
INSTRUCTIONS = tuple(sorted((TASK_RED_ON_GREEN, TASK_GREEN_ON_RED)))

# cell -> (pick colour, place colour, arrangement). Arrangement says which colour is in region A.
CELLS = {
    'C1': dict(pick='red', place='green', arrangement='familiar'),
    'C2': dict(pick='red', place='green', arrangement='swapped'),
    'C3': dict(pick='green', place='red', arrangement='familiar'),
    'C4': dict(pick='green', place='red', arrangement='swapped'),
}
CELL_IDS = tuple(sorted(CELLS))
ARRANGEMENTS = ('familiar', 'swapped')

HOME_JITTER_DEG_MAX = 1.0
HOVER_Z_JITTER_M_MAX = .03
PHASES = ('pre_grasp', 'pre_place')

TRAIN_LHS_COUNT = 160
TRAIN_BASE_COUNT = 0  # B1 uses 200 distinct geometries; no repeated base geometry.
VALIDATION_COUNT = 40
EVAL_HOLDOUT_COUNT = 80
ADMIT_COUNT = TRAIN_LHS_COUNT + TRAIN_BASE_COUNT + VALIDATION_COUNT
ATTEMPT_LIMIT = 300
REJECTION_BUDGET = 100
REPLAY_LIMIT = 2 * ADMIT_COUNT
UNIQUE_LAYOUTS = TRAIN_LHS_COUNT + VALIDATION_COUNT + EVAL_HOLDOUT_COUNT  # 280
# Every reserved evaluation layout is banked in BOTH arrangements, so it contributes two scenes.
UNIQUE_BANK_IDENTITIES = ADMIT_COUNT + 2*EVAL_HOLDOUT_COUNT + 1  # 361
EXPECTED_FRAMES = 528
FRAME_ROWS = ADMIT_COUNT * EXPECTED_FRAMES  # 105_600
DECODED_FRAMES = 2 * FRAME_ROWS  # 211_200

# Populated only if a layout proves uncompletable during collection; see broad_workspace #11.
# validation_037 (slot 197, C3 green-on-red) exhausted all 9 feasible home-pose draws with
# `expert_execution` rejections after 197 clean admits; the scripted expert cannot complete that
# geometry. Resample index 0 substitutes the layout via amend_frozen_plan on resume.
INFEASIBLE_LAYOUT_RESAMPLES = {'validation_037': 0}


def instruction_for(pick, place):
    """Single source of the instruction string; never hardcode it at a call site."""
    if pick == place or {pick, place} != {'red', 'green'}:
        raise ValueError('pick/place must be the two distinct cube colours')
    return f'pick up the {pick} cube and place it on top of the {place} cube'


def task_index_for(instruction):
    if instruction not in INSTRUCTIONS:
        raise ValueError(f'Unapproved instruction: {instruction!r}')
    return INSTRUCTIONS.index(instruction)


def colour_positions(arrangement, region_a_xy, region_b_xy):
    """cube_xy_m is always [red_x, red_y, green_x, green_y]; the arrangement permutes the regions."""
    if arrangement == 'familiar':
        red, green = region_a_xy, region_b_xy
    elif arrangement == 'swapped':
        red, green = region_b_xy, region_a_xy
    else:
        raise ValueError(f'Unapproved arrangement: {arrangement!r}')
    return [float(red[0]), float(red[1]), float(green[0]), float(green[1])]


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
                cube_xy_m=[float(BASE_REGION_A_XY[0]), float(BASE_REGION_A_XY[1]),
                           float(BASE_REGION_B_XY[0]), float(BASE_REGION_B_XY[1])])


def assign_cells(split_name, n, seeds):
    """Exactly n/4 episodes per cell, in a deterministic shuffle.

    Balance is enforced by construction rather than by sampling, so no split can drift toward one
    arrangement — an imbalance would reintroduce the position shortcut this dataset exists to remove.
    """
    if n % len(CELL_IDS):
        raise ValueError(f'{split_name}: {n} episodes do not divide evenly into {len(CELL_IDS)} cells')
    identity = f'cells:{split_name}'
    seeds[identity] = seed(identity)
    deck = np.array([CELL_IDS[k % len(CELL_IDS)] for k in range(n)])
    order = np.random.RandomState(seeds[identity]).permutation(n)
    return [str(x) for x in deck[order]]


def make_plan() -> dict:
    seeds = {}
    layouts = []
    split_map = (('training_varied', 'train', TRAIN_LHS_COUNT),
                 ('validation', 'validation', VALIDATION_COUNT),
                 ('evaluation', 'evaluation', EVAL_HOLDOUT_COUNT))
    for stream_name, split, n in split_map:
        xy = _lhs(stream_name, n, seeds)
        layouts.extend(dict(layout_id=f'{split}_{i:03d}', split=split, candidate_index=i,
                            region_xy_m=xy[i].tolist()) for i in range(n))
    for layout in layouts:
        index = INFEASIBLE_LAYOUT_RESAMPLES.get(layout['layout_id'])
        if index is not None:
            layout['region_xy_m'] = resample_layout_xy(layout['layout_id'], index)
    training_lhs = [x for x in layouts if x['split'] == 'train']
    validation_lhs = [x for x in layouts if x['split'] == 'validation']
    evaluation_lhs = [x for x in layouts if x['split'] == 'evaluation']

    train_cells = assign_cells('train', TRAIN_LHS_COUNT, seeds)
    validation_cells = assign_cells('validation', VALIDATION_COUNT, seeds)
    by_id = {x['layout_id']: x for x in layouts}

    episodes = []
    for i in range(TRAIN_LHS_COUNT):
        layout = training_lhs[i]
        episode = _episode(seeds, i, layout['layout_id'], 'train', train_cells[i])
        region = layout['region_xy_m']
        layout['cube_xy_m'] = colour_positions(episode['arrangement'], region[:2], region[2:])
        layout['arrangement'] = episode['arrangement']
        episodes.append(episode)
    for j in range(VALIDATION_COUNT):
        i = TRAIN_LHS_COUNT + TRAIN_BASE_COUNT + j
        layout = validation_lhs[j]
        episode = _episode(seeds, i, layout['layout_id'], 'validation', validation_cells[j])
        region = layout['region_xy_m']
        layout['cube_xy_m'] = colour_positions(episode['arrangement'], region[:2], region[2:])
        layout['arrangement'] = episode['arrangement']
        episodes.append(episode)

    # Reserved layouts are never collected, and each must be evaluable in BOTH arrangements, so
    # they carry region positions plus a per-arrangement scene identity instead of one cube_xy_m.
    evaluation_holdout = {}
    for layout in evaluation_lhs:
        lid = layout['layout_id']
        region = layout['region_xy_m']
        home = [_draw_eval_home_jitter(seeds, lid, j) for j in range(7)]
        hover = [_draw_eval_hover_jitter(seeds, lid, phase) for phase in PHASES]
        evaluation_holdout[lid] = dict(
            home_pose_jitter_deg=home, hover_z_jitter_m=hover,
            hover_z_offset=dict(zip(PHASES, hover)),
            region_xy_m=list(region),
            arrangements={a: colour_positions(a, region[:2], region[2:]) for a in ARRANGEMENTS})
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
        envelope=dict(name='F7', region_a=dict(low=list(REGION_A_LOW), high=list(REGION_A_HIGH)),
                      region_b=dict(low=list(REGION_B_LOW), high=list(REGION_B_HIGH))),
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


def _episode(seeds, i, layout_id, split, cell):
    identity = f'expert:{i}'
    seeds[identity] = seed(identity)
    home = [_draw_home_jitter(seeds, i, j) for j in range(7)]
    hover = [_draw_hover_jitter(seeds, i, phase) for phase in PHASES]
    spec = CELLS[cell]
    instruction = instruction_for(spec['pick'], spec['place'])
    return dict(episode_index=i, seed=seeds[identity], layout_id=layout_id, split=split,
                cell=cell, arrangement=spec['arrangement'],
                pick=spec['pick'], place=spec['place'], colors=['red', 'green'],
                instruction=instruction, task_index=task_index_for(instruction),
                source_episode_index=None,
                home_pose_jitter_deg=home, hover_z_jitter_m=hover,
                hover_z_offset=dict(zip(PHASES, hover)))


def validate_plan(plan):
    if not _same(plan, make_plan()):
        raise ValueError('Noncanonical broad-workspace plan; no count/seed/layout overrides')
