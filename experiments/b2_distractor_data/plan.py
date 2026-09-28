"""B2 distractors: envelope F7 target pair plus 0-2 non-target cubes.

One axis is under test: **distractor presence**. Instruction and colour/region arrangement still
vary per episode as background randomisation — removing them would reopen the position confound
B1 exists to have closed (FAILURES #27) — but they are not reported factors here.

The distractor bodies already exist in the pinned scene and are parked far outside the workspace
(z ~ 30 m) whenever unused, which is exactly the state every prior dataset was collected in. The
control arm therefore reproduces the B1 visual condition **without editing the hash-bound scene
XML**, which is sealed into `b1_language_data` and `broad_workspace_data` provenance.
"""
import hashlib

import numpy as np

from experiments.fixed_layout.contract import (
    ROOT, IMAGE, SOURCE, SOURCE_SHA256, SOURCE_SCENE_SHA256, TASK, _same,
)

OUTPUT = ROOT / 'outputs/b2_distractor_data/b2_distractor_20260928'
VERSION = 'b2_distractor_data_v1'
DATASET_REPO_ID = 'b2_distractor_data'
MASTER = 20260928
START_BYTES = 60 * 1024**3
FLOOR_BYTES = 40 * 1024**3
MAX_SECONDS = 12 * 60 * 60
SPEC = 'docs/superpowers/specs/2026-09-28-b2-distractors-design.md'
PLAN = 'docs/superpowers/plans/2026-09-28-b2-distractors.md'
EXPERT_VERSION = 'paired_scripted_expert_b2_distractor_v1'
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

# Distractors may occupy the full y span, including the middle band the target pair never uses.
# Restricting them to the middle band would make "distractor" predictable from position and
# trivially ignorable -- the same class of confound as FAILURES #27.
DISTRACTOR_COLOURS = ('blue', 'yellow')
DISTRACTOR_LOW = (.20, -.45)
DISTRACTOR_HIGH = (.35, .45)
MIN_CLEARANCE_M = .15
DISTRACTOR_DRAW_LIMIT = 500
# Rejected if the expert disturbs a distractor by more than this; measures collision rate
# instead of silently admitting a scrambled scene.
DISTRACTOR_DISTURBANCE_M = .02

TASK_RED_ON_GREEN = TASK
TASK_GREEN_ON_RED = 'pick up the green cube and place it on top of the red cube'
INSTRUCTIONS = tuple(sorted((TASK_RED_ON_GREEN, TASK_GREEN_ON_RED)))

CELLS = {
    'C1': dict(pick='red', place='green', arrangement='familiar'),
    'C2': dict(pick='red', place='green', arrangement='swapped'),
    'C3': dict(pick='green', place='red', arrangement='familiar'),
    'C4': dict(pick='green', place='red', arrangement='swapped'),
}
CELL_IDS = tuple(sorted(CELLS))
ARRANGEMENTS = ('familiar', 'swapped')

CONDITIONS = ('control', 'distractor')
# Deck of four: half the distractor-present episodes carry two cubes, and the single-cube half
# splits evenly between the two colours.
DISTRACTOR_DECK = (('blue',), ('yellow',), DISTRACTOR_COLOURS, DISTRACTOR_COLOURS)

HOME_JITTER_DEG_MAX = 1.0
HOVER_Z_JITTER_M_MAX = .03
PHASES = ('pre_grasp', 'pre_place')

TRAIN_CONTROL_COUNT = 32
TRAIN_DISTRACTOR_COUNT = 128
VALIDATION_CONTROL_COUNT = 8
VALIDATION_DISTRACTOR_COUNT = 32
TRAIN_LHS_COUNT = TRAIN_CONTROL_COUNT + TRAIN_DISTRACTOR_COUNT
TRAIN_BASE_COUNT = 0
VALIDATION_COUNT = VALIDATION_CONTROL_COUNT + VALIDATION_DISTRACTOR_COUNT
ADMIT_COUNT = TRAIN_LHS_COUNT + TRAIN_BASE_COUNT + VALIDATION_COUNT
CONTROL_COUNT = TRAIN_CONTROL_COUNT + VALIDATION_CONTROL_COUNT
DISTRACTOR_COUNT = TRAIN_DISTRACTOR_COUNT + VALIDATION_DISTRACTOR_COUNT

EVAL_SCREEN_COUNT = 20
EVAL_REPORT_COUNT = 40
EVAL_HOLDOUT_COUNT = EVAL_SCREEN_COUNT + EVAL_REPORT_COUNT
EVAL_TIERS = ('screen', 'report')

ATTEMPT_LIMIT = 300
REJECTION_BUDGET = 100
REPLAY_LIMIT = 2 * ADMIT_COUNT
UNIQUE_LAYOUTS = TRAIN_LHS_COUNT + VALIDATION_COUNT + EVAL_HOLDOUT_COUNT  # 260
# Every reserved evaluation layout is banked in BOTH conditions so the report contrast is
# within-layout; arrangement is fixed per layout because it is background, not a reported factor.
UNIQUE_BANK_IDENTITIES = ADMIT_COUNT + len(CONDITIONS)*EVAL_HOLDOUT_COUNT + 1  # 321
EXPECTED_FRAMES = 528
FRAME_ROWS = ADMIT_COUNT * EXPECTED_FRAMES  # 105_600
DECODED_FRAMES = 2 * FRAME_ROWS  # 211_200

# Populated only if a layout proves uncompletable during collection; see broad_workspace #11.
INFEASIBLE_LAYOUT_RESAMPLES = {}


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


def validate_distractors(colours):
    if type(colours) is not tuple or not colours:
        raise ValueError('Distractor colours must be a non-empty tuple')
    if len(set(colours)) != len(colours) or any(c not in DISTRACTOR_COLOURS for c in colours):
        raise ValueError(f'Unapproved distractor colours: {colours!r}')


def draw_distractor_xy(seeds, identity, colours, cube_xy):
    """Rejection-sample each distractor clear of both targets and of every other distractor."""
    if not colours:
        return {}
    validate_distractors(colours)
    occupied = [list(cube_xy[:2]), list(cube_xy[2:])]
    placed = {}
    for colour in colours:
        key = f'distractor:{identity}:{colour}'
        seeds[key] = seed(key)
        stream = np.random.RandomState(seeds[key])
        for _ in range(DISTRACTOR_DRAW_LIMIT):
            xy = [float(DISTRACTOR_LOW[a] + (DISTRACTOR_HIGH[a]-DISTRACTOR_LOW[a])*stream.uniform())
                  for a in range(2)]
            if all(float(np.hypot(xy[0]-o[0], xy[1]-o[1])) >= MIN_CLEARANCE_M for o in occupied):
                occupied.append(xy)
                placed[colour] = xy
                break
        else:
            raise ValueError(f'Distractor {colour} unplaceable for {identity}; stop without redraw')
    return dict(sorted(placed.items()))


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
    return dict(layout_id='base', split='base', candidate_index=None, condition='control',
                distractors=[], distractor_xy_m={},
                cube_xy_m=[float(BASE_REGION_A_XY[0]), float(BASE_REGION_A_XY[1]),
                           float(BASE_REGION_B_XY[0]), float(BASE_REGION_B_XY[1])])


def _balanced_deck(split_name, kind, n, deck, seeds):
    if n % len(deck):
        raise ValueError(f'{split_name}: {n} episodes do not divide evenly into {len(deck)} {kind}')
    identity = f'{kind}:{split_name}'
    seeds[identity] = seed(identity)
    cards = [deck[k % len(deck)] for k in range(n)]
    order = np.random.RandomState(seeds[identity]).permutation(n)
    return [cards[i] for i in order]


def assign_cells(split_name, n, seeds):
    """Exactly n/4 episodes per cell, in a deterministic shuffle.

    Balance is enforced by construction rather than by sampling, so no split can drift toward one
    arrangement — an imbalance would reintroduce the position shortcut B1 removed.
    """
    return [str(x) for x in _balanced_deck(split_name, 'cells', n, CELL_IDS, seeds)]


def assign_conditions(split_name, control_n, distractor_n, seeds):
    """Deterministic interleave of the two arms; the arms are asymmetric by design (40/160)."""
    identity = f'conditions:{split_name}'
    seeds[identity] = seed(identity)
    deck = ['control']*control_n + ['distractor']*distractor_n
    order = np.random.RandomState(seeds[identity]).permutation(len(deck))
    return [deck[i] for i in order]


def assign_distractor_specs(split_name, n, seeds):
    return [tuple(x) for x in _balanced_deck(split_name, 'distractors', n, DISTRACTOR_DECK, seeds)]


def _smoke_profiles():
    """Riskiest-first ordering: FAILURES #29 — make the first minutes of the run the smoke test."""
    return (('distractor', 2), ('distractor', 1), ('control', 0), ('distractor', 2))


def promote_smoke_order(entries):
    """Swap the riskiest profiles into the leading slots without changing the multiset."""
    entries = list(entries)
    for slot, (condition, count) in enumerate(_smoke_profiles()):
        if slot >= len(entries):
            break
        found = next((k for k in range(slot, len(entries))
                      if entries[k]['condition'] == condition and len(entries[k]['distractors']) == count), None)
        if found is None:
            raise ValueError(f'No episode matches smoke profile {condition}/{count}')
        entries[slot], entries[found] = entries[found], entries[slot]
    return entries


def _split_entries(split_name, layouts, control_n, distractor_n, seeds):
    """Condition, cell and distractor-colour assignment for one split, before positions exist."""
    conditions = assign_conditions(split_name, control_n, distractor_n, seeds)
    cells = {c: iter(assign_cells(f'{split_name}:{c}', n, seeds))
             for c, n in (('control', control_n), ('distractor', distractor_n))}
    specs = iter(assign_distractor_specs(split_name, distractor_n, seeds))
    entries = []
    for layout, condition in zip(layouts, conditions):
        entries.append(dict(layout=layout, condition=condition, cell=next(cells[condition]),
                            distractors=list(next(specs)) if condition == 'distractor' else []))
    return promote_smoke_order(entries)


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

    train_entries = _split_entries('train', training_lhs, TRAIN_CONTROL_COUNT,
                                   TRAIN_DISTRACTOR_COUNT, seeds)
    validation_entries = _split_entries('validation', validation_lhs, VALIDATION_CONTROL_COUNT,
                                        VALIDATION_DISTRACTOR_COUNT, seeds)

    episodes = []
    for split, entries in (('train', train_entries), ('validation', validation_entries)):
        base = 0 if split == 'train' else TRAIN_LHS_COUNT + TRAIN_BASE_COUNT
        for j, entry in enumerate(entries):
            i = base + j
            layout = entry['layout']
            episode = _episode(seeds, i, layout['layout_id'], split, entry)
            region = layout['region_xy_m']
            layout['cube_xy_m'] = colour_positions(episode['arrangement'], region[:2], region[2:])
            layout['arrangement'] = episode['arrangement']
            layout['condition'] = episode['condition']
            layout['distractors'] = list(episode['distractors'])
            layout['distractor_xy_m'] = draw_distractor_xy(
                seeds, f'episode:{i}', tuple(episode['distractors']), layout['cube_xy_m'])
            episode['distractor_xy_m'] = dict(layout['distractor_xy_m'])
            episodes.append(episode)

    # Reserved layouts are never collected. Each is banked in both conditions so the report
    # contrast is within-layout; arrangement is fixed per layout (background, not a factor).
    eval_arrangements = _balanced_deck('evaluation', 'arrangement', EVAL_HOLDOUT_COUNT,
                                       ARRANGEMENTS, seeds)
    eval_specs = _balanced_deck('evaluation', 'holdout_distractors', EVAL_HOLDOUT_COUNT,
                                DISTRACTOR_DECK, seeds)
    evaluation_holdout = {}
    for n, layout in enumerate(evaluation_lhs):
        lid = layout['layout_id']
        region = layout['region_xy_m']
        arrangement = str(eval_arrangements[n])
        cube_xy = colour_positions(arrangement, region[:2], region[2:])
        colours = tuple(eval_specs[n])
        home = [_draw_eval_home_jitter(seeds, lid, j) for j in range(7)]
        hover = [_draw_eval_hover_jitter(seeds, lid, phase) for phase in PHASES]
        layout['arrangement'] = arrangement
        layout['cube_xy_m'] = list(cube_xy)
        evaluation_holdout[lid] = dict(
            home_pose_jitter_deg=home, hover_z_jitter_m=hover,
            hover_z_offset=dict(zip(PHASES, hover)),
            region_xy_m=list(region), arrangement=arrangement, cube_xy_m=list(cube_xy),
            tier=EVAL_TIERS[0] if n < EVAL_SCREEN_COUNT else EVAL_TIERS[1],
            distractors=list(colours),
            conditions={'control': {},
                        'distractor': draw_distractor_xy(seeds, f'evaluation:{lid}', colours, cube_xy)})
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
        evaluation_screen_ids=[k for k, v in evaluation_holdout.items() if v['tier'] == 'screen'],
        evaluation_report_ids=[k for k, v in evaluation_holdout.items() if v['tier'] == 'report'],
        policy_seeds=policy, expert_attempts=ADMIT_COUNT, attempt_limit=ATTEMPT_LIMIT,
        rejection_budget=REJECTION_BUDGET, feasibility_redraws=8,
        infeasible_layout_resamples=dict(INFEASIBLE_LAYOUT_RESAMPLES),
        replay_modes=['reference', 'persisted_float32'], replay_limit=REPLAY_LIMIT,
        unique_layouts=UNIQUE_LAYOUTS, unique_bank_identities=UNIQUE_BANK_IDENTITIES,
        reset_policy='per_episode_arm_qpos_and_cube_xy_no_settle',
        envelope=dict(name='F7', region_a=dict(low=list(REGION_A_LOW), high=list(REGION_A_HIGH)),
                      region_b=dict(low=list(REGION_B_LOW), high=list(REGION_B_HIGH)),
                      distractor=dict(low=list(DISTRACTOR_LOW), high=list(DISTRACTOR_HIGH))),
        conditions=list(CONDITIONS), control_episodes=CONTROL_COUNT,
        distractor_episodes=DISTRACTOR_COUNT, distractor_colours=list(DISTRACTOR_COLOURS),
        min_clearance_m=float(MIN_CLEARANCE_M),
        distractor_disturbance_m=float(DISTRACTOR_DISTURBANCE_M),
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


def _episode(seeds, i, layout_id, split, entry):
    identity = f'expert:{i}'
    seeds[identity] = seed(identity)
    home = [_draw_home_jitter(seeds, i, j) for j in range(7)]
    hover = [_draw_hover_jitter(seeds, i, phase) for phase in PHASES]
    spec = CELLS[entry['cell']]
    instruction = instruction_for(spec['pick'], spec['place'])
    return dict(episode_index=i, seed=seeds[identity], layout_id=layout_id, split=split,
                cell=entry['cell'], arrangement=spec['arrangement'],
                pick=spec['pick'], place=spec['place'], colors=['red', 'green'],
                condition=entry['condition'], distractors=list(entry['distractors']),
                instruction=instruction, task_index=task_index_for(instruction),
                source_episode_index=None,
                home_pose_jitter_deg=home, hover_z_jitter_m=hover,
                hover_z_offset=dict(zip(PHASES, hover)))


def validate_plan(plan):
    if not _same(plan, make_plan()):
        raise ValueError('Noncanonical B2 distractor plan; no count/seed/layout overrides')
