"""Exact approved budget and separate SHA-derived RandomState streams."""
import hashlib

import numpy as np

from experiments.fixed_layout.contract import (
    ROOT, IMAGE, SOURCE, SOURCE_SHA256, SOURCE_SCENE_SHA256, TASK, EXPERT_VERSION, _same,
)

OUTPUT = ROOT / 'outputs/local_varied_data/local_varied_20260919'
VERSION = 'local_varied_data_v1'
MASTER = 20260919
START_BYTES = 40 * 1024**3
FLOOR_BYTES = 30 * 1024**3
MAX_SECONDS = 6 * 60 * 60
SPEC = 'docs/superpowers/specs/2026-09-19-local-varied-data-design.md'
PLAN = 'docs/superpowers/plans/2026-09-19-local-varied-data.md'


def seed(identity):
    return int.from_bytes(hashlib.sha256(f'{VERSION}:{MASTER}:{identity}'.encode('ascii')).digest()[:4], 'big')


def make_plan() -> dict:
    seeds = {}
    layouts = [dict(layout_id='center', split='center', candidate_index=None, offsets_m=[0., 0., 0., 0.])]
    for split, n in (('train', 40), ('validation', 12), ('evaluation', 24)):
        values = np.empty((n, 4), np.float64)
        for axis in range(4):
            streams = {}
            for kind in ('permutation', 'jitter'):
                identity = f'layout:{split}:{axis}:{kind}'
                seeds[identity] = seed(identity)
                streams[kind] = np.random.RandomState(seeds[identity])
            values[:, axis] = -.010 + .020 * (streams['permutation'].permutation(n) + streams['jitter'].uniform(size=n)) / n
        layouts.extend(dict(layout_id=f'{split}_{i:03d}', split=split, candidate_index=i,
                            offsets_m=values[i].tolist()) for i in range(n))
    train = [x for x in layouts if x['split'] == 'train']
    selected = []
    for axis in range(4):
        for sign in (1, -1):
            order = sorted(train, key=lambda x: (sign*x['offsets_m'][axis], x['candidate_index']))
            selected.append(next(x['layout_id'] for x in order if x['layout_id'] not in selected))
    ids = ['center']*4 + selected + ['center']*4
    ids += [x['layout_id'] for x in train if x['layout_id'] not in selected]
    ids += [x['layout_id'] for x in layouts if x['split'] == 'validation']
    episodes = []
    for i, layout_id in enumerate(ids):
        identity = f'expert:{i}'; seeds[identity] = seed(identity)
        episodes.append(dict(episode_index=i, seed=seeds[identity], layout_id=layout_id,
                             split='train' if i < 48 else 'validation', pick='red', place='green',
                             colors=['red', 'green'], instruction=TASK, task_index=0, source_episode_index=7))
    policy = []
    for i in range(3):
        identity = f'policy:{i}'; seeds[identity] = seed(identity); policy.append(seeds[identity])
    expert = [e['seed'] for e in episodes]
    if len(set(seeds.values())) != len(seeds) or set(policy) & (set(expert) | set(range(20260918, 20260938))):
        raise ValueError('Seed collision; stop without redraw')
    return dict(version=VERSION, master_seed=MASTER, rng='numpy.random.RandomState', rng_seeds=seeds,
                seed_derivation='SHA256(namespace:master:identity) first four bytes big endian',
                source=dict(path=SOURCE, sha256=SOURCE_SHA256, episode_index=7, scene_hash=SOURCE_SCENE_SHA256),
                layouts=layouts, episodes=episodes, train_ids=list(range(48)), validation_ids=list(range(48, 60)),
                preparation_ids=list(range(12)), evaluation_layout_ids=[x['layout_id'] for x in layouts if x['split'] == 'evaluation'],
                candidate_episode_ids={x['layout_id']: [e['episode_index'] for e in episodes if e['layout_id'] == x['layout_id']] for x in layouts},
                policy_seeds=policy, expert_attempts=60, replay_modes=['reference', 'persisted_float32'],
                replay_limit=120, unique_layouts=77, reset_policy='only_four_cube_XY_qpos_coordinates_no_resettle',
                expert_version=EXPERT_VERSION, waypoint_jitter_sigma_m=.005, offset_limit_per_axis_m=.010,
                output=str(OUTPUT.relative_to(ROOT)), start_free_bytes=START_BYTES, floor_free_bytes=FLOOR_BYTES,
                max_seconds=MAX_SECONDS, image=IMAGE, expected_episode_frames=528, fps=30, physics_hz=600,
                substeps_per_control=20, state_dim=9, action_dim=8, policy_dtype='float32', camera_size=[512, 512],
                action_semantics='absolute_joint_commands_gripper_0_closed_255_open',
                observation_alignment='observe_command_record_pre_step',
                failure_policy='first_failure_stop_no_retry_no_resume', automatic_training=False,
                validation_is_spatial_holdout=False, statistics_episode_ids=list(range(48)))


def validate_plan(plan):
    if not _same(plan, make_plan()):
        raise ValueError('Noncanonical local-varied plan; no count/seed/layout overrides')