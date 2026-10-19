"""Fresh paired layouts for the P2 re-test of checkpoint 101,520 on milk and bread. Pure; admission image.

120 layouts per object, each run under two bin instructions (trained and held-out) with identical geometry, so a layout
carries both arms of a pair. Written in the format of the sealed P2 plan (`evaluation_holdout`, `evaluation_report_ids`),
which is what the unchanged P2 evaluator reads. Samplers are P2's own (`props_p2_data.plan`, imported read-only), drawn under
identities `retest:...`, a namespace P2 never used. Design: docs/superpowers/specs/2026-10-19-props-p2-retest-design.md.
"""
import hashlib
import json

from experiments.props_p2_data import plan as p2

OBJECTS = ("milk", "bread")
PER_OBJECT = 120
ARMS = ("trained", "held")
# (trained bin, held-out bin): milk is trained into orange and held out of blue; bread the reverse
BINS = {"milk": ("orange", "blue"), "bread": ("blue", "orange")}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _count_and_side(i):
    """Index i -> (distractor count, blue slot): the four combinations cycle, so each gets PER_OBJECT / 4."""
    return (1, 2)[i % 2], ("left", "right")[(i // 2) % 2]


def make_plan():
    seeds = {}
    holdout, pairs, ids = {}, [], []
    for obj in OBJECTS:
        decks = {count: p2._deck(obj, count, PER_OBJECT // 2, f"retest:{obj}:deck{count}", seeds) for count in (1, 2)}
        taken = {1: 0, 2: 0}
        for i in range(PER_OBJECT):
            count, side = _count_and_side(i)
            distractors = decks[count][taken[count]]
            taken[count] += 1
            identity = f"retest:{obj}:{i}"
            geometry = dict(blue_slot=side, distractor_count=count, **p2._draw_layout(obj, distractors, identity, seeds),
                            home_pose_jitter_deg=p2._jitter(identity, seeds))
            expert_seed = p2.seed(f"{identity}:expert")
            seeds[f"{identity}:expert"] = expert_seed
            pair = dict(pair_id=f"retest_{obj}_{i:03d}", object=obj)
            for arm, bin_colour in zip(ARMS, BINS[obj]):
                key = f"{pair['pair_id']}_{arm}"
                holdout[key] = dict(tier="report", layout_id=key, held_out=(obj, bin_colour) in p2.HELD_OUT_PAIRS,
                                    object=obj, bin=bin_colour, cell=[obj, bin_colour, side],
                                    instruction=p2.instruction(obj, bin_colour), seed=expert_seed, **geometry)
                pair[arm] = key
                ids.append(key)
            pairs.append(pair)
    return dict(version="props_p2_retest_v1", master_namespace="retest", spec="docs/superpowers/specs/2026-10-19-props-p2-retest-design.md",
                evaluation_holdout=holdout, evaluation_report_ids=ids, pairs=pairs, seeds=seeds)
