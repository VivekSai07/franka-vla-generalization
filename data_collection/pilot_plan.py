"""Canonical balanced-pilot manifests; standard library only, no simulation.

``slots`` contains training demonstrations first, then validation demonstrations.
``splits`` maps those split names to ordered slot-ID lists. ``scene_slots`` is a
separate list (120 tuning, then 240 final), never demonstration rows. Scene seeds
must not be retried or selected by expert/policy success. Distinct seeds do not
prove geometric independence; initialization/duplicate-scene checks belong to
the eventual collector, not this manifest generator.

Version changes are required for any change to the canonical ordering, quotas,
IDs or seed algorithm. Preparation is separate from, not a subset of, the pilot.
"""
import argparse
import hashlib
import json


VERSION = "balanced_pilot_v1"
MAX_ATTEMPTS_PER_SLOT = 10
_COLORS = ("red", "green", "blue", "yellow")
_PAIRS = tuple((pick, place) for pick in _COLORS for place in _COLORS if pick != place)
_MODES = ("pilot", "preparation")
_STREAMS = ("train", "validation", "tuning", "final")
_UINT32_MAX = 2**32 - 1
_STREAM_SIZE = 2**20


def _require_uint32(seed):
    if type(seed) is not int or not 0 <= seed <= _UINT32_MAX:
        raise ValueError("seed must be an exact int in 0..2**32-1 (not bool)")


def _quotas(mode, split, pair_id):
    if mode == "preparation" and split in ("train", "validation"):
        assigned = "train" if pair_id < 9 else "validation"
        return [int(split == assigned and count == pair_id % 3) for count in range(3)]
    base = {"train": 8, "validation": 3, "tuning": 3, "final": 6}[split]
    counts = [base, base, base]
    counts[pair_id % 3] += 1
    if split == "final":
        counts[(pair_id + 1) % 3] += 1
    return counts


def make_plan(mode: str, seed: int) -> dict:
    """Return a fresh canonical pilot (420-demo) or preparation (12-demo) plan.

    Both modes include independent 120-tuning/240-final scene slots. Invalid
    modes or non-exact/out-of-range uint32 master seeds raise ValueError.
    """
    if type(mode) is not str or mode not in _MODES:
        raise ValueError("mode must be 'pilot' or 'preparation'")
    _require_uint32(seed)

    # Hash-derived affine permutation of uint32 counters: an odd multiplier is
    # invertible modulo 2**32, so distinct counters cannot collide. Partition by
    # mode and stream BEFORE permutation; do not hash each attempt independently
    # into uint32 (which would introduce birthday collisions). These are layout
    # seeds, not cryptographic secrets or proof of independent sampled geometry.
    key = hashlib.sha256(f"{VERSION}:seeds:{seed}".encode("ascii")).digest()
    multiplier = int.from_bytes(key[:4], "big") | 1
    offset = int.from_bytes(key[4:8], "big")
    seen_seeds = set()
    streams = {}
    for stream_id, split in enumerate(_STREAMS):
        partition = (_MODES.index(mode) * len(_STREAMS) + stream_id) * _STREAM_SIZE
        retries = MAX_ATTEMPTS_PER_SLOT if split in ("train", "validation") else 1
        slots = []
        for pair_id, (pick, place) in enumerate(_PAIRS):
            eligible = [color for color in _COLORS if color not in (pick, place)]
            identity_key = f"{VERSION}:extras:{mode}:{seed}:{split}:{pair_id}"
            if hashlib.sha256(identity_key.encode("ascii")).digest()[0] & 1:
                eligible.reverse()
            for count, quota in enumerate(_quotas(mode, split, pair_id)):
                for occurrence in range(quota):
                    extras = [eligible[occurrence % 2]] if count == 1 else eligible[:count]
                    index = len(slots)
                    retry_seeds = []
                    for retry in range(retries):
                        local = index * MAX_ATTEMPTS_PER_SLOT + retry
                        if local >= _STREAM_SIZE:
                            raise ValueError("Seed stream partition exhausted")
                        attempt_seed = (multiplier * (partition + local) + offset) & _UINT32_MAX
                        if attempt_seed in seen_seeds:
                            raise ValueError("Seed collision across attempt/scene streams")
                        seen_seeds.add(attempt_seed)
                        retry_seeds.append(attempt_seed)
                    slots.append({
                        "slot_id": f"{mode}/{split}/{index:04d}",
                        "split": split,
                        "pair_id": pair_id,
                        "pick": pick,
                        "place": place,
                        "colors": [pick, place] + extras,
                        "retry_seeds": retry_seeds,
                    })
        streams[split] = slots

    # Never interleave validation or scenes into the first 300 pilot IDs.
    return {
        "version": VERSION,
        "mode": mode,
        "seed": seed,
        "max_attempts_per_slot": MAX_ATTEMPTS_PER_SLOT,
        "slots": streams["train"] + streams["validation"],
        "splits": {split: [slot["slot_id"] for slot in streams[split]]
                   for split in ("train", "validation")},
        "scene_slots": streams["tuning"] + streams["final"],
    }


def _canonical_equal(actual, expected):
    # Python equality alone accepts True == 1 and 1.0 == 1. Reject those and
    # non-JSON containers while ignoring dictionary insertion order only.
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (actual.keys() == expected.keys()
                and all(_canonical_equal(actual[key], value) for key, value in expected.items()))
    if type(expected) is list:
        return (len(actual) == len(expected)
                and all(_canonical_equal(a, b) for a, b in zip(actual, expected)))
    return actual == expected


def validate_plan(plan: dict) -> None:
    """Return None only for a type-exact canonical regeneration; else ValueError.

    Reject altered seeds, identities, order, split assignments, missing/extra
    fields and unsupported versions, not merely incorrect aggregate quotas.
    This is integrity checking, not authentication of an externally pinned seed.
    """
    if type(plan) is not dict:
        raise ValueError("plan must be a dict")
    expected = make_plan(plan.get("mode"), plan.get("seed"))
    if not _canonical_equal(plan, expected):
        raise ValueError("Plan does not match canonical regeneration")


def _canonical_json(plan):
    return json.dumps(plan, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def plan_digest(plan: dict) -> str:
    """Validate, then return lowercase SHA-256 of compact sorted-key UTF-8 JSON.

    Dictionary key order and file whitespace are excluded; list order is not.
    The canonical byte string has no trailing newline and uses ASCII escaping.
    """
    validate_plan(plan)
    return hashlib.sha256(_canonical_json(plan).encode("utf-8")).hexdigest()


def slot_seed(slot: dict, retry: int) -> int:
    """Read a zero-based bounded attempt seed without changing the condition.

    Demo retries are 0..9; tuning/final slots permit only 0. Invalid retry
    indices or malformed seed lists raise ValueError. Validate the whole plan
    first when consuming externally supplied slots.
    """
    if type(retry) is not int or not 0 <= retry < MAX_ATTEMPTS_PER_SLOT:
        raise ValueError("retry must be an exact int in 0..9 (not bool)")
    if type(slot) is not dict or slot.get("split") not in _STREAMS:
        raise ValueError("slot must have a recognized split")
    seeds = slot.get("retry_seeds")
    expected = 1 if slot["split"] in ("tuning", "final") else MAX_ATTEMPTS_PER_SLOT
    if type(seeds) is not list or len(seeds) != expected:
        raise ValueError("slot has an invalid number of retry seeds")
    for seed in seeds:
        _require_uint32(seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("slot contains duplicate retry seeds")
    if retry >= len(seeds):
        raise ValueError("Scene slots have one seed and do not permit success retries")
    return seeds[retry]


def main() -> None:
    """Write a fresh manifest using exclusive creation; never overwrite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        plan = make_plan(args.mode, args.seed)
        validate_plan(plan)
        content = _canonical_json(plan) + "\n"
        with open(args.output, "x", encoding="utf-8") as stream:
            stream.write(content)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()