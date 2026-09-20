"""Immutable approved paired-data schedule; stdlib only, no simulator side effects."""
import hashlib
import itertools
import json

COLORS = ("red", "green", "blue", "yellow")
MASTER_SEED = 20260917
VERSION = "paired_demonstrations_v1"
IMAGE = "sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def instruction(pick, place):
    if pick not in COLORS or place not in COLORS or pick == place:
        raise ValueError("Invalid ordered cube pair")
    return f"pick up the {pick} cube and place it on top of the {place} cube"


def make_plan():
    def seed(identity):
        return int.from_bytes(hashlib.sha256(
            f"{VERSION}:{MASTER_SEED}:{identity}".encode()).digest()[:4], "big")

    layouts, episodes, control = [], [], []
    for split, count in (("train", 2), ("heldout", 1)):
        for pair_index, pair in enumerate(itertools.combinations(COLORS, 2)):
            for replicate in range(count):
                identity = f"{split}/{'-'.join(pair)}/{replicate}"
                layouts.append({"layout_id": identity, "ordinal": len(layouts), "split": split,
                                "colors": list(pair), "replicate": replicate, "seed": seed(identity)})
                for direction, ordered in (("forward", pair), ("reverse", pair[::-1])):
                    episode_id = f"{identity}/{direction}"
                    index = len(episodes)
                    episodes.append({"episode_index": index, "episode_id": episode_id,
                                     "layout_id": identity, "split": split, "colors": list(pair),
                                     "direction": direction, "pick": ordered[0], "place": ordered[1],
                                     "instruction": instruction(*ordered), "seed": seed(episode_id),
                                     "task_index": pair_index * 2 + (direction == "reverse")})
                    if split == "train" and direction == ("forward" if replicate == 0 else "reverse"):
                        control.append(index)
    return {"version": VERSION, "master_seed": MASTER_SEED,
            "seed_derivation": "sha256(version:master:identity) first 4 bytes big endian",
            "layouts": layouts, "episodes": episodes,
            "subsets": {"paired": list(range(24)), "control": control, "heldout": list(range(24, 36))},
            "failure_policy": "first_failure_stop_no_retry_no_resume",
            "fps": 30, "chunk_size": 50, "expected_episode_frames": 528}


def validate_plan(plan):
    # Canonical JSON distinguishes bool/float from integer as well as extra fields.
    if digest(plan) != digest(make_plan()):
        raise ValueError("Plan differs from approved paired schedule")


def complete_starts(length, horizon=50):
    if type(length) is not int or length < 0 or type(horizon) is not int or horizon <= 0:
        raise ValueError("Invalid complete-chunk dimensions")
    return range(max(0, length - horizon + 1))


if __name__ == "__main__":
    print(json.dumps(make_plan(), indent=2))