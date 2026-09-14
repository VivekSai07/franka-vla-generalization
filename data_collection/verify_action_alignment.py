"""
Verify that recorded actions predict the NEXT state, not the current one.

Run this against a small dataset collected with the fixed collectors to
confirm the action/observation timestep-alignment fix
(docs/superpowers/specs/2026-09-14-fix-action-observation-alignment-design.md)
actually changed what's on disk.

Usage:
    python data_collection/verify_action_alignment.py --root <path_to_dataset_dir> --repo-id <repo_id>
"""
import argparse
import sys

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Verify action/observation alignment")
    parser.add_argument("--root", type=str, required=True, help="Local dataset directory")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo_id used at collection time")
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    num_episodes = dataset.meta.total_episodes
    if num_episodes == 0:
        print("FAIL: dataset has zero episodes — nothing to verify.")
        sys.exit(1)

    diffs_action_to_current_state = []
    diffs_action_to_next_state = []

    for ep_idx in range(num_episodes):
        ep = dataset.meta.episodes[ep_idx]
        start = ep["dataset_from_index"]
        end = ep["dataset_to_index"]

        states = np.stack([dataset[i]["observation.state"][:7].numpy() for i in range(start, end)])
        actions = np.stack([dataset[i]["action"][:7].numpy() for i in range(start, end)])

        # Compare action_t against state_t (current) and state_{t+1} (next),
        # for every t where a "next" frame exists in this episode.
        diffs_action_to_current_state.append(np.abs(actions[:-1] - states[:-1]).mean())
        diffs_action_to_next_state.append(np.abs(actions[:-1] - states[1:]).mean())

    mean_to_current = float(np.mean(diffs_action_to_current_state))
    mean_to_next = float(np.mean(diffs_action_to_next_state))

    print(f"mean |action_t - state_t|     (current, should be LARGER): {mean_to_current:.6f}")
    print(f"mean |action_t - state_t+1|   (next, should be SMALLER):   {mean_to_next:.6f}")

    if mean_to_next < mean_to_current:
        print("PASS: recorded actions are closer to the NEXT state than the current one — alignment fix confirmed.")
        sys.exit(0)
    else:
        print("FAIL: recorded actions are closer to the CURRENT state — stale-action bug may still be present.")
        sys.exit(1)


if __name__ == "__main__":
    main()
