"""Small, explicit contracts for the approved local 20k-update pilot."""
import math

import torch


def checked_splits(manifest, total_episodes):
    train = list(manifest["splits"]["train"])
    validation = list(manifest["splits"]["validation"])
    all_ids = train + validation
    if (len(train) != 300 or len(validation) != 120 or total_episodes != 420
            or any(type(i) is not int for i in all_ids)
            or len(set(all_ids)) != 420 or set(all_ids) != set(range(total_episodes))):
        raise ValueError("Pilot requires disjoint complete 300/120 episode IDs")
    return train, validation


def frame_indices(episodes, selected):
    return [i for ep in selected for i in range(int(episodes[ep]["dataset_from_index"]),
                                               int(episodes[ep]["dataset_to_index"]))]


def probe_indices(episodes, selected):
    indices = []
    for ep in selected:
        start, end = int(episodes[ep]["dataset_from_index"]), int(episodes[ep]["dataset_to_index"])
        for offset in (60, 107, 347):
            if start + offset + 50 > end:
                raise ValueError("Fixed validation probe would cross episode boundary")
            indices.append(start + offset)
    return indices


def learning_rate(update):
    if type(update) is not int or not 1 <= update <= 20000:
        raise ValueError("Update must be in [1, 20000]")
    if update <= 1000:
        return 1e-4 * update / 1000
    progress = (update - 1000) / 19000
    return 1e-5 + .5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * progress))


def check_batch(batch):
    shapes = {"action": (8, 50, 8), "observation.state": (8, 9),
              "observation.images.image": (8, 3, 512, 512),
              "observation.images.image2": (8, 3, 512, 512)}
    for key, shape in shapes.items():
        if key not in batch or tuple(batch[key].shape) != shape or not torch.isfinite(batch[key]).all():
            raise ValueError(f"Invalid physical training batch: {key}")
    mask = batch.get("action_is_pad")
    if mask is None or tuple(mask.shape) != (8, 50) or mask.any():
        raise ValueError("Missing or nonempty action padding mask")


def backward_microbatch(loss):
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("Nonfinite or nonscalar loss")
    (loss / 4).backward()