"""The four corruptions of the oracle box (arms A6-A9) and the constants that define them, fixed before any trial they govern.

A box is S0's `(u_min, v_min, u_max, v_max, in_frame)` per camera, in [0, 1] (the image divided by 512); `apply` takes the oracle's
(2, 5) pair (front camera, then gripper camera) and returns the (2, 5) pair to draw, always a box `overlay.pixel_box` accepts.
Randomness is seeded by sha256 of (condition, the oracle box's float64 bytes, the tick), so a trial reproduces whatever the worker
order or a resume. Pure numpy: either image, CPU.

* `jitter_small` / `jitter_large`: Gaussian noise of `sigma_px` pixels on each of the four coordinates of each in-frame box, then
  ordered, grown to a minimum side of `MIN_SIDE_PX` about its centre and clipped to the image (shifted inward if that leaves it thinner).
  With `sigma_px` 0 the box is returned unchanged (as for any oracle box at least `MIN_SIDE_PX` wide).
* `dropout`: the gripper-camera box is not drawn (all zeros) with probability `probability` per call; the front box is unchanged.
* `swap`: the distractor's box is drawn on both cameras instead of the target's (the caller supplies it).
"""
import hashlib

import numpy as np

IMAGE = 512
MIN_SIDE_PX = 2.0
IDENTITY = "identity"
CONDITIONS = {
    "jitter_small": dict(kind="jitter", sigma_px=6.0),
    "jitter_large": dict(kind="jitter", sigma_px=18.0),
    "dropout": dict(kind="dropout", probability=0.5),
    "swap": dict(kind="swap"),
}
ARMS = {"A6": "jitter_small", "A7": "jitter_large", "A8": "dropout", "A9": "swap"}


def _pair(box, what="box"):
    out = np.array(box, dtype=np.float64)
    if out.shape != (2, 5) or not np.isfinite(out).all():
        raise ValueError(f"{what} must be a finite (2, 5) pair: the front camera's box, then the gripper camera's")
    if not np.isin(out[:, 4], (0.0, 1.0)).all() or (out[:, :4] < 0).any() or (out[:, :4] > 1).any():
        raise ValueError(f"{what}: coordinates must lie in [0, 1] and in_frame must be 0 or 1")
    return out


def _rng(condition, box, tick):
    material = condition.encode() + b"\0" + np.ascontiguousarray(box, dtype=np.float64).tobytes() + int(tick).to_bytes(8, "little")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _valid_interval(low, high):
    """Ordered, at least MIN_SIDE_PX wide, inside [0, 1]."""
    side = MIN_SIDE_PX / IMAGE
    low, high = min(low, high), max(low, high)
    if high - low < side:
        middle = (low + high) / 2
        low, high = middle - side / 2, middle + side / 2
    low, high = min(max(low, 0.0), 1.0), min(max(high, 0.0), 1.0)
    if high - low < side:
        low = min(low, 1.0 - side)
        high = low + side
    return low, high


def _jitter(box, rng, sigma_px):
    noise = rng.normal(0.0, sigma_px / IMAGE, size=(2, 4))
    out = box.copy()
    if sigma_px == 0:
        return out
    for camera in range(2):
        if box[camera, 4] == 0.0:
            continue
        u0, u1 = _valid_interval(box[camera, 0] + noise[camera, 0], box[camera, 2] + noise[camera, 2])
        v0, v1 = _valid_interval(box[camera, 1] + noise[camera, 1], box[camera, 3] + noise[camera, 3])
        out[camera, :4] = u0, v0, u1, v1
    return out


def apply(condition, box, tick, distractor_box=None):
    """The (2, 5) float64 pair to draw instead of the oracle `box` at `tick` under `condition`. Never modifies `box`."""
    box = _pair(box, "box")
    if condition == IDENTITY:
        return box
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; one of {sorted(CONDITIONS)}")
    spec = CONDITIONS[condition]
    if spec["kind"] == "swap":
        if distractor_box is None:
            raise ValueError("swap needs the distractor's box")
        return _pair(distractor_box, "distractor_box")
    rng = _rng(condition, box, tick)
    if spec["kind"] == "jitter":
        return _jitter(box, rng, spec["sigma_px"])
    out = box.copy()
    if rng.random() < spec["probability"]:
        out[1] = 0.0
    return out


def nearest_distractor(target, positions):
    """The prop other than `target` closest to it in 3D (`positions`: {name: xyz}); a tie goes to the first name in sorted order."""
    if target not in positions or len(positions) < 2:
        raise ValueError("a target and at least one other prop are needed")
    here = np.asarray(positions[target], dtype=np.float64)
    return min(sorted(n for n in positions if n != target), key=lambda n: float(np.linalg.norm(np.asarray(positions[n], dtype=np.float64) - here)))


def iou(a, b):
    """Intersection over union of two single-camera boxes; two absent boxes agree (1.0), one absent box gives 0.0."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a[4] == 0.0 or b[4] == 0.0:
        return 1.0 if a[4] == b[4] else 0.0
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    inside = max(width, 0.0) * max(height, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inside
    return float(inside / union)


def mean_iou(condition, boxes, distractor_boxes=None):
    """Mean over cameras and over `boxes` (tick = position) of the IoU between the corrupted box and the true one."""
    values = []
    for tick, box in enumerate(boxes):
        other = None if distractor_boxes is None else distractor_boxes[tick]
        drawn = apply(condition, box, tick, distractor_box=other)
        values += [iou(drawn[camera], np.asarray(box)[camera]) for camera in range(2)]
    return float(np.mean(values))
