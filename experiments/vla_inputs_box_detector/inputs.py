"""Frame preprocessing, box geometry, exact translation augmentation and the instruction parser of the box detector.

Pure numpy: the cache builder, the trainer, the gate (on raw renders) and the rollout adapter all call these functions, so the frame
the detector sees in training and at rollout is produced by the same code. See `constants.py` for the geometry.
"""
import re

import numpy as np

from experiments.vla_inputs_box_detector import constants as k


def preprocess(images):
    """(front, gripper): the front camera's fixed native-resolution crop (288, 480, 3) and the gripper camera 2 x 2 area-averaged
    (rounded half up) to (256, 256, 3), both uint8, from the two uint8 512 x 512 x 3 renders (front camera first)."""
    if len(images) != 2 or any(not isinstance(i, np.ndarray) or i.dtype != np.uint8 or i.shape != (k.IMAGE, k.IMAGE, 3) for i in images):
        raise ValueError("preprocess needs two uint8 512 x 512 x 3 renders: the front camera, then the gripper camera")
    y0, y1, x0, x1 = k.FRONT_CROP
    front = np.ascontiguousarray(images[0][y0:y1, x0:x1])
    f = k.GRIPPER_FACTOR
    s = k.GRIPPER_SHAPE
    mean = images[1].reshape(s[0], f, s[1], f, 3).astype(np.float32).mean(axis=(1, 3))
    return front, np.floor(mean + 0.5).astype(np.uint8)


def _to_input(box, scale, origin):
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (5,):
        raise ValueError("a box is (u_min, v_min, u_max, v_max, in_frame)")
    if box[4] == 0.0:
        return None
    x0, y0, x1, y1 = box[:4] * k.IMAGE
    return np.array([(x0 - origin[1]) / scale, (y0 - origin[0]) / scale, (x1 - origin[1]) / scale, (y1 - origin[0]) / scale])


def front_box_to_input(box):
    """S0's box -> (x0, y0, x1, y1) in the front crop's pixels, or None when `in_frame` is 0."""
    return _to_input(box, 1.0, (k.FRONT_CROP[0], k.FRONT_CROP[2]))


def gripper_box_to_input(box):
    return _to_input(box, float(k.GRIPPER_FACTOR), (0, 0))


def _from_input(pixels, scale, origin):
    x0, y0, x1, y1 = np.asarray(pixels, dtype=np.float64)
    full = np.clip(np.array([x0 * scale + origin[1], y0 * scale + origin[0], x1 * scale + origin[1], y1 * scale + origin[0]]), 0.0, k.IMAGE)
    if full[2] <= full[0] or full[3] <= full[1]:
        return np.zeros(5)
    return np.concatenate([full / k.IMAGE, [1.0]])


def input_to_front_box(pixels):
    """(x0, y0, x1, y1) in the front crop's pixels -> S0's box in [0, 1] of the full frame, clipped; no area -> all zeros."""
    return _from_input(pixels, 1.0, (k.FRONT_CROP[0], k.FRONT_CROP[2]))


def input_to_gripper_box(pixels):
    return _from_input(pixels, float(k.GRIPPER_FACTOR), (0, 0))


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer number of pixels")
    return int(value)


def shift_image(image, dx, dy):
    """A copy of `image` (H, W, C) with its content moved right by `dx` and down by `dy` pixels (either may be negative); the rest is zero."""
    dx, dy = _integer(dx, "dx"), _integer(dy, "dy")
    height, width = image.shape[:2]
    out = np.zeros_like(image)
    if abs(dx) >= width or abs(dy) >= height:
        return out
    out[max(dy, 0):height + min(dy, 0), max(dx, 0):width + min(dx, 0)] = image[max(-dy, 0):height - max(dy, 0), max(-dx, 0):width - max(dx, 0)]
    return out


def shift_box(box, dx, dy, shape):
    """(box, present): the pixel box `(x0, y0, x1, y1)` moved by (dx, dy), clipped to `shape` (H, W). Absent (zeros) when what remains
    is thinner than `MIN_VISIBLE_PX` in either dimension."""
    dx, dy = _integer(dx, "dx"), _integer(dy, "dy")
    height, width = shape
    moved = np.asarray(box, dtype=np.float64) + np.array([dx, dy, dx, dy])
    clipped = np.array([max(moved[0], 0.0), max(moved[1], 0.0), min(moved[2], float(width)), min(moved[3], float(height))])
    if clipped[2] - clipped[0] < k.MIN_VISIBLE_PX or clipped[3] - clipped[1] < k.MIN_VISIBLE_PX:
        return np.zeros(4), False
    return clipped, True


_INSTRUCTION = re.compile(r"^put the (" + "|".join(k.OBJECTS) + r") in the (" + "|".join(k.BINS) + r") bin$")


def parse_instruction(text):
    """Index into `OBJECTS` of the object `put the <object> in the <colour> bin` names; anything else raises."""
    match = _INSTRUCTION.match(text) if isinstance(text, str) else None
    if not match:
        raise ValueError(f"not an instruction of the form 'put the <object> in the <colour> bin': {text!r}")
    return k.OBJECTS.index(match.group(1))
