"""Draw the target's box on a camera frame: a magenta band outside the box, the only thing that differs from A2's input.

Pure: numpy and torch only, so the admission image (the training loader) and the robosuite image (the rollout adapter) run the
very same code. The box is S0's `(u_min, v_min, u_max, v_max, in_frame)` in [0, 1] (u right, v down, divided by 512); it is
turned into the half-open pixel box `[floor(512 u_min), ceil(512 u_max)) x [floor(512 v_min), ceil(512 v_max))`, which contains
every pixel the box touches. The band is `THICKNESS` pixels wide, drawn outside that box (so it covers background, not the target),
clipped to the image; nothing is drawn when `in_frame` is 0. Frames are float32 CHW in [0, 1] and are returned as copies.
"""
import math

import numpy as np
import torch

THICKNESS = 4
COLOUR = (1.0, 0.0, 1.0)
SIZE = 512
FRONT, GRIPPER = 'observation.images.image', 'observation.images.image2'


def pixel_box(box):
    """(x0, y0, x1, y1) half-open pixel box, or None when `in_frame` is 0. Anything not a valid box raises."""
    values = np.asarray(box, dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError('A box is five finite values: u_min, v_min, u_max, v_max, in_frame')
    u_min, v_min, u_max, v_max, in_frame = (float(x) for x in values)
    if not all(0.0 <= x <= 1.0 for x in (u_min, v_min, u_max, v_max)) or in_frame not in (0.0, 1.0):
        raise ValueError('Box coordinates must lie in [0, 1] and in_frame must be 0 or 1')
    if in_frame == 0.0:
        return None
    x0, y0 = math.floor(SIZE*u_min), math.floor(SIZE*v_min)
    x1, y1 = math.ceil(SIZE*u_max), math.ceil(SIZE*v_max)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('An in-frame box must cover at least one pixel')
    return x0, y0, x1, y1


def band(box):
    """(512, 512) bool, indexed [y, x]: the pixels the overlay paints for this box."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    pixels = pixel_box(box)
    if pixels is None:
        return mask
    x0, y0, x1, y1 = pixels
    mask[max(y0-THICKNESS, 0):min(y1+THICKNESS, SIZE), max(x0-THICKNESS, 0):min(x1+THICKNESS, SIZE)] = True
    mask[y0:y1, x0:x1] = False
    return mask


def draw(image, box):
    """A copy of `image` (float32 (3, 512, 512) in [0, 1]) with the box's band painted magenta."""
    if (not isinstance(image, torch.Tensor) or image.dtype != torch.float32 or tuple(image.shape) != (3, SIZE, SIZE)
            or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1):
        raise ValueError('Overlay needs a finite float32 (3, 512, 512) frame in [0, 1]')
    mask = torch.from_numpy(band(box))
    out = image.clone()
    out[:, mask] = torch.tensor(COLOUR, dtype=torch.float32).unsqueeze(1)
    return out


def near_magenta(images, tolerance):
    """How many pixels of `images` (float32, channels first, any leading dimensions) are within `tolerance` of the band colour.

    L-infinity distance over the three channels. Used on frames that are NOT drawn, to ask whether the scene already contains
    the band's colour.
    """
    reference = torch.tensor(COLOUR, dtype=images.dtype).reshape(3, 1, 1)
    return int(((images - reference).abs() <= tolerance).all(dim=-3).sum())


def draw_frame(frame, target_box):
    """`frame` with the front-camera box drawn on `observation.images.image` and the gripper-camera box on `...image2`.

    `target_box` is (2, 5): the front camera's box then the gripper camera's. Everything else passes through; the input
    mapping and its tensors are untouched.
    """
    boxes = np.asarray(target_box)
    if boxes.shape != (2, 5):
        raise ValueError('target_box must be (2, 5): the front camera, then the gripper camera')
    if FRONT not in frame or GRIPPER not in frame:
        raise ValueError('Both camera frames are required')
    out = dict(frame)
    out[FRONT] = draw(frame[FRONT], boxes[0])
    out[GRIPPER] = draw(frame[GRIPPER], boxes[1])
    return out
