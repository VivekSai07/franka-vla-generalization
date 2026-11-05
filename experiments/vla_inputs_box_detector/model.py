"""The box detector (M3b spec D4): a class-conditional, two-camera centre-point network, trained from scratch (no downloaded weights).

Each camera has its own ResNet-18-shaped backbone with a small top-down neck to stride 4 and three heads: a heat map with one channel
per object, a sub-cell centre offset and a box size. The network does not take the instruction: the instruction names the object, and
the object names the heat-map channel that is trained and read (`make_target`, `loss`, `decode`), so only the instructed object's
channel is ever supervised. The presence of the target is the sigmoid of that channel's peak logit against `PRESENCE_THRESHOLD`.
Fully convolutional; the real inputs are 288 x 480 (front crop) and 256 x 256 (gripper, 2 x 2 averaged), see `constants.py`.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.vla_inputs_box_detector import constants as k

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
OBJECT_COUNT = len(k.OBJECTS)


def standardise(x):
    """float32 (N, 3, H, W) in [0, 1] -> standardised with `MEAN`, `STD`."""
    mean = torch.tensor(MEAN, dtype=torch.float32, device=x.device).reshape(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32, device=x.device).reshape(1, 3, 1, 1)
    return (x - mean) / std


def normalise(images):
    """uint8 (N, H, W, 3) tensor -> float32 (N, 3, H, W), divided by 255 and standardised with `MEAN`, `STD`."""
    return standardise(images.permute(0, 3, 1, 2).float() / 255.0)


class Block(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.first = nn.Sequential(nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))
        self.second = nn.Sequential(nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False), nn.BatchNorm2d(outputs))
        self.skip = None if stride == 1 and inputs == outputs else nn.Sequential(nn.Conv2d(inputs, outputs, 1, stride, bias=False), nn.BatchNorm2d(outputs))

    def forward(self, x):
        out = self.second(self.first(x))
        return F.relu(out + (x if self.skip is None else self.skip(x)))


def stage(inputs, outputs, stride):
    return nn.Sequential(Block(inputs, outputs, stride), Block(outputs, outputs, 1))


def conv_bn_relu(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, 1, 1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))


class CameraNet(nn.Module):
    def __init__(self, widths=k.WIDTHS, head_channels=k.HEAD_CHANNELS, objects=OBJECT_COUNT):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 7, 2, 3, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.layers = nn.ModuleList([stage(widths[0], widths[0], 1), stage(widths[0], widths[1], 2), stage(widths[1], widths[2], 2),
                                     stage(widths[2], widths[3], 2)])
        self.lateral = nn.ModuleList([nn.Conv2d(w, head_channels, 1) for w in widths])
        self.smooth = conv_bn_relu(head_channels, head_channels)
        self.heat = nn.Sequential(conv_bn_relu(head_channels, head_channels), nn.Conv2d(head_channels, objects, 1))
        self.offset = nn.Sequential(conv_bn_relu(head_channels, head_channels), nn.Conv2d(head_channels, 2, 1))
        self.size = nn.Sequential(conv_bn_relu(head_channels, head_channels), nn.Conv2d(head_channels, 2, 1))
        nn.init.constant_(self.heat[-1].bias, -2.19)  # a prior of 0.1 per cell: the usual focal-loss start

    def forward(self, x):
        features, y = [], self.stem(x)
        for layer in self.layers:
            y = layer(y)
            features.append(y)
        top = self.lateral[3](features[3])
        for level in (2, 1, 0):
            top = self.lateral[level](features[level]) + F.interpolate(top, size=features[level].shape[-2:], mode="nearest")
        top = self.smooth(top)
        return dict(heat=self.heat(top), offset=self.offset(top), size=self.size(top))


class BoxDetector(nn.Module):
    def __init__(self, widths=k.WIDTHS, head_channels=k.HEAD_CHANNELS):
        super().__init__()
        self.front = CameraNet(widths, head_channels)
        self.gripper = CameraNet(widths, head_channels)

    def forward(self, front, gripper):
        return self.front(front), self.gripper(gripper)


def count_parameters(net):
    return sum(p.numel() for p in net.parameters())


# --- targets -------------------------------------------------------------------------------------------------------------------

def make_target(box, shape):
    """The training target of one camera's frame: `box` (x0, y0, x1, y1) in input pixels or None (absent), `shape` (H, W) of the input.

    A Gaussian heat map (peak exactly 1 at the cell holding the box centre; sigma grows with the smaller side), the centre cell, the
    sub-cell offset of the centre and the size in units of `SIZE_UNIT_PX`. An absent box has an empty heat map and no box target.
    """
    height, width = shape[0] // k.HEAD_STRIDE, shape[1] // k.HEAD_STRIDE
    heat = np.zeros((height, width), np.float32)
    if box is None:
        return dict(heat=heat, present=False, iy=0, ix=0, offset=np.zeros(2, np.float32), size=np.zeros(2, np.float32))
    x0, y0, x1, y1 = (float(v) for v in box)
    cx, cy = (x0 + x1) / 2 / k.HEAD_STRIDE, (y0 + y1) / 2 / k.HEAD_STRIDE
    ix, iy = min(max(int(math.floor(cx)), 0), width - 1), min(max(int(math.floor(cy)), 0), height - 1)
    sigma = max(1.0, min(x1 - x0, y1 - y0) / k.HEAD_STRIDE / 6.0)
    ys, xs = np.mgrid[0:height, 0:width]
    heat[:] = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2.0 * sigma ** 2))
    heat[iy, ix] = 1.0
    return dict(heat=heat, present=True, iy=iy, ix=ix, offset=np.array([cx - ix, cy - iy], np.float32),
                size=np.array([x1 - x0, y1 - y0], np.float32) / k.SIZE_UNIT_PX)


def collate_targets(targets, classes):
    """Stack per-sample targets and the instructed object indices into batch tensors."""
    return dict(heat=torch.from_numpy(np.stack([t["heat"] for t in targets])), present=torch.tensor([t["present"] for t in targets]),
                index=torch.tensor([[t["iy"], t["ix"]] for t in targets], dtype=torch.long),
                offset=torch.from_numpy(np.stack([t["offset"] for t in targets])), size=torch.from_numpy(np.stack([t["size"] for t in targets])),
                cls=torch.as_tensor(classes, dtype=torch.long))


def _gather(maps, index):
    """(B, C, H, W) maps at each sample's (iy, ix) -> (B, C)."""
    batch = torch.arange(maps.shape[0])
    return maps[batch, :, index[:, 0], index[:, 1]]


def loss(outputs, targets):
    """Focal loss on the instructed object's heat-map channel plus L1 on the centre offset and the size at the centre cell of a present
    box; an absent frame has only the focal (all-negative) term. Normalised by the number of present boxes in the batch (at least 1)."""
    batch = torch.arange(outputs["heat"].shape[0])
    logits = outputs["heat"][batch, targets["cls"]].float()
    target = targets["heat"].to(logits.device)
    positive = (target == 1.0).float()
    log_p, log_not_p = F.logsigmoid(logits), F.logsigmoid(-logits)
    focal = -(positive * (1 - torch.sigmoid(logits)) ** k.FOCAL_ALPHA * log_p
              + (1 - positive) * (1 - target) ** k.FOCAL_BETA * torch.sigmoid(logits) ** k.FOCAL_ALPHA * log_not_p).sum()
    present = targets["present"].to(logits.device)
    count = torch.clamp(present.sum().float(), min=1.0)
    index = targets["index"].to(logits.device)
    mask = present.float().unsqueeze(1)
    off = (torch.abs(_gather(outputs["offset"], index).float() - targets["offset"].to(logits.device)) * mask).sum()
    size = (torch.abs(_gather(outputs["size"], index).float() - targets["size"].to(logits.device)) * mask).sum()
    heat, box = focal / count, (k.OFFSET_WEIGHT * off + k.SIZE_WEIGHT * size) / count
    return dict(total=heat + box, heat=heat, box=box)


def decode(outputs, classes):
    """(boxes, peak logits): per sample the box (x0, y0, x1, y1) in input pixels at the peak of the instructed object's channel, and that
    peak's logit (`is_present` thresholds it)."""
    classes = torch.as_tensor(classes, dtype=torch.long, device=outputs["heat"].device)
    batch = torch.arange(outputs["heat"].shape[0], device=classes.device)
    heat = outputs["heat"][batch, classes]
    width = heat.shape[-1]
    peak, flat = heat.reshape(heat.shape[0], -1).max(dim=1)
    index = torch.stack([flat // width, flat % width], dim=1)
    offset, size = _gather(outputs["offset"], index), _gather(outputs["size"], index) * k.SIZE_UNIT_PX
    cx, cy = (index[:, 1].float() + offset[:, 0]) * k.HEAD_STRIDE, (index[:, 0].float() + offset[:, 1]) * k.HEAD_STRIDE
    boxes = torch.stack([cx - size[:, 0] / 2, cy - size[:, 1] / 2, cx + size[:, 0] / 2, cy + size[:, 1] / 2], dim=1)
    return boxes, peak


def is_present(logits):
    return torch.sigmoid(logits) >= k.PRESENCE_THRESHOLD


def pixel_iou(a, b):
    """IoU of two pixel boxes (x0, y0, x1, y1)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    inside = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    union = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0) + max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0) - inside
    return float(inside / union) if union > 0 else 0.0
