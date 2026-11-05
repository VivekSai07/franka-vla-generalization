"""Inference with the trained box detector: the one place that turns two raw renders and an instruction into the (2, 5) box pair.

Used by the gate (on raw renders and on cached frames) and by the rollout adapter of arm A5, so the gate measures exactly the code that
runs in a rollout. Preprocessing is `inputs.preprocess`, the network is `model.BoxDetector`, the box is decoded at the peak of the
instructed object's heat-map channel, mapped back to the full 512 x 512 frame and clipped (`inputs.input_to_*_box`), and a camera whose
peak logit is below the presence threshold gets the absent box (all zeros). Float32 inference; weights are loaded with
`weights_only=True`.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.vla_inputs_box_detector import constants as k
from experiments.vla_inputs_box_detector import inputs, model

WEIGHTS, DESCRIPTION = "detector.pt", "detector.json"


def constants_digest():
    """sha256 of every pinned constant (the upper-case names of `constants.py`): a detector is only valid under the constants it trained with."""
    values = {name: getattr(k, name) for name in dir(k) if name.isupper()}
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=list).encode()).hexdigest()


class Predictor:
    def __init__(self, net, device="cpu"):
        self.net, self.device = net.to(device).eval(), torch.device(device)

    @classmethod
    def load(cls, directory, device="cpu"):
        """The detector saved in `directory` (`detector.pt` and `detector.json`); refuses one trained under other constants."""
        directory = Path(directory)
        description = json.loads((directory / DESCRIPTION).read_text())
        if description["constants_sha256"] != constants_digest():
            raise ValueError(f"{directory} was trained under other constants than the current ones")
        net = model.BoxDetector(tuple(description.get("widths", k.WIDTHS)), description.get("head_channels", k.HEAD_CHANNELS))
        net.load_state_dict(torch.load(directory / WEIGHTS, map_location="cpu", weights_only=True)["state"])
        return cls(net, device)

    @torch.inference_mode()
    def predict_arrays(self, fronts, grippers, classes):
        """(boxes (N, 2, 5) float64, peak logits (N, 2) float32) for preprocessed uint8 frames (N, 288, 480, 3) and (N, 256, 256, 3)
        and the instructed object indices (N,)."""
        front_out, gripper_out = self.net(model.normalise(torch.as_tensor(fronts).to(self.device)), model.normalise(torch.as_tensor(grippers).to(self.device)))
        classes = torch.as_tensor(classes, dtype=torch.long, device=self.device)
        front_px, front_logit = model.decode(front_out, classes)
        gripper_px, gripper_logit = model.decode(gripper_out, classes)
        logits = torch.stack([front_logit, gripper_logit], dim=1).float().cpu()
        present = model.is_present(logits).numpy()
        out = np.zeros((len(logits), 2, 5))
        for i in range(len(logits)):
            if present[i, 0]:
                out[i, 0] = inputs.input_to_front_box(front_px[i].cpu().numpy())
            if present[i, 1]:
                out[i, 1] = inputs.input_to_gripper_box(gripper_px[i].cpu().numpy())
        return out, logits.numpy()

    def predict(self, images, instruction):
        """(2, 5) float64 boxes (front camera, then gripper camera) for two uint8 512 x 512 x 3 renders and the instruction."""
        front, gripper = inputs.preprocess(images)
        boxes, _ = self.predict_arrays(front[None], gripper[None], [inputs.parse_instruction(instruction)])
        return boxes[0]
