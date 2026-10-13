"""P2 expert: the accepted P0 `PropsExpert`, unchanged except the carry height.

During `lift` and `move` the held object's lowest point is carried at `plan.CARRY_BOTTOM_M` (0.19 m:
the tallest prop plus 3 cm) instead of P0's wall + 0.05 = 0.12 m, so it clears a standing distractor.
Every other phase, the grasp plan, durations and the success rule are P0's. Robosuite image.
"""
import numpy as np

from experiments.props_p2_data.plan import CARRY_BOTTOM_M
from experiments.props_scene.expert import PropsExpert

EXPERT_VERSION = "props_p2_expert_v1"


class P2Expert(PropsExpert):
    def __init__(self, objects, blue_slot):
        super().__init__(objects, blue_slot)
        self.config["expert_version"] = EXPERT_VERSION

    def _target(self, name, data, hand, obj, plan, grasp_rotation, home_pos, home_rotation, bin_xy, held):
        if name == "lift":
            held.update(self._measure(data, hand, obj))
            held["bottom_at_lift"] = held["bottom"]
            held["lift_rise"] = CARRY_BOTTOM_M - held["bottom"]
            return np.array([hand[0], hand[1], CARRY_BOTTOM_M + held["d"]]), grasp_rotation, False
        if name == "move":
            held.update(self._measure(data, hand, obj))
            xy = bin_xy - held["off"]
            return np.array([xy[0], xy[1], CARRY_BOTTOM_M + held["d"]]), grasp_rotation, True
        return super()._target(name, data, hand, obj, plan, grasp_rotation, home_pos, home_rotation, bin_xy, held)
