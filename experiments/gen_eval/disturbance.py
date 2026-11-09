"""When a G0 disturbance fires. Pure; imports no simulator.

`move`: at the start of the first tick at which the grasp centre (the midpoint of the two fingers) is within 6 cm horizontally of
the target and no close command has been applied, the target is put at the condition's `move_to` (8 cm from its start).
`drop`: at the start of the first tick at which the target is 5 cm or more above where it started, the gripper command is forced
open for 10 ticks (that tick and the nine after it); the policy keeps control of the arm and sees the scene throughout.
Each fires at most once. A "close command" is an applied gripper command at or below the open threshold
(`props_scene.constants.GRIPPER_OPEN_CTRL`), the same threshold the success rule uses for "open".
"""
import math

from experiments.gen_scene import levels
from experiments.props_scene import constants as psc

OPEN_THRESHOLD = psc.GRIPPER_OPEN_CTRL
FORCED_OPEN_COMMAND = 255.0   # the expert's open command


class Disturbance:
    def __init__(self, condition):
        self.kind = condition.disturbance
        self.move_to = None if condition.move_to is None else tuple(condition.move_to)
        self.fired = False
        self.tick = None
        self._closed = False

    def note_applied_gripper(self, value):
        """Record the gripper command the physics received this tick."""
        if float(value) <= OPEN_THRESHOLD:
            self._closed = True

    def move_due(self, tick, grasp_xy, target_xy):
        if self.kind != "move" or self.fired or self._closed:
            return False
        return bool(math.hypot(grasp_xy[0] - target_xy[0], grasp_xy[1] - target_xy[1]) <= levels.MOVE_TRIGGER_M)

    def drop_due(self, tick, target_z, start_z):
        if self.kind != "drop" or self.fired:
            return False
        return bool(target_z - start_z >= levels.DROP_LIFT_M)

    def fire(self, tick):
        if self.fired:
            raise RuntimeError("a disturbance fires at most once")
        self.fired, self.tick = True, int(tick)

    def forced_open(self, tick):
        """True on the ticks at which a fired `drop` overrides the policy's gripper command."""
        return bool(self.kind == "drop" and self.fired and self.tick <= tick < self.tick + levels.DROP_OPEN_TICKS)

    def summary(self):
        return dict(disturbance_fired=bool(self.fired), disturbance_tick=self.tick)
