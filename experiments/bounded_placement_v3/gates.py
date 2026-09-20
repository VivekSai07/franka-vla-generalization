"""Measured readiness, finite deadlines and tracking/grasp protection."""
import numpy as np
from .solver import PlanningFailure, finite


def ready(*, grasped, xy, z, hand_error, rotation_error, max_speed, unexpected):
    return bool(np.isfinite([xy, z, hand_error, rotation_error, max_speed]).all()
                and grasped and 0 <= xy <= .01 and .04 <= z <= .09
                and 0 <= hand_error <= .005 and 0 <= rotation_error <= .03
                and 0 <= max_speed <= .10 and not unexpected)


class ReleaseGate:
    def __init__(self):
        self.streak = 0

    def update(self, **measurement):
        self.streak = self.streak+1 if ready(**measurement) else 0
        return self.streak >= 10


class ExecutionGuard:
    def __init__(self, elapsed_ticks):
        self.elapsed_ticks, self.loss_ticks = elapsed_ticks, 0
        self.opened, self.can_advance = False, False

    def check(self, measured, target, grasped, unexpected):
        finite(measured, target)
        if self.elapsed_ticks >= 1200:
            raise PlanningFailure('total_deadline', ticks=self.elapsed_ticks)
        if unexpected:
            raise PlanningFailure('execution_collision')
        self.can_advance = bool(np.max(np.abs(measured-target)) <= .05)
        if not self.opened:
            self.loss_ticks = 0 if grasped else self.loss_ticks+1
            if self.loss_ticks >= 10:
                raise PlanningFailure('sustained_grasp_loss', ticks=self.loss_ticks)

    def tick(self):
        self.elapsed_ticks += 1


def validate_release_geometry(release_hand, relative_cube, base_position):
    center = (release_hand*relative_cube).translation
    finite(center, base_position)
    z = float(center[2]-base_position[2])
    if not .04 <= z <= .09:
        raise PlanningFailure('release_height_contract_conflict', predicted_cube_center_z_difference_m=z,
                              approved_range_m=[.04, .09])
    return z


class PlacementExecution:
    """Consumes one checked target per tick; lag holds do not consume a target.

    Readiness is sampled only after the final target has been sent. At most 60
    additional closed ticks follow arrival. A command decision consumes one
    tick; the caller must perform exactly one control step for that decision.
    """
    def __init__(self, schedule, elapsed_ticks, previous_target=None):
        if not schedule:
            raise PlanningFailure('empty_execution_schedule')
        self.schedule = [(s, q.copy()) for s, q in schedule]
        self.guard = ExecutionGuard(elapsed_ticks)
        self.release = ReleaseGate()
        self.index, self.settle_ticks = -1, 0
        self.current = (schedule[0][1] if previous_target is None else previous_target).copy()
        self.stage = schedule[0][0]

    def command(self, measured_q, measurement):
        self.guard.check(measured_q, self.current, measurement['grasped'], measurement['unexpected'])
        if self.guard.opened:
            raise PlanningFailure('placement_already_released')
        if self.index == len(self.schedule)-1:
            if self.settle_ticks >= 60:
                raise PlanningFailure('placement_not_ready', readiness_streak=self.release.streak,
                                      settling_ticks=self.settle_ticks, measurement=measurement)
            self.settle_ticks += 1
            if self.release.update(**measurement):
                if self.guard.elapsed_ticks+90 > 1200:
                    raise PlanningFailure('post_release_deadline', elapsed_ticks=self.guard.elapsed_ticks)
                self.guard.opened = True
        elif self.guard.can_advance:
            self.index += 1
            self.stage, self.current = self.schedule[self.index]
        self.guard.tick()
        return self.current.copy(), 255 if self.guard.opened else 0, self.stage