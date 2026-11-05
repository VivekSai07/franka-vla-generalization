"""Arms A6-A9 of the M0 evaluator: A3's frozen policy fed a corrupted oracle box (M3b, spec D2/D3). No policy is trained.

M0 loads an arm by `module:Class` from a module named `experiments.vla_inputs_arm_*` and constructs it from the checkpoint path alone,
so each condition is a class: `XVLAJitterSmall` (A6), `XVLAJitterLarge` (A7), `XVLADropout` (A8), `XVLASwap` (A9). All of them are A3's
`XVLAForceBoxAdapter` (loading, M2's 17-value force state, processors, `act`, the overlay) with one change: the box the state builder
stores for `act` to draw is `corruption.apply(condition, oracle box, tick)` instead of the oracle box. The oracle box stays in
`oracle_box` as a diagnostic that `act` never reads (M0 records it in every trace anyway). `tick` counts state-builder calls since
`reset()`, so a trial's noise is the same whatever the worker order or a resume. `XVLANoCorruption` draws the oracle box unchanged
(identity); it exists for the live equality gate (a trial reproduces A3's recorded trace bit for bit) and is not an arm.

`swap` draws, on both cameras, the live box of the distractor nearest the target in 3D at the first tick of the trial (fixed for
the whole episode; the first of equal distances in name order), never the target's.
"""
import mujoco
import numpy as np

from experiments.props_scene import constants as scene_constants
from experiments.vla_inputs_arm_xvla_force_box.adapter import INPUT_DIRS, XVLAForceBoxAdapter, state_force
from experiments.vla_inputs_arm_xvla_force_box_noisy import corruption
from experiments.vla_inputs_signals import signals

__all__ = ["INPUT_DIRS", "NoisyBoxAdapter", "XVLADropout", "XVLAJitterLarge", "XVLAJitterSmall", "XVLANoCorruption", "XVLASwap",
           "prop_positions", "state_force"]


def prop_positions(model, data):
    """{object name: world position} of every prop in the model (bodies `prop_<name>`)."""
    out = {}
    for name in scene_constants.OBJECTS:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"prop_{name}")
        if body >= 0:
            out[name] = np.array(data.xpos[body])
    return out


class NoisyBoxAdapter(XVLAForceBoxAdapter):
    """A3's adapter drawing `corruption.apply(self.condition, ...)` of the live oracle box; `condition` None draws it unchanged."""
    name = "xvla_force_box_noisy"
    condition = None

    def __init__(self, checkpoint, state_builder=state_force, enabled=True):
        self.enabled = enabled
        self._tick = 0
        self.distractor = None
        self.oracle_box = None
        super().__init__(checkpoint, state_builder=state_builder)

    def reset(self):
        self._tick, self.distractor, self.oracle_box = 0, None, None
        super().reset()

    def state_and_box(self, model, data, target):
        state = super().state_and_box(model, data, target)
        oracle = self._box
        self.oracle_box = oracle.copy()
        condition = self.condition if self.enabled and self.condition else corruption.IDENTITY
        other = None
        if condition == "swap":
            if self.distractor is None:
                self.distractor = corruption.nearest_distractor(target, prop_positions(model, data))
            other = signals.target_box(model, data, self.distractor)
        self._box = corruption.apply(condition, oracle, self._tick, distractor_box=other)
        self._tick += 1
        return state


class XVLANoCorruption(NoisyBoxAdapter):
    name = "xvla_force_box_noisy_identity"


class XVLAJitterSmall(NoisyBoxAdapter):
    name, condition = "xvla_force_box_jitter_small", "jitter_small"


class XVLAJitterLarge(NoisyBoxAdapter):
    name, condition = "xvla_force_box_jitter_large", "jitter_large"


class XVLADropout(NoisyBoxAdapter):
    name, condition = "xvla_force_box_dropout", "dropout"


class XVLASwap(NoisyBoxAdapter):
    name, condition = "xvla_force_box_swap", "swap"
