"""Arm A2 of the M0 evaluator: X-VLA trained with the wrist wrench and finger forces in the robot state.

M0 loads an arm only from a module named `experiments.vla_inputs_arm_*`, by the identifier
`experiments.vla_inputs_arm_xvla_force.adapter:XVLAForceAdapter`. The code is `experiments/xvla_p2_force/eval_adapter.py`,
deliberately outside the training contract's `SOURCE_FILES`; this module only re-exports it, so that M0's arm closure binds
this package, its README and every repository module the import loads, and `INPUT_DIRS` binds the base bundle and the sealed
sidecar the adapter reads.
"""
from experiments.xvla_p2_force.eval_adapter import INPUT_DIRS, XVLAForceAdapter, state_force

__all__ = ["INPUT_DIRS", "XVLAForceAdapter", "state_force"]
