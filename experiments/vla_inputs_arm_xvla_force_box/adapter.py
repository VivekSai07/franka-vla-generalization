"""Arm A3 of the M0 evaluator: X-VLA trained with M2's force state and the target's box drawn on both camera frames.

M0 loads an arm only from a module named `experiments.vla_inputs_arm_*`, by the identifier
`experiments.vla_inputs_arm_xvla_force_box.adapter:XVLAForceBoxAdapter`. The code is
`experiments/xvla_p2_force_box/eval_adapter.py`, deliberately outside the training contract's `SOURCE_FILES`; this module only
re-exports it, so that M0's arm closure binds this package, its README and every repository module the import loads, and
`INPUT_DIRS` binds the base bundle and the sealed sidecar the adapter reads.
"""
from experiments.xvla_p2_force_box.eval_adapter import INPUT_DIRS, XVLAForceBoxAdapter, state_force

__all__ = ["INPUT_DIRS", "XVLAForceBoxAdapter", "state_force"]
