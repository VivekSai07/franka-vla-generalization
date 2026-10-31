"""The X-VLA continuation-pilot arm of the M0 evaluator.

M0 loads an arm only from a module named `experiments.vla_inputs_arm_*`, by the identifier
`experiments.vla_inputs_arm_xvla_pilot.adapter:XVLAPilotAdapter`. The code is `experiments/xvla_p2_pilot/eval_adapter.py`,
deliberately outside the training contract's `SOURCE_FILES`; this module only re-exports it, so that M0's arm closure binds
this package, its README and every repository module the import loads, and `INPUT_DIRS` binds the base bundle and the sealed
sidecar the adapter reads. The variant is read from the checkpoint's marker, so one identifier serves every pilot variant.
"""
from experiments.xvla_p2_pilot.eval_adapter import INPUT_DIRS, LIVE, XVLAPilotAdapter

__all__ = ["INPUT_DIRS", "LIVE", "XVLAPilotAdapter"]
