# vla_inputs_pilot_plan

Layouts for the overnight input pilots: 192 comparison layouts (48 cells x 4), 2 smoke layouts, an empty selection tier,
namespace `vla_inputs:pilot`. Same P2 samplers and layout schema as M0's plan (`vla_inputs_eval.plan`, imported read-only);
every seed, start pose and distractor pose is disjoint from the sealed P2 plan, the P2 re-test and both M0 tiers.
M0's frozen core serves a campaign root holding this plan unchanged (tested in `tests/test_plan.py`).

`python -m experiments.vla_inputs_pilot_plan.plan` is not provided; the night driver calls `plan.write(<root>/plan)`.
Tests (admission image, no simulator): `python -m unittest discover -s experiments/vla_inputs_pilot_plan/tests -t experiments/vla_inputs_pilot_plan/tests`
with the sealed P2 root mounted at `outputs/props_p2_data`.
