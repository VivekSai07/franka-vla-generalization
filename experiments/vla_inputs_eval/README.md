# M0 — common evaluation protocol and the control arm

The shared evaluator for the VLA-inputs programme: fresh P2-style layouts, a policy-agnostic rollout that records the S0
signals for every arm, exact paired statistics, and the control arm A0 (X-VLA 101,520 from the P2 bundle).
Design: `docs/superpowers/specs/2026-10-30-vla-inputs-m0-evaluation-design.md`; plan:
`docs/superpowers/plans/2026-10-30-vla-inputs-m0-evaluation.md`.

## Frozen shared core and separately bound arms

This package (its non-test modules, not this README) plus the P2 classifier `xvla_props_p2_eval/analysis.py` and the whole
S0 signal source (`vla_inputs_signals.contract.provenance()`, which covers P2's frozen scene, expert, controllers and `src/`)
is the **shared core**; `contract.core_provenance()` hashes it. `prepare` records the core hash, the layout and feasibility
manifests, and the arm's own closure (adapter package, declared extra sources, weights, processors, runtime). `compare`
refuses unless both arms' cores equal each other and the repository, the manifests are unchanged and each arm's closure is
unchanged. A core correction is a new protocol with a new control; it is never resolved by ignoring a mismatch.

Arms live in `experiments/vla_inputs_arm_*` and are loaded lazily by `module:Class`, so an environment without an arm's
dependencies never imports it and adding an arm never edits the core. Adapter interface: `reset()`, `act(obs) -> (8,)`.

## Modules

| file | role |
|---|---|
| `plan.py` | 80 selection and 576 comparison layouts (48 cells x 12) from P2's samplers under `vla_inputs:m0:*`, disjoint from P2 and the P2 re-test |
| `scenes.py`, `feasibility.py` | restore a layout (1 mm / 1 degree proof); unrendered expert on every layout, sharded; dropped layouts counted by cell |
| `adapters.py` | the frozen adapter interface, the lazy loader, `state_p2` |
| `rollout.py` | P2's loop with an adapter, plus `wrist_wrench`, `finger_force`, `target_box` in `trace.npz` |
| `protocol.py`, `evaluate.py` | trials and stages: `prepare`, `worker` (one checkpoint per worker), `collect`, `select`, `compare`, `final` |
| `stats.py`, `power.py` | exact McNemar, stratified paired bootstrap, Holm over A1..A4, selection with recorded ties; planning power |
| `contract.py` | the core closure |

## Run

`scripts/vla_inputs_eval.sh` (see its header): `plan`, `feasibility`, `prepare`, then `stage` per arm/tier/checkpoint, which
launches one pinned worker pool for one checkpoint, polls `docker ps`, collects, and resumes once. Tests: pure ones in the
admission image, `-m unittest discover -s experiments/vla_inputs_eval/tests -t experiments/vla_inputs_eval/tests`; the
simulator tests (`test_scenes`, `test_rollout`, the simulator closure) in the robosuite image; the GPU gates
(`vla_inputs_arm_xvla/tests`) with `--runtime nvidia --gpus all` and the P2 bundle at `/bundle`.
