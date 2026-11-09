# gen_eval — the G0 generalization benchmark evaluator

Forked from M0's evaluator `experiments/vla_inputs_eval/` (frozen core `c4f6c303f2ae`, never edited). Design:
[G0 spec](../../docs/superpowers/specs/2026-11-06-g0-generalization-benchmark-design.md) (D1-D14),
[plan](../../docs/superpowers/plans/2026-11-06-g0-generalization-benchmark.md). The scene comes from the randomization engine
[gen_scene](../gen_scene/README.md); arms are M0's adapters (`experiments.vla_inputs_arm_*`), imported read-only.

**Frozen after the first arm is prepared.** `contract.core_provenance()` hashes this package's modules, every `gen_scene` module, the
texture manifest, the M0 and P2 evaluation modules it imports and S0's whole provenance. `prepare` also binds the 120 texture files
(sha256 each) and refuses a second arm whose core differs from the first's; every worker re-verifies core, arm, manifests, textures
and checkpoint, and `compare` re-verifies them in full. A core change is a new protocol and re-measures every arm.

## What is measured

| | |
|---|---|
| layouts | namespace `gen:g0`: **benchmark** 144 (M0's 48 comparison cells x 3), **selection** 24 (every other cell x 1; for a later arm's checkpoint, never for claims), **smoke** 2. Fresh seeds and poses, disjoint from P2, its re-test, M0 and the pilot (`plan.check_disjoint`) |
| trials | every layout runs once under `nominal` and once under each of five axes, with its own rollout seed (shared by every condition and arm): 864 benchmark trials per arm. Layout `n` of a tier gets level `(n + offset) % levels` on each axis (`gen_scene.levels`) |
| feasibility (D8) | the unrendered expert on every layout x condition that changes geometry; a layout infeasible at nominal leaves with all six trials; `lighting`, `background` and `drop` inherit the nominal's; `move` also needs the expert to solve a fresh episode from the post-move pose. Dropped trials are counted by axis and level, for every arm, never replaced |
| tick caps | 900; 1,200 for the disturbance axis |
| success | M0's: the target resting in the named bin for 30 stable ticks (P2's `in_bin`, relative to the bin's own height); `grasp_lift` = the target rose 5 cm or more at any tick; `cause` = P2's classifier |
| disturbances | `move`: at the start of the first tick the grasp centre (finger midpoint) is within 6 cm horizontally of the target and no close command (applied gripper <= 127) has been applied, the target is put at `move_to`, 8 cm from its start; `drop`: at the start of the first tick the target is 5 cm or more above its start, the gripper command is forced open (255) for that tick and the nine after it. Each fires at most once; the policy keeps the arm and sees everything. Recorded: `disturbance_fired`, `disturbance_tick`, `recovered` = success and fired |
| statistics | per arm, axis and level: success under the perturbation against nominal on the same layouts; exact McNemar; the paired drop with a 95% bootstrap interval stratified by object (10,000, seed 0); **held** iff the drop's interval upper bound <= 10 points; Holm over the five axes is reported (no family-wise claim) |

Caveat recorded with the rule: a layout whose start lies within 6 cm of the home grasp centre (12 of 144 benchmark layouts, about 6 of
72 `move` layouts) has its target moved before the first observation (`disturbance_tick` 0). The rule is literal; read the per-tick
field when interpreting `move`.

## Gates (spec D9) and what has run

(i) engine nominal equals `props_scene` and (ii) forked expert reproduces three sealed P2 episodes: `gen_scene.gates`, passed 2026-11-09
(see [gen_scene README](../gen_scene/README.md)). (iii) **`identity_gate.py`**: A2's M0 comparison trials 1 and 291 re-run through this
package's rollout with the nominal condition, bit for bit in every trace array, under M0's worker runtime (deterministic algorithms,
7 cores and 7 threads: FAILURES #70). It needs the GPU and runs through `scripts/gen_eval.sh identity_gate <tag>`. A CPU stand-in
(`tests/test_rollout.py`, `M0Parity`) runs M0's frozen rollout and this one on the same replayed action sequence and compares every array.

## Running it

`scripts/gen_eval.sh` (tests in `tests/test_gen_eval_script.py`, against a fake `docker`): `plan`, `feasibility <shards> <cores>`,
`prepare <arm>`, `smoke <arm>`, `selection <arm>`, `select <arm>`, `benchmark <arm>`, `compare <arm>`, `final`, `identity_gate <tag>`.
Run from a dedicated worktree at the merge commit; poll with `docker ps` and `result.json`, never `docker wait`. Arms: A0 (P2 101,520),
A2 (88,830), A3 (101,520, descriptive).
