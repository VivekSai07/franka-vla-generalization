# X-VLA continuation pilots on the props P2 dataset (overnight pilots)

Training package for the overnight pilots of the VLA-inputs programme: **10,000 updates continuing the bundle's P2 X-VLA
checkpoint 101,520 with one extra-state variant**, judged on a separate 192-layout pilot tier. A pilot is a screen
(promote to a full run or drop), never a reported result.

Data: the sealed props P2 collection `outputs/props_p2_data/props_p2_20261013` (540 scripted robosuite episodes of 452 frames, 480
training and 60 validation; seal `b1d642b3edeebe15d462709ebeb8b92afa6d7eabcca3be5514e5deeecaf4d898`), bound exactly as M2 binds it.

Design: [docs/superpowers/specs/2026-10-31-vla-inputs-overnight-pilots-design.md](../../docs/superpowers/specs/2026-10-31-vla-inputs-overnight-pilots-design.md).
Plan: [docs/superpowers/plans/2026-10-31-vla-inputs-overnight-pilots.md](../../docs/superpowers/plans/2026-10-31-vla-inputs-overnight-pilots.md).
Layouts: [experiments/vla_inputs_pilot_plan](../vla_inputs_pilot_plan/README.md).

It reads M2 (`experiments/xvla_p2_force`, frozen, **never edited**) read-only: P2's constants, split, instructions and
sampler (`sample_batches`), the dataset checks and the force transform (`forces.py`, whose hash is recorded in every force
run's record). The update loop and its plumbing are M2's runner functions copied unchanged; `tests/test_runner.py` asserts the
copies stay equal to M2's by AST. Nothing from the original P2 package is imported.

## The recipe (fixed before any pilot ran)

| | |
|---|---|
| start | the bundle's P2 checkpoint 101,520 (`outputs/downloads_20261017/props_p2/checkpoints/step_101520`, model `59d02d0a578f3b70590534837f65b802764e68d02e05ae513c334ed99975932e`), weights only, each of its files pinned by hash (`contract.START_FILES`) |
| optimizer | a **fresh** AdamW (the bundle keeps no optimizer state): P2's betas (0.9, 0.95), eps 1e-8, weight decay 1e-4 (VLM group 1e-5), clip 10, bf16 autocast over FP32 parameters and gradients; the VLM group runs at a tenth of the rate |
| schedule | linear warm-up over 200 updates to **1e-5** (the floor of P2's own schedule), then constant; `contract.learning_rate` |
| horizon | 10,000 updates at effective batch 8 (2 x 4): 80,000 samples, 0.39 epoch of P2's 203,040 windows; `contract.UPDATES`, read by the runner |
| samples | M2's `sample_batches` (fixed seed 20261007): the same order for every variant |
| action space | the action statistics the processors normalise with must equal the ones saved in the start checkpoint, byte for byte (`contract.require_start_stats`, checked by every `--check` and `--train`) |
| output | `checkpoints/step_010000` only: weights, config, processors and a marker; no optimizer state, no snapshots. A crashed pilot is re-run in a new directory; it is deterministic |
| decision | promote iff Δ ≥ +5 points on the paired pilot layouts, otherwise drop; mechanism counts are reported beside Δ and never decide |

The control (`variants/control.json`) is the same run with every extra zero: the start checkpoint continued under this recipe,
trained once and reused by every pilot of this recipe version (`contract.RECIPE_VERSION`).

## The state: 19 values, whichever variant

9 joint positions, then 10 extras padded with zeros. The model pads proprioception to 20 slots with zeros anyway, so `control`
reaches the model exactly as P2's 9-value state does, and `force` exactly as M2's 17-value state does. State normalisation stays
`IDENTITY`: the loader applies each builder's transform, LeRobot's normaliser does not.

| variant | extras | width | source | transform |
|---|---|---|---|---|
| `control` | none | 0 | | |
| `force` | `wrist_wrench` (6) then `finger_force` (2) | 8 | sealed S0 sidecar `outputs/vla_inputs_signals/p2_signals_20261030` (seal `7e2e06db6f71883992cb0e66375e2da899ada4297359d87bd0e2a2f278f58e6c`) | M2's clip to the training rows' 0.1/99.9 percentiles, then standardise (`forces.py`) |
| `box` | `target_box`: front camera then gripper camera, (u_min, v_min, u_max, v_max, in_frame) each | 10 | the same sidecar | none: raw, already in [0, 1]; `in_frame` is 1 on every sealed frame, so those two inputs are constant |

The box table is pinned: `contract.BOX_TABLE_SHA256` = `2475c91ef951c1fe2fd2ea60feea2008c7aa260bfdf0940b7564c010d72dfbfe`
(the (540, 452, 10) float32 table). Joins are by explicit (episode, frame) keys; a row that is not at `index == episode*452 +
frame`, a sidecar whose seal does not verify and an episode file S0's validator rejects are errors.

## Variants are data

`variants/<name>.json` has exactly the keys `schema`, `name`, `state_builder`, `loss_term`, `loss_weight`, `note`
(`contract.check_variant_spec`); `state_builder` is one of `contract.BUILDERS`; `loss_term` must be `null` in this package
version (a loss term is code and needs a new package version) and `loss_weight` 0. A file outside the schema, with a duplicate
key, NaN, an unknown builder or a name that differs from its file name is rejected; a name is an identifier, never a path. The
file's sha256 and the builder's record (for `force`, the hash of `forces.py` and the constants' digest) go into each run's
`result.json` and the file is copied to `variant.json` in the run directory. The files are **not** in `SOURCE_FILES`, so adding a
variant never edits what a run is bound to. A new state builder is code and goes through `BUILDERS` with a package version.

## Run (admission image, GPU only for the last two modes)

```bash
python -m experiments.xvla_p2_pilot.runner --check            --variant force --out /out/check_force
python -m experiments.xvla_p2_pilot.runner --zero-equivalence --variant force --out /out/gate_force
python -m experiments.xvla_p2_pilot.runner --train            --variant force --out /out/force
python -m experiments.xvla_p2_pilot.runner --train            --variant force --out /out/dry --updates 300   # dry run: a prefix
```

`--out` is a new, empty, bind-mounted directory (the claim refuses anything else; no overwrite). Mounts: the repository read-only,
`outputs/props_p2_data` (the sealed P2 collection), `outputs/vla_inputs_signals` (S0), `outputs/downloads_20261017/props_p2` (the
start checkpoint) and `.cache/xvla_trial/bundle` (tokenizer and base files), all read-only; `/tmp` writable. Disk: the guard is
15 GiB free for the 3.3 GB checkpoint and its reload; 28 GiB of GPU memory free.

* **`--check`**: no GPU. Recomputes the parent's fingerprint and S0's, rehashes the binding, reads every row, checks the model
  inputs through the real processor, compares the action statistics with the start checkpoint's, strict-loads the start weights
  and builds a fresh optimizer.
* **`--zero-equivalence`**: 30 updates of the variant with every extra forced to zero against 30 updates of the control from the
  same start: per-update metrics, weights and Adam moments must be byte-equal, so the night's treatment differs from the control
  by the variant only. A leaking builder fails it (tested).
* **`--train`**: admission inline, then the continuation, then the checkpoint with a byte-exact FP32 reload. The marker records
  the variant's name and hash, the start's, `optimizer_saved: false`. The binding is rechecked before the checkpoint is written.

## What it binds

`current_binding()`: the parent's seal record, plan, substitutions, source provenance and validation record (as M2's and P2's),
the sidecar (its seal, report, S0's own provenance, the derived constants), the box table digest, the start checkpoint (exactly its
files, each at its pinned hash, the weights at their marker's), the digest of the recipe, every file in `SOURCE_FILES` and the
base model bundle. `SOURCE_FILES` lists exactly the repository modules that importing `contract` and `runner` loads, plus this
package's tests and README (`tests/test_fork_hygiene.py` asserts it).

**The rollout side is deliberately not bound:** `eval_adapter.py`, `eval_tests/` and `variants/` are outside `SOURCE_FILES`:
fixing an evaluation adapter after training must not invalidate the training binding. The evaluator binds them per arm (through
`experiments/vla_inputs_arm_xvla_pilot`).

## What a pilot cannot show

A continuation asks whether information helps a policy that already works, not what training from scratch would do; effects
below about 10 points are mostly invisible (P(promote) is about 0.5 at +5); and nothing here is reportable. A promoted idea earns
a full run on M0's comparison tier. Calibration (the force pilot against M2's full result) comes before a pilot may drop an idea.

## Tests

```bash
python -m unittest discover -s experiments/xvla_p2_pilot/tests -t experiments/xvla_p2_pilot/tests
```

CPU only (admission image, with the mounts above). The synthetic-policy tests run the real `train` and `zero_equivalence` with a
one-layer model and patched GPU calls; nothing here exercises a real GPU update, so the first GPU evidence is the zero-equivalence
gate and the dry run of plan Task 6.
