# X-VLA fine-tune on the sealed G1 randomized dataset (G1)

Training package for G1 of the generalization programme: X-VLA fine-tuned from the base checkpoint on the sealed
`outputs/gen_g1_data/g1_20261110` (seal `845ac94063fd367cc315b8d287fc141df76ea171f9227f5794eef0bd8e4e161e`): 3,000 scripted
episodes of 452 frames in the randomized robosuite scene (1,356,000 frames); episodes 0-2699 train, 2700-2999 validation, read
from the plan; ten trained object->bin instructions, `milk->blue` and `bread->orange` never collected as targets. The state is
M2's 17 values: the 9 joint positions and 8 force/torque values. Design: [spec](../../docs/superpowers/specs/2026-11-06-g1-randomized-data-design.md),
[plan](../../docs/superpowers/plans/2026-11-06-g1-randomized-data.md); the collection package is `experiments/gen_g1_data/`.

A fork of `experiments/xvla_p2_force/` (M2, which trained A2), re-bound from P2's dataset to the G1 dataset. **The recipe is
M2's, unchanged**: `tests/test_m2_reference.py` compares M2's contract and runner with this one's value by value (constants,
schedule, checkpoint grid, sampler, optimiser, model config, and the update function's code), so the one variable that changes
between A2's run and this one is the data. Caveat: "the data" is three things at once (size, the randomization of the scene,
and a lower exposure per start, below); this package cannot separate them.

## The data and the 17-value state

| columns | content | source | transform |
|---|---|---|---|
| 0-8 | the 9 joint positions (7 arm joints, 2 finger joints) | dataset `observation.state` | none (raw radians) |
| 9-14 | `wrist_wrench`: Fx, Fy, Fz (N), Tx, Ty, Tz (N·m) | `signals/episode_NNNN.npz` of the sealed G1 dataset | clip, then standardise |
| 15-16 | `finger_force`: left, right normal force (N) | same | clip, then standardise |

The signals are the dataset's own sealed files, joined by explicit (episode, frame) keys (`contract.JoinedDataset`); there is no
separate signals dataset. X-VLA takes 20 proprioception values: 17 real and 3 zeros.

* **Signals** (`forces.py`, numpy only): every file is checked by S0's episode validator; a missing, duplicated, unexpected,
  short or shifted episode, a split that is not 2,700 train then 300 validation, and a seal that is not the pinned
  `forces.SEAL_RECORD` are errors. Values are clipped to the 0.1 and 99.9 percentiles of the **training rows only** (2,700
  episodes, 1,220,400 rows), then standardised with the mean and standard deviation of the clipped training rows (exactly
  rounded sums, so every image derives the same bytes). The constants come from demonstrations, never from rollouts, and are pinned
  by `forces.CONSTANTS_SHA256` (`b6e229658c04d6462cfe96bf71425b314608066835a4d21b33054ce5df2a2d15`, pinned before any training).
* **Normalisation stays `IDENTITY`** for the state: the loader applies the transform, LeRobot's normaliser does not.
* **Statistics** (`contract.selected_stats`): 17 values for the state, 8 for the action; the first 9 equal M2's computation byte for byte.
* **Model-input check** (`contract.check_model_inputs`): proprio is (n, 20); `[:, :9]` is the raw state; `[:, 9:17]` equals the
  standardised extras recomputed from the sealed signals for the rows being checked; `[:, 17:]` is zero.

## What it binds

`current_binding()` checks on every admission: the parent's `seal.json` equals the pinned record; `plan.json` is the canonical
G1 plan and hashes to the sealed digest; `substitutions.json` (9 substituted slots) and `validation.json` are the sealed ones;
`provenance.json` equals a fresh `gen_g1_data.contract.provenance()` and hashes to the sealed digest; the signals' seal and
constants digest; every file in `SOURCE_FILES`; the base model bundle. `--check` also recomputes the parent's fingerprint over every
file (about 33 GB) and refuses unless it equals the seal.

`SOURCE_FILES` lists exactly the repository modules that importing `contract` and `runner` loads, this package's files and tests,
and `REFERENCE_FILES`: M2's `__init__`, contract, forces and runner, plus the two P2 collection modules they add, loaded only by
the M2 gate, so that proof is of equality with exactly those bytes. `tests/test_fork_hygiene.py` checks all of it. There is no
`eval_adapter.py` here: the rollout side is never bound by the training contract, so fixing it after training cannot invalidate a run.

## Budget

126,900 updates x effective batch 8 = 1,015,200 starts drawn = **0.89 epoch** of the 1,142,100 complete starts (2,700 x 423; no
start repeats). The budget is M2's updates, not a fraction of the data: A2 was 5.0 epochs of 203,040 starts (480 x 423), the same
126,900 updates. Microbatch 4 x accumulation 2, chunk 30, peak LR 1e-4 (500 warm-up updates, cosine to 1e-5), seed 20261007.
Checkpoints at 1, 50,760, 76,140, 88,830, 101,520, 114,210, 126,900 (about 29.7 GB); snapshots every 2,600 updates; disk guard 65 GiB.
Offline metrics never select a checkpoint, stop a run early or support a claim.

## Gates

In order, each with its own output directory, in the admission image (`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`), with the sealed dataset
(`outputs/gen_g1_data/g1_20261110`) and the base bundle mounted read-only, `--ulimit nofile=65536:65536` on GPU runs and a large `/tmp`.

| gate | output | proves |
|---|---|---|
| `--check` | `outputs/train/xvla_g1_rand_v1_admission` | fingerprint, binding, train-only statistics, the 17-value input check, clipping rates (CPU) |
| `--preflight` | `outputs/train/xvla_g1_rand_v1_preflight` | 2 GPU updates, FP32 weight and Adam round trip, one snapshot |
| `--resume-proof` | `outputs/train/xvla_g1_rand_v1_resumeproof` | a resumed process reproduces update 2 exactly |
| `--equivalence` | `outputs/train/xvla_g1_rand_v1_equivalence` | the fast loop equals the serial loop byte for byte over 30 updates |
| `--m2-equivalence` | `outputs/train/xvla_g1_rand_v1_m2equivalence` | M2's runner and this one, from the same fresh state on M2's own inputs, agree byte for byte over 30 updates (metrics, weights, Adam moments) |
| `--train [--resume] [--stop-at T]` | `outputs/train/xvla_g1_rand_v1` | the run; refuses without the other proofs, each bound to the same sources and data |

`--m2-equivalence` replaces M2's zero-equivalence gate. It additionally mounts, read-only, P2's sealed data
(`outputs/props_p2_data/props_p2_20261013`) and its signals (`outputs/vla_inputs_signals/p2_signals_20261030`), which M2's loader reads.
For its second loop `contract.reference_profile` rebinds exactly `TRAIN`, `VAL` and `planned_instructions` to M2's, restores them on exit
(also on error) and refuses if instructions, frame count, chunk or seed differ.

```bash
python -m experiments.xvla_g1_rand.runner --check | --preflight | --resume-proof | --equivalence | --m2-equivalence | --train [--resume] [--stop-at T]
```

## Overnight batches and resuming

`--train --stop-at <unix time>` pauses cleanly (status `paused`) at a snapshot after which the next one could not be written before
the deadline; a crash, or a GPU that stays hot for 30 minutes (the guard sleeps from 83 C until 75 C), leaves `failure.json`.
`--train --resume` continues either from the newest verified snapshot in the original bind-mounted output (`run_claim.json`; each resume leaves
`resume_claim_NN.json`). A snapshot whose bytes no longer hash to its record falls back to the previous one; what the chosen
snapshot does not vouch for is moved to `quarantine/`, never deleted; a changed source or dataset binding refuses. A fix that
touches a file in `SOURCE_FILES` is a different run, not a resume.

## Files

| file | content |
|---|---|
| `forces.py` | the signals, the clip and the standardisation, the constants and their pin; numpy only |
| `contract.py` | binding, `JoinedDataset`, 17-value `observation`, statistics, `check_model_inputs`, horizon, schedule, `reference_profile`, `SOURCE_FILES` |
| `runner.py` | the gates, `one_update`, `verify_precision_fast`, `Prefetcher`, the pause and the thermal guard |
| `tests/` | contract, runner, signals and join, statistics and input check, M2 reference and gate, fork hygiene; no GPU or simulator |
