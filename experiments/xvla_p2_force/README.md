# X-VLA fine-tune on the props P2 dataset with force/torque in the robot state (M2)

Training package for M2 of the VLA-inputs programme: X-VLA fine-tuned from the base checkpoint on the sealed props P2
collection `outputs/props_p2_data/props_p2_20261013` (540 scripted episodes in the robosuite scene: 480 training, 60
validation; 10 trained object->bin instructions; `milk->blue` and `bread->orange` never collected as targets; seal
`b1d642b3edeebe15d462709ebeb8b92afa6d7eabcca3be5514e5deeecaf4d898`), **with one change: the robot state carries 8 more
values**, the wrist wrench and the two finger contact forces from the sealed S0 sidecar
`outputs/vla_inputs_signals/p2_signals_20261030` (seal `7e2e06db6f71883992cb0e66375e2da899ada4297359d87bd0e2a2f278f58e6c`).
Evaluated as arm A2 of the M0 protocol.

Design: [docs/superpowers/specs/2026-10-30-vla-inputs-m2-force-design.md](../../docs/superpowers/specs/2026-10-30-vla-inputs-m2-force-design.md).
Plan: [docs/superpowers/plans/2026-10-30-vla-inputs-m2-force.md](../../docs/superpowers/plans/2026-10-30-vla-inputs-m2-force.md).
S0 report: [docs/VLA_INPUTS_S0_20261030.md](../../docs/VLA_INPUTS_S0_20261030.md).

A fork of `experiments/xvla_props_p2/`. **The recipe is P2's, unchanged** (`tests/test_p2_reference.py` reads P2's contract and
compares the recipe, the schedule, the checkpoint grid, the sampler's batches and the model config, apart from the state's
width), so this run and P2's differ in one variable. The checkpoint writer, strict FP32 reload, snapshots, resume, resource
guards, faster loop and thermal guard are P2's.

## The state: 17 values

| columns | content | source | transform |
|---|---|---|---|
| 0-8 | the 9 joint positions (7 arm joints, 2 finger joints) | sealed dataset `observation.state` | none (raw radians, as P2) |
| 9-14 | `wrist_wrench`: Fx, Fy, Fz (N), Tx, Ty, Tz (N·m) | S0 sidecar | clip, then standardise |
| 15-16 | `finger_force`: left, right normal force (N) | S0 sidecar | clip, then standardise |

X-VLA takes 20 proprioception values; P2 used 9 and zero padding, this run uses 17 and 3 zeros.

* **Join** (`contract.JoinedDataset`, `contract.joined_state`): by explicit (episode, frame) keys, never by position alone.
  A row that is not at `index == episode*452 + frame`, a missing or duplicated sidecar episode, a shifted frame index and a
  sidecar whose seal does not verify are all errors (`forces.load_raw`, S0's own episode validator).
* **Clip and standardise** (`forces.py`): clip to the 0.1 and 99.9 percentiles of the training rows that S0's sealed
  `report.json` records, then subtract the mean and divide by the standard deviation of the clipped training rows. The
  constants come from demonstrations only (never from rollouts), are derived from the sealed sidecar with exactly rounded
  sums (so every image gets the same bytes) and are pinned by `forces.CONSTANTS_SHA256`. Clipping rates per dimension for the
  training and the validation episodes are recorded by `--check` and are never used to move the ranges.
* **Normalisation stays `IDENTITY`** for the state, as in P2 (`xvla_single_task.make_config` raises otherwise): the loader
  applies the transform, LeRobot's normaliser does not. The state feature of the model config is widened to 17.
* **Statistics** (`contract.selected_stats`): 17 values for the state, 8 for the action. The first 9 state entries are computed
  exactly as P2's and equal them byte for byte; the extras' are of the standardised training rows.
* **Model-input check** (`contract.check_model_inputs`, a local copy: the shared one asserts 9 real values and is bound by
  sealed runs): proprio is (n, 20); `proprio[:, :9]` equals the raw state; `proprio[:, 9:17]` equals the standardised extras,
  which are recomputed from the sealed sidecar for the dataset rows being checked; `proprio[:, 17:]` is zero.

What it cannot show and what it is: the signals are exact simulator quantities; the wrist wrench is a sensor-frame interaction
wrench including the hand's own weight. The policy replans at 3 Hz (ten queued actions per inference), so this tests force at
chunk boundaries, not reactive force control.

## What it binds

`current_binding()` checks, on every admission, everything P2's does (the parent's seal record, plan, substitutions, source
provenance, validation record, every file in `SOURCE_FILES` and the base model bundle) and, new, the sidecar: its seal equals the
pinned record, `provenance.json` equals a fresh `vla_inputs_signals.contract.provenance()` (which covers P2's sealed files and
S0's package) and hashes to the sealed digest, `report.json`'s hash, and the derived constants' digest. `--check` first
recomputes both fingerprints (P2's parent and S0's sidecar) and refuses unless they equal their seals; as P2: the split comes
from the plan, and tasks are matched by text.

`SOURCE_FILES` lists exactly the repository modules that importing `contract` and `runner` loads, plus this package's tests and
README. **The rollout side is deliberately not bound:** `eval_adapter.py` and `eval_tests/` are outside `SOURCE_FILES`, so
fixing an evaluation adapter after training never invalidates the training binding; M0's arm manifest binds them.

## Recipe and horizon (P2's, unchanged)

Effective batch 8 (microbatch 4 x accumulation 2), chunk 30, `n_action_steps` 10, peak LR 1e-4 with 500 warm-up updates and
cosine decay to 1e-5, VLM LR scale 0.1, weight decay 1e-4, betas (0.9, 0.95), eps 1e-8, clip 10, FP32 parameters with bf16
autocast, seed 20261007, base bundle in `.cache/xvla_trial/bundle`. 480 x 423 = 203,040 starts; **126,900 updates = 5.0
epochs**. Checkpoints at 1, 50,760, 76,140, 88,830, 101,520, 114,210, 126,900 (epochs 0, 2, 3, 3.5, 4, 4.5, 5); about 29.7 GB. Disk
guard 65 GiB, 28 GiB of free GPU memory. Offline metrics never select a checkpoint, stop a run early or support a claim.

## The faster loop, overnight batches, resuming

As P2's: `one_update` is one function used by both loops; `verify_precision_fast` and `Prefetcher` change plumbing only, and
`--equivalence` proves the fast loop equals the serial one byte for byte over 30 updates before `--train` is allowed.
`--train --stop-at <unix time>` pauses cleanly at a snapshot; `--train --resume` continues from the newest verified snapshot.
A thermal guard sleeps at 83 °C until 75 °C (no effect on the math). A fix that needs a file in `SOURCE_FILES` changed is a
different run, not a resume.

## Gates

In order, each with its own output directory, in the admission image
(`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`), `--ulimit nofile=65536:65536` on every GPU run
(FAILURES #42), a large `/tmp`, and the sidecar mounted read-only.

| gate | output | proves |
|---|---|---|
| `--check` | `outputs/train/xvla_p2_force_v1_admission` | both fingerprints, binding, train-only statistics, the 17-value input check, clipping rates (CPU) |
| `--preflight` | `..._v1_preflight` | 2 GPU updates, FP32 weight + Adam round trip, one snapshot |
| `--resume-proof` | `..._v1_resumeproof` | a resumed process reproduces update 2 exactly |
| `--equivalence` | `..._v1_equivalence` | the fast loop equals the serial loop byte for byte over 30 updates |
| zero-equivalence (not written yet) | | with the 8 extras forced to 0 the loop reproduces P2's byte for byte, so the run differs only by those values |
| `--train [--resume] [--stop-at T]` | `outputs/train/xvla_p2_force_v1` | the run |

```bash
python -m experiments.xvla_p2_force.runner --check | --preflight | --resume-proof | --equivalence | --train [--resume] [--stop-at T]
```

## Files

| file | content |
|---|---|
| `forces.py` | the sidecar join, the clip and the standardisation, the constants and their pin; numpy only, shared with the rollout |
| `contract.py` | parent and sidecar binding, `JoinedDataset`, 17-value `observation`, statistics and `check_model_inputs`, horizon, schedule, `SOURCE_FILES` |
| `runner.py` | the gates, `one_update`, `verify_precision_fast`, `Prefetcher`, the pause and the thermal guard |
| `eval_adapter.py`, `eval_tests/` | rollout side: `state_force` and the X-VLA adapter for M0, and the train/rollout parity tests (robosuite image); not bound by the training contract |
| `tests/` | contract, runner, sidecar join, statistics and input check, fork hygiene, and the reference comparison with P2 |
