# X-VLA fine-tune on the props P3 dataset

Training package for props P3, Food vs Drinks with a fixed prompt: X-VLA fine-tuned from the base checkpoint on
the sealed collection `outputs/props_p3_data/props_p3_20261020` (544 scripted episodes in the robosuite scene:
480 training, 64 validation; one object per episode; can and milk go to the orange bin, bread and lemon to the
blue bin, the bins swap sides; bottle and cereal never collected). Every frame carries the single prompt
`sort the item`. Seal fingerprint `6430d34ea2a13c9c49a6621b0a2509350cc5e84c489209154367dc27cd151acf`.

Design: [docs/superpowers/specs/2026-10-19-props-p3-food-drinks-design.md](../../docs/superpowers/specs/2026-10-19-props-p3-food-drinks-design.md).
Plan: [docs/superpowers/plans/2026-10-21-props-p3-training-eval.md](../../docs/superpowers/plans/2026-10-21-props-p3-training-eval.md).
Collection report: [docs/PROPS_P3_COLLECTION_20261020.md](../../docs/PROPS_P3_COLLECTION_20261020.md).

A fork of `experiments/xvla_props_p2/`. **The recipe is unchanged** (P1's, through P2), so P3 differs from them in
its data and in one place of code: there is **one instruction**, so the machinery that matched ten instructions by
text is re-derived for a single string (below). The checkpoint writer, strict FP32 reload, snapshots, resume, the
faster loop, the pause and the resource guards are P2's.

## What it binds

`current_binding()` checks, on every admission:

1. `seal.json` equals the six-field record in `contract.SEAL_RECORD`.
2. `plan.json` passes `props_p3_data.plan.validate_plan` and hashes to the sealed plan digest.
3. `substitutions.json` is the sealed empty record (P3 needed no substitutions).
4. `provenance.json` equals a fresh `props_p3_data.contract.provenance()` and hashes to the sealed digest. This
   freezes `experiments/props_p3_data/**`, `experiments/props_scene/*.py`, `experiments/robosuite_scene/*.py`,
   `src/**`, the shared modules the collection imported, and the P3 spec and collection plan (FAILURES #52: do not
   edit them, ever).
5. `validation.json` equals the sealed record (544 episodes, 480 + 64, 60 per cell in training, 8 per cell in
   validation, 136 episodes of each of the four objects, 0 substitutions, 0 held-out objects, the prompt).
6. Every file in `SOURCE_FILES` and the base model bundle (by hash).

`--check` also recomputes the dataset fingerprint over every file of the parent and refuses unless it equals the
seal. As before, the split comes from the plan (LeRobot calls all 544 episodes `train`).

## One instruction (what is new)

`INSTRUCTIONS` is `('sort the item',)`. `planned_instructions()` reads the sealed plan and refuses unless every
slot carries exactly that prompt; `task_index_map()` refuses unless the dataset's task table is exactly
`{'sort the item': 0}`; `approved_instruction()` and `policy_row()` accept only it, so the expert's own
`put the X in the Y bin` text, which the collection replaced, is rejected if it ever appears. Tests were
mutation-checked for each of these.

## Recipe and horizon (unchanged)

Effective batch 8 (microbatch 4 x accumulation 2), chunk 30, `n_action_steps` 10, peak LR 1e-4 with 500 warm-up
updates and cosine decay to 1e-5, VLM LR scale 0.1, weight decay 1e-4, betas (0.9, 0.95), eps 1e-8, clip 10, FP32
parameters with bf16 autocast, seed 20261007, base bundle in `.cache/xvla_trial/bundle`. 480 x 423 = 203,040
starts; **126,900 updates = 5.0 epochs** (the same count as P1 and P2, because the training split is the same
480 x 452 frames). Checkpoints at 1, 50,760, 76,140, 88,830, 101,520, 114,210, 126,900 (epochs 0, 2, 3, 3.5, 4,
4.5, 5); about 29.7 GB. Disk guard 65 GiB, 28 GiB of free GPU memory.

Offline metrics never select a checkpoint, stop a run early or support a claim about progress.

## The faster loop (P2's, proven again before `--train`)

`one_update` is P1's update unchanged; `verify_precision_fast`, the background `Prefetcher` and the fused
finiteness checks only remove waiting. **`--equivalence`** is a gate: 30 updates with the serial loop, then 30 with
the fast loop from the same fresh state; per-update metrics, model weights and Adam moments must be byte-equal.
`--train` refuses without it. P3 repeats it because this is a new package, a new binding and a new dataset.

## Overnight batches and resuming

`--train --stop-at <unix time>` pauses at a snapshot (resume snapshot after every 2,600th update, about 10.5 GB,
newest two kept) when the next one could not be written before the deadline; continue with
`--train --resume --stop-at <time>` (proven by `--resume-proof`). Every 20 updates `nvidia-smi` is read inside the
container; at 83 degrees C or any thermal slowdown the loop sleeps until 75 degrees C. A fix that needs a file in
`SOURCE_FILES` changed is a different run, not a resume.

## Gates

In order, each with its own output directory, in the admission image
(`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`), `--ulimit nofile=65536:65536` on every
GPU run (FAILURES #42), a large `/tmp`.

| gate | output | proves |
|---|---|---|
| `--check` | `outputs/train/xvla_props_p3_v1_admission` | fingerprint, binding, train-only statistics, processor checks (CPU) |
| `--preflight` | `..._v1_preflight` | 2 GPU updates, FP32 weight + Adam round trip, one snapshot |
| `--resume-proof` | `..._v1_resumeproof` | a resumed process reproduces update 2 exactly |
| `--equivalence` | `..._v1_equivalence` | the fast loop equals the serial loop byte for byte over 30 updates |
| `--train [--resume] [--stop-at T]` | `outputs/train/xvla_props_p3_v1` | the run |

```bash
python -m experiments.xvla_props_p3.runner --check | --preflight | --resume-proof | --equivalence | --train [--resume] [--stop-at T]
```

## Files

| file | content |
|---|---|
| `contract.py` | parent binding, data reader, the one-instruction checks, horizon, schedule, equivalence validator, `SOURCE_FILES` |
| `runner.py` | the gates, `one_update`, `verify_precision_fast`, `Prefetcher`, the pause and the thermal guard |
| `tests/test_contract.py`, `test_runner.py`, `test_fork_hygiene.py` | contract, runner and fork-residue tests |

**This README is in `SOURCE_FILES`: finish it before `--check`. Nothing in `SOURCE_FILES` is edited between
`--check` and the end of training.**
