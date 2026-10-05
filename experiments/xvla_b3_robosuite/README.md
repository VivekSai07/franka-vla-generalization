# X-VLA fine-tune on the B3 robosuite dataset

Training package for B3, the last cube experiment: X-VLA fine-tuned from the base checkpoint on the
sealed B3 collection, `outputs/b3_robosuite_data/b3_robosuite_20261003` (560 scripted episodes in the
robosuite scene: 500 training, 60 validation; 10 colour-pair instructions; one to two distractors in
80% of the training episodes; extra picks at the region edges). Seal fingerprint
`e14af864596caf7cb515ebc500957e970a3cf12a40a3a65c1692696d92b8e6c0`.

Design: [docs/superpowers/specs/2026-10-03-b3-robosuite-cubes-design.md](../../docs/superpowers/specs/2026-10-03-b3-robosuite-cubes-design.md)
section 3. Plan: [docs/superpowers/plans/2026-10-03-b3-xvla-training.md](../../docs/superpowers/plans/2026-10-03-b3-xvla-training.md).
Collection report: [docs/B3_ROBOSUITE_COLLECTION_20261003.md](../../docs/B3_ROBOSUITE_COLLECTION_20261003.md).

It is a fork of the B2 training package (contract + runner + tests). The training loop, the
checkpoint writer, the strict FP32 reload and the resource guards are unchanged; what changed is
what is bound, what is read, and the horizon.

## What it binds

`current_binding()` checks, on every admission:

1. `seal.json` equals the seven-field record in `contract.SEAL_RECORD`.
2. `plan.json` passes `b3_robosuite_data.plan.validate_plan` and hashes to the sealed plan digest.
3. `substitutions.json` is `{}` and hashes to the sealed digest.
4. `provenance.json` equals a fresh `b3_robosuite_data.contract.provenance()` and hashes to the
   sealed digest. This freezes `experiments/b3_robosuite_data/**`, `experiments/robosuite_scene/*.py`,
   `src/**`, the shared modules the collection imported, the B3 spec and the collection plan.
5. `validation.json` equals the sealed counts (560 episodes, 500 + 60, 25 + 3 per cell, 100 / 200 /
   200 distractors, 400 / 50 / 50 pick positions, 0 substitutions, 0 held-out pairs).
6. Every file in `SOURCE_FILES` and the base model bundle (by hash).

`--check` additionally recomputes the dataset fingerprint over every file of the parent
(`contract.parent_fingerprint()`, about 3 GB) and refuses unless it equals the seal.

Two facts about the data that the contract pins by test:

- **The split comes from the plan.** LeRobot's `meta/info.json` marks all 560 episodes `train`
  (`{'train': '0:560'}`). Episodes 0-499 are training and 500-559 validation, from `plan.json`.
- **Tasks are matched by text.** LeRobot numbers the 10 instructions by first appearance, not in the
  plan's sorted order, so a task number is never trusted.

The approved instruction set is exactly the 10 trained instructions. The two held-out pairs
(`yellow -> red`, `green -> blue`) exist only in the evaluation package.

## Recipe and horizon

Identical to B2 except the horizon and the data: effective batch 8 (microbatch 4 x accumulation 2),
chunk 30, `n_action_steps` 10, peak LR 1e-4 with 500 warm-up updates and cosine decay to 1e-5, VLM LR
scale 0.1, weight decay 1e-4, betas (0.9, 0.95), eps 1e-8, clip 10, FP32 parameters with bf16
autocast, seed 20260917, from the base bundle in `.cache/xvla_trial/bundle`; no automatic resume, but a
manual one from a verified snapshot (below).

**156,000 updates = 5.0 epochs.** 500 episodes x 499 complete chunk starts = 249,500 starts;
x 5 / 8 = 155,937.5, rounded to 156,000 = 5 x 31,200 so that checkpoints fall on whole-epoch
boundaries (31,200 x 8 = 249,600 chunks = 1.0004 epochs). That is the exposure B2 had (5.01
epochs); it is held constant, not the update count.

Checkpoints: `[1, 31200, 62400, 93600, 124800, 156000]` (about 3.3 GB each, 9.9 GB for the final one,
which alone carries `training_state.pt`; 26.4 GB in all). The evaluation screens the last four. The
disk guard is 60 GiB for training (checkpoints plus the resume snapshots, below), 28 GiB for
preflight and the resume proof, and 28 GiB of free GPU memory.

Offline metrics (loss, action error, gripper accuracy) never select a checkpoint, stop a run early
or support a claim about progress; only rollouts do.

## Resuming a failed run

The first launch (`..._v1`) saved the optimizer only on the final update, so a crash at hour 30 meant
starting again from zero. This package (`..._v2`) can continue instead.

**Snapshots.** After every 2,600th update (about 36 min; it divides 31,200, so each epoch checkpoint
is also a snapshot) the runner writes `resume/snap_NNNNNN/` holding the FP32 weights (`model.pt`), the
Adam state (`optimizer.pt`), the torch RNG state (`rng.pt`), the Python and NumPy RNG state and the
SHA-256 of every file (`snapshot.json`), and the run report (`report.json`). About 10.5 GB each. It
is written under `tmp_NNNNNN/` and renamed only when complete, and the newest two are kept. A crash
loses at most 2,600 updates.

**`--train --resume`** restores the newest snapshot whose bytes still hash to its record (a damaged
one falls back to the previous), refuses if the sources or data differ from the run the snapshot
belongs to, replays the sampler to its position (it is a pure function of the seed), restores the RNG
last, and continues. Everything the snapshot does not vouch for (checkpoints and partial snapshots
written after it, `failure.json`, `result.json`, the extra `metrics.jsonl` rows) moves to
`quarantine/resume_NN/`; nothing is deleted. Each resume leaves `resume_claim_NN.json`, a copy of the
previous `result.json` as `result_before_resume_NN.json`, and an entry in `result.json['resumes']`.
A second failure writes `failure_2.json`, never over the first.

**It is manual.** There is no restart policy: read `failure.json` and `docker logs` first. A fix that
needs a file in `SOURCE_FILES` changed is a different run, not a resume: the binding no longer
matches and `--resume` refuses. A fix in the container flags (memory, `--ulimit`, tmpfs) is fine.

**`--resume-proof`** is the gate that earns the right to trust it. It copies the preflight run's
update-1 snapshot into a fresh directory, resumes it in a new process for update 2, and requires the
logged loss, gradient norm, sampled rows and learning rates, the update-2 weights and the Adam state
to equal the uninterrupted preflight run's byte for byte. `--train` refuses without it.

## Gates

Always in order, each with its own output directory, in the admission image
(`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`; never the robosuite
image). Mount the parent read-only and give `/tmp` 32 GB (the Hugging Face arrow cache of a
295,680-row dataset lands there).

| gate | output | proves | cost |
|---|---|---|---|
| `--check` | `outputs/train/xvla_franka_stacking_b3_robosuite_v2_admission` | fingerprint recomputed, binding, statistics from the training rows only, processor checks | CPU, minutes to hours |
| `--preflight` | `..._v2_preflight` | 2 GPU updates, FP32 weight + Adam round trip, one snapshot | about 5 min |
| `--resume-proof` | `..._v2_resumeproof` | a resumed process reproduces update 2 exactly | about 5 min |
| `--train` | `outputs/train/xvla_franka_stacking_b3_robosuite_v2` | the run | about 36 h if the speed matches B2 (0.84 s per update) |

```bash
python -m experiments.xvla_b3_robosuite.runner --check | --preflight | --resume-proof | --train [--resume]
```

Container flags and the pre-launch checklist: `.github/skills/launching-a-run/SKILL.md`. Add
`--ulimit nofile=65536:65536` to every GPU run (FAILURES #42). `--resume-proof` mounts the preflight
directory read-only and the empty proof directory read-write; `--train` mounts both proof
directories read-only and the output directory read-write, and so does `--train --resume`.

## Files

| file | content |
|---|---|
| `contract.py` | parent binding, data reader, instruction map, horizon, schedule, `SOURCE_FILES` |
| `runner.py` | `--check`, `--preflight`, `--resume-proof`, `--train [--resume]` |
| `tests/test_contract.py`, `test_runner.py`, `test_fork_hygiene.py` | contract, runner and fork-residue tests |

**This README is in `SOURCE_FILES`: finish it before `--check`. Nothing in `SOURCE_FILES` is edited
between `--check` and the end of training.**
