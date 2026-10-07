# X-VLA fine-tune on the props P1 dataset

Training package for props P1, instruction sorting with one object and no distractors: X-VLA
fine-tuned from the base checkpoint on the sealed collection,
`outputs/props_p1_data/props_p1_20261006` (540 scripted episodes in the robosuite scene: 480
training, 60 validation; 10 object->bin instructions; two held-out pairs, `milk->blue` and
`bread->orange`, never collected). Seal fingerprint
`86e3aadb678963eda8cb27d0104d6e22778d4cc25c497829e8fb47937ac53bcf`.

Design: [docs/superpowers/specs/2026-10-05-props-sorting-design.md](../../docs/superpowers/specs/2026-10-05-props-sorting-design.md).
Plan: [docs/superpowers/plans/2026-10-07-props-p1-training-eval.md](../../docs/superpowers/plans/2026-10-07-props-p1-training-eval.md).
Collection report: [docs/PROPS_P1_COLLECTION_20261007.md](../../docs/PROPS_P1_COLLECTION_20261007.md).

It is a fork of `experiments/xvla_b3_robosuite/` (contract + runner + tests). The training loop, the
checkpoint writer, the strict FP32 reload, the resumable snapshot machinery and the resource guards
are unchanged; what changed is what is bound, what is read, the episode length, and the horizon.

## What it binds

`current_binding()` checks, on every admission:

1. `seal.json` equals the six-field record in `contract.SEAL_RECORD`.
2. `plan.json` passes `props_p1_data.plan.validate_plan` and hashes to the sealed plan digest.
3. `substitutions.json` hashes to the sealed digest (`{"74": 1, "261": 1}`, two real substitutions —
   unlike B3's empty record).
4. `provenance.json` equals a fresh `props_p1_data.contract.provenance()` and hashes to the sealed
   digest. This freezes `experiments/props_p1_data/**`, `experiments/props_scene/*.py`, `src/**`,
   the shared modules the collection imported, and the P1 spec and collection-plan docs
   (`plan.py`'s `SPEC`/`PLAN` constants pull these into provenance even though they live outside
   the package — see FAILURES #52; do not edit them, ever, for any reason).
5. `validation.json` equals the sealed counts (540 episodes, 480 + 60, 24 + 3 per cell, 2
   substituted slots, 0 held-out pairs collected).
6. Every file in `SOURCE_FILES` and the base model bundle (by hash).

`--check` additionally recomputes the dataset fingerprint over every file of the parent
(`contract.parent_fingerprint()`, about 1.1 GB) and refuses unless it equals the seal.

Two facts about the data that the contract pins by test:

- **The split comes from the plan.** LeRobot's `meta/info.json` marks all 540 episodes `train`
  (`{'train': '0:540'}`). Episodes 0-479 are training and 480-539 validation, from `plan.json`.
- **Tasks are matched by text.** LeRobot numbers the 10 instructions by first appearance, not in the
  plan's sorted order, so a task number is never trusted.

The approved instruction set is exactly the 10 trained instructions, derived at import time from
the sealed plan's `TRAINED_PAIRS` (the sealed `plan.py` itself stores no `INSTRUCTIONS` constant and
cannot be edited to add one). The two held-out pairs (`milk -> blue`, `bread -> orange`) exist only
in the evaluation package.

## Recipe and horizon

Identical to B3 except the horizon, the episode length and the data: effective batch 8 (microbatch
4 x accumulation 2), chunk 30, `n_action_steps` 10, peak LR 1e-4 with 500 warm-up updates and cosine
decay to 1e-5, VLM LR scale 0.1, weight decay 1e-4, betas (0.9, 0.95), eps 1e-8, clip 10, FP32
parameters with bf16 autocast, seed 20261007, from the base bundle in `.cache/xvla_trial/bundle`; no
automatic resume, but a manual one from a verified snapshot (below).

**Episodes are 452 frames (not 528).** 480 training episodes x 423 complete chunk starts (chunk 30)
= 203,040 starts. **126,900 updates = 5.0 epochs exactly** (203,040 x 5 / 8 = 126,900, no rounding
needed, unlike B3's 155,937.5).

**Checkpoint schedule is dense from epoch 2**, chosen from this repo's own checkpoint-peak evidence
rather than round numbers (B1 peaked at 3.35 epochs, B2 at 5.0, B3 at 4.0 with 5.0 its worst by 22
points):

| update | epoch |
|---|---|
| 1 | — (machinery sanity) |
| 50,760 | 2.0 |
| 76,140 | 3.0 |
| 88,830 | 3.5 |
| 101,520 | 4.0 |
| 114,210 | 4.5 |
| 126,900 | 5.0 |

Seven checkpoints total (about 3.3 GB each intermediate, 9.9 GB for the final one, which alone
carries `training_state.pt`; about 29.7 GB in all). Unlike B3's whole-epoch schedule, these do not
all land on a snapshot update (snapshot cadence is unchanged from B3, every ~2,600 updates, not
realigned to this denser grid). The disk guard is 65 GiB for training (checkpoints 29.7 GB plus the
resume snapshot working set of 2 kept + 1 being written, 3x~10.5 GB = 31.5 GB, peaking at ~61.2 GB;
a first-pass plan estimate of 55 GB missed the transient third snapshot), 28 GiB for preflight and
the resume proof, and 28 GiB of free GPU memory.

Offline metrics (loss, action error, gripper accuracy) never select a checkpoint, stop a run early
or support a claim about progress; only rollouts do.

## Resuming a failed run

**Snapshots.** After every 2,600th update (about 37 min at B3's measured 0.848 s/update) the runner
writes `resume/snap_NNNNNN/` holding the FP32 weights (`model.pt`), the Adam state (`optimizer.pt`),
the torch RNG state (`rng.pt`), the Python and NumPy RNG state and the SHA-256 of every file
(`snapshot.json`), and the run report (`report.json`). About 10.5 GB each. It is written under
`tmp_NNNNNN/` and renamed only when complete, and the newest two are kept. A crash loses at most
2,600 updates.

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
image). Mount the parent read-only and give `/tmp` plenty of headroom (the Hugging Face arrow cache
of a 244,080-row dataset lands there).

| gate | output | proves | cost |
|---|---|---|---|
| `--check` | `outputs/train/xvla_props_p1_v1_admission` | fingerprint recomputed, binding, statistics from the training rows only, processor checks | CPU, minutes to hours |
| `--preflight` | `..._v1_preflight` | 2 GPU updates, FP32 weight + Adam round trip, one snapshot | about 5 min |
| `--resume-proof` | `..._v1_resumeproof` | a resumed process reproduces update 2 exactly | about 5 min |
| `--train` | `outputs/train/xvla_props_p1_v1` | the run | about 30 h at B3's measured speed (0.848 s/update); preflight measures the real rate first |

```bash
python -m experiments.xvla_props_p1.runner --check | --preflight | --resume-proof | --train [--resume]
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
