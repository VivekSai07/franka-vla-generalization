# X-VLA fine-tune on the props P2 dataset

Training package for props P2, instruction sorting with 1–2 distractor props: X-VLA fine-tuned from the
base checkpoint on the sealed collection `outputs/props_p2_data/props_p2_20261013` (540 scripted
episodes in the robosuite scene: 480 training, 60 validation, 270 with one distractor and 270 with two;
10 trained object->bin instructions; `milk->blue` and `bread->orange` never collected as targets). Seal
fingerprint `b1d642b3edeebe15d462709ebeb8b92afa6d7eabcca3be5514e5deeecaf4d898`.

Design: [docs/superpowers/specs/2026-10-13-props-p2-distractors-design.md](../../docs/superpowers/specs/2026-10-13-props-p2-distractors-design.md).
Plan: [docs/superpowers/plans/2026-10-13-props-p2-training.md](../../docs/superpowers/plans/2026-10-13-props-p2-training.md).
Collection report: [docs/PROPS_P2_COLLECTION_20261013.md](../../docs/PROPS_P2_COLLECTION_20261013.md).

A fork of `experiments/xvla_props_p1/`. **The recipe is P1's, unchanged**, so P1 and P2 differ only in
their data. The checkpoint writer, strict FP32 reload, snapshots, resume and resource guards are P1's.
What changed: what is bound, and the training loop's plumbing (below), which is proven not to change
the math.

## What it binds

`current_binding()` checks, on every admission:

1. `seal.json` equals the six-field record in `contract.SEAL_RECORD`.
2. `plan.json` passes `props_p2_data.plan.validate_plan` and hashes to the sealed plan digest.
3. `substitutions.json` is the sealed empty record (P2 needed no substitutions).
4. `provenance.json` equals a fresh `props_p2_data.contract.provenance()` and hashes to the sealed
   digest. This freezes `experiments/props_p2_data/**`, `experiments/props_scene/*.py`,
   `experiments/robosuite_scene/*.py`, `src/**`, the shared modules the collection imported, and the P2
   spec and collection plan (FAILURES #52: do not edit them, ever).
5. `validation.json` equals the sealed record (540 episodes, 480 + 60, 12 per cell per distractor
   count, 3 per cell in validation, 270/270 distractor counts, 0 substitutions, 0 held-out pairs).
6. Every file in `SOURCE_FILES` and the base model bundle (by hash).

`--check` also recomputes the dataset fingerprint over every file of the parent and refuses unless it
equals the seal. As P1: the split comes from the plan (LeRobot calls all 540 episodes `train`), and
tasks are matched by text (LeRobot numbers them by first appearance).

## Recipe and horizon (P1's, unchanged)

Effective batch 8 (microbatch 4 x accumulation 2), chunk 30, `n_action_steps` 10, peak LR 1e-4 with 500
warm-up updates and cosine decay to 1e-5, VLM LR scale 0.1, weight decay 1e-4, betas (0.9, 0.95), eps
1e-8, clip 10, FP32 parameters with bf16 autocast, seed 20261007, base bundle in `.cache/xvla_trial/bundle`.
480 x 423 = 203,040 starts; **126,900 updates = 5.0 epochs**. Checkpoints at 1, 50,760, 76,140, 88,830,
101,520, 114,210, 126,900 (epochs 0, 2, 3, 3.5, 4, 4.5, 5); about 29.7 GB. Disk guard 65 GiB, 28 GiB of
free GPU memory.

Offline metrics never select a checkpoint, stop a run early or support a claim about progress.

## The faster loop: same math, less waiting

Measured on a real P1 update (0.896 s): forward+backward 0.332 s; `verify_precision` twice 0.321 s
(one GPU sync per tensor, ~4,500 per update); decoding the 8 samples 0.129 s with the GPU idle; Adam
0.090 s; clipping 0.017 s. The update itself is one function, `one_update`, used by both loops:

- **`verify_precision_fast`** runs every check `xvla_trial.preflight.verify_precision` runs, every
  update: presence, FP32 and `requires_grad` on the host, finiteness of parameters, gradients and both
  Adam moments in one fused max-abs pass (`torch._foreach_norm(..., inf)`, NaN/inf propagate). The
  shared helper is bound by other runs, so this is a local override; a test requires both to accept and
  reject exactly the same fixtures.
- **`Prefetcher`** decodes the next microbatches in a background thread, in exactly the sampler's
  order. One re-entrant lock serializes dataset reads and the shared-global instruction swap with the
  checkpoint probe.
- **`--equivalence`** is a gate: 30 updates with P1's loop (serial decoding, per-tensor checks), then
  30 with the fast loop from the same fresh state; per-update metrics, model weights and Adam moments
  must be byte-equal. `--train` refuses without it. It records both loops' seconds per update.

## Overnight batches

`--train --stop-at <unix time>` pauses at a snapshot: after each snapshot, if the next snapshot could
not be written before the deadline at this session's pace, it records `status: paused` and exits 0.
Continue the next night with `--train --resume --stop-at <time>` (P1's resume path, proven by
`--resume-proof`). A paused run is complete up to its snapshot; nothing is lost.

**Thermal guard.** Every 20 updates `nvidia-smi` is read inside the container (temperature, power,
thermal slowdown) and logged with the update row. At 83 °C or any thermal slowdown the loop sleeps
until the GPU is at 75 °C or below; sleeping changes no number in the run. After 30 minutes hot it
stops with an error, resumable from the last snapshot. The card's own limits stay in force (300 W
hard cap, 85 °C firmware target).

## Resuming

Snapshots after every 2,600th update (`resume/snap_NNNNNN/`: weights, Adam, RNG, hashes, report; ~10.5
GB, newest two kept). `--train --resume` restores the newest snapshot that still hashes to its record,
refuses if sources or data changed, replays the sampler to its position, restores the RNG last, and
continues; what the snapshot does not vouch for moves to `quarantine/resume_NN/`. A fix that needs a
file in `SOURCE_FILES` changed is a different run, not a resume.

## Gates

In order, each with its own output directory, in the admission image
(`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`), `--ulimit nofile=65536:65536`
on every GPU run (FAILURES #42), a large `/tmp`.

| gate | output | proves |
|---|---|---|
| `--check` | `outputs/train/xvla_props_p2_v1_admission` | fingerprint, binding, train-only statistics, processor checks (CPU) |
| `--preflight` | `..._v1_preflight` | 2 GPU updates, FP32 weight + Adam round trip, one snapshot |
| `--resume-proof` | `..._v1_resumeproof` | a resumed process reproduces update 2 exactly |
| `--equivalence` | `..._v1_equivalence` | the fast loop equals P1's loop byte for byte over 30 updates |
| `--train [--resume] [--stop-at T]` | `outputs/train/xvla_props_p2_v1` | the run |

```bash
python -m experiments.xvla_props_p2.runner --check | --preflight | --resume-proof | --equivalence | --train [--resume] [--stop-at T]
```

## Files

| file | content |
|---|---|
| `contract.py` | parent binding, data reader, instruction map, horizon, schedule, equivalence validator, `SOURCE_FILES` |
| `runner.py` | the gates, `one_update`, `verify_precision_fast`, `Prefetcher`, the pause and the thermal guard |
| `tests/test_contract.py`, `test_runner.py`, `test_fork_hygiene.py` | contract, runner and fork-residue tests |

**This README is in `SOURCE_FILES`: finish it before `--check`. Nothing in `SOURCE_FILES` is edited
between `--check` and the end of training.**
