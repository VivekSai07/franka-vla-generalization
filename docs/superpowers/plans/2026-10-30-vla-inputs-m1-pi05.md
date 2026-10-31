# M1 — pi0.5 on P2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> This plan fixes files, interfaces, tests and gates; the code is written at execution time, test first, in the order below.

**Goal:** pi0.5 fine-tuned on the sealed P2 dataset with its own recipe at 5.0 epochs, evaluated as arm A1 by M0.

**Architecture:** two new images (existing image + transformers@`dcddb97` and tokenizers 0.21.4 only); hash-pinned assets; a training package
`experiments/pi05_props_p2/` with the X-VLA runner's gate structure; a `Pi05Adapter` in M0's evaluator.

**Tech stack:** LeRobot 0.4.4 `pi05`; new images `admission-pi` (training, tests) and `robosuite-pi` (rollouts).

**Spec:** [../specs/2026-10-30-vla-inputs-m1-pi05-design.md](../specs/2026-10-30-vla-inputs-m1-pi05-design.md).
**Depends on:** M0 Tasks 1–6 before Task 6 here.

## Global constraints

- The existing images are never rebuilt, retagged or removed. Images belonging to other users are never touched.
- Docker builds: classic builder (`DOCKER_BUILDKIT=0`), host proxy as build args (`--build-arg http_proxy=... https_proxy=...`).
- The Hugging Face token is typed by the user into a host terminal; it is never pasted into chat, written to a file in the
  repository, or passed into a container.
- Exposure: 5.0 epochs over 193,440 windows (chunk 50) = 967,200 samples = **120,900 updates** at effective batch 8, derived in
  `recipe()`, never retyped. Checkpoints 1, 48,360, 72,540, 84,630, 96,720, 108,810, 120,900.
- Execute 10 actions per inference at evaluation (`n_action_steps` 10).
- The memory strategy (spec D5) is fixed in `recipe()` before `--check`; changing it afterwards is a new run.
- Nothing in `SOURCE_FILES` is edited between `--check` and the end of training.

---

### Task 1: the two images

**Files:** create `docker/pi05/Dockerfile.admission-pi`, `docker/pi05/Dockerfile.robosuite-pi`, `docker/pi05/README.md`,
`experiments/pi05_props_p2/tests/test_image.py`.

- [x] **Step 0 (done 2026-10-30; tokenizer files downloaded, `pi05_base` weights not yet):** the user requests access to `google/paligemma-3b-pt-224` on Hugging Face
  (gated, **manual** approval, checked 2026-10-30). Task 2 cannot start without it.
- [x] **Step 1 (resolved 2026-10-30 on the RTX machine; re-verify, do not re-derive):** LeRobot 0.4.4's `pi` extra pins the
  branch `fix/lerobot_openpi`, which is commit `dcddb970176382c0fcf4521b0c0e6fc15894dfe0` (transformers 4.53.3). That commit
  requires `tokenizers>=0.21,<0.22`; the parent image has 0.22.2. Install exactly
  `transformers @ https://github.com/huggingface/transformers/archive/dcddb970176382c0fcf4521b0c0e6fc15894dfe0.zip` and
  `tokenizers==0.21.4`, both `--no-deps`. A throwaway probe with these two gave `siglip check` True and a constructed
  `PI05Policy`. Record both in `docker/pi05/README.md`. If `git ls-remote` now shows a different branch head, keep the
  recorded commit.
- [x] **Step 2: failing test** `test_image.py` (run in the new image): `transformers.models.siglip.check
  .check_whether_transformers_replace_is_installed_correctly()` is true; `PI05Policy(PI05Config(...))` constructs on CPU with
  the default variants and has 3,616,757,520 parameters with the P2 features; `pip check` passes; `pip freeze` equals the
  parent image's except the transformers and tokenizers lines (the parent's freeze is captured to
  `docker/pi05/parent_freeze_<digest>.txt` first).
- [x] **Step 3:** build both images FROM the existing digests (`DOCKER_BUILDKIT=0`, proxy build args), installing the two pinned
  packages with `--no-deps`; record both new digests in `docker/pi05/README.md` and `docs/OPERATIONS.md`. A container that
  runs code from `/tmp` needs `--tmpfs /tmp:rw,exec` (Docker's tmpfs is `noexec` by default; the probe failed on it once).
- [x] **Step 4:** run the test in both images; pass. **Physics unchanged:** run `props_p2_data.record.determinism` (12 rendered
  attempts) in `robosuite-pi` and in the robosuite image, into scratch directories; the 12 digests must be equal.
- [x] **Step 5:** commit `M1: pi0.5 images`. *Done 2026-10-30:* `vla-pi05:admission` `sha256:a229f2fb…290eae`, `vla-pi05:robosuite`
  `sha256:2abb18a5…b21`; installed with `--require-hashes` (the freeze shows `#sha256=` on the transformers line); the test
  fails in the unmodified parents; 12 of 12 determinism digests equal. `docker/pi05/README.md`.

### Task 2: assets

**Files:** create `experiments/pi05_props_p2/assets.py`, `tests/test_assets.py`.

- [x] **Step 1:** on the host, the user logs in to Hugging Face (PaliGemma access approved, Task 1 Step 0) and downloads
  `lerobot/pi05_base` at revision `9e55186ad36e66b95cda57bc47818d9e6237ae30` (weights sha256
  `0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f`, 14,467,165,872 bytes; the latest revision's
  preprocessor uses a `relative_actions_processor` step that LeRobot 0.4.4 lacks) and the tokenizer files of
  `google/paligemma-3b-pt-224` into `outputs/research/pi05_assets/`. Disk: measure with `du -sh`.
- [x] **Step 1b:** test that the weights load into `PI05Policy` with zero missing and zero unexpected keys (the loader
  prints missing keys and continues, so assert on the returned lists), and that processors are built from the local config
  and P2 statistics, never from the hub's `policy_preprocessor.json`.
- [x] **Step 2: failing test:** `assets.require_assets()` hashes every file and compares with the pinned table; a one-byte change
  in a copied fixture fails it.
- [x] **Step 3–4:** write the pinned table from the downloaded files; pass.
- [x] **Step 5:** commit `M1: pinned pi0.5 assets` (hashes only; never the weights). *Done 2026-10-30:* downloaded anonymously (the
  repository is public), 14 GB. Step 1b: `from_pretrained(strict=True)` always lists the tied `embed_tokens.weight` as missing and only
  prints a warning, so the test asserts on the key sets (a missing key is accepted only if it shares storage with a loaded one) and
  on every tensor. The processor half of Step 1b is in Task 3's `test_processor.py`.

### Task 3: contract

**Files:** create `experiments/pi05_props_p2/__init__.py`, `contract.py`, `README.md`, `tests/test_contract.py`,
`tests/test_fork_hygiene.py`.

**Interfaces:** `contract.SOURCE_FILES`; the P2 dataset binding copied from `experiments/xvla_props_p2/contract.py`
(`PARENT`, `SEAL_RECORD`, splits, instruction checks) — copied, not imported, so P2's file stays untouched and the copy is
tested against the original on the real dataset; `contract.windows()` (start frames with a full 50-step chunk inside the
episode); `contract.statistics()` (training rows only: `mean`, `std`, `min`, `max`, `q01`, `q99` for `observation.state` (9)
and `action` (8)); `contract.recipe(preflight: bool)`; `contract.make_config(device)` (a `PI05Config` with the spec's values,
`n_action_steps` 10, the two image features, `max_state_dim` 32); `contract.current_binding()`.
- [x] **Step 1: failing tests:** `len(windows()) == 193_440`; `recipe(False)["max_optimizer_updates"] == 120_900` computed as
  `round(5.0 * len(windows()) / 8)`; checkpoint grid equals the fractions (0.4, 0.6, 0.7, 0.8, 0.9, 1.0) of the run plus 1;
  statistics read no validation row (count the rows); the copied dataset checks accept the sealed P2 dataset and reject a
  fixture with one instruction changed; on CPU, the processor matches the installed reference prompt for a real training frame and quantile-boundary/tail
  fixtures (including values below q01), has 32 state tokens, and preserves the full P2 instruction without truncation.
  Do not assume quantile scaling clips; record native out-of-range token behavior and any explicit recipe deviation. Fork hygiene as the forking skill (residue grep,
  symtable scan).
- [x] **Step 2–4:** fail, implement, pass, mutate (typed 120900, validation rows in statistics, chunk 30 windows).
- [x] **Step 5 (done 2026-10-30; 69 tests OK in `admission-pi`; mutations of the typed horizon, validation rows in the statistics and chunk 30 each
  fail the suite):** README written now (it is in `SOURCE_FILES`; confirm non-zero size). Commit `M1: contract`.

### Task 4: runner and the memory probe

**Files:** create `runner.py`, `tests/test_runner.py`.

**Interfaces:** CLI modes `--probe-memory` (not a gate; scratch output; never after `--check`), `--check`, `--preflight`,
`--resume-proof`, `--train [--stop-at T] [--resume]`. The loop, prefetcher, snapshot/resume, thermal guard and pause-at-snapshot
follow `experiments/xvla_props_p2/runner.py` (copied and adapted, not imported). Disk and GPU guards derived from measured
sizes in `recipe()`.
- [x] **Step 1: failing tests (pure):** mutually exclusive modes; `--resume` only with `--train`; the horizon comes from
  `recipe()`; the disk guard equals the formula over measured sizes; snapshots kept = 1 plus the one being written.
- [x] **Step 2–4:** fail, implement, pass, mutate. *Done 2026-10-31* (tests were written with or just before the code, not run red
  first; seven mutations each fail the suite). The runner also has the gate modes, unexercised until a decision exists.
- [ ] **Step 5 (GPU, admission-pi):** `--probe-memory` for spec D5's candidates (a) all-bf16, (b) bf16 on GPU with FP32
  master weights and moments offloaded to CPU, (c) `train_expert_only`: peak memory over 20 updates, seconds per
  update, checkpoint and snapshot sizes, and for (a) and (b) the per-tensor fraction of elements changed by one update at the
  peak and the floor learning rate. FP32 on-GPU full tuning (57.9 GB before activations) and frozen-vision FP32 (~52 GB) are
  not probed: they cannot fit. Write `outputs/train/pi05_props_p2_probe/probe.json`.
- [ ] **Step 6:** **user decision** on the memory strategy and the time/disk projection, recorded in `README.md` and
  `recipe()`. Commit `M1: runner, memory strategy <x>`.

### Task 4b: numerical and reference-path proof

- [ ] Pin installed source hashes, optimizer implementation and transformers commit; `pip check`, offline load and actual
  forward/backward/update/inference must pass. If the two-package install (transformers + tokenizers) cannot satisfy dependencies, stop and amend
  the manifest before training; do not weaken the gate or mutate the old images.
- [x] *(CPU, done 2026-10-31; the real-model optimizer update is measured by the probe)* Compare custom runner loss, gradients and update against the installed policy/processor reference on a fixed batch;
  check padding/masks, real action dimensions, camera order, prompt length and physical-action round-trip.
- [ ] Record parameter/gradient/moment/master-weight dtypes. Both bf16-weight candidates can lose sub-ULP updates;
  fraction-changed is diagnostic only. Verify the optimizer supports its actual dtype configuration and resumes exactly.
- [ ] Decide effective batch independently from microbatch memory (accumulation can retain 32); freeze the chosen recipe
  before check. A supported LoRA/frozen-backbone fallback is a named different arm recipe, not full-tuning equivalence.
- [ ] Report complete-window starts, observation samples, action-target positions and schedule phase. Different chunks have
  different start support despite five epochs each. Include all retained gate artifacts and image/assets/video storage in
  disk projections, plus temporary atomic writes and the final optimizer save.

### Task 5: gates and training

- [ ] Pre-launch checklist ([launching-a-run](../../../.github/skills/launching-a-run/SKILL.md) section 1); disk decision by the user.
- [ ] `--check` → `--preflight` (2 updates, weight and optimizer round trip) → `--resume-proof` (a resumed process reproduces
  update 2 byte for byte) → `--train --stop-at <deadline>` in nights from a detached worktree that is never switched.
- [ ] Each night: status entry in `docs/BENCHMARK_STATUS.md` (updates, pace, pauses; no offline metric used for anything).

### Task 6: evaluation as A1

**Files:** create `experiments/pi05_props_p2/eval_adapter.py` and `experiments/pi05_props_p2/eval_tests/test_pi05_adapter.py`.
Neither is listed in the training contract's `SOURCE_FILES` (an explicit tuple; a test asserts their absence), so writing
them after training does not invalidate the training binding; M0's arm manifest binds them. Nothing is added to
`experiments/vla_inputs_eval/` (its tests directory belongs to the frozen core).

- [ ] **Step 1: failing test (robosuite-pi, GPU):** `Pi05Adapter(checkpoint, state_p2)` returns (8,) actions; with the
  checkpoint's statistics the unnormalised action of a fixed observation equals the policy's own `select_action` →
  postprocessor path; the adapter's queue re-plans every 10 calls.
- [ ] **Step 2–4:** fail, implement, pass, mutate (n_action_steps 50 → test fails).
- [ ] **Step 5:** smoke in `robosuite-pi` (memory per worker → worker count); S on the six candidates; `select`; C with the
  winner and predeclared 4.0-epoch secondary checkpoint (deduplicated); `compare --arm A1`.
  Load the separately bound adapter lazily; do not edit M0 core. Reserve GPU time separately from training.
- [ ] **Step 6:** `docs/VLA_INPUTS_M1_<date>.md` (recipe, memory strategy, pace, selection, primary result, per object,
  failure causes, against the spec's prediction); docs pass over `RESULTS.md`, `BENCHMARK_STATUS.md`, `TODO.md`, `INDEX.md`;
  commit.
