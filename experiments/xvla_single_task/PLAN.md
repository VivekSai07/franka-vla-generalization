# Approved XVLA single-task implementation plan — 2026-09-17

**Goal:** implement the already-approved fresh-base 5,000-update comparison, with a separate mandatory two-update exact-batch scratch gate. This session performs CPU verification only; main reviewer runs scratch and launch later.

**Architecture:** three production modules: contract (admission/batches/schedule), runner (scratch/full training and strict reload), evaluate (explicit XVLA port of baseline rollout). Tests and this readable plan live only in this new package. No old source, installed runtime, cache, or recorded report is edited. No commits, installs, downloads, credentials, or agent-launched GPU jobs.

## Fixed contract

- Seed 20260917; train episodes 0–8, validation 300–303, red-on-green/two-cube only. Statistics from all 4,752 training rows; 4,491 unpadded length-30 training starts. No validation sampling or checkpoint selection.
- 5,000 updates; microbatch 4, accumulation 2, effective 8; scratch exactly 2 updates with the same schedule/batches/precision. Main always loads pinned original base, never scratch weights or optimizer. No resume, retry, OOM fallback, autotune, extension, or intermediate rollouts.
- All 901 named parameter tensors / 879,482,456 scalars trainable FP32, FP32 gradients/Adam, BF16 autocast compute. Native clean-action continuous MSE across all 8 real action coordinates; no gripper sigmoid.
- Real state **9** (7 joints + 2 fingers), physical action **8** (7 joints + gripper); preserve existing contract despite transposed dimensions in request shorthand. Chunk30, execution horizon10, domain0, two ordered RGB512 cameras, native padded resize224/ImageNet, masked zero third view, tokenizer64; STATE IDENTITY, ACTION MEAN_STD. No LIBERO rotation or object truth in policy inputs.
- Policy/soft prompts peak1e-4, VLM0.1x; warmup updates1–500, cosine to 10% at5000. Native named groups; weight decay1e-4 except VLM1e-5, betas(.9,.95), eps1e-8, clip10.
- Store model/config/processors at updates1,1000,2000,3000,4000,5000 (scratch1,2). **Adam only at final update of each run** to fit disk. Scratch strict FP32 model+Adam exact restoration and zero prediction error; full strict final model reload and zero prediction error, no full Adam restoration. Save stats, sampling and checkpoint metadata/hashes.
- Minimum free disk: scratch15GiB; full32GiB (six ~3.28GiB models + final ~6.56GiB Adam + logs/videos/headroom). Recheck before every save. Stop on disk failure; never delete artifacts automatically.
- Admission verifies every *listed* old source hash individually, not recursive old `source_hashes()`. Preserve and bind old GPU result and reviewed CPU report bytes. New explicit source list binds these modules/tests/plan plus extra installed dependencies. Full data replay/runtime/final validation and pinned asset checks before/after scratch, training, evaluation. New files never repair or rewrite old report hashes.
- Final-only13 original-reset rollouts: seen9, held-out4. Same30Hz/20physics substeps,1200ticks/40s, strict30tick stack score, threshold127 preserving0/255, unchanged clipping diagnostics/videos/traces. Report rates and provisional8/9+3/4 gate, never statistical generalization.

## Implementation / verification steps

- [x] Write CPU tests for immutable listed bindings, missing/mutated sources, exact recipe and LR bounds, train-only sample/stats admission, real processor batch4 and masked third view, 4x2 gradient equivalence on a tiny objective, strict checkpoint/Adam restoration and rollout queue semantics.
- [x] Observe tests fail for absent new implementation; implement only the three modules.
- [x] Run new CPU tests plus all old XVLA helper tests offline inside unchanged Docker with no GPU exposure. Include real dataset batch4 processing, not a full model gradient allocation. **44 tests passed: 24 new +20 preserved.**
- [x] Run read-only full source/data/cache admission: **101 old listed sources +11 explicit new/additional sources pass**. AST comparison proves rollout differs only by documented adaptations. The final `--check` saves CPU_ADMISSION.json; both GPU modes require that exact frozen binding. Reports do not hash themselves.
- [x] Leave both GPU output directories absent. Provide exact manual Docker scratch-then-full CLI and failure handling. Reviewer checks source first; source edits after CPU freeze or successful scratch block launch. Never silently regenerate a proof to admit changes.
- [ ] **Main reviewer, not this coding session:** review source, run exact scratch, inspect result, then launch fresh main. GPU batch4 feasibility and capability remain unmeasured.

## Rollout port differences from validation/evaluate_pilot.py::run_trial

An explicit source copy, not a config-spoofing subclass. Change only chunk assertion/report50→30; explicit seed20260917+episode_index; BF16 inference autocast; a strict observation-only frame/preprocessor/model-input precheck (state9, action8, camera order, domain0, zero padding/masked view). Preserve original policy.reset(), native select_action(), temporary _get_action_chunk counter, physics, scoring, gripper adapter, trace/video capture and ceil(attempts/10) check. Baseline source hash stays individually bound. CPU AST comparison verifies the rest of the port.

## Exact manual launch — NOT executed by implementation agent

Use the canonical **/vla** mount: the existing immutable dataset validation binds its scene-bank path there. A /workspace-only mount fails that gate (sources/runtime still match, but the recorded scene-bank path is absent). The fix is the mount, never a rewritten old validation report. Use the already-installed local image; `--pull=never`, no network, no env-file or credentials. No parallel GPU jobs. The original successful micro1 preflight used NVIDIA runtime and8GiB shared memory.

After main review, run scratch in foreground (container retained for exit/OOM inspection):

```bash
cd /home/utxowqr/smolvla_franka
docker run --pull=never --name xvla-single-task-preflight-20260917 --runtime=nvidia --gpus 'device=0' --network=none --shm-size=8g -e NVIDIA_VISIBLE_DEVICES=0 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -e MUJOCO_GL=egl -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 -v "$PWD:/vla" -w /vla --entrypoint python smolvla:rtc-offline -u -m experiments.xvla_single_task.runner --preflight
```

Inspect before launching main: exit0/noOOM; status complete; completed_updates2; microbatch4/accumulation2; strict_reload_fp32_exact=true; adam_reload_fp32_exact=true; reload_max_absolute_error=0; final_data_binding_passed=true; peak memory and latency. Scratch has **zero rollouts**. Main re-verifies these proof fields, artifact inventory, source/data/cache binding and CPU report; it never initializes from scratch.

```bash
docker inspect xvla-single-task-preflight-20260917 --format '{{.State.ExitCode}} {{.State.OOMKilled}}'
jq '{status,completed_updates,microbatch,accumulation,strict_reload_fp32_exact,adam_reload_fp32_exact,reload_max_absolute_error,final_data_binding_passed,final_memory}' outputs/train/xvla_single_task_20260917_preflight/result.json
docker run -d --pull=never --name xvla-single-task-5000-20260917 --runtime=nvidia --gpus 'device=0' --network=none --shm-size=8g -e NVIDIA_VISIBLE_DEVICES=0 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -e MUJOCO_GL=egl -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 -v "$PWD:/vla" -w /vla --entrypoint python smolvla:rtc-offline -u -m experiments.xvla_single_task.runner --train
```

Main outputs: outputs/train/xvla_single_task_20260917. Training result is separate from evaluation/result.json: the latter contains the final13 seen/held-out rates and provisional gate. Program integrity is not capability success. All planned budgets stop regardless of capability.

**Failure handling:** OOM/nonfinite/missing asset/stale binding/disk error stops immediately with exit1 and a best-effort safe result plus CUDA memory; no microbatch change, retry, resume, cleanup or automatic main launch. Kernel/container OOM-kill cannot write a Python failure report: inspect retained Docker exit/OOMKilled and preserve partial output. A nonempty output directory is never reused. Request a new decision after any failure; do not delete proof directories to rerun. Minimum disk15GiB scratch/32GiB main,12GiB before final model+Adam save,5GiB before other model saves,2GiB before each rollout.

CPU verification uses the same local image with `--runtime=runc --network=none`, `NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=` and the /vla mount. Test modules: experiments.xvla_single_task.tests.test_contract, experiments.xvla_trial.tests.test_contract, experiments.xvla_trial.tests.test_preflight, experiments.xvla_trial.tests.test_assets. The old downloader test mocks network failure inside a temporary directory; no downloads occurred. `--check` performs read-only admission then writes only the new exclusive CPU report. Do not call the old preflight --inspect or run_gpu entrypoint.