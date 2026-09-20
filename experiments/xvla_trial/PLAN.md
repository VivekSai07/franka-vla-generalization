# Bounded XVLA trial — implementation and authorization plan

**Goal:** compatibility/resource evidence for the approved existing-data model-plus-recipe comparison; not a training launch.

**Isolation:** all new source/tests/docs here. Existing sources, datasets, checkpoints and benchmark cache remain untouched. No commits, credentials, installs, alternate routes, collection or rollouts.

## Stages

1. Inspect installed LeRobot 0.4.4 XVLA configuration, preprocessing, clean-action objective, inference, optimizer and storage. Inspect public HF metadata and pin revisions before download.
2. TDD: declarative episode admission first; then public downloader confinement/integrity; then real processor, padding, objective, optimizer and checkpoint tests. CPU Python runs only inside the unchanged Docker image, runtime runc, no GPU access.
3. Download only fixed public assets into `.cache/xvla_trial`, using public HF API/standard download, explicit token=False, fresh HF home and proxy environment names only. Stop on dependency/proxy/download failure, including 407. No route or host workaround. Record hashes/bytes and safe failure status.
4. Implement a separately gated preflight: fresh `outputs/validation/xvla_preflight_20260917`, microbatch1, accumulation1, exactly two maximum optimizer updates, full-model gradients, FP32 parameters/Adam with BF16 autocast. Record memory/latency and strict save/reload prediction preservation. OOM stops; no autotuning or resume. **Do not execute GPU work in this session. Main review/explicit authorization required.**
5. After successful actual preflight, review measured hardware cost, batch/accumulation, LR schedule and resource budget with user. Only then consider fresh-base 5,000 updates and final-only 13 rollouts (seen0–8/held-out300–303). No full-training or evaluation entrypoint is part of this implementation.

## Explicit recipe choices

- Exact train0–8 and held-out300–303: red-on-green, two cubes, 528 frames/episode; train-only raw statistics, 4,752 rows and 4,491 complete length30 training chunks. Keep 9-state/8-action real schemas and 20D model capacities.
- `auto`: clean normalized action MSE over all eight real coordinates including gripper; padded output coordinates excluded. The installed model predicts clean action, not flow velocity. Temporal padded chunks are rejected because stock forward does not consume their mask.
- Two ordered real 512RGB cameras; stock ImageNet preprocessing and padded resize224. `num_image_views=3`, `empty_cameras=0`: missing third view is zero and masked by policy preparation, with zero auxiliary tokens retained. Do not introduce a fake real feature or LIBERO transforms.
- Domain0 matches actual processor/policy default; no claim it is a pretrained Franka domain. Full fine-tuning, not a LIBERO adapter.
- Preserve checkpoint ACTION MEAN_STD and STATE IDENTITY mapping, compute/save state statistics too. Tokenizer max64 explicitly replaces published1024: transformer maximum512 includes language, vision and actions. Verify instruction is untruncated and sequence fits.
- Full fine-tuning with XVLA named-parameter AdamW: VLM LR0.1x, other/soft prompts1x; base LR1e-4, betas(.9,.95), decay1e-4 for the two-update scratch only. Main schedule/batch remain unapproved.
- Set config dtype=float32 before construction/loading; do not use SmolVLA precision loader. Check every trainable parameter, gradient and Adam moment, plus saved tensor dtype and exact strict reload. BF16 autocast is compute-only. No native BF16 fallback.

## Verification checklist

- [ ] Tests written and observed failing before implementation.
- [ ] CPU contracts and installed-code integration pass.
- [ ] Public asset download result recorded (complete or safely blocked).
- [ ] CPU full-checkpoint schema/dtype inspection if assets available.
- [ ] Dataset replay/provenance checked read-only; preflight rechecks before/after.
- [ ] Source review and exact runnable CLI documented.
- [ ] GPU preflight **not run; awaits review**.
- [ ] Full recipe/resource approval **not granted**.