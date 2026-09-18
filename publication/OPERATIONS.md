# Baseline publication operations

This is the remote-operations addendum to [README.md](README.md), whose offline
packager/loader/verification tools remain unchanged. The earlier W&B attempt
blocked at `personal_identity_check`; the latest attempt passed authentication and
the approved-entity gate but blocked at `project_privacy_check` before mutation.
Main's authenticated read-only introspection proved `Project.access: String` and
`UpsertModelInput.access: String` (with `entityName` and `name`); the old code had
incorrectly required `visibility`. The access-schema correction below is
offline-tested only; **main reruns W&B after review, preserving prior attempt
reports**. Model HF publication remains blocked
by HTTP 407; no retry is part of this correction. Private visibility was already
approved; no new permission request is needed. Never start a new training,
evaluation, monitor, or upload job implicitly from the historical importer.

## Exact CLI

Run inside the pinned CPU Docker image, from the repository mounted at `/vla`:

```sh
python -m publication.publish --registry
python -m publication.publish --hf
python -m publication.publish --wandb
```

Flags may be combined (`--registry --hf --wandb`). Independent service invocations
are recommended so a transport failure is easy to review. There are no arbitrary
repository, package, endpoint, token or API-key CLI options. Both service flags
target exactly the two baselines. No flag means no action, exit 2. Exit 0 means
all requested operations met their stated gates; exit 1 means blocked/partial.
Do not blindly rerun a partial remote attempt: collisions deliberately fail closed.

`--registry` is offline and never calls the credential parser. It writes a fresh
two-row JSON registry and attempt report in the ignored publication output. Each
row includes selected config/actual recipe, 5,000 original numeric update records,
20 counted trial outcomes, train-only statistics, probe summaries, source hashes,
documented original times, and explicit missing telemetry. It is not an index of
all historical experiments.

## Main-agent Docker launch contract

Image: `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`.
Use runtime `runc`, no GPUs, read-only root, current UID/GID, 4 GiB memory,
2 CPUs, and a private writable `/tmp` tmpfs (1–2 GiB). Mount the repository
read-only at `/vla`. Mount only the publication output directory writable at
`/vla/outputs/publication/baselines_20260918`; overlay its `packages` subdirectory
read-only again. Set `PYTHONDONTWRITEBYTECODE=1`, `HOME=/tmp/home`,
`CUDA_VISIBLE_DEVICES=` and `NVIDIA_VISIBLE_DEVICES=void`.

For `--registry` and tests: use `--network none`, no credential mounts.
For main's explicitly requested remote run only: use the normal approved network
and existing proxy settings, passed by environment **name**, never printed or
rewritten. Do not add a proxy bypass or alternate API endpoint. Mount the approved
private credential file read-only at `/run/secrets/publishing.env`; never use
`--env-file`, shell `source`, shell expansion of its contents, or a key CLI option.
Do not mount host home/auth caches. Do not open the source credential file or its
template in tools. The process uses `dotenv_values(..., interpolate=False)` and
extracts only the requested service key (`HF_TOKEN` and/or `WANDB_API_KEY`) plus
optional `WANDB_ENTITY`; all other parsed values are discarded. Ambient service
keys are not a fallback. The fixed secret mount has no override.

The CLI replaces inherited service/capture settings, switches to private temporary
HOME/cache/config/work directories, masks outputs, disables Python logging,
discards inherited subprocess stdout/stderr, and tears down W&B before cleanup.
SDK temporary debug files stay in the private tmpfs and are not upload inputs.
Only static phases, exception class, and numeric HTTP status are retained on
failure, never raw exception text/URLs/headers/signatures. Use normal CLI reports,
not raw SDK debugging output.

## HF gates and failure handling

Finalized local cards and legal notices are documented in
[card-finalization-report.md](../outputs/publication/baselines_20260918/card-finalization-report.md).
The exact optional additions to the release allowlist are `LICENSE` and `NOTICE`;
they receive the same text scan and inventory verification as other metadata.
No arbitrary legal-file suffixes, extra scripts or parent-directory files are allowed.
`reviewed_files` checks the historical numerical PASS/bit-exact/zero-error proof,
the verified weight hash and the **current** audited inventory. It does not compare
the entire package to historical metadata hashes, counts or sizes. Those reports
remain immutable historical evidence. The finalization audit separately proves
all weights, loader, configs, processor states, tokenizer/VLM assets and runtime
unchanged; it is not another inference run. The existing publisher gate alone is
not an immutable numerical binding for arbitrary future loader/config changes;
such changes require fresh numerical verification, not just rehashing metadata.

Only these private model repositories are allowed:

- `VivekSai07/smolvla-franka-stacking-fixed-layout-v1`
- `VivekSai07/xvla-franka-stacking-fixed-layout-v1`

1. Recheck existing CPU packaging proof and exact weights digest; reject symlinks,
   run `audit_release`, and hash the exact allowed files. No parent/folder upload.
2. Authenticate with `HfApi.whoami()` and require name exactly `VivekSai07`.
3. Refuse any existing repository, including empty ones; there is no reliable
   previous-attempt ownership exception. Creation is `private=True, exist_ok=False`.
4. Verify initial privacy/commit/inventory, preupload the explicit list using native
   SDK operations with one worker, recheck local hashes, then commit with expected
   parent SHA. Never shell Git or `upload_folder`.
5. Verify private immutable commit SHA and every remote file: LFS SHA256 **and**
   size, otherwise bounded small-blob download + Git-blob SHA1 + SHA256 + size.
   Only server-generated `.gitattributes` is allowed beyond the release allowlist;
   its downloaded hash/size are recorded too. Recheck private HEAD after verification.

API endpoint is fixed `https://huggingface.co`. Blob transport permits HTTPS HF
domains and the one previously observed native presigned HF LFS storage host,
`hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com`, only with a signature parameter.
HF bearer authorization is removed from every non-API host. An unrecognized
storage destination is a blocker, not authority to widen the allowlist or bypass
the proxy. This preserves the standard SDK upload route; it does not substitute
an alternate route. Requests have 10-second connect/60-second read bounds and a
30-minute per-model deadline. SDK transient retries remain bounded by that deadline;
HTTP 407 is not retried through another route.

Reports are exclusive attempt files, atomically updated at phase boundaries.
A preupload HTTP 407 after creation leaves `repo_created_no_modelcommit` with
`commit_verified=false`. An uncertain commit response is explicitly
`repo_created_modelcommit_unconfirmed`, not falsely asserted absent or successful.
Never delete a partial repo or automatically retry it. Preserve the report and
local packages; main must inspect remote state before any separately reviewed repair.

## W&B historical import

Fixed API `https://api.wandb.ai`, fixed project `franka-fixed-layout-baselines`.
The only approved entity is `vivekchaitanya07-university-of-stuttgart`.
Authenticate via native GraphQL `viewer { username entity }`: require a nonempty
username and default `entity` exactly equal to that approved slug, **not** to the
username. Optional `WANDB_ENTITY`, when provided, must also equal the approved
slug. Personal accounts and other teams are not implicitly allowlisted. Missing
viewer data or null/empty credentials fail closed before project access/mutation;
there is no ambient credential fallback.

Installed W&B 0.24.2's internal `default_entity` property returns
`viewer().get("entity")`; its native Viewer query includes both `username` and
`entity` (verified by offline installed-source inspection). This narrow gate uses
the authenticated approved default, not a guessed membership/`viewerCan...`
field. A different default blocks even when `WANDB_ENTITY` requests the approved
slug; non-default membership support is deferred. The new report phase is
`approved_entity_check`; preserve older `personal_identity_check` reports unchanged.

Offline verification of this correction: seven new tests; remote-operation/security
coverage increased from 34 to 41 tests, with all 63 publication tests passing in the
pinned CPU image (`--network none`, no credential mount). The approved-team and
null-safety regressions were observed failing before the production fix. No remote
retry, GPU access, upload, or commit was performed for this correction.

Public SDK Project/GetProject omits privacy and 0.24.2's `upsert_project` helper
has no access argument. Introspection selects one privacy field present on **both**
`Project` and `UpsertModelInput`: prefer `access`, otherwise `visibility`.
The actual schema uses `visibility: access` in project queries and mutation
returns, with `access: $visibility` in the native `upsertModel` input. Aliasing
keeps the importer/report contract unchanged. String and introspected enum types
are supported; enums must explicitly contain `PRIVATE`. Missing/mismatched fields,
unavailable introspection, or unsupported types halt W&B before project mutation.

The API value is uppercase **`PRIVATE`**, not `private`. Offline installed-source
inspection of W&B 0.24.2 confirmed its registry path supplies
`UpsertModelInput(access=Visibility.from_python(...).value)`, where the SDK enum
documents backend values and maps `organization` to `"PRIVATE"`. This is evidence
for the shared native input's casing, not use of registry semantics in this importer.
No unsupported `upsert_project(access=...)` call is made. Mutation responses must
return the exact project name and exact `PRIVATE`; lowercase/unknown values are
not normalized into permission. Independent privacy queries still run before
creation (including the immediate absence recheck), before any `wandb.init`, and
after SDK finish. Existing nonprivate/unknown projects are never modified.
Do not assume account defaults. Main should avoid concurrent creation of the same
project: the absence check/upsert is not a server-side compare-and-swap operation.

Access-schema correction verification: nine added regressions plus strengthened
payload assertions; all **72 publication tests passed** in the pinned CPU Docker
image (`--network none`, no credential mount). TDD reproduced the access-only
schema rejection, wrong field preference, and unchecked mutation response before
the fix. Tests cover actual String access schemas, legacy visibility fallback,
shared-field selection, enum validation, query/return aliases, exact mutation
payload/casing, nonprivate rejection, and pre/post-import privacy queries. Existing
credential and approved-entity protections remain unchanged. No credentials were
read, and no authenticated/network call, upload, or commit was made for this fix.

Run IDs are deterministic model + source-inventory hash; check absence and use
`resume='never'`. Any existing ID is refused, including failed/partial imports.
Exactly updates 1–5000 are logged at their original steps. Offline probes,
20-trial numeric table, success/failure counts and elapsed times are separate
summary data, not additional training steps. No model weights, videos, images,
source artifacts, entire bindings, or environment dictionaries are uploaded.

Historical config records `logging_mode=historical_import` plus original UTC job
times: SmolVLA 17:41:15–19:32:07, X-VLA 15:14:07–16:56:38 on 2026-09-18. These
are full container job times, not training-only durations. W&B's own run creation
time remains import time, additionally recorded as `import_started_utc`; history
is not assigned fabricated per-update wall timestamps.

Settings inspected and accepted by SDK 0.24.2: `x_disable_stats`,
`x_disable_meta`, `x_disable_machine_info`, `x_disable_viewer`, `disable_git`,
`disable_code`, `disable_job_creation`; `save_code=False`, `console='off'`,
quiet/silent, fixed host/program labels, explicit timeouts, zero GraphQL/file-stream/
file-transfer retries. No ambient settings may re-enable capture. Requests in the
control plane have 10/60-second bounds, init 60 seconds, per-run deadline 15 minutes,
and teardown/failure cleanup 30 seconds. No `wandb.login` or host login file.

`imported` means all local logs/summary were submitted, SDK `finish()` returned,
and private visibility was rechecked. The report explicitly states that exhaustive
server history readback is **not** implemented. Main has verified the actual schema
read-only; corrected remote creation/privacy readback and SDK synchronization still
await main's authorized execution.

## Interpretation and future live logging

SmolVLA: 18/20, frozen VLM, 99,880,992 trainable / 450,046,176 total parameters,
flow matching, chunk50, effective batch32, 160,000 sampled chunks. X-VLA: 20/20,
879,482,456 all trainable, native MSE, chunk30, effective batch8, 40,000 chunks.
Both use 5,000 updates and actual 500-update warmup, not stale serialized scheduler
defaults. Same geometry and 20 seed values are not spatial generalization, identical
noise, equal compute, or an architecture-only comparison.

Original logged PyTorch CUDA allocator counters remain under historical training
metrics. Past GPU utilization/power/energy are null/not_recorded. No current GPU
sampling is performed to fill history. Future live logging needs a separate,
new-version wrapper and reviewed execution budget; these custom trainers do not
natively integrate W&B. SDK/NVML live utilization/memory/power can only be recorded
when sensors are accessible and are device-wide on a shared GPU, not per-run energy.
No live adapter, frozen-runner edits, or monitor job is installed here.