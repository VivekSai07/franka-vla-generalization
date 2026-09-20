# Baseline archive, private publication and tracking

User approved W&B logging, publishing both final baselines to their HF account,
and the ordered research goals on2026-09-18. Explicit visibility response: PRIVATE
repositories. Preparation, tests and authenticated publication are authorized;
no model retraining or new perturbation rollouts in this publication stage.

## Scope

1. Portable immutable release copies of SmolVLA5000 and X-VLA5000. Private repos:
   VivekSai07/smolvla-franka-stacking-fixed-layout-v1 and
   VivekSai07/xvla-franka-stacking-fixed-layout-v1. No force overwrite of an existing
   nonempty repo. Final weights/processors only, not optimizer/training-state,
   credentials, entire datasets, absolute machine paths or indiscriminate source.
2. Exact source checkpoint hashes and explicit config rewrite map; portable loader
   promotes SmolVLA trainables toFP32 BEFORE strict weight copy, preservesXVLAFP32,
   uses packaged tokenizer/config assets. No hidden downloaded replacement weights.
   Test loading and prediction equivalence offline before upload. CPU tests may
   compare onCPU; do not claimGPUequivalence unless tested onGPU.
3. Model cards:20/20XVLA and18/20Smol on20samplingseeds/same selectedgeometry,
   20train4samegeometryvalidation, absolute7joints+gripper contract, nohardware
   safety/generalization claim. Actual customschedule500warmup/5000updates must
   override interpretation of stale serialized default scheduler fields in card,
   not silently rewrite original evaluated config. Record model/base license sources
   and distinguish missing/uncertain metadata. Model release is not fullsimulation
   reproduction package; dataset/envassets are not being uploaded here.
4. Local machine-readable registry of the two validated baselines initially, links
   to earlier experiment index. No claim that every historical attempt is indexed.
   Missing telemetry recorded null/not_recorded, not inferred.
5. W&B historical import of both baselines:5000loggedupdates/config/trainstats
   summaries/offlineprobes/all20outcomes. No cloud weights/videos or source auto
   capture. Disable system stats/code/git/console capture for imports; preserve
   original start/end and mark importtime distinct. Private W&B project
   franka-fixed-layout-baselines in authenticated personal account if available;
   verify private project before logging, otherwise stop W&B and report blocker.
6. Future live W&B integration must be external/newversioned wrappers, not edits
   to frozen trainers. Record GPUutil/memory/power only when sensors accessible;
   sharedGPUmetrics devicewide not per-run energy. Design next, not fabricate past.

## Credentials and remote controls

User explicitly authorizes process-only credential loading from .env. Never open,
display,diff or upload it/.env.example; do not shell-source it. A reviewed Python
process may parse with dotenv(interpolate=False), extract only necessary key,
never echo values, log headers/URLs/tracebacks or persist key. Token used only for
the corresponding service. Temporary writableHOME/HFcache for auth, no loginfile
written to repository. Authenticated HFwhoami must equal VivekSai07 beforecreation.
Never print identity payload, tokenpermissions or configenvironment; only expected
username/safe status and canonical project/repoURLs. Redirect SDKoutput and suppress
network exceptiontext; report exceptiontype+HTTPstatus only.

No proxybypass. Previous HF largeuploads hitHTTP407; standard endpoint failure
must stop, preserving localbundles and remote partialstate. Use bounded retries
for normaltransport only; no alternate route or credential escalation.

Original source/checkpoints/data/caches readonly. New package/tests/docs only;
release and audit output under outputs/publication/baselines_20260918. Reviewed
allowlists and secret-pattern scans before uploads. Verify remote committed files
byLFSsha256 or Gitblob hash plus sizes and repo visibility/revision. No success
claim until verification. Localmilestone commits solelyuser,noGitpush.

## Execution checklist

- [x] Public/private clarified: private.
- [x] Existing packages available:wandb0.24.2,huggingface_hub0.35.3,dotenv1.2.3.
- [ ] Portable packages and offline verification.
- [ ] Private authenticated remote publication and fileverification.
- [ ] Historical W&B import and privateproject verification.
- [ ] Results/registry and status handoff; unresolved transport gaps explicit.
- [ ] Separate next bounded perturbation design; no rolloutuntil exactbudgetreview.
