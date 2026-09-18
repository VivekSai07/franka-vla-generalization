# Historical CPU report archive — 2026-09-18

The seven reports below were previously untracked. They are committed unchanged
as provenance for completed experiments, not regenerated or presented as new
launch authorization. Their timestamps and `not_started`/GPU-unmeasured fields
describe the time they were written. See [current status](BENCHMARK_STATUS.md)
for completed training and policy results.

| Report | Interpretation |
|---|---|
| [Single-task verification](../experiments/single_task/CPU_VERIFICATION.json) | Historical 219-test admission; its six recorded source hashes still match. |
| [Initial X-VLA report](../experiments/xvla_trial/CPU_REPORT.json) | Superseded: two recorded source hashes differ from current preflight code/tests. |
| [Preserved initial X-VLA report](../experiments/xvla_trial/CPU_REPORT_PRESERVED.json) | Byte-identical preserved copy of the initial report; same two historical mismatches. |
| [Reviewed X-VLA report](../experiments/xvla_trial/CPU_REPORT_REVIEWED.json) | Its 101 recorded source hashes match current repository/installed sources. |
| [Initial X-VLA single-task tests](../experiments/xvla_single_task/CPU_TEST_REPORT.json) | Historical 44-test result before later additions; not a current test-count claim. |
| [Initial X-VLA single-task admission](../experiments/xvla_single_task/CPU_ADMISSION.json) | Superseded: four recorded new-source hashes differ from current code/tests. |
| [September 18 admission](../experiments/xvla_single_task/CPU_ADMISSION_20260918.json) | Its 114 recorded source/report references match. This is not a new full data/asset revalidation. |

Fresh checks before archiving:

- Seven JSON documents parse; common credential/private-key/signed-URL pattern
  scan found no matches. Credential files were not opened.
- **78 focused CPU tests passed**: single-task 31, X-VLA trial 20, X-VLA
  single-task 27; zero failures, errors or skips.
- Tests ran in the existing offline Docker image using runc, no network/GPU,
  read-only repository, temporary caches and separate Python processes per suite.
- The first verification harness passed 31 tests then hit duplicate test-module
  names across suites; separate processes resolved the harness import collision.
  Installed-source paths were resolved from the package rather than assumed.
  No project code was changed to make verification pass.
- No training, collection, model download or learned-policy rollout occurred.

The six existing machine-local offline build/runbook files remain on disk and
are narrowly ignored in [.gitignore](../.gitignore), retaining the prior decision
not to commit machine-specific setup. Datasets, checkpoints, videos, weights,
caches and wheels remain excluded. No push was requested or performed.