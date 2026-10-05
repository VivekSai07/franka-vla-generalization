# Documentation index

## Live documents

Kept current; read these first.

| file | content |
|---|---|
| [../README.md](../README.md) | entry point and navigation |
| [../TODO.md](../TODO.md) | live plan, known debt (the pre-launch checklist is in `.github/skills/launching-a-run/SKILL.md`) |
| [RESULTS.md](RESULTS.md) | every benchmark number, with protocols |
| [FAILURES.md](FAILURES.md) | failures, fixes, meta-lessons |
| [OPERATIONS.md](OPERATIONS.md) | how to run things now |
| [BENCHMARK_STATUS.md](BENCHMARK_STATUS.md) | chronological status log, newest first |
| [../DATASET_DESIGN.md](../DATASET_DESIGN.md) | v1 failure analysis, v2 design rationale |

## Naming convention

| kind | pattern | example |
|---|---|---|
| live document | `ALL_CAPS.md` | `RESULTS.md` |
| point-in-time report | `TOPIC_YYYYMMDD.md` | `LOCAL_VARIED_COLLECTION_20260919.md` |
| design spec | `superpowers/specs/YYYY-MM-DD-topic-design.md` | |
| implementation plan | `superpowers/plans/YYYY-MM-DD-topic.md` | |
| superseded material | `archive/<topic>.md` | `archive/docker-compose-pipeline.md` |

Dated reports are immutable records of what was true on that date. Do not edit them to reflect
later findings — add a new one and update the live documents.

## Point-in-time reports by topic

**Datasets and collection**
`LOCAL_VARIED_COLLECTION_20260919.md` · `DATASET_UPLOAD_20260916.md` ·
`DATA_COVERAGE_AND_PUBLIC_REFERENCES_20260918.md` · `LOCAL_POSITION_PROBE_20260919.md` ·
`LOCAL_VARIED_DATASET_PUBLICATION_20260921.md` · `B1_AND_BROADWORKSPACE_PUBLICATION_20260926.md` ·
`B3_ROBOSUITE_COLLECTION_20261003.md` · `B3_PUBLICATION_20261005.md`

**Training runs**
`SMOLVLA_PILOT_TRAINING_20260916.md` · `SINGLE_TASK_20260917.md` ·
`SMOLVLA_FIXED_LAYOUT_20260918.md` · `XVLA_FIXED_LAYOUT_20260918.md` · `XVLA_TRIAL_20260917.md`

**Evaluation and validation**
`B3_XVLA_EVALUATION_20261005.md` · `ROBOSUITE_SIM2SIM_20261002.md` · `B2_DISTRACTOR_EVALUATION_20260930.md` (+ `B2_DISTRACTOR_FAILURES_20260930.csv`) ·
`PILOT_EVALUATION_20260916.md` · `FULL_EVALUATION_REVIEW_20260917.md` ·
`VALIDATION_20260915.md` · `OVERFIT_VALIDATION_20260915.md` ·
`PRECISION_OVERFIT_20260915.md` · `FITTING_AUDIT_20260915.md`

**Simulator / scene**
`ROBOSUITE_SCENE_BRINGUP_20261002.md` · `PROPS_P0_20261005.md` · `PROPS_P0_RERUN_20261005.md`

**Expert / controller development**
`ADAPTIVE_PLACEMENT_V4_20260917.md` · `BOUNDED_PLACEMENT_V3_20260917.md` ·
`PLACEMENT_REACHABILITY_20260917.md` · `POSE_FIX_AND_EXTERNAL_ADVICE_20260917.md` ·
`JOINT_LIMIT_PREPARATION_20260915.md` · `REPLANNING_DIAGNOSTIC_20260915.md` ·
`TIMING_VALIDATION_20260915.md` · `PAIRED_DIAGNOSTIC_20260917.md` ·
`PAIRED_LEARNING_PREPARATION_20260917.md`

**Publication**
`BASELINE_PUBLICATION_20260918.md` · `LOCAL_VARIED_DATASET_PUBLICATION_20260921.md`

**Background and analysis**
`SMOLVLA_DATASET_SIZE_ANALYSIS.md` · `POLICY_INPUTS_AND_GROUNDING_OPTIONS.md` ·
`INSTRUCTION_AUDIT_20260917.md` · `CPU_REPORT_ARCHIVE_20260918.md` ·
`OVERNIGHT_PILOT_20260915.md` · `PILOT_PREPARATION_20260915.md` ·
`FIXED_LAYOUT_LAUNCH_20260918.md` · `RTC_*.md`
