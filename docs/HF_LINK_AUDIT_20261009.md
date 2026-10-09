# HF documentation link audit - 2026-10-09

Repository reviewed: `C:\d drive\RTX trained models\franka-vla-generalization`.
Base Git commit: `dc0e5eb4b7a456007fc6c3fe829c51c20b88241d`.
Local evidence directory: `C:\d drive\RTX trained models\_publish_work\repo_hf_links_20261009`.

Added current HF navigation for the published Franka programme and preserved the historical/sealed evidence. Changes are local documentation edits; no GitHub push, HF upload, training, or evaluation is part of this documentation pass.

## Documentation coverage

The pre-change inventory covers **148 tracked Markdown documents, 1,437,283 bytes**: 90 historical/report/plan/spec documents, 48 experiment/instruction/publication documents, and 10 root/live documents. Hidden .github instructions and skills were included; .claude contains no Markdown files. Long historical and live docs were read in bounded chunks, with truncated reads repeated. Coverage ledgers and content hashes are retained in the local evidence directory.

## What was added

**Two new files and 18 modified files (20 total)**. The table lists the concrete additions; all paths are relative to the Windows repository named above.

| File | Addition |
|---|---|
| [AGENTS.md](../AGENTS.md) | Added HF artifact navigation for future agents without changing repository rules. |
| [CLAUDE.md](../CLAUDE.md) | Added the current artifact map to Claude orientation. |
| [DATASET_DESIGN.md](../DATASET_DESIGN.md) | Added verified legacy v1/v2 links and clarified that proposed collection changes are not current uploaded contents. |
| [README.md](../README.md) | Added seven-axis HF download table, checkpoint/model associations, privacy/missing-model notes, and navigation to the full artifact index. |
| [TODO.md](../TODO.md) | Updated current P1/B3 publication links and status, corrected stale B2 publication item while retaining incomplete-archive follow-up, and linked current artifact navigation. |
| [docs/BENCHMARK_STATUS.md](BENCHMARK_STATUS.md) | Added a newest-first dated publication outcome for P1/B3 and dataset repairs, with current artifact mapping and historical blockers preserved. |
| [docs/FAILURES.md](FAILURES.md) | Added artifact navigation beside the lessons, preserving every failure and result. |
| [docs/HUGGING_FACE_ARTIFACTS.md](HUGGING_FACE_ARTIFACTS.md) | Created the central map for seven programme axes, legacy artifacts, all uploaded checkpoint subdirectories, historical reports/packages, immutable revisions/access, download recipe, and portability/missing-publication caveats. |
| [docs/INDEX.md](INDEX.md) | Added the live HF index and dated link-addition audit to documentation navigation. |
| [docs/OPERATIONS.md](OPERATIONS.md) | Added artifact download/verification navigation and direct B3/P1 repository links alongside the bundle census. |
| [docs/RESULTS.md](RESULTS.md) | Added dataset/model links at P1, B3, B2, B1, matched-exposure, sim2sim, fixed-layout, and local-varied result sections; kept numerical results/protocol caveats unchanged. |
| [experiments/robosuite_scene/README.md](../experiments/robosuite_scene/README.md) | Added navigation from scene bring-up to later B3/P1 publications, preserving the diagnostic/policy distinction. |
| [experiments/robosuite_sim2sim/README.md](../experiments/robosuite_sim2sim/README.md) | Linked sim2sim to B2 weights/data without inventing a separate model. |
| [experiments/xvla_b2_distractors_eval/README.md](../experiments/xvla_b2_distractors_eval/README.md) | Added correct B2 dataset/root checkpoint links and archive-gap navigation. |
| [experiments/xvla_b2_distractors_eval/analysis/README.md](../experiments/xvla_b2_distractors_eval/analysis/README.md) | Added the analyzed B2 model/dataset and earlier-checkpoint mapping. |
| [experiments/xvla_b3_robosuite_eval/README.md](../experiments/xvla_b3_robosuite_eval/README.md) | Added B3 dataset, primary and alternate checkpoint links with original-evaluator portability caveat. |
| [experiments/xvla_props_p1_eval/README.md](../experiments/xvla_props_p1_eval/README.md) | Added P1 dataset, confirmed/alternate checkpoint links and bundle-wrapper reconstruction navigation. |
| [publication/OPERATIONS.md](../publication/OPERATIONS.md) | Added a current download/status callout while preserving historical HTTP407 details and the publisher -v1 allowlist. |
| [publication/README.md](../publication/README.md) | Added actual fixed-layout dataset/model downloads and live publication navigation; offline tooling behavior unchanged. |
| `docs/HF_LINK_AUDIT_20261009.md` | This detailed report: coverage, per-file additions, exact mappings, verification, exclusions, and Windows handoff paths. |

## Artifact links and exact mappings

The central [HUGGING_FACE_ARTIFACTS.md](HUGGING_FACE_ARTIFACTS.md) covers **19 verified repositories: 10 datasets and 9 models**. Seven programme datasets map to eight programme model repositories; three older Franka datasets and the legacy 80,000-update SmolVLA model are in a separate legacy section.

| Programme stage | Dataset | Published root checkpoint |
|---|---|---|
| Fixed-layout baseline | [24 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-fixed-layout-24) | [SmolVLA 5,000](https://huggingface.co/VivekSai07/smolvla-franka-stacking-fixed-layout); [X-VLA 5,000](https://huggingface.co/VivekSai07/xvla-franka-stacking-fixed-layout) |
| Local-varied | [60 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-local-varied-60) | [SmolVLA 5,000](https://huggingface.co/VivekSai07/smolvla-franka-stacking-local-varied) |
| Broad-workspace, matched exposure | [200 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-broad-workspace-200) | [X-VLA 16,700](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace) |
| B1 language grounding | [200 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b1-language-200) | [X-VLA 33,400](https://huggingface.co/VivekSai07/xvla-franka-b1-language) |
| B2 distractors | [200 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b2-distractors-200) | [X-VLA 50,000](https://huggingface.co/VivekSai07/xvla-franka-b2-distractors-50000) |
| B3 robosuite cubes | [560 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b3-robosuite-560) | [X-VLA 124,800](https://huggingface.co/VivekSai07/xvla-franka-b3-robosuite-124800) |
| Props P1 instruction sorting | [540 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-props-p1-540) | [X-VLA 126,900](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900) |

The index links all **24 additional uploaded checkpoint folders**, including B3 step 156,000 and P1 step 88,830, at immutable model commits. Together with eight programme roots, this maps 32 stored programme checkpoints. The legacy SmolVLA root is separate.

Each programme row also has an episode-0 LeRobot visualizer link, experiment package navigation, and original dated report links. The revision/access table records all 19 immutable snapshot commits. P1 reconstruction, verifier, and original bundle README links point to the exact published model files.

## Corrections to navigation and current status

- Use actual B1 model `xvla-franka-b1-language`, not the older suggested `xvla-franka-b1-language-33400`. Root checkpoint is 33,400.
- Use actual broad-workspace model `xvla-franka-broad-workspace`, not suggested `xvla-franka-broadworkspace-16700`. Root checkpoint is 16,700. It is the matched-exposure run; do not attach it to the earlier 5,000-update result.
- Actual fixed-layout models omit the historical publisher -v1 suffix. The old publisher allowlist and blocked attempts are preserved, with current links added above the historical tooling text.
- P1 now says published from Windows on 2026-10-09 rather than only staged/transferred. B3 publication links distinguish the primary 124,800 root and alternate 156,000 checkpoint.
- The stale B2 publication checkbox now records the existing dataset and six checkpoint weights; a separate unchecked item preserves the missing evaluation/training archive work. This pass did not upload those missing model files.
- DATASET_DESIGN.md now links legacy v1/v2 and distinguishes the proposed 300-episode/12-instruction design from both currently uploaded 50-episode/one-task snapshots.
- Sim2sim links B2 step 50,000, not a new B3 policy. Scene bring-up/P0 and preparation diagnostics do not acquire invented model links.

## Preserved evidence and remaining gaps

All **90 original historical/report/plan/spec documents** are byte-identical to the pre-change inventory. All hash-bound collection and training READMEs, specs/plans, source code, tests, runtime recipes, dataset contents, and checkpoint weights are unchanged. Existing experimental result tables and numerical findings were not altered. Only confirmed unbound evaluator/scene READMEs were given navigation links.

The following are recorded rather than disguised by links:

- No current account model repository was found for X-VLA local-varied, SmolVLA broad-workspace, or the short (5,000-update) X-VLA broad-workspace run. No guessed repository links were added.
- The old pilot-420 upload link remains a dated blocked/empty publication record, not a verified current dataset.
- B2 dataset metadata is repaired, but the 2026-10-06 model audit found 1,065 missing evaluation/training files; complete archive publication is still not established.
- B3 and earlier X-VLA model archives retain tokenizer/runtime/evaluation-code portability gaps. Existence of downloadable weights does not establish clean-machine benchmark re-evaluation.
- P1 includes runtime/base/tokenizer/code and reconstruction helpers, but Docker images/offline wheels are not bundled and retraining is untested. RTX reported a bit-identical 2-trial clean-container smoke; this pass did not repeat it.
- P1 preserved metadata labels all episodes train; the collection plan defines the actual 480/60 research split.
- Access was verified at the time of this audit. B3 model and both P1 repositories are private; legacy v1/v2 are now public. Older cards may contain historical PRIVATE wording. No privacy setting was changed here.

## Verification

The link-validation and preservation results are saved as `validation_results.json` in the local evidence directory. Checks cover new relative file/heading links, exact HF repository IDs, immutable commit and checkpoint/helper paths, dataset v3.0 metadata/branches, root checkpoint metadata, preservation of historical hashes and result tables, and git diff whitespace. This is a documentation/link audit, not a new full model download or simulation test.


| Check | Result |
|---|---|
| New link occurrences | 286 checked: 197 HF and 89 relative links |
| Confirmed HF repositories | 19; commit and access match audit snapshot |
| Additional checkpoint folders | 24 directories confirmed at pinned commits |
| Original historical documents | 90 byte-identical |
| All unmodified original documents | 130 pre-change hashes match |
| Existing result tables | Unchanged |
| Git diff whitespace | Pass |
| Source/runtime/test changes | None |

## Absolute Windows paths for handoff

- Edited checkout: `C:\d drive\RTX trained models\franka-vla-generalization`.
- Artifact index: `C:\d drive\RTX trained models\franka-vla-generalization\docs\HUGGING_FACE_ARTIFACTS.md`.
- This report: `C:\d drive\RTX trained models\franka-vla-generalization\docs\HF_LINK_AUDIT_20261009.md`.
- Change manifest: `C:\d drive\RTX trained models\_publish_work\repo_hf_links_20261009\changes.json`.
- Pre-change documentation hashes: `C:\d drive\RTX trained models\_publish_work\repo_hf_links_20261009\before_document_inventory.json`.
- HF inventory/verification inputs: `C:\d drive\RTX trained models\_publish_work\repo_hf_links_20261009\verified_hf_repositories.json`.
- B1/broad bundle: `C:\d drive\RTX trained models\downloads_20260926`.
- B2 bundle: `C:\d drive\RTX trained models\downloads_20261001\b2_distractor`.
- B3 bundle: `C:\d drive\RTX trained models\B3`.
- P1 bundle: `C:\d drive\RTX trained models\P1`.
- Previous checkpoint portability audit: `C:\d drive\RTX trained models\_publish_work\audit_20261006\README.md`.
- P1 publication evidence: `C:\d drive\RTX trained models\_publish_work\p1_publication_20261009\README.md`.
- Dataset repairs: `C:\d drive\RTX trained models\_publish_work\dataset_viewer_repairs_20261009\README.md`.

The local documentation edits have not been committed or pushed. Review this report and the Git diff, then transfer the documentation changes through your normal Git workflow. Apply no changes to frozen experiment source or dataset provenance just to add a link.
