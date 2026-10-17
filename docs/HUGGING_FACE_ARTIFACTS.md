# Hugging Face datasets and checkpoints

Current artifact map, checked against the authenticated Hub API on **2026-10-17**. The links below use the repositories actually present on `VivekSai07`, rather than names suggested in older plans. Repository existence, file paths, access, and root checkpoint records were checked; this documentation pass did not train or re-evaluate a policy.

| Experiment | Dataset | Published checkpoint |
|---|---|---|
| Fixed-layout baseline | [24 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-fixed-layout-24) | [SmolVLA step 5,000](https://huggingface.co/VivekSai07/smolvla-franka-stacking-fixed-layout); [X-VLA step 5,000](https://huggingface.co/VivekSai07/xvla-franka-stacking-fixed-layout) |
| Local-varied | [60 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-local-varied-60) | [SmolVLA step 5,000](https://huggingface.co/VivekSai07/smolvla-franka-stacking-local-varied) |
| Broad-workspace, matched exposure | [200 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-broad-workspace-200) | [X-VLA step 16,700](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace) |
| B1 language grounding | [200 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b1-language-200) | [X-VLA step 33,400](https://huggingface.co/VivekSai07/xvla-franka-b1-language) |
| B2 distractors | [200 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b2-distractors-200) | [X-VLA step 50,000](https://huggingface.co/VivekSai07/xvla-franka-b2-distractors-50000) |
| B3 robosuite cubes | [560 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b3-robosuite-560) | [X-VLA step 124,800](https://huggingface.co/VivekSai07/xvla-franka-b3-robosuite-124800) |
| Props P1 instruction sorting | [540 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-props-p1-540) | [X-VLA step 126,900](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900) |
| Props P2 instruction sorting + distractors | [540 episodes](https://huggingface.co/datasets/VivekSai07/franka-panda-props-p2-540) | [X-VLA step 101,520](https://huggingface.co/VivekSai07/xvla-franka-props-p2-101520) |

All eight programme dataset repositories contain `meta/info.json` and a `v3.0` branch. The local-varied X-VLA model, broad-workspace SmolVLA model, and original 5,000-update broad-workspace X-VLA model are **not present in the account inventory**. Their measured results remain in [RESULTS.md](RESULTS.md), but the 16,700-update model must not be used as a download link for the 5,000-update result.

Dataset links open dataset cards; model links open the winning/root checkpoint cards. **A model repo root is a particular checkpoint, not necessarily the last training update**: B3 places step 124,800 at root and keeps 156,000 in a subdirectory. The seven earlier dataset repositories and their published models were switched to public on 2026-10-09; P2 was published private on 2026-10-17. The access table below records those checks. Older cards may still say PRIVATE in their text after visibility changed.

## Additional checkpoint directories

These links are pinned to the verified model commit, and only include checkpoints whose `model.safetensors` appears in that repository. Training schedules in old plans may list additional checkpoints that were not kept or uploaded.

| Model | Root checkpoint | Other uploaded checkpoints |
|---|---|---|
| [smolvla-franka-stacking-fixed-layout](https://huggingface.co/VivekSai07/smolvla-franka-stacking-fixed-layout) | 5,000 | Root only |
| [xvla-franka-stacking-fixed-layout](https://huggingface.co/VivekSai07/xvla-franka-stacking-fixed-layout) | 5,000 | Root only |
| [smolvla-franka-stacking-local-varied](https://huggingface.co/VivekSai07/smolvla-franka-stacking-local-varied) | 5,000 | [1](https://huggingface.co/VivekSai07/smolvla-franka-stacking-local-varied/tree/5c7a8a46d12f5abeb1be86708b8be9b68fb430e7/checkpoints/step_000001); [1,000](https://huggingface.co/VivekSai07/smolvla-franka-stacking-local-varied/tree/5c7a8a46d12f5abeb1be86708b8be9b68fb430e7/checkpoints/step_001000); [2,000](https://huggingface.co/VivekSai07/smolvla-franka-stacking-local-varied/tree/5c7a8a46d12f5abeb1be86708b8be9b68fb430e7/checkpoints/step_002000); [3,000](https://huggingface.co/VivekSai07/smolvla-franka-stacking-local-varied/tree/5c7a8a46d12f5abeb1be86708b8be9b68fb430e7/checkpoints/step_003000); [4,000](https://huggingface.co/VivekSai07/smolvla-franka-stacking-local-varied/tree/5c7a8a46d12f5abeb1be86708b8be9b68fb430e7/checkpoints/step_004000) |
| [xvla-franka-broad-workspace](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace) | 16,700 | [1](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace/tree/988a69eb72044b3a4d115e7a75f9b5be98f8514c/checkpoints/step_000001); [2,000](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace/tree/988a69eb72044b3a4d115e7a75f9b5be98f8514c/checkpoints/step_002000); [4,000](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace/tree/988a69eb72044b3a4d115e7a75f9b5be98f8514c/checkpoints/step_004000); [6,000](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace/tree/988a69eb72044b3a4d115e7a75f9b5be98f8514c/checkpoints/step_006000); [8,000](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace/tree/988a69eb72044b3a4d115e7a75f9b5be98f8514c/checkpoints/step_008000); [10,000](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace/tree/988a69eb72044b3a4d115e7a75f9b5be98f8514c/checkpoints/step_010000); [12,000](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace/tree/988a69eb72044b3a4d115e7a75f9b5be98f8514c/checkpoints/step_012000); [14,000](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace/tree/988a69eb72044b3a4d115e7a75f9b5be98f8514c/checkpoints/step_014000) |
| [xvla-franka-b1-language](https://huggingface.co/VivekSai07/xvla-franka-b1-language) | 33,400 | [1](https://huggingface.co/VivekSai07/xvla-franka-b1-language/tree/cf38e2efa8cc6318ed1260adb9be3af390e73558/checkpoints/step_000001); [8,000](https://huggingface.co/VivekSai07/xvla-franka-b1-language/tree/cf38e2efa8cc6318ed1260adb9be3af390e73558/checkpoints/step_008000); [16,700](https://huggingface.co/VivekSai07/xvla-franka-b1-language/tree/cf38e2efa8cc6318ed1260adb9be3af390e73558/checkpoints/step_016700); [25,000](https://huggingface.co/VivekSai07/xvla-franka-b1-language/tree/cf38e2efa8cc6318ed1260adb9be3af390e73558/checkpoints/step_025000) |
| [xvla-franka-b2-distractors-50000](https://huggingface.co/VivekSai07/xvla-franka-b2-distractors-50000) | 50,000 | [1](https://huggingface.co/VivekSai07/xvla-franka-b2-distractors-50000/tree/2f7c6fa26b67076ec1879b02c47c370f5b038e37/checkpoints/step_000001); [16,700](https://huggingface.co/VivekSai07/xvla-franka-b2-distractors-50000/tree/2f7c6fa26b67076ec1879b02c47c370f5b038e37/checkpoints/step_016700); [25,000](https://huggingface.co/VivekSai07/xvla-franka-b2-distractors-50000/tree/2f7c6fa26b67076ec1879b02c47c370f5b038e37/checkpoints/step_025000); [33,400](https://huggingface.co/VivekSai07/xvla-franka-b2-distractors-50000/tree/2f7c6fa26b67076ec1879b02c47c370f5b038e37/checkpoints/step_033400); [41,700](https://huggingface.co/VivekSai07/xvla-franka-b2-distractors-50000/tree/2f7c6fa26b67076ec1879b02c47c370f5b038e37/checkpoints/step_041700) |
| [xvla-franka-b3-robosuite-124800](https://huggingface.co/VivekSai07/xvla-franka-b3-robosuite-124800) | 124,800 | [156,000](https://huggingface.co/VivekSai07/xvla-franka-b3-robosuite-124800/tree/a71a75c988c3915777b8ccaf80482e112881f4d0/checkpoints/step_156000) |
| [xvla-franka-props-p1-126900](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900) | 126,900 | [88,830](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900/tree/bfc9e062d71e77a6e9d8358e576f5a0ed4609aef/checkpoints/step_088830) |
| [xvla-franka-props-p2-101520](https://huggingface.co/VivekSai07/xvla-franka-props-p2-101520) | 101,520 | [126,900](https://huggingface.co/VivekSai07/xvla-franka-props-p2-101520/tree/55499baadd46012d314938618a3c90bd031dfe67/checkpoints/step_126900) |

## Experiment packages and historical reports

Collection/training READMEs and some specs/plans are hash-bound. Dated reports describe what was true at the time. They are preserved unchanged; use this live index to reach their later HF publications. A linked historical upload blocker is not a claim that the current repository is unavailable.

### Fixed-layout baseline

Dataset: [franka-panda-cube-stacking-fixed-layout-24](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-fixed-layout-24). Model: [smolvla-franka-stacking-fixed-layout (root step 5,000)](https://huggingface.co/VivekSai07/smolvla-franka-stacking-fixed-layout); [xvla-franka-stacking-fixed-layout (root step 5,000)](https://huggingface.co/VivekSai07/xvla-franka-stacking-fixed-layout).

Packages: [`fixed_layout`](../experiments/fixed_layout/), [`smolvla_fixed_layout`](../experiments/smolvla_fixed_layout/), [`xvla_fixed_layout`](../experiments/xvla_fixed_layout/).

Records: [SMOLVLA_FIXED_LAYOUT_20260918.md](SMOLVLA_FIXED_LAYOUT_20260918.md), [XVLA_FIXED_LAYOUT_20260918.md](XVLA_FIXED_LAYOUT_20260918.md), [BASELINE_PUBLICATION_20260918.md](BASELINE_PUBLICATION_20260918.md).

[Visualize episode 0](https://huggingface.co/spaces/lerobot/visualize_dataset?path=/VivekSai07/franka-panda-cube-stacking-fixed-layout-24/episode_0).

### Local-varied

Dataset: [franka-panda-cube-stacking-local-varied-60](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-local-varied-60). Model: [smolvla-franka-stacking-local-varied (root step 5,000)](https://huggingface.co/VivekSai07/smolvla-franka-stacking-local-varied).

Packages: [`local_varied_data`](../experiments/local_varied_data/), [`smolvla_local_varied`](../experiments/smolvla_local_varied/), [`xvla_local_varied`](../experiments/xvla_local_varied/).

Records: [LOCAL_VARIED_COLLECTION_20260919.md](LOCAL_VARIED_COLLECTION_20260919.md), [LOCAL_VARIED_DATASET_PUBLICATION_20260921.md](LOCAL_VARIED_DATASET_PUBLICATION_20260921.md).

[Visualize episode 0](https://huggingface.co/spaces/lerobot/visualize_dataset?path=/VivekSai07/franka-panda-cube-stacking-local-varied-60/episode_0).

The later collection seal and the original SmolVLA run have documented provenance differences; publication does not repair them. See the dated publication report and [FAILURES.md](FAILURES.md) #2/#17. No X-VLA local-varied publication was found. A 2026-10-09 repair pass added missing `evaluation_broad_holdout*` evidence plus `runtime/smolvla_base_bundle/` (`lerobot/smolvla_base` + `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`) and `code/` portability sections to the SmolVLA model repository.

### Broad-workspace, matched exposure

Dataset: [franka-panda-cube-stacking-broad-workspace-200](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-broad-workspace-200). Model: [xvla-franka-broad-workspace (root step 16,700)](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace).

Packages: [`broad_workspace_data`](../experiments/broad_workspace_data/), [`xvla_broad_workspace_long`](../experiments/xvla_broad_workspace_long/), [`xvla_broad_workspace_long_eval`](../experiments/xvla_broad_workspace_long_eval/).

Records: [B1_AND_BROADWORKSPACE_PUBLICATION_20260926.md](B1_AND_BROADWORKSPACE_PUBLICATION_20260926.md).

[Visualize episode 0](https://huggingface.co/spaces/lerobot/visualize_dataset?path=/VivekSai07/franka-panda-cube-stacking-broad-workspace-200/episode_0).

This model is the matched-exposure run (16,700 updates, 62/72 on the broad holdout). The earlier 5,000-update X-VLA result (49/72) and the SmolVLA result (23/72) are different runs without confirmed model repositories here. A 2026-10-09 repair pass added missing `evaluation_{placement_modes,step_004000,step_008000,step_012000}` evidence plus `runtime/` and `code/` portability sections.

### B1 language grounding

Dataset: [franka-panda-cube-stacking-b1-language-200](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b1-language-200). Model: [xvla-franka-b1-language (root step 33,400)](https://huggingface.co/VivekSai07/xvla-franka-b1-language).

Packages: [`b1_language_data`](../experiments/b1_language_data/), [`xvla_b1_language`](../experiments/xvla_b1_language/), [`xvla_b1_language_eval`](../experiments/xvla_b1_language_eval/).

Records: [B1_AND_BROADWORKSPACE_PUBLICATION_20260926.md](B1_AND_BROADWORKSPACE_PUBLICATION_20260926.md).

[Visualize episode 0](https://huggingface.co/spaces/lerobot/visualize_dataset?path=/VivekSai07/franka-panda-cube-stacking-b1-language-200/episode_0).

A 2026-10-09 repair pass added missing `evaluation_{report_step_033400,screen_step_016700,
screen_step_025000,screen_step_033400,smoke_step_016700}` evidence plus `runtime/` and `code/`
portability sections.

### B2 distractors

Dataset: [franka-panda-cube-stacking-b2-distractors-200](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b2-distractors-200). Model: [xvla-franka-b2-distractors-50000 (root step 50,000)](https://huggingface.co/VivekSai07/xvla-franka-b2-distractors-50000).

Packages: [`b2_distractor_data`](../experiments/b2_distractor_data/), [`xvla_b2_distractors`](../experiments/xvla_b2_distractors/), [`xvla_b2_distractors_eval`](../experiments/xvla_b2_distractors_eval/), [`robosuite_sim2sim`](../experiments/robosuite_sim2sim/).

Records: [B2_DISTRACTOR_EVALUATION_20260930.md](B2_DISTRACTOR_EVALUATION_20260930.md), [ROBOSUITE_SIM2SIM_20261002.md](ROBOSUITE_SIM2SIM_20261002.md).

[Visualize episode 0](https://huggingface.co/spaces/lerobot/visualize_dataset?path=/VivekSai07/franka-panda-cube-stacking-b2-distractors-200/episode_0).

Sim2sim reuses this B2 step-50,000 policy; it is not a separate model trained in robosuite. The dataset metadata repair on 2026-10-09 restored all 803 original dataset files. The 2026-10-06 Windows audit found 1,065 evaluation/training files missing from the B2 model archive; a repair pass later on 2026-10-09, run directly from the RTX machine via the Hub API, added the 1,565 missing evaluation/training files found by a fresh local-vs-HF diff, plus `runtime/` (the X-VLA base model and tokenizer this run trained from) and `code/` (a repository snapshot). See `PORTABILITY_REPAIR_20261009.md` in the model repository for the exact file list and what this does not establish.

### B3 robosuite cubes

Dataset: [franka-panda-cube-stacking-b3-robosuite-560](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b3-robosuite-560). Model: [xvla-franka-b3-robosuite-124800 (root step 124,800)](https://huggingface.co/VivekSai07/xvla-franka-b3-robosuite-124800).

Packages: [`b3_robosuite_data`](../experiments/b3_robosuite_data/), [`xvla_b3_robosuite`](../experiments/xvla_b3_robosuite/), [`xvla_b3_robosuite_eval`](../experiments/xvla_b3_robosuite_eval/).

Records: [B3_ROBOSUITE_COLLECTION_20261003.md](B3_ROBOSUITE_COLLECTION_20261003.md), [B3_XVLA_EVALUATION_20261005.md](B3_XVLA_EVALUATION_20261005.md), [B3_PUBLICATION_20261005.md](B3_PUBLICATION_20261005.md).

[Visualize episode 0](https://huggingface.co/spaces/lerobot/visualize_dataset?path=/VivekSai07/franka-panda-cube-stacking-b3-robosuite-560/episode_0).

Windows publication completed 2026-10-06, after the dated staging report. Both kept checkpoints and recorded evaluation evidence are downloadable. The original bundle lacked the training-time tokenizer/base bundle and evaluator/scene code; a repair pass on 2026-10-09, run directly from the RTX machine via the Hub API, added `runtime/xvla_base_bundle/` and `code/` (repository snapshot) to close that gap. See `PORTABILITY_REPAIR_20261009.md` in the model repository. A from-scratch clean-container re-evaluation from this repository alone has still not been independently run (unlike props P1, below).

### Props P1 instruction sorting

Dataset: [franka-panda-props-p1-540](https://huggingface.co/datasets/VivekSai07/franka-panda-props-p1-540). Model: [xvla-franka-props-p1-126900 (root step 126,900)](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900).

Packages: [`props_p1_data`](../experiments/props_p1_data/), [`xvla_props_p1`](../experiments/xvla_props_p1/), [`xvla_props_p1_eval`](../experiments/xvla_props_p1_eval/).

Records: [PROPS_P1_COLLECTION_20261007.md](PROPS_P1_COLLECTION_20261007.md), [PROPS_P1_TRAINING_20261009.md](PROPS_P1_TRAINING_20261009.md), [PROPS_P1_EVALUATION_20261009.md](PROPS_P1_EVALUATION_20261009.md).

[Visualize episode 0](https://huggingface.co/spaces/lerobot/visualize_dataset?path=/VivekSai07/franka-panda-props-p1-540/episode_0).

Windows publication completed 2026-10-09. The model archive contains `runtime/`, `code/`, `evaluation/`, `training/`, and `reevaluate/`. Follow the [model card](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900) and [reconstruct_bundle.py](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900/blob/bfc9e062d71e77a6e9d8358e576f5a0ed4609aef/reevaluate/reconstruct_bundle.py) to reconstruct the split dataset/model archive, then use [verify_bundle.py](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900/blob/bfc9e062d71e77a6e9d8358e576f5a0ed4609aef/reevaluate/verify_bundle.py) and the [original bundle recipes](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900/blob/bfc9e062d71e77a6e9d8358e576f5a0ed4609aef/bundle_README.md).

The clean-container 2-trial smoke and trace equality were reported on RTX; Windows verified hashes and reconstructed layout, but did not run GPU rollouts. Docker images and offline wheels are not bundled, and the retrain recipe is untested. The research split is 480 train / 60 validation; preserved `meta/info.json` labels all 540 as train, so use the collection plan to select episodes.

### Props P2 instruction sorting with distractors

Dataset: [franka-panda-props-p2-540](https://huggingface.co/datasets/VivekSai07/franka-panda-props-p2-540). Model: [xvla-franka-props-p2-101520 (root step 101,520)](https://huggingface.co/VivekSai07/xvla-franka-props-p2-101520). Both repositories are private.

Packages: [`props_p2_data`](../experiments/props_p2_data/), [`xvla_props_p2`](../experiments/xvla_props_p2/), [`xvla_props_p2_eval`](../experiments/xvla_props_p2_eval/).

Records: [PROPS_P2_COLLECTION_20261013.md](PROPS_P2_COLLECTION_20261013.md), [training package](../experiments/xvla_props_p2/README.md), and [PROPS_P2_EVALUATION_20261016.md](PROPS_P2_EVALUATION_20261016.md).

[Visualize episode 0](https://huggingface.co/spaces/lerobot/visualize_dataset?path=/VivekSai07/franka-panda-props-p2-540/episode_0) after authenticating for the private dataset.

Windows publication completed 2026-10-17. Dataset commit `6263e62c7c4f26da02ec9b8e5cd359ef7335f42a` is shared by `main` and `v3.0`; model commit is `55499baadd46012d314938618a3c90bd031dfe67`. The model root is the preregistered winner at step 101,520 and `checkpoints/step_126900/` holds the final checkpoint. The two snapshots reconstruct the original bundle with `reevaluate/reconstruct_bundle.py`; that reconstructed layout passed `verify_bundle.py` on Windows. RTX reported the clean-container 2-trial smoke as bit-identical to the original. Windows did not rerun GPU rollouts or retraining.

## Legacy datasets and model

These belong to the archived docker-compose-era work, rather than the sealed fixed-layout/B1/B2/B3/P1 programme. The archived TODO and [DATASET_DESIGN.md](../DATASET_DESIGN.md) preserve the original alignment/design caveats. Repairing viewer metadata does not repair observation/action alignment or prove the proposed 300-episode, 12-instruction design was collected.

| Artifact | Verified HF link | Association |
|---|---|---|
| Legacy stacking v1 | [franka_libero_stacking_v1](https://huggingface.co/datasets/VivekSai07/franka_libero_stacking_v1) | 50 episodes; one stored red-on-blue instruction |
| Legacy stacking v2 | [franka_libero_stacking_v2](https://huggingface.co/datasets/VivekSai07/franka_libero_stacking_v2) | 50 episodes; one stored red-on-blue instruction |
| Legacy SmolVLA, 80,000 updates | [smolvla_franka_libero_stacking_finetuned_b32_s80k](https://huggingface.co/VivekSai07/smolvla_franka_libero_stacking_finetuned_b32_s80k) | Saved train_config.json names franka_libero_stacking_v2 |
| Other 50-episode Franka archive | [franka_stacking_50](https://huggingface.co/datasets/VivekSai07/franka_stacking_50) | No specific sealed run association established; not the 24-episode fixed-layout dataset |

The v1/v2 and franka_stacking_50 `main` branches were restored from their intact `v3.0` datasets on 2026-10-09, and cards now support episode visualization. Existing branch histories and tags remain. SO100 pick-and-place is a separate account project; it is not assigned to a Franka experiment here.

## Runs without confirmed corresponding publications

- Pilot-420 and the 9-episode single-task experiments: the historical upload record does not establish a complete current pilot dataset/model publication. No replacement link is invented.
- Paired-demonstration, bounded/adaptive-placement, local-position probes, and X-VLA trial: preparation/diagnostic work, not confirmed fine-tuned releases.
- Robosuite scene bring-up (5 demos) and props P0: expert/scene gates, not published learned-policy checkpoints.
- X-VLA local-varied, SmolVLA broad-workspace, and short-run X-VLA broad-workspace: results exist locally/in reports, but corresponding model repositories were not found in this account inventory.

## Revision pins and access

Use these immutable commits when downloading for reproduction. Visibility is a dated observation, not a promise about later settings. The snapshot table records file metadata/path checks, not a new end-to-end evaluation.

| Type | Repository | Verified commit | Access when checked |
|---|---|---|---|
| Dataset | [franka-panda-cube-stacking-b1-language-200](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b1-language-200) | [029a66927171fdca96f09bef5993e0f1a11133d3](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b1-language-200/tree/029a66927171fdca96f09bef5993e0f1a11133d3) | Public |
| Dataset | [franka-panda-cube-stacking-b2-distractors-200](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b2-distractors-200) | [81e3af023e88327cd415156c6b67b1243a8c849b](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b2-distractors-200/tree/81e3af023e88327cd415156c6b67b1243a8c849b) | Public |
| Dataset | [franka-panda-cube-stacking-b3-robosuite-560](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b3-robosuite-560) | [075e0848130c34640656def559743a02feef279d](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b3-robosuite-560/tree/075e0848130c34640656def559743a02feef279d) | Public |
| Dataset | [franka-panda-cube-stacking-broad-workspace-200](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-broad-workspace-200) | [56c42c0e89230f041512ff9229daa1b1bccd89e4](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-broad-workspace-200/tree/56c42c0e89230f041512ff9229daa1b1bccd89e4) | Public |
| Dataset | [franka-panda-cube-stacking-fixed-layout-24](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-fixed-layout-24) | [aeb84bb0683914473c8932d4334f9ce18a4c2fce](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-fixed-layout-24/tree/aeb84bb0683914473c8932d4334f9ce18a4c2fce) | Public |
| Dataset | [franka-panda-cube-stacking-local-varied-60](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-local-varied-60) | [8e0000fd692172370bb2481774cd2d251bc6aaf2](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-local-varied-60/tree/8e0000fd692172370bb2481774cd2d251bc6aaf2) | Public |
| Dataset | [franka-panda-props-p1-540](https://huggingface.co/datasets/VivekSai07/franka-panda-props-p1-540) | [95ffafc45feb84f4c5832ec3f703fb5f30b64f1c](https://huggingface.co/datasets/VivekSai07/franka-panda-props-p1-540/tree/95ffafc45feb84f4c5832ec3f703fb5f30b64f1c) | Public (switched 2026-10-09) |
| Dataset | [franka-panda-props-p2-540](https://huggingface.co/datasets/VivekSai07/franka-panda-props-p2-540) | [6263e62c7c4f26da02ec9b8e5cd359ef7335f42a](https://huggingface.co/datasets/VivekSai07/franka-panda-props-p2-540/tree/6263e62c7c4f26da02ec9b8e5cd359ef7335f42a) | Private (published 2026-10-17) |
| Dataset | [franka_libero_stacking_v1](https://huggingface.co/datasets/VivekSai07/franka_libero_stacking_v1) | [7857a63fe6752bd78c21319c8838514f7954ceaa](https://huggingface.co/datasets/VivekSai07/franka_libero_stacking_v1/tree/7857a63fe6752bd78c21319c8838514f7954ceaa) | Public |
| Dataset | [franka_libero_stacking_v2](https://huggingface.co/datasets/VivekSai07/franka_libero_stacking_v2) | [a9addfec8cf52adc8bc322938e740b81f2b74fdd](https://huggingface.co/datasets/VivekSai07/franka_libero_stacking_v2/tree/a9addfec8cf52adc8bc322938e740b81f2b74fdd) | Public |
| Dataset | [franka_stacking_50](https://huggingface.co/datasets/VivekSai07/franka_stacking_50) | [fc3931acd6e54649700f9f83b31ba899cff74914](https://huggingface.co/datasets/VivekSai07/franka_stacking_50/tree/fc3931acd6e54649700f9f83b31ba899cff74914) | Public |
| Model | [smolvla-franka-stacking-fixed-layout](https://huggingface.co/VivekSai07/smolvla-franka-stacking-fixed-layout) | [ec591c0ef196b8868cd85cad2291f5417f9bc392](https://huggingface.co/VivekSai07/smolvla-franka-stacking-fixed-layout/tree/ec591c0ef196b8868cd85cad2291f5417f9bc392) | Public |
| Model | [smolvla-franka-stacking-local-varied](https://huggingface.co/VivekSai07/smolvla-franka-stacking-local-varied) | [0db4be898a4230b4795862c63e7e045dbaa52c88](https://huggingface.co/VivekSai07/smolvla-franka-stacking-local-varied/tree/0db4be898a4230b4795862c63e7e045dbaa52c88) (portability repair 2026-10-09) | Public |
| Model | [smolvla_franka_libero_stacking_finetuned_b32_s80k](https://huggingface.co/VivekSai07/smolvla_franka_libero_stacking_finetuned_b32_s80k) | [787aaa55bf9400e3cc0fc30e2d318813ab7c1a7f](https://huggingface.co/VivekSai07/smolvla_franka_libero_stacking_finetuned_b32_s80k/tree/787aaa55bf9400e3cc0fc30e2d318813ab7c1a7f) | Public |
| Model | [xvla-franka-b1-language](https://huggingface.co/VivekSai07/xvla-franka-b1-language) | [244acc0374947a162f9f0835e4ada95e70885ef6](https://huggingface.co/VivekSai07/xvla-franka-b1-language/tree/244acc0374947a162f9f0835e4ada95e70885ef6) (portability repair 2026-10-09) | Public |
| Model | [xvla-franka-b2-distractors-50000](https://huggingface.co/VivekSai07/xvla-franka-b2-distractors-50000) | [7d8a0edbcb44bb137d674601bf8730398b901895](https://huggingface.co/VivekSai07/xvla-franka-b2-distractors-50000/tree/7d8a0edbcb44bb137d674601bf8730398b901895) (completeness + portability repair 2026-10-09) | Public |
| Model | [xvla-franka-b3-robosuite-124800](https://huggingface.co/VivekSai07/xvla-franka-b3-robosuite-124800) | [31976a43d72958a835200b8a99466877299d8469](https://huggingface.co/VivekSai07/xvla-franka-b3-robosuite-124800/tree/31976a43d72958a835200b8a99466877299d8469) (portability repair 2026-10-09) | Public |
| Model | [xvla-franka-broad-workspace](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace) | [1ba739ee95b473d3936664ec36b4aa1af96523ee](https://huggingface.co/VivekSai07/xvla-franka-broad-workspace/tree/1ba739ee95b473d3936664ec36b4aa1af96523ee) (portability repair 2026-10-09) | Public |
| Model | [xvla-franka-props-p1-126900](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900) | [bfc9e062d71e77a6e9d8358e576f5a0ed4609aef](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900/tree/bfc9e062d71e77a6e9d8358e576f5a0ed4609aef) | Public (switched 2026-10-09) |
| Model | [xvla-franka-props-p2-101520](https://huggingface.co/VivekSai07/xvla-franka-props-p2-101520) | [55499baadd46012d314938618a3c90bd031dfe67](https://huggingface.co/VivekSai07/xvla-franka-props-p2-101520/tree/55499baadd46012d314938618a3c90bd031dfe67) | Private (published 2026-10-17) |
| Model | [xvla-franka-stacking-fixed-layout](https://huggingface.co/VivekSai07/xvla-franka-stacking-fixed-layout) | [65d1d5651f6f1f8f258b83db6b7c00f84f91af6a](https://huggingface.co/VivekSai07/xvla-franka-stacking-fixed-layout/tree/65d1d5651f6f1f8f258b83db6b7c00f84f91af6a) | Public |

## Download the complete snapshot

Run downloads in a network-enabled staging environment; keep the experimental containers offline. Use your existing HF login or HF_TOKEN environment variable. Do not put token values in commands, documentation, or source control.

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id="VivekSai07/xvla-franka-props-p1-126900",
    repo_type="model",
    revision="bfc9e062d71e77a6e9d8358e576f5a0ed4609aef",
    local_dir="p1_model",
)
dataset_dir = snapshot_download(
    repo_id="VivekSai07/franka-panda-props-p1-540",
    repo_type="dataset",
    revision="95ffafc45feb84f4c5832ec3f703fb5f30b64f1c",
    local_dir="p1_dataset",
)
```

Download all files, not only model.safetensors: configs, processors, normalization state, and tokenizer/runtime dependencies are necessary for loading. Then follow the linked model card to restore the original bundle layout. For other experiments, use their row's exact repo and commit and review the archive gaps above.

Documentation audit and complete addition report: [HF_LINK_AUDIT_20261009.md](HF_LINK_AUDIT_20261009.md).
