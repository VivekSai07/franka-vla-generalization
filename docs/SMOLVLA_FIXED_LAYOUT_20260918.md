# SmolVLA fixed-layout comparison — 2026-09-18

## Completed: 18/20 stacks, capability gate passed

Container exited0/noOOM at2026-09-18T19:32:07Z, about1h50m52s after launch.
All5000updates/160000sampledchunks and20final-checkpoint trials completed.
**18/20successes (90%); agreed>=18/20gate passed.** No retries or extensions.

All20trials achieved sustained closure, bilateral contact and contact-associated
lift>50mm. Trials6(seed20260923) and16(seed20260933) released but never reached
30consecutive stable-stack ticks; both timed out at40s. FinalXYerrors42.167mm
and42.360mm respectively. This identifies later-stage placement failures, not
an established causal diagnosis of why placement failed.

Successful trials finished in12.30–12.97simulated seconds, finalXY1.084–15.091mm.
No arm-joint clipping in any trial. Gripper-range clipping changed zero binary
open/close decisions; adapter clipping is not human assistance.

| Fixed-layout comparison | SmolVLA | X-VLA |
|---|---:|---:|
| Successful stacks |18/20|20/20|
| Capability gate |Passed|Passed|
| Total parameters |450,046,176|879,482,456|
| Trainable parameters |99,880,992|879,482,456|
| Peak GPU allocation (approx.) |5.58GB|21.3GB|
| Successful final XY error |1.08–15.09mm|0.63–4.68mm|
| Arm-clipped ticks |0|0|
| Optimizer updates |5000|5000|
| Sampled chunks / native length |160000 /50|40000 /30|

One geometry and one training seed per recipe do not establish architecture
superiority or general reliability. Same rollout seed numbers do not imply
identical random noise across models. Neither result tests new object positions.

Final offlineprobes:train60full50armMAE0.0096005rad,first10 0.0109410rad,
gripper96.87%; validation12full50armMAE0.0120813rad,first10 0.0130845rad,
gripper95.83%. Validation shares geometry, not a spatialholdout. Loss first/last
200updates mean0.1489025/0.00322122, not directly comparable toX-VLA's objective.
All six GPU checkpoint reloads had exactFP32weights and predictiondifference0.
Peaktraining allocation5,576,195,072bytes.

Fresh read-only completion audit passed currentdata/source/cache/CPU/scratch/
statistics and sixcheckpointinventories, all5000finiteorderedloss/gradientrecords,
trainingbytesunchanged,20individualJSONrecords, exactinitialintegrationstates,
recomputed stages andsuccessstreaks. No newGPUinference/physics/video decoding.
FinalmodelSHA256:
`9adf89327fb5b0be5cf886edcb8ca18a5dc44f6bfba47f7b73c7a6d716884916`.
Parenttraining report's unevaluated fields are frozen pre-evaluation history;
the separate completed evaluation report is authoritative. Never rewrite them.

Evidence: [evaluation](../outputs/train/smolvla_fixed_layout_20260918/evaluation/result.json),
[successful trial1](../outputs/train/smolvla_fixed_layout_20260918/evaluation/trial_01_seed_20260918/cameras.mp4),
[failed trial6](../outputs/train/smolvla_fixed_layout_20260918/evaluation/trial_06_seed_20260923/cameras.mp4),
[failed trial16](../outputs/train/smolvla_fixed_layout_20260918/evaluation/trial_16_seed_20260933/cameras.mp4).
No further jobs started. Preserve both baselines; any small-position evaluation
or collection requires a new bounded protocol, not an automatic training extension.

The launch text below is historical; this job is no longer running.

User approved SmolVLA after X-VLA20/20, same dataset and evaluation seeds, before
introducing position variation. This is a model-plus-recipe comparison, not an
architecture-only or compute-matched test.

## Verification and launch

- Implementation commit78fbfa5, isolated experiments/smolvla_fixed_layout.
-88CPUtests passed (31new+57regressions); main independently reran31newtests.
 Independent static review plus main source inspection found no concrete blocker.
- Fresh fullsemantic fixed-layoutCPUadmission exited0, actual10560training rows,
 9580complete50action starts, train-onlystats from20episodes. Fourvalidation
 episodes share geometry and are excluded from optimization/statistics.
- GPUtwo-update scratch exited0/noOOM. BothcheckpointFP32weights/actionreload
 errors0; actualAdamroundtrip exactFP32; source/data/cache/statsbindings rechecked.
 Peakallocation5,575,401,472bytes. Zero scratchrollouts; neverresume scratchweights.
- Fullrun container `smolvla-fixed-layout-5000-20260918`, ID
 `b129f9a53d89d468b43df519995ac6951e575ecda18606d011dc7f8a753fa7b0`,
 started2026-09-18T17:41:15Z, verifiedrunning. Host~77GiBfree andGPU45,287MiBfree
 beforelaunch, above35GiBdisk/12GiBGPUstart floors.

PinnedofflineDockerimage08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc,
read-onlyrepo/data/cache/root, only new output writable. No credentialfiles,
downloads, oldartifactdeletion or unrelatedcontainer changes. Logs/container retained.
Output `outputs/train/smolvla_fixed_layout_20260918`; sibling_admission and_preflight
hold verifiedproofs. No source/test/README/design/plan edits while running.

## Frozen recipe

Fresh pinnedSmolVLAbase,450046176total/99880992trainable parameters; frozenVLM,
FP32expert/projections andAdam, BF16trainingautocast.5000updates micro8xaccum4,
effective32/160000sampledchunks,seed20260917,stockflowmatchingloss,chunk50/execute10.
LR500warmup1e-4cosine1e-5;AdamW(.9,.95),eps1e-8,decay1e-10,clip10.
Checkpoint1/1000..5000, allsavedstatefiles retained; final5000onlyselection.

Aftertraining:72finalofflineprobes at60/107/347on20train+4val, then20rollouts
seeds20260918..20260937 with exactsame savedepisode7reset asX-VLA.40smax,
1200ticks,strict30consecutive released-stack ticks,goal>=18/20afterall20finish.
No actiontargets/objectposes sent to policy, no newpositions/retries/extensions.
Reportclosure/contact/lift/release/stable-stack/clipping separately.

Differences retained versusX-VLA:chunk50vs30,effective32vs8,160kvs40kchunks,
frozenVLMvsalltrainable, nativeobjective/processors. Sameinstructions,demonstrations,
reset,seeds andstrictscoring; matching seedvalues does not match noise acrossmodels.

This report records LAUNCH, not completedupdates or tasksuccess. No automatic
follow-on experiments. All olderresults preserved; no push.
