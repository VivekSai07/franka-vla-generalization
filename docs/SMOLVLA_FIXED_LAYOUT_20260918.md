# SmolVLA fixed-layout comparison — 2026-09-18

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
