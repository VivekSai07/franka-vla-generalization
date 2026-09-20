# SmolVLA fixed-layout implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development; TDD then independent review, main-agent-only launch.

**Goal:** Run the approved fixed-layout SmolVLA comparison safely using the old verified recipe.
**Architecture:** New explicit contract/dataadapter,runner,evaluator. Reuse pure
SmolVLAprecision/checkpoint helpers and fixed-layout reset/scoring/seed functions;
never monkeypatch legacy globals or edit frozenpackages.
**Tech Stack:** Existing Python/Torch/LeRobot/MuJoCo in pinnedofflineDocker.

## Global constraints

Read design2026-09-18-smolvla-fixed-layout-training-design.md verbatim. User
approved comparison execution; do not ask redundant mode prompts. Existing in-place
featurebranch preference retained. Newpackage only; no secrets/hostPython/install/
deletion/push. WorkerCPUonly; mainGPUafterreview. Same20train4val/20seeds, no newscenes.

## Task1: isolated adapter with TDD

Files: experiments/smolvla_fixed_layout/{__init__,contract,runner,evaluate}.py,
tests/test_contract.py,test_runner.py,test_evaluate.py,README.md.
Interfaces: load_data()->ds,starts,stats; current_binding/admission; prepare_data;
recipe(preflight); CLI --check/--preflight/--train; evaluation_plan/summarize.

- [ ] RED: assert10560rows/9580starts, lastoffset478, validation excluded from
  sampler/statistics, actionchunk50, realstate9/action8, rejectold9demoindices.
```python
rows,starts=indices(metadata)
assert len(rows)==10560 and len(starts)==9580
assert all(i//528<20 and i%528<=478 for i in starts)
assert [t['seed'] for t in evaluation_plan()]==list(range(20260918,20260938))
```
- [ ] Adapt previous single_task runner explicitly only where globallybound;
  use newdataset50deltas and whitelistedPolicyRows, train-onlycompute_stats.
  Reuse LR,backward_microbatch(loss/4),check_batch,save_checkpoint and precision
  loader if schema-independent. Do not reuse oldpilotrequire_complete/admission.
- [ ] Fresh CPUsemantic require_fixed_layout_pass once; later exact fullfingerprint/
  currentfixedprovenance/cache bytes/source/CPUproof checks, no fullsemantic repeat
  eachcheckpoint. Bind all importedSmolVLA/config/processors/cache/newsource/spec
  individually; never invoke oldrecursiveXVLAhashscan or depend onXVLAweights.
- [ ] FullcachedSmolVLAbundle verifiedmanifest+hashes; freshconfig/schema and
  actualCPUprocessor check real8microbatch withoutmodelconstruction/backward.
  CPUproof exclusive writableadmissionmount, same mountedemptyrootclaim semantics.
- [ ] Two-updatescratch stores/reloads stricttrainableFP32weights andAdam evidence,
  original <=1e-5absoluteactionlimit enforced on bothmarkers; verifyall checkpoint
  fileinventories, CPUstats/cachestamps beforemain. Never reusepreflightweights.
- [ ] Main5kpriorrecipe, model450046176trainable99880992, frozenVLMnograd,
  finiteprecisionstates,checkpoint1/1000..5k, derived10560/9580counts. All old data
  hashbound aftertraining, no override/resumeCLI. Keep failureartifacts.
- [ ] Eval port originalSmolVLAloop orXVLAfixedloop with explicitSmolVLAprocessor,
  chunk50 and no inappropriateXVLAautocast/model_inputcheck. Realexplicitseed,
  fixedreset unchanged, no fakeepisodeIDs. ASTregression retains physics/scoring.
 72final observation-onlyprobes, all20rollouts, noactions/oracles inprediction;
 logs/videos/stagemetrics/checkpointbinding/trainingbytesunchanged,>=18after20only.
- [ ] Tests tinyCPUmodels verify save/reload/dtype/failure/noevalonscratch;
  reject staleCPU/scratch/cache/data/statshashes, leakingval, wrongseeds/partial20.
  Test preprocessing actualrows, hashalloldconsumedfiles unchanged, noactualphysics.
- [ ] GREEN new+priorSmolVLA/single_task/precision suites in separateprocesses,
  pinnedCPU Dockerread-onlyrepo/tmpcache/networknone,GPUoff. Write evidence under
  outputs/implementation/smolvla_fixed_layout_20260918 only. Worker nocommit.
- [ ] Independent review spec+quality, mainreruntests/commit verifiedcode.

## Task2: main launch

- [ ] Disk/GPU/outputuniqueness check; priorXVLAfinished and artifacts preserved.
- [ ] CPUfullsemanticadmission actual20trainstats9580starts.
- [ ] GPUtwo-update scratch exact recipe, strictweights/actionagreement/FP32Adam.
  Stoponfailure; no repair/retry disguised as another output.
- [ ] IfPASSfreshbase5k+20eval detachedretainedcontainer, boundinputsreadonly,
  onlymainoutputRW, pinnedimage no credentials/network.
- [ ] Document actualstart/proofs/commit and registerdockerwait, no poll/duplicate.

Selfreview: recipe differences documented; newdata remains immutable; no
randomization/workspacerange changes or automatic extra experiments.
