# X-VLA fixed-layout training implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Execute the implementation task with TDD, independent review, then main-agent-only launch.

**Goal:** Adapt the already verified X-VLA recipe to the validated fixed-layout dataset and evaluate20sampling seeds.

**Architecture:** New explicit contract/data adapter, runner and evaluation module.
Reuse pure model/precision/scheduler helpers from previous experiment; never
monkeypatch its globals or silently invoke old data/admission paths. An explicit
copy of the small training loop is acceptable where legacy global `c` prevents
injection; document exact changes. New gate binds old imports individually.

**Tech Stack:** Python3.10,LeRobot0.4.4,Torch2.7.1,MuJoCo/OSMesa,pinnedDocker,
offline existingX-VLAweights. No dependencies.

## Global constraints

Read approved spec `docs/superpowers/specs/2026-09-18-xvla-fixed-layout-training-design.md`.
Same feature branch/in-place preference; newpackage only, no boundoldfiles. User
approved actualtraining+20eval after gates, not a new round of permission prompts.
Only main agent launches GPUjobs; worker implements/tests CPU. No credentials,
hostPython, downloads, retries, checkpoint deletion, oldresume, pushes or training
on validation. Keep newdata/spec source immutable once admission/scratch starts.

## Task1 — integrated adaptation and tests

Files: new `experiments/xvla_fixed_layout/{__init__,contract,runner,evaluate}.py`,
`tests/test_contract.py`, `tests/test_runner.py`, `tests/test_evaluate.py`,README.
Additional small data helper allowed if clarity demands; no unrelated framework.
Reports/ledger: ignored `outputs/implementation/xvla_fixed_layout_20260918/`.

Interfaces: `prepare_data() -> (ds,starts,stats,gate)`, `admission() -> dict`,
`recipe(preflight) -> dict`, `validate_scratch(...)`, `evaluation_plan() -> list`,
`summarize(rows) -> dict`. CLI `--check`, `--preflight`, `--train` exclusive.

- [ ] TDD RED tests first: exact20/4split/9980starts, no crossboundary, train-only
  stats exclude extreme validation values, row/inputwhitelist, seeds20unique and
  samefullreset, reject inherited4752rows/4491starts/pilotmanifest/oldstats.

```python
rows, starts = indices(metadata)
self.assertEqual(len(rows), 10560)
self.assertEqual(len(starts), 9980)
self.assertTrue(all(i // 528 < 20 and i % 528 <= 498 for i in starts))
self.assertEqual([r['seed'] for r in evaluation_plan()], list(range(20260918,20260938)))
self.assertFalse(summarize([{'task_success': True}] * 18)['capability_passed'])
```

- [ ] Newcontract constants/data reader use fixed parent seal and `require_fixed_layout_pass`
  ONLY in fresh CPUcheck. Persist fullsemantic gate once in admissionroot; binding
  includes current seal+recomputedfingerprint/currentfixedprovenance and actual
  assets bytes, explicit importedhelper/installedpolicyfiles and newsource/spec.
  No savedPASS if any binding differs. No repeated fullmediaaudit pergradient/reload.
- [ ] Reuse old config/scheduler/precision helpers only if datasetindependent.
  `old.p.policy_row`, `old.p.selected_stats`, `old.p.prepare_data`, oldadmission and
  oldsample_batches are DATASETBOUND and MUSTNOT be reused unchanged. Build real
  CPUmicro4processor check from new20train data without wholepolicy/backward.
- [ ] Implement fresh exclusive scratch and mainrunner matching prior optimizer
  recipe, checkpoint+processor save/strictreload/Adamproof; CPU and scratchadmission
  required. Freshstats not compared to old9demo stats. Allactualrowcounts derived
  and asserted. Trainingdata/settings/CPUproof frozen afterscratch.
- [ ] Finaloffline probes use20train/4val at60/107/347, observation-onlypredict and
  real30x8 targets. Preserve RNG; report physical errors separately; val never
  enters optimizer/stats/sampler. Integration test uses tiny fakepolicy, notGPU.
- [ ] Evaluation can reuse prior run_trial with explicit newreset entry/seed adapter
  ONLY if no runtime monkeypatch/globals or fakeepisodeIDs needed. Otherwise port
  run_trial literally, change explicitseedargument and newcontractimport, then
  compareAST/diff to preserve timing/action/scoring. Actual20trials same reset,
  newrollout seed each, no20different-layout fiction. Countall20, gate18/20 only
  after complete. Expose stage metrics/gripperclosure from savedactions/contact.
- [ ] Tests reject staleCPU/scratch proofs, wrongupdate/microbatch/stats/hash,
  optimizerreload precision drift, missingevaltrials/nonfinites, checkpointselection
  changes, observationoraclefields, partialsuccess and anyresume/overrideCLI.
- [ ] Run newCPUtests plus relevant oldsingle_task/XVLA suites, separateprocess
  perunittest discovery. DockerGPUoff/read-onlyrepo/tmpcaches/16GiB/4CPU. Noactual
  modeltraining/physicsrollout. Record RED/GREEN and allsourcehashes. Readonly
  realdataset/preprocessor inspection allowed; CPUadmissionfullaudit main later.
- [ ] Independent review spec+quality. Fix concrete blockers with regression tests.
  Main verifies tests/hashes, commits code+docs with useridentity, no push.

## Task2 — main-agent execution only

- [ ] Check GPU/disk/unique roots, no competing training; unrelatedcontainersuntouched.
- [ ] Run newCPUcheck with readonlyrepo and onlyadmissionroot writable; assert
  fullfreshdataadmission+20trainstats+9980starts, CPUproofsourcebinding.
- [ ] Run exactly2updates scratch onGPU; verifyexit0/reloadweights+Adam exact and
  actionerror0, no rollouts. IfFAILstop andreport, no retry/autotune.
- [ ] IfPASSlaunchmain freshbase5000then20trials detached retainedcontainer,
  source/datasetcache readonly/outputonlyRW. BindexistingGPU workload headroom.
- [ ] Record actualcontainer/image/commit/proofs/statistics and startup only.
  Registerdockerwaitnotification; no polling/duplicatelaunch or in-flightedits.

## Progress

- [x] User approved X-VLAfirst/5000updates/20fixedresetseeds and18/20gate.
- [x] Priorrecipe and localinterfaces examined; disk~121GiB, GPU~3.2GiBused.
- [ ] Implementation and review.
- [ ] CPUadmission and GPU scratch.
- [ ] Main launch.

Self-review: stats/split/seed/interfaces fixed; no unapproved secondmodel or
changedphysics, and no olddata helper accidentally treated as generic. This
plan documents execution detail within the user's approved design.
