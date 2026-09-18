# Local position probe implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development; TDD plus independent review. Only main launches actual policy trials.

**Goal:** Measure small-position sensitivity of the two frozen baselines in exactly68trials.
**Architecture:** One isolated package creates17nonsteppedreset variants and calls
the existing frozen model-specific run_trial functions without monkeypatching.
One model is loaded once at a time, controls first in each model block. Reuse
oldprovenance/admission/FP32loaders. LocalGPUtelemetry cannot affect policyRNG.
**Tech Stack:** Existing pinnedDockerPython3.10/LeRobot/Torch/MuJoCo/OSMesa; no installs.

## Global constraints

- Approvedspec docs/superpowers/specs/2026-09-18-local-position-probe-design.md.
-17conditions(control +red/green x5/10mm x+X/-X/+Y/-Y),2seeds20260918/20260923,
 34/model68total. ModelorderSmolVLA thenX-VLA, conditionorder canonical aslisted.
 Bothcontrols first permodel; exactpriortrace match required beforeitsperturbations.
- Frozenfinal005000weights from outputs/train/{smolvla,xvla}_fixed_layout_20260918.
 No expert,optimizer,training,demodata,oldfinalscene access,newmodel/download.
- Samecontrol40s1200ticks30Hz/20substeps/30strictstabletickscore; preservefailures.
- CPUworker only: noGPUinference/physicssteps/renderer. Nonsteppingfullreset
 derivation/forward andsyntheticunit tests allowed. Main launch aftertests/review.
- AllPython pinnedimage08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc,
 offlineDocker/read-onlyrepo,exclusiveoutput writable, nohostPython/credentials.
- Newpackage experiments/local_position_probe only; tests inside it, notoldtests.
 Newdocs/ownignoredimplementationlogs allowed; no oldboundcode/spec/report edits.
- User cancelledcheckpointGittransfer: no binaries/branches/push; directWindowsdownload.
- Existingfeaturebranch/inplace preference; main alone commits underuseridentity.

## Task1: contract, reset bank, runner, telemetry and report

Files: experiments/local_position_probe/{__init__,contract,scenes,runner}.py,
optional smalltelemetry.py; tests/{test_contract,test_scenes,test_runner}.py,README.
Artifacts: outputs/diagnostics/local_position_probe_20260919 (exclusive wholejob),
CPUcheck evidence outputs/implementation/local_position_probe_20260919 (separate).
Interfaces: make_plan()->dict; derive_scene(model,source_reset,condition)->dict;
preflight()->dict (no writes/steps/models); run()->report; summarize(rows)->dict.

- [ ] REDtests: immutable17conditions/68slots, truesourceepisode7 unchanged,
 seedordinal=1for20260918 and6for20260923 (oldSmolrun_trial enforces this pairing).
 Studytrialindex separate from baselineordinal; never spoofepisodeIDs orseed.
```python
plan=make_plan()
assert len(plan['conditions'])==17 and len(plan['trials'])==68
assert {t['seed'] for t in plan['trials']}=={20260918,20260923}
assert all(t['source_episode_index']==7 for t in plan['trials'])
```
- [ ] Derivefull17bank from sealedreset:restorestate, selectcube freejointqposXY
 and alter onlyonecoordinate ±.005/.010m, preserving allotherintegration fields
 INCLUDINGqvel,ctrl,warmstart,time,quaternion. captureandsynchronize via existing
 statepreservinghelper. Assert entireintegrationvector differs onlyexpectedqpos
 scalar (controlbitexact); compare actualworlddelta, cameraextrinsics same,settledZ,
 workspacebounds/clearance andnoinitialrobotcubecontacts. Inspecteachjoint type.
 Oldsource qpos37, state size retrievednotguessed, no resettle/mj_step. Save all
 sceneNPZ+hashes+plan beforepolicycalls, reuse identical files bothmodels.
- [ ] Tests actualnonstepping17variants, wrongaxis/body/quaternion/warmstart
 corruption, nonfinite, invalidbounds/collision. No successfilter/resampling.
- [ ] CPUpreflight oldmodel c.admission +verify_final both completebeforelaunch,
 exactpackagehashes+spec/plan/imports and bank/source binding. Testreadonlynooutputs.
 Don'tretrain/reloadcheckGPU. SHA source checks requiredbothbefore/afterrun.
- [ ] Runnerclaim exclusiveemptyoutputbindmount asprevious pattern; noresume,
 no commandlinebudget/checkpoint/modeloverride. --check nonsteppingpreflight,
 --run approvedjob. Assert>=20GiB disk and>=12GiB GPUfree (inferenceonly) before
 launch, diskbetween trials;4hour wall guard forinfrastructure only,nobudgetextension.
- [ ] Policy loading uses priorSmolload_counted andXVLA c.p.load_weights(CountedPolicy)
 toCUDA withsavedprepost, eval/no grad, Torchdeterministicalgorithms samebaseline.
 Do not copyrun_trial loops unnecessarily: passderivedbankentryfile/hash/scenehash,
 sourceepisode7,baselineordinalandexplicitseed to unchangedfunctions. Nativechunk
 queue resets everytrial, real seeds unchanged. No monkeypatcholdglobals/functions.
- [ ] Comparebothcontrol NPZs to originalsamebaselinefolder eachmodel: identical
 arraykeys,dtypes,shapes,values bit-for-bit across everystoredarray, not justoutcome.
 Compare episodefields excluding wallseconds andartifactpath/hash (comparesemantic
 selectionexplicitly). Videoencoding byteequalityNOT required. Smolseed23failure
 must reproduce, not forcecontrolsuccess. Infrastructure/seed/nonfinite/repro
 mismatch STOPbefore nexttrial; tasktimeoutordinaryfalse recordedcontinue.
- [ ] Maintain append-only trialstart/end journal and currentstatus, preservepartial
 output on error. Only68run_trial invocations max; failclosed anyrelaunch/existing
 path/extraresult. Reportcomplete onlyexactcanonicalallslots; aggregatepermodel/
 condition/object/magnitude with denominators; controlsseparate, pairedoutcomes,
 stagefailures derivedexistingstages, no newthreshold or successclaims frompartial.
- [ ] Local telemetry using boundednvidia-smi query only UUID/name/utilization/
 memoryused/total/powerdraw/temp every5s withUTC+monotonic+currentphase, explicit
 devicewide/sharedGPU. CaptureCPUwall/resourceinfo andtiming. No credentials,
 processcommandlines/environment/codeauto capture. MissingNVMLfield→null reason,
 telemetryfailure nonfatal butrecorded. Backgroundsamplerterminate/joinfinally,
 never changesRNG or readsmodel tensors. PythonEvent.wait okay, no terminalsleep.
 W&Bcloudlogging deferred until completedresults ready; localrawmeasurements
 authoritative, no delayingcompute for newnetworklogger. Explicitly document.
- [ ] RED/GREEN tests mocktrial/modelhardware seams not scoring, regressionagainst
 exactoldresults forcontrolcomparer (syntheticarrays detect everymutation),count68,
 noextra onfailure, ordinal/seedpair andall17samebank acrossmodels. Testtelemetry
 parser missing/nonfinite and subprocessbounds. No actualpolicy/physics calls.
- [ ] CPUchecksusingactual17states andoldbindings; publish ownreport andsourcehashes
 underignoredimplementationroot. Mainindependentreview spec+quality thenrerun
 CPUtestcounts/fullpreflight. Freeze/codecommit beforeGPUjob.

## Task2: main only execution

- [ ] Review resources, no otheragentjobs, olddata/sourceimmutable.
- [ ] Run exactlyone68-trialjob retainedofflineGPUcontainer,onlyoutputRW.
- [ ] Confirmfirstcontrolstart, recordimage/commit/ID, registerdockerwait once.
- [ ] Oncompletion inspectcontrolreproductions/all68outcomes/fingerprints/telemetry,
 reportbothmodels percondition rather thanoverallonly. No new collection until
 resultsreview and concretevarieddata design. Noautomaticretries/extensions.

## Progress

- [x] User approved68trialscope20260919.
- [x] User withdrewcheckpointGitcommitrequest; directoriesprovided forWindows.
- [ ] Implementation/TDD/review/CPUpreflight.
- [ ] ActualGPUlaunch and outcome verification.

Selfreview: permodelcontrolsfirstresolves loadonce ordering; seedordinalkept from
baseline, studyindexseparate. Translationdoesnotresettle and doesnotpurport to
recover arbitraryrealworldreset distribution.68hardbudget includes4controls.
