# Local varied data implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development. TDD, independent review, then main-only real collection.

**Goal:** Collect the approved60demos with12preparation withinbudget, freezing24future evaluation scenes and3policyseeds beforeexecution.
**Architecture:** Isolated newpackage adapts fixed_layout record/validation lifecycle,
reusing generic functions unchanged; explicit schedule/reset/count wrappers replace
hardcoded oldcontracts. No monkeypatch or source rewriting of oldmodules.
**Tech Stack:** PinnedDocker08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc,
Python3.10/NumPy/MuJoCo/LeRobot/PyAV CPUrunc/OSMesa only. No newpackages.

## Global constraints

Exactapprovedspec docs/superpowers/specs/2026-09-19-local-varied-data-design.md.
60expertsmax,48train(40LHS+8center)12valLHS,24evalLHS noexpert;77uniquegeometries.
Onlyred/green4XYcoords vary±10mm/axis. Originalsealedreset/allotherintegration
state preserved inclwarmstart, noresettle. First12=fourcenter+8deterministicmarginal
extremes; remainingfourcenter+32varied+12val. Stopfirstfailure,noretry/replacement.
40GiBstart/30GiBremaining,6hcap,120controlreplaysmax, no training/GPU/policycalls.
Alloldboundcode/spec/data/cache/checkpoints/readmes immutable. Newpackageonly.
Existingfeaturebranch/inplacecanonical/vla; mainonlycommits,soleuser,nopush.
NEVERread.env/.env.example orhostPython/install. WorkerCPUtests/noactualexpert/
physics stepping/renderer; main launches only aftertests/review/realpreflight.

## Task1: integrated bounded collector

Files new experiments/local_varied_data/{__init__,plan,scenes,record,validate}.py,
tests/test_plan.py,test_scenes.py,test_record.py,test_validate.py,README.md.
Smalladditionalmodules onlyifneeded. Output outputs/local_varied_data/local_varied_20260919;
CPUlogs outputs/implementation/local_varied_data_20260919. CLI --preflight readonly
and --collect exclusive. No arbitrarypaths/seeds/count/timeout/retry overrides.

- [ ] Write REDtests beforeimplementation. Interfaces make_plan()->dict,
 validate_plan(plan), derive_scene(model,source,layout)->arrays, preflight()->dict,
 collect()->report, require_local_varied_pass(root)->report.
```python
p=make_plan()
assert len(p['episodes'])==60 and len(p['layouts'])==77
assert p['train_ids']==list(range(48)) and p['validation_ids']==list(range(48,60))
assert p['preparation_ids']==list(range(12))
assert len(p['evaluation_layout_ids'])==24 and len(set(p['policy_seeds']))==3
```
- [ ] Immutableplan exacttypes/counts; separate40/12/24LHS permutations+jitter per
coordinate usingSHAseed namespace/master/identity fromspec. Freezeallresolvedseeds,
offsetvalues, candidate-to-episode mapping, policyseeds disjoint fromexpert/base.
Noautomaticredraw evengeometryfailure. Tests marginalbincoverage/extremauniqueness/
8centertrainingonly/splitdisjointness/source identity/floatsboolspoof/reproducibility.
- [ ] Sceneadapter reuses oldsource resetloader/geometryhelpers ifgeneric; alteronly
4freejointXYvalues andassert fullintegrationmask exactly, cameras/Z/etc unchanged.
Excludeexact17oldprobegeometries except explicitcentertraining; no240sealedbank.
RealnonstepCPUcheck all77geometryvalid,noduplicates, all60demoresetrefs resolve,
24evaluation neverinexpert list. Saveplan/all77states/hashmanifest BEFOREdataset
ortrajectory, compare oldsource hashes before/after. No exposureoraclepolicyfields.
- [ ] Reuse record_attempt/TickSink/media boundencoder fromfixed_layout unchanged
whenindependent ofroot/count; ownrun_schedule/provenance/checkoutput/resources.
Perattempt actualunchangedpose-snapshot-v2Expert, preservepartialrawframes/terminal,
firstfailurestop, step/resourcecheckinsink. StoreperattemptelapsedCPUmetrics,
stage/contact/release/placement diagnostics and command saturation. NoGPUutilclaim.
- [ ] Reuse fixed validate.read_rows/load_attempt/replay_controls/audit_replay/
validate_video/contactgates/media binding onlywhen no24count/fixedreset assumptions.
New metadata/schedule wrapper60count/48split/perlayoutreset; no globalpatching.
Exactlytwo savedcontrolreplays peracceptedepisode, reusedexclusive modebudgettokens;
doNOT calloldexpertregeneration helper. Governingtolerances unchanged.
- [ ] Finalaudit independentlyall60persistedepisodeentries/31680rows/63360video
frames and120unique successfulreplayevidences, noextra/missingfiles, duplicatefull
action+state rejection, all77resetbindings and24reservedunexecutedscenes unchanged,
first12preparationevent/order, fullfingerprint/seal. Futurepassgate must compare
freshsource/runtime/actualdata, not trustsavedsuccess. No trainingstatsglobal.
- [ ] Failures: preserveoriginalexception despitefinalizeerror; rootclaimexclusive
emptybindmount allowedasreviewedoldpattern; noresume/delete/clearfailedstates;
sixhourmonotonic+alarm anddisk30floor insideencoding/replay/checks aswellasboundary.
- [ ] Tests syntheticfakeexpert toprove60count/allfrozenbeforecalls/prep12/validation
beforeattempt13/24banknotrun/noextraonfailure/diskfloor/nooverride/replaymodecounts.
Real2episode syntheticLeRobot persistence withnewadapter/AV1 andmedia bindings,
no physics/model; finalfull60scheduler/metadata gates syntheticarrays, not60actual
encodes ifunnecessary. Independentlytestsharedgenericvalidators forlayout-specific
reset injection and changedcounts, realLHS77nonstepping valid. No newactualexperts.
- [ ] RunGREENnew+relevantoldunit suites separatePythonprocesses pinnedCPUreadonlyrepo,
tmpcacheandownimplementationRWonly. ReportRED/GREEN/oldhashes/APIs/limitations.
- [ ] Independentreview spec+quality, mainrerunCPUtests+nonsteppingpreflight, commit.

## Task2: main collection

- [ ] Check40GiBstartspace, runningjobs/sourceuniqueness and no pendingchanges.
- [ ] LaunchoneCPUcollector,noGPU/network/credentials, readonlyrepo/bankinputs,
newoutputonlyRW; retainlogs/container. First12thenremainingautomaticallyonlyifpass.
- [ ] Inspectstartup andregisterdockerwaitnotification once,no polling/duplicate.
- [ ] Oncompletion verifycounts/replays/seal/holdout untouched; stop onfailure.
No training/variedevaluation/newattempts without nextdecision. W&Bsummaryexport
separate reviewedoperation aftercompletion, notnetwork insidecollector.

Selfreview: validation snapshots useeachplannedreset notoldcenter; centergeometry
duplicationintentional, trajectoryduplicationblocked; all77statesfreezepriorcalls,
futurepolicyseedsfixed, only60demo IDs allowedexperts. No blockedlegacy rewrites.
