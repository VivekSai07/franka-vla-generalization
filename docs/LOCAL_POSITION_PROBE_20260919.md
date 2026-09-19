# Local position probe — 2026-09-19

## Completed: 68/68 trials, controls exactly reproduced

Container exited0/noOOM at2026-09-19T00:24:37Z, about1h50m37s afterlaunch.
All68canonical trials finished,136journalstart/end events. All4unchangedcontrols
matched every savedarray byte and checkedsemanticfield againstoriginalbaselines:
SmolVLAseed20260918success,seed20260923placementfailure; bothX-VLAcontrolssuccess.
No retries, newtraining, expertcollection or checkpointchanges.

### Results (controls excluded from shifted-scene totals)

| Condition group | SmolVLA | X-VLA |
|---|---:|---:|
| Unchanged controls |1/2|2/2|
| All 5/10mm shifts |19/32 (59.4%)|31/32 (96.9%)|
| Pick/red moved |11/16|16/16|
| Destination/green moved |8/16|15/16|
| 5mm shifts |9/16|16/16|
| 10mm shifts |10/16|15/16|

### Complete per-condition results

Each entry is successes /2predeclared seeds. Directions are worldaxes.

| Moved cube | Offset | SmolVLA | X-VLA |
|---|---|---:|---:|
| red |+X5mm|2/2|2/2|
| red |-X5mm|0/2|2/2|
| red |+Y5mm|2/2|2/2|
| red |-Y5mm|1/2|2/2|
| red |+X10mm|2/2|2/2|
| red |-X10mm|1/2|2/2|
| red |+Y10mm|2/2|2/2|
| red |-Y10mm|1/2|2/2|
| green |+X5mm|0/2|2/2|
| green |-X5mm|2/2|2/2|
| green |+Y5mm|1/2|2/2|
| green |-Y5mm|1/2|2/2|
| green |+X10mm|0/2|2/2|
| green |-X10mm|2/2|1/2|
| green |+Y10mm|1/2|2/2|
| green |-Y10mm|1/2|2/2|

ShiftedSmolVLAfailures:10placement-stage,3grasp-stage under the predefined
stage classifier (sustainedclosure/contact/lift/stable-stack sequence).
X-VLA's sole shiftedfailure:green-X10mm,seed20260918,placement-stage.
These are recorded failure stages, not causal diagnoses of visual grounding,
control error or release dynamics.

Seed breakdown over16offsets each:SmolVLA13/16(seed20260918),6/16(seed20260923);
X-VLA15/16 and16/16. Seed20260923 was intentionally selected as a knownSmolVLA
fixed-layout failure. Do not directly interpret59.4% as a population degradation
from the earlier18/20=90% benchmark, which used20differentseeds. The +5mm/+10mm
nonmonotonic results also preclude claiming a single learned distance boundary.

### Verification and telemetry

Freshread-only completion audit rechecked both frozenmodeladmissions/finalweights,
source/runtime/data/cache/controlbindings, all17bank/plan/proof hashes, recomputed
the fullsummary, checked all68individualJSON/trace reset/stage/result matches,
successfinal30stackticks or1200ticktimeouts, and all4bitexactcontrols. Everytrial
has a nonemptyvideo file. This audit did not rerun inference/physics or fully
decode the videos. Existingsealedfinal bank remains untouched.

1315localtelemetrysamples,0reportedqueryerrors,0samplererrors. Acrossallphases,
devicewide(sharedGPU) utilization0–57%, samplemean16.71%; GPUmemory3659–7389MiB;
power49.53–114.56W, samplemean94.75W; temperature64–75C. These includeotherGPU
workloads andphaseoverheads: notper-policyutilization, energy or performance
attribution. Rawtimestamps/phases retained forlateranalysis. W&Bsyncnotperformed
for this study yet; local telemetry/results are authoritative.

Evidence: [result](../outputs/diagnostics/local_position_probe_20260919/result.json),
[journal](../outputs/diagnostics/local_position_probe_20260919/journal.jsonl),
[telemetry](../outputs/diagnostics/local_position_probe_20260919/telemetry.jsonl).

### Recommended next step (not launched)

Proceed toward **small, deliberately varied local data**, not fullworkspace
randomization or longertraining ontheunchangedfixeddata. Preservebothbaselines.
Propose firstneighborhood up to10mm along eachXYaxis aroundbothnominalcube
positions, varyingpick/destinationindependently and includingcenterexamples.
This broaderlocalrectangle/simultaneousmovement is a proposednewtrainingdomain,
not something already evaluated by thissingle-axisstudy. Validategeometry and
expertcompletion first on afixedsmallpreparation set; nostartuntil exactbudget
andacceptanceprotocol agreed. Keep task/cameras/objects/actioninterfacefixed.

Predeclare fresh independent evaluationlayouts ANDseeds beforecollection. Existing
16offsetconditions are nowdevelopmentprobes, notuntouchedheldoutdata. Retain all
expertattemptfailures andunfilledcells; never silentlysuccessfilter hardregions.
Ifexpertgatepasses, collect a boundedstratifiedlocaldataset, trainnewcomparisons
underfixedrecipes andscoretheuntouchedbank plusoriginalfixedcontrol. SmolVLA
placement/closure mustremainseparatelyreported. X-VLAcanserveasstrongerreference,
but thisstudydoesnotprove a modelarchitecture effect orvisuallyadaptivebehavior.

Offsets5/10mm are smallerthan20mmXYsuccess tolerance: X-VLAcould remain successful
partly throughplacementmargin ratherthan fullytracking destination shifts. Future
analysis should compare release/placement position response, not infer visual
grounding from successalone. No newcollection/trainingjob started afterthisstudy.

The launch notes below are historical; this job is no longer running.

User approved the exact68-trial evaluation-only study. No newtraining or varied
data collection is authorized yet; collection follows resultreview and a separate
bounded design. User withdrewcheckpointGitcommitrequest and choseWindowsdownload
of portablepublication packages; no checkpointbranch/binarycommit/push was made.

## Verification and launch

Implementation54c03c9, isolated experiments/local_position_probe, no frozenbaseline
source/data edits.35CPUtests and real17resetnonsteppingpreflight passed; independent
staticreview plus maininspection found no concreteblocker. Mainfirsttestcommand
omitted writablefixturemount (15filesystemerrors); correctedmount passedall35
withoutsourcechanges. That command hadalreadycommitted the worker-verifiedcode;
mainlaunch waited for the corrected independenttests andfreshpreflightPASS.

PlanSHA256 `92916c08bd92932cd629388bd8695781a7977df09e8770d1913932fb1700f65b`.
17conditions,68trials,0policyloads/physicssteps duringCPUpreflight.

Container `franka-local-position-probe-68-20260919`, ID
`996f301e2e0424f341ec526dfdc241c875bb816d25308eb4cd447703a17b4c7e`,
started2026-09-18T22:34:00Z (studydate2026-09-19), verifiedrunning.
Pinnedimage08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc,
offlineGPUcontainer, readonlyrepo/data/cache/root, onlynewoutput writable.
Host59GiBfree/GPU45,287MiBfree beforelaunch;20GiBdisk/12GiBGPUfloors unchanged.

Output: `outputs/diagnostics/local_position_probe_20260919`.
No network/credentialfiles, no optimizer or expert calls. Oldartifacts preserved.
No boundpackage/README/spec/plan edits while running; no retry/resume/duplicatejob.

## Experiment

SmolVLA thenX-VLA,34trials each: unchangedcontrol plus redpick ORgreendestination
 x5/10mm xworld+X/-X/+Y/-Y, eachseed20260918 and20260923. All17savedinitializations
shared betweenmodels. TranslationchangesonlyonecubeXYqposscalar, noresettle;
otherintegrationstate includingwarmstart,velocities,controls andtime preserved.
Geometry/noinitialrobotcubecontact/camerapreservation gates passednonstepping.

Eachmodelmustreproducebothunchangedcontrol traces byteexact beforeitsoffsets.
Smolseed20260923 is a knownfailed baseline and must match thatfailure, notbecome
successful. Ordinarytimeouts countasoutcomes and continue; reproduction/nonfinite/
infrastructurefailuresstop.68trialcapincludes4controls,4hwallguard,noextensions.
Originalmodelrun_trial functions unchanged,baselineordinals1/6 mapthetwoseeds;
studyindex separate,episodeidentity always7.40smax/30strictstabletickscore unchanged.

Local5secondGPUtelemetry records UTC/monotonic time,utilization,memory,power,
temperature andphase whenavailable. Devicewide/sharedGPU, notperrunenergy.
No modelRNGsideeffects; errors/missingfields explicit, telemetrynonfatal.
W&Bcloudlogging deferred; rawlocalmeasurements authoritative.

Currententry isLAUNCHonly. Reportpercondition/object/magnitude andfailurestage,
controlsseparately. Two developmentselectedseeds/condition form a smallsensitivity
screen, not preciseboundaryestimation, populationreliability or spatialgeneralization.
