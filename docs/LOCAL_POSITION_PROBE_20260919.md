# Local position probe — 2026-09-19

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
