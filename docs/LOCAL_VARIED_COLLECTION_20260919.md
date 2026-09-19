# Local varied collection — 2026-09-19

## Completed: 60/60 demonstrations and 120/120 replays passed

Container exited0/noOOM at2026-09-19T12:03:39Z, about2h56m28s afterlaunch.
Exactly60started/60admitted episodes and onefirst12preparation-pass event. No
replacement or retry. Final validator reports training_ready=true.

-48training episodes (40varied+8center),12validation;31,680total rows at30Hz.
-63,360camera frames fully decoded by the final collector validator.
-120/120saved-control replays passed: maximumreferencefull-qposerror0;
 maximum persistedfloat32policy-stateerror0.000014249901249645258 (<0.001).
-Minimum accepted episode:240closed ticks,217bilateral-contact ticks,
 0.2010556754m contact-associated lift and180finalstable-stack ticks.
-No exact duplicate action/state trajectories.77frozen scene geometries preserved.
-24reserved evaluation scenes were not executed by an expert or policy. Future
 policy seeds remain2912411705,944268569,2782447885; no evaluation budget executed.
-LeRobot dataset341,017,635bytes; complete evidence root4,948,039,087bytes.
 Approximately54GiB free aftercollection; futuretraining must check its own budget.

Freshread-only completion audit recomputed fullfilefingerprint/currentprovenance,
regenerated all77initializations withoutphysicssteps, checked exactfrozenbank,
all60first-rowqpos/slot/report identities, orderedjournal/preparation,120unique
successfulreplayrecords and exclusionof24evaluation IDs fromeveryattempt.
It did not rerun experts, physicsreplays or video decoding. An initial audit
command used the wrong helperimport and failed beforechecks; correctedimport
completed unchanged-source audit successfully.

Seal: `f8b10a498798217a63564416ab6561e25bd7e9c87e4fdddcf42f288cdb9e9a28`.
[Final validation](../outputs/local_varied_data/local_varied_20260919/validation.json),
[dataset](../outputs/local_varied_data/local_varied_20260919/dataset/),
[journal](../outputs/local_varied_data/local_varied_20260919/journal.jsonl).

No training or policy evaluation started. Recommended next protocol: first
test whether a fresh SmolVLA trained on48local training demos improves over
the frozen fixed-layoutSmolVLA on the same24reserved scenes/3frozenpolicyseeds,
with original-center regression checks. Freeze recipe/budget and finalcheckpoint
selection before bankexecution, never use results to select/tune checkpoints.
This is a proposed comparison, not authorization or evidence of policy improvement.
LaterX-VLA comparison remains useful; no broadworkspace expansion yet.
W&Bcollection-summary export remains pending; local reports are authoritative.

Earlier launch text below is historical; collector is no longer running.

## Approved and launched

User approved the exact [60-demo design](superpowers/specs/2026-09-19-local-varied-data-design.md).
Implementation9eff884 is isolated in experiments/local_varied_data; all old
source/data/checkpoint/release artifacts remain unchanged. No training authorized.

Worker85CPUtests passed (22new+63helper); independent staticreview andmaininspection
found no concreteblocker. Main independently reran all22newtests (145.318s),
including synthetic60-slot orchestration/admission and realtwo-episode AV1write/read,
then realnonsteppingpreflight:77uniquevalidlayouts,60expert slots,24evaluation
layouts,outputabsent,storageeligible62,522,449,920bytesfree. No actualexpert or
physics call occurred before the authorized collector launch.

Container `franka-local-varied-60-20260919`, ID
`3eca3bf4aedca403f9ec04af17708519dd61a83693f34e9b56ee67bfde38f0be`,
started2026-09-19T09:07:12Z, confirmedrunning, exactpinnedimage
`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`.
CPU/runc/OSMesa,4CPUs16GiBmemory,noGPU/network/credentials. Read-onlyrepository
andoldoutputs, onlynewexclusivecollectionroot writable. Retaincontainer/logs.

## Outputs and gates

Root: `outputs/local_varied_data/local_varied_20260919`.
LeRobot dataset lives in its `dataset` child; rawattempts/replays under`attempts`,
all77frozeninitializations under`scenes`, plan/bank/journal/provenance alongside.
Allscenesandthreefuturepolicyseeds freeze beforefirstexpert;24futureevaluation
scenes have noexpert/policyexecution. No old240final-bank access.

60expertattempts,48training(40varied+8center),12validation. IndependentcubeXY
variation±10mm/axis via separatesplitLHS; allotherintegrationstate preserved,
noresettle. Unchanged snapshot-v2expert/waypointjitter, state9/action8,two512AV1,
30Hz/600Hz andstrictscore. First12(4center+8trainingextremes)mustallpass before13.
Eachdemo validated includingtwo savedcontrolreplays before nextattempt.

Firstexpert/geometry/media/replay/integrityfailure stops and preserves evidence,
no replacement/retry/resume. Budget60experts/120replaysmax,40GiBstart30GiBfloor,
6hourwallcap. Completion requires60/60+120/120 andfreshseal. No claim oftraining
readiness oranyacceptedcount at startup. No automatictraining afterward.

Future W&Bsummary export is separate from offlinecollection; localCPUresources
andstage metrics recorded, no collectorGPUutilizationclaim. Newdata size alone
does not guarantee improvedpolicybehavior. Validation/evaluation are localdomain
interpolation, not fullworkspace or visualgrounding proof.

Do not edit newpackage/README/spec/plan oroldimportedhelpers while running; all
arebound. This report and status/handoff can be updated independently. Checkactual
journal,failure/finalvalidation andcontainerexit before reportingcompletion.
