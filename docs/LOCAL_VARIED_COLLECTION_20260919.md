# Local varied collection — 2026-09-19

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
