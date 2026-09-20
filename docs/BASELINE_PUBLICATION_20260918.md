# Baseline archive and publication — 2026-09-18

User approved authenticated service use, privately parsed credentials, bothHFmodels
PRIVATE, and W&Bentity `vivekchaitanya07-university-of-stuttgart`. Credential files
were never displayed/diffed/staged; process-only parsing occurred in isolatedDocker.
Alloriginalcheckpoint/data/runtime evidence remains unchanged. No newtraining,
policyrollouts or physics experiments were performed.

## W&B historical runs: imported

Private project:
https://wandb.ai/vivekchaitanya07-university-of-stuttgart/franka-fixed-layout-baselines

- [SmolVLA18/20](https://wandb.ai/vivekchaitanya07-university-of-stuttgart/franka-fixed-layout-baselines/runs/smolvla-c830749e504b041f3a3b6027)
- [X-VLA20/20](https://wandb.ai/vivekchaitanya07-university-of-stuttgart/franka-fixed-layout-baselines/runs/xvla-87cc4f92079d8abc10818e59)

Each imported5000originalupdate records, selectedactualrecipe/trainstats,
offlineprobes and all20trial outcomes. `logging_mode=historical_import` distinguishes
importtime from originaljobtime. MissingpastGPUutilization/power/energy remain
null/not_recorded. System/code/git/console/machine capture disabled forimport;
no currentGPUtelemetry presented as trainingmeasurement, no weights/videos/code
artifacts uploaded toW&B. SDKfinish returned, privateprojectaccess rechecked.
Exhaustive server-side history readback was not performed; do not claim that audit.

Initial attempts stopped beforelogging: defaultentity differedfromusername;
user specified exactapprovedentity, then observedW&Bschema used`access:String`
rather than`visibility`. Readonlyschema checks and regressiontests resolved both,
retaining failclosedprivatechecks. Oldattempt reports preserved.

Evidence: [successful import](../outputs/publication/baselines_20260918/publisher-attempt-f8b6c0a036ec412eb82fe0773b0e9f34.json).
Local [two-baseline registry](../outputs/publication/baselines_20260918/baseline-registry-17c5ca3dc92a4647a740d5d02d85e8c9.json).
This registry covers two completed baselines, not every historicalexperiment.

## Hugging Face: BLOCKED, no weights published

Attempt authenticated expectedaccountVivekSai07 and createdPRIVATE
`VivekSai07/smolvla-franka-stacking-fixed-layout-v1`, butLFSpreupload returned
HTTP407proxy authentication. No modelcommitverified/published. X-VLAcreation/upload
not attempted after firstblockedmodel; do not implyitsrepoexists.
No proxybypass, broadretry, remotevisibilitychange or partialrepo deletion.

Evidence: [blocked upload](../outputs/publication/baselines_20260918/publisher-attempt-6a8106ddcbc9495bb047bd83ee156116.json).
User/IT must restore approved upload-route proxy access. StandardHFroute attempted
includes hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com. After repair, inspect
existingprivateSmolrepoHEAD/files before authorizing a narrowlyscoped continuation;
the current publisher deliberately refuses existingrepos, evenempty. Do not rerun
blindly or weaken overwrite/collision checks.

## Portable local releases: verified and ready

- [SmolVLA package](../outputs/publication/baselines_20260918/packages/smolvla/)
- [X-VLA package](../outputs/publication/baselines_20260918/packages/xvla/)

Finalweights remain byteidentical to evaluatedcheckpoints. Independentrelocated
CPUloading with originalrepo/caches unavailable verifies all500SmolVLAtensors
and903serializedXVLAtensors bitexact, processor states and oneobservation/seed
actionchunk equivalence maximumerror0. This is CPUoneprobe packagingverification,
not GPUequivalence/generalization or repeatedtaskevaluation.

Relative bundledtokenizer/config paths and standaloneFP32-awareloader avoid
original/vla cache dependencies. Explicitconfigrewrite map; originalcheckpoints
untouched. Actual500warmup/5000customschedule documented separately from stale
serializedschedulerdefaults. Apache2text/upstreamnotices and pinnedlicensecitations
included; historicalSmolcard lackedlicense,currentupstreamdeclaration pinned
d9f33c94a60fb382c90dea2164c96845bd955e28. No blanketlegal-clearanceclaim.
Optimizerstates/credentials/datasets/simulatorassets excluded byallowlist.

72publicationtests pass; independent static safetyreview plus main runtimechecks.
No fullsimulationreproduction package uploaded; localDocker/data/backupsstillneeded.

## Next work

FutureliveW&B logging is NOT yet integrated into frozen customtrainers. Add an
explicit newversioned wrapper before nexttraining; use liveSDKsystemmetrics only
whenGPU/NVMLavailable, devicewide/sharedGPU caveat. PreserveSDKlocalspoolonfailure
and keep localstructuredreports authoritative. Never rerun pasttraining forgraphs.

Proposednextcapabilitystudy: boundedsmallposition evaluationonly, frozenmodels,
no newdemonstrations/training. Exactdesignawaitsreview; no rolloutstarted here.
