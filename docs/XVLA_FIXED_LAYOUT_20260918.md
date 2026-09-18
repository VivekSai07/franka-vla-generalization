# X-VLA fixed-layout training — 2026-09-18

## Authorized and launched

User approved X-VLAfirst5000updates followed by20fixed-reset sampling trials,
initial capability>=18/20. No automaticSmolVLAjob, trainingextension, collection
or upload. [Design](superpowers/specs/2026-09-18-xvla-fixed-layout-training-design.md)
and [implementation plan](superpowers/plans/2026-09-18-xvla-fixed-layout-training.md).

New implementation committed02c936c, isolated under experiments/xvla_fixed_layout.
Independent static spec/quality review found no concrete blocker.110CPUtests
passed (32new+78existing); main independently reran all32newtests successfully.
No old provenance-bound source/data/spec/report changes.

### Real admission gates

- Fresh fullsemantic fixed-layout readmission and realCPUprocessor checks passed,
  container xvla-fixed-layout-cpu-admission-20260918 exited0.
- Exactly10560training rows from IDs0-19,9980complete30action chunks. IDs20-23
  excluded from optimization and stats; they share the same geometry.
- Two-update micro4xaccum2GPU scratch exited0/noOOM. StrictFP32weights+Adam
  reload exact, prediction error0, final source/data/cache/statistics bindingPASS.
  Zero scratch rollouts; scratch weights not reused for initialization.
- Scratch peak allocation21,333,082,112bytes; free hostdisk~108GiB andGPUfree
  45,287MiB afterscratch. UnrelatedmoveitFrancesco remains untouched.

### Full run

Container `xvla-fixed-layout-5000-20260918`, ID
`016ab837e837e58957211770e0a693429f6d61e0bda8785161aceef9d73e7af4`,
started2026-09-18T15:14:07Z, verifiedrunning on pinnedofflineimage
`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`.
Read-onlyrepo/data/cache and containerroot; only newtrainingroot writable,
temporarycaches, no network/credentials/downloads. GPUmemory/resource checks
remain enforced. Container/logs retained after exit.

Output: `outputs/train/xvla_fixed_layout_20260918`.
CPUproof sibling `_admission`, scratch sibling `_preflight`.
Do not modify newsource/tests/README/design or olddata while running; they are
hashed in admission. This report/status/handoff are outside that binding.

FreshpinnedXVLAbase,allFP32trainable/BF16autocast,nativeMSE,trainseed20260917,
micro4xaccum2/effective8,5000updates/40000chunks,500warmup1e-4→1e-5cosine,
VLM0.1x; previous verifiedoptimizer settings unchanged. Checkpoints1/1000..5000,
optimizer onlyfinal. Final5000selected, no validation-selected checkpoint.

After strict finalreload:72offlineprobes (frames60/107/347 on20train+4validation),
observation-onlyprediction and physicalerrors; then20rollouts with distinct
seeds20260918..20260937, same sourceepisode7 fullreset,40smax,30Hz,
chunk30/execute10, unchanged strictreleasedstack30consecutive ticks.
Report approach/closure/contact/lift/release/stack/clipping separately.
Gate>=18successes requires all20trials completed. No retry/resume/extension.

Current entry records LAUNCH ONLY, not completedupdates or policy success.
Check actual result and container state. Future main completion should preserve
training bytes across evaluation and report data integrity separately from task
success. Same-layout sampling reliability is not spatial generalization;
dataset size/expert changes mean no randomization-only causal comparison.
