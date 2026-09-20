# X-VLA fixed-layout training — 2026-09-18

## Completed: 20/20 fixed-layout stacks

Full container exited0/noOOM at2026-09-18T16:56:38Z, approximately1h42m31s
after launch. All5000optimizer updates and20final-checkpoint trials completed.
**20/20successes; capability gate>=18/20 passed.** No budget extension/retries,
checkpoint selection changes or automaticSmolVLA run.

All20distinct seeds20260918..20260937 used the same exact saved integration reset.
Each saved trace has contact/lift/release and30terminal released-stack ticks.
Completion times12.23–12.83simulated seconds; finalXYerror0.628–4.682mm.
Arm-joint clipping:0ticks in every trial. Gripper commands were clipped to their
allowed range; this is adapter processing, not human assistance. Historical
`unassisted_success` fields count any adapter clipping, not human intervention.

Final offline probes (not success rates):

| Split | Probes | Full30 arm MAE(rad) | First10 MAE(rad) | Gripper binary agreement |
|---|---:|---:|---:|---:|
| Train20episodes |60|0.0064924|0.0091442|96.28%|
| Validation4episodes |12|0.0082725|0.0121941|92.78%|

Fresh CPU completion audit verified current full dataset/source/cache/CPU/scratch/
statistics bindings, final configuration and six checkpoint file inventories;
all5000loss/gradient records finite and ordered; training bytes unchanged during
evaluation; all20episode reports agree with aggregate counts, exact initial
integration state, recomputed stage metrics and terminal stack streaks. Strict
GPU final reload had recorded prediction difference0. No additional policy query,
physics replay or video decode was performed by this completion audit.

Evidence: [evaluation report](../outputs/train/xvla_fixed_layout_20260918/evaluation/result.json),
[offline probes](../outputs/train/xvla_fixed_layout_20260918/evaluation/offline_probes.json),
[first rollout video](../outputs/train/xvla_fixed_layout_20260918/evaluation/trial_01_seed_20260918/cameras.mp4).
The parent training result's rollouts0 is its frozen pre-evaluation state; the
separate evaluation report records all20completed trials. Do not rewrite either.

Interpretation: this establishes successful learned stacking under the fixed
arrangement and sampled policy seeds. It does not establish arbitrary-layout
performance, language grounding, or general reliability from20samples. Fixing
layout coincided with changed demonstration count/expert; not a pure causal
ablation of randomization. Recommended next comparison is SmolVLA on this same
dataset/reset/seeds, separately approved; do not expand randomization yet.

Earlier launch text below is historical; no training job remains running.

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
