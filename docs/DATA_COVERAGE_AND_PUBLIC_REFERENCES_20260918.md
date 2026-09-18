# Dataset coverage and public VLA references — 2026-09-18

## Scope and decision

User authorized two research threads: investigate our training-layout coverage
and failed approaches, and inspect the supplied Ethan model/dataset plus other
public Hugging Face references. No training, new policy rollouts, collection,
physics stepping, new model weights or videos were downloaded/executed.

**Conclusion: spatial coverage is sparse, but more random demonstrations are
not a demonstrated solution. Closure timing/state and placement coverage need
separate attention.** Public model names are not evidence of high performance;
the supplied Ethan checkpoint lacks a published measured result in the inspected
repositories. A native successful benchmark is a useful positive control, not
a plug-in replacement for our custom joint-action interface.

## 1. Exact local coverage and behavior

Compared nine training IDs0–8 with held-out300–303 only, same red-on-green task,
two cubes. Earlier 300-demo coverage numbers are not substituted for this subset.

| Held-out episode | Pick within train convex hull | Nearest train pick distance | Nearest combined pick/place XY distance |
|---|---|---:|---:|
|300|Yes|147.3 mm|162.0 mm|
|301|No|134.8 mm|194.5 mm|
|302|No|120.9 mm|266.2 mm|
|303|No|29.4 mm|32.8 mm|

All destination locations lie within the training destination hull. Episode300
also lies inside the combined pick/place hull, so pure outside-hull extrapolation
cannot explain all four failures. Hull membership is not local density or proof
of easy interpolation. Episode303 is close to successful training episode2 yet
still fails. Layout distance is descriptive, not a success threshold.

Actual MuJoCo FK of recorded joints agrees with saved body positions and logged
finger-origin distances. The finger-body midpoint is not the fingertip midpoint:
use the expert-defined cube+105mm hand goal for hand-position comparisons.
Initial robot positions/velocities/controls and checked robot fields match across
all13scenes, so differing robot reset postures do not explain the split gap.

### SmolVLA: wrong approach locations

First-close XY errors on300–303: **149.6,524.2,415.5,66.7mm**. The first three
close locations are only20.7,21.2,30.7mm from some training pick positions.
This is consistent with layout-biased behavior but does not prove memorization:
nearest-of-nine comparisons, shared starts and phase timing bias similarity.

### X-VLA: missing or late closure, not only incorrect positions

- Episodes300–302: **zero closed commands in1,200ticks**; raw gripper minima
  254.42,249.46,245.98, all above the actual127threshold. No first-close position
  exists. Closest hand-goal distances239.9,204.1,203.9mm.
- Episode303: comes within **10.07mm** of the hand goal at frame81 but stays
  open; its only closed command occurs at frame919 for one tick, with43.6mm
  lateral target error. This is a close approach, not proof of feasible grasp.
- Seen episode8: approaches within **7.9mm** of the hand goal yet never closes.
  This failure cannot be attributed solely to unseen spatial coverage.
- Seen placement failures remain: five SmolVLA and four X-VLA trials contact/lift
  the target but do not finish the correct stack.

Checks: 181selected input files unchanged, all26saved policy cases covered, five
numerical tests and independent geometry/metric checks passed. Raw action checks
independently confirmed the never-close and one-tick-close counts. No policy
inference or physics step occurred in this analysis.

Evidence: [findings](../outputs/diagnostics/spatial_coverage_20260918/FINDINGS.md),
[all-case report](../outputs/diagnostics/spatial_coverage_20260918/REPORT.md),
[spatial map](../outputs/diagnostics/spatial_coverage_20260918/spatial_layout.png),
[verification](../outputs/diagnostics/spatial_coverage_20260918/final_audit.json).

## 2. User-supplied Ethan checkpoint and dataset

- [Model](https://huggingface.co/ethanCSL/svla_franka_pick_n_place_vla_steering_libero_osc_natural),
  inspected revision `eadc20523fe204b2683348a828e4cc99c0a18ee1`.
- [Dataset](https://huggingface.co/datasets/ethanCSL/svla_franka_pick_n_place_vla_steering_libero_osc_natural),
  inspected revision `67cd6fca412bdac3da7630294a09f3a4f5774660`.
- **50episodes,4,089frames,20FPS,one task**, two256RGB cameras,8state/7action.
  Task: pick the Akita black bowl between plate and ramekin and put it on plate.
  Episode lengths63–167frames; variable duration, not evidence of recovery by itself.
- Saved intended training: SmolVLA base,20kupdates,batch16,expert-only fine-tuning,
  frozen VLM,chunk50/execution50. Completion and actual runtime recipe unverified.
- **No measured evaluation success rate or raw outcome report found.** A configured
  evaluation count of50 is not50 successful policy trials. No held-out split declared.
- Dataset and normalizer state width8 conflict with saved declared state6;
  two real cameras coexist with extra configured camera slots. These may be
  inherited defaults, but exact runtime behavior is not established.
- Action values are inside[-1,1] in published stats and suggest6motion+gripper
  OSC commands; exact units/frame/scaling/polarity remain undocumented. They are
  not our7absolute joint targets+gripper. Do not trim/copy them into our controller.

Pinned sources: [info](https://huggingface.co/datasets/ethanCSL/svla_franka_pick_n_place_vla_steering_libero_osc_natural/raw/67cd6fca412bdac3da7630294a09f3a4f5774660/meta/info.json),
[training configuration](https://huggingface.co/ethanCSL/svla_franka_pick_n_place_vla_steering_libero_osc_natural/raw/eadc20523fe204b2683348a828e4cc99c0a18ee1/train_config.json).
[Detailed evidence and12pinned citations](../outputs/research/hf_franka_20260918/ethan_review.md).
Only bounded metadata was inspected, including one small episode-metadata parquet
and a small normalization tensor file, not model weights or videos.

## 3. Public shortlist: evidence strength, not a universal leaderboard

Bounded HF searches for smolvla/franka/libero/xvla and targeted namespaces;
six additional pairs inspected. Not an exhaustive crawl or independent
reproduction. Likes/downloads were not used as task-performance evidence.

| Reference model | Dataset reference | Public evidence | Usefulness/limits |
|---|---|---|---|
|[LeRobot pi05 v044](https://huggingface.co/lerobot/pi05_libero_finetuned_v044)|[HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero)|Official guide links97.5%to checkpoint; documented400trial reproduction protocol|Strongest checkpoint-linked positive reference reviewed; different EEF control/task/pretraining/compute|
|[LeRobot xvla-libero](https://huggingface.co/lerobot/xvla-libero)|[HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero), transformed for absoluteEEF|Guide advertises93%; exact score-specific protocol less complete|Same model family; requires original domain3, absoluteEEF preprocessing and control; not ourjoint8|
|[Beeface smolvla-libero-spatial](https://huggingface.co/Beeface/smolvla-libero-spatial)|[HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero)|Author reports36%Spatial; no bound trial/seed denominator found|20kbatch8 modest-compute reference; config episodes=null on full40task source, not provenSpatial-only|
|[LeRobot smolvla_libero](https://huggingface.co/lerobot/smolvla_libero)|[lerobot/libero](https://huggingface.co/datasets/lerobot/libero)|No checkpoint-specific score found|Saved25kbatch32/full-tuning recipe; do not attach separate paper87.3%to weights|
|[IDEAS SmolVLA Jar](https://huggingface.co/IDEAS-Lab-Northwestern/smolvla-base-datagen-v1-jar-joint-2cam)|[datagen-jar-v1-joint-5cam](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam)|No card checkpoint metric; associated paper poorJar results, exactweight binding unresolved|Closest joint8/Panda control reference, not a proven high performer; commanded vsnext-achieved actions differ|
|[LeRobot pi0 v044](https://huggingface.co/lerobot/pi0_libero_finetuned_v044)|[HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero)|No checkpoint-specific numerical score located|Secondary candidate, weaker evidence thanpi05; initialization history partly unresolved|

LIBERO dataset metadata: **1,693episodes,273,465frames,40tasks,10FPS**, state8/action7,
two256RGB views. It is not automatically directly compatible with our30Hz joint
controller. X-VLA absoluteEEF training uses a derived representation, not raw
relative-action rows unchanged. A50step chunk has different duration at10vs30Hz.

Pi05's97.5% follows **additional6ktraining updates on a LIBERO-trained base**,
not6kfrom a general base or nine demonstrations. The guide's current starter
command is not identical to the published run. Treat reported success as a
first-party benchmark claim, not a reproduction performed here.

IDEAS data includes1,040episodes/26base tasks, with separate commanded and
next-achieved joint targets. Its schema/cards/paper contain material conflicts
about views, state size, FPS and batch. Similar robot/control naming does not
guarantee successful learning or interchangeability. License restrictions also
apply separately to datasets, weights, tokenizers and assets.

Primary performance reference: [pi05 guide](https://huggingface.co/docs/lerobot/pi05#libero-benchmark-results).
Full pinned revisions, recipes, caveats and source links for all six pairs:
[comparison review](../outputs/research/hf_franka_20260918/comparison_review.md).

## 4. What to do next with our dataset

Do not relabel existing data, substitute a downloaded OSC vector for joint
commands, change the gripper threshold or promise more episodes will fix this.

If a new data experiment is approved, the evidence favors **one-task two-cube
spatial densification with complete successful closure/placement supervision**:

- Predefine a feasible workspace region and deliberate pick/place coverage,
  including interior gaps; vary both locations, not just task quotas/ranges.
- Require meaningful grasp-ready states, sustained closure/contact/lift and
  verified release/stack behavior. Record failures, do not silently filter
  difficult cells away or label noisy/failed commands as successful corrections.
- Do not expand instructions/distractors simultaneously. Keep model/recipe fixed
  for the data comparison, and set a finite episode/training/evaluation budget.
- Separate approach localization, sustained-close decision, grasp, transport,
  release and stable-stack metrics. Closely matching an expert hand position
  is not sufficient when the policy never closes.
- The four former held-out scenes have now informed diagnosis. If data is
  collected around them, treat them as **development probes**, not untouched
  held-out evidence. Reserve new independent layouts before any new collection;
  keep the240sealed final scenes untouched.

A second useful option is a **native public-benchmark positive control**, using
one published model in its intended environment with exact processors/control,
without custom fine-tuning first. Pi05 has the strongest reviewed evaluation
provenance; X-VLA is closer to our installed family. This can establish that a
reference pipeline operates as reported, but does not itself repair custom data.
Do not turn this into another long model sweep or assume native success transfers.

These are recommendations, not launch authorization. Exact new data budget and
expert feasibility remain unresolved, given the known failed scripted placements.
No additional training, collection or benchmark rollout began during this review.