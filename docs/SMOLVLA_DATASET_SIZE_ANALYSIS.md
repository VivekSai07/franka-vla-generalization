# SmolVLA dataset-size assessment for custom Franka stacking

Research date: 2026-09-15. Scope: the custom MuJoCo Panda collector in this
repository, not physical-robot deployment or the hyphenated research project.
No collection, deletion, training or model-weight download was performed for this
assessment. The initially missing “Pasted text #1” was subsequently supplied;
its primary references and corrections are assessed in section 7 below.

## Recommendation

Treat **300 successful demonstrations as a pilot**, not a proven sufficient
dataset. For the existing twelve colour-pair variants and two/three/four-cube
scenes, budget **600 training demonstrations plus 120 held-out expert
demonstrations**, collected in stages. Start with a small train/reload/rollout
integration check, then 300 training demonstrations, evaluate, and expand to
600 only if needed. Reserve a further expansion to 1,200 training demonstrations
for observed coverage gaps or a learning curve that is still improving.

These are engineering recommendations, not thresholds established by the cited
papers. It is possible that a well-designed 300-demo set is sufficient; no
available source establishes that it will be sufficient for this environment.
Likewise, 600 is a budget, not a success guarantee.

## 1. Primary-source findings

### Official SmolVLA documentation and paper

- [LeRobot SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla): recommends
  approximately 50 episodes as a starting point for a task, with enough examples
  of each introduced variation. Its example uses five positions with ten
  demonstrations each; a similar 25-episode dataset performed poorly.
- [SmolVLA paper v1, sections 3.2 and 4](https://arxiv.org/html/2506.01844v1):
  community pretraining uses 481 datasets, 22.9K episodes and 10.6M frames.
  This is pretraining scale, not a requirement for each downstream task.
- The paper's real-world experiments describe 50 demonstrations per task.
  The SO100 stacking result of 90 is under a multitask setting with robotics
  pretraining, and real-world scoring permits partial credit. It must not be
  advertised as 90% binary success on this custom Franka task from 50 demos.
- The paper uses 1,693 LIBERO episodes across 40 tasks and 2,500 Meta-World
  episodes across 50 tasks. Simulation training is described as 100K updates
  at batch 64, with no robotics pretraining for its simulation table.
- The paper's SmolVLA-0.45B LIBERO table reports 90/96/92/71 for
  Spatial/Object/Goal/Long, average 87.3. This is a paper experiment, not verified
  performance of every checkpoint bearing the SmolVLA name.
- [Official launch blog](https://huggingface.co/blog/smolvla) illustrates
  fine-tuning at 20K updates and batch 64. The paper's larger experimental
  budgets and the blog's starter recipe are distinct, not universal defaults.

Source-version caveats: the launch blog says 487 community datasets whereas the
paper table says 481. The current public SO100 stacking dataset metadata reports
56 episodes, not exactly the 50 described for the paper experiment. Do not mix
these snapshots as though they describe the same immutable dataset revision.

### HuggingFaceVLA/smolvla_libero

[Model card](https://huggingface.co/HuggingFaceVLA/smolvla_libero), inspected
revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.

The card is sparse and contains generic examples, including `policy.type=act`.
It does not establish a fine-tuning dataset-size recommendation, and no
`train_config.json` was listed or retrievable. The Hub metadata presents it as
a derivative of smolvla_base, but that is not enough to reconstruct training
history or identify it with a particular paper result.

[Published config](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/6721902bc4d61e50a3bfdb11dfb4cb626f05d102/config.json)
has two visual inputs, 8 state dimensions, 7 action dimensions, maximum padded
state/action dimensions of 32, expert width multiplier 0.5, chunk size 50 and
`n_action_steps=1`. Its VLM/layer settings also differ from smolvla_base.

It is a reasonable first transfer candidate for a Panda simulation, conditional
on an actual weight-loading and custom-action adaptation test. It is not a
ready-to-run policy for this repository's joint-position action convention.

### Beeface/smolvla-libero-spatial

[Model card](https://huggingface.co/Beeface/smolvla-libero-spatial), inspected
revision `0728a96dcb64e1a16ed7fe3723fe880bfaba598b`.

The author reports base initialization from smolvla_base, 20K updates, batch 8,
an L4 24 GB GPU and about 2.5 hours of training. Reported Spatial success is
36% overall, with task scores from 10% to 70%. The reported step sweep is
2K:2%, 6K:17%, 10K:31%, 20K:36%. These are self-reported results; the evaluation
trial count, confidence intervals and repeated-seed evidence are not provided.

Crucially, the [saved training configuration](https://huggingface.co/Beeface/smolvla-libero-spatial/blob/0728a96dcb64e1a16ed7fe3723fe880bfaba598b/train_config.json)
sets `dataset.repo_id=HuggingFaceVLA/libero`, `episodes=null`, `revision=null`
and `env=null`. That identifies no Spatial-only training subset. The full
referenced dataset has 1,693 episodes and 40 tasks. An external data alteration
cannot be excluded, but the public artifacts do not support asserting that this
model trained on exactly 500 Spatial demonstrations.

The [model config](https://huggingface.co/Beeface/smolvla-libero-spatial/blob/0728a96dcb64e1a16ed7fe3723fe880bfaba598b/config.json)
uses expert width 0.75, 16 VLM layers and `n_action_steps=50`. Consequently its
reported score is not a controlled dataset-size comparison against the official
LIBERO checkpoint. More steps, different data and different action execution
settings are confounded. Do not infer that convergence or a sufficient data
budget was established at 20K steps. Use this model as an optional comparison,
not the primary justification for choosing 300 demonstrations.

### lerobot/smolvla_base

[Model card](https://huggingface.co/lerobot/smolvla_base), inspected revision
`c83c3163b8ca9b7e67c509fffd9121e66cb96205`.

This is the official general-purpose fine-tuning starting point. The community
robotics pretraining described by the paper focuses on SO100, not native Panda
joint-target control. Its [published config](https://huggingface.co/lerobot/smolvla_base/blob/c83c3163b8ca9b7e67c509fffd9121e66cb96205/config.json)
specifies 6 state and 6 action dimensions, three named camera inputs, padded
dimensions of 32, expert width 0.75 and `n_action_steps=50`.

It is the most directly documented baseline and should be compared against the
LIBERO initialization on the same pilot data if resources allow. Its feature
metadata needs adaptation; the three-camera schema does not mean every
downstream dataset must collect three cameras. Both actual schema compatibility
and normalization must be checked in the installed LeRobot version.

## 2. Why LIBERO episode counts do not transfer directly

[Official LIBERO documentation](https://huggingface.co/docs/lerobot/libero)
and [dataset metadata](https://huggingface.co/datasets/HuggingFaceVLA/libero/blob/86958911c0f959db2bbbdb107eb3e17c5f9c798e/meta/info.json)
identify 1,693 episodes, 273,465 frames, 40 tasks and 10 FPS. The default LIBERO
interface uses end-effector deltas plus a gripper command; its 8-dimensional
state represents end-effector pose and gripper positions.

This repository records 7 absolute joint targets plus a 0/255 gripper command
(8 actions), 7 joint positions plus 2 finger positions (9 states), at 30 FPS.
The same robot and two cameras therefore do not imply the same control problem.
Padding to 32 dimensions makes adaptation architecturally possible; it does not
give old output coordinates the correct new meanings or normalization.

Before committing to a large budget, verify pretrained key/shape loading,
custom input/output metadata, training-split statistics, 8-dimensional inference
outputs, camera ordering and chunk execution. The width multiplier is an expert
architecture setting, not a camera-count setting. Do not use 0.5 indiscriminately
for checkpoints whose architecture uses 0.75.

The installed LeRobot 0.4.4 factory was inspected read-only: when
`pretrained_path` is supplied it loads pretrained policy weights with the passed
config, derives outputs from dataset metadata, and derives inputs only when the
config has none. Thus a generic command or inherited config is not evidence that
all source/target feature mismatches have been handled. No checkpoint loading
or training experiment was executed during this research.

## 3. Deriving a practical budget for this collector

There are 12 ordered colour pairs and three distractor-count conditions:
12 × 3 = 36 coarse conditions, before spatial layout is considered.

| Training demonstrations | Mean per pair | Mean per pair/count condition | Frames at 520/episode | Collection time at 85 s/episode |
|---:|---:|---:|---:|---:|
| 300 | 25 | 8.3 | 156,000 | 7.1 h |
| 600 | 50 | 16.7 | 312,000 | 14.2 h |
| 1,200 | 100 | 33.3 | 624,000 | 28.3 h |

Times are extrapolations from two successful smoke episodes, not measured bulk
throughput; retries, encoding and load can increase them. An additional 120
held-out expert demonstrations add approximately 2.8 hours at that rate.

The twelve colour pairs share the same stacking mechanics, so multiplying every
single-task guideline by twelve is not a theorem. Conversely, 300 nearly
duplicate demonstrations are not equivalent to 300 diverse demonstrations.
The proposed 600 balances repeated per-instruction coverage with manageable
collection cost. It applies to the current fixed-camera simulated task, not
unseen physical robots, arbitrary objects or broad sim-to-real generalization.

### Coverage must be measured among saved successes

The collector samples a new instruction and layout on each attempt. Failed
attempts are discarded, then another instruction/layout is sampled. Therefore
uniform attempted-task sampling does not guarantee uniform saved-task counts.
With equally successful tasks, 300 attempts give an expected 25 per pair but a
standard deviation of about 4.8. Different task/layout success probabilities
introduce additional selection bias. Merely running longer does not remove that
bias.

The distractor count is sampled uniformly from 0, 1, 2 before filtering, not
50%/25%/25%. Jitter is Gaussian with standard deviation 5 mm, not a hard ±5 mm
bound. It is applied to approach, descent, move-to-place and lower-to-place
trajectory targets; it is not a general perturbation/recovery data generator.

For a 600-demo training target, aim for 50 saved demonstrations per pair with
approximately 16–17 in each distractor-count condition. Within those groups,
cover near/far, left/right and varied relative pick/place layouts. Record seeds,
initial conditions, instruction counts and failed-attempt reasons. The current
collector has no quota or seed CLI, so guaranteed quotas and reproducible splits
would require a separately approved change; this research did not implement it.

## 4. Experimental plan and stopping criteria

1. Inspect videos and data semantics, and run a tiny train/save/reload/rollout
   integration test. A two-episode alignment PASS does not validate training.
2. Build a 300-training-demo pilot, with balanced successful coverage where
   practical. Keep the two-episode smoke set separate from headline evaluation.
3. Reserve 120 held-out expert demos (10 per pair) for offline validation; never
   put frames from one trajectory into different splits. Use training-only
   normalization statistics. Group related/counterfactual layouts into one split.
4. Evaluate checkpoints in closed loop on fixed, independent scene seeds, with
   at least 10 trials per pair for tuning. Lock a separate final test set of
   20 trials per pair (240 trials), balanced across distractor conditions as
   closely as possible. These policy trials are not training demonstrations.
5. Report binary released-and-stable stacking success, correct object selection,
   grasp success and placement error separately, plus per-pair and worst-group
   performance. Use consistent geometry/time criteria in comparison runs.
6. If the pilot meets the user's predefined target on independent evaluation,
   stop; if not, diagnose the failure before adding data. Expand to 600, then
   possibly 1,200 using targeted conditions when evidence indicates a coverage
   limitation. Near-zero performance warrants pipeline investigation first.

Use the same pilot/splits/evaluation seeds to compare base and LIBERO
initialization. Repeat promising settings with multiple training seeds. With
20 final trials per pair, per-pair uncertainty is still large; report confidence
intervals rather than treating a point estimate as a guarantee.

Current evaluation mismatches must be addressed before claiming twelve-condition
performance: the client always spawns four cubes while training varies cube count;
it accepts XY <3 cm for 30 consecutive ticks, while collection accepts XY <2 cm
at the final check. A request per tick also does not guarantee a new predicted
chunk per tick because `select_action` can consume its internal action queue.

## 5. Dataset size is not training duration

For a single GPU without gradient accumulation, approximate sample-equivalent
epochs are updates × batch size / training frames. This is only an accounting
proxy: temporal chunks overlap and nearby frames are highly correlated.

| Training episodes | 20K updates, batch 32 | 80K updates, batch 32 |
|---:|---:|---:|
| 300 | 4.10 | 16.41 |
| 600 | 2.05 | 8.21 |
| 1,200 | 1.03 | 4.10 |

Thus retaining 80K updates while doubling the dataset halves the average frame
exposure. Beeface's 20K × 8 is 160K samples, whereas 20K × 64 is 1.28M samples.
The paper's 100K × 64 is 6.4M samples. Equal update counts are not equal training
budgets, and differing inference settings further confound score comparisons.

Use 20K updates as a checkpoint to evaluate, not an automatic stop. Consider
40K and 80K only while closed-loop validation improves, and check the learning
rate schedule: the published configs specify a 30K-step decay schedule, so
requesting 80K steps alone does not establish an 80K-step decay schedule.

## 6. Corrections to previous project assumptions

The historical dataset-design document attributes failure to insufficient data
and a single instruction too definitively. Those claims are hypotheses without
controlled ablations, especially because historical training data also had an
alignment bug. A fixed instruction can be sufficient for a genuinely fixed task;
it cannot establish the requested multi-instruction grounding ability. The
paper itself includes fixed-instruction stacking experiments.

The new RTC v1 smoke dataset is post-fix and passed verification; it must not be
confused with an older pre-fix dataset also named v1. Dataset names alone do not
establish provenance. No pre-fix checkpoint may be resumed.

## 7. Addendum: supplied Franka summary and its original references

The supplied text is a generated secondary summary, not an experimental report.
Its claims were checked against the linked model/dataset cards, their saved
configuration and the original issue discussions/blog article. Aggregator sites
are not treated as independent evidence for a demonstration-count threshold.

### IDEAS Lab joint-space Franka example

[Jar checkpoint card](https://huggingface.co/IDEAS-Lab-Northwestern/smolvla-base-datagen-v1-jar-joint-2cam),
revision `a2983a52e65eece73cfe577d5edd3015d6b2d21b`, explicitly describes
fine-tuning `lerobot/smolvla_base` for simulated Franka absolute joint targets
with an overview and wrist view. This strengthens the case for including
smolvla_base as a first-class joint-control baseline, rather than assuming a
LIBERO checkpoint must be superior merely because LIBERO uses Panda.

The linked [dataset card](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam)
and metadata, revision `080dddb4f51ab060e9127e322c06e8ba4147d83c`, report:

| Property | Published value |
|---|---|
| Successful, safety-filtered episodes | 1,040 |
| Base task configurations | 26, with 40 demonstrations per configuration |
| Unique language tasks | 21 |
| Frames | 946,870 |
| FPS | 30 |
| Raw camera streams | 5; model card describes a prepared 2-camera copy |
| Training | 3,699 updates, global batch 512, approximately 2 epochs |

The 26 base configurations and 21 language tasks are different counts, not a
contradiction. The approximately 910 frames/episode also differ from our measured
520. Training sample exposure is 3,699 × 512 = 1,893,888, approximately twice the
published frame count. The model card does not supply a dataset-size ablation or
a success-rate guarantee for our stacking task. Therefore 1,040 is evidence of
one actual training budget, not a minimum requirement for Franka.

There are important reproduction caveats:

- Despite the card's 8-D state/two-camera description, its published config
  retains a 6-D state and three base camera entries, while output action is 8-D.
  The preprocessor maps top/wrist to camera1/camera2 but also retains those
  base feature declarations. This is an artifact inconsistency to investigate,
  not proof that the experiment failed or that the checkpoint cannot run.
- The raw dataset separates `actions` (next-achieved joint positions) from
  `actions_commanded` (planner-commanded targets). Its card also mentions
  training-time joint-delta choices, whereas the checkpoint card describes
  absolute commands at evaluation. The exact prepared-data conversion must be
  inspected before borrowing this recipe. Our collector records current-tick
  commanded targets; these targets are not interchangeable with next-achieved
  positions under tracking error.
- The model card's usage example references a different `-yanZ` repository
  suffix. Use verified artifact paths, not that example verbatim.
- Dataset license is CC BY-NC 4.0; the checkpoint card flags upstream weight
  licensing as undeclared. Review applicable terms before reuse, particularly
  commercial use. No data or weights from this project were imported here.

### Xavier O'Keefe's custom Panda experiment

The [original article](https://medium.com/correll-lab/fine-tuning-smolvla-for-new-environments-code-included-af266c56d632)
uses Unity/ROS2/MoveIt with the LIBERO checkpoint. It reports starting with 25
demonstrations, then collecting 100 more (approximately 125 total). In roughly
30 informal evaluation trials, the author observed about 95% touching and 40%
grasp success. Those outcomes are not complete stacking success and were not
presented as a rigorous evaluation.

The current linked `Xavier033/new_libero` metadata instead reports 127 episodes,
11,905 frames, four task labels and 10 FPS: a later dataset snapshot should not
be conflated with the article's rounded narrative. It records 8-D end-effector
pose/gripper state and 7-D pose-delta/gripper actions, not our joint-target
interface. The article's references to 2,500/12,000 “steps” are not used here
as verified optimizer-update counts.

The strongest transferable lesson is replay and normalization validation. The
author reports materially improving behavior after replacing LIBERO statistics
with statistics for the target dataset and validating recorded trajectories.
Our numerical alignment test is useful but is not a full action replay through
the deployment path. Such a replay should reproduce the initial scene/robot
state and action cadence; deterministic resets or captured simulator states
may be needed, since the current collector does not persist a complete replay
initialization record.

### The linked GitHub issues are cautionary evidence

- [Issue 2259](https://github.com/huggingface/lerobot/issues/2259) starts with a
  report of 500 Franka RLBench demonstrations and 0% SmolVLA success despite
  decreasing loss. A later comment identifies stale cached dataset features as
  one error, but still reports poor performance. This is not a controlled proof
  that 500 is insufficient; it demonstrates why interface/cache/evaluation
  debugging must precede interpreting dataset size or training loss.
- [Issue 1567](https://github.com/huggingface/lerobot/issues/1567) includes an FR3
  lab report that SmolVLA was not smooth after fine-tuning and that a Diffusion
  Policy performed better on their task. No universal sample budget follows
  from that anecdote. Historical discussion of hardware-driver availability is
  not proof of the current support matrix.

### Corrections to the generated summary

1. **Franka does not have one universal 8-D state representation.** The jar
   project uses seven arm joints plus one gripper scalar. Our state includes
   both finger positions and is 9-D; our action is 8-D. Standard LIBERO is
   8-D state/7-D action with end-effector semantics.
2. **Padding is not embodiment conversion.** It accommodates vector lengths,
   but does not align coordinate conventions, units, gripper meaning, action
   semantics, cameras or normalization statistics.
3. **Fine-tuning does not normally reinitialize the action expert.** Loading
   smolvla_base transfers its pretrained action expert and updates its weights.
   Training a fresh expert on a pretrained VLM is a different experimental setup.
   Freezing only the vision encoder and freezing the whole VLM are also distinct;
   inspect the actual configuration/trainable parameter list.
4. **Smolvla_base is a valid default, not the only usable version for Franka.**
   Base, LIBERO-adapted and other task-adapted checkpoints exist; the paper also
   studies different model sizes. These sources do not prove a universal best
   checkpoint for joint-space stacking.
5. **Asynchronous inference is optional deployment infrastructure.** Loading
   weights does not automatically enable it. Our server/client requests are
   synchronous, and internal action chunking is a separate concept.
6. **Joint outputs are not sufficient for safe real-hardware deployment.**
   Matching control mode, limits, cadence, gripper conventions and appropriate
   robot safety mechanisms are still required. Our evidence is simulation-only.

### Effect on the recommendation

The budget remains **300 training demos as a pilot, 600 as a conditional target,
plus 120 held-out expert demos**. The new joint-space example makes the base
checkpoint a stronger baseline candidate; it does not establish that 300 is
enough or that 1,040 is required. Before bulk collection, add a replay/integration
gate to the plan, then compare base versus LIBERO on identical corrected data
and independent closed-loop evaluation. No checkpoint choice or collection
configuration was changed as part of this assessment.

## Decision summary

**Recommended approval:** staged data collection, 300-demo training pilot,
600-demo training budget plus 120 held-out expert demonstrations, with expansion
conditioned on closed-loop results. Start from the official LIBERO checkpoint
only after a transfer-compatibility check; compare smolvla_base as the documented
baseline. Do not use the Beeface result as a numerical sufficiency guarantee.

No source supports promising a particular Franka success rate from 300 or 600
demonstrations. The decisive evidence will be a controlled learning curve on
this exact task and evaluation distribution.