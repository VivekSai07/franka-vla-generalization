# Paired-instruction and training-scene diagnostic — 2026-09-17

## Verified completion and interpretation

Container exited **0**, no OOM, at 12:33:54 UTC (start 10:03:47; about
2 h 30 min). All **48 trials**, **12 original comparisons**, and **48 CPU
replay checks** completed. Saved completion flags indicate unchanged inputs.
Fresh post-completion inspection rechecked baseline inventory, plan/source and
derived-state hashes, input preflight fingerprints, summaries and all original
trace comparisons: **12/12 bit-exact**. No new policy runs were made by that
inspection. No model, dataset or final-scene edits.

| Condition | Correct stacks | Requested first bilateral-contact cube | No first bilateral contact |
|---|---:|---:|---:|
| Original instruction | 2/12 | 3/12 | 8/12 |
| Reversed instruction, same seed | 1/12 | 1/12 | 8/12 |
| Original instruction, alternate seed | 1/12 | 1/12 | 9/12 |
| Seen training scenes | 1/12 | 2/12 | 7/12 |

Reversed conditions had **zero first-object switches**: four scene pairs had
bilateral contact in both runs and chose the same physical cube; eight pairs
had no contact in either run. Do not describe these eight null pairs as proof
of instruction invariance. There were also zero switches between two non-null
choices for the alternate-seed comparison, but three pairs changed whether any
bilateral contact occurred. Seeds affect execution reliability, not demonstrated
correct language responsiveness. One alternate seed is not a statistical study.

Concrete examples, referring to original baseline scene ordinals:

- Scene 0: original red-on-green picks red but fails placement; reversed
  green-on-red still picks red and forms the wrong red-on-green stack.
- Scene 30: original green-on-red picks red and fails; reversed red-on-green
  still picks red and succeeds. A reversed-condition success alone therefore
  does not demonstrate a change in selected object.
- Scene 50: original green-on-yellow succeeds; reversed yellow-on-green first
  contacts green and fails. Alternate seed/original instruction succeeds.
- Scene 90: original yellow-on-red succeeds; reversed red-on-yellow still
  picks yellow and fails. Alternate seed loses the grasp altogether.

The only successful seen training scene was episode **150** (blue on red),
diagnostic trial 42. Episode 75 contacted and lifted the requested green cube
but did not stack successfully. Three other training trials first contacted a
wrong cube; seven had no bilateral contact. Only two training trials reached
the sustained lifted-object criterion. These twelve predefined two-cube scenes
are a limited fit check, not a success estimate over all 300 training episodes.

Evidence: [final result](../outputs/diagnostics/paired_instruction_20260917/result.json)
and [frozen plan](../outputs/diagnostics/paired_instruction_20260917/plan.json).
Do not pool the four condition blocks as independent benchmark trials; original
runs deliberately repeat baseline scenes and all paired runs share layouts.

### Next decision, not yet an authorized job

The results support **poor instruction-dependent target selection together with
poor closed-loop reproduction even on seen scenes**. They do not isolate a sole
cause among optimization, representations, data design and compounding errors.
They do not prove language is absent from the network or that all sampled
trajectories are invariant. Scene count and seed coverage are small.

There is sufficient evidence to move from broad diagnosis to a bounded learning
intervention. Recommended next design: a small two-cube contrastive demonstration
experiment, using multiple pick/place instructions from identical initial layouts
and keeping layout families together within a split. Preserve the original 420
dataset. Before scaling, require a fresh controlled model to execute both roles
on seen layouts, then test separate held-out layouts. Explicitly compare an
equal-budget training control so any improvement is not automatically credited
to contrastive data rather than extra optimization or reduced task complexity.

Do not launch another 300+ demo collection or an unrestricted parameter sweep.
Exact demonstration count, fixed training budget and closed-loop acceptance
criteria need design approval. Poor seen-scene performance means that adding
contrastive examples alone cannot be assumed sufficient; failure on the small
paired learning task would redirect effort toward the training/model interface.
No further training, collection, upload or final evaluation was launched here.

The launch details below are historical; this diagnostic is no longer running.

User explicitly approved the **48-rollout** diagnostic after clarification that
this is evaluation, not new expert demonstration collection or training.

## Frozen protocol

- Same completed 020000 checkpoint and saved processors; no optimizer updates.
- Lowest-ordinal two-cube tuning scene for each ordered pair, independent of
  outcomes: baseline ordinals 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110.
- Three conditions per scene: original instruction/seed, reversed roles/same
  seed and original instruction/alternate seed (+1), **36 rollouts** total.
- Twelve two-cube training scenes, lowest episode ID per ordered pair:
  0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275. These test reproduction
  on seen scenes, not held-out performance; no demonstration targets enter policy inputs.
- Seed 20260916 + source ordinal/episode ID (seedcontrol adds one). All cases
  reset full initial simulator state and policy queues independently.
- Unchanged baseline trial loop: 40 seconds, 1,200 ticks maximum, 30 Hz,
  20 physics substeps, 50-action prediction, ten-action execution horizon.
- Reversed instructions change requested pick/place roles only, not cube colors
  or positions. Scoring uses the requested roles for that condition.
- Each original-condition trajectory must reproduce its saved baseline metadata
  and full trace (action absolute tolerance 1e-7, state/other float arrays 1e-8,
  relative tolerance zero; exact equality recorded separately). Stop on first
  unexplained mismatch, no automatic retry or relaxed tolerances.
- CPU saved-action replay after each policy trial measures all-cube first bilateral
  contact, ten-consecutive-tick contact, sustained contact plus lift above 5 cm,
  and all ordered-pair 30-tick stable stacks. These are diagnostic proxies.
- Null/tied object choices stay in denominators; never treat two null choices
  as evidence of successful instruction switching. Report seed and instruction
  effects separately. This small test does not establish causal certainty.
- Training resets are adapted to new output-only initialization files from
  hash-verified source traces; original demonstrations remain untouched.

## Verification and launched job

90 CPU Docker tests passed (21 new), including selection, pairing, failure
denominators, source immutability and original-trace mismatch handling. The
corrupt-video rejection test emits an expected decoder error. Static review
found no confirmed critical defect. CPU preflight verified the baseline and
restored all twelve derived training initializations; this is not GPU rollout evidence.

- Container: `smolvla-paired-diagnostic-48`.
- ID: `670ac9744bf3330ebdc198e55cf45bffdb5243f751bc3a8dff7ce1094567aa2a`.
- Output: `outputs/diagnostics/paired_instruction_20260917`.
- Network disabled; project/baseline/dataset/checkpoint read-only, diagnostic
  outputs writable. GPU was idle and approximately 85 GB disk free before launch.
- Full immutable trial plan is written before policy execution. Follow
  `result.json` for active/finished trial IDs, original comparisons and conditions.

Launch is not completion or a task-success claim. Completion requires all
48 outcomes, 12 original comparisons, 48 saved-action replays and unchanged
baseline/input/plan/source fingerprints. Preserve failures and stop for review
if protocol checks fail. No new training, collection, upload or final-scene trial.
Do not modify evaluator/probe/bound sources while the job runs.

After completion use object-role responsiveness and seen-scene reproduction to
choose one targeted intervention. Do not automatically launch another experiment.