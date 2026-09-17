# Full pilot evaluation review and next decision — 2026-09-17

## Executive decision

**The baseline is valid as an executed experiment, but policy performance is
poor: 4/120 instructed stacks (3.33%).** Do not equate validation/infrastructure
PASS with task success. No demonstrated normalization, timing, label or scorer
bug warrants replacing this result. Do not discard the existing dataset or
start another large collection/parameter sweep blindly.

The highest-value next step is the bounded same-scene instruction comparison
already proposed, with a small training-scene reproduction check if needed to
separate failure to fit demonstrations from failure to generalize. Use the
result to select one targeted intervention. Full success cannot be promised.

## 1. Complete baseline and integrity inspection

- Container `smolvla-pilot-eval-tuning-120`: exit **0**, no OOM, from
  2026-09-16 21:43:33 UTC to 2026-09-17 04:24:39 UTC (about 6 h 41 min).
- Saved result complete, all 120 tuning trials, post-run fingerprints unchanged.
- Fresh preflight rechecked source/runtime, dataset, checkpoint, processors,
  local tokenizer assets and bank fingerprints against the baseline records.
- Independently replayed **all 120** saved action trajectories on CPU. All
  recorded pre/post 9-D states and terminal integration states matched to the
  audit tolerance; maximum observed 9-D error **0.0**.
- Reconstructed applied commands from raw predictions and verified exact match.
- Recomputed strict stack predicates for **every ordered pair at every tick**,
  including the required 30-tick stability. All requested-pair outcomes agree
  with the original scorer.
- Decoded all **141,121** frames across 120 two-camera videos. Dimensions and
  per-trial frame counts match action attempts plus one terminal frame.
  This is a full decode audit, not a claim of manually watching every frame.
- No new policy-driven rollout or optimizer update was executed in this review.
  Separate frozen-checkpoint offline probes used recorded held-out observations.
- Final evaluation scenes were not rolled out or used to choose changes.

Evidence: [baseline result](../outputs/evaluation/smolvla_pilot_tuning_20260916/result.json),
[full audit](../outputs/diagnostics/full_evaluation_audit_20260917/analysis.json),
[audit script](../outputs/diagnostics/full_evaluation_audit_20260917/inspect_results.py).
Generated artifacts remain local and are not committed.

## 2. Results by difficulty and failure stage

| Present cubes | Correct stacks | Rate |
|---|---:|---:|
| 2 | 4/40 | 10% |
| 3 | 0/40 | 0% |
| 4 | 0/40 | 0% |
| All | 4/120 | 3.33% |

Successful zero-based trial ordinals: **22, 50, 61, 90** (red->yellow,
green->yellow, blue->red, yellow->red). Each of those ordered pairs has 1/10;
the other eight pairs have 0/10. Ten trials/pair is too few to establish a
reliable ranking of pair difficulty. Likewise, cube-count results are suggestive
of distractor difficulty, not a controlled causal ablation of distractors.

Mutually exclusive outcome classification:

| Stage reached | Trials |
|---|---:|
| No bilateral contact with instructed pick cube | **110** |
| Target bilateral contact, but no target lift above 5 cm | **1** |
| Target contact and lift above 5 cm, no correct stable stack | **5** |
| Correct stable stack | **4** |

Contact and lift are diagnostic proxies, not a complete grasp-quality test.
First bilateral-contacted cube: requested target in **9**, another object in
**17**, no cube in **94**. A trial can contact a wrong cube first and later the
target, so these counts differ from the exclusive stage table.

Eight trials formed a wrong-pair stack stable for at least 30 ticks, correctly
rejected by the scorer: **2, 8, 35, 56, 81, 83, 97, 105**. This confirms
instructional object-role mistakes coexist with missed approaches/grasps.

18 trials had at least one clipped joint command. Gripper clipping changed
**zero** binary open/close decisions over all 120 traces. The baseline reports
0/120 intervention-free successes because its definition penalizes any clipping,
including numerically small gripper overshoots. That definition must remain
visible, but does not mean an external operator assisted the four successes.

## 3. Dataset inspection: what is and is not deficient

The train-only statistics, physical command semantics, labels, timing and replay
passed their checks. Earlier exact processor comparisons remain relevant:
[instruction audit](INSTRUCTION_AUDIT_20260917.md). Recomputing the same correct
normalization statistics is not an identified fix.

New measurements:

- All **300 training initial color-labeled cube layouts are distinct**. There
  is no exact same-layout/multiple-instruction contrast in the current set.
- All **300** have gripper command transitions at **frames 107 and 347**.
- Existing balancing supplies 25 demos/pair, but only **8 or 9** per
  pair/distractor-count condition. Balanced quotas do not imply dense spatial
  or interaction coverage.
- Training and tuning pick/place XY extrema span essentially the same configured
  workspace. There is no obvious gross workspace-range shift from this check.
- Nearest training-layout distance within the same pair and cube count has
  median **17.47 cm**, p90 **27.87 cm**, using RMS pick/place XY displacement.
  This ignores distractors and is descriptive only: it is not a required
  coverage radius or proof the dataset is too small.
- Accepted demonstrations are filtered by expert success; tuning states are not.
  Expert reachability on these tuning states is not yet established by the
  policy replay audit. Keep this alternative explanation open.

The fixed phase timing and unique layouts make state/phase-conditioned motion
continuation without strong language use a plausible shortcut. We have not
proved which cues the model relies on. Changing timing randomly without regard
to grasp state is not automatically a better expert.

Evidence: [dataset coverage audit](../outputs/diagnostics/full_evaluation_audit_20260917/dataset_coverage.json).

## 4. Offline validation had important blind spots

Original probes are at frames 60, 107 and 347. They did not evaluate the initial
decision from the reset state, complete autonomous approach or recovery after
the policy's own errors. Training itself **does include frames 0–59**; do not
mistake a validation-probe omission for a training-data exclusion.

The model's final first-ten arm MAE across the original probes is **0.008329 rad**.
A naive baseline that repeats current measured joint positions achieves
**0.010059 rad** on those same targets: the model reduces that error by only
about 17.2%. Full-50 comparison is stronger: hold-state **0.070816**
versus model **0.016913 rad**. The model learned useful prediction, but the
short-horizon metric alone was weak evidence of autonomous competence.

New frozen-model probes across all 120 held-out demonstrations:

| Observation frame | Full-50 arm MAE (rad) | First-ten arm MAE (rad) |
|---|---:|---:|
| 0 | **0.141925** | 0.009975 |
| 10 | 0.039970 | 0.010998 |
| 30 | 0.013977 | 0.012130 |
| 60 | 0.027896 | 0.011961 |
| 107 | 0.013460 | 0.007751 |
| 347 | 0.009383 | 0.005276 |

At initialization the longer intended trajectory is substantially less accurate;
short immediate actions can still match because early motion is shared. Since
execution uses ten actions before replanning, full-50 initialization error is
not by itself the cause of rollout failure. The contrast motivates measuring
target selection directly rather than relying only on averaged joint errors.

Complete-only 50-action chunks allow observation starts 0–478 of 528 frames.
The final 49 frames are absent **as starting observations**, but still appear
as target actions in preceding chunks. They are not proven recovery examples;
our successful scripted demonstrations do not teach broad recovery from policy
mistakes. Do not remove padding safeguards without fixing and testing the
upstream temporal mask convention.

Evidence: [phase probe results](../outputs/diagnostics/full_evaluation_audit_20260917/phase_probe_results.json).
Checkpoint hash was unchanged. This is a single checkpoint/seed-method diagnosis,
not a multi-seed model comparison. Old BF16-update-loss observations do not apply
as an unfixed defect: this completed run already used verified FP32 trainable
parameters and optimizer moments.

## 5. Answer: new data, new parameters, or something else?

### First recommendation: something else — a bounded discriminating experiment

Run the proposed **36** comparisons: 12 preselected two-cube tuning layouts,
original instruction/seed, reversed instruction/same seed, original instruction/
alternate seed. Select without conditioning on success. Preserve the baseline;
verify original-condition reproduction and measure actual object selection.

Optionally add **12** training-scene reproductions, one predefined training
scene per ordered pair, as a separately stated 48-rollout total budget. This
distinguishes whether the trained policy can reproduce demonstrations it saw
from whether it transfers to novel layouts. Neither block changes weights.
If scene feasibility remains in doubt, compare the scripted expert on the
same selected layouts under a separately approved bounded expert test.

There is a stopping rule: after this diagnostic block, choose one intervention
or acknowledge remaining uncertainty; do not run an indefinite series of tiny
experiments or claim a cause from one favorable video.

### New data: potentially useful, but change supervision rather than only counts

If the matched tests confirm poor language-conditioned object selection, prefer
an **additional versioned contrastive demonstration set**: multiple instructions
from the exact same initial layout, with reset state and visual setup held fixed.
Two-cube role reversal is the simplest case; later add target/destination choices
with distractors. Keep the entire layout family in one split to avoid leakage.

Add contact-aware timing and expert recovery demonstrations only when the
failure analysis supports them. Preserve the original 300 as a nested baseline,
fit statistics on the new training subset only, retain independent evaluation
and never relabel failed trajectories as successful expert actions.

If training-scene reproduction is poor, collecting another large similar set
may be premature. If reproduction succeeds but novel layouts fail, targeted
spatial/distractor coverage becomes more justified. No fixed new dataset size
is currently proven sufficient.

### New training: controlled ablations, not arbitrary parameter sweeps

Candidate interventions depend on the diagnostic outcome:

- Failure to fit even training scenes: inspect observation/action fidelity and
  compare a longer optimization budget at fixed data/recipe with closed-loop
  tuning checks. 4.45 passes may be insufficient, but low success alone does
  not prove that more epochs are the solution.
- Weak instruction grounding: test early-decision/phase-balanced sampling or
  target-role auxiliary supervision; if representations are limiting, test
  bounded adaptation of relevant VLM layers rather than assuming frozen VLM
  is inherently wrong. Architecture/compute compatibility needs its own check.
- Correct object choice with grasp/placement misses: then test execution horizon,
  feedback or recovery coverage. Earlier single-scene horizon-1 failure does
  not establish the best horizon for this checkpoint, but horizon tuning alone
  cannot be assumed to fix wrong roles.

Change one factor at a time. Keep an untouched baseline checkpoint, explicit
sampling/exposure budget and identical independent tuning trials. Checkpoints
should be assessed with closed-loop task/selection metrics, not selected solely
by the old three-phase action MAE. No new training is launched by this review.

## 6. Practical route versus raw-image benchmark route

If the priority becomes a reliable task-specific robot rather than demonstrating
raw-image VLA grounding, an explicit camera-based color/object detector + language
role selector + geometric controller is a reasonable **separate system**. It
uses additional perception assumptions and must be reported separately; perfect
simulator target poses are only an oracle diagnostic. See the
[input options note](POLICY_INPUTS_AND_GROUNDING_OPTIONS.md).

For the current shared SmolVLA/pi benchmark, keep raw inputs and success rules
unchanged until a versioned experiment is approved. No route guarantees 100%
success; estimate performance through held-out trials and report limitations.