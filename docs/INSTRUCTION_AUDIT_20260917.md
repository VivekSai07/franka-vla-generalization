# Instruction versus rollout audit — 2026-09-17

## Scope and conclusion

User reported videos contradicting instructions during the ongoing 120-scene
tuning evaluation. Investigated saved artifacts, rendered colors, task labels,
processors, exact saved-action replay and fixed-observation instruction swaps.
No optimizer updates, new policy-driven trials, changes to the active evaluator,
changed success thresholds, checkpoint edits or final-scene evaluation.

**Confirmed: the policy sometimes builds a stack using the wrong ordered color
pair. These inspected cases are scored as failures, not successes.** No label,
color-ID, processor-normalization or success-scoring defect was demonstrated by
these checks. The model-level cause of poor instruction following remains a
hypothesis, not a proven implementation bug or a verified fix.

## Snapshot: first 41 completed trials

- Correct instructed successes: **1/41**, trial 022; intervention-free successes
  at the earlier 40-trial result snapshot: **0/40**.
- Target bilateral contact observed in **3/41**; instructed target lift above
  5 cm in **3/41**. Do not extrapolate these partial, pair-ordered results to
  the full benchmark. Most failures begin at target selection/approach/grasp,
  not merely at final placement.
- Three wrong-pair stacks detected at the terminal state:

| Trial directory | Instruction | Terminal stack | Recorded outcome |
|---|---|---|---|
| trial_002 | red on green | green on red | timeout / failure |
| trial_008 | red on green | green on yellow | timeout / failure |
| trial_035 | green on red | yellow on red | timeout / failure |
| trial_022 | red on yellow | red on yellow | success |

Indices above are zero-based directory ordinals; console trial numbering is
one-based (trial_022 is console trial 23). The videos contain front and wrist
views of the **same trial**, not two separate executions.

Evidence: [all completed-trial audit](../outputs/diagnostics/instruction_audit_20260917/completed_trial_audit.json),
[wrong-pair terminal frames](../outputs/diagnostics/instruction_audit_20260917/wrong_pair_terminal_frames.jpg),
[correct success sequence](../outputs/diagnostics/instruction_audit_20260917/trial_022_contact.jpg).
Wrong-pair detection here checks terminal geometry/contact; it does not claim
every wrong stack was stable for 30 ticks or exhaust all earlier transient stacks.

## Checks and measured outcomes

### Color names and training language labels

Runtime model is the recorded MuJoCo scene. Cube bodies and geometry RGBA agree:
red (0.8,0.2,0.2), green (0.2,0.8,0.2), blue (0.2,0.2,0.8), yellow (0.8,0.8,0.2).
There are no material overrides on those cube geometries. Actual decoded video
frames visually agree with those colors.

All **221,760 Parquet rows** resolve through the task-index mapping to their
episode manifest instruction, with **zero mismatches**. All 12 tokenized task
strings retain both colors and their order at the configured length of 48.
Actual model-processed instruction token sequences in the counterfactual tests
were distinct. No evidence of truncated colors or one shared instruction.

### Saved processors and gripper clipping

Every saved state/action statistic in the checkpoint's normalizer and
unnormalizer exactly matches the saved train-only statistics. Saved-versus-fresh
postprocessors produced **zero maximum difference** on a test tensor spanning
normalized values -3 to 3. Unbounded model outputs do not by themselves imply a
normalization bug.

Across the 41 completed traces, clipping the gripper into [0,255] changed
**zero binary decisions** relative to directly thresholding the raw output at
127. It affects the intervention-free metric, but is not evidence that the
gripper received the opposite open/close command.

Trial 034 raw gripper range was approximately -19.46 to 255.57, with 87 open
ticks and 1,120 clipped ticks. That alone does not explain the wrong approach.

### Independent saved-action replay and scoring

Replayed every saved command for trials 000, 022 and 034. At every tick the
measured 9-D state matched the recorded trace exactly; the complete terminal
integration state also matched exactly. Evaluated every ordered pair among the
present cubes with the strict stack predicate:

- Trial 022: red grasped bilaterally for 379 ticks, lifted 20.51 cm; correct
  red-on-yellow predicate held for 30 consecutive ticks. Reverse pair streak 0.
- Trial 000: correct red target grasped for 259 ticks and lifted 20.02 cm, but
  neither ordered pair achieved the strict stack predicate. Video shows a
  placement miss rather than incorrect pick identity.
- Trial 034: no cube had bilateral contact or meaningful lift. Minimum gripper
  midpoint distances: green **19.90 cm**, red **26.86 cm**, blue **12.09 cm**.
  This is an empty/missed grasp toward the wrong area, not a completed blue
  grasp. Green-on-red and all other strict pair streaks were zero.

Evidence: [physical replay audit](../outputs/diagnostics/instruction_audit_20260917/physical_replay_audit.json),
[trial 034 sequence](../outputs/diagnostics/instruction_audit_20260917/trial_034_contact.jpg),
[trial 000 sequence](../outputs/diagnostics/instruction_audit_20260917/trial_000_contact.jpg).

### Controlled instruction swaps

Reconstructed exact live observations at ticks 0 and 90 for trials 022 and 034
by replaying saved actions, then queried the frozen 020000 checkpoint with only
the instruction changed and identical sampling noise. No new policy-controlled
trajectory was executed. Using each original trial's seed reproduced its saved
first ten actions **exactly**, tying the diagnostic to the actual running path.

At tick 90:

- Trial 022, reversing red-on-yellow to yellow-on-red changed the first-ten
  arm predictions by mean **0.000996 rad**. Changing only the sampling seed
  for the correct instruction changed them by **0.005881 rad**.
- Trial 034, five alternative instructions changed first-ten arm predictions
  by **0.001224–0.003830 rad**; the one different-noise control was **0.004175 rad**.
  All alternatives still produced a closed-gripper first action.

These are evidence of **weak immediate instruction sensitivity on the sampled
states**, not proof that language is entirely ignored. At tick 0, some full
50-action chunks changed substantially (up to 0.1024 rad mean arm difference).
Early actions can be shared across tasks, and changing instructions mid-approach
creates counterfactual states outside the demonstrations. One alternate noise
draw is a control, not a statistical significance test.

Evidence: [instruction-swap results](../outputs/diagnostics/instruction_audit_20260917/instruction_counterfactual.json),
with full prediction arrays stored beside it. Model SHA256 remained unchanged.

## Interpretation and next decision

The evidence supports weak/reliably incorrect object-role selection alongside
missed grasps and placement failures. A plausible mechanism is learning local
motion continuation from measured robot state and nearby visual objects without
robust use of the instructed pick/place colors. **That mechanism is not yet
causally established.** Low offline action error on expert states did not
establish robust language grounding or closed-loop recovery.

Leave the authorized 120-scene baseline unchanged; no demonstrated evaluator
bug justifies invalidating it. After completion, a bounded paired-instruction
experiment from identical initial tuning scenes and common seeds would test
whether changing the requested pair changes the selected object. Include a
same-instruction seed control and an expert reachability/control comparison
before choosing retraining, more data, an objective change or a horizon change.
Obtain approval for new policy-driven experiments; leave the final bank sealed.

Do not loosen success criteria to count any stack, relabel demonstrations,
assume clipping caused the failures, or promise extra epochs will repair this.