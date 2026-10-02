---
name: diagnosing-policy-failures
description: "Use when a finished evaluation has failed rollouts to explain, when a result looks odd, or before stating a cause for a failure pattern. Covers how to separate layout, feasibility, interference and mechanism, and how not to fool yourself."
---

# Diagnosing policy failures

The evaluation gives a success count and, for every non-success, the label `timeout`. **That label
is not a cause**; it only means no stable stack within 1,200 ticks. The cause has to be recovered
from the traces (`trace.npz`, `episode.json`, `cameras.mp4`). Worked example:
`docs/B2_DISTRACTOR_EVALUATION_20260930.md`, scripts in
`experiments/xvla_b2_distractors_eval/analysis/`.

## Order of questions

1. **Random or layout-specific?** Layout is the independent unit. Permute the failure labels over
   trials and compare a concentration statistic (sum of squared per-layout failure counts). B2:
   five layouts held all 13 failures at 50,000 (p = 0.0002). Check which layouts fail at *every*
   checkpoint.
2. **Feasible?** Reserved layouts were never expert-verified (FAILURES #14). Run the scripted
   expert on every failing layout, same jitter and scene arrays. B2: 23/23, so nothing was
   unreachable. Do this before any claim about the policy.
3. **Interference?** Measure displacement of every non-target cube and of the base cube. Give the
   measure a positive control (the pick cube must register motion on a success).
4. **Mechanism.** Classify each failure by *first cause*, mutually exclusive, from the trace:
   acquisition (never lifted), hesitation (stall before lift, or near-miss on the clock), release
   (offset at the last release after a carry), wrong cube (which cube was lifted first). Use lift
   and release events, **not** `first_bilateral_contact_tick`, which failed grasps also satisfy.
5. **Decision-point quantity vs outcome.** For the dominant mechanism, plot success against the
   continuous quantity at the moment of decision (B2: release offset: 98% at < 1.5 cm, 22% at
   2.0–2.5, 0% beyond 2.5; cube edge 4.2 cm so the tipping limit is 2.1 cm). A cliff at a physical
   limit means the failure is real, not a scoring artefact.
6. **Compare with the expert** on the same quantity from the sealed dataset. A fatter policy tail
   than the expert's says what the demonstrations did and did not teach.

## Rules that would have saved time

- **Validate every diagnostic on the successes first.** A measure identical for successes and
  failures is noise. Three B2 measures failed this (FAILURES #38).
- **Read the raw trace or video of the odd cases** before finalising a taxonomy. Two "impossible"
  numbers each exposed a class the rules missed.
- **Compare checkpoints paired**: the same trial ordinal shares layout, condition and rollout seed
  (McNemar on the discordant pairs).
- **Count your tests** and mark anything not backed by mechanism or physics as exploratory.
  B2 ran about thirty uncorrected tests; the arrangement effect (p = 0.003) does not survive that.
- **Do hand-frame geometry** for grasp questions (cube centre in the gripper's own frame at first
  closure); world-frame distances conflate the arm pose with the grasp.
- **Retract wrong interim claims in the written record**, where they were made.

## Outputs

A dated report (`docs/<TOPIC>_<YYYYMMDD>.md`) with the headline, the mechanism table, what was
ruled out and why, a section on checks that did not hold up, limitations, and directions labelled
as hypotheses. Also update `RESULTS.md`, `BENCHMARK_STATUS.md`, `FAILURES.md` and `TODO.md` in the
same pass.

## See also

[evaluating-a-checkpoint](../evaluating-a-checkpoint/SKILL.md) for designing the protocol this
diagnoses, [collecting-a-dataset](../collecting-a-dataset/SKILL.md) if the cause is in the
demonstrations.
