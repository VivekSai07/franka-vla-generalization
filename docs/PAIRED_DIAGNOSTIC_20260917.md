# Paired-instruction and training-scene diagnostic — 2026-09-17

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