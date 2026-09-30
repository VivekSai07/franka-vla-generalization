# B2 distractor evaluation

Fork of `experiments/xvla_b1_language_eval/` — repointed at the B2 training run
(`xvla_franka_stacking_b2_distractors_v1_20260928`), scored on the **condition** axis (control
vs distractor-present) instead of instruction.

## Protocol

- **Screening**: 20 layouts (`evaluation_screen_ids`), distractor condition only, run against
  each of the 5 non-trivial checkpoints (`16700, 25000, 33400, 41700, 50000`) = 100 trials. Picks
  a checkpoint on capability alone, before any contrast is measured.
- **Report**: 40 layouts (`evaluation_report_ids`), **both conditions** at every layout (2 seeds
  each) = 160 trials, on the winning checkpoint only. Same layout + same weights + same seed pair
  across conditions makes the contrast within-layout, per the design spec.

## Pick/instruction assignment

The sealed collection plan fixes each reserved layout's **arrangement** (familiar/swapped) but
never assigns a pick colour or instruction there — that axis is eval-only background for B2 (design
spec Non-goals), never part of the frozen/sealed plan. `eval_picks()` derives it here,
deterministically: balanced 15/15/15/15 across the four (pick × arrangement) cells, stratified
within each arrangement so the balance is exact. This gives the same red/green × familiar/swapped
coverage collection itself used, without touching the sealed plan/bank/provenance.

## Usage

`experiments.xvla_b2_distractors_eval.evaluate` accepts `--mode` (`smoke`/`screen`/`report`) and
`--checkpoint` (one of `CANDIDATES`). Run in order: `smoke` (4 trials, both conditions, one
screen + one report layout) → `screen` × 5 checkpoints → pick the winner by success count →
`report` on the winner. See `/home/utxowqr/b2_eval_chain.sh`, which gates each stage on the
previous one succeeding.

Outputs land under the training run's own directory:
`outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928/evaluation_<mode>_step_<NNNNNN>/`.
