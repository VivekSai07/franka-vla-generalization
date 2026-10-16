# Props P2 — training evaluation Implementation Plan

> Executed inline in the session that wrote it. Rules below are fixed **before the first rollout** and
> are not changed after any result is seen. They operationalize the P2 spec
> ([specs/2026-10-13-props-p2-distractors-design.md](../specs/2026-10-13-props-p2-distractors-design.md),
> sealed: not edited) and follow the P1 evaluation ([2026-10-07-props-p1-training-eval.md](2026-10-07-props-p1-training-eval.md)).

**Goal:** Evaluate the P2 checkpoints on the 120 reserved layouts, compare against P1's checkpoint 126,900 on
the same trials, and report the four pre-registered criteria, as fast as the machine allows.

**Architecture:** `experiments/xvla_props_p2_eval/` forked from `xvla_props_p1_eval/` (not bound by any seal; it
imports `props_p2_data` read-only). Rollouts are independent trials, so a stage is run by 8 parallel worker
containers (one policy copy each, ~4.3 GB GPU, ~5 CPU cores) taking interleaved jobs; per-trial folders are
written to one directory per (mode, policy, checkpoint); `--collect` merges them into `result.json`.
Checkpoints and the sealed parent are verified once (`--prepare`), not per worker.

## Fixed rules

- **Tiers (reserved layouts, never collected):** screen 20 (all 6 candidates, 1 seed) · report 60 layouts × 2
  seeds = 120 trials (20 trained × 1 distractor, 20 trained × 2, 20 held-out; 1/2 alternating) · control 20
  (0 distractors) · confirm 20. Candidates: 50,760 · 76,140 · 88,830 · 101,520 · 114,210 · 126,900.
- **Order:** expert feasibility of all 120 layouts (unrendered) → classifier validation → smoke → screen →
  report (survivors, and P1 126,900 on the same 120 trials) → control (winner and P1) → confirm (winner, once).
- **Excluded layouts:** any layout the scripted expert cannot solve is dropped from every read-out and counted.
- **Screen survivors:** every candidate within **2 successes** of the best screen count goes to the report.
- **Report winner:** the highest report-tier overall success count; **ties go to the later checkpoint**. Confirm
  runs once on the winner. The four criteria are computed on the winner's report tier (the spec's tier); the
  confirm and control results are reported beside them, and the report number carries the selection caveat
  (the report both selects and measures, B3 lesson), which is what the confirm tier is for.
- **Success rule, 900-tick cap, seeds, instructions:** as P1 (`props_scene.metrics`, 30 stable ticks); the same
  trial list and rollout seeds are used for P1's policy, so the comparison is paired.

## Failure classifier (written and validated before any rollout)

An object counts as **lifted** when its centre rises ≥ **2 cm** above its start and stays up **15 ticks**
(the spec's 2 cm). Measured on the 540 sealed expert episodes (`outputs/p2_speed_probe/rise_probe.py`): an
untouched distractor rises ≤ 0.02 cm (lemon 0.00), a carried target 18.3–20.8 cm, so the threshold is far from
both. First cause, mutually exclusive, in this order: `success` · `never_lifted` (no object lifted) ·
`wrong_object` (the first object lifted is a distractor) · then P1's: `wrong_bin` · `unstable_in_target` ·
`timeout_carrying` · `dropped_outside`. Also recorded per trial: `first_lifted`, and the maximum displacement
of each distractor. The **wrong-object rate** is the share of trials whose first lifted object is a
distractor, whatever the outcome.

## The four criteria (report tier, trials with 1–2 distractors, winner)

1. overall success ≥ **70%**;
2. wrong-object rate ≤ **5%**;
3. |trained-pair rate − held-out-pair rate| ≤ **10 points**;
4. P2 rate > P1's on the same trials with **non-overlapping layout-clustered 95% intervals**
   (10,000-resample bootstrap over layouts, seed 0). Paired exact McNemar p is reported with it.

All four are required. Also reported, not gated: success by distractor count (0 / 1 / 2), by bin arrangement
(`blue_slot`, the physical side), wrong-bin rate, causes, and the control tier for P2 and P1.

## Tasks

- [ ] 1. Package fork + rewritten tests (`protocol`, `scenes`, `analysis`, `rollout`, `feasibility`, `evaluate`).
- [ ] 2. Feasibility of 120 layouts; classifier validated on the 540 sealed expert traces (all first-lifted =
  target; 0 distractor lifts) and on constructed wrong-object traces.
- [ ] 3. `--prepare`, smoke (P2 and P1), measure workers' GPU/CPU, then the stages above via
  `scripts/xvla_props_p2_eval.sh`.
- [ ] 4. Report `docs/PROPS_P2_EVALUATION_<date>.md`, docs pass, FAILURES if anything recurs.
