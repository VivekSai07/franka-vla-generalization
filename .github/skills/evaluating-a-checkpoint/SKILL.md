---
name: evaluating-a-checkpoint
description: "Use when running or designing a rollout evaluation — screening vs report tiers, holdout independence, checkpoint selection, and reporting statistics across reserved layouts."
---

# Evaluating a checkpoint

Evaluation packages fork a training package's eval sibling (`<pkg>_eval`), sharing the rollout
body, gates, and scene-restore/geometry proof, but pinned to their own reserved layouts.
`evaluate.py` exposes `run(output, mode=..., checkpoint=...)`; modes are commonly
`('smoke', 'screen', 'report')`.

## 1. Never select a checkpoint from offline metrics

Loss, action MAE, and gripper accuracy have all improved monotonically through a checkpoint that
scored **0/72** on rollouts, then recovered to 86.1% at a later checkpoint — a total capability
collapse invisible in every offline number. Only rollouts count. Do not stop early, pick a
checkpoint, or report progress from training-loop metrics.

## 2. Screening vs report are different holdout sets

- **Screening** runs a small held-out layout set against every non-trivial checkpoint to choose
  which one to report on — this measures the axis whose difficulty is uncertain, not the final
  number.
- **Report** runs a disjoint, larger held-out set — never shared with screening's or another
  package's holdout — at the winning checkpoint(s).
- If two conditions are compared (e.g. control vs distractor-present), evaluate **both at every
  report layout** (within-layout contrast) rather than splitting layouts across conditions — this
  is what makes a contrast diagnostic instead of confounded by layout difficulty.

## 3. A holdout failure is not automatically a policy failure

Reserved layouts are re-derived via `derive_scene` and proven bit-exact against the sealed
`bank.json`, but they were never expert-verified end to end. If a checkpoint fails a holdout
layout, screen it with the scripted expert before attributing the failure to the policy — 8
"unsolvable" B1 layouts all turned out solvable, which is what made the reported success rate
trustworthy.

## 4. Scene restoration

Do not restore F7-envelope scenes with a pilot-era `restore_scene` — it caps `|y| <= 0.40`, which
rejects roughly a third of an F7 holdout (precisely the widest, most discriminative layouts). Use
the collector's own geometry gate (`_validate_broad_geometry` in the package's `scenes.py`).

## 5. Report the statistic that matches the sampling unit

Report **geometry-clustered intervals**, not a naive binomial — a spread pooled across layouts
describes the layouts, not the sampling variability of the policy. Before calling any spread
"tight" or "consistent", check how many independent layouts it was computed over; a handful of
trials from a handful of layouts describes between-layout variation, not measurement precision.

## See also

[training-a-policy](../training-a-policy/SKILL.md) for checkpoint schedule design,
[launching-a-run](../launching-a-run/SKILL.md) for the container/gate mechanics shared with
training.
