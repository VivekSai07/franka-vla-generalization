---
name: evaluating-a-checkpoint
description: "Use when running or designing a rollout evaluation — screening vs report tiers, holdout independence, checkpoint selection, and reporting statistics across reserved layouts."
---

# Evaluating a checkpoint

Evaluation packages fork a training package's eval sibling (`<pkg>_eval`), sharing the rollout
body, gates, and scene-restore/geometry proof, but pinned to their own reserved layouts.
`evaluate.py` exposes `run(output, mode=..., checkpoint=...)`; modes are commonly
`('smoke', 'screen', 'report')`.

## 1. `rc=0` is not evidence a rollout ran — verify wall-clock time and output files too

A forked `evaluate.py` missing its `if __name__ == '__main__':` guard imports cleanly, defines
every function, and exits 0 having called nothing — no model load, no rollout, no output
directory. A launch chain that gates stage N+1 only on stage N's exit code "passed" five
20-trial screening stages in under 30 seconds total before failing at the very end with no
result files to read. Before trusting any `docker run ... evaluate ...` stage: check the elapsed
wall-clock time is plausible for a real GPU rollout (tens of seconds per trial, not per stage),
and check the expected `evaluation_<mode>_step_<N>/result.json` actually exists.

## 2. Never select a checkpoint from offline metrics

Loss, action MAE, and gripper accuracy have all improved monotonically through a checkpoint that
scored **0/72** on rollouts, then recovered to 86.1% at a later checkpoint — a total capability
collapse invisible in every offline number. Only rollouts count. Do not stop early, pick a
checkpoint, or report progress from training-loop metrics.

## 3. Screening vs report are different holdout sets

- **Screening** runs a small held-out layout set against every non-trivial checkpoint to choose
  which one to report on — this measures the axis whose difficulty is uncertain, not the final
  number.
- **Report** runs a disjoint, larger held-out set — never shared with screening's or another
  package's holdout — at the winning checkpoint(s).
- If two conditions are compared (e.g. control vs distractor-present), evaluate **both at every
  report layout** (within-layout contrast) rather than splitting layouts across conditions — this
  is what makes a contrast diagnostic instead of confounded by layout difficulty.

## 4. A holdout failure is not automatically a policy failure

Reserved layouts are re-derived via `derive_scene` and proven bit-exact against the sealed
`bank.json`, but they were never expert-verified end to end. If a checkpoint fails a holdout
layout, screen it with the scripted expert before attributing the failure to the policy — 8
"unsolvable" B1 layouts all turned out solvable, which is what made the reported success rate
trustworthy.

## 5. Background factors the sealed plan never assigned still need a decision

If a reserved layout fixes one factor (e.g. arrangement) but leaves another (e.g. pick colour)
unassigned because it's eval-only background, derive it in the eval package deterministically —
balanced across the population, never touching the sealed plan/bank/provenance to add it. Confirm
the intended coverage with the user before spending GPU time: "background, not a reported factor"
still needs to decide *what* the background looks like across trials.

## 6. Scene restoration

Do not restore F7-envelope scenes with a pilot-era `restore_scene` — it caps `|y| <= 0.40`, which
rejects roughly a third of an F7 holdout (precisely the widest, most discriminative layouts). Use
the collector's own geometry gate (`_validate_broad_geometry` in the package's `scenes.py`).

## 7. Report the statistic that matches the sampling unit

Report **geometry-clustered intervals**, not a naive binomial — a spread pooled across layouts
describes the layouts, not the sampling variability of the policy. Before calling any spread
"tight" or "consistent", check how many independent layouts it was computed over; a handful of
trials from a handful of layouts describes between-layout variation, not measurement precision.

## See also

[training-a-policy](../training-a-policy/SKILL.md) for checkpoint schedule design,
[launching-a-run](../launching-a-run/SKILL.md) for the container/gate mechanics shared with
training.
