---
name: training-a-policy
description: "Use when writing or reviewing a training contract/runner, sizing a checkpoint schedule or disk guard, or comparing two training runs — covers constant duplication, disk-guard sizing, and matching runs for a fair comparison."
---

# Training a policy

`contract.py` declares what a run is bound to and its recipe (horizon, checkpoint schedule);
`runner.py` executes it. This skill covers designing/reviewing that pair. For container flags and
the `--check`/`--preflight`/`--train` gate sequence, see [launching-a-run](../launching-a-run/SKILL.md).

## 1. The contract is the only source of truth

Every number the runner needs that also appears in `recipe()` must be **derived**, not re-typed:

```python
maximum = 2 if preflight else c.recipe(False)['max_optimizer_updates']
```

A runner that hardcoded its own horizon separately from the contract trained to the wrong number
of updates and only failed afterwards, at `verify_checkpoints` — after the GPU time was spent. Add
a regression test asserting the runner contains no hardcoded fallback for this value, and confirm
it fails against the unfixed file (`git stash`).

## 2. Size the disk guard to this package's own schedule

A disk guard sized for a 9-checkpoint schedule (inherited from a fork's contract) rejected a
5-checkpoint package's genuinely-sufficient disk. Measure this package's actual checkpoint sizes
and set the guard from that, not from the source package's number.

## 3. Preflight checkpoints are evidence, not scratch

`train()` re-audits the preflight checkpoints as a precondition (`validate_scratch` →
`verify_checkpoints(report, SCRATCH, True)`). Deleting `<run>_preflight/checkpoints` to free disk
after preflight passes makes training fail immediately with `FileNotFoundError`. Grep for a path
before deleting anything a stage wrote — a later stage may re-verify it.

## 4. Comparing two runs

- Hold **exposure (epochs)** constant across runs being compared, not update count — the same
  update count meant a 3.3x difference in exposure once the dataset size changed, and nearly
  produced a false conclusion.
- To compare runs mid-training, match **schedule phase** (fraction annealed), not raw step count —
  the same step number was 80% annealed in a short run and 24% annealed in a long one; matching
  phase instead reversed an "indistinguishable" result into a decisive one.
- Change **one variable at a time** — a failure with two new axes (e.g. a data axis and an
  environment change) in the same run is not attributable to either.

## 5. Don't bundle a speculative change with a necessary fix

Bundling a correct fix (disk guard) with a speculative one (deleting "obviously temporary" output)
made one failure look like two, and the speculative half was the one that broke training. Land
them separately, in order of confidence.

## See also

[evaluating-a-checkpoint](../evaluating-a-checkpoint/SKILL.md) — no offline metric here predicts
rollout success; the only way to know whether training is progressing is to run rollouts.
