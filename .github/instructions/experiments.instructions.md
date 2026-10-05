---
applyTo: "experiments/**"
description: "Use when editing experiment packages: contracts, runners, SOURCE_FILES, provenance, seals, checkpoint schedules or evaluation protocols."
---

# Experiment packages

Each package is self-contained: `contract.py` declares what is bound and verified, `runner.py`
executes, `tests/` covers both. Evaluation lives in a separate `*_eval` package.

## Hash-bound surfaces

Before editing anything here, determine whether it is bound:

- **`SOURCE_FILES`** in a contract — an explicit tuple. It includes the package's **own README**.
- **`provenance()`** in a dataset package — `rglob('*.py')` over its own directory, **plus its
  README explicitly** (`paths += [Path(__file__).with_name('README.md')]`), plus the base source
  module's provenance.

Editing a bound file:
- **while a run is live** → kills it at its final `admission()` re-check
- **after `--check`** → invalidates the admission, forcing a ~1h re-run

Currently frozen and uncorrectable (editing would invalidate sealed runs):
`experiments/broad_workspace_data/README.md`, `experiments/{xvla,smolvla}_local_varied/README.md`.

Hash-bound by the B3 seal (2026-10-03): `experiments/b3_robosuite_data/` (a real `provenance()`:
every `.py` incl. tests, its README) and, through it, `experiments/robosuite_scene/*.py`, the shared
modules it imports, `src/`, and the B3 spec and collection plan. Editing any of them invalidates
the seal; fork instead. Its `plan.py` and `contract.py` import no simulator so training can import
them in the admission image.

`experiments/xvla_b3_robosuite/` (B3 training) is bound by its own `SOURCE_FILES`, like every
training package; the B3 evaluation imports that contract, so edit neither.

Not hash-bound: `experiments/robosuite_sim2sim/` (sim2sim rollouts), `experiments/xvla_b3_robosuite_eval/`
(B3 evaluation) and the README of `experiments/robosuite_scene/` (scene bring-up). No contract lists them; the bring-up `seal.json` is
a file fingerprint only. `robosuite_scene` runs only in the robosuite image
(`contract.IMAGE`, MuJoCo 3.4.0) — except `test_r1_image`, `test_seal` and `test_expert_fork`, which
run in the admission image — and its tests are run by module name, not by `discover`. It imports
sealed modules read-only; `test_r1_image` fails if a sealed path changes on that branch.

## Shared helpers

`experiments/xvla_single_task/contract.py` is bound by **four** contracts. It hardcodes
`learning_rate` to `[1,5000]` and `scheduler_decay_steps=5000`. Never edit it to accommodate one
package — define a local override in the new package instead, and verify it reproduces the shared
schedule where they overlap.

The same reasoning applies to anything under `data_collection/`, `validation/`, `training/`,
`src/` and `experiments/fixed_layout/`: assume multiple sealed runs depend on it.

## Contract / runner duplication

Constants must have one home. `recipe()` declaring 16700 updates while `runner.py` held
`maximum = 2 if preflight else 5000` truncated a run at 5000 and only failed afterwards, wasting
70 minutes. Derive from the contract:

```python
maximum = 2 if preflight else c.recipe(False)['max_optimizer_updates']
```

When adding a package by copying another, grep the copy for every literal that also lives in
`recipe()` — horizon, checkpoint schedule, output paths, purpose strings — and for the old
package name in module imports and test paths. **The copied tests are the trap**: they keep
importing the source module and patching its symbols, so they pass while testing the wrong code.
Derive constants from the contract (`recipe(False)['max_optimizer_updates']`) rather than copying
the number, and assert agreement with `checkpoint_updates(False)[-1]`.

## Tests

Run with the tests directory as top-level dir (`-t experiments/<pkg>/tests`) and `PYTHONPATH=/vla`;
they import siblings directly and have no `__init__.py`.

Contract tests exercise `recipe()`; runner tests use synthetic fixtures. **Neither automatically
covers the code path that actually runs.** When fixing a bug, add a test that reads the real
source or the real wiring, and confirm it fails before the fix.

## Datasets

Collection packages are append-only once sealed. The seal covers data files; `provenance()` covers
source. Both must verify before training admits.

If a layout is uncompletable or irreproducible, use the deterministic resample mechanism
(`INFEASIBLE_LAYOUT_RESAMPLES`) rather than loosening a tolerance — and prefer resampling layouts
that fall in the validation split so training data is untouched.

## Evaluation

Reserved evaluation layouts are never executed during collection. Scenes are re-derived via
`derive_scene` and proven bit-exact against the sealed `bank.json` before use.

Do not restore F7 scenes with `validation.evaluate_pilot.restore_scene` — it applies pilot-era
workspace bounds (|y| ≤ 0.40) that reject a third of the envelope-F7 holdout. Use the collector's
own geometry gate.
