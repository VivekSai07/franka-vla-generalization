# gen_g1_compare — G1 against A2 (outside the frozen cores)

Arm A1 (X-VLA trained on G1's randomized data, `experiments/vla_inputs_arm_xvla_g1/`) read against A2 on G0's benchmark tier, plus the
secondary nominal comparison on M0's comparison tier. Design: [G1 spec](../../docs/superpowers/specs/2026-11-06-g1-randomized-data-design.md)
(D9, D10, R4). Nothing here is hash-bound: `gen_eval` (core `512806282b1f…`) and `vla_inputs_eval` (core `c4f6c303f2ae…`) are imported
read-only, and both cores are pinned in `compare.py`.

## What it computes

`python -m experiments.gen_g1_compare.compare g0 --root <G0 campaign> --checkpoint <update>`: per axis 1-5, A1 against A2 (88,830) on the
axis's perturbed trials paired by layout, exact McNemar of the discordant pairs, **Holm over the five axes**, alpha 0.05. Criterion per axis:
A1 **holds** the axis (G0 D13: A1's own paired drop, nominal minus perturbed, has a 95% bootstrap upper bound of at most 10 points) **and**
beats A2 (more successes on the perturbed trials, Holm p below 0.05). Levels are descriptive and unadjusted. Nominal is reported with the
paired gain and interval; A1 more than 5 points below A2 is flagged `cost`.

`... m0 --root <M0 campaign> --g0-root <G0 campaign> --checkpoint <winner>`: A1 at G0's selection winner against A2 (88,830) on M0's
comparison tier (573 layouts), M0's own `mcnemar` and `paired_effect`; secondary.

## What it refuses

Both arms must carry the pinned core and the same plan, feasibility and dropped trials; both closures, the core and the manifests must
still verify (deep hashes); A1's checkpoint must be the selection winner (`arms/A1/selection.json`) or 101,520 (G0) / the winner (M0);
every planned trial of both stages must be present and identical in key. The benchmark tier never chooses a checkpoint.

## Driver and chain

`scripts/gen_g1_eval.sh` is a fork of `scripts/gen_eval.sh` for arm A1 only (the original names A0/A2/A3 and its tests, in the frozen
`gen_eval` package, refuse any other arm). `scripts/gen_g1_chain.sh` runs the whole evaluation detached and resumable: prepare (six
candidates 50,760 ... 126,900), smoke, selection tier, select, benchmark tier at the winner and at 101,520, compare, then M0's comparison.

## Tests (`tests/`)

| file | what it proves |
|---|---|
| `test_compare.py` | hand-built counts for the statistics, held / beats / criterion, Holm, direction, the nominal cost flag, every refusal, the files written (verification calls mocked) |
| `test_driver.py` | the driver's docker commands (dry run) and its polling, collecting, skipping and resuming against a fake `docker` |
| `test_chain.py` | order, skipping, resuming, stop at the first failure, the GPU and disk guards, the winner deciding the second benchmark (fake driver) |

Caveat that goes with every figure: A1 saw 0.89 epoch of its data (126,900 updates at effective batch 8 over 1,142,100 starts), A2 5.0 epochs
of P2's. Only the data differs, not the recipe.
