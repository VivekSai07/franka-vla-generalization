# X-VLA fine-tune on the B2 distractor dataset

Fork of `experiments/xvla_b1_language/` — repointed at the sealed parent
`outputs/b2_distractor_data/b2_distractor_20260928`. The two-instruction handling (task index
map, per-episode instruction checks, `approved_instruction`) is inherited **unchanged**: B2 also
carries two instructions as background randomisation across the same four cells, so every check
B1 needed for that axis already holds here with no code change.

## Why this package exists

B2 measures whether distractor cubes in the scene reduce success, independent of language. See
[docs/superpowers/specs/2026-09-28-b2-distractors-design.md](../../docs/superpowers/specs/2026-09-28-b2-distractors-design.md)
for the design and [docs/FAILURES.md](../../docs/FAILURES.md) #32–#35 for what collection hit.

## What differs from the fork source

**Budget is uncertain, not doubled by a known ratio.** B1 doubled its horizon because splitting
160 episodes across two instructions halved each task's data, and the shorter run was still
climbing steeply. B2's finest per-condition split (128 distractor-present episodes / 4 cells =
32/cell) is only ~20% thinner than B1's 40/cell — suggestive that B1's horizon is a floor, not
proof of it. `TOTAL_UPDATES` = 50,000, extending 25%/50% past B1's own winner (33,400) so the
curve can be read past it if still climbing.

| update | epochs | rationale |
|---|---|---|
| 1 | ~0.0001 | checkpoint machinery sanity |
| 16,700 | 1.67 | B1's matched-exposure point — direct comparison at identical update count |
| 25,000 | 2.50 | fills the same gap B1 screened |
| 33,400 | 3.35 | B1's winning checkpoint — does B2 land visibly below here? |
| 41,700 | 4.18 | +25% past B1's winner |
| 50,000 | 5.01 | +50% past B1's winner — current ceiling |

If the screening curve is still climbing at 50,000 the way B1's was at 33,400, a further
extension is not yet budgeted — an open decision in the design spec, not a promise.

**Disk budget recomputed for 6 checkpoints, not copied from B1's 5.** ~3.3 GB per regular
checkpoint and 9.9 GB for the final one (optimizer state): 5×3.3 + 9.9 = 26.4 GB. `require_resources`
gates training at 33 GiB free (B1 used 30 GiB for its 23.1 GB actual) — measured from this
package's own schedule per FAILURES #31, not copied from the fork source.

Everything else (contract structure, X-VLA architecture constants, processor/precision checks,
checkpoint verification, Adam roundtrip) is unchanged from `xvla_b1_language`.

## Parent binding

`SEAL` is injected from the sealed B2 collection
(`outputs/b2_distractor_data/b2_distractor_20260928/seal.json`,
`b6cfff3f27be92ba6fe12fef406cb481c384a521c9d2aeade41aa25da0c12340`). `current_binding()` rehashes
the full parent and compares against this seal on every admission.

Consequence: `experiments/b2_distractor_data/**/*.py` is frozen for this run's lifetime.

## Usage

`experiments.xvla_b2_distractors.runner` accepts exactly one of `--check`, `--preflight`,
`--train`. Run `--check` then `--preflight` (2 updates) and only launch `--train` if both pass, so
a fork bug costs minutes rather than the full ~11.7-hour run.

Outputs:

- `outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928_admission/`
- `outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928_preflight/`
- `outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928/`
