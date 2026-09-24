# B1 language grounding — implementation plan

**Date:** 2026-09-25
**Design:** [../specs/2026-09-24-b1-language-grounding-design.md](../specs/2026-09-24-b1-language-grounding-design.md)
**Package:** `experiments/b1_language_data`
**Output:** `outputs/b1_language_data/b1_language_20260925`

This document is hashed into the dataset's provenance. It describes what was built, not what might
be built.

## What the dataset is

200 episodes over 200 distinct geometries, balanced across a 2×2 factorial of instruction ×
colour arrangement. 160 train (40 per cell) / 40 validation (10 per cell), plus 80 reserved
evaluation layouts that collection never executes.

| cell | instruction | arrangement | pick colour is in |
|---|---|---|---|
| C1 | red→green | familiar | region A (−y) |
| C2 | red→green | swapped | region B (+y) |
| C3 | green→red | familiar | region B (+y) |
| C4 | green→red | swapped | region A (−y) |

Each region is correct in exactly two cells and each instruction maps to region A once and region B
once, so neither position nor instruction alone predicts the target. This is the property the
dataset exists to create; `test_neither_position_nor_instruction_alone_predicts_the_target` asserts
it directly.

## What changed from `broad_workspace_data`

- **`plan.py`** — axes renamed from colours to **regions** (identical numbers); `assign_cells`
  gives exactly n/4 episodes per cell by construction; `colour_positions` permutes region → colour
  by arrangement; `instruction_for` is the single source of instruction text; `task_index_for`
  indexes the **sorted** instruction tuple; `TRAIN_BASE_COUNT = 0` so every episode has a distinct
  geometry; `UNIQUE_BANK_IDENTITIES = 361`.
- **`scenes.py`** — layout validation is region-based, so either colour may occupy either region;
  `eval_bank_key(layout_id, arrangement)` keys reserved layouts as `{layout_id}:{arrangement}` and
  `make_bank` banks each reserved layout in **both** arrangements.
- **`validate.py`** — `cell` and `arrangement` added to the immutable-field checks;
  `require_task_table` pins LeRobot's `meta/tasks.jsonl` to the planned mapping;
  `require_cell_balance` asserts balance survived collection.
- **`expert.py`, `record.py`** — unchanged behaviour. The expert was already colour-parameterised
  and the instruction already flowed per-episode.

## Decisions and their reasons

- **Fully self-contained.** B1 does not read, import, extend or re-seal `broad_workspace_data`.
  Sharing the holdout would have given a paired comparison against 86.1% but required importing
  another dataset's layout bank; this repository has been damaged three times by that kind of
  cross-package reach.
- **`task_index` from sorted instructions, not collection order.** LeRobot allocates indices by
  first appearance, so a rejected episode could silently shift the mapping. Validation pins the two
  against each other.
- **Both arrangements banked, not derived at evaluation time.** Keeps the "every scene is
  hash-proven against a frozen bank before use" invariant and puts both arrangements under the seal.
- **Balance enforced by construction.** An imbalanced split would partially restore the position
  shortcut, which is the one thing this dataset must not permit.

## Verification before collection

- 50/50 package tests green, including `test_fork_hygiene` — a `symtable` scan for names read but
  never bound, a check that no module still references the fork source, and a check that no
  colour-keyed envelope constant survives. The two previous forks in this repository shipped bugs
  that only appeared at runtime; both were statically detectable.
- Bank built end-to-end: 361 identities = 200 episodes + 80 × 2 arrangements + 1 base, all digests
  unique, and the two arrangements of a layout verified to be mutually transposed.
- Plan verified balanced: 50 per cell overall, 40 per cell in train, 10 per cell in validation,
  100/100 per instruction, 100/100 per arrangement, 100/100 per pick colour.

## Sequence

1. `--preflight` — read-only: plan, bank, provenance, disk.
2. `--collect` — 200 admits with the feasibility gate and rejection budget. ~6.6 h at the measured
   1.97 min/episode.
3. CPU admission (`--check`) for the training package. ~1 h.
4. Preflight, then train X-VLA from base for 16,700 updates (1.67 epochs). ~3.9 h.

Steps 2–4 run chained; each stage refuses to start unless the previous one verified.

## Known limitations recorded at creation

- **Terminal heights are constant.** Cube resting height and stack height are fixed by a flat table
  and identical cubes. Approach altitude is varied (±3 cm hover jitter) but the target is not, so an
  absolute-Z policy satisfies every episode. Deliberately not addressed here — it would be a second
  new axis and failures would not be attributable. See TODO B1b.
- **Half the per-instruction data.** 80 episodes per instruction against 160 for the
  single-instruction model. The C1-vs-C2 contrast is within-model so attribution is unaffected;
  only the absolute level is exposed. Escalate to 400 episodes only if the absolute result is
  ambiguous.
