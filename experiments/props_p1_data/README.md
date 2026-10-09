# props_p1_data

Collection of the props P1 instruction-sorting dataset: one object, two bins, swapping sides, the
instruction `put the <object> in the <colour> bin`. Built on the accepted props P0 package
(`experiments/props_scene/`, go 2026-10-05, re-confirmed 2026-10-06) — the scene, the scripted expert
and the success rule are P0's, unchanged. Spec:
[docs/superpowers/specs/2026-10-05-props-sorting-design.md](../../docs/superpowers/specs/2026-10-05-props-sorting-design.md)
(P1 section). Plan: [docs/superpowers/plans/2026-10-06-props-p1-collection.md](../../docs/superpowers/plans/2026-10-06-props-p1-collection.md).

**Image:** robosuite 1.5.2 / MuJoCo 3.4.0 (`sha256:54d861aa...`) for collection; `test_plan.py`,
`test_contract.py` and `test_validate.py` import no simulator and also run in the admission image
(`sha256:08de1e09...`).

## Plan

`plan.make_plan()`: 12 object×colour pairs, each in both bin arrangements, 24 cells; 2 pairs
(`milk→blue`, `bread→orange`) held out of training entirely (4 cells). 480 training episodes (20
trained cells × 24), 60 validation (20 × 3), **540 admitted**. 80 reserved evaluation layouts, never
collected: 20 screen (one per trained cell), 40 report (20 trained + 20 held-out, 5 per held-out
cell), 20 confirm (one per trained cell, untouched until the winning checkpoint). Starts: the P0
amended region (x ∈ [0.27, 0.40], radius ≥ 0.285 m), yaw ±90° except lemon ±75°.

## Admission

Per episode, in order: `expert_success` (P0's rule, ≥ 30 stable ticks), `frames` (452, as recorded),
`replay` (`props_scene.validate.replay`, < 1e-3 on the arm and the object), `labels` (instruction
matches the plan slot, every frame's `task` matches), `schema` (dtypes and shapes). A rejected
attempt retries the same slot (`plan.attempt_draw`); after 3 attempts the slot is resampled to a new
start pose (`plan.resample_start`), up to 3 substitutions.

## Collection

```bash
python -m experiments.props_p1_data.record --pilot <path> --slots 0 1 2 ...
python -m experiments.props_p1_data.record --collect <path>
python -m experiments.props_p1_data.record --resume <path>
python -m experiments.props_p1_data.record --status <path>
```

One journal line per attempt; frames are buffered and only written to the dataset on admission; the
dataset is finalized once per episode without reconstructing the dataset object, so a crash leaves
the completed episodes readable without the per-episode reopen cost that filled `/tmp` in B3
(FAILURES #41). `--collect` refuses an existing output and writes `plan.json`/`provenance.json`
first; `--resume` refuses a dataset/journal mismatch.

## Tests

```bash
run $OLD -m unittest experiments.props_p1_data.tests.test_plan        # also _contract, _validate, _record
run $NEW -m unittest experiments.props_p1_data.tests.test_episode
```
