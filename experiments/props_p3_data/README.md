# props_p3_data

Collection of the props P3 dataset: Food vs Drinks with a fixed prompt. One object per episode, no distractors,
the prompt `sort the item` in every frame, P0's two bins as containers (drinks → orange, food → blue) swapping
sides between episodes. Trained objects: can and milk (drinks), bread and lemon (food). Bottle (drink) and cereal
(food) are never collected, in training or validation. Built on the accepted P0 package (`experiments/props_scene/`,
imported, never edited); the expert is P0's `PropsExpert` unchanged. Spec:
[docs/superpowers/specs/2026-10-19-props-p3-food-drinks-design.md](../../docs/superpowers/specs/2026-10-19-props-p3-food-drinks-design.md).
Plan: [docs/superpowers/plans/2026-10-19-props-p3-collection.md](../../docs/superpowers/plans/2026-10-19-props-p3-collection.md).

**Image:** robosuite 1.5.2 / MuJoCo 3.4.0 (`sha256:54d861aa...`) for everything that builds the scene or merges;
`test_plan`, `test_contract`, `test_validate`, `test_record` import no simulator and run in the admission image
(`sha256:08de1e09...`). Rendering is OSMesa, as P1 and P2; the EGL image is not used.

## Plan

544 admitted: 480 training (8 trained cells × 60) and 64 validation (8 × 8). A cell is an object and the side of
the blue bin; the container follows from the object. 288 reserved layouts, never collected: screen 24 (3 per
trained cell), report 240 (10 per trained cell, 40 per held-out cell: 80 each for bottle and cereal), confirm 24.
288 gate attempts for the expert re-test (6 objects × 2 arrangements × 24).

## The prompt

The expert writes its own text, `put the <object> in the <colour> bin`, into every frame. `episode.run_attempt`
replaces it with the fixed prompt before any sink sees it and raises on any other text, so the expert's wording
never reaches the dataset. The validator then requires every frame, every episode and the dataset's task table to
carry exactly the fixed prompt.

## Admission

P1's gates in order: `expert_success`, `frames`, `replay` (< 1e-3 on the arm and the prop), `labels` (the fixed
prompt), `schema`. Three attempts per layout, then a fresh start for the same slot, up to three substitutions;
150 rejections per shard. The bin and the held-out split are checked on the plan and again on the merged dataset.

## Running

```bash
python -m experiments.props_p3_data.record --gate /out/gate --shard K          # K = 0..15, unrendered
python -m experiments.props_p3_data.record --determinism /out/gate --tag a     # twice, different tags
python -m experiments.props_p3_data.record --gate-summary /out/gate
python -m experiments.props_p3_data.record --pilot /out/pilot --shard K --slots ...
python -m experiments.props_p3_data.record --merge /out/pilot --out /out/pilot_merged --pilot-merge
python -m experiments.props_p3_data.record --collect /out/<run>_shards --shard K [--resume]
python -m experiments.props_p3_data.record --status /out/<run>_shards
python -m experiments.props_p3_data.record --merge /out/<run>_shards --out /out/<run>
```

One container per shard (16 shards of 34 slots, each pinned to 3 cores with `LP_NUM_THREADS=3`, proven pixel-neutral
against default threads), each journalling every attempt, writing frames only on admission,
finalizing per episode without reopening (FAILURES #41) and resuming alone. The merge copies videos byte for byte,
then `merge.verify` checks every merged episode against its shard before validation and the seal.

## Tests

```bash
run $OLD -m unittest discover -s experiments/props_p3_data/tests -t experiments/props_p3_data/tests -p 'test_[pcvr]*.py'
run $NEW -m unittest discover -s experiments/props_p3_data/tests -t experiments/props_p3_data/tests -p 'test_[me]*.py'
```
