# props_p2_data

Collection of the props P2 dataset: instruction sorting with 1–2 distractor objects. One target, the
instruction `put the <object> in the <colour> bin`, two bins swapping sides, `milk→blue` and
`bread→orange` never collected as targets. Built on the accepted P0 package (`experiments/props_scene/`,
imported, never edited); the only expert change is a 0.19 m carry so the held object clears a standing
distractor. Spec: [docs/superpowers/specs/2026-10-13-props-p2-distractors-design.md](../../docs/superpowers/specs/2026-10-13-props-p2-distractors-design.md).
Plan: [docs/superpowers/plans/2026-10-13-props-p2-collection.md](../../docs/superpowers/plans/2026-10-13-props-p2-collection.md).

**Image:** robosuite 1.5.2 / MuJoCo 3.4.0 (`sha256:54d861aa...`) for everything that builds the scene
or merges; `test_plan`, `test_contract`, `test_validate`, `test_record` import no simulator and run in
the admission image (`sha256:08de1e09...`).

## Plan

540 admitted: 480 training (20 trained cells × 24, half with one distractor, half with two), 60
validation (20 × 3, 30/30). Distractors are distinct other props from shuffled decks. Every object in
the P1 start region; every pair at least `max(r_a + r_b + 0.03, 0.065 + max(r_a, r_b))` apart —
symmetric, so spacing does not mark the target. 120 reserved layouts, never collected: screen 20,
report 60 (20 + 20 trained with one/two distractors, 20 held-out), control 20 (no distractor),
confirm 20. 288 gate attempts for the expert re-test.

## Admission

P1's gates in order — `expert_success`, `frames`, `replay` (< 1e-3 on the arm and every prop),
`labels`, `schema` — plus `distractors`: no distractor touched by anything but the table at any physics
substep (a pose threshold cannot work: the lemon rocks on its own after reset, up to 13° untouched).
Three attempts
per layout, then a fresh layout with the same objects, up to three substitutions; 150 rejections per shard.

## Running

```bash
python -m experiments.props_p2_data.record --gate /out/gate --shard K          # K = 0..11, unrendered
python -m experiments.props_p2_data.record --determinism /out/gate --tag a     # twice, different tags
python -m experiments.props_p2_data.record --gate-summary /out/gate
python -m experiments.props_p2_data.record --pilot /out/pilot --shard K --slots ...
python -m experiments.props_p2_data.record --merge /out/pilot --out /out/pilot_merged --pilot-merge
python -m experiments.props_p2_data.record --collect /out/<run>_shards --shard K [--resume]
python -m experiments.props_p2_data.record --status /out/<run>_shards
python -m experiments.props_p2_data.record --merge /out/<run>_shards --out /out/<run>
```

One container per shard, `LP_NUM_THREADS=4` each. Each shard journals every attempt, writes frames
only on admission, finalizes per episode without reopening (FAILURES #41) and resumes alone. The
merge copies videos byte for byte, then `merge.verify` checks every merged episode against its shard
before validation and the seal.

## Tests

```bash
run $OLD -m unittest discover -s experiments/props_p2_data/tests -t experiments/props_p2_data/tests -p 'test_[pcvr]*.py'
run $NEW -m unittest discover -s experiments/props_p2_data/tests -t experiments/props_p2_data/tests -p 'test_[me]*.py'
```
