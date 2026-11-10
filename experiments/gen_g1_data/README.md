# gen_g1_data — G1's randomized collection (3,000 episodes, live force signals)

Design: [G1 spec](../../docs/superpowers/specs/2026-11-06-g1-randomized-data-design.md) (D1-D10, revision R1-R4, the 2026-11-10 mapping
note), [plan](../../docs/superpowers/plans/2026-11-06-g1-randomized-data.md). A fork of `experiments/props_p2_data/` (sealed, never
edited). The engine is `experiments/gen_scene/` (frozen by G0's core hash, imported read-only). Results and the expert gate live in
`docs/GEN_G1_COLLECTION_<date>.md`, not here: this README is hash-bound (`contract.provenance()`) and cannot change after `--collect`.

| module | what | image |
|---|---|---|
| `plan.py` | 3,000 slots (2,700 train / 300 validation, P2's 10 trained instruction pairs x 2 sides, milk-blue and bread-orange never collected) and 96 expert-gate attempts, namespace `gen:g1`, plan digest pinned in `PLAN_SHA256` | either |
| `contract.py` | images, shard count, `provenance()`; the live design documents are deliberately not bound (FAILURES #52) | either |
| `episode.py` | one attempt: `compose.model_for` + `GenExpert`, distractor motion and contacts, and S0's signals by exact replay (see below) | robosuite |
| `validate.py` | `admit` (expert success, undisturbed distractors, frame count, replay, labels, schema, signals), dataset validation, rejection report, fingerprint, seal | either (imports no simulator) |
| `record.py` | the collector: `--gate`, `--gate-summary`, `--determinism`, `--pilot`, `--collect`, `--resume`, `--status`; 16 shards, journal per shard | robosuite |
| `merge.py` | `--merge`: aggregate the shards, verify every episode against its shard, validate, seal | robosuite |
| `sheet.py` | contact sheets of 24 rendered gate attempts for the user's approval | robosuite |

**Signals.** `GenExpert.run` (frozen) has no observation hook, so the force and box signals are computed by an exact in-process replay
of the attempt: at every tick the replayed `qpos` and every object's pose must equal the recorded trace byte for byte, S0's
`wrist_wrench`, `finger_force` and `props_box` (imported read-only) are evaluated at the observation instant, and the state after the
last tick must equal the recorded one, or the attempt is rejected. Under the nominal condition the result equals S0's sealed sidecar byte
for byte (`tests/test_episode.py`, sealed P2 episodes 0, 1 and 5).

**Gate.** The D4 bars (overall at least 95%, at least 90% for every object, table-height stratum and distractor count) are applied to the
first-attempt *admission* of the 96 gate attempts, which is stricter than the expert's success alone; both are reported.

Each slot is one instruction cell and one scene drawn from the **train** pools only, as a `gen_scene.Condition(pool="train")`: lights,
table/floor/wall textures, table height (five equal strata of -7 to +12 cm), bin shift, target start (P2's region, or G0's outer ring
for 30%), 0-5 distractors (props of the six and pool A primitives), and a +-10 degree home pose. Balance is by construction (shuffled
decks): cells 135/15, distractor counts 135/405/675/675/540/270 (train), height strata 540 each, start regions 70/30. The layout carries
the target, its used start, the prop distractors and the home jitter; the condition carries everything else, so
`gen_scene.compose.model_for(layout, condition)` rebuilds the scene. `Condition.start` and `extra_props` are unused.

A rejected attempt retries the same slot: attempt a uses scene r = a // 3 (r > 0 redraws the whole scene inside the slot's stratum)
and expert draw k = a % 3 (seed and home jitter), so every stratum total of the admitted data is the plan's. Shards are contiguous
blocks of slots: 16 shards, the first eight of 188 slots, the last eight of 187.

Tests: `tests/` (admission image, with the sealed P2 data and the texture folder mounted read-only; see
[docs/OPERATIONS.md](../../docs/OPERATIONS.md)). The tests for the recorder run in the robosuite image.
