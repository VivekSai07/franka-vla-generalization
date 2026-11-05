# VLA inputs M3: the target's box drawn on both camera images, on top of force (arm A3) — 2026-11-05

Written 2026-11-05 from the recorded outputs (`outputs/train/xvla_p2_force_box_v1/`,
`outputs/vla_inputs_eval/m0_20261030/arms/A3/`, `final_amended.json`) and the chain log
(`outputs/train/xvla_p2_force_box_eval_chain.log`). Dated record: do not edit; later findings go into the live documents.
[Spec](superpowers/specs/2026-10-30-vla-inputs-m3-box-design.md) · [plan](superpowers/plans/2026-10-30-vla-inputs-m3-box.md) ·
package [`experiments/xvla_p2_force_box/`](../experiments/xvla_p2_force_box/README.md) (hash-bound), arm
`experiments/vla_inputs_arm_xvla_force_box/`.

## Question

With everything else identical to M2's run (arm A2: P2's data and recipe, 17-value state with wrist force/torque and finger
forces), does drawing the target object's box on both camera images (4 px magenta band, outside the box) raise success on M0's
frozen comparison tier? Pre-registered criterion (spec D10/D11): **A3's paired effect against A2 is positive and its
Holm-adjusted p < 0.05 over {A1 vs A0, A2 vs A0, A3 vs A2, A4 vs A0}**, unrun arms at p = 1.

## Result: the criterion is met

Comparison tier, 573 feasible layouts of 576 planned (3 dropped for every arm, counted by cell), one trial each, paired.
Core `c4f6c303f2ae` (unchanged), arm `150d3970b3b4`. **One training seed per arm**; layouts are the units, so the interval and
test do not include seed-to-seed variation of training.

| | A2 at 88,830 (primary control) | A2 at 101,520 (matched phase) | A0 (P2, 101,520) |
|---|---|---|---|
| A3 at 101,520 | **571/573 = 99.65%** (99.1 – 100.0) | same A3 run | same A3 run |
| control | 523/573 = 91.3% | 517/573 = 90.2% | 451/573 = 78.7% |
| paired difference, 95% interval | **+8.4 points (6.1 – 10.6)** | +9.4 points (7.0 – 11.9) | +20.9 points (17.8 – 24.3) |
| discordant (A3 only / control only) | 50 / 2 | 56 / 2 | 121 / 1 |
| exact McNemar p | **6.1 × 10⁻¹³** | 1.2 × 10⁻¹⁴ | 4.6 × 10⁻³⁵ |
| Holm-adjusted p | **2.4 × 10⁻¹² (amended family, decides)** | secondary | 1.9 × 10⁻³⁴ (M0's own family, secondary) |

- **Primary test:** A3 at its selection winner (101,520) against A2 at its selected checkpoint (88,830): positive effect,
  adjusted p = 2.4 × 10⁻¹² < 0.05. `final_amended.json`: `criterion_met: true`. A2 vs A0 stays at 3.3 × 10⁻¹⁰ in the same family.
- A3 fails 2 layouts (`lemon-orange-right-d2_11`: timeout while carrying; `milk-blue-left-d1_11`: unstable in the target); A2
  passed both. Every one of A2's 50 failures at 88,830 is a pass for A3.
- Selection tier (79 trials each): 57 / 78 / 77 / 78 / 74 / 70 at 50,760 / 76,140 / 88,830 / 101,520 / 114,210 / 126,900.
  76,140 and 101,520 tied; the rule sends a tie to the later checkpoint (administrative, not evidence), which is also the
  matched-phase checkpoint, so the comparison tier ran at 101,520 only. The selection tier is nearly saturated and does not
  rank the late checkpoints; the comparison tier at 76,140 was not run.

## Where the gain is (exploratory, unadjusted, paired)

| | A2 at 88,830 | A3 at 101,520 |
|---|---|---|
| never lifted | 33 | **0** (A3 only: 0; A2 only: 33) |
| timeout while carrying | 8 | 1 |
| dropped outside | 3 | 0 |
| wrong object | 4 | 0 |
| unstable in target | 2 | 1 |
| milk | 77/96 = 80.2% | 95/96 = 99.0% |
| held-out pairs | 84/96 = 87.5% | 95/96 = 99.0% |
| trained pairs | 439/477 = 92.0% | 476/477 = 99.8% |
| bottle / cereal / lemon / can / bread | 92 / 94 / 84 / 89 / 87 of 96, 96, 94, 96, 95 | 96 / 96 / 93 / 96 / 95 of the same |

A3-only successes by object (50): milk 19, lemon 10, bread 8, can 7, bottle 4, cereal 2. Against A2 at 101,520: never lifted
41 → 0, wrong object 5 → 0, held-out 82 → 95, milk 74 → 95. The gripper-camera box left the frame in none of A3's two failing
trials (the read-out looks only at failures, so it says nothing about successes).

## The prediction against the result

**Recorded 2026-11-03, before any run: "0 to +4 points, probably not significant; wrong-object can add at most 0.7".** The
result is +8.4, more than twice the top of that range. The mechanism half of the prediction held: the gain did not come from
wrong-object (4 → 0, worth 0.7 points) but from approach and alignment, and never-lifted fell, from 33 to 0. The magnitude half
was wrong, in the same direction as M2's miss (M2: "small or no effect", measured +12.6). The headroom argument in the spec was
right and is what bounds the result: A2 fails 50 layouts, so the largest possible gain was 8.7 points, and A3 took 8.4 of them
(all 50 failures reversed, 2 new ones). The prediction treated "the box names the target" as the only channel; the box
also marks where the object is in the gripper camera, which is where A2 lost 33 of its 50 layouts. That is a reading of the
numbers, not a tested mechanism: no run separates "locates the object" from "names the object".

## Run record

- Training: 126,900 updates, 29.6 h of training time (`elapsed_seconds` 106,631) across three container launches and two clean
  `--stop-at` pauses (no failure, no supervisor restart). `final_data_binding_passed: true`; `adam_reload_fp32_exact: false`
  as in A2's run. Gates: `--check` (0 of 75,497,472 undrawn pixels near magenta), `--preflight`, `--resume-proof`,
  `--equivalence` and `--zero-equivalence` (overlay off reproduces M2's runner byte for byte over 30 updates; overlay on is
  unequal), so the box is the only difference from A2.
- Evaluation chain (one script, no step retried): `prepare` 22 s, `smoke` 126 s (2 of 2 at 101,520), `selection` 6,366 s
  (six checkpoints in parallel), `select`, `comparison` 7,210 s (573 of 573, 6 workers), `compare`, `compare_a2`, `final`.
  Started 2026-11-05 09:18, complete 13:07.

## Caveats

- **The box is privileged.** It is computed from simulator state at every tick (train/rollout parity proven, D8), amodal, and
  perfect. 99.65% is an upper bound for what a perfect detector could give through this encoding, not a deployable result.
  Nothing here says what a detector's or a policy's predicted box would do (M3b's question).
- **Box given force, not box alone.** The box-only arm on A0's 9-value state was not run; the effect is conditional on A2's
  force state.
- One training seed per arm and one rollout per layout. Rollout noise at checkpoint 101,520 is not separated from training-seed
  variation. The interval is layout-clustered only.
- A3's checkpoint was chosen by a selection tier on which it is nearly saturated (78/79 at two checkpoints), A2's by one where
  it was not. The matched-phase pair (+9.4) is consistent, so the verdict does not depend on that choice.
- Near the ceiling (2 failures in 573) the comparison tier cannot rank anything further above A3: a later arm that matches A3
  cannot show a gain here, and a drop is measurable only down from 99.65%.
- Exploratory read-outs are unadjusted and were not part of the pre-registered test.

## Programme consequences

- **M3b start condition (A3 meets the criterion) is met.** Its spec is not written; it needs the user's decision on the form
  (two-stage predicted box drawn onto A3's frames, or end to end). A two-stage M3b would measure the cost of a predicted box
  against this oracle with no new policy training.
- **M4 (output side) now has little headroom on this layout set:** A3 leaves 2 failures. Whether to keep M4 on these layouts, or
  re-base it on a harder protocol, is open and should be settled before M4 is built.
- Order remains M3 → M3b → M4 → M1 (pi0.5) last.
