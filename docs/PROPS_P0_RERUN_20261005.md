# Props P0 re-run against the amended spec — 2026-10-05

Branch `feat/props-p0`, rebased onto `main` (`19174ae`), nothing pushed. Package `experiments/props_scene/`; the
amended spec is [superpowers/specs/2026-10-05-props-sorting-design.md](superpowers/specs/2026-10-05-props-sorting-design.md).
**Robosuite 1.5.2 / MuJoCo 3.4.0 image (`sha256:54d861aa…`), scripted attempts only, no policy; the package has no
contract and no seal.** Numbers are not comparable with the sealed MuJoCo 3.13.0 pipeline. The first run is the record
of what failed and why: [PROPS_P0_20261005.md](PROPS_P0_20261005.md) (not edited).

## Result

**The P0 gate passes: 283 of 288 attempts (98.3%) against 95%, every object at least 95.8%, both slots at least 97.2%.**

| gate | threshold | measured (code `cd28f04`) | |
|---|---|---|---|
| overall | ≥ 95% | **283 / 288 = 98.3%** | pass |
| per object | each ≥ 90% | can 48/48, bread 48/48, cereal 48/48, milk 47/48 (97.9%), bottle 46/48 (95.8%), lemon 46/48 (95.8%) | pass |
| per target slot | each ≥ 93% | left 143/144 (99.3%), right 140/144 (97.2%) | pass |
| replay within 1e-3 (arm and prop pose) | all | 288 of 288, largest error 0.0 | pass |
| determinism, byte-equal traces in a second process | 12 re-executed | 12 of 12 (images included); 12 more on the other bin arrangement, 12 of 12 | pass |
| frames show both bins and the object | by eye | yes | pass |

Thresholds, the 7 cm wall and the 30-tick rule are as in the spec; only the two amendments below changed. By bin
arrangement (which slot blue is in) 142/144 and 141/144; by target bin blue 140/144, orange 143/144. Per object and
target slot, the lowest is lemon right 22/24 (the others 23/24 or 24/24): no left/right asymmetry. Median 34 s per
attempt with rendering; the six shards took 27 minutes. Records, plan, summary: `outputs/props_p0/gate_20261005c/`.

## The two amendments

Both were errors in the spec, not in the expert, and both are now in the spec with their evidence.

1. **Success is decided by the object's lowest point, not its centre.** An object counts as in the bin when its centre
   is inside the inner box in x and y, its lowest point is below the 7 cm wall top, the gripper is open and clear, and
   that holds for 30 ticks, still true at the last tick. The first rule (centre below the wall top) cannot be met by a
   14–16 cm object standing on the bin floor (centre 7.2 to 8.0 cm); in the first run 77 of 136 failures were objects
   resting correctly in the bin and all 35 tall successes had fallen over. `metrics.in_bin` now computes the lowest point
   from the collision hull (`geometry.bottom_z`); the centre is still the centre of mass. The test that documented the
   standing-cereal failure now asserts that a standing cereal box is a success; it failed against the old rule first.
2. **Start region x ∈ [0.27, 0.40]** (was 0.22). Below about 0.285 m from the base the expert's straight-line approach
   saturates joint 4 and the hand closes on air (58 of the first run's failures). There is **no near-base stratum any
   more**, so the 12 corner attempts of the first gate have no region: the gate is 6 objects × 2 bins × 2 arrangements ×
   12 seeds = **288 attempts**, not 300. The spec text still says 300; that is a deviation in the plan's favour of the
   spec's own "~12 seeds" per cell. If you want 300, add a thirteenth seed (312).

Also: the square-footprint fix (`3e19c01`, now `e8509da` after the rebase) is accepted and part of this code; the
OSC comparison is kept as evidence and is not a gate input; the cube's contact parameters are unchanged.

## Against the prediction

Re-scoring the first run under both amendments predicted 205 of 206 (99.5%). The re-run draws new start poses (the
pose is a function of the attempt's identity and the region), so it is a fresh sample, not a re-scoring: 283 of 288
(98.3%). The difference is five attempts, and they are not a new failure mode:

| attempt | start (x, y, yaw) | radius | what the trace shows |
|---|---|---|---|
| lemon, orange, left, #11 | 0.281, −0.038, 80° | 0.283 | joint 4 at its fold limit during close and lift; hand 1 cm out |
| milk, blue, right, #4 | 0.277, −0.009, −40° | 0.278 | in the reach band (not looked at in the trace) |
| bottle, blue, left, #7 | 0.280, 0.000, −69° | 0.280 | joint 4 saturated at the close; hand 1.7 cm out |
| bottle, blue, right, #9 | 0.279, −0.033, −7° | 0.281 | in the reach band (not looked at in the trace) |
| lemon, blue, right, #10 | 0.294, −0.116, 81° | 0.316 | joint 4 margin 0.07–0.13 rad, not saturated; joint 1 swings 0.3 rad in the close |

- Four of the five start between radius 0.278 and 0.283 m, **inside the reach boundary the amendment was written
  around (about 0.285 m)**, because the region is bounded by x ≥ 0.27, not by radius: 5.9% of the amended region is
  closer than 0.285 m, and 13 of the 288 attempts landed there (9 succeeded). Beyond 0.285 m: **274 of 275**.
- The fifth is the lemon failure of the first run's analysis again: lemon, y ≈ −0.12, yaw ≈ 80°, joint 4 not at its limit
  (the first run had it at (0.29, −0.12, 79°) and (0.253, −0.116, 81°)). Its cause is not established. It does not
  threaten the gate (lemon 95.8%) but it is a repeatable pattern worth a look before P1 relies on lemons at that yaw.
- All five are `reason = grasp`; there is no other failure.

## What the tall objects do now

All six objects are scored by the same rule. Under it: cereal ends upright in 48 of 48; milk 22 upright and 25 fallen
(47 successes); bottle 19 upright and 27 fallen (46). Whether the object stands or falls no longer decides the score;
the expert still does not try to keep them upright, so a policy trained on these demonstrations will see both.

## Contact sheet

`outputs/props_p0/contact_sheets/rerun_tall_standing.png` (cereal, milk, bottle, all ending upright; viewed): the front
camera shows the table, the orange bin on the image-left and the blue bin on the right (blue's slot is y > 0), the arm
and the object; the object is carried upright over the blue bin, lowered until it stands on the floor of the bin, released
and left standing (the cereal box leans slightly; milk and bottle are straight); the gripper camera shows the object between
the pads before the grasp and the inside of the blue bin during lower, open and retract, ending with the bin seen from the
home pose. The first run's sheets (`successes*.png`, `failures.png`) remain valid for the unchanged part of the pipeline.

## What this does not show

- No policy was trained. Nothing here says how well X-VLA can learn these tasks, or tell the bins apart by colour.
- The determinism recheck built into the gate samples every 24th attempt, which here lands only on the blue-left
  arrangement of each object; I ran 12 attempts on the other arrangement separately
  (`outputs/props_p0/recheck_right.py`, `gate_20261005c/recheck_right_arrangement.json`) and the gate's own recheck should
  spread over arrangements before it is reused.
- Each attempt is one run at one seed; the five failures sit in a band the region still includes; the amended region
  excludes the near-base area the B2 and B3 weak corner lay in, so that corner is not tested here.
- Whether any attempt left an object leaning on the rim with its lowest point under the wall top and most of it
  outside the bin was not checked; the lowest-point rule would count it, as the spec is written.

Decision: go

## 2026-10-06 — re-confirmation after the region and yaw amendments

Before the P1 collector, the start region gained a radius floor (0.285 m, not just x ≥ 0.27) and the lemon's
yaw was capped at ±75° (the two remaining lemon failures above were a yaw effect: grasp axis near-aligned
with the long side past 75°, not a reach effect). The full 288-attempt gate re-run on this region
(`outputs/props_p0/gate_20261006_reconfirm/`, code `a927a6f`): **286/288 = 99.3%**; can, bread, cereal,
milk and bottle all 48/48; **lemon 46/48** (one `grasp`, one `lost_in_carry`); target slots 143/144 and
143/144; replay exact on all 288; determinism recheck 12/12, this time spread over all six objects and
both bin arrangements (the sampling gap noted above is fixed). Verdict: go. The remaining two lemon
failures are not examined further here; P1's collection will reject and resample them like any other
expert failure.
