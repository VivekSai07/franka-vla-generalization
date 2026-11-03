# VLA inputs overnight pilots: night 1 (force) and its calibration against full M2 — 2026-11-02

Written 2026-11-03 from the recorded outputs (`outputs/vla_inputs_eval/m0_pilot_night1_20261102/summary.json`, `SUMMARY.md`) and
the status log. Dated record: do not edit; later findings go into the live documents.
[Spec](superpowers/specs/2026-10-31-vla-inputs-overnight-pilots-design.md) ·
[plan](superpowers/plans/2026-10-31-vla-inputs-overnight-pilots.md) · packages `experiments/vla_inputs_pilot_plan/`,
`experiments/xvla_p2_pilot/`, `experiments/vla_inputs_arm_xvla_pilot/`, driver `scripts/vla_inputs_pilot_night.sh`.

## What a pilot is

A 10,000-update continuation (recipe `pilot_v1`: fresh AdamW, 200-update warm-up to a constant 1e-5) of the P2 X-VLA
checkpoint 101,520 with one input change, against a shared control continued the same way with the extra inputs zeroed, on 192
fresh paired layouts (190 feasible) disjoint from P2, its re-test and both M0 tiers. Rule fixed before the night: **promote iff
Δ ≥ +5 points**. In the pilot campaign the arm labels are local: `A0` is the continuation control, not M0's A0, and `A1` is the
first variant.

## What ran

| when | what | outcome |
|---|---|---|
| 2026-11-01 night | dry night (40 updates, GPU path only), inside the overnight chain | failed after 4 minutes on the pilot contract's checkpoint file list ([FAILURES #68](FAILURES.md)); the chain skipped the pilots and ran M2's selection tier instead |
| 2026-11-02 morning | fix merged (`96fb9d8`); dry night re-run (`m0_pilot_dry2_20261102`) | green |
| 2026-11-02 09:02 | pilot night 1, variant `force` only (the context arm A9 skipped; the `box` variant never run) | decided, below |

Wall clock: force zero-equivalence 181 s, training 2.4 h, evaluation 0.9 h; control 2.3 h (`wall_clock_s` 8,135); night 6.6 h.

## Result

| | control (`A0`) | `force` (`A1`) |
|---|---|---|
| successes / 190 | 136 (71.6%) | 142 (74.7%) |
| Δ, 95% paired interval | – | **+3.16 points** [−1.58, +7.89] |
| discordant (variant only / control only) | – | 13 / 7 |
| exact McNemar p | – | 0.263 |
| never lifted | 22 | 23 |
| timeout while carrying | 20 | 12 |
| wrong object | 5 | 4 |
| dropped outside | 2 | 4 |
| wrong bin / unstable in target | 3 / 2 | 3 / 2 |

**Decision by the rule: drop**, provisional until calibration. Recorded in the status log at 15:50 on 2026-11-02, before M2's
comparison tier was started (`PILOT_DECISION_RECORDED=1`).

## Calibration (plan Task 7), done 2026-11-03

The full M2 run (same 8 force values, trained from the base for 126,900 updates) measured, on M0's 573 comparison layouts,
**+12.6 points [8.9, 16.4]** at its selected checkpoint and +11.5 [7.9, 15.4] at the matched phase
([VLA_INPUTS_M2_20261103.md](VLA_INPUTS_M2_20261103.md)).

- Same sign; the pilot's interval [−1.6, +7.9] does not overlap the full run's [8.9, 16.4]. The pilot under-called the effect by
  about a factor of four and its rule produced a false negative.
- The mechanism moved the same way in both: timeout while carrying fell (pilot 20 → 12; full 36 → 8), so the pilot saw the right
  direction at the wrong size. Never lifted did not move in the pilot (22 → 23) but did in the full run (59 → 33).
- Per the spec's calibration rule, **a pilot may promote but must not drop.** A likely cause is that a 10,000-update continuation
  of a model trained without the signal adapts to it only partly; that is a reading, not a measurement.

## Consequences

- The `box` pilot was not run. Since a pilot can no longer drop an idea, it could not have spared M3's full run, so M3 (redesigned
  2026-11-03) goes straight to full training ([M3 spec](superpowers/specs/2026-10-30-vla-inputs-m3-box-design.md)).
- The pilot machinery stays on `main` for ideas that need a cheap promotion screen.

## Caveats

One seed per arm, 190 layouts, 10,000 updates (about 8% of a full run's 126,900); the layouts differ from M0's, so pilot and full
rates are not comparable as levels, only as paired differences against their own controls.
