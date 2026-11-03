# VLA inputs M2: wrist force/torque and finger forces in the robot state (arm A2) — 2026-11-03

Written 2026-11-03 from the recorded outputs (`outputs/train/xvla_p2_force_v1/`, `outputs/vla_inputs_eval/m0_20261030/arms/A2/`)
and the status log. Dated record: do not edit; later findings go into the live documents.
[Spec](superpowers/specs/2026-10-30-vla-inputs-m2-force-design.md) · [plan](superpowers/plans/2026-10-30-vla-inputs-m2-force.md) ·
package [`experiments/xvla_p2_force/`](../experiments/xvla_p2_force/README.md) (frozen and hash-bound since its `--check`), arm
`experiments/vla_inputs_arm_xvla_force/`.

## Question

With everything else identical to P2's X-VLA run (data, images, instructions, recipe, sample order), does adding the wrist
force/torque (6) and the two finger contact forces (2) to the robot state change success on M0's comparison tier?

## What was trained

- State 17 = P2's 9 joint positions + the 8 values from the sealed S0 sidecar (seal `7e2e06db…f58e6c`), joined by
  (episode, frame), clipped to S0's training-row 0.1/99.9 percentiles and standardised with fixed constants (digest `98d86821…`);
  LeRobot's state normalisation stays `IDENTITY`, as in P2.
- Recipe P2's, asserted equal value by value: 126,900 updates (5.0 epochs over 203,040 windows), effective batch 8, chunk 30,
  ten executed actions (3 Hz replanning), checkpoint grid 50,760 / 76,140 / 88,830 / 101,520 / 114,210 / 126,900.
- Gates, 2026-10-31, from the detached worktree `.worktrees/m2-run` at `f1c49b3`: `--check`, `--preflight`, `--resume-proof`,
  `--equivalence` (30 updates byte-equal) and **`--zero-equivalence`**: P2's own runner and this one with the 8 extras forced to 0
  agree byte for byte over 30 updates (metrics, weights, Adam moments); the mutation (one column left un-zeroed) was reported
  unequal. Train/rollout state parity byte-equal on 1,808 replayed frames, and on the normalised, padded `proprio` tensor.
  Inference boundaries proven: the policy runs at ticks 0, 10, 20 of 30.
- Training 2026-10-31 17:02 to 2026-11-01 21:42, **28.6 h** wall clock (0.52 s per update raw; the rest is the thermal guard,
  83 → 75 °C, checkpoints and snapshots), under a supervisor that resumes from the newest verified snapshot after a death.

## Evaluation on M0 (frozen core `c4f6c303f2ae`, arm `c04a8145c120`)

**Selection tier** (79 feasible trials per checkpoint, 2026-11-01/02, done 00:51): 56 / 61 / **70** / 69 / 69 / 65 at 50,760 /
76,140 / **88,830** / 101,520 / 114,210 / 126,900. The comparison tier ran the winner 88,830 and the matched-phase
101,520 (A0's checkpoint).

**Comparison tier** (573 feasible layouts of 576, the same layouts and trials as A0, paired; 2026-11-02 15:50 to 21:16, about
5.4 h, 6 workers):

| | A0 (101,520) | A2 at 88,830 (primary) | A2 at 101,520 (matched phase) |
|---|---|---|---|
| overall | 451/573 = 78.7% | **523/573 = 91.3%** (88.8 – 93.5) | **517/573 = 90.2%** (87.8 – 92.5) |
| paired difference, 95% interval | – | **+12.6 points** (8.9 – 16.4) | +11.5 points (7.9 – 15.4) |
| discordant layouts (A2 only / A0 only) | – | 100 / 28 | 100 / 34 |
| exact McNemar p | – | 1.1 × 10⁻¹⁰ (Holm over A1–A4: at most 4.4 × 10⁻¹⁰) | 9.9 × 10⁻⁹ |
| trained pairs | 384/477 = 80.5% | 439/477 = 92.0% | 435/477 = 91.2% |
| held-out pairs (milk→blue, bread→orange) | 67/96 = 69.8% | 84/96 = 87.5% | 82/96 = 85.4% |
| milk | 51/96 = 53.1% | 77/96 = 80.2% | 74/96 = 77.1% |
| bottle / cereal / lemon / can / bread | 83% / 81% / 78% / 89% / 88% | 96% / 98% / 89% / 93% / 92% | 99% / 99% / 85% / 93% / 88% |

| failure | A0 | A2 at 88,830 | A2 at 101,520 |
|---|---|---|---|
| never lifted | 59 | 33 | 41 |
| timeout while carrying | 36 | 8 | 6 |
| dropped outside | 11 | 3 | 3 |
| wrong bin | 7 | 0 | 0 |
| wrong object | 4 | 4 | 5 |
| unstable in target | 5 | 2 | 1 |

Exploratory, untested: paired never-lifted layouts A0 only / A2 only / both are 50 / 24 / 9 at 88,830 and 47 / 29 / 12 at
101,520, so A2 also fails some layouts A0 lifted; milk never-lifted 22 → 12 / 14. The Holm figure is a bound: only A2 has been
evaluated among A1–A4, and the smallest p is multiplied by the family size.

## Against the prediction

The spec predicted (2026-10-30) "small or no effect is plausible". **That was wrong by a wide margin.** The gain is not confined
to the hold: carrying timeouts and wrong-bin outcomes nearly vanished as well.

The overnight force pilot (a 10,000-update continuation with the same 8 values) measured +3.2 points (p = 0.26) and was dropped
by its +5 rule; the full run is about four times larger. Calibration and its consequence (pilots may promote, never drop):
[VLA_INPUTS_PILOTS_20261102.md](VLA_INPUTS_PILOTS_20261102.md).

## Caveats

- **One training seed.** The test shows the difference is not layout noise; it does not bound seed-to-seed variation of training.
- Exact simulator forces: nothing follows about a real wrist sensor.
- 3 Hz replanning with ten queued actions: force conditioning at chunk boundaries, not reactive control.
- Which of the 8 values matters, and whether history or another normalisation would do better, is not shown.

## Next

M3 builds on A2: the target's box drawn on both camera images over this 17-value state, tested against A2
([M3 spec](superpowers/specs/2026-10-30-vla-inputs-m3-box-design.md)).
