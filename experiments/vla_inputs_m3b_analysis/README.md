# vla_inputs_m3b_analysis: the M3b analysis

M3b ([spec](../../docs/superpowers/specs/2026-11-05-vla-inputs-m3b-predicted-box-design.md), D9 and D10) asks how much of A3's gain over A2
survives a box predicted from pixels. M0's `compare` is hard-wired to A0, so this module reads the arms' comparison stages through M0's
own loaders (read only) and applies M0's statistics. Admission image, no simulator, no GPU. Not bound by M0's core or any contract.

```
python -m experiments.vla_inputs_m3b_analysis.analyse --root <campaign> analyse [--no-verify] [--resamples N] [--box-log-dir DIR]
```

It writes `m3b_analysis.json` under the campaign root (one new file, never over an existing one, never inside an arm directory).

## What it computes

| quantity | definition |
|---|---|
| **R** | (S - F) / (C - F): S = A5's successes at checkpoint 101,520, F = A2's at 88,830 (its selected checkpoint; 523), C = A3's at 101,520 (571). Interval: layouts resampled stratified by object, the three arms together (numerator and denominator resampled together), 10,000 resamples, seed 0, percentile 2.5 / 97.5 over resamples with C* > F*. Undefined (an error) when C <= F. A secondary R uses A2 at 101,520 (517) |
| A5 against A2 | exact McNemar and M0's paired effect with its stratified bootstrap; the one pre-registered test (alpha 0.05) |
| A5 against A3 | the same, descriptive |
| A6-A9 against A3 | the same, Holm over the four, exploratory; an arm not run enters as p = 1 and is listed in `not_run` |
| failure classes | per arm, M0's causes with timeout-while-carrying and unstable-in-target pooled as the spec's prediction does; A5's counts against the recorded ranges (`PREDICTION`) |
| decision | `branch(R, A5 vs A2)`: R >= 0.75 closes M3b; 0.25 <= R < 0.75 partial (M3b-2 considered); R < 0.25 or A5 not above A2 (positive effect and p < 0.05) points to M4a; a threshold the interval straddles is flagged |
| A9 reading | >= 90% language dominates, 40 to 90% mixed, < 40% the box is followed |
| detector error | from A5's predicted-box logs and M0's traces: IoU against the live oracle box per camera (mean, share below 0.5, a 10-bin histogram), share of ticks with no predicted box, misses and false alarms, and the mean change in pixels between consecutive replans (every tenth tick), for the prediction and for the oracle itself |

## The log format `detector_error` reads

A directory of `*.jsonl` files (A5's adapter writes them), one JSON object per `act`: `{"trial": <string unique to a trial within its file>, "tick": 0..n-1,
"predicted": [[5], [5]], "oracle": [[5], [5]]}` with S0's `(u_min, v_min, u_max, v_max, in_frame)` per camera. A trial's group is matched to its trace by the oracle
sequence, which must equal the trace's `target_box` byte for byte; a trace without a log and a log without a trace are reported, never ignored.

## What it refuses

Unless every arm was prepared with core `c4f6c303f2ae` (the prepared record and the stage both), the core, the manifests and each arm still verify against the
repository (skipped by `--no-verify`, for tests), the layouts, objects and dropped layouts are identical, and the stages A2 at 88,830 and 101,520, A3 and A5 at
101,520 exist and are complete.

## The live identity gate

`identity_gate.py` (robosuite image, GPU, through `scripts/vla_inputs_m3b_eval.sh identity_gate`) is spec D7 (iv): `XVLANoCorruption` (A3's frozen
policy and force state, the oracle box drawn unchanged) re-runs A3's comparison trials 1 and 291 with M0's own rollout, and every trace array must equal A3's
recorded `trace.npz` byte for byte (dtype, shape, bytes; a NaN equals a NaN in the same place), as must `episode.json` without the wall clock. The video is
recorded, not compared. The real campaign is mounted read-only at `/campaign` and only read; the re-runs and the verdict (`identity_gate/result.json`) go to a
scratch `--out` that must not be the campaign and is never overwritten. Any difference is a failed gate (exit code 1): no A5-A9 trial counts until it passes.

## Tests

`tests/test_analyse.py` (admission image): hand-computed R (S = 559 against 523 and 571 is exactly 0.75), a bootstrap with a known stratified answer, exact McNemar, Holm over
four, the pinned thresholds and prediction, the branches, failure classes, IoU, the detector read-out on synthetic logs, every refusal, and the command line. Every check was confirmed to fail
against a deliberate mutation (floor and ceiling swapped, unstratified bootstrap, Holm over five, wrong control, a moved threshold, a denominator not resampled with the numerator, a
missing arm entering as 0, the replan stride, the IoU union, misses against false alarms, each refusal removed, matching by a prefix of the oracle sequence).
`tests/test_identity_gate.py` (a stand-in runner copies or alters A3's recorded trial: tolerance instead of bit equality, a dtype or key mismatch, the wall clock, `any` instead of `all`, an overwritten
verdict, a skipped verification and a gate built on an arm adapter are each caught) and `tests/test_m3b_eval_script.py` (the driver's dry-run commands, `EXTRA_MOUNTS` and `gpu_module` in M0's driver).
