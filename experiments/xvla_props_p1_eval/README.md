# X-VLA evaluation on the props P1 reserved layouts

Published artifacts: [P1 dataset](https://huggingface.co/datasets/VivekSai07/franka-panda-props-p1-540) and
[X-VLA root step 126,900](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900); alternate [88,830](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900/tree/bfc9e062d71e77a6e9d8358e576f5a0ed4609aef/checkpoints/step_088830).
Use the [P1 artifact/bundle instructions](../../docs/HUGGING_FACE_ARTIFACTS.md#props-p1-instruction-sorting)
to reconstruct the HF archive. Bundle re-evaluation uses the supplied wrapper; the
original evaluator below still requires its original run layout and checks.

Evaluation package for props P1: screen -> report -> confirm a trained checkpoint on the 80
reserved layouts from `outputs/props_p1_data/props_p1_20261006/plan.json` (never executed by the
collector). Scene, success rule and object geometry are imported unchanged from the accepted
`experiments/props_scene/`. Robosuite image (MuJoCo 3.4.0); rollouts need a GPU, `feasibility.py`
does not.

Design: [docs/superpowers/specs/2026-10-05-props-sorting-design.md](../../docs/superpowers/specs/2026-10-05-props-sorting-design.md)
(the "P1 training and evaluation" section). Plan: [docs/superpowers/plans/2026-10-07-props-p1-training-eval.md](../../docs/superpowers/plans/2026-10-07-props-p1-training-eval.md).

It follows `experiments/xvla_b3_robosuite_eval/` (read, not imported — neither package is
hash-bound, and a later edit there must not change this protocol). What differs: one object and
two bins instead of pick/place cubes, no distractor conditions, a `confirm` tier (props P1 has
three tiers, not two), and reserved layouts restored directly from the plan's pose parameters
(`props_scene.scene.reset`) instead of a precomputed physics bank — the P1 collection never built
one, since scenes are cheap and deterministic to construct fresh.

## Tiers

| tier | layouts | trials | composition |
|---|---|---|---|
| screen | 20 | 20 | all 20 trained cells, one seed each |
| report | 40 | 80 | 20 trained-pair layouts + 20 held-out-cell layouts, 2 seeds each |
| confirm | 20 | 20 | all 20 trained cells (different layouts than screen), one seed each |

Candidates for screening: the six non-trivial checkpoints `(50760, 76140, 88830, 101520, 114210,
126900)` — update 1 is machinery sanity only. Tie rule: every candidate within 2 successes of the
best screen count goes to the report. The confirm tier is touched once, on the report's winner,
after the report — that is the number quoted.

## Files

| file | responsibility |
|---|---|
| `protocol.py` | trial generation, seeds, the 12 approved instructions (10 trained + 2 held-out), output dirs |
| `scenes.py` | restore a reserved layout from the plan's pose parameters; asserts 1 mm / 1 deg |
| `feasibility.py` | the scripted expert on all 80 layouts, CPU only, no GPU — run while training runs |
| `rollout.py` | one policy rollout: 900-tick cap, `props_scene.metrics` success rule, `placed_bin`/`wrong_bin` tracking |
| `analysis.py` | `classify` (success / wrong_bin / never_lifted / timeout_carrying / dropped_outside), `readouts`, clustered intervals |
| `evaluate.py` | CLI `--mode smoke/screen/report/confirm --checkpoint N [--resume]`; refuses without feasibility, an unverified checkpoint, or MuJoCo 3.4.0 |

## Read-outs

Overall success with layout-clustered intervals (layouts, not trials, are the independent units);
success by `kind` (trained vs held-out); success by `blue_slot` — **the physical side the blue bin
is on, not the instructed colour** (Review Focus 5: a policy using position instead of colour
collapses on one arrangement; grouping by colour instead would hide this, since the evaluation
layouts balance target colour evenly within each arrangement); the wrong-bin rate (placed stably in
the *other* bin — the clean signature of a colour that was not read); and failure causes.

## Running

```bash
# CPU, no GPU, safe alongside other GPU work — run any time after the package is built:
python -m experiments.xvla_props_p1_eval.feasibility --out outputs/xvla_props_p1_eval/feasibility

# GPU, after training completes and feasibility has run:
python -m experiments.xvla_props_p1_eval.evaluate --mode smoke --checkpoint 126900
python -m experiments.xvla_props_p1_eval.evaluate --mode screen --checkpoint 50760
# ... one screen run per candidate, then report on the carried candidates, then confirm once.
```

Every trial writes `cameras.mp4`, `trace.npz`, `episode.json` under its own `trial_NNN_<layout_id>/`
folder inside `outputs/xvla_props_p1_eval/<mode>_step_<candidate>/`; `result.json` is rewritten
after every trial with every read-out so far. `--resume` continues a `running` or `failed` run
without rerunning finished trials; it is never automatic.
