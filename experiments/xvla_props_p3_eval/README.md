# Props P3 evaluation

Rollout evaluation of the X-VLA checkpoints trained by `experiments/xvla_props_p3/` on the 288 reserved P3 layouts (never
collected), in the robosuite scene. Not hash-bound: no contract, no `provenance()`; the sealed parent, the training output and the
checkpoints are read-only inputs. Robosuite image (`sha256:54d861aa…ccf27e`, MuJoCo 3.4.0), a GPU for the workers, **OSMesa
rendering, the renderer the checkpoints were trained on; never the EGL image** (its frames differ from OSMesa's on half the pixels).

Spec and pre-registration: [docs/superpowers/specs/2026-10-19-props-p3-food-drinks-design.md](../../docs/superpowers/specs/2026-10-19-props-p3-food-drinks-design.md)
(revised 2026-10-20). Plan, with the one decision made while building:
[docs/superpowers/plans/2026-10-21-props-p3-training-eval.md](../../docs/superpowers/plans/2026-10-21-props-p3-training-eval.md).
A fork of `experiments/xvla_props_p2_eval/` with the distractor, control and cross-policy machinery removed.

## Protocol

288 layouts, one rollout seed each (the layout is the independent unit): screen 24 (3 per trained cell), report 240 (80
trained-object layouts, 80 bottle, 80 cereal), confirm 24. Every trial carries the prompt `sort the item`. The correct bin
follows from the object (drinks orange, food blue); the blue bin's side is balanced inside every object.

1. **Feasibility** (`feasibility.py`, scripted expert, CPU, unrendered): a layout the expert cannot solve is dropped and counted,
   never replaced; the policy is never blamed for it.
2. **Screen**: every non-trivial checkpoint (50,760 … 126,900) on the 24 screen layouts; every checkpoint within 2 successes of the
   best goes to the report (a tie is a tie).
3. **Report**: the survivors on the 240 layouts. The winner is the highest success count on the **trained-object** layouts, ties to
   the later checkpoint: held-out layouts never decide which checkpoint is reported.
4. **Confirm**: the 24 confirm layouts, once, on the winner. Selected on nothing: the unbiased estimate.

## What is gated (pre-registered; `analysis.criteria`)

The gate is **bin choice**, not overall success: motor failures (never lifted, hold, carry) dominate this robot and say nothing about
Food vs Drinks. Bin choice is read on **stable placements**, the trials in which the object came to rest inside either bin
(`placed_bin` is not None at the end), correct when it is the container of the object's category.

| | rule |
|---|---|
| 1. trained objects (80 layouts) | wrong-bin share of stable placements at most 5% (point estimate, Wilson interval reported); at least **60** stable placements, else `not_interpretable` (neither pass nor fail) |
| 2. held-out objects, exploratory | for bottle and cereal each, at least **30** stable placements and the Wilson 95% lower bound of bin-choice accuracy above 0.5; `evidence` only if both show it, `not_shown` otherwise (inconclusive, not a negative below about 65% true accuracy), `not_interpretable` below the floor |

Reported, not gated: overall and per-object success (milk read on its own), by blue-bin side, the failure causes
(`never_lifted`, `unstable_in_target`, `timeout_carrying`, `dropped_outside`, `wrong_bin`), and the count of successes without a
correct stable placement (must be 0). With six objects, "category" cannot be separated from visual similarity: a pass is evidence
about the object-to-container mapping on these objects, not about semantic food versus drinks.

## Running

```bash
scripts/xvla_props_p3_eval.sh feas        # 288 layouts, 12 CPU shards, unrendered
scripts/xvla_props_p3_eval.sh prepare     # checkpoints, seal, feasibility verified once (needs the finished training run)
scripts/xvla_props_p3_eval.sh smoke 1     # two trials on 126,900
scripts/xvla_props_p3_eval.sh all 8       # screen -> decide -> report -> decide -> confirm -> final.json
```

Workers are pinned to 5 cores each and sized by peak GPU memory (about 7.3 GiB after a checkpoint switch: FAILURES #57); a stage is
resumable and a resumed trial equals an uninterrupted one. Output under `outputs/xvla_props_p3_eval/` (gitignored): one folder per
trial (`cameras.mp4`, `trace.npz`, `episode.json`, `telemetry.json`), `decisions/`, `feasibility/`, `final.json`.

## Tests

```bash
run $OLD -m unittest discover -s experiments/xvla_props_p3_eval/tests -t experiments/xvla_props_p3_eval/tests -p 'test_[pae][nrv]*.py'
run $NEW -m unittest discover -s experiments/xvla_props_p3_eval/tests -t experiments/xvla_props_p3_eval/tests -p 'test_rollout.py'
```

The pure suites (protocol, analysis, evaluate) need the sealed parent mounted (they skip otherwise); the rollout suite restores all
288 layouts, replays the expert's actions as the policy for a drink and a food among the trained and the held-out objects, and
produces a `wrong_bin` outcome with real physics.
