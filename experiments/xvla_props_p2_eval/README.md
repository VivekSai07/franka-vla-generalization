# Props P2 evaluation

Rolls the props P2 checkpoints (X-VLA trained on 540 sealed episodes with 0-2 distractors) on the reserved
layouts of the sealed P2 plan, against P1's 126,900 checkpoint on the same trials. Not hash-bound: it imports
`props_p2_data` and `props_scene` read-only and never writes under the dataset or the training output. The rules
were fixed and committed before any rollout:
[docs/superpowers/plans/2026-10-16-props-p2-evaluation.md](../../docs/superpowers/plans/2026-10-16-props-p2-evaluation.md).

After the 2026-10-17 disk reclaim only P1 88,830 / 126,900 and P2 101,520 / 126,900 remain on RTX, so `prepare` (which verifies every
checkpoint of both runs) can no longer pass in the repo; re-evaluate from the archive bundle with `reevaluate/bundle_evaluate.py`.

Robosuite image (MuJoCo 3.4.0) with a GPU for rollouts; everything else (`prepare`, `collect`, `decide`, `final`,
the pure tests) runs in the admission image.

| module | what it holds |
|---|---|
| `protocol.py` | the screen / report / control / confirm trials, seeds, jobs, interleaved sharding, the 12-instruction context |
| `scenes.py` | restores a reserved layout (target plus distractors) and asserts 1 mm / 1 deg against the plan |
| `rollout.py` | one policy rollout; traces every prop so the first-lifted object is known |
| `analysis.py` | lift rule, failure classifier, layout-clustered intervals, selection rules, the four criteria |
| `feasibility.py` | the scripted expert on all 120 layouts, sharded; infeasible layouts are excluded, never blamed on the policy |
| `evaluate.py` | `prepare`, `worker`, `collect`, `decide`, `final` |
| `../../scripts/xvla_props_p2_eval.sh` | the staged parallel driver (`smoke`, `all`) |

## Stages

| tier | layouts | trials / checkpoint | purpose |
|---|---|---|---|
| screen | 20 (10 one-distractor, 10 two) | 20 | pick candidates; 6 checkpoints |
| report | 60 (20 trained x 1 distractor, 20 trained x 2, 20 held-out cells) | 120 (2 rollout seeds) | survivors and P1 126,900 on the same trials; picks the winner |
| control | 20 trained cells, no distractor | 20 | what the distractor training cost on the old task; winner and P1 |
| confirm | 20 | 20 | touched once, on the winner; the only estimate not selected on |

Screen survivors: every checkpoint within 2 successes of the best. Report winner: the highest success count,
ties to the later checkpoint. The criteria (all on the report tier, winner vs P1): at least 70% overall, wrong
object at most 5%, trained-versus-held-out gap at most 10 points, and a better rate than P1 with
non-overlapping layout-clustered 95% intervals. Intervals resample layouts, not trials; the layout is the
independent unit.

## Parallelism

A worker is one process (about 4 GB GPU, about 4 CPU cores) running an interleaved share of a stage's
(checkpoint, trial) jobs, so every worker sees every checkpoint's mix of easy and hard trials and loads each
checkpoint once. Workers write disjoint folders `trial_<ordinal>_<layout>/` (`cameras.mp4`, `trace.npz` with
`props_xyz`, `episode.json`, `telemetry.json`) under `outputs/xvla_props_p2_eval/<mode>_<policy>_step_<N>/`.
`collect` merges them into `result.json` and is `complete` only if every planned trial exists and matches the
plan. A worker keeps finished trials and moves a half-written folder to `quarantine/`; a resumed trial equals an
uninterrupted one because each trial reseeds itself. A trial is never retried. Before each trial a worker pauses
while the GPU is at 83 C or above until it falls to 75 C.

## Failure classes

First cause, mutually exclusive: `success`, `never_lifted`, `wrong_object` (the first object lifted is not the
target), `wrong_bin`, `unstable_in_target`, `timeout_carrying`, `dropped_outside`. An object is lifted when its
centre rises 2 cm above its start and stays up 15 ticks; on the 540 sealed expert episodes an untouched
distractor rises at most 0.02 cm and a carried target 18-21 cm.

## Tests

```bash
docker run --rm --runtime runc --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=4g \
  --ulimit nofile=65536:65536 -e MUJOCO_GL=osmesa -e LP_NUM_THREADS=4 -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache \
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/vla -v "$PWD:/vla:ro" \
  -v "$PWD/outputs/props_p2_data:/vla/outputs/props_p2_data:ro" -w /vla --entrypoint python \
  sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e \
  -m unittest discover -s experiments/xvla_props_p2_eval/tests -t experiments/xvla_props_p2_eval/tests
```

The robosuite image runs all 45 (about 2.5 minutes, CPU only). The admission image runs the 40 that import no
simulator and skips the rest. The rollout tests drive the real loop with stub policies: a parked policy never lifts
anything; the scripted expert's own recorded actions replayed as the policy must succeed; and an expert that carries
a *distractor* to the bin must be classified `wrong_object` (a real physical event, not a fabricated trace). The
classifier is also checked against all 540 sealed expert traces before any rollout: the target is always the first
object lifted and no distractor ever is.
