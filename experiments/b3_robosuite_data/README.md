# b3_robosuite_data

Collection of the B3 cube dataset in the robosuite scene: 500 training and 60 validation scripted
episodes (20 cells of colour pair x side, distractors none/one/two, edge-biased picks) plus 60
reserved evaluation layouts banked but never collected. Design:
[docs/superpowers/specs/2026-10-03-b3-robosuite-cubes-design.md](../../docs/superpowers/specs/2026-10-03-b3-robosuite-cubes-design.md);
plan: [docs/superpowers/plans/2026-10-03-b3-robosuite-collection.md](../../docs/superpowers/plans/2026-10-03-b3-robosuite-collection.md).

**Hash-bound once sealed.** `contract.provenance()` hashes every `.py` file here (tests included),
this README, `experiments/robosuite_scene/*.py`, the shared modules it imports, `src/`, the spec and
the plan. Editing any of them after `--collect` starts aborts the run; editing them afterwards
invalidates the seal. Fork the package, do not edit it.

## Images

| stage | image |
|---|---|
| `test_plan`, `test_contract`, `test_expert_fork`, `test_record`, `test_validate` | admission (`contract.ADMISSION_IMAGE`, MuJoCo 3.13.0) |
| `test_scenes`, `test_expert`, every mode of `record` | robosuite (`contract.IMAGE`, MuJoCo 3.4.0), CPU |

`plan.py` and `contract.py` import no simulator, so a training contract can import them in the
admission image.

## Files

| file | content |
|---|---|
| `plan.py` | every count, seed, layout, episode and reserved layout; `make_plan`, `validate_plan`, `attempt_draw`, `resample_layout` |
| `scenes.py` | `build_model` (cached per colour set), `reset`, `validate_geometry`, `make_bank` |
| `expert.py` | `B3Expert` (`run` copied from the B2 expert, AST-checked) and `run_attempt` |
| `validate.py` | `admit` (six gates), `validate_dataset`, canonical-JSON `fingerprint`, `seal` |
| `record.py` | scheduler, journal and the `--pilot` / `--collect` / `--resume` / `--status` modes |
| `contract.py` | image digests and `provenance()` |

## Collection rules

- Frames are buffered in memory and written to the dataset only after the episode is admitted.
- A rejected attempt retries the same slot (attempt `a`: layout `a // 3`, expert draw `a % 3`).
  Three failed attempts move to a deterministic substitute layout of the same slot (cell, profile,
  edge class, colours, instruction), up to three substitutions; the result is in
  `substitutions.json`. A slot that fails all 12 attempts raises `SlotExhausted`.
- 150 rejected attempts in total is the budget; the 151st raises `BudgetExhausted`.
- Dataset episodes are one file each and the dataset is finalized after every admitted episode, so
  a kill between episodes leaves a readable dataset.
- Admission gates: `expert_success`, `frames`, `stack`, `distractors_undisturbed` (no distractor
  moved more than 2 cm), `replay` (state error < 1e-3) and `labels`. Offline metrics play no part.

## Running

From the worktree root (see the plan's "Running things" for the `run` helper):

```bash
run "$OLD" -m unittest experiments.b3_robosuite_data.tests.test_plan -v
run "$NEW" -m experiments.b3_robosuite_data.record --pilot --slots 0 3 500 --out /out/<scratch>
run "$NEW" -m experiments.b3_robosuite_data.record --collect --out /out/b3_robosuite_<YYYYMMDD>
run "$NEW" -m experiments.b3_robosuite_data.record --resume  --out /out/b3_robosuite_<YYYYMMDD>
run "$NEW" -m experiments.b3_robosuite_data.record --status  --out /out/b3_robosuite_<YYYYMMDD>
```

`--collect` refuses an existing output directory. `--resume` refuses unless the dataset's episode
count equals the journal's admitted count and `plan.json` / `provenance.json` match fresh values.
The `--pilot` output is scratch and never sealed.

## Not comparable

This scene runs on robosuite 1.5.2 and MuJoCo 3.4.0. Numbers from it are not physics-comparable
with the sealed MuJoCo 3.13.0 datasets.
