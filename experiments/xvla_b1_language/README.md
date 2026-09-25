# X-VLA fine-tune on the B1 language-grounding dataset

Fork of `experiments/xvla_broad_workspace_long/` — the run that reached 86.1% on the broad
holdout — repointed at the sealed parent
`outputs/b1_language_data/b1_language_20260925` and widened to two instructions.

## Why this package exists

The broad-workspace dataset made colour non-diagnostic: the picked cube was always at `y < 0`
and the placed cube always at `y > 0`, in 200/200 episodes (FAILURES #27). A policy that never
perceives colour and never reads the instruction could score 100%, so **no result to date
measures language grounding.** B1 decorrelates colour from position with a 2x2 factorial
(instruction x arrangement) and asks whether the policy follows the instruction.

## What differs from the fork source

**Two instructions.** The parent carries `pick up the {red,green} cube and place it on top of
the {green,red} cube`. Every single-task assumption inherited from the broad-workspace contract
is replaced by a per-episode instruction check driven by the plan:

| site | was | now |
|---|---|---|
| `indices` | `tasks == [TASK]` | `tasks == [planned_instructions()[i]]` |
| `selected_stats` | `task_index == 0` | `task_index == task_of_episode[i//528]` |
| `policy_row` | `row['task'] == TASK` | `row['task'] == planned_instructions()[episode]` |
| `load_data` | `total_tasks == 1` | `total_tasks == len(INSTRUCTIONS)` |
| `load_data` | `tasks.index == [TASK]` | `task_index_map` bijection over both |
| `load_data` | raw `task_index == 0` | per-episode expected index |

`task_index_map` reads the index assignment from the dataset rather than assuming it matches
the plan's sorted order, because **LeRobot numbers tasks by first appearance**.

**Doubled horizon: 33,400 updates (3.3 epochs), was 16,700 (1.67).** Three reasons:

1. B1 splits the same 160 training episodes across two instructions and four cells, so each
   task sees half the data the 86.1% run gave its single task.
2. That run was still climbing steeply at its endpoint — 70.8% at 12,000 to 86.1% at 16,700.
   The plateau was never located.
3. The failure mode to avoid is not a low score but an *uninterpretable* one. If the policy is
   incompetent, all four cells collapse toward the floor and "doesn't ground language" cannot
   be distinguished from "can't do the task".

The cosine schedule is stretched over the full horizon, so update 16,700 is mid-decay here and
is **not** numerically identical to the 86.1% run's endpoint. It remains a useful capability
comparison, not an exact replication.

## Checkpoints

`[1, 8000, 16700, 25000, 33400]` — disk-bound at ~3.4 GB each plus ~9.9 GB for the final one
(which carries optimizer state), against roughly 36 GB free once collection lands.

- `16700` — matched exposure; comparability with 86.1% and a safety net.
- `25000`, `33400` — the longer candidates.
- `8000` — where the broad-workspace run collapsed to 0/72. A reproducible mid-training collapse
  is the reason this run checkpoints at all rather than trusting its endpoint.
- `1` — checkpoint machinery sanity.

Evaluation screens `16700 / 25000 / 33400` on 40 of the 80 held-out layouts, then reports the
grounding contrast on the untouched other 40. Offline metrics cannot be used for selection:
MAE improved monotonically straight through the 0/72 collapse (FAILURES #21).

## Parent binding

`SEAL` is `PENDING_B1_ADMISSION` in source and is injected by `b1_followup.sh` once admission
seals the parent. `current_binding()` refuses to run while the placeholder is present.

Note the limitation: a seal the chain read from the artifact it is verifying cannot detect
corruption that predates injection. It pins the dataset against change *after* training starts,
which is its actual purpose. The full semantic audit
(`b1_language_data.validate.require_broad_workspace_pass`) is what establishes correctness.

Consequence: `experiments/b1_language_data/**/*.py` is frozen for this run's lifetime.

## Usage

`experiments.xvla_b1_language.runner` accepts exactly one of `--check`, `--preflight`, `--train`.
The chain runs `--check` then `--preflight` (2 updates) and only launches `--train` if both pass,
so a fork bug costs minutes rather than the full 7.7-hour run.

Outputs:

- `outputs/train/xvla_franka_stacking_b1_language_v1_20260925_admission/`
- `outputs/train/xvla_franka_stacking_b1_language_v1_20260925_preflight/`
- `outputs/train/xvla_franka_stacking_b1_language_v1_20260925/`
