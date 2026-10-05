# Agent instructions

**Read [.github/copilot-instructions.md](.github/copilot-instructions.md).** It is the single
source of truth for working in this repository, and applies to every agent regardless of tool.

Additional context loaded automatically where supported:

| file | scope |
|---|---|
| [.github/instructions/experiments.instructions.md](.github/instructions/experiments.instructions.md) | `experiments/**` — contracts, bindings, shared helpers |
| [.github/skills/launching-a-run/SKILL.md](.github/skills/launching-a-run/SKILL.md) | starting training, admission, preflight, evaluation |
| [.github/skills/diagnosing-a-binding-failure/SKILL.md](.github/skills/diagnosing-a-binding-failure/SKILL.md) | admission / provenance / seal mismatches |
| [.github/skills/collecting-a-dataset/SKILL.md](.github/skills/collecting-a-dataset/SKILL.md) | `--preflight`/`--collect`, rejection budget, resume/rollback |
| [.github/skills/training-a-policy/SKILL.md](.github/skills/training-a-policy/SKILL.md) | contract/runner design, checkpoint schedule, disk guards, comparing runs |
| [.github/skills/evaluating-a-checkpoint/SKILL.md](.github/skills/evaluating-a-checkpoint/SKILL.md) | screening vs report, holdout independence, checkpoint selection, ties |
| [.github/skills/diagnosing-policy-failures/SKILL.md](.github/skills/diagnosing-policy-failures/SKILL.md) | explaining failed rollouts: layout, feasibility, mechanism, validating measures |
| [.github/skills/forking-an-experiment-package/SKILL.md](.github/skills/forking-an-experiment-package/SKILL.md) | copying a package — residue traps, symtable check |
| [.github/skills/branching-for-an-experiment-axis/SKILL.md](.github/skills/branching-for-an-experiment-axis/SKILL.md) | one branch per axis, safe recovery if work landed on `main` |
| [.github/skills/transferring-artifacts-off-rtx/SKILL.md](.github/skills/transferring-artifacts-off-rtx/SKILL.md) | stage/transfer/verify/publish/reclaim disk for datasets & checkpoints |
| [.github/skills/working-on-the-robosuite-path/SKILL.md](.github/skills/working-on-the-robosuite-path/SKILL.md) | `experiments/robosuite_*` and `experiments/props_scene` — which image, crossing between the two MuJoCo versions, the traps hit so far |

Orientation: [README.md](README.md) · [docs/INDEX.md](docs/INDEX.md) ·
[docs/OPERATIONS.md](docs/OPERATIONS.md) · [docs/FAILURES.md](docs/FAILURES.md) ·
[TODO.md](TODO.md)

## The three that matter most

1. Never edit a hash-bound file while a run is live — package READMEs are bound.
2. Never delete datasets, and never weaken a verification gate to make something pass.
3. Never claim something works without running it and reading the output.
