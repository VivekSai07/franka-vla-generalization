# Agent instructions

**Read [.github/copilot-instructions.md](.github/copilot-instructions.md).** It is the single
source of truth for working in this repository, and applies to every agent regardless of tool.

Additional context loaded automatically where supported:

| file | scope |
|---|---|
| [.github/instructions/experiments.instructions.md](.github/instructions/experiments.instructions.md) | `experiments/**` — contracts, bindings, shared helpers |
| [.github/skills/launching-a-run/SKILL.md](.github/skills/launching-a-run/SKILL.md) | starting training, admission, preflight, evaluation |
| [.github/skills/diagnosing-a-binding-failure/SKILL.md](.github/skills/diagnosing-a-binding-failure/SKILL.md) | admission / provenance / seal mismatches |

Orientation: [README.md](README.md) · [docs/INDEX.md](docs/INDEX.md) ·
[docs/OPERATIONS.md](docs/OPERATIONS.md) · [docs/FAILURES.md](docs/FAILURES.md) ·
[TODO.md](TODO.md)

## The three that matter most

1. Never edit a hash-bound file while a run is live — package READMEs are bound.
2. Never delete datasets, and never weaken a verification gate to make something pass.
3. Never claim something works without running it and reading the output.
