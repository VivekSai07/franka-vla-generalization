---
name: diagnosing-a-binding-failure
description: "Use when a run aborts with a binding, provenance, admission or seal mismatch — such as 'Current full CPU admission required', 'provenance changed', or a source/weight hash mismatch."
---

# Diagnosing a binding failure

A binding failure means the code or data now differs from what the run was admitted against.
**The check is correct.** Never weaken or bypass it — diagnose what drifted.

## Symptoms

- `Current full CPU admission required`
- `provenance changed` / `semantic.get('provenance') != current`
- source file, dataset seal, or model weight sha256 mismatch
- a job that ran for hours failing at its *final* verification step

## 1. Identify which surface drifted

Binding covers three independent things. Read the error to see which:

| surface | source |
|---|---|
| source files | `SOURCE_FILES` in the contract — explicit tuple, **includes the package README** |
| dataset semantics | dataset package `provenance()` — `rglob('*.py')` + its own README |
| weights / data | seal and checkpoint sha256 |

## 2. Diff stored against fresh

```python
import json, importlib
from pathlib import Path
c = importlib.import_module('experiments.<pkg>.contract')
stored = json.loads(Path('<output>/admission.json').read_text())
fresh  = c.data_binding()          # or c.current_binding()
for k in fresh:
    if stored['binding'].get(k) != fresh[k]:
        print(k, stored['binding'].get(k), '->', fresh[k])
```

For per-file resolution, hash each `SOURCE_FILES` entry and compare against the recorded map.

## 3. Attribute the change

```bash
git log --oneline -5 -- <drifted file>
git status --short
```

Three common causes, in order of frequency:

1. **A README was edited.** READMEs are bound. This is the single most common cause — a one-line
   documentation edit has killed a completed 3-hour evaluation.
2. **A shared helper was edited** — `experiments/xvla_single_task/contract.py` is bound by four
   contracts; `data_collection/`, `validation/`, `src/` are bound by many.
3. **A stale worktree** — the mounted directory is at an older commit than expected. Check
   `git log --oneline -1` *inside the mounted path*, not the main checkout.

## 4. Choose a remedy

| situation | remedy |
|---|---|
| edit was unnecessary | revert it, re-run |
| edit is needed and no run is live | make it, then re-run `--check` in full (~1h) |
| edit is needed but a run is live | **wait** — do not touch the file until it finishes |
| run needs a clean tree while you keep working | run from a dedicated `git worktree` at the pinned commit, and mount that path |
| the binding is genuinely unreproducible (e.g. the dataset was re-sealed after the run) | write a **weakened, explicit** binding — see below |

## 5. Weakened bindings

Only when reproduction is impossible for a recorded reason — for example a run that predates a
re-seal. Then:

- recompute whatever *can* be verified independently (e.g. fresh train-only statistics) and require
  equality with the values the original admission recorded
- make the weakening explicit in the returned binding:
  `{'metadata_binding_reproducible': False}`
- state it beside every number the run produces, not only in the commit message

Precedent: `experiments/smolvla_localvaried_broad_eval/contract.py`.

## 6. Prevention

- Finish all edits — including READMEs — before `--check`.
- Override shared helpers locally rather than editing them.
- Know what is bound before typing: see
  [.github/instructions/experiments.instructions.md](../../instructions/experiments.instructions.md).
