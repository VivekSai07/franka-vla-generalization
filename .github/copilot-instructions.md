# Working in this repository

Single source of agent truth. `AGENTS.md` and `CLAUDE.md` point here.

This is a research repository where **results must be reproducible and provable**. Experiments are
hash-bound: datasets carry seals, training runs record bindings over their source files, and a run
refuses to start if it cannot reproduce what it was built against. Most rules below exist because
violating them destroyed hours of compute. See [docs/FAILURES.md](../docs/FAILURES.md).

## Non-negotiable

- **Never edit a hash-bound file while a run is active.** `SOURCE_FILES` and `provenance()` reach
  into package READMEs and shared helpers. A finished 3-hour evaluation has been killed at its
  final gate by a one-line README edit.
- **Never delete datasets** or anything under `datasets/`. Deleting the pilot manifest permanently
  broke `local_varied` provenance — it is still unfixable.
- **Never relax a verification gate to make something pass.** If a gate fails, either the code is
  wrong or the claim is wrong. Weakening a check to get a green result is the one unrecoverable
  mistake here.
- **Never force-push, never amend published commits, never discard user changes.**
- Commits are authored solely by the user. No AI or co-author trailers. Short, simple messages.
- No secrets, datasets, checkpoints or generated videos in commits.

## Evidence before claims

- Never say something passed without running it and reading the output.
- When a test is added for a bug, **confirm it fails against the unfixed code** (`git stash`).
  A green suite has four times coexisted with broken production code here, because the tests
  exercised a fixture or a constant rather than the path that actually runs.
- Report the number you measured, not the number you expected. Negative results are results.
- State caveats next to figures, not only in commit messages: confounds, weakened bindings,
  protocols that are not comparable.

## Before launching any long run

Work through the checklist in [TODO.md](../TODO.md#before-launching-any-long-run--checklist).
Each item cost real GPU time at least once. Highlights:

- package README updated — it is in `SOURCE_FILES`
- no constant duplicated between `contract.py` and `runner.py`
- `--check` and `--preflight` before `--train`; editing any `SOURCE_FILES` member after `--check`
  invalidates it
- all gitignored mounts present: `outputs`, `datasets`, `.cache`

## Monitoring

`docker wait` has silently failed to notify in this environment. Poll actively; never treat
"launched" as "will report". Runs have sat dead for 45+ minutes this way.

## Experiment design

- Change **one variable at a time**. A failure with two new axes is not attributable.
- Hold **exposure (epochs) constant**, not update count. Holding updates fixed while the dataset
  grew silently changed exposure by 3.3× and nearly produced a false conclusion.
- A result is uninterpretable without knowing what the *absence* of the treatment scores on the
  same protocol. Run the cross-evaluation.
- Do not move a gate after seeing the result.

## Documentation

- Update docs at each phase boundary, not in a batch at the end.
- Live documents are `ALL_CAPS.md`; dated reports are immutable records — never rewrite them to
  reflect later findings.
- Conventions and file map: [docs/INDEX.md](../docs/INDEX.md).

## Operations

Container flags, mounts, gate sequence, storage costs: [docs/OPERATIONS.md](../docs/OPERATIONS.md).
Nothing runs outside the digest-pinned image.
