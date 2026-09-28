---
name: branching-for-an-experiment-axis
description: "Use when starting work on a new experiment axis (a new *_data/*_eval package, a new dataset variation) or when that work already landed directly on main and needs to move to a branch."
---

# Branching for an experiment axis

Each new experiment axis (language grounding, distractors, ...) gets its own branch, merged into
`main` once the axis has a measured result. Work directly on `main` only for housekeeping (docs,
TODO, FAILURES) that isn't part of the axis itself.

## 1. Branch before the first commit of a new axis

```bash
git switch -c feat/<axis-name>
```

Name it after the axis, not the package (`feat/b2-distractors`, not `feat/b2-distractor-data`) —
one axis can span a collection package, a training fork, and an eval fork.

## 2. If work already landed on `main` unbranched

This is recoverable **without touching any file**, as long as the commits are not yet pushed
(`git log --oneline origin/main..main` to confirm first). Branch pointers and `HEAD` can move
independently of the working tree:

```bash
git branch feat/<axis-name>          # ref only; same tree as HEAD
git switch feat/<axis-name>          # HEAD moves; tree is identical, so no files change
git branch -f main origin/main       # main ref only; main is not checked out, so no files change
```

**While any job has this directory mounted, do not run *any* further `git checkout`/`switch`/
`reset` in this checkout for any reason, full stop — not "only the destructive ones".** The three
commands above are safe only because, at that exact moment, the destination tree is provably
identical to what's already checked out. That fact does not persist: the very next branch
operation (e.g. switching back to `main` for an unrelated commit) checks out a *different* tree
and silently deletes/reverts whatever the job depends on — this has actually happened, mid-run, to
this exact job (FAILURES #34). If you need to commit to a different branch while a job is live,
use a separate `git worktree` instead of switching this checkout.

Verify after any ref-only move: `git status --short` is unchanged, and the job's next status poll
still succeeds.

**Do not** do this by checking out `main` and running `git reset --hard origin/main` — that
rewrites the working-tree files main was pointing at, which is exactly the kind of mid-run source
edit that [diagnosing-a-binding-failure](../diagnosing-a-binding-failure/SKILL.md) warns kills a
live job at its next re-hash.

## 3. Commits already pushed

If `git log --oneline origin/main..main` is empty for the commits in question, they are already
shared history. Moving them off `main` now needs a force-push and rewrites a public branch — that
requires the user's explicit approval; do not do it as part of routine cleanup.

## 4. Merging back

Once the axis has a measured result (or is explicitly abandoned), merge into `main` with a merge
commit naming the axis, and update the live docs (`TODO.md`, `docs/RESULTS.md`,
`docs/FAILURES.md`) in the same pass — see the Documentation section of
[.github/copilot-instructions.md](../../copilot-instructions.md).
