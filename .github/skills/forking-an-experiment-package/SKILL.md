---
name: forking-an-experiment-package
description: "Use when copying an experiment package (collection, training, or eval) to start a new one — covers the residue traps that let a green test suite pass while testing the wrong code."
---

# Forking an experiment package

Every package here started as a copy of another. The fork itself is not the risk; residue from
the source package silently surviving the copy is, because tests can pass while exercising code
that no longer runs.

## 1. Grep for the source package's name

Copied `import experiments.<source_pkg>...` lines, `repo_id='<source_pkg>'` literals, and test
docstrings naming the old package are the most common residue. Forked tests have imported the
*source* module instead of the new one three separate times in this repo's history, passing
26-68 "green" tests while testing the wrong code, because nothing forced them to import the fork.

## 2. Grep for duplicated constants, not just names

Anything that also lives in the new package's `recipe()`/`plan()` — update horizon, checkpoint
schedule, output paths, `EXPERT_VERSION` — must be **derived from that one source**, not
re-typed. A runner that hardcoded `maximum = 2 if preflight else 5000` while its contract's
`recipe()` said 16700 trained to the wrong horizon and only failed afterwards, at the
`verify_checkpoints` gate — after the GPU time was already spent.

## 3. Run the symtable dead-name check

A fork that deletes a code path can leave a reference to it behind (e.g. an `if` branch testing a
flag the fork's rewrite removed), which raises `NameError` only when that exact branch executes —
often past every test, and after the run has already claimed the GPU and created its output
directory, which then blocks a retry under the no-overwrite rule. Before spending GPU time on a
new fork, run a static scan asserting no function reads a name that is never bound anywhere in the
module (Python's `symtable`) — it catches this class of bug in milliseconds, with no rollout
needed.

## 4. Rebind literals through a context, not a shared-file edit

If the fork varies something the shared validator hardcodes (a colour pair, a task index), do not
edit the shared file — it is very likely inside another package's sealed `provenance()` or
`SOURCE_FILES`. Rebind the module globals it reads through a scoped context manager instead, and
test that every patched global is restored, including when the body raises.

## 5. Confirm a new regression test would actually have caught the bug

When adding a test for a fork-hygiene bug, confirm it **fails** against the pre-fix file
(`git stash` the fix, rerun). A test that imports the wrong module, patches the wrong symbol, or
never drives the code path with the bug will pass either way — which is not evidence.

## See also

[.github/instructions/experiments.instructions.md](../../instructions/experiments.instructions.md)
for what is hash-bound before you edit anything,
[branching-for-an-experiment-axis](../branching-for-an-experiment-axis/SKILL.md) for putting the
fork on its own branch.
