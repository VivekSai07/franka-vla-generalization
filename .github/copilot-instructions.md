# Working in this repository

Single source of agent truth. `AGENTS.md` and `CLAUDE.md` point here.

This is a research repository where **results must be reproducible and provable**. Experiments are
hash-bound: datasets carry seals, training runs record bindings over their source files, and a run
refuses to start if it cannot reproduce what it was built against. Most rules below exist because
violating them destroyed hours of compute. See [docs/FAILURES.md](../docs/FAILURES.md).

## Orientation

Do not reconstruct history or current numbers from this file. Read [TODO.md](../TODO.md) for where
things stand, [docs/BENCHMARK_STATUS.md](../docs/BENCHMARK_STATUS.md) newest-first for the log, and
[docs/RESULTS.md](../docs/RESULTS.md) for every quoted figure. The map of live docs versus immutable
dated reports is [docs/INDEX.md](../docs/INDEX.md).

`main`'s `experiments/` is not the whole code. Unmerged axes live on their own branches and
worktrees — `git worktree list` before concluding a package is missing, and `docker ps` before
editing anything a run might hash. A live container may mount a worktree, not this checkout.

`.ua/` holds a generated code graph of `main` for navigation. Query it with
`python3 scripts/code_graph.py` (never read the 2 MB JSON whole). It is incomplete, not evidence, and
does not know what is hash-bound; rules in
[skills/using-the-code-graph/SKILL.md](skills/using-the-code-graph/SKILL.md).

## Non-negotiable

- **Never edit a hash-bound file while a run is active.** `SOURCE_FILES` and `provenance()` reach
  into package READMEs and shared helpers. A finished 3-hour evaluation has been killed at its
  final gate by a one-line README edit.
- **Never `git checkout`, `switch`, or `reset` in a directory a container has bind-mounted.**
  Whether the destination tree matches does not persist to the next command. Use
  `git worktree add` instead. This crashed a live collection (FAILURES #34). The only copy of the
  procedure is [skills/branching-for-an-experiment-axis/SKILL.md](skills/branching-for-an-experiment-axis/SKILL.md).
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
  A green suite has five times coexisted with broken production code here, because the tests
  exercised a fixture, a constant, or the wrong module rather than the path that actually runs.
- **Offline metrics are not capability.** Loss, action MAE and gripper accuracy improved
  monotonically through a checkpoint that scored 0/72 on rollouts. Never select a checkpoint,
  stop early, or claim progress from them.
- Report the number you measured, not the number you expected. Negative results are results, and
  a refuted hypothesis is worth recording with the evidence that killed it.
- **Check how many independent units a statistic was computed over before calling it tight.** A
  spread pooled across groups describes the groups, not the units inside them — a "2 mm consistent"
  offset turned out to be a between-layout number masquerading as within-layout precision.
- State caveats next to figures, not only in commit messages: confounds, weakened bindings,
  protocols that are not comparable.

## Before launching any long run

Work through the checklist in section 1 of
[skills/launching-a-run/SKILL.md](skills/launching-a-run/SKILL.md) — the only copy. Each item cost
real GPU time at least once.

## Monitoring

`docker wait` has silently failed to notify in this environment. Poll actively; never treat
"launched" as "will report". Runs have sat dead for 45+ minutes this way.

## Experiment design

- Change **one variable at a time**. A failure with two new axes is not attributable.
- Hold **exposure (epochs) constant**, not update count. Holding updates fixed while the dataset
  grew silently changed exposure by 3.3× and nearly produced a false conclusion.
- To compare two runs mid-training, match **schedule phase**, not step count — the same update
  number meant 80% annealed in one run and 24% in another.
- A result is uninterpretable without knowing what the *absence* of the treatment scores on the
  same protocol. Run the cross-evaluation.
- Do not move a gate after seeing the result.

## Documentation

**Every finding triggers a documentation pass.** Whenever you produce a result, learn something,
refute a hypothesis, or make a mistake, immediately ask which documents now contain stale claims —
and fix them in the same turn, not later. Findings and docs ship together.

The pass, every time:
1. Grep the live docs for the **specific numbers and claims** your finding changes
   (`grep -rn "<old figure>" README.md docs/*.md TODO.md .github/`).
2. Update every hit. A number corrected in one document and left standing in three others is worse
   than not correcting it, because the stale copies still read as current.
3. When a claim is **superseded rather than simply wrong**, say so explicitly and keep the history:
   state what was believed, what replaced it, and why the first reading was mistaken.
4. Add the mistake to [FAILURES.md](../docs/FAILURES.md) if it could recur — including your own
   analysis errors, not just pipeline failures.
5. Re-check links after moving or renaming anything.

Also:
- Update docs at each phase boundary, not in a batch at the end.
- Live documents are `ALL_CAPS.md`; dated reports are immutable records — never rewrite them to
  reflect later findings. Correct the live documents instead.
- Conventions and file map: [docs/INDEX.md](../docs/INDEX.md).

## Operations

Container flags, mounts, gate sequence, storage costs: [docs/OPERATIONS.md](../docs/OPERATIONS.md).
Nothing runs outside one of two digest-pinned images.

There are two images. The admission image (MuJoCo 3.13.0) runs everything sealed, B3 training
(`experiments/xvla_b3_robosuite/`, no simulator needed) and every test that imports no simulator. A
second image (robosuite 1.5.2, MuJoCo 3.4.0) runs everything that builds the robosuite scene: `experiments/robosuite_scene/`, the rollout stage of
`experiments/robosuite_sim2sim/`, B3 collection (`experiments/b3_robosuite_data/`) and B3 evaluation
rollouts (`experiments/xvla_b3_robosuite_eval/`). Numbers from the two
are not physics-comparable. Digests, build notes and which runs need `--ulimit nofile=65536:65536` or a
large `/tmp`: OPERATIONS.md.

- **A version pin is proven by running the thing, not by importing it.** robosuite 1.5.2 imports on
  MuJoCo 3.13.0 and cannot reset an environment there (FAILURES #39).
- **When a check reports failure, confirm the physical state before tuning.** A "grasp failed"
  verdict stood while the cube was held 0.21 m in the air; the check was blind (FAILURES #40).
