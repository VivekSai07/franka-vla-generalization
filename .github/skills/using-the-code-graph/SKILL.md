---
name: using-the-code-graph
description: "Use when navigating an unfamiliar package, tracing what depends on a shared helper before editing it, onboarding a human, or when asked to generate, refresh, view or commit the Understand Anything graph in .ua/."
---

# Using the code graph

`.ua/knowledge-graph.json` is a generated map of the code and docs on `main`: files, functions,
classes, imports/calls, architectural layers and a guided tour. It comes from the
[Understand Anything](https://github.com/Egonex-AI/Understand-Anything) plugin (tree-sitter for
structure, an LLM for summaries). It is a **navigation aid**, not a source of truth.

## 1. Using it (agents and humans)

- **Agents:** never read `.ua/knowledge-graph.json` whole (2 MB, about 540k tokens). Query it:
  `python3 scripts/code_graph.py status | layers | show PATH | find TEXT`. Use it for **orientation**
  (`layers`) and **triage** (`show` a file instead of opening it), where it measured well. Do not use it
  for "who imports X": use `git grep` (see the limits below). Then read the real files.
- **Humans:** `npx https://github.com/Egonex-AI/Understand-Anything/releases/download/v2.9.0/understand-anything-viewer.tgz .`
  from the repo root (Node >= 18; no LLM, no API key; serves `.ua/` read-only on 127.0.0.1). The URL is
  pinned to a release tag on purpose — `releases/latest/download/` would run whatever was published last.
- **Staleness:** `.ua/meta.json` holds `gitCommitHash`. If `git log <hash>..main -- experiments src`
  is non-empty the graph is behind. Say so rather than trusting it.
- **No graph, or branch work:** the graph describes `main` only. Unmerged axes live in worktrees
  (`git worktree list`) and are absent from it. Fall back to grep; do not conclude a package is missing.

## 2. What it must never be used for

- **Not evidence.** Node summaries are LLM-written and can be wrong. A number, a pass/fail or a
  cause comes from `docs/RESULTS.md` and the code you read, never from a summary.
- **Does not know about hash-binding.** The graph cannot tell you whether a file is in
  `SOURCE_FILES` or reachable from `provenance()`. Before editing a shared helper, run
  `/understand-diff` for the ripple, then check bindings yourself
  ([experiments.instructions.md](../../instructions/experiments.instructions.md)).
- Do not copy graph text into live docs as if it were a finding.
- **Incomplete.** Measured 2026-10-21 by `scripts/code_graph_bench.py` over 14 modules with a repo-unique
  name: median import recall 63% (min 0%, max 100%), 5 of 14 had no importer found at all, precision
  100%. Function-level `calls` edges across files are mostly absent (162 dropped at merge). So `importers`
  is a lower bound at best, and "nothing imports this" is never proof: grep before you conclude, and
  always grep before editing a shared helper ([FAILURES.md](../../../docs/FAILURES.md) #63).
- **Measured gains** (same script, tokens approximated as bytes/4, a proxy and not a session
  measurement): `layers` 636 tokens against 3,360 for `ls` plus README heads (5.3x); `show` returns a
  median 151 tokens where the file is 1,098 (about 968 saved per file not opened, if the summary is
  enough to decide); concept `find` was cheaper for 2 of 4 queries and costlier for 2. The always-loaded
  cost of this integration is about 126 tokens per session.

## 3. Generating and refreshing (manual only)

Needs Node >= 22 and pnpm >= 10 on the host (`node --version`, `pnpm --version`). This is tooling for
reading the repo; it is not an experiment and runs outside the two pinned images. On the RTX machine
they live in a dedicated conda env, never in `base`:
`conda create -n ua-node -c conda-forge --override-channels nodejs=24 pnpm=10`, then
`conda activate ua-node` before `/understand`.

1. **Refresh in a throwaway worktree cut from `main`**, never by switching a checkout:
   `git worktree add .worktrees/graph-refresh -b chore/graph-refresh main`, then run with
   `UNDERSTAND_NO_WORKTREE_REDIRECT=1`. Without it, from a worktree the plugin silently switches to the
   main checkout, which is usually on a feature branch here, and writes `.ua/` there for the wrong tree.
2. **Scope with `.ua/.understandignore`** (committed; gitignore syntax). It must exclude `outputs/`,
   `datasets/`, `huggingface/`, `publication/` and `.worktrees/`. The first full run pauses for you to
   confirm this file. Review it; do not accept the starter blindly.
3. **Never `--auto-update`.** The post-commit hook would spend tokens on every commit and rewrite
   `.ua/` in the middle of unrelated work. Keep `autoUpdate` false in `.ua/config.json`.
4. Refresh with plain `/understand` (incremental: only files changed since `.ua/meta.json`'s commit are
   re-analysed). Refresh after an axis merges into `main`, not during one. `scripts/code_graph.py
   status` says when it is due.
   **Above 30 structurally changed files (or 50% of the repo) the plugin refuses to go incremental** and
   demands a full rebuild (hardcoded in its `change-classifier.ts`; `finalize-incremental.mjs` rejects
   such a plan). One axis such as P3 crosses 30, so refresh before then. If you are already past it, the
   cheaper path, used on 2026-10-30 for 54 files (14 batches, 12k lines, against 682 files and 112k
   lines for a rebuild), is `python3 scripts/code_graph_patched_plugin.py` (builds `/tmp/ua-patched`: the
   real plugin by symlink, with a 5-line patch that turns `FULL_UPDATE` into `ARCHITECTURE_UPDATE`; the
   install is untouched) and then the incremental steps by hand with that copy of `prepare-incremental.mjs`:
   `prepare-incremental.mjs <root> <meta hash>`, `compute-batches.mjs <root> --changed-files=.ua/intermediate/changed-files.json`,
   one analyzer per batch, `merge-batch-graphs.py <root>`, then `node scripts/code_graph_cover.cjs <root>`
   (every changed file has a node, nothing dangles, unchanged nodes identical; it compares against the
   old graph, so run it before finalizing), layers and tour passes keeping the previous
   ones as the base, `finalize-incremental.mjs <root>` (the real one), then
   `node scripts/code_graph_verify.mjs <root>`. Symbol-loss validation and the schema check still run.
   Fall back to a full `/understand --full` if verification fails.
5. The first full run is token-heavy. Say so before starting; do not start it unprompted.

## 4. Committing it

- Commit `.ua/` **except** `intermediate/`, `tmp/`, `.trash-*/` and `diff-overlay.json` (gitignored).
  Over 10 MB: track `.ua/*.json` with git-lfs.
- A refresh is its own commit ("Refresh code graph"), never mixed with experiment, doc or contract
  changes. `.ua/` is outside every `SOURCE_FILES`; if a contract ever globs the repo root, re-check.
- Never edit `.ua/*.json` by hand.

## 5. What is automatic and what is not

- **Automatic, zero tokens:** staleness detection. `scripts/code_graph.py status` counts analysed files
  changed since `.ua/meta.json`'s commit (stale from 10). Agents run it first. The opt-in git hook
  `.githooks/post-merge` prints it after a merge into `main`; enable it once per clone with
  `git config core.hooksPath .githooks` (applies to every worktree; it only prints, never writes or
  blocks, and no other hooks exist to override).
- **Not automatic, by decision:** the refresh itself. It needs an LLM, a git hook cannot start one, and the
  plugin's own `--auto-update` is a Claude Code hook that makes the model rewrite `.ua/` unprompted after
  every `git commit|merge|cherry-pick|rebase` (checked in its source), which would spend tokens and
  touch `.ua/` mid-experiment, and never fires in Copilot. Revisit only with a way to bound the cost.
- A refresh costs about the number of files `status` reports: 18 files cost 9 small analyzer batches,
  a layers pass and a tour pass; 54 files cost 14 batches plus the same two passes. The deterministic
  prepare/merge/save steps cost no tokens.
- Benchmark: `python3 scripts/code_graph_bench.py` reproduces every figure above.

## 6. Installing the plugin

Never `curl ... | bash`. Clone the repo, read `install.sh`, then run it. On 2026-10-21 `install.sh`
(plugin commit `b8c807c`) was reviewed and run for `vscode`: it only clones to
`~/.understand-anything/repo` and symlinks skills into `~/.copilot/skills` plus
`~/.understand-anything-plugin`; nothing inside this repo. Re-read it after `--update`.
The first `/understand` also runs `pnpm install` inside that clone (network).
