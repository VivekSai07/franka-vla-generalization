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

- **Agents:** before grepping across `experiments/`, read `.ua/knowledge-graph.json` (or ask
  `/understand-chat`) to find the package, its layer and what imports it. Then read the real files.
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

## 3. Generating and refreshing (manual only)

Needs Node >= 22 and pnpm >= 10 on the host (`node --version`, `pnpm --version`). This is tooling for
reading the repo; it is not an experiment and runs outside the two pinned images.

1. **Run from a checkout whose HEAD is `main`, with the redirect disabled:**
   `UNDERSTAND_NO_WORKTREE_REDIRECT=1`. Without it, from a worktree the plugin silently switches to the
   main checkout, which is usually on a feature branch here, and writes `.ua/` there for the wrong tree.
2. **Scope with `.ua/.understandignore`** (committed; gitignore syntax). It must exclude `outputs/`,
   `datasets/`, `huggingface/`, `publication/` and `.worktrees/`. The first full run pauses for you to
   confirm this file. Review it; do not accept the starter blindly.
3. **Never `--auto-update`.** The post-commit hook would spend tokens on every commit and rewrite
   `.ua/` in the middle of unrelated work. Keep `autoUpdate` false in `.ua/config.json`.
4. Refresh with plain `/understand` (incremental, only changed files). Refresh after an axis merges
   into `main`, not during one.
5. The first full run is token-heavy. Say so before starting; do not start it unprompted.

## 4. Committing it

- Commit `.ua/` **except** `intermediate/`, `tmp/`, `.trash-*/` and `diff-overlay.json` (gitignored).
  Over 10 MB: track `.ua/*.json` with git-lfs.
- A refresh is its own commit ("Refresh code graph"), never mixed with experiment, doc or contract
  changes. `.ua/` is outside every `SOURCE_FILES`; if a contract ever globs the repo root, re-check.
- Never edit `.ua/*.json` by hand.

## 5. Installing the plugin

Never `curl ... | bash`. Clone the repo, read `install.sh`, then run it. As of 2026-10-21 it only
clones to `~/.understand-anything/repo` and symlinks skills into `~/.copilot/skills` (for `vscode`)
plus `~/.understand-anything-plugin`; nothing inside this repo. Re-read it after `--update`.
The first `/understand` also runs `pnpm install` inside that clone (network).
