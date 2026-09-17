# Agent handoff

Read [docs/BENCHMARK_STATUS.md](docs/BENCHMARK_STATUS.md) first, then the
design/plan/research it indexes. Read [CLAUDE.md](CLAUDE.md) for architecture;
its historical command defaults are not current authorization.

- Current status: the authorized 420-episode pilot completed and passed final
  validation on 2026-09-16 (300 train / 120 validation). User deferred the proxy-
  blocked upload and approved a fresh 20,000-update local SmolVLA training run,
  now complete (exit 0, all completion gates passed). See current status before
  any jobs; the 120-scene tuning evaluation is complete: 4/120 correct successes.
  The approved 48-rollout paired-instruction/training-scene diagnostic is also
  complete; see current results and obtain approval for the next learning design. Preserve
  bound sources. No further training, collection, uploads or final-scene evaluation.
  Never resume pre-fix checkpoints.
- Run Python and install dependencies only inside Docker, not host Conda.
- Preserve secrets and existing artifacts. Never display credential files or
  commit datasets, checkpoints, wheels, generated videos or model caches.
- Preserve the action/state contract and test observe-command-record alignment.
- Keep simulator ground truth in replay/evaluation metadata, not policy inputs.
- Distinguish measured findings from suggestions, and partial gates from passes.
- Update the status/report when blocked so another agent can continue safely.
- Commits must be solely authored by the configured user; no assistant
  co-authorship or Co-authored-by trailers. Keep messages short; no force push.