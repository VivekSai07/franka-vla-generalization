# Handoff to a new session: props P3 (Food vs Drinks), written 2026-10-19

**How to use.** Paste the prompt below into a new Copilot session in this repository. Everything after it is the context the
prompt points to. Numbers here are a convenience copy: **the repo's rule is to read the live documents, not reconstruct
history**, and a number copied here may have gone stale. Verify against [TODO.md](../../../TODO.md),
[docs/RESULTS.md](../../RESULTS.md) and the reports it links.

## The prompt

> You are continuing a research project in `/home/utxowqr/smolvla_franka` (comparing how much data a VLA needs; currently X-VLA
> sorting real-life objects into bins in a robosuite scene). Before anything else read, in this order:
> `.github/copilot-instructions.md`, `AGENTS.md`, `CLAUDE.md`, `TODO.md`, `docs/BENCHMARK_STATUS.md` (newest first),
> `docs/RESULTS.md`, `docs/FAILURES.md` (all of it: most rules exist because violating one destroyed hours of compute), then
> `docs/superpowers/handoffs/2026-10-19-props-p3-handoff.md` (this file), the P3 spec
> `docs/superpowers/specs/2026-10-19-props-p3-food-drinks-design.md` and plan
> `docs/superpowers/plans/2026-10-19-props-p3-collection.md`, and the reports
> `docs/PROPS_P2_EVALUATION_20261016.md`, `docs/PROPS_P2_DIAGNOSTICS_20261019.md`, `docs/EGL_RENDERING_CHECK_20261019.md`.
> Load the skills under `.github/skills/` when their topic comes up. **Your first job is not to write code:** run the state
> checks in the handoff, then go through the spec's decisions D1–D7 with the user and ask what is unclear. The P3 spec and plan
> are drafts written without running anything; treat every assumption in them as unverified. Hold yourself to the repo's rules:
> never edit a hash-bound file, never delete datasets, never `git checkout` in a mounted directory, never weaken a gate, never
> claim anything passed without running it and reading the output, commits are authored solely by the user with no AI trailers,
> and do not push. Update the documentation in the same turn as every finding.

## What this repository is

A research repository where **results must be reproducible and provable**. Experiments are hash-bound: datasets carry seals,
training runs record bindings over their source files, and a run refuses to start if it cannot reproduce what it was built
against. The central question: how much and what kind of data does a VLA need to generalise? Cube programme (B1–B3) closed
2026-10-05. Now **real-life props in robosuite**: P0 scene and scripted expert; P1 instruction sorting (one object, bins named by
colour); P2 the same with 1–2 distractors; P3 (next) Food vs Drinks with a fixed prompt.

Two digest-pinned Docker images, nothing runs outside them: the **admission image** (`sha256:08de1e09…714abc`, MuJoCo 3.13.0:
everything sealed, training, tests that import no simulator) and the **robosuite image** (`sha256:54d861aa…ccf27e`, robosuite
1.5.2 / MuJoCo 3.4.0: anything that builds the scene). Their results are not physics-comparable. A third, **check-only EGL
image** exists (below). Full digests, tags and flags: [docs/OPERATIONS.md](../../OPERATIONS.md).

## Where things stand (verify; this was true on 2026-10-19)

**P2** (X-VLA, 540 sealed episodes) was trained and evaluated: checkpoint 101,520 report **75.0% (90/120)**, two of four
pre-registered criteria met (wrong-object 5.8% against 5%, held-out gap 30 points against 10: both failed), wrong-bin 0 of 120. P1's
checkpoint scores 0/120 on the same trials. P2 is closed; the user merged `feat/props-p2` into `main` (origin `fa72b56`).

**Work done in the session that wrote this** (branch `feat/props-p2-followup`, 19+ commits ahead of `origin/main`, **not pushed**):

1. **P2 failure diagnostics** ([report](../../PROPS_P2_DIAGNOSTICS_20261019.md)), replay-only, code in
   `experiments/xvla_props_p2_eval/diagnostics/` (not hash-bound, reusable). Replay is exact (Gate 0, 440/440 trials), but the
   pre-registered failure taxonomy **was not produced**: its validation Gate 1 failed twice and the 0.80 threshold turned out
   unattainable (two identical policies score 0.605/0.696 on it); the spec's stop rule forbade a third definition. Envelope-free
   findings (descriptive, not pre-registered): all 12 never-lifted report trials of 101,520 closed the gripper, 9 held the target
   without lifting; milk is 7 of those 12 and fails at the hold, not the approach; no layout concentration for 101,520 (p = 0.45);
   wrong-object association not supported (p = 0.17). FAILURES #58.
2. **EGL (GPU) rendering check** (package `experiments/egl_rendering/`; merged into `main` 2026-10-21 from `feat/egl-rendering`): new image `vla-robosuite:1.5.2-mj3.4.0-egl`
   `sha256:fec83621c46bdae862a60dbfa32607226f8e4afb7610f3b85acf1094a757ade5` (robosuite image + `libegl1`, built through the host
   proxy as build args). OSMesa unchanged byte for byte; physics bit-identical across renderers; EGL repeats differ by 1 to 2
   pixels of 904 images; 7.7× faster; **but EGL and OSMesa frames differ on half the pixels.** Check-only: **not adopted for any
   run**, and **the P2 checkpoints must not be evaluated under EGL.** ([report](../../EGL_RENDERING_CHECK_20261019.md))
3. **P3 spec and plan**, revised 2026-10-20 at the user's request after the re-test: the gate is now bin choice on trained objects, not overall success; one seed per layout; 288 reserved layouts with 80 per held-out object; a power table fixes the held-out rule (the first draft had almost no power: about 20 independent held-out layouts per object). D1–D7 are otherwise unchanged and still need confirming.

**Sub-project B (fresh-layout re-test) is done** (2026-10-20, `docs/PROPS_P2_RETEST_20261020.md`, package `experiments/xvla_props_p2_retest/`). After a power analysis the user chose the pooled paired penalty as primary (milk-versus-bread was unresolvable): 238 paired layouts, penalty **2.5 points [−4.2, +9.2]**, "small, existence not shown" under the pre-registered reading. The P2 30-point held-out gap therefore mostly reflects which objects the cells contain; criterion 3 stays failed. Milk is still the weak object (57% against bread's 79%). B and P3 training both need the GPU: it is free now.

## What to do first (state checks; do not assume)

```bash
git worktree list; git branch -vv | head -20; git stash list; git status --short
docker ps -a --format '{{.Names}} {{.Image}} {{.Status}}'      # any live container may mount a tree you are about to edit
df -h /home | tail -1                                           # was 103 GB free, 88% full
```

- `feat/props-p2-followup` and `feat/egl-rendering` are both merged into `main` (2026-10-21; push status is the user's). Create the P3
  branch from `main`: `git worktree add -b feat/props-p3 .worktrees/feat-props-p3 <base>`. **Never
  `git checkout`/`switch`/`reset` in a directory a container has mounted** (FAILURES #34 crashed a live 200-episode collection).
- `git stash@{0}` holds the user's own edit to `AGENTS.md` (removes the HF-artifacts link), deliberately not committed. Leave it.
- Local `main` is behind `origin/main` by 24 commits; do not build on it.
- `/tmp/histwt` is a detached worktree from an earlier session; leave it.
- The helper scripts `/tmp/dtest.sh`, `/tmp/rtest.sh`, `/tmp/drun.sh`, `/tmp/etest.sh`, `/tmp/egl_run.sh` are ephemeral and may be
  gone; the commands are in `CLAUDE.md` (tests) and the plans (runs). Recreate, do not trust.
- Disk: `outputs/` is about 57 GB. `outputs/egl_rendering_check/` (3.6 GB, raw EGL check runs, reproducible) is scratch that the
  user may let you reclaim; ask. **Never delete anything under `datasets/` or any sealed dataset root.**

## The P3 design in one paragraph (details and reasons in the spec)

Fixed prompt `sort the item`; one object per episode; P0's blue/orange bins as the two containers, **drinks → orange, food →
blue**, sides swap between episodes; trained objects can and milk (drinks), bread and lemon (food); **held out entirely: bottle
(drink) and cereal (food)**, because the policy handles both (P2: 0.93, 0.86) and cereal is a tall box like the trained milk carton,
so a shape cue points to the wrong bin. 544 episodes (480 train + 64 validation), OSMesa, 5 epochs. Gated criteria apply to
trained objects only; held-out objects are exploratory with bin-choice accuracy as the read-out. **All of D1–D7 are proposals the
user has not confirmed.**

## What to be careful about (each one cost real time)

**Repository rules (non-negotiable, see `.github/copilot-instructions.md`)**
- Never edit a hash-bound file while a run is live; package READMEs are bound. `experiments/props_scene/*.py`,
  `experiments/robosuite_scene/*.py`, `experiments/props_p1_data/**`, `experiments/props_p2_data/**` and the P1/P2 specs and
  plans are bound through the seals: **fork, do not edit.** Before editing anything under `experiments/`, check `SOURCE_FILES`
  and `provenance()`.
- Never relax a verification gate to get green. If a gate fails, the code or the claim is wrong.
- Offline metrics (loss, MAE) are not capability; never select a checkpoint from them.
- Evidence before claims: run it, read the output. When adding a test for a bug, confirm it fails against the unfixed code.
- Docs pass on every finding, in the same turn: grep the live docs for the figures your finding changes and update every hit.
- Commits: authored solely by the user, short messages, no AI or co-author trailers, no secrets, datasets, checkpoints or videos.
  **Do not push** (no GitHub credentials on this machine; the user pushes). Do not amend published commits or force-push.
- `docker wait` has silently failed to notify here: poll actively.

**Traps specific to P3** (the first three are in the plan's review focus)
- The expert's own text "put the X in the Y bin" is written into every frame by frozen code; the fixed prompt must replace it in
  the sink, with a test proven to fail without the fix.
- P2's validators and training contract assume 10 instructions matched to episodes by text; a single shared prompt breaks them in
  specific places (listed in the plan). Re-derive each check, scoped; never relax generally.
- Held-out objects must never be a target in training or validation (test it against real plan output, not a fixture).
- Milk is weak even when trained; its failure must not be read as a category failure. Per-object tables, and bin choice
  conditioned on a stable placement.
- "Category" cannot be separated from visual similarity with six objects; do not claim semantics.
- Do not change the renderer in P3 (one new variable per phase). The EGL image is for a later phase that adopts it end to end.

**Methodological lessons from this session (my own mistakes; do not repeat them)**
- I fixed the diagnostics' validation threshold (0.80) without first computing what the sample sizes permit, then spent the one
  allowed amendment on the wrong thing. **Simulate the statistic under "nothing differs" before fixing a threshold.**
- An envelope built from the scripted expert described the expert, not what is enough to succeed.
- A pooled comparison (closest approach, never-lifted against successes) was an object-mix effect; re-read per object first.
- I read "the object is being carried off the table" in rendered frames where the trace showed the target never rose more than
  1.2 cm. **Where a picture and a trace disagree, the trace decides.**
- I asserted that `PYOPENGL_PLATFORM=egl` was required, wrote it into a spec and a guard, and a four-way test disproved it (the real
  cause was a missing `Renderer.close()`). Test a suspected cause before writing it down.
- Several of my first tests were vacuous (compared a value with itself, took the expected value from the code under test).
  **Mutation-check every new test**: break the code, see it fail. Surviving mutations exposed real gaps three times. Some
  survivors are equivalent mutants (an implied condition): remove the redundancy instead of testing the impossible.
- The passing-gate branch of my analysis, which the real data never reached, crashed on an object that appeared only as a
  distractor. Test the branches the real data does not exercise.
- Layout, not trial, is the independent unit. A pooled spread describes groups, not the units inside them.

**Operational lessons (FAILURES.md has the full entries)**
- Parallel evaluation: pin cores per worker (`--cpuset-cpus`, `OMP_NUM_THREADS`), size the pool by peak GPU memory after a
  checkpoint switch (about 7.3 GiB per worker), not the first-load figure (#57). Collection: at most about 30 workers (#50).
- A forked package can pass its green tests while exercising the source package (symtable dead-name scan, grep for the source
  name): `.github/skills/forking-an-experiment-package/SKILL.md`. A hash-bound README was once committed empty and the residue test
  passed it: confirm the file's size on disk before testing or hashing it (#56).
- Docker builds here need the host proxy as build args (`--build-arg http_proxy="$http_proxy" --build-arg https_proxy=...`); use
  the classic builder (`DOCKER_BUILDKIT=0`). Never remove, retag or prune an image that is not ours; the machine is shared
  (`determined_wozniak` and `isaac-sim` containers belong to other people).
- Long robosuite runs need a large `/tmp` (16–32 GB) and, for per-episode-video datasets, `--ulimit nofile=65536:65536`.

## Artifacts and where they are

- P2 dataset (sealed): `outputs/props_p2_data/props_p2_20261013/` plus 12 shard datasets (kept). **Never delete.**
- P2 checkpoints still on this machine: 101,520 and 126,900 (and P1 88,830 and 126,900). Intermediate weights and optimizer states
  were deleted after the user verified Windows copies; `verify_checkpoints` and `evaluate.py prepare` cannot pass in the repo.
  Re-evaluate from the bundle `outputs/downloads_20261017/props_p2/` (`reevaluate/bundle_evaluate.py`, `verify_bundle.py`).
- P2 evaluation trials: `outputs/xvla_props_p2_eval/<mode>_<policy>_step_<N>/trial_NNN_<layout>/`
  (`cameras.mp4`, `trace.npz`, `episode.json`, `telemetry.json`); diagnostics output under `.../diagnostics/` (measures, `reference.json`,
  `analysis_descriptive.json`, contact sheets).
- Hugging Face artifacts: [docs/HUGGING_FACE_ARTIFACTS.md](../../HUGGING_FACE_ARTIFACTS.md) (private; publication is the user's decision).

## Open decisions for the user

1. Confirm or change P3 decisions D1–D7 and the proposed success thresholds (they are pre-registration).
2. ~~B: numbers for the pre-registered rule~~ done 2026-10-20 (see above).
3. ~~Push and merge `feat/props-p2-followup` and `feat/egl-rendering`~~ done: both are in `main` (the EGL merge is local until the user pushes).
4. Whether to reclaim `outputs/egl_rendering_check/` and other scratch before P3 training (about 18.5 h of projected training
   needs the snapshots' space: 10.6 GB each in P2).
5. Whether to publish the P2 diagnostics or any P2 artifact.

## How the user works

Terse, decisive, wants recommended options with reasons and tends to approve a clear recommendation. Keeps every protocol
honest (pre-registered thresholds, stop rules, no moving a gate after a result). Prefers periodic, logically scoped local commits
on the relevant branch, simple short messages, nothing pushed without permission. Wants docs kept in sync as a standing rule, and
negative results recorded with the evidence. Avoid time estimates; ask when a design choice is genuinely theirs to make.
