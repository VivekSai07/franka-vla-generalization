# SmolVLA / X-VLA Franka TODO

## Where things stand (2026-10-30)

- **Next: the VLA inputs programme — specs and plans drafted 2026-10-30, under external review; nothing built.** The data stays
  P2's 540 sealed episodes; what changes is what the model sees: M0 a common paired evaluation (fresh layouts, A0 = X-VLA 101,520
  as control), S0 a sealed sidecar of wrist force/torque, finger forces and target boxes for every P2 frame, M1 pi0.5 against
  X-VLA, M2 force/torque in the state, M3 the target box in the state, M4 an auxiliary box head (M4b force outputs scoped only).
  Order approved by the user: M0 and S0 alongside M1 training, then M2, M3, M4a. Overview:
  [docs/superpowers/specs/2026-10-30-vla-inputs-programme-design.md](docs/superpowers/specs/2026-10-30-vla-inputs-programme-design.md)
  (links every spec and plan). Two facts measured while writing it: replaying the sealed P2 actions from the sealed initial state
  reproduces the sealed `qpos` byte for byte (3 of 3 episodes probed), and **pi0.5 cannot be constructed in the pinned admission
  image** (it needs LeRobot's patched transformers; `transformers.models.siglip.check` is missing), so M1 needs a new image.
- **Props P3 closed as a failed attempt (2026-10-27), artifacts deleted by the user.** It was trained (126,900 updates) and
  evaluated: best checkpoint 101,520, 85.0% on trained objects, 5.7% on held-out ones, milk (trained) only 10/20; numbers in
  [docs/RESULTS.md](docs/RESULTS.md). The dataset, checkpoints, gate scratch and evaluation outputs were deleted on the user's instruction and were never
  copied to Windows or published, so nothing in P3 can be re-verified or re-evaluated; what is kept is the code and docs, merged into `main`
  (`0ce1698`, on origin), which hold the evidence and the lessons (FAILURES #59, #60). The `feat/props-p3` branch no longer exists locally.
- **Props P3 (Food vs Drinks, fixed prompt `sort the item`) dataset collected and sealed 2026-10-20 (deleted 2026-10-27): 544/544 episodes,
  0 rejected, fingerprint `6430d34e…d151acf`; trained and evaluated, failed, deleted (bullet above).** Decisions D1–D7 confirmed by the user 2026-10-20 (unchanged).
  Expert re-test 286/288 = 99.3% (lemon with the blue bin on the left is the only miss), the 16-worker thread cap proven
  pixel-neutral, 46 min 43 s in 16 parallel shards, seal re-verified in a second process (held-out objects bottle and cereal: 0
  episodes, checked from the recorded arrays). Report: [docs/PROPS_P3_COLLECTION_20261020.md](docs/PROPS_P3_COLLECTION_20261020.md);
  branch `feat/props-p3` (not pushed). The training and evaluation packages for it are built (bullet above).
- **Props P2 (instruction sorting with 1–2 distractors) trained and evaluated 2026-10-16: two of the four
  pre-registered criteria met.** Dataset sealed 2026-10-13 (540/540, fingerprint `b1d642b3…f4d898`), trained 126,900
  updates. Winner by the fixed rule, checkpoint 101,520, report tier 90/120 = **75.0%** [65.0, 84.2]: overall ≥70%
  **pass**, beats P1 126,900 (**0/120**) **pass**, wrong-object ≤5% **fail** (7/120 = 5.8%, one trial over),
  trained-vs-held-out gap ≤10 points **fail** (trained 85.0%, held-out 55.0%; `milk→blue` 7/20). Confirm (touched
  once) 16/20 = 80%; no-distractor control 17/20 (P1: 14/20); 0 wrong-bin in 120. 101,520 and 126,900 (83/120) are
  not distinguishable (McNemar p = 0.21). Report (with every trial by set and checkpoint):
  [docs/PROPS_P2_EVALUATION_20261016.md](docs/PROPS_P2_EVALUATION_20261016.md). **P2 is closed (user, 2026-10-17):** the
  criteria were not loosened. Complete archive bundle `outputs/downloads_20261017/props_p2/` (14 GB: dataset, both kept
  checkpoints 101,520 and 126,900, all 540 trials, runtime, code snapshot; weights, seal and 540/540 trials hash-verified, and both weights hash-verified on Windows 2026-10-17;
  smoke re-evaluation from the bundle alone in a clean offline container reproduced both trials bit for bit). Branch
  `feat/props-p2`: pushed to origin (2026-10-17; the RTX machine has no GitHub credentials, so the push and the pull request are made from elsewhere); merged into `main` by the user.
  **Diagnostics 2026-10-19** ([report](docs/PROPS_P2_DIAGNOSTICS_20261019.md), replay only): the pre-registered failure
  taxonomy was not produced (replay exact, 440/440; the validation gate failed twice and its 0.80 threshold was
  unattainable, FAILURES #58). What it did show, descriptive and untested: all 12 never-lifted report trials of 101,520
  closed the gripper, 9 held the target without lifting it; milk (7 of the 12) fails at the hold, not the approach;
  no layout concentration for 101,520; wrong-object association not supported (p = 0.17).
  **Fresh-layout re-test 2026-10-20** ([report](docs/PROPS_P2_RETEST_20261020.md)), 101,520, 238 paired layouts of milk and
  bread under both bin instructions: pooled unseen-pair penalty **2.5 points [−4.2, +9.2]**, "small, existence not shown"
  (pre-registered reading); milk 57.6% / 56.8%, bread 80.8% / 76.7%. The P2 30-point gap compared different objects; criterion 3
  stays failed. Answers the "unseen pair or hard object" question: mostly object.
  **Open:** why a closed, holding gripper does not lift (milk fails at the hold), and whether to publish P2 (the P2 branch was
  merged to `main` by the user).
- **Props P1 trained and evaluated (2026-10-09): confirm 100% (20/20), report 75% (60/80) on checkpoint
  126,900**; zero wrong-bin outcomes in 480 report trials. Reports:
  [evaluation](docs/PROPS_P1_EVALUATION_20261009.md), [training](docs/PROPS_P1_TRAINING_20261009.md),
  [collection](docs/PROPS_P1_COLLECTION_20261007.md) (540/540, sealed 2026-10-07, 9 h 38 m serial). P0 and P1
  (`feat/props-p0`) were merged into `main` by PR #1 (`582b107`).
- **Props P0 is GO (re-run 2026-10-05): 283/288 = 98.3%** after two spec amendments (success by lowest
  point; start region clear of the arm's working envelope). Reports:
  [docs/PROPS_P0_RERUN_20261005.md](docs/PROPS_P0_RERUN_20261005.md), first run
  [docs/PROPS_P0_20261005.md](docs/PROPS_P0_20261005.md).
- **The cube programme is closed.** B3 (four colours, two held-out pairs, 0–2 distractors, robosuite scene)
  was the last cube experiment: step 124,800 about 85% overall and about 75% with distractors; held-out
  pairs as good as trained ones. Results arc: [docs/RESULTS.md](docs/RESULTS.md).
- **Everything published is public** (switched 2026-10-09, verified via the Hub API): seven datasets
  and the published models, including B3 and props P1. A 2026-10-09 repair pass closed B2's
  missing-archive gap and added `runtime/` + `code/` portability sections to B1, broad-workspace, B2, B3
  and local-varied's SmolVLA model (clean-container re-evaluation from those five not yet run; props P1's
  was). Map, pins and caveats: [docs/HUGGING_FACE_ARTIFACTS.md](docs/HUGGING_FACE_ARTIFACTS.md).
- Items from Phases A and B that the closed cube programme made moot are ticked with
  *(closed 2026-10-05, not pursued)* and kept for their reasoning.

Live plan is below. The original `docker-compose` + `franka_libero_stacking_v1/v2` work is
preserved as an archive at the bottom; see [HF artifact links](docs/HUGGING_FACE_ARTIFACTS.md), [README](README.md) and
[docs/BENCHMARK_STATUS.md](docs/BENCHMARK_STATUS.md) for current status.

---

## Phase A — maximise the current single-task stacking problem

Goal: establish the ceiling on one axis of variation (cube position in envelope F7) before
adding any new axis. Ordering matters — a failure with two new axes is not attributable.

### A0. Finish what is running
- [x] Cross-evals complete (local-varied checkpoints on `broad_holdout`, 72 trials each)
- [x] Record the full 2×2 + confidence intervals in README
- [x] Note the `metadata_binding_reproducible: False` caveat beside the SmolVLA cross-eval figure

### A1. Measure the checkpoint curve BEFORE training longer
Rationale: X-VLA broad saw **0.50 epochs** (40,000 chunks / 79,840 starts) versus 1.67 epochs on
local-varied — we tripled the data and held updates at 5000. Train/val arm MAE 0.01655/0.01748
(5.6% gap) shows underfitting, not overfitting. But confirm the trend before spending GPU.

- [x] Add a `--checkpoint` option to the broad eval packages. Currently 5000 is hard-pinned in
      four places: `offline_probes` `checkpoint_update=`, `verify_final`'s `completed_updates`
      and `selected_checkpoint_update` requirements, and `checkpoint_directory(5000)`.
      `verify_final` must still pin the chosen checkpoint's `model_sha256` to its training marker.
- [x] Evaluate X-VLA broad checkpoints **3000** and **4000** on the identical 72-trial holdout
- [x] Plot success vs updates (1000/2000/3000/4000/5000 as available)

**Decision gate: PASSED (2026-09-23).** 0.0% @3000 -> 20.8% @4000 -> 68.1% @5000, still
accelerating at the final checkpoint, val/train 1.01-1.06 (no overfitting). Proceed to A2.
Caveat recorded: part of the final jump is the cosine anneal, so do not extrapolate linearly.

### A2. Train to matched exposure (only if A1 says climbing)
- [x] New package (`xvla_broad_workspace_long`) — did NOT edit the existing one. `recipe()`
      hardcodes `max_optimizer_updates=5000` and `schedule.total=5000`; changing them alters
      `SOURCE_FILES` hashes and would invalidate the sealed 5000-update run.
- [x] **16,700 updates** to match local-varied epoch exposure (79,840 starts × 1.67 / 8)
- [x] Fresh run from base, not a resume
- [x] Checkpoint schedule extended so the curve stays measurable
- [x] Re-evaluated on the same holdout: **62/72 = 86.1%** vs 49/72 = 68.1% (Fisher p = 0.0165)
- [x] *(closed 2026-10-05, not pursued: the cube programme ended with B3)* Repeat for SmolVLA (2.09 epochs currently) — **hold**; see A4, the remaining failure is not
      obviously exposure-limited and a SmolVLA run would cost ~4 h to answer a different question

**Outcome (2026-09-24): exposure was the deficit, and it is now largely paid off.** Two findings
changed the plan and are recorded in [docs/RESULTS.md](docs/RESULTS.md) and
[docs/FAILURES.md](docs/FAILURES.md) #21–#23:
- the checkpoint curve is **non-monotonic** (25.0% → 0.0% → 86.1%) while offline MAE improves
  monotonically — offline metrics are invalid as capability proxies
- all 10 remaining failures are a single placement offset of roughly one cube width, and the
  expert demonstrations are clean (max 1.98 cm, 0 episodes > 2 cm), so it is policy-side

### A3. Close out
- [x] Update docs with the exposure finding and the curve
- [x] Evaluate checkpoint **12,000** — the LR-matched comparison against the short run's 4,000.
      **Result: 51/72 = 70.8% vs 20.8%, p = 2.2e-9.** Exposure drives capability, not the anneal;
      at matched epochs a 4.4× LR difference changes nothing (p = 0.69).
- [x] *(closed 2026-10-05, not pursued: the cube programme ended with B3)* Decide whether the 65/72 gate is right for this protocol — argue it on its own terms,
      not by fitting it to whatever number we got. Note 62/72 misses by three trials and every
      miss is the same 4.5 cm offset, so the gate question is now entangled with A4.

### A4. The 4.5 cm placement offset (new — highest-value open question)

**Design spec: [docs/superpowers/specs/2026-09-24-placement-offset-design.md](docs/superpowers/specs/2026-09-24-placement-offset-design.md)**
Approved ordering: **Tier 2 → Tier 1 → B1 collection design.**

All 10 failures at 86.1% land the cube one cube-width beside the target. Decomposed about the robot
base, the systematic component is **tangential** (a ~5° bearing error) rather than a reach error.
Authoritative magnitudes from the 128-trial T2a run: successes +0.150 cm (sd 0.715, n=96), failures
+3.893 cm (sd 0.723, n=32). Failure is layout-specific, not a global bias.

**Cause: action-sampling multimodality.** 6 of 8 failing layouts are split decisions — same scene,
same weights, different sampling seed, opposite outcome. `modeling_xvla.py:252` draws fresh
`torch.randn` per chunk over `num_denoising_steps=10`; `predict_action_chunk` accepts injected
`noise`, which is the hook for every Tier 1 fix.

Ruled out by test, not assumption:
- **imitating a sloppy expert** — refuted; all 200 demos clean (max 1.98 cm, 0 above 2 cm)
- **unreachable far layouts** — refuted and inverted (failing 0.424 m vs solved 0.472 m reach)
- **sparse training coverage** — refuted; identical 4-D NN distance, τ = 0.000
- ⇒ **more data will not close this gap**; this is the primary input to B1 planning

#### Tier 2 — measurement (first)
- [x] **T2a** mode-probability run: 128 trials complete (rc=0, 223 min). **Control 64/64 = 100%**,
      failing group 32/64. Three layouts (008, 017, 029) draw the wrong mode 7 times in 8; the bad
      mode is **absent** from control layouts. No global bias (successes +0.15 cm). No geometric
      feature separates the hard layouts.
- [x] **T2b** seed budget **established**: 2 seeds cannot distinguish 12.5% from 87.5% per layout
      (024 and 029 both read 1/2). **≥8 seeds per layout is required for any per-layout claim.**
      Aggregate rates survive at 2 seeds. Lock this into the B1 protocol before collecting.

#### Tier 1 — inference-time (downgraded by T2a)
- [x] *(closed 2026-10-05, not pursued: the cube programme ended with B3)* **T1b** best-of-N — projected 62 → ~65 of 72, and cannot help the three bad-dominant layouts.
      Marginal; **not a blocker for B1**. Do only if the gate matters more than N× inference cost.
- [x] *(closed 2026-10-05, not pursued: the cube programme ended with B3)* **T1a** `num_denoising_steps` 10 → 30; 16-trial probe (~40 min) — cheapest remaining probe
- [x] *(closed 2026-10-05, not pursued: the cube programme ended with B3)* **T1c** temporal ensembling — **averaging is unsafe**: the mean of the two modes lands inside
      the empty band and fails everywhere. Cluster-then-pick-largest only.
- Each is a separate protocol in its own package; never compare to the sealed 86.1% unless the
  layout roster and seeds are identical

#### Tier 3 — data design for B1 (low expected value here)
- [x] *(closed 2026-10-05, not pursued: the cube programme ended with B3)* Audit expert approach-direction consistency against the sealed dataset (CPU-only, cheap) —
      the only data-side lever with a plausible mechanism
- [x] *(closed 2026-10-05, not pursued: the cube programme ended with B3)* Terminal-phase supervision density — hypothesis only
- [x] **Not planned:** more episodes, denser layout coverage

Only after Tier 2 + Tier 1: decide whether the 65/72 gate is right, and finalise B1 collection.

---

## Phase B — generalisation axes (deferred until Phase A closes)

One axis at a time, each with its own dataset and its own holdout.

### B1. Language grounding (first axis — **measured 2026-09-26**, see RESULTS.md) [x]

**Design spec: [docs/superpowers/specs/2026-09-24-b1-language-grounding-design.md](docs/superpowers/specs/2026-09-24-b1-language-grounding-design.md)**

**Result: 151/160 = 94.4% on held-out layouts; familiar vs. swapped arrangement p = 1.0 (not
significant) — no evidence of the positional shortcut.** Full numbers, checkpoint screening, and
per-cell breakdown in [docs/RESULTS.md](docs/RESULTS.md) and
[docs/BENCHMARK_STATUS.md](docs/BENCHMARK_STATUS.md). Text below is the original design
rationale, kept for context.

**The language pathway has never been load-bearing.** `plan.py` draws the cubes from disjoint
regions — red always at y<0, green always at y>0, verified 200/200 episodes and 80/80 reserved
layouts, separated by 40 cm. "Pick the cube at −y" solves 100% of the data, so the 86.1% spatial
result says nothing about language.

**The original plan here would have produced a false positive.** Reversing the instruction over the
same layouts makes "red→green" ≡ "pick −y" and "green→red" ≡ "pick +y"; a colour-blind model scores
100% by learning instruction→side.

**Design: 2×2 factorial** — instruction (red→green / green→red) × arrangement (familiar / swapped
colours). Each region is correct in 2 of 4 cells and each instruction maps to each region once, so
neither position nor instruction alone suffices. The *pattern* of passed cells names the strategy:
all four = grounding; C1+C3 = instruction→side; C1+C2 = ignores instruction; C1+C4 = ignores both.

No expert changes needed — it is already colour-parameterised and generates the instruction from
`pick`/`place`. Only `plan.py`'s hardcoded `pick='red', place='green'` changes.

#### BLOCKED — the shared validation stack is single-task and hash-bound
Collection cannot run until this is resolved. See [docs/FAILURES.md](docs/FAILURES.md) #28.

`experiments/fixed_layout/validate.py` assumes one task in five places. A local `metadata()`
override in the B1 package handles three; `read_rows` (per-frame `task_index == 0`) and the
`raw["task"] == TASK` frame check remain, both reached through `generic._episode_inputs`.

**The file must not be edited** — it is inside the sealed provenance of `broad_workspace_data`
(confirmed in `provenance.json`), so changing it invalidates the sealed dataset and the 86.1%
result.

Two viable options, needing a decision:
- [x] *(resolved: B1 was collected and sealed on 2026-09-25)* **(A) Duplicate locally.** Fork `read_rows` and `_episode_inputs` into the B1 package with
      task-aware checks. ~90 lines of deep validation logic copied. Risk: a transcription slip
      yields a validator *weaker* than intended, which still passes — the worst failure mode here.
      Mitigate by diffing the copy against the original and asserting equivalence on a single-task
      fixture before trusting it.
- [x] *(resolved: B1 was collected and sealed on 2026-09-25)* **(B) Parameterise the shared module** with a `tasks=(TASK,)` default that preserves current
      behaviour byte-for-byte, then re-seal the affected datasets. Cleaner long-term, but re-sealing
      `broad_workspace_data` invalidates every binding built on it.

Recommendation: **(A)**, with the equivalence check. (B) is the better design but its cost lands on
already-sealed results.

#### Budgets (measured rates: 1.97 min/episode, 0.83 s/update, 2.2 min/trial)
B1 is a **fully self-contained dataset** — own plan, own layout bank, own reserved holdout, own
seal. It does not read, import, extend or re-seal `broad_workspace_data`.
- [x] **Collect** 200 episodes, 50/cell, 200 distinct geometries, balanced across C1–C4. Reserve
      B1's own evaluation layouts from envelope F7. — **6.6 h** (actual: 6.4 h + resume; slot 197
      needed an infeasible-layout substitution, FAILURES #30)
- [x] **Admission + preflight** — ~1 h CPU + 5 min GPU (actual: 28 min CPU + 1 min GPU)
- [x] **Train** X-VLA from base, **33,400 updates = 3.35 epochs — 7.8 h** at the measured
      0.84 s/update. Doubled from the matched-exposure 16,700/1.67 epochs/3.9 h because B1 splits
      the same 160 training episodes across two instructions and four cells, and because the
      86.1% run was still climbing at its endpoint (70.8% at 12k → 86.1% at 16.7k).
      Checkpoints `[1, 8000, 16700, 25000, 33400]`; 16,700 is retained for comparability.
      (actual: 472 min = 7.9 h, rc=0, zero-error FP32 reload on the final checkpoint)
- [x] **Evaluate** the three candidates (16700/25000/33400). Screen on 40 of the 80 held-out
      layouts, then report the grounding contrast on the untouched other 40 — a disjoint
      selection set, because offline metrics cannot pick checkpoints here (FAILURES #21).
      (actual: screen 22/40, 34/40, **40/40** — winner 33,400; report **151/160 = 94.4%**,
      familiar 93.8% vs swapped 95.0% p=1.0, red 96.2% vs green 92.5% p=0.495 — see RESULTS.md)
- [x] Report **geometry-clustered** intervals, not naive binomial (per-geometry p ranges
      12.5%–87.5%, so trials cluster hard within geometry)
      (actual: much milder clustering than T2a — 34/40 report layouts at 4/4, 3 at 3/4,
      3 at 2/4, none below 50%)
- Total ≈ **25 h** (actual, collection start to eval finish: ≈ 33.5 h across 2026-09-25–26)

Comparing B1's C1 cell to the sealed 86.1% is a same-envelope, **different-layouts** comparison — a
rough sanity check only, never a precise delta. Sharing the holdout was considered and rejected:
it would drag cross-dataset binding into a package with no other reason to touch it.

#### Controls
- Data starvation: 80 episodes per instruction vs 160 for the single-instruction model. The C1-vs-C2
  contrast is within-model so attribution is safe; only the absolute level is exposed. Escalate to
  400 episodes only if the absolute result is ambiguous.
  (actual: absolute level 94.4% overall, no cell below 92.5% — not ambiguous; no escalation needed)
- Do **not** evaluate existing checkpoints on the swapped cells as a grounding test — they have
  never seen a red cube at +y, so that measures distribution shift, not grounding.

### B1b. Vertical variation (new axis — currently a constant, therefore untestable)

**Deferred to "next week" per user, 2026-10-01.** Separate loop test; do not fold into B3.

Measured across the sealed 200 episodes: initial pick cube Z and place cube Z each take **exactly
one value** (spread 0.00e+00), and the final stack height spreads 1.5e-05 m. Approach altitude *is*
varied (±3 cm hover jitter, 200 distinct values) and home pose is varied (±1°, 1400 values), but the
terminal heights are constant — a flat table with identical rigid cubes fixes them.

Consequence: the policy can satisfy every episode with an **absolute-Z target** and has never needed
to perceive height. The hover jitter does not rescue this — it varies the descent *distance*, not the
target. Same class of problem as the colour confound (#27): a variable held constant makes the
capability untestable and the success metric cannot see it.

Deliberately **not** folded into B1 — that would be two new axes at once and a failure would not be
attributable. Options when it gets its own phase, each with different expert implications:
- [x] *(closed 2026-10-05, not pursued: the cube programme ended with B3)* cube-size variation — changes grasp aperture the controller assumes, and the stack height
- [x] *(closed 2026-10-05, not pursued: the cube programme ended with B3)* table-height variation — uniform shift of both cubes; touches the scene
- [x] *(closed 2026-10-05, not pursued: the cube programme ended with B3)* variable-height place target (platform/riser) — changes place Z but not pick Z

### B2. Distractors (second axis)

**Design spec: [docs/superpowers/specs/2026-09-28-b2-distractors-design.md](docs/superpowers/specs/2026-09-28-b2-distractors-design.md)**
**Plan: [docs/superpowers/plans/2026-09-28-b2-distractors.md](docs/superpowers/plans/2026-09-28-b2-distractors.md)**

- [x] Package `experiments/b2_distractor_data/` implemented and tested (90/90 in-container).
      200 slots: 40 control + 160 distractor-present (1-2 cubes), cells balanced *within* each
      arm, riskiest profiles ordered first, 60 reserved layouts (20 screen / 40 report) banked in
      both conditions. First launch died at episode 0 on inherited parked-cube velocity; fixed
      before the collection that followed (FAILURES #32, #33). A later `git switch main` crashed
      the running collection at episode 49 by checking out a tree missing the package/docs;
      recovered via rollback + `--resume`, no data lost (FAILURES #34). Slot 124 (`train_124`)
      was unsolvable by the expert (9/9 `expert_execution` rejections, same class as B1's
      `validation_037`); substituting it exposed a fork-hygiene bug in `amend_frozen_plan`'s
      per-episode equality check, which didn't allow for B2's `distractor_xy_m` recompute — fixed
      and resumed cleanly, no data lost (FAILURES #35).
- [x] **Pilot** — the first ~15 admits ran clean (episode 0 admitted first try after the velocity
      fix); the full run then rejected only `train_124` (9/9, substituted).
- [x] Full 200-episode collection: 200/200 admitted, 9 rejections (all on `train_124`), sealed
      `b6cfff3f...0c12340`.
- [x] Training package `experiments/xvla_b2_distractors/` (forked from `xvla_b1_language`), 50/50
      tests passing. CPU admission, GPU preflight, and 50,000-update training all completed
      cleanly (11.68 h, all 6 checkpoints byte-exact FP32 reload).
- [x] Eval package `experiments/xvla_b2_distractors_eval/` (forked from `xvla_b1_language_eval`).
      First launch of the chain silently no-op'd on every stage — the forked `evaluate.py` was
      missing its `if __name__ == '__main__':` guard, so `python -m ...evaluate` exited 0 having
      done nothing; every screening checkpoint "passed" in seconds (FAILURES #36). Fixed, with a
      regression test confirmed to fail against the pre-fix commit, before relaunching.
- [x] **Evaluation complete (2026-09-30).** Screening tied 33,400 and 50,000 at 17/20; reports:
      33,400 **132/160 = 82.5%**, 50,000 **147/160 = 91.9%** (paired p = 0.004; FAILURES #37).
      Control vs distractor: +2.5 / +1.25 pts, not detectable. Full diagnosis of all 41 failures:
      [docs/B2_DISTRACTOR_EVALUATION_20260930.md](docs/B2_DISTRACTOR_EVALUATION_20260930.md).
- [x] Measures distractor rejection independently of language — **answer: no detectable effect**,
      for distractors that never physically interfere (max displacement 0.064 mm).
- [x] *(answered 2026-10-02: the user folded these into B3 — pick-coverage near `054`/`048`, harder distractors; the release-precision gate was dropped)* **Decide next steps** (see section 8 of the report). Candidates, most
      failure mass first: tighten the expert's release precision (addresses 21/41 failures), add
      pick-coverage near the `054`/`048` regions, a harder distractor test with possible
      interference (B2b), and understand why `054` got worse with training and why the *familiar*
      arrangement fails more. Change one variable per run.
- [x] *(closed 2026-10-05, not pursued: the cube programme ended with B3)* Confirm 50,000's 91.9% on a fresh holdout (it is the larger of two reported results).
- [x] `feat/b2-distractors` merged into `main` (2026-10-01, `511ab25`), since pushed. The
      branch is kept, as the other axis branches are. The B2 packages are now hash-bound to the
      trained run: **fork, do not edit** (`experiments/b2_distractor_data/**`,
      `experiments/xvla_b2_distractors/**`, the B2 spec and plan).
- [x] **Transferred to Windows (2026-10-01).** Staged at
      `outputs/downloads_20261001/b2_distractor/` (**deleted from RTX 2026-10-17**, with its checkpoint originals; the
      Windows copy and the Hub are the archive) and copied to
      `C:\d drive\RTX trained models\downloads_20261001\b2_distractor`. All six
      `model.safetensors` matched `checkpoint_result.json` `model_sha256`; `seal.json`
      is `b6cfff3f...0c12340`. RTX originals not deleted.
- [x] **B2 HF dataset and checkpoint repositories published (verified 2026-10-09):**
      [dataset](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b2-distractors-200) and [model (root 50,000)](https://huggingface.co/VivekSai07/xvla-franka-b2-distractors-50000).
      The dataset's 153 missing metadata files were restored from Windows on 2026-10-09.
      The original transfer check read `seal.json`; it did not re-derive the dataset fingerprint.
- [x] Complete the B2 evaluation/training archive: the 2026-10-06 Windows audit found
      1,065 missing model-repo files despite all six checkpoint weights matching; a repair
      pass on 2026-10-09, run directly from RTX via the Hub API, added the 1,565 files a
      fresh local-vs-HF diff found missing, plus `runtime/` and `code/` portability sections
      (also added to B1, broad-workspace, B3 and local-varied's SmolVLA model). See
      [archive gaps](docs/HUGGING_FACE_ARTIFACTS.md#b2-distractors).

**Three spec errors were corrected before implementation** (see the spec's *Corrections* section):
the floor-colour prerequisite was withdrawn as both a no-op and seal-breaking; "16 episodes per
cell" is really 32; and screening is 100 trials, not 80.

### Environment housekeeping
- [x] *(closed 2026-10-05: moot — later work moved to the robosuite scene, which has its own table and floor)* **Floor colour: blue → grey — BLOCKED, and not a prerequisite for anything.**
      `src/robots/franka_panda/model/scene.xml:21` groundplane checker texture
      (`rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"`) reads bluish. Two findings from the B2 review
      (2026-09-28) change this item:
      (a) `scene.xml` is **not** the scene that runs — the packages load `smolvla_scene.xml`
      (pinned in `experiments/fixed_layout/source.py`), so editing `scene.xml` is a silent no-op;
      (b) **both** XML files are recorded in the sealed `provenance.json` of `b1_language_data`
      and `broad_workspace_data`, so editing either invalidates the provenance behind the 94.4%
      and 86.1% results and would block any future run bound to those datasets.
      Doing this at all means accepting that those two datasets can no longer be re-verified, or
      deferring until nothing further is needed from them. B2 does **not** need it: the distractor
      cubes are already in the scene, and leaving it alone makes B2's control arm visually
      identical to B1. Still never in the same collection run as a new data axis.

### B3. Combined-axis dataset on a new (robosuite) scene

**Design spec (Phase 1 only): [docs/superpowers/specs/2026-10-01-robosuite-scene-bringup-design.md](docs/superpowers/specs/2026-10-01-robosuite-scene-bringup-design.md)**
**Plan:** [docs/superpowers/plans/2026-10-01-robosuite-scene-bringup.md](docs/superpowers/plans/2026-10-01-robosuite-scene-bringup.md)
**Branch:** `feat/robosuite-scene-bringup` (created from `10f2524`; the spec's `624b156` base is stale).

Two-phase project. User accepted the one-variable-per-run attribution trade-off: a strong result from one experiment if it works, limited attribution if it doesn't.

> **Direction (user, 2026-10-02; clarified 2026-10-02).** The cube work has answered the question
> it was for: X-VLA learns this task when given the right data in the right format. **B3 on cubes,
> collected in the robosuite scene, is the final cube work**, and every axis that was meant to be
> tested goes into that one dataset. After it, new task ideas use real-life props in robosuite (see
> "After B3" below). Consequences:
> - B3 cube collection was the next thing to do; it was designed, collected, trained and evaluated
>   2026-10-03 to 2026-10-05 (below).
> - No separate effort goes into the placement offset or a bigger cube. The offset is understood
>   (the expert's own jitter); how the B3 release gate treats it is one decision inside the B3
>   design, not a project of its own.
> - Sim2sim is **not a gate**. It may be repeated, as a matter of interest, once a policy has been
>   trained on the robosuite scene.

- [x] **Phase 1 — robosuite scene bring-up.** Prove the simulator swap works end-to-end on a trivial scene (one layout, two cubes, two cameras replicated from the current scene, scripted expert, 5-episode PoC, deterministic resets, LeRobot schema parity). Nothing new is varied. Go/no-go report decides whether Phase 2 starts. See spec above for risks R1-R7.
      **Result (2026-10-02): go.** It rests on the fingertip-pad re-parenting, a model-contract fix outside the tuning protocol (whose six allowed attempts all failed), allowed by the executing AI controller's ruling and **ratified by the user on 2026-10-02**. R1–R7 all pass; 5/5 scripted episodes sealed at `outputs/robosuite_scene/robosuite_bringup_20261002/` (fingerprint `543e300c257d…`). Report: [docs/ROBOSUITE_SCENE_BRINGUP_20261002.md](docs/ROBOSUITE_SCENE_BRINGUP_20261002.md). The robosuite path runs in its own image on MuJoCo 3.4.0 (robosuite 1.5.2 cannot reset on 3.13.0), so robosuite-scene results are not physics-comparable with sealed results.
- [x] **Before Phase 2 is designed — carried over from Phase 1** — all settled by 2026-10-03: (a) the jitter-vs-gate question was dropped with the tighter gate (the 2 cm window stayed); (b) done as the sim2sim check (3/20 vs 19/20); (c) pad friction and geometry stayed robosuite's; (d) B3 got `b3_robosuite_data.contract.provenance()` and a seal; (e) ratified; (f) fixed — the B3 fingerprint hashes JSON files in canonical form. Original items:
      (a) decide jitter versus gate: the 8.6–12.7 mm xy placement offset (all five seeds, 20 mm window) is the scripted expert's own ±5 mm (1σ) target jitter, reproduced from the seeds to 0.65 mm, not a scene defect. Seed 2 at 12.65 mm fails Phase 2's planned ≤ 1.2 cm expert gate, and about 15% of episodes would (calculated) — reduce the jitter, loosen the gate, or accept the rejections;
      (b) run the B2-checkpoint smoke test on the robosuite scene (deferred from Phase 1 to a follow-up plan, not run);
      (c) decide whether finger pad friction/geometry stay robosuite's;
      (d) the package is not hash-bound — Phase 2 needs a contract before any result is reported from it;
      (e) ~~ratify or reject the fingertip-pad re-parenting~~ — ratified by the user on 2026-10-02 after the mechanism was explained (pad collision geoms sat in jointless `_tip` bodies, so the body-id contact check could not see a held cube);
      (f) the collected dataset is not byte-reproducible: two collector runs differ in `meta/stats.json` key order (all other files identical), so the fingerprint changes between runs. Fix or drop the "identical bytes" requirement before a contract binds a robosuite dataset.
- [x] **Sim2sim validation (branch `feat/robosuite-sim2sim`).** The MuJoCo-trained B2 checkpoint (step 50,000) on 20 of B2's own control report trials, both scenes, one loop: **MuJoCo 19/20, robosuite 3/20** (paired 16 vs 0, p = 3 × 10⁻⁵; the control matches B2's own report on these trials). On robosuite, 11 failures lift the cube and never carry it, 6 never lift it. Report: [docs/ROBOSUITE_SIM2SIM_20261002.md](docs/ROBOSUITE_SIM2SIM_20261002.md). Not hash-bound; counts from ten layouts.
- Not planned (user, 2026-10-02): the blue-checker-table rerun that would test whether missing visual texture breaks the carry. Sim2sim is not a priority; revisit only if wanted after a robosuite-trained policy exists.
- [x] **B3 collection done and sealed (2026-10-03): 560/560, fingerprint `e14af864…`**, 7 rejections, 0 substituted layouts, one crash at episode 541 recovered without data loss (FAILURES #41). Report: [docs/B3_ROBOSUITE_COLLECTION_20261003.md](docs/B3_ROBOSUITE_COLLECTION_20261003.md). Branch `feat/b3-robosuite-cubes`, merged into `main` 2026-10-05. Seal, provenance, plan, journal and per-episode instructions independently re-verified the same day. **Then: plan 2, training ([docs/superpowers/plans/2026-10-03-b3-xvla-training.md](docs/superpowers/plans/2026-10-03-b3-xvla-training.md)) and plan 3, evaluation ([docs/superpowers/plans/2026-10-03-b3-xvla-evaluation.md](docs/superpowers/plans/2026-10-03-b3-xvla-evaluation.md)): both done 2026-10-05.** `experiments/robosuite_scene/*.py` and the B3 spec/plan are now bound by the seal: do not edit them.
- [x] **B3 trained and evaluated (2026-10-05).** Training v2, 156,000 updates; report on reserved layouts, 160 trials per checkpoint: **110 / 116 / 141 / 105 of 160 = 68.8 / 72.5 / 88.1 / 65.6%** at steps 62,400 / 93,600 / 124,800 / 156,000 (robosuite 1.5.2, not comparable with earlier results). Held-out colour pairs within 10 points of trained, wrong-cube rate under 5%, at all four. Final checkpoint worst; 124,800 best by 16–23 points (paired). **Quote 124,800 as about 85% overall and about 75% with distractors present**: it was both selected and measured on the report, and the independent screen gives 13/20 on the distractor condition (report addendum). Report: [docs/B3_XVLA_EVALUATION_20261005.md](docs/B3_XVLA_EVALUATION_20261005.md). Branch `feat/b3-robosuite-cubes` (with the training-resume work of `feat/b3-resume`), merged into `main` 2026-10-05.
- [ ] **B3 follow-ups the report left open** (none started; each is a hypothesis, not a plan): (a) why side B stalls 12–14 cm above the target and side A does not (stalls at 156,000: 36 of 160 trials, all side B, also with no distractor): compare the expert demonstrations on the two sides, joint-limit margins on the side-B reach; (b) log cube orientation in the evaluation trace before any further rollouts: 23 of 69 near-miss placements released within 2 cm of centre and still toppled, yaw untested; (c) a distractor 8–12 cm from the pick raised pick-phase failures from 2/160 to 20/160 on the same layouts: needs a randomised contrast, not a between-layout reading; (d) confirm the 124,800 figure on fresh layouts — partly done: the screen, never used for selection, gives 13/20 on its one condition (report addendum); the three persistently failing layouts are all side-B picks near the base, the corner B2's `evaluation_054` failed at; (e) ~~decide whether to publish~~ decided 2026-10-05: steps 124,800 and 156,000 go to Windows and Hugging Face (below).
- [x] ~~B3 design done (2026-10-03); collection is next.~~ Superseded by the line above; kept for the design. Spec: [docs/superpowers/specs/2026-10-03-b3-robosuite-cubes-design.md](docs/superpowers/specs/2026-10-03-b3-robosuite-cubes-design.md) (reviewed and approved by the user). Plan 1 of 3, collection: [docs/superpowers/plans/2026-10-03-b3-robosuite-collection.md](docs/superpowers/plans/2026-10-03-b3-robosuite-collection.md). Branch `feat/b3-robosuite-cubes` (created 2026-10-03). Decided with the user: colour randomisation over four colours with 2 of 12 pairs held out, harder distractors (none / one / two at 20 / 40 / 40%), denser edge coverage in the same two regions, 500 training episodes, the 2 cm window unchanged. Training is sized at 156,000 updates, about 36 h.
- [x] ~~Re-think the B3 axis list before designing Phase 2~~ (user, 2026-10-02) — done in the spec above; kept for the reasoning. The list below was written before Phase 1 and before the sim2sim result. Inputs that change it: the ≤ 1.2 cm gate conflicts with the expert's own jitter (item a); the scene change is itself a large visual axis; the sim2sim result will say how much of B2's capability survives it.
      Idea to evaluate there (user): a larger cube, still within the 8 cm finger opening, to make placement more forgiving. Notes for that discussion: the in-hand offset comes from jitter along the pad surface, which a larger cube does not remove; a wider cube tips later, so the physical tolerance grows with the edge length (B2 measured the cliff at half the 4.2 cm edge); but `stack_metrics`' 2 cm xy window is a fixed constant in shared code, so the scored tolerance does not grow unless a new metric is defined; and a wider cube leaves less finger clearance on approach.
- Superseded by the 2026-10-03 spec above (kept as the original sketch): **Phase 2 — B3 combined-axis collection**, designed only after Phase 1 passes. Will layer: tighter release precision (expert gate ≤ 1.2 cm), harder distractors (reduced clearance / in the carry path), full colour randomisation, extra demonstrations near the acquisition-weak `054`/`048` pick regions, and a floor/prop change. Horizon likely 75-100 k updates.
- [x] **Publish B3 steps 124,800 and 156,000** (user, 2026-10-05; done 2026-10-06: copied to Windows byte-identical, published as private HF repos (both switched to public 2026-10-09; the missing tokenizer/runtime and evaluation code were added by the 2026-10-09 portability repair) [dataset](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b3-robosuite-560) and [model](https://huggingface.co/VivekSai07/xvla-franka-b3-robosuite-124800), 5,010 files hash-verified on the Windows side; tokenizer path and external evaluation code left as documented portability gaps): staged and hash-verified at `outputs/downloads_20261005/b3_robosuite/` (zero extra disk); copy to Windows, verify the two hashes, upload as a private model repo — commands in the staging `README.md`, record in [docs/B3_PUBLICATION_20261005.md](docs/B3_PUBLICATION_20261005.md). Keep all RTX originals.
- [x] **Next project — robosuite's built-in props / real-life objects** (user, 2026-10-02) — *underway*: P0 (scene + expert) GO 2026-10-05, P1 trained and evaluated 2026-10-09, P2 dataset sealed 2026-10-13; roadmap below. Original note: move from cubes to robosuite's object library (cans, bottles, boxes, cereal, bread, milk; `robosuite.models.objects`). This is a jump, not an increment: the scripted expert assumes a cube it can grasp top-down at a fixed height and scores a cube-on-cube stack, so it needs new grasp strategies per object, a new task definition and success metric, and instructions that name objects.
- **Props P1 dataset SEALED (2026-10-07): 540/540 episodes**, fingerprint `86e3aadb…`, 7 rejections of 547 (all lemon), 2 substitutions, 0 held-out pairs collected. Seal, cell balance and reserved tiers independently re-verified. Report: [docs/PROPS_P1_COLLECTION_20261007.md](docs/PROPS_P1_COLLECTION_20261007.md). **Trained and evaluated 2026-10-09: confirm 100% (20/20), report 75% (60/80), checkpoint 126,900 — [docs/PROPS_P1_EVALUATION_20261009.md](docs/PROPS_P1_EVALUATION_20261009.md).**
- **Props P0 PASSED (2026-10-05): 283/288 = 98.3%** after two spec amendments I had got wrong (success by lowest point, not centre; start region clear of the arm's working envelope). Report: [docs/PROPS_P0_RERUN_20261005.md](docs/PROPS_P0_RERUN_20261005.md). ~~Next: P1 collection~~ — done 2026-10-07 (plan: [docs/superpowers/plans/2026-10-06-props-p1-collection.md](docs/superpowers/plans/2026-10-06-props-p1-collection.md)). Branch `feat/props-p0`, merged into `main` by PR #1.
- **Props roadmap (agreed direction, user 2026-10-05; each phase gets its own spec → plan → go/no-go):**
      Spec: [docs/superpowers/specs/2026-10-05-props-sorting-design.md](docs/superpowers/specs/2026-10-05-props-sorting-design.md). P0 plan: [docs/superpowers/plans/2026-10-05-props-p0-scene-expert.md](docs/superpowers/plans/2026-10-05-props-p0-scene-expert.md) (ends with a report and a handoff prompt). P1 bins are told apart by colour (blue / orange), not by side; they swap sides.
      - **P0 — scene and expert bring-up (no policy). GO 2026-10-05: 283/288 = 98.3%** after the spec amendments below (first run 164/300, no-go). Two-bin scene composed like `robosuite_scene`
        (table at z = 0, our two cameras), 4–6 robosuite objects with explicit masses, our joint-target
        IK expert extended with per-object grasp height and yaw-aligned grasps. Gate: the expert drops
        every object into either bin from random yaw, ≥ 95% over a few hundred scripted attempts,
        deterministic replay, yaw logged. The controller question is settled here cheaply: switch to
        OSC only if the IK expert cannot reach that rate.
      - **P1 — instruction sorting, one object, no distractors. Dataset collected and sealed
        2026-10-07: 540/540 episodes (480 train, 60 validation), 7 rejections (all lemon), 2
        substitutions, true wall-clock 9h38m (the "~6 h" estimate only timed the scripted attempt, not
        the per-episode video encode/commit).** Report: [docs/PROPS_P1_COLLECTION_20261007.md](docs/PROPS_P1_COLLECTION_20261007.md).
        "put the <object> in the <bin> bin"; bins swap sides every episode; `milk→blue` and
        `bread→orange` held out of training entirely (0 collected, confirmed). **Trained and evaluated
        2026-10-09** (confirm 20/20, report 60/80 on 126,900; 80 reserved layouts expert-feasible 80/80):
        [docs/PROPS_P1_EVALUATION_20261009.md](docs/PROPS_P1_EVALUATION_20261009.md).
      - **P2 — instruction sorting with 1–2 distractor objects. Dataset collected and sealed 2026-10-13:
        540/540 episodes (480 train, 60 validation; 270 one-distractor, 270 two-distractor), 0 rejected,
        0 substitutions, 0 distractors touched, fingerprint `b1d642b3…f4d898`, re-verified in a second
        process; 12 parallel shards, 54 min wall-clock (10.8 h of serial work).** Report:
        [docs/PROPS_P2_COLLECTION_20261013.md](docs/PROPS_P2_COLLECTION_20261013.md). Spec (now bound):
        [docs/superpowers/specs/2026-10-13-props-p2-distractors-design.md](docs/superpowers/specs/2026-10-13-props-p2-distractors-design.md).
        Branch `feat/props-p2`, merged into `main`. Training package `experiments/xvla_props_p2/` (P1's recipe,
        byte-equal faster loop, overnight pause, thermal guard) passed every gate 2026-10-13 and trained
        126,900 updates. **Evaluated 2026-10-16** with `experiments/xvla_props_p2_eval/` (parallel workers,
        `wrong_object` decided by what is lifted, 120/120 layouts expert-feasible): winner 101,520, report
        75.0% (90/120), confirm 16/20; two of four pre-registered criteria met (wrong-object 5.8% against 5%,
        held-out gap 30 points against 10):
        [docs/PROPS_P2_EVALUATION_20261016.md](docs/PROPS_P2_EVALUATION_20261016.md). The original roadmap line follows.
      - *(roadmap)* **P2 — instruction sorting with 1–2 distractor objects** (the main dataset, B3-sized). The
        policy must find the named object among others. Random yaw; held-out pairs; several
        checkpoints + an untouched confirmation set. **No near-base stratum** (spec decision
        2026-10-06): P0 found the expert cannot reach starts closer than about 0.285 m to the base,
        so P2's layouts start at x ≥ 0.27 and the near-base corner is not tested unless the approach
        path is rebuilt. **Collection runs as parallel CPU-rendered shards (user decision 2026-10-13,
        option A):** contiguous slot blocks per worker, merged in slot order, OSMesa kept so pixels and
        byte-determinism match P1. Measured on one P1 episode: physics 0.6 s, OSMesa rendering ~37 s,
        frame write + AV1 encode 22.9 s; P1 ran all of it serially (9 h 38 m).
      - [x] **GPU (EGL) image built and checked 2026-10-19 (merged into `main` 2026-10-21):** one P1 attempt,
        five runs: OSMesa in the new image is byte-identical to the base image, physics and recorded data are bit-identical
        across all runs, EGL repeats differ by 1 to 2 pixels of 904 images (1 level), 7.7× faster (4.7 s against 36.5 s).
        New image `sha256:fec83621…ade5`. **Still not adopted:** EGL and OSMesa frames differ on 50% of pixels, so a phase must
        use one renderer end to end, the P2 checkpoints must not be evaluated under EGL without a paired test, and the
        byte-equal determinism gate must be replaced for that phase.
        [docs/EGL_RENDERING_CHECK_20261019.md](docs/EGL_RENDERING_CHECK_20261019.md). Original deferral (user, 2026-10-13), kept:
        **do it at a phase boundary with its own side-by-side check, never mixed into a phase's other changes**. Measured 2026-10-13 on one P1
        episode (probe scripts in gitignored `outputs/p2_speed_probe/`): EGL 2.2 s vs OSMesa 37.8 s per
        attempt (~17×), same expert outcome. Three things to settle first: (1) the robosuite image lacks
        `libEGL.so.1` (Ubuntu `libegl1`; the NVIDIA EGL driver is injected by the runtime) — adding it
        means a new image digest; (2) EGL is not byte-deterministic: across 3 identical runs 4–6 of 904
        images differed, each by 1 pixel by 1 level — the P0/P1 byte-equal determinism gate would need
        an explicit tolerance decided before data; (3) EGL vs OSMesa differ on 72% of pixels (mean 1.2,
        max 105 levels), so training data and evaluation rollouts must use the same renderer, and the
        switch must not coincide with another variable. Also worth it for evaluation, which renders too.
      - **P3 — rule-based Food vs Drinks — dataset sealed 2026-10-20 (544/544), trained and evaluated, failed; deleted 2026-10-27**
        ([spec](docs/superpowers/specs/2026-10-19-props-p3-food-drinks-design.md),
        [collection plan](docs/superpowers/plans/2026-10-19-props-p3-collection.md),
        [collection report](docs/PROPS_P3_COLLECTION_20261020.md),
        [handoff to a new session](docs/superpowers/handoffs/2026-10-19-props-p3-handoff.md)): fixed prompt, one object, P0's
        blue/orange bins (drinks orange, food blue), bottle and cereal held out; decisions D1–D7 confirmed 2026-10-20. Original
        roadmap text: two visually distinct containers. First one object per
        episode with a fixed prompt; then held-out objects as a stretch test (zero-shot category from
        six objects is a long shot and is read as exploratory). Multi-object sequential sorting only
        after that.
- **Props P0 result and its resolution (2026-10-05)** — package `experiments/props_scene/` (robosuite image,
  no contract of its own; frozen through the P1 and P2 seals; merged into `main` by PR #1). First run, written spec: 164/300 = 54.7% no-go ([docs/PROPS_P0_20261005.md](docs/PROPS_P0_20261005.md)).
  Decisions taken by the owner of the plan the same day, and what they gave ([docs/PROPS_P0_RERUN_20261005.md](docs/PROPS_P0_RERUN_20261005.md)):
      1. **Success** is now: centre inside the bin in x and y, **lowest point** below the wall top, gripper open and clear, 30 ticks. Re-run: milk 47/48, bottle 46/48, cereal 48/48.
      2. **Start region** is x ≥ 0.27 (was 0.22), no near-base stratum, 288 attempts. 4 of the 5 remaining failures start at radius 0.278–0.283 m, inside the reach band the region still admits (5.9% of its area); beyond 0.285 m: 274/275.
      3. The square-footprint fix is accepted; centre = centre of mass; the cube's contact parameters are kept; OSC is dropped as a gate input (148/300 on the first run's attempts) and kept as evidence.
      Open from the re-run: a lemon at y ≈ −0.12, yaw ≈ 80° fails repeatably with joint 4 not saturated (cause unknown); the gate's built-in determinism recheck only lands on one bin arrangement.
- **Inputs for the props project, collected 2026-10-05** (read before its brainstorm):
      - *Carried over from B3, as questions to design in, not as more cube work:* (1) the +y side near
        the robot base is the policy's weak corner in both B2 and B3 — put a stratum of layouts there
        and check the expert and policy for left/right asymmetry before trusting a layout
        distribution; (2) object orientation decides grasps and placements — log yaw (and full
        orientation) in every trace from the start; (3) the place phase was the unstable part of B3
        (topples, stalls) — bins make placement forgiving, which should remove the topple mode but not
        the stall.
      - *Process lessons:* evaluate several checkpoints spread through training and keep an untouched
        set of layouts to confirm the winner (B3's report both selected and measured); define a
        detailed failure-mode classifier before the first rollout (FAILURES #45); training on one
        video file per episode needs `--ulimit nofile=65536:65536` (#42); size `/tmp` for the
        HuggingFace cache (#41); resumable training from the start (#43).
      - *What robosuite 1.5.2 ships (checked in our image):* a ready **bin-sorting setup** —
        environment `PickPlace` with `BinsArena`: a source bin and a target bin split into four
        compartments, four objects (milk, bread, cereal, can), success = object inside its
        compartment volume with the gripper away. It assumes a 0.8 m table and robosuite's OSC
        controller, so for us it is a template: compose our own `BinsArena` scene the way
        `robosuite_scene` composes `TableArena`, and drive it with our expert.
        Other objects: lemon, bottle, plus `HammerObject`, `PotWithHandlesObject`, nuts/pegs (assembly,
        not sorting), and primitives (box, cylinder, ball, capsule). Each food object also has a
        visual-only twin (`*VisualObject`) for showing a target.
      - *Measured object sizes (bounding box) and masses:* milk 4.0 × 4.0 × 14.4 cm, 0.021 kg;
        bread 4.8 × 4.0 × 4.8 cm, 0.004 kg; cereal 10 × 3 × 15 cm, 0.067 kg; can 5.2 × 5.2 × 8 cm,
        0.016 kg; lemon 7 × 4 × 4.3 cm, 0.003 kg; bottle 5.5 × 5.9 × 16 cm, 0.014 kg. Our gripper
        opens 8 cm, so all of them fit across their narrow side, but cereal (3 cm wide, 10 cm long)
        and lemon must be grasped at the right yaw, and the tall ones (milk, cereal, bottle) need a
        per-object grasp height. Several are implausibly light (bread 4 g, lemon 3 g) — set masses
        explicitly.
      - *User's idea (2026-10-05): sort objects into two bins.* Design questions for the brainstorm:
        what decides the bin (an instruction naming object and bin, or a rule such as food vs drink);
        swap the bins' sides between episodes so position never predicts the answer (the B1 lesson,
        FAILURES #27); hold out some object × bin combinations, as B3 held out colour pairs; how many
        objects per scene and whether the rest act as distractors; random yaw as an axis; and the
        controller decision below.
- [x] **Decide the controller and action space for the props project: our joint-target IK or robosuite's OSC_POSE** (user, 2026-10-02). *Settled at P0 (2026-10-05): our joint-target IK expert passed the gate (283/288); OSC was dropped as a gate input (148/300 on the first run's attempts) and kept as evidence. P1 and P2 use 7 absolute joint targets + gripper.* Original considerations:
      - *Ours* (pinocchio IK → 7 absolute joint targets + gripper, 30 Hz): every existing dataset and checkpoint uses it, so old policies can still be rolled out and sim2sim stays possible; robosuite's runtime is not needed (`env.step` is never called). Note from the sim2sim result: the old checkpoints roll out on the robosuite scene but score 3/20, so "existing checkpoints stay usable" is a weak reason to keep this action space.
      - *OSC_POSE* (robosuite built-in: 6 end-effector deltas + gripper): the format public robosuite / robomimic / LIBERO data uses, and compliant in contact (the SmolVLA checkpoint this repo started from, `smolvla_libero`, is LIBERO-finetuned). Costs: a new action space (no reuse of existing checkpoints or datasets), a scripted expert rewritten to emit deltas, `env.step` and robosuite's control loop in the collection path, and re-proving determinism and replay.
      - Questions to answer in the brainstorm: do the props need contact-rich or tilted grasps that joint targets handle badly? Do we want to mix in public data or compare against public baselines? Is rolling out the cube-trained checkpoints on the props scene still wanted?
      - A cheap experiment that would inform it: have the existing expert drive one prop grasp with joint targets, and the same grasp through OSC_POSE, and compare success and smoothness.
- [x] *(closed 2026-10-05, not pursued: the cube programme ended with B3)* Confirm 50,000's 91.9% on a fresh holdout as part of Phase 2's eval, since the B2 50k number is the larger of two reported results.

### B3 (original, superseded) — colour randomisation as a standalone axis
- [x] Superseded by B3, which randomised four colours with two held-out pairs.

### Not planned
- **Bounding-box / segmentation-mask injection.** Would likely raise numbers but hands the model
  the spatial answer, defeating the goal of the vision encoder learning spatial grounding.
  Also incompatible: our models are not autoregressive token decoders (SmolVLA uses flow
  matching, X-VLA a soft-prompt/action-hub head), so coordinate-text injection does not transfer
  from RT-2/OpenVLA. A 4th mask channel would change the observation schema and force
  recollection of every sealed dataset.

---

## Known debt

- [ ] `experiments/{xvla,smolvla}_local_varied/README.md` state seal `f8b10a49…`; actual is
      `ae8828ea…`. Hash-bound — cannot be corrected without invalidating those runs' bindings.
- [ ] `experiments/broad_workspace_data/README.md` still reads as pre-collection. Same constraint
      (`scenes.provenance()` appends it).
- [ ] SmolVLA local-varied metadata binding is unreproducible (pre-re-seal). Cross-eval uses a
      data-level binding instead; documented in the result JSON.

---

## Before launching any long run — checklist

The checklist lives in one place:
[.github/skills/launching-a-run/SKILL.md](.github/skills/launching-a-run/SKILL.md), section 1.
It was moved there on 2026-10-02 because three copies (here, the skill, and
`copilot-instructions.md`) had started to differ. Every item cost real GPU time at least once.

## After every finding — documentation pass

Results, learnings and mistakes all trigger this, in the same turn they are produced:

- [ ] Grep the live docs for the specific figures and claims the finding changes
- [ ] Update **every** hit — README.md, docs/RESULTS.md, docs/FAILURES.md, docs/BENCHMARK_STATUS.md,
      TODO.md, the relevant spec
- [ ] Where a claim is superseded rather than wrong, keep the history: what was believed, what
      replaced it, why the first reading was mistaken
- [ ] Add recurrable mistakes to docs/FAILURES.md — including analysis errors, not just pipeline ones
- [ ] Re-check relative links

---
---

# ARCHIVE — original docker-compose pipeline (last accurate ~2026-09-14)

Historical record of the `franka_libero_stacking_v1/v2` HF datasets and the v1 0% /
language-grounding findings. Superseded by the `experiments/` packages above.

## Environment Setup
- [x] Identify RTX 5090 + CUDA 13.0 (driver 580.142) on new lab machine
- [x] Confirm Docker 29.4.2 + NVIDIA runtime already configured
- [x] Write `Dockerfile` — CUDA 12.8.1 + PyTorch nightly cu128 (required for Blackwell sm_120)
- [x] Write `docker-compose.yml` — 5 services: `collect`, `push`, `train`, `server`, `client`
- [x] Write `.env.example` + `.gitignore`
- [x] Build `smolvla:latest` image
- [x] Verify GPU inside container — PyTorch 2.10.0+cu128, no sm_120 warning, RTX 5090 detected

## Security / Housekeeping
- [x] Remove hardcoded HF token from `huggingface/push_dataset_to_hub.py`
- [x] Fix Windows path → container path `/vla/datasets/franka_libero_stacking_v1`
- [x] `SERVER_URL` in `inference/franka_inference_libero.py` reads from env var
- [x] Remove `datasets/` from git — push to HF Hub instead
- [x] Ignore `.local/`, `.bash_history`, generated PNGs in `.gitignore`

## Dataset v1 (deprecated)
- [x] 50 episodes collected on Windows GTX 1650 (single red→blue instruction)
- [x] Push `franka_libero_stacking_v1` to HF Hub (`VivekSai07/franka_libero_stacking_v1`, branch `v3.0`)
- [x] Train + infer on v1 → **0% success rate** (root cause: single instruction, no language grounding)

## Dataset v2
> Root cause of v1 failure: single fixed instruction, 50 episodes, deterministic trajectories.
> Full analysis in DATASET_DESIGN.md.

- [x] Document v1 failure analysis and v2 design rationale → `DATASET_DESIGN.md`
- [x] Upgrade `data_collection/collect_libero_dataset.py`:
  - [x] 12 instruction variants (all colour-pair combinations of red/green/blue/yellow)
  - [x] 300 episodes target (~25 per variant)
  - [x] Variable distractor count per scene (0–2 extra cubes)
  - [x] Wider spawn range: x=[0.20,0.60], y=[-0.40,0.40]
  - [x] Min cube clearance: 15 cm (up from 12 cm)
  - [x] ±5 mm Gaussian jitter on approach trajectories
  - [x] Success threshold tightened: xy < 2 cm (was 3 cm)
  - [x] Success verification uses instructed colour pair (not hardcoded red/blue)
  - [x] 512×512 images (matches SmolVLM2 max_image_size.longest_edge=512)
  - [x] Rolling ETA + live `\r` status line (phase bar, sim time, ETA overwrites in place)
- [x] Collect 300 episodes — completed overnight (~3.8h, avg 41s/ep)
  ```bash
  docker compose run --rm collect
  ```
- [x] Push `franka_libero_stacking_v2` to HF Hub (`VivekSai07/franka_libero_stacking_v2`, branch `v3.0`)
  ```bash
  docker compose run --rm push
  ```

## Training v2
- [x] Hyperparameter analysis based on actual model config (SmolVLM2-500M backbone, 1.2GB weights):
  - batch_size: 24 → **32** (safe for 32GB VRAM, est. ~25GB peak)
  - steps: 20K → **80K** (3 epochs → 16 epochs; 3 epochs confirmed undertrained)
  - LR: keep default 1e-4 (no linear scaling for fine-tuning)
- [x] Train on v2 dataset — **completed overnight (~8h, batch=32, steps=80K)**
  ```bash
  docker compose run --rm train   # auto-resumes from latest checkpoint if interrupted
  ```
- [x] Model pushed to HF Hub: `VivekSai07/smolvla_franka_libero_stacking_finetuned_b32_s80k`
- [x] Monitor via W&B — project `smolvla_franka_project`

## Inference v2
- [x] Fix inference script mismatches vs training:
  - [x] Renderer: 480×640 → 512×512
  - [x] Instruction: removed "perfectly" to match training format exactly
  - [x] Colour combos: expanded from 2 (red↔blue only) to all 12
  - [x] Spawn range + cube clearance: matched to training (0.15m clearance)
  - [x] MAX_STEPS: 600 → 1200 (40s window for suboptimal paths)
  - [x] Video folders: `inference_videos/<policy_name>/run_N/` naming convention
- [x] Fix server URL: `server:8000` → `localhost:8000` (client uses `network_mode: host`)
- [ ] Run 30-episode evaluation and record success rate
  ```bash
  # Terminal 1
  docker compose up server   # wait for "✅ Policy & Processors loaded!"

  # Terminal 2
  xhost +local:docker
  docker compose run --rm client
  ```
- [ ] Review videos in `inference/inference_videos/smolvla_franka_libero_stacking_finetuned_b32_s80k/`
- [ ] Compare success rate vs v1 (0%)

## Next Steps (if inference success rate still low)
- [ ] Collect dataset v3 — 600 episodes (50 per instruction variant) for reliable placement
  - Update `docker-compose.yml` collect command: `--episodes 600`
  - Retrain with steps=120K
- [ ] Try `smolvla_base` (not fine-tuned on LIBERO) — compare zero-shot vs fine-tuned

## Action/Observation Alignment Fix (2026-09-14) — READ THIS FIRST if inference still fails to pick

Root cause found: all three data collectors (`collect_libero_dataset.py`, `collect_v2_dataset.py`,
`collect_stacking_dataset.py`) recorded each frame's `action` as the *previous* control tick's
stale commanded joint target instead of the action actually computed and sent that tick — so every
episode in `franka_libero_stacking_v1` and `v2` has `(state_t, action_{t-1})` instead of the correct
`(state_t, action_t)`. Confirmed against lerobot's own `lerobot_record.py` reference loop
(`observe → act → send → save`, always in that order). This is a strong candidate for why the
fine-tuned model in v1/v2 "reaches around" the cube but never picks it — see
`docs/superpowers/specs/2026-09-14-fix-action-observation-alignment-design.md` for the full
analysis and `docs/superpowers/plans/2026-09-14-fix-action-observation-alignment.md` for exactly
what changed.

- [x] Fix the reorder bug in all 3 collector scripts so `action` matches what was actually sent
- [x] Add `data_collection/verify_action_alignment.py` — sanity-checks that a collected dataset's
      `action_t` is closer to `state_{t+1}` than to `state_t`
- [ ] **On the actual GPU/Linux collection rig** (this fix was authored on a Windows dev machine
      with no Docker/GPU access — none of this has been runtime-verified yet):
  ```bash
  docker compose run --rm collect python data_collection/collect_libero_dataset.py --episodes 2 --headless
  docker compose run --rm collect python data_collection/verify_action_alignment.py \
    --root datasets/<the_new_versioned_dir> --repo-id <same_name>
  ```
  Confirm it prints `PASS`. If `FAIL`, something in the reorder is still wrong — re-check
  `collect_libero_dataset.py`'s per-tick loop against the design doc before going further.
- [ ] Once verified: re-collect a fresh `franka_libero_stacking_v3` (or overwrite v2 collection
      logic is unchanged otherwise — same 12 instructions, same episode count) with the fixed
      collector, push to HF Hub (`v3.0` branch), and retrain from scratch (previous checkpoints
      were trained on mislabeled data — do not resume from them)
- [ ] Re-run the 30-episode inference evaluation and compare success rate against v1/v2 (both 0%
      or near-0%) to confirm whether this was in fact the root cause
- [ ] If success rate is still poor after this fix, the next suspects (not yet investigated) are:
      the one-tick control latency between `d.ctrl` write and `d.qpos` read at the top of the next
      loop iteration, and whether SmolVLA's `delta_timestamps`/action-chunking config expects a
      different temporal convention than a single-step `(state_t, action_t)` pair
