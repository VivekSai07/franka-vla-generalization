# B3 — the final cube dataset, collected in the robosuite scene — design

**Date:** 2026-10-03
**Status:** design sections 1 (dataset) approved in conversation; sections 2–4 (collection
mechanics, training, evaluation) written from B2's precedents and awaiting the user's review of
this document
**Supersedes:** the "Phase 2 — B3 combined-axis collection" bullet in `TODO.md` and the Phase 2
sketch in [2026-10-01-robosuite-scene-bringup-design.md](2026-10-01-robosuite-scene-bringup-design.md)
**Branch:** `feat/b3-robosuite-cubes`

## What the user wants

B3 is the **last** cube experiment. Every axis that was meant to be tested on cubes goes into this
one dataset, collected in the robosuite scene. The trained policy must show **both language
grounding and spatial understanding**; more demonstrations and more training time are acceptable
to get there. After B3, new tasks use real-life props in robosuite.

Success is a measured rollout result on held-out layouts in the robosuite scene, structured so that
when something fails it is possible to say which axis failed.

## Decisions made with the user (2026-10-02/03)

| question | decision |
|---|---|
| axes | full colour randomisation, harder distractors, extra pick coverage at the region edges |
| not included | tighter release gate (the 2 cm window stays), bigger cube, vertical variation, new cube sizes |
| language test | 2 of the 12 colour pairs are never trained and are evaluated |
| size | 500 training episodes |
| distractors | mixed: none / one / two, 20% / 40% / 40% |
| workspace | B1/B2's two regions, sampled more densely at their edges |
| controller | the repo's joint-target IK expert (the OSC question belongs to the props project) |

Assumptions the user has not been asked about: X-VLA only (no SmolVLA run); B2's reserved-layout
structure (20 screen + 40 report); training exposure matched to B2 (5 epochs).

## Why three axes at once is acceptable here

The repo's rule is one variable at a time. The user accepted the attribution trade-off for B3
(recorded in `TODO.md`): a strong result from one experiment if it works, limited attribution if
it does not. The design recovers what attribution it can **inside the evaluation**: every report
layout is run with and without distractors, half the report layouts use held-out colour pairs, and
a fixed share are edge layouts, so each axis has its own contrast.

---

## 1. The dataset

### Scene

The Phase 1 robosuite scene (`experiments/robosuite_scene/`): Panda, minimal mount, table top at
z = 0, `front_cam` and `gripper_cam` at 512 × 512, the scripted-expert model contract. Two
extensions:

- **Four cubes** — red, green, blue, yellow; half-size 0.02 m, mass 0.05 kg, rgba copied from
  `smolvla_scene.xml` (`0.8 0.2 0.2`, `0.2 0.8 0.2`, `0.2 0.2 0.8`, `0.8 0.8 0.2`).
- **Only the cubes in an episode exist in its model.** One compiled model per set of colours
  present (11 sets of two, three or four colours), built once and cached. No cube is ever "parked"
  off-table: parked cubes carried residual velocity in B2 (FAILURES #32, #33).

### Colours and instructions

- An episode names a pick colour and a different place colour: 12 ordered pairs.
- **Held out of training: `yellow→red` and `green→blue`.** In the 10 trained pairs red and blue
  are the pick colour three times and the place colour twice; green and yellow the reverse. Every
  colour is trained in both roles.
- Instruction text: `pick up the <pick> cube and place it on top of the <place> cube` — the
  wording every earlier dataset used. `red→green` stays a trained pair.

### Layout

- Pick and place cubes sit in B1/B2's two regions: A = x ∈ [0.20, 0.35], y ∈ [−0.45, −0.20];
  B = x ∈ [0.20, 0.35], y ∈ [0.20, 0.45]. One target cube in each region.
- **Side** (which region holds the pick cube) is balanced, so colour never predicts position.
- **Cells:** 10 trained pairs × 2 sides = 20 cells. **25 training episodes per cell = 500**, plus
  **3 validation episodes per cell = 60** (training-loss monitoring only; not a capability test).
- **Edge coverage (the "extra pick coverage" axis).** In each cell 20 training episodes draw both
  cube positions by Latin hypercube over the full regions, as before. The other 5 put the **pick
  cube** in an edge band and draw the place cube normally:
  - inner corner: x ∈ [0.20, 0.25], |y| ∈ [0.20, 0.25] — where B2's `evaluation_054` failed
    (pick at (0.242, 0.203), reach 0.316 m);
  - outer corner: x ∈ [0.32, 0.35], |y| ∈ [0.40, 0.45] — where `evaluation_048` failed
    (pick at (0.339, 0.425), reach 0.544 m).
  Cells alternate 3 inner + 2 outer and 2 inner + 3 outer, giving 50 + 50 over the dataset (20%).

### Distractors

- The cubes of the two colours that are neither pick nor place.
- Per cell of 25: **5 with none, 10 with one, 10 with two.** With one distractor the two
  candidate colours alternate. Validation: 1 / 1 / 1 per cell.
- Distractors are drawn in the band x ∈ [0.20, 0.35], y ∈ [−0.45, 0.45] (B2's band).
- **Harder than B2.** B2 kept every distractor ≥ 15 cm from both target cubes and from each other.
  B3: minimum centre-to-centre clearance **8 cm**, and in half of the distractor episodes of each
  cell one distractor is **near**: 8–12 cm from the pick cube (5 episodes) or the place cube (5).
  The other half keep B2's ≥ 15 cm from both targets.
- **8 cm is an estimate, fixed by a pilot before the full run.** The open gripper spans about
  10 cm across its closing axis, so a neighbour 8 cm away along that axis leaves about 1 cm. The
  pilot measures the expert's rejection rate on near-distractor layouts; the ladder is 8 → 9 →
  10 cm, taking the smallest value with a rejection rate ≤ 20%. The value is frozen in the plan
  before collection starts and never changed afterwards.
- An attempt is rejected if the expert moves any distractor more than 2 cm (B2's rule).

### Per-episode variation kept from B2

Home-pose jitter (each arm joint ±1°), hover-height jitter on the two approach phases (±3 cm),
the expert's own ±5 mm target jitter. 528 frames, 30 Hz, 9-value state, 8-value action,
`FEATURES` / `ENCODING` from `experiments.paired_demonstrations.media`.

### Admission of one episode

All of: the expert reports success (30 stable ticks, the unchanged `stack_metrics` window:
xy < 2 cm, 3.5 cm < z < 4.5 cm, gripper open, no finger contact); exactly 528 frames; replay of
the recorded actions reproduces the trajectory (state error < 1e-3); no distractor moved > 2 cm;
the recorded instruction, pick and place match the plan slot. Offline metrics play no part.

### Reserved evaluation layouts (never collected)

60 layouts drawn in the same plan: 20 **screen**, 40 **report**.

| tier | layouts | colour pairs | conditions banked |
|---|---|---|---|
| screen | 20 | the 10 trained pairs, one layout per side | two distractors |
| report | 20 | the 10 trained pairs, one per side | none, and two distractors |
| report | 20 | the 2 held-out pairs, 10 layouts each, 5 per side | none, and two distractors |

In the two-distractor condition one distractor is near (8–12 cm from the pick cube on half the
layouts, from the place cube on the other half) and one is far. A quarter of each tier (5 screen,
10 report) are edge layouts, split between the two corners. Each banked scene stores its settled
start state, so evaluation does not depend on re-deriving it.

---

## 2. Collection mechanics

New package **`experiments/b3_robosuite_data/`**, built on `experiments/robosuite_scene/`
(imported, not copied). It is not a fork of `b2_distractor_data`: that package is 2,300 lines tied
to the MuJoCo scene and to bit-exact replay on MuJoCo 3.13, and forks are where this repo's
green-tests-wrong-code failures came from. It runs in the robosuite image (MuJoCo 3.4.0), CPU.

- **`plan.py`** — the deterministic plan: every count above as a named constant, seeds derived
  as `SHA256(version:master:identity)`, balance enforced by construction (decks, not sampling),
  `make_plan()` / `validate_plan()`. Imports no robosuite and no MuJoCo, so the training contract
  can import it in the admission image.
- **`scenes.py`** — `build_model(colours)` (cached per colour set), `reset(model, layout, jitter)`
  (arm at `Q_HOME` + home jitter, cubes at their planned xy, 0.2 s settle, no RNG),
  `make_bank()` for the reserved layouts.
- **`expert.py`** — the Phase 1 robosuite expert plus B2's hover-height jitter. Its `run` is a copy
  of `b2_distractor_data.expert.Expert.run` with the one grasp-height constant, guarded by an AST
  test as in Phase 1.
- **`record.py`** — gates `--pilot`, `--collect`, `--resume`, `--status`. Frames are buffered in
  memory and written to the dataset only after the episode passes admission, so a rejected attempt
  never touches the dataset and no rollback path is needed. A journal line is appended per
  attempt. A rejected attempt retries the **same slot** with the next home-jitter and expert-seed
  draw (never skips a slot); 3 attempts per layout. A layout that fails all three is substituted
  automatically by a deterministic redraw of the same slot (same cell, profile and edge class; up
  to 3 substitutions), recorded in `substitutions.json` — B2 did this by hand, and the collection is
  meant to run unattended. A global budget of 150 rejected attempts stops a run that is failing
  systematically.
- **`validate.py`** — per-episode admission, final dataset validation (counts, cell balance,
  condition balance, edge share, instruction ↔ plan agreement), `fingerprint()`, `seal()`.
- **`contract.py`** — `provenance()`: hashes of this package's files and README, of
  `experiments/robosuite_scene/*.py`, of the shared modules it imports, and of this spec and its
  plan. Recorded at collection start; any drift before the seal aborts.

Two fixes the bring-up left open are part of this package:

- **Hash-bound.** The seal records the fingerprint, `provenance()`, the image digest and the
  frozen plan hash. Training refuses a dataset whose seal or provenance does not match.
- **Reproducible fingerprint.** Phase 1's two collections differed only in the key order of
  `meta/stats.json`. The fingerprint hashes JSON files by their canonical form (sorted keys) and
  every other file by its bytes.

Per-episode audit evidence is one small `trace.npz` (qpos, ctrl, initial state); no raw frames are
kept beside the encoded videos.

**Gate order:** unit tests → `--pilot` (clearance ladder, 2 episodes from each of the riskiest
profiles, one full replay) → user looks at pilot frames → `--collect` → seal → report.

---

## 3. Training

New package **`experiments/xvla_b3_robosuite/`**, forked from `experiments/xvla_b2_distractors/`
under the forking skill's checks. Training needs no simulator, so it runs in the admission image
with the usual `--check` → `--preflight` → `--train` gates.

- Binds the B3 seal, `b3_robosuite_data` provenance, its own source files and the base model.
- Same recipe as B2 (effective batch 8, chunk 30, peak LR 1e-4 with 500 warm-up updates and cosine
  decay to 1e-5, FP32 parameters with bf16 autocast), from the base checkpoint, no resume.
- **Exposure matched to B2: 5.0 epochs.** 500 episodes × 499 complete starts = 249,500 starts;
  × 5 / 8 = **156,000 updates** (B2: 50,000 updates on 79,840 starts = 5.0 epochs). At B2's
  measured 0.84 s per update that is about **36 hours**.
- Checkpoints at update 1 and at each whole epoch: `[1, 31200, 62400, 93600, 124800, 156000]`,
  about 30 GB.
- The 10 trained instructions are the dataset's task table; the 2 held-out instructions are added
  to the approved set only in the evaluation package.

Offline metrics are not used to select a checkpoint or to judge progress.

---

## 4. Evaluation

New package **`experiments/xvla_b3_robosuite_eval/`**, robosuite image with GPU, using the rollout
loop written for the sim2sim check. 1,200-tick cap, 30-tick stable-stack rule.

1. **Feasibility first.** The scripted expert runs every reserved scene (100 banked scenes). A
   layout the expert cannot solve is reported and excluded before any policy number is read.
2. **Screen.** The four checkpoints from epoch 2 on (`62400, 93600, 124800, 156000`), 20 screen
   layouts, two-distractor condition, one seed: 80 trials. A tie is a tie: candidates within two
   trials of the best are all carried to the report.
3. **Report.** The winner(s) on the 40 report layouts × 2 conditions × 2 seeds: **160 trials.**
4. **Read-outs, fixed now:**
   - overall success, with layout-clustered intervals;
   - **language:** held-out pairs against trained pairs (80 trials each), and the **wrong-cube
     rate** — how often the first cube lifted is not the instructed pick colour;
   - **distractors:** two-distractor against none, within layout;
   - **space:** edge layouts against interior layouts;
   - failures classified by first cause (never lifted / lifted not carried / released off target /
     wrong cube), validated on the successes first.
5. **No pass mark is set.** B3 is the last cube experiment and is reported as measured. What the
   read-outs will be taken to mean is fixed here so it cannot drift afterwards:
   - held-out-pair success within 10 points of trained-pair success, and a wrong-cube rate under
     5%, reads as *the policy follows colour words it has not seen combined*;
   - a held-out drop of more than 10 points with correct cubes picked reads as *grounding holds,
     execution on new combinations is weaker*; with wrong cubes picked it reads as *the policy
     memorised instructions*;
   - per-layout claims are not made (2 seeds per layout; the repo's rule is ≥ 8).

---

## Budget

| stage | cost |
|---|---|
| packages and tests for collection | implementation, CPU only |
| pilot | under 1 h CPU |
| collection | 560 admitted episodes at about 72 s each (Phase 1's measured rate) ≈ **11–12 h CPU** plus rejections; resumable |
| dataset on disk | about 3 GB (B2: 1.1 GB for 200 episodes) plus small traces |
| training | about **36 h GPU**, about 30 GB |
| evaluation | feasibility ≈ 2 h CPU; screen 80 trials + report 160 trials at 1–3.5 min each ≈ **4–14 h GPU** |

The machine is shared; check `nvidia-smi` and `df -h` before each long stage.

## Risks

| risk | handling |
|---|---|
| near distractors make the expert knock cubes over | the clearance ladder in the pilot; rejection budget; the value is frozen before collection |
| edge-corner picks are unreachable or unstable on the robosuite Panda (shorter fingers) | checked before the plan was handed over (2026-10-03, Phase 1 expert, red→green, no distractors): 14 of 14 episodes succeeded at inner and outer corners on both sides, including each band's extreme points (reach 0.297–0.563 m; final xy 6.8–11.7 mm). The pilot repeats it with the B3 code; an infeasible layout is substituted, never forced by loosening a gate |
| the fingertip-pad contract fix hides a real grasp problem | admission needs a *released, stable stack*, which does not depend on the grasp check alone |
| four cubes change the model per episode | one cached model per colour set; each episode's trace records its set; replay uses the same model |
| three axes at once make a failure hard to attribute | the within-evaluation contrasts in section 4 |
| 36 h of training on a shared GPU | no resume exists; confirm the machine is free for that window before `--train` |
| the white table gave a MuJoCo-trained policy little to navigate by (sim2sim) | not a risk for a policy trained in this scene; noted because it is the one visual property that could also limit B3. If the screen tier shows carry failures, a textured table becomes the first follow-up |

## Out of scope

SmolVLA; a tighter release gate; cube size or height variation; OSC control; real-life props;
sim2sim; publishing the dataset (a separate step after a measured result).

## Plans

One spec, three implementation plans, written in order because each depends on what the previous
stage measured:

1. `docs/superpowers/plans/2026-10-03-b3-robosuite-collection.md` — the collection package through
   a sealed dataset.
2. Training package and run — written once the dataset is sealed.
3. Evaluation package and report — written once training has started.
