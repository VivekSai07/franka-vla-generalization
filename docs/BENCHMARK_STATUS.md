# Franka benchmark: start here

## M0 evaluator built, frozen, A0 prepared and smoked; stopped before the A0 comparison run - 2026-10-31

Branch `feat/vla-inputs` (`experiments/vla_inputs_eval/`, `experiments/vla_inputs_arm_xvla/`, `scripts/vla_inputs_eval.sh`; core
hash `3cc8c623…`). Equivalence gate passed: the new rollout and X-VLA adapter reproduce the two lowest-ordinal successes and
failures of P2's recorded report trials of 101,520 byte for byte (702, 719, 900, 900 ticks), and an expert replay through the
real loop records the S0 signals byte-equal to the sealed sidecar. **Feasibility (unrendered expert, 656 layouts): 652
feasible, 4 dropped for every arm** (comparison 573 of 576: bread-blue-right-d1 11/12, lemon-orange-right-d2 10/12; selection 79
of 80: lemon-blue-left-d2 1/2; reasons `grasp` x3, `lost_in_carry` x1). Power recomputed at N = 573 before any policy outcome:
at alpha .0125 and a 10-point gain 0.999 / 0.969 / 0.895 for D = .19 / .30 / .40, at 8 points 0.972 / 0.831 / 0.684, at 6 points
0.771 / 0.524 / 0.387 (exact enumeration, Monte Carlo agrees within 0.0012). A0 smoke 2/2; an 8-worker one-checkpoint probe on
the selection layouts used 33.6 GB of the 48 GB card, hit 84 C and paused workers for 585 s in total. A0 has not been run on
the comparison tier. Details: the M0 report to be written after A0.

## M1 Tasks 1-3 done: pi0.5 images, pinned assets, contract - 2026-10-30

Branch `feat/vla-inputs-m1` (not pushed). **Images:** `vla-pi05:admission` (`sha256:a229f2fb…290eae`) and `vla-pi05:robosuite`
(`sha256:2abb18a5…b21`) are the existing images plus transformers@`dcddb970` and tokenizers 0.21.4, installed `--no-deps
--require-hashes` (92.6 MB each; the existing images were not touched). `test_image.py` passes in both (siglip check, pi0.5
constructs with 3,616,757,520 parameters, `pip check` equal to the parent's, `pip freeze` equal to the parent's except those two
lines) and fails in the unmodified parents. **Physics unchanged:** the 12 rendered determinism attempts of P2's collection gave
12 equal trace digests in `robosuite-pi` and in the robosuite image. **Assets:** `pi05_base` revision `9e55186a` downloaded
anonymously (14.5 GB; weights hash `0eb11ca9…` as pinned) and every asset file hashed in `assets.py`; disk 109 GB free
afterwards. **Finding:** LeRobot 0.4.4's `PI05Policy.from_pretrained(strict=True)` always reports the tied
`embed_tokens.weight` missing (the checkpoint stores it under `lm_head`) and, like any failure inside it, only prints a
warning, so the runner must not rely on it; `test_weights_load.py` asserts on key sets and every tensor. **Contract:**
`experiments/pi05_props_p2/` (dataset binding copied from X-VLA's and checked against the original on the real dataset, 193,440
windows, horizon derived to 5.0 epochs, native quantile behaviour recorded: below q01 maps to token -1, above q99 saturates
at 255, nothing clips). 69 tests OK in `admission-pi`. No runner, no GPU use, no run started; Task 4 (memory probe) is next and
needs an idle GPU.

## M2/M3/M4 specs corrected after S0; S0 merged to main - 2026-10-30

Two corrections before any M2 code. (1) **P2's X-VLA recipe does not normalise the state** (`STATE` is `IDENTITY`, enforced by
`xvla_single_task.make_config`); the M2 and M3 specs had assumed a mean/std normaliser. M2 now clips its 8 extras to S0's
percentiles and standardises them with fixed training-row constants in its own loader and rollout state builder, leaving the
9 P2 values raw; M3's boxes stay raw (already in [0, 1]). (2) S0's findings are carried into the specs: the object's weight is not
readable from the mean wrist force (M2's prediction rewritten), and `in_frame` is constant 1 (kept as a constant input in M3, the
BCE term dropped from M4a's head). S0 tests re-run independently: 70 OK. `feat/vla-inputs` fast-forwarded into `main`.

## VLA inputs S0 sidecar built and sealed - 2026-10-30

[Report](VLA_INPUTS_S0_20261030.md). All 540 sealed P2 episodes were replayed from their sealed actions with `qpos`, prop
poses and `final_qpos` byte-equal at every tick (G1; the sealed `observation.state` clause checked separately, 0 differences),
and every frame now has wrist wrench, finger forces and target/prop boxes: 244,080 frames, 87 MB, seal `7e2e06db…f58e6c`,
re-verified in a second container, built in 3.5 min on 6 pinned shards. G2 to G4 passed (G3 below 1e-9 against MuJoCo's own
sensors; G4 IoU 0.949 mean). Findings that change later plans: the wrist force is dominated by the hand's 10 N weight, so
object weight is **not** readable from the mean `fz` (M2 can rely only on transients and pose loading); finger force does
separate objects (4.1 to 6.6 N in the hold); `in_frame` is 1 for every frame (carries no information for M3); the signals
have heavy tails (max |z| 17 to 58), so the 0.1/99.9 percentile clip fixed in the report is needed. P2's provenance lists 142
files, not 76, and omits four transitive imports, which S0 binds itself. PaliGemma access was granted and the tokenizer files
(pinned revision `35e4f464`) downloaded to `outputs/research/pi05_assets/`; the 14 GB `pi05_base` weights are not.

## VLA inputs plans revised with RTX evidence - 2026-10-30

Read-only and throwaway-container checks on the workstation, written into the specs and plans before any build:
(1) the sealed P2 NPZ has no `phase_index` (the audit's R6 suggestion is refuted; segments are inferred); (2) pi0.5 needs
transformers@`dcddb970` (4.53.3, LeRobot's `fix/lerobot_openpi` head) **and** tokenizers 0.21.x (the image has 0.22.2), so
"transformers alone" was wrong; with both, the siglip check passes and pi0.5 builds with **3,616,757,520 parameters**, which puts
FP32 full tuning at 57.9 GB before activations (does not fit 48 GB); (3) pin `pi05_base` revision `9e55186a` (weights
`0eb11ca9…`, same in every revision): the latest hub preprocessor uses a step LeRobot 0.4.4 lacks; (4) PaliGemma access is
manual approval, so request it early; (5) R2 (only view 0 reaches the language encoder) is confirmed on the installed X-VLA
source; (6) eval workers that never switch checkpoint ran at ~4.2 GB each in P3, so M0 assigns one checkpoint per worker pool;
(7) the M0 C size is fixed at 576 (48 cells × 12; user, same day, before any layout was drawn; it supersedes the 384 and the
"up to 384 extra trials" in the audit entry below); (8) per-arm eval adapters and their tests
stay out of training `SOURCE_FILES` and out of the frozen M0 core. Disk 123 GB free, GPU idle, `/var/log/syslog` absent.

## VLA inputs pre-implementation audit - 2026-10-30

[Audit and research roadmap](VLA_INPUTS_AUDIT_20261030.md) records eight findings and verified primary sources.
All seven specs and six plans revised: frozen evaluation core with separately bound adapters; both camera feature paths
for M4a; instantaneous force inputs at 3 Hz replanning; fixed multiplicity family and single-seed claim limits; stronger
signal/reference checks and realistic resource scheduling. An added fixed-phase secondary comparison costs up to 384
extra trials per arm. The previous “M0 CPU alongside training” description and unqualified power claims are superseded.
Exact planning-power enumeration confirms the old unadjusted table but gives lower power at the conservative four-test
threshold. No policy/simulator tests ran on this laptop (Docker unavailable); workstation runtime gates remain pending.
No experiment source or historical result report was changed.


## VLA inputs programme designed (specs and plans, under review) - 2026-10-30

Next direction (user): hold P2's 540 sealed episodes fixed and change what the model sees. Six specs and plans, nothing built:
M0 common paired evaluation (80 selection + 384 comparison layouts, power from `2026-10-30-vla-inputs-m0-power.py`), S0 a sealed
sidecar of wrist force/torque, finger forces and target boxes, M1 pi0.5 against X-VLA, M2 force/torque in the state, M3 the target
box in the state, M4 an auxiliary box head and a force-output scoping note. Measured while writing (read-only): replaying the sealed
actions from the sealed initial state reproduces the sealed `qpos` byte for byte on 3 of 3 probed episodes; two P2 checkpoints
disagree on 23 of 120 identical report trials (discordance 0.19, a power input); and pi0.5 cannot be constructed in the pinned
admission image (LeRobot's patched transformers is missing), so M1 needs a new image. Overview:
[superpowers/specs/2026-10-30-vla-inputs-programme-design.md](superpowers/specs/2026-10-30-vla-inputs-programme-design.md).

## Props P3 closed and deleted - 2026-10-27

The user closed P3 as a failed attempt (85% on trained objects but 5.7% on held-out ones, milk only 10/20) and had its artifacts deleted, without copying them to
Windows or publishing them: the sealed dataset, all training output including the 23 GB `_preflight` and `_resumeproof` scratch, the evaluation outputs, the run
worktree and its stopped containers (about 101 GB; disk 84 to 184 GiB free). Kept: the code and docs on branch `feat/props-p3` (not pushed), RESULTS.md's figures and
FAILURES #59 and #60. The recorded figures can no longer be re-verified.

## Props P3 evaluated: trained objects 85%, held-out objects 6% - 2026-10-27

Training finished 2026-10-26 (126,900 updates, six checkpoints); `prepare` and the smoke stage passed; screen, report and confirm ran in 8 parallel
workers (about 6 h of rollouts). Winner 101,520 by the trained-only rule: report **77/239 = 32.2%**, trained objects **68/80 = 85.0%**, held-out
**9/159 = 5.7%**, confirm 22/24 = 91.7%. Trained-object bin choice was never wrong (68/68). Held-out: bottle 0/79 (77 never lifted), cereal 9/80 with
22 wrong-bin of 31 stable placements. Milk is the weak trained object (10/20 at 101,520; 12, 10, 8, 8 across the four report checkpoints; never-lifted
3, 6, 7, 11). The second pre-registered criterion is not interpretable (bottle never reaches a bin). Numbers and caveats: RESULTS.md; report: [PROPS_P3_EVALUATION_20261027.md](PROPS_P3_EVALUATION_20261027.md).

## Props P3 training, night 1: paused at 46,800 of 126,900 - 2026-10-23

The first batch ran 19:00 to 06:00 and paused itself at the last snapshot before the deadline (exit 0, snapshot 46,800; checkpoint 1 only so far, the next is 50,760).
Pace 0.515 s/update, but 474 cool-down pauses (GPU max 84 C, 5 thermal slowdowns) make the effective rate about 0.85 s/update; about 11.5 h of work remain.
Training loss 0.0024 is an offline figure and says nothing about capability. Resume is blocked: 30 GB free against the runner's 65 GiB guard (syslog 152 GB, root-owned)
and another user's process holds 37 GiB of GPU memory against the 28 GiB the runner requires. Neither check is to be relaxed.

## Props P3 training launched - 2026-10-22

All four gates passed: `--check`, `--preflight` (23 GB scratch, measured), `--resume-proof` (metrics, model and training state bit-equal after resuming
from the update-1 snapshot) and `--equivalence` (30 updates, metrics, model and optimizer equal). `--train` started 19:00 from the detached tree
`.worktrees/p3-train` at `06d1d98`; update 1 reproduces the preflight loss (0.82804). The first batch stops itself at 06:00 on 2026-10-23.
No result yet. Getting to this point needed ~87 GB reclaimed (OPERATIONS.md): `/var/log/syslog` had grown to 134 GB and the user is not in sudoers,
so the pre-P2 datasets and a Qwen cache were deleted instead; the log still grows ~33 GB/day.

## Props P3 training and evaluation packages built - 2026-10-21

`experiments/xvla_props_p3/` (admission image; fork of `xvla_props_p2`, recipe unchanged: 126,900 updates = 5.0 epochs, same checkpoint grid)
bound to the sealed P3 dataset (fingerprint `6430d34e…d151acf`), with the P2 ten-instruction machinery re-derived for one prompt: the plan
must carry only `sort the item`, the task table must be exactly `{prompt: 0}`, the expert's own wording is rejected. 72 tests, every
P3-specific change mutation-checked. `experiments/xvla_props_p3_eval/` (robosuite image + GPU): 288 reserved layouts, one seed each, the
pre-registered bin-choice criteria on stable placements with interpretability floors, 64 pure tests and 12 real-physics tests (all 288
layouts restore; the expert's actions replayed as the policy succeed for a drink and a food among trained and held-out objects; a
`wrong_bin` outcome is produced by real physics; four more cover `run_job`'s skip, quarantine and no-retry rules), every rule mutation-checked.
Expert feasibility of the 288 reserved layouts (unrendered, 43 s): **287 of 288**; `report_062` (bottle) is infeasible and is dropped and
counted, not replaced (the bottle then has 79 report layouts). An unexplained edit to `rollout.py` that thinned the recorded object trace
to every second tick was caught by the rollout test (shape) and reverted. Writing its tests found that a stage with no stable placement crashed `collect` (NaN in
a Wilson interval, FAILURES #60). No GPU run yet. Plan: [superpowers/plans/2026-10-21-props-p3-training-eval.md](superpowers/plans/2026-10-21-props-p3-training-eval.md).

## Props P3 dataset sealed: 544 Food-vs-Drinks episodes, 46 min 43 s - 2026-10-20

`experiments/props_p3_data/` (branch `feat/props-p3`): one object per episode, the fixed prompt `sort the item`, P0's bins (drinks
orange, food blue, sides swap); can and milk (drinks), bread and lemon (food) trained, bottle and cereal never collected (user
confirmed D1–D7 unchanged the same day). 544/544 admitted, 0 rejected, 0 substitutions, fingerprint `6430d34e…d151acf`, re-verified in a
second process from the recorded arrays. Expert re-test 286/288 = 99.3% (only misses: lemon, blue bin on the left); the thread cap for
16 pinned workers is pixel-neutral (12 rendered attempts, capped against default, byte-equal). 16 parallel CPU-rendered shards, 12.4 h of
serial work. The first launch died at once (shards directory not created, nothing written; FAILURES #59). Not a policy result: training and
evaluation plans are next. Report: [PROPS_P3_COLLECTION_20261020.md](PROPS_P3_COLLECTION_20261020.md).

## Props P2 fresh-layout re-test: the unseen-pair penalty is small on the same objects - 2026-10-20

Checkpoint 101,520 from the archive bundle on 240 fresh layouts of milk and bread, each under its trained and its held-out bin (480 rollouts, 8 pinned workers, 4 expert-infeasible entries dropped as 2 whole pairs). Pooled penalty **2.5 points [−4.2, +9.2]** over 238 pairs: small, existence not shown (pre-registered). Milk 57.6% / 56.8%, bread 80.8% / 76.7%. The design came from a power analysis before any rollout (the milk-versus-bread contrast first proposed was unresolvable). One deviation: the evaluator's hard-coded 120-layout feasibility check was replaced by a re-test loader, the evaluator itself unchanged. Report: [PROPS_P2_RETEST_20261020.md](PROPS_P2_RETEST_20261020.md).

## Props P2 failure diagnostics: replay is exact, the pre-registered taxonomy was not produced - 2026-10-19

Replay-only analysis of the 440 recorded P2/P1 trials (`experiments/xvla_props_p2_eval/diagnostics/`, branch
`feat/props-p2-followup`). Gate 0 passed (440/440 exact). Gate 1, the check that a grasp measure agrees with the
successes before any failure is read through it, failed twice (expert envelope 0.225; cross-checkpoint success
envelope 0.486 / 0.634 against 0.80), and the 0.80 turned out to be unattainable (two identical policies score 0.605 /
0.696 on it, FAILURES #58), so the stop rule applied and no failure was classified. Envelope-free findings: all 12
never-lifted report trials of 101,520 closed the gripper and 9 held the target without lifting; milk fails at the
hold, not the approach; never-lifted trials are not layout-concentrated for 101,520; the wrong-object association is
not supported (p = 0.17). Report: [PROPS_P2_DIAGNOSTICS_20261019.md](PROPS_P2_DIAGNOSTICS_20261019.md).

## Disk reclaim: 59 GB to 203 GB free - 2026-10-17

User-approved. Props P2 resume snapshots and the heavy preflight / resume-proof scratch (70.4 GB, every `result.json`
kept), six unused VS Code server versions, and, after the user confirmed the Windows copies were verified, the B1,
broad-workspace, SmolVLA local-varied and B2 bundles together with their checkpoints (83.8 GB). The P2 bundle
(`outputs/downloads_20261017/props_p2/`) was left in place and still passes `verify_bundle.py`; the user then hash-verified its two
weights on Windows (both matched), after which the P1 and P2 intermediate checkpoints and optimizer states were removed as well
(248 GB free, 71% full; those weights now exist nowhere, and `verify_checkpoints` cannot pass here for P1 or P2). On the user's
go-ahead the remaining unbundled weights and optimizer states (B3 intermediates, B2/B3/local-varied `training_state.pt`, the
superseded X-VLA local-varied checkpoints) followed: `outputs/` is 53 GB, down from 292 GB, and the disk is 64% full. Hardlink accounting mattered: the SmolVLA local-varied weights were linked from a second older bundle, so removing
one bundle would have freed nothing. A safety assertion also fired once on a substring match against bundled source code
(`local_varied_data/`) before any deletion and was tightened. Details: [OPERATIONS.md](OPERATIONS.md) Storage.

## Props P2 trained and evaluated: 75.0% report, two of four criteria met - 2026-10-16

X-VLA trained 126,900 updates on the P2 seal, then evaluated with `experiments/xvla_props_p2_eval/` (parallel workers:
540 rollouts in about 3 h instead of ~12 h serial). All 120 reserved layouts were expert-feasible. Screen picked
101,520 (19/20) and 126,900 (18/20); the report tier chose **101,520: 90/120 = 75.0%** [65.0, 84.2] over 83/120 (not
distinguishable, p = 0.21); the same P1 checkpoint scored **0/120** on those trials (14/20 on its own task in the
no-distractor control), confirm 16/20. Criteria: overall pass, beats-P1 pass, **wrong-object 5.8% against 5% fail,
held-out gap 30 points against 10 fail**; no wrong-bin outcome in 120. The weak spot is `milk→blue` (7/20), a held-out
cell on an object that is also the weakest when trained, so pair and object are confounded. Eight unpinned workers
thrashed the CPU (load 142) and one died of GPU out of memory mid-screen; the first was stopped and the second
absorbed by the driver's resume (FAILURES #57). Report: [PROPS_P2_EVALUATION_20261016.md](PROPS_P2_EVALUATION_20261016.md).

## Props P2 training package: every gate passed, 39% faster per update - 2026-10-13

`experiments/xvla_props_p2/` (branch `feat/props-p2`): P1's recipe unchanged on the P2 seal. Gates in
the admission image: `--check` passed (fingerprint recomputed); `--preflight` passed (exact FP32 reload,
snapshot 10.6 GB in 15 s); `--resume-proof` byte-equal; `--equivalence` byte-equal metrics, weights and
Adam moments over 30 updates between P1's loop and the new one (fused finiteness checks, background
decoding) — **0.524 s/update against 0.858 s** on the same GPU, projecting ~18.5 h for 126,900 updates.
The new loop has not yet run a full session; the night reports record the real pace. Training is split
into two nights by `--stop-at`, which pauses at a snapshot; a thermal guard sleeps at 83 °C. The package
README was first committed empty and every test passed; caught before `--check` (FAILURES #56).

## Props P2 dataset sealed: 540 episodes in 54 minutes - 2026-10-13

Instruction sorting with 1–2 distractor objects (`experiments/props_p2_data/`, branch `feat/props-p2`):
540/540 admitted, 0 rejected, 0 substitutions, 0 distractors touched; 480 train / 60 validation, 270
one-distractor / 270 two-distractor; held-out pairs `milk→blue`, `bread→orange` never collected.
Fingerprint `b1d642b3…f4d898`, re-verified in a second process. Collected by 12 parallel CPU-rendered
shards (user decision; GPU rendering deferred, see TODO) and merged with every episode verified
against its shard: 54 min against P1's 9 h 38 m for the same budget. The first expert re-test failed
(261/288, hand housing clipping tall neighbours); a symmetric hand-clearance rule passed it
(284/288 = 98.6%) without changing the bar. Report:
[PROPS_P2_COLLECTION_20261013.md](PROPS_P2_COLLECTION_20261013.md); lessons FAILURES #54–55.

## Repository visibility, archive repair, and RTX disk reclaim - 2026-10-09

All seven dataset repositories and all eight published model repositories were switched from
private to public (owner action, verified via the Hub API). B2's documented 1,065-file model-repo
gap was closed: a repair pass run directly from RTX via the Hub API added the 1,565 missing
evaluation/training files a fresh local-vs-HF diff found, matching the audit's figure. The same
pass added `runtime/` (base model + tokenizer, X-VLA or SmolVLA as appropriate) and `code/`
(repository snapshot) portability sections — the gap the props P1 bundle closed — to B1,
broad-workspace, B3 and local-varied's SmolVLA model repositories; B3's and local-varied's
evaluation evidence were already complete, B1 and broad-workspace had smaller missing-evidence
gaps also closed in the same pass. None of these five repairs re-ran a from-scratch clean-container
re-evaluation (unlike props P1); see each repository's `PORTABILITY_REPAIR_20261009.md`.

Separately, ~111 GB of RTX disk was reclaimed: finished-run preflight/resumeproof gate directories
and resume snapshots for props P1 and B3 (neither referenced by their published bundles), the
raw per-attempt collection audit trail for local-varied and fixed-layout (both HF-published,
payload untouched), the earliest 2026-09-15 SmolVLA validation directory, and 138 stopped Docker
containers. See [HF artifact map](HUGGING_FACE_ARTIFACTS.md) for updated access/revision pins.

## HF publication and dataset-link update - 2026-10-09

Props P1 is now published from the verified Windows bundle: [dataset](https://huggingface.co/datasets/VivekSai07/franka-panda-props-p1-540) and
[model (root 126,900; alternate 88,830)](https://huggingface.co/VivekSai07/xvla-franka-props-p1-126900) (both switched to public later on 2026-10-09; see the entry above).
The model archive includes the runtime/tokenizer, code snapshot, evaluation evidence and
bundle reconstruction/verifier helpers. Publication verification did not repeat GPU rollouts.

B3 publication completed 2026-10-06: [dataset](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b3-robosuite-560) and [model](https://huggingface.co/VivekSai07/xvla-franka-b3-robosuite-124800).
The original B3 portability gaps were closed later the same day; see the entry above. B2's 153
missing dataset metadata files were repaired on 2026-10-09; its model evidence archive gap was
also closed later the same day (see above).

[Current HF artifact map](HUGGING_FACE_ARTIFACTS.md) links all published stages to their
historical reports, including the fixed-layout and local-varied repositories that now exist
after the earlier upload blockers. The entries below retain the history at their original dates.


## Props P1 trained and evaluated: confirm 100% (20/20), report 75% (60/80) — 2026-10-09

`experiments/xvla_props_p1/` trained X-VLA for 126,900 updates (5.0 epochs) on the sealed P1
dataset; `experiments/xvla_props_p1_eval/` screened all 6 non-trivial checkpoints (20 layouts
each), reported all 6 on 80 trials each (a user-directed, documented deviation from the original
tie-rule-only plan — see the eval report), and confirmed the winner once on 20 untouched layouts.
**Checkpoint 126,900 won the pre-committed tie-break (exactly tied with 88,830 at 75% on report;
rule favoured the later checkpoint) and scored 100% (20/20) on confirm.** Zero `wrong_bin` outcomes
across all 480 report trials on every checkpoint — the policy never confuses the two bin colours;
every failure is a motor/timing problem. Held-out pairs (`milk→blue`, `bread→orange`) perform at or
above trained pairs on every checkpoint. One classifier bug found by video inspection and fixed
before these numbers were final (a correct-bin rest was mislabelled `dropped_outside`); the 900-tick
rollout cap was evidence-checked and left unchanged (recorded as a limitation, not retroactively
loosened). Reports: [PROPS_P1_TRAINING_20261009.md](PROPS_P1_TRAINING_20261009.md),
[PROPS_P1_EVALUATION_20261009.md](PROPS_P1_EVALUATION_20261009.md).

## Props P1 dataset collected and sealed: 540/540 episodes — 2026-10-07

Not a policy result. `experiments/props_p1_data/` (branch `feat/props-p0`, robosuite image, now hash-bound/sealed)
collected the instruction-sorting dataset: one object, two bins swapping sides,
`put the <object> in the <colour> bin`. **540/540 episodes admitted** (480 train, 60 validation), 7 of 547
attempts rejected (all lemon, the P0 success gate), 2 start-pose substitutions, seal fingerprint verified
independently in a second process. `milk→blue` and `bread→orange` held out of training entirely (0 collected).
True wall-clock 9h38m (the plan's "~6 h" estimate only timed the scripted attempt, not the per-episode video
encode and commit). Report: [PROPS_P1_COLLECTION_20261007.md](PROPS_P1_COLLECTION_20261007.md). Trained and
evaluated 2026-10-09 (above).

## Props P0 re-run against the amended spec: GO, 283/288 = 98.3% — 2026-10-05

Not a policy result. The first run (below) failed because of two errors in the spec; both were amended
(success by the object's lowest point below the wall top; start region x ≥ 0.27, no near-base stratum, 288 attempts)
and the gate re-run on the same expert plus the square-footprint fix. **283/288 (98.3%)** against 95%: can, bread, cereal
48/48, milk 47/48, bottle 46/48, lemon 46/48; target slots 143/144 and 140/144; replay exact on all 288; determinism 12/12
in the gate's recheck and 12/12 on the other bin arrangement. Four of the five failures start at radius 0.278–0.283 m, inside
the reach band the region still admits; beyond 0.285 m, 274/275. Report: [PROPS_P0_RERUN_20261005.md](PROPS_P0_RERUN_20261005.md).
Next: P1 training and evaluation design ([TODO.md](../TODO.md)).

## Props P0 (two-bin scene and scripted expert): no-go, 164/300 = 54.7% — 2026-10-05 (first run, superseded by the re-run above)

Not a policy result. `experiments/props_scene/` (branch `feat/props-p0`, robosuite image, no contract, not merged)
builds the two-bin scene with six robosuite objects and a scripted expert that sorts any of them into either bin.
300 scripted attempts against the 95% gate: **164 (54.7%)**; can 45/50, bread 45, lemon 39, milk 20, bottle 14,
cereal 1; target slots 85/150 and 79/150; every replay exact; 12 of 12 re-executions byte-equal. Two causes the
plan's tuning list cannot reach: the written success definition (centre below the 7 cm wall) cannot be met by a
tall object standing in the bin (77 of 136 failures), and the start region within 0.285 m of the base is not
reached along the expert's straight-line approach (58). Outside the verdict, 241/300 end resting in the bin and 198 of
the 199 attempts starting 0.285 m or more from the base do. No tuning change was adopted; robosuite's OSC scored
148/300 on the same attempts. Report: [PROPS_P0_20261005.md](PROPS_P0_20261005.md); the two causes were then
resolved by amending the spec (above).

## Cube programme closed; B3 reviewed, merged and staged for publication — 2026-10-05

An independent check of the B3 evaluation reproduced every headline figure from the raw results and
added two points to the report (dated addendum in
[B3_XVLA_EVALUATION_20261005.md](B3_XVLA_EVALUATION_20261005.md)):

- **Headline lowered to about 85% overall, about 75% with distractors.** The report both chose and
  measured step 124,800 (141/160 = 88.1%); the independent screen gives 13/20 on its one condition
  (trained pairs, distractors) against 31/40 in the report.
- **The persistent failures sit in B2's weak corner.** The three layouts that fail at every
  checkpoint all put the pick on the +y side near the robot base, where B2's `evaluation_054` failed.

`feat/b3-robosuite-cubes` (which contains the training-resume work of `feat/b3-resume`) was merged
into `main`. Steps 124,800 and 156,000 are staged at `outputs/downloads_20261005/b3_robosuite/` for
the Windows copy and a Hugging Face model repo; nothing was deleted. B3 was the last cube experiment
(user, 2026-10-02); next is the props project ([TODO.md](../TODO.md)).

## B3 X-VLA trained and evaluated in the robosuite scene — 2026-10-03 to 2026-10-05

`xvla_b3_robosuite` v2 trained 156,000 updates (5.0 epochs, 132,298 s) on the sealed 560-episode dataset
after v1 died at update 227 (FAILURES #42) and could not resume (#43); six checkpoints verify. The
expert solves all 100 banked evaluation scenes. Screen (20 layouts × distractor, one seed): 13 / 12 / 13 /
13 of 20 at steps 62,400 / 93,600 / 124,800 / 156,000, a tie, so all four were carried. Report (160 trials
each): **110 / 116 / 141 / 105 of 160 = 68.8 / 72.5 / 88.1 / 65.6%.** Held-out pairs within 10 points of
trained with wrong-cube rate under 5% at every checkpoint; the final checkpoint is the worst; 124,800
beats the others by 16–23 points. Failures: near-miss placements that topple and stalls above the target
dominate; the stall is a side effect (FAILURES #44), the pre-fixed cause labels hide several mechanisms
(#45). Not physics-comparable with MuJoCo 3.13.0 results. See [B3_XVLA_EVALUATION_20261005.md](B3_XVLA_EVALUATION_20261005.md).

## B3 cube dataset collected and sealed in the robosuite scene — 560/560 episodes — 2026-10-03

Not a policy result. `experiments/b3_robosuite_data/` collected the B3 dataset in the robosuite
scene: 500 training + 60 validation scripted episodes, 20 cells (10 trained colour pairs × 2 sides) at
25 + 3 each, training distractors none / one / two = 100 / 200 / 200, pick position interior / inner
corner / outer corner = 400 / 50 / 50, the two held-out pairs (`yellow→red`, `green→blue`) never
collected, 60 evaluation layouts banked and never executed. Distractor clearance 8 cm (frozen after
a 23/23 pilot). 7 rejected attempts of a 150 budget, 0 substituted layouts; final stack offset
0.19 / 6.9 / 18.9 mm (min / median / max, window 20 mm). Sealed at
`outputs/b3_robosuite_data/b3_robosuite_20261003/`, fingerprint `e14af864…`, 2.9 GB.

The run crashed once at episode 541 (the `/tmp` tmpfs filled by a per-episode dataset reopen,
FAILURES #41) and was resumed with no data lost and no code change. The seal binds
`experiments/robosuite_scene/*.py`, so that package is now frozen as well.
See [B3_ROBOSUITE_COLLECTION_20261003.md](B3_ROBOSUITE_COLLECTION_20261003.md).

**Next (done):** training and evaluation ran 2026-10-03 to 2026-10-05; see the section above.

## Sim2sim: the B2 policy on the robosuite scene — 3/20 against 19/20 — 2026-10-02

The MuJoCo-trained B2 checkpoint (step 50,000) was rolled out on 20 of B2's own control-condition
report trials (10 layouts × 2 seeds, B2's seeds), once on the MuJoCo scene and once on the
robosuite scene, with one loop, in one image (MuJoCo 3.4.0). **MuJoCo scene 19/20, robosuite scene
3/20**; paired, 16 trials succeed only on MuJoCo and none only on robosuite (exact McNemar
p = 3 × 10⁻⁵). The control reproduces B2's own report on these trials (19/20 there too, 18 agreeing
trial for trial), so the drop is the scene, not the harness or the MuJoCo version.

Of the 17 robosuite failures, 11 grasp and lift the cube and then never carry it to the target or
release it, and 6 never lift it. Which scene difference causes this is not isolated; "the white
table gives the wrist camera nothing to navigate by" is a hypothesis. Counts from ten layouts, not
rates; the package has no contract. See
[ROBOSUITE_SIM2SIM_20261002.md](ROBOSUITE_SIM2SIM_20261002.md).

**Consequence:** a policy trained on the MuJoCo cube scene cannot be reused on a robosuite scene
as it is. B3 and everything after it need their own data collected in robosuite.

## Robosuite scene bring-up (B3 Phase 1): go — 2026-10-02 (the one fix it rests on was ratified 2026-10-02)

Not a policy result. A scene composed by robosuite 1.5.2 (`experiments/robosuite_scene/`) is now
driven by the `pose_snapshot_v2` scripted expert and recorded in the existing 9-state / 8-action
LeRobot schema. Gates R1–R7 all pass (8 tests in the admission image, 16 in the robosuite image).
Five scripted red-on-green episodes, 5/5 successful, 528 frames each, are sealed at
`outputs/robosuite_scene/robosuite_bringup_20261002/` (fingerprint `543e300c257d…`); the dataset
loads and both AV1 videos decode (2640 frames each).

The robosuite path runs in a **second image** on MuJoCo 3.4.0, because robosuite 1.5.2 cannot reset
an environment on 3.13.0 (FAILURES #39); the sealed pipeline stays on 3.13.0. Robosuite-scene
numbers are therefore not physics-comparable with any sealed result.

The go is conditional: all six attempts the tuning protocol allows failed, and the pass came from a
model-contract correction outside it (fingertip pads re-parented into the finger bodies), allowed
by the executing AI controller's ruling as physically equivalent. The user ratified it on 2026-10-02.

Every seed lands 8.6–12.7 mm off-centre in xy (20 mm window). That is the scripted expert's own
±5 mm (1σ) target jitter, not the robosuite scene: the offset predicted from the seed's draws
matches the measured signed offset to 0.65 mm in all five seeds, and its direction changes with
the seed. A ≤ 12 mm expert gate would reject about 15% of episodes with this jitter (calculated),
so Phase 2 must choose between the jitter and the gate.

A second collection reproduced every file except `meta/stats.json` (key order), so the dataset is
not byte-reproducible and its fingerprint differs between runs (`08c807bf0095…` the second time).

Open: the B2-checkpoint smoke test was not run; finger pad friction and geometry are robosuite's. The first
"grasp failed" verdict was a blind contact check, not a failed grasp (FAILURES #40). See
[ROBOSUITE_SCENE_BRINGUP_20261002.md](ROBOSUITE_SCENE_BRINGUP_20261002.md).

**Next:** sim2sim validation of the B2 checkpoint on the robosuite scene (branch
`feat/robosuite-sim2sim`), then decide jitter versus gate and re-think the B3 axis list before
designing Phase 2. The placement offset needs no diagnosis: it is the expert's own jitter.
*(Outcome: sim2sim done 2026-10-02; the jitter-vs-gate question was dropped with the tighter gate;
B3 designed 2026-10-03 and evaluated 2026-10-05 — entries above.)*

## B2 distractors: no detectable distractor effect; 50,000 updates 91.9% vs 82.5% at 33,400 — 2026-09-30

Trained X-VLA from base on `b2_distractor_20260928` (200 episodes, 40 control + 160 with 1–2
distractor cubes) for 50,000 updates (11.7 h). Screened five checkpoints on 20 layouts
(distractor condition): 8 / 16 / 17 / 16 / 17 of 20 at 16.7k / 25k / 33.4k / 41.7k / 50k.
33,400 and 50,000 tied; the chain reported 33,400 by array-order tie-break, and a queued follow-up
reported 50,000 as well. Report (160 trials, 40 held-out layouts, both conditions, 2 seeds):
**33,400 132/160 = 82.5%, 50,000 147/160 = 91.9%** (paired McNemar p = 0.004).

Control vs distractor within layout: +2.5 pts (p = 0.82) and +1.25 pts (p = 1.0). Not detectable;
no distractor was ever touched, so this is a visual-distraction result only. Diagnosis of all 41
failures: layout-specific, none infeasible (scripted expert 23/23); half are release precision near
the 2.1 cm tipping limit of the cube, 27% are acquisition failures in two layouts, 17% are
hesitation. Collection hit four problems of its own (FAILURES #32–#35) and the evaluation chain
two more (#36, #37). See [B2_DISTRACTOR_EVALUATION_20260930.md](B2_DISTRACTOR_EVALUATION_20260930.md).

## B1 language grounding: 94.4% report success, no positional shortcut detected — 2026-09-26

Trained X-VLA from base on `b1_language_20260925` (200 episodes, 2×2 factorial: instruction ×
colour/region arrangement, designed to break the #27 colour-position confound). Screened three
checkpoints on cell C1 only (40 disjoint layouts each): 16,700 updates 22/40 (55.0%), 25,000
34/40 (85.0%), 33,400 **40/40 (100%)** — monotonic and pairwise significant (Fisher p ≤ 0.026 at
every step). Matched exposure (16,700, same budget as the 86.1% broad-workspace run) was clearly
undertrained here, as expected: B1 splits the same 160 training episodes across two instructions.

Reported the winner (33,400 updates, 3.35 epochs) on the untouched other 40 layouts, all four
cells, 160 trials: **151/160 = 94.4%** [89.7, 97.0]. The load-bearing contrast is
familiar-vs-swapped arrangement — 93.8% vs 95.0%, Fisher p = 1.0, not worse under swap. A policy
still using the old positional shortcut would fail specifically on swapped layouts, since swapping
relocates the correct-coloured cube to the region the shortcut ignores. It didn't. Pick-colour
contrast (red 96.2% vs green 92.5%, p = 0.495) is likewise not significant. **First evidence in
this repo that the policy conditions on the instruction rather than cube position.**

Four fork issues surfaced and were fixed along the way, each with an equivalence test proving the
fork matches the sealed original wherever the original applies (FAILURES #28–#31): the shared
validator's five single-task assumptions, its hardcoded red-on-green cube binding, one infeasible
evaluation layout requiring a recorded substitution, and two resource guards inherited from the
fork source that didn't fit B1's schedule. See [RESULTS.md](RESULTS.md) for the full table.

**Next:** none planned yet — B1's own follow-up (height/vertical grounding, deferred per TODO
"B1b") is unscheduled. Read [RESULTS.md](RESULTS.md) and [FAILURES.md](FAILURES.md) before
proposing the next experiment.
*(Outcome: B2 distractors followed, 2026-09-28; B1b was never scheduled and was closed with the cube
programme on 2026-10-05.)*

## Matched exposure: 68.1% → 86.1%, and the failure mode identified — 2026-09-24

Retrained X-VLA on the identical sealed `broad_workspace` data at **1.67 epochs** (16,700 updates)
instead of 0.50 (5,000), with everything else held constant. On the same 36-layout × 2-seed
holdout: **62/72 = 86.1%** [76.3, 92.3], versus 49/72 = 68.1% [56.6, 77.7]. Fisher exact
**p = 0.0165**. `fixed_parity` unchanged at 20/20. The 65/72 gate still fails, by three trials, and
was again not lowered.

So the earlier 68.1% was undertraining, not a ceiling. Across the whole programme updates were
pinned at 5,000 while datasets grew, making exposure fall as data rose — 8.91 epochs at 9 episodes
down to 0.50 at 160. Data volume and exposure had been perfectly anti-correlated, and no run had
ever isolated either.

**The checkpoint curve is non-monotonic and offline metrics are actively misleading.** Success went
25.0% (4,000) → **0.0%** (8,000) → 86.1% (16,700) while train/val arm MAE improved *monotonically*
throughout (0.03390 → 0.02621 → 0.01022). At 8,000 the policy still grasped 72/72 but lifted only
17/72, leaving the cube a median 63 cm from target. Loss, MAE and gripper accuracy all showed
steady improvement through a total capability collapse. Rollouts are the only valid measure.

**All 10 remaining failures are one failure.** The cube is grasped, lifted ~29 cm, carried up to
0.80 m across the workspace, and then set down roughly one cube width from the target — beside it
rather than on it. Decomposed about the robot base the offset is **tangential** (a ~5° bearing
error), not a reach error. The "unreachable far layouts" explanation is refuted and inverted:
failing layouts average 0.424 m reach, fully-solved ones 0.472 m.

Authoritative magnitudes come from the 128-trial T2a measurement, not from these 10 trials:
successes +0.150 cm (sd 0.715, n=96), failures +3.893 cm (sd 0.723, n=32). The originally reported
"4.51 cm ± 0.20 cm" mixed a between-layout spread with a within-layout one, and the claim that
failures land toward the robot base did not survive the larger sample.

**The expert is not the cause.** All 200 sealed demonstrations were checked: mean final placement
error 0.679 cm, max 1.981 cm, zero episodes above 2 cm, mean signed offset 0.1 mm, correct stack
height in 200/200. The offset is introduced by the policy, so collecting more of the same data
should not be expected to fix it.

Pending: the update-12,000 checkpoint, which is the genuinely LR-matched comparison against the
short run's update-4,000 (2.74e-5 vs 2.05e-5) and will separate exposure from annealing.

**Resolved same day — it is exposure, not annealing.** Checkpoint 12,000 scored **51/72 = 70.8%**.
Against the short run's update-4,000 at a comparable learning rate (2.74e-5 vs 2.05e-5, 72% vs 80%
annealed) but a third of the exposure, that is 70.8% vs 20.8%, **p = 2.2e-9**. Conversely, at
matched *epochs* (0.40) a 4.4× difference in learning rate changes nothing (25.0% vs 20.8%,
p = 0.69). Exposure drives capability; the anneal does not.

The full curve is 25.0% → 0.0% → 70.8% → 86.1%. The mid-training collapse is **reproducible**: the
short run also passed through 0/72 at update 3,000, 60% of its schedule. Two independent runs both
hit exactly zero near mid-schedule and both recovered, so mid-training checkpoints cannot be used
as a proxy for the endpoint in either direction.

## Broad-workspace trained and evaluated — 2026-09-22

`broad_workspace_data` sealed at 200/200 episodes (105,600 frames, 160 train / 40 validation,
400 replays audited, 0 duplicates, seal `0351d3da…`). Both policies retrained on it at identical
hyperparameters and evaluated on the 36 reserved F7 holdout layouts (×2 frozen policy seeds).

X-VLA **49/72 = 68.1%** broad_holdout, 20/20 fixed_parity. SmolVLA **23/72 = 31.9%** broad_holdout.
Both fail the 65/72 gate; the gate was set for the easier local-varied protocol and was NOT lowered
to match. All failures are timeouts. X-VLA leads by 36.2 pp — a much wider margin than the 9.7 pp
seen on the narrow local-varied protocol.

The 8 layouts X-VLA fails on both seeds were screened with the scripted expert and all 8 are
solvable, so the holdout is sound and 68.1% is a genuine policy limit. Likely cause is coverage
density, not volume: 160 episodes spread across a wide 4-D envelope is sparser than 48 episodes
inside a ±1 cm neighbourhood. fixed_parity staying at 100% supports this — the policy is perfect on
geometry it knows and degrades as layouts leave training density.

**Not yet answerable:** whether the 200-episode collection improved generalisation at all. A
cross-evaluation of the local-varied X-VLA on the identical holdout is built (`experiments/
xvla_localvaried_broad_eval`) and queued. Note it conflates breadth with episode count (48 vs 160),
so it gives the direction and rough magnitude, not a controlled ablation.

**Cross-evaluation answered (2026-09-22).** The local-varied checkpoints were run on the identical
72-trial holdout: X-VLA **2/72 = 2.8%** [0.8, 9.6], SmolVLA **2/72 = 2.8%** [0.8, 9.6]. Against
68.1% [56.6, 77.7] and 31.9% [22.3, 43.4] for the broad-trained models, the intervals do not
overlap — the 200-episode collection is what creates wide-workspace capability (+65.3 pp X-VLA,
+29.2 pp SmolVLA). Both local-varied models solved exactly one layout, `evaluation_031`, on both
seeds; it is the closest of all 80 holdout layouts to their training base (0.0297 vs 0.2209 max),
so the 2.8% is the one holdout that falls inside their ±1 cm training neighbourhood rather than
noise. Caveats: breadth is confounded with volume (48 vs 160 episodes), and the SmolVLA cross-eval
uses a data-level binding because its metadata binding predates the re-seal.

**68.1% is probably not a ceiling.** X-VLA saw 0.50 epochs on the broad dataset versus 1.67 on
local-varied — updates stayed at 5000 while the data tripled. Next step is measuring the checkpoint
curve before training longer; see TODO Phase A.

## Local-varied Hugging Face publication blocked — 2026-09-21

User requested dataset publication. Authenticated `VivekSai07` with a write
token; proposed private dataset `franka-panda-cube-stacking-local-varied-60`
does not exist. No remote mutations/upload. Storage-origin probe still fails
HTTP 407 through the configured proxy. Current full-file seal verifies, but
strict provenance fails on two shared collector files changed September 20;
matching historical versions are recoverable from Git. Current seal also
differs from the September 19 completion report and requires reconciliation.
See [publication blockers and recovery](LOCAL_VARIED_DATASET_PUBLICATION_20260921.md).
Dataset/source/secrets preserved; no new experiment or validation replay.

## Local varied dataset complete — 2026-09-19

60/60expert demonstrations admitted,120/120saved-control replaysPASS; collector
exit0/noOOM12:03:39Z (~2h56m).48train/12validation,31,680rows,63,360decoded
camera frames,noexactduplicates. All77frozenstates verified and24reserved eval
scenes unexecuted;3policyseeds preserved. Freshfullseal/source/bank/journal/slot
auditPASS. See [completion report](LOCAL_VARIED_COLLECTION_20260919.md).
No training/evaluation automaticallystarted;54GiBfree. Older runningnotes historical.

## Local varied60 collection launched — 2026-09-19

Exactspecapproved;85workerCPUtests andmain22newtests+77nonstepinitializationsPASS.
Collectorfranka-local-varied-60-20260919 ID3eca3bf4aedc started09:07:12Z, confirmed
running pinnedCPUofflineimage.48train(40varied8center)/12val,24reserved evalscenes,
first12gatewithin60,120replaysmax,40GiBstart30floor6hcapfirstfailurestop.
See [launch](LOCAL_VARIED_COLLECTION_20260919.md). No training or policybankruns.
No duplicate/retry/resume/boundsourceedits. Startup only, notcompletion.

## Local varied-data direction approved — 2026-09-19

User approved proceeding toward local varied data, not broad workspace randomization.
Exact [data-stage design](superpowers/specs/2026-09-19-local-varied-data-design.md)
proposed for review:48train(40varied+8center),12validation,24fresh reserved evaluation
scenes, independent cubeXY offsets within10mm/axis; first12preparation counted
within60expert attempts,120controlreplaysmax, firstfailurestop. No newcode/jobs/data
for this stage yet. Approximately59GiBfree. No automatictraining/collectionlaunch
before exactbudget review. Portablemodeldownloadfolders reverified; no Gitbinaries.

## Local position screen completed — 2026-09-19

68/68trials complete exit0/noOOM00:24:37Z (~1h51m);4controltracesbyteexact.
Shiftedscenes only:SmolVLA19/32(10placement+3grasp failures),X-VLA31/32(1placement).
5mm:9/16vs16/16;10mm:10/16vs15/16. Two development-selectedseeds/condition,
oneknownSmolfailingseed: descriptive sensitivity, not90%→59%populationclaim or
clean distanceboundary. Fullsource/bank/result/traceauditPASS;1315telemetrysamples,
0queryerrors, sharedGPUmetrics notperrunattribution. See
[complete results and proposed local-data next step](LOCAL_POSITION_PROBE_20260919.md).
No newcollection/training authorized by completion. Older runningnotes historical.

## Bounded position probe launched — 2026-09-19

User approved exact68trial frozenmodel evaluation.35CPUtests+real17resetnonstep
preflightPASS. Containerfranka-local-position-probe-68-20260919 ID996f301e2e04
started2026-09-18T22:34:00Z. See [study launch](LOCAL_POSITION_PROBE_20260919.md).
SmolthenXVLA controlsbyteexact firstpermodel,5/10mm pickORplaceXYoffsets,2seeds.
No training/collection/retry/extension; preserveboundsources; no outcomeclaimyet.
UsercancelledcheckpointGittransfer and will downloadportablepackages onWindows.

## Baseline archive/publication update — 2026-09-18

Both historicalbaselines imported toPRIVATE W&Bproject under user-approvedentity
vivekchaitanya07-university-of-stuttgart:5000updates+20trials each, systemstatsOFF
(no fabricatedpastGPUutil). PortableCPUverifiedreleasecopies ready. HFprivateSmol
repo created but weights blockedHTTP407LFSproxy; no modelcommit, XVLAnotattempted.
See [publication report](BASELINE_PUBLICATION_20260918.md).72publicationtestsPASS.
Next [smallpositionstudy](superpowers/specs/2026-09-18-local-position-probe-design.md)
is proposed68trials, notlaunched; no training/newcollection. FutureliveW&Bhookpending.

## Both fixed-layout capability gates passed — 2026-09-18

SmolVLA5000updates+20trials complete exit0/noOOM19:32:07Z (~1h51m):18/20stacks,
>=18gatePASS. All20grasp/lift; trials6/16releasedwithoutstable stack,40stimeout.
0armclipping; successfulXY1.08–15.09mm; peak5.58GB. X-VLA remains20/20,
XY0.63–4.68mm,peak~21.3GB. Same20train/4samegeometryval and20seedresetprotocol,
but different nativechunks/objectives/batches/trainableparameters: notarchitecture
onlycomparison or spatialgeneralization. Freshcompletionbindings/metrics/traces
verified. See [SmolVLA comparison](SMOLVLA_FIXED_LAYOUT_20260918.md).
No furthertraining/positionvariation/collection authorized; oldrunningnotes historical.

## SmolVLA fixed-layout comparison launched — 2026-09-18

User approved comparison afterX-VLA20/20. CPUsemanticadmission+GPU2update scratch
PASS (bothreloaderrors0,FP32Adam exact),88CPUtests. Fullfreshbase5000then20same-reset
seeds launched17:41:15Z,containersmolvla-fixed-layout-5000-20260918 IDb129f9a53d89.
Read [SmolVLA launch](SMOLVLA_FIXED_LAYOUT_20260918.md). No boundcode/spec/plan edits,
duplicate/resume/extension, newpositions or automaticjobs. No resultyet.

## X-VLA fixed-layout capability passed — 2026-09-18

Full5000updates+20trials complete, exit0/noOOM16:56:38Z (~1h43m).
**20/20correct stable stacks**, above18/20gate; samefixedreset20distinct policy
seeds.0armclipping, finalXY0.628–4.682mm. Freshcompletion audit verifies all
training/checkpoint/data/source bindings and20savedtrace stages/reset/streaks.
See [completed result](XVLA_FIXED_LAYOUT_20260918.md). Not spatialgeneralization.
No automaticSmolVLA, furthertraining/evaluation/collection/extension launched.
Older running notes below superseded; preserve frozen source/spec/artifacts.

## X-VLA fixed-layout training launched — 2026-09-18

User approved freshbase5k then20same-reset trials,18/20capabilitygate. CPUdata
admission andGPU2update scratch passed:FP32weights+Adam exactreload,prediction0,
10560trainrows/9980starts,20train-onlystats. Fullcontainer
xvla-fixed-layout-5000-20260918 ID016ab837e837 started15:14:07Z, verifiedrunning.
See [training launch](XVLA_FIXED_LAYOUT_20260918.md). Do not edit boundsource/spec,
duplicate/resume/extend jobs, or launchSmolVLAautomatically. Old no-trainingnotes
below are historical; collectiondata remains immutable. No policy result yet.

## Fixed-layout collection complete — 2026-09-18

Collector exited0/noOOM at14:33:22Z (~1h02m):24/24expert demonstrations admitted,
48/48control replays pass,12,672rows,25,344decoded camera frames,20train/4same-layout
validation, no exactduplicates. Fresh complete-file seal/source/runtime/reset audit
passes; source/spec unchanged. See [completion report](FIXED_LAYOUT_LAUNCH_20260918.md).
Disk~121GiB free. Dataset readiness is NOT learned-policy success or spatial
generalization. No training/retry/newcollection automatically authorized.
Historical running entries below are superseded; do not relaunch the collector.

## Fixed-layout collector launched — 2026-09-18

Container smolvla-fixed-layout-24-20260918 ID728c7a71c5da started13:31:39Z,
verified running in pinned offline CPU sandbox.122.22GiB free at preflight.
Read [launch report](FIXED_LAYOUT_LAUNCH_20260918.md). Startup is not24/24success;
inspect journal/failure/validation and actual container exit. Do not restart,
duplicate, edit bound sources/spec or start training. Firstfailure stops.

## Fixed-layout collection authorized and software verified — 2026-09-18

User authorized safe cache cleanup and collection. Docker build-cache-only cleanup
increased free space18GiB→123GiB; no experiment artifacts/images/containers/volumes
deleted. All64distinct software tests passed, including synthetic24-slot final
validation/readmission and8corruption checks. See [launch report](FIXED_LAYOUT_LAUNCH_20260918.md).
Launch target smolvla-fixed-layout-24-20260918,24fixed-reset expert attempts,
first3gate within24, two saved-control replays/episode,30GiBfloor,3hcap, firstfailure
stop. No retries/planner restart/automatic training. Old pending notes below are
historical. Do not duplicate a running collector or modify bound source/spec.

## Fixed-layout implementation verification in progress — 2026-09-18

User approved the exact24-execution data-stage spec and subagent-led code/tests,
with collection blocked until30GiB free; no training or deletion. Reset contract
committed106ad6f after20focusedtests. Recorder/validator implemented; review fixes
cover finite video intervals, actual output-filesystem reserve, real LeRobot
save/finalize/continue and ordered encoder-input/media binding. A synthetic24-slot
full admission test is running; do not claim final software or physical gates pass.
See outputs/implementation/fixed_layout_20260918/progress.md for execution ledger.
No real expert/collection/physics replay/model job has been launched. Existing
unrelated container moveitFrancesco must remain untouched. Storage~18GiB free.

## Fixed-layout direction approved — 2026-09-18

User approved replacing spatial expansion with one fixed arrangement first.
The [fixed-layout data-stage design](superpowers/specs/2026-09-18-fixed-layout-baseline-design.md)
uses the exact pilot episode7 reset (expert and both policies previously succeeded),
proposes24 repeated expert executions (20train/4same-geometry validation), with
first-three preparation, unchanged snapshot-v2 expert and no automatic training.
Exact budget/spec review precedes implementation. Only about19 GB free: existing
30 GiB collection reserve blocks launch. No deletion/reserve reduction authorized.
The 12-layout proposal below is superseded, not an execution instruction.

## Targeted dataset preparation proposed — 2026-09-18

User approved the dataset-first direction after the coverage review. The exact
first-stage [preparation design](superpowers/specs/2026-09-18-targeted-dataset-preparation-design.md)
is now written for review: 12 fixed red-on-green demonstrations, separate closure,
contact/lift and strict placement gates, plus 24 independently frozen future
evaluation initializations. No new collection/training has started; exact design
review precedes implementation. No automatic planner restart or budget expansion.
Historical CPU reports are now [archived](CPU_REPORT_ARCHIVE_20260918.md) unchanged;
78 focused CPU tests freshly passed. Machine-local setup remains ignored on disk.

## Spatial audit and public references — 2026-09-18

User authorized saved-artifact spatial analysis and publicHFmodel/datasetresearch.
Exact9train/4heldout audit:3pickpositions outsidehull,300insidecombinedhull;
X-VLA300–302neverclose,303onlyclosesonetickat919despiteearlier10.07mmhandgoal
approach; seen8alsoapproaches7.9mmwithoutclosing. Sparsecoverage alone insufficient.
181oldfilesunchanged,26caseschecked,noGPU/policy/physicsjobs. Ethan50demo bowlplate
pair hasno verifiedperformance;6otherreferences researched withpinnedcitations.
See [coverage and reference synthesis](DATA_COVERAGE_AND_PUBLIC_REFERENCES_20260918.md).
No new collection/training authorized. Targetedcoverage plusclosure/placement
supervision or nativebenchmarkpositivecontrol are proposals, not launched jobs.

## X-VLA complete; capability gate failed — 2026-09-18

Fresh5k+13rollouts finished exit0/noOOM (~1h46min). Seen4/9correct stacks,
held-out0/4; provisional8/9+3/4gate failed. Eightseen trials contacted/lifted
target, fourstacked; allheldoutlackbilateraltargetcontact. Freshdata/source/cache/
checkpoint/stats/inventorychecks passed, all5kmetricsfinite, recordedreloaderror0.
See [final X-VLA comparison](XVLA_TRIAL_20260917.md#completed-result--2026-09-18).
Better observed seen performance thanSmolVLA1/9, noheldoutgain; recipeconfounds
prevent architecture-onlyclaim. Job stopped normally; no automaticextensions,
collection, planner work, uploads or final-scene rollout. Awaituserdecision.

## Current X-VLA full run — 2026-09-18

Corrected two-update micro4xaccum2preflight PASSED exactFP32model/Adam/prediction
reload(error0) and data binding. User authorized deleting only failedscratch
weight/optimizer artifacts for space; all reports and other checkpoints preserved.
Freshbase5k+final13rollouts job `xvla-single-task-5000-20260918` launched into
`outputs/train/xvla_single_task_20260918`. See [current launch](XVLA_TRIAL_20260917.md#current-launch--2026-09-18).
No source/admission edits, duplicates, recipe changes, uploads or extensions
while running. Prior blocked messages below are historical. No task result yet.

## Current X-VLA trial — 2026-09-17

User approved bounded X-VLA comparison, then explicit5k micro4xaccum2 fullFP32/
BF16 recipe+13rollouts. Publicpinnedassets downloaded. Initialmicro1two-updateGPU
preflight passed exactweights/Adam/predictionreload. Exactbatch4scratch completed
2updates peak21.33GB, then failed assistant checker on action=None duringreload.
Checkerfixed with3regressions;47CPUtests passed. Failedscratch/evidence preserved,
CPUadmission now intentionally stale; nofulltraining orrollouts launched.
See [X-VLA status and correction](XVLA_TRIAL_20260917.md). Needapprovalforonefresh
corrected scratch underno-retryagreement; do notoverwritefailureroot or bypass
admission. SmolVLA/planner/recollection remainpaused.

## Single-task experiment complete; stop for decision — 2026-09-17

5,000 updates and13rollouts complete, exit0/noOOM. Capability **failed**:
seen1/9, held-out0/4 (required8/9,3/4). Six seen trials lifted requested target;
only one stacked; all four held-out trials lacked target bilateral contact.
Fresh model/data/source/cache/stat/inventory checks pass; program integrity is
not task success. See [completion findings](SINGLE_TASK_20260917.md#completed-capability-gate-failed).
No retries/extensions/newtraining/collection/planner work. The agreed stopping
rule applies; wait for a decision before starting another experiment.

## Current priority: single-task capability — 2026-09-17

User paused planner debugging and approved one simplified existing-data run:
red-on-green/two-cube train0–8, held-out300–303, freshbase5kupdates batch8x4,
FP32trainable/BF16autocast,500warmup1e-4decay1e-5, final-only13rollouts.
219CPUtests andtwo-updateGPUpreflight passed, reloaderror0. Detached job
`smolvla-single-task-5000` launched; see [single-task protocol](SINGLE_TASK_20260917.md).
No recollection, planner changes, extensions or final-bank evaluation. Do not
duplicate job or edit its sources. Historical v3/v4 and paired-learning plans
below are paused, not instructions for further work.

## Adaptive subdivision numerical stop — 2026-09-17

Approved bounded subdivision executed one saved-state numerical route, no physics.
It passed the original 0.069763rad continuity rejection, completed93/113 original
waypoints, then failed lowering IK backtracking at1.707mm position residual
(required<=1mm), joints1/3atinteriorbounds. Limits unchanged, no retry. 188 tests
passed and independent review confirmed narrow numerical claims. See
[v4 result](ADAPTIVE_PLACEMENT_V4_20260917.md). No complete path or dynamic task
success. Next requires route/posture redesign rather than tolerance relaxation;
no new collection/training is authorized by this failed gate.

## Bounded placement v3 stopped — 2026-09-17

Approved v3 implemented in isolated package; 168 CPU tests and independent review
passed. One actual case (episode4) stopped after 197 pick/lift ticks at
waypoint_branch_jump: max joint displacement0.0697633rad > approved0.05rad.
Rejected waypoint not executed, no intentional release, no remaining four cases
or retry. Reference replay qposerror0/fullstatesexact; old hashes unchanged.
Container exit1/noOOM. See [v3 report](BOUNDED_PLACEMENT_V3_20260917.md).
Expert placement still unresolved; new adaptive-subdivision design would need
approval. No collection/training or final-scene rollout follows this failure.

## Pose-copy correction tested — 2026-09-17

User approved a versioned endpoint-copy fix and one same-layout test. New
`experiments/pose_snapshot_v2` preserves old bound sources. 129 existing +8 new
CPU tests passed; exactly one corrected expert invocation on failed episode4
still failed placement (exit1/noOOM): saturation202->169ticks, finalXY80.95->97.77mm,
stable0. Alias defect fixed, task not fixed. No retries/recollection/training.
See [results and external-advice assessment](POSE_FIX_AND_EXTERNAL_ADVICE_20260917.md).
Next proposed bounded change: joint-limit-aware continuous placement planning
and measured release gating; design approval required. Existing paired data stays
failed/not-training-ready. Old source and artifact hashes were preserved.

## Placement investigation — 2026-09-17

Bounded CPU IK found joint-valid solutions for all three exact failed placement
endpoints; no continuous-path feasibility claim. Independently confirmed a
planner bug: FK returns a mutable frame pose and TaskSpaceTrajectory retains
it, so later FK calls move its stored start. Minimal reproduction moved t=0
start 6.13 cm; previous-tick-start reconstruction matches failed trace velocities
exactly. See [investigation](PLACEMENT_REACHABILITY_20260917.md).
No source fix, expert rerun, recollection or training was performed. Recommend
versioned immutable-pose correction and bounded same-layout test first, not
disabling limits or replacing the layout. Correction requires approval.

## Paired collection stopped — 2026-09-17

`smolvla-paired-collect-36` record exited 1 after four successful saved demos;
fifth `train/red-blue/0/forward` (seed 488840035) failed placement. Saved-control
CPU replay qpos error 0: grasp/lift succeeded, joint 4 target lower-bound
saturation persisted 202 ticks, time-driven release occurred off-target;
final XY gap 8.09 cm, no stack. Full validator and training never started.
Preserve all partial artifacts; no retry/resample/resume or loosened criteria.
See [failure diagnosis](PAIRED_LEARNING_PREPARATION_20260917.md#current-stopped-on-expert-placement-failure).
Next proposed step is bounded same-layout expert planning diagnosis, not model
training or automatic recollection. Historical running banners below superseded.

## Current paired collection — 2026-09-17

User approved the exact three-legacy-episode geometry exception. Fresh 129-test
CPU suite passed. Container `smolvla-paired-collect-36` launched into
`outputs/paired_demonstrations/paired_20260917`, automatically running full paired
validation only after successful collection. Startup confirmed; no completed-demo
or training-readiness claim yet. See [live preparation handoff](PAIRED_LEARNING_PREPARATION_20260917.md).
No duplicate jobs, source/spec edits, retries, new training or uploads while this
gate runs. Two-model training/evaluation remain gated subsequent stages.

## Paired stage one prepared — launch held for review, 2026-09-17

User explicitly approved the narrow exception for two legacy v1 episodes and one
v2 episode without initial geometry. Implemented test-first: literal-message
allowlist only, visible **unchecked** gaps, approval reason and actual metadata
file hashes in inventory/manifest/results, fresh validation and readiness-gate
rechecks. Unknown gaps and all non-geometric integrity errors still fail.
Old sources/artifacts remain unchanged; all new train/eval uses new paired data.

**129 CPU Docker tests pass. Actual read-only CLI preflight exits 0:** configuration
passed, collection `not_started`, 1,906 files / 1,164 geometry hashes, both approved
missing messages retained, zero expert trajectories and `training_ready: false`.
See [approved amendment and current preparation evidence](PAIRED_LEARNING_PREPARATION_20260917.md).
**No collection, GPU jobs or commits. Main launch follows review.** The 36-demo
record/validate commands are prepared, not executed; downstream two-model training
and 72-rollout evaluation implementation remain pending.

## Historical paired preparation blocker — superseded by approval above

User approved the paired-demonstration two-model design. Isolated stage-one
collection/validation implementation is prepared; 124 CPU tests reported passing.
Fresh configuration-only preflight passes configuration but blocks collection:
two old v1 episodes and one incomplete v2 episode lack full initial geometry
needed by the approved historical duplicate check. No expert trajectories or
training jobs executed. See [preparation and proposed narrow amendment](PAIRED_LEARNING_PREPARATION_20260917.md).
Do not bypass this gate or claim training readiness. Downstream training/evaluation
implementation remains pending; new stage-one files are uncommitted.

## Paired diagnostic complete — 2026-09-17

48/48 trials, exit 0/no OOM, 12/12 original traces bit-exact, all 48 saved-action
replays and unchanged final fingerprints. Correct stacks: original 2/12,
reversed 1/12, alternate seed 1/12, seen training scenes 1/12. Four original/
reversed scene pairs contacted a cube in both runs and chose the same cube;
eight pairs had no contact in either run. No demonstrated object-choice switch.
Poor performance on seen scenes rules out an exclusively unseen-layout problem.
See [results and next design recommendation](PAIRED_DIAGNOSTIC_20260917.md#verified-completion-and-interpretation).
No job is active from this diagnostic. Proposed small contrastive two-cube
learning intervention needs exact design/budget approval; no bulk collection,
training, upload or final-scene evaluation automatically follows completion.

## Current authorized diagnostic — 2026-09-17

User approved **48 policy rollouts**: 36 same-scene original/reversed/seed-control
tuning comparisons plus 12 two-cube training-scene reproductions. Container
`smolvla-paired-diagnostic-48` launched with unchanged 020000 checkpoint,
40-second timeout and horizon 10. See [protocol and handoff](PAIRED_DIAGNOSTIC_20260917.md).
90 tests passed; original-condition reproduction is a live gate, not yet a
completion claim. No new training, demonstration collection, upload or final-scene
evaluation. Do not change bound sources or duplicate the running diagnostic.

## Full tuning evaluation complete — 2026-09-17

**4/120 correct stacks (3.33%), 0/120 without any clipping.** All four successes
are two-cube scenes (4/40); three/four cubes each 0/40. Container exited 0/no OOM,
post-run fingerprints unchanged. Independent all-120 saved-action replay matched
recorded states exactly, all pairwise 30-tick scoring checks agree, 141,121 video
frames decoded. 110 trials had no bilateral instructed-target contact; first
contact was wrong-object in 17, requested-target in 9, absent in 94. Eight wrong
stable ordered-pair stacks correctly failed. See
[full review and decision tree](FULL_EVALUATION_REVIEW_20260917.md).

Additional inspection found 300 unique training layouts, identical gripper
transition frames 107/347 in all train demos, and original offline probe blind
spots. No new collection/training/policy rollouts started. Recommended next:
bounded paired-instruction comparison, optionally training-scene reproduction,
then one evidence-selected intervention. Exact follow-up budget needs confirmation;
final 240 scenes remain sealed. Historical running notes below are superseded.

## Instruction audit — 2026-09-17

User reported mismatched instruction/video behavior. [Detailed audit](INSTRUCTION_AUDIT_20260917.md)
confirms wrong-pair terminal stacks in trial_002 (green on red), trial_008
(green on yellow), trial_035 (yellow on red); all correctly scored failures.
At the 41-completed snapshot only trial_022 succeeds, correctly red on yellow.
No label/color/normalizer/scoring mismatch demonstrated; all 221,760 row tasks
match, saved stats exact, three full saved-action replays exact. Controlled
fixed-observation instruction swaps suggest weak immediate language sensitivity,
not a proven sole cause. No training/scoring/source changes or new policy rollouts.
Authorized 120-scene baseline continues unchanged; final bank remains sealed.

## Current status and authorization — 2026-09-16

**CURRENT AUTHORIZED JOB:** user approved rollouts. Selected 020000 checkpoint
is running on exactly 120 tuning scenes in `smolvla-pilot-eval-tuning-120`,
40-second timeout/horizon 10; final 240 scenes are not evaluated. Model loaded,
first trial started; launch is not a success score. See
[evaluation handoff](PILOT_EVALUATION_20260916.md). No training/upload; preserve
bound sources and do not duplicate this job. The pending-approval note below
is historical and superseded by this authorization.

**TRAINING COMPLETE:** `smolvla-pilot-train-20000` exited 0/no OOM at
21:16:00 UTC. Verified 20,000 consecutive finite updates, all 11 checkpoint
model hashes and fresh dataset/runtime/training-source bindings. Selected
checkpoint **020000** by predefined full-chunk held-out arm MAE: **0.016913 rad**;
first-ten **0.008329 rad**, gripper accuracy **93.60%**. No closed-loop success
claim. See [completion report](SMOLVLA_PILOT_TRAINING_20260916.md#verified-completion).
No active training remains from this job. Next proposed task: selected-checkpoint
evaluation on 120 independent tuning scenes, 40 seconds/horizon 10; approval
required. Final 240 scenes remain untouched. No restart, collection or upload.

### Historical training launch — completed

**CURRENT AUTHORIZATION / LAUNCH:** user deferred Hub upload and explicitly
approved local **20,000-update SmolVLA pilot training**, batch 8 x accumulation 4,
FP32 trainable/BF16 autocast, 1e-4 peak LR, 1,000-step warmup/cosine to 1e-5.
54 tests and the two-update GPU/save-reload/data-binding preflight passed.
Fresh-base detached container `smolvla-pilot-train-20000` launched, output
`outputs/train/smolvla_pilot_20260916`. See
[training protocol and live handoff](SMOLVLA_PILOT_TRAINING_20260916.md).
No upload, further collection or final-scene evaluation. Launch is not completion;
do not modify training/provenance-bound source or start a duplicate job.

### Historical upload-first hold — superseded by training authorization above

**LATEST AUTHORIZATION / BLOCKER:** user authorized a safely named private
Hugging Face dataset upload, followed by SmolVLA protocol preparation once the
upload completes. Repository `VivekSai07/franka-panda-cube-stacking-pilot-420`
was created privately, but LFS upload failed with proxy HTTP 407. Fresh remote
inspection found only the initial Git attributes file, no dataset commit.
See [upload status and continuation](DATASET_UPLOAD_20260916.md). Local data is
unchanged. Training execution remains unauthorized; resolve upload first.

**PILOT COMPLETE — FINAL VALIDATION PASSED.** Independently inspected the
persisted runner, validator, replay result and container state on September 16.
Container `smolvla-pilot-420-overnight` exited 0 at 08:01:57 UTC, no OOM.
The dataset contains **420 episodes / 221,760 frames**, with **300 training /
120 validation** episodes, collected in 469 attempts. All 840 replay results
(420 episodes in each of two modes) passed. Alignment errors: current
0.021869, next 0.016087. Independent scene banks: 120 tuning / 240 final.
See [completion evidence](OVERNIGHT_PILOT_20260915.md#verified-completion--2026-09-16).

No training or further collection is authorized; upload is authorized above. Preserve
the dataset, scene banks and provenance-bound sources. Next proposed milestone:
prepare a fresh SmolVLA pilot training protocol with explicit manifest train IDs,
train-only statistics, held-out validation and an untouched final scene bank.
Never resume pre-fix checkpoints or blindly use the default Compose trainer.
Data-integrity PASS is not evidence of learned-policy stacking success.

## Historical authorization and progress — superseded by status above

**CURRENT LIVE JOB:** [overnight pilot](OVERNIGHT_PILOT_20260915.md) launched
after corrected preparation and full validation PASS. Container
`smolvla-pilot-420-overnight` is collecting **300 train + 120 validation** into
`datasets/franka_pilot_20260915`; runner state is
`outputs/pilot_overnight_20260915/run.json`. Automatic final validation follows
recording; no training/uploads. Do not modify hashed sources, run a duplicate,
or treat startup as completion. The historical holds/pending notes below are
superseded only by this conditional, validated launch authorization.

**Overnight launch approval:** the user requested proceeding with full dataset
collection tonight. Launch the reviewed **300 training + 120 validation** pilot
only after the running corrected 12-slot preparation completes and its full
validator passes. No waiver of integrity gates, no duplicate collection, no
training or uploads. At the latest check preparation had saved 8/12 episodes;
its manifest was still incomplete, so no pilot launch has occurred.

**Current task:** user approved joint-limit-aware expert correction and a fresh
12-slot preparation run. See [JOINT_LIMIT_PREPARATION_20260915.md](JOINT_LIMIT_PREPARATION_20260915.md).
The bounded command governor passed 48 tests and the formerly failing seed's
first approach; full recording and final validation remain pending. No pilot
or training authorization is implied.

Active container `smolvla-pilot-preparation-limits-12`; root
`datasets/franka_pilot_preparation_limits_20260915`; fresh banks under
`outputs/benchmark_preparation_limits_20260915`. Await completion, then run
the full pilot validator with `preparation_scenes`. Do not use the previous
incomplete three-episode root or claim validation before the final gate.

**Current preparation:** approved design implementation is underway; see
[PILOT_PREPARATION_20260915.md](PILOT_PREPARATION_20260915.md). Forty-four tests
pass, 420-slot and 12-slot plans and independent scene banks are generated.
`smolvla-pilot-preparation-12` **stopped with exit 1 after 3/12 episodes** in
`datasets/franka_pilot_preparation_20260915`. Joint 4 target -3.09553671 rad
violated lower actuator bound -3.0718 on slot preparation/train/0003,
seed 4281541513, tick 28. Deterministically reproduced without recording data.
The manifest is incomplete and rejected; no final validation PASS. No jobs run.
Both independent scene banks passed reload/provenance/geometry checks, but
cross-validation with a full demo set awaits the expert correction and fresh run.
Next decision: joint-limit-aware expert correction; do not weaken/skip bounds.
No 420-demonstration pilot launch, training or upload in this step. Historical
diagnostic sections below are not instructions to start another job.

**Current direction:** the user approved proceeding toward structured pilot
collection, not further automatic fitting diagnostics. The exact
[balanced-pilot specification](superpowers/specs/2026-09-15-balanced-pilot-design.md)
is recorded for review: 300 training + 120 validation demonstrations, successful
condition quotas, fixed evaluation banks and fail-closed validation. Implementation
and the 12-episode preparation gate follow review; no pilot launched yet.
Policy success is not guaranteed by dataset structure. Historical diagnostic
authorization/results below are completed, not instructions to start new jobs.

**Active authorization:** [precision-only overfit repeat](PRECISION_OVERFIT_20260915.md).
Same bounded 2,000-update single-episode test, FP32 trainable/optimizer storage
with the existing BF16 training autocast. No loss/data/timing change, no resume
or bulk collection. **Completed, exit 0:** all 33 expert normalization tensors
changed on step 1 and at the end; FP32 Adam moments verified; exact saved/reloaded
weights and predictions. Fixed-probe arm MAE 0.047295 rad vs prior 0.044300;
training-scene task result still 0/1. No arm clipping or bilateral grasp.
Evidence: `outputs/validation/smolvla_overfit_fp32_20260915`. Twenty tests pass.
Use `validation.precision.load_fp32_checkpoint` for its weights; generic older
loaders may round FP32 weights into BF16 parameter storage. No active jobs;
no additional training/loss changes automatically authorized. Preserve prior checkpoints.

Latest completed step: [fitting audit](FITTING_AUDIT_20260915.md), no optimizer
updates/rollouts/collection. Saved statistics and checked target/processor
roundtrips agree. VLM frozen as intended; all expert/projection tensors have
nonzero gradients in an audit batch. **33 BF16 expert normalization tensors
remained unchanged over the old 2,000-update run**; BF16 update rounding is an
actionable precision concern, not proven task-failure causality. The loss includes
24 padded action dimensions; temporal mask naming also mismatches but prior
training excluded padded chunks. Evidence:
`outputs/validation/smolvla_fitting_audit_20260915_final`.
Recommended next change is a controlled FP32-trainable-storage/BF16-autocast
test with the same bounded protocol; no new training is automatically authorized.

Current step: [paired replanning/grasp diagnostic](REPLANNING_DIAGNOSTIC_20260915.md),
evaluation only. Same overfit checkpoint/training scene, horizons 10 versus 1;
no additional training or collection. **Completed: both horizons failed (0/1
each), with 60 versus 600 measured chunk predictions.** Initial state/action
matched exactly; horizon 10 reproduced the prior rollout exactly. No arm-joint
clipping or bilateral target grasp in either run. Horizon 1 was slower and did
not rescue stacking. Expert-observation probes show remaining joint errors and
three incorrect release-adjacent gripper decisions. Eighteen tests pass.
Results/videos: `outputs/validation/smolvla_replanning_20260915`. No active jobs.

Latest authorized step: [single-training-scene overfit diagnostic](OVERFIT_VALIDATION_20260915.md),
episode 0 only, capped at 2,000 updates, fresh base initialization and one
training-scene rollout. No data collection or held-out evaluation in this step.

**Overfit diagnostic completed:** fixed nine-observation arm MAE decreased from
0.214860 to 0.044300 rad (79.4%), binary gripper agreement reached 99.33%, and
save/reload error was zero. **Training-scene task reproduction failed (0/1)**:
no bilateral target grasp, red cube displaced about 9.5 cm, no stack. No joint
clipping; 121 gripper-clipped ticks. Output:
`outputs/validation/smolvla_overfit_20260915`. Sixteen tests pass; no active jobs.
This shows improved fitting, not successful memorization or generalization.

The user approved a reusable joint-control benchmark design and a **small
end-to-end validation batch only**. No bulk collection, dataset upload or
training from a pre-fix checkpoint is authorized by this validation milestone.
Work is on `feat/reusable-validation`, in the original underscore checkout.

## Reading order for humans and agents

1. This status file: current decisions and execution evidence.
2. [Approved design](superpowers/specs/2026-09-15-reusable-validation-design.md).
3. [Implementation plan](superpowers/plans/2026-09-15-reusable-validation.md).
4. [Dataset-size research](SMOLVLA_DATASET_SIZE_ANALYSIS.md): source links,
   checkpoint revisions, comparison caveats and the supplied-text addendum.
5. [Current validation results](VALIDATION_20260915.md): measured gates and failures.
6. [Original two-episode report](RTC_TWO_EPISODE_REPORT.md): historical gate.
7. [Local-wheel environment](RTC_LOCAL_WHEELS.md): reproducible image setup.

## Decisions

- Goal: SmolVLA first, then pi0-FAST, pi0 and pi05, evaluated on the same
  immutable demonstrations and independent test scenes.
- Canonical data: 9-D measured joint/finger state, 8-D absolute joint-target /
  gripper command, two 512x512 RGB views, nominal 30 Hz. No normalized/padded
  actions or FAST tokens in the authoritative dataset.
- Preserve seed, full simulation initialization, model/source provenance,
  command/timestep trace and success diagnostics separately from policy inputs.
  Do not expose simulator object poses to policies.
- Proposed later budget: 300 training-demo pilot, conditional expansion to 600,
  120 held-out expert demos; these are recommendations, not guarantees.
- Full benchmark quotas, large splits and pi-family adapters are subsequent
  milestones. Small validation does not establish trained-policy success.
- Prior v1 smoke data is post-fix, two episodes. Prior v2 is interrupted bulk data,
  not training-ready. Preserve both during the new milestone.
- All Python execution/installations happen inside Docker. No host pip.
- Never read or print credentials; never commit datasets, wheels, videos or
  checkpoints. Commits use only the user-approved Git identity, no coauthors.
- Original Compose training auto-resumes and points at a fixed dataset: do not
  use its default command for this milestone.

## Execution status

**Current work:** the user authorized timing correction and another small batch.
See [TIMING_VALIDATION_20260915.md](TIMING_VALIDATION_20260915.md). Historical
results below refer to the old 34 ms contract; do not mix their dataset or
checkpoints with corrected-timing runs. The old source/assets were archived
beside the old output before editing. No bulk run is authorized.

Corrected timing unit tests: **15 pass**. Current bounded container:
`smolvla-validation-timing-v2`; target root
`datasets/franka_benchmark_validation_20260915_timing_v2`.
Collection completed: six successes / 3,168 frames, exit 0. Both replay modes
passed for all episodes with synchronized geometry and 30 Hz timestamps.
Alignment PASS: current 0.023667, next 0.017389. Fresh training/rollout container
`smolvla-learning-timing-v2` completed with exit 0: 100 finite updates, actual
weight changes and zero seeded save/reload error. **Task result remains 0/2**.
All 600-tick rollouts now span exactly 20 simulator seconds. No arm joints were
clipped; gripper clips were 204/207 ticks, bilateral target grasp counts were
zero and target lift was numerical zero. Raw/applied traces and new videos are
under `outputs/validation/smolvla_20260915_timing_v2`. No running jobs remain.
See the timing report for the next exact gate order; do not duplicate the job.

Implementation is on `feat/reusable-validation`. Eight unit tests pass, covering
simulator snapshots, deterministic conditions, invalid actions, dataset lifecycle,
fingerprints, fresh scoring, training-only statistics and chunk boundaries.

- The first foreground six-episode attempt was interrupted with exit 130 during
  terminal reuse. Its root `datasets/franka_benchmark_validation_20260915` is
  incomplete and preserved; do not train on it.
- Completed replacement: `smolvla-validation-6-retry`, exit 0, wrote
  `datasets/franka_benchmark_validation_20260915_retry`. Six successes from eight
  attempts, 3,120 frames, seed 20260915, four train/two validation episodes.
- Public model assets downloaded to `.cache/benchmark_models`, with pinned
  policy and VLM-metadata revisions. No separate VLM weights are needed when
  loading the full SmolVLA policy snapshot.
- Strict loading of `lerobot/smolvla_base` with our 9-D state, 8-D action and
  two-camera schema passed: 450,046,176 total / 99,880,992 trainable parameters.
- LeRobot 0.4.4 config loading must use `PreTrainedConfig.from_pretrained`, not
  `SmolVLAConfig.from_pretrained`, to decode the tagged `type` field correctly.
- Replay PASS: six episodes in both reference and saved-action modes; exact
  reference drift zero, maximum saved-action policy-state drift about 1.27e-7.
- Alignment PASS: current error 0.023539, next error 0.017174.
- `smolvla-learning-smoke` exited 0: 100 updates, batch 2, finite gradients,
  action-projection weight change 0.00345515, exact seeded save/reload agreement.
- **Policy task result: 0/2 successes.** Both held-out rollouts reached the
  600-tick limit. Range clipping occurred on 310 and 252 ticks respectively.
  This is a completed integration pipeline, not a successful benchmark policy.
- Outputs: `outputs/validation/smolvla_20260915` contains checkpoint, processors,
  training-only statistics, machine-readable metrics and two rollout videos.
- No containers remain running. No bulk collection/upload/new commit was made
  during this milestone. Changes remain on the working branch for review.

## Before the next milestone

1. Timing and clipping attribution were addressed in the timing-v2 repeat.
  The old-run notes above are historical; use the timing report for new results.
  Gripper clipping before binary thresholding does not alter the final binary
  command, so it is not an established explanation for failed grasps.
2. Overfit and paired replanning diagnostics completed without successful task
  reproduction. Consolidate findings and audit remaining training reconstruction
  errors before choosing another bounded fitting test; do not keep launching
  horizon sweeps or training extensions without explicit approval.
3. Add final benchmark quotas, independent seeded evaluation and consistent
   released-stack criteria. Validate pi-family adapters separately.
4. Choose a proper training budget only after these checks; 100 updates on four
   demos is not evidence that SmolVLA cannot learn the task or needs more data.

Bulk collection is still on hold. Read the evidence report before proposing
next experiments. Never resume a pre-fix checkpoint.