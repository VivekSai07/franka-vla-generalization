# B3 robosuite collection — 2026-10-03

Collection of the B3 cube dataset in the robosuite scene
(`experiments/b3_robosuite_data/`, plan
[2026-10-03-b3-robosuite-collection.md](superpowers/plans/2026-10-03-b3-robosuite-collection.md),
spec [2026-10-03-b3-robosuite-cubes-design.md](superpowers/specs/2026-10-03-b3-robosuite-cubes-design.md)).
Dated report: do not edit it to reflect later findings.

**Result: sealed, 560 of 560 episodes admitted, 0 substituted layouts, 7 rejected attempts of a
budget of 150.** Not a policy result; no policy has been trained on it. The scene runs on robosuite
1.5.2 and MuJoCo 3.4.0, so nothing measured on it is physics-comparable with the sealed MuJoCo 3.13.0
datasets.

## What was collected

| item | measured | planned |
|---|---|---|
| episodes | 560 (500 training, 60 validation) | 560 |
| per cell (10 trained pairs × 2 sides) | 25 training + 3 validation in all 20 cells | 25 + 3 |
| training distractors none / one / two | 100 / 200 / 200 | 100 / 200 / 200 |
| training pick position interior / inner corner / outer corner | 400 / 50 / 50 | 400 / 50 / 50 |
| held-out pairs (`yellow→red`, `green→blue`) collected | 0 | 0 |
| frames | 295,680 (528 per episode, 30 fps) | 528 per episode |
| videos | 1,120 AV1, every one decodes to 528 frames | 2 per episode |
| reserved evaluation scenes banked (never collected) | 100 (60 layouts; 40 report layouts in two conditions, 20 screen in one) | 100 |
| size on disk | 2.9 GB | about 3 GB |

`MIN_CLEARANCE_M` is **0.08 m**, frozen after the pilot and never changed. The dataset loads with
`LeRobotDataset` (first and last frame read).

## Seal

| field | value |
|---|---|
| fingerprint | `e14af864596caf7cb515ebc500957e970a3cf12a40a3a65c1692696d92b8e6c0` |
| image | `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e` (robosuite, MuJoCo 3.4.0) |
| `provenance_sha256` | `d99d2c629099bfbbc9eac723cda48623aa0931dafa94527cef4316f5dd2de7ba` |
| `plan_sha256` | `ea6b572f69d15e1b28a1bc70293817f8f33098aa2984ace1ba8acaad0150140f` |
| `substitutions_sha256` | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` (empty) |

Verified in a second process after the run: the recomputed fingerprint equals the seal; the plan of
record equals a fresh `make_plan()`; the current `provenance()` equals `provenance.json` and the
sealed digest. Output: `outputs/b3_robosuite_data/b3_robosuite_20261003/`. The journal admits every
slot 0–559 exactly once and in order, and every admitted line has all six gates true.

## Pilot and the frozen clearance

Single attempt per slot at 0.08 m: 23 of 23 admitted (first 10 `near_pick` and 10 `near_place`
slots, 0 of 20 rejected; first 5 inner-corner and 5 outer-corner picks, 0 of 5 each). Frames of
three pilot episodes (two-distractor near-pick at an outer corner, a control, two-distractor
near-place at an outer corner) showed the right cubes in the right colours and the instructed pick
ending on the instructed place cube; the wrist camera cuts a cube at its frame edge at the home
pose. Frames of the 560 collected episodes were not inspected beyond that.

## Rejections

7 rejected attempts, all on attempt 0 and all admitted on the next attempt of the same layout:

| reason | count | slots |
|---|---|---|
| `expert_success` | 6 | 43, 196, 211, 307, 315, 383 |
| `distractors_undisturbed` | 1 | 61 |

By profile (not by reason): none/interior 1, one-far/interior 1, one-near-pick/inner 1,
two-far/interior 1, two-near-pick/interior 2, two-near-place/interior 1. Seven attempts of 567 is
1.2%. The causes of the six `expert_success` rejections were not diagnosed (the gate name is all
the journal holds); with so few, no rate by profile is claimed.

## Quality of the admitted episodes

- Final stack offset (`final_xy_m`) min / median / max: **0.19 / 6.9 / 18.9 mm** against the 20 mm
  window; z 0.04096–0.04100 m (window 0.035–0.045). The maximum is 1.1 mm inside the gate. The
  median is the scripted expert's own ±5 mm target jitter plus settling, as measured in the bring-up.
- No distractor of an admitted episode moved more than **0.29 mm** (660 distractor instances; limit
  20 mm). At 8–12 cm the gripper does not touch a near distractor; the clearance was not tested
  by contact. Whether crowded layouts are harder for a *policy* is not measured here.
- Wall time per attempt (expert plus replay) 44.2 / 46.0 / 52.4 s (min / median / max); with
  encoding and saving about 73 s per admitted episode.

## What went wrong: the run crashed at episode 541

The collector finalized and reopened the dataset after every admitted episode so that a kill
between episodes would leave a readable dataset. Each reopen makes the Hugging Face `datasets`
library build an arrow cache of the whole dataset under `HF_HOME`, which the run helper put on
the container's 8 GB `/tmp` tmpfs. One open of 541 episodes writes 31 MB (measured), so the sum over
about 540 reopens is roughly 8 GB. At 12:32 CEST, after 541 episodes had been saved, the reopen
raised `OSError: [Errno 28] No space left on device` (the disk itself had 209 GB free). The pilot
ran 3–4 episodes and could not show it. [FAILURES.md](FAILURES.md) #41.

State after the crash: the dataset held 541 complete episodes; the journal held 540, because the
journal line is written after the reopen. `--resume` refuses that mismatch by design.

Recovery, with no code change (so `provenance()` and the seal binding were unaffected):

1. Slot 540 was re-executed through the collector's own `execute` (attempt 0, planned layout). Its
   trace file (qpos, ctrl, initial state, final qpos) and its dataset rows (state, action, task
   index, 528 rows) were bit-identical to what the crashed run had saved, and all six gates passed;
   only then was its journal line appended. That line carries a `note` field saying so; it is the
   only journal line with extra content. No dataset file was edited or deleted. (The first attempt
   at this script failed one of its own checks because of a tensor-versus-int type bug in the check;
   it wrote nothing, and the check was corrected, not relaxed.)
2. `--resume` was run once, detached, with a 24 GB `/tmp` (a `docker run` flag).

Timeline (CEST): launch 01:05, crash 12:32, resume 13:16, sealed about 13:41. The 44 minutes between
crash and resume are the time the run sat stopped; running time was about 11.9 h.

## Caveats

- **Hash-bound from now on.** The seal's provenance covers every `.py` and the README of
  `experiments/b3_robosuite_data/`, `experiments/robosuite_scene/*.py`, the shared modules it
  imports, `src/`, the spec and the plan. Editing any of them invalidates the seal for training.
  This makes `experiments/robosuite_scene/*.py` effectively frozen; the robosuite scene package is
  no longer "not hash-bound".
- The training and evaluation packages (plans 2 and 3 of the spec) do not exist yet.
- The tests for the plan and the scene module were written after the code, not before (the plan
  says test-first); the plan test then caught one real bug (a shared alternation counter left the
  one-distractor colours 4/6 instead of 5/5), and the validator and scheduler tests were
  mutation-checked.
- Reserved evaluation layouts were banked in `bank.npz` / `bank.json` and never executed.
- The scripted expert ran from a single process per container; the machine has 48 cores and the
  collection used about five.
