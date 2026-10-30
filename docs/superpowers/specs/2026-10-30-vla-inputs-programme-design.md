# VLA inputs programme — same data, new information — design

**Date:** 2026-10-30. **Status: draft for external review.** Direction, sub-projects and order approved by the user on
2026-10-30; everything below the "Approved" table is proposed and open to review. Nothing has been built or run except
the read-only checks cited here. Follows P3, closed as a failed attempt
([PROPS_P3_EVALUATION_20261027.md](../../PROPS_P3_EVALUATION_20261027.md)).

## Review revision — 2026-10-30

Read the [pre-implementation audit and research roadmap](../../VLA_INPUTS_AUDIT_20261030.md) before execution.
The approved programme and broad order remain. Revised specs/plans correct evaluator binding, M4a camera access, signal
semantics and statistical/resource claims. No code or runs have been added. Additional research arms in the audit are
recommendations only. Runtime gates and choice of pi0.5 memory/batch recipe remain outstanding on the RTX workstation.

M0's shared core is frozen before A0; arm adapters are separately bound. Results are single-training-seed selected-system
pilots, with a predeclared 4.0-epoch secondary contrast and a fixed A1–A4 multiplicity family. This adds up to 384 evaluation
trials per arm when the selected checkpoint differs; budget this before launching. No broad method claim without replication.

Only M0 preparation/feasibility and S0 are CPU work. M0 GPU rollouts need separate scheduling from M1 training unless
measured combined peaks fit. Historical workstation free-space and GPU figures below must be remeasured, not assumed.

## Question

Every phase so far changed the **dataset**. This programme holds the data fixed and changes **what the model sees** (and,
later, what it emits). With the 540 sealed P2 episodes unchanged:

1. Does a different VLA family (pi0.5) do better than X-VLA on the same data and inputs? (M1)
2. Does contact information in the robot state (wrist force/torque, finger contact forces) improve success? (M2)
3. Does the target's 2D bounding box in the robot state improve success? (M3)
4. Can the box be learned as an auxiliary output, so it is not needed at test time? (M4a). Later: action outputs that carry
   force (M4b, scoped only).

## Approved 2026-10-30 (user)

| item | decision |
|---|---|
| sub-projects | M0 evaluation protocol and baseline, S0 extra-signal sidecar, M1 pi0.5 vs X-VLA, M2 force/torque in state, M3 bounding box in state, M4 output side |
| order | M0 preparation/feasibility and S0 (CPU) alongside M1 training (GPU); M0 GPU rollouts separately resource-gated; then M2, M3, M4a |
| pi variant | pi0.5 (`pi05` in the pinned LeRobot 0.4.4), base weights `pi05_base` |
| pi recipe | each model's own recipe at matched exposure (epochs), stated as "model plus its recipe" |
| F/T signals | wrist 6-axis force/torque + 2 finger contact forces, per frame at 30 Hz, per-dimension normalisation, no history |
| box form | numbers in the state, target object only, both cameras, normalised to [0, 1]; drawn-on-image later as a variant |
| framing | M2 and M3 are information upper bounds (privileged simulator signals at test time), not deployable gains |
| evaluation | fresh reserved layouts disjoint from every P2 layout, size from a power analysis, every arm on the same layouts, McNemar, per-object tables, pre-registered criteria per sub-project |
| where | specs and plans committed on `main` for review first (user, 2026-10-30); work happens later on `feat/vla-inputs` in a worktree |

## Why this is possible without new data

The P2 root keeps, per episode, the full simulator trace: `episodes/episode_NNN.npz` holds `qpos` (every joint and every
free body, float64), `ctrl`, `props_pose` for each of the 452 frames, and the `initial_state`. **Measured 2026-10-30:**
replaying the sealed actions from the sealed initial state reproduces the sealed `qpos` byte for byte (3 of 3 episodes probed,
every frame, error 0.0). This supports reconstruction at the observation instant; position equality alone does not independently validate historical forces. Every extra quantity is reconstructed, with the replay
proven equal to the sealed trace for all 540 episodes before anything is read from it (S0). The images and the actions the
model is trained on stay the sealed ones.

## Sub-projects

| id | spec | needs | GPU | output |
|---|---|---|---|---|
| M0 | [m0 common evaluation](2026-10-30-vla-inputs-m0-evaluation-design.md) | P2 bundle | eval only | a policy-agnostic evaluator, two reserved tiers, the control arm A0 measured |
| S0 | [s0 signal sidecar](2026-10-30-vla-inputs-s0-signals-design.md) | P2 seal | none | sealed per-frame F/T, finger forces, boxes for all 540 episodes; the same signal code for rollouts |
| M1 | [m1 pi0.5 on P2](2026-10-30-vla-inputs-m1-pi05-design.md) | M0 (for evaluation) | training | arm A1 |
| M2 | [m2 force/torque in state](2026-10-30-vla-inputs-m2-force-design.md) | S0, M0 | training | arm A2 |
| M3 | [m3 box in state](2026-10-30-vla-inputs-m3-box-design.md) | S0, M0, M2's package | training | arm A3 |
| M4 | [m4 output side](2026-10-30-vla-inputs-m4-outputs-design.md) | M3 result | training | arm A4 (M4a); M4b scoped only |

Plans: [M0](../plans/2026-10-30-vla-inputs-m0-evaluation.md) · [S0](../plans/2026-10-30-vla-inputs-s0-signals.md) ·
[M1](../plans/2026-10-30-vla-inputs-m1-pi05.md) · [M2](../plans/2026-10-30-vla-inputs-m2-force.md) ·
[M3](../plans/2026-10-30-vla-inputs-m3-box.md) · [M4](../plans/2026-10-30-vla-inputs-m4-outputs.md).

```mermaid
flowchart LR
  M0[M0 evaluator + A0] --> A1E[A1 eval]
  S0[S0 sidecar] --> M2[M2 train] --> A2E[A2 eval]
  M1[M1 image + train] --> A1E
  M2 --> M3[M3 train] --> A3E[A3 eval]
  A3E --> M4a[M4a aux box head]
```

## Arms (all evaluated by M0 on the same layouts)

| arm | model | state | images | data | trained by |
|---|---|---|---|---|---|
| A0 | X-VLA, P2 checkpoint 101,520 (bundle) | 9 | sealed P2 | P2 | existing (2026-10-16) |
| A1 | pi0.5 | 9 | sealed P2 | P2 | M1 |
| A2 | X-VLA | 9 + 8 (F/T 6, finger forces 2) = 17 | sealed P2 | P2 + S0 | M2 |
| A3 | X-VLA | 9 + 10 (box 4 + in_frame 1, per camera) = 19 | sealed P2 | P2 + S0 | M3 |
| A4 | X-VLA + auxiliary box head | 9 | sealed P2 | P2 + S0 (targets only) | M4a |

A2, A3 and A4 use P2's recipe unchanged (126,900 updates = 5.0 epochs, effective batch 8, the same checkpoint grid and the
same sample order). X-VLA has 20 proprioception slots (`check_model_inputs` pads 9 to 20 with zeros), so 17 and 19 fit
without an architecture change.

## Rules that hold in every sub-project

- **One variable per arm.** A1 changes the model (and its recipe), A2 only the state contents, A3 only the state contents, A4
  only the training loss. Renderer stays OSMesa (EGL is not adopted; A0 must not be evaluated under EGL).
- **Nothing sealed is edited.** P2's `provenance()` hashes 76 files (`experiments/props_p2_data/**`, `experiments/props_scene/*.py`,
  `experiments/robosuite_scene/*.py`, `src/**`, the P2 spec and plan, shared helpers). They are imported read-only; where
  behaviour must differ the code is forked into the new package, never patched on disk.
- **Every new training package proves equality before it trains:** with its extra inputs set to zero it reproduces P2's
  training byte for byte over 30 updates (M2, M3, M4a). That makes the extra inputs the only difference in the real run.
- **Selection never touches the comparison tier.** Checkpoints are chosen on M0's selection tier; the comparison tier evaluates the S winner and the predeclared fixed-phase secondary checkpoint,
  deduplicated if identical. C never chooses between them. Recipes/analysis freeze before C; C-informed redesign needs fresh confirmation.
- **Offline metrics are never used** to choose, stop, or claim.
- **Predictions are written before rollouts** (each spec has one) and results are reported against them, including nulls.
- Docs pass on every finding; commits by the user only; nothing pushed by the agent.

## Predictions (written 2026-10-30, before any run)

- P2's failures at 101,520 are dominated by "never lifted" (12 of the 30 report failures; all 12 closed the gripper, 9 held the
  target), with milk the weakest object (57.6% on 118 fresh trained-bin layouts). Wrong object is 7 of 120.
- **M3 (box):** wrong-object choices should fall toward zero (the target box says which object); little effect on milk's hold.
  A gain around the wrong-object rate (about 6 points) is a hypothesis, not an upper bound; boxes may also change alignment.
  M0 has low unadjusted power at that effect size.
- **M2 (F/T):** an arm intended to help the hold, sampled at 30 Hz but used at 3 Hz replanning (ten queued actions). But the state already contains both finger positions, so the
  policy already "knows" the gripper closed; the new information is load (object weight 0.5–2 N at the wrist, masses
  0.05–0.20 kg) and grip force. The expert never fails a grasp, so demonstrations contain no recovery. **A null is plausible.**
- **M1 (pi0.5):** no confident prediction; larger VLM, different action head, state given as text tokens.

## What this programme cannot show

- That a deployable robot benefits: the extra signals are exact simulator quantities (an upper bound). F/T has a real-robot
  analogue (estimated external wrench); a detector-fed box policy needs its own error-distribution evaluation; M4a separately tests auxiliary supervision.
- Anything about other tasks, objects, or more data. The data is P2's, unchanged.
- That pi0.5 is a better *model* in general: A1 is pi0.5 with its recipe on the 480 training episodes (60 validation episodes), against X-VLA with P2's recipe.

## Known blockers and costs (measured or read, 2026-10-30)

| item | finding | where it lands |
|---|---|---|
| pi0.5 cannot be constructed in the pinned image | `PI05Pytorch.__init__` requires `transformers.models.siglip.check` (LeRobot's patched transformers); the image has transformers 4.57.6 and the import fails. **Resolved in a throwaway probe (2026-10-30):** transformers@`dcddb970176382c0fcf4521b0c0e6fc15894dfe0` (4.53.3) plus tokenizers 0.21.4 pass the check and construct pi0.5 (3,616,757,520 parameters) | M1 Task 1: two new images, own digests |
| pi0.5 base weights and the PaliGemma tokenizer are not in the image | `lerobot/pi05_base` is public (pin revision `9e55186a…`, weights `0eb11ca9…`, 14.5 GB); `google/paligemma-3b-pt-224` is gated with **manual** approval; containers run offline | M1 Task 1 Step 0: request access now; Task 2: fetched on the host through the proxy, hash-pinned, token never committed |
| disk | 123 GB free on 2026-10-30 evening. The `/var/log/syslog` growth (~30 GB/day, October) is no longer visible: only `syslog.1` (32 MB, 2026-10-25) exists. Re-measure before every gate | every training plan has a disk gate; pi0.5's working set is larger than X-VLA's (M1) |
| GPU | idle on 2026-10-30 (363 MiB used, no other compute process); shared, and another user's job held 37 of 48 GiB on 2026-10-23 | the runners' free-memory guards stay; not relaxed |
| evaluation memory | 8 X-VLA workers that never switch checkpoint ran at ~4.2 GB each (P3); workers that switched reached 7.3 GiB and one died (P2) | M0: one checkpoint per worker pool |
| host | 48 cores, 125 GB RAM | worker pinning; CPU-offloaded optimizer is feasible for M1 (≈ 43 GB) |
| X-VLA training time | 0.524 s/update raw, ~0.9 effective with thermal pauses → about 32 h per X-VLA arm | M2, M3, M4a |
