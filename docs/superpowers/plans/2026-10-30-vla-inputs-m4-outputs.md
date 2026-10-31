# M4 — auxiliary box head (M4a) and force-output scoping (M4b) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> This plan fixes files, interfaces, tests and gates; the code is written at execution time, test first, in the order below.

**Goal:** (M4a) X-VLA trained as P2's run plus an auxiliary loss predicting the target's box, inputs unchanged, evaluated as A4;
(M4b) a written scoping note on force-carrying outputs, no training.

**Architecture (M4a):** `experiments/xvla_p2_auxbox/`, forked from `experiments/xvla_props_p2/`, with a local `XVLAPolicy`
subclass adding an MLP head on separately pooled `vlm_features` and `aux_visual_inputs` and the weighted loss; S0's `target_box` as training targets only.

**Spec:** [../specs/2026-10-30-vla-inputs-m4-outputs-design.md](../specs/2026-10-30-vla-inputs-m4-outputs-design.md).
**Depends on:** M3's result (start condition), S0, M0.

## Global constraints

- M4a starts only if M4 spec D1 holds (A3 beats A0 on M0's primary test, or A3's `wrong_object` count is lower at p < 0.05,
  paired exact); otherwise the user decides, and the reason is recorded in `TODO.md`. This remains a budget heuristic,
  not evidence that auxiliary supervision requires a positive M3. The secondary raw p-value is exploratory. A4 is not
  detector-fed A3; redesign after viewing C needs fresh confirmation.
- Recipe identical to P2's (derived and asserted equal, as M2). λ fixed once from the first preflight batch, then a constant.
- LeRobot's installed `modeling_xvla.py` is not edited; the head lives in the fork.
- The head's box error is never used to choose a checkpoint.

---

### Task 1 (M4a): start condition

- [ ] Read A3's `compare` record and the paired `wrong_object` test; write the decision with numbers into `TODO.md` and
  `docs/BENCHMARK_STATUS.md`. Stop here if the condition fails and the user does not override.

### Task 2 (M4a): fork, head, loss

**Files:** copy `experiments/xvla_props_p2/` → `experiments/xvla_p2_auxbox/`; create `head.py`, `tests/test_head.py`; modify
`contract.py` (targets from the S0 sidecar), `runner.py` (loss, λ).

**Interfaces:** `head.BoxHead(hidden: int = 256)` — separately mean-pool front/language and wrist features, concatenate
in fixed camera order → Linear → GELU → Linear(8);
`head.box_loss(pred (n, 8), target (n, 8)) -> Tensor` = L1 over the four coordinates of each camera, averaged over the two
cameras (no `in_frame` term: constant 1 in S0's training rows); `head.AuxBoxXVLA(XVLAPolicy)` whose `forward(batch)` returns
`action_loss + LAMBDA * box_loss` and logs both; the head initialised from `torch.Generator().manual_seed(HEAD_SEED)`.
- [ ] **Step 1: failing tests:** the loss weights both cameras equally and both views reach the head (zeroing either view
  changes the prediction); the global torch random state after building `AuxBoxXVLA` equals the state after building `XVLAPolicy`;
  `select_action` output with the head present equals the plain policy's for the same weights and seed; recipe equals P2's.
- [ ] **Step 2:** forking-skill pass.
- [ ] **Step 3–4:** implement, pass, mutate (head drawn from the global generator; mask removed).
- [ ] **Step 5:** commit `M4a: auxiliary box head`.

### Task 2b (M4a): prove the auxiliary path reaches both views and shared parameters

- [ ] Inspect installed `forward_vlm`: identify view-0/language tokens and separate wrist-view visual tokens. Test changes
  to each view can affect auxiliary output; use explicit token masks and camera order. Avoid a duplicate VLM forward.
- [ ] Assert nonzero finite auxiliary gradients reach shared trainable visual parameters and every head parameter is in
  exactly one optimizer group at the intended LR. Keep the action path and global RNG stream unchanged at lambda zero.
- [ ] Specify FP32 loss reductions as in revised spec, including a zero coordinate loss for empty validity masks.
- [ ] Prove checkpoint save/load into ordinary inference removes only declared auxiliary keys and checks all base weights;
  no permissive load that silently ignores missing/mismatched base weights.

### Task 3 (M4a): λ and zero-equivalence

- [ ] `--probe-lambda` (not a gate; scratch output; run **before** `--check`): action and box losses on the first training batch at
  initialisation, recording batch ID/RNG, loss reductions and gradient norms; finite nonzero denominator required;
  `LAMBDA = 0.1 × action / box`, written to `README.md` and `recipe()` as a constant (both are in `SOURCE_FILES`,
  so this happens before `--check` and is never changed afterwards).
- [ ] `--zero-equivalence` (λ = 0): 30 updates equal to P2's runner for metrics, base weights and their Adam moments; the head's
  parameters are excluded. Mutation run recorded.
- [ ] Commit `M4a: lambda fixed, zero-equivalence`.

### Task 4 (M4a): training and evaluation

- [ ] Pre-launch checklist; `--check` → `--preflight` → `--resume-proof` → `--equivalence` → `--zero-equivalence` → `--train`.
- [ ] M0: `prepare --arm A4` with `XVLAAdapter(checkpoint, state_p2)` (the head is unused at test time) → smoke → S → `select` →
  C with winner plus fixed 4.0-epoch secondary checkpoint (deduplicate) → `compare --arm A4`; A4 against A3 reported as exploratory.
- [ ] `docs/VLA_INPUTS_M4A_<date>.md`; docs pass; commit.

### Task 5 (M4b): scoping note, no training

- [ ] From S0's `report.json`: grip force and wrist load by object and segment in the demonstrations.
- [ ] From M0's recorded traces: grip force and wrist load in A0's and A2's failing milk trials against their successes
  (descriptive; layouts are the unit).
- [ ] Read robosuite 1.5.2's operational-space controller with variable impedance and the gripper actuator definition in the
  robosuite image; list what a force-carrying action would need from the controller, the expert and the scene, and which frozen
  files each option would have to fork.
- [ ] Write `docs/VLA_INPUTS_M4B_SCOPING_<date>.md` with a recommendation (new phase with its own spec, or stop); update
  `TODO.md`; commit.
