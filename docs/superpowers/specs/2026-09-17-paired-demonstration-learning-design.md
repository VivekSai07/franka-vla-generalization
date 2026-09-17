# Paired-demonstration learning experiment

Date: 2026-09-17. User approved the two-model controlled budget in the design
question. This written specification is the review gate before implementation.

## Goal and evidence boundary

Test whether a small paired-instruction demonstration set improves closed-loop
object-role selection and reproduction on seen layouts, followed by transfer to
unseen layouts. The completed baseline scored 4/120; the paired diagnostic
showed no first-object switches in four non-null reversed pairs and only 1/12
seen-scene successes. Neither establishes a single root cause or a guaranteed fix.

## Alternatives and selected comparison

1. **Selected: two fresh models at equal optimization budget.** Same training
   layouts, paired directions versus one direction per layout. Allows a practical
   comparison of the paired-data package against a simpler control.
2. Paired model only: cheaper, but cannot attribute gains to paired supervision
   rather than task reduction, extra exposure or other changes.
3. Bulk random collection or unrestricted hyperparameter tuning: deferred because
   neither directly addresses the measured role-selection/seen-scene failures.

The selected control has fewer unique demonstrations. This is **not** an isolated
causal test of pairing with unique-demo count held fixed. Statistics also differ
because each condition fits its own training subset. Report those limitations.

## Data budget: 36 demonstrations on 18 fresh layouts

- Colors: red, green, blue, yellow. Enumerate the six unordered combinations in
  that order. Exactly two present cubes, no distractors in this experiment.
- Per unordered pair: two training layouts and one held-out layout.
- Total: **12 training layouts + 6 held-out layouts**.
- On each layout execute both ordered instructions from the **identical complete
  initial simulator state**. Total **24 train demos + 12 held-out demos**.
- Pair order and layout identity are fixed before recording. Physical reset,
  camera geometry, appearances and initial gripper controls are identical for
  both directions. Do not just reverse labels on an existing action sequence.
- Fresh seeded streams use master seed 20260917 and explicit split/layout/direction
  identifiers. Store the exact generated seeds and full reset states before any
  expert execution. Compare exact geometry hashes against previous demos/banks
  and all new layouts; reject duplicates rather than accepting seed distinctness
  alone as independence. Do not restore final-bank states for rollouts.
- Preserve existing workspace bounds, clearance, timing, robot model, joint
  governor, camera order/resolution, action/state semantics and success rule.
- A layout is not silently resampled because one instruction fails. On the first
  expert task failure or integrity error, preserve the incomplete batch and stop
  for a design decision. No retry, resume or quota substitution in this experiment.
- Layout families stay entirely in one split. Record both-direction selection
  and any expert feasibility filtering limitations explicitly.
- Canonical old datasets remain unchanged. Create a new exclusive dataset root
  and separate scene records. No data or model uploads.

## Approved historical-geometry amendment — 2026-09-17

The user explicitly approved proceeding with stage-one preparation despite exactly
three legacy episodes lacking full initial cube geometry. This narrowly amends
the historical duplicate-comparison requirement above, not any other gate:

- Allow only the exact inventory message
  `datasets/franka_libero_stacking_v1: 2 episodes without full initial geometry`.
- Allow only the exact inventory message
  `datasets/franka_libero_stacking_v2: 1 episodes without full initial geometry`.
- Keep both messages visible. These episodes remain **geometrically unchecked**;
  no claim of geometric non-overlap with them is possible. Do not infer poses,
  mark them checked, delete them, or modify old metadata/manifests/artifacts.
- Continue checking every recoverable historical layout, including all prior
  paired reset sets and partial runs, and all new layouts against one another.
  Exclude only the current paired root when rescanning its historical inventory.
- Record the observed exceptions machine-readably in the historical inventory,
  new run manifest and validation results, with reason `userapproved20260917`,
  dataset/count, exact message, `geometry_checked: false`, explicit exclusion
  from new train/eval, and actual metadata SHA256 values from that inventory.
  The current existing inventory binds these files:
  - [v1 metadata](../../../datasets/franka_libero_stacking_v1/meta/info.json):
    `7561ab94498836e48a26e182f98aa9d59ed94fcb09b2bd5e64b64c5edc18d649`.
  - [v2 metadata](../../../datasets/franka_libero_stacking_v2/meta/info.json):
    `67f2bee9ece73ad6a85cf08b40b5a0b901d3e299a3aaf44831dc36fabe26229d`.
  These are file bindings, **not recovered geometry hashes**. No additional
  legacy file or episode geometry is claimed checked by this exception.
- Validation and the final paired readiness gate must reconstruct this same
  exception policy, verify inventory/manifest/result bindings, and compare a
  fresh historical inventory including file hashes. Missing/tampered exception
  evidence, new gaps, changed counts/paths, metadata/replay count mismatches,
  corrupt/hash-mismatched/nonfinite/unindexed artifacts and all other integrity
  errors still fail. No broad ignore-missing flag or root exclusion is allowed.
- All training/evaluation in this design consumes **only the new paired dataset
  and its reset states**. The three legacy episodes are never candidates for
  either model's train/held-out subsets or any of the 72 evaluation trials.

This approval permits the narrow code/spec/test preparation, not a launch in this
implementation session. CPU Docker tests and read-only CLI preflight only; no
collection, GPU jobs or commits. Main launch follows review. The full 36-demo
validation and downstream training/evaluation remain pending; configuration
preflight alone never grants `training_ready`.

## Training conditions

**Paired:** all 24 training demonstrations, both directions per training layout.

**Control:** one direction from each of the same 12 layouts. For each unordered
pair, choose forward on its first training layout and reverse on its second.
This produces 12 demos, one per ordered pair. The remaining demonstrations are
not used by this model. Uniform sampling is repeated across epochs to match
optimizer exposure; do not write duplicate demonstrations into the dataset.
For the control, the opposite direction on a training layout is a **seen-layout,
unseen-layout/instruction combination**, not a demonstration-reproduction test.
Report control trained-direction and opposite-direction results separately.

Common recipe, explicitly approved:

- Two independent fresh initializations from cached `lerobot/smolvla_base`,
  revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`. Never resume old weights.
- **5,000 optimizer updates per model**, microbatch **8**, accumulation **4**,
  effective batch **32**, 160,000 sampled chunks per model.
- Seed **20260917** per run. Same seed does not imply identical minibatches or
  gradients when subsets differ; record actual sample exposure.
- AdamW betas (0.9,0.95), weight decay 1e-10, gradient norm cap 10.
- Linear **500-update warmup** to **1e-4**, cosine decay to **1e-5** at 5,000.
- Frozen VLM; FP32 trainable expert/projection weights and optimizer moments,
  BF16 training autocast. Keep architecture and stock flow-matching loss unchanged.
- 9-D measured state, 8-D absolute joint/gripper action, two 512x512 RGB views.
  No simulator object poses or phase labels as model inputs; no augmentation.
- Chunk length 50; complete-only chunk starts and explicit empty temporal-pad
  assertions. Document stock dummy action dimensions, do not patch loss here.
- Fit state/action statistics on each condition's own selected train episodes;
  never dataset-global statistics or held-out trajectories.
- Run conditions sequentially, control then paired; no hyperparameter revision
  based on the first model's performance.
- Save update 1 verification checkpoint and updates 1,000 through 5,000. Check
  FP32 storage/optimizer state, finite values, actual updates and exact reload.
  Always evaluate the **final 5,000 checkpoint** for the primary comparison,
  not a retrospectively chosen best checkpoint.
- Save fixed offline probes at frames 0, 10, 30, 60, 107 and 347 when complete
  chunks fit. Report early-phase and per-direction results separately. Offline
  probes are diagnostics, not stopping or model-selection criteria.

## Closed-loop evaluation budget: 72 rollouts

For each of two final checkpoints:

- Both requested directions on 12 training layouts: **24 seen-layout trials**.
- Both requested directions on 6 held-out layouts: **12 held-out trials**.
- Total **36 per model, 72 overall**. No retries or favorable-seed selection.
- Reuse identical saved initial states and sampling seed for each model and each
  direction on a layout. Seed = 20260917 + global layout ordinal (train 0–11,
  held-out 12–17). Common seeds align draws by call index, not semantic phase.
- Same 40-second/1,200-tick timeout, 30 Hz/20 substeps, 50 predicted actions and
  ten-action execution horizon. Strict requested-pair released-stack predicate
  sustained 30 ticks; record actual prediction counts.
- Save camera videos, raw/applied actions, state/time traces, first bilateral
  and sustained grasped/lifted object, all-pair stable stacks and interventions.
- Nonfinite policy actions count as failed trials. Infrastructure or provenance
  failures stop the experiment rather than becoming task failures or exclusions.
- CPU replay validates completed traces and scores, with recorded artifact hashes.

## Primary outcomes and decision rule

Report for each condition, separately on train and held-out layouts:

1. Correct instructed success count with all scheduled trials in the denominator.
2. **Both-directions-successful layout count** (out of 12 seen / 6 held-out).
3. Requested first-contact/lifted object, no-grasp outcomes, wrong ordered stacks.
4. Paired direction responsiveness, preserving null/tied outcomes rather than
   treating failures as object-choice agreement or switches.
5. Intervention-free score separately from ordinary success; no metric redefinition.

This small one-seed pilot supports a feasibility decision, not population-level
significance or a 100% success claim. No numerical success threshold has been
authorized as an automatic launch trigger for a larger experiment.

- If paired training reproduces both roles more reliably on seen and held-out
  layouts than control, propose a larger versioned paired-data experiment.
- If both fail on seen layouts, do not scale collection: investigate fitting,
  model representation and closed-loop observation/action mismatch.
- If seen reproduction is strong but held-out transfer weak, prioritize targeted
  layout coverage before claiming a grounding solution.
- If results are mixed or control equally strong, report ambiguity; do not credit
  pairing automatically or silently extend budgets.

## Implementation and safety gates

Implement separately from provenance-bound baseline collectors/trainers/evaluators
or archive and explicitly version any necessary shared change. Never rewrite old
manifests to make stale bindings pass. Reuse verified physical helpers where possible.

Before optimizer work, all 36 demos must pass schema/label/timing/video checks,
same-reset direction identity, split disjointness, joint-command reconstruction,
full reference/persisted-action replay and original alignment validation. Tests
must cover selection, direction labels, family-level splits, control counts,
fixed update budget, padding, incomplete data rejection and denominator handling.

After CPU tests, run a tiny isolated GPU save/reload preflight whose weights are
not resumed. Log exact containers, source/runtime/model/data hashes and resource
use. Require fresh output paths and adequate disk space; stop on failures.

No existing final/tuning benchmark scene is added to training. The 240 final
scenes remain sealed. Preserve old checkpoints, data and results. Local milestone
commits use only the user's configured identity; push only with explicit permission.

## Budget and authorization boundary

Collection: 36 demos; actual render/encoding time measured after first examples.
Training: 10,000 optimizer updates total, approximately 2.3 hours at previous
throughput before setup/probes/checkpoints; this is an estimate.
Evaluation: up to 72 x 40 simulated seconds; roughly 4 hours at previous CPU
rendering rates, plus replay/validation. Reserve at least 30 GiB before execution.

The user approved this exact two-model budget. Review this written spec before
implementation planning/execution. No new collection or learning job was started
while writing it. Do not expand the approved budget automatically.