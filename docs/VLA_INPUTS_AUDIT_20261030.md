# VLA inputs programme: pre-implementation audit

Reviewed 2026-10-30 against commit `67d5376`. Scope: all seven design documents, all six implementation plans, the power script, relevant collection/training/evaluation source, and primary online sources. No experiment implementation or training was performed; numerical power calculations are planning analysis.

## Verdict

Keep the programme, its sealed P2 data, and its broad order. Do not execute the original plans unchanged. The strongest parts are paired layouts, train-only statistics, action replay rather than expert regeneration, zero-treatment equivalence, and explicit privileged-input framing. The most serious defects concern evaluator provenance, M4a's missing camera input, the interpretation of force feedback, and inference beyond what one training run can establish.

The revisions in the linked specs/plans address those defects. Further research options below are recommendations, not additional approved runs. Runtime gates remain outstanding on the RTX workstation.

## Findings and required corrections

### R1 — High: the evaluation plan invalidates its own comparisons

M0 Task 6 hashes `vla_inputs_eval/*.py` and requires equality with A0. M1 Task 6, M2 Task 4 and M3 Task 2 subsequently modify `adapters.py`. Every resulting comparison would fail its intended provenance gate. Calling the evaluator “not hash-bound” does not resolve this contradiction.

Freeze the shared rollout, scene restoration, labels, statistics and layout manifest before A0. Put each adapter/state builder in an independently bound arm module. Compare identical shared-core hashes while checking each arm against its own declared adapter, preprocessing, weights, runtime and source closure. Reject undeclared changes; never ignore mismatches. A shared-core correction requires a versioned protocol and a new comparison campaign, including its control.

The core cannot import all arm modules eagerly: X-VLA's image must not need pi0.5's dependencies. Validate the plugin boundary before A0. Later adapters must pass interface/parity tests without editing the frozen core.

### R2 — High: M4a predicts two camera boxes from a one-camera representation

The inspected [LeRobot X-VLA source](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/src/lerobot/policies/xvla/modeling_xvla.py) sends view 0 plus text to `vlm_features`; remaining views are in `aux_visual_inputs`. The original mean-pooled head reads only the former. It therefore lacks direct wrist-camera information for half its targets. This must be rechecked against the installed source hashes, not assumed from a version string.

Use separate pooled view-0/language and view-1 visual representations, concatenate in a fixed order, and predict the ten targets. Assert that both views influence the auxiliary prediction, box loss reaches trainable shared visual parameters, every new parameter is in exactly one optimizer group, and action inference remains unchanged when the head is unused. These are wiring checks, not claims that the model learns useful grounding.

M4a is an auxiliary-supervision experiment, not a deployable replacement for M3's input. A detector-fed M3 policy has a separate train/test distribution problem. Nor does a failed M3 prove auxiliary supervision cannot help. M3's significance gate is retained only as the original budget heuristic, with that limitation explicit; changing it is a decision before viewing comparison results.

### R3 — High: 30 Hz recording does not provide 30 Hz force feedback

`experiments/xvla_props_p2_eval/rollout.py` observes every tick, but the policy executes ten queued actions per inference. At 30 Hz the policy replans at 3 Hz. New force samples between replans do not change the queued actions. M2 tests instantaneous force conditioning at chunk boundaries, not a reactive force controller. Raw 30 Hz sampling can also miss contact transients from the 600 Hz physics.

A null M2 result cannot rule out causal force history, filtered signals, faster replanning, or a downstream compliant controller. History is worth a later separate arm even if instantaneous force fails; do not require a positive M2 as scientific proof of its necessity. First inspect successful demonstrations and existing failed rollouts for saturation, phase confounding and signal support. Training contains no slip-recovery demonstrations, which limits what can be learned.

### R4 — High: the planned tests overstate the strength and scope of inference

384 layouts are independent evaluation units, not independent training runs. Pairing one deterministic training seed helps attribution within that seed but cannot estimate variability across training. Describe effects as conditional on these checkpoints, recipes and seed. Replicate a promising treatment and a matched control over additional training seeds before generalizing to a method.

The power table is for one unadjusted test at alpha .05. It does not establish familywise power for A1–A4 with Holm, nor power for wrong-object reduction, nor power after feasibility exclusions. Freeze a four-hypothesis family before C; report unrun arms conservatively as p=1 for the family adjustment. Treat the original power table as a sensitivity calculation, not a guarantee.

An independent exact enumeration performed during this audit gives the following planning sensitivities at N=384:

| Effect | Discordance | Power at .05 | Power at .0125 |
|---|---|---|---|
| +6 points | .19 | .743 | .540 |
| +6 points | .30 | .540 | .330 |
| +6 points | .40 | .429 | .238 |
| +10 points | .19 | .996 | .980 |
| +10 points | .30 | .943 | .851 |
| +10 points | .40 | .861 | .708 |

Calculation: sum over M ~ Binomial(N,D), then B | M ~ Binomial(M,(D+delta)/(2D)), counting B > M/2
and two-sided exact McNemar p < alpha. This is deterministic enumeration, not rerunning the original Monte Carlo script.
Alpha .0125 is the conservative first Holm threshold, not the exact joint power of the four correlated tests.
The original .05 Monte Carlo estimates are consistent with this calculation. In the planning D=.30–.40 range,
10-point effects therefore have about .71–.85 power at this conservative threshold, not .85–.95. Recalculate at the frozen feasible sample count before policy outcomes; no outcome-driven enlargement.

S excludes held-out pairs while C includes them. This is a defensible objective but must be named: select for trained-pair performance and then report a different, broader population. A0 is a historical selected checkpoint; treatment arms have a fresh six-checkpoint selection procedure. This compares selected systems, not only input information. Add the already-available 4.0-epoch treatment checkpoint as a predeclared secondary matched-phase comparison with A0, independently of the S winner, with its evaluation cost budgeted. Do not use its C results to choose either checkpoint.

Fresh seeds alone do not guarantee a new distribution. Preserve the actual geometric collision checks, name the target distribution, retain exclusions by cell, and require complete paired trials. Infrastructure crashes resume; policy nonfinite actions are outcomes, not missing data to silently drop. Once C is inspected, revisions motivated by it require a fresh confirmation set. The original fixed recipe may continue unchanged on C.

### R5 — High: resource scheduling contradicts the approved concurrency description

M0 rollouts require GPU inference; only layout generation and expert feasibility are CPU work. Eight workers at the cited 7.3 GiB peak need 58.4 GiB before headroom, exceeding the 48 GiB card. The original 1.1/2.5-hour estimates are conditional throughput arithmetic, not feasible scheduling commitments. Run CPU preparation alongside M1 only within measured CPU/RAM limits; schedule GPU evaluation separately from full training unless measured combined peaks fit. Budget video storage, image build/cache layers, asset downloads, preflight and resume-proof artifacts, final optimizer saves, and temporary atomic writes, as well as normal snapshots.

### R6 — Medium: S0's exactness and force sanity checks need narrower claims

Exact `qpos` replay establishes the recorded kinematic trajectory, not independently measured historical forces: per-frame velocity, solver state and wrench were not all sealed. The replay recipe and full initial integration state produce a reproducible reconstruction; validate the new signal against MuJoCo's reference sensor, including sign, reference point and torque translation. Preserve the existing exact replay gate.

The live `synchronize_observation` forwards derived quantities then restores integration state. Comparing its scratch data with a sensor model forwarded from the restored state can involve a different warm start. Capture/control the same pre-forward integration state for both reference paths and diagnose any mismatch; do not enlarge the tolerance.

The plan's assertion that both fingers have positive force on every frame above 2 cm is not a physical invariant: an object can remain elevated after release or briefly lose a contact. Likewise, averaging a moving grasp and a different pre-grasp pose does not isolate weight. Use controlled static, contact and no-contact fixtures for correctness; make real-trajectory distributions descriptive. MuJoCo's [sensor documentation](https://mujoco.readthedocs.io/en/3.4.0/XMLreference.html#sensor-force) specifies a child-to-parent interaction wrench in the sensor frame, not a gravity-compensated external-load estimate.

The source collector declares `phase_index` in `TRACE_KEYS` (`experiments/props_p2_data/record.py`). The plan's “trace has no phase index” is not established without inspecting the actual sealed NPZ keys. Prefer the recorded phase if present and verified; otherwise use the explicitly documented inferred segments.

### R7 — Medium: box validity is not visibility, and containment is insufficient

The proposed box ignores occlusion; `visible` really means projected/in-frame validity. Call it `in_frame` in the new schema. Test full occlusion separately and describe the privilege supplied by an amodal box. Projection needs near-plane clipping, correct compiled mesh transforms and renderer geom-group conventions. Projecting only front-facing vertices is not generally sufficient when an edge crosses the near plane.

Segmentation containment alone accepts a full-image box on every frame. Add analytic known-box fixtures, finite/ordered/range checks, out-of-frame and behind-camera cases, a full-image mutation, and tightness checks on isolated non-occluded geometry. Preserve segmentation containment as an additional check.

The claim that M3 can improve by at most 5.8–6 points is not an upper bound. Boxes also encode size and location and can help approach, alignment or carrying, or hurt performance. Wrong-object frequency motivates a hypothesis; it does not bound the treatment effect. With raw boxes and one seed, a null is evidence about this encoding and recipe, not about spatial information generally.

### R8 — Medium: M1 needs a real reference path and a numerically defined recipe

The epoch arithmetic is correct: 480 × 403 × 5 / 8 = 120,900 updates for complete 50-action windows. However, chunk 50 removes 20 observation starts per episode relative to chunk 30. Five epochs over each model's valid windows therefore does not mean identical observation exposure or identical action-target exposure. Report starts, samples and supervised action positions separately.

Effective batch is not dictated by microbatch memory: gradient accumulation can retain batch 32. Batch 8 remains a proposed controlled choice, not a hardware necessity. Matching ten executed actions is reasonable for a controlled-replanning comparison; name the deviations from the native recipe.

Both all-bf16 parameters/moments and bf16 parameters with FP32 moments can lose sub-ULP weight updates without FP32 master weights or a proven update mechanism. Fraction-changed probes are useful but do not validate optimization. Benchmark a supported numerical path against FP32 for loss, gradients and updates, plus serialization/resume. Do not promise that manually changing Adam-state dtypes works with the actual optimizer kernels. Consider supported LoRA/frozen-backbone options as explicitly different recipes if full tuning cannot fit. [OpenPI](https://github.com/Physical-Intelligence/openpi) reports substantially different requirements for LoRA and full tuning and notes backend-specific feature limits; these are reference estimates, not measurements of this LeRobot build.

Use the installed LeRobot policy and processors as the reference for a fixed batch, forward/backward/update and queued action path. Constructor/import success is not enough. The inspected [quantile normalizer](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/src/lerobot/processor/normalize_processor.py) scales without clipping; the [state tokenizer](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/src/lerobot/policies/pi05/processor_pi05.py) can produce -1 below the lower bound. Replace the blanket assertion of 0–255 tokens with parity and boundary tests, explicitly deciding any clipping as a recipe deviation. Verify this against installed hashes.

The [release dependency declaration](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/pyproject.toml) identifies a special transformers branch, but a mutable branch is not a reproducibility pin. Pin its commit, test dependencies and execute the actual offline forward/backward/inference path. If changing transformers alone cannot produce a consistent environment, stop and amend the dependency manifest before any run; retain the physics-equivalence gate.

## Research sources: what to learn and what not to infer

The Google response mixes valid ideas with imprecise citations. It should be corrected, not dismissed wholesale.

| Primary source | Verified point and relevance |
|---|---|
| [pi0.5 paper, IV-C](https://arxiv.org/html/2504.16054v1) | Includes box prediction and localization data. This is not evidence that the released LeRobot fine-tuning path exposes a ready-made auxiliary box head. |
| [GR00T N1, Appendix E](https://arxiv.org/html/2503.14734v1) | Auxiliary localization uses normalized box centers with a linear head and squared loss; describing it as tokenized full-box prediction is inaccurate. This is a closer conceptual reference for M4a. |
| [GraspVLA paper](https://arxiv.org/html/2505.03233v2) | Predicts boxes, then grasp pose, then conditions action generation on intermediate tokens. Study this for an explicit grounding-to-action path, rather than treating a discarded auxiliary head as equivalent. |
| [GraspVLA repository](https://github.com/PKU-EPIC/GraspVLA) and [playground](https://github.com/MiYanDoris/GraspVLA-playground) | Useful implementation and separate grasping benchmark references. The playground has its own LIBERO/robosuite/cuRobo stack; it is not a drop-in P2 baseline. Preserve environment and action semantics if comparing. |
| [ForceVLA](https://arxiv.org/abs/2505.22159) | Real six-axis force fusion and contact-rich data. It motivates testing useful force information, not assuming force channels alone solve P2's milk failures. |
| [TaF-VLA](https://arxiv.org/abs/2601.20321) | A tactile encoder aligned with synchronized force supervision and temporal dynamics. It is not simply a recipe for concatenating eight scalar signals. |
| [CR-VLA-Force / CC-VLA](https://arxiv.org/abs/2609.05832) | A real September 2026 source explicitly concerned with chunk latency and reactive compliance; the name differs between title and framework. Relevant to M4b's controller boundary. |
| [LAP paper](https://arxiv.org/html/2602.10556v1) and [code](https://github.com/lihzha/lap) | Represents chunk-level end-effector motion in language, including coordinate-frame conventions. Useful for later action-representation research; P2 joint targets cannot simply be renamed as Cartesian commands. |
| [2505.04769](https://arxiv.org/abs/2505.04769) | A survey, not the original pi0.5 paper. Use primary papers for architecture claims. |
| [REA](https://github.com/morluto/rea) | Binary/application/runtime reverse engineering. Source is available here; the existing graph plus direct source inspection suffices. Inspected the README; did not install or run REA. |

## Suggested learning and experimentation sequence

1. **Build the interface map first.** For X-VLA and pi0.5, trace one physical observation through image transforms, state normalization/tokenization, padding, action prediction, inverse transforms, clipping and queued execution. Record dimensions, units, coordinate frames and timestamps. This is the most useful preparation for all later changes.
2. **Complete S0 and evaluation infrastructure.** Plot example boxes and force traces across approach, close, lift and release. Measure validity, clipping and contact transients. These checks determine whether the proposed channels actually carry the information you intend.
3. **Run the approved M1/M2/M3 pilots with bounded claims.** Keep matched replanning, existing controller and fixed data. Report selected-system performance and the additional matched-phase contrast. Replicate promising effects before broad claims.
4. **Separate spatial information from its representation.** Future independent arms: numeric box versus image overlay; target-only versus all-object proposals with instruction-based selection; oracle versus predicted boxes, then dropout/jitter/occlusion robustness. A detector-fed experiment needs training inputs with realistic detector errors, not only an inference-time swap. Avoid using C to tune these choices.
5. **Separate force availability from temporal access.** Future independent arms: causal history ending at the observation timestamp; filtered/compensated wrench; or faster replanning applied to both treatment and control. Keep sensor and action clocks explicit. Any new high-rate signal storage must be versioned rather than editing a sealed S0 package.
6. **Study output representations before force control.** Joint-target versus joint-delta output can be explored through a reversible encoding and an explicitly tested decoder. Cartesian output requires a controller and reachable-target validation; forward kinematics of commanded joints is not the same as an observed next pose. LAP-inspired motion descriptions could first be auxiliary supervision, while preserving the physical action decoder. New force/stiffness control needs an appropriate expert and data, as M4b already recognizes.

Do not launch all these variants. Choose one next question from the completed pilot evidence and write a fresh spec. More information, a better representation, and a better feedback controller are distinct hypotheses.

## Verification and limits of this audit

The checkout was clean at the start; only `main` was listed by `git worktree list`. The graph reports 21 changed analyzed files since `0ce16984`; it was used for navigation, not evidence, and was not regenerated. Docker is not installed/available as a command here, so no pinned-image tests, replay checks, installed-model checks or GPU memory measurements were run. The workstation's three-episode probe is a recorded claim, not independently reproduced in this audit. Online source findings require installed-source parity before implementation. Historical result reports and experiment source remain unchanged. Exact power enumeration was run in the tool JavaScript runtime; the original Monte Carlo script was read but not executed. Documentation links, change scope and whitespace were checked separately.

Documentation verification: 18 changed/new Markdown files at review time; all 12 newly introduced local links resolve.
Two pre-existing unavailable links remain in historical status/failure text (`RTC_LOCAL_WHEELS.md` and
`outputs/train/STATUS_b3.md`); they were not introduced or rewritten by this audit. `git diff --check` passed.
