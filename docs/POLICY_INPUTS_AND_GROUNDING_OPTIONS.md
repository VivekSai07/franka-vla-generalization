# Policy inputs, language grounding, and possible improvements

Date: 2026-09-17. Purpose: knowledge note and research options, **not an
approved model/dataset change or a claim that any option will solve stacking**.

## 1. The question to answer first

Does the robot lack information, or fail to use information already available?
Our current policy receives two RGB camera images, nine measured joint/finger
positions, and the natural-language instruction. Images contain cube appearance
and location; language specifies pick and destination roles. The
[instruction audit](INSTRUCTION_AUDIT_20260917.md) confirms wrong-pair behavior
in some rollouts but has not established its underlying learning mechanism.

Adding inputs changes the policy interface, preprocessing, statistics and often
architecture. Appending numbers to the dataset does not automatically cause the
model to use them. Inputs must also exist at inference time, with consistent
units, timestamps and coordinate frames.

## 2. Recent observation and action history

**Possible information:** previous images, joint positions and previously issued
commands, with explicit timestamps or a fixed sampling interval.

**Benefit:** distinguishes motion direction, approach from retreat, an object
moving with the gripper from a stationary object, and a stable grasp from slip.
A single image/state describes where things are more readily than what is
happening. Temporal context may help recovery after imperfect actions.

**Limits:** history alone does not explain which color the instruction selects.
Past actions can reinforce motion-continuation shortcuts if not balanced by
visual and language grounding. Longer histories increase memory/compute and
require a compatible temporal model, rather than an arbitrary concatenation.

## 3. Joint velocities and gripper feedback

**Possible information:** seven joint velocities, finger velocities, measured
gripper force, motor current or contact estimates available from hardware.

**Benefit:** helps distinguish an empty closure from fingers stopped by an
object, detect slipping, and identify disagreement between commanded and
achieved motion. Velocities can come from telemetry or filtered differences;
force/current availability depends on the robot.

**Limits:** these signals help execution, not necessarily correct object
selection. A closed command is not proof of a grasp. Filter delay, noise,
sampling cadence and hardware-specific units matter. Simulator contact labels
are not equivalent to real force sensors or deployment-available contact estimates.

## 4. Explicit visual object information

**Possible information:** object color/class, bounding boxes, segmentation masks,
visibility/confidence, depth and estimated 3-D object pose.

**Benefit:** a perception module can supply an explicit object list instead of
requiring the action policy to infer all identities and geometry from pixels.
Depth or calibrated multi-view reconstruction may improve approach positioning.

Example conceptual representation:

| Object | Observed location | Visibility |
|---|---|---|
| Green cube | Left side of workspace | Clear |
| Red cube | Near front edge | Clear |
| Blue cube | Behind gripper | Partial |

**Limits:** detections can be wrong, absent or occluded. Confidence and missing
values need explicit handling. Perfect simulator poses during training followed
by noisy estimates at inference create a distribution mismatch. A perception
pipeline has its own calibration and failure modes.

## 5. Instruction-conditioned object roles

**Possible information:** a module maps “put green on red” to pick=green,
destination=red, other objects=distractors. It then associates these roles with
detected objects, supplying target-specific masks, highlighted regions or
structured target features.

**Benefit:** makes the language-to-object association explicit. This is more
directly related to wrong-color selection than supplying more joint telemetry.
Separating pick and destination roles is important: recognizing two colors does
not establish their correct order.

**Limits:** a color ID appended to joint state is not enough by itself; the
policy still needs to link that identifier to the visible cube. Target masks
or locations make the link more explicit but require a reliable role-grounding
module. Incorrect grounding can consistently direct an otherwise capable
controller to the wrong object. This changes the overall system, not just a
hyperparameter of the existing VLA.

## 6. Relative geometry

**Possible information:** gripper pose relative to the selected pick object,
pick-object pose relative to the destination, and desired grasp orientation.

For positions in the same calibrated frame:

$$\Delta p_{\text{pick}} = p_{\text{pick object}} - p_{\text{gripper}}.$$

**Benefit:** relative geometry can simplify learning how to reach a target
across layouts, compared with jointly inferring geometry from pixels and absolute
joint angles. It can also support placement and approach alignment.

**Limits:** the correct target must already have been selected. Pose estimates,
coordinate conventions, orientation representation and calibration must agree.
Providing target-relative observations does not require silently replacing our
absolute-joint action contract; any action-space change is a separate experiment.

## 7. Task-progress and grasp-status information

**Possible information:** estimated approach/grasp/lift/transfer/release/retreat
phase, object-in-gripper probability or evidence that the previous phase failed.

**Benefit:** after closing on empty space, a robot should retry or recover rather
than proceed as though it were carrying the object. Explicit phase estimates
can help a hierarchical policy choose the appropriate behavior.

**Limits:** phase must be inferred from deployment-available observations. The
scripted expert's internal phase or elapsed demonstration frame cannot simply
be supplied at training time and omitted during deployment. A phase classifier
can itself make mistakes; elapsed time alone is a weak substitute for progress.

## 8. Match information to failure mode

| Observed failure | Potentially relevant information |
|---|---|
| Picks wrong color | Instruction-conditioned identity or target mask |
| Reaches beside correct cube | Depth, object pose, relative target geometry |
| Closes on empty space | Gripper feedback and temporal visual observations |
| Drops grasped cube | Force/tactile feedback and history |
| Continues after missed grasp | Estimated grasp status and task progress |

These are hypotheses about usefulness, not demonstrated corrections for this run.

## 9. Three distinct experiment categories

### Deployment-available additional inputs

Use camera-derived detections, telemetry velocities or measured gripper feedback
in both training and deployment. This can be a practical system enhancement,
but changes its sensing/perception assumptions and must be reported explicitly.

### Training-only auxiliary supervision

Keep inference inputs as images/state/instruction, but use training labels for
object localization, color-role prediction or grasp status in auxiliary objectives.
Simulator truth may be useful supervision without becoming a privileged policy
input. This requires objective/head design, train-only labels, loss weighting
and evaluation of whether grounding improves without damaging control.

### Oracle diagnostics

Temporarily supply perfect target identity/pose to test whether perception and
grounding are a bottleneck. A successful oracle-guided controller would be a
diagnostic upper bound for that system, not a fair raw-image VLA benchmark score.
It also would not prove that a deployable perception module is accurate enough.

## 10. Preserve the shared benchmark

Our canonical recordings remain immutable: nine state values, eight absolute
joint/gripper commands, two 512x512 RGB cameras at 30 Hz. Privileged simulator
state stays in replay/evaluation metadata, not policy input. Raw demonstrations
can support separate versioned adapters without rewriting the canonical data.

- Fit statistics, learned perception/tokenization and auxiliary objectives only
  using the designated training data; never use final-scene outcomes for tuning.
- Keep 300 training / 120 validation demonstration IDs explicit.
- Keep the 120 tuning and 240 final scene banks distinct.
- Label oracle-assisted, additional-sensor and raw-image-only results separately.
- Compare models under the same input availability, controller, scoring and
  compute-budget accounting; raw data reuse is not proof of adapter compatibility.
- Preserve noisy/raw predictions and record interventions. Do not count any
  visually plausible stack as success unless the requested ordered pair meets
  the released-stack stability criterion.

## 11. Why low offline error was not sufficient

Expert observations may already reveal the current movement and target through
arm posture, camera view or grasp state. A policy can predict locally plausible
continuations without reliably choosing the instructed target from initialization.
Closed-loop execution additionally exposes states caused by its own errors.

Our controlled fixed-observation probes showed weak immediate instruction
sensitivity on sampled states, but some longer chunks changed appreciably.
That does not establish total language neglect. Reversing instructions midway
through a grasp also creates unusual states. Initial-state paired rollouts are
a more direct next test of instructed object selection.

## 12. Agreed next sequence and proposed bounded comparison

The user approved the sequence: **finish the baseline unchanged → compare
same-scene swapped instructions with matched seeds → choose a targeted correction
from evidence**. No input augmentation or retraining is authorized by this note.

At the documentation snapshot, the baseline is still running with 49/120 trials
complete and one correct instructed success. This is partial and pair-ordered,
not the final benchmark result. Do not edit its bound sources or start a duplicate.

Proposed exact follow-up for review:

1. Select the lowest-ordinal **two-cube** tuning scene for each of the 12 ordered
   instruction pairs, regardless of outcome. Reversing roles then introduces no
   distractor-choice ambiguity. Record all selected IDs before any new rollout.
2. Run three conditions from each identical saved initial state: original
   instruction with seed S, reversed instruction with the same S, original
   instruction with alternate seed S+1. Use S=20260916+original scene ordinal,
   matching the baseline; use local RNG reset for each independent trial.
3. Budget **36 policy rollouts**, no optimizer steps or checkpoint changes.
   Use the selected 020000 model, saved processors, 40-second timeout,
   50-action predictions and ten-action execution horizon unchanged.
4. First reproduce the original-condition baseline action/state trace. Stop
   on unexplained mismatch; do not interpret a broken reproduction as language
   sensitivity. Common seeds align randomness by prediction-call index, not
   necessarily by semantic phase after trajectories diverge.
5. Record actual first grasped color, all-cube contact/lift, terminal ordered
   stack pair, correct-instruction success and interventions for each condition.
   Recompute requested roles for reversed trials; never score them with the
   original instruction. Save paired videos and all raw traces.
6. Compare instruction-induced object-selection changes with seed-induced
   variation. Keep each scene as a paired unit; report denominators and failures,
   not only successful examples. This is a small diagnostic, not a new final score.

If roles reverse reliably but execution fails, focus on reaching/grasping/recovery.
If object choice barely responds to language, investigate instruction-dependent
supervision or explicit role grounding. If seed variation dominates, broaden
matched-seed evidence before declaring a cause. Expert reachability comparisons
can help distinguish difficult scenes from policy failure in a separately bounded
follow-up. Do not introduce multiple corrections simultaneously.

**Review boundary:** the sequence is approved; the exact 12-scene/36-rollout
budget above is proposed for confirmation before implementing or launching it.
Finish the original baseline first. Keep final scenes sealed and do not silently
change success criteria, dataset labels, execution horizon or training budget.