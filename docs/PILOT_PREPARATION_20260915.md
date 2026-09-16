# Balanced pilot preparation: execution record

Date: 2026-09-15. User approved the balanced-pilot specification and preparation.
The current run is **12 demonstrations only**, not the 420-demonstration pilot.
No optimizer updates, model rollouts, upload or checkpoint resume in this step.

## Implemented controls

- Canonical plans define exact successful episode slots, split/pair/count/identity
  quotas and collision-free uint32 attempt seeds. Training slots precede
  validation, preserving the first 300 as the eventual pilot training subset.
- Failed attempts retain the slot's instruction and distractor identities.
  Stop at ten failed attempts per slot; do not replace conditions or auto-resume.
- Plan mode derives count and seed from the plan, rejects conflicting benchmark
  options, uses an exclusive root and preserves timing-v2/controller semantics.
- Expert action range/finiteness/gripper checks fail rather than clip targets.
- Initial-state geometry hashes detect exact duplicates. Full state, runtime,
  source/assets, synchronized geometry and attempt outcomes are recorded.
- Generalized replay supports variable counts. It checks persisted actions
  exactly against float32-converted controls, separately from simulation drift.
- Final validation checks persisted row instructions, per-episode frame/media
  association, complete video decoding, exact plan acceptance and retry ledgers,
  split isolation and every episode's replay plus original alignment gate.
- `recording complete` and `validation passed` are separate. Training refuses
  planned data lacking a current final validation result bound to the dataset,
  scene bank, source and runtime. Failed checks do not authorize training.

## Test and review evidence before launch

44 unittest cases passed in the existing verified Docker image. The schedule
tests cover full 420-slot arithmetic, deterministic seeds, strict plan mutation
rejection and bounded retries. Integrity tests reject missing/wrong slots,
duplicate geometry, invalid commands, corrupt media and overlapping/gapped
episode video intervals. Existing timing and learning-data regressions pass.

Read-only review identified missing persisted-label and per-episode media-map
checks, an omitted alignment-script source hash, overly loose same-tick action
comparison and plan/CLI seed inconsistency. All were addressed before freezing
the source used by the run. This is not a claim that tests prove absence of all
bugs; real preparation validation remains a gate.

## Frozen plans and independent scene banks

Generated under `outputs/benchmark_preparation_20260915`:

| Plan | Demo split | Scene banks | Plan SHA-256 |
|---|---|---|---|
| preparation_plan.json | 9 train / 3 validation | 120 tuning / 240 final | c5591ab379276f3ed95e8325e197ee7e536e49e5c5c6ca136cf0fc50b9a38451 |
| pilot_plan.json | 300 train / 120 validation | 120 tuning / 240 final | 6cb795e04da110bc56343dfedef4c79ceaa683a32e22cd6b5ef0a2ae718775d1 |

Master seed: 20260915. Both scene banks generated successfully with fresh
deterministic simulator states, geometry/clearance checks and no expert/policy
success filtering. The preparation and pilot streams are disjoint. Each plan's
final validator additionally compares scene geometry with collected demonstrations.
These are state banks, not hundreds of extra expert demonstrations.

## Preparation outcome: stopped at expert-command integrity gate

- Container: `smolvla-pilot-preparation-12`.
- ID: `f51cf6d0be150c0a63deadcf5f72b451c8408986329462e4a55549a2c59f21a5`.
- Image: `smolvla:rtc-offline` (same verified local-wheel environment).
- Plan: `outputs/benchmark_preparation_20260915/preparation_plan.json`.
- Root: `datasets/franka_pilot_preparation_20260915`.
- One successful episode per ordered colour pair, rotating distractor counts.
- Prepared validation command: `validation/validate_pilot.py --root
  datasets/franka_pilot_preparation_20260915 --scene-bank
  outputs/benchmark_preparation_20260915/preparation_scenes` inside Docker.

The container exited **1**, with no OOM, after saving **3 episodes / 1,584
frames**. The fourth slot (green on red, two cubes) produced an out-of-range
scripted expert command. The manifest remains incomplete; the completion guard
was explicitly checked and correctly rejected the partial dataset. No final
replay/video/alignment PASS is claimed for this batch. It must not be used for
training or counted toward a pilot.

Saved conditions were red on green (two cubes), red on blue (three cubes),
and red on yellow (four cubes), one successful attempt each. The fatal fourth
attempt is recorded with its slot ID, seed and reason. Files and container
are retained. There was no automatic retry, clipping or condition substitution
after this integrity violation.

### Deterministic reproduction of the failure

`validation/diagnose_expert_bounds.py` reproduced only the initial approach,
without rendering, policies or dataset recording:

| Field | Value |
|---|---|
| Slot | preparation/train/0003 |
| Seed | 4281541513 |
| Control tick | 28 (zero-based) |
| Elapsed approach time | 0.933333 s |
| Joint | Panda joint 4 |
| Commanded target | -3.09553671 rad |
| Permitted actuator interval | [-3.0718, -0.0698] rad |
| Measured joint position | -2.94803032 rad |
| Computed clipped velocity | -1.00798887 rad/s |

The commanded target is approximately 0.02374 rad beyond the lower bound,
not float32 noise. The collector integrates velocity into `q_target` without
position bounds; the differential IK clips velocity but does not constrain
joint position. A valid measured position and velocity cap therefore do not
guarantee valid commanded targets. This is a scripted-expert/control issue,
not a learned-policy failure and not evidence that the new range check is wrong.

The new preparation guard exposed a problem not exercised by the earlier
six-episode seeds. Do not bypass it by clipping saved labels or discarding
the hard condition silently. A joint-limit-aware expert correction requires
an explicit design change and a new preparation run; no such correction or
run was attempted here.

## Independently verified scene-bank deliverables

Both banks were fully reloaded and checked against the corresponding plans,
source/runtime provenance, timing, each state-file hash, exact qpos roundtrip,
geometry/clearance and duplicate-state rejection. Each contains 120 tuning
and 240 final scenes, with no success filtering.

| Bank | Content hash |
|---|---|
| preparation_scenes | eb56a673a96a74c9deb61a3470bd2741f8229637aed2d8a38db686c2a8e0d113 |
| pilot_scenes | 81495ecca740de63ca03d1eef3e73f05b98a955e7c9c0d881f572dda1bcb6cc0 |

These bank checks do not replace the unexecuted complete preparation validator
or its cross-check against a full accepted demonstration set. The pilot plan
is structurally tested, but no pilot demonstrations were collected.

## Next decision

Hold the pilot launch. Review a joint-limit-aware expert integration/planning
correction, test the failing seed and preserve command-versus-recorded-action
equality. Then regenerate provenance-bound plans/banks as needed and rerun a
fresh 12-slot preparation gate. No controller changes, training or further
collection are automatically authorized by this report.

## Preservation and remaining boundaries

The prior timing-v2 source/assets were archived after matching their recorded
hashes at `outputs/validation/timing_v2_before_pilot.zip`. Existing datasets and
checkpoint files are not overwritten. New schedules, banks and datasets remain
Git-ignored; code and reports remain on `feat/reusable-validation`.

The pi-family adapters and a final model-training recipe are separate gates.
The 420-demonstration pilot is not launched by this preparation command. Report
the real small-batch results before the launch decision. Dataset integrity can
be verified; successful learned task performance cannot be guaranteed.