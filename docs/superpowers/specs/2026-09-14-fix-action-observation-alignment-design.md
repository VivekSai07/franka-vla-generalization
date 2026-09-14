# Fix action/observation timestep alignment in data collectors

**Status:** Approved (bounded design, agreed in chat 2026-09-14)

## Problem

All three data collection scripts (`data_collection/collect_libero_dataset.py`,
`data_collection/collect_v2_dataset.py`, `data_collection/collect_stacking_dataset.py`)
pack each frame's `observation.state` and `action` **before** computing that
tick's IK/trajectory update. At the point the frame is packed, `q_target`
still holds the value computed on the *previous* tick (the value that
produced the just-read `q_current`/`obs_state`). So every saved frame is
really `(state_t, action_{t-1})` instead of `(state_t, action_t)`.

This was confirmed against `lerobot`'s own `record_loop()`
(`src/lerobot/scripts/lerobot_record.py`), which does, every tick:
`get_observation() → get_action() → send_action() → add_frame()` — the
action saved is always the one just computed and sent *this* tick, paired
with the observation read at the top of the same tick.

Effect: with a 30 Hz control loop and a smooth cubic IK trajectory, the
per-tick change in commanded joint position is tiny, so `action_{t-1}` is
almost indistinguishable from `state_t`. The model is trained on a
near-identity state→action mapping instead of "what to do next," which is
consistent with the observed inference failure (policy barely moves, never
commits to a pick).

## Fix

In each collector's per-tick loop, reorder so the IK/trajectory block (which
computes `dq`, updates `q_target`, and writes `d.ctrl`) runs **before**
`act_state` is built and the frame is appended to `episode_buffer`. Build
`act_state` from the freshly updated `q_target` — the same value just
written to `d.ctrl`, i.e. the action actually sent this tick — so the saved
pair becomes `(state_t, action_t)`.

`obs_state` construction (reading `q_current`/`d.qpos`) stays exactly where
it is — that part was already correct. Camera rendering order, gripper
command/grasp-check, state-transition logic, and physics stepping are
unchanged in content, only shifted later in the tick (they were already
downstream of the original frame-packing point in the source, and remain
downstream of it here).

No feature/schema changes: `observation.state`, `action`, and the camera
keys are byte-for-byte identical in shape/dtype/names before and after.
Training, inference (`inference/policy_server_libero.py`,
`inference/franka_inference_libero.py`), and `huggingface/push_dataset_to_hub.py`
require no changes.

## Scope

Fix applies to all three collector scripts (confirmed: all three have the
identical bug via the identical pattern). Re-collecting the dataset,
retraining, and touching inference scripts are explicitly out of scope —
follow-on work once this lands.

## Verification

No existing test suite covers these MuJoCo/hardware-adjacent scripts.
Verification is a new standalone script,
`data_collection/verify_action_alignment.py`, that loads a just-collected
episode via `LeRobotDataset` and checks, for consecutive frames `t, t+1`:

```
mean(|action_t[:7] - state_{t+1}[:7]|) < mean(|action_t[:7] - state_t[:7]|)
```

i.e. the recorded action for frame `t` should predict where the arm ends up
*next*, not where it already was. This is run against a small (2-episode)
headless collection from the fixed `collect_libero_dataset.py` as the
concrete pass/fail check for the fix.
