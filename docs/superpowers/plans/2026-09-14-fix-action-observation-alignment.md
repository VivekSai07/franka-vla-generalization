# Fix Action/Observation Timestep Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three Franka data-collection scripts so each saved frame's `action` is the joint target actually computed and sent *this* control tick (matching `lerobot_record.py`'s `observe → act → send → save` ordering), instead of the stale target left over from the previous tick.

**Architecture:** In each collector's per-tick `while current_state_idx < len(states)` loop, move the IK/trajectory block (which computes `dq`, updates `q_target`, and writes `d.ctrl`) to run *before* `act_state` is built and appended to `episode_buffer`. `obs_state` (built from `q_current`, read at the top of the tick) does not move. No dataset schema, training, or inference code changes — this only reorders when each in-memory value is read within an unchanged control loop.

**Tech Stack:** Python, MuJoCo, Pinocchio (`pin`), `lerobot.datasets.lerobot_dataset.LeRobotDataset`. No test framework in this repo — verification is a standalone runtime script plus manual headless smoke runs.

**Spec:** `docs/superpowers/specs/2026-09-14-fix-action-observation-alignment-design.md`

## Global Constraints

- No changes to `observation.state`, `action`, or camera feature keys/shapes/dtypes in any collector — training/inference code must not need changes.
- No changes to physics stepping, gripper control, state-transition, or success-verification logic — only *when* within the tick these run relative to frame-packing.
- Fix applies to all three collectors: `data_collection/collect_libero_dataset.py`, `data_collection/collect_v2_dataset.py`, `data_collection/collect_stacking_dataset.py`.
- Re-collecting the dataset and retraining are out of scope for this plan.

---

### Task 1: Reorder `collect_libero_dataset.py`'s per-tick loop

**Files:**
- Modify: `data_collection/collect_libero_dataset.py:252-316`

**Interfaces:**
- Consumes: nothing new — uses existing `robot`, `ik`, `q_target`, `ACTIVE_JOINTS`, `dt_control`, `q_home_pin`, `fixed_rotation`, `HOME_POS`, `pick_color`, `place_color`, `get_cube_body_name`, `TaskSpaceTrajectory`, `grasp_sys`, all already in scope in `main()`.
- Produces: `episode_buffer` frames where `action`'s first 7 values equal the `q_target` written to `d.ctrl` in that same tick (previously they equaled the *previous* tick's `q_target`). No new functions or files — this task changes control flow within `main()` only.

- [ ] **Step 1: Locate and replace the tick body**

Open `data_collection/collect_libero_dataset.py`. Find this block (currently lines 252-316):

```python
        while current_state_idx < len(states):
            step_start = time.time()

            # --- Observations ---
            renderer.update_scene(d, camera="front_cam")
            img_agent = renderer.render()
            renderer.update_scene(d, camera="gripper_cam")
            img_gripper = renderer.render()

            q_current = d.qpos[:robot.model.nq].copy()
            obs_state = np.zeros(9, dtype=np.float32)
            act_state = np.zeros(8, dtype=np.float32)
            obs_state[:7] = q_current[:7]
            obs_state[7] = q_current[7]
            obs_state[8] = q_current[8]
            act_state[:7] = q_target[:7]
            act_state[7] = float(states[current_state_idx]["gripper"])

            episode_buffer.append({
                "observation.images.image":  img_agent,
                "observation.images.image2": img_gripper,
                "observation.state": obs_state,
                "action": act_state,
                "task": instruction,
            })

            # --- Controller ---
            state = states[current_state_idx]

            if current_trajectory is None:
                if state["phase"] == "pick":
                    body_id = m.body(get_cube_body_name(pick_color)).id
                    base_pos = d.xpos[body_id].copy()
                elif state["phase"] == "place":
                    body_id = m.body(get_cube_body_name(place_color)).id
                    base_pos = d.xpos[body_id].copy()
                else:
                    base_pos = HOME_POS - state["offset"]

                target_pos = base_pos + state["offset"]

                # Approach jitter: ±5 mm noise on jitter-enabled states.
                # Prevents the model from memorising a single canonical path.
                if state["jitter"]:
                    target_pos += np.random.normal(0, 0.005, 3)

                start_se3 = robot.forward_kinematics(q_current)
                end_se3 = pin.SE3(fixed_rotation, target_pos)
                current_trajectory = TaskSpaceTrajectory(
                    start_pose=start_se3, end_pose=end_se3,
                    duration=state["duration"], method="cubic"
                )

            t_state = d.time - state_start_time
            target_se3 = current_trajectory.get_pose(t_state)

            dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
            dq = np.clip(dq, -2.0, 2.0)
            for idx in ACTIVE_JOINTS:
                q_target[idx] += dq[idx] * dt_control
                d.ctrl[idx] = q_target[idx]

            grasp_sys.command(state["gripper"])
            is_holding = grasp_sys.is_grasped()
```

Replace it with (controller block moved before frame packing; `act_state` now built from the freshly-updated `q_target`):

```python
        while current_state_idx < len(states):
            step_start = time.time()

            # --- Observations ---
            renderer.update_scene(d, camera="front_cam")
            img_agent = renderer.render()
            renderer.update_scene(d, camera="gripper_cam")
            img_gripper = renderer.render()

            q_current = d.qpos[:robot.model.nq].copy()
            obs_state = np.zeros(9, dtype=np.float32)
            obs_state[:7] = q_current[:7]
            obs_state[7] = q_current[7]
            obs_state[8] = q_current[8]

            # --- Controller: compute and send this tick's action BEFORE
            # recording it, so the saved action matches what was actually
            # sent (mirrors lerobot_record.py's observe -> act -> send -> save).
            state = states[current_state_idx]

            if current_trajectory is None:
                if state["phase"] == "pick":
                    body_id = m.body(get_cube_body_name(pick_color)).id
                    base_pos = d.xpos[body_id].copy()
                elif state["phase"] == "place":
                    body_id = m.body(get_cube_body_name(place_color)).id
                    base_pos = d.xpos[body_id].copy()
                else:
                    base_pos = HOME_POS - state["offset"]

                target_pos = base_pos + state["offset"]

                # Approach jitter: ±5 mm noise on jitter-enabled states.
                # Prevents the model from memorising a single canonical path.
                if state["jitter"]:
                    target_pos += np.random.normal(0, 0.005, 3)

                start_se3 = robot.forward_kinematics(q_current)
                end_se3 = pin.SE3(fixed_rotation, target_pos)
                current_trajectory = TaskSpaceTrajectory(
                    start_pose=start_se3, end_pose=end_se3,
                    duration=state["duration"], method="cubic"
                )

            t_state = d.time - state_start_time
            target_se3 = current_trajectory.get_pose(t_state)

            dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
            dq = np.clip(dq, -2.0, 2.0)
            for idx in ACTIVE_JOINTS:
                q_target[idx] += dq[idx] * dt_control
                d.ctrl[idx] = q_target[idx]

            # --- Record the action that was just computed and sent above ---
            act_state = np.zeros(8, dtype=np.float32)
            act_state[:7] = q_target[:7]
            act_state[7] = float(state["gripper"])

            episode_buffer.append({
                "observation.images.image":  img_agent,
                "observation.images.image2": img_gripper,
                "observation.state": obs_state,
                "action": act_state,
                "task": instruction,
            })

            grasp_sys.command(state["gripper"])
            is_holding = grasp_sys.is_grasped()
```

Everything from `# --- State transitions ---` onward (the `if t_state > state["duration"]:` block and the physics-step section) stays exactly as-is, unmoved, immediately following this block.

- [ ] **Step 2: Static sanity check — confirm the reorder didn't drop or duplicate any statement**

Run:
```bash
python -c "import ast; ast.parse(open('data_collection/collect_libero_dataset.py').read())"
```
Expected: no output (file parses as valid Python — confirms no syntax break from the edit).

- [ ] **Step 3: Headless smoke run — 1 episode, must complete without crashing**

```bash
docker compose run --rm collect python data_collection/collect_libero_dataset.py --episodes 1 --headless
```
Expected: the episode either succeeds (prints `SUCCESS (xy=...)`) or fails and retries (prints `FAILURE (...) — discarding.` and starts another attempt) — either way, no Python traceback, and it eventually prints `Data collection complete: 1 episodes saved.` Let it run until that line (it may retry a few times; a single episode collects in well under a minute).

- [ ] **Step 4: Commit**

```bash
git add data_collection/collect_libero_dataset.py
git commit -m "fix: record actual sent action instead of stale previous-tick target in collect_libero_dataset.py"
```

---

### Task 2: Reorder `collect_v2_dataset.py`'s per-tick loop

**Files:**
- Modify: `data_collection/collect_v2_dataset.py:189-262`

**Interfaces:**
- Consumes: existing `robot`, `ik`, `q_target`, `ACTIVE_JOINTS`, `dt_control`, `q_home_pin`, `fixed_rotation`, `HOME_POS`, `TaskSpaceTrajectory`, `grasp_sys`, all already in scope in `main()`.
- Produces: same alignment fix as Task 1, applied to this collector's frame dict (no `"task"` key here — `INSTRUCTION` is passed separately to `dataset.add_frame(frame, task=INSTRUCTION)` at save time, unchanged).

- [ ] **Step 1: Locate and replace the tick body**

Open `data_collection/collect_v2_dataset.py`. Find this block (currently lines 189-262):

```python
        while current_state_idx < len(states):
            step_start = time.time()

            # ==========================================
            # 1. RENDER & LOG EXISTING STATE
            # ==========================================
            renderer.update_scene(d, camera="top_cam")
            img_top = renderer.render()
            
            renderer.update_scene(d, camera="front_cam")
            img_front = renderer.render()
            
            renderer.update_scene(d, camera="gripper_cam")
            img_gripper = renderer.render()
            
            # Pack observation and action (before taking the step)
            q_current = d.qpos[:robot.model.nq].copy()
            
            obs_state = np.zeros(9, dtype=np.float32)
            act_state = np.zeros(8, dtype=np.float32)
            
            # observation.state requires 7 arm joints + 2 finger joints
            obs_state[:7] = q_current[:7]
            obs_state[7] = q_current[7]  # finger1.pos
            obs_state[8] = q_current[8]  # finger2.pos
            
            # action requires 7 arm targets + 1 gripper command
            act_state[:7] = q_target[:7]
            act_state[7] = states[current_state_idx]["gripper"] if current_state_idx < len(states) else 0.0
            
            episode_buffer.append({
                "observation.images.up": img_top,
                "observation.images.wrist": img_gripper,
                "observation.images.front": img_front,
                "observation.state": obs_state,
                "action": act_state
            })
            
            # ==========================================
            # 2. RUN CONTROLLERS
            # ==========================================
            state = states[current_state_idx]

            if current_trajectory is None:
                if state["target"] == "red":
                    target_id = m.body("target_cube_red").id
                    base_pos = d.xpos[target_id].copy()
                elif state["target"] == "blue":
                    target_id = m.body("target_cube_blue").id
                    base_pos = d.xpos[target_id].copy()
                else:
                    base_pos = HOME_POS - state["offset"]
                    
                target_pos = base_pos + state["offset"]

                start_se3 = robot.forward_kinematics(q_current)
                end_se3 = pin.SE3(fixed_rotation, target_pos)
                current_trajectory = TaskSpaceTrajectory(
                    start_pose=start_se3, end_pose=end_se3, 
                    duration=state["duration"], method="cubic"
                )

            t_state = d.time - state_start_time
            target_se3 = current_trajectory.get_pose(t_state)
            
            dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
            dq = np.clip(dq, -2.0, 2.0)
            
            for idx in ACTIVE_JOINTS:
                q_target[idx] += dq[idx] * dt_control
                d.ctrl[idx] = q_target[idx]

            grasp_sys.command(state["gripper"])
            is_holding = grasp_sys.is_grasped()
```

Replace it with:

```python
        while current_state_idx < len(states):
            step_start = time.time()

            # ==========================================
            # 1. RENDER & READ CURRENT STATE
            # ==========================================
            renderer.update_scene(d, camera="top_cam")
            img_top = renderer.render()
            
            renderer.update_scene(d, camera="front_cam")
            img_front = renderer.render()
            
            renderer.update_scene(d, camera="gripper_cam")
            img_gripper = renderer.render()
            
            q_current = d.qpos[:robot.model.nq].copy()
            
            obs_state = np.zeros(9, dtype=np.float32)
            
            # observation.state requires 7 arm joints + 2 finger joints
            obs_state[:7] = q_current[:7]
            obs_state[7] = q_current[7]  # finger1.pos
            obs_state[8] = q_current[8]  # finger2.pos

            # ==========================================
            # 2. RUN CONTROLLERS — compute and send this tick's action
            #    BEFORE recording it, so the saved action matches what
            #    was actually sent.
            # ==========================================
            state = states[current_state_idx]

            if current_trajectory is None:
                if state["target"] == "red":
                    target_id = m.body("target_cube_red").id
                    base_pos = d.xpos[target_id].copy()
                elif state["target"] == "blue":
                    target_id = m.body("target_cube_blue").id
                    base_pos = d.xpos[target_id].copy()
                else:
                    base_pos = HOME_POS - state["offset"]
                    
                target_pos = base_pos + state["offset"]

                start_se3 = robot.forward_kinematics(q_current)
                end_se3 = pin.SE3(fixed_rotation, target_pos)
                current_trajectory = TaskSpaceTrajectory(
                    start_pose=start_se3, end_pose=end_se3, 
                    duration=state["duration"], method="cubic"
                )

            t_state = d.time - state_start_time
            target_se3 = current_trajectory.get_pose(t_state)
            
            dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
            dq = np.clip(dq, -2.0, 2.0)
            
            for idx in ACTIVE_JOINTS:
                q_target[idx] += dq[idx] * dt_control
                d.ctrl[idx] = q_target[idx]

            # ==========================================
            # 3. RECORD the action that was just computed and sent above
            # ==========================================
            act_state = np.zeros(8, dtype=np.float32)
            act_state[:7] = q_target[:7]
            act_state[7] = state["gripper"]

            episode_buffer.append({
                "observation.images.up": img_top,
                "observation.images.wrist": img_gripper,
                "observation.images.front": img_front,
                "observation.state": obs_state,
                "action": act_state
            })

            grasp_sys.command(state["gripper"])
            is_holding = grasp_sys.is_grasped()
```

Note: the original `act_state[7] = states[current_state_idx]["gripper"] if current_state_idx < len(states) else 0.0` guard is dead code — the enclosing `while current_state_idx < len(states):` already guarantees the condition, so the replacement simplifies it to `state["gripper"]` directly (same value, `state` is already `states[current_state_idx]`).

Everything from `# State Transitions` onward stays exactly as-is, unmoved.

- [ ] **Step 2: Static sanity check**

```bash
python -c "import ast; ast.parse(open('data_collection/collect_v2_dataset.py').read())"
```
Expected: no output.

- [ ] **Step 3: Headless smoke run — 1 episode**

```bash
docker compose run --rm collect python data_collection/collect_v2_dataset.py --episodes 1 --headless
```
Expected: no traceback; eventually prints `✅ Data Collection Complete!` after 1 episode is saved (may retry on grasp failure first).

- [ ] **Step 4: Commit**

```bash
git add data_collection/collect_v2_dataset.py
git commit -m "fix: record actual sent action instead of stale previous-tick target in collect_v2_dataset.py"
```

---

### Task 3: Reorder `collect_stacking_dataset.py`'s per-tick loop

**Files:**
- Modify: `data_collection/collect_stacking_dataset.py:189-263`

**Interfaces:**
- Consumes: existing `robot`, `ik`, `q_target`, `ACTIVE_JOINTS`, `dt_control`, `q_home_pin`, `fixed_rotation`, `HOME_POS`, `TaskSpaceTrajectory`, `grasp_sys`, `INSTRUCTION`, all already in scope in `main()`.
- Produces: same alignment fix as Tasks 1-2, applied to this collector's frame dict (this file keeps `"task": INSTRUCTION` inline in the frame dict, unlike `collect_v2_dataset.py`).

- [ ] **Step 1: Locate and replace the tick body**

Open `data_collection/collect_stacking_dataset.py`. Find this block (currently lines 189-263):

```python
        while current_state_idx < len(states):
            step_start = time.time()

            # ==========================================
            # 1. RENDER & LOG EXISTING STATE
            # ==========================================
            renderer.update_scene(d, camera="top_cam")
            img_top = renderer.render()
            
            renderer.update_scene(d, camera="front_cam")
            img_front = renderer.render()
            
            renderer.update_scene(d, camera="gripper_cam")
            img_gripper = renderer.render()
            
            # Pack observation and action (before taking the step)
            q_current = d.qpos[:robot.model.nq].copy()
            
            obs_state = np.zeros(9, dtype=np.float32)
            act_state = np.zeros(8, dtype=np.float32)
            
            # observation.state requires 7 arm joints + 2 finger joints
            obs_state[:7] = q_current[:7]
            obs_state[7] = q_current[7]  # finger1.pos
            obs_state[8] = q_current[8]  # finger2.pos
            
            # action requires 7 arm targets + 1 gripper command
            act_state[:7] = q_target[:7]
            act_state[7] = states[current_state_idx]["gripper"] if current_state_idx < len(states) else 0.0
            
            episode_buffer.append({
                "observation.images.up": img_top,
                "observation.images.wrist": img_gripper,
                "observation.images.front": img_front,
                "observation.state": obs_state,
                "action": act_state,
                "task": INSTRUCTION
            })
            
            # ==========================================
            # 2. RUN CONTROLLERS
            # ==========================================
            state = states[current_state_idx]

            if current_trajectory is None:
                if state["target"] == "red":
                    target_id = m.body("target_cube_red").id
                    base_pos = d.xpos[target_id].copy()
                elif state["target"] == "blue":
                    target_id = m.body("target_cube_blue").id
                    base_pos = d.xpos[target_id].copy()
                else:
                    base_pos = HOME_POS - state["offset"]
                    
                target_pos = base_pos + state["offset"]

                start_se3 = robot.forward_kinematics(q_current)
                end_se3 = pin.SE3(fixed_rotation, target_pos)
                current_trajectory = TaskSpaceTrajectory(
                    start_pose=start_se3, end_pose=end_se3, 
                    duration=state["duration"], method="cubic"
                )

            t_state = d.time - state_start_time
            target_se3 = current_trajectory.get_pose(t_state)
            
            dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
            dq = np.clip(dq, -2.0, 2.0)
            
            for idx in ACTIVE_JOINTS:
                q_target[idx] += dq[idx] * dt_control
                d.ctrl[idx] = q_target[idx]

            grasp_sys.command(state["gripper"])
            is_holding = grasp_sys.is_grasped()
```

Replace it with:

```python
        while current_state_idx < len(states):
            step_start = time.time()

            # ==========================================
            # 1. RENDER & READ CURRENT STATE
            # ==========================================
            renderer.update_scene(d, camera="top_cam")
            img_top = renderer.render()
            
            renderer.update_scene(d, camera="front_cam")
            img_front = renderer.render()
            
            renderer.update_scene(d, camera="gripper_cam")
            img_gripper = renderer.render()
            
            q_current = d.qpos[:robot.model.nq].copy()
            
            obs_state = np.zeros(9, dtype=np.float32)
            
            # observation.state requires 7 arm joints + 2 finger joints
            obs_state[:7] = q_current[:7]
            obs_state[7] = q_current[7]  # finger1.pos
            obs_state[8] = q_current[8]  # finger2.pos

            # ==========================================
            # 2. RUN CONTROLLERS — compute and send this tick's action
            #    BEFORE recording it, so the saved action matches what
            #    was actually sent.
            # ==========================================
            state = states[current_state_idx]

            if current_trajectory is None:
                if state["target"] == "red":
                    target_id = m.body("target_cube_red").id
                    base_pos = d.xpos[target_id].copy()
                elif state["target"] == "blue":
                    target_id = m.body("target_cube_blue").id
                    base_pos = d.xpos[target_id].copy()
                else:
                    base_pos = HOME_POS - state["offset"]
                    
                target_pos = base_pos + state["offset"]

                start_se3 = robot.forward_kinematics(q_current)
                end_se3 = pin.SE3(fixed_rotation, target_pos)
                current_trajectory = TaskSpaceTrajectory(
                    start_pose=start_se3, end_pose=end_se3, 
                    duration=state["duration"], method="cubic"
                )

            t_state = d.time - state_start_time
            target_se3 = current_trajectory.get_pose(t_state)
            
            dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
            dq = np.clip(dq, -2.0, 2.0)
            
            for idx in ACTIVE_JOINTS:
                q_target[idx] += dq[idx] * dt_control
                d.ctrl[idx] = q_target[idx]

            # ==========================================
            # 3. RECORD the action that was just computed and sent above
            # ==========================================
            act_state = np.zeros(8, dtype=np.float32)
            act_state[:7] = q_target[:7]
            act_state[7] = state["gripper"]

            episode_buffer.append({
                "observation.images.up": img_top,
                "observation.images.wrist": img_gripper,
                "observation.images.front": img_front,
                "observation.state": obs_state,
                "action": act_state,
                "task": INSTRUCTION
            })

            grasp_sys.command(state["gripper"])
            is_holding = grasp_sys.is_grasped()
```

Everything from `# State Transitions` onward stays exactly as-is, unmoved.

- [ ] **Step 2: Static sanity check**

```bash
python -c "import ast; ast.parse(open('data_collection/collect_stacking_dataset.py').read())"
```
Expected: no output.

- [ ] **Step 3: Headless smoke run — 1 episode**

```bash
docker compose run --rm collect python data_collection/collect_stacking_dataset.py --episodes 1 --headless
```
Expected: no traceback; eventually prints `✅ Data Collection Complete! Finalizing Dataset...` after 1 episode is saved.

- [ ] **Step 4: Commit**

```bash
git add data_collection/collect_stacking_dataset.py
git commit -m "fix: record actual sent action instead of stale previous-tick target in collect_stacking_dataset.py"
```

---

### Task 4: Verification script — prove the alignment fix actually changed the data

**Files:**
- Create: `data_collection/verify_action_alignment.py`

**Interfaces:**
- Consumes: a dataset directory produced by `collect_libero_dataset.py` (post-Task-1 fix), loaded via `lerobot.datasets.lerobot_dataset.LeRobotDataset(repo_id=..., root=...)`. Reads each episode's `action` and `observation.state` tensors in frame order.
- Produces: a pass/fail print + nonzero exit code on failure. No other code depends on this script — it's a standalone check, not imported elsewhere.

- [ ] **Step 1: Write the script**

Create `data_collection/verify_action_alignment.py`:

```python
"""
Verify that recorded actions predict the NEXT state, not the current one.

Run this against a small dataset collected with the fixed collectors to
confirm the action/observation timestep-alignment fix
(docs/superpowers/specs/2026-09-14-fix-action-observation-alignment-design.md)
actually changed what's on disk.

Usage:
    python data_collection/verify_action_alignment.py --root <path_to_dataset_dir> --repo-id <repo_id>
"""
import argparse
import sys

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Verify action/observation alignment")
    parser.add_argument("--root", type=str, required=True, help="Local dataset directory")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo_id used at collection time")
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    num_episodes = dataset.meta.total_episodes
    if num_episodes == 0:
        print("FAIL: dataset has zero episodes — nothing to verify.")
        sys.exit(1)

    diffs_action_to_current_state = []
    diffs_action_to_next_state = []

    for ep_idx in range(num_episodes):
        ep = dataset.meta.episodes[ep_idx]
        start = ep["dataset_from_index"]
        end = ep["dataset_to_index"]

        states = np.stack([dataset[i]["observation.state"][:7].numpy() for i in range(start, end)])
        actions = np.stack([dataset[i]["action"][:7].numpy() for i in range(start, end)])

        # Compare action_t against state_t (current) and state_{t+1} (next),
        # for every t where a "next" frame exists in this episode.
        diffs_action_to_current_state.append(np.abs(actions[:-1] - states[:-1]).mean())
        diffs_action_to_next_state.append(np.abs(actions[:-1] - states[1:]).mean())

    mean_to_current = float(np.mean(diffs_action_to_current_state))
    mean_to_next = float(np.mean(diffs_action_to_next_state))

    print(f"mean |action_t - state_t|     (current, should be LARGER): {mean_to_current:.6f}")
    print(f"mean |action_t - state_t+1|   (next, should be SMALLER):   {mean_to_next:.6f}")

    if mean_to_next < mean_to_current:
        print("PASS: recorded actions are closer to the NEXT state than the current one — alignment fix confirmed.")
        sys.exit(0)
    else:
        print("FAIL: recorded actions are closer to the CURRENT state — stale-action bug may still be present.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Collect a tiny fixed-collector dataset to verify against**

```bash
docker compose run --rm collect python data_collection/collect_libero_dataset.py --episodes 2 --headless
```
Note the printed `Initializing dataset: <path>` line at the start of output — it names the versioned dataset directory (e.g. `datasets/franka_libero_stacking_v3`). Use that path in the next step.

- [ ] **Step 3: Run the verification script and confirm PASS**

```bash
docker compose run --rm collect python data_collection/verify_action_alignment.py \
  --root datasets/franka_libero_stacking_v3 \
  --repo-id franka_libero_stacking_v3
```
(Substitute the actual versioned directory name from Step 2's output.)

Expected: script prints both mean-diff values with `mean |action_t - state_t+1|` smaller than `mean |action_t - state_t|`, and prints `PASS: ...`, exiting 0.

If it prints `FAIL`, stop and re-check Task 1's edit before proceeding — do not continue to Step 4.

- [ ] **Step 4: Commit**

```bash
git add data_collection/verify_action_alignment.py
git commit -m "test: add action/observation alignment verification script"
```

---

## Post-plan note (not part of this plan's scope)

This plan fixes the collection *code*. It does not re-collect
`franka_libero_stacking_v1`/`v2` (both still contain the mislabeled data)
or retrain. Once this plan's commits land, re-collecting a fresh dataset
with the fixed `collect_libero_dataset.py` and retraining SmolVLA on it is
the next step to actually validate whether this was the root cause of the
"never picks the object" inference failure — track that as a separate
follow-up, not an extension of this plan.
