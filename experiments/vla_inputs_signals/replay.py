"""Exact replay of a sealed P2 episode from its sealed initial state and sealed actions.

Runs in the robosuite image. Nothing here is tolerant: at every tick the live `qpos` and prop poses must equal the
sealed ones byte for byte, and the state after the last tick must equal the sealed `final_qpos`. The loop is the
policy rollout's (`xvla_props_p2_eval.rollout`): synchronize, read the observation, act, step.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

from data_collection.benchmark import apply_action, restore_state, step_control, synchronize_observation
from experiments.props_p2_data.episode import expert_for, names_of
from experiments.props_scene import constants as c
from experiments.props_scene.expert import HOLDING
from experiments.vla_inputs_signals import contract
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF

NPZ_KEYS = ("qpos", "ctrl", "props_pose", "initial_state", "final_qpos")


class ReplayMismatch(ValueError):
    def __init__(self, index, tick, field):
        super().__init__(f"episode {index}: sealed and replayed {field} differ at tick {tick}")
        self.index, self.tick, self.field = index, tick, field


def _plan_episode(index):
    plan = json.loads((contract.P2_ROOT / "plan.json").read_text())
    return plan["episodes"][index]


def _dataset_rows(index):
    import pyarrow.parquet as pq
    root = contract.P2_ROOT / "dataset"
    rows = [r for path in sorted((root / "meta/episodes").rglob("*.parquet")) for r in pq.read_table(path).to_pylist()
            if r["episode_index"] == index]
    if len(rows) != 1:
        raise ValueError(f"episode {index} is not in the sealed dataset exactly once")
    path = root / f"data/chunk-{rows[0]['data/chunk_index']:03d}/file-{rows[0]['data/file_index']:03d}.parquet"
    table = pq.read_table(path, columns=["episode_index", "frame_index", "action", "observation.state"]).to_pandas()
    table = table[table["episode_index"] == index].sort_values("frame_index")
    if table["frame_index"].tolist() != list(range(rows[0]["length"])):
        raise ValueError(f"episode {index}: frame_index is not 0..length-1")
    return np.stack(table["action"].to_numpy()).astype(np.float32), np.stack(table["observation.state"].to_numpy()).astype(np.float32)


def load_episode(index):
    """The plan episode, the sealed trace arrays, and the sealed `action` and `observation.state` rows (float32)."""
    episode = _plan_episode(index)
    with np.load(contract.P2_ROOT / "episodes" / f"episode_{index:03d}.npz") as npz:
        arrays = {key: npz[key] for key in NPZ_KEYS}
        identity = (str(npz["object"]), str(npz["bin"]), str(npz["blue_slot"]), [str(d) for d in npz["distractors"]])
    if episode["episode_index"] != index or identity != (
            episode["object"], episode["bin"], episode["blue_slot"], [d["object"] for d in episode["distractors"]]):
        raise ValueError(f"episode {index}: the sealed trace and the plan of record disagree")
    action, state = _dataset_rows(index)
    if not (len(action) == len(state) == len(arrays["qpos"]) == len(arrays["props_pose"]) == len(arrays["ctrl"])):
        raise ValueError(f"episode {index}: sealed arrays differ in length")
    return {"index": index, "episode": episode, "action": action, "state": state, **arrays}


def frames(index, on_frame):
    """Replay episode `index`; `on_frame(tick, model, data)` runs at the observation instant of every tick.

    `on_frame` must not write `data` (S0 computes on a scratch copy). Returns {"frames", "index", "phase_index"}.
    The sealed NPZ has no phase index, so it is rebuilt from the simulation clock by the expert's own rule and
    must agree with the sealed gripper command at every tick.
    """
    sealed = load_episode(index)
    expert = expert_for(names_of(sealed["episode"]), sealed["episode"]["blue_slot"])
    model = expert.model
    bodies = [model.body(f"prop_{name}").id for name in expert.objects]
    data = mujoco.MjData(model)
    restore_state(model, data, sealed["initial_state"])
    phase, phase_start, phases = 0, data.time, []
    for tick, action in enumerate(sealed["action"]):
        synchronize_observation(model, data)
        if not np.array_equal(data.qpos, sealed["qpos"][tick]):
            raise ReplayMismatch(index, tick, "qpos")
        props = np.array([np.concatenate((data.xpos[b], data.xquat[b])) for b in bodies])
        if not np.array_equal(props, sealed["props_pose"][tick]):
            raise ReplayMismatch(index, tick, "props_pose")
        name = c.PHASE_NAMES[phase]
        if (float(action[7]) == 0.0) != (name in HOLDING):
            raise ReplayMismatch(index, tick, "phase")
        phases.append(phase)
        if data.time - phase_start > c.PHASE_DURATION_S[name]:
            phase, phase_start = phase + 1, data.time
        on_frame(tick, model, data)
        apply_action(model, data, action, ACTIVE_JOINTS)
        step_control(model, data, 1 / c.FPS, ARM_DOF)
    if not np.array_equal(data.qpos, sealed["final_qpos"]):
        raise ReplayMismatch(index, len(sealed["action"]), "final_qpos")
    return {"frames": len(sealed["action"]), "index": index, "phase_index": np.asarray(phases, dtype=np.int64)}
