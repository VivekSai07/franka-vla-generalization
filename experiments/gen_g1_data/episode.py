"""One G1 attempt: a randomized scene, the engine's expert, how far each distractor moved, and the live signals.

Runs in the robosuite image. The scene is `gen_scene.compose.model_for(episode, condition)`, the expert is `gen_scene.expert.GenExpert`
(both frozen with G0's core, imported read-only). A scene is unique to its slot, so a model is built per attempt; `compose` keeps at
most `MAX_CACHED_MODELS` compiled models and clears them when full (a rebuild is bit-identical), and the expert is dropped with the
attempt.

Signals: `replay_signals` re-applies the recorded actions from the recorded initial state and computes S0's `wrist_wrench`,
`finger_force` and `props_box` (imported from `vla_inputs_signals.signals`, read-only) at the observation instant of every tick, exactly
where S0 computes them. `GenExpert.run` is frozen and has no observation hook, so the signals are computed from the exact replay of the
attempt in the same process, which is also the exactness gate: at every tick the replayed `qpos` and every object's pose must equal the
recorded trace byte for byte, and the state after the last tick must equal the recorded final state, or the attempt is rejected.
"""
import mujoco
import numpy as np

from data_collection.benchmark import apply_action, restore_state, step_control, synchronize_observation
from experiments.gen_g1_data import plan as p
from experiments.gen_scene import compose
from experiments.gen_scene.expert import GenExpert
from experiments.props_scene import constants as c
from experiments.props_scene.metrics import in_bin, success_streak
from experiments.vla_inputs_signals import signals
from experiments.vla_inputs_signals.sidecar import segments_of
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF

GRIPPER_COLUMN = 7


def scene_of(episode):
    """(model, resolved scene) of a plan slot or one of its substitutes."""
    return compose.model_for(episode, p.condition_of(episode))


def _tilt_deg(data, model, name):
    zz = data.xmat[model.body(f"prop_{name}").id].reshape(3, 3)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def _poses(model, state, names):
    data = mujoco.MjData(model)
    restore_state(model, data, state)
    mujoco.mj_forward(model, data)
    return {n: (data.xpos[model.body(f"prop_{n}").id][:2].copy(), _tilt_deg(data, model, n)) for n in names}


def run_attempt(episode, draw, sink, render=True):
    """One attempt with `draw`'s seed and home jitter; `sink(frame, trace)` per tick."""
    model, resolved = scene_of(episode)
    expert = GenExpert(model, resolved["primitive_names"])
    scene = compose.reset(model, resolved["poses"], draw["home_pose_jitter_deg"])
    run_episode = {"object": episode["object"], "bin": episode["bin"], "seed": draw["seed"],
                   "home_pose_jitter_deg": list(draw["home_pose_jitter_deg"])}
    result = expert.run(scene["initial_state"], run_episode, sink, render=render)
    distractors = [n for n in resolved["names"] if n != episode["object"]]
    before, after = _poses(model, scene["initial_state"], distractors), _poses(model, result["final_state"], distractors)
    motion = [{"object": n, "displacement_m": float(np.linalg.norm(after[n][0] - before[n][0])), "tilt_deg": after[n][1]}
             for n in distractors]
    return {**result, "reset": scene, "distractors": distractors, "props": list(expert.objects),
            "primitives": list(resolved["primitive_names"]), "distractor_motion": motion}


def distractor_contacts(result, actions, traces):
    """Re-run the recorded actions substep by substep and list every distractor (prop or primitive) touched by anything but the table.

    The loop is `data_collection.benchmark.step_control` unrolled so each substep's contacts can be read; every tick's state is
    required to equal the recorded trace exactly, so this is the recorded physics. The lemon rocks on its own after reset (P2,
    FAILURES #54), so a pose threshold cannot tell disturbance from rest; contact can.
    """
    model = result["model"]
    distractor_ids = {model.body(f"prop_{n}").id: n for n in result["distractors"]}
    allowed = {model.body("world").id, model.body("table").id}
    data = mujoco.MjData(model)
    restore_state(model, data, result["initial_state"])
    touched = {}
    for tick, (action, trace) in enumerate(zip(actions, traces)):
        synchronize_observation(model, data)
        if not np.array_equal(data.qpos, trace["qpos"]):
            raise ValueError(f"contact re-run diverged from the recorded trace at tick {tick}")
        apply_action(model, data, action, ACTIVE_JOINTS)
        if int(trace["substeps"]) == 0:
            continue
        synchronize_observation(model, data)
        for _ in range(int(trace["substeps"])):
            data.qfrc_applied[:ARM_DOF] = data.qfrc_bias[:ARM_DOF]
            mujoco.mj_step(model, data)
            for i in range(data.ncon):
                a = int(model.geom_bodyid[data.contact[i].geom1])
                b = int(model.geom_bodyid[data.contact[i].geom2])
                for mine, other in ((a, b), (b, a)):
                    if mine in distractor_ids and other not in allowed and distractor_ids[mine] not in touched:
                        touched[distractor_ids[mine]] = {"object": distractor_ids[mine], "partner": model.body(other).name,
                                                         "tick": tick}
        synchronize_observation(model, data)
    return {"touched": [touched[n] for n in result["distractors"] if n in touched]}


class ReplayMismatch(ValueError):
    pass


def replay_signals(episode, result, actions, traces):
    """(arrays, details): the S0 sidecar arrays of this attempt by exact replay. Raises `ReplayMismatch` on the first byte that differs.

    `arrays` has the keys and dtypes of `vla_inputs_signals.sidecar.build_episode`; `prop_names` is the props only (the target's box
    column is found by name), primitives are not boxed. The check order is the observation instant of S0's replay: synchronize,
    compare, compute, act, step.
    """
    model, target = result["model"], episode["object"]
    props, bodies = result["props"], [model.body(f"prop_{n}").id for n in (*result["props"], *result["primitives"])]
    column = props.index(target)
    data = mujoco.MjData(model)
    restore_state(model, data, result["initial_state"])
    wrench, force, boxes, stable, ticks = [], [], [], 0, len(actions)
    for tick, (action, trace) in enumerate(zip(actions, traces)):
        synchronize_observation(model, data)
        if not np.array_equal(data.qpos, trace["qpos"]):
            raise ReplayMismatch(f"qpos differs at tick {tick}")
        if not np.array_equal(np.array([np.concatenate((data.xpos[b], data.xquat[b])) for b in bodies]), trace["props_pose"]):
            raise ReplayMismatch(f"props_pose differs at tick {tick}")
        wrench.append(signals.wrist_wrench(model, data))
        force.append(signals.finger_force(model, data))
        boxes.append(signals.props_box(model, data, props))
        apply_action(model, data, action, ACTIVE_JOINTS)
        if int(trace["substeps"]) == 0:
            raise ReplayMismatch(f"tick {tick} was recorded but never stepped")
        if step_control(model, data, 1 / c.FPS, ARM_DOF) != int(trace["substeps"]):
            raise ReplayMismatch(f"substeps differ at tick {tick}")
        metrics = in_bin(model, data, target, episode["bin"])
        if metrics["success_tick"] != bool(trace["success_tick"]):
            raise ReplayMismatch(f"success tick differs at tick {tick}")
        stable, _ = success_streak(stable, metrics)
    if not np.array_equal(data.qpos, result["final_qpos"]):
        raise ReplayMismatch("final qpos differs")
    if stable != result["stable_ticks"]:
        raise ReplayMismatch(f"stable ticks {stable} differ from the expert's {result['stable_ticks']}")
    props_box = np.stack(boxes)
    arrays = {
        "episode_index": np.array(episode["episode_index"]), "frame_index": np.arange(ticks),
        "wrist_wrench": np.stack(wrench), "finger_force": np.stack(force), "target_box": props_box[:, column].copy(),
        "props_box": props_box, "prop_names": np.array(props),
        "segment": segments_of(np.asarray(actions), np.stack([t["props_pose"] for t in traces])[:, column, 2]),
        "phase_index": np.array([int(t["phase_index"]) for t in traces], dtype=np.int64),
        "object": np.array(target), "split": np.array(episode["split"]),
    }
    return arrays, {"ticks": ticks, "stable_ticks": stable}
