"""One P2 attempt: target plus distractors in one scene, the P2 expert, and how far each distractor moved.

Runs in the robosuite image. A props scene model costs ~390 MB and `props_scene.scene` caches every
one for the process lifetime; a worker meets up to 45 object sets, so this module keeps at most
`MAX_CACHED_MODELS` and clears both caches when full. A rebuild is bit-identical (measured
2026-10-13: all 434 compiled arrays equal), so eviction cannot change an episode.
"""
import gc

import mujoco
import numpy as np

from data_collection.benchmark import apply_action, restore_state, synchronize_observation
from experiments.props_p2_data.expert import P2Expert
from experiments.props_scene import constants as c
from experiments.props_scene import scene as props_scene
from experiments.props_scene.scene import object_pose, reset
from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF

MAX_CACHED_MODELS = 4
_EXPERTS = {}


def names_of(episode):
    return [episode["object"], *(d["object"] for d in episode["distractors"])]


def expert_for(objects, blue_slot):
    key = (tuple(sorted(objects, key=c.OBJECTS.index)), blue_slot)
    if key not in _EXPERTS:
        if len(_EXPERTS) >= MAX_CACHED_MODELS:
            _EXPERTS.clear()
            props_scene._built.cache_clear()
            gc.collect()
        _EXPERTS[key] = P2Expert(key[0], blue_slot)
    return _EXPERTS[key]


def _tilt_deg(data, model, name):
    zz = data.xmat[model.body(f"prop_{name}").id].reshape(3, 3)[2, 2]
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def _poses(model, state, names):
    data = mujoco.MjData(model)
    restore_state(model, data, state)
    mujoco.mj_forward(model, data)
    return {n: (object_pose(model, data, n)["xy"], _tilt_deg(data, model, n)) for n in names}


def run_attempt(episode, draw, sink, render=True):
    """One attempt with `draw`'s seed and home jitter; `sink(frame, trace)` per tick."""
    expert = expert_for(names_of(episode), episode["blue_slot"])
    poses = {episode["object"]: tuple(episode["start"]), **{d["object"]: tuple(d["pose"]) for d in episode["distractors"]}}
    scene = reset(expert.model, poses, draw["home_pose_jitter_deg"])
    run_episode = {"object": episode["object"], "bin": episode["bin"], "seed": draw["seed"],
                   "home_pose_jitter_deg": list(draw["home_pose_jitter_deg"])}
    result = expert.run(scene["initial_state"], run_episode, sink, render=render)
    distractors = [d["object"] for d in episode["distractors"]]
    before = _poses(expert.model, scene["initial_state"], distractors)
    after = _poses(expert.model, result["final_state"], distractors)
    motion = [{"object": n, "displacement_m": float(np.linalg.norm(after[n][0] - before[n][0])),
               "tilt_deg": after[n][1]} for n in distractors]
    return {**result, "reset": scene, "distractors": distractors, "distractor_motion": motion}


def distractor_contacts(result, actions, traces):
    """Re-run the recorded actions substep by substep and list every distractor touched by anything but the table.

    The loop is `data_collection.benchmark.step_control` unrolled so each substep's contacts can be read;
    every tick's state is required to equal the recorded trace exactly, so this is the recorded physics.
    Measured 2026-10-13: the lemon rocks on its own after reset (up to 13 deg, 4.6 mm, never touched), so
    a pose threshold cannot tell disturbance from rest; contact can.
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
