"""One P3 attempt: a single object in the P0 scene, the accepted P0 expert, and the fixed prompt in every frame.

Runs in the robosuite image. The expert is `props_scene.expert.PropsExpert`, unchanged. It writes its own
text ("put the X in the Y bin") into every frame; that wording must never reach a dataset, so `run_attempt`
hands the sink frames carrying `plan.PROMPT` instead, and raises if the expert ever writes anything else.

A props scene model costs ~390 MB and `props_scene.scene` caches every one for the process lifetime; a
worker meets up to 12 (object, arrangement) models, so this module keeps at most `MAX_CACHED_MODELS` and
clears both caches when full. A rebuild is bit-identical (measured for P2 on 2026-10-13), so eviction
cannot change an episode.
"""
import gc

from experiments.props_p3_data import plan as p
from experiments.props_scene import scene as props_scene
from experiments.props_scene.expert import PropsExpert, task_text
from experiments.props_scene.scene import reset

MAX_CACHED_MODELS = 4
_EXPERTS = {}


def expert_for(obj, blue_slot):
    key = (obj, blue_slot)
    if key not in _EXPERTS:
        if len(_EXPERTS) >= MAX_CACHED_MODELS:
            _EXPERTS.clear()
            props_scene._built.cache_clear()
            gc.collect()
        _EXPERTS[key] = PropsExpert((obj,), blue_slot)
    return _EXPERTS[key]


def relabelled(sink, episode):
    """`sink` behind a guard: the expert's own wording is replaced by the fixed prompt, anything else raises."""
    expected = task_text(episode["object"], episode["bin"])

    def labelled(frame, trace):
        if frame["task"] != expected:
            raise ValueError(f"expert wrote {frame['task']!r}, expected {expected!r}")
        sink({**frame, "task": p.PROMPT}, trace)

    return labelled


def run_attempt(episode, draw, sink, render=True):
    """One attempt with `draw`'s seed and home jitter; `sink(frame, trace)` per tick, frames labelled with the prompt."""
    obj, bin_colour = episode["object"], episode["bin"]
    expert = expert_for(obj, episode["blue_slot"])
    scene = reset(expert.model, {obj: tuple(episode["start"])}, draw["home_pose_jitter_deg"])
    run_episode = {"object": obj, "bin": bin_colour, "seed": draw["seed"],
                   "home_pose_jitter_deg": list(draw["home_pose_jitter_deg"])}
    result = expert.run(scene["initial_state"], run_episode, relabelled(sink, episode), render=render)
    return {**result, "reset": scene}
