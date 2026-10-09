"""One P1 episode: scene, expert, recorded frames. Runs in the robosuite image.

Reuses the accepted P0 package unchanged: `props_scene.scene.build_model/reset` composes and resets
the scene, `props_scene.expert.PropsExpert.run` scripts the episode and already labels each frame with
the instruction `props_scene.expert.task_text` produces — identical to `plan.instruction`.
"""
from experiments.props_scene.expert import PropsExpert
from experiments.props_scene.scene import reset

_EXPERTS = {}


def _expert(obj, blue_slot):
    key = (obj, blue_slot)
    if key not in _EXPERTS:
        _EXPERTS[key] = PropsExpert((obj,), blue_slot)
    return _EXPERTS[key]


def run_attempt(episode, draw, sink):
    """One attempt at `episode`'s slot with `draw`'s seed and home jitter. `sink(frame, trace)` per tick."""
    obj, bin_colour, blue_slot = episode["object"], episode["bin"], episode["blue_slot"]
    expert = _expert(obj, blue_slot)
    x, y, yaw = episode["start"]
    scene = reset(expert.model, {obj: (x, y, yaw)}, draw["home_pose_jitter_deg"])
    run_episode = {"object": obj, "bin": bin_colour, "seed": draw["seed"],
                   "home_pose_jitter_deg": list(draw["home_pose_jitter_deg"])}
    result = expert.run(scene["initial_state"], run_episode, sink, render=True)
    return {**result, "reset": scene}
