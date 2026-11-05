"""Equality with A3 and parity with the sidecar, on replayed sealed episodes (M3b spec D7 i). Robosuite image, CPU, OSMesa rendering.

The sealed episodes 0, 18, 271 and 539 are replayed exactly (S0's `replay.frames`). At every frame each adapter's state builder runs
at the instant M0's rollout calls it; at four frames per episode the two cameras are rendered (512 x 512) and every adapter's `act` runs
on the same render with a stub policy that returns a function of everything it was given. Proven:

* with the corruption off (`XVLANoCorruption`, and every class built with `enabled=False`), the frames the processors receive, every
  other processor input and the stub's action equal A3's byte for byte;
* with a corruption on, the frames are the render drawn with `corruption.apply(condition, sidecar box of this frame, frame index)`,
  and for `swap` with the live box of the nearest other prop (one of the episode's distractors), never the target's.
"""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

EPISODES = (0, 18, 271, 539)
FRAMES = 452
TICKS = (0, 120, 260, 451)
CONDITIONS = ("jitter_small", "jitter_large", "dropout", "swap")


def modules():
    from experiments.vla_inputs_arm_xvla_force_box_noisy import adapter, corruption
    from experiments.vla_inputs_signals import replay, signals
    from experiments.xvla_p2_force import forces
    from experiments.xvla_p2_force_box import boxes, contract, eval_adapter, overlay
    return adapter, corruption, replay, signals, forces, boxes, contract, eval_adapter, overlay


def mounted():
    adapter, corruption, replay, signals, forces, boxes, contract, eval_adapter, overlay = modules()
    return all(path.exists() for path in (forces.SIDECAR / "seal.json", contract.PARENT / "seal.json"))


def build(cls, **kwargs):
    seen = {}

    class Policy:
        config = SimpleNamespace()

        def reset(self):
            pass

        def select_action(self, processed):
            seen["processed"] = processed
            total = sum(float(v.double().sum()) for v in processed.values() if torch.is_tensor(v))
            return torch.full((1, 8), total, dtype=torch.float32)

    def pre(observation):
        seen["observation"] = observation
        return observation

    with patch("experiments.vla_inputs_arm_xvla.adapter.load_policy", return_value=Policy()), \
            patch("lerobot.policies.factory.make_pre_post_processors", return_value=(pre, lambda action: action)):
        adapter = cls(Path("/nonexistent/checkpoint"), **kwargs)
    return adapter, seen


def replay_episode(index, adapters):
    """{name: {tick: {state, observation, processed, action}}} for every adapter, plus the renders and the distractor names."""
    import mujoco
    from experiments.vla_inputs_eval.rollout import CAMERA_NAMES
    adapter_module, corruption, replay, signals, forces, boxes, contract, eval_adapter, overlay = modules()
    sealed = replay.load_episode(index)
    target, instruction = sealed["episode"]["object"], contract.planned_instructions()[index]
    distractors = [d["object"] for d in sealed["episode"]["distractors"]]
    out = {"renders": {}, "swap_live": {}, "nearest": None, "names": {name: {} for name in adapters}, "target": target}
    holder = {}

    def on_frame(tick, model, data):
        states = {name: a.state_builder(model, data, target) for name, (a, _) in adapters.items()}
        if tick == 0:
            here = np.array(data.xpos[model.body(f"prop_{target}").id])
            distance = {n: float(np.linalg.norm(np.array(data.xpos[model.body(f"prop_{n}").id]) - here)) for n in distractors}
            out["nearest"] = min(sorted(distance), key=distance.get)
        out["swap_live"][tick] = signals.target_box(model, data, out["nearest"])
        if tick not in TICKS:
            return
        if "renderer" not in holder:
            holder["renderer"] = mujoco.Renderer(model, height=512, width=512)
        frames = []
        for camera in CAMERA_NAMES:
            holder["renderer"].update_scene(data, camera=camera)
            frames.append(holder["renderer"].render().copy())
        out["renders"][tick] = frames
        for name, (a, seen) in adapters.items():
            action = a.act(dict(images=frames, state=states[name], instruction=instruction))
            out["names"][name][tick] = dict(state=states[name], observation=seen["observation"], processed=seen["processed"], action=action)

    try:
        replay.frames(index, on_frame)
    finally:
        if "renderer" in holder:
            holder["renderer"].close()
    return out


def same(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.tobytes() == b.tobytes()
    return a == b


class ReplayParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not mounted():
            raise unittest.SkipTest("P2 parent or S0 sidecar not mounted")
        (cls.m, cls.corruption, cls.replay, cls.signals, cls.forces, cls.boxes, cls.c, cls.a3, cls.overlay) = modules()
        cls.cache = {}
        for index in EPISODES:
            adapters = {"a3": build(cls.a3.XVLAForceBoxAdapter), "identity": build(cls.m.XVLANoCorruption)}
            for condition, name in zip(CONDITIONS, ("XVLAJitterSmall", "XVLAJitterLarge", "XVLADropout", "XVLASwap")):
                adapters[f"off_{condition}"] = build(getattr(cls.m, name), enabled=False)
                adapters[condition] = build(getattr(cls.m, name))
            cls.cache[index] = replay_episode(index, adapters)

    def test_with_the_corruption_off_every_input_the_policy_receives_equals_a3s_byte_for_byte(self):
        for index, got in self.cache.items():
            for name in ("identity", *(f"off_{c}" for c in CONDITIONS)):
                for tick in TICKS:
                    mine, theirs = got["names"][name][tick], got["names"]["a3"][tick]
                    self.assertTrue(same(mine["state"], theirs["state"]), (index, name, tick))
                    self.assertEqual(sorted(mine["observation"]), sorted(theirs["observation"]))
                    for key in theirs["observation"]:
                        self.assertTrue(same(mine["observation"][key], theirs["observation"][key]), (index, name, tick, key))
                    for key in theirs["processed"]:
                        self.assertTrue(same(mine["processed"][key], theirs["processed"][key]), (index, name, tick, key))
                    self.assertEqual(mine["action"].tobytes(), theirs["action"].tobytes(), (index, name, tick))

    def test_a3s_own_frames_are_the_render_drawn_with_the_sidecar_box(self):
        """The reference is right: A3's frames are the replayed render with the sealed box on it, so equality with A3 means the sealed box."""
        for index, got in self.cache.items():
            for tick in TICKS:
                box = self.boxes.table()[index, tick]
                seen = got["names"]["a3"][tick]["observation"]
                for camera, key in enumerate(self.c.CAMERAS):
                    undrawn = torch.from_numpy(got["renders"][tick][camera]).permute(2, 0, 1).float() / 255
                    self.assertTrue(torch.equal(seen[key], self.overlay.draw(undrawn, box[camera])), (index, tick, key))

    def test_a_corrupted_arm_draws_the_corruption_of_the_sidecar_box_at_this_frames_tick(self):
        for index, got in self.cache.items():
            for condition in ("jitter_small", "jitter_large", "dropout"):
                for tick in TICKS:
                    want = self.corruption.apply(condition, self.boxes.table()[index, tick], tick)
                    seen = got["names"][condition][tick]["observation"]
                    for camera, key in enumerate(self.c.CAMERAS):
                        undrawn = torch.from_numpy(got["renders"][tick][camera]).permute(2, 0, 1).float() / 255
                        self.assertTrue(torch.equal(seen[key], self.overlay.draw(undrawn, want[camera])), (index, condition, tick, key))

    def test_the_corruptions_actually_change_the_frames(self):
        changed = {c: 0 for c in CONDITIONS}
        for index, got in self.cache.items():
            for tick in TICKS:
                a3 = got["names"]["a3"][tick]["observation"]
                for condition in CONDITIONS:
                    seen = got["names"][condition][tick]["observation"]
                    changed[condition] += int(any(not torch.equal(seen[k], a3[k]) for k in self.c.CAMERAS))
        for condition, count in changed.items():
            self.assertGreater(count, 0, condition)
        self.assertEqual(changed["swap"], len(EPISODES) * len(TICKS))  # a distractor's box is never the target's

    def test_swap_draws_the_live_box_of_the_nearest_distractor_on_both_cameras(self):
        for index, got in self.cache.items():
            self.assertIn(got["nearest"], [d["object"] for d in self.replay.load_episode(index)["episode"]["distractors"]])
            self.assertNotEqual(got["nearest"], got["target"])
            for tick in TICKS:
                seen = got["names"]["swap"][tick]["observation"]
                for camera, key in enumerate(self.c.CAMERAS):
                    undrawn = torch.from_numpy(got["renders"][tick][camera]).permute(2, 0, 1).float() / 255
                    self.assertTrue(torch.equal(seen[key], self.overlay.draw(undrawn, got["swap_live"][tick][camera])), (index, tick, key))


if __name__ == "__main__":
    unittest.main()
