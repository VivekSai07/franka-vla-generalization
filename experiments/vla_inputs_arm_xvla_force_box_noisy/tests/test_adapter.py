"""The noisy-box adapters with stub model parts: which box each draws, when, and that the diagnostic oracle never reaches the frames.

Robosuite image, CPU (it imports the A3 adapter's stack); loading a real checkpoint needs the GPU and is not done here.
"""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments.vla_inputs_arm_xvla_force_box_noisy import corruption
from experiments.vla_inputs_eval import adapters

FRONT_BOX = np.array([0.25, 0.125, 0.5, 0.375, 1.0])
GRIPPER_BOX = np.array([0.5, 0.5, 0.75, 0.75, 1.0])
BOX = np.stack([FRONT_BOX, GRIPPER_BOX])
DISTRACTOR_BOX = np.stack([np.array([0.0625, 0.5, 0.125, 0.625, 1.0]), np.array([0.125, 0.125, 0.25, 0.25, 1.0])])
PREFIX = "experiments.vla_inputs_arm_xvla_force_box_noisy.adapter:"
CLASSES = {"jitter_small": "XVLAJitterSmall", "jitter_large": "XVLAJitterLarge", "dropout": "XVLADropout", "swap": "XVLASwap"}


def images(front=0, gripper=255):
    return [np.full((512, 512, 3), front, np.uint8), np.full((512, 512, 3), gripper, np.uint8)]


class Fixture(unittest.TestCase):
    def setUp(self):
        from experiments.vla_inputs_arm_xvla_force_box_noisy import adapter
        from experiments.xvla_p2_force_box import contract, eval_adapter, overlay
        self.adapter_module, self.contract, self.a3, self.overlay = adapter, contract, eval_adapter, overlay
        self.instruction = contract.INSTRUCTIONS[0]

    def build(self, cls, **kwargs):
        seen = {"reset": 0, "pre": []}
        kwargs.setdefault("state_builder", lambda model, data, target: np.linspace(-1., 1., 17))

        class Policy:
            config = SimpleNamespace()

            def reset(self):
                seen["reset"] += 1

            def select_action(self, processed):
                return torch.arange(8, dtype=torch.float32).reshape(1, 8)

        def pre(observation):
            seen["pre"].append(observation)
            return observation

        with patch("experiments.vla_inputs_arm_xvla.adapter.load_policy", return_value=Policy()), \
                patch("lerobot.policies.factory.make_pre_post_processors", return_value=(pre, lambda action: action)):
            adapter = cls(Path("/nonexistent/checkpoint"), **kwargs)
        return adapter, seen

    def cls(self, condition):
        return getattr(self.adapter_module, CLASSES[condition])

    def tick(self, adapter, box=BOX, other=None, **kwargs):
        """What M0's rollout does at one tick: the state builder, then act."""
        live = {"milk": box, "can": DISTRACTOR_BOX if other is None else other}
        with patch.object(self.adapter_module.signals, "target_box", side_effect=lambda model, data, name: live[name].copy()), \
                patch.object(self.a3.signals, "target_box", side_effect=lambda model, data, name: live[name].copy()), \
                patch.object(self.adapter_module, "prop_positions", return_value={"milk": np.zeros(3), "can": np.array([0.1, 0, 0])}):
            built = adapter.state_builder("model", "data", "milk")
        obs = dict(images=images(**kwargs), state=built, instruction=self.instruction)
        return built, adapter.act(obs)

    def drawn(self, seen, tick=-1):
        given = seen["pre"][tick]
        return [given[key] for key in self.contract.CAMERAS]

    def expected(self, box, **kwargs):
        undrawn = [torch.from_numpy(i).permute(2, 0, 1).float() / 255 for i in images(**kwargs)]
        return [self.overlay.draw(undrawn[0], box[0]), self.overlay.draw(undrawn[1], box[1])]


class Interface(Fixture):
    def test_every_class_implements_the_m0_interface_and_the_identifier_m0_loads(self):
        for condition, name in CLASSES.items():
            adapter, seen = self.build(self.cls(condition))
            adapters.check_adapter(adapter)
            self.assertTrue(adapters._IDENTIFIER.match(PREFIX + name))
            self.assertEqual(adapter.condition, condition)
            self.assertTrue(callable(adapter.state_builder))
            adapter.reset()
            self.assertEqual(seen["reset"], 1)

    def test_the_arm_names_map_to_the_classes_by_the_conditions(self):
        for arm, condition in corruption.ARMS.items():
            self.assertEqual(self.cls(condition).condition, condition, arm)

    def test_it_extends_a3s_frozen_adapter_and_only_that(self):
        from experiments.vla_inputs_arm_xvla_force_box.adapter import XVLAForceBoxAdapter
        for condition in CLASSES:
            self.assertTrue(issubclass(self.cls(condition), XVLAForceBoxAdapter))
        self.assertIs(self.adapter_module.XVLAForceBoxAdapter, XVLAForceBoxAdapter)
        self.assertNotIn("act", vars(self.adapter_module.NoisyBoxAdapter))  # the policy call and its processors are A3's own

    def test_the_inputs_and_the_state_builder_are_a3s(self):
        self.assertEqual(self.adapter_module.INPUT_DIRS, (".cache/xvla_trial/bundle", "outputs/vla_inputs_signals/p2_signals_20261030"))
        self.assertIs(self.adapter_module.state_force, self.a3.state_force)


class DrawsTheCorruptedBox(Fixture):
    def test_jitter_draws_the_corrupted_box_with_the_tick_counted_from_reset(self):
        for condition in ("jitter_small", "jitter_large"):
            adapter, seen = self.build(self.cls(condition))
            for tick in range(4):
                self.tick(adapter)
                want = corruption.apply(condition, BOX, tick)
                for got, expect in zip(self.drawn(seen), self.expected(want)):
                    self.assertTrue(torch.equal(got, expect), (condition, tick))
            adapter.reset()
            self.tick(adapter)  # a new trial counts from tick 0 again
            for got, expect in zip(self.drawn(seen), self.expected(corruption.apply(condition, BOX, 0))):
                self.assertTrue(torch.equal(got, expect), condition)

    def test_the_same_trial_on_a_second_adapter_draws_the_same_frames(self):
        first, seen_a = self.build(self.cls("jitter_large"))
        second, seen_b = self.build(self.cls("jitter_large"))
        for _ in range(3):
            self.tick(first)
        for _ in range(3):
            self.tick(second)
        for a, b in zip(seen_a["pre"], seen_b["pre"]):
            for key in self.contract.CAMERAS:
                self.assertTrue(torch.equal(a[key], b[key]))

    def test_dropout_leaves_the_gripper_frame_undrawn_exactly_when_the_corruption_says_so(self):
        adapter, seen = self.build(self.cls("dropout"))
        gone = 0
        for tick in range(40):
            self.tick(adapter)
            front, gripper = self.drawn(seen)
            want = corruption.apply("dropout", BOX, tick)
            gone += int(want[1, 4] == 0)
            expect = self.expected(want)
            self.assertTrue(torch.equal(front, expect[0]) and torch.equal(gripper, expect[1]), tick)
            self.assertTrue(torch.equal(front, self.expected(BOX)[0]))  # the front box is always drawn
            if want[1, 4] == 0:
                self.assertTrue(torch.equal(gripper, torch.from_numpy(images()[1]).permute(2, 0, 1).float() / 255))
        self.assertTrue(5 < gone < 35)

    def test_swap_draws_the_distractors_live_box_and_never_the_targets(self):
        adapter, seen = self.build(self.cls("swap"))
        self.tick(adapter)
        front, gripper = self.drawn(seen)
        expect = self.expected(DISTRACTOR_BOX)
        self.assertTrue(torch.equal(front, expect[0]) and torch.equal(gripper, expect[1]))
        for target_frame in self.expected(BOX):
            self.assertFalse(torch.equal(front, target_frame))
        self.assertEqual(adapter.distractor, "can")
        moved = np.stack([np.array([0.5, 0.0625, 0.625, 0.125, 1.0]), DISTRACTOR_BOX[1]])  # the distractor moved: its live box follows
        self.tick(adapter, other=moved)
        self.assertTrue(torch.equal(self.drawn(seen)[0], self.expected(moved)[0]))

    def test_the_distractor_is_chosen_once_per_trial_at_the_first_tick_and_reset_forgets_it(self):
        adapter, _ = self.build(self.cls("swap"))
        calls = []

        def positions(model, data):
            calls.append(1)
            return {"milk": np.zeros(3), "can": np.array([0.3, 0, 0]), "bread": np.array([0.1, 0, 0])}

        live = lambda model, data, name: BOX.copy()
        with patch.object(self.adapter_module.signals, "target_box", side_effect=live), \
                patch.object(self.a3.signals, "target_box", side_effect=live), \
                patch.object(self.adapter_module, "prop_positions", side_effect=positions):
            for _ in range(3):
                adapter.state_builder("m", "d", "milk")
            self.assertEqual((adapter.distractor, len(calls)), ("bread", 1))
            adapter.reset()
            self.assertIsNone(adapter.distractor)
            adapter.state_builder("m", "d", "milk")
            self.assertEqual(len(calls), 2)

    def test_the_oracle_box_is_kept_as_a_diagnostic_and_is_not_what_is_drawn(self):
        for condition in ("jitter_large", "dropout", "swap"):
            adapter, seen = self.build(self.cls(condition))
            self.tick(adapter)
            np.testing.assert_array_equal(adapter.oracle_box, BOX)
        adapter, seen = self.build(self.cls("swap"))
        self.tick(adapter)
        self.tick(adapter, box=np.stack([GRIPPER_BOX, FRONT_BOX]))  # change the oracle: the swap's frames do not move
        self.assertTrue(torch.equal(self.drawn(seen, 0)[0], self.drawn(seen, 1)[0]))
        np.testing.assert_array_equal(adapter.oracle_box, np.stack([GRIPPER_BOX, FRONT_BOX]))

    def test_the_state_the_builder_returns_is_the_inner_one_untouched(self):
        state = np.linspace(-2., 2., 17)
        for condition in CLASSES:
            adapter, _ = self.build(self.cls(condition), state_builder=lambda model, data, target: state)
            built, _ = self.tick(adapter)
            np.testing.assert_array_equal(built, state)


class WithTheCorruptionOffItIsA3(Fixture):
    def a3_frames(self, box, **kwargs):
        adapter, seen = self.build(self.a3.XVLAForceBoxAdapter)
        with patch.object(self.a3.signals, "target_box", return_value=box.copy()):
            built = adapter.state_builder("model", "data", "milk")
        adapter.act(dict(images=images(**kwargs), state=built, instruction=self.instruction))
        return seen["pre"][-1]

    def test_identity_and_every_class_with_enabled_false_give_a3s_inputs_byte_for_byte(self):
        want = self.a3_frames(BOX)
        for cls, kwargs in ([(self.adapter_module.XVLANoCorruption, {})]
                            + [(self.cls(c), {"enabled": False}) for c in CLASSES]):
            adapter, seen = self.build(cls, **kwargs)
            for _ in range(3):
                self.tick(adapter)
                for key in want:
                    got = seen["pre"][-1][key]
                    self.assertTrue(torch.equal(got, want[key]) if torch.is_tensor(got) else got == want[key], (cls.__name__, key))

    def test_with_enabled_false_swap_never_looks_for_a_distractor(self):
        adapter, _ = self.build(self.cls("swap"), enabled=False)
        with patch.object(self.adapter_module, "prop_positions", side_effect=AssertionError("looked")):
            with patch.object(self.a3.signals, "target_box", return_value=BOX.copy()):
                adapter.state_builder("model", "data", "milk")

    def test_the_corruption_at_zero_gives_a3s_inputs_for_the_jitter_and_dropout_classes(self):
        want = self.a3_frames(BOX)
        for condition, spec in (("jitter_small", dict(kind="jitter", sigma_px=0.0)), ("dropout", dict(kind="dropout", probability=0.0))):
            with patch.dict(corruption.CONDITIONS, {condition: spec}):
                adapter, seen = self.build(self.cls(condition))
                self.tick(adapter)
            for key in want:
                got = seen["pre"][-1][key]
                self.assertTrue(torch.equal(got, want[key]) if torch.is_tensor(got) else got == want[key], (condition, key))

    def test_the_enabled_classes_differ_from_a3_where_the_condition_says_so(self):
        want = self.a3_frames(BOX)
        adapter, seen = self.build(self.cls("jitter_large"))
        self.tick(adapter)
        self.assertFalse(torch.equal(seen["pre"][-1][self.contract.CAMERAS[0]], want[self.contract.CAMERAS[0]]))


class NearestDistractorOnARealModel(Fixture):
    XML = """<mujoco><worldbody>
      <body name="prop_milk" pos="0 0 0"><freejoint/><geom type="box" size="0.01 0.01 0.01"/></body>
      <body name="prop_can" pos="0.30 0 0"><freejoint/><geom type="box" size="0.01 0.01 0.01"/></body>
      <body name="prop_bread" pos="0 0.12 0.05"><freejoint/><geom type="box" size="0.01 0.01 0.01"/></body>
      <body name="not_a_prop" pos="0.01 0 0"><geom type="box" size="0.01 0.01 0.01"/></body>
    </worldbody></mujoco>"""

    def test_positions_are_read_for_the_props_of_the_model_only(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string(self.XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        positions = self.adapter_module.prop_positions(model, data)
        self.assertEqual(sorted(positions), ["bread", "can", "milk"])
        np.testing.assert_allclose(positions["can"], [0.30, 0, 0])
        self.assertEqual(corruption.nearest_distractor("milk", positions), "bread")


if __name__ == "__main__":
    unittest.main()
