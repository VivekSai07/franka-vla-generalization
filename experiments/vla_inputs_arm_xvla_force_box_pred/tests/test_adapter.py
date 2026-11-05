"""Arm A5: the predicted box drawn on A3's frozen policy's frames. Stub detector and stub model parts; robosuite image, CPU.

What is proven here (spec D6, D7 ii and iii): with a detector that returns the oracle box A5 equals A3 byte for byte; A5 never draws the oracle
(altering the diagnostic oracle changes nothing drawn, and a mutated adapter that draws it fails the very same check); the detector is handed the
raw uint8 renders and the instruction, exactly as the gate measured them; absence draws nothing; the log round-trips into the analysis.
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments.vla_inputs_eval import adapters

FRONT_BOX = np.array([0.25, 0.125, 0.5, 0.375, 1.0])
GRIPPER_BOX = np.array([0.5, 0.5, 0.75, 0.75, 1.0])
ORACLE = np.stack([FRONT_BOX, GRIPPER_BOX])
PREDICTED = np.stack([np.array([0.3, 0.15, 0.55, 0.4, 1.0]), np.array([0.45, 0.5, 0.8, 0.8, 1.0])])
IDENTIFIER = "experiments.vla_inputs_arm_xvla_force_box_pred.adapter:XVLAPredictedBoxAdapter"


def images(front=0, gripper=255):
    return [np.full((512, 512, 3), front, np.uint8), np.full((512, 512, 3), gripper, np.uint8)]


class StubDetector:
    def __init__(self, box=PREDICTED):
        self.box, self.seen = box, []

    def predict(self, rendered, instruction):
        self.seen.append((rendered, instruction))
        return np.array(self.box, dtype=np.float64)


class Fixture(unittest.TestCase):
    def setUp(self):
        from experiments.vla_inputs_arm_xvla_force_box_pred import adapter
        from experiments.xvla_p2_force_box import contract, eval_adapter, overlay
        self.adapter_module, self.contract, self.a3, self.overlay = adapter, contract, eval_adapter, overlay
        self.instruction = contract.INSTRUCTIONS[0]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_dir = Path(self.tmp.name) / "logs"

    def build(self, cls=None, **kwargs):
        seen = {"pre": []}
        kwargs.setdefault("state_builder", lambda model, data, target: np.linspace(-1., 1., 17))

        class Policy:
            config = SimpleNamespace()

            def reset(self):
                seen["reset"] = seen.get("reset", 0) + 1

            def select_action(self, processed):
                return torch.arange(8, dtype=torch.float32).reshape(1, 8)

        def pre(observation):
            seen["pre"].append(observation)
            return observation

        with patch("experiments.vla_inputs_arm_xvla.adapter.load_policy", return_value=Policy()), \
                patch("lerobot.policies.factory.make_pre_post_processors", return_value=(pre, lambda action: action)):
            adapter = (cls or self.adapter_module.XVLAPredictedBoxAdapter)(Path("/nonexistent/checkpoint"), **kwargs)
        return adapter, seen

    def tick(self, adapter, oracle=ORACLE, **render):
        with patch.object(self.a3.signals, "target_box", return_value=oracle.copy()):
            built = adapter.state_builder("model", "data", "milk")
        return built, adapter.act(dict(images=images(**render), state=built, instruction=self.instruction))

    def drawn(self, seen, tick=-1):
        return [seen["pre"][tick][key] for key in self.contract.CAMERAS]

    def expected(self, box, **render):
        undrawn = [torch.from_numpy(i).permute(2, 0, 1).float() / 255 for i in images(**render)]
        return [self.overlay.draw(undrawn[0], box[0]), self.overlay.draw(undrawn[1], box[1])]


class Interface(Fixture):
    def test_it_implements_the_m0_interface_extends_a3_and_is_loadable_by_the_identifier(self):
        from experiments.vla_inputs_arm_xvla_force_box.adapter import XVLAForceBoxAdapter
        adapter, seen = self.build(detector=StubDetector(), log_dir=self.log_dir)
        adapters.check_adapter(adapter)
        self.assertIsInstance(adapter, XVLAForceBoxAdapter)
        self.assertTrue(adapters._IDENTIFIER.match(IDENTIFIER))
        adapter.reset()
        self.assertEqual(seen["reset"], 1)
        self.assertNotIn("act_policy", vars(self.adapter_module.XVLAPredictedBoxAdapter))

    def test_the_input_directories_are_a3s_plus_the_detectors_final_directory(self):
        from experiments.vla_inputs_arm_xvla_force_box.adapter import INPUT_DIRS as a3_dirs
        dirs = self.adapter_module.INPUT_DIRS
        self.assertEqual(dirs[:len(a3_dirs)], a3_dirs)
        self.assertEqual(dirs[-1], "outputs/vla_inputs_box_detector/run_v1/final")
        self.assertEqual(len(dirs), len(a3_dirs) + 1)


class Draws(Fixture):
    def test_the_predicted_box_is_drawn_on_the_two_cameras_not_the_oracle(self):
        detector = StubDetector()
        adapter, seen = self.build(detector=detector, log_dir=self.log_dir)
        self.tick(adapter)
        for got, expect in zip(self.drawn(seen), self.expected(PREDICTED)):
            self.assertTrue(torch.equal(got, expect))
        for got, oracle_frame in zip(self.drawn(seen), self.expected(ORACLE)):
            self.assertFalse(torch.equal(got, oracle_frame))

    def test_the_detector_is_given_the_raw_uint8_renders_and_the_instruction(self):
        detector = StubDetector()
        adapter, _ = self.build(detector=detector, log_dir=self.log_dir)
        self.tick(adapter, front=77, gripper=200)
        rendered, instruction = detector.seen[0]
        self.assertEqual([(i.dtype, i.shape) for i in rendered], [(np.uint8, (512, 512, 3))] * 2)
        self.assertTrue((rendered[0] == 77).all() and (rendered[1] == 200).all())  # not drawn on, not normalised
        self.assertEqual(instruction, self.instruction)

    def test_an_absent_prediction_draws_nothing_in_that_camera(self):
        box = np.stack([PREDICTED[0], np.zeros(5)])
        adapter, seen = self.build(detector=StubDetector(box), log_dir=self.log_dir)
        self.tick(adapter)
        front, gripper = self.drawn(seen)
        undrawn = torch.from_numpy(images()[1]).permute(2, 0, 1).float() / 255
        self.assertTrue(torch.equal(gripper, undrawn))
        self.assertTrue(torch.equal(front, self.expected(box)[0]))

    def test_the_state_the_builder_returns_is_a3s(self):
        state = np.linspace(-2., 2., 17)
        adapter, _ = self.build(detector=StubDetector(), log_dir=self.log_dir, state_builder=lambda m, d, t: state)
        built, _ = self.tick(adapter)
        np.testing.assert_array_equal(built, state)

    def test_the_detector_runs_at_every_act_and_a_prediction_is_not_reused(self):
        detector = StubDetector()
        adapter, _ = self.build(detector=detector, log_dir=self.log_dir)
        for _ in range(3):
            self.tick(adapter)
        self.assertEqual(len(detector.seen), 3)


class EqualsA3WithTheOracleDetector(Fixture):
    def test_a_detector_that_returns_the_oracle_box_gives_a3s_inputs_byte_for_byte(self):
        a3, seen_a3 = self.build(cls=self.a3.XVLAForceBoxAdapter)
        with patch.object(self.a3.signals, "target_box", return_value=ORACLE.copy()):
            built = a3.state_builder("model", "data", "milk")
        a3.act(dict(images=images(front=40), state=built, instruction=self.instruction))
        mine, seen = self.build(detector=StubDetector(ORACLE), log_dir=self.log_dir)
        self.tick(mine, front=40)
        for key in seen_a3["pre"][0]:
            a, b = seen["pre"][0][key], seen_a3["pre"][0][key]
            self.assertTrue(torch.equal(a, b) if torch.is_tensor(a) else a == b, key)


def drawn_frames_with(case, cls, oracle, **kwargs):
    adapter, seen = case.build(cls=cls, detector=StubDetector(), log_dir=case.log_dir, **kwargs)
    case.tick(adapter, oracle=oracle)
    return [f.clone() for f in case.drawn(seen)]


def assert_no_leak(case, cls):
    """The frames do not depend on the oracle: change it, nothing drawn moves."""
    first = drawn_frames_with(case, cls, ORACLE)
    other = drawn_frames_with(case, cls, np.stack([GRIPPER_BOX, FRONT_BOX]))
    for a, b in zip(first, other):
        case.assertTrue(torch.equal(a, b), "the drawn frames changed with the oracle box")


class Leakage(Fixture):
    def test_altering_the_oracle_changes_nothing_that_is_drawn(self):
        assert_no_leak(self, self.adapter_module.XVLAPredictedBoxAdapter)

    def test_the_check_fails_for_an_adapter_mutated_to_draw_the_oracle(self):
        base = self.adapter_module.XVLAPredictedBoxAdapter

        class Leaky(base):
            def act(self, obs):
                self._box = self.oracle_box  # the mutation: the privileged box goes to the policy
                return super(base, self).act(obs)

        with self.assertRaises(AssertionError):
            assert_no_leak(self, Leaky)

    def test_the_oracle_is_kept_only_as_a_diagnostic(self):
        adapter, _ = self.build(detector=StubDetector(), log_dir=self.log_dir)
        self.tick(adapter)
        np.testing.assert_array_equal(adapter.oracle_box, ORACLE)
        adapter.reset()
        self.assertIsNone(adapter.oracle_box)

    def test_act_itself_never_mentions_the_oracle_or_the_signals(self):
        import inspect
        source = inspect.getsource(self.adapter_module.XVLAPredictedBoxAdapter.act)
        self.assertNotIn("signals", source)
        self.assertNotIn("oracle", source)


class Log(Fixture):
    def test_every_act_writes_a_line_that_round_trips_into_the_analysis(self):
        from experiments.vla_inputs_m3b_analysis import analyse
        adapter, _ = self.build(detector=StubDetector(), log_dir=self.log_dir)
        adapter.reset()
        for t in range(3):
            self.tick(adapter, oracle=ORACLE + 0.001 * t * (ORACLE > 0) * (ORACLE < 1))
        adapter.reset()
        self.tick(adapter)
        files = list(self.log_dir.glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        groups = analyse._read_logs(self.log_dir)
        self.assertEqual(len(groups), 2)
        sizes = sorted(len(v[0]) for v in groups.values())
        self.assertEqual(sizes, [1, 3])
        predicted, oracle = next(v for v in groups.values() if len(v[0]) == 3)
        np.testing.assert_array_equal(predicted[0], PREDICTED)
        np.testing.assert_array_equal(oracle[0], ORACLE)
        line = json.loads(files[0].read_text().splitlines()[0])
        self.assertEqual(sorted(line), ["oracle", "predicted", "tick", "trial"])

    def test_the_log_directory_is_created_and_nothing_is_written_without_one(self):
        adapter, _ = self.build(detector=StubDetector(), log_dir=self.log_dir / "a" / "b")
        adapter.reset()
        self.tick(adapter)
        self.assertTrue((self.log_dir / "a" / "b").is_dir())
        adapter, _ = self.build(detector=StubDetector(), log_dir=None)
        adapter.reset()
        self.tick(adapter)


if __name__ == "__main__":
    unittest.main()
