import unittest

import mujoco
import numpy as np

from data_collection.benchmark import (
    capture_state, restore_state, validation_condition, validate_action,
    BenchmarkRecorder, require_complete,
)


class BenchmarkTests(unittest.TestCase):
    def test_conditions_cover_counts_and_retain_pair_on_retry(self):
        conditions = [validation_condition(i) for i in range(6)]
        self.assertEqual([len(c[2]) for c in conditions], [2, 3, 4, 2, 3, 4])
        self.assertEqual(len({(c[0], c[1]) for c in conditions}), 6)
        self.assertEqual(validation_condition(0), validation_condition(0))

    def test_integration_state_round_trip(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="sphere" '
            'size="0.1" mass="1"/></body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        initial = capture_state(model, data)
        mujoco.mj_step(model, data)
        restore_state(model, data, initial)
        np.testing.assert_array_equal(capture_state(model, data), initial)
        with self.assertRaises(ValueError):
            restore_state(model, data, initial[:-1])

    def test_action_rejects_invalid_shape_and_nonfinite_values(self):
        np.testing.assert_array_equal(validate_action(np.zeros(8)), np.zeros(8))
        for bad in (np.zeros(7), np.full(8, np.nan), np.full(8, np.inf)):
            with self.assertRaises(ValueError):
                validate_action(bad)

    def test_unfinished_or_existing_dataset_fails_closed(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises((ValueError, FileNotFoundError)):
                require_complete(root)
            recorder = BenchmarkRecorder(root, {"requested_episodes": 6})
            with self.assertRaises(FileExistsError):
                BenchmarkRecorder(root, {"requested_episodes": 6})
            with self.assertRaises(ValueError):
                recorder.finalize(0, {"train": [0, 1, 2, 3], "validation": [4, 5]})

    def test_fingerprint_detects_data_changes_but_not_result_updates(self):
        import tempfile
        from pathlib import Path
        from validation.gates import dataset_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "replay").mkdir()
            (root / "data/test.parquet").write_bytes(b"original")
            before = dataset_fingerprint(root)
            (root / "replay/replay_result.json").write_text('{"passed":true}')
            self.assertEqual(dataset_fingerprint(root), before)
            (root / "data/test.parquet").write_bytes(b"changed")
            self.assertNotEqual(dataset_fingerprint(root), before)

    def test_fresh_score_does_not_mutate_replay_state(self):
        from pathlib import Path
        from src.robots.franka_panda.config import SCENE_PATH
        from validation.gates import fresh_stack_metrics
        model = mujoco.MjModel.from_xml_path(str(Path(SCENE_PATH)))
        data, scratch = mujoco.MjData(model), mujoco.MjData(model)
        mujoco.mj_step(model, data)
        before = capture_state(model, data)
        result = fresh_stack_metrics(model, data, scratch, "red", "green")
        np.testing.assert_array_equal(capture_state(model, data), before)
        self.assertIn("released_stack", result)


if __name__ == "__main__":
    unittest.main()