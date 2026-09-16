import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_collection.pilot_plan import make_plan
from data_collection.pilot_integrity import scene_digest, validate_accepted_slots, validate_expert_commands


class PilotIntegrityTests(unittest.TestCase):
    def test_scene_digest_ignores_time_but_not_geometry(self):
        qpos = np.arange(12, dtype=np.float64)
        self.assertEqual(scene_digest(qpos), scene_digest(qpos.copy()))
        changed = qpos.copy()
        changed[0] += .01
        self.assertNotEqual(scene_digest(qpos), scene_digest(changed))

    def test_plan_acceptance_enforces_counts_identity_and_seed(self):
        plan = make_plan("preparation", 20260915)
        entries = [{"episode_index": i, "slot_id": s["slot_id"], "split": s["split"],
                    "pick": s["pick"], "place": s["place"], "colors": s["colors"],
                    "seed": s["retry_seeds"][0], "scene_hash": str(i)} for i, s in enumerate(plan["slots"])]
        validate_accepted_slots(plan, entries)
        for mutation in ("missing", "seed", "identity", "duplicate"):
            bad = copy.deepcopy(entries)
            if mutation == "missing": bad.pop()
            if mutation == "seed": bad[0]["seed"] = -1
            if mutation == "identity": bad[0]["colors"] = ["blue", "green"]
            if mutation == "duplicate": bad[-1]["scene_hash"] = bad[0]["scene_hash"]
            with self.assertRaises(ValueError): validate_accepted_slots(plan, bad)

    def test_expert_commands_reject_bounds_nan_and_gripper_encoding(self):
        bounds = np.array([[-1., 1.]] * 7)
        valid = np.zeros((3, 8))
        valid[1, 7] = 255
        validate_expert_commands(valid, bounds)
        for column, value in ((0, 2.), (1, np.nan), (7, 127.)):
            bad = valid.copy(); bad[0, column] = value
            with self.assertRaises(ValueError): validate_expert_commands(bad, bounds)

    def test_truncated_video_is_rejected(self):
        from validation.validate_pilot import decode_video
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.mp4"
            path.write_bytes(b"not a video")
            with self.assertRaises(Exception): decode_video(path, 30, (512, 512))

    def test_episode_video_intervals_reject_overlap_gap_and_wrong_counts(self):
        from validation.validate_pilot import validate_video_intervals
        validate_video_intervals([(0., 1., 30), (1., 2., 30)], 60, 30)
        for intervals, count in (([(0., 1., 30), (0., 1., 30)], 60),
                                 ([(0., 1., 30), (2., 3., 30)], 60),
                                 ([(0., 1., 30)], 29)):
            with self.assertRaises(ValueError): validate_video_intervals(intervals, count, 30)


if __name__ == "__main__": unittest.main()