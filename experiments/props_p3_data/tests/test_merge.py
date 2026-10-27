import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from experiments.props_p3_data import contract
from experiments.props_p3_data import merge as merge_module
from experiments.props_p3_data import plan as p
from experiments.props_p3_data.merge import aggregate, merge_and_seal, verify, VIDEO_KEYS
from experiments.props_p3_data.record import create_dataset
from experiments.props_p3_data.validate import read_dataset_meta

PLAN = p.make_plan()


def _shard(root, k, episodes, frames=3, task=p.PROMPT):
    dataset = create_dataset(root, f"props_p3_shard_{k:02d}")
    rng = np.random.RandomState(k)
    for e in range(episodes):
        for _ in range(frames):
            dataset.add_frame({"observation.state": rng.rand(9).astype(np.float32),
                               "action": rng.rand(8).astype(np.float32), "task": task,
                               "observation.images.image": rng.randint(0, 255, (512, 512, 3), dtype=np.uint8),
                               "observation.images.image2": rng.randint(0, 255, (512, 512, 3), dtype=np.uint8)})
        dataset.save_episode()
        dataset.finalize()
    return Path(root) / "dataset"


class MergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.shards = [_shard(base / "s0", 0, 2), _shard(base / "s1", 1, 3)]
        cls.out = base / "out"
        aggregate(cls.shards, cls.out / "dataset")
        cls.merged = cls.out / "dataset"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tampered(self):
        copy = Path(self.tmp.name) / "tampered"
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(self.merged, copy)
        return copy

    def test_merge_equals_the_shards_in_order(self):
        # Review Focus 5.
        report = verify(self.shards, self.merged)
        self.assertEqual((report["episodes"], report["frames"], report["shards"], report["videos_verified"]), (5, 15, 2, 10))

    def test_the_merged_dataset_carries_only_the_fixed_prompt_and_reads_back(self):
        meta = read_dataset_meta(self.out)
        self.assertEqual(meta["task_table"], [p.PROMPT])
        self.assertEqual(meta["tasks"], [[p.PROMPT]] * 5)
        self.assertEqual((meta["total_episodes"], meta["total_frames"], meta["fps"]), (5, 15, 30))
        self.assertEqual(meta["indices"], [0, 1, 2, 3, 4])
        self.assertEqual(meta["lengths"], [3] * 5)

    def test_verify_refuses_the_wrong_shard_order(self):
        with self.assertRaises(ValueError):
            verify(list(reversed(self.shards)), self.merged)

    def test_verify_refuses_a_missing_shard(self):
        with self.assertRaisesRegex(ValueError, "merged holds"):
            verify(self.shards[:1], self.merged)

    def test_verify_refuses_an_altered_video(self):
        copy = self.tampered()
        video = sorted(copy.glob(f"videos/{VIDEO_KEYS[0]}/chunk-*/file-*.mp4"))[3]
        data = bytearray(video.read_bytes())
        data[len(data) // 2] ^= 0xFF
        video.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, "video"):
            verify(self.shards, copy)

    def test_verify_refuses_an_altered_action_column(self):
        self.assert_column_tamper_refused("action", lambda column: column + 1.0)

    def test_verify_refuses_every_other_altered_column(self):
        self.assert_column_tamper_refused("observation.state", lambda column: column + 1.0)
        self.assert_column_tamper_refused("timestamp", lambda column: column + 1.0)
        self.assert_column_tamper_refused("frame_index", lambda column: column + 1)
        self.assert_column_tamper_refused("index", lambda column: column + 1)

    def assert_column_tamper_refused(self, name, change):
        copy = self.tampered()
        path = sorted(copy.glob("data/chunk-*/file-*.parquet"))[2]
        frame = pd.read_parquet(path)
        column = np.stack(frame[name].to_numpy())
        frame[name] = list(change(column)) if column.ndim > 1 else change(column)
        frame.to_parquet(path)
        with self.assertRaises(ValueError, msg=name):
            verify(self.shards, copy)

    def test_verify_refuses_a_different_task_string_in_a_shard(self):
        other = _shard(Path(self.tmp.name) / "alt0", 0, 2, task="put the can in the orange bin")
        with self.assertRaises(ValueError):
            verify([other, self.shards[1]], self.merged)

    def edit_episode_metadata(self, change):
        copy = self.tampered()
        path = sorted(copy.glob("meta/episodes/chunk-*/file-*.parquet"))[0]
        frame = pd.read_parquet(path)
        change(frame)
        frame.to_parquet(path)
        return copy

    def test_verify_refuses_altered_episode_length_or_task_metadata(self):
        def length(frame):
            frame["length"] = frame["length"] + 1

        def tasks(frame):
            frame["tasks"] = [["put the can in the orange bin"]] * len(frame)
        for change in (length, tasks):
            with self.assertRaisesRegex(ValueError, "length or task"):
                verify(self.shards, self.edit_episode_metadata(change))

    def test_verify_refuses_altered_video_timestamps(self):
        for bound in ("from_timestamp", "to_timestamp"):
            column = f"videos/{VIDEO_KEYS[0]}/{bound}"

            def change(frame, column=column):
                frame[column] = frame[column] + 1.0
            with self.assertRaisesRegex(ValueError, bound):
                verify(self.shards, self.edit_episode_metadata(change))

    def test_verify_refuses_altered_info_totals(self):
        for key in ("total_frames", "total_episodes"):
            copy = self.tampered()
            info = json.loads((copy / "meta/info.json").read_text())
            info[key] += 1
            (copy / "meta/info.json").write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, "totals", msg=key):
                verify(self.shards, copy)

    def test_verify_refuses_a_renamed_task_in_the_merged_task_table(self):
        copy = self.tampered()
        path = copy / "meta/tasks.parquet"
        table = pd.read_parquet(path)
        table.index = pd.Index(["put the can in the orange bin"], name=table.index.name)
        table.to_parquet(path)
        with self.assertRaisesRegex(ValueError, "task strings"):
            verify(self.shards, copy)

    def test_verify_refuses_a_missing_data_row(self):
        copy = self.tampered()
        path = sorted(copy.glob("data/chunk-*/file-*.parquet"))[2]
        pd.read_parquet(path).iloc[:-1].to_parquet(path)
        with self.assertRaisesRegex(ValueError, "data rows"):
            verify(self.shards, copy)


def _fake_shard(root, k, plan=None, provenance=None, done=True, admitted=True, block=None):
    root.mkdir(parents=True)
    slots = p.shard_slots(k if block is None else block, contract.SHARDS)
    (root / "plan.json").write_text(json.dumps(PLAN if plan is None else plan))
    (root / "provenance.json").write_text(json.dumps(contract.provenance() if provenance is None else provenance))
    (root / "slots.json").write_text(json.dumps(slots))
    (root / "journal.jsonl").write_text("".join(json.dumps({"slot": s, "admitted": admitted}) + "\n" for s in slots))
    if done:
        (root / "done.json").write_text("{}")
    return root


class PilotMergeTests(unittest.TestCase):
    def test_a_pilot_merge_aggregates_verifies_and_collects_the_journals_and_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            shards, out = Path(tmp) / "shards", Path(tmp) / "out"
            for k, episodes in ((0, 2), (1, 3)):
                root = shards / f"shard_{k:02d}"
                _shard(root, k, episodes)
                (root / "plan.json").write_text(json.dumps(PLAN))
                (root / "provenance.json").write_text(json.dumps(contract.provenance()))
                (root / "slots.json").write_text(json.dumps(list(range(episodes))))
                lines = [{"slot": s, "admitted": True, "r": 1 if (k, s) == (1, 2) else 0} for s in range(episodes)]
                (root / "journal.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
                (root / "episodes").mkdir()
                for s in range(episodes):
                    np.savez(root / "episodes" / f"episode_{k}{s}.npz", x=np.zeros(1))
            with mock.patch.object(merge_module, "verify", wraps=verify) as spy:
                report = merge_and_seal(shards, out, contract.SHARDS, pilot=True)
            spy.assert_called_once_with([shards / "shard_00" / "dataset", shards / "shard_01" / "dataset"], out / "dataset")
            self.assertEqual((report["episodes"], report["frames"]), (5, 15))
            self.assertEqual(verify([shards / "shard_00/dataset", shards / "shard_01/dataset"], out / "dataset")["episodes"], 5)
            self.assertEqual(len((out / "journal.jsonl").read_text().splitlines()), 5)
            self.assertEqual(sorted(q.name for q in (out / "episodes").iterdir()),
                             ["episode_00.npz", "episode_01.npz", "episode_10.npz", "episode_11.npz", "episode_12.npz"])
            self.assertEqual(json.loads((out / "plan.json").read_text()), json.loads(json.dumps(PLAN)))
            self.assertEqual(json.loads((out / "provenance.json").read_text()), contract.provenance())
            self.assertEqual(json.loads((out / "substitutions.json").read_text()), {"2": 1})
            self.assertEqual(json.loads((out / "merge.json").read_text())["shard_roots"], ["shard_00", "shard_01"])
            self.assertFalse((out / "seal.json").exists())


class MergeAndSealGuardTests(unittest.TestCase):
    """The checks `merge_and_seal` makes before it aggregates anything."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.shards, self.out = Path(self.tmp.name) / "shards", Path(self.tmp.name) / "out"
        self.shards.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def full(self, bad=None):
        bad = bad or {}
        for k in range(contract.SHARDS):
            _fake_shard(self.shards / f"shard_{k:02d}", k, **bad.get(k, {}))

    def test_a_shard_without_done_json_stops_a_full_merge(self):
        self.full({7: {"done": False}})
        with self.assertRaisesRegex(SystemExit, "incomplete"):
            merge_and_seal(self.shards, self.out, contract.SHARDS)
        self.assertFalse(self.out.exists())

    def test_a_shard_that_did_not_admit_its_whole_block_stops_a_full_merge(self):
        self.full({3: {"admitted": False}})
        with self.assertRaisesRegex(SystemExit, "incomplete"):
            merge_and_seal(self.shards, self.out, contract.SHARDS)

    def test_a_shard_holding_another_shards_block_stops_a_full_merge(self):
        self.full({3: {"block": 4}})
        with self.assertRaisesRegex(SystemExit, "incomplete or not shard 3"):
            merge_and_seal(self.shards, self.out, contract.SHARDS)

    def test_a_shard_from_another_plan_stops_the_merge(self):
        self.full({2: {"plan": {**PLAN, "prompt": "put the can in the orange bin"}}})
        with self.assertRaisesRegex(SystemExit, "plan differs"):
            merge_and_seal(self.shards, self.out, contract.SHARDS)

    def test_a_shard_collected_from_other_source_stops_the_merge(self):
        self.full({5: {"provenance": {"experiments/props_p3_data/plan.py": "0" * 64}}})
        with self.assertRaisesRegex(SystemExit, "provenance differs"):
            merge_and_seal(self.shards, self.out, contract.SHARDS)

    def test_an_existing_output_is_never_overwritten(self):
        self.full()
        self.out.mkdir()
        with self.assertRaisesRegex(SystemExit, "No overwrite"):
            merge_and_seal(self.shards, self.out, contract.SHARDS)


if __name__ == "__main__":
    unittest.main()
