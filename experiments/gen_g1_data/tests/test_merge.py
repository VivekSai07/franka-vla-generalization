import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.gen_g1_data.merge import aggregate, link_recorded_files, verify
from experiments.gen_g1_data.record import create_dataset

TASKS = ("put the can in the blue bin", "put the milk in the orange bin", "put the lemon in the blue bin")


def _shard(root, k, episodes, frames=3):
    dataset = create_dataset(root, f"gen_g1_shard_{k:02d}")
    rng = np.random.RandomState(k)
    for e in range(episodes):
        task = TASKS[(k + e) % len(TASKS)]
        for _ in range(frames):
            dataset.add_frame({"observation.state": rng.rand(9).astype(np.float32), "action": rng.rand(8).astype(np.float32),
                               "task": task,
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
        cls.merged = base / "merged"
        aggregate(cls.shards, cls.merged)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_merge_equals_the_shards_in_order(self):
        report = verify(self.shards, self.merged)
        self.assertEqual((report["episodes"], report["frames"], report["videos_verified"]), (5, 15, 10))

    def test_verify_refuses_the_wrong_shard_order(self):
        with self.assertRaises(ValueError):
            verify(list(reversed(self.shards)), self.merged)

    def test_verify_refuses_a_missing_shard(self):
        with self.assertRaises(ValueError):
            verify(self.shards[:1], self.merged)

    def test_verify_refuses_a_video_that_differs_by_one_byte(self):
        video = next((self.merged / "videos/observation.images.image").rglob("*.mp4"))
        original = video.read_bytes()
        try:
            video.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaises(ValueError):
                verify(self.shards, self.merged)
        finally:
            video.write_bytes(original)
        verify(self.shards, self.merged)


def _edit(path, change):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    frame = table.to_pandas()
    change(frame)
    pq.write_table(pa.Table.from_pandas(frame, schema=table.schema, preserve_index=False), path)


class TamperedMergeTests(unittest.TestCase):
    """One thing changed in a copy of the merged dataset: `verify` must name it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.shards = [_shard(base / "s0", 0, 2), _shard(base / "s1", 1, 3)]
        cls.merged = base / "merged"
        aggregate(cls.shards, cls.merged)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tampered(self, name, change, shards=None, target="data"):
        """verify() on a copy of the merged tree where `change(frame)` edited every parquet file under `target`."""
        import shutil
        copy = Path(self.tmp.name) / f"copy_{name}"
        shutil.copytree(self.merged, copy)
        paths = sorted((copy / target).rglob("*.parquet")) if target != "info" else []
        for path in paths:
            _edit(path, change)
        if target == "info":
            info = json.loads((copy / "meta/info.json").read_text())
            change(info)
            (copy / "meta/info.json").write_text(json.dumps(info))
        return lambda: verify(shards or self.shards, copy)

    def test_the_untouched_copy_verifies(self):
        self.tampered("clean", lambda f: None)()

    def test_a_changed_action_state_timestamp_or_frame_index_is_refused(self):
        def bump(column):
            def change(frame):
                frame.at[0, column] = frame[column][0] + 1
            return change
        for column in ("action", "observation.state", "timestamp", "frame_index"):
            with self.assertRaises(ValueError, msg=column):
                self.tampered(f"col_{column}", bump(column))()

    def test_frames_that_name_another_task_are_refused(self):
        def change(frame):
            frame["task_index"] = frame["task_index"].map(lambda t: 1 - t if t in (0, 1) else t)
        with self.assertRaises(ValueError):
            self.tampered("task_index", change)()

    def test_an_index_column_that_is_not_contiguous_is_refused(self):
        def change(frame):
            frame["index"] = frame["index"] + 1
        with self.assertRaises(ValueError):
            self.tampered("index", change)()

    def test_an_episode_whose_task_or_length_or_video_times_changed_is_refused(self):
        def tasks(frame):
            frame.at[0, "tasks"] = ["put the lemon in the blue bin"] if list(frame["tasks"][0]) != ["put the lemon in the blue bin"] else ["x"]

        def length(frame):
            frame.at[0, "length"] = frame["length"][0] + 1

        def times(frame):
            frame.at[0, "videos/observation.images.image2/to_timestamp"] = frame["videos/observation.images.image2/to_timestamp"][0] + 0.5

        for name, change in (("tasks", tasks), ("length", length), ("times", times)):
            with self.assertRaises(ValueError, msg=name):
                self.tampered(name, change, target="meta/episodes")()

    def test_totals_in_info_json_must_match(self):
        for key in ("total_episodes", "total_frames"):
            def change(info, key=key):
                info[key] += 1
            with self.assertRaises(ValueError, msg=key):
                self.tampered(f"info_{key}", change, target="info")()

    def test_the_episode_list_must_match_the_shards_even_when_the_totals_were_made_to_agree(self):
        def change(info):
            info["total_episodes"], info["total_frames"] = 5, 6
        with self.assertRaises(ValueError):
            self.tampered("short", change, shards=self.shards[:1], target="info")()


class LinkTests(unittest.TestCase):
    def shard(self, base, name, files):
        root = Path(base) / name
        for folder in ("episodes", "signals"):
            (root / folder).mkdir(parents=True)
            for n in files:
                (root / folder / f"episode_{n:04d}.npz").write_bytes(f"{folder}{n}".encode())
        return root

    def test_every_shards_files_are_linked_under_their_slot_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [self.shard(tmp, "a", [0, 1]), self.shard(tmp, "b", [2])]
            out = Path(tmp) / "out"
            out.mkdir()
            link_recorded_files(roots, out)
            for folder in ("episodes", "signals"):
                self.assertEqual(sorted(q.name for q in (out / folder).iterdir()),
                                 ["episode_0000.npz", "episode_0001.npz", "episode_0002.npz"])
            self.assertEqual((out / "signals" / "episode_0002.npz").read_bytes(), b"signals2")
            self.assertTrue((out / "episodes" / "episode_0000.npz").samefile(roots[0] / "episodes" / "episode_0000.npz"))

    def test_a_slot_in_two_shards_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [self.shard(tmp, "a", [0, 1]), self.shard(tmp, "b", [1])]
            out = Path(tmp) / "out"
            out.mkdir()
            with self.assertRaises(ValueError):
                link_recorded_files(roots, out)


if __name__ == "__main__":
    unittest.main()
