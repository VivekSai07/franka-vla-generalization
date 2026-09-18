"""Review regressions: real persistence/media, synthetic RGB, no native simulation."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.fixed_layout import record, source, validate
from experiments.fixed_layout.contract import MIN_FREE_BYTES, TASK
from experiments.paired_demonstrations.media import CAMERAS, ENCODING, FEATURES, PairedDataset
from experiments.paired_demonstrations.media import pixels_digest


def synthetic_frame(episode, tick):
    images = {}
    for camera, key in enumerate(CAMERAS):
        rgb = np.zeros((512, 512, 3), np.uint8)
        rgb[:, :, camera] = 80 + episode * 20
        rgb[:, (tick % 32) * 16:(tick % 32 + 1) * 16, :] = 240
        images[key] = rgb
    return {"action": np.full(8, episode, np.float32),
            "observation.state": np.full(9, episode, np.float32), "task": TASK, **images}


def create_dataset(root, cls=PairedDataset):
    dataset = cls.create(repo_id="synthetic_fixed_review", root=root / "dataset", fps=30,
        features=copy.deepcopy(FEATURES), vcodec=ENCODING["vcodec"],
        encoder_threads=ENCODING["encoder_threads"], metadata_buffer_size=1)
    dataset.meta.info["data_files_size_in_mb"] = 0
    dataset.meta.info["video_files_size_in_mb"] = 0
    return dataset


class ReviewTests(unittest.TestCase):
    def test_creation_binding_rejects_wrong_camera_unrelated_and_reordered_pngs(self):
        from PIL import Image
        for corruption in ("wrong_camera", "unrelated", "reordered"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                dataset = create_dataset(root, record.PairedDataset)
                attempt = root / "attempts/000000"
                attempt.mkdir(parents=True)
                sink = record.TickSink(attempt, dataset, lambda: None)
                for tick in range(4):
                    frame = synthetic_frame(0, tick)
                    sink(frame, {"image_sha256": np.array([pixels_digest(frame[key]) for key in CAMERAS])})
                first, second = CAMERAS
                paths = [dataset._get_image_file_path(0, first, i) for i in range(4)]
                if corruption == "wrong_camera":
                    paths[0].write_bytes(dataset._get_image_file_path(0, second, 0).read_bytes())
                elif corruption == "unrelated":
                    Image.fromarray(np.full((512, 512, 3), 33, np.uint8)).save(paths[0])
                else:
                    a, b = paths[0].read_bytes(), paths[1].read_bytes()
                    paths[0].write_bytes(b)
                    paths[1].write_bytes(a)
                try:
                    with self.assertRaisesRegex(ValueError, "[Ee]ncoder.*(source|input)|PNG"):
                        dataset.save_episode()
                finally:
                    dataset.finalize()

    def test_creation_binding_pins_raw_order_codec_and_final_video_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = create_dataset(root, record.PairedDataset)
            attempt = root / "attempts/000000"
            attempt.mkdir(parents=True)
            sink = record.TickSink(attempt, dataset, lambda: None)
            expected = {key: [] for key in CAMERAS}
            for tick in range(4):
                frame = synthetic_frame(0, tick)
                hashes = [pixels_digest(frame[key]) for key in CAMERAS]
                for key, digest in zip(CAMERAS, hashes):
                    expected[key].append(digest)
                sink(frame, {"image_sha256": np.array(hashes)})
            try:
                dataset.save_episode()
                dataset.finalize()
                for key in CAMERAS:
                    evidence = attempt / "media" / f"{key}.json"
                    self.assertTrue(evidence.is_file(), "creation-time encoder binding missing")
                    binding = json.loads(evidence.read_text())
                    self.assertEqual(binding["encoding"], ENCODING)
                    self.assertEqual([f["raw_pixel_sha256"] for f in binding["frames"]], expected[key])
                    self.assertEqual([f["frame_index"] for f in binding["frames"]], list(range(4)))
                    video = root / "dataset" / binding["video_path"]
                    self.assertEqual(binding["video_sha256"], record.file_hash(video))
                    self.assertEqual(validate.validate_video(video, [(0., 4 / 30, 4)], lambda: None), 4)
                    trace = {"image_sha256": np.array(list(zip(*expected.values())))}
                    validate._media_binding(attempt, key, binding["video_path"], (0., 4 / 30, 4), trace, lambda: None)
                    original = video.read_bytes()
                    video.write_bytes(original + b"changed-media")
                    with self.assertRaisesRegex(ValueError, "media bytes"):
                        validate._media_binding(attempt, key, binding["video_path"], (0., 4 / 30, 4), trace, lambda: None)
                    video.write_bytes(original)
                    changed = copy.deepcopy(binding)
                    changed["frames"].reverse()
                    evidence.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, "frame-order"):
                        validate._media_binding(attempt, key, binding["video_path"], (0., 4 / 30, 4), trace, lambda: None)
                    evidence.write_text(json.dumps(binding))
            finally:
                dataset.finalize()

    def test_output_filesystem_before_creation_and_after_each_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            output = parent / "new-parent/run"
            def usage(path):
                return SimpleNamespace(free=MIN_FREE_BYTES - 1 if Path(path) == parent else MIN_FREE_BYTES + 1)
            with patch.object(record, "disk_usage", side_effect=usage), patch.object(record, "OUTPUT", output), \
                    patch.object(record, "Expert", side_effect=AssertionError("repo disk must not admit output")):
                with self.assertRaisesRegex(ValueError, "30 GiB"):
                    record.collect()
            self.assertFalse(output.parent.exists())
            with patch.object(record, "disk_usage", return_value=SimpleNamespace(free=MIN_FREE_BYTES)) as disk:
                budget = record.Budget(output)
                budget.check()
                self.assertEqual(disk.call_args.args[0], parent)
                output.mkdir(parents=True)
                budget.check()
                self.assertEqual(disk.call_args.args[0], output)
                disk.return_value.free -= 1
                with self.assertRaises(ValueError):
                    budget.check()

    def test_preflight_reports_destination_not_repo_free_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "missing/run"
            def usage(path):
                return SimpleNamespace(free=MIN_FREE_BYTES - 1 if Path(path) == Path(tmp) else MIN_FREE_BYTES + 1)
            with patch.object(source, "OUTPUT", output), patch.object(source, "disk_usage", side_effect=usage), \
                    patch.object(source, "provenance", side_effect=ValueError("no source work in storage test")):
                result = source.preflight()
            self.assertFalse(result["storage_eligible"])
            self.assertEqual(result["free_bytes"], MIN_FREE_BYTES - 1)
            self.assertFalse(output.parent.exists())

    def test_low_repo_space_does_not_replace_healthy_destination_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            output = parent / "missing/run"
            def usage(path):
                return SimpleNamespace(free=MIN_FREE_BYTES if Path(path) == parent else 0)
            with patch.object(record, "disk_usage", side_effect=usage):
                record.Budget(output).check()
            with patch.object(source, "OUTPUT", output), patch.object(source, "disk_usage", side_effect=usage), \
                    patch.object(source, "provenance", side_effect=ValueError("no source work in storage test")):
                result = source.preflight()
            self.assertTrue(result["storage_eligible"])
            self.assertEqual(result["free_bytes"], MIN_FREE_BYTES)
            self.assertFalse(output.parent.exists())

    def test_parquet_nonfinite_interval_endpoints_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.parquet"
            for bad in (float("nan"), float("inf"), float("-inf")):
                for endpoint in (0, 1):
                    interval = [0., 528 / 30]
                    interval[endpoint] = bad
                    pq.write_table(pa.table({"start": [interval[0]], "end": [interval[1]]}), path)
                    row = pq.read_table(path).to_pylist()[0]
                    with self.subTest(bad=bad, endpoint=endpoint), \
                            patch.object(validate.av, "open", side_effect=AssertionError("invalid interval reached decoder")), \
                            self.assertRaisesRegex(ValueError, "finite"):
                        validate.validate_video(Path(tmp) / "unused.mp4", [(row["start"], row["end"], 528)], lambda: None)

    def test_interval_scalar_type_spoofs_rejected(self):
        for value in (True, "0", None, np.array([0.])):
            with self.subTest(value=value), patch.object(validate.av, "open", side_effect=AssertionError("decoder reached")), \
                    self.assertRaisesRegex(ValueError, "finite"):
                validate.validate_video("unused", [(value, 17.6, 528)], lambda: None)

    def test_per_file_intervals_reject_shared_camera_file_before_next_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episodes = []
            for i in range(2):
                ep = {"episode_index": i, "length": 528, "data/chunk_index": 0, "data/file_index": i}
                for key in CAMERAS:
                    ep.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                               f"videos/{key}/from_timestamp": 0., f"videos/{key}/to_timestamp": 17.6})
                episodes.append(ep)
            pq.write_table(pa.Table.from_pylist(episodes), root / "episodes.parquet")
            metadata = pq.read_table(root / "episodes.parquet").to_pylist()
            info = {"data_path": "data/{chunk_index}/{file_index}.parquet",
                    "video_path": "videos/{video_key}/{chunk_index}/{file_index}.mp4"}
            with patch.object(validate, "read_rows", return_value=(None, None, None)), \
                    patch.object(validate, "load_attempt", return_value=({}, {}, {})), \
                    patch.object(validate, "validate_video", return_value=528), \
                    self.assertRaisesRegex(ValueError, "[Ii]ndependent|[Ss]hared|[Oo]verlap"):
                validate._episode_inputs(root, {"episode_index": 0}, None, None, lambda: None, info, metadata)

    def test_two_episode_real_paired_dataset_finalize_read_continue_reopen(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = create_dataset(root)
            first = {}
            try:
                for episode in range(2):
                    for tick in range(528):
                        dataset.add_frame(synthetic_frame(episode, tick))
                    dataset.save_episode()
                    dataset.finalize()
                    dataset._writer_closed_for_reading = True
                    info, metadata = validate._metadata(root, episode + 1)
                    for index, ep in enumerate(metadata):
                        rows = root / "dataset" / info["data_path"].format(
                            chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])
                        actions, states, _ = validate.read_rows(rows, index)
                        np.testing.assert_array_equal(actions, np.full((528, 8), index, np.float32))
                        np.testing.assert_array_equal(states, np.full((528, 9), index, np.float32))
                        for key in CAMERAS:
                            prefix = "videos/" + key
                            video = root / "dataset" / info["video_path"].format(video_key=key,
                                chunk_index=ep[prefix + "/chunk_index"], file_index=ep[prefix + "/file_index"])
                            interval = (ep[prefix + "/from_timestamp"], ep[prefix + "/to_timestamp"], 528)
                            self.assertEqual(validate.validate_video(video, [interval], lambda: None), 528)
                    if episode == 0:
                        first = {p: record.file_hash(p) for folder in ("data", "videos", "meta/episodes")
                                 for p in (root / "dataset" / folder).rglob("*") if p.is_file()}
                    else:
                        self.assertEqual(first, {p: record.file_hash(p) for p in first})
                    # Reopen using the actual installed reader, not the writer's cache.
                    reopened = LeRobotDataset(repo_id="synthetic_fixed_review", root=root / "dataset", video_backend="pyav")
                    self.assertEqual(len(reopened), (episode + 1) * 528)
                    self.assertEqual(reopened.meta.total_episodes, episode + 1)
                    self.assertEqual(int(reopened[episode * 528]["episode_index"]), episode)
                    for key in CAMERAS:
                        self.assertEqual(tuple(reopened[episode * 528][key].shape), (3, 512, 512))
                    reopened.finalize()
                self.assertEqual(len(list((root / "dataset/data").rglob("*.parquet"))), 2)
                self.assertEqual(len(list((root / "dataset/videos").rglob("*.mp4"))), 4)
                self.assertEqual(len(first), 4)
            finally:
                dataset.finalize()


if __name__ == "__main__":
    unittest.main()