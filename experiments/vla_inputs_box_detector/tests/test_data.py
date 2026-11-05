"""The detector's frame cache and training set: the cache format and its verification, exact augmentation, determinism, the split, the
disk guard, and (where the sealed data is mounted) that the cache holds what the dataset decodes and S0 labelled.

Admission image, CPU. The real-data tests need `outputs/props_p2_data` and `outputs/vla_inputs_signals` mounted and skip without them.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.vla_inputs_box_detector import constants as k
from experiments.vla_inputs_box_detector import data, inputs, model

ROWS = [(0, 0), (0, 8), (1, 0), (1, 8), (480, 0), (480, 8), (481, 0), (481, 8)]
OBJECT_OF_EPISODE = {0: 3, 1: 0, 480: 5, 481: 2}


def box_for(row):
    """A deterministic in-frame box pair (front, gripper) per row, inside the front crop."""
    r = np.random.default_rng(row)
    x0, y0 = r.uniform(0.15, 0.6), r.uniform(0.4, 0.55)
    gx0, gy0 = r.uniform(0.0, 0.5), r.uniform(0.2, 0.6)
    return np.array([[x0, y0, x0 + r.uniform(0.04, 0.1), y0 + r.uniform(0.05, 0.12), 1.0],
                     [gx0, gy0, gx0 + r.uniform(0.15, 0.4), gy0 + r.uniform(0.15, 0.35), 1.0]])


def synthetic(directory, rows=ROWS):
    """A tiny cache: black frames with a white rectangle exactly where the label says the box is."""
    n = len(rows)
    arrays = data.create_arrays(directory, n)
    labels = np.stack([box_for(r) for r in range(n)])
    for i, (episode, frame) in enumerate(rows):
        front, gripper = np.zeros((*k.FRONT_SHAPE, 3), np.uint8), np.zeros((*k.GRIPPER_SHAPE, 3), np.uint8)
        x0, y0, x1, y1 = np.round(inputs.front_box_to_input(labels[i, 0])).astype(int)
        front[y0:y1, x0:x1] = 255
        x0, y0, x1, y1 = np.round(inputs.gripper_box_to_input(labels[i, 1])).astype(int)
        gripper[y0:y1, x0:x1] = 255
        arrays["front"][i], arrays["gripper"][i] = front, gripper
        arrays["labels"][i] = labels[i]
        arrays["index"][i] = (episode, frame)
        arrays["objects"][i] = OBJECT_OF_EPISODE[episode]
    data.finish(directory, arrays, dict(source="synthetic", episodes=[]))
    return directory


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / "cache"
        synthetic(self.directory)


class CacheFormat(Fixture):
    def test_round_trip_and_the_pinned_shapes(self):
        cache = data.open_cache(self.directory, verify=True)
        self.assertEqual(cache.front.shape, (8, *k.FRONT_SHAPE, 3))
        self.assertEqual(cache.gripper.shape, (8, *k.GRIPPER_SHAPE, 3))
        self.assertEqual((cache.labels.shape, cache.labels.dtype), ((8, 2, 5), np.float64))
        self.assertEqual(cache.index.tolist(), [list(r) for r in ROWS])
        self.assertEqual(cache.objects.tolist(), [OBJECT_OF_EPISODE[e] for e, _ in ROWS])
        self.assertEqual(cache.front.dtype, np.uint8)
        self.assertEqual(cache.manifest["stride"], k.STRIDE_K)

    def test_the_manifest_records_a_hash_of_every_array_and_the_constants(self):
        manifest = json.loads((self.directory / "manifest.json").read_text())
        for name in ("front", "gripper", "labels", "index", "objects"):
            self.assertEqual(manifest["arrays"][name]["sha256"], hashlib.sha256((self.directory / f"{name}.npy").read_bytes()).hexdigest())
        self.assertEqual(manifest["constants"]["front_crop"], list(k.FRONT_CROP))

    def test_a_changed_byte_a_missing_marker_or_another_stride_is_refused(self):
        with (self.directory / "front.npy").open("r+b") as stream:
            stream.seek(-5, 2)
            stream.write(b"\x07")
        with self.assertRaisesRegex(ValueError, "front"):
            data.open_cache(self.directory, verify=True)
        data.open_cache(self.directory, verify=False)  # the fast open does not hash
        (self.directory / "complete").unlink()
        with self.assertRaisesRegex(ValueError, "complete"):
            data.open_cache(self.directory, verify=False)

    def test_a_cache_made_under_other_constants_is_refused(self):
        path = self.directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["stride"] = 4
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "stride|constants"):
            data.open_cache(self.directory, verify=False)

    def test_a_cache_directory_is_never_reused(self):
        with self.assertRaises(FileExistsError):
            data.create_arrays(self.directory, 3)

    def test_the_split_is_the_plans(self):
        cache = data.open_cache(self.directory)
        self.assertEqual(data.rows_of(cache, "train").tolist(), [0, 1, 2, 3])
        self.assertEqual(data.rows_of(cache, "validation").tolist(), [4, 5, 6, 7])
        with self.assertRaises(ValueError):
            data.rows_of(cache, "test")


class TrainingSet(Fixture):
    def setUp(self):
        super().setUp()
        self.cache = data.open_cache(self.directory)

    def test_it_refuses_a_validation_row_and_the_batches_never_contain_one(self):
        with self.assertRaisesRegex(ValueError, "validation"):
            data.TrainingSet(self.cache, np.array([0, 4]))
        training = data.TrainingSet(self.cache, data.rows_of(self.cache, "train"))
        for g in range(len(training) * 6):
            self.assertLess(int(training.sample_row(g)), 4)
            self.assertLess(training.episode(g), 480)

    def test_a_sample_is_deterministic_from_its_global_id(self):
        training = data.TrainingSet(self.cache, data.rows_of(self.cache, "train"))
        a, b = training[7], training[7]
        np.testing.assert_array_equal(a["front"], b["front"])
        self.assertEqual(a["front_box"] is None, b["front_box"] is None)
        self.assertEqual(training[7 + len(training)]["cls"], training[7]["cls"])

    def test_a_new_epoch_draws_new_shifts_and_about_the_pinned_share_is_shifted(self):
        training = data.TrainingSet(self.cache, data.rows_of(self.cache, "train"))
        shifted = {0: 0, 1: 0}
        total = 0
        for epoch in range(300):
            for i in range(len(training)):
                sample = training[epoch * len(training) + i]
                shifted[0] += int(sample["front_shift"] != (0, 0))
                shifted[1] += int(sample["gripper_shift"] != (0, 0))
                total += 1
        for camera in (0, 1):
            self.assertAlmostEqual(shifted[camera] / total, k.SHIFT_PROBABILITY, delta=0.06)
        self.assertNotEqual(training[0]["front_shift"], training[len(training)]["front_shift"])

    def test_the_shift_stays_within_the_pinned_fraction_of_the_frame(self):
        training = data.TrainingSet(self.cache, data.rows_of(self.cache, "train"))
        limit_front = (int(k.SHIFT_MAX_FRACTION * k.FRONT_SHAPE[1]), int(k.SHIFT_MAX_FRACTION * k.FRONT_SHAPE[0]))
        for g in range(400):
            dx, dy = training[g]["front_shift"]
            self.assertLessEqual(abs(dx), limit_front[0])
            self.assertLessEqual(abs(dy), limit_front[1])

    def test_image_and_box_agree_after_every_augmentation(self):
        """The cache has a white rectangle exactly at the box; whatever the shift, the rectangle's extent is the sample's box, or absent."""
        training = data.TrainingSet(self.cache, data.rows_of(self.cache, "train"))
        absent = {0: 0, 1: 0}
        for g in range(600):
            sample = training[g]
            for camera, image, box in ((0, sample["front"], sample["front_box"]), (1, sample["gripper"], sample["gripper_box"])):
                ys, xs = np.nonzero(image[..., 0])
                if box is None:
                    absent[camera] += 1
                    self.assertTrue(len(xs) == 0 or xs.max() + 1 - xs.min() < k.MIN_VISIBLE_PX or ys.max() + 1 - ys.min() < k.MIN_VISIBLE_PX)
                else:
                    seen = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)
                    np.testing.assert_allclose(seen, box, atol=1.0)  # the label is not integer; the rectangle was drawn on rounded coordinates
        self.assertGreater(absent[0], 0)
        self.assertGreater(absent[1], 0)  # the augmentation does produce frames without the object

    def test_an_unshifted_sample_is_the_cached_frame_and_the_label_of_s0(self):
        training = data.TrainingSet(self.cache, data.rows_of(self.cache, "train"), shift_probability=0.0)
        sample = training[2]
        np.testing.assert_array_equal(sample["front"], self.cache.front[2])
        np.testing.assert_allclose(sample["front_box"], inputs.front_box_to_input(self.cache.labels[2, 0]))
        np.testing.assert_allclose(sample["gripper_box"], inputs.gripper_box_to_input(self.cache.labels[2, 1]))
        self.assertEqual(sample["cls"], OBJECT_OF_EPISODE[1])
        self.assertEqual(sample["front_shift"], (0, 0))

    def test_the_cached_frame_is_not_modified_by_augmentation(self):
        before = self.cache.front[:4].copy()
        training = data.TrainingSet(self.cache, data.rows_of(self.cache, "train"))
        for g in range(80):
            training[g]
        np.testing.assert_array_equal(self.cache.front[:4], before)

    def test_collate_builds_the_batch_the_loss_expects(self):
        training = data.TrainingSet(self.cache, data.rows_of(self.cache, "train"), shift_probability=0.0)
        batch = data.collate([training[i] for i in range(4)])
        self.assertEqual((tuple(batch["front"].shape), batch["front"].dtype), ((4, *k.FRONT_SHAPE, 3), torch.uint8))
        self.assertEqual(tuple(batch["gripper"].shape), (4, *k.GRIPPER_SHAPE, 3))
        self.assertEqual(tuple(batch["front_targets"]["heat"].shape), (4, 72, 120))
        self.assertEqual(tuple(batch["gripper_targets"]["heat"].shape), (4, 64, 64))
        self.assertEqual(batch["cls"].tolist(), [OBJECT_OF_EPISODE[0], OBJECT_OF_EPISODE[0], OBJECT_OF_EPISODE[1], OBJECT_OF_EPISODE[1]])
        self.assertEqual(batch["front_targets"]["cls"].tolist(), batch["cls"].tolist())
        self.assertTrue(bool(batch["front_targets"]["present"].all()))
        loss = model.loss({"heat": torch.zeros(4, 6, 72, 120), "offset": torch.zeros(4, 2, 72, 120), "size": torch.zeros(4, 2, 72, 120)},
                          batch["front_targets"])
        self.assertTrue(torch.isfinite(loss["total"]))


class Photometric(unittest.TestCase):
    def test_it_stays_in_range_keeps_the_shape_and_is_reproducible(self):
        x = torch.rand(4, 3, 8, 8)
        a = data.photometric(x, torch.Generator().manual_seed(1))
        b = data.photometric(x, torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a.shape, x.shape)
        self.assertGreaterEqual(float(a.min()), 0.0)
        self.assertLessEqual(float(a.max()), 1.0)
        self.assertFalse(torch.equal(a, x))
        self.assertLess(float((a - x).abs().max()), 0.25)

    def test_the_input_is_not_modified(self):
        x = torch.rand(2, 3, 4, 4)
        before = x.clone()
        data.photometric(x, torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(x, before))


class ToUint8(unittest.TestCase):
    def test_every_grey_level_survives_the_float_round_trip(self):
        levels = torch.arange(256, dtype=torch.float32)
        pixels = (levels / 255).reshape(1, 1, 1, 256).repeat(1, 3, 1, 1)
        out = data.to_uint8(pixels)
        self.assertEqual(out.shape, (1, 1, 256, 3))
        np.testing.assert_array_equal(out[0, 0, :, 0], np.arange(256))


class DiskGuard(unittest.TestCase):
    def test_the_needed_bytes_are_the_arrays_exactly(self):
        n = 30_780
        per_row = int(np.prod(k.FRONT_SHAPE)) * 3 + int(np.prod(k.GRIPPER_SHAPE)) * 3 + 2 * 5 * 8 + 2 * 4 + 1
        self.assertEqual(data.needed_bytes(n), n * per_row + data.RESERVE_BYTES)
        self.assertLess(data.needed_bytes(n), 20 * 2**30)  # about 19 GB for all 540 episodes

    def test_extraction_refuses_when_the_cache_would_leave_less_than_the_guard_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "cache"
            need = data.needed_bytes(30_780)
            with self.assertRaisesRegex(RuntimeError, "free"):
                data.extract(target, free_bytes=lambda path: need + k.CACHE_DISK_GUARD_GIB * 2**30 - 1)
            self.assertFalse(target.exists())  # nothing was created

    def test_extraction_refuses_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                data.extract(Path(tmp), free_bytes=lambda path: 10**15)


def real_data_mounted():
    try:
        from experiments.xvla_p2_force_box import boxes, contract
        return (contract.DATA / "meta/info.json").is_file() and (boxes.forces.SIDECAR / "seal.json").is_file()
    except Exception:
        return False


@unittest.skipUnless(real_data_mounted(), "the sealed P2 dataset and the S0 sidecar are not mounted")
class RealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name) / "cache"
        data.extract(cls.directory, episodes=(3, 481), workers=1, free_bytes=lambda path: 10**15)
        cls.cache = data.open_cache(cls.directory, verify=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_it_holds_every_stride_th_frame_of_each_episode_in_order(self):
        self.assertEqual(self.cache.index.tolist(), [[e, f] for e in (3, 481) for f in k.CACHE_FRAMES])
        self.assertEqual(len(self.cache.front), 2 * 57)

    def test_labels_are_s0s_target_box_of_that_episode_and_frame(self):
        from experiments.xvla_p2_force_box import boxes
        for row, (episode, frame) in enumerate(self.cache.index):
            self.assertEqual(self.cache.labels[row].tobytes(), boxes.table()[episode, frame].tobytes())

    def test_the_object_is_the_one_the_plans_instruction_names(self):
        from experiments.xvla_p2_force_box import contract
        for row, (episode, _) in enumerate(self.cache.index):
            self.assertEqual(int(self.cache.objects[row]), inputs.parse_instruction(contract.planned_instructions()[episode]))

    def test_frames_equal_what_the_dataset_decodes_for_the_same_row(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from experiments.xvla_p2_force_box import contract
        ds = LeRobotDataset(repo_id=contract.DATA.name, root=contract.DATA)
        for row in (0, 1, 56, 57, 113):
            episode, frame = (int(v) for v in self.cache.index[row])
            item = ds[episode * 452 + frame]
            renders = [(item[key] * 255).round().to(torch.uint8).permute(1, 2, 0).numpy() for key in contract.CAMERAS]
            front, gripper = inputs.preprocess(renders)
            np.testing.assert_array_equal(self.cache.front[row], front)
            np.testing.assert_array_equal(self.cache.gripper[row], gripper)

    def test_the_manifest_records_each_episodes_source_videos(self):
        from experiments.xvla_p2_force_box import contract
        for record in self.cache.manifest["episodes"]:
            for key, digest in record["videos"].items():
                path = contract.DATA / "videos" / key / "chunk-000" / f"file-{record['episode']:03d}.mp4"
                self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual([r["episode"] for r in self.cache.manifest["episodes"]], [3, 481])

    def test_the_gripper_boxes_are_clipped_geometry_and_every_label_is_in_frame(self):
        self.assertTrue((self.cache.labels[:, :, 4] == 1.0).all())


class NoOverlapWithTheEvaluationLayouts(unittest.TestCase):
    def test_no_p2_episode_shares_a_start_pose_with_any_m0_layout(self):
        plan_path = Path(__file__).resolve().parents[3] / "outputs/vla_inputs_eval/m0_20261030/plan/plan.json"
        p2_path = Path(__file__).resolve().parents[3] / "outputs/props_p2_data/props_p2_20261013/plan.json"
        if not (plan_path.is_file() and p2_path.is_file()):
            self.skipTest("the M0 plan or the P2 plan is not mounted")
        m0 = json.loads(plan_path.read_text())
        p2 = json.loads(p2_path.read_text())
        seen = {tuple(np.round(e["start"], 6)) for e in p2["episodes"] if "start" in e}
        self.assertGreater(len(seen), 100, "the P2 plan has no start poses to compare; the test would prove nothing")
        for layout in m0["layouts"].values():
            self.assertNotIn(tuple(np.round(layout["start"], 6)), seen, layout["layout_id"])


if __name__ == "__main__":
    unittest.main()
