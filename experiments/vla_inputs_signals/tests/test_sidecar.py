import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_collection.benchmark import capture_state
from experiments.vla_inputs_signals import contract, replay, sidecar, signals

FRAMES = 6
EPISODES = 4
ROOT = Path(__file__).resolve().parents[3]


def fake(index, frames=FRAMES):
    """A well-formed episode without simulating one: exercises merge, report, seal and verify."""
    rng = np.random.default_rng(index)
    return {
        "episode_index": np.array(index), "frame_index": np.arange(frames),
        "wrist_wrench": rng.normal(size=(frames, 6)), "finger_force": np.abs(rng.normal(size=(frames, 2))),
        "target_box": rng.random((frames, 2, 5)), "props_box": rng.random((frames, 3, 2, 5)),
        "prop_names": np.array(["can", "milk", "bread"]),
        "segment": np.array([0, 0, 1, 2, 2, 3][:frames], dtype=np.int8),
        "phase_index": np.arange(frames), "object": np.array("can" if index % 2 == 0 else "milk"),
        "split": np.array("train" if index < 3 else "validation"),
    }


def build(tmp, shards=2, episodes=EPISODES, builder=fake):
    work = Path(tmp) / "shards"
    for k in range(shards):
        sidecar.build_shard(work, k, shards, episodes=episodes, frames=FRAMES, builder=builder)
    return work


def pipeline(tmp, shards=2, episodes=EPISODES):
    work = build(tmp, shards, episodes)
    out = Path(tmp) / "out"
    sidecar.merge(work, out, shards, episodes=episodes, frames=FRAMES)
    sidecar.report(out)
    return work, out


class Shards(unittest.TestCase):
    def test_a_shard_holds_its_own_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = build(tmp)
            self.assertEqual(sorted(p.name for p in (work / "shard_01/episodes").glob("*.npz")),
                             ["episode_001.npz", "episode_003.npz"])

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = build(tmp)
            with self.assertRaises(FileExistsError):
                sidecar.build_shard(work, 0, 2, episodes=EPISODES, frames=FRAMES, builder=fake)

    def test_malformed_arrays_are_rejected_before_they_are_written(self):
        def shifted(index):
            arrays = fake(index)
            arrays["frame_index"] = arrays["frame_index"] + 1
            return arrays

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                sidecar.build_shard(Path(tmp) / "s", 0, 2, episodes=EPISODES, frames=FRAMES, builder=shifted)
            self.assertEqual(list((Path(tmp) / "s").rglob("*.npz")), [])

    def test_wrong_episode_index_is_rejected(self):
        def wrong(index):
            return fake(index + 1)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                sidecar.build_shard(Path(tmp) / "s", 0, 2, episodes=EPISODES, frames=FRAMES, builder=wrong)


class Merge(unittest.TestCase):
    def test_merge_refuses_a_missing_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "shards"
            sidecar.build_shard(work, 0, 2, episodes=EPISODES, frames=FRAMES, builder=fake)
            with self.assertRaises((ValueError, FileNotFoundError, SystemExit)):
                sidecar.merge(work, Path(tmp) / "out", 2, episodes=EPISODES, frames=FRAMES)
            self.assertFalse((Path(tmp) / "out").exists())

    def test_merge_refuses_changed_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = build(tmp)
            path = work / "shard_01/provenance.json"
            value = json.loads(path.read_text())
            value[sorted(value)[0]] = "0" * 64
            path.write_text(json.dumps(value))
            with self.assertRaises((ValueError, SystemExit)):
                sidecar.merge(work, Path(tmp) / "out", 2, episodes=EPISODES, frames=FRAMES)

    def test_merge_refuses_an_extra_or_duplicated_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = build(tmp)
            extra = work / "shard_00/episodes/episode_002.npz"
            os.link(work / "shard_01/episodes/episode_001.npz", work / "shard_00/episodes/episode_001.npz")
            self.assertTrue(extra.exists())
            with self.assertRaises((ValueError, SystemExit)):
                sidecar.merge(work, Path(tmp) / "out", 2, episodes=EPISODES, frames=FRAMES)

    def test_merge_has_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, out = pipeline(tmp)
            with self.assertRaises((FileExistsError, SystemExit)):
                sidecar.merge(work, out, 2, episodes=EPISODES, frames=FRAMES)

    def test_index_lists_every_file_with_its_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            index = json.loads((out / "index.json").read_text())
            self.assertEqual(sorted(index), [f"episode_{i:03d}.npz" for i in range(EPISODES)])
            for name, digest in index.items():
                self.assertEqual(digest, sidecar.sha256_file(out / "episodes" / name))


class Seal(unittest.TestCase):
    def test_seal_changes_when_one_byte_of_one_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            first = sidecar.seal(out, episodes=EPISODES, frames=FRAMES)["fingerprint"]
            path = out / "episodes/episode_002.npz"
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            os.unlink(path)  # the file is a hard link into the shard directory: do not write through it
            path.write_bytes(bytes(data))
            (out / "seal.json").unlink()
            second = sidecar.fingerprint(out)
            self.assertNotEqual(first, second)

    def test_verify_passes_then_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            sidecar.seal(out, episodes=EPISODES, frames=FRAMES)
            self.assertEqual(sidecar.verify(out, episodes=EPISODES, frames=FRAMES)["episodes"], EPISODES)
            report = json.loads((out / "report.json").read_text())
            report["tampered"] = True
            (out / "report.json").write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                sidecar.verify(out, episodes=EPISODES, frames=FRAMES)

    def test_verify_in_a_fresh_process_recomputes_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            sealed = sidecar.seal(out, episodes=EPISODES, frames=FRAMES)
            env = {**os.environ, "PYTHONPATH": str(ROOT)}
            code = ("import sys; from experiments.vla_inputs_signals import sidecar; "
                    f"print(sidecar.verify(sys.argv[1], episodes={EPISODES}, frames={FRAMES})['fingerprint'])")
            fresh = subprocess.run([sys.executable, "-c", code, str(out)], capture_output=True, text=True, env=env, cwd=ROOT)
            self.assertEqual(fresh.returncode, 0, fresh.stderr[-500:])
            self.assertEqual(fresh.stdout.strip().splitlines()[-1], sealed["fingerprint"])
            (out / "episodes/episode_000.npz").unlink()  # same process model: a missing file fails the fresh verify
            broken = subprocess.run([sys.executable, "-c", code, str(out)], capture_output=True, text=True, env=env, cwd=ROOT)
            self.assertNotEqual(broken.returncode, 0)

    def _rewrite(self, out, name, arrays, reindex=True):
        path = out / "episodes" / name
        os.unlink(path)  # break the hard link before writing
        sidecar.write_episode(path, arrays)
        if reindex:  # a consistent rewrite: only the structural checks can object
            index = json.loads((out / "index.json").read_text())
            index[name] = sidecar.sha256_file(path)
            (out / "index.json").write_text(json.dumps(index))

    def test_seal_refuses_segments_that_run_backwards(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            self._rewrite(out, "episode_001.npz", fake(1) | {"segment": fake(1)["segment"][::-1].copy()})
            with self.assertRaises(ValueError):
                sidecar.seal(out, episodes=EPISODES, frames=FRAMES)

    def test_seal_refuses_a_stray_or_a_missing_episode_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            sidecar.write_episode(out / "episodes/episode_004.npz", fake(3))
            with self.assertRaises(ValueError):
                sidecar.seal(out, episodes=EPISODES, frames=FRAMES)
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            (out / "episodes/episode_003.npz").unlink()
            with self.assertRaises(ValueError):
                sidecar.seal(out, episodes=EPISODES, frames=FRAMES)

    def test_seal_refuses_a_file_that_differs_from_the_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            self._rewrite(out, "episode_002.npz", fake(2) | {"wrist_wrench": fake(2)["wrist_wrench"] + 1.0}, reindex=False)
            with self.assertRaises(ValueError):
                sidecar.seal(out, episodes=EPISODES, frames=FRAMES)

    def test_seal_refuses_shifted_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            arrays = fake(2)
            arrays["frame_index"] = np.roll(arrays["frame_index"], 1)
            self._rewrite(out, "episode_002.npz", arrays)
            with self.assertRaises(ValueError):
                sidecar.seal(out, episodes=EPISODES, frames=FRAMES)
            self.assertFalse((out / "seal.json").exists())

    def test_seal_refuses_a_duplicated_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            self._rewrite(out, "episode_003.npz", fake(2))
            with self.assertRaises(ValueError):
                sidecar.seal(out, episodes=EPISODES, frames=FRAMES)

    def test_seal_refuses_non_finite_values_and_truncated_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            arrays = fake(1)
            arrays["wrist_wrench"][2, 3] = np.nan
            self._rewrite(out, "episode_001.npz", arrays)
            with self.assertRaises(ValueError):
                sidecar.seal(out, episodes=EPISODES, frames=FRAMES)
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            self._rewrite(out, "episode_001.npz", {**fake(1), "finger_force": fake(1)["finger_force"][:-1]})
            with self.assertRaises(ValueError):
                sidecar.seal(out, episodes=EPISODES, frames=FRAMES)

    def test_seal_binds_the_p2_seal_and_the_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            sealed = sidecar.seal(out, episodes=EPISODES, frames=FRAMES)
            self.assertEqual(sealed["p2_seal"], contract.P2_SEAL)
            self.assertEqual(sealed["provenance_sha256"], sidecar.digest(contract.provenance()))
            self.assertEqual(sealed["image"], contract.IMAGE)


class Report(unittest.TestCase):
    def test_clip_percentiles_come_from_training_rows_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            report = json.loads((out / "report.json").read_text())
            train = [i for i in range(EPISODES) if fake(i)["split"] == "train"]
            rows = np.concatenate([np.concatenate([fake(i)["wrist_wrench"], fake(i)["finger_force"]], axis=1) for i in train])
            self.assertEqual(report["training_frames"], len(rows))
            self.assertEqual(len(report["force_names"]), 8)
            self.assertTrue(np.allclose(report["clip_percentiles"]["p0.1"], np.percentile(rows, 0.1, axis=0), rtol=0, atol=1e-12))
            self.assertTrue(np.allclose(report["clip_percentiles"]["p99.9"], np.percentile(rows, 99.9, axis=0), rtol=0, atol=1e-12))
            z = np.abs((rows - rows.mean(0)) / rows.std(0)).max(0)
            self.assertTrue(np.allclose(report["max_abs_z"], z, rtol=0, atol=1e-12))

    def test_per_object_and_segment_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = pipeline(tmp)
            report = json.loads((out / "report.json").read_text())
            cell = report["by_object_segment"]["can"]["closed_and_lifted"]
            self.assertEqual(cell["frames"], 2 * 2)  # episodes 0 and 2 are cans, segment 2 has two frames each
            self.assertEqual(sorted(cell["wrist_wrench"]), ["max", "mean", "min", "p0.1", "p99.9", "std"])
            self.assertEqual(report["segments"], list(sidecar.SEGMENTS))


class NonPerturbation(unittest.TestCase):
    def test_signals_do_not_change_the_replay(self):
        """G2: episode 0 with and without every signal computed at every frame: same integration state, every tick."""
        def run(compute):
            states = []

            def on_frame(tick, model, data):
                states.append(capture_state(model, data).copy())
                if compute:
                    signals.wrist_wrench(model, data)
                    signals.finger_force(model, data)
                    signals.target_box(model, data, "cereal")
                    signals.props_box(model, data, replay.objects_of(0))

            replay.frames(0, on_frame)
            return np.stack(states)

        self.assertEqual(replay.load_episode(0)["episode"]["object"], "cereal")
        plain, with_signals = run(False), run(True)
        self.assertEqual(plain.shape[0], contract.FRAMES)
        self.assertTrue(np.array_equal(plain, with_signals))


class RealEpisode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays = sidecar.build_episode(271)

    def test_arrays_are_well_formed_and_physical(self):
        a = self.arrays
        sidecar.check_arrays(a, 271, contract.FRAMES)
        n = len(a["prop_names"])
        self.assertEqual(a["props_box"].shape, (contract.FRAMES, n, 2, 5))
        self.assertEqual(str(a["object"]), "can")
        self.assertIn("can", list(a["prop_names"]))
        self.assertTrue(all(np.isfinite(a[k]).all() for k in ("wrist_wrench", "finger_force", "target_box", "props_box")))
        self.assertEqual(sorted(set(a["segment"].tolist())), [0, 1, 2, 3])
        self.assertEqual(a["segment"].tolist(), sorted(a["segment"].tolist()))
        # the hold segment carries the can: the fingers squeeze and the wrist carries its weight
        lifted = a["segment"] == 2
        self.assertTrue((a["finger_force"][lifted] > 0.05).all())
        self.assertGreater(float(np.abs(a["wrist_wrench"][lifted][:, :3]).max()), 1.0)
        self.assertTrue((a["finger_force"][a["segment"] == 0] == 0).all())

    def test_the_target_box_is_the_target_columns_of_props_box(self):
        a = self.arrays
        column = list(a["prop_names"]).index(str(a["object"]))
        self.assertTrue(np.array_equal(a["target_box"], a["props_box"][:, column]))

    def test_a_second_build_is_identical(self):
        again = sidecar.build_episode(271)
        for key, value in self.arrays.items():
            self.assertTrue(np.array_equal(value, again[key]), key)

    def test_build_shard_writes_the_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            sidecar.build_shard(Path(tmp) / "s", 271, contract.EPISODES)
            with np.load(Path(tmp) / "s/shard_271/episodes/episode_271.npz") as stored:
                for key, value in self.arrays.items():
                    self.assertTrue(np.array_equal(stored[key], value), key)


if __name__ == "__main__":
    unittest.main()
