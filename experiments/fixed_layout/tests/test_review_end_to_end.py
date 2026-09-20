"""Full 24-slot synthetic evidence fixture. Never an actual expert or physics step.

Only source/reset and geometric simulation seams are replaced. The real writer,
encoder, rows, metadata, traces, replay evidence audit, reports and seal remain.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.fixed_layout import record, validate
from experiments.fixed_layout.contract import MIN_FREE_BYTES, make_plan
from experiments.paired_demonstrations.media import CAMERAS
from test_review import create_dataset, synthetic_frame
from test_validate import SyntheticSimulator


class EndToEndReviewTests(unittest.TestCase):
    def test_real_synthetic_dataset_validate_seal_require_then_mutate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "attempts").mkdir()
            source_bytes = b"explicit synthetic source; not the approved reset"
            (root / "source.npz").write_bytes(source_bytes)
            bound = {"synthetic_fixture": True, "files": {"source.npz": record.file_hash(root / "source.npz")}}
            plan = make_plan()
            record.write_json(root / "plan.json", plan)
            record.write_json(root / "provenance.json", bound)
            sims = [SyntheticSimulator(velocity=.001 + i * .00001) for i in range(24)]
            expert = sims[0]
            record.write_json(root / "configuration.json", expert.config)
            dataset = create_dataset(root, record.PairedDataset)
            events, fake_steps = [], 0
            images = {key: synthetic_frame(0, 0)[key] for key in CAMERAS}
            try:
                for i, sim in enumerate(sims):
                    attempt = root / "attempts" / f"{i:06d}"
                    with sim.patches(validate):
                        reset, _ = sim.fixture(attempt, validate, episode_index=i, images=images)
                        # validate_episode owns creation of its replay token directory.
                        (attempt / "replays").rmdir()
                        if i == 0:
                            frozen_reset = copy.deepcopy(reset)
                            record.write_npz(root / "reset.npz", frozen_reset)
                            hashes = {**record.snapshot_hashes(root), **{name: record.file_hash(root / name)
                                      for name in ("configuration.json", "source.npz")}}
                            record.write_json(root / "snapshots.json", hashes)
                        else:
                            for key in reset:
                                np.testing.assert_array_equal(reset[key], frozen_reset[key])
                        for tick in range(528):
                            raw = validate._npz(attempt / "ticks" / f"{tick:06d}.npz")
                            frame = {key: raw[key] for key in (*CAMERAS, "action", "observation.state")}
                            frame["task"] = raw["task"].item()
                            dataset.add_frame(frame)
                        dataset.save_episode()
                        dataset.finalize()
                        dataset._writer_closed_for_reading = True
                        # Real orchestration, including two pure-numeric replay writers.
                        validate.validate_episode(root, plan["episodes"][i], sim, reset, lambda: None)
                        fake_steps += sim.steps
                    events.extend([{"event": "started", **plan["episodes"][i], "elapsed_seconds": i},
                                   {"event": "admitted", "episode_index": i, "elapsed_seconds": i}])
                    if i == 2:
                        events.append({"event": "preparation_pass", "elapsed_seconds": i})
                (root / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
                info_path = root / "dataset/meta/info.json"
                info = json.loads(info_path.read_text())
                info["splits"] = {"train": "0:20", "validation": "20:24"}
                info_path.write_text(json.dumps(info))
                with expert.patches(validate), patch.object(validate, "provenance", return_value=bound), \
                        patch.object(record, "provenance", return_value=bound), \
                        patch.object(validate, "SOURCE_SHA256", record.file_hash(root / "source.npz")), \
                        patch.object(validate, "Expert", return_value=expert), \
                        patch.object(validate, "load_reset", return_value=frozen_reset), \
                        patch.object(record, "disk_usage") as disk:
                    disk.return_value.free = MIN_FREE_BYTES
                    result = validate.validate_dataset(root, expert, frozen_reset, lambda: None)
                    self.assertEqual((result["episode_count"], result["replay_count"], result["decoded_camera_frames"]),
                                     (24, 48, 25344))
                    self.assertEqual(result["duplicates"]["exact_duplicates"], [])
                    record.write_json(root / "validation.json", result)
                    record.write_json(root / "seal.json", {"fingerprint": validate.fingerprint(root)})
                    self.assertEqual(validate.require_fixed_layout_pass(root), result)
                    self.assertEqual(sum(s.run_calls for s in sims), 0)
                    self.assertEqual(sum(s.steps for s in sims), fake_steps)
                    self.assertEqual(fake_steps, 25344)

                    # Stale seals AND independently re-audited, freshly resealed
                    # mutations must fail. No fake validate_dataset PASS shortcut.
                    first = root / "attempts/000000"
                    metadata_path = sorted((root / "dataset/meta/episodes").rglob("*.parquet"))[0]
                    metadata = pq.read_table(metadata_path).to_pylist()
                    video_paths = [root / "dataset" / result["episodes"][0]["media"]["videos"][j] for j in range(2)]
                    rows_path = root / "dataset" / result["episodes"][0]["media"]["rows"]
                    replay_path = first / "replays/reference/000000.npz"
                    for kind, path in (("interval_nan", metadata_path), ("wrong_camera", video_paths[0]),
                                       ("persisted_row", rows_path), ("replay_control", replay_path)):
                        original = path.read_bytes()
                        if kind == "interval_nan":
                            changed = copy.deepcopy(metadata)
                            changed[0][f"videos/{next(iter(CAMERAS))}/from_timestamp"] = float("nan")
                            pq.write_table(pa.Table.from_pylist(changed), path)
                        elif kind == "wrong_camera":
                            path.write_bytes(video_paths[1].read_bytes())
                        elif kind == "persisted_row":
                            table = pq.read_table(path)
                            actions = np.array(table["action"].to_pylist(), np.float32)
                            actions[0, 0] += .01
                            table = table.set_column(table.schema.get_field_index("action"), "action",
                                                     pa.array(actions.tolist(), type=pa.list_(pa.float32(), 8)))
                            pq.write_table(table, path)
                        else:
                            changed = validate._npz(path)
                            changed["ctrl"][0] += .01
                            with path.open("wb") as stream:
                                np.savez_compressed(stream, **changed)
                        with self.subTest(kind=kind, seal="stale"), self.assertRaisesRegex(ValueError, "stale"):
                            validate.require_fixed_layout_pass(root)
                        (root / "seal.json").write_text(json.dumps({"fingerprint": validate.fingerprint(root)}))
                        with self.subTest(kind=kind, seal="fresh"), self.assertRaises(ValueError):
                            validate.require_fixed_layout_pass(root)
                        path.write_bytes(original)
                        (root / "seal.json").write_text(json.dumps({"fingerprint": validate.fingerprint(root)}))
                    print("SYNTHETIC_E2E episodes=24 rows=12672 camera_frames=25344 replay_evidence=48 "
                          "numeric_fake_steps=25344 actual_expert_calls=0 native_steps=0 mutations=8", flush=True)
            finally:
                dataset.finalize()


if __name__ == "__main__":
    unittest.main()