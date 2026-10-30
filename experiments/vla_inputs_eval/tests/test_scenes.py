import copy
import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from experiments.vla_inputs_eval import feasibility, plan, scenes
from experiments.xvla_props_p2_eval import scenes as p2_scenes

P2_ROOT = Path(__file__).resolve().parents[3] / "outputs/props_p2_data/props_p2_20261013"


def full_state(model, data):
    state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return state


class RestoreEqualsP2(unittest.TestCase):
    def test_a_p2_reserved_layout_restores_to_the_same_full_state(self):
        sealed = json.loads((P2_ROOT / "plan.json").read_text())
        for key in ("report_000", "report_059", "screen_003"):
            layout = sealed["evaluation_holdout"][key]
            ours = scenes.restore(layout)
            theirs = p2_scenes.restore(key)
            self.assertEqual(ours[2], theirs[2])
            self.assertTrue(np.array_equal(full_state(ours[0], ours[1]), full_state(theirs[0], theirs[1])), key)

    def test_the_target_is_first(self):
        layout = json.loads((P2_ROOT / "plan.json").read_text())["evaluation_holdout"]["report_000"]
        self.assertEqual(scenes.restore(layout)[2][0], layout["object"])


class Tolerance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = json.loads((P2_ROOT / "plan.json").read_text())["evaluation_holdout"]["report_000"]

    def patched(self, dx=0.0, dyaw=0.0):
        original = scenes.object_pose

        def shifted(model, data, name):
            pose = dict(original(model, data, name))
            pose["xy"] = pose["xy"] + np.array([dx, 0.0])
            pose["yaw_deg"] = pose["yaw_deg"] + dyaw
            return pose

        return shifted

    def restore_with(self, **kwargs):
        original = scenes.object_pose
        scenes.object_pose = self.patched(**kwargs)
        try:
            return scenes.restore(self.layout)
        finally:
            scenes.object_pose = original

    def test_a_sub_tolerance_offset_is_accepted(self):
        self.restore_with(dx=0.0005, dyaw=0.5)

    def test_a_two_millimetre_offset_is_refused(self):
        with self.assertRaises(ValueError):
            self.restore_with(dx=0.002)

    def test_a_two_degree_yaw_offset_is_refused(self):
        with self.assertRaises(ValueError):
            self.restore_with(dyaw=2.0)

    def test_yaw_difference_wraps_around(self):
        self.assertAlmostEqual(scenes._yaw_difference_deg(179.5, -179.5), 1.0)


class EveryM0LayoutRestores(unittest.TestCase):
    def test_all_656_layouts_restore_within_tolerance(self):
        layouts = plan.make_plan()["layouts"]
        ordered = sorted(layouts.values(), key=lambda l: (tuple(sorted([l["object"], *(d["object"] for d in l["distractors"])])), l["blue_slot"]))
        for layout in ordered:
            model, data, names = scenes.restore(layout)
            self.assertEqual(names[0], layout["object"])
            self.assertEqual(len(names), 1 + layout["distractor_count"])


class Feasibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.plan_dir = Path(cls.tmp.name) / "plan"
        plan.write(cls.plan_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_two_layouts_run_and_write_two_rows(self):
        ids = sorted(json.loads((self.plan_dir / "plan.json").read_text())["tiers"]["smoke"])
        out = Path(self.tmp.name) / "run"
        rows = feasibility.run_shard(out, 0, 1, self.plan_dir, only=ids)
        self.assertEqual(sorted(r["layout_id"] for r in rows), ids)
        lines = [json.loads(x) for x in (out / "feasibility.shard_0.jsonl").read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertEqual(set(line) >= {"layout_id", "feasible", "reason"}, True)
            self.assertIsInstance(line["feasible"], bool)

    def test_a_shard_never_overwrites(self):
        ids = sorted(json.loads((self.plan_dir / "plan.json").read_text())["tiers"]["smoke"])[:1]
        out = Path(self.tmp.name) / "twice"
        feasibility.run_shard(out, 0, 1, self.plan_dir, only=ids)
        with self.assertRaises(FileExistsError):
            feasibility.run_shard(out, 0, 1, self.plan_dir, only=ids)

    def test_shards_partition_the_layouts_interleaved(self):
        ids = [f"l{i}" for i in range(10)]
        parts = [feasibility.shard_ids(ids, k, 3) for k in range(3)]
        self.assertEqual(sorted(sum(parts, [])), ids)
        self.assertEqual(parts[1], ["l1", "l4", "l7"])
        with self.assertRaises(ValueError):
            feasibility.shard_ids(ids, 3, 3)

    def fake_run(self, out, flags):
        """Write shard files from `flags` {layout_id: feasible} with an injected runner."""
        all_ids = sorted(json.loads((self.plan_dir / "plan.json").read_text())["layouts"])
        for k in range(2):
            feasibility.run_shard(out, k, 2, self.plan_dir, only=all_ids,
                                  runner=lambda layout: dict(feasible=flags.get(layout["layout_id"], True),
                                                             reason="ok" if flags.get(layout["layout_id"], True) else "stub",
                                                             frames=1, stable_ticks=1, seconds=0.0))

    def test_merge_lists_feasible_and_dropped_by_cell(self):
        out = Path(self.tmp.name) / "merge"
        layouts = json.loads((self.plan_dir / "plan.json").read_text())["layouts"]
        victim = sorted(layouts)[5]
        self.fake_run(out, {victim: False})
        merged = feasibility.merge(out, self.plan_dir)
        self.assertEqual(merged["dropped"], [victim])
        self.assertEqual(len(merged["feasible"]) + 1, 656)
        self.assertEqual(merged["dropped_by_cell"], {"-".join(map(str, layouts[victim]["cell"])): 1})
        self.assertTrue((out / "feasibility.json").exists())
        with self.assertRaises(FileExistsError):
            feasibility.merge(out, self.plan_dir)

    def test_merge_refuses_a_missing_layout(self):
        out = Path(self.tmp.name) / "partial"
        ids = sorted(json.loads((self.plan_dir / "plan.json").read_text())["layouts"])
        feasibility.run_shard(out, 0, 2, self.plan_dir, only=feasibility.shard_ids(ids, 0, 2),
                              runner=lambda layout: dict(feasible=True, reason="ok", frames=1, stable_ticks=1, seconds=0.0))
        with self.assertRaises(ValueError):
            feasibility.merge(out, self.plan_dir)
        self.assertFalse((out / "feasibility.json").exists())

    def test_a_changed_plan_is_refused(self):
        other = Path(self.tmp.name) / "tampered"
        other.mkdir()
        value = json.loads((self.plan_dir / "plan.json").read_text())
        value["layouts"][sorted(value["layouts"])[0]]["seed"] += 1
        (other / "plan.json").write_text(json.dumps(value))
        (other / "plan.sha256").write_text((self.plan_dir / "plan.sha256").read_text())
        with self.assertRaises(ValueError):
            feasibility.run_shard(Path(self.tmp.name) / "x", 0, 1, other, only=[])


if __name__ == "__main__":
    unittest.main()
