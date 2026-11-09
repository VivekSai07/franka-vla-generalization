import unittest
from collections import Counter

from experiments.gen_eval import protocol
from experiments.gen_scene import levels

TIER = "benchmark"


def synthetic_plan(n=24):
    """n layouts with every axis a trivial condition; the protocol only reads layout fields and the axes' dicts."""
    layouts, order = {}, []
    for i in range(n):
        lid = f"L{i:03d}"
        order.append(lid)
        axes = {axis: dict(level=levels.level_of(axis, i), condition=dict(disturbance="move" if axis == "disturbance" and levels.level_of(axis, i) == "move"
                                                                          else ("drop" if axis == "disturbance" else "none"), table_offset_m=0.0))
                for axis in levels.AXES}
        layouts[lid] = dict(layout_id=lid, tier=TIER, seed=100 + i, object="milk" if i % 2 else "can", bin="blue", blue_slot="left",
                            cell=["can", "blue", "left", 1], instruction="put the can in the blue bin", distractor_count=1,
                            distractors=[dict(object="bread", pose=[0.3, 0.1, 0.0])], start=[0.3, 0.0, 0.0],
                            home_pose_jitter_deg=[0.0] * 7, axes=axes)
    return dict(layouts=layouts, tiers={"smoke": [], "selection": [], TIER: order})


def feasibility_of(plan, drop=()):
    keys = {protocol.trial_key(lid, axis) for lid in plan["layouts"] for axis in protocol.AXES}
    return dict(feasible_trials=sorted(keys - set(drop)))


class Trials(unittest.TestCase):
    def setUp(self):
        self.plan = synthetic_plan()

    def test_six_trials_per_layout_in_plan_order_nominal_first(self):
        trials = protocol.stage_trials(self.plan, feasibility_of(self.plan), TIER)
        self.assertEqual(len(trials), 24 * 6)
        self.assertEqual([t["axis"] for t in trials[:6]], ["nominal", "lighting", "background", "distractors", "spatial", "disturbance"])
        self.assertEqual([t["ordinal"] for t in trials], list(range(1, 24 * 6 + 1)))
        self.assertEqual([t["layout_id"] for t in trials[:12]], ["L000"] * 6 + ["L001"] * 6)

    def test_the_trial_carries_its_condition_level_and_tick_cap(self):
        trials = protocol.stage_trials(self.plan, feasibility_of(self.plan), TIER)
        first = trials[:6]
        self.assertEqual((first[0]["level"], first[0]["condition"]["table_offset_m"], first[0]["condition"]["disturbance"]), ("nominal", 0.0, "none"))
        self.assertEqual([t["level"] for t in first[1:]], [levels.level_of(a, 0) for a in levels.AXES])
        self.assertEqual([t["max_ticks"] for t in first], [900, 900, 900, 900, 900, 1200])
        self.assertEqual({t["max_ticks"] for t in trials if t["axis"] != "disturbance"}, {900})
        self.assertEqual({t["max_ticks"] for t in trials if t["axis"] == "disturbance"}, {1200})

    def test_every_trial_of_a_layout_shares_its_seed(self):
        trials = protocol.stage_trials(self.plan, feasibility_of(self.plan), TIER)
        self.assertEqual({t["seed"] for t in trials if t["layout_id"] == "L003"}, {103})
        self.assertTrue(all(t["seed_index"] == 0 for t in trials))

    def test_a_dropped_trial_leaves_a_gap_and_the_others_keep_their_ordinals(self):
        feas = feasibility_of(self.plan, drop=[protocol.trial_key("L001", "spatial")])
        trials = protocol.stage_trials(self.plan, feas, TIER)
        self.assertEqual(len(trials), 24 * 6 - 1)
        ordinals = [t["ordinal"] for t in trials]
        self.assertNotIn(6 + 4 + 1, ordinals)
        self.assertIn(6 + 5 + 1, ordinals)

    def test_a_whole_layout_dropped(self):
        feas = feasibility_of(self.plan, drop=[protocol.trial_key("L002", a) for a in protocol.AXES])
        trials = protocol.stage_trials(self.plan, feas, TIER)
        self.assertEqual(len(trials), 23 * 6)
        self.assertNotIn("L002", {t["layout_id"] for t in trials})

    def test_kind_and_tier_validation(self):
        trial = protocol.stage_trials(self.plan, feasibility_of(self.plan), TIER)[0]
        self.assertEqual(trial["kind"], "trained")
        with self.assertRaises(ValueError):
            protocol.stage_trials(self.plan, feasibility_of(self.plan), "comparison")
        with self.assertRaises(ValueError):
            protocol.trial_of(self.plan["layouts"]["L000"], "colour", 1)

    def test_ordinal_formula(self):
        self.assertEqual(protocol.ordinal_of(0, "nominal"), 1)
        self.assertEqual(protocol.ordinal_of(0, "disturbance"), 6)
        self.assertEqual(protocol.ordinal_of(143, "disturbance"), 864)


class Sharding(unittest.TestCase):
    def setUp(self):
        self.plan = synthetic_plan(144)
        self.trials = protocol.stage_trials(self.plan, feasibility_of(self.plan), TIER)

    def test_every_shard_count_partitions_the_stage(self):
        for shards in range(1, 9):
            seen = []
            for shard in range(shards):
                seen += [t["ordinal"] for t in protocol.shard_trials(self.trials, shard, shards)]
            self.assertEqual(sorted(seen), [t["ordinal"] for t in self.trials], shards)

    def test_six_workers_each_get_every_axis_equally_often(self):
        for shard in range(6):
            share = protocol.shard_trials(self.trials, shard, 6)
            self.assertEqual(len(share), 144)
            self.assertEqual(set(Counter(t["axis"] for t in share).values()), {24}, shard)
            self.assertEqual({t["axis"] for t in share}, set(protocol.AXES))

    def test_a_layouts_six_trials_go_to_six_workers(self):
        owner = {}
        for shard in range(6):
            for t in protocol.shard_trials(self.trials, shard, 6):
                owner.setdefault(t["layout_id"], set()).add(shard)
        self.assertTrue(all(len(s) == 6 for s in owner.values()))

    def test_the_longer_disturbance_trials_spread_evenly(self):
        for shard in range(6):
            share = protocol.shard_trials(self.trials, shard, 6)
            self.assertEqual(sum(t["axis"] == "disturbance" for t in share), 24)

    def test_invalid_shards(self):
        for shard, shards in ((0, 0), (3, 3), (-1, 3), (1.0, 3), (0, "3")):
            with self.assertRaises(ValueError):
                protocol.shard_trials(self.trials, shard, shards)


class Folders(unittest.TestCase):
    def test_names_and_completeness(self):
        import tempfile
        from pathlib import Path
        trial = dict(ordinal=7, layout_id="L001", axis="lighting")
        self.assertEqual(protocol.trial_folder("/x", trial).name, "trial_0007_L001_lighting")
        self.assertEqual(protocol.stage_dir("/r", "A2", "benchmark", 88830), Path("/r/arms/A2/benchmark/step_088830"))
        with self.assertRaises(ValueError):
            protocol.stage_dir("/r", "A2", "comparison", 1)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in protocol.TRIAL_FILES[:-1]:
                (folder / name).write_text("x")
            self.assertFalse(protocol.trial_is_complete(folder))
            (folder / protocol.TRIAL_FILES[-1]).write_text("x")
            self.assertTrue(protocol.trial_is_complete(folder))


if __name__ == "__main__":
    unittest.main()
