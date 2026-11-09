import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiments.gen_eval import feasibility, plan as plan_module, protocol
from experiments.gen_scene import levels
from experiments.gen_scene.conditions import NOMINAL, Condition

_CACHE = {}


def the_plan():
    if not _CACHE:
        _CACHE["plan"] = plan_module.make_plan()
    return _CACHE["plan"]


def write_plan(directory):
    directory = Path(directory)
    directory.mkdir(parents=True)
    plan = the_plan()
    from experiments.props_p2_data.plan import digest
    (directory / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True, allow_nan=False) + "\n")
    (directory / "plan.sha256").write_text(digest(plan) + "\n")
    return plan


def always(feasible=True, reason="success"):
    return lambda layout, job: dict(feasible=feasible, reason=reason if not feasible else "success", frames=452, stable_ticks=30, seconds=0.0)


class Jobs(unittest.TestCase):
    def test_every_layout_has_nominal_distractors_spatial_and_a_move_job_only_for_move_layouts(self):
        plan = the_plan()
        for layout in plan["layouts"].values():
            jobs = feasibility.jobs_of(layout)
            self.assertEqual(jobs[:3], ["nominal", "distractors", "spatial"])
            self.assertEqual("move" in jobs, layout["axes"]["disturbance"]["level"] == "move")
        counts = Counter(job for _, job in feasibility.all_jobs(plan))
        self.assertEqual(counts["nominal"], counts["distractors"])
        self.assertEqual(counts["nominal"], 170)
        self.assertEqual(counts["move"], 85)

    def test_job_scenes(self):
        plan = the_plan()
        layout = plan["layouts"][plan["tiers"]["benchmark"][0]]
        self.assertEqual(feasibility.job_scene(layout, "nominal"), (layout, NOMINAL))
        for job in ("distractors", "spatial"):
            scene_layout, condition = feasibility.job_scene(layout, job)
            self.assertIs(scene_layout, layout)
            self.assertEqual(condition, Condition.from_dict(layout["axes"][job]["condition"]))
        move = next(l for l in plan["layouts"].values() if l["axes"]["disturbance"]["level"] == "move")
        moved, condition = feasibility.job_scene(move, "move")
        x, y = move["axes"]["disturbance"]["condition"]["move_to"]
        self.assertEqual((moved["start"], condition), ([x, y, move["start"][2]], NOMINAL))
        self.assertEqual(move["start"][:2] != moved["start"][:2], True)      # the original layout is untouched
        with self.assertRaises(ValueError):
            feasibility.job_scene(layout, "lighting")

    def test_sharding_partitions_the_jobs(self):
        jobs = feasibility.all_jobs(the_plan())
        seen = []
        for k in range(7):
            seen += feasibility.shard_jobs(jobs, k, 7)
        self.assertEqual(sorted(seen), jobs)
        with self.assertRaises(ValueError):
            feasibility.shard_jobs(jobs, 7, 7)


class Decide(unittest.TestCase):
    """Spec D8, rule by rule."""

    def layout(self, move=False):
        plan = the_plan()
        return next(l for l in plan["layouts"].values() if (l["axes"]["disturbance"]["level"] == "move") == move)

    def ok(self, **kw):
        base = dict(nominal=True, distractors=True, spatial=True, move=True)
        base.update(kw)
        return base

    def reasons(self, layout, **kw):
        return {k.split(":")[1]: v for k, v in feasibility.decide(layout, self.ok(**kw)).items()}

    def test_everything_feasible(self):
        self.assertEqual(set(self.reasons(self.layout()).values()), {None})

    def test_a_nominal_failure_drops_all_six_trials(self):
        out = self.reasons(self.layout(), nominal=False)
        self.assertEqual(set(out), set(protocol.AXES))
        self.assertEqual(set(out.values()), {"nominal infeasible"})

    def test_lighting_and_background_inherit_the_nominal(self):
        out = self.reasons(self.layout(), distractors=False, spatial=False)
        self.assertEqual((out["nominal"], out["lighting"], out["background"]), (None, None, None))

    def test_a_geometry_failure_drops_only_that_axis(self):
        out = self.reasons(self.layout(), distractors=False)
        self.assertEqual(out["distractors"], "distractors infeasible")
        self.assertEqual([k for k, v in out.items() if v], ["distractors"])
        out = self.reasons(self.layout(), spatial=False)
        self.assertEqual([k for k, v in out.items() if v], ["spatial"])

    def test_move_needs_the_fresh_episode_and_drop_needs_only_the_nominal(self):
        move = self.layout(move=True)
        self.assertEqual(self.reasons(move, move=False)["disturbance"], "post-move pose infeasible")
        self.assertIsNone(self.reasons(move)["disturbance"])
        drop = self.layout(move=False)
        self.assertEqual(drop["axes"]["disturbance"]["level"], "drop")
        out = feasibility.decide(drop, dict(nominal=True, distractors=True, spatial=True))   # no move job exists for a drop layout
        self.assertIsNone(out[protocol.trial_key(drop["layout_id"], "disturbance")])


class Shards(unittest.TestCase):
    def test_a_shard_writes_one_row_per_job_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_plan(Path(tmp) / "plan")
            out = Path(tmp) / "feasibility"
            plan = the_plan()
            only = feasibility.all_jobs(plan)[:9]
            rows = feasibility.run_shard(out, 1, 3, Path(tmp) / "plan", only=only, runner=always())
            self.assertEqual([(r["layout_id"], r["job"]) for r in rows], feasibility.shard_jobs(only, 1, 3))
            lines = (out / "feasibility.shard_1.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertFalse((out / "feasibility.shard_1.partial.jsonl").exists())
            with self.assertRaises(FileExistsError):
                feasibility.run_shard(out, 1, 3, Path(tmp) / "plan", only=only, runner=always())

    def test_a_plan_that_does_not_match_its_digest_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_plan(Path(tmp) / "plan")
            (Path(tmp) / "plan/plan.sha256").write_text("0" * 64 + "\n")
            with self.assertRaises(ValueError):
                feasibility.load_plan(Path(tmp) / "plan")


class Merge(unittest.TestCase):
    def campaign(self, tmp, runner=None, shards=4):
        write_plan(Path(tmp) / "plan")
        out = Path(tmp) / "feasibility"
        for k in range(shards):
            feasibility.run_shard(out, k, shards, Path(tmp) / "plan", runner=runner or always())
        return out

    def test_all_feasible_gives_864_trials_per_benchmark_tier_and_a_gapless_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.campaign(tmp)
            summary = feasibility.merge(out, Path(tmp) / "plan")
            self.assertEqual(len(summary["feasible_trials"]), 170 * 6)
            self.assertEqual(summary["dropped_trials"], {})
            bench = summary["counts"]["benchmark"]
            self.assertEqual(bench["nominal"]["nominal"], {"planned": 144, "feasible": 144})
            self.assertEqual({lv: c["planned"] for lv, c in bench["lighting"].items()}, {"dark": 36, "bright": 36, "tinted": 36, "low_angle": 36})
            self.assertEqual({lv: c["planned"] for lv, c in bench["disturbance"].items()}, {"move": 72, "drop": 72})
            self.assertEqual(json.loads((out / "feasibility.json").read_text()), summary)

    def test_drops_follow_the_rules_and_are_counted_by_axis_and_level(self):
        plan = the_plan()
        bad_nominal = plan["tiers"]["benchmark"][3]
        bad_distractors = plan["tiers"]["benchmark"][5]
        bad_move = next(i for i in plan["tiers"]["benchmark"] if plan["layouts"][i]["axes"]["disturbance"]["level"] == "move" and i not in (bad_nominal, bad_distractors))

        def runner(layout, job):
            fail = (layout["layout_id"], job) in {(bad_nominal, "nominal"), (bad_distractors, "distractors"), (bad_move, "move")}
            return always(feasible=not fail, reason="grasp")(layout, job)

        with tempfile.TemporaryDirectory() as tmp:
            out = self.campaign(tmp, runner)
            summary = feasibility.merge(out, Path(tmp) / "plan")
            dropped = summary["dropped_trials"]
            self.assertEqual(sorted(k for k in dropped if k.startswith(bad_nominal)), sorted(f"{bad_nominal}:{a}" for a in protocol.AXES))
            self.assertEqual({k: v for k, v in dropped.items() if k.startswith(bad_distractors)}, {f"{bad_distractors}:distractors": "distractors infeasible"})
            self.assertEqual({k: v for k, v in dropped.items() if k.startswith(bad_move)}, {f"{bad_move}:disturbance": "post-move pose infeasible"})
            self.assertEqual(len(dropped), 6 + 1 + 1)
            self.assertEqual(len(summary["feasible_trials"]), 170 * 6 - 8)
            bench = summary["counts"]["benchmark"]
            self.assertEqual(bench["nominal"]["nominal"]["feasible"], 143)
            level = plan["layouts"][bad_distractors]["axes"]["distractors"]["level"]
            self.assertEqual(bench["distractors"][level]["planned"] - bench["distractors"][level]["feasible"], 1 + (plan["layouts"][bad_nominal]["axes"]["distractors"]["level"] == level))
            self.assertIn(f"{bad_nominal}:nominal", summary["expert_reasons"])
            self.assertEqual(summary["expert_reasons"][f"{bad_nominal}:nominal"], "grasp")

    def test_a_dropped_trial_is_never_replaced_and_leaves_a_gap_in_the_stage(self):
        plan = the_plan()
        bad = plan["tiers"]["benchmark"][7]
        with tempfile.TemporaryDirectory() as tmp:
            out = self.campaign(tmp, lambda l, j: always(not (l["layout_id"] == bad and j == "nominal"))(l, j))
            summary = feasibility.merge(out, Path(tmp) / "plan")
            trials = protocol.stage_trials(plan, summary, "benchmark")
            self.assertEqual(len(trials), 143 * 6)
            self.assertNotIn(bad, {t["layout_id"] for t in trials})
            self.assertEqual(trials[7 * 6]["ordinal"], 8 * 6 + 1)

    def test_missing_or_repeated_jobs_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.campaign(tmp, shards=4)
            (out / "feasibility.shard_2.jsonl").unlink()
            with self.assertRaises(ValueError):
                feasibility.merge(out, Path(tmp) / "plan")
        with tempfile.TemporaryDirectory() as tmp:
            out = self.campaign(tmp, shards=4)
            extra = out / "feasibility.shard_9.jsonl"
            extra.write_text((out / "feasibility.shard_1.jsonl").read_text())
            with self.assertRaises(ValueError):
                feasibility.merge(out, Path(tmp) / "plan")

    def test_merge_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.campaign(tmp)
            feasibility.merge(out, Path(tmp) / "plan")
            with self.assertRaises(FileExistsError):
                feasibility.merge(out, Path(tmp) / "plan")


if __name__ == "__main__":
    unittest.main()
