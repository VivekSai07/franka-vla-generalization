import json
import hashlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.props_scene import benchmark as b
from experiments.props_scene import constants as c


def _fake_record(attempt, success=True, replay_ok=True, resting=True, reason=None):
    return {"id": attempt["id"], "index": attempt["index"], "object": attempt["object"], "bin": attempt["bin"],
            "blue_slot": attempt["blue_slot"], "target_slot": attempt["target_slot"], "stratum": attempt["stratum"],
            "success": success, "reason": reason or ("success" if success else "not_in_bin"),
            "replay_ok": replay_ok, "resting_in_bin": resting, "trace_sha256": "0" * 64}


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = b.build_plan()

    def test_size_and_balance(self):
        plan = self.plan
        self.assertEqual(len(plan), c.GATE_ATTEMPTS)
        self.assertEqual(Counter(a["stratum"] for a in plan), {"grid": 288})
        cells = Counter((a["object"], a["bin"], a["blue_slot"]) for a in plan)
        self.assertEqual(len(cells), 24)
        self.assertEqual(set(cells.values()), {c.GATE_SEEDS_PER_CELL})
        self.assertEqual(Counter(a["target_slot"] for a in plan), {"left": 144, "right": 144})
        self.assertEqual(Counter(a["object"] for a in plan), {o: 48 for o in c.OBJECTS})

    def test_target_slot_is_where_the_target_bin_is(self):
        for a in self.plan:
            other = "right" if a["blue_slot"] == "left" else "left"
            self.assertEqual(a["target_slot"], a["blue_slot"] if a["bin"] == "blue" else other)

    def test_poses_stay_in_their_regions(self):
        for a in self.plan:
            x, y, yaw = a["pose"]
            xr, yr = c.START_X_M, c.START_Y_M
            self.assertTrue(xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1], a["id"])
            self.assertGreaterEqual(float(np.hypot(x, y)), c.START_RADIUS_MIN_M, a["id"])
            cap = c.YAW_MAX_DEG_BY_OBJECT[a["object"]]
            self.assertTrue(-cap <= yaw < cap, a["id"])
            self.assertEqual(len(a["home_pose_jitter_deg"]), 7)
            self.assertTrue(all(abs(j) <= c.HOME_JITTER_DEG_MAX for j in a["home_pose_jitter_deg"]))

    def test_lemon_yaw_is_narrower_than_the_rest(self):
        lemon = [a for a in self.plan if a["object"] == "lemon"]
        self.assertTrue(all(abs(a["pose"][2]) < 75.0 for a in lemon))
        self.assertTrue(any(abs(a["pose"][2]) > 75.0 for a in self.plan if a["object"] != "lemon"))

    def test_ids_and_seeds_are_unique_and_the_plan_is_reproducible(self):
        self.assertEqual(len({a["id"] for a in self.plan}), c.GATE_ATTEMPTS)
        self.assertEqual(len({a["seed"] for a in self.plan}), c.GATE_ATTEMPTS)
        self.assertEqual(json.dumps(self.plan, sort_keys=True), json.dumps(b.build_plan(), sort_keys=True))
        self.assertEqual([a["index"] for a in self.plan], list(range(c.GATE_ATTEMPTS)))

    def test_shards_partition_the_plan_evenly(self):
        shards = [b.shard_of(self.plan, k, 6) for k in range(1, 7)]
        self.assertEqual([len(s) for s in shards], [48] * 6)
        self.assertEqual(sorted(a["id"] for s in shards for a in s), sorted(a["id"] for a in self.plan))
        for s in shards:
            self.assertEqual(len({a["object"] for a in s}), 6)
        with self.assertRaises(ValueError):
            b.shard_of(self.plan, 0, 6)
        with self.assertRaises(ValueError):
            b.shard_of(self.plan, 7, 6)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "gate"
        self.plan = b.build_plan()
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def _executor(self, attempt):
        self.calls.append(attempt["id"])
        return _fake_record(attempt)

    def test_run_writes_one_line_per_attempt_and_resume_skips_finished_ones(self):
        mine = b.shard_of(self.plan, 2, 6)
        b.run(self.out, 2, 6, resume=False, executor=self._executor, limit=5)
        lines = (self.out / "attempts_shard2of6.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line)["id"] for line in lines], [a["id"] for a in mine[:5]])
        self.calls.clear()
        b.run(self.out, 2, 6, resume=True, executor=self._executor)
        self.assertEqual(self.calls, [a["id"] for a in mine[5:]])
        lines = (self.out / "attempts_shard2of6.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line)["id"] for line in lines], [a["id"] for a in mine])

    def test_run_refuses_to_overwrite_without_resume(self):
        b.run(self.out, 1, 6, resume=False, executor=self._executor, limit=2)
        with self.assertRaises(FileExistsError):
            b.run(self.out, 1, 6, resume=False, executor=self._executor, limit=2)

    def test_an_exception_in_an_attempt_is_recorded_as_a_failure_and_the_run_continues(self):
        def flaky(attempt):
            if attempt["index"] == 0:
                raise RuntimeError("boom")
            return _fake_record(attempt)
        b.run(self.out, 1, 6, resume=False, executor=flaky, limit=3)
        records = [json.loads(line) for line in (self.out / "attempts_shard1of6.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertFalse(records[0]["success"])
        self.assertTrue(records[0]["reason"].startswith("exception"))
        self.assertIn("boom", records[0]["error"])

    def test_recheck_compares_digests_of_attempts_spread_over_the_recorded_ones(self):
        def digest(attempt):
            return {**_fake_record(attempt), "trace_sha256": hashlib.sha256(attempt["id"].encode()).hexdigest()}
        b.run(self.out, 1, 6, resume=False, executor=digest, limit=10)
        result = b.recheck(self.out, 5, executor=digest)
        self.assertEqual((result["count"], result["matched"]), (5, 5))
        self.assertEqual(len({d["id"] for d in result["details"]}), 5)
        drifting = lambda attempt: {**digest(attempt), "trace_sha256": "f" * 64}
        result = b.recheck(self.out, 5, executor=drifting)
        self.assertEqual((result["count"], result["matched"]), (5, 0))
        with self.assertRaises(ValueError):
            b.recheck(self.out, 11, executor=digest)

    def test_plan_is_pinned_next_to_the_results_and_must_match_on_resume(self):
        b.run(self.out, 1, 6, resume=False, executor=self._executor, limit=1)
        pinned = json.loads((self.out / "plan.json").read_text())
        self.assertEqual(pinned["digest"], b.plan_digest(self.plan))
        (self.out / "plan.json").write_text(json.dumps({"digest": "other"}))
        with self.assertRaises(ValueError):
            b.run(self.out, 1, 6, resume=True, executor=self._executor, limit=1)

    def test_recheck_spreads_over_both_arrangements_and_several_objects(self):
        # The first gate's recheck landed on blue_slot="left" every time (FAILURES); this must not recur.
        b.run(self.out, 1, 1, resume=False, executor=self._executor)
        result = b.recheck(self.out, 12, executor=self._executor)
        self.assertEqual(result["count"], 12)
        self.assertEqual(len({d["id"] for d in result["details"]}), 12)
        self.assertEqual({d["blue_slot"] for d in result["details"]}, {"left", "right"})
        self.assertGreaterEqual(len({d["object"] for d in result["details"]}), 4)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.plan = b.build_plan()

    def _all(self, **kw):
        return [_fake_record(a, **kw) for a in self.plan]

    def test_all_pass_is_go(self):
        s = b.summarise(self._all(), self.plan, recheck={"count": 12, "matched": 12})
        self.assertEqual(s["verdict"], "go")
        self.assertEqual(s["overall"]["rate"], 1.0)

    def test_overall_below_95_percent_fails(self):
        records = self._all()
        for r in records[:16]:                         # 16 failures spread over the plan -> 272/288 = 94.4%
            r["success"], r["reason"] = False, "not_in_bin"
        records = [r for r in records]
        failing = {r["id"] for r in records if not r["success"]}
        self.assertEqual(len(failing), 16)
        s = b.summarise(records, self.plan, recheck={"count": 12, "matched": 12})
        self.assertEqual(s["verdict"], "no-go")
        self.assertFalse(s["gates"]["overall"])

    def test_one_object_hidden_behind_the_others_fails_the_per_object_gate(self):
        records = self._all()
        milk = [r for r in records if r["object"] == "milk"]
        for r in milk[:6]:                              # 42 of 48 = 87.5% for milk, 282 of 288 = 97.9% overall
            r["success"], r["reason"] = False, "not_in_bin"
        s = b.summarise(records, self.plan, recheck={"count": 12, "matched": 12})
        self.assertTrue(s["gates"]["overall"])
        self.assertFalse(s["gates"]["per_object"])
        self.assertEqual(s["verdict"], "no-go")

    def test_one_slot_worse_fails_the_per_slot_gate(self):
        records = self._all()
        left = [r for r in records if r["target_slot"] == "left"]
        for r in left[:11]:                             # 133 of 144 = 92.4% < 93%, overall 277/288 = 96.2%
            r["success"], r["reason"] = False, "not_in_bin"
        s = b.summarise(records, self.plan, recheck={"count": 12, "matched": 12})
        self.assertTrue(s["gates"]["overall"])
        self.assertFalse(s["gates"]["per_slot"])
        self.assertEqual(s["verdict"], "no-go")

    def test_a_replay_failure_fails_the_gate_even_when_everything_succeeds(self):
        records = self._all()
        records[3]["replay_ok"] = False
        s = b.summarise(records, self.plan, recheck={"count": 12, "matched": 12})
        self.assertFalse(s["gates"]["replay"])
        self.assertEqual(s["verdict"], "no-go")

    def test_missing_attempts_are_incomplete_not_go(self):
        s = b.summarise(self._all()[:-1], self.plan, recheck={"count": 12, "matched": 12})
        self.assertEqual(s["verdict"], "incomplete")

    def test_determinism_recheck_must_exist_and_match(self):
        self.assertEqual(b.summarise(self._all(), self.plan, recheck=None)["verdict"], "incomplete")
        self.assertEqual(b.summarise(self._all(), self.plan, recheck={"count": 12, "matched": 11})["verdict"], "no-go")

    def test_the_diagnostic_never_changes_the_verdict(self):
        records = self._all(success=False, resting=True)
        s = b.summarise(records, self.plan, recheck={"count": 12, "matched": 12})
        self.assertEqual(s["verdict"], "no-go")
        self.assertEqual(s["diagnostic_resting_in_bin"]["overall"]["rate"], 1.0)

    def test_reasons_are_reported(self):
        records = self._all()
        s = b.summarise(records, self.plan, recheck={"count": 12, "matched": 12})
        self.assertEqual(s["by_stratum"]["grid"]["n"], 288)
        self.assertEqual(s["by_reason"], {"success": 288})
        self.assertEqual(set(s["by_target_slot"]), {"left", "right"})


if __name__ == "__main__":
    unittest.main()
