import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.props_p3_data import contract
from experiments.props_p3_data import plan as p
from experiments.props_p3_data import record as rec
from experiments.props_p3_data.record import (
    BudgetExhausted, SlotExhausted, read_journal, run_schedule, substitutions_of,
)

PLAN = p.make_plan()
SHARDS = contract.SHARDS
PER_SHARD = p.ADMIT_COUNT // SHARDS


class Interrupted(Exception):
    pass


class Fake:
    """execute/commit stand-ins; `reject` is a set of (slot, attempt) that fail, `stop_at` raises."""

    def __init__(self, reject=(), stop_at=None):
        self.reject, self.stop_at = set(reject), stop_at
        self.executed, self.committed = [], []

    def attempt_of(self, episode, draw):
        base = PLAN["episodes"][episode["episode_index"]]
        k = [p.attempt_draw(base, i)["seed"] for i in range(3)].index(draw["seed"])
        r = int(episode["layout_id"].rsplit("_r", 1)[1]) if "_r" in episode["layout_id"] else 0
        return r * 3 + k

    def execute(self, episode, draw):
        slot, attempt = episode["episode_index"], self.attempt_of(episode, draw)
        if self.stop_at == (slot, attempt):
            raise Interrupted()
        self.executed.append((slot, attempt, episode["layout_id"], draw["seed"], tuple(episode["start"])))
        ok = (slot, attempt) not in self.reject
        verdict = {"admitted": ok, "reason": "admitted" if ok else "expert_success", "gates": {"expert_success": ok},
                   "details": {}}
        return {}, ["frame"], [{}], verdict

    def commit(self, episode, frames, result, traces):
        self.committed.append((episode["episode_index"], episode["layout_id"]))


class ShardSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.journal = Path(self.tmp.name) / "journal.jsonl"
        self.log = []

    def tearDown(self):
        self.tmp.cleanup()

    def run_(self, fake, slots, **kwargs):
        return run_schedule(PLAN, fake.execute, self.journal, fake.commit, slots, log=self.log.append, **kwargs)

    def test_a_shard_admits_exactly_its_block_in_order(self):
        fake = Fake()
        slots = p.shard_slots(3, SHARDS)
        self.assertEqual(self.run_(fake, slots), {"admitted": PER_SHARD, "rejected": 0})
        self.assertEqual([s for s, _ in fake.committed], slots)
        self.assertEqual([j["slot"] for j in read_journal(self.journal)], slots)

    def test_the_shards_together_admit_every_slot_once(self):
        admitted = []
        for k in range(SHARDS):
            journal = Path(self.tmp.name) / f"j{k}.jsonl"
            fake = Fake()
            run_schedule(PLAN, fake.execute, journal, fake.commit, p.shard_slots(k, SHARDS), log=self.log.append)
            admitted += [j["slot"] for j in read_journal(journal) if j["admitted"]]
        self.assertEqual(admitted, list(range(544)))

    def test_a_rejected_attempt_never_commits_and_retries_the_same_slot(self):
        # Review Focus 5.
        fake = Fake(reject={(5, 0), (5, 1)})
        self.assertEqual(self.run_(fake, [5, 6]), {"admitted": 2, "rejected": 2})
        self.assertEqual([s for s, _ in fake.committed], [5, 6])
        self.assertEqual([c[1] for c in fake.executed if c[0] == 5], [0, 1, 2])
        self.assertEqual(read_journal(self.journal)[0]["reason"], "expert_success")

    def test_three_rejections_resample_the_start_keeping_the_slot(self):
        fake = Fake(reject={(40, 0), (40, 1), (40, 2)})
        self.run_(fake, [40])
        line = read_journal(self.journal)[-1]
        self.assertTrue(line["admitted"] and line["r"] == 1)
        self.assertEqual(len({c[4] for c in fake.executed}), 2)
        self.assertEqual(substitutions_of(read_journal(self.journal)), {"40": 1})

    def test_a_slot_failing_every_layout_stops_the_shard(self):
        with self.assertRaises(SlotExhausted):
            self.run_(Fake(reject={(20, a) for a in range(12)}), [20, 21])

    def test_a_pilot_may_skip_an_exhausted_slot(self):
        fake = Fake(reject={(20, a) for a in range(12)})
        self.assertEqual(self.run_(fake, [20, 21], strict=False), {"admitted": 1, "rejected": 12})

    def test_the_shard_budget(self):
        slots = p.shard_slots(0, SHARDS)
        everything = {(s, a) for s in slots for a in range(3)}
        self.assertEqual(self.run_(Fake(reject=everything), slots, budget=3 * PER_SHARD)["rejected"], 3 * PER_SHARD)
        self.journal.unlink()
        with self.assertRaises(BudgetExhausted):
            self.run_(Fake(reject=everything), slots, budget=3 * PER_SHARD - 1)

    def test_a_resumed_journal_already_over_budget_is_refused(self):
        self.run_(Fake(reject={(0, 0), (0, 1)}), [0, 1], budget=10)
        with self.assertRaises(BudgetExhausted):
            self.run_(Fake(), [0, 1], resume=True, budget=1, dataset_count=lambda: 2)

    def test_resume_continues_and_refuses_mismatches(self):
        slots = p.shard_slots(1, SHARDS)
        first = Fake(reject={(37, 0)}, stop_at=(37, 1))
        with self.assertRaises(Interrupted):
            self.run_(first, slots)
        with self.assertRaises(ValueError):
            self.run_(Fake(), slots, resume=True, dataset_count=lambda: 1)
        second = Fake(reject={(37, 0)})
        self.run_(second, slots, resume=True, dataset_count=lambda: 3)
        self.assertEqual([c[:2] for c in second.executed][0], (37, 1))
        self.assertEqual([s for s, _ in first.committed + second.committed], slots)
        with self.assertRaises(ValueError):
            self.run_(Fake(), p.shard_slots(2, SHARDS), resume=True)

    def test_fresh_run_over_a_journal_and_resume_without_one_refuse(self):
        with self.assertRaises(ValueError):
            self.run_(Fake(), [0], resume=True)
        self.run_(Fake(), [0])
        with self.assertRaises(FileExistsError):
            self.run_(Fake(), [0])

    def test_truncated_last_journal_line_is_dropped_but_a_bad_middle_line_raises(self):
        self.run_(Fake(), [0, 1])
        with self.journal.open("a") as stream:
            stream.write('{"slot": 2, "adm')
        self.assertEqual(len(read_journal(self.journal)), 2)
        self.journal.write_text('{"slot": 0\n{"slot": 1}\n')
        with self.assertRaises(json.JSONDecodeError):
            read_journal(self.journal)

    def test_journal_records_the_object_and_has_no_distractor_fields(self):
        self.run_(Fake(), [0], tag="P3 shard=0")
        line = read_journal(self.journal)[0]
        self.assertEqual(line["object"], PLAN["episodes"][0]["object"])
        self.assertFalse([k for k in line if "distractor" in k])
        self.assertTrue(self.log[0].startswith("P3 shard=0 "))
        self.assertIn("commit_seconds", line)

    def test_the_check_runs_at_the_start_and_every_fifth_admit(self):
        calls = []
        self.run_(Fake(), list(range(11)), check=calls.append)
        self.assertEqual(calls, [0, 5, 10])


class FakeDataset:
    def __init__(self):
        self.events = []
        self.meta = type("Meta", (), {"total_episodes": 7})()

    def add_frame(self, frame):
        self.events.append("frame")

    def save_episode(self):
        self.events.append("save")

    def finalize(self):
        self.events.append("finalize")


class CommitterTests(unittest.TestCase):
    def test_an_admitted_episode_is_written_and_finalized_and_its_arrays_saved(self):
        episode = PLAN["episodes"][12]
        traces = [{"qpos": np.zeros(9), "ctrl": np.zeros(8), "props_pose": np.zeros((1, 7))} for _ in range(3)]
        result = {"initial_state": np.zeros(4), "final_qpos": np.zeros(9)}
        with tempfile.TemporaryDirectory() as tmp:
            dataset = FakeDataset()
            committer = rec.Committer(tmp, dataset)
            committer(episode, ["a", "b", "c"], result, traces)
            self.assertEqual(dataset.events, ["frame"] * 3 + ["save", "finalize"])
            self.assertEqual(committer.count(), 7)
            with np.load(Path(tmp) / "episodes" / "episode_012.npz") as saved:
                self.assertEqual(str(saved["object"]), episode["object"])
                self.assertEqual(str(saved["bin"]), episode["bin"])
                self.assertEqual(str(saved["blue_slot"]), episode["blue_slot"])
                np.testing.assert_array_equal(saved["start"], episode["start"])
                self.assertEqual(saved["props_pose"].shape, (3, 1, 7))
                self.assertFalse([k for k in saved.files if "distractor" in k])


class CheckTests(unittest.TestCase):
    def test_low_disk_and_changed_provenance_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            provenance = contract.provenance()
            with mock.patch.object(contract, "MIN_FREE_BYTES", 1):
                rec.make_check(tmp, provenance)(0)
                with self.assertRaisesRegex(RuntimeError, "provenance changed"):
                    rec.make_check(tmp, {**provenance, "extra": "x"})(0)
            with mock.patch.object(contract, "MIN_FREE_BYTES", 1 << 60):
                with self.assertRaisesRegex(RuntimeError, "GB free"):
                    rec.make_check(tmp, provenance)(0)


def _gate_rows(failures=()):
    """One row per gate attempt; `failures` are attempt indices that the expert missed."""
    return [{"index": g["episode_index"], "object": g["object"], "bin": g["bin"], "blue_slot": g["blue_slot"],
             "admitted": g["episode_index"] not in failures,
             "reason": "admitted" if g["episode_index"] not in failures else "expert_success",
             "expert_reason": "success" if g["episode_index"] not in failures else "grasp", "gates": {}}
            for g in PLAN["gate_attempts"]]


def _spread(n):
    """n failed attempts spread evenly over the 12 object-and-arrangement cells (no per-object or per-slot bar trips)."""
    cells = len(p.CELLS)
    picked = {c * p.GATE_PER_CELL + round_ for round_ in range(n // cells + 1) for c in range(cells)
              if round_ * cells + c < n}
    assert len(picked) == n
    return picked


def _indices(obj=None, slot=None, n=1):
    picked = [g["episode_index"] for g in PLAN["gate_attempts"]
              if (obj is None or g["object"] == obj) and (slot is None or g["blue_slot"] == slot)]
    return picked[:n]


class GateSummaryTests(unittest.TestCase):
    def summary(self, failures=(), determinism=(("a", "x"), ("b", "x")), rows=None, split=4):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rows = _gate_rows(failures) if rows is None else rows
            for k in range(split):
                (out / f"gate_{k:02d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[k::split]))
            for tag, digest in determinism:
                (out / f"determinism_{tag}.json").write_text(json.dumps({"lp_num_threads": "3", "rows": {"0": {"digest": digest}}}))
            return rec.gate_summary(out)

    def test_the_bars_are_the_p0_bars(self):
        self.assertEqual((rec.GATE_OVERALL, rec.GATE_PER_OBJECT, rec.GATE_PER_SLOT), (0.95, 0.90, 0.93))

    def test_a_perfect_gate_with_equal_determinism_passes(self):
        summary = self.summary()
        self.assertTrue(summary["passed"])
        self.assertEqual((summary["attempts"], summary["admitted"]), (288, 288))
        self.assertEqual(len(summary["by_cell"]), 12)

    def test_the_overall_bar(self):
        self.assertTrue(self.summary(failures=_spread(12))["passed"])
        self.assertTrue(self.summary(failures=_spread(14))["passed"])           # 274/288 = 95.1%
        self.assertFalse(self.summary(failures=_spread(15))["passed"])          # 273/288 = 94.8%

    def test_the_per_object_bar_fails_when_the_overall_rate_would_pass(self):
        four = set(_indices(obj="milk", n=4))
        five = set(_indices(obj="milk", n=5))
        self.assertTrue(self.summary(failures=four)["passed"])                  # 44/48 = 91.7%
        summary = self.summary(failures=five)                                    # 43/48 = 89.6%, overall 98.3%
        self.assertGreaterEqual(summary["overall"], rec.GATE_OVERALL)
        self.assertFalse(summary["passed"])

    def test_the_per_slot_bar(self):
        right = [g["episode_index"] for g in PLAN["gate_attempts"] if g["blue_slot"] == "right"]
        by_object = {}
        for i in right:
            by_object.setdefault(PLAN["gate_attempts"][i]["object"], []).append(i)
        spread = [i for o in p.OBJECTS for i in by_object[o][:2]]               # two per object: 12 failures
        self.assertTrue(self.summary(failures=set(spread[:10]))["passed"])      # 134/144 = 93.06%
        summary = self.summary(failures=set(spread[:11]))                        # 133/144 = 92.4%
        self.assertGreaterEqual(summary["overall"], rec.GATE_OVERALL)
        self.assertGreaterEqual(min(summary["by_object"].values()), rec.GATE_PER_OBJECT)
        self.assertFalse(summary["passed"])

    def test_determinism_must_be_run_twice_and_equal(self):
        self.assertFalse(self.summary(determinism=(("a", "x"),))["passed"])
        self.assertFalse(self.summary(determinism=())["passed"])
        self.assertFalse(self.summary(determinism=(("a", "x"), ("b", "y")))["passed"])
        self.assertTrue(self.summary(determinism=(("a", "x"), ("b", "x"), ("c", "x")))["passed"])
        self.assertFalse(self.summary(determinism=(("a", "x"), ("b", "x"), ("c", "y")))["passed"])

    def test_an_incomplete_or_duplicated_gate_is_refused(self):
        rows = _gate_rows()
        with self.assertRaises(SystemExit):
            self.summary(rows=rows[:-1])
        with self.assertRaises(SystemExit):
            self.summary(rows=rows[:-1] + [rows[0]])

    def test_reasons_are_reported_for_the_failures(self):
        summary = self.summary(failures={3, 4})
        self.assertEqual(summary["reasons"], {"expert_success": 2})
        self.assertEqual(summary["expert_reasons"], {"grasp": 2})
        self.assertEqual(summary["determinism_lp_num_threads"], ["3", "3"])

    def test_determinism_covers_one_attempt_per_object_and_arrangement(self):
        picked = [PLAN["gate_attempts"][i] for i in rec.determinism_indices()]
        self.assertEqual(sorted((g["object"], g["blue_slot"]) for g in picked),
                         sorted((o, s) for o in p.OBJECTS for s in p.SLOTS))


class StatusAndJsonTests(unittest.TestCase):
    def test_substitutions_list_only_admitted_slots_that_moved(self):
        journal = [{"slot": 1, "admitted": True, "r": 0}, {"slot": 2, "admitted": True, "r": 2},
                   {"slot": 3, "admitted": False, "r": 1}, {"slot": 4, "admitted": True}]
        self.assertEqual(substitutions_of(journal), {"2": 2})
    def test_status_counts_admitted_and_rejected_per_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            for k, lines in enumerate((
                    [{"slot": 0, "admitted": True, "reason": "admitted"}, {"slot": 1, "admitted": False, "reason": "replay"},
                     {"slot": 1, "admitted": True, "reason": "admitted"}],
                    [{"slot": 34, "admitted": False, "reason": "labels"}])):
                root = rec.shard_root(tmp, k)
                root.mkdir()
                (root / "journal.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
            (rec.shard_root(tmp, 0) / "done.json").write_text("{}")
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                rec.status(tmp)
            text = buffer.getvalue()
            self.assertIn("shard_00: admitted=2 rejected=1 done=True {'replay': 1}", text)
            self.assertIn("shard_01: admitted=0 rejected=1 done=False {'labels': 1}", text)
            self.assertIn("TOTAL admitted=2/544 rejected=2", text)

    def test_write_json_replaces_atomically_and_leaves_no_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            rec._write_json(path, {"a": 1})
            rec._write_json(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text()), {"a": 2})
            self.assertEqual([q.name for q in Path(tmp).iterdir()], ["x.json"])
            with self.assertRaises(ValueError):
                rec._write_json(path, {"a": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"a": 2})


class ConstantsTests(unittest.TestCase):
    def test_slot_and_budget_constants(self):
        self.assertEqual(rec.MAX_ATTEMPTS_PER_SLOT, 12)
        self.assertEqual(rec.REJECTION_BUDGET, 150)
        self.assertEqual(rec.MUJOCO_VERSION, "3.4.0")

    def test_the_shard_count_divides_the_slots(self):
        self.assertEqual(p.ADMIT_COUNT % contract.SHARDS, 0)


if __name__ == "__main__":
    unittest.main()
