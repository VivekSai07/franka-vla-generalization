import json
import tempfile
import unittest
from pathlib import Path

from experiments.props_p2_data import plan as p
from experiments.props_p2_data.record import (
    BudgetExhausted, SlotExhausted, read_journal, run_schedule, substitutions_of,
)

PLAN = p.make_plan()


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
        verdict = {"admitted": ok, "reason": "admitted" if ok else "distractors", "gates": {"distractors": ok},
                   "details": {"distractor_motion": []}}
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
        slots = p.shard_slots(3, 12)
        self.assertEqual(self.run_(fake, slots), {"admitted": 45, "rejected": 0})
        self.assertEqual([s for s, _ in fake.committed], slots)
        self.assertEqual([j["slot"] for j in read_journal(self.journal)], slots)

    def test_twelve_shards_together_admit_every_slot_once(self):
        admitted = []
        for k in range(12):
            journal = Path(self.tmp.name) / f"j{k}.jsonl"
            fake = Fake()
            run_schedule(PLAN, fake.execute, journal, fake.commit, p.shard_slots(k, 12), log=self.log.append)
            admitted += [j["slot"] for j in read_journal(journal) if j["admitted"]]
        self.assertEqual(admitted, list(range(540)))

    def test_a_rejected_attempt_never_commits_and_retries_the_same_slot(self):
        # Review Focus 4.
        fake = Fake(reject={(5, 0), (5, 1)})
        self.assertEqual(self.run_(fake, [5, 6]), {"admitted": 2, "rejected": 2})
        self.assertEqual([s for s, _ in fake.committed], [5, 6])
        self.assertEqual([c[1] for c in fake.executed if c[0] == 5], [0, 1, 2])
        self.assertEqual(read_journal(self.journal)[0]["reason"], "distractors")

    def test_three_rejections_resample_the_whole_layout_keeping_the_objects(self):
        fake = Fake(reject={(40, 0), (40, 1), (40, 2)})
        self.run_(fake, [40])
        line = read_journal(self.journal)[-1]
        self.assertTrue(line["admitted"] and line["r"] == 1)
        starts = {c[4] for c in fake.executed}
        self.assertEqual(len(starts), 2)
        self.assertEqual(substitutions_of(read_journal(self.journal)), {"40": 1})

    def test_a_slot_failing_every_layout_stops_the_shard(self):
        with self.assertRaises(SlotExhausted):
            self.run_(Fake(reject={(20, a) for a in range(12)}), [20, 21])

    def test_the_shard_budget(self):
        slots = p.shard_slots(0, 12)
        self.assertEqual(self.run_(Fake(reject={(s, a) for s in slots for a in range(3)}), slots, budget=135)["rejected"], 135)
        self.journal.unlink()
        with self.assertRaises(BudgetExhausted):
            self.run_(Fake(reject={(s, a) for s in slots for a in range(3)}), slots, budget=134)

    def test_resume_continues_and_refuses_mismatches(self):
        slots = p.shard_slots(1, 12)
        first = Fake(reject={(47, 0)}, stop_at=(47, 1))
        with self.assertRaises(Interrupted):
            self.run_(first, slots)
        with self.assertRaises(ValueError):
            self.run_(Fake(), slots, resume=True, dataset_count=lambda: 1)
        second = Fake(reject={(47, 0)})
        self.run_(second, slots, resume=True, dataset_count=lambda: 2)
        self.assertEqual([c[:2] for c in second.executed][0], (47, 1))
        self.assertEqual([s for s, _ in first.committed + second.committed], slots)
        with self.assertRaises(ValueError):
            self.run_(Fake(), p.shard_slots(2, 12), resume=True)

    def test_fresh_run_over_a_journal_and_resume_without_one_refuse(self):
        with self.assertRaises(ValueError):
            self.run_(Fake(), [0], resume=True)
        self.run_(Fake(), [0])
        with self.assertRaises(FileExistsError):
            self.run_(Fake(), [0])

    def test_truncated_last_journal_line_is_dropped(self):
        self.run_(Fake(), [0, 1])
        with self.journal.open("a") as stream:
            stream.write('{"slot": 2, "adm')
        self.assertEqual(len(read_journal(self.journal)), 2)

    def test_journal_records_distractor_motion_and_log_tag(self):
        self.run_(Fake(), [0], tag="P2 shard=0")
        self.assertIn("distractor_motion", read_journal(self.journal)[0])
        self.assertTrue(self.log[0].startswith("P2 shard=0 "))
        json.dumps(read_journal(self.journal))


if __name__ == "__main__":
    unittest.main()
