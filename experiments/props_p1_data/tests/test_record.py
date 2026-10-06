import json
import tempfile
import unittest
from pathlib import Path

from experiments.props_p1_data import plan as p
from experiments.props_p1_data.record import (
    BudgetExhausted, SlotExhausted, read_journal, run_schedule,
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
        self.executed.append((slot, attempt, episode["layout_id"], draw["seed"]))
        ok = (slot, attempt) not in self.reject
        verdict = {"admitted": ok, "reason": "admitted" if ok else "replay", "gates": {"replay": ok}}
        return {"initial_state": None}, ["frame"], [{}], verdict

    def commit(self, episode, frames, result, traces):
        self.committed.append((episode["episode_index"], episode["layout_id"]))


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.journal = Path(self.tmp.name) / "journal.jsonl"
        self.log = []

    def tearDown(self):
        self.tmp.cleanup()

    def run_(self, fake, **kwargs):
        return run_schedule(PLAN, fake.execute, self.journal, fake.commit, log=self.log.append, **kwargs)

    def test_every_slot_admitted_in_order(self):
        fake = Fake()
        summary = self.run_(fake)
        self.assertEqual(summary, {"admitted": 540, "rejected": 0})
        self.assertEqual([s for s, _ in fake.committed], list(range(540)))
        self.assertEqual([j["slot"] for j in read_journal(self.journal)], list(range(540)))
        self.assertTrue(all(line.startswith("P1 ") for line in self.log))

    def test_rejection_retries_the_same_slot_with_a_different_draw(self):
        fake = Fake(reject={(3, 0)})
        self.run_(fake, slots=[3])
        calls = [c for c in fake.executed if c[0] == 3]
        self.assertEqual([c[1] for c in calls], [0, 1])
        self.assertNotEqual(calls[0][3], calls[1][3])
        self.assertEqual(calls[0][2], calls[1][2])  # same start pose until three attempts fail
        self.assertEqual(fake.committed.count((3, PLAN["episodes"][3]["layout_id"])), 1)

    def test_a_rejected_attempt_never_commits(self):
        # Review Focus 4.
        fake = Fake(reject={(5, 0), (5, 1), (9, 0)})
        summary = self.run_(fake, slots=[5, 9])
        self.assertEqual(summary, {"admitted": 2, "rejected": 3})
        self.assertEqual([s for s, _ in fake.committed], [5, 9])
        lines = read_journal(self.journal)
        self.assertEqual([j["admitted"] for j in lines], [False, False, True, False, True])
        self.assertEqual(lines[0]["reason"], "replay")

    def test_slot_failing_every_start_and_attempt_stops_the_run(self):
        fake = Fake(reject={(20, a) for a in range(12)})
        with self.assertRaises(SlotExhausted):
            self.run_(fake)
        lines = read_journal(self.journal)
        self.assertEqual(sum(j["slot"] == 20 for j in lines), 12)
        self.assertEqual(sorted({j["r"] for j in lines if j["slot"] == 20}), [0, 1, 2, 3])
        self.assertEqual(len([j for j in lines if j["admitted"]]), 20)

    def test_the_151st_rejection_stops_the_run(self):
        fake = Fake(reject={(s, 0) for s in range(151)})
        with self.assertRaises(BudgetExhausted):
            self.run_(fake)
        self.assertEqual(sum(not j["admitted"] for j in read_journal(self.journal)), 151)

    def test_150_rejections_are_allowed(self):
        fake = Fake(reject={(s, 0) for s in range(150)})
        self.assertEqual(self.run_(fake)["rejected"], 150)

    def test_substitute_start_after_three_rejections(self):
        fake = Fake(reject={(40, 0), (40, 1), (40, 2)})
        self.run_(fake, slots=[40])
        line = read_journal(self.journal)[-1]
        self.assertTrue(line["admitted"] and line["r"] == 1 and line["attempt"] == 3)
        self.assertTrue(line["layout_id"].endswith("_r1"))
        self.assertEqual(fake.committed, [(40, PLAN["episodes"][40]["layout_id"] + "_r1")])
        substitutions = json.loads((self.journal.with_name("substitutions.json")).read_text())
        self.assertEqual(substitutions, {"40": 1})

    def test_interrupted_run_resumes_each_slot_exactly_once(self):
        first = Fake(stop_at=(7, 0))
        with self.assertRaises(Interrupted):
            self.run_(first)
        second = Fake()
        self.run_(second, resume=True)
        committed = [s for s, _ in first.committed] + [s for s, _ in second.committed]
        self.assertEqual(committed, list(range(540)))
        self.assertEqual([j["slot"] for j in read_journal(self.journal) if j["admitted"]], list(range(540)))

    def test_resume_continues_a_half_finished_slot_at_its_next_attempt(self):
        first = Fake(reject={(4, 0), (4, 1)}, stop_at=(4, 2))
        with self.assertRaises(Interrupted):
            self.run_(first)
        second = Fake(reject={(4, 0), (4, 1)})
        self.run_(second, resume=True, slots=[4])
        self.assertEqual([c[:2] for c in second.executed if c[0] == 4], [(4, 2)])

    def test_resume_keeps_a_substitution(self):
        first = Fake(reject={(2, 0), (2, 1), (2, 2)}, stop_at=(6, 0))
        with self.assertRaises(Interrupted):
            self.run_(first)
        self.journal.with_name("substitutions.json").unlink()
        second = Fake()
        self.run_(second, resume=True, slots=range(0, 8))
        self.assertEqual(json.loads(self.journal.with_name("substitutions.json").read_text()), {"2": 1})
        self.assertEqual([s for s, _ in first.committed + second.committed], list(range(8)))

    def test_resume_refuses_a_dataset_that_disagrees_with_the_journal(self):
        first = Fake(stop_at=(7, 0))
        with self.assertRaises(Interrupted):
            self.run_(first)
        second = Fake()
        with self.assertRaises(ValueError):
            self.run_(second, resume=True, dataset_count=lambda: 8)
        self.assertEqual(second.executed, [])
        self.run_(Fake(), resume=True, dataset_count=lambda: 7)

    def test_resume_without_a_journal_and_a_fresh_run_over_a_journal_both_refuse(self):
        with self.assertRaises(ValueError):
            self.run_(Fake(), resume=True)
        self.run_(Fake(), slots=[0])
        with self.assertRaises(FileExistsError):
            self.run_(Fake(), slots=[1])

    def test_a_truncated_last_journal_line_is_ignored_but_a_bad_middle_line_is_not(self):
        self.run_(Fake(), slots=[0, 1])
        with self.journal.open("a") as stream:
            stream.write('{"slot": 2, "adm')  # killed mid-write
        self.assertEqual(len(read_journal(self.journal)), 2)
        with self.journal.open("a") as stream:
            stream.write('garbage\n{"slot": 3, "admitted": true}\n')
        with self.assertRaises(json.JSONDecodeError):
            read_journal(self.journal)


if __name__ == "__main__":
    unittest.main()
