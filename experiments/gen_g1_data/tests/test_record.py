import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.gen_g1_data import plan as p
from experiments.gen_g1_data import record as r
from experiments.gen_g1_data import validate as v
from experiments.vla_inputs_signals.sidecar import KEYS

PLAN = p.make_plan()
FRAMES = p.EXPECTED_EPISODE_FRAMES


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
        resample = int(episode["layout_id"].rsplit("_r", 1)[1]) if "_r" in episode["layout_id"] else 0
        return resample * 3 + k

    def execute(self, episode, draw):
        slot, attempt = episode["episode_index"], self.attempt_of(episode, draw)
        if self.stop_at == (slot, attempt):
            raise Interrupted()
        self.executed.append((slot, attempt, episode["layout_id"], draw["seed"], tuple(episode["start"]), episode["condition_sha256"]))
        ok = (slot, attempt) not in self.reject
        verdict = {"admitted": ok, "reason": "admitted" if ok else "distractors", "gates": {"distractors": ok},
                   "details": {"distractor_motion": [], "expert_reason": "success"}}
        return {}, ["frame"], [{}], verdict

    def commit(self, episode, draw, frames, result, traces):
        self.committed.append((episode["episode_index"], episode["layout_id"], draw["seed"]))


class ShardSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.journal = Path(self.tmp.name) / "journal.jsonl"
        self.log = []

    def tearDown(self):
        self.tmp.cleanup()

    def run_(self, fake, slots, **kwargs):
        return r.run_schedule(PLAN, fake.execute, self.journal, fake.commit, slots, log=self.log.append, **kwargs)

    def test_a_shard_admits_exactly_its_block_in_order(self):
        fake = Fake()
        slots = p.shard_slots(3, 16)
        self.assertEqual(self.run_(fake, slots), {"admitted": 188, "rejected": 0})
        self.assertEqual([c[0] for c in fake.committed], slots)
        self.assertEqual([j["slot"] for j in r.read_journal(self.journal)], slots)

    def test_sixteen_shards_together_admit_every_slot_once(self):
        admitted = []
        for k in range(16):
            journal = Path(self.tmp.name) / f"j{k}.jsonl"
            fake = Fake()
            r.run_schedule(PLAN, fake.execute, journal, fake.commit, p.shard_slots(k, 16), log=self.log.append)
            admitted += [j["slot"] for j in r.read_journal(journal) if j["admitted"]]
        self.assertEqual(admitted, list(range(3000)))

    def test_a_rejected_attempt_never_commits_and_retries_the_same_slot_with_the_next_draw(self):
        fake = Fake(reject={(5, 0), (5, 1)})
        self.assertEqual(self.run_(fake, [5, 6]), {"admitted": 2, "rejected": 2})
        self.assertEqual([c[0] for c in fake.committed], [5, 6])
        self.assertEqual([c[1] for c in fake.executed if c[0] == 5], [0, 1, 2])
        self.assertEqual(len({c[3] for c in fake.executed if c[0] == 5}), 3)
        self.assertEqual(r.read_journal(self.journal)[0]["reason"], "distractors")
        self.assertEqual(fake.committed[0][2], fake.executed[2][3])

    def test_three_rejections_redraw_the_whole_scene_inside_the_stratum(self):
        fake = Fake(reject={(40, 0), (40, 1), (40, 2)})
        self.run_(fake, [40])
        line = r.read_journal(self.journal)[-1]
        self.assertTrue(line["admitted"] and line["r"] == 1 and line["k"] == 0)
        self.assertEqual(len({c[5] for c in fake.executed}), 2)
        self.assertEqual(len({c[4] for c in fake.executed}), 2)
        self.assertEqual(r.substitutions_of(r.read_journal(self.journal)), {"40": 1})
        self.assertEqual(fake.committed[0][1], f"{PLAN['episodes'][40]['layout_id']}_r1")

    def test_a_journal_line_never_claims_an_episode_the_dataset_lacks(self):
        fake = Fake()

        def failing_commit(episode, draw, frames, result, traces):
            raise Interrupted()

        with self.assertRaises(Interrupted):
            r.run_schedule(PLAN, fake.execute, self.journal, failing_commit, [5], log=self.log.append)
        self.assertFalse(self.journal.exists() and r.read_journal(self.journal))

    def test_a_slot_failing_every_scene_stops_the_shard(self):
        with self.assertRaises(r.SlotExhausted):
            self.run_(Fake(reject={(20, a) for a in range(12)}), [20, 21])

    def test_a_slot_that_fails_every_scene_is_skipped_only_in_a_pilot(self):
        fake = Fake(reject={(20, a) for a in range(12)})
        self.assertEqual(self.run_(fake, [20, 21], strict=False), {"admitted": 1, "rejected": 12})
        self.assertEqual([c[0] for c in fake.committed], [21])

    def test_the_shard_budget_is_per_shard_and_counts_rejections(self):
        slots = p.shard_slots(0, 16)[:45]
        reject = {(s, a) for s in slots for a in range(3)}
        self.assertEqual(self.run_(Fake(reject=reject), slots, budget=135)["rejected"], 135)
        self.journal.unlink()
        with self.assertRaises(r.BudgetExhausted):
            self.run_(Fake(reject=reject), slots, budget=134)

    def test_a_resumed_shard_over_budget_refuses_to_start(self):
        slots = p.shard_slots(0, 16)[:3]
        self.run_(Fake(reject={(0, 0), (0, 1)}), slots)
        with self.assertRaises(r.BudgetExhausted):
            self.run_(Fake(), slots, resume=True, budget=1)

    def test_resume_continues_and_refuses_mismatches(self):
        slots = p.shard_slots(1, 16)
        first = Fake(reject={(slots[2], 0)}, stop_at=(slots[2], 1))
        with self.assertRaises(Interrupted):
            self.run_(first, slots)
        with self.assertRaises(ValueError):
            self.run_(Fake(), slots, resume=True, dataset_count=lambda: 1)
        second = Fake(reject={(slots[2], 0)})
        self.run_(second, slots, resume=True, dataset_count=lambda: 2)
        self.assertEqual(second.executed[0][:2], (slots[2], 1))
        self.assertEqual([c[0] for c in first.committed + second.committed], slots)
        with self.assertRaises(ValueError):
            self.run_(Fake(), p.shard_slots(2, 16), resume=True)

    def test_fresh_run_over_a_journal_and_resume_without_one_refuse(self):
        with self.assertRaises(ValueError):
            self.run_(Fake(), [0], resume=True)
        self.run_(Fake(), [0])
        with self.assertRaises(FileExistsError):
            self.run_(Fake(), [0])

    def test_truncated_last_journal_line_is_dropped_and_an_earlier_bad_line_raises(self):
        self.run_(Fake(), [0, 1])
        with self.journal.open("a") as stream:
            stream.write('{"slot": 2, "adm')
        self.assertEqual(len(r.read_journal(self.journal)), 2)
        with self.journal.open("a") as stream:
            stream.write('\n{"slot": 3}\n')
        with self.assertRaises(json.JSONDecodeError):
            r.read_journal(self.journal)

    def test_the_journal_records_the_expert_reason_motion_and_the_log_tag(self):
        self.run_(Fake(reject={(0, 0)}), [0], tag="G1 shard=0")
        first = r.read_journal(self.journal)[0]
        self.assertEqual((first["expert_reason"], first["distractor_motion"]), ("success", []))
        self.assertTrue(self.log[0].startswith("G1 shard=0 "))

    def test_the_disk_and_provenance_check_runs_at_the_start_and_every_fifth_admission(self):
        calls = []
        self.run_(Fake(), list(range(11)), check=calls.append)
        self.assertEqual(calls, [0, 5, 10])


class TraceFilesTests(unittest.TestCase):
    def setUp(self):
        self.episode = PLAN["episodes"][7]
        self.draw = p.attempt_draw(self.episode, 0)
        self.traces = [{"qpos": np.full(18, t, np.float64), "ctrl": np.zeros(8), "sim_time": 0.03 * t,
                        "props_pose": np.zeros((3, 7)), "phase_index": t // 60, "substeps": 5, "success_tick": t > 440}
                       for t in range(FRAMES)]
        self.result = {"initial_state": np.arange(5.0), "final_qpos": np.zeros(18), "primitives": ["cube_red"]}

    def arrays(self):
        return r.trace_arrays(self.episode, self.draw, self.result, self.traces)

    def test_the_arrays_hold_the_trace_and_the_scene(self):
        arrays = self.arrays()
        self.assertEqual(arrays["qpos"].shape, (FRAMES, 18))
        self.assertEqual(arrays["phase_index"][-1], (FRAMES - 1) // 60)
        self.assertEqual(arrays["success_tick"].dtype, bool)
        self.assertEqual(json.loads(str(arrays["condition"])), self.episode["condition"])
        self.assertEqual(json.loads(str(arrays["draw"])), {"seed": self.draw["seed"], "home_pose_jitter_deg": self.draw["home_pose_jitter_deg"]})
        self.assertEqual(list(arrays["primitives"]), ["cube_red"])

    def test_the_trace_records_the_draw_that_ran_not_the_planned_one(self):
        draw = p.attempt_draw(self.episode, 2)
        self.assertNotEqual(draw["home_pose_jitter_deg"], self.episode["home_pose_jitter_deg"])
        arrays = r.trace_arrays(self.episode, draw, self.result, self.traces)
        self.assertEqual(list(arrays["home_pose_jitter_deg"]), draw["home_pose_jitter_deg"])
        self.assertEqual(json.loads(str(arrays["draw"]))["seed"], draw["seed"])

    def test_what_the_recorder_writes_is_what_the_validator_reads(self):
        episodes = [PLAN["episodes"][7], PLAN["episodes"][8]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "episodes").mkdir()
            (root / "signals").mkdir()
            for e in episodes:
                draw = p.attempt_draw(e, 1)
                r.write_npz(root / "episodes" / r.episode_name(e), r.trace_arrays(e, draw, self.result, self.traces))
                r.write_npz(root / "signals" / r.episode_name(e), _signals(e))
            v.check_recorded_files(root, episodes)
            self.assertEqual(sorted(q.name for q in (root / "episodes").iterdir()), ["episode_0007.npz", "episode_0008.npz"])

    def test_a_substitute_scene_is_what_the_trace_says(self):
        scene = p.resample_start(self.episode, 1)
        draw = p.attempt_draw(self.episode, 0)
        arrays = r.trace_arrays(scene, draw, self.result, self.traces)
        self.assertEqual(str(arrays["layout_id"]), scene["layout_id"])
        self.assertNotEqual(str(arrays["condition_sha256"]), self.episode["condition_sha256"])
        self.assertEqual(r.episode_name(scene), r.episode_name(self.episode))

    def test_write_npz_is_atomic_and_replaces_what_a_killed_run_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode_0001.npz"
            r.write_npz(path, {"x": np.arange(3)})
            r.write_npz(path, {"x": np.arange(5)})
            with np.load(path) as stored:
                self.assertEqual(len(stored["x"]), 5)
            self.assertEqual([q.name for q in Path(tmp).iterdir()], ["episode_0001.npz"])


def _signals(episode):
    arrays = {"episode_index": np.array(episode["episode_index"]), "frame_index": np.arange(FRAMES),
              "wrist_wrench": np.zeros((FRAMES, 6)), "finger_force": np.zeros((FRAMES, 2)), "target_box": np.zeros((FRAMES, 2, 5)),
              "props_box": np.zeros((FRAMES, 3, 2, 5)), "prop_names": np.array([episode["object"], "x", "y"]),
              "segment": np.repeat(np.arange(4), FRAMES // 4).astype(np.int8), "phase_index": np.zeros(FRAMES, np.int64),
              "object": np.array(episode["object"]), "split": np.array(episode["split"])}
    assert set(arrays) == set(KEYS)
    return arrays


class CommitterTests(unittest.TestCase):
    class Dataset:
        def __init__(self, root, order):
            self.root, self.order = Path(root), order
            self.meta = type("Meta", (), {"total_episodes": 0})()

        def add_frame(self, frame):
            self.order.append(("add_frame", sorted(q.name for q in (self.root / "episodes").iterdir()),
                               sorted(q.name for q in (self.root / "signals").iterdir())))

        def save_episode(self):
            self.order.append("save_episode")

        def finalize(self):
            self.order.append("finalize")

    def commit(self, tmp, episode=PLAN["episodes"][3], root_dir=None):
        order = []
        root = Path(tmp)
        committer = r.Committer(root, self.Dataset(root, order))
        draw = p.attempt_draw(episode, 0)
        result = {"initial_state": np.zeros(3), "final_qpos": np.zeros(18), "primitives": [], "signals": _signals(episode)}
        traces = [{"qpos": np.zeros(18), "ctrl": np.zeros(8), "sim_time": 0.0, "props_pose": np.zeros((3, 7)), "phase_index": 0,
                   "substeps": 5, "success_tick": False}] * 3
        committer(episode, draw, [{}] * 3, result, traces)
        return order

    def test_both_files_exist_before_the_dataset_episode_and_the_episode_is_finalized_at_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            order = self.commit(tmp)
        self.assertEqual(order[0], ("add_frame", ["episode_0003.npz"], ["episode_0003.npz"]))
        self.assertEqual(order[-2:], ["save_episode", "finalize"])
        self.assertEqual(len(order), 3 + 2)

    def test_a_file_left_by_a_killed_run_is_replaced_not_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for folder in ("episodes", "signals"):
                (Path(tmp) / folder).mkdir()
                (Path(tmp) / folder / "episode_0003.npz").write_bytes(b"left behind")
            self.commit(tmp)
            with np.load(Path(tmp) / "signals" / "episode_0003.npz") as stored:
                self.assertEqual(set(stored.files), set(KEYS))


def _rows(failing=()):
    """The 96 gate attempts as gate rows; `failing` is a set of attempt indices that were rejected."""
    return [{"index": a["episode_index"], "object": a["object"], "quintile": a["quintile"], "distractor_count": a["distractor_count"],
             "region": a["region"], "admitted": a["episode_index"] not in failing, "expert_success": a["episode_index"] not in failing,
             "reason": "admitted" if a["episode_index"] not in failing else "expert_success",
             "expert_reason": "success" if a["episode_index"] not in failing else "grasp",
             "distractor_motion": [{"object": "x", "displacement_m": 0.001, "tilt_deg": 0.0}]} for a in PLAN["gate_attempts"]]


def _of(field, value):
    return [a["episode_index"] for a in PLAN["gate_attempts"] if a[field] == value]


class GateVerdictTests(unittest.TestCase):
    def test_the_gate_is_the_96_attempts_balanced_on_what_it_reports(self):
        rows = _rows()
        self.assertEqual(len(rows), 96)
        verdict = r.gate_verdict(rows)
        self.assertTrue(verdict["passed"], verdict["failed_bars"])
        self.assertEqual({k: s["n"] for k, s in verdict["by_object"].items()}, {o: 16 for o in p.OBJECTS})
        self.assertEqual({k: s["n"] for k, s in verdict["by_distractor_count"].items()}, {str(n): 16 for n in range(6)})
        self.assertEqual(sorted(s["n"] for s in verdict["by_quintile"].values()), [19, 19, 19, 19, 20])

    def test_four_failures_pass_overall_five_do_not(self):
        four = r.gate_verdict(_rows(set(self._spread(4))))
        self.assertEqual(four["overall"], 92 / 96)
        self.assertTrue(four["passed"], four["failed_bars"])
        verdict = r.gate_verdict(_rows(set(self._spread(5))))
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["failed_bars"], [f"overall {91 / 96:.3f}"])

    def _spread(self, n):
        """n failing attempts, no two sharing an object, a quintile or a distractor count where avoidable."""
        chosen, used = [], [set(), set(), set()]
        for a in PLAN["gate_attempts"]:
            keys = (a["object"], a["quintile"], a["distractor_count"])
            if all(k not in u for k, u in zip(keys, used)):
                chosen.append(a["episode_index"])
                for k, u in zip(keys, used):
                    u.add(k)
            if len(chosen) == n:
                return chosen
        raise AssertionError("no spread of failures")

    def test_two_failures_in_one_object_or_count_or_stratum_fail_the_bar(self):
        for field, value in (("object", "lemon"), ("distractor_count", 3), ("quintile", 2)):
            ids = _of(field, value)
            one = r.gate_verdict(_rows(set(ids[:1])))
            self.assertTrue(one["passed"], (field, one["failed_bars"]))
            two = r.gate_verdict(_rows(set(ids[:2])))
            self.assertFalse(two["passed"], field)
            self.assertTrue(any(f.startswith(f"{field}={value}") for f in two["failed_bars"]), two["failed_bars"])

    def test_exactly_ninety_five_percent_overall_passes(self):
        rows = [{**_rows()[0], "index": i, "admitted": i != 0, "expert_success": i != 0, "reason": "admitted" if i else "grasp"}
                for i in range(20)]
        verdict = r.gate_verdict(rows)
        self.assertEqual(verdict["overall"], 0.95)
        self.assertTrue(verdict["passed"], verdict["failed_bars"])
        rows[1].update(admitted=False, expert_success=False)
        self.assertFalse(r.gate_verdict(rows)["passed"])

    def test_exactly_ninety_percent_of_a_stratum_passes(self):
        twenty = next(q for q in range(5) if len(_of("quintile", q)) == 20)
        two = r.gate_verdict(_rows(set(_of("quintile", twenty)[:2])))
        self.assertEqual(two["by_quintile"][str(twenty)]["rate"], 0.9)
        self.assertTrue(two["passed"], two["failed_bars"])
        three = r.gate_verdict(_rows(set(_of("quintile", twenty)[:3])))
        self.assertFalse(three["passed"])

    def test_the_bar_is_on_admission_and_the_expert_success_is_reported_beside_it(self):
        rows = _rows()
        for row in rows[:6]:
            row.update(admitted=False, reason="distractors", expert_success=True)
        verdict = r.gate_verdict(rows)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["expert_success_overall"], 1.0)
        self.assertEqual(verdict["reasons"], {"distractors": 6})

    def test_the_bars_are_the_registered_ones(self):
        self.assertEqual((r.GATE_OVERALL, r.GATE_PER_STRATUM, r.GATE_FIELDS), (0.95, 0.90, ("object", "quintile", "distractor_count")))
        self.assertEqual(r.gate_verdict(_rows())["bars"], {"overall": 0.95, "per_stratum": 0.90, "strata": ["object", "quintile", "distractor_count"]})

    def test_the_region_is_reported_but_is_not_a_bar(self):
        verdict = r.gate_verdict(_rows())
        self.assertEqual(set(verdict["by_region"]), {"outer", "p2"})
        rows = _rows(set(_of("region", "outer")[:6]))
        self.assertNotIn("region", "".join(r.gate_verdict(rows)["failed_bars"]))

    def test_the_largest_distractor_displacement_is_reported(self):
        rows = _rows()
        rows[3]["distractor_motion"].append({"object": "y", "displacement_m": 0.02, "tilt_deg": 3.0})
        self.assertEqual(r.gate_verdict(rows)["max_distractor_displacement_m"], 0.02)


class GateFilesTests(unittest.TestCase):
    def write(self, out, rows, parts=2, determinism=()):
        for k in range(parts):
            with (Path(out) / f"gate_{k:02d}.jsonl").open("w") as stream:
                for row in rows[k::parts]:
                    stream.write(json.dumps(row) + "\n")
        for tag, digest in determinism:
            (Path(out) / f"determinism_{tag}.json").write_text(json.dumps({"lp_num_threads": "4", "rows": {"0": {"digest": digest}}}))

    def summary(self, out):
        with contextlib.redirect_stdout(io.StringIO()):
            r.gate_summary(out)
        return json.loads((Path(out) / "gate_summary.json").read_text())

    def test_the_summary_reads_every_part_and_judges_determinism_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp, _rows(), determinism=(("a", "x"), ("b", "x")))
            summary = self.summary(tmp)
            self.assertTrue(summary["passed"])
            self.assertTrue(summary["determinism_equal"])
            self.assertEqual(summary["determinism_lp_num_threads"], ["4", "4"])
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp, _rows(), determinism=(("a", "x"), ("b", "y")))
            summary = self.summary(tmp)
            self.assertFalse(summary["determinism_equal"])
            self.assertTrue(summary["passed"])
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp, _rows(), determinism=(("a", "x"),))
            self.assertFalse(self.summary(tmp)["determinism_equal"])

    def test_an_incomplete_or_duplicated_gate_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp, _rows()[:-1])
            with self.assertRaises(SystemExit):
                self.summary(tmp)
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp, _rows() + _rows()[:1])
            with self.assertRaises(SystemExit):
                self.summary(tmp)

    def test_gate_rows_name_what_the_gate_needs(self):
        attempt = PLAN["gate_attempts"][4]
        verdict = {"admitted": False, "reason": "distractors", "gates": {"expert_success": True},
                   "details": {"distractor_touched": [{"object": "x"}]}}
        row = r.gate_row(attempt, {"reason": "success", "distractor_motion": []}, verdict, 3.14159)
        self.assertEqual((row["index"], row["object"], row["quintile"], row["distractor_count"], row["region"]),
                         (4, attempt["object"], attempt["quintile"], attempt["distractor_count"], attempt["region"]))
        self.assertEqual((row["admitted"], row["expert_success"], row["wall_seconds"]), (False, True, 3.1))
        json.dumps(row)

    def test_determinism_samples_twelve_distinct_gate_attempts(self):
        indices = r.determinism_indices()
        self.assertEqual((len(indices), len(set(indices))), (12, 12))
        self.assertTrue(all(0 <= i < p.GATE_ATTEMPTS for i in indices))


class StatusTests(unittest.TestCase):
    def test_status_totals_admissions_and_rejections_per_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            for k, lines in ((0, [{"admitted": True, "reason": "admitted"}, {"admitted": False, "reason": "frames"}]),
                             (1, [{"admitted": True, "reason": "admitted"}])):
                root = r.shard_root(tmp, k)
                root.mkdir()
                (root / "journal.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                r.status(tmp)
            text = out.getvalue()
            self.assertIn("shard_00: admitted=1 rejected=1 done=False {'frames': 1}", text)
            self.assertIn("TOTAL admitted=2/3000 rejected=1", text)


if __name__ == "__main__":
    unittest.main()
