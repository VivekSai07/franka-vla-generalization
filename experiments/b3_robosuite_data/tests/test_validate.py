import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.b3_robosuite_data import plan as p
from experiments.b3_robosuite_data.validate import (
    admit, distractor_disturbance, fingerprint, seal, validate_dataset,
)

COLOURS = ("red", "green", "blue", "yellow")


class _Joint:
    def __init__(self, adr):
        self.qposadr = [adr]


class _Body:
    def __init__(self, k):
        self.jntadr = [k]


class FakeModel:
    def body(self, name):
        return _Body(COLOURS.index(name.rsplit("_", 1)[1]))

    def joint(self, k):
        return _Joint(9 + 7 * k)


def qpos_with(moves=None):
    qpos = np.zeros(9 + 7 * 4)
    for colour, (x, y) in (moves or {}).items():
        adr = 9 + 7 * COLOURS.index(colour)
        qpos[adr:adr + 2] = (x, y)
    return qpos


class Fixture:
    def __init__(self):
        self.plan = p.make_plan()
        self.episode = next(e for e in self.plan["episodes"] if len(e["distractors"]) == 2)
        self.text = self.episode["instruction"]

    def good(self):
        e = self.episode
        result = {"success": True, "stable_ticks": 30, "frames": 528, "final_xy_m": 0.004, "final_z_m": 0.041,
                  "model": FakeModel(), "reset": {"qpos": qpos_with()}, "final_qpos": qpos_with()}
        frames = [{"observation.state": np.zeros(9, np.float32), "action": np.zeros(8, np.float32),
                   "task": self.text} for _ in range(528)]
        return copy.deepcopy(self.episode), result, frames, [{}] * 528


def replay_ok(result, frames, traces, pick, place):
    return {"max_policy_state_error": 1e-5}


class AdmitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = Fixture()

    def run_admit(self, mutate=None, replay=replay_ok):
        episode, result, frames, traces = self.fx.good()
        if mutate:
            mutate(episode, result, frames, traces)
        return admit(episode, result, frames, traces, replay=replay)

    def test_all_gates_pass(self):
        verdict = self.run_admit()
        self.assertTrue(verdict["admitted"], verdict)
        self.assertEqual(verdict["reason"], "admitted")
        self.assertTrue(all(verdict["gates"].values()))

    def test_each_gate_fails_alone_and_names_itself(self):
        cases = {
            "expert_success": lambda e, r, f, t: r.update(success=False),
            "frames": lambda e, r, f, t: (f.pop(), t.pop(), r.update(frames=527)),
            "stack": lambda e, r, f, t: r.update(final_xy_m=0.021),
            "distractors_undisturbed": lambda e, r, f, t: r.update(
                final_qpos=qpos_with({e["distractors"][0]: (0.025, 0.0)})),
            "labels": lambda e, r, f, t: f[10].update(task="pick up the red cube"),
        }
        for gate, mutate in cases.items():
            verdict = self.run_admit(mutate)
            self.assertEqual(verdict["reason"], gate, verdict)
            self.assertFalse(verdict["gates"][gate])
        verdict = self.run_admit(replay=lambda *a: {"max_policy_state_error": 2e-3})
        self.assertEqual(verdict["reason"], "replay")
        self.assertEqual([g for g, ok in verdict["gates"].items() if not ok], ["replay"])

    def test_gates_other_than_the_named_one_still_pass(self):
        verdict = self.run_admit(lambda e, r, f, t: r.update(final_z_m=0.05))
        self.assertEqual([g for g, ok in verdict["gates"].items() if not ok], ["stack"])

    def test_stable_ticks_below_thirty_fails_the_expert_gate(self):
        verdict = self.run_admit(lambda e, r, f, t: r.update(stable_ticks=29))
        self.assertEqual(verdict["reason"], "expert_success")

    def test_every_gate_is_recorded_after_the_first_failure(self):
        verdict = self.run_admit(lambda e, r, f, t: (r.update(final_xy_m=0.5),
                                                    f[0].update(task="x")))
        self.assertEqual(set(verdict["gates"]), {"expert_success", "frames", "stack",
                                                 "distractors_undisturbed", "replay", "labels"})
        self.assertEqual(verdict["reason"], "stack")
        self.assertFalse(verdict["gates"]["labels"])

    def test_distractor_moved_2_5_cm_rejects_but_a_moving_pick_cube_does_not(self):
        def distractor(e, r, f, t):
            r["final_qpos"] = qpos_with({e["distractors"][0]: (0.025, 0.0)})

        def pick(e, r, f, t):
            r["final_qpos"] = qpos_with({e["pick"]: (0.2, 0.3)})

        self.assertEqual(self.run_admit(distractor)["reason"], "distractors_undisturbed")
        self.assertTrue(self.run_admit(pick)["admitted"])
        episode, result, _, _ = self.fx.good()
        moved = distractor_disturbance(FakeModel(), qpos_with(), qpos_with({"blue": (0.0, 0.02)}), ["blue", "red"])
        self.assertAlmostEqual(moved["blue"], 0.02)
        self.assertEqual(moved["red"], 0.0)

    def test_exactly_two_centimetres_is_allowed(self):
        verdict = self.run_admit(lambda e, r, f, t: r.update(
            final_qpos=qpos_with({e["distractors"][0]: (0.02, 0.0)})))
        self.assertTrue(verdict["gates"]["distractors_undisturbed"])

    def test_replay_error_is_recorded_not_raised(self):
        def broken(*args):
            raise ValueError("Replay clock drift at tick 3")
        verdict = self.run_admit(replay=broken)
        self.assertEqual(verdict["reason"], "replay")
        self.assertIn("drift", verdict["details"]["replay_error"])

    def test_labels_reject_wrong_dtypes(self):
        verdict = self.run_admit(lambda e, r, f, t: f[5].update(action=np.zeros(8, np.float64)))
        self.assertEqual(verdict["reason"], "labels")


class FingerprintTests(unittest.TestCase):
    def make_root(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name) / "ds"
        (root / "meta").mkdir(parents=True)
        (root / "meta/stats.json").write_text(json.dumps({"a": 1, "b": 2}))
        (root / "data.bin").write_bytes(b"abc")
        return tmp, root

    def test_ignores_seal_and_key_order_but_not_values_or_other_files(self):
        tmp, root = self.make_root()
        with tmp:
            base = fingerprint(root)
            (root / "seal.json").write_text("{}")
            self.assertEqual(fingerprint(root), base)
            (root / "meta/stats.json").write_text(json.dumps({"b": 2, "a": 1}, indent=4))
            self.assertEqual(fingerprint(root), base)
            (root / "meta/stats.json").write_text(json.dumps({"a": 1, "b": 3}))
            self.assertNotEqual(fingerprint(root), base)
            (root / "meta/stats.json").write_text(json.dumps({"a": 1, "b": 2}))
            self.assertEqual(fingerprint(root), base)
            (root / "data.bin").write_bytes(b"abd")
            self.assertNotEqual(fingerprint(root), base)

    def test_rejects_symlinks(self):
        tmp, root = self.make_root()
        with tmp:
            (root / "link").symlink_to(root / "data.bin")
            with self.assertRaises(ValueError):
                fingerprint(root)

    def test_seal_records_what_training_will_check(self):
        tmp, root = self.make_root()
        with tmp:
            (root / "substitutions.json").write_text(json.dumps({"12": 1}))
            plan = {"min_clearance_m": 0.08, "episodes": [{}] * 560}
            sealed = seal(root, plan, {"x.py": "0" * 64})
            on_disk = json.loads((root / "seal.json").read_text())
            self.assertEqual(sealed, on_disk)
            self.assertEqual(set(on_disk), {"fingerprint", "provenance_sha256", "plan_sha256",
                                            "substitutions_sha256", "image", "min_clearance_m", "episodes"})
            self.assertEqual(fingerprint(root), on_disk["fingerprint"])


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = p.make_plan()

    def journal(self, substitute=None):
        return [{"slot": i, "admitted": True, "r": 1 if i == substitute else 0} for i in range(560)]

    def reader(self, mutate=None):
        meta = {"total_episodes": 560, "total_frames": 560 * 528, "fps": 30,
                "task_table": list(self.plan["instructions"]),
                "tasks": [[e["instruction"]] for e in self.plan["episodes"]], "lengths": [528] * 560,
                "indices": list(range(560))}
        if mutate:
            mutate(meta)
        return lambda root: meta

    def test_accepts_the_planned_dataset_and_a_substituted_slot(self):
        summary = validate_dataset("x", self.plan, self.journal(), self.reader())
        self.assertEqual(summary["episodes"], 560)
        self.assertEqual(summary["substituted_slots"], 0)
        self.assertEqual(validate_dataset("x", self.plan, self.journal(substitute=40), self.reader())["substituted_slots"], 1)

    def test_rejects_wrong_episode_count(self):
        def short(meta):
            meta["total_episodes"] = 559
        with self.assertRaises(ValueError):
            validate_dataset("x", self.plan, self.journal(), self.reader(short))
        with self.assertRaises(ValueError):
            validate_dataset("x", self.plan, self.journal()[:-1], self.reader())

    def test_rejects_an_instruction_that_disagrees_with_its_slot(self):
        def swap(meta):
            meta["tasks"][17] = [self.plan["instructions"][0 if meta["tasks"][17] != [self.plan["instructions"][0]] else 1]]
        with self.assertRaises(ValueError):
            validate_dataset("x", self.plan, self.journal(), self.reader(swap))

    def test_rejects_wrong_cell_balance(self):
        bad = copy.deepcopy(self.plan)
        victim = next(e for e in bad["episodes"] if e["split"] == "train")
        victim["side"] = "B" if victim["side"] == "A" else "A"
        with self.assertRaises(ValueError):
            validate_dataset("x", bad, self.journal(), self.reader())

    def test_rejects_wrong_distractor_balance_and_edge_share(self):
        bad = copy.deepcopy(self.plan)
        victim = next(e for e in bad["episodes"] if e["split"] == "train" and e["edge"] == "interior")
        victim["edge"] = "inner"
        with self.assertRaises(ValueError):
            validate_dataset("x", bad, self.journal(), self.reader())
        bad = copy.deepcopy(self.plan)
        victim = next(e for e in bad["episodes"] if e["split"] == "train" and len(e["distractors"]) == 0)
        victim["distractors"] = ["blue"]
        with self.assertRaises(ValueError):
            validate_dataset("x", bad, self.journal(), self.reader())

    def test_rejects_a_held_out_pair(self):
        bad = copy.deepcopy(self.plan)
        victim = bad["episodes"][0]
        victim["pick"], victim["place"] = p.HELD_OUT_PAIRS[0]
        with self.assertRaises(ValueError):
            validate_dataset("x", bad, self.journal(), self.reader())

    def test_rejects_journal_that_skips_or_repeats_a_slot(self):
        journal = self.journal()
        journal[5]["slot"] = 4
        with self.assertRaises(ValueError):
            validate_dataset("x", self.plan, journal, self.reader())


if __name__ == "__main__":
    unittest.main()
