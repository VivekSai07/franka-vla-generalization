"""The lemon-only gate (spec note 2026-11-10): 96 fresh lemon attempts, pass at >= 90% first-attempt admission."""
import unittest
from collections import Counter

from experiments.gen_g1_data import lemon_gate as lg
from experiments.gen_g1_data import plan as p


def _rows(failed=()):
    return [{"index": i, "admitted": i not in failed, "expert_success": True, "reason": "expert_success" if i in failed else "ok",
             "expert_reason": "grasp" if i in failed else "success", "distractor_count": i % 6, "quintile": i % 5,
             "bin": "blue" if i % 2 else "orange", "region": "outer" if i % 3 == 0 else "p2", "distractor_motion": []}
            for i in range(lg.ATTEMPTS)]


class LemonAttemptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attempts = lg.make_attempts()
        cls.plan = p.make_plan()

    def test_96_lemon_attempts_with_the_gate_balance(self):
        a = self.attempts
        self.assertEqual(len(a), 96)
        self.assertEqual({x["object"] for x in a}, {"lemon"})
        self.assertEqual(Counter(x["distractor_count"] for x in a), {n: 16 for n in range(6)})
        self.assertEqual(Counter(tuple(x["cell"]) for x in a), {c: 24 for c in lg.LEMON_CELLS})
        self.assertEqual(sorted(Counter(x["quintile"] for x in a).values()), [19, 19, 19, 19, 20])
        self.assertEqual(sum(x["region"] == "outer" for x in a), 29)
        self.assertEqual([x["episode_index"] for x in a], list(range(96)))

    def test_every_attempt_is_a_train_pool_condition(self):
        for x in self.attempts:
            self.assertEqual(p.condition_of(x).pool, "train")

    def test_disjoint_from_every_slot_and_every_gate_attempt(self):
        earlier = [*self.plan["episodes"], *self.plan["gate_attempts"]]
        mine = self.attempts
        for key in ("layout_id", "identity", "seed", "condition_sha256"):
            self.assertFalse({x[key] for x in mine} & {x[key] for x in earlier}, key)
            self.assertEqual(len({x[key] for x in mine}), 96, key)
        self.assertFalse({x["seed"] for x in mine} & set(self.plan["rng_seeds"].values()))

    def test_the_attempts_are_fixed_by_a_pinned_digest(self):
        self.assertEqual(lg.make_attempts(), self.attempts)
        self.assertEqual(p.digest(self.attempts), lg.SHA256)

    def test_the_plan_itself_is_unchanged(self):
        self.assertEqual(p.digest(self.plan), p.PLAN_SHA256)


class LemonVerdictTests(unittest.TestCase):
    def test_87_of_96_passes_and_86_fails(self):
        self.assertTrue(lg.verdict(_rows(set(range(9))))["passed"])
        fail = lg.verdict(_rows(set(range(10))))
        self.assertFalse(fail["passed"])
        self.assertEqual(fail["admitted"], 86)

    def test_the_bar_is_the_published_90_percent(self):
        self.assertEqual(lg.BAR, 0.90)

    def test_admission_decides_not_the_expert_success_alone(self):
        rows = _rows()
        for r in rows[:20]:
            r["admitted"] = False
            r["reason"] = "distractors"
        verdict = lg.verdict(rows)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["expert_success_overall"], 1.0)

    def test_strata_are_reported_but_never_barred(self):
        rows = _rows({i for i in range(96) if i % 6 == 0})      # every failure in one distractor count: 0% there
        verdict = lg.verdict(rows)
        self.assertEqual(verdict["by_distractor_count"]["0"], {"n": 16, "admitted": 0})
        self.assertEqual(verdict["admitted"], 80)
        self.assertFalse(verdict["passed"])
        spread = lg.verdict(_rows({0, 1, 2, 3, 4, 5, 6, 7, 8}))
        self.assertTrue(spread["passed"])
        self.assertEqual(set(spread["by_quintile"]), {"0", "1", "2", "3", "4"})
        self.assertEqual(set(spread["by_bin"]), {"blue", "orange"})
        self.assertEqual(set(spread["by_region"]), {"outer", "p2"})

    def test_failure_reasons_are_counted(self):
        verdict = lg.verdict(_rows({0, 1, 2}))
        self.assertEqual(verdict["expert_reasons"], {"grasp": 3})
        self.assertEqual(verdict["reasons"], {"expert_success": 3})

    def test_an_incomplete_set_of_rows_is_refused(self):
        with self.assertRaises(ValueError):
            lg.verdict(_rows()[:95])
        rows = _rows()
        rows[5]["index"] = 4
        with self.assertRaises(ValueError):
            lg.verdict(rows)


if __name__ == "__main__":
    unittest.main()
