"""G1's comparison against A2: the statistics by hand-built counts, every refusal, and the file it writes. Admission image, CPU."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.gen_g1_compare import compare as c

AXES = c.AXES


def stage(n, success, nominal=None, levels=("a", "b")):
    """Episodes of `n` layouts x (nominal + five axes). `success(axis, i)` -> bool; nominal defaults to all success."""
    nominal = nominal or (lambda i: True)
    out = []
    for i in range(n):
        obj = "milk" if i % 2 else "bread"
        out.append(dict(layout_id=f"L{i}", axis="nominal", level="nominal", object=obj, task_success=nominal(i)))
        for axis in AXES:
            out.append(dict(layout_id=f"L{i}", axis=axis, level=levels[i % len(levels)], object=obj, task_success=success(axis, i)))
    return out


def prepared(core=c.G0_CORE_SHA256, plan="p", feas="f", dropped=()):
    return dict(core=dict(sha256=core), plan_sha256=plan, feasibility_sha256=feas, dropped_trials=list(dropped),
                arm=dict(sha256="arm"), checkpoints={"126900": dict(model_sha256="m"), "101520": dict(model_sha256="n")})


def done(episodes, dropped=(), planned=None):
    return dict(status="complete", episodes=episodes, planned_trials=planned or len(episodes), missing_ordinals=[],
                dropped_trials=list(dropped))


class Analyse(unittest.TestCase):
    def test_holds_and_beats(self):
        new = stage(100, lambda a, i: not (a == "lighting" and i == 0))           # lighting: one failure of 100
        old = stage(100, lambda a, i: not (a == "lighting" and i < 40))           # lighting: 40 failures
        r = c.analyse(new, old)
        pooled = r["axes"]["lighting"]["pooled"]
        self.assertEqual((pooled["new_successes"], pooled["control_successes"]), (99, 60))
        self.assertEqual(pooled["discordant"], dict(new_only=39, control_only=0))
        self.assertTrue(pooled["new_held"])
        self.assertAlmostEqual(pooled["gain_over_control"]["estimate"], 0.39)
        self.assertLess(r["axes"]["lighting"]["holm_p"], 0.05)
        self.assertTrue(r["criterion_met"]["lighting"])

    def test_held_but_not_better_is_not_the_criterion(self):
        r = c.analyse(stage(100, lambda a, i: True), stage(100, lambda a, i: True))
        self.assertTrue(r["axes"]["background"]["pooled"]["new_held"])
        self.assertFalse(r["axes"]["background"]["beats_control"])
        self.assertFalse(any(r["criterion_met"].values()))

    def test_better_but_not_held_is_not_the_criterion(self):
        new = stage(100, lambda a, i: not (a == "spatial" and i < 30))             # spatial drop 30 points
        old = stage(100, lambda a, i: not (a == "spatial" and i < 80))
        r = c.analyse(new, old)
        self.assertFalse(r["axes"]["spatial"]["pooled"]["new_held"])
        self.assertTrue(r["axes"]["spatial"]["beats_control"])
        self.assertFalse(r["criterion_met"]["spatial"])

    def test_a_worse_arm_never_beats_even_at_small_p(self):
        new = stage(100, lambda a, i: not (a == "distractors" and i < 40))
        old = stage(100, lambda a, i: True)
        r = c.analyse(new, old)
        self.assertLess(r["axes"]["distractors"]["pooled"]["mcnemar_exact_p"], 1e-6)
        self.assertFalse(r["axes"]["distractors"]["beats_control"])

    def test_holm_over_the_five_axes_is_applied(self):
        # seven layouts better on one axis only: raw p = 2/128 = 0.0156 < 0.05 but Holm over five is 0.078
        new = stage(100, lambda a, i: True)
        old = stage(100, lambda a, i: not (a == "lighting" and i < 7))
        r = c.analyse(new, old)
        self.assertAlmostEqual(r["axes"]["lighting"]["pooled"]["mcnemar_exact_p"], 2 / 128)
        self.assertAlmostEqual(r["axes"]["lighting"]["holm_p"], 5 * 2 / 128)
        self.assertFalse(r["axes"]["lighting"]["beats_control"])

    def test_levels_are_reported_per_level(self):
        new = stage(100, lambda a, i: True)
        old = stage(100, lambda a, i: not (a == "lighting" and i % 2 == 0))
        levels = c.analyse(new, old)["axes"]["lighting"]["levels"]
        self.assertEqual(sorted(levels), ["a", "b"])
        self.assertEqual((levels["a"]["layouts"], levels["a"]["control_successes"]), (50, 0))
        self.assertEqual((levels["b"]["layouts"], levels["b"]["control_successes"]), (50, 50))

    def test_nominal_cost_flag(self):
        old = stage(100, lambda a, i: True, nominal=lambda i: i < 90)
        low = c.analyse(stage(100, lambda a, i: True, nominal=lambda i: i < 80), old)["nominal"]
        near = c.analyse(stage(100, lambda a, i: True, nominal=lambda i: i < 87), old)["nominal"]
        self.assertTrue(low["cost"])
        self.assertFalse(near["cost"])
        self.assertEqual((low["new_successes"], low["control_successes"]), (80, 90))
        self.assertAlmostEqual(low["gain_over_control"]["estimate"], -0.10)

    def test_a_layout_without_a_nominal_trial_in_both_arms_is_left_out_of_that_pair(self):
        new = stage(10, lambda a, i: True)
        old = [e for e in stage(10, lambda a, i: True) if not (e["layout_id"] == "L3" and e["axis"] == "nominal")]
        with self.assertRaises(ValueError):
            c.analyse(new, old)                                                    # different trial sets are refused outright
        both = [e for e in stage(10, lambda a, i: True) if not (e["layout_id"] == "L3" and e["axis"] == "nominal")]
        self.assertEqual(c.analyse(both, both)["axes"]["lighting"]["pooled"]["layouts"], 9)

    def test_duplicates_are_refused(self):
        s = stage(5, lambda a, i: True)
        with self.assertRaises(ValueError):
            c.analyse(s + s[:1], s)


class CheckPair(unittest.TestCase):
    def args(self, **over):
        s = done(stage(2, lambda a, i: True))
        base = dict(new=prepared(), old=prepared(), new_stage=s, old_stage=s, checkpoint=126900, winner=126900)
        base.update(over)
        return base

    def test_roles(self):
        self.assertEqual(c.check_pair(**self.args()), "primary")
        self.assertEqual(c.check_pair(**self.args(checkpoint=101520)), "matched-phase secondary")
        self.assertEqual(c.check_pair(**self.args(checkpoint=101520, winner=101520)), "primary and matched-phase")

    def test_a_checkpoint_the_selection_did_not_allow_is_refused(self):
        with self.assertRaises(ValueError):
            c.check_pair(**self.args(checkpoint=114210))

    def test_a_different_core_is_refused_for_either_arm(self):
        with self.assertRaises(ValueError):
            c.check_pair(**self.args(new=prepared(core="0" * 64)))
        with self.assertRaises(ValueError):
            c.check_pair(**self.args(old=prepared(core="0" * 64)))

    def test_different_plan_feasibility_or_drops_are_refused(self):
        for over in (dict(plan="x"), dict(feas="x"), dict(dropped=["t"])):
            with self.subTest(over=over), self.assertRaises(ValueError):
                c.check_pair(**self.args(new=prepared(**over)))

    def test_incomplete_or_different_stages_are_refused(self):
        s = stage(2, lambda a, i: True)
        for new_stage in (done(s[:-1], planned=len(s)), done(s, planned=len(s) + 1), done(s, dropped=["t"]),
                          dict(done(s), status="incomplete"), dict(done(s), missing_ordinals=[3])):
            with self.subTest(), self.assertRaises(ValueError):
                c.check_pair(**self.args(new_stage=new_stage, old_stage=done(s)))


class AnalyseM0(unittest.TestCase):
    def by(self, n, ok):
        return {f"L{i}": dict(object="milk" if i % 2 else "bread", task_success=ok(i)) for i in range(n)}

    def test_counts_effect_and_cost(self):
        r = c.analyse_m0(self.by(100, lambda i: i < 80), self.by(100, lambda i: i < 90))
        self.assertEqual((r["new_successes"], r["control_successes"]), (80, 90))
        self.assertEqual(r["discordant"], dict(new_only=0, control_only=10))
        self.assertAlmostEqual(r["effect"]["estimate"], -0.10)
        self.assertTrue(r["cost"])
        self.assertFalse(c.analyse_m0(self.by(100, lambda i: i < 88), self.by(100, lambda i: i < 90))["cost"])

    def test_different_layout_sets_are_refused(self):
        with self.assertRaises(ValueError):
            c.analyse_m0(self.by(10, lambda i: True), self.by(9, lambda i: True))


class CompareWritesItsFile(unittest.TestCase):
    def test_writes_once_and_verifies_both_arms(self):
        new_eps, old_eps = stage(20, lambda a, i: True), stage(20, lambda a, i: i > 5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {"arms/A1/prepared.json": prepared(), "arms/A2/prepared.json": prepared(),
                     "arms/A1/selection.json": dict(winner=126900),
                     "arms/A1/benchmark/step_126900/result.json": done(new_eps),
                     "arms/A2/benchmark/step_088830/result.json": done(old_eps)}
            for rel, value in files.items():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                c.g0.write_json(root / rel, value)
            with mock.patch.object(c.g0, "verify_core") as core, mock.patch.object(c.g0, "verify_manifests") as manifests, \
                    mock.patch.object(c.g0, "verify_textures") as textures, mock.patch.object(c.g0, "verify_arm") as arm:
                result = c.compare(root, 126900, repo=root)
            self.assertEqual((core.call_count, manifests.call_count, textures.call_count, arm.call_count), (2, 2, 2, 2))
            self.assertTrue(all(call.kwargs.get("deep") for call in arm.call_args_list))
            self.assertEqual(result["role"], "primary")
            self.assertEqual(result["model_sha256"], "m")
            written = c.g0.read_json(root / "arms/A1/compare_A2_step_126900.json")
            self.assertEqual(written["nominal"]["new_successes"], 20)
            with mock.patch.object(c.g0, "verify_core"), mock.patch.object(c.g0, "verify_manifests"), \
                    mock.patch.object(c.g0, "verify_textures"), mock.patch.object(c.g0, "verify_arm"):
                with self.assertRaises(FileExistsError):
                    c.compare(root, 126900, repo=root)


class CompareM0WritesItsFile(unittest.TestCase):
    def write(self, root, files):
        for rel, value in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            c.g0.write_json(root / rel, value)

    def episodes(self, ok):
        return [dict(layout_id=f"L{i}", object="milk" if i % 2 else "bread", task_success=ok(i)) for i in range(20)]

    def test_refuses_anything_but_the_g0_winner_then_writes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            g0_root, m0_root = Path(tmp) / "g0", Path(tmp) / "m0"
            self.write(g0_root, {"arms/A1/selection.json": dict(winner=126900)})
            stage_new = dict(done(self.episodes(lambda i: True)), dropped_layouts=[])
            stage_old = dict(done(self.episodes(lambda i: i > 3)), dropped_layouts=[])
            self.write(m0_root, {"arms/A1/prepared.json": prepared(core=c.M0_CORE_SHA256),
                                 "arms/A2/prepared.json": prepared(core=c.M0_CORE_SHA256),
                                 "arms/A1/comparison/step_126900/result.json": stage_new,
                                 "arms/A2/comparison/step_088830/result.json": stage_old})
            with self.assertRaises(ValueError):
                c.compare_m0(m0_root, g0_root, 101520, repo=m0_root)
            with mock.patch.object(c.m0, "verify_core") as core, mock.patch.object(c.m0, "verify_manifests"), \
                    mock.patch.object(c.m0, "verify_arm") as arm:
                result = c.compare_m0(m0_root, g0_root, 126900, repo=m0_root)
            self.assertEqual((core.call_count, arm.call_count), (2, 2))
            self.assertEqual((result["new_successes"], result["control_successes"]), (20, 16))
            self.assertTrue((m0_root / "arms/A1/compare_A2_step_126900_vs_088830.json").is_file())

    def test_a_core_other_than_the_pinned_one_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            g0_root, m0_root = Path(tmp) / "g0", Path(tmp) / "m0"
            self.write(g0_root, {"arms/A1/selection.json": dict(winner=126900)})
            self.write(m0_root, {"arms/A1/prepared.json": prepared(core="0" * 64),
                                 "arms/A2/prepared.json": prepared(core=c.M0_CORE_SHA256)})
            with mock.patch.object(c.m0, "verify_core"), mock.patch.object(c.m0, "verify_manifests"), mock.patch.object(c.m0, "verify_arm"):
                with self.assertRaises(ValueError):
                    c.compare_m0(m0_root, g0_root, 126900, repo=m0_root)


if __name__ == "__main__":
    unittest.main()
