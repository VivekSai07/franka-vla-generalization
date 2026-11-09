import hashlib
import importlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.gen_eval import contract, evaluate, feasibility, plan as plan_module, protocol, stats
from experiments.gen_scene import levels

REPO = contract.ROOT
FAKE_ADAPTER = '''import numpy as np


class FakeAdapter:
    name = "fake"

    def __init__(self, checkpoint, scale=1.0):
        self.checkpoint = checkpoint

    def reset(self):
        pass

    def act(self, obs):
        return np.zeros(8)
'''
_TEMPLATE = {}


def template():
    """A campaign template built once: plan dir, a feasibility.json with every trial feasible, a repo copy with the textures."""
    if not _TEMPLATE:
        root = Path(tempfile.mkdtemp(prefix="g0_template_"))
        plan_module.write(root / "campaign/plan")
        shard = lambda layout, job: dict(feasible=True, reason="success", frames=452, stable_ticks=30, seconds=0.0)
        feasibility.run_shard(root / "campaign/feasibility", 0, 1, root / "campaign/plan", runner=shard)
        feasibility.merge(root / "campaign/feasibility", root / "campaign/plan")
        for rel in contract.core_provenance():
            target = root / "repo" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / rel, target)
        shutil.copytree(REPO / ".cache/gen_textures", root / "repo/.cache/gen_textures")
        _TEMPLATE["root"] = root
    return _TEMPLATE["root"]


def tearDownModule():
    if _TEMPLATE:
        shutil.rmtree(_TEMPLATE["root"], ignore_errors=True)


def make_checkpoint(directory, name, content=b"weights"):
    directory = Path(directory) / name
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(content)
    (directory / "config.json").write_text("{}")
    (directory / "checkpoint_result.json").write_text(json.dumps(dict(status="complete", model_sha256=hashlib.sha256(content).hexdigest())))
    return directory


def position(trial):
    return (trial["ordinal"] - 1) // len(protocol.AXES)


def stub_runner(rule, lift=lambda t, ok: ok, log=None):
    """A rollout stand-in: success by `rule(trial)`, grasp-lift by `lift(trial, success)`; a disturbance trial always fires."""
    def run(adapter, state_builder, layout, trial, folder):
        success = bool(rule(trial))
        fired = trial["axis"] == "disturbance"
        (folder / "cameras.mp4").write_bytes(b"x")
        (folder / "trace.npz").write_bytes(b"x")
        row = dict(trial, outcome="success" if success else "timeout", task_success=success, ticks=1, placed_bin=None,
                   final_gripper_open=True, cause="success" if success else "never_lifted", first_lifted=None,
                   wrong_object=False, distractor_displacement_m={}, names=[trial["object"]], wall_seconds=0.0,
                   max_lift_m=0.0, grasp_lift=bool(lift(trial, success)), disturbance_fired=fired,
                   disturbance_tick=10 if fired else None, recovered=bool(success and fired))
        (folder / "episode.json").write_text(json.dumps(row))
        if log is not None:
            log.append(trial["ordinal"])
        return row
    return run


class Campaign(unittest.TestCase):
    """A fresh campaign root and repo copy per test, with a fake arm package registered under the real namespace."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="g0_test_"))
        base = template()
        self.root, self.repo = self.tmp / "campaign", self.tmp / "repo"
        shutil.copytree(base / "campaign", self.root)
        shutil.copytree(base / "repo", self.repo)
        sys.path.append(str(self.repo))
        importlib.invalidate_caches()
        self.addCleanup(self.cleanup)
        self.add_arm("vla_inputs_arm_fake")
        self.ckpt = make_checkpoint(self.tmp, "ckpt_a", b"a")

    def cleanup(self):
        for name in [m for m in sys.modules if m.startswith("experiments.vla_inputs_arm_fake")]:
            del sys.modules[name]
        if str(self.repo) in sys.path:
            sys.path.remove(str(self.repo))
        import experiments
        for name in ("vla_inputs_arm_fake", "vla_inputs_arm_fake2"):
            if hasattr(experiments, name):
                delattr(experiments, name)
        importlib.invalidate_caches()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def add_arm(self, name, source=FAKE_ADAPTER):
        package = self.repo / "experiments" / name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("")
        (package / "adapter.py").write_text(source)
        (package / "README.md").write_text("# fake arm\n")

    def prepare(self, arm="A2", name="vla_inputs_arm_fake", checkpoints=None):
        return evaluate.prepare(self.root, arm, f"experiments.{name}.adapter:FakeAdapter", checkpoints or {1000: self.ckpt}, repo=self.repo)

    def run_worker(self, arm="A2", tier="smoke", checkpoint=1000, shard=0, shards=1, rule=lambda t: True, log=None, **kwargs):
        return evaluate.worker(self.root, arm, tier, checkpoint, shard, shards, repo=self.repo,
                               factory=kwargs.pop("factory", lambda prepared, c: _fake_instance(c)),
                               runner=kwargs.pop("runner", None) or stub_runner(rule, log=log), cooler=lambda: 0.0, **kwargs)

    def stage(self, arm, tier, checkpoint=1000, rule=lambda t: True, **kwargs):
        self.run_worker(arm, tier, checkpoint, rule=rule, **kwargs)
        return evaluate.collect(self.root, arm, tier, checkpoint)


def _fake_instance(checkpoint):
    module = sys.modules[next(m for m in sys.modules if m.startswith("experiments.vla_inputs_arm_fake") and m.endswith(".adapter"))]
    return module.FakeAdapter(checkpoint)


class Prepare(Campaign):
    def test_records_core_textures_arm_manifests_and_checkpoint_facts(self):
        record = self.prepare()
        self.assertEqual(record["core"]["sha256"], contract.core_sha256(self.repo))
        self.assertEqual(sorted(record["checkpoints"]), ["1000"])
        self.assertEqual(record["checkpoints"]["1000"]["model_sha256"], hashlib.sha256(b"a").hexdigest())
        self.assertEqual(len(record["textures"]), 120)
        self.assertEqual(record["dropped_trials"], {})
        self.assertEqual(record["feasibility_counts"]["benchmark"]["nominal"]["nominal"], {"planned": 144, "feasible": 144})
        self.assertTrue(any(f.endswith("adapter.py") for f in record["arm"]["files"]))
        self.assertEqual((self.root / "arms/A2/prepared.json").exists(), True)

    def test_the_core_closure_includes_the_engine_the_manifest_the_signal_source_and_the_p2_analysis(self):
        bound = set(contract.core_provenance(self.repo))
        for rel in ("experiments/gen_scene/compose.py", "experiments/gen_scene/levels.py", "experiments/gen_scene/expert.py",
                    "experiments/gen_scene/assets_manifest.json", "experiments/gen_eval/rollout.py", "experiments/gen_eval/evaluate.py",
                    "experiments/vla_inputs_signals/signals.py", "experiments/xvla_props_p2_eval/analysis.py",
                    "experiments/vla_inputs_eval/adapters.py", "validation/rollout_diagnostics.py"):
            self.assertIn(rel, bound)
        self.assertFalse([r for r in bound if r.startswith(("experiments/gen_eval/tests/", "experiments/gen_scene/tests/"))])

    def test_no_overwrite(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_refuses_weights_that_do_not_match_their_marker(self):
        (self.ckpt / "model.safetensors").write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_refuses_an_incomplete_marker(self):
        (self.ckpt / "checkpoint_result.json").write_text(json.dumps(dict(status="running", model_sha256="x")))
        with self.assertRaises(ValueError):
            self.prepare()

    def test_refuses_a_non_arm_module(self):
        with self.assertRaises(ValueError):
            evaluate.prepare(self.root, "A2", "experiments.gen_eval.protocol:trial_of", {1000: self.ckpt}, repo=self.repo)

    def test_refuses_an_arm_name_outside_a0_to_a9(self):
        with self.assertRaises(ValueError):
            evaluate.prepare(self.root, "arm2", "experiments.vla_inputs_arm_fake.adapter:FakeAdapter", {1000: self.ckpt}, repo=self.repo)

    def test_every_arm_of_a_campaign_shares_one_core(self):
        self.prepare("A2")
        self.add_arm("vla_inputs_arm_fake2")
        (self.repo / "experiments/gen_eval/stats.py").write_text((self.repo / "experiments/gen_eval/stats.py").read_text() + "\n# edited\n")
        with self.assertRaises(ValueError) as caught:
            self.prepare("A0", name="vla_inputs_arm_fake2")
        self.assertIn("shared core differs from A2", str(caught.exception))

    def test_a_different_declared_adapter_with_an_identical_core_is_accepted(self):
        self.prepare("A2")
        self.add_arm("vla_inputs_arm_fake2")
        record = self.prepare("A0", name="vla_inputs_arm_fake2")
        self.assertEqual(record["core"]["sha256"], evaluate.read_json(self.root / "arms/A2/prepared.json")["core"]["sha256"])
        self.assertNotEqual(record["arm"]["sha256"], evaluate.read_json(self.root / "arms/A2/prepared.json")["arm"]["sha256"])

    def test_a_changed_texture_file_is_refused(self):
        victim = next((self.repo / ".cache/gen_textures").iterdir())
        victim.write_bytes(victim.read_bytes()[:-1] + b"\x00")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_a_missing_texture_file_is_refused(self):
        next((self.repo / ".cache/gen_textures").iterdir()).unlink()
        with self.assertRaises(ValueError):
            self.prepare()

    def test_the_signal_source_must_be_the_sealed_one(self):
        (self.repo / "experiments/vla_inputs_signals/signals.py").write_text("# edited\n")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_a_frozen_p2_file_edit_is_refused(self):
        target = self.repo / "experiments/props_scene/expert.py"
        target.write_text(target.read_text() + "\n# edited\n")
        with self.assertRaises(ValueError):
            self.prepare()


class Worker(Campaign):
    def setUp(self):
        super().setUp()
        self.prepare("A2")

    def test_runs_every_trial_of_its_share_and_writes_telemetry(self):
        rows = self.run_worker(tier="smoke")
        self.assertEqual(len(rows), 12)
        directory = protocol.stage_dir(self.root, "A2", "smoke", 1000)
        for trial in protocol.stage_trials(*evaluate.load_manifests(self.root), "smoke"):
            folder = protocol.trial_folder(directory, trial)
            self.assertTrue(protocol.trial_is_complete(folder))
            self.assertIn("rss_max_mib", json.loads((folder / "telemetry.json").read_text()))

    def test_shards_partition_the_stage_and_spread_the_axes(self):
        log = []
        for shard in range(6):
            self.run_worker(tier="selection", shard=shard, shards=6, log=log)
        self.assertEqual(sorted(log), list(range(1, 24 * 6 + 1)))
        self.assertEqual(evaluate.collect(self.root, "A2", "selection", 1000)["status"], "complete")

    def test_a_resumed_worker_skips_complete_trials_and_redoes_incomplete_ones(self):
        log = []
        self.run_worker(tier="smoke", log=log)
        directory = protocol.stage_dir(self.root, "A2", "smoke", 1000)
        trials = protocol.stage_trials(*evaluate.load_manifests(self.root), "smoke")
        (protocol.trial_folder(directory, trials[2]) / "episode.json").unlink()
        log.clear()
        self.run_worker(tier="smoke", log=log)
        self.assertEqual(log, [trials[2]["ordinal"]])
        self.assertTrue((directory / "quarantine").is_dir())

    def test_a_finished_stage_builds_no_adapter_at_all(self):
        self.run_worker(tier="smoke")
        built = []
        self.run_worker(tier="smoke", factory=lambda p, c: built.append(c))
        self.assertEqual(built, [])

    def test_one_worker_builds_exactly_one_adapter_for_exactly_one_checkpoint(self):
        built = []
        self.run_worker(tier="smoke", factory=lambda p, c: built.append(c) or _fake_instance(c))
        self.assertEqual(built, [1000])
        with self.assertRaises(ValueError):
            evaluate.worker(self.root, "A2", "smoke", "1000", 0, 1, repo=self.repo)

    def test_an_unprepared_checkpoint_is_refused(self):
        with self.assertRaises(ValueError):
            self.run_worker(checkpoint=2000)

    def test_a_changed_checkpoint_is_refused(self):
        (self.ckpt / "model.safetensors").write_bytes(b"changed!")
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_a_core_edit_after_prepare_stops_the_worker(self):
        target = self.repo / "experiments/gen_scene/levels.py"
        target.write_text(target.read_text() + "\n# edited\n")
        with self.assertRaises(ValueError) as caught:
            self.run_worker()
        self.assertIn("core changed", str(caught.exception))

    def test_an_arm_edit_after_prepare_rejects_that_arm(self):
        (self.repo / "experiments/vla_inputs_arm_fake/adapter.py").write_text(FAKE_ADAPTER + "\n# edited\n")
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_a_manifest_edit_after_prepare_stops_the_worker(self):
        path = self.root / "feasibility/feasibility.json"
        value = json.loads(path.read_text())
        value["feasible_trials"] = value["feasible_trials"][1:]
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_a_texture_edit_after_prepare_stops_the_worker(self):
        victim = sorted((self.repo / ".cache/gen_textures").iterdir())[5]
        victim.write_bytes(victim.read_bytes()[:-1] + b"\x01")
        with self.assertRaises(ValueError) as caught:
            self.run_worker()
        self.assertIn("texture", str(caught.exception))

    def test_an_infrastructure_error_leaves_a_failure_file_named_by_shard_and_is_not_swallowed(self):
        def boom(adapter, state_builder, layout, trial, folder):
            raise RuntimeError("renderer died")
        with self.assertRaises(RuntimeError):
            self.run_worker(shard=1, shards=3, runner=boom)
        failure = protocol.stage_dir(self.root, "A2", "smoke", 1000) / "failure_shard1.json"
        self.assertEqual(json.loads(failure.read_text())["error"], "renderer died")

    def test_no_file_is_named_by_a_process_id(self):
        self.run_worker(tier="smoke")
        names = [p.name for p in self.tmp.rglob("*") if p.is_file()]
        self.assertFalse([n for n in names if ".tmp" in n])           # atomic writes leave no temporary behind
        self.assertEqual(evaluate.write_json.__code__.co_names.count("getpid"), 0)

    def test_the_eval_line_names_axis_and_level(self):
        import io
        from contextlib import redirect_stdout
        out = io.StringIO()
        with redirect_stdout(out):
            self.run_worker(tier="smoke")
        line = next(l for l in out.getvalue().splitlines() if l.startswith("EVAL") and '"lighting"' in l)
        self.assertIn('"level"', line)


class ArmInputs(Campaign):
    def setUp(self):
        super().setUp()
        self.inputs = self.repo / "bundle"
        (self.inputs / "tok").mkdir(parents=True)
        (self.inputs / "tok/vocab.json").write_text("{}")
        (self.inputs / "tok/merges.txt").write_text("a b")
        self.add_arm("vla_inputs_arm_fake", FAKE_ADAPTER + '\nINPUT_DIRS = ("bundle",)\n')

    def test_every_file_of_the_input_directory_is_recorded_with_hash_size_and_mtime(self):
        record = self.prepare()
        files = record["arm"]["inputs"]["bundle"]
        self.assertEqual(sorted(files), ["tok/merges.txt", "tok/vocab.json"])
        self.assertEqual(files["tok/vocab.json"]["sha256"], hashlib.sha256(b"{}").hexdigest())

    def test_a_changed_input_file_stops_the_worker(self):
        self.prepare()
        (self.inputs / "tok/vocab.json").write_text('{"a": 1}')
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_a_missing_or_empty_input_directory_is_refused(self):
        shutil.rmtree(self.inputs)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_a_symlink_in_an_input_directory_is_refused(self):
        (self.inputs / "link").symlink_to(self.inputs / "tok/vocab.json")
        with self.assertRaises(ValueError):
            self.prepare()


class Collect(Campaign):
    def setUp(self):
        super().setUp()
        self.prepare("A2")

    def test_complete_only_if_every_planned_trial_exists(self):
        self.run_worker(tier="smoke", shard=0, shards=2)
        report = evaluate.collect(self.root, "A2", "smoke", 1000)
        self.assertEqual((report["status"], report["planned_trials"], len(report["episodes"])), ("incomplete", 12, 6))
        self.run_worker(tier="smoke", shard=1, shards=2)
        report = evaluate.collect(self.root, "A2", "smoke", 1000)
        self.assertEqual((report["status"], report["missing_ordinals"]), ("complete", []))
        self.assertEqual(sorted(report["by_axis"]), sorted(protocol.AXES))

    def test_a_trial_that_does_not_match_the_plan_is_refused(self):
        self.run_worker(tier="smoke")
        directory = protocol.stage_dir(self.root, "A2", "smoke", 1000)
        trial = protocol.stage_trials(*evaluate.load_manifests(self.root), "smoke")[1]
        path = protocol.trial_folder(directory, trial) / "episode.json"
        row = json.loads(path.read_text())
        row["condition"]["table_offset_m"] = 0.5
        path.write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            evaluate.collect(self.root, "A2", "smoke", 1000)

    def test_dropped_trials_leave_the_stage_and_are_recorded(self):
        plan, feas = evaluate.load_manifests(self.root)
        lid = plan["tiers"]["smoke"][0]
        key = protocol.trial_key(lid, "spatial")
        feas["feasible_trials"] = [k for k in feas["feasible_trials"] if k != key]
        feas["dropped_trials"] = {key: "spatial infeasible"}
        evaluate.write_json(self.root / "feasibility/feasibility.json", feas)
        self.fresh_prepare()
        self.run_worker(arm="A5", tier="smoke")
        report = evaluate.collect(self.root, "A5", "smoke", 1000)
        self.assertEqual((report["status"], report["planned_trials"], report["dropped_trials"]), ("complete", 11, {key: "spatial infeasible"}))
        self.assertNotIn(5, [e["ordinal"] for e in report["episodes"]])

    def fresh_prepare(self):
        (self.root / "arms/A2/prepared.json").unlink()
        self.prepare("A5")


class Select(Campaign):
    def setUp(self):
        super().setUp()
        self.ckpts = {1000: self.ckpt, 2000: make_checkpoint(self.tmp, "ckpt_b", b"b"), 3000: make_checkpoint(self.tmp, "ckpt_c", b"c")}
        self.prepare("A2", checkpoints=self.ckpts)

    def stage_all(self, rules):
        for checkpoint, rule in rules.items():
            self.stage("A2", "selection", checkpoint, rule=rule)

    def test_picks_the_most_successful_checkpoint_over_every_trial_of_the_selection_tier(self):
        self.stage_all({1000: lambda t: position(t) < 5, 2000: lambda t: position(t) < 9, 3000: lambda t: position(t) < 2})
        decision = evaluate.select(self.root, "A2")
        self.assertEqual((decision["winner"], decision["counts"]), (2000, {"1000": 30, "2000": 54, "3000": 12}))
        self.assertEqual(json.loads((self.root / "arms/A2/selection.json").read_text()), decision)

    def test_a_tie_goes_to_the_later_checkpoint_and_is_recorded(self):
        self.stage_all({1000: lambda t: position(t) < 5, 2000: lambda t: position(t) < 5, 3000: lambda t: position(t) < 1})
        decision = evaluate.select(self.root, "A2")
        self.assertEqual((decision["winner"], decision["tied"]), (2000, [1000, 2000]))
        self.assertIn("administrative", decision["rule"])

    def test_reads_only_the_selection_tier(self):
        self.stage_all({1000: lambda t: True, 2000: lambda t: False, 3000: lambda t: False})
        self.stage("A2", "smoke", 2000, rule=lambda t: True)       # a better smoke result must not matter
        self.assertEqual(evaluate.select(self.root, "A2")["winner"], 1000)

    def test_an_incomplete_candidate_blocks_selection(self):
        self.stage("A2", "selection", 1000)
        self.stage("A2", "selection", 2000)
        self.run_worker(tier="selection", checkpoint=3000, shard=0, shards=2)
        evaluate.collect(self.root, "A2", "selection", 3000)
        with self.assertRaises(RuntimeError):
            evaluate.select(self.root, "A2")

    def test_the_decision_is_never_overwritten(self):
        self.stage_all({1000: lambda t: True, 2000: lambda t: False, 3000: lambda t: False})
        evaluate.select(self.root, "A2")
        with self.assertRaises(FileExistsError):
            evaluate.select(self.root, "A2")

    def test_the_benchmark_tier_never_chooses(self):
        self.stage_all({1000: lambda t: False, 2000: lambda t: False, 3000: lambda t: False})
        self.stage("A2", "benchmark", 3000, rule=lambda t: True)
        self.assertEqual(evaluate.select(self.root, "A2")["winner"], 3000)       # all tied at 0 -> later; the benchmark result is not read
        self.assertEqual(evaluate.read_json(self.root / "arms/A2/selection.json")["counts"], {"1000": 0, "2000": 0, "3000": 0})


class Compare(unittest.TestCase):
    """A synthetic benchmark whose paired outcomes are known exactly (position p = layout number in the tier); built once."""

    @staticmethod
    def rule(trial):
        p, axis, level = position(trial), trial["axis"], trial["level"]
        nominal = p % 10 != 0                                  # 15 of 144 layouts fail at nominal
        if axis == "nominal":
            return nominal
        if axis == "lighting":
            return nominal and level != "dark"                 # dark loses every nominal success
        if axis == "background":
            return nominal                                     # no change
        if axis == "distractors":
            return nominal or level == "five"                  # five gains the nominal failures
        if axis == "spatial":
            return nominal and level != "table_up10"
        return level == "move" and p % 4 == 1                  # move recovers half; drop never

    @staticmethod
    def lift(trial, success):
        return success or position(trial) % 3 != 0

    @classmethod
    def setUpClass(cls):
        cls.env = Campaign("run_worker")
        cls.env.setUp()
        cls.env.prepare("A2")
        cls.env.run_worker(arm="A2", tier="benchmark", runner=stub_runner(cls.rule, lift=cls.lift))
        evaluate.collect(cls.env.root, "A2", "benchmark", 1000)
        cls.result = evaluate.compare(cls.env.root, "A2", 1000, repo=cls.env.repo)

    @classmethod
    def tearDownClass(cls):
        cls.env.doCleanups()

    def test_nominal(self):
        n = self.result["nominal"]
        self.assertEqual((n["trials"], n["successes"]), (144, 129))
        self.assertAlmostEqual(n["rate"], 129 / 144)
        self.assertTrue(n["interval"][0] < n["rate"] < n["interval"][1])
        self.assertEqual(self.result["layouts_planned"], 144)
        self.assertEqual(self.result["layouts_with_nominal"], 144)

    def test_lighting_dark_loses_what_nominal_had_and_the_axis_is_not_held(self):
        lighting = self.result["axes"]["lighting"]
        pooled, dark = lighting["pooled"], lighting["levels"]["dark"]
        self.assertEqual((pooled["layouts"], pooled["nominal_successes"], pooled["perturbed_successes"]), (144, 129, 101))
        self.assertEqual(pooled["discordant"], dict(nominal_only=28, perturbed_only=0))
        self.assertAlmostEqual(pooled["drop"]["estimate"], 28 / 144)
        self.assertFalse(pooled["held"])
        self.assertAlmostEqual(pooled["mcnemar_exact_p"], stats.mcnemar(0, 28))
        self.assertEqual((dark["layouts"], dark["nominal_successes"], dark["perturbed_successes"]), (36, 28, 0))
        self.assertAlmostEqual(dark["drop"]["estimate"], 28 / 36)
        for level in ("bright", "tinted", "low_angle"):
            self.assertEqual(lighting["levels"][level]["discordant"], dict(nominal_only=0, perturbed_only=0))
            self.assertTrue(lighting["levels"][level]["held"])

    def test_an_unchanged_axis_has_zero_drop_and_is_held(self):
        bg = self.result["axes"]["background"]["pooled"]
        self.assertEqual((bg["drop"]["estimate"], bg["drop"]["lo"], bg["drop"]["hi"], bg["held"]), (0.0, 0.0, 0.0, True))
        self.assertEqual(bg["mcnemar_exact_p"], 1.0)

    def test_a_gain_is_a_negative_drop(self):
        d = self.result["axes"]["distractors"]["pooled"]
        self.assertEqual(d["discordant"], dict(nominal_only=0, perturbed_only=5))
        self.assertAlmostEqual(d["drop"]["estimate"], -5 / 144)
        self.assertTrue(d["held"])
        self.assertEqual(self.result["axes"]["distractors"]["levels"]["five"]["discordant"]["perturbed_only"], 5)

    def test_spatial_up10_collapses_and_the_pooled_axis_reads_it(self):
        s = self.result["axes"]["spatial"]
        self.assertEqual(s["levels"]["table_up10"]["discordant"], dict(nominal_only=36, perturbed_only=0))
        self.assertAlmostEqual(s["pooled"]["drop"]["estimate"], 36 / 144)
        self.assertFalse(s["pooled"]["held"])
        self.assertEqual({lv: s["levels"][lv]["layouts"] for lv in s["levels"]}, {"table_up5": 36, "table_down5": 36, "table_up10": 36, "outer_start": 36})

    def test_disturbance_counts_fired_and_recovered(self):
        d = self.result["axes"]["disturbance"]
        self.assertEqual(d["pooled"]["layouts"], 144)
        self.assertEqual(d["levels"]["move"]["disturbance"], dict(fired=72, recovered=36, not_fired=0))
        self.assertEqual(d["levels"]["drop"]["disturbance"], dict(fired=72, recovered=0, not_fired=0))
        self.assertEqual(d["levels"]["move"]["perturbed_successes"], 36)
        self.assertNotIn("disturbance", self.result["axes"]["lighting"]["pooled"])

    def test_grasp_lift_is_paired_the_same_way(self):
        lift = self.result["axes"]["lighting"]["pooled"]["grasp_lift"]
        nominal_lift = lambda p: (p % 10 != 0) or p % 3 != 0     # success or (p % 3 != 0), success being p % 10 != 0
        dark_lift = lambda p: p % 3 != 0                         # a dark trial never succeeds
        self.assertEqual(lift["nominal"], sum(nominal_lift(p) for p in range(144)))
        self.assertEqual(lift["perturbed"], sum(dark_lift(p) if p % 4 == 0 else nominal_lift(p) for p in range(144)))
        self.assertEqual((lift["nominal"], lift["perturbed"]), (139, 130))
        self.assertAlmostEqual(lift["drop"]["estimate"], 9 / 144)

    def test_causes_are_counted_per_cell(self):
        dark = self.result["axes"]["lighting"]["levels"]["dark"]
        self.assertEqual(dark["causes"], {"never_lifted": 36})
        self.assertEqual(dark["nominal_causes"], {"never_lifted": 8, "success": 28})

    def test_holm_over_the_five_axes_is_reported_and_no_family_claim_is_made(self):
        holm = self.result["holm_over_axes"]
        self.assertEqual(sorted(holm), sorted(levels.AXES))
        raw = {a: self.result["axes"][a]["pooled"]["mcnemar_exact_p"] for a in levels.AXES}
        self.assertEqual(holm, stats.holm(raw))
        self.assertIn("no family-wise claim", self.result["caveat"])

    def test_held_flags_follow_the_rule(self):
        self.assertEqual(self.result["held"], {"lighting": False, "background": True, "distractors": True, "spatial": False, "disturbance": False})

    def test_the_population_and_the_exclusions_are_stated(self):
        self.assertIn("never replaced", self.result["population"])
        self.assertEqual(self.result["planned_and_feasible"]["lighting"]["dark"], {"planned": 36, "feasible": 36})

    def test_a_result_is_never_overwritten_and_final_collects_it(self):
        env = self.env
        with self.assertRaises(FileExistsError):
            evaluate.compare(env.root, "A2", 1000, repo=env.repo)
        out = evaluate.final(env.root)
        entry = out["arms"]["A2@1000"]
        self.assertEqual(entry["file"], "compare_step_001000.json")
        self.assertAlmostEqual(entry["nominal"], 129 / 144)
        self.assertEqual(entry["held"], self.result["held"])
        self.assertAlmostEqual(entry["drops"]["lighting"]["estimate"], 28 / 144)
        self.assertIn("no cross-arm test", out["note"])


class CompareRefusals(Campaign):
    def setUp(self):
        super().setUp()
        self.prepare("A2")
        self.stage("A2", "benchmark", rule=lambda t: True)

    def test_refuses_incomplete_stages(self):
        shutil.rmtree(protocol.stage_dir(self.root, "A2", "benchmark", 1000))
        self.run_worker(arm="A2", tier="benchmark", shard=0, shards=3)
        evaluate.collect(self.root, "A2", "benchmark", 1000)
        with self.assertRaises(RuntimeError):
            evaluate.compare(self.root, "A2", 1000, repo=self.repo)

    def test_each_core_edit_rejects_the_comparison(self):
        for rel in ("experiments/gen_eval/rollout.py", "experiments/gen_scene/compose.py", "experiments/gen_scene/assets_manifest.json",
                    "experiments/vla_inputs_signals/signals.py", "experiments/xvla_props_p2_eval/analysis.py"):
            target = self.repo / rel
            original = target.read_bytes()
            target.write_bytes(original + b"\n")
            try:
                with self.assertRaises(ValueError, msg=rel):
                    evaluate.compare(self.root, "A2", 1000, repo=self.repo)
            finally:
                target.write_bytes(original)

    def test_a_layout_or_feasibility_manifest_edit_rejects_the_comparison(self):
        path = self.root / "plan/plan.json"
        original = path.read_text()
        path.write_text(original + " ")
        with self.assertRaises(ValueError):
            evaluate.compare(self.root, "A2", 1000, repo=self.repo)

    def test_a_texture_edit_rejects_the_comparison(self):
        victim = sorted((self.repo / ".cache/gen_textures").iterdir())[0]
        victim.write_bytes(victim.read_bytes()[:-1] + b"\x07")
        with self.assertRaises(ValueError):
            evaluate.compare(self.root, "A2", 1000, repo=self.repo)

    def test_an_arm_edit_after_prepare_rejects_that_arm(self):
        (self.repo / "experiments/vla_inputs_arm_fake/adapter.py").write_text(FAKE_ADAPTER + "\n# edited\n")
        with self.assertRaises(ValueError):
            evaluate.compare(self.root, "A2", 1000, repo=self.repo)

    def test_a_core_hash_that_differs_between_the_prepared_arms_is_refused(self):
        record = evaluate.read_json(self.root / "arms/A2/prepared.json")
        other = dict(record, core=dict(record["core"], sha256="0" * 64), arm=dict(record["arm"], name="A0"))
        (self.root / "arms/A0").mkdir()
        evaluate.write_json(self.root / "arms/A0/prepared.json", other)
        with self.assertRaises(ValueError):
            evaluate.compare(self.root, "A2", 1000, repo=self.repo)


class CompareDropped(Campaign):
    def test_a_layout_dropped_at_nominal_leaves_every_axis_and_is_counted_once(self):
        plan, feas = evaluate.load_manifests(self.root)
        lid = plan["tiers"]["benchmark"][5]
        keys = {protocol.trial_key(lid, a) for a in protocol.AXES}
        feas["feasible_trials"] = [k for k in feas["feasible_trials"] if k not in keys]
        feas["dropped_trials"] = {k: "nominal infeasible" for k in sorted(keys)}
        evaluate.write_json(self.root / "feasibility/feasibility.json", feas)
        self.prepare("A2")
        self.stage("A2", "benchmark", rule=lambda t: True)
        result = evaluate.compare(self.root, "A2", 1000, repo=self.repo)
        self.assertEqual((result["layouts_planned"], result["layouts_with_nominal"], result["nominal"]["trials"]), (144, 143, 143))
        for axis in levels.AXES:
            self.assertEqual(result["axes"][axis]["pooled"]["layouts"], 143, axis)


class Final(Campaign):
    def test_two_arms_side_by_side_with_no_cross_arm_test(self):
        for arm, name in (("A2", "vla_inputs_arm_fake"), ("A0", "vla_inputs_arm_fake2")):
            if arm == "A0":
                self.add_arm("vla_inputs_arm_fake2")
            self.prepare(arm, name=name)
            self.stage(arm, "benchmark", rule=lambda t, arm=arm: t["axis"] != "lighting" or arm == "A2")
            evaluate.compare(self.root, arm, 1000, repo=self.repo)
        out = evaluate.final(self.root)
        self.assertEqual(sorted(out["arms"]), ["A0@1000", "A2@1000"])
        self.assertEqual(out["arms"]["A2@1000"]["held"]["lighting"], True)
        self.assertEqual(out["arms"]["A0@1000"]["held"]["lighting"], False)


XVLA_ONLY = ("import sys; import experiments.gen_eval.evaluate as e; "
             "print(sorted(m for m in ('torch', 'lerobot', 'transformers', 'robosuite', 'mujoco') if m in sys.modules), "
             "sorted(m for m in sys.modules if 'vla_inputs_arm_' in m))")
CLOSURE_SCRIPT = """
import importlib, json, sys
from pathlib import Path
from experiments.gen_eval import contract
for name in NAMES:
    importlib.import_module(name)
bound = set(contract.core_provenance())
root = contract.ROOT
loaded = set()
for module in list(sys.modules.values()):
    path = getattr(module, "__file__", None)
    if path and Path(path).is_absolute() and Path(path).resolve().is_relative_to(root):
        loaded.add(Path(path).resolve().relative_to(root).as_posix())
print(json.dumps(sorted(p for p in loaded if p not in bound and "/tests/" not in p and not p.endswith("__init__.py"))))
"""


def unbound_after_importing(names):
    """Repository modules a fresh interpreter loads for `names` that the core closure does not bind."""
    import subprocess
    code = f"NAMES = {list(names)!r}\n" + CLOSURE_SCRIPT
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    if out.returncode:
        raise RuntimeError(out.stderr[-800:])
    return json.loads(out.stdout.strip().splitlines()[-1])


class Closure(unittest.TestCase):
    def test_importing_the_stages_loads_no_model_stack_and_no_arm(self):
        import subprocess
        out = subprocess.run([sys.executable, "-c", XVLA_ONLY], capture_output=True, text=True, cwd=REPO)
        self.assertEqual(out.stdout.strip(), "[] []", out.stderr[-400:])

    def test_the_core_closure_covers_everything_the_pure_core_loads(self):
        names = [f"experiments.gen_eval.{n}" for n in ("plan", "protocol", "stats", "feasibility", "evaluate", "contract", "disturbance")]
        names += ["experiments.gen_scene.levels", "experiments.gen_scene.pools", "experiments.gen_scene.conditions"]
        self.assertEqual(unbound_after_importing(names), [])

    def test_the_simulator_side_of_the_core_is_covered_too(self):
        try:
            import robosuite  # noqa: F401
        except ImportError:
            self.skipTest("needs the robosuite image")
        names = [f"experiments.gen_eval.{n}" for n in ("rollout", "scenes", "feasibility", "evaluate", "identity_gate")]
        names += ["experiments.gen_scene.gates", "experiments.gen_scene.expert"]
        self.assertEqual(unbound_after_importing(names), [])


if __name__ == "__main__":
    unittest.main()
