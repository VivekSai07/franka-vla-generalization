import hashlib
import importlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.vla_inputs_eval import adapters, contract, evaluate, plan as plan_module, protocol, stats

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
    """A campaign template built once: plan dir, a feasibility.json with every layout feasible, a repo copy."""
    if not _TEMPLATE:
        root = Path(tempfile.mkdtemp(prefix="m0_template_"))
        plan_module.write(root / "campaign/plan")
        plan = json.loads((root / "campaign/plan/plan.json").read_text())
        (root / "campaign/feasibility").mkdir()
        (root / "campaign/feasibility/feasibility.json").write_text(json.dumps(
            dict(plan_sha256=contract.digest(plan), feasible=sorted(plan["layouts"]), dropped=[], dropped_by_cell={}, reasons={})))
        for rel in contract.core_provenance():
            target = root / "repo" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / rel, target)
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
    (directory / "checkpoint_result.json").write_text(json.dumps(
        dict(status="complete", model_sha256=hashlib.sha256(content).hexdigest())))
    return directory


def stub_runner(rule, log=None):
    def run(adapter, state_builder, layout, trial, folder):
        success = bool(rule(trial))
        (folder / "cameras.mp4").write_bytes(b"x")
        (folder / "trace.npz").write_bytes(b"x")
        row = dict(trial, outcome="success" if success else "timeout", task_success=success, ticks=1, placed_bin=None,
                   final_gripper_open=True, cause="success" if success else "never_lifted", first_lifted=None,
                   wrong_object=False, distractor_displacement_m={}, names=[trial["object"]], wall_seconds=0.0)
        (folder / "episode.json").write_text(json.dumps(row))
        if log is not None:
            log.append(trial["ordinal"])
        return row
    return run


class Campaign(unittest.TestCase):
    """A fresh campaign root and repo copy per test, with a fake arm package registered under the real namespace."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="m0_test_"))
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

    def prepare(self, arm="A0", name="vla_inputs_arm_fake", checkpoints=None, phase=None):
        checkpoints = checkpoints or {1000: self.ckpt}
        if arm != "A0" and phase is None:
            phase = max(checkpoints)
        return evaluate.prepare(self.root, arm, f"experiments.{name}.adapter:FakeAdapter", checkpoints, repo=self.repo,
                                phase_checkpoint=phase)

    def fake_selection(self, arm, winner=1000, phase=1000):
        evaluate.write_json(self.root / "arms" / arm / "selection.json",
                            dict(arm=arm, winner=winner, phase_checkpoint=phase, comparison_checkpoints=sorted({winner, phase})))

    def run_worker(self, arm="A0", tier="smoke", checkpoint=1000, shard=0, shards=1, rule=lambda t: True, log=None, **kwargs):
        return evaluate.worker(self.root, arm, tier, checkpoint, shard, shards, repo=self.repo,
                               factory=kwargs.pop("factory", lambda prepared, c: _fake_instance(c)),
                               runner=stub_runner(rule, log), cooler=lambda: 0.0, **kwargs)

    def stage(self, arm, tier, checkpoint=1000, rule=lambda t: True):
        self.run_worker(arm, tier, checkpoint, rule=rule)
        return evaluate.collect(self.root, arm, tier, checkpoint)


def _fake_instance(checkpoint):
    module = sys.modules[next(m for m in sys.modules if m.startswith("experiments.vla_inputs_arm_fake") and m.endswith(".adapter"))]
    return module.FakeAdapter(checkpoint)


class Prepare(Campaign):
    def test_records_core_arm_manifests_and_checkpoint_facts(self):
        record = self.prepare()
        self.assertEqual(record["core"]["sha256"], contract.core_sha256(self.repo))
        self.assertEqual(sorted(record["arm"]["files"]), ["experiments/vla_inputs_arm_fake/README.md",
                                                           "experiments/vla_inputs_arm_fake/__init__.py",
                                                           "experiments/vla_inputs_arm_fake/adapter.py"])
        self.assertEqual(record["checkpoints"]["1000"]["model_sha256"], hashlib.sha256(b"a").hexdigest())
        self.assertEqual(record["dropped"], [])
        self.assertTrue(record["s0_seal"])
        self.assertTrue((self.root / "arms/A0/prepared.json").exists())

    def test_the_core_closure_includes_the_signal_source_and_the_p2_analysis(self):
        files = contract.core_provenance(self.repo)
        for rel in ("experiments/vla_inputs_signals/signals.py", "experiments/xvla_props_p2_eval/analysis.py",
                    "experiments/vla_inputs_eval/rollout.py", "experiments/vla_inputs_eval/stats.py",
                    "experiments/props_scene/scene.py"):
            self.assertIn(rel, files)
        self.assertNotIn("experiments/vla_inputs_eval/README.md", files)
        self.assertFalse([f for f in files if "/tests/" in f and "vla_inputs_eval" in f])

    def test_no_overwrite(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_refuses_weights_that_do_not_match_their_marker(self):
        (self.ckpt / "model.safetensors").write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_refuses_an_incomplete_marker(self):
        marker = json.loads((self.ckpt / "checkpoint_result.json").read_text())
        marker["status"] = "partial"
        (self.ckpt / "checkpoint_result.json").write_text(json.dumps(marker))
        with self.assertRaises(ValueError):
            self.prepare()

    def test_refuses_a_non_arm_module(self):
        with self.assertRaises(ValueError):
            evaluate.prepare(self.root, "A0", "os:system", {1000: self.ckpt}, repo=self.repo)
        with self.assertRaises(ValueError):
            evaluate.prepare(self.root, "control", "experiments.vla_inputs_arm_fake.adapter:FakeAdapter", {1000: self.ckpt}, repo=self.repo)

    def test_a_new_arm_must_declare_its_matched_phase_checkpoint_and_the_control_must_not(self):
        self.prepare("A0")
        for bad in (None, 5000):
            with self.assertRaises(ValueError, msg=bad):
                evaluate.prepare(self.root, "A1", "experiments.vla_inputs_arm_fake.adapter:FakeAdapter", {1000: self.ckpt},
                                 repo=self.repo, phase_checkpoint=bad)
        with self.assertRaises(ValueError):
            evaluate.prepare(self.root, "A0", "experiments.vla_inputs_arm_fake.adapter:FakeAdapter", {1000: self.ckpt},
                             repo=self.repo, phase_checkpoint=1000)

    def test_exclusions_by_cell_are_recorded(self):
        self.assertEqual(self.prepare()["dropped_by_cell"], {})

    def test_adding_a_future_arm_leaves_the_core_hash_unchanged(self):
        before = contract.core_sha256(self.repo)
        self.add_arm("vla_inputs_arm_fake3", FAKE_ADAPTER.replace("scale=1.0", "scale=3.0") + "\nEXTRA = 1\n")
        self.assertEqual(contract.core_sha256(self.repo), before)
        a0 = self.prepare("A0")
        a3 = self.prepare("A2", "vla_inputs_arm_fake3")
        self.assertEqual(a0["core"]["sha256"], before)
        self.assertEqual(a3["core"]["sha256"], before)

    def test_a_second_arm_needs_the_same_core_as_a0(self):
        self.prepare("A0")
        rollout = self.repo / "experiments/vla_inputs_eval/rollout.py"
        rollout.write_text(rollout.read_text() + "\n# a shared-core edit\n")
        with self.assertRaises(ValueError):
            self.prepare("A1")

    def test_a_different_declared_adapter_with_an_identical_core_is_accepted(self):
        self.add_arm("vla_inputs_arm_fake2", FAKE_ADAPTER.replace("scale=1.0", "scale=2.0"))
        a0 = self.prepare("A0")
        a1 = self.prepare("A1", "vla_inputs_arm_fake2")
        self.assertEqual(a0["core"]["sha256"], a1["core"]["sha256"])
        self.assertNotEqual(a0["arm"]["sha256"], a1["arm"]["sha256"])

    def test_the_signal_source_must_be_the_sealed_one(self):
        signals = self.repo / "experiments/vla_inputs_signals/signals.py"
        signals.write_text(signals.read_text() + "\n# edited after the seal\n")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_a_frozen_p2_file_edit_is_refused(self):
        scene = self.repo / "experiments/props_scene/scene.py"
        scene.write_text(scene.read_text() + "\n# edited\n")
        with self.assertRaises(ValueError):
            self.prepare()


class Worker(Campaign):
    def setUp(self):
        super().setUp()
        self.prepare()

    def trial_count(self, tier):
        plan, feas = evaluate.load_manifests(self.root)
        return len(protocol.stage_trials(plan, feas, tier))

    def test_runs_every_trial_of_its_share_and_writes_telemetry(self):
        log = []
        rows = self.run_worker(tier="smoke", log=log)
        self.assertEqual(len(rows), 2)
        folders = sorted((self.root / "arms/A0/smoke/step_001000").glob("trial_*"))
        self.assertEqual(len(folders), 2)
        for folder in folders:
            self.assertTrue(protocol.trial_is_complete(folder))
            self.assertIn("rss_max_mib", json.loads((folder / "telemetry.json").read_text()))

    def test_shards_partition_the_stage_and_are_interleaved(self):
        a, b = [], []
        self.run_worker(tier="selection", shard=0, shards=2, log=a)
        self.run_worker(tier="selection", shard=1, shards=2, log=b)
        self.assertEqual(sorted(a + b), list(range(1, 81)))
        self.assertEqual(a[:3], [1, 3, 5])
        self.assertEqual(b[:3], [2, 4, 6])

    def test_a_resumed_worker_skips_complete_trials_and_redoes_incomplete_ones(self):
        self.run_worker(tier="smoke")
        directory = self.root / "arms/A0/smoke/step_001000"
        trials = sorted(directory.glob("trial_*"))
        (trials[1] / "episode.json").unlink()  # one trial was interrupted before its record
        marker = (trials[0] / "cameras.mp4").stat().st_mtime_ns
        log = []
        self.run_worker(tier="smoke", log=log)
        self.assertEqual(len(log), 1)
        self.assertEqual((trials[0] / "cameras.mp4").stat().st_mtime_ns, marker)
        self.assertTrue(protocol.trial_is_complete(trials[1]))
        self.assertEqual(len(list((directory / "quarantine").iterdir())), 1)

    def test_a_finished_stage_builds_no_adapter_at_all(self):
        self.run_worker(tier="smoke")
        built = []
        self.run_worker(tier="smoke", factory=lambda prepared, c: built.append(c))
        self.assertEqual(built, [])

    def test_one_worker_builds_exactly_one_adapter_for_exactly_one_checkpoint(self):
        built = []

        def factory(prepared, checkpoint):
            built.append(checkpoint)
            return _fake_instance(checkpoint)

        self.run_worker(tier="selection", shard=0, shards=4, factory=factory)
        self.assertEqual(built, [1000])
        with self.assertRaises(ValueError):
            evaluate.worker(self.root, "A0", "smoke", [1000, 2000], 0, 1, repo=self.repo)
        with self.assertRaises(SystemExit):
            evaluate.parse_args(["--root", str(self.root), "worker", "--arm", "A0", "--tier", "smoke", "--checkpoint", "1000,2000",
                                 "--shard", "0", "--shards", "1"])

    def test_an_unprepared_checkpoint_is_refused(self):
        with self.assertRaises(ValueError):
            self.run_worker(checkpoint=2000)

    def test_a_changed_checkpoint_is_refused(self):
        (self.ckpt / "model.safetensors").write_bytes(b"other bytes")
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_a_core_edit_after_prepare_stops_the_worker(self):
        stats_file = self.repo / "experiments/vla_inputs_eval/stats.py"
        stats_file.write_text(stats_file.read_text() + "\n# edited after prepare\n")
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_an_arm_edit_after_prepare_rejects_that_arm(self):
        adapter = self.repo / "experiments/vla_inputs_arm_fake/adapter.py"
        adapter.write_text(adapter.read_text() + "\n# edited after prepare\n")
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_a_manifest_edit_after_prepare_stops_the_worker(self):
        path = self.root / "plan/plan.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaises(ValueError):
            self.run_worker()

    def test_an_infrastructure_error_leaves_a_failure_file_and_is_not_swallowed(self):
        def broken(adapter, state_builder, layout, trial, folder):
            raise RuntimeError("renderer died")

        with self.assertRaises(RuntimeError):
            evaluate.worker(self.root, "A0", "smoke", 1000, 0, 1, repo=self.repo, factory=lambda p, c: _fake_instance(c),
                            runner=broken, cooler=lambda: 0.0)
        failure = json.loads(next((self.root / "arms/A0/smoke/step_001000").glob("failure_shard0.json")).read_text())
        self.assertEqual(failure["error_type"], "RuntimeError")

    def test_a_nonfinite_policy_output_is_a_recorded_failure_not_a_missing_trial(self):
        rows = self.run_worker(tier="smoke", rule=lambda t: False)
        self.assertTrue(all(not r["task_success"] for r in rows))
        report = evaluate.collect(self.root, "A0", "smoke", 1000)
        self.assertEqual((report["status"], len(report["episodes"])), ("complete", 2))


class ArmInputs(Campaign):
    """An arm's input directories (the X-VLA base bundle: tokenizer files) are bound in its manifest."""

    def setUp(self):
        super().setUp()
        bundle = self.repo / "assets/bundle"
        (bundle / "tokenizer").mkdir(parents=True)
        (bundle / "tokenizer/tokenizer.json").write_text('{"vocab": 1}')
        (bundle / "manifest.json").write_text("{}")
        self.add_arm("vla_inputs_arm_fake", FAKE_ADAPTER + 'INPUT_DIRS = ("assets/bundle",)\n')
        self.bundle = bundle

    def test_every_file_of_the_input_directory_is_recorded_with_hash_size_and_mtime(self):
        record = self.prepare()
        files = record["arm"]["inputs"]["assets/bundle"]
        self.assertEqual(sorted(files), ["manifest.json", "tokenizer/tokenizer.json"])
        self.assertEqual(files["manifest.json"]["sha256"], hashlib.sha256(b"{}").hexdigest())
        self.assertEqual(files["manifest.json"]["bytes"], 2)
        self.assertIn("mtime_ns", files["manifest.json"])

    def test_an_arm_without_input_directories_records_none(self):
        self.add_arm("vla_inputs_arm_fake2")
        self.assertEqual(self.prepare("A0", "vla_inputs_arm_fake2")["arm"]["inputs"], {})

    def test_the_arm_hash_covers_the_inputs(self):
        a0 = self.prepare("A0")
        (self.bundle / "manifest.json").write_text('{"x":1}')
        a1 = self.prepare("A1")
        self.assertEqual(a0["core"]["sha256"], a1["core"]["sha256"])
        self.assertNotEqual(a0["arm"]["sha256"], a1["arm"]["sha256"])

    def test_a_missing_or_empty_input_directory_is_refused(self):
        shutil.rmtree(self.bundle)
        with self.assertRaises(ValueError):
            self.prepare()
        self.bundle.mkdir(parents=True)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_a_symlink_in_an_input_directory_is_refused(self):
        (self.bundle / "link.json").symlink_to(self.bundle / "manifest.json")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_a_worker_rejects_a_changed_added_or_removed_input_file(self):
        self.prepare()
        target = self.bundle / "tokenizer/tokenizer.json"
        original = target.read_text()
        target.write_text('{"vocab": 2}')  # same size, different bytes
        with self.assertRaises(ValueError):
            self.run_worker()
        target.write_text(original)
        (self.bundle / "extra.txt").write_text("new")
        with self.assertRaises(ValueError):
            self.run_worker()
        (self.bundle / "extra.txt").unlink()
        target.unlink()
        with self.assertRaises(ValueError):
            self.run_worker()
        target.write_text(original)
        self.assertEqual(len(self.run_worker(tier="smoke")), 2)

    def test_large_files_get_a_cheap_worker_check_and_a_full_hash_in_the_comparison(self):
        import os
        record = self.prepare()
        original_limit = evaluate.LARGE_BYTES
        evaluate.LARGE_BYTES = 1  # treat every file as large
        try:
            target = self.bundle / "tokenizer/tokenizer.json"
            stat = target.stat()
            target.write_text('{"vocab": 2}')
            os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same size, same mtime, other bytes
            evaluate.verify_arm(record, self.repo)  # the worker's cheap check does not see it
            with self.assertRaises(ValueError):
                evaluate.verify_arm(record, self.repo, deep=True)
            target.write_text('{"vocab": 1}')
            os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            with self.assertRaises(ValueError):  # a new mtime is caught cheaply
                evaluate.verify_arm(record, self.repo)
        finally:
            evaluate.LARGE_BYTES = original_limit

    def test_a_size_change_with_an_unchanged_mtime_is_caught_cheaply_on_a_large_file(self):
        import os
        record = self.prepare()
        original_limit = evaluate.LARGE_BYTES
        evaluate.LARGE_BYTES = 1
        try:
            target = self.bundle / "tokenizer/tokenizer.json"
            stat = target.stat()
            target.write_text('{"vocab": 11}')
            os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with self.assertRaises(ValueError):
                evaluate.verify_arm(record, self.repo)
        finally:
            evaluate.LARGE_BYTES = original_limit

    def _compared_with_silent_edit(self, edited):
        """Both arms prepared with their own input directory; one file there changes bytes only (size and mtime kept)."""
        import os
        bundle2 = self.repo / "assets/bundle2"
        bundle2.mkdir(parents=True)
        (bundle2 / "manifest.json").write_text("{}")
        self.add_arm("vla_inputs_arm_fake2", FAKE_ADAPTER.replace("scale=1.0", "scale=2.0") + 'INPUT_DIRS = ("assets/bundle2",)\n')
        self.prepare("A0")
        self.prepare("A1", "vla_inputs_arm_fake2")
        self.fake_selection("A1")
        self.stage("A0", "comparison", 1000, rule=lambda t: False)
        self.stage("A1", "comparison", 1000, rule=lambda t: True)
        target = (self.bundle if edited == "control" else bundle2) / "manifest.json"
        stat = target.stat()
        original_limit = evaluate.LARGE_BYTES
        evaluate.LARGE_BYTES = 1  # every input file counts as large: only the comparison's full hash can see the edit
        try:
            target.write_text("[]")
            os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            return evaluate.compare(self.root, "A1", 1000, repo=self.repo)
        finally:
            evaluate.LARGE_BYTES = original_limit

    def test_the_comparison_hashes_the_control_inputs_in_full(self):
        with self.assertRaises(ValueError):
            self._compared_with_silent_edit("control")

    def test_the_comparison_hashes_the_new_arms_inputs_in_full(self):
        with self.assertRaises(ValueError):
            self._compared_with_silent_edit("new")


class Collect(Campaign):
    def setUp(self):
        super().setUp()
        self.prepare()

    def test_complete_only_if_every_planned_trial_exists(self):
        self.run_worker(tier="smoke", shard=0, shards=2)
        report = evaluate.collect(self.root, "A0", "smoke", 1000)
        self.assertEqual((report["status"], report["missing_ordinals"]), ("incomplete", [2]))
        with self.assertRaises(RuntimeError):
            evaluate.load_stage(self.root, "A0", "smoke", 1000)
        self.run_worker(tier="smoke", shard=1, shards=2)
        self.assertEqual(evaluate.collect(self.root, "A0", "smoke", 1000)["status"], "complete")

    def test_a_trial_that_does_not_match_the_plan_is_refused(self):
        self.run_worker(tier="smoke")
        folder = next((self.root / "arms/A0/smoke/step_001000").glob("trial_001_*"))
        row = json.loads((folder / "episode.json").read_text())
        row["seed"] += 1
        (folder / "episode.json").write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            evaluate.collect(self.root, "A0", "smoke", 1000)

    def test_dropped_layouts_leave_the_stage_and_are_recorded(self):
        plan, feas = evaluate.load_manifests(self.root)
        victim = plan["tiers"]["smoke"][0]
        feas["feasible"] = [i for i in feas["feasible"] if i != victim]
        feas["dropped"], feas["dropped_by_cell"], feas["reasons"] = [victim], {"x": 1}, {victim: "stub"}
        path = self.root / "feasibility/feasibility.json"
        path.write_text(json.dumps(feas))
        self.assertEqual(len(protocol.stage_trials(plan, feas, "smoke")), 1)


class Select(Campaign):
    def setUp(self):
        super().setUp()
        self.second = make_checkpoint(self.tmp, "ckpt_b", b"b")
        self.prepare("A0")
        self.prepare("A1", checkpoints={1000: self.ckpt, 2000: self.second}, phase=2000)

    def test_picks_the_most_successful_checkpoint_on_the_selection_tier(self):
        self.stage("A1", "selection", 1000, rule=lambda t: t["ordinal"] <= 30)
        self.stage("A1", "selection", 2000, rule=lambda t: t["ordinal"] <= 50)
        decision = evaluate.select(self.root, "A1")
        self.assertEqual((decision["winner"], decision["counts"], decision["tied"]), (2000, {"1000": 30, "2000": 50}, [2000]))
        self.assertTrue((self.root / "arms/A1/selection.json").exists())

    def test_the_matched_phase_checkpoint_joins_the_comparison_list_unless_it_is_the_winner(self):
        self.stage("A1", "selection", 1000, rule=lambda t: t["ordinal"] <= 60)
        self.stage("A1", "selection", 2000, rule=lambda t: t["ordinal"] <= 20)
        decision = evaluate.select(self.root, "A1")  # phase checkpoint is 2000 (the later one), the winner is 1000
        self.assertEqual((decision["winner"], decision["phase_checkpoint"], decision["comparison_checkpoints"]), (1000, 2000, [1000, 2000]))

    def test_a_coinciding_winner_and_phase_checkpoint_is_run_once(self):
        self.stage("A1", "selection", 1000, rule=lambda t: t["ordinal"] <= 20)
        self.stage("A1", "selection", 2000, rule=lambda t: t["ordinal"] <= 60)
        self.assertEqual(evaluate.select(self.root, "A1")["comparison_checkpoints"], [2000])

    def test_a_tie_goes_to_the_later_checkpoint_and_is_recorded(self):
        self.stage("A1", "selection", 1000, rule=lambda t: t["ordinal"] <= 40)
        self.stage("A1", "selection", 2000, rule=lambda t: t["ordinal"] <= 40)
        decision = evaluate.select(self.root, "A1")
        self.assertEqual((decision["winner"], decision["tied"]), (2000, [1000, 2000]))

    def test_reads_only_the_selection_tier(self):
        self.stage("A1", "selection", 1000)
        self.stage("A1", "selection", 2000)
        tiers = []
        original = evaluate.load_stage
        evaluate.load_stage = lambda root, arm, tier, ckpt: (tiers.append(tier), original(root, arm, tier, ckpt))[1]
        try:
            evaluate.select(self.root, "A1")
        finally:
            evaluate.load_stage = original
        self.assertEqual(set(tiers), {"selection"})

    def test_a_held_out_layout_never_reaches_selection(self):
        self.stage("A1", "selection", 1000)
        self.stage("A1", "selection", 2000)
        path = self.root / "arms/A1/selection/step_001000/result.json"
        report = json.loads(path.read_text())
        report["episodes"][0]["kind"] = "held_out"
        path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            evaluate.select(self.root, "A1")

    def test_an_incomplete_candidate_blocks_selection(self):
        self.stage("A1", "selection", 1000)
        with self.assertRaises(RuntimeError):
            evaluate.select(self.root, "A1")

    def test_the_selection_tier_contains_no_held_out_layout(self):
        plan, feas = evaluate.load_manifests(self.root)
        trials = protocol.stage_trials(plan, feas, "selection")
        self.assertEqual({t["kind"] for t in trials}, {"trained"})


class Compare(Campaign):
    def setUp(self):
        super().setUp()
        self.add_arm("vla_inputs_arm_fake2", FAKE_ADAPTER.replace("scale=1.0", "scale=2.0"))
        self.prepare("A0")
        self.prepare("A1", "vla_inputs_arm_fake2")
        self.fake_selection("A1")

    def both(self, rule_new, rule_control):
        self.stage("A0", "comparison", 1000, rule=rule_control)
        self.stage("A1", "comparison", 1000, rule=rule_new)

    def test_known_discordance_gives_the_exact_p_and_effect(self):
        # control succeeds on ordinals 1..300; the new arm on 1..250 and 301..360: new-only 60, control-only 50
        self.both(lambda t: t["ordinal"] <= 250 or 301 <= t["ordinal"] <= 360, lambda t: t["ordinal"] <= 300)
        result = evaluate.compare(self.root, "A1", 1000, repo=self.repo)
        self.assertEqual(result["discordant"], dict(new_only=60, control_only=50))
        self.assertEqual((result["new_successes"], result["control_successes"], result["layouts"]), (310, 300, 576))
        self.assertAlmostEqual(result["mcnemar_exact_p"], stats.mcnemar(60, 50), places=12)
        self.assertAlmostEqual(result["effect"]["estimate"], 10 / 576)
        self.assertEqual(len(result["discordant_layouts"]["new_only"]), 60)

    def test_reads_only_the_comparison_tier(self):
        self.both(lambda t: True, lambda t: False)
        tiers = []
        original = evaluate.load_stage
        evaluate.load_stage = lambda root, arm, tier, ckpt: (tiers.append(tier), original(root, arm, tier, ckpt))[1]
        try:
            evaluate.compare(self.root, "A1", 1000, repo=self.repo)
        finally:
            evaluate.load_stage = original
        self.assertEqual(set(tiers), {"comparison"})

    def test_the_comparison_tier_never_chooses_a_checkpoint(self):
        self.both(lambda t: True, lambda t: False)
        with self.assertRaises(ValueError):
            evaluate.compare(self.root, "A1", 2000, repo=self.repo)
        (self.root / "arms/A1/selection.json").unlink()
        with self.assertRaises(FileNotFoundError):
            evaluate.compare(self.root, "A1", 1000, repo=self.repo)

    def test_the_two_comparison_checkpoints_are_labelled_primary_and_matched_phase(self):
        second = make_checkpoint(self.tmp, "ckpt_phase", b"p")
        self.prepare("A2", "vla_inputs_arm_fake2", checkpoints={1000: self.ckpt, 2000: second}, phase=2000)
        self.fake_selection("A2", winner=1000, phase=2000)
        self.stage("A0", "comparison", 1000, rule=lambda t: t["ordinal"] % 2 == 0)
        for ckpt in (1000, 2000):
            self.stage("A2", "comparison", ckpt, rule=lambda t: True)
        roles = {c: evaluate.compare(self.root, "A2", c, repo=self.repo)["role"] for c in (1000, 2000)}
        self.assertEqual(roles, {1000: "primary", 2000: "matched-phase secondary"})
        self.assertEqual(sorted(p.name for p in (self.root / "arms/A2").glob("compare_step_*.json")),
                         ["compare_step_001000.json", "compare_step_002000.json"])

    def test_the_population_and_the_exclusions_are_stated(self):
        self.both(lambda t: True, lambda t: False)
        result = evaluate.compare(self.root, "A1", 1000, repo=self.repo)
        self.assertIn("576 feasible comparison layouts of 576 planned; 0 dropped", result["population"])
        self.assertEqual(result["dropped_by_cell"], {})

    def test_refuses_the_control_against_itself(self):
        with self.assertRaises(ValueError):
            evaluate.compare(self.root, "A0", 1000, repo=self.repo)

    def test_refuses_incomplete_pairs(self):
        self.stage("A0", "comparison", 1000)
        self.run_worker("A1", "comparison", 1000, shard=0, shards=2)
        evaluate.collect(self.root, "A1", "comparison", 1000)
        with self.assertRaises(RuntimeError):
            evaluate.compare(self.root, "A1", 1000, repo=self.repo)

    def test_a_pair_set_mismatch_is_refused(self):
        self.both(lambda t: True, lambda t: True)
        path = self.root / "arms/A1/comparison/step_001000/result.json"
        report = json.loads(path.read_text())
        report["episodes"] = report["episodes"][:-1]
        path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            evaluate.compare(self.root, "A1", 1000, repo=self.repo)

    def test_each_shared_core_edit_rejects_the_comparison(self):
        self.both(lambda t: True, lambda t: False)
        for rel in ("experiments/vla_inputs_eval/rollout.py", "experiments/xvla_props_p2_eval/analysis.py",
                    "experiments/vla_inputs_signals/signals.py", "experiments/vla_inputs_eval/stats.py",
                    "experiments/vla_inputs_eval/adapters.py", "experiments/vla_inputs_eval/scenes.py"):
            path = self.repo / rel
            original = path.read_text()
            path.write_text(original + "\n# edited\n")
            try:
                with self.assertRaises(ValueError, msg=rel):
                    evaluate.compare(self.root, "A1", 1000, repo=self.repo)
            finally:
                path.write_text(original)

    def test_a_layout_or_feasibility_manifest_edit_rejects_the_comparison(self):
        self.both(lambda t: True, lambda t: False)
        for rel in ("plan/plan.json", "feasibility/feasibility.json"):
            path = self.root / rel
            original = path.read_text()
            path.write_text(original + " ")
            try:
                with self.assertRaises(ValueError, msg=rel):
                    evaluate.compare(self.root, "A1", 1000, repo=self.repo)
            finally:
                path.write_text(original)

    def test_an_arm_adapter_edit_after_prepare_rejects_that_arm_only(self):
        self.both(lambda t: True, lambda t: False)
        adapter = self.repo / "experiments/vla_inputs_arm_fake2/adapter.py"
        adapter.write_text(adapter.read_text() + "\n# edited after prepare\n")
        with self.assertRaises(ValueError):
            evaluate.compare(self.root, "A1", 1000, repo=self.repo)

    def test_the_control_adapter_edit_after_prepare_rejects_the_comparison(self):
        self.both(lambda t: True, lambda t: False)
        adapter = self.repo / "experiments/vla_inputs_arm_fake/adapter.py"
        adapter.write_text(adapter.read_text() + "\n# edited after prepare\n")
        with self.assertRaises(ValueError):
            evaluate.compare(self.root, "A1", 1000, repo=self.repo)

    def test_a_core_hash_that_differs_between_the_prepared_arms_is_refused(self):
        self.both(lambda t: True, lambda t: False)
        path = self.root / "arms/A1/prepared.json"
        prepared = json.loads(path.read_text())
        prepared["core"]["sha256"] = "0" * 64
        path.write_text(json.dumps(prepared))
        with self.assertRaises(ValueError):
            evaluate.compare(self.root, "A1", 1000, repo=self.repo)

    def test_a_result_is_never_overwritten(self):
        self.both(lambda t: True, lambda t: False)
        evaluate.compare(self.root, "A1", 1000, repo=self.repo)
        with self.assertRaises(FileExistsError):
            evaluate.compare(self.root, "A1", 1000, repo=self.repo)


class CompareWithDroppedLayouts(Campaign):
    """Three comparison layouts fail the expert feasibility pass: they leave every arm's population, counted by cell."""

    def setUp(self):
        super().setUp()
        self.add_arm("vla_inputs_arm_fake2", FAKE_ADAPTER.replace("scale=1.0", "scale=2.0"))
        plan, feas = evaluate.load_manifests(self.root)
        self.victims = [plan["tiers"]["comparison"][i] for i in (0, 1, 300)]
        feas["feasible"] = [i for i in feas["feasible"] if i not in self.victims]
        feas["dropped"] = sorted(self.victims)
        cells = {}
        for v in self.victims:
            key = "-".join(map(str, plan["layouts"][v]["cell"]))
            cells[key] = cells.get(key, 0) + 1
        feas["dropped_by_cell"], feas["reasons"] = dict(sorted(cells.items())), {v: "stub" for v in self.victims}
        (self.root / "feasibility/feasibility.json").write_text(json.dumps(feas))
        self.expected_cells = feas["dropped_by_cell"]
        self.prepare("A0")
        self.prepare("A1", "vla_inputs_arm_fake2")
        self.fake_selection("A1")

    def test_the_population_excludes_them_for_both_arms_and_names_them_by_cell(self):
        self.stage("A0", "comparison", 1000, rule=lambda t: t["ordinal"] % 3 == 0)
        self.stage("A1", "comparison", 1000, rule=lambda t: True)
        result = evaluate.compare(self.root, "A1", 1000, repo=self.repo)
        self.assertEqual(result["layouts"], 573)
        self.assertEqual(sorted(result["dropped_layouts"]), sorted(self.victims))
        self.assertEqual(result["dropped_by_cell"], self.expected_cells)
        self.assertIn("573 feasible comparison layouts of 576 planned; 3 dropped", result["population"])
        folders = {p.name for p in (self.root / "arms/A1/comparison/step_001000").glob("trial_*")}
        self.assertEqual(len(folders), 573)
        self.assertFalse(any(name.endswith("_" + v) for v in self.victims for name in folders))

    def test_ordinals_keep_their_plan_position_so_a_drop_leaves_a_gap(self):
        plan, feas = evaluate.load_manifests(self.root)
        ordinals = [t["ordinal"] for t in protocol.stage_trials(plan, feas, "comparison")]
        self.assertNotIn(1, ordinals)
        self.assertNotIn(2, ordinals)
        self.assertNotIn(301, ordinals)
        self.assertEqual(ordinals[0], 3)


class Final(Campaign):
    def test_holm_over_the_family_with_unrun_arms_as_one(self):
        for arm, p in (("A1", 0.004), ("A2", 0.2)):
            (self.root / "arms" / arm).mkdir(parents=True)
            self.fake_selection(arm, winner=1000, phase=2000)
            (self.root / "arms" / arm / "compare_step_001000.json").write_text(json.dumps(dict(mcnemar_exact_p=p)))
            (self.root / "arms" / arm / "compare_step_002000.json").write_text(json.dumps(dict(mcnemar_exact_p=0.0001)))
        out = evaluate.final(self.root)
        self.assertEqual(out["files"], {"A1": "compare_step_001000.json", "A2": "compare_step_001000.json"})
        self.assertAlmostEqual(out["holm"]["A1"], 0.016)
        self.assertAlmostEqual(out["holm"]["A2"], 0.6)
        self.assertEqual((out["holm"]["A3"], out["holm"]["A4"]), (1.0, 1.0))


CLOSURE_SCRIPT = """
import importlib, json, sys
from pathlib import Path
from experiments.vla_inputs_eval import contract
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


class XVLAOnlyEnvironment(unittest.TestCase):
    def test_importing_the_stages_loads_no_model_stack_and_no_arm(self):
        import subprocess
        code = ("import sys; import experiments.vla_inputs_eval.evaluate as e; "
                "print(sorted(m for m in ('torch', 'lerobot', 'transformers', 'robosuite', 'mujoco') if m in sys.modules), "
                "sorted(m for m in sys.modules if 'vla_inputs_arm_' in m))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
        self.assertEqual(out.stdout.strip(), "[] []", out.stderr[-400:])


class Closure(unittest.TestCase):
    def test_the_core_closure_covers_everything_the_pure_core_loads(self):
        names = [f"experiments.vla_inputs_eval.{n}" for n in ("plan", "protocol", "stats", "adapters", "feasibility", "evaluate",
                                                                "contract")]
        self.assertEqual(unbound_after_importing(names), [])

    def test_the_simulator_side_of_the_core_is_covered_too(self):
        try:
            import robosuite  # noqa: F401
        except ImportError:
            self.skipTest("needs the robosuite image")
        names = [f"experiments.vla_inputs_eval.{n}" for n in ("rollout", "scenes", "feasibility", "evaluate")]
        names.append("experiments.props_p2_data.episode")
        self.assertEqual(unbound_after_importing(names), [])


if __name__ == "__main__":
    unittest.main()
