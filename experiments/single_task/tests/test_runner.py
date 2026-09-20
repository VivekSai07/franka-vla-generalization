import copy
from contextlib import ExitStack, contextmanager, nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from experiments.single_task import contract as c, evaluate, runner
from test_contract import manifest


class RunnerTests(unittest.TestCase):
    @contextmanager
    def scratch_fixture(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            cache, scratch = root/".cache/benchmark_models", root/"scratch"
            for name in ("policy", "vlm"):
                (cache/name).mkdir(parents=True)
                (cache/name/"model.safetensors").write_bytes(b"original cached weights")
            c.h.write_json(cache/"manifest.json", {
                "policy": dict(repo_id="lerobot/smolvla_base", revision=c.REVISION, path=str(cache/"policy")),
                "vlm": dict(revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467", path=str(cache/"vlm"))})
            markers = [dict(status="complete", update=i, reload_fp32_weights_exact=True,
                            reload_max_action_error=0.) for i in (1, 2)]
            for marker in markers:
                checkpoint = scratch/f"checkpoints/{marker['update']:06d}"
                checkpoint.mkdir(parents=True)
                c.h.write_json(checkpoint/"checkpoint_result.json", marker)
            report = dict(status="complete", purpose="single_task_preflight", completed_training_steps=2,
                          final_data_binding_passed=True, source_hashes={"code": "hash"},
                          dataset_fingerprint="dataset", checkpoints=markers,
                          final_checkpoint_files=c.inventory(scratch/"checkpoints/000002"),
                          asset_files=c.inventory(cache))
            c.h.write_json(scratch/"result.json", report)
            stack.enter_context(patch.object(c, "ROOT", root))
            stack.enter_context(patch.object(c, "SCRATCH", scratch))
            stack.enter_context(patch.object(c, "sources", return_value={"code": "hash"}))
            stack.enter_context(patch.object(c.h, "require_complete", return_value={**manifest(), "timing": {}}))
            stack.enter_context(patch.object(c.h, "require_replay_pass", return_value={"dataset_fingerprint": "dataset"}))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
            yield root, cache, scratch, report

    def test_matching_scratch_cache_reaches_cuda_gate(self):
        with self.scratch_fixture() as (root, cache, scratch, report):
            with self.assertRaisesRegex(RuntimeError, "CUDA with BF16"):
                runner.train(root, False, {})

    def test_tampered_cache_rejected_before_full_training(self):
        for name in ("policy", "vlm"):
            for mutation in ("bytes", "added", "removed"):
                with self.subTest(asset=name, mutation=mutation), self.scratch_fixture() as (root, cache, scratch, report):
                    path = cache/name/"model.safetensors"
                    if mutation == "bytes":
                        path.write_bytes(b"changed under the same pinned revision")
                    elif mutation == "added":
                        (cache/name/"extra.json").write_text("{}")
                    else:
                        path.unlink()
                    with self.assertRaisesRegex((ValueError, RuntimeError), "scratch preflight"):
                        runner.train(root, False, {})

    def test_stale_scratch_asset_report_rejected(self):
        for assets in (None, {}, {"policy/model.safetensors": "stale"}):
            with self.subTest(assets=assets), self.scratch_fixture() as (root, cache, scratch, report):
                if assets is None:
                    report.pop("asset_files")
                else:
                    report["asset_files"] = assets
                c.h.write_json(scratch/"result.json", report)
                with self.assertRaisesRegex((ValueError, RuntimeError), "scratch preflight"):
                    runner.train(root, False, {})

    def test_scratch_acceptance_rechecks_absolute_reload_error(self):
        for update in (1, 2):
            for error in (2e-5, float("nan"), float("inf"), -1.):
                with self.subTest(update=update, error=error), self.scratch_fixture() as (root, cache, scratch, report):
                    report["checkpoints"][update-1]["reload_max_action_error"] = error
                    # Include nonfinite stale JSON from external/older writers.
                    (scratch/"result.json").write_text(json.dumps(report))
                    with self.assertRaisesRegex((ValueError, RuntimeError), "absolute reload"):
                        runner.train(root, False, {})

    def test_every_orchestration_save_rejects_relative_only_gripper_agreement(self):
        expected = torch.full((1, 50, 8), 255.)
        actual = expected.clone()
        actual[..., 7] += .001
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        error = (actual-expected).abs().max().item()
        self.assertGreater(error, 1e-5)
        for preflight, bad_update in ((True, 1), (True, 2), *((False, i) for i in (1, 1000, 2000, 3000, 4000, 5000))):
            with self.subTest(preflight=preflight, update=bad_update), self.scratch_fixture() as (root, cache, scratch, saved), ExitStack() as stack:
                # Synthetic CPU orchestration only: no pretrained model, CUDA,
                # real dataset batches, checkpoint writer or optimizer execution.
                policy = torch.nn.Module()
                policy.action_out_proj = torch.nn.Linear(1, 1, bias=False)
                parameter = policy.action_out_proj.weight
                parameter.numel = lambda: 450046176
                policy.forward = lambda batch: (parameter.square().mean(), {})
                config = SimpleNamespace(chunk_size=50, n_obs_steps=1, save_pretrained=lambda path: None)
                from lerobot.configs.types import FeatureType
                features = {k: SimpleNamespace(type=FeatureType.STATE if k == "observation.state" else FeatureType.VISUAL,
                                               shape=shape) for k, shape in c.OBS.items()}
                features["action"] = SimpleNamespace(type=FeatureType.ACTION, shape=(8,))
                raw = Mock()
                optimizer = Mock(param_groups=[{}])
                def step():
                    with torch.no_grad():
                        parameter.add_(.01)
                optimizer.step.side_effect = step
                updates = []
                def save(*args):
                    update = args[7]
                    updates.append(update)
                    return dict(status="complete", update=update, reload_fp32_weights_exact=True,
                                reload_max_action_error=error if update == bad_update else 0.)
                patches = [
                    patch.object(torch.cuda, "is_available", return_value=True),
                    patch.object(torch.cuda, "is_bf16_supported", return_value=True),
                    patch.object(torch.cuda, "synchronize"),
                    patch.object(torch.cuda, "max_memory_allocated", return_value=0),
                    patch.object(torch, "autocast", side_effect=lambda *a, **k: nullcontext()),
                    patch.object(torch.optim, "AdamW", return_value=optimizer),
                    patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=100*1024**3)),
                    patch("lerobot.configs.policies.PreTrainedConfig.from_pretrained", return_value=config),
                    patch("lerobot.datasets.lerobot_dataset.LeRobotDataset", return_value=raw),
                    patch("lerobot.datasets.utils.dataset_to_policy_features", return_value=features),
                    patch("lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy.from_pretrained", return_value=policy),
                    patch("lerobot.policies.factory.make_pre_post_processors", return_value=(lambda x: x, lambda x: x)),
                    patch("torch.utils.data.DataLoader", return_value=[{}]),
                    patch.object(c, "indices", return_value=([], [0], {"training": [0], "validation": [0]})),
                    patch.object(c.h, "frame_indices", return_value=[]),
                    patch.object(c.h, "compute_training_stats", return_value={}),
                    patch.object(c.h, "promote_trainable_parameters", return_value={"trainable_parameters": 99880992}),
                    patch.object(c.h, "fixed_training_probes", return_value={}),
                    patch.object(c.h, "check_batch"),
                    patch.object(c.h, "backward_microbatch"),
                    patch.object(c.h, "verify_optimizer_precision"),
                    patch.object(c.h, "save_checkpoint", side_effect=save),
                    patch("builtins.print"),
                ]
                for item in patches:
                    stack.enter_context(item)
                report = {}
                with self.assertRaisesRegex(ValueError, "absolute reload"):
                    runner.train(root, preflight, report)
                self.assertEqual(updates[-1], bad_update)
                self.assertEqual([m["update"] for m in report["checkpoints"]], updates[:-1])
                self.assertNotEqual(report["status"], "complete")

    def test_preflight_dispatch_never_evaluates(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "train") as train, \
                patch.object(runner, "evaluate") as evaluate_mock, patch.object(c, "SCRATCH", Path(tmp)/"scratch"):
            runner.main(["--preflight"])
            self.assertEqual(train.call_args.args[1], True)
            evaluate_mock.run.assert_not_called()
            self.assertTrue((Path(tmp)/"scratch/result.json").is_file())

    def test_full_dispatch_evaluates_once_after_training(self):
        events = []
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, "OUTPUT", Path(tmp)/"full"), \
                patch.object(runner, "train", side_effect=lambda *a: events.append("train")), \
                patch.object(runner.evaluate, "run", side_effect=lambda *a: events.append("eval")):
            runner.main([])
        self.assertEqual(events, ["train", "eval"])

    def test_training_failure_stops_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, "OUTPUT", Path(tmp)/"full"), \
                patch.object(runner, "train", side_effect=ValueError("failed gate")), \
                patch.object(runner.evaluate, "run") as run:
            with self.assertRaises(ValueError):
                runner.main([])
            run.assert_not_called()
            self.assertEqual(json.loads((Path(tmp)/"full/failure.json").read_text())["error"], "failed gate")

    def test_observation_whitelist_excludes_metadata(self):
        frame = {k: torch.zeros(shape) for k, shape in c.OBS.items()}
        frame.update(task=c.TASK, episode_index=torch.tensor(300), action=torch.zeros(50, 8),
                     action_is_pad=torch.zeros(50, dtype=torch.bool), initial_state="privileged")
        selected = c.PolicyRows([frame])[0]
        self.assertEqual(set(selected), set(c.OBS)|{"task", "action", "action_is_pad"})
        self.assertNotIn("action", c.h.observation_only(selected))
        frame["observation.objects"] = torch.zeros(3)
        with self.assertRaises(ValueError):
            c.PolicyRows([frame])[0]

    def test_observation_rejects_other_episode_or_task(self):
        frame = {k: torch.zeros(shape) for k, shape in c.OBS.items()}
        frame.update(task=c.TASK, episode_index=torch.tensor(9))
        with self.assertRaises(ValueError):
            c.PolicyRows([frame])[0]
        frame.update(episode_index=torch.tensor(0), task="wrong")
        with self.assertRaises(ValueError):
            c.PolicyRows([frame])[0]

    def checkpoint_fixture(self, out):
        checkpoint = out/"checkpoints/005000"
        checkpoint.mkdir(parents=True)
        (checkpoint/"model.safetensors").write_bytes(b"fake weights: no model loaded")
        config = dict(chunk_size=50, n_action_steps=10, n_obs_steps=1, type="smolvla",
                      input_features={k: {"shape": list(s)} for k, s in c.OBS.items()},
                      output_features={"action": {"shape": [8]}})
        for path in (checkpoint/"config.json", out/"initial_config/config.json"):
            path.parent.mkdir(exist_ok=True)
            c.h.write_json(path, config)
        marker = dict(status="complete", update=5000, reload_fp32_weights_exact=True,
                      reload_max_action_error=0., model_sha256=c.h.file_hash(checkpoint/"model.safetensors"))
        c.h.write_json(checkpoint/"checkpoint_result.json", marker)
        c.h.write_json(out/"training_stats.json", {})
        report = dict(status="complete", purpose="single_task", training_steps=5000,
                      completed_training_steps=5000, selected_checkpoint_update=5000,
                      final_data_binding_passed=True, dataset_root=str(c.DATA.resolve()),
                      dataset_fingerprint="dataset", timing={}, source_hashes=c.sources(),
                      train_episodes=list(c.TRAIN), validation_episodes=list(c.VAL),
                      training_frames=4752, valid_training_chunk_starts=4311,
                      checkpoints=[marker], final_checkpoint_files=c.inventory(checkpoint),
                      asset_files=c.inventory(c.ROOT/".cache/benchmark_models"),
                      training_stats_sha256=c.h.file_hash(out/"training_stats.json"))
        c.h.write_json(out/"result.json", report)
        return checkpoint, report

    def test_final_checkpoint_report_marker_binding(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, "sources", return_value={"code": "hash"}):
            out = Path(tmp)
            checkpoint, report = self.checkpoint_fixture(out)
            self.assertEqual(evaluate.verify_checkpoint(out, {"dataset_fingerprint": "dataset"}, {"timing": {}}), checkpoint)
            for key, value in (("completed_training_steps", 4999), ("selected_checkpoint_update", 1000),
                               ("final_data_binding_passed", False), ("source_hashes", {}),
                               ("train_episodes", [0, 300]), ("asset_files", {})):
                bad = {**report, key: value}
                c.h.write_json(out/"result.json", bad)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    evaluate.verify_checkpoint(out, {"dataset_fingerprint": "dataset"}, {"timing": {}})

    def test_final_checkpoint_processor_and_stats_bytes_bound(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, "sources", return_value={}):
            out = Path(tmp)
            checkpoint, _ = self.checkpoint_fixture(out)
            (checkpoint/"processor.json").write_text("changed")
            with self.assertRaises(ValueError):
                evaluate.verify_checkpoint(out, {"dataset_fingerprint": "dataset"}, {"timing": {}})

    def test_held_out_initial_adapter_is_exact_and_source_immutable(self):
        import mujoco
        from data_collection.benchmark import require_complete, restore_timing
        m = require_complete(c.DATA)
        model = mujoco.MjModel.from_xml_path(str(c.ROOT/m["scene_path"]))
        restore_timing(model, m)
        before = {e["file"]: c.h.file_hash(c.DATA/"replay"/e["file"]) for e in c.select_entries(m)}
        with tempfile.TemporaryDirectory() as tmp:
            for entry in c.select_entries(m):
                adapted, evidence = evaluate.derive_initial(model, c.DATA/"replay", entry, Path(tmp))
                evaluate.baseline.restore_scene(model, Path(tmp), adapted)
                self.assertEqual(evidence["source_sha256"], entry["sha256"])
                self.assertEqual(adapted["scene_hash"], entry["scene_hash"])
                self.assertEqual(adapted["episode_index"], entry["episode_index"])
                with self.assertRaises(FileExistsError):
                    evaluate.derive_initial(model, c.DATA/"replay", entry, Path(tmp))
            bad = copy.deepcopy(c.select_entries(m)[-1])
            for key, value in (("sha256", "0"*64), ("split", "train"), ("episode_index", 304),
                               ("scene_hash", "0"*64), ("file", "../escape.npz")):
                changed = {**bad, key: value}
                with self.subTest(key=key), self.assertRaises((ValueError, FileNotFoundError)):
                    evaluate.derive_initial(model, c.DATA/"replay", changed, Path(tmp))
        self.assertEqual(before, {e["file"]: c.h.file_hash(c.DATA/"replay"/e["file"]) for e in c.select_entries(m)})

    def exercise_evaluation(self, mutate=False):
        # Only model loading/inference and heavyweight gates are stubbed; real reset
        # adapter, output binding, dispatch, seed validation and accounting execute.
        calls = []
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                protected = out/"training.bin"
                protected.write_bytes(b"immutable")
                policy_stub = torch.nn.Linear(1, 1)
                policy_stub.config = None
                def trial(policy, pre, post, model, m, entry, initial, folder, ordinal):
                    calls.append(ordinal)
                    if mutate:
                        protected.write_bytes(b"changed")
                    return dict(entry, ordinal=ordinal, rollout_seed=20260916+ordinal,
                                task_success=True, outcome="success")
                with patch.object(c.h, "require_replay_pass", return_value={"dataset_fingerprint": "test"}), \
                        patch.object(evaluate, "verify_checkpoint", return_value=out/"checkpoints/005000"), \
                        patch.object(c.h, "load_fp32_checkpoint", return_value=policy_stub), \
                        patch("lerobot.policies.factory.make_pre_post_processors", return_value=(None, None)), \
                        patch.object(evaluate.baseline, "run_trial", side_effect=trial):
                    if mutate:
                        with self.assertRaisesRegex(ValueError, "Training bytes changed"):
                            evaluate.run(out)
                        self.assertEqual(json.loads((out/"evaluation/failure.json").read_text())["status"], "failed")
                    else:
                        # Config is used only by the processor factory stub.
                        evaluate.run(out)
                        result = json.loads((out/"evaluation/result.json").read_text())
                        self.assertTrue(result["training_bytes_unchanged"])
                        self.assertEqual(result["finished_trials"], 13)
                        self.assertEqual(protected.read_bytes(), b"immutable")
                self.assertEqual(calls, [i+1 for i in c.TRAIN+c.VAL])
        finally:
            torch.use_deterministic_algorithms(deterministic)

    def test_evaluation_exactly_thirteen_preserves_training_bytes(self):
        self.exercise_evaluation()

    def test_evaluation_detects_training_byte_mutation(self):
        self.exercise_evaluation(mutate=True)


if __name__ == "__main__":
    unittest.main()