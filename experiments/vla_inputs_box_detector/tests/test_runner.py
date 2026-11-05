"""The detector's training run on a tiny synthetic cache (CPU, a small network), the learning-rate schedule, the checkpoint choice, the saved
detector's round trip through `Predictor`, the guards, and the binding against the real seals where mounted."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.vla_inputs_box_detector import constants as k
from experiments.vla_inputs_box_detector import contract, data, gate, inputs, model, predictor, runner

import test_data  # noqa: E402

ROWS = [(e, f) for e in (0, 1, 2, 3) for f in (0, 8)] + [(e, f) for e in (480, 481) for f in (0, 8)]
SMALL = dict(widths=(8, 16, 32, 64), head_channels=16)


def synthetic_cache(directory):
    test_data.OBJECT_OF_EPISODE.update({2: 1, 3: 4})
    test_data.synthetic(directory, rows=ROWS)
    return data.open_cache(directory)


class Schedule(unittest.TestCase):
    def test_warmup_then_cosine_to_zero(self):
        steps, epochs = 10, 5
        rates = [runner.learning_rate(s, steps, epochs) for s in range(steps * epochs)]
        self.assertAlmostEqual(rates[0], k.LEARNING_RATE / 10)
        self.assertAlmostEqual(rates[steps - 1], k.LEARNING_RATE)
        self.assertTrue(all(a >= b for a, b in zip(rates[steps:], rates[steps + 1:])))
        self.assertLess(rates[-1], k.LEARNING_RATE * 0.01)

    def test_the_score_is_the_mean_of_the_two_cameras_mean_iou(self):
        self.assertAlmostEqual(runner.validation_score(dict(front=dict(mean_iou=0.8), gripper=dict(mean_iou=0.9))), 0.85)


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cache = synthetic_cache(Path(cls.tmp.name) / "cache")
        cls.run_dir = Path(cls.tmp.name) / "run"
        cls.messages = []
        cls.final = runner.train(cls.cache, cls.run_dir, epochs=3, batch=4, binding="b" * 64, log=lambda *a, **kw: cls.messages.append(a), **SMALL)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_it_writes_the_history_and_a_final_directory_and_the_run_is_never_reused(self):
        history = json.loads((self.run_dir / "history.json").read_text())
        self.assertEqual([h["epoch"] for h in history], [1, 2, 3])
        self.assertTrue(all(np.isfinite(h["train_loss"]) for h in history))
        self.assertEqual(sorted(p.name for p in self.final.iterdir()), [predictor.DESCRIPTION, predictor.WEIGHTS])
        before = (self.run_dir / "history.json").read_bytes()
        with self.assertRaises(FileExistsError):
            runner.train(self.cache, self.run_dir, epochs=1, batch=4, **SMALL)
        self.assertEqual((self.run_dir / "history.json").read_bytes(), before)  # nothing of the existing run was touched

    def test_the_final_is_the_epoch_with_the_best_validation_iou_not_the_last(self):
        history = json.loads((self.run_dir / "history.json").read_text())
        description = json.loads((self.final / predictor.DESCRIPTION).read_text())
        best = max(history, key=lambda h: h["validation_score"])
        self.assertEqual(description["epoch"], best["epoch"])
        self.assertAlmostEqual(description["validation_score"], best["validation_score"])
        self.assertEqual(description["binding_sha256"], "b" * 64)
        self.assertEqual(description["constants_sha256"], predictor.constants_digest())
        self.assertIn("never by rollouts", description["chosen_by"])

    def test_the_best_epoch_is_chosen_even_when_it_is_not_the_last(self):
        scores = iter([0.2, 0.9, 0.5])
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runner, "validation_score", side_effect=lambda report: next(scores)):
                final = runner.train(self.cache, Path(tmp) / "run", epochs=3, batch=4, log=lambda *a, **kw: None, **SMALL)
            self.assertEqual(json.loads((final / predictor.DESCRIPTION).read_text())["epoch"], 2)

    def test_the_saved_detector_loads_and_predicts_a_box_pair_through_the_rollout_path(self):
        loaded = predictor.Predictor.load(self.final)
        renders = [np.zeros((512, 512, 3), np.uint8), np.zeros((512, 512, 3), np.uint8)]
        out = loaded.predict(renders, "put the can in the blue bin")
        self.assertEqual((out.shape, out.dtype), ((2, 5), np.float64))
        self.assertTrue(np.isin(out[:, 4], (0.0, 1.0)).all())
        for box in out:
            if box[4] == 1.0:
                self.assertTrue((box[:4] >= 0).all() and (box[:4] <= 1).all() and box[2] > box[0] and box[3] > box[1])
        with self.assertRaises(ValueError):
            loaded.predict(renders, "put the cube in the blue bin")

    def test_a_detector_trained_under_other_constants_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "final"
            copy.mkdir()
            for name in (predictor.WEIGHTS, predictor.DESCRIPTION):
                (copy / name).write_bytes((self.final / name).read_bytes())
            description = json.loads((copy / predictor.DESCRIPTION).read_text())
            description["constants_sha256"] = "0" * 64
            (copy / predictor.DESCRIPTION).write_text(json.dumps(description))
            with self.assertRaisesRegex(ValueError, "constants"):
                predictor.Predictor.load(copy)

    def test_validation_episodes_are_never_trained_on(self):
        seen = []
        original = data.TrainingSet.__getitem__

        def spy(self, i):
            sample = original(self, i)
            seen.append(int(self.cache.index[sample["row"], 0]))
            return sample

        with tempfile.TemporaryDirectory() as tmp, patch.object(data.TrainingSet, "__getitem__", spy):
            runner.train(self.cache, Path(tmp) / "run", epochs=2, batch=4, log=lambda *a, **kw: None, **SMALL)
        self.assertTrue(seen)
        self.assertTrue(all(e < 480 for e in seen))

    def test_the_predictor_reads_the_same_decoding_the_gate_measures(self):
        loaded = predictor.Predictor.load(self.final)
        rows = data.rows_of(self.cache, "validation")
        report = gate.decoded_report(loaded, self.cache, rows)
        for name in gate.CAMERAS:
            self.assertEqual(report[name]["frames"], len(rows))


class Guards(unittest.TestCase):
    def test_the_disk_guard_refuses_below_the_pinned_free_space(self):
        with patch.object(runner.shutil, "disk_usage", return_value=type("U", (), {"free": 65 * 2**30 - 1})()):
            with self.assertRaisesRegex(RuntimeError, "refusing"):
                runner.disk_guard("/")
        with patch.object(runner.shutil, "disk_usage", return_value=type("U", (), {"free": 65 * 2**30})()):
            runner.disk_guard("/")

    def test_an_admission_for_another_binding_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "manifest.json").write_text("{}")
            (Path(tmp) / "result.json").write_text(json.dumps(dict(binding_sha256="a" * 64, cache_manifest_sha256=contract.file_sha256(Path(tmp) / "manifest.json"))))
            with patch.object(contract, "binding_sha256", return_value="b" * 64):
                with self.assertRaisesRegex(ValueError, "binding changed"):
                    runner.require_admission(Path(tmp), Path(tmp))

    def test_training_needs_a_current_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                runner.require_admission(Path(tmp) / "none", Path(tmp))


def seals_mounted():
    try:
        from experiments.xvla_p2_force_box import contract as p2
        return (p2.PARENT / "seal.json").is_file() and (p2.forces.SIDECAR / "seal.json").is_file()
    except Exception:
        return False


class Source(unittest.TestCase):
    def test_source_files_are_exactly_the_packages_non_test_files(self):
        package = Path(contract.__file__).parent
        actual = sorted(f"{contract.PACKAGE}/{p.name}" for p in package.iterdir() if p.is_file() and p.suffix in (".py", ".md"))
        self.assertEqual(sorted(contract.SOURCE_FILES), actual)

    def test_the_readme_is_bound_and_not_empty(self):
        self.assertIn(f"{contract.PACKAGE}/README.md", contract.SOURCE_FILES)
        self.assertGreater(len((Path(contract.__file__).parent / "README.md").read_text()), 2000)

    def test_no_function_reads_a_name_that_is_never_bound(self):
        """The symtable scan of the forking skill: a name read in a function that is bound nowhere in the module is a NameError waiting for its branch."""
        import builtins
        import symtable
        for name in ("constants", "contract", "data", "gate", "inputs", "model", "predictor", "runner"):
            path = Path(contract.__file__).parent / f"{name}.py"
            top = symtable.symtable(path.read_text(), str(path), "exec")
            module_names = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}

            def walk(table):
                for symbol in table.get_symbols():
                    if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in module_names and not hasattr(builtins, symbol.get_name()) and not symbol.get_name().startswith('__'):
                        yield table.get_name(), symbol.get_name()
                for child in table.get_children():
                    yield from walk(child)

            self.assertEqual(list(walk(top)), [], name)


@unittest.skipUnless(seals_mounted(), "the sealed P2 parent and S0 sidecar are not mounted")
class Binding(unittest.TestCase):
    def test_the_binding_names_the_seals_the_split_the_constants_and_the_sources(self):
        binding = contract.binding()
        self.assertEqual(binding["p2_seal"], "b1d642b3edeebe15d462709ebeb8b92afa6d7eabcca3be5514e5deeecaf4d898")
        self.assertEqual(binding["boxes"]["target_box_sha256"], "d168977668b8d3bb61acdc1aac1216564c5707c12b674740360fe82454a2a1d7")
        self.assertEqual(binding["constants_sha256"], predictor.constants_digest())
        self.assertEqual(sorted(binding["sources"]), sorted(contract.SOURCE_FILES))
        self.assertEqual(binding["s0"]["seal"]["fingerprint"], "7e2e06db6f71883992cb0e66375e2da899ada4297359d87bd0e2a2f278f58e6c")
        self.assertEqual(contract.binding_sha256(), contract.binding_sha256())

    def test_the_plan_split_is_the_detectors(self):
        from experiments.xvla_p2_force_box import contract as p2
        plan = json.loads((p2.PARENT / "plan.json").read_text())
        contract.check_split(plan)
        bad = dict(plan, validation_ids=list(plan["validation_ids"])[:-1])
        with self.assertRaises(ValueError):
            contract.check_split(bad)

    def test_a_changed_constant_changes_the_digest(self):
        before = predictor.constants_digest()
        with patch.object(k, "STRIDE_K", 4):
            self.assertNotEqual(predictor.constants_digest(), before)

    def test_the_admission_derives_from_a_cache_and_a_changed_cache_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            data.extract(cache_dir, episodes=(0, 480), workers=1, free_bytes=lambda p: 10**15)
            record = runner.admission_result(cache_dir, expect_episodes=2)
            self.assertEqual(record["rows"], 2 * 57)
            self.assertLessEqual(record["parameters"], k.MAX_PARAMETERS)
            self.assertTrue(np.isfinite(record["batch_loss"]))
            admission = Path(tmp) / "admission"
            admission.mkdir()
            (admission / "result.json").write_text(json.dumps(record))
            self.assertEqual(runner.require_admission(admission, cache_dir)["binding_sha256"], record["binding_sha256"])
            with self.assertRaisesRegex(ValueError, "does not hold every"):
                runner.admission_result(cache_dir, expect_episodes=540)
            (cache_dir / "manifest.json").write_text((cache_dir / "manifest.json").read_text() + " ")
            with self.assertRaises(ValueError):
                runner.require_admission(admission, cache_dir)


if __name__ == "__main__":
    unittest.main()
