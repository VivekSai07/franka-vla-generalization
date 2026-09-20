"""No physics: seeds, probe leakage/RNG, AST scoring and final-only gate."""
import ast
import importlib
import inspect
import json
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from test_contract import frame


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).parents[1]/'evaluate.py').exists(), 'new evaluator missing')
        self.e = importlib.import_module('experiments.smolvla_fixed_layout.evaluate')

    def test_seeds_are_genuine_explicit_and_reset_identity_remains_seven(self):
        plan = self.e.evaluation_plan()
        self.assertEqual([r['seed'] for r in plan], list(range(20260918, 20260938)))
        self.assertEqual({r['source_episode_index'] for r in plan}, {7})
        self.assertIn('seed', inspect.signature(self.e.run_trial).parameters)
        draws = []
        for row in plan:
            self.e.seed_trial(row['seed']); draws.append((random.random(), np.random.rand(), torch.rand(()).item()))
        self.assertEqual(len(set(draws)), 20)
        self.e.seed_trial(20260918)
        self.assertEqual(draws[0], (random.random(), np.random.rand(), torch.rand(()).item()))
        with self.assertRaises(ValueError): self.e.seed_trial(7)

    def test_gate_requires_all_twenty_and_eighteen_successes(self):
        rows = [dict(seed=20260918+i, task_success=i<18, outcome='success' if i<18 else 'timeout') for i in range(20)]
        self.assertTrue(self.e.summarize(rows)['capability_passed'])
        for bad in (rows[:18], rows[:19], rows[:-1]+[rows[0]], rows+[rows[0]],
                    [{**r, 'outcome': 'nonfinite_action'} for r in rows],
                    [{**r, 'value': float('nan')} for r in rows]):
            self.assertFalse(self.e.summarize(bad)['capability_passed'])
            with self.assertRaises(ValueError): self.e.require_complete_trials(bad)
        rows[17].update(task_success=False, outcome='timeout')
        self.assertFalse(self.e.summarize(rows)['capability_passed'])

    def test_72_probes_observation_only_physical_metrics_rng_and_mode(self):
        class Dataset:
            def __getitem__(self, i): return {**frame(i//528), 'success': True, 'objects': [1, 2, 3]}
        class Tiny(torch.nn.Module):
            calls = 0
            def predict_action_chunk(self, batch):
                self.calls += 1
                if set(batch) != {'observation.state', 'observation.images.image', 'observation.images.image2', 'task'}:
                    raise AssertionError('prediction input leakage')
                random.random(); np.random.rand(); torch.rand(())
                return torch.ones(1, 50, 8)
        model = Tiny(); model.train()
        py, npstate, tensor = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        result = self.e.offline_probes(model, lambda x:x, lambda x:x, Dataset(), device='cpu')
        self.assertEqual(model.calls, 72)
        self.assertEqual((result['train']['probes'], result['validation']['probes']), (60, 12))
        self.assertEqual({r['frame'] for r in result['rows']}, {60, 107, 347})
        self.assertEqual(result['train']['arm_mae_rad'], 1.)
        self.assertEqual(result['validation']['first10_arm_mae_rad'], 1.)
        self.assertEqual(result['validation']['gripper_binary_accuracy'], 1.)
        self.assertTrue(model.training)
        self.assertEqual(py, random.getstate()); np.testing.assert_array_equal(npstate[1], np.random.get_state()[1])
        self.assertTrue(torch.equal(tensor, torch.get_rng_state()))

    def test_nonfinite_probe_restores_rng_on_failure(self):
        class Dataset:
            def __getitem__(self, i): return frame(i//528)
        class Tiny(torch.nn.Module):
            def predict_action_chunk(self, batch):
                torch.rand(()); return torch.full((1, 50, 8), float('nan'))
        model = Tiny(); before = torch.get_rng_state().clone()
        with self.assertRaises(ValueError): self.e.offline_probes(model, lambda x:x, lambda x:x, Dataset(), device='cpu')
        self.assertTrue(model.training); self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_scoring_action_timing_ast_unchanged_no_xvla_autocast(self):
        from validation.evaluate_pilot import run_trial as old
        def physics(function):
            loop = next(n for n in ast.walk(ast.parse(inspect.getsource(function)))
                        if isinstance(n, ast.For) and ast.unparse(n.iter) == 'range(1200)')
            start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                         and ast.unparse(n.targets[0]) == '(bounded, clips)')
            return [ast.dump(n) for n in loop.body[start:]]
        self.assertEqual(physics(old), physics(self.e.run_trial))
        source = inspect.getsource(self.e.run_trial)
        self.assertNotIn('autocast', source)
        self.assertNotIn('policy._get_action_chunk =', source)
        self.assertNotIn('check_model_inputs', source)
        self.assertIn('seed_trial(seed)', source)

    def test_stages_separate_closure_contact_lift_and_release(self):
        actions = np.zeros((40, 8)); actions[35:, 7] = 255
        result = self.e.stage_metrics(dict(applied_action=actions, bilateral_target_contact=np.arange(40)>=5,
                                          pick_lift_m=np.arange(40)*.01, released_stack=np.arange(40)>=36))
        self.assertEqual(result['first_closure_tick'], 0)
        self.assertEqual(result['sustained_closure_30_tick'], 29)
        self.assertEqual(result['first_bilateral_contact_tick'], 5)
        self.assertEqual(result['first_contact_lift_gt_5cm_tick'], 6)
        self.assertEqual(result['first_release_after_closure_tick'], 35)
        self.assertIsNone(result['stable_stack_30_tick'])

    def test_counted_loader_promotes_fp32_before_copy(self):
        from types import SimpleNamespace
        from safetensors.torch import save_model
        class Tiny(torch.nn.Module):
            def __init__(self, config=None):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        with tempfile.TemporaryDirectory() as tmp:
            model = Tiny(); self.e.c.h.promote_trainable_parameters(model)
            with torch.no_grad(): model.weight.fill_(1.000123)
            save_model(model, str(Path(tmp)/'model.safetensors'))
            with patch.object(self.e, 'CountedPolicy', Tiny), \
                    patch('lerobot.configs.policies.PreTrainedConfig.from_pretrained', return_value=SimpleNamespace(device='cpu')):
                restored = self.e.load_counted(Path(tmp))
            self.assertEqual(restored.weight.dtype, torch.float32)
            self.assertTrue(torch.equal(restored.weight, model.weight))

    def test_reset_entry_hash_identity_and_no_inference_ground_truth(self):
        c = self.e.c
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); (parent/'reset.npz').write_bytes(b'synthetic')
            binding = {'reset_sha256': c.sha256(parent/'reset.npz')}
            with patch.object(c, 'PARENT', parent):
                self.assertEqual(self.e.reset_entry(binding)['episode_index'], 7)
                (parent/'reset.npz').write_bytes(b'changed')
                with self.assertRaises(ValueError): self.e.reset_entry(binding)


if __name__ == '__main__': unittest.main()