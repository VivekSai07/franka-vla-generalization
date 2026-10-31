"""The tie-aware loader on small fixtures, and on the real checkpoint for each memory strategy's model structure."""
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from experiments.pi05_props_p2 import assets, weights

REAL = (assets.WEIGHTS_DIR/'model.safetensors').is_file()


class Tied(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed = torch.nn.Embedding(5, 3)
        self.model.head = torch.nn.Linear(3, 5, bias=False)
        self.model.head.weight = self.model.embed.weight
        self.model.norm = torch.nn.LayerNorm(3).to(torch.bfloat16)


def write(tmp, tensors):
    path = Path(tmp)/'w.safetensors'
    save_file(tensors, str(path))
    return path


def stored():
    return {'embed.weight': torch.arange(15, dtype=torch.float32).reshape(5, 3)/10,
            'norm.weight': torch.full((3,), 1.0009), 'norm.bias': torch.full((3,), 0.1234567)}


class LoaderTests(unittest.TestCase):
    def load(self, tensors, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Tied()
            return policy, weights.load_base_weights(policy, write(tmp, tensors), **kwargs)

    def test_a_tied_tensor_stored_once_loads_and_is_reported_as_an_alias(self):
        policy, (report, _) = self.load(stored())
        self.assertEqual(report['tied_aliases'], ['model.head.weight'])
        self.assertTrue(torch.equal(policy.model.embed.weight, stored()['embed.weight']))
        self.assertTrue(torch.equal(policy.model.head.weight, policy.model.embed.weight))

    def test_values_are_rounded_into_the_policy_dtype_and_the_masters_are_exact(self):
        policy, (report, masters) = self.load(stored(), keep_masters=True)
        self.assertEqual(policy.model.norm.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(policy.model.norm.weight, stored()['norm.weight'].to(torch.bfloat16)))
        self.assertEqual(set(masters), {'model.embed.weight', 'model.norm.weight', 'model.norm.bias'})
        self.assertTrue(torch.equal(masters['model.norm.bias'], stored()['norm.bias']))
        self.assertEqual(masters['model.norm.weight'].dtype, torch.float32)
        self.assertNotEqual(float(masters['model.norm.weight'][0]), float(policy.model.norm.weight[0]))
        self.assertEqual(report['trainable_masters'], 3)

    def test_masters_follow_requires_grad(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Tied()
            policy.model.norm.requires_grad_(False)
            _, masters = weights.load_base_weights(policy, write(tmp, stored()), keep_masters=True)
        self.assertEqual(set(masters), {'model.embed.weight'})

    def test_every_malformed_checkpoint_is_an_error(self):
        good = stored()
        cases = {
            'unexpected key': {**good, 'extra.weight': torch.zeros(2)},
            'missing key': {k: v for k, v in good.items() if k != 'norm.bias'},
            'shape': {**good, 'norm.bias': torch.zeros(4)},
            'dtype': {**good, 'norm.bias': good['norm.bias'].to(torch.bfloat16)},
            'renamed key': {**good, 'state_proj.weight': torch.zeros(2)},
        }
        for label, tensors in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                self.load(tensors)

    def test_two_stored_keys_for_one_tensor_are_an_error(self):
        with self.assertRaises(ValueError):
            self.load({**stored(), 'model.embed.weight': stored()['embed.weight']})

    def test_a_missing_tied_pair_is_an_error_only_if_neither_is_stored(self):
        with self.assertRaises(ValueError):
            self.load({k: v for k, v in stored().items() if k != 'embed.weight'})

    def test_the_installed_loaders_renames_are_refused_not_applied(self):
        for key in ('state_proj.weight', 'action_time_mlp_in.weight', 'action_time_mlp_out.bias'):
            with self.assertRaises(ValueError):
                weights.map_key(key)
        self.assertEqual(weights.map_key('paligemma_with_expert.x'), 'model.paligemma_with_expert.x')
        self.assertEqual(weights.map_key('model.x'), 'model.x')


@unittest.skipUnless(REAL, 'pi05 weights are not mounted')
class RealCheckpointTests(unittest.TestCase):
    """Builds the real 3.6 B policy on the CPU once per strategy (about 1 min and 15 GB each)."""

    def build(self, name):
        from experiments.pi05_props_p2 import strategy
        return strategy.build_policy(name)

    def test_fp32_policy_loads_every_tensor_with_one_tie(self):
        from experiments.pi05_props_p2 import contract as c
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from safetensors import safe_open
        policy = PI05Policy(c.make_config('cpu'))
        report, _ = weights.load_base_weights(policy)
        self.assertEqual(report['stored_tensors'], 812)
        self.assertEqual(report['tied_aliases'], ['model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'])
        state = policy.state_dict()
        with safe_open(str(assets.WEIGHTS_DIR/'model.safetensors'), framework='pt', device='cpu') as f:
            for key in f.keys():
                self.assertTrue(torch.equal(f.get_tensor(key), state[weights.map_key(key)]), key)

    def test_bf16_cpu_master_holds_exact_fp32_masters_and_a_rounded_policy(self):
        from safetensors import safe_open
        policy, config, masters, report = self.build('bf16_cpu_master')
        self.assertEqual(len(masters), 812)
        self.assertEqual(sum(m.numel() for m in masters.values()), 3_616_757_520)
        self.assertEqual({m.dtype for m in masters.values()}, {torch.float32})
        self.assertTrue(config.gradient_checkpointing)
        with safe_open(str(assets.WEIGHTS_DIR/'model.safetensors'), framework='pt', device='cpu') as f:
            key = 'paligemma_with_expert.gemma_expert.model.layers.3.mlp.up_proj.weight'
            exact = f.get_tensor(key)
        name = 'model.'+key
        self.assertTrue(torch.equal(masters[name], exact))
        parameter = dict(policy.named_parameters())[name]
        self.assertEqual(parameter.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(parameter, exact.to(torch.bfloat16)))
        kept = {n: p.dtype for n, p in policy.named_parameters() if p.dtype == torch.float32}
        self.assertTrue(any('input_layernorm' in n for n in kept))
        self.assertTrue(any('action_in_proj' in n for n in kept))

    def test_expert_only_freezes_the_paligemma_side_and_keeps_the_expert_fp32(self):
        policy, config, masters, report = self.build('expert_only')
        self.assertEqual(masters, {})
        trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
        self.assertTrue(all('gemma_expert' in n or '.action_' in n or '.time_mlp_' in n for n in trainable))
        self.assertEqual({p.dtype for p in trainable.values()}, {torch.float32})
        frozen = [p for n, p in policy.named_parameters() if not p.requires_grad]
        self.assertTrue(all('paligemma.' in n for n, p in policy.named_parameters() if not p.requires_grad))
        self.assertIn(torch.bfloat16, {p.dtype for p in frozen})
        count = sum(p.numel() for p in trainable.values())
        self.assertEqual(count, 691_256_320+2_165_792)  # expert + action_in/out + time_mlp_in/out
        self.assertFalse(config.gradient_checkpointing)
        key = 'paligemma_with_expert.gemma_expert.model.layers.3.mlp.up_proj.weight'
        from safetensors import safe_open
        with safe_open(str(assets.WEIGHTS_DIR/'model.safetensors'), framework='pt', device='cpu') as f:
            self.assertTrue(torch.equal(trainable['model.'+key], f.get_tensor(key)))


if __name__ == '__main__':
    unittest.main()
