"""The three memory strategies on a tiny CPU model: update math, dtypes, clipping, snapshots and resume.

The real 3.6 B-parameter model is covered by `test_weights.py` (structure of each strategy's model) and, on the GPU, by
the runner's `--probe-memory`.
"""
import unittest

import torch

from experiments.pi05_props_p2 import contract as c
from experiments.pi05_props_p2 import strategy as s


class Tiny(torch.nn.Module):
    """A bf16 linear, an FP32 norm and an FP32 head: the dtype mix `dtype='bfloat16'` gives the real policy."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = torch.nn.Linear(16, 16).to(torch.bfloat16)
        self.norm = torch.nn.LayerNorm(16)
        self.head = torch.nn.Linear(16, 4)
        self.frozen = torch.nn.Linear(4, 4)
        for p in self.frozen.parameters():
            p.requires_grad = False

    def forward(self, x):
        h = self.norm(self.lin(x.to(torch.bfloat16)).float())
        return self.frozen(self.head(h)).pow(2).mean()


def masters_of(model):
    return {n: p.detach().float().clone() for n, p in model.named_parameters() if p.requires_grad}


def batches(n=6):
    g = torch.Generator().manual_seed(7)
    return [torch.randn(8, 16, generator=g) for _ in range(n)]


def run(strategy, model, updates, start=1, lr=None):
    rows = []
    for i, x in zip(range(start, start+updates), batches(start+updates-1)[start-1:]):
        strategy.zero_grad()
        model(x).backward()
        rows.append(strategy.step(i, lr))
    return rows


class UpdateMathTests(unittest.TestCase):
    def test_cpu_master_equals_an_fp32_reference_adamw_and_the_gpu_copy_is_its_rounding(self):
        model = Tiny()
        strategy = s.CpuMasterAdam(model, masters_of(model), fused=False)
        reference = {n: p.clone().requires_grad_() for n, p in masters_of(Tiny()).items()}
        optimizer = torch.optim.AdamW(list(reference.values()), lr=c.learning_rate(1), betas=s.BETAS, eps=s.EPS,
                                      weight_decay=s.WEIGHT_DECAY, foreach=False)
        for i, x in enumerate(batches(), 1):
            strategy.zero_grad()
            model(x).backward()
            grads = {n: p.grad.float().clone() for n, p in strategy.params.items()}
            norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads.values()]))
            scale = min(1.0, s.CLIP/(float(norm)+1e-6))
            for n, p in reference.items():
                p.grad = grads[n]*scale
            c.set_learning_rate(optimizer, i)
            optimizer.step()
            strategy.step(i)
            for n, p in reference.items():
                torch.testing.assert_close(strategy.masters[n].detach(), p.detach(), rtol=1e-5, atol=1e-8)
                self.assertTrue(torch.equal(strategy.params[n].detach(), p.detach().to(strategy.params[n].dtype)))

    def test_bf16_all_loses_updates_that_cpu_master_keeps(self):
        weights = {}
        for name in ('bf16_all', 'bf16_cpu_master'):
            model = Tiny()
            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            strategy = s.make_strategy(name, model, masters_of(model), 'cpu')
            for update in (1, 2):
                strategy.zero_grad()
                model(batches()[0]).backward()
                strategy.step(update, lr=2.5e-6)
            weights[name] = (before, strategy)
        before, all_bf16 = weights['bf16_all']
        changed = s.changed_fractions(before, {n: p.detach() for n, p in all_bf16.params.items()})
        bf16_row = next(r for r in changed if r['name'] == 'lin.weight')
        before, master = weights['bf16_cpu_master']
        moved = (master.masters['lin.weight'].detach() != masters_of(Tiny())['lin.weight']).float().mean()
        self.assertEqual(bf16_row['dtype'], 'torch.bfloat16')
        self.assertLess(bf16_row['fraction_changed'], 0.5)
        self.assertEqual(float(moved), 1.0)

    def test_gradient_clipping_matches_the_reference_and_does_not_touch_small_gradients(self):
        for scale, expect_clipped in ((100.0, True), (1e-3, False)):
            model = Tiny()
            strategy = s.GpuAdam(model, 'bf16_all', 'cpu')
            (model(batches()[0])*scale).backward()
            raw = [p.grad.float().clone() for p in strategy.params.values()]
            norm = s.clip_gradients(list(strategy.params.values()))
            reference = float(torch.linalg.vector_norm(torch.stack([g.norm() for g in raw])))
            self.assertAlmostEqual(norm, reference, delta=1e-3*reference)
            after = float(torch.linalg.vector_norm(torch.stack(
                [p.grad.float().norm() for p in strategy.params.values()])))
            if expect_clipped:
                self.assertAlmostEqual(after, 1.0, delta=1e-2)
            else:
                self.assertAlmostEqual(after, reference, delta=1e-2*reference)

    def test_a_missing_gradient_or_a_nonfinite_norm_is_an_error(self):
        model = Tiny()
        strategy = s.GpuAdam(model, 'bf16_all', 'cpu')
        with self.assertRaises(ValueError):
            s.clip_gradients(list(strategy.params.values()))
        model(batches()[0]).backward()
        next(iter(strategy.params.values())).grad.fill_(float('inf'))
        with self.assertRaises(ValueError):
            s.clip_gradients(list(strategy.params.values()))


class StructureTests(unittest.TestCase):
    def test_dtype_reports(self):
        model = Tiny()
        a = s.GpuAdam(model, 'bf16_all', 'cpu')
        run(a, model, 1)
        report = a.dtype_report()
        self.assertEqual(set(report['parameters']), {'torch.bfloat16', 'torch.float32'})
        self.assertEqual(report['moments'], {k: 2*v for k, v in report['parameters'].items()})
        self.assertEqual(report['gradients'], report['parameters'])
        model = Tiny()
        b = s.CpuMasterAdam(model, masters_of(model), fused=False)
        run(b, model, 1)
        report = b.dtype_report()
        self.assertEqual(set(report['parameters']), {'torch.bfloat16', 'torch.float32'})
        self.assertEqual(set(report['masters']), {'torch.float32'})
        self.assertEqual(set(report['moments']), {'torch.float32'})
        self.assertEqual(set(report['gradients']), {'torch.float32'})
        self.assertEqual(set(report['gpu_gradients']), {'torch.bfloat16', 'torch.float32'})

    def test_masters_must_match_the_trainable_parameters_exactly(self):
        model = Tiny()
        good = masters_of(model)
        for bad in ({k: v for k, v in good.items() if k != 'lin.weight'}, {**good, 'frozen.weight': torch.zeros(4, 4)},
                    {**good, 'lin.weight': good['lin.weight'].bfloat16()},
                    {**good, 'lin.weight': torch.zeros(3, 3)}):
            with self.assertRaises(ValueError):
                s.CpuMasterAdam(model, bad, fused=False)

    def test_the_frozen_tensor_is_neither_trained_nor_moved(self):
        model = Tiny()
        before = model.frozen.weight.detach().clone()
        for name in ('bf16_all', 'bf16_cpu_master'):
            model = Tiny()
            strategy = s.make_strategy(name, model, masters_of(model), 'cpu')
            run(strategy, model, 2)
            self.assertNotIn('frozen.weight', strategy.params)
            self.assertTrue(torch.equal(model.frozen.weight.detach(), before))

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            s.make_strategy('fp32_everything', Tiny(), {}, 'cpu')

    def test_every_strategy_named_in_the_contract_is_buildable_and_nothing_else(self):
        self.assertEqual(set(c.STRATEGIES), {'bf16_all', 'bf16_cpu_master', 'expert_only'})
        for name in ('bf16_all', 'bf16_cpu_master'):
            model = Tiny()
            self.assertEqual(s.make_strategy(name, model, masters_of(model), 'cpu').name, name)

    def test_change_fractions(self):
        before = {'a': torch.zeros(10), 'b': torch.zeros(4)}
        after = {'a': torch.tensor([1.] * 3 + [0.] * 7), 'b': torch.zeros(4)}
        rows = s.changed_fractions(before, after)
        self.assertEqual([r['fraction_changed'] for r in rows], [0.3, 0.0])
        summary = s.summarize_changes(rows)
        self.assertAlmostEqual(summary['element_weighted'], 3/14)
        self.assertEqual(summary['tensors_with_no_change'], 1)


class SnapshotTests(unittest.TestCase):
    def check_resume(self, name):
        model = Tiny()
        full = s.make_strategy(name, model, masters_of(model), 'cpu')
        run(full, model, 2, lr=1e-2)
        model_state = {k: v.clone() for k, v in full.model_state().items()}
        optimizer_state = __import__('copy').deepcopy(full.optimizer_state())
        run(full, model, 2, start=3, lr=1e-2)
        resumed_model = Tiny()
        with torch.no_grad():
            for p in resumed_model.parameters():
                if p.requires_grad:
                    p.add_(1.0)
        resumed = s.make_strategy(name, resumed_model, masters_of(resumed_model), 'cpu')
        resumed.restore(model_state, optimizer_state)
        run(resumed, resumed_model, 2, start=3, lr=1e-2)
        moved = 0
        for (n, a), (_, b) in zip(model.named_parameters(), resumed_model.named_parameters()):
            self.assertTrue(torch.equal(a, b), n)
            moved += int(not torch.equal(a, Tiny().state_dict()[n]))
        self.assertGreater(moved, 0)

    def test_bf16_all_resumes_bit_exactly(self):
        self.check_resume('bf16_all')

    def test_cpu_master_resumes_bit_exactly(self):
        self.check_resume('bf16_cpu_master')


if __name__ == '__main__':
    unittest.main()
