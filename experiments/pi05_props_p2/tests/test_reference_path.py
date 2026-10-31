"""Task 4b, CPU parts: the runner's update against a reference, and the real batch through the installed policy.

* `OneUpdateTests`: gradient accumulation over microbatches equals one batch of 8 (tiny model, deterministic loss).
* `RealBatchTests`: real rows through the real processors and the real 3.6 B policy on the CPU: padding, real action
  dimensions, camera order, prompt length, the physical-action round trip, and the runner's loss and gradients against a
  hand-written reference that draws its noise and time explicitly. Needs the assets and the sealed P2 root and about
  50 GB of RAM and several minutes; skipped without them.
The optimizer update against an FP32 reference is in `test_strategy.py`; on the real model it is measured by the GPU probe.
"""
import unittest

import torch

from experiments.pi05_props_p2 import assets, contract as c, runner as r, strategy as st, weights

HAVE_REAL = ((assets.WEIGHTS_DIR/'model.safetensors').is_file() and (assets.TOKENIZER_DIR/'tokenizer.json').is_file()
             and (c.DATA/'meta/info.json').is_file())


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.net = torch.nn.Linear(6, 3)

    def forward(self, batch):
        loss = (self.net(batch['x'])-batch['action'][:, 0, :3]).pow(2).mean()
        return loss, {}


class ListFetch:
    def __init__(self, raws):
        self.items = iter([(list(range(8)), raws)])

    def __next__(self):
        return next(self.items)


def identity_pre(raw):
    return {'x': raw['x'], 'action': raw['action']}


class OneUpdateTests(unittest.TestCase):
    def test_accumulation_over_microbatches_equals_one_batch_of_eight(self):
        g = torch.Generator().manual_seed(11)
        x, action = torch.randn(8, 6, generator=g), torch.randn(8, 50, 8, generator=g)
        results = {}
        for microbatch in (8, 4, 2, 1):
            policy = TinyPolicy()
            strategy = st.GpuAdam(policy, 'bf16_all', 'cpu')
            raws = [{'x': x[k:k+microbatch], 'action': action[k:k+microbatch]} for k in range(0, 8, microbatch)]
            row = r.one_update(1, strategy, policy, identity_pre, ListFetch(raws), microbatch, lr=1e-3)
            results[microbatch] = (row, {n: p.detach().clone() for n, p in policy.named_parameters()})
        reference_row, reference = results[8]
        for microbatch, (row, params) in results.items():
            self.assertAlmostEqual(row['loss'], reference_row['loss'], places=6, msg=microbatch)
            self.assertAlmostEqual(row['grad_norm'], reference_row['grad_norm'], places=5, msg=microbatch)
            for name, value in params.items():
                torch.testing.assert_close(value, reference[name], rtol=1e-5, atol=1e-7)

    def test_the_wrong_number_of_microbatches_or_shapes_is_an_error(self):
        policy = TinyPolicy()
        strategy = st.GpuAdam(policy, 'bf16_all', 'cpu')
        raw = {'x': torch.zeros(4, 6), 'action': torch.zeros(4, 50, 8)}
        with self.assertRaisesRegex(ValueError, 'count'):
            r.one_update(1, strategy, policy, identity_pre, ListFetch([raw]), 4)
        bad = {'x': torch.zeros(4, 6), 'action': torch.zeros(4, 30, 8)}
        with self.assertRaisesRegex(ValueError, 'shape'):
            r.one_update(1, strategy, policy, identity_pre, ListFetch([bad, bad]), 4)


@unittest.skipUnless(HAVE_REAL, 'assets or the sealed P2 root are not mounted')
class RealBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        cls.ds, cls.starts, cls.stats = c.load_data()
        cls.rows = [cls.starts[0], cls.starts[403*100+17]]
        cls.raw = c.collate_rows([cls.ds[i] for i in cls.rows])
        cls.config = c.make_config('cpu')
        cls.pre, cls.post = c.make_processors(cls.config, cls.stats)
        cls.batch = cls.pre(cls.raw)
        cls.policy = PI05Policy(cls.config)
        weights.load_base_weights(cls.policy)
        cls.policy.train()

    def test_batch_layout_padding_and_real_action_dimensions(self):
        b, p = self.batch, self.policy
        self.assertEqual(tuple(b['action'].shape), (2, 50, 8))
        self.assertEqual(tuple(b['observation.state'].shape), (2, 9))
        self.assertEqual(tuple(b['observation.language.tokens'].shape), (2, 200))
        padded = p.prepare_action(b)
        self.assertEqual(tuple(padded.shape), (2, 50, 32))
        self.assertTrue(torch.equal(padded[..., :8], b['action']))
        self.assertTrue((padded[..., 8:] == 0).all())
        self.assertEqual(p.config.output_features['action'].shape[0], 8)

    def test_camera_order_follows_the_datasets_and_each_image_reaches_its_slot(self):
        p = self.policy
        self.assertEqual(list(p.config.image_features), list(c.CAMERAS))
        images, masks = p._preprocess_images(self.batch)
        self.assertEqual([tuple(i.shape) for i in images], [(2, 3, 224, 224)]*2)
        self.assertTrue(all(bool(m.all()) for m in masks))
        changed = dict(self.batch)
        changed[c.CAMERAS[1]] = torch.zeros_like(self.batch[c.CAMERAS[1]])
        other, _ = p._preprocess_images(changed)
        self.assertTrue(torch.equal(other[0], images[0]))
        self.assertFalse(torch.equal(other[1], images[1]))
        self.assertTrue(torch.isfinite(images[0]).all() and images[0].min() >= -1 and images[0].max() <= 1)

    def test_the_prompt_keeps_the_instruction_and_the_physical_action_round_trips(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(assets.TOKENIZER_DIR))
        tokens, masks = self.batch['observation.language.tokens'], self.batch['observation.language.attention_mask']
        for text, t, m in zip(self.raw['task'], tokens, masks):
            self.assertTrue(tokenizer.decode(t[m], skip_special_tokens=True).startswith(f'Task: {text}, State: '))
            self.assertLess(int(m.sum()), 200)
        torch.testing.assert_close(self.post(self.batch['action']), self.raw['action'], rtol=1e-5, atol=1e-5)

    def reference_loss(self, seed):
        p, b = self.policy, self.batch
        images, masks = p._preprocess_images(b)
        actions = p.prepare_action(b)
        torch.manual_seed(seed)
        noise = p.model.sample_noise(actions.shape, actions.device)
        time = p.model.sample_time(actions.shape[0], actions.device)
        losses = p.model.forward(images, masks, b['observation.language.tokens'],
                                 b['observation.language.attention_mask'], actions, noise=noise, time=time)
        return losses[:, :, :8].mean(), losses.mean()

    def test_the_runners_loss_and_gradients_equal_a_reference_with_explicit_noise_and_time(self):
        seed = 20261031
        reference, over_all_32 = self.reference_loss(seed)
        reference.backward()
        expected = {n: p.grad.clone() for n, p in self.policy.named_parameters() if p.grad is not None}
        self.policy.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        loss, _ = self.policy(self.batch)
        (loss/1).backward()
        self.assertTrue(torch.isfinite(loss) and float(loss) > 0)
        self.assertEqual(float(loss), float(reference))
        self.assertNotEqual(float(loss), float(over_all_32))  # the padded action dimensions are not in the loss
        grads = {n: p.grad for n, p in self.policy.named_parameters() if p.grad is not None}
        self.assertEqual(set(grads), set(expected))
        self.assertGreater(len(grads), 800)
        for name, value in grads.items():
            self.assertTrue(torch.equal(value, expected[name]), name)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values()))

    def test_inference_queues_ten_physical_actions_per_replan_and_stays_finite(self):
        p = self.policy
        p.eval()
        try:
            p.reset()
            torch.manual_seed(5)
            with torch.no_grad():
                first = p.select_action(self.batch)
            self.assertEqual(tuple(first.shape), (2, 8))
            self.assertEqual(len(p._action_queue), c.N_ACTION_STEPS-1)
            physical = self.post(first)
            self.assertEqual(tuple(physical.shape), (2, 8))
            self.assertTrue(torch.isfinite(first).all() and torch.isfinite(physical).all())
            with torch.no_grad():
                chunk = p.predict_action_chunk(self.batch)
            self.assertEqual(tuple(chunk.shape), (2, c.CHUNK_SIZE, 8))
            self.assertTrue(torch.isfinite(chunk).all())
        finally:
            p.train()


if __name__ == '__main__':
    unittest.main()
