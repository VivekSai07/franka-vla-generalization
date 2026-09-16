import unittest

import torch

from validation.precision import promote_trainable_parameters, verify_optimizer_precision


class PrecisionTests(unittest.TestCase):
    def test_promotes_only_trainable_values_and_keeps_parameter_identity(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)).bfloat16()
        model[0].requires_grad_(False)
        before = {n: (id(p), p.detach().clone()) for n, p in model.named_parameters()}
        promote_trainable_parameters(model)
        for name, param in model.named_parameters():
            self.assertEqual(id(param), before[name][0])
            self.assertEqual(param.dtype, torch.float32 if param.requires_grad else torch.bfloat16)
            torch.testing.assert_close(param.float(), before[name][1].float(), rtol=0, atol=0)

    def test_small_update_survives_and_adam_moments_are_fp32(self):
        model = torch.nn.Linear(1, 1, bias=False).bfloat16()
        with torch.no_grad():
            model.weight.fill_(1)
        promote_trainable_parameters(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        self.assertLess(model.weight.item(), 1.)
        self.assertTrue(verify_optimizer_precision(optimizer))


if __name__ == "__main__":
    unittest.main()