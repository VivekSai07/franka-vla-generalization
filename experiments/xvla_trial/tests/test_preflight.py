"""CPU tests: actual stock processors/forward with small objective-only test fixtures."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.xvla_trial import contract as c


def frame():
    return {**{k: torch.zeros(shape) for k, shape in c.OBS.items()},
            'task': c.TASK, 'episode_index': torch.tensor(0),
            'action': torch.ones(30, 8), 'action_is_pad': torch.zeros(30, dtype=torch.bool)}


class PreflightTests(unittest.TestCase):
    def test_batch_admission_rejects_padding_bad_schema_and_privileged_inputs(self):
        from experiments.xvla_trial.preflight import policy_row
        good = policy_row(frame(), training=True)
        self.assertEqual(set(good), set(c.OBS) | {'task', 'action', 'action_is_pad'})
        for change in [{'action_is_pad': torch.ones(30, dtype=torch.bool)},
                       {'action': torch.zeros(29, 8)}, {'action': torch.zeros(30, 20)},
                       {'observation.state': torch.zeros(20)},
                       {c.CAMERAS[0]: torch.full((3, 512, 512), float('nan'))},
                       {'observation.oracle': torch.zeros(3)}, {'episode_index': torch.tensor(300)}]:
            with self.subTest(keys=list(change)), self.assertRaises(ValueError):
                policy_row({**frame(), **change}, training=True)

    def test_config_preserves_capacity_but_real_features_and_all_trainable(self):
        from experiments.xvla_trial.preflight import make_config
        cfg = make_config('cpu')
        cfg.validate_features()
        cfg.validate_features()  # empty_cameras=0 avoids repeated feature-count inflation
        self.assertEqual((cfg.action_mode, cfg.chunk_size, cfg.max_action_dim, cfg.max_state_dim), ('auto', 30, 20, 20))
        self.assertEqual({k: tuple(v.shape) for k,v in cfg.input_features.items()}, c.OBS)
        self.assertEqual(tuple(cfg.output_features['action'].shape), (8,))
        self.assertEqual((cfg.num_image_views, cfg.empty_cameras, cfg.dtype), (3, 0, 'float32'))
        self.assertFalse(cfg.freeze_vision_encoder or cfg.freeze_language_encoder or cfg.push_to_hub)

    def test_actual_processors_normalization_cameras_domain_and_state_padding(self):
        from experiments.xvla_trial.preflight import make_config, policy_row, process_row
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
        cfg = make_config('cpu')
        stats = {'action': {'mean': torch.full((8,), 3.), 'std': torch.full((8,), 2.)}}
        pre, post = make_xvla_pre_post_processors(cfg, stats)
        sample = frame()
        sample[c.CAMERAS[1]].fill_(1.)
        sample['observation.state'] = torch.arange(9).float()
        batch = process_row(pre, policy_row(sample, training=True))
        torch.testing.assert_close(batch['action'], torch.full((1,30,8), -1.))
        torch.testing.assert_close(post(batch['action']), torch.ones(1,30,8))
        torch.testing.assert_close(batch['observation.state'], torch.arange(9).float()[None])
        self.assertEqual(batch['domain_id'].tolist(), [0])
        self.assertEqual(tuple(batch['observation.language.tokens'].shape), (1,64))
        # Exercise actual policy preparation without allocating its full backbone.
        policy = XVLAPolicy.__new__(XVLAPolicy)
        torch.nn.Module.__init__(policy)
        policy.config, policy.model = cfg, SimpleNamespace(dim_proprio=20, dim_action=20)
        inputs = policy._build_model_inputs(batch)
        self.assertEqual(tuple(inputs['image_input'].shape), (1,3,3,224,224))
        self.assertEqual(inputs['image_mask'].tolist(), [[True, True, False]])
        self.assertEqual(torch.count_nonzero(inputs['image_input'][:,2]).item(), 0)
        torch.testing.assert_close(inputs['image_input'][0,0,:,100,100],
                                   torch.tensor([-.485/.229, -.456/.224, -.406/.225]))
        torch.testing.assert_close(inputs['image_input'][0,1,:,100,100],
                                   torch.tensor([.515/.229, .544/.224, .594/.225]))
        torch.testing.assert_close(inputs['proprio'][0,:9], torch.arange(9).float())
        self.assertEqual(torch.count_nonzero(inputs['proprio'][:,9:]).item(), 0)
        target = policy._prepare_action_targets(batch)
        self.assertEqual(tuple(target.shape), (1,30,20))
        self.assertEqual(torch.count_nonzero(target[:,:,8:]).item(), 0)

    def test_stock_forward_uses_clean_target_all_real8_and_ignores_padded_outputs(self):
        from experiments.xvla_trial.preflight import make_config
        from lerobot.policies.xvla.modeling_xvla import XVLAModel
        from lerobot.policies.xvla.action_hub import AutoActionSpace

        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pred = torch.nn.Parameter(torch.zeros(1,30,20))
            def forward(self, **kwargs):
                self.noisy = kwargs['action_with_noise'].detach()
                return self.pred

        class ObjectiveOnly(XVLAModel):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.config = make_config('cpu')
                self.action_space = AutoActionSpace(real_dim=8, max_dim=20)
                self.transformer = Head()
            def forward_vlm(self, *args):
                return {}  # vision is deliberately outside this objective-only fixture

        model = ObjectiveOnly()
        action = torch.zeros(1,30,20)
        action[:,:,:8] = torch.arange(1,9)
        loss = sum(model(torch.ones(1,2,dtype=torch.long), torch.zeros(1,3,3,4,4),
                         torch.tensor([[True,True,False]]), torch.tensor([0]), torch.zeros(1,20), action).values())
        self.assertAlmostEqual(loss.item(), 25.5)  # sum(1^2..8^2)/8, NOT a noise/velocity target
        loss.backward()
        self.assertTrue(torch.all(model.transformer.pred.grad[:,:,:8] != 0))
        self.assertEqual(torch.count_nonzero(model.transformer.pred.grad[:,:,8:]).item(), 0)
        self.assertGreater(torch.count_nonzero(model.transformer.noisy[:,:,8:]).item(), 0)
        with torch.no_grad():
            model.transformer.pred[:,:,8:] = 10000
        self.assertEqual(model.action_space.compute_loss(model.transformer.pred, action)['joints_loss'].item(), 25.5)
        self.assertEqual(tuple(model.action_space.postprocess(model.transformer.pred).shape), (1,30,8))

    def test_optimizer_keeps_vlm_ratio_fp32_and_updates_vlm(self):
        from experiments.xvla_trial.preflight import make_config, verify_precision
        params = {n: torch.nn.Parameter(torch.ones(2)) for n in
                  ['model.vlm.weight', 'model.transformer.soft_prompt_hub.weight', 'model.transformer.weight']}
        optimizer = make_config('cpu').get_optimizer_preset().build(params)
        before = params['model.vlm.weight'].detach().clone()
        sum(p.square().sum() for p in params.values()).backward()
        optimizer.step()
        verify_precision(params, optimizer, require_gradients=True)
        rates = {g['name']:g['lr'] for g in optimizer.param_groups}
        self.assertAlmostEqual(rates['vlm']/rates['other'], .1)
        self.assertFalse(torch.equal(before, params['model.vlm.weight']))
        optimizer.state[params['model.vlm.weight']]['exp_avg'] = torch.ones(2, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            verify_precision(params, optimizer, require_gradients=True)

    def test_strict_load_preserves_fp32_prediction_and_rejects_missing_key(self):
        from experiments.xvla_trial.preflight import load_weights
        from safetensors.torch import save_model, save_file
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.safetensors'
            original, reloaded = torch.nn.Linear(3,8), torch.nn.Linear(3,8)
            save_model(original, str(path))
            load_weights(reloaded, path)
            probe = torch.arange(3).float()
            torch.testing.assert_close(original(probe), reloaded(probe), rtol=0, atol=0)
            save_file({'weight': original.weight.detach()}, str(path))
            with self.assertRaises(RuntimeError):
                load_weights(reloaded, path)

    def test_gpu_gate_rejects_missing_approval_before_output_creation(self):
        from experiments.xvla_trial.preflight import main
        with self.assertRaises(SystemExit) as caught:
            main([])
        self.assertEqual(caught.exception.code, 2)

    def test_adam_save_reload_preserves_fp32_moments_and_group_rates(self):
        from experiments.xvla_trial.preflight import make_config, restore_adam
        params = {n: torch.nn.Parameter(torch.ones(2)) for n in ['model.vlm.weight', 'model.head.weight']}
        preset = make_config('cpu').get_optimizer_preset()
        optimizer = preset.build(params)
        sum(p.square().sum() for p in params.values()).backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'optimizer.pt'
            torch.save(optimizer.state_dict(), path)
            restored = restore_adam(params, preset, path)
            self.assertEqual([g['lr'] for g in restored.param_groups], [1e-5,1e-4])
            for p in params.values():
                for key in ('exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(restored.state[p][key], optimizer.state[p][key], rtol=0, atol=0)

    def test_saved_config_and_processors_preserve_real_schema_and_predictions(self):
        from experiments.xvla_trial.preflight import make_config, process_row, policy_row
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
        cfg = make_config('cpu')
        stats = {'action': {'mean': torch.arange(8).float(), 'std': torch.ones(8)*2}}
        pre, post = make_xvla_pre_post_processors(cfg, stats)
        row = policy_row(frame(), training=True)
        expected = process_row(pre, row)
        with tempfile.TemporaryDirectory() as d:
            cfg.save_pretrained(d)
            pre.save_pretrained(d)
            post.save_pretrained(d)
            restored = PreTrainedConfig.from_pretrained(d, local_files_only=True)
            self.assertEqual({k:tuple(v.shape) for k,v in restored.input_features.items()}, c.OBS)
            self.assertEqual((tuple(restored.action_feature.shape), restored.max_action_dim,
                              restored.max_state_dim, restored.num_image_views, restored.dtype),
                             ((8,),20,20,3,'float32'))
            rpre, rpost = make_pre_post_processors(restored, pretrained_path=d)
            actual = process_row(rpre, row)
            for key in (*c.OBS, 'action', 'domain_id', 'observation.language.tokens'):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            torch.testing.assert_close(rpost(actual['action']), post(expected['action']), rtol=0, atol=0)

    def test_train_stats_exclude_extreme_validation_rows(self):
        from experiments.xvla_trial.preflight import selected_stats
        rows = [{**{'episode_index': 0, 'task_index': 0}, 'observation.state': torch.ones(9)*v,
                 'action': torch.ones(8)*v} for v in [1.,3.,1000000.]]
        rows[2]['episode_index'] = 300
        stats = selected_stats(rows, [0,1])
        torch.testing.assert_close(stats['action']['mean'], torch.full((8,),2.))
        torch.testing.assert_close(stats['action']['std'], torch.ones(8))
        with self.assertRaises(ValueError):
            selected_stats(rows, [0,2])