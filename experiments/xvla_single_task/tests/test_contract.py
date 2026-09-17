"""CPU-only tests; tiny objectives and real batch preprocessing, never a full backward."""
import ast
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from experiments.xvla_trial import contract as old
from experiments.xvla_trial.tests.test_preflight import frame


class ContractTests(unittest.TestCase):
    def test_new_contract_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.xvla_single_task.contract'))

    def test_listed_hashes_ignore_added_files_but_reject_mutation(self):
        from experiments.xvla_single_task import contract as c
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'old.py').write_text('old')
            binding = {'old.py': c.sha256(root/'old.py')}
            (root/'new.py').write_text('new')
            self.assertEqual(c.verify_listed(binding, root=root), binding)
            (root/'old.py').write_text('changed')
            with self.assertRaises(ValueError):
                c.verify_listed(binding, root=root)

    def test_listed_missing_escape_and_symlink_rejected(self):
        from experiments.xvla_single_task import contract as c
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'real').write_text('x')
            (root/'link').symlink_to(root/'real')
            for name in ('missing', '../escape', '/absolute', 'link'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    c.verify_listed({name: c.sha256(root/'real')}, root=root)

    def test_prior_success_requires_exact_micro1_proof(self):
        from experiments.xvla_single_task import contract as c
        report = json.loads(c.OLD_RESULT.read_text())
        c.validate_prior(report)
        for key, value in [('completed_updates', 1), ('microbatch', 4), ('status', 'stopped'),
                           ('reload_max_absolute_error', 1e-9), ('adam_reload_fp32_exact', False),
                           ('final_data_binding_passed', False)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_prior({**report, key: value})

    def test_schedule_bounds_warmup_cosine_and_group_ratio(self):
        from experiments.xvla_single_task import contract as c
        self.assertAlmostEqual(c.learning_rate(1), 2e-7)
        self.assertAlmostEqual(c.learning_rate(500), 1e-4)
        self.assertAlmostEqual(c.learning_rate(5000), 1e-5)
        rates = [c.learning_rate(i) for i in range(501,5001)]
        self.assertTrue(all(1e-5 <= x <= 1e-4 for x in rates))
        self.assertTrue(all(a >= b for a,b in zip(rates,rates[1:])))
        for bad in (0,5001,True,1.0):
            with self.assertRaises(ValueError):
                c.learning_rate(bad)

    def test_config_and_native_optimizer_groups(self):
        from experiments.xvla_single_task import contract as c
        cfg = c.make_config('cpu')
        self.assertEqual((cfg.chunk_size,cfg.n_action_steps,cfg.tokenizer_max_length), (30,10,64))
        self.assertEqual(cfg.normalization_mapping['STATE'].value, 'IDENTITY')
        self.assertEqual(cfg.normalization_mapping['ACTION'].value, 'MEAN_STD')
        params = {n: torch.nn.Parameter(torch.ones(2)) for n in
                  ('model.vlm.x','model.soft_prompt_hub.x','model.head.x')}
        opt = cfg.get_optimizer_preset().build(params)
        for update in (1,2,500,501,5000):
            c.set_learning_rate(opt, update)
            for g in opt.param_groups:
                scale = .1 if g['name'] == 'vlm' else 1.
                self.assertAlmostEqual(g['lr'], c.learning_rate(update)*scale)
                self.assertAlmostEqual(g['weight_decay'], 1e-4*scale)
                self.assertEqual(g['betas'], (.9,.95))
        opt.param_groups[0]['weight_decay'] = 0
        with self.assertRaises(ValueError):
            c.set_learning_rate(opt, 3)

    def test_four_rows_really_collate_B30x8(self):
        from experiments.xvla_single_task import contract as c
        from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
        pre,_ = make_xvla_pre_post_processors(c.make_config('cpu'), {'action': {'mean':torch.zeros(8),'std':torch.ones(8)}})
        batch = c.collate_rows([frame() for _ in range(4)])
        self.assertEqual(batch['action'].shape, (4,30,8))
        actual = pre(batch)
        self.assertEqual(actual['action'].shape, (4,30,8))
        self.assertEqual(actual['observation.state'].shape, (4,9))
        self.assertEqual(actual['domain_id'].tolist(), [0]*4)

    def test_batch_rejects_validation_and_partial_microbatch(self):
        from experiments.xvla_single_task import contract as c
        for rows in ([frame()]*3, [frame()]*5, [frame()]*3+[{**frame(),'episode_index':300}]):
            with self.assertRaises(ValueError):
                c.collate_rows(rows)

    def test_counterfactual_object_truth_metadata_never_enters_policy(self):
        from experiments.xvla_single_task import contract as c
        a = frame()
        b = {**a,'object_positions':torch.ones(99),'success':True,'target_id':99}
        first, second = c.collate_rows([a]*4), c.collate_rows([b]*4)
        self.assertEqual(set(first),set(second))
        self.assertFalse({'episode_index','object_positions','success','target_id'} & set(second))
        for k in first:
            if isinstance(first[k],torch.Tensor):
                torch.testing.assert_close(first[k],second[k],rtol=0,atol=0)
        with self.assertRaises(ValueError):
            c.observation({**{k:a[k] for k in (*old.OBS,'task')},'object_positions':torch.ones(3)})

    def test_4x2_gradient_equals_effective8(self):
        from experiments.xvla_single_task import contract as c
        torch.manual_seed(old.SEED)
        x, y = torch.randn(8,3), torch.randn(8,30,8)
        whole, micro = torch.nn.Linear(3,240), torch.nn.Linear(3,240)
        micro.load_state_dict(whole.state_dict())
        torch.nn.functional.mse_loss(whole(x).reshape(8,30,8),y).backward()
        for lo in (0,4):
            c.backward_microbatch(torch.nn.functional.mse_loss(micro(x[lo:lo+4]).reshape(4,30,8),y[lo:lo+4]))
        for a,b in zip(whole.parameters(),micro.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-7)

    def test_nonfinite_loss_rejected(self):
        from experiments.xvla_single_task import contract as c
        with self.assertRaises(ValueError):
            c.backward_microbatch(torch.tensor(float('nan'),requires_grad=True))

    def test_model_input_prechecks_real_native_preparation(self):
        from experiments.xvla_single_task import contract as c
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
        cfg = c.make_config('cpu')
        policy = XVLAPolicy.__new__(XVLAPolicy)
        torch.nn.Module.__init__(policy)
        policy.config, policy.model = cfg, SimpleNamespace(dim_proprio=20,dim_action=20)
        pre,_ = make_xvla_pre_post_processors(cfg, {'action': {'mean':torch.zeros(8),'std':torch.ones(8)}})
        raw = c.collate_rows([frame()]*4)
        batch = pre(raw)
        evidence = c.check_model_inputs(policy,batch,raw['observation.state'])
        self.assertEqual(evidence['image_mask'], [[True,True,False]]*4)
        batch['domain_id'].fill_(1)
        with self.assertRaises(ValueError):
            c.check_model_inputs(policy,batch,raw['observation.state'])

    def saved_observation_inputs(self):
        from experiments.xvla_single_task import contract as c
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        cfg = c.make_config('cpu')
        # Use real native input preparation, without allocating the backbone.
        policy = XVLAPolicy.__new__(XVLAPolicy)
        torch.nn.Module.__init__(policy)
        policy.config, policy.model = cfg, SimpleNamespace(dim_proprio=20,dim_action=20)
        pre,_ = make_pre_post_processors(
            cfg,pretrained_path=str(c.ROOT/'outputs/train/xvla_single_task_20260917_preflight/checkpoints/000002'),
            preprocessor_overrides={'device_processor':{'device':'cpu'}})
        sample = frame()
        sample['observation.state'] = torch.arange(9).float()
        sample[c.CAMERAS[1]].fill_(1.)
        obs = c.observation({k:sample[k] for k in (*c.OBS,'task')})
        return policy,c.p.process_row(pre,obs),obs['observation.state'][None]

    def test_saved_observation_processor_none_or_absent_action_passes_precheck(self):
        from experiments.xvla_single_task import contract as c
        policy,batch,state = self.saved_observation_inputs()
        self.assertIn('action',batch)
        self.assertIsNone(batch['action'])
        for present in (True,False):
            with self.subTest(action_present=present):
                actual = dict(batch)
                if not present:
                    del actual['action']
                evidence = c.check_model_inputs(policy,actual,state)
                self.assertEqual(evidence['image_shape'],[1,3,3,224,224])
                self.assertEqual(evidence['image_mask'],[[True,True,False]])
                self.assertEqual(evidence['proprio_shape'],[1,20])
                self.assertEqual(evidence['token_shape'],[1,64])
                self.assertTrue(evidence['state_identity_exact'])

    def test_model_precheck_present_action_tensor_shape_stays_strict(self):
        from experiments.xvla_single_task import contract as c
        policy,batch,state = self.saved_observation_inputs()
        c.check_model_inputs(policy,{**batch,'action':torch.zeros(1,30,8)},state)
        for shape in ((30,8),(1,29,8),(1,30,20),(4,30,8),(0,30,8)):
            with self.subTest(shape=shape),self.assertRaisesRegex(ValueError,'action batch shape'):
                c.check_model_inputs(policy,{**batch,'action':torch.zeros(shape)},state)

    def test_training_collate_still_requires_action_on_every_row(self):
        from experiments.xvla_single_task import contract as c
        for action,error in ((None,AttributeError),(torch.zeros(29,8),ValueError),
                             (torch.zeros(30,20),ValueError)):
            with self.subTest(action=action),self.assertRaises(error):
                c.collate_rows([frame()]*3+[{**frame(),'action':action}])
        missing = frame()
        del missing['action']
        with self.assertRaises(KeyError):
            c.collate_rows([frame()]*3+[missing])

    def test_sampler_only_complete_training_chunks_and_reproducible(self):
        from experiments.xvla_single_task import contract as c
        episodes = [dict(episode_index=i,dataset_from_index=i*528,dataset_to_index=(i+1)*528) for i in range(420)]
        rows, starts = old.indices(episodes)
        self.assertEqual((len(rows),len(starts)),(4752,4491))
        a,b = c.sample_batches(starts),c.sample_batches(starts)
        for _ in range(1130):
            batch = next(a)
            self.assertEqual(batch,next(b))
            self.assertEqual(len(batch),4)
            self.assertTrue(all(i//528 in old.TRAIN and i%528 <= 498 for i in batch))

    def test_stats_cannot_include_validation(self):
        from experiments.xvla_trial.preflight import selected_stats
        rows = [{'episode_index':0,'task_index':0,'action':torch.ones(8)*i,'observation.state':torch.ones(9)*i} for i in (1,3)]
        rows.append({**rows[0],'episode_index':300,'action':torch.ones(8)*1e9})
        torch.testing.assert_close(selected_stats(rows,[0,1])['action']['mean'],torch.ones(8)*2)
        with self.assertRaises(ValueError):
            selected_stats(rows,[0,1,2])

    def test_real_dataset_batch_four_cpu_no_full_model(self):
        from experiments.xvla_single_task import contract as c
        from experiments.xvla_trial.preflight import selected_stats
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
        ds = LeRobotDataset(repo_id=old.DATA.name,root=old.DATA,delta_timestamps={'action':[i/30 for i in range(30)]})
        rows,starts = old.indices(ds.meta.episodes)
        ds._ensure_hf_dataset_loaded()
        stats = selected_stats(ds.hf_dataset,rows,int(ds.meta.tasks.loc[old.TASK,'task_index']))
        pre,post = make_xvla_pre_post_processors(c.make_config('cpu'),stats)
        raw = c.collate_rows([ds[i] for i in (starts[0],starts[498],starts[499],starts[-1])])
        batch = pre(raw)
        self.assertEqual(batch['action'].shape,(4,30,8))
        torch.testing.assert_close(post(batch['action']),raw['action'],rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(batch['observation.state'],raw['observation.state'],rtol=0,atol=0)

    def test_scratch_proof_rejects_different_recipe_or_inventory(self):
        from experiments.xvla_single_task import contract as c
        proof = {**c.recipe(True),'status':'complete','completed_updates':2,'binding':{'test':1},
                 'final_data_binding_passed':True,'strict_reload_fp32_exact':True,
                 'adam_reload_fp32_exact':True,'reload_max_absolute_error':0.,
                 'final_checkpoint_files':{'model':'hash'},'training_stats_sha256':'stats',
                 'checkpoints':[{'update':1},{'update':2}]}
        c.validate_scratch(proof,{'test':1},{'model':'hash'},'stats')
        for change in ({'microbatch':1},{'accumulation':1},{'completed_updates':3},{'resume':True},
                       {'binding':{}},{'reload_max_absolute_error':1e-8},{'adam_reload_fp32_exact':False},
                       {'final_checkpoint_files':{}},{'training_stats_sha256':'bad'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                c.validate_scratch({**proof,**change},{'test':1},{'model':'hash'},'stats')

    def test_scheduled_adam_exact_restore_using_old_helper(self):
        from experiments.xvla_single_task import contract as c
        from experiments.xvla_trial.preflight import restore_adam
        params = {n:torch.nn.Parameter(torch.ones(2)) for n in ('model.vlm.x','model.soft_prompt.x','model.head.x')}
        cfg = c.make_config('cpu')
        opt = cfg.get_optimizer_preset().build(params)
        c.set_learning_rate(opt,2)
        sum(p.square().sum() for p in params.values()).backward()
        opt.step()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'optimizer.pt'
            torch.save(opt.state_dict(),path)
            preset = cfg.get_optimizer_preset()
            preset.lr = c.learning_rate(2)
            restored = restore_adam(params,preset,path)
            for p in params.values():
                torch.testing.assert_close(restored.state[p]['exp_avg'],opt.state[p]['exp_avg'],rtol=0,atol=0)

    def test_cpu_review_binding_must_match_before_gpu(self):
        from experiments.xvla_single_task import contract as c
        proof = dict(status='passed',gpu_executed=False,optimizer_updates=0,binding={'source':'hash'})
        c.validate_cpu(proof,{'source':'hash'})
        for change in ({'binding':{}},{'status':'stopped'},{'gpu_executed':True},{'optimizer_updates':1}):
            with self.assertRaises(ValueError):
                c.validate_cpu({**proof,**change},{'source':'hash'})

    def test_rollout_port_ast_only_documented_changes(self):
        from experiments.xvla_single_task.evaluate import run_trial
        from validation.evaluate_pilot import run_trial as baseline
        # Normalize ONLY the documented XVLA adaptations; physics/score/trace code must be identical.
        class Normalize(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value == 30:
                    # Only the chunk tuple/dict and message are changed below, not physics30Hz.
                    return node
                if node.value == 'Expected chunk 30/action steps 10':
                    node.value = 'Expected chunk 50/action steps 10'
                return node
            def visit_Tuple(self,node):
                if ast.dump(node) == ast.dump(ast.parse('(30, 10)',mode='eval').body):
                    return ast.parse('(50, 10)',mode='eval').body
                return self.generic_visit(node)
            def visit_BinOp(self,node):
                if ast.unparse(node) == "c.SEED + entry['episode_index']":
                    return ast.parse('SEED + ordinal',mode='eval').body
                return self.generic_visit(node)
            def visit_Assign(self,node):
                if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='processed':
                    return None
                return self.generic_visit(node)
            def visit_If(self,node):
                if ast.unparse(node.test) == '_ == 0':
                    return None
                return self.generic_visit(node)
            def visit_With(self,node):
                node.items = [x for x in node.items if not ast.unparse(x.context_expr).startswith('torch.autocast(')]
                return self.generic_visit(node)
            def visit_Name(self,node):
                return ast.parse('pre(frame)',mode='eval').body if node.id=='processed' else node
            def visit_Dict(self,node):
                for i,key in enumerate(node.keys):
                    if isinstance(key,ast.Constant) and key.value=='chunk_size':
                        node.values[i] = ast.Constant(50)
                return self.generic_visit(node)
        actual = Normalize().visit(ast.parse(inspect.getsource(run_trial)))
        expected = ast.parse(inspect.getsource(baseline))
        self.assertEqual(ast.dump(actual),ast.dump(expected))

    def test_rollout_rejects_chunk50_without_spoofing_or_simulation(self):
        from experiments.xvla_single_task.evaluate import run_trial
        policy = SimpleNamespace(config=SimpleNamespace(chunk_size=50,n_action_steps=10))
        with self.assertRaisesRegex(ValueError,'chunk 30'):
            run_trial(policy,None,None,None,None,None,None,None,None)
        self.assertEqual(policy.config.chunk_size,50)

    def test_gripper_contract_is_continuous_denormalization_then_threshold127(self):
        from validation.rollout_diagnostics import bound_action
        import numpy as np
        limits = np.array([[-2.,2.]]*7)
        for value,expected in ((0.,0.),(127.,0.),(127.01,255.),(255.,255.)):
            action = np.zeros(8)
            action[7] = value
            bounded,_ = bound_action(action,limits)
            self.assertEqual(bounded[7],expected)

    def test_cli_requires_explicit_mode(self):
        from experiments.xvla_single_task.runner import parse_args
        for args in ([],['--preflight','--train'],['--train','--updates','6000']):
            with self.assertRaises(SystemExit):
                parse_args(args)
        self.assertTrue(parse_args(['--preflight']).preflight)

    def test_summary_rates_provisional_and_no_extra_trials(self):
        from experiments.xvla_single_task.evaluate import summarize
        rows = [{'episode_index':i,'task_success':i not in (8,303)} for i in old.TRAIN+old.VAL]
        result = summarize(rows)
        self.assertEqual(result['seen']['success_rate'],8/9)
        self.assertEqual(result['held_out']['success_rate'],3/4)
        self.assertTrue(result['provisional_capability'])
        self.assertFalse(result['statistical_generalization'])
        with self.assertRaises(ValueError):
            summarize(rows+[rows[0]])

    def test_native_queue_reset_and_ceil_ten(self):
        from experiments.xvla_single_task import contract as c
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        class QueueOnly(XVLAPolicy):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.config = c.make_config('cpu')
                self.calls = 0
                self.reset()
            def _get_action_chunk(self,batch):
                self.calls += 1
                return torch.arange(30).float()[None,:,None].expand(1,30,8)
        policy = QueueOnly()
        for i in range(23):
            self.assertEqual(policy.select_action({})[0,0].item(),i%10)
        self.assertEqual(policy.calls,3)
        policy.reset()
        self.assertEqual(policy.select_action({})[0,0].item(),0)
        self.assertEqual(policy.calls,4)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()