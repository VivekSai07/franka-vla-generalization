import ast
import importlib
import inspect
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from experiments.xvla_fixed_layout.tests.test_contract import frame


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).parents[1]/'evaluate.py').is_file(),'new evaluator required')
        self.e=importlib.import_module('experiments.xvla_fixed_layout.evaluate')

    def test_twenty_real_seeds_one_reset_identity(self):
        plan=self.e.evaluation_plan()
        self.assertEqual([r['seed'] for r in plan],list(range(20260918,20260938)))
        self.assertEqual({r['reset_file'] for r in plan},{'reset.npz'})
        self.assertEqual({r['source_episode_index'] for r in plan},{7})
        draws=[]
        for trial in plan:
            self.e.seed_trial(trial['seed']); draws.append((random.random(),np.random.random(),torch.rand(()).item()))
        self.assertEqual(len(set(draws)),20)
        self.e.seed_trial(plan[0]['seed'])
        self.assertEqual(draws[0],(random.random(),np.random.random(),torch.rand(()).item()))

    def test_gate_no_partial_success_and_exact_complete_unique_seeds(self):
        self.assertFalse(self.e.summarize([{'task_success':True}]*18)['capability_passed'])
        rows=[{'seed':r['seed'],'task_success':i<18,'outcome':'success' if i<18 else 'timeout'}
              for i,r in enumerate(self.e.evaluation_plan())]
        self.assertTrue(self.e.summarize(rows)['capability_passed'])
        for bad in (rows[:19],rows[:-1]+[rows[0]],rows+[rows[0]],
                    [{**r,'outcome':'nonfinite_action'} for r in rows]):
            self.assertFalse(self.e.summarize(bad)['capability_passed'])
        with self.assertRaises(ValueError): self.e.require_complete_trials(rows[:19])

    def test_probes_all20plus4_frames_observation_only_and_rng_preserved(self):
        class Dataset:
            def __getitem__(self,index):
                return {**frame(index//528),'success':True,'object_positions':torch.ones(3)}
        class Tiny(torch.nn.Module):
            def __init__(self): super().__init__(); self.calls=0
            def predict_action_chunk(self,batch):
                self.calls+=1
                if 'action' in batch or 'episode_index' in batch or 'success' in batch:
                    raise AssertionError('target/metadata leakage')
                random.random(); np.random.random(); torch.rand(())
                return torch.zeros(1,30,8)
        policy=Tiny(); policy.train()
        random.seed(91); np.random.seed(91); torch.manual_seed(91)
        python=random.getstate(); numpy=np.random.get_state(); tensor=torch.get_rng_state().clone()
        result=self.e.offline_probes(policy,lambda x:x,lambda x:x,Dataset(),device='cpu')
        self.assertEqual(policy.calls,72)
        self.assertEqual((result['train']['probes'],result['validation']['probes']),(60,12))
        self.assertEqual({r['frame'] for r in result['rows']},{60,107,347})
        self.assertEqual(result['train']['arm_mae_rad'],0.)
        self.assertEqual(result['validation']['gripper_binary_accuracy'],1.)
        self.assertTrue(policy.training)
        self.assertEqual(random.getstate(),python)
        self.assertTrue(np.array_equal(np.random.get_state()[1],numpy[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(),tensor))

    def test_stage_metrics_are_saved_action_contact_not_distance_oracle(self):
        actions=np.zeros((40,8)); actions[35:,7]=255
        trace={'applied_action':actions,'bilateral_target_contact':np.arange(40)>=5,
               'pick_lift_m':np.arange(40)*.01,'released_stack':np.arange(40)>=36}
        result=self.e.stage_metrics(trace)
        self.assertEqual(result['first_closure_tick'],0)
        self.assertEqual(result['sustained_closure_30_tick'],29)
        self.assertEqual(result['first_bilateral_contact_tick'],5)
        self.assertEqual(result['first_release_after_closure_tick'],35)
        self.assertIsNone(result['stable_stack_30_tick'])

    def test_nonfinite_outcome_evidence_never_passes_complete_gate(self):
        rows=[dict(seed=r['seed'],task_success=True,outcome='success',sim_seconds=float('nan'))
              for r in self.e.evaluation_plan()]
        self.assertFalse(self.e.summarize(rows)['capability_passed'])
        with self.assertRaises(ValueError): self.e.require_complete_trials(rows)

    def test_reset_adapter_same_bytes_and_identity_for_every_seed(self):
        c=self.e.c
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp); (parent/'reset.npz').write_bytes(b'synthetic-reset')
            binding={'reset_sha256':c.sha256(parent/'reset.npz')}
            with patch.object(c,'PARENT',parent):
                entries=[self.e.reset_entry(binding) for _ in self.e.evaluation_plan()]
                self.assertTrue(all(entry==entries[0] for entry in entries))
                self.assertEqual(entries[0]['episode_index'],7)
                self.assertEqual(entries[0]['file'],'reset.npz')
                (parent/'reset.npz').write_bytes(b'changed')
                with self.assertRaises(ValueError): self.e.reset_entry(binding)

    def test_final_checkpoint_gate_rejects_selection_schedule_and_proof_drift(self):
        c=self.e.c
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/'checkpoints/005000'; checkpoint.mkdir(parents=True)
            (root/'initial_config').mkdir(); (root/'scratch').mkdir()
            (root/'scratch/result.json').write_text('scratch')
            (root/'cpu.json').write_text('cpu'); (root/'training_stats.json').write_text('stats')
            config=dict(type='xvla',chunk_size=30,n_action_steps=10,dtype='float32',
                        input_features={k:{'shape':list(v)} for k,v in c.OBS.items()},
                        output_features={'action':{'shape':[8]}})
            for path in (checkpoint/'config.json',root/'initial_config/config.json'):
                path.write_text(json.dumps(config))
            report={**c.recipe(False),'status':'complete','completed_updates':5000,
                    'selected_checkpoint_update':5000,'final_data_binding_passed':True,
                    'strict_reload_fp32_exact':True,'reload_max_absolute_error':0.,'binding':{'bound':True},
                    'training_rows':10560,'complete_training_chunks':9980,'scratch_used_for_initialization':False,
                    'cpu_admission_sha256':c.sha256(root/'cpu.json'),
                    'training_stats_sha256':c.sha256(root/'training_stats.json'),
                    'scratch_result_sha256':c.sha256(root/'scratch/result.json'),
                    'final_checkpoint_files':c.inventory(checkpoint),
                    'checkpoints':[{'update':i} for i in c.checkpoint_updates(False)]}
            with patch.object(c,'CPU_ADMISSION',root/'cpu.json'),patch.object(c,'SCRATCH',root/'scratch'):
                (root/'result.json').write_text(json.dumps(report))
                self.assertEqual(self.e.verify_final(root,{'bound':True}),checkpoint)
                for change in ({'selected_checkpoint_update':1000},{'checkpoints':[{'update':5000}]},
                               {'cpu_admission_sha256':'bad'},{'training_stats_sha256':'old'},
                               {'completed_updates':4999},{'final_checkpoint_files':{}},
                               {'scratch_used_for_initialization':True}):
                    (root/'result.json').write_text(json.dumps({**report,**change}))
                    with self.assertRaises(ValueError): self.e.verify_final(root,{'bound':True})

    def test_probe_nonfinite_prediction_restores_rng_and_mode_on_failure(self):
        class Dataset:
            def __getitem__(self,index): return frame(index//528)
        class Tiny(torch.nn.Module):
            def predict_action_chunk(self,batch):
                torch.rand(()); random.random(); np.random.random()
                return torch.full((1,30,8),float('nan'))
        policy=Tiny(); policy.train()
        python=random.getstate(); tensor=torch.get_rng_state().clone(); numpy=np.random.get_state()
        with self.assertRaises(ValueError):
            self.e.offline_probes(policy,lambda x:x,lambda x:x,Dataset(),device='cpu')
        self.assertTrue(policy.training)
        self.assertEqual(python,random.getstate())
        self.assertTrue(torch.equal(tensor,torch.get_rng_state()))
        self.assertTrue(np.array_equal(numpy[1],np.random.get_state()[1]))

    def test_rollout_port_retains_timing_action_scoring_ast(self):
        from experiments.xvla_single_task.evaluate import run_trial as old
        # Only reset, real seed, nonmonkeypatch call counting and extra reporting differ.
        def loop(function):
            tree=ast.parse(inspect.getsource(function))
            return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(1200)')
        self.assertEqual(ast.dump(loop(old)),ast.dump(loop(self.e.run_trial)))
        source=inspect.getsource(self.e.run_trial)
        self.assertNotIn('policy._get_action_chunk =',source)
        self.assertNotIn("c.SEED + entry['episode_index']",source)


if __name__=='__main__': unittest.main()