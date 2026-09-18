"""Synthetic CPU boundaries; never construct a model or advance physics."""
import copy
import importlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


def metadata():
    return [dict(episode_index=i, length=528, tasks=['pick up the red cube and place it on top of the green cube'],
                 dataset_from_index=i*528, dataset_to_index=(i+1)*528) for i in range(24)]


def frame(episode=0):
    return {'episode_index': torch.tensor(episode), 'task': metadata()[0]['tasks'][0],
            'observation.state': torch.arange(9).float(),
            'observation.images.image': torch.zeros(3,512,512),
            'observation.images.image2': torch.ones(3,512,512),
            'action': torch.zeros(30,8), 'action_is_pad': torch.zeros(30,dtype=torch.bool)}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).parents[1]/'contract.py').is_file(), 'new fixed-layout contract required')
        self.c = importlib.import_module('experiments.xvla_fixed_layout.contract')

    def test_exact_split_and_complete_starts(self):
        rows, starts = self.c.indices(metadata())
        self.assertEqual((len(rows),len(starts)), (10560,9980))
        self.assertEqual(rows,list(range(10560)))
        self.assertTrue(all(i//528 < 20 and i%528 <= 498 for i in starts))
        self.assertEqual((self.c.TRAIN,self.c.VAL),(tuple(range(20)),tuple(range(20,24))))

    def test_reject_pilot_metadata_and_bad_boundaries(self):
        for bad in (metadata()[:9],metadata()*2,metadata()[:23]):
            with self.assertRaises(ValueError): self.c.indices(bad)
        for key,value in [('length',527),('episode_index',True),('dataset_from_index',1),('tasks',['wrong'])]:
            bad=metadata(); bad[0][key]=value
            with self.assertRaises(ValueError): self.c.indices(bad)

    def test_stats_exclude_extreme_validation_and_require_all_train_rows(self):
        class Rows:
            def __getitem__(self,index):
                value = index%528 if index//528 < 20 else 1e20
                return dict(episode_index=index//528,task_index=0,
                            action=torch.full((8,),float(value)),
                            **{'observation.state':torch.full((9,),float(value))})
        rows,_=self.c.indices(metadata())
        stats=self.c.selected_stats(Rows(),rows)
        self.assertEqual(stats['action']['max'].tolist(),[527.]*8)
        self.assertEqual(stats['action']['mean'].tolist(),[263.5]*8)
        for bad in (rows[:4752],rows+[10560],rows[:-1]+[10560]):
            with self.assertRaises(ValueError): self.c.selected_stats(Rows(),bad)

    def test_train_row_whitelist_and_validation_rejection(self):
        source={**frame(19),'success':True,'object_positions':torch.ones(3)}
        batch=self.c.collate_rows([source]*4)
        self.assertEqual(set(batch),set(self.c.OBS)|{'task','action','action_is_pad'})
        self.assertEqual(batch['action'].shape,(4,30,8))
        for bad in (frame(20),{**frame(),'observation.oracle':torch.ones(1)},
                    {**frame(),'action':torch.zeros(29,8)}, {**frame(),'action':None},
                    {**frame(),'action_is_pad':torch.ones(30,dtype=torch.bool)}):
            with self.assertRaises(ValueError): self.c.collate_rows([bad]*4)
        with self.assertRaises(ValueError): self.c.collate_rows([frame()]*3)

    def test_observations_strictly_exclude_actions_and_oracles(self):
        obs={k:frame()[k] for k in (*self.c.OBS,'task')}
        self.c.observation(obs)
        for key in ('action','episode_index','success','object_positions'):
            with self.assertRaises(ValueError): self.c.observation({**obs,key:0})
        with self.assertRaises(ValueError):
            self.c.observation({**obs,'observation.state':torch.full((9,),float('nan'))})

    def test_sampler_one_epoch_all_9980_no_drop_or_val_and_reproducible(self):
        _,starts=self.c.indices(metadata())
        a,b=self.c.sample_batches(starts),self.c.sample_batches(starts)
        epoch=list(itertools.islice(a,2495))
        self.assertEqual(sorted(itertools.chain.from_iterable(epoch)),starts)
        self.assertEqual(epoch,list(itertools.islice(b,2495)))
        for bad in (starts[:4491],starts[:-1]+[10560],starts[:-1]+[starts[0]]):
            with self.assertRaises(ValueError): next(self.c.sample_batches(bad))

    def test_fixed_recipe_and_checkpoint_schedule(self):
        recipe=self.c.recipe(False)
        self.assertEqual((recipe['seed'],recipe['microbatch'],recipe['accumulation']), (20260917,4,2))
        self.assertEqual(recipe['max_optimizer_updates'],5000)
        self.assertEqual(self.c.checkpoint_updates(False),[1,1000,2000,3000,4000,5000])
        self.assertEqual(self.c.checkpoint_updates(True),[1,2])
        self.assertEqual(recipe['train_episodes'],list(range(20)))
        self.assertAlmostEqual(self.c.learning_rate(5000),1e-5)

    def test_cpu_proof_requires_semantics_stats_processor_and_exact_binding(self):
        semantic={'status':'passed','training_ready':True}
        binding={'semantic_sha256':self.c.digest_json(semantic)}
        proof=self.c.cpu_proof(binding,semantic,{'action':{'mean':[1.]*8}}, {'checked':True})
        self.c.validate_cpu(proof,binding)
        for change in ({'binding':{}},{'status':'stopped'},{'semantic_gate':{}},
                       {'training_stats':{}},{'processor_check':{}},{'optimizer_updates':1},
                       {'training_rows':4752},{'complete_training_chunks':4491}):
            with self.assertRaises(ValueError): self.c.validate_cpu({**proof,**change},binding)

    def test_light_binding_rejects_changed_seal_fingerprint_and_current_provenance(self):
        from experiments.fixed_layout import source,validate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,value in [('seal.json',{'fingerprint':self.c.SEAL}),
                               ('validation.json',{'status':'passed','training_ready':True,'provenance':{'v':1}}),
                               ('plan.json',{}),('reset.npz',{})]:
                (root/name).write_text(json.dumps(value))
            with patch.object(self.c,'PARENT',root), patch.object(self.c,'source_hashes',return_value={'x':'y'}), \
                 patch.object(self.c,'require_assets',return_value={'asset':'bytes'}), \
                 patch.object(source,'provenance',return_value={'v':1}), \
                 patch.object(validate,'fingerprint',return_value=self.c.SEAL), \
                 patch.object(self.c,'validate_plan'):
                good=self.c.current_binding()
                self.assertEqual(good['dataset_fingerprint'],self.c.SEAL)
                with patch.object(validate,'fingerprint',return_value='changed'),self.assertRaises(ValueError):
                    self.c.current_binding()
                with patch.object(source,'provenance',return_value={'v':2}),self.assertRaises(ValueError):
                    self.c.current_binding()
                (root/'seal.json').write_text('{}')
                with self.assertRaises(ValueError): self.c.current_binding()

    def test_admission_light_only_and_exact_cpu_bytes(self):
        semantic={'status':'passed','training_ready':True}
        binding={'semantic_sha256':self.c.digest_json(semantic)}
        proof=self.c.cpu_proof(binding,semantic,{'a':[1]}, {'checked':True})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json'; path.write_text(json.dumps(proof))
            with patch.object(self.c,'CPU_ADMISSION',path),patch.object(self.c,'current_binding',return_value=binding), \
                 patch('experiments.fixed_layout.validate.require_fixed_layout_pass',side_effect=AssertionError('expensive gate forbidden')):
                self.assertEqual(self.c.admission(),binding)
                proof['binding']={}; path.write_text(json.dumps(proof))
                with self.assertRaises(ValueError): self.c.admission()

    def test_scratch_proof_rejects_recipe_stats_and_hash_drift(self):
        proof={**self.c.recipe(True),'status':'complete','completed_updates':2,'binding':{'source':'hash'},
               'final_data_binding_passed':True,'strict_reload_fp32_exact':True,'adam_reload_fp32_exact':True,
               'reload_max_absolute_error':0.,'final_checkpoint_files':{'model':'hash'},
               'training_stats_sha256':'stats','checkpoints':[{'update':1},{'update':2}],
               'training_rows':10560,'complete_training_chunks':9980,'cpu_admission_sha256':'cpu'}
        self.c.validate_scratch(proof,{'source':'hash'},{'model':'hash'},'stats','cpu')
        for change in ({'microbatch':1},{'accumulation':1},{'resume':True},{'completed_updates':1},
                       {'training_stats_sha256':'old'},{'cpu_admission_sha256':'stale'},
                       {'binding':{}},{'adam_reload_fp32_exact':False},{'reload_max_absolute_error':float('nan')},
                       {'checkpoints':[{'update':2}]},{'training_rows':4752}):
            with self.assertRaises(ValueError):
                self.c.validate_scratch({**proof,**change},{'source':'hash'},{'model':'hash'},'stats','cpu')

    def test_explicit_sources_not_old_report_or_recursive_experiments(self):
        names=self.c.SOURCE_FILES
        self.assertIn('experiments/xvla_trial/preflight.py',names)
        self.assertIn('experiments/xvla_single_task/contract.py',names)
        self.assertIn('experiments/xvla_fixed_layout/evaluate.py',names)
        self.assertFalse(any('CPU_REPORT' in n or 'CPU_ADMISSION' in n for n in names))
        self.assertEqual(len(names),len(set(names)))

    def test_fractional_boolean_or_nonscalar_episode_is_not_an_integer_id(self):
        for episode in (19.5,True,torch.tensor([0,1]),torch.tensor(float('nan'))):
            with self.subTest(episode=episode),self.assertRaises(ValueError):
                self.c.policy_row({**frame(),'episode_index':episode},True)

    def test_prepare_data_rejects_inherited_stats_even_with_valid_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cpu.json'; path.write_text(json.dumps({'training_stats':{'old9demo':True}}))
            with patch.object(self.c,'CPU_ADMISSION',path),patch.object(self.c,'admission',return_value={'bound':True}), \
                 patch.object(self.c,'load_data',return_value=({},[],{'action':{'mean':torch.zeros(8)}})):
                with self.assertRaises(ValueError): self.c.prepare_data()


if __name__=='__main__': unittest.main()