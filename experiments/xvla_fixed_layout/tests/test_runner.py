import importlib
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from experiments.xvla_fixed_layout.tests.test_contract import frame


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((Path(__file__).parents[1]/'runner.py').is_file(),'new runner required')
        self.r=importlib.import_module('experiments.xvla_fixed_layout.runner')
        self.c=self.r.c

    def test_cli_no_resume_override_or_combined_modes(self):
        for args in ([],['--train','--resume'],['--train','--steps','1'],['--check','--preflight']):
            with self.assertRaises(SystemExit): self.r.parse_args(args)
        for mode in ('check','preflight','train'):
            self.assertTrue(getattr(self.r.parse_args(['--'+mode]),mode))

    def test_empty_output_bind_mount_claim_is_exclusive_and_never_resumes(self):
        self.assertTrue(hasattr(self.c,'claim_output'),'empty mounted output needs an exclusive claim')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'output'
            self.c.claim_output(root)
            self.assertTrue((root/'run_claim.json').is_file())
            with self.assertRaises(FileExistsError): self.c.claim_output(root)
            mounted=Path(tmp)/'mounted'; mounted.mkdir()
            with self.assertRaises(FileExistsError): self.c.claim_output(mounted)
            with patch.object(self.c,'is_mountpoint',return_value=True):
                self.c.claim_output(mounted)
                with self.assertRaises(FileExistsError): self.c.claim_output(mounted)
            linked=Path(tmp)/'link'; linked.symlink_to(root,target_is_directory=True)
            with self.assertRaises(ValueError): self.c.claim_output(linked)

    def test_resource_floor_main_60g_disk_28g_gpu(self):
        for disk,gpu,passes in ((60,28,True),(59.99,28,False),(60,27.99,False)):
            with patch.object(self.r.shutil,'disk_usage',return_value=type('Disk',(),{'free':int(disk*1024**3)})()), \
                 patch.object(torch.cuda,'mem_get_info',return_value=(int(gpu*1024**3),48*1024**3)):
                if passes: self.r.require_resources(Path('/tmp'),False)
                else:
                    with self.assertRaises(RuntimeError): self.r.require_resources(Path('/tmp'),False)

    def test_checkpoint_saves_processors_once_and_optimizer_final_only_exclusive(self):
        class Saved:
            def __init__(self,name): self.name,self.calls=name,0
            def save_pretrained(self,path):
                self.calls+=1; (path/self.name).write_text('saved')
        policy,pre,post=Saved('model'),Saved('pre'),Saved('post')
        parameter=torch.nn.Parameter(torch.ones(1))
        opt=torch.optim.AdamW([{'params':[parameter],'name':'other'}])
        with tempfile.TemporaryDirectory() as tmp,patch.object(self.r,'require_disk'):
            root=Path(tmp); report={'checkpoints':[]}
            one=self.r.save_checkpoint(policy,pre,post,opt,root,1,False,report)
            final=self.r.save_checkpoint(policy,pre,post,opt,root,2,True,report)
            self.assertFalse((one/'optimizer.pt').exists())
            self.assertTrue((final/'optimizer.pt').is_file())
            self.assertEqual([m['sampled_chunks'] for m in report['checkpoints']],[8,16])
            self.assertEqual((policy.calls,pre.calls,post.calls),(2,2,2))
            with self.assertRaises(FileExistsError): self.r.save_checkpoint(policy,pre,post,opt,root,2,True,report)

    def test_tiny_strict_weights_and_scheduled_adam_reload_precision(self):
        from safetensors.torch import save_model
        from experiments.xvla_trial.preflight import load_weights,restore_adam
        cfg=self.c.make_config('cpu')
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model=torch.nn.ModuleDict({name:torch.nn.Linear(2,2)
                                               for name in ('vlm','soft_prompt_hub','head')})
            def forward(self,x): return sum(layer(x) for layer in self.model.values())
        model=Tiny(); params=dict(model.named_parameters())
        opt=cfg.get_optimizer_preset().build(params)
        for step in (1,2):
            self.c.set_learning_rate(opt,step); opt.zero_grad()
            for _ in range(2): self.c.backward_microbatch(model(torch.ones(4,2)).square().mean())
            opt.step()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); save_model(model,str(root/'model.safetensors')); torch.save(opt.state_dict(),root/'optimizer.pt')
            loaded=load_weights(Tiny(),root/'model.safetensors')
            self.assertTrue(torch.equal(model(torch.ones(1,2)),loaded(torch.ones(1,2))))
            preset=cfg.get_optimizer_preset(); preset.lr=self.c.learning_rate(2)
            restored=restore_adam(dict(loaded.named_parameters()),preset,root/'optimizer.pt')
            self.assertEqual(len(restored.state),6)
            saved=opt.state_dict(); saved['state'][0]['exp_avg']=saved['state'][0]['exp_avg'].bfloat16()
            torch.save(saved,root/'bad.pt')
            with self.assertRaises(ValueError): restore_adam(dict(loaded.named_parameters()),preset,root/'bad.pt')

    def test_cpu_micro4_actual_processor_no_model_construction(self):
        stats={'action':{'mean':torch.arange(8).float(),'std':torch.ones(8)}}
        rows={i:frame(0) for i in (0,498,528,10530)}
        evidence=self.r.processor_check(rows,[0,498,528,10530],stats)
        self.assertEqual(evidence['image_shape'],[4,3,3,224,224])
        self.assertEqual(evidence['image_mask'],[[True,True,False]]*4)
        self.assertTrue(evidence['state_identity_exact'])
        self.assertTrue(evidence['physical_action_roundtrip'])

    def test_check_exclusive_and_full_semantics_exactly_once(self):
        semantic={'status':'passed','training_ready':True}
        binding={'semantic_sha256':self.c.digest_json(semantic)}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'admission'
            with patch.object(self.c,'ADMISSION',root),patch.object(self.c,'CPU_ADMISSION',root/'result.json'), \
                 patch.object(self.c,'current_binding',return_value=binding), \
                 patch('experiments.fixed_layout.validate.require_fixed_layout_pass',return_value=semantic) as gate, \
                 patch.object(self.c,'load_data',return_value=({},list(range(9980)),{'action':{'mean':torch.zeros(8)}})), \
                 patch.object(self.r,'processor_check',return_value={'checked':True}):
                self.r.check()
                self.assertEqual(gate.call_count,1)
                self.assertEqual(json.loads((root/'result.json').read_text())['status'],'passed')
                with self.assertRaises(FileExistsError): self.r.check()
                self.assertEqual(gate.call_count,1)

    def test_no_dataset_bound_old_helpers_in_new_runtime(self):
        import ast
        forbidden={'p.prepare_data','p.policy_row','p.selected_stats','p.source_hashes',
                   'old.admission','old.sample_batches','c.p.prepare_data','c.p.policy_row'}
        for name in ('contract.py','runner.py','evaluate.py'):
            tree=ast.parse((Path(__file__).parents[1]/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call): self.assertNotIn(ast.unparse(node.func),forbidden)

    def test_reload_phase_tiny_policy_one_weight_load_scratch_only_adam_restore(self):
        from safetensors.torch import save_model
        from lerobot.policies.xvla import modeling_xvla
        from lerobot.policies import factory
        c=self.c
        config=c.make_config('cpu')
        class Tiny(torch.nn.Module):
            constructions=0
            def __init__(self,cfg):
                super().__init__(); type(self).constructions+=1; self.config=cfg
                self.model=torch.nn.ModuleDict({n:torch.nn.Linear(2,2) for n in ('vlm','soft_prompt_hub','head')})
            def to(self,device):
                if device!='cuda': raise AssertionError('unexpected requested device')
                return self  # Test-only fake device; all tensors remain on CPU.
            def get_optim_params(self): return dict(self.named_parameters())
            def predict_action_chunk(self,batch):
                return sum(layer(torch.ones(1,2)) for layer in self.model.values()).repeat(1,120).reshape(1,30,8)
            def save_pretrained(self,path):
                save_model(self,str(path/'model.safetensors')); self.config.save_pretrained(path)
        class Processor:
            def __init__(self,name): self.name=name
            def __call__(self,value): return value
            def save_pretrained(self,path): (path/self.name).write_text('processor')
        pre,post=Processor('pre'),Processor('post')
        model=Tiny(config); optimizer=config.get_optimizer_preset().build(model.get_optim_params())
        c.set_learning_rate(optimizer,2)
        model.predict_action_chunk({}).square().mean().backward(); optimizer.step()
        expected=model.predict_action_chunk({}).detach()
        stats={'action':{'mean':torch.zeros(8)}}
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp); cpu=parent/'cpu.json'; cpu.write_text('cpu')
            scratch=parent/'scratch'; scratch.mkdir(); (scratch/'result.json').write_text('scratch')
            for preflight in (True,False):
                with self.subTest(preflight=preflight),ExitStack() as stack:
                    output=parent/str(preflight); output.mkdir()
                    config.save_pretrained(output/'initial_config')
                    c.write_json(output/'training_stats.json',self.r.statistics_json(stats))
                    report=dict(checkpoints=[],binding={'fixture':True},cpu_admission_sha256=c.sha256(cpu),
                                training_stats_sha256=c.sha256(output/'training_stats.json'),reload_probe_row=0,
                                scratch_result_sha256=c.sha256(scratch/'result.json'))
                    stack.enter_context(patch.object(self.r,'require_disk'))
                    for update in c.checkpoint_updates(preflight):
                        path=self.r.save_checkpoint(model,pre,post,optimizer,output,update,
                                                    update==c.checkpoint_updates(preflight)[-1],report)
                    torch.save(expected,path/'reload_reference.pt')
                    stack.enter_context(patch.object(modeling_xvla,'XVLAPolicy',Tiny))
                    stack.enter_context(patch.object(factory,'make_pre_post_processors',return_value=(pre,post)))
                    stack.enter_context(patch.object(c,'prepare_data',return_value=({0:frame()},[0],stats,report['binding'])))
                    stack.enter_context(patch.object(c,'admission',return_value=report['binding']))
                    stack.enter_context(patch.object(c,'CPU_ADMISSION',cpu))
                    stack.enter_context(patch.object(c,'SCRATCH',scratch))
                    stack.enter_context(patch.object(c,'check_model_inputs',return_value={'synthetic':True}))
                    stack.enter_context(patch.object(c.p,'memory',return_value={}))
                    stack.enter_context(patch.object(torch,'autocast',side_effect=lambda *a,**k:nullcontext()))
                    stack.enter_context(patch.object(torch.cuda,'empty_cache'))
                    stack.enter_context(patch.object(torch.cuda,'synchronize'))
                    weights=stack.enter_context(patch.object(c.p,'load_weights',wraps=c.p.load_weights))
                    adam=stack.enter_context(patch.object(c.p,'restore_adam',wraps=c.p.restore_adam))
                    constructions=Tiny.constructions
                    self.r.reload_final(output,preflight,report)
                    self.assertEqual(Tiny.constructions-constructions,1)
                    self.assertEqual(weights.call_count,1)
                    self.assertEqual(adam.call_count,1 if preflight else 0)
                    self.assertEqual(report['reload_max_absolute_error'],0.)
                    self.assertEqual(report['selected_checkpoint_update'],None if preflight else 5000)
                    self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__': unittest.main()