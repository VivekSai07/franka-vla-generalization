"""CPU --check, exactly two scratch updates, or fresh5000; main agent launches GPU."""
import argparse
import gc
import json
import os
import random
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch

from experiments.xvla_trial.assets import safe_error
from . import contract as c


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--train',action='store_true')
    return parser.parse_args(argv)


def require_disk(output,gib):
    if shutil.disk_usage(output).free < gib*1024**3:
        raise RuntimeError(f'Require {gib} GiB free; no deletion/fallback')


def require_resources(output,preflight):
    require_disk(output,15 if preflight else 60)
    free,total=torch.cuda.mem_get_info()
    if free < 28*1024**3:
        raise RuntimeError('Require 28 GiB GPU free; do not interfere with other workloads')
    return dict(disk_free=shutil.disk_usage(output).free,gpu_free=free,gpu_total=total,
                previous_peak_gb=21.34)


def failure(output,report,exc):
    report.update(status='stopped',**safe_error(exc),retry_allowed=False,autotune=False)
    if torch.cuda.is_initialized(): report['failure_memory']=c.p.memory()
    try: c.write_json(output/'result.json',report)
    except OSError: pass  # Preserve original failure on disk exhaustion; never clean/retry.
    print(json.dumps(dict(status='stopped',stage=report.get('stage'),**safe_error(exc))),flush=True)


def statistics_json(stats):
    return {key:{name:value.tolist() for name,value in values.items()} for key,values in stats.items()}


def processor_check(ds,selected,stats):
    """Real CPU processor + native input preparation; no backbone allocation/forward."""
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
    config=c.make_config('cpu')
    policy=XVLAPolicy.__new__(XVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config,policy.model=config,SimpleNamespace(dim_proprio=20,dim_action=20)
    pre,post=make_xvla_pre_post_processors(config,stats)
    raw=c.collate_rows([ds[i] for i in selected])
    batch=pre(raw)
    evidence=c.check_model_inputs(policy,batch,raw['observation.state'])
    if not torch.allclose(post(batch['action']),raw['action'],rtol=1e-5,atol=1e-5):
        raise ValueError('Physical action normalization roundtrip failed')
    return {**evidence,'physical_action_roundtrip':True,'dataset_rows':selected,
            'whole_policy_constructed':False,'backward_executed':False}


def check():
    from experiments.fixed_layout.validate import require_fixed_layout_pass
    c.claim_output(c.ADMISSION)
    report=dict(status='preparing',stage='cpu_semantic_admission')
    try:
        c.write_json(c.CPU_ADMISSION,report)
        before=c.current_binding()
        semantic=require_fixed_layout_pass(c.PARENT)  # ONLY full semantic call in this package.
        ds,starts,stats=c.load_data()
        evidence=processor_check(ds,[starts[0],starts[498],starts[499],starts[-1]],stats)
        if c.current_binding() != before:
            raise ValueError('Source/data/assets changed during CPU admission')
        report=c.cpu_proof(before,semantic,statistics_json(stats),evidence)
        c.validate_cpu(report,before)
        c.write_json(c.CPU_ADMISSION,report)
    except BaseException as exc:
        failure(c.ADMISSION,report,exc)
        raise
    print('CPU data/processor/semantic admission passed; no GPU work',flush=True)


def save_checkpoint(policy,pre,post,optimizer,output,update,final,report):
    require_disk(output,12 if final else 5)
    checkpoint=output/f'checkpoints/{update:06d}'
    checkpoint.mkdir(parents=True,exist_ok=False)
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    if final: torch.save(optimizer.state_dict(),checkpoint/'optimizer.pt')
    marker=dict(update=update,files=c.inventory(checkpoint),optimizer_saved=final,
                learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},
                completed_microbatches=update*2,sampled_chunks=update*8)
    c.write_json(checkpoint/'checkpoint_result.json',marker)
    report['checkpoints'].append(marker)
    c.write_json(output/'result.json',report)
    return checkpoint


def train(output,preflight,report):
    """Explicit old runner loop copy: new data/binding/counts/resources; no old globals."""
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors

    def progress(**values):
        report.update(values)
        c.write_json(output/'result.json',report)

    ds,starts,stats,binding=c.prepare_data()
    cpu_hash=c.sha256(c.CPU_ADMISSION)
    encoded_stats=statistics_json(stats)
    if not preflight:
        proof=json.loads((c.SCRATCH/'result.json').read_text())
        c.validate_scratch(proof,binding,c.inventory(c.SCRATCH/'checkpoints/000002'),
                           c.sha256(c.SCRATCH/'training_stats.json'),cpu_hash)
        if json.loads((c.SCRATCH/'training_stats.json').read_text()) != encoded_stats:
            raise ValueError('Scratch training statistics changed')
        progress(scratch_result_sha256=c.sha256(c.SCRATCH/'result.json'),scratch_used_for_initialization=False)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required; no fallback')
    resources=require_resources(output,preflight)
    torch.cuda.reset_peak_memory_stats()
    random.seed(c.SEED)
    np.random.seed(c.SEED)
    torch.manual_seed(c.SEED)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    c.write_json(output/'training_stats.json',encoded_stats)
    rows,verified_starts=c.indices(ds.meta.episodes)
    if starts != verified_starts or (len(rows),len(starts)) != (10560,9980):
        raise ValueError('Derived training counts changed')
    progress(binding=binding,cpu_admission_sha256=cpu_hash,resource_admission=resources,
             training_stats_sha256=c.sha256(output/'training_stats.json'),
             training_rows=len(rows),complete_training_chunks=len(starts),checkpoints=[],
             completed_updates=0,stage='fresh_base_load',
             device=dict(name=torch.cuda.get_device_name(),properties=str(torch.cuda.get_device_properties(0))),
             initial_memory=c.p.memory(),base_weights_sha256=c.sha256(c.CACHE/'model/model.safetensors'))
    config=c.make_config('cpu')
    policy=c.p.load_weights(XVLAPolicy(config),c.CACHE/'model/model.safetensors').to('cuda')
    config.device='cuda'
    config.save_pretrained(output/'initial_config')
    pre,post=make_xvla_pre_post_processors(config,stats)
    params=policy.get_optim_params()
    if (len(params) != 901 or len(dict(policy.named_parameters())) != 901
            or sum(x.numel() for x in params.values()) != 879482456
            or set(params) != set(dict(policy.named_parameters()))):
        raise ValueError('Expected all901 named FP32 tensors /879482456 trainable parameters')
    c.p.verify_precision(params)
    optimizer=config.get_optimizer_preset().build(params)
    c.set_learning_rate(optimizer,1)
    grouped=[x for group in optimizer.param_groups for x in group['params']]
    if len(grouped) != 901 or {id(x) for x in grouped} != {id(x) for x in params.values()}:
        raise ValueError('Missing/duplicated optimizer parameter')
    del grouped
    raw=c.collate_rows([ds[i] for i in (starts[0],starts[498],starts[499],starts[-1])])
    checked=pre(raw)
    progress(model_input_precheck=c.check_model_inputs(policy,checked,raw['observation.state']),
             parameter_tensors=len(params),parameter_count=sum(x.numel() for x in params.values()),
             optimizer_groups=[dict(name=g['name'],parameter_tensors=len(g['params']),
                                    weight_decay=g['weight_decay']) for g in optimizer.param_groups],
             after_load_memory=c.p.memory(),status='training',stage='training')
    del raw,checked
    maximum=2 if preflight else 5000
    samples=c.sample_batches(starts)
    began=time.monotonic()
    policy.train()
    with (output/'metrics.jsonl').open('x') as log:
        for update in range(1,maximum+1):
            tick=time.monotonic()
            c.set_learning_rate(optimizer,update)
            optimizer.zero_grad(set_to_none=True)
            losses,indices=[],[]
            for _ in range(2):
                selected=next(samples)
                batch=pre(c.collate_rows([ds[i] for i in selected]))
                if batch['action'].shape != (4,30,8):
                    raise ValueError('Processed microbatch shape changed')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,_=policy(batch)
                c.backward_microbatch(loss)
                losses.append(float(loss.detach()))
                indices.append(selected)
            c.p.verify_precision(params,require_gradients=True)
            norm=torch.nn.utils.clip_grad_norm_(list(params.values()),10.,error_if_nonfinite=True)
            optimizer.step()
            c.p.verify_precision(params,optimizer,require_gradients=True)
            torch.cuda.synchronize()
            row=dict(update=update,loss=sum(losses)/2,grad_norm=float(norm),dataset_rows=indices,
                     learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},
                     seconds=time.monotonic()-tick,memory=c.p.memory(),sampled_chunks=update*8)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            log.flush()
            report['completed_updates']=update
            if update == 1 or update%20 == 0 or update == maximum:
                progress(last_metrics=row,elapsed_seconds=time.monotonic()-began)
                print('TRAIN',json.dumps(row,allow_nan=False),flush=True)
            if update in c.checkpoint_updates(preflight):
                checkpoint=save_checkpoint(policy,pre,post,optimizer,output,update,update==maximum,report)
    sample=ds[starts[0]]
    obs=c.observation({key:sample[key] for key in (*c.OBS,'task')})
    policy.eval()
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(c.SEED)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        expected=post(policy.predict_action_chunk(c.p.process_row(pre,obs))).cpu()
    if expected.shape != (1,30,8) or not torch.isfinite(expected).all():
        raise ValueError('Expected finite real8 prediction before reload')
    torch.save(expected,checkpoint/'reload_reference.pt')
    progress(reload_probe_row=starts[0],after_save_memory=c.p.memory(),stage='release_training_model')
    # Return releases model/Adam/graphs together, BEFORE replacement allocation.


def reload_final(output,preflight,report):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint=output/f'checkpoints/{2 if preflight else 5000:06d}'
    before=c.inventory(checkpoint)
    # Admission before constructing a replacement: light full-file binding, not media/physics.
    ds,starts,stats,binding=c.prepare_data()
    if binding != report['binding'] or c.sha256(c.CPU_ADMISSION) != report['cpu_admission_sha256']:
        raise ValueError('Phase admission changed')
    config=PreTrainedConfig.from_pretrained(checkpoint,local_files_only=True)
    if json.loads((checkpoint/'config.json').read_text()) != json.loads((output/'initial_config/config.json').read_text()):
        raise ValueError('Saved config changed')
    if config.dtype != 'float32' or config.chunk_size != 30 or config.action_feature.shape != (8,):
        raise ValueError('Saved precision/action/chunk contract changed')
    config.device='cpu'
    policy=c.p.load_weights(XVLAPolicy(config),checkpoint/'model.safetensors').to('cuda')
    policy.config.device='cuda'
    report.update(stage='strict_final_reload',strict_reload_fp32_exact=True,
                  adam_reload_fp32_exact=False,full_adam_restore_required=preflight)
    if preflight:
        preset=config.get_optimizer_preset()
        preset.lr=c.learning_rate(2)
        optimizer=c.p.restore_adam(policy.get_optim_params(),preset,checkpoint/'optimizer.pt')
        report.update(adam_reload_fp32_exact=True,after_adam_reload_memory=c.p.memory())
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=str(checkpoint))
    sample=ds[report['reload_probe_row']]
    obs=c.observation({key:sample[key] for key in (*c.OBS,'task')})
    batch=c.p.process_row(pre,obs)
    report['reloaded_input_precheck']=c.check_model_inputs(policy,batch,obs['observation.state'][None])
    expected=torch.load(checkpoint/'reload_reference.pt',map_location='cpu',weights_only=True)
    policy.eval()
    torch.manual_seed(c.SEED)
    tick=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        actual=post(policy.predict_action_chunk(batch)).cpu()
    torch.cuda.synchronize()
    error=float((actual-expected).abs().max())
    if actual.shape != (1,30,8) or not torch.isfinite(actual).all() or not torch.equal(actual,expected):
        raise ValueError('Strict reload prediction must have exactly zero error')
    if (c.admission() != binding or before != c.inventory(checkpoint)
            or c.sha256(c.CPU_ADMISSION) != report['cpu_admission_sha256']
            or c.sha256(output/'training_stats.json') != report['training_stats_sha256']
            or statistics_json(stats) != json.loads((output/'training_stats.json').read_text())):
        raise ValueError('Post-training source/data/cache/checkpoint/statistics changed')
    if not preflight and c.sha256(c.SCRATCH/'result.json') != report['scratch_result_sha256']:
        raise ValueError('Scratch proof changed during main')
    if [m['update'] for m in report['checkpoints']] != c.checkpoint_updates(preflight):
        raise ValueError('Checkpoint selection/schedule changed')
    for marker in report['checkpoints']:
        directory=output/f"checkpoints/{marker['update']:06d}"
        c.verify_listed(marker['files'],root=directory)
        if json.loads((directory/'checkpoint_result.json').read_text()) != marker:
            raise ValueError('Checkpoint metadata changed')
    report.update(status='complete',stage='training_complete',reload_max_absolute_error=error,
                  inference_seconds=time.monotonic()-tick,final_memory=c.p.memory(),
                  final_data_binding_passed=True,final_checkpoint_files=before,
                  selected_checkpoint_update=None if preflight else 5000,rollouts=0)
    c.write_json(output/'result.json',report)


def main(argv=None):
    args=parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    torch.set_num_threads(4)
    if args.check:
        check()
        return
    output=c.SCRATCH if args.preflight else c.OUTPUT
    c.claim_output(output)
    report=dict(status='preparing',stage='admission',**c.recipe(args.preflight))
    try:
        c.write_json(output/'result.json',report)
        train(output,args.preflight,report)
        reload_final(output,args.preflight,report)
    except BaseException as exc:
        failure(output,report,exc)
        raise SystemExit(1) from None
    gc.collect()
    torch.cuda.empty_cache()
    if not args.preflight:
        from .evaluate import run
        run(output)
    print(json.dumps(dict(status='complete',purpose=report['purpose'],completed_updates=report['completed_updates'])),flush=True)


if __name__=='__main__': main()