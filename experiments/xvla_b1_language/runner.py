"""CPU admission, exact two-update scratch, or fresh matched-exposure X-VLA fine-tune. No automatic resume."""
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

EXPECTED_PARAM_TENSORS = 901
EXPECTED_PARAM_NUMEL = 879482456


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('check', 'preflight', 'train'):
        mode.add_argument('--'+name, action='store_true')
    return parser.parse_args(argv)


def require_disk(output, gib):
    if shutil.disk_usage(output).free < gib*1024**3:
        raise RuntimeError(f'Require {gib} GiB free; no deletion/fallback')


def require_resources(output, preflight):
    require_disk(output, 15 if preflight else 60)
    free, total = torch.cuda.mem_get_info()
    if free < 28*1024**3:
        raise RuntimeError('Require 28 GiB GPU free; do not interfere with other workloads')
    return dict(disk_free=shutil.disk_usage(output).free, gpu_free=free, gpu_total=total)


def failure(output, report, exc):
    report.update(status='failed', **safe_error(exc), error=str(exc), retries=False)
    if torch.cuda.is_initialized():
        report['failure_memory'] = c.p.memory()
    try:
        with (output/'failure.json').open('x') as stream:
            json.dump(report, stream, allow_nan=False)
        c.write_json(output/'result.json', report)
    except (OSError, ValueError):
        pass  # Preserve original failure, never cleanup/retry.
    print(json.dumps(dict(status='failed', stage=report.get('stage'), **safe_error(exc))), flush=True)


def statistics_json(stats):
    return {key: {name: value.tolist() for name, value in values.items()} for key, values in stats.items()}


def processor_check(ds, selected, stats):
    """Real CPU processor + native input preparation; no backbone allocation/forward."""
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
    config = c.make_config('cpu')
    policy = XVLAPolicy.__new__(XVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config, policy.model = config, SimpleNamespace(dim_proprio=20, dim_action=20)
    pre, post = make_xvla_pre_post_processors(config, stats)
    raw = c.collate_rows([ds[i] for i in selected])
    batch = pre(raw)
    evidence = c.check_model_inputs(policy, batch, raw['observation.state'])
    if not torch.allclose(post(batch['action']), raw['action'], rtol=1e-5, atol=1e-5):
        raise ValueError('Physical action normalization roundtrip failed')
    return {**evidence, 'physical_action_roundtrip': True, 'dataset_rows': selected,
            'whole_policy_constructed': False, 'backward_executed': False}


def check():
    from experiments.b1_language_data.validate import require_broad_workspace_pass
    c.claim_output(c.ADMISSION)
    report = dict(status='preparing', stage='full_cpu_semantic_admission')
    try:
        c.write_json(c.CPU_ADMISSION, report)
        before = c.current_binding()
        semantic = require_broad_workspace_pass(c.PARENT)  # Only full semantic audit call this phase.
        ds, starts, stats = c.load_data()
        evidence = processor_check(ds, [starts[0], starts[498], starts[499], starts[-1]], stats)
        if c.current_binding() != before:
            raise ValueError('Source/data/assets changed during CPU admission')
        report = c.cpu_proof(before, semantic, statistics_json(stats), evidence)
        c.validate_cpu(report, before)
        c.write_json(c.CPU_ADMISSION, report)
    except BaseException as exc:
        failure(c.ADMISSION, report, exc)
        raise
    print('CPU data/processor/semantic admission passed; no GPU work', flush=True)


def load_fresh_policy(config):
    """Fresh X-VLA base with strict safetensors FP32 load; verified before any Adam allocation."""
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    policy = c.p.load_weights(XVLAPolicy(config), c.CACHE/'model/model.safetensors')
    params = policy.get_optim_params()
    if (len(params) != EXPECTED_PARAM_TENSORS or len(dict(policy.named_parameters())) != EXPECTED_PARAM_TENSORS
            or sum(x.numel() for x in params.values()) != EXPECTED_PARAM_NUMEL
            or set(params) != set(dict(policy.named_parameters()))):
        raise ValueError('Expected all 901 named FP32 tensors / 879482456 trainable parameters')
    return policy, params


def adam_roundtrip(optimizer, path, update):
    """Byte-verify a saved X-VLA optimizer.state_dict roundtrip on CPU."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if len(saved['param_groups']) != len(optimizer.param_groups):
        raise ValueError('Adam parameter groups changed')
    current = optimizer.state_dict()
    if saved['param_groups'] != current['param_groups'] or set(saved['state']) != set(current['state']):
        raise ValueError('Adam parameter groups/state membership changed')
    for key, values in saved['state'].items():
        if set(values) != set(current['state'][key]):
            raise ValueError('Missing Adam state')
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                if value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError('Invalid saved FP32 Adam')
                if not torch.equal(value, current['state'][key][name].cpu()):
                    raise ValueError('Adam state bytes changed')
    # CPU-side load_state_dict roundtrip on empty parameters — no GPU allocation.
    parameters = [torch.nn.Parameter(torch.empty_like(p, device='cpu'))
                  for group in optimizer.param_groups for p in group['params']]
    restored = torch.optim.AdamW(parameters, lr=c.learning_rate(1))
    # Restore identical group structure (name/lr/betas/eps/weight_decay) before load_state_dict.
    restored.param_groups = []
    offset = 0
    for original, group in zip(optimizer.param_groups, saved['param_groups'], strict=True):
        count = len(group['params'])
        restored.add_param_group({**{k: original[k] for k in original if k != 'params'},
                                  'params': parameters[offset:offset+count]})
        offset += count
    restored.load_state_dict(saved)
    roundtrip = restored.state_dict()
    if roundtrip['param_groups'] != saved['param_groups']:
        raise ValueError('Adam group reload mismatch')
    for key, values in saved['state'].items():
        for name, value in values.items():
            if isinstance(value, torch.Tensor) and not torch.equal(value, roundtrip['state'][key][name]):
                raise ValueError('Adam reload mismatch')
    return dict(adam_reload_fp32_exact=True, update=update, training_state_sha256=c.sha256(path))


def _reload_and_verify(checkpoint, expected):
    """Local port of the X-VLA strict FP32 reload with byte-exact prediction equality."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    if json.loads((checkpoint/'config.json').read_text()).get('dtype') != 'float32':
        raise ValueError('Saved precision must remain float32')
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = 'cpu'
    if config.dtype != 'float32' or config.chunk_size != 30 or tuple(config.action_feature.shape) != (8,):
        raise ValueError('Saved dtype/chunk/action feature changed')
    reloaded = c.p.load_weights(XVLAPolicy(config), checkpoint/'model.safetensors').to('cuda')
    reloaded.config.device = 'cuda'
    reloaded.eval()
    rpre, rpost = make_pre_post_processors(reloaded.config, pretrained_path=str(checkpoint))
    torch.manual_seed(c.SEED)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        actual = rpost(reloaded.predict_action_chunk(rpre(expected['observation']))).cpu()
    if actual.shape != (1, 30, 8) or not torch.isfinite(actual).all():
        raise ValueError('Invalid reload prediction shape/values')
    if not torch.equal(actual, expected['prediction']):
        raise ValueError('Strict reload prediction must match training-time prediction bytewise')
    error = float((actual-expected['prediction']).abs().max())
    del reloaded, rpre, rpost
    gc.collect()
    torch.cuda.empty_cache()
    return error


def save_checkpoint(policy, pre, post, optimizer, ds, probe, output, update, final):
    """Local port: step_ prefix, per-checkpoint FP32 byte-exact reload, optimizer only on final."""
    require_disk(output, 12 if final else 5)
    checkpoint = output/'checkpoints'/c.checkpoint_directory(update)
    checkpoint.mkdir(parents=True, exist_ok=False)
    was_training = policy.training
    policy.eval()
    sample = ds[probe]
    obs = c.observation({key: sample[key] for key in (*c.OBS, 'task')})
    torch.manual_seed(c.SEED)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = post(policy.predict_action_chunk(c.p.process_row(pre, obs))).cpu()
    if expected.shape != (1, 30, 8) or not torch.isfinite(expected).all():
        raise ValueError('Invalid pre-save prediction')
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    if final:
        torch.save(optimizer.state_dict(), checkpoint/'training_state.pt')
    torch.cuda.synchronize()
    error = _reload_and_verify(checkpoint, {'observation': obs, 'prediction': expected})
    policy.train(was_training)
    marker = {'status': 'complete', 'update': update, 'reload_fp32_weights_exact': True,
              'reload_max_action_error': error, 'model_sha256': c.sha256(checkpoint/'model.safetensors'),
              'optimizer_saved': final,
              'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups},
              'completed_microbatches': update*2, 'sampled_chunks': update*8}
    c.write_json(checkpoint/'checkpoint_result.json', marker)
    print('CHECKPOINT', json.dumps(marker), flush=True)
    return marker


def train(output, preflight, report):
    """Port of the xvla_fixed_layout runner adapted to the B1 language-grounding dataset only."""
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors

    def progress(**fields):
        report.update(fields)
        c.write_json(output/'result.json', report)

    ds, starts, stats, binding = c.prepare_data()
    encoded_stats, cpu_hash = statistics_json(stats), c.sha256(c.CPU_ADMISSION)
    scratch_hash = None
    if not preflight:
        scratch_hash = c.sha256(c.SCRATCH/'result.json')
        proof = json.loads((c.SCRATCH/'result.json').read_text())
        c.validate_scratch(proof, binding, c.SCRATCH, encoded_stats, cpu_hash)
        if json.loads((c.SCRATCH/'training_stats.json').read_text()) != encoded_stats:
            raise ValueError('Scratch training statistics changed')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required; no fallback')
    resources = require_resources(output, preflight)
    torch.cuda.reset_peak_memory_stats()
    random.seed(c.SEED)
    np.random.seed(c.SEED)
    torch.manual_seed(c.SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    c.write_json(output/'training_stats.json', encoded_stats)
    rows, verified = c.indices(ds.meta.episodes)
    if starts != verified or (len(rows), len(starts)) != (c.TRAINING_ROWS, c.COMPLETE_STARTS):
        raise ValueError('Derived training counts changed')
    progress(status='training', stage='fresh_base_load', binding=binding, cpu_admission_sha256=cpu_hash,
             scratch_result_sha256=scratch_hash, scratch_used_for_initialization=False,
             training_rows=len(rows), complete_training_chunks=len(starts),
             resource_admission=resources,
             training_stats_sha256=c.sha256(output/'training_stats.json'),
             checkpoints=[], checkpoint_files={}, completed_updates=0, rollouts=0,
             device=dict(name=torch.cuda.get_device_name(),
                         properties=str(torch.cuda.get_device_properties(0))),
             initial_memory=c.p.memory(),
             base_weights_sha256=c.sha256(c.CACHE/'model/model.safetensors'))
    config = c.make_config('cpu')
    policy, params = load_fresh_policy(config)
    policy = policy.to('cuda')
    config.device = 'cuda'
    config.save_pretrained(output/'initial_config')
    c.p.verify_precision(params)
    pre, post = make_xvla_pre_post_processors(config, stats)
    optimizer = config.get_optimizer_preset().build(params)
    c.set_learning_rate(optimizer, 1)
    grouped = [x for group in optimizer.param_groups for x in group['params']]
    if len(grouped) != EXPECTED_PARAM_TENSORS or {id(x) for x in grouped} != {id(x) for x in params.values()}:
        raise ValueError('Missing/duplicated optimizer parameter')
    del grouped
    raw = c.collate_rows([ds[i] for i in (starts[0], starts[498], starts[499], starts[-1])])
    checked = pre(raw)
    progress(model_input_precheck=c.check_model_inputs(policy, checked, raw['observation.state']),
             parameter_tensors=len(params), parameter_count=sum(x.numel() for x in params.values()),
             optimizer_groups=[dict(name=g['name'], parameter_tensors=len(g['params']),
                                    weight_decay=g['weight_decay']) for g in optimizer.param_groups],
             after_load_memory=c.p.memory(), stage='training')
    del raw, checked
    samples = c.sample_batches(starts)
    began = time.monotonic()
    policy.train()
    # Horizon comes from the contract; hardcoding it here silently truncated a 16700-update run at 5000.
    maximum = 2 if preflight else c.recipe(False)['max_optimizer_updates']
    with (output/'metrics.jsonl').open('x') as log:
        for update in range(1, maximum+1):
            tick = time.monotonic()
            c.set_learning_rate(optimizer, update)
            optimizer.zero_grad(set_to_none=True)
            losses, indices = [], []
            for _ in range(2):
                selected = next(samples)
                batch = pre(c.collate_rows([ds[i] for i in selected]))
                if batch['action'].shape != (4, 30, 8):
                    raise ValueError('Processed microbatch shape changed')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, _ = policy(batch)
                c.backward_microbatch(loss)
                losses.append(float(loss.detach()))
                indices.append(selected)
            c.p.verify_precision(params, require_gradients=True)
            norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 10., error_if_nonfinite=True)
            optimizer.step()
            c.p.verify_precision(params, optimizer, require_gradients=True)
            torch.cuda.synchronize()
            row = dict(update=update, loss=sum(losses)/2, grad_norm=float(norm),
                       dataset_rows=indices,
                       learning_rates={g['name']: g['lr'] for g in optimizer.param_groups},
                       seconds=time.monotonic()-tick, memory=c.p.memory(), sampled_chunks=update*8)
            log.write(json.dumps(row, allow_nan=False)+'\n')
            log.flush()
            report['completed_updates'] = update
            if update == 1 or update % 20 == 0 or update == maximum:
                progress(last_metrics=row, elapsed_seconds=time.monotonic()-began)
                print('TRAIN', json.dumps(row, allow_nan=False), flush=True)
            if update in c.checkpoint_updates(preflight):
                marker = save_checkpoint(policy, pre, post, optimizer, ds, starts[0], output, update,
                                         final=(update == maximum))
                c.require_marker(marker, update)
                key = c.checkpoint_directory(update)
                directory = output/'checkpoints'/key
                if preflight and update == maximum:
                    evidence = adam_roundtrip(optimizer, directory/'training_state.pt', update)
                    c.write_json(directory/'adam_reload.json', evidence)
                report['checkpoints'].append(marker)
                report['checkpoint_files'][key] = c.inventory(directory)
                progress(startup_save_reload_passed=True)
    if c.admission() != binding or c.sha256(c.CPU_ADMISSION) != cpu_hash:
        raise ValueError('Post-training dataset/source/cache/CPU proof changed')
    if c.sha256(output/'training_stats.json') != report['training_stats_sha256']:
        raise ValueError('Training statistics bytes changed')
    if not preflight and c.sha256(c.SCRATCH/'result.json') != scratch_hash:
        raise ValueError('Scratch proof changed during main')
    c.verify_checkpoints(report, output, preflight)
    if not preflight:
        final_config = json.loads((output/'checkpoints'/c.checkpoint_directory(maximum)/'config.json').read_text())
        initial_config = json.loads((output/'initial_config/config.json').read_text())
        if final_config != initial_config:
            raise ValueError('Final X-VLA config differs from initial saved config')
    progress(status='complete', stage='training_complete', completed_updates=maximum,
             final_data_binding_passed=True,
             selected_checkpoint_update=None if preflight else maximum,
             adam_reload_fp32_exact=preflight, rollouts=0,
             final_memory=c.p.memory())


def main(argv=None):
    args = parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    torch.set_num_threads(4)
    if args.check:
        check()
        return
    output = c.SCRATCH if args.preflight else c.OUTPUT
    c.claim_output(output)
    report = dict(status='preparing', stage='admission', **c.recipe(args.preflight))
    try:
        c.write_json(output/'result.json', report)
        train(output, args.preflight, report)
    except BaseException as exc:
        failure(output, report, exc)
        raise
    gc.collect()
    torch.cuda.empty_cache()
    # Evaluation/rollout is a separate protocol; --train never launches it.


if __name__ == '__main__':
    main()
