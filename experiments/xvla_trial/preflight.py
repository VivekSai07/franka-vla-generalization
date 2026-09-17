"""CPU inspection or explicitly approved two-update scratch preflight; no main training."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import shutil
import time

import torch

from . import contract as c
from .assets import claim_directory, require_assets, safe_error, sha256, write_json

CPU_REPORT = Path(__file__).parent/'CPU_REPORT_REVIEWED.json'


def make_config(device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig
    cfg = PreTrainedConfig.from_pretrained(c.CACHE/'model', local_files_only=True)
    if (not isinstance(cfg, XVLAConfig) or cfg.chunk_size != 30 or cfg.max_state_dim != 20
            or cfg.max_action_dim != 20 or cfg.num_image_views != 3
            or tuple(cfg.resize_imgs_with_padding) != (224,224)):
        raise ValueError('Unexpected pinned base architecture')
    cfg.input_features = {k: PolicyFeature(type=FeatureType.STATE if k == 'observation.state'
                         else FeatureType.VISUAL, shape=shape) for k, shape in c.OBS.items()}
    cfg.output_features = {'action': PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    cfg.action_mode, cfg.dtype, cfg.device = 'auto', 'float32', device
    cfg.num_image_views, cfg.empty_cameras, cfg.n_action_steps = 3, 0, 10
    cfg.freeze_vision_encoder = cfg.freeze_language_encoder = cfg.push_to_hub = False
    cfg.train_policy_transformer = cfg.train_soft_prompts = True
    cfg.tokenizer_name, cfg.tokenizer_max_length = str(c.CACHE/'tokenizer'), 64
    cfg.domain_feature_key = 'domain_id'
    cfg.validate_features()
    return cfg


def policy_row(row, training):
    ids = c.TRAIN if training else c.TRAIN+c.VAL
    if (int(row['episode_index']) not in ids or row['task'] != c.TASK
            or {k for k in row if k.startswith('observation.')} != set(c.OBS)):
        raise ValueError('Unapproved episode/task/observation')
    for key, shape in {**c.OBS, 'action': (30,8), 'action_is_pad': (30,)}.items():
        x = row[key]
        if tuple(x.shape) != shape or not torch.isfinite(x).all():
            raise ValueError('Nonfinite or wrong-shaped input')
    if row['action_is_pad'].dtype != torch.bool or row['action_is_pad'].any():
        raise ValueError('Stock XVLA requires complete unpadded chunks here')
    for key in c.CAMERAS:
        if row[key].dtype != torch.float32 or row[key].min() < 0 or row[key].max() > 1:
            raise ValueError('Expected decoded float32 RGB in [0,1]')
    return {k: row[k] for k in (*c.OBS, 'task', 'action', 'action_is_pad')}


def process_row(pre, row):
    # Stock AddBatchDimension leaves a [T,A] chunk alone (it assumes [B,A]).
    from torch.utils.data import default_collate
    return pre(default_collate([row]))


def selected_stats(rows, indices, task_index=0):
    values = {k: [] for k in ('observation.state', 'action')}
    for i in indices:
        row = rows[i]
        if (int(row['episode_index']) not in c.TRAIN or int(row['episode_index']) != i//528
                or int(row['task_index']) != task_index):
            raise ValueError('Statistics must contain selected training rows only')
        for key, size in [('observation.state',9), ('action',8)]:
            x = torch.as_tensor(row[key]).float()
            if x.shape != (size,) or not torch.isfinite(x).all():
                raise ValueError('Invalid raw training row')
            values[key].append(x)
    return {k: {'mean': (x := torch.stack(v)).mean(0), 'std': x.std(0, unbiased=False),
                'min': x.amin(0), 'max': x.amax(0)} for k,v in values.items()}


def verify_precision(params, optimizer=None, require_gradients=False):
    for p in params.values():
        if not p.requires_grad or p.dtype != torch.float32 or not torch.isfinite(p).all():
            raise ValueError('All parameters must be finite FP32 and trainable')
        if require_gradients and (p.grad is None or p.grad.dtype != torch.float32
                                  or not torch.isfinite(p.grad).all()):
            raise ValueError('Missing/nonfinite/non-FP32 gradient')
        if optimizer is not None:
            state = optimizer.state[p]
            for key in ('exp_avg', 'exp_avg_sq'):
                if (key not in state or state[key].dtype != torch.float32
                        or not torch.isfinite(state[key]).all()):
                    raise ValueError('Missing/nonfinite/non-FP32 Adam moment')


def load_weights(policy, path):
    """XVLA-specific FP32 construction followed by strict safetensors load; no dtype downcast."""
    from safetensors import safe_open
    from safetensors.torch import load_model
    with safe_open(path, framework='pt', device='cpu') as stored:
        if any(stored.get_slice(n).get_dtype() != 'F32' for n in stored.keys()):
            raise ValueError('Expected FP32 checkpoint storage')
    load_model(policy, str(path), strict=True, device='cpu')
    verify_precision(dict(policy.named_parameters()))
    with safe_open(path, framework='pt', device='cpu') as stored:
        tensors = policy.state_dict()
        for name in stored.keys():
            if not torch.equal(tensors[name].detach().cpu(), stored.get_tensor(name)):
                raise ValueError('Loaded checkpoint differs from FP32 stored tensors')
    return policy


def restore_adam(params, preset, path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    optimizer = preset.build(params)
    if len(saved['param_groups']) != len(optimizer.param_groups):
        raise ValueError('Saved optimizer groups changed')
    for original, group in zip(optimizer.param_groups, saved['param_groups'], strict=True):
        if any(original[k] != group[k] for k in ('name', 'lr', 'betas', 'eps', 'weight_decay')):
            raise ValueError('Saved optimizer recipe changed')
    for state in saved['state'].values():
        for key in ('exp_avg', 'exp_avg_sq'):
            if state[key].dtype != torch.float32 or not torch.isfinite(state[key]).all():
                raise ValueError('Saved Adam moments must remain finite FP32')
    optimizer.load_state_dict(saved)
    verify_precision(params, optimizer)
    for group, stored_group in zip(optimizer.param_groups, saved['param_groups'], strict=True):
        for p, index in zip(group['params'], stored_group['params'], strict=True):
            for key in ('exp_avg', 'exp_avg_sq', 'step'):
                if not torch.equal(optimizer.state[p][key].cpu(), saved['state'][index][key]):
                    raise ValueError('Restored Adam state differs from stored tensors')
    return optimizer


def source_hashes():
    # Bind reused modules and existing sources without touching any bound experiment.
    folders = ('training', 'validation', 'data_collection', 'src', 'experiments')
    paths = sorted(p for folder in folders for p in (c.ROOT/folder).rglob('*.py'))
    result = {str(p.relative_to(c.ROOT)): sha256(p) for p in paths}
    import lerobot
    installed = Path(lerobot.__file__).parent
    bound = list((installed/'policies/xvla').glob('*.py'))
    bound += list((installed/'processor').glob('*.py'))
    bound += [installed/'policies/pretrained.py', installed/'optim/optimizers.py']
    result.update({f'installed/lerobot/{p.relative_to(installed)}': sha256(p) for p in bound})
    return result


def prepare_data():
    from data_collection.benchmark import require_complete
    from validation.gates import require_replay_pass
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    manifest = require_complete(c.DATA)
    c.select_entries(manifest)
    gate = require_replay_pass(c.DATA, c.ROOT)
    ds = LeRobotDataset(repo_id=c.DATA.name, root=c.DATA,
                        delta_timestamps={'action': [i/30 for i in range(30)]})
    features = dataset_to_policy_features(ds.features)
    if ({k: tuple(v.shape) for k,v in features.items() if k.startswith('observation.')} != c.OBS
            or tuple(features['action'].shape) != (8,)):
        raise ValueError('Dataset real feature schema mismatch')
    rows, starts = c.indices(ds.meta.episodes)
    ds._ensure_hf_dataset_loaded()
    stats = selected_stats(ds.hf_dataset, rows, int(ds.meta.tasks.loc[c.TASK, 'task_index']))
    return ds, starts, stats, gate


def memory():
    free, total = torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                device_free=free, device_total=total)


def inspect_cpu():
    """Actual full strict CPU load plus vision-token check; no backward or optimizer step."""
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
    from transformers import AutoTokenizer
    assets, before = require_assets(), source_hashes()
    ds, starts, stats, gate = prepare_data()
    config = make_config('cpu')
    policy = load_weights(XVLAPolicy(config), c.CACHE/'model/model.safetensors')
    pre, _ = make_xvla_pre_post_processors(config, stats)
    sample = process_row(pre, policy_row(ds[starts[0]], training=True))
    inputs = policy._build_model_inputs(sample)
    policy.eval()
    with torch.inference_mode():
        vision_shape = list(policy.model.vlm._encode_image(inputs['image_input'][:,0]).shape)
    tokens = AutoTokenizer.from_pretrained(config.tokenizer_name, local_files_only=True)(c.TASK)['input_ids']
    sequence = config.chunk_size + config.tokenizer_max_length + config.num_image_views*vision_shape[1]
    if len(tokens) > 64 or sequence > config.max_len_seq:
        raise ValueError('Language/vision/action sequence exceeds approved capacity')
    count = sum(p.numel() for p in policy.parameters())
    from validation.gates import require_replay_pass
    if gate != require_replay_pass(c.DATA, c.ROOT) or before != source_hashes():
        raise ValueError('Dataset or source changed during inspection')
    return dict(status='cpu_compatible_gpu_unmeasured', assets=assets,
                source_hashes=before, dataset_fingerprint=gate['dataset_fingerprint'],
                train_episodes=list(c.TRAIN), validation_episodes=list(c.VAL), training_rows=4752,
                complete_chunks=len(starts), parameter_count=count, trainable_count=count,
                fp32_parameter_gradient_adam_bytes_lower_bound=count*16,
                strict_full_checkpoint_load=True, checkpoint_dtype='F32',
                image_mask=inputs['image_mask'].tolist(), vision_shape=vision_shape,
                instruction_tokens=len(tokens), transformer_sequence_tokens=sequence,
                training_stats={k:{n:v.tolist() for n,v in s.items()} for k,s in stats.items()},
                gpu_executed=False, optimizer_updates=0)


def run_gpu(report):
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from validation.gates import require_replay_pass
    assets, sources = require_assets(), source_hashes()
    cpu = json.loads(CPU_REPORT.read_text())
    ds, starts, stats, gate = prepare_data()
    if (cpu['status'] != 'cpu_compatible_gpu_unmeasured' or cpu['assets'] != assets
            or cpu['source_hashes'] != sources or cpu['dataset_fingerprint'] != gate['dataset_fingerprint']):
        raise ValueError('Matching CPU compatibility inspection required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 support required; no fallback')
    if shutil.disk_usage(c.OUTPUT).free < 15*1024**3:
        raise RuntimeError('Less than 15 GiB checkpoint headroom')
    torch.cuda.reset_peak_memory_stats()
    report.update(device=dict(name=torch.cuda.get_device_name(), index=torch.cuda.current_device(),
                              visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                              nvidia_visible_devices=os.environ.get('NVIDIA_VISIBLE_DEVICES'),
                              properties=str(torch.cuda.get_device_properties(0))),
                  source_hashes=sources, assets=assets, dataset_fingerprint=gate['dataset_fingerprint'],
                  initial_memory=memory(), updates=[])
    write_json(c.OUTPUT/'result.json', report)
    write_json(c.OUTPUT/'training_stats.json', {k:{n:v.tolist() for n,v in s.items()} for k,s in stats.items()})
    config = make_config('cpu')
    policy = load_weights(XVLAPolicy(config), c.CACHE/'model/model.safetensors').to('cuda')
    config.device = 'cuda'
    config.save_pretrained(c.OUTPUT/'initial_config')
    pre, post = make_xvla_pre_post_processors(config, stats)
    params = policy.get_optim_params()
    if len(params) != len(dict(policy.named_parameters())):
        raise ValueError('Full-model fine-tuning required')
    optimizer = config.get_optimizer_preset().build(params)
    groups = {g['name']:g['lr'] for g in optimizer.param_groups}
    if abs(groups['vlm']/groups['other']-.1) > 1e-12:
        raise ValueError('VLM LR ratio must remain 0.1')
    report.update(parameter_count=sum(p.numel() for p in params.values()), learning_rates=groups,
                  after_load_memory=memory())
    with (c.OUTPUT/'metrics.jsonl').open('x') as log:
        for update in (1,2):
            torch.cuda.synchronize()
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            # Fixed first/last full training chunks, microbatch1; no validation sampling.
            index = starts[0 if update == 1 else -1]
            batch = process_row(pre, policy_row(ds[index], training=True))
            policy.train()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            verify_precision(params, require_gradients=True)
            norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 10., error_if_nonfinite=True)
            optimizer.step()
            verify_precision(params, optimizer, require_gradients=True)
            torch.cuda.synchronize()
            row = dict(update=update, dataset_row=index, loss=float(loss.detach()), grad_norm=float(norm),
                       seconds=time.monotonic()-tick, memory=memory())
            log.write(json.dumps(row, allow_nan=False)+'\n')
            log.flush()
            report['updates'].append(row)
            write_json(c.OUTPUT/'result.json', report)
    # Predict from observations only with fixed RNG; the policy ignores its noise argument.
    observation = {k:v for k,v in policy_row(ds[starts[0]], training=True).items() if k in c.OBS or k == 'task'}
    policy.eval()
    torch.manual_seed(c.SEED)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = post(policy.predict_action_chunk(process_row(pre, observation))).cpu()
    if expected.shape != (1,30,8) or not torch.isfinite(expected).all():
        raise ValueError('Invalid real8 prediction')
    checkpoint = c.OUTPUT/'checkpoint'
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    torch.save(optimizer.state_dict(), checkpoint/'optimizer.pt')
    report['after_save_memory'] = memory()
    # Release the first model/Adam before constructing its replacement; no duplicate GPU model.
    del policy, optimizer, params, batch, loss, pre, post
    gc.collect()
    torch.cuda.empty_cache()
    from lerobot.configs.policies import PreTrainedConfig
    restored_config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    restored_config.device = 'cpu'
    if restored_config.dtype != 'float32' or tuple(restored_config.action_feature.shape) != (8,):
        raise ValueError('Saved XVLA precision/feature schema changed')
    restored = load_weights(XVLAPolicy(restored_config), checkpoint/'model.safetensors').to('cuda')
    restored.config.device = 'cuda'
    restored_adam = restore_adam(restored.get_optim_params(), restored.config.get_optimizer_preset(),
                                 checkpoint/'optimizer.pt')
    report['after_adam_reload_memory'] = memory()
    del restored_adam
    gc.collect()
    torch.cuda.empty_cache()
    rpre, rpost = make_pre_post_processors(restored.config, pretrained_path=str(checkpoint))
    restored.eval()
    torch.manual_seed(c.SEED)
    tick = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        actual = rpost(restored.predict_action_chunk(process_row(rpre, observation))).cpu()
    torch.cuda.synchronize()
    inference_seconds = time.monotonic()-tick
    error = float((actual-expected).abs().max())
    if actual.shape != (1,30,8) or not torch.isfinite(actual).all() or error > 1e-5:
        raise ValueError('Strict reload prediction preservation failed')
    if require_replay_pass(c.DATA, c.ROOT) != gate or source_hashes() != sources or require_assets() != assets:
        raise ValueError('Final data/source/asset binding failed')
    report.update(status='complete', completed_updates=2, strict_reload_fp32_exact=True,
                  reload_max_absolute_error=error, inference_seconds=inference_seconds,
                  adam_reload_fp32_exact=True, optimizer_sha256=sha256(checkpoint/'optimizer.pt'),
                  final_memory=memory(), final_data_binding_passed=True,
                  checkpoint_sha256=sha256(checkpoint/'model.safetensors'),
                  main_training_authorized=False, rollouts=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inspect', action='store_true', help='CPU-only strict compatibility inspection')
    mode.add_argument('--approved-two-update-preflight', action='store_true', help='Requires explicit post-review GPU authorization')
    args = parser.parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    torch.set_num_threads(8)
    random.seed(c.SEED)
    torch.manual_seed(c.SEED)
    if args.inspect:
        output = CPU_REPORT
        if output.exists():
            raise FileExistsError('Preserve existing CPU inspection report')
        report = inspect_cpu()
        write_json(output, report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('source_hashes','assets','training_stats')}))
        return
    claim_directory(c.OUTPUT)
    report = dict(status='preparing', purpose='xvla_two_update_preflight', seed=c.SEED,
                  max_optimizer_updates=2, microbatch=1, accumulation=1,
                  precision='fp32_trainable_and_adam_bf16_autocast', autotune=False, resume=False)
    try:
        write_json(c.OUTPUT/'result.json', report)
        run_gpu(report)
    except Exception as exc:
        report.update(status='stopped', **safe_error(exc))
        if torch.cuda.is_initialized():
            report['failure_memory'] = memory()
        write_json(c.OUTPUT/'result.json', report)
        print(json.dumps(dict(status='stopped', **safe_error(exc))))
        raise SystemExit(1) from None
    write_json(c.OUTPUT/'result.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()