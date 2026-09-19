"""CPU admission, exact two-update scratch, or fresh5000 fine-tune. No automatic resume."""
import argparse
import gc
import json
import os
import random
import shutil
import time

import numpy as np
import torch

from . import contract as c


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('check', 'preflight', 'train'): mode.add_argument('--'+name, action='store_true')
    return parser.parse_args(argv)


def require_disk(output, gib):
    if shutil.disk_usage(output).free < gib*1024**3: raise RuntimeError(f'Require {gib}GiB free; no deletion/fallback')


def require_resources(output):
    require_disk(output, 30)
    free, total = torch.cuda.mem_get_info()
    if free < 12*1024**3: raise RuntimeError('Require12GiB GPU free; preserve other workloads')
    return dict(disk_free=shutil.disk_usage(output).free, gpu_free=free, gpu_total=total)


def failure(output, report, exc):
    # Fixed local code only; no environment/credential contents are included.
    report.update(status='failed', error_type=type(exc).__name__, error=str(exc), retries=False)
    try:
        with (output/'failure.json').open('x') as stream: json.dump(report, stream, allow_nan=False)
        c.write_json(output/'result.json', report)
    except (OSError, ValueError): pass  # Preserve original failure, never cleanup/retry.


def processor_check(ds, selected, stats):
    """Actual8-row SmolVLA processor; no model constructor/forward/backward."""
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    if selected != c.PROCESSOR_ROWS: raise ValueError('CPU processor probe rows changed')
    config = c.make_config('cpu')
    pre, post = make_smolvla_pre_post_processors(config, stats)
    raw = c.collate_rows([ds[i] for i in selected])
    batch = pre(raw)
    c.h.check_batch(batch)
    if batch['observation.language.tokens'].shape != (8, 48): raise ValueError('SmolVLA language schema changed')
    if not torch.allclose(post(batch['action']), raw['action'], rtol=1e-5, atol=1e-5):
        raise ValueError('Physical action normalization roundtrip failed')
    expected = (raw['observation.state']-stats['observation.state']['mean'])/(stats['observation.state']['std']+1e-8)
    torch.testing.assert_close(batch['observation.state'], expected)
    probe = inference_batch(pre, c.observation(ds[selected[0]]))
    if 'action' in probe or 'action_is_pad' in probe: raise ValueError('Inference action leakage')
    return dict(batch_size=8, action_shape=list(batch['action'].shape), state_shape=list(batch['observation.state'].shape),
                token_shape=list(batch['observation.language.tokens'].shape), dataset_rows=selected,
                physical_action_roundtrip=True, observation_only=True,
                whole_policy_constructed=False, backward_executed=False)


def inference_batch(pre, observation):
    batch = pre(observation)
    if batch.get('action') is not None: raise ValueError('Inference processor introduced target action')
    # Native converters can emit an empty action key; never even pass None to model.
    return {key: value for key, value in batch.items() if key != 'action' and key != 'action_is_pad'}


def check():
    from experiments.local_varied_data.validate import require_local_varied_pass
    c.claim_output(c.ADMISSION)
    report = dict(status='preparing', stage='full_cpu_semantic_admission')
    try:
        c.write_json(c.CPU_ADMISSION, report)
        before = c.current_binding()
        semantic = require_local_varied_pass(c.PARENT)  # Only full semantic audit call; later phases rehash.
        ds, starts, stats = c.load_data()
        evidence = processor_check(ds, c.PROCESSOR_ROWS, stats)
        if c.current_binding() != before: raise ValueError('Binding changed during CPU admission')
        report = c.cpu_proof(before, semantic, c.statistics_json(stats), evidence)
        c.validate_cpu(report, before)
        c.write_json(c.CPU_ADMISSION, report)
    except BaseException as exc:
        failure(c.ADMISSION, report, exc)
        raise


def make_optimizer(params):
    return torch.optim.AdamW(params, lr=c.learning_rate(1), betas=(.9, .95), eps=1e-8, weight_decay=1e-10)


def load_fresh_policy(config):
    """Fresh original base only, with verified parameter counts before Adam."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(c.CACHE/'policy', config=config, local_files_only=True, strict=True)
    promotion = c.h.promote_trainable_parameters(policy)
    if promotion['trainable_parameters'] != 99880992 or sum(p.numel() for p in policy.parameters()) != 450046176:
        raise ValueError('Unexpected450046176total/99880992trainable architecture')
    return policy, promotion


def adam_roundtrip(optimizer, path, update):
    """Read trusted own state; verify raw FP32 before Adam's implicit casting."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['update'] != update: raise ValueError('Adam update mismatch')
    state = saved['optimizer']
    current = optimizer.state_dict()
    if state['param_groups'] != current['param_groups'] or set(state['state']) != set(current['state']):
        raise ValueError('Adam parameter groups/state membership changed')
    for key, values in state['state'].items():
        if set(values) != set(current['state'][key]): raise ValueError('Missing Adam state')
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                if value.dtype != torch.float32 or not torch.isfinite(value).all(): raise ValueError('Invalid saved FP32 Adam')
                if not torch.equal(value, current['state'][key][name].cpu()): raise ValueError('Adam state bytes changed')
    # True optimizer.load_state_dict roundtrip on CPU, not a claimed weights-only check.
    parameters = [torch.nn.Parameter(torch.empty_like(p, device='cpu')) for group in optimizer.param_groups for p in group['params']]
    restored = make_optimizer(parameters)
    restored.load_state_dict(state)
    c.h.verify_optimizer_precision(restored)
    roundtrip = restored.state_dict()
    if roundtrip['param_groups'] != state['param_groups']: raise ValueError('Adam group reload mismatch')
    for key, values in state['state'].items():
        for name, value in values.items():
            if not torch.equal(value, roundtrip['state'][key][name]): raise ValueError('Adam reload mismatch')
    return dict(adam_reload_fp32_exact=True, update=update, training_state_sha256=c.sha256(path))


def save_checkpoint(policy, pre, post, optimizer, ds, probe, output, update,
                    generator, epoch_generator_state, epoch, batches_in_epoch):
    """Local port of the pilot saver with step_ prefix; exact FP32 reload agreement enforced."""
    from lerobot.policies.factory import make_pre_post_processors
    from safetensors import safe_open
    from validation.smolvla_smoke import observation_only

    checkpoint = output/'checkpoints'/c.checkpoint_directory(update)
    checkpoint.mkdir(parents=True, exist_ok=False)
    was_training = policy.training
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        policy.eval()
        observation = observation_only(ds[probe])
        torch.manual_seed(123)
        with torch.inference_mode():
            expected = post(policy.predict_action_chunk(pre(observation))).cpu()
        if expected.shape != (1, 50, 8) or not torch.isfinite(expected).all():
            raise ValueError('Invalid pre-save prediction')
        policy.save_pretrained(checkpoint)
        pre.save_pretrained(checkpoint)
        post.save_pretrained(checkpoint)
        reloaded = c.h.load_fp32_checkpoint(checkpoint)
        with safe_open(checkpoint/'model.safetensors', framework='pt', device='cpu') as saved:
            for name, parameter in reloaded.named_parameters():
                if parameter.requires_grad:
                    if parameter.dtype != torch.float32:
                        raise ValueError('Reload lost FP32 storage')
                    torch.testing.assert_close(parameter.detach().cpu(), saved.get_tensor(name), rtol=0, atol=0)
        rpre, rpost = make_pre_post_processors(reloaded.config, pretrained_path=str(checkpoint))
        torch.manual_seed(123)
        with torch.inference_mode():
            actual = rpost(reloaded.predict_action_chunk(rpre(observation))).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        error = (actual-expected).abs().max().item()
        del reloaded, rpre, rpost, actual, expected
        gc.collect(); torch.cuda.empty_cache()
    policy.train(was_training)
    torch.save({'optimizer': optimizer.state_dict(), 'update': update,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'loader_generator': generator.get_state(), 'epoch_generator_state': epoch_generator_state,
                'epoch': epoch, 'batches_in_epoch': batches_in_epoch,
                'resume_supported': False}, checkpoint/'training_state.pt')
    marker = {'status': 'complete', 'update': update, 'reload_fp32_weights_exact': True,
              'reload_max_action_error': error, 'model_sha256': c.sha256(checkpoint/'model.safetensors')}
    c.write_json(checkpoint/'checkpoint_result.json', marker)
    print('CHECKPOINT', json.dumps(marker), flush=True)
    return marker


def train(output, preflight, report):
    """Explicit port of the fixed_layout runner adapted to the local-varied dataset only."""
    from lerobot.policies.factory import make_pre_post_processors
    from torch.utils.data import DataLoader, Subset

    def progress(**fields):
        report.update(fields); c.write_json(output/'result.json', report)

    raw, starts, stats, binding = c.prepare_data()
    encoded_stats, cpu_hash = c.statistics_json(stats), c.sha256(c.CPU_ADMISSION)
    scratch_hash = None
    if not preflight:
        scratch_hash = c.sha256(c.SCRATCH/'result.json')
        proof = json.loads((c.SCRATCH/'result.json').read_text())
        c.validate_scratch(proof, binding, c.SCRATCH, encoded_stats, cpu_hash)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA BF16 required; no fallback')
    resources = require_resources(output)
    random.seed(c.SEED); np.random.seed(c.SEED); torch.manual_seed(c.SEED)
    torch.set_num_threads(8); torch.backends.cudnn.benchmark = False
    c.write_json(output/'training_stats.json', encoded_stats)
    config = c.make_config('cuda')
    # Always original base, never scratch/final checkpoint as initialization.
    policy, promotion = load_fresh_policy(config)
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    config.save_pretrained(output/'initial_config')
    params = [p for p in policy.parameters() if p.requires_grad]
    tracked = next(p for n, p in policy.named_parameters() if p.requires_grad and 'action_out_proj.weight' in n)
    original = tracked.detach().clone()
    ds = c.PolicyRows(raw)
    rows, verified = c.indices(raw.meta.episodes)
    if starts != verified: raise ValueError('Sampler starts changed')
    progress(status='training', stage='fresh_base', binding=binding, cpu_admission_sha256=cpu_hash,
             scratch_result_sha256=scratch_hash, scratch_used_for_initialization=False,
             training_rows=len(rows), complete_training_chunks=len(starts), resource_admission=resources,
             training_stats_sha256=c.sha256(output/'training_stats.json'), precision_promotion=promotion,
             checkpoints=[], checkpoint_files={}, completed_updates=0, rollouts=0)
    generator = torch.Generator().manual_seed(c.SEED)
    loader = DataLoader(Subset(ds, starts), batch_size=8, shuffle=True, drop_last=True, num_workers=0, generator=generator)
    epoch_state, epoch, batches = generator.get_state(), 0, 0
    iterator = iter(loader)
    optimizer = make_optimizer(params)
    policy.train()
    maximum, began = (2 if preflight else 5000), time.monotonic()
    with (output/'metrics.jsonl').open('x') as log:
        for update in range(1, maximum+1):
            tick = time.monotonic()
            for group in optimizer.param_groups: group['lr'] = c.learning_rate(update)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(4):
                try: batch = next(iterator)
                except StopIteration:
                    epoch, batches, epoch_state = epoch+1, 0, generator.get_state()
                    iterator = iter(loader); batch = next(iterator)
                batches += 1
                c.h.check_batch(batch)
                batch = pre(batch)
                with torch.autocast('cuda', dtype=torch.bfloat16): loss, _ = policy(batch)
                c.h.backward_microbatch(loss)
                losses.append(loss.detach().item())
            norm = torch.nn.utils.clip_grad_norm_(params, 10., error_if_nonfinite=True)
            optimizer.step()
            if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in params): raise ValueError('Nonfinite/nonFP32 trainable parameter')
            if any(p.grad is not None for p in policy.parameters() if not p.requires_grad): raise ValueError('Frozen VLM received gradients')
            c.h.verify_optimizer_precision(optimizer)
            if update == 1 and torch.equal(original, tracked): raise ValueError('First projection update absent')
            torch.cuda.synchronize()
            row = dict(update=update, loss=sum(losses)/4, grad_norm=float(norm), lr=c.learning_rate(update),
                       seconds=time.monotonic()-tick, sampled_chunks=update*32, epoch=epoch,
                       completed_microbatches=update*4, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
            if update == 1 or update % 20 == 0 or update == maximum:
                progress(completed_updates=update, last_metrics=row, elapsed_seconds=time.monotonic()-began)
                print('TRAIN', json.dumps(row), flush=True)
            if update in c.checkpoint_updates(preflight):
                require_disk(output, 5)
                marker = save_checkpoint(policy, pre, post, optimizer, ds, c.PROCESSOR_ROWS[0], output, update,
                                         generator, epoch_state, epoch, batches)
                c.require_marker(marker, update)  # Stronger than prior helper's relative tolerance.
                checkpoint = output/'checkpoints'/c.checkpoint_directory(update)
                if preflight:
                    evidence = adam_roundtrip(optimizer, checkpoint/'training_state.pt', update)
                    c.write_json(checkpoint/'adam_reload.json', evidence)
                report['checkpoints'].append(marker)
                report['checkpoint_files'][c.checkpoint_directory(update)] = c.inventory(checkpoint)
                progress(startup_save_reload_passed=True)
    if c.admission() != binding or c.sha256(c.CPU_ADMISSION) != cpu_hash:
        raise ValueError('Post-training dataset/source/cache/CPU proof changed')
    if c.sha256(output/'training_stats.json') != report['training_stats_sha256']:
        raise ValueError('Training statistics bytes changed')
    if not preflight:
        if c.sha256(c.SCRATCH/'result.json') != scratch_hash: raise ValueError('Scratch proof changed during main')
        c.validate_scratch(proof, binding, c.SCRATCH, encoded_stats, cpu_hash)
    c.verify_checkpoints(report, output, preflight)
    # Final saved config matches initial saved config (dtype/chunk/action-step schema).
    if not preflight:
        final_config = json.loads((output/'checkpoints'/c.checkpoint_directory(5000)/'config.json').read_text())
        initial_config = json.loads((output/'initial_config/config.json').read_text())
        if (final_config != initial_config
                or (final_config['type'], final_config['chunk_size'], final_config['n_action_steps'], final_config['n_obs_steps']) != ('smolvla', 50, 10, 1)
                or {k: tuple(v['shape']) for k, v in final_config['input_features'].items()} != c.OBS
                or {k: v['shape'] for k, v in final_config['output_features'].items()} != {'action': [8]}):
            raise ValueError('Final SmolVLA config/schema changed')
    progress(status='complete', stage='training_complete', completed_updates=maximum,
             final_data_binding_passed=True, selected_checkpoint_update=None if preflight else 5000,
             adam_reload_fp32_exact=preflight, task_success_evaluated=False)


def main(argv=None):
    args = parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if args.check:
        check(); return
    output = c.SCRATCH if args.preflight else c.OUTPUT
    c.claim_output(output)
    report = dict(status='preparing', stage='admission', **c.recipe(args.preflight))
    try:
        c.write_json(output/'result.json', report)
        train(output, args.preflight, report)
    except BaseException as exc:
        failure(output, report, exc); raise
    gc.collect(); torch.cuda.empty_cache()
    # Evaluation/policy rollout is a separate protocol; --train never launches it.


if __name__ == '__main__': main()
