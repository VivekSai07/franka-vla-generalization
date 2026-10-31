"""CPU admission, zero-equivalence gate, or an X-VLA continuation pilot: the P2 checkpoint 101,520 plus one extra-state variant.

`python -m experiments.xvla_p2_pilot.runner (--check | --zero-equivalence | --train) --variant NAME --out DIR`

The update math is M2's, unchanged, in `one_update`, and so is the plumbing around it (the fused precision checks, the
prefetching thread, the thermal guard, the byte-exact reload of the saved weights): those functions below are copied from
`experiments/xvla_p2_force/runner.py` and `tests/test_runner.py` asserts they stay equal to M2's. What is the pilot's own is
the start (the bundle's checkpoint 101,520, hash-pinned, with a fresh optimizer), the schedule (`contract.py`), the variant
(a data file, hashed into the run record) and the output: one checkpoint, no snapshots and no optimizer state. A pilot that
crashes is re-run in a new directory; it is deterministic.

* `--check`: the full CPU admission of the variant (seal, sidecar, binding, data, processor and model-input check, action
  statistics against the start checkpoint's) and the strict load of the start weights with a fresh optimizer. No GPU.
* `--zero-equivalence`: the variant with every extra forced to zero against the control for a few updates: metrics, weights and
  Adam moments must be byte-equal, so the real run differs from the control by the variant only.
* `--train`: admission, then the continuation, then the checkpoint. `--updates N` runs a prefix of the schedule (a dry run).
"""
import argparse
import contextlib
import gc
import hashlib
import json
import os
import queue
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.xvla_trial.assets import safe_error
from . import contract as c

EXPECTED_PARAM_TENSORS = 901
EXPECTED_PARAM_NUMEL = 879482456
ZERO_EQUIVALENCE_UPDATES = 30
TELEMETRY_EVERY = 20
HOT_C, COOL_C, HOT_LIMIT_S = 83, 75, 1800  # firmware target is 85 C on this card (300 W hard cap)


# --- plumbing copied unchanged from the M2 runner (test_runner.py asserts it stays equal) ------------

def verify_precision_fast(params, optimizer=None, require_gradients=False):
    """Exactly `xvla_trial.preflight.verify_precision`'s checks, with one GPU sync instead of one per tensor.

    Presence, dtype and requires_grad are host-side; finiteness is one fused max-abs pass (NaN and inf
    propagate through it; max-abs cannot overflow the way an L2 norm can).
    """
    tensors, grads, moments = [], [], []
    for p in params.values():
        if not p.requires_grad or p.dtype != torch.float32:
            raise ValueError('All parameters must be finite FP32 and trainable')
        tensors.append(p)
        if require_gradients:
            if p.grad is None or p.grad.dtype != torch.float32:
                raise ValueError('Missing/nonfinite/non-FP32 gradient')
            grads.append(p.grad)
        if optimizer is not None:
            state = optimizer.state[p]
            for key in ('exp_avg', 'exp_avg_sq'):
                if key not in state or state[key].dtype != torch.float32:
                    raise ValueError('Missing/nonfinite/non-FP32 Adam moment')
                moments.append(state[key])
    for group, message in ((tensors, 'All parameters must be finite FP32 and trainable'),
                           (grads, 'Missing/nonfinite/non-FP32 gradient'),
                           (moments, 'Missing/nonfinite/non-FP32 Adam moment')):
        if group and not bool(torch.isfinite(torch.stack(torch._foreach_norm(group, float('inf')))).all()):
            raise ValueError(message)


class LockedDataset:
    """One re-entrant lock for every dataset read and every `approved_instruction` swap of the shared global."""

    def __init__(self, ds):
        self.ds, self.lock = ds, threading.RLock()

    def __getitem__(self, i):
        with self.lock:
            return self.ds[i]

    def __getattr__(self, name):
        return getattr(self.ds, name)


def serial_fetch(ds, samples):
    """P1's fetch: decode a microbatch right before it is used."""
    lock = getattr(ds, 'lock', None) or contextlib.nullcontext()
    for selected in samples:
        with lock:
            raw = c.collate_rows([ds[i] for i in selected])
        yield selected, raw


class Prefetcher:
    """Decodes microbatches ahead in a thread, in exactly the sampler's order; errors re-raise in the caller."""

    def __init__(self, ds, samples, depth=4):
        self.queue, self.stop = queue.Queue(maxsize=depth), threading.Event()
        self.thread = threading.Thread(target=self._run, args=(ds, samples), daemon=True)
        self.thread.start()

    def _run(self, ds, samples):
        try:
            for item in serial_fetch(ds, samples):
                if not self._put(item):
                    return
            self._put(StopIteration())
        except BaseException as exc:
            self._put(exc)

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=1)
                return True
            except queue.Full:
                continue
        return False

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.stop.set()
        self.thread.join(timeout=30)


def one_update(update, policy, params, optimizer, pre, fetch, verify):
    """P1's update, unchanged: two microbatches of 4, bf16 autocast, two verifies, clip 10, AdamW."""
    c.set_learning_rate(optimizer, update)
    optimizer.zero_grad(set_to_none=True)
    losses, indices = [], []
    for _ in range(2):
        selected, raw = next(fetch)
        batch = pre(raw)
        if batch['action'].shape != (4, 30, 8):
            raise ValueError('Processed microbatch shape changed')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = policy(batch)
        c.backward_microbatch(loss)
        losses.append(float(loss.detach()))
        indices.append(selected)
    verify(params, require_gradients=True)
    norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 10., error_if_nonfinite=True)
    optimizer.step()
    verify(params, optimizer, require_gradients=True)
    torch.cuda.synchronize()
    return dict(update=update, loss=sum(losses)/2, grad_norm=float(norm), dataset_rows=indices,
                learning_rates={g['name']: g['lr'] for g in optimizer.param_groups})


def gpu_telemetry():
    """Temperature, power and thermal-throttle flags from nvidia-smi; None if it cannot be read."""
    fields = ('temperature.gpu', 'power.draw', 'power.limit', 'clocks_throttle_reasons.hw_thermal_slowdown',
              'clocks_throttle_reasons.sw_thermal_slowdown', 'utilization.gpu')
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu='+','.join(fields), '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_telemetry(out)


def parse_telemetry(line):
    values = [v.strip() for v in line.strip().splitlines()[0].split(',')]
    return dict(temperature_c=int(values[0]), power_w=float(values[1]), power_limit_w=float(values[2]),
                thermal_slowdown=values[3] == 'Active' or values[4] == 'Active', utilization=int(values[5]))


def too_hot(telemetry):
    return telemetry is not None and (telemetry['temperature_c'] >= HOT_C or telemetry['thermal_slowdown'])


def cool_down(read=gpu_telemetry, sleep=time.sleep, clock=time.monotonic):
    """Sleep until the GPU is at or below COOL_C; returns seconds slept, or None after HOT_LIMIT_S hot."""
    began = clock()
    while True:
        telemetry = read()
        if telemetry is None or (telemetry['temperature_c'] <= COOL_C and not telemetry['thermal_slowdown']):
            return clock()-began
        if clock()-began >= HOT_LIMIT_S:
            return None
        sleep(30)


def state_digest(policy):
    hasher = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def require_disk(output, gib):
    if shutil.disk_usage(output).free < gib*1024**3:
        raise RuntimeError(f'Require {gib} GiB free; no deletion/fallback')


def failure(output, report, exc):
    report.update(status='failed', **safe_error(exc), error=str(exc), retries=False)
    if torch.cuda.is_initialized():
        report['failure_memory'] = c.p.memory()
    try:
        path, number = output/'failure.json', 1
        while path.exists():  # a resumed run that fails again must not lose its own traceback
            number += 1
            path = output/f'failure_{number}.json'
        with path.open('x') as stream:
            json.dump(report, stream, allow_nan=False)
        c.write_json(output/'result.json', report)
    except (OSError, ValueError):
        pass  # Preserve original failure, never cleanup/retry.
    print(json.dumps(dict(status='failed', stage=report.get('stage'), **safe_error(exc))), flush=True)


def statistics_json(stats):
    return {key: {name: value.tolist() for name, value in values.items()} for key, values in stats.items()}


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


# --- what is the pilot's own ----------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('check', 'train', 'zero-equivalence'):
        mode.add_argument('--'+name, action='store_true')
    parser.add_argument('--variant', required=True, help='name of a file in variants/ (no path, no suffix)')
    parser.add_argument('--out', required=True, help='an empty bind-mounted directory for this run')
    parser.add_argument('--updates', type=int, default=c.UPDATES,
                        help='dry run: a prefix of the pilot schedule with this many updates (--train only)')
    args = parser.parse_args(argv)
    if args.updates != c.UPDATES and not args.train:
        parser.error('--updates only applies to --train')
    if not 1 <= args.updates <= c.UPDATES:
        parser.error(f'--updates must be in [1,{c.UPDATES}]')
    return args


def require_resources(output):
    # One 3.3 GB checkpoint is all a run keeps; 15 GiB is a margin over the checkpoint and its byte-exact reload.
    require_disk(output, 15)
    free, total = torch.cuda.mem_get_info()
    if free < 28*1024**3:
        raise RuntimeError('Require 28 GiB GPU free; do not interfere with other workloads')
    return dict(disk_free=shutil.disk_usage(output).free, gpu_free=free, gpu_total=total)


def processor_check(ds, selected, stats, builder, zero=False):
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
    evidence = c.check_model_inputs(policy, batch, raw['observation.state'], selected, builder, zero)
    if not torch.allclose(post(batch['action']), raw['action'], rtol=1e-5, atol=1e-5):
        raise ValueError('Physical action normalization roundtrip failed')
    return {**evidence, 'physical_action_roundtrip': True, 'dataset_rows': selected,
            'whole_policy_constructed': False, 'backward_executed': False}


def admit(variant, zero=False):
    """Everything CPU that must hold before a GPU is touched, for one variant: the seal, the sidecar, the binding, the data."""
    builder = c.BUILDERS[variant['spec']['state_builder']]
    fingerprint = c.parent_fingerprint()  # first: nothing else is trusted until the seal reproduces
    if fingerprint != c.SEAL:
        raise ValueError('Props P2 parent fingerprint differs from the seal')
    sidecar = c.forces.verify_sidecar()['fingerprint']  # and S0's seal, before the binding or any data
    before = c.current_binding()
    semantic = c.semantic_gate()  # the collection's own proof, bound by digest in `before`
    ds, starts, stats = c.load_data(builder, zero)
    if (len(starts), len(ds)) != (c.COMPLETE_STARTS, c.TOTAL_FRAMES):
        raise ValueError('Derived training counts changed')
    selected = [starts[0], starts[421], starts[422], starts[-1]]
    evidence = processor_check(ds, selected, stats, builder, zero)
    if c.current_binding() != before:
        raise ValueError('Source/data/assets changed during CPU admission')
    return dict(ds=ds, starts=starts, stats=stats, binding=before, builder=builder, evidence=evidence,
                semantic=semantic, fingerprint=fingerprint, sidecar_fingerprint=sidecar)


def load_start_policy(config):
    """The bundle's checkpoint 101,520: strict safetensors FP32 load, verified before any Adam allocation."""
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    policy = c.p.load_weights(XVLAPolicy(config), c.START/'model.safetensors')
    params = policy.get_optim_params()
    if (len(params) != EXPECTED_PARAM_TENSORS or len(dict(policy.named_parameters())) != EXPECTED_PARAM_TENSORS
            or sum(x.numel() for x in params.values()) != EXPECTED_PARAM_NUMEL
            or set(params) != set(dict(policy.named_parameters()))):
        raise ValueError('Expected all 901 named FP32 tensors / 879482456 trainable parameters')
    return policy, params


def fresh_optimizer(config, params):
    """A new AdamW at the first warm-up rate: no moments, no step counts (the bundle keeps weights only)."""
    optimizer = config.get_optimizer_preset().build(params)
    c.set_learning_rate(optimizer, 1)
    if optimizer.state:
        raise ValueError('The continuation starts from a fresh optimizer; this one already holds state')
    grouped = [x for group in optimizer.param_groups for x in group['params']]
    if len(grouped) != EXPECTED_PARAM_TENSORS or {id(x) for x in grouped} != {id(x) for x in params.values()}:
        raise ValueError('Missing/duplicated optimizer parameter')
    return optimizer


def save_checkpoint(policy, pre, post, optimizer, ds, probe, output, variant, updates):
    """Local port: step_ prefix, FP32 byte-exact reload, no optimizer state (a pilot is re-run, never resumed)."""
    require_disk(output, 5)
    checkpoint = output/'checkpoints'/c.checkpoint_directory(updates)
    checkpoint.mkdir(parents=True, exist_ok=False)
    was_training = policy.training
    policy.eval()
    with getattr(ds, 'lock', None) or contextlib.nullcontext():
        sample = ds[probe]
        with c.approved_instruction(c.planned_instructions()[int(sample['episode_index'])]):
            obs = c.observation({key: sample[key] for key in (*c.OBS, 'task')})
    torch.manual_seed(c.SEED)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = post(policy.predict_action_chunk(c.p.process_row(pre, obs))).cpu()
    if expected.shape != (1, 30, 8) or not torch.isfinite(expected).all():
        raise ValueError('Invalid pre-save prediction')
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    torch.cuda.synchronize()
    error = _reload_and_verify(checkpoint, {'observation': obs, 'prediction': expected})
    policy.train(was_training)
    marker = {'status': 'complete', 'update': updates, 'reload_fp32_weights_exact': True,
              'reload_max_action_error': error, 'model_sha256': c.sha256(checkpoint/'model.safetensors'),
              'optimizer_saved': False,
              'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups},
              'completed_microbatches': updates*2, 'sampled_chunks': updates*8,
              'start_update': c.START_UPDATE, 'start_model_sha256': c.START_MODEL_SHA256,
              'variant': variant['name'], 'variant_sha256': variant['sha256']}
    c.write_json(checkpoint/'checkpoint_result.json', marker)
    print('CHECKPOINT', json.dumps(marker), flush=True)
    return marker


def optimizer_digest(optimizer):
    digest = hashlib.sha256()
    for group in optimizer.param_groups:
        for param in group['params']:
            for key in ('exp_avg', 'exp_avg_sq'):
                digest.update(optimizer.state[param][key].cpu().numpy().tobytes())
    return digest.hexdigest()


def run_updates(ds, starts, stats, updates):
    """`updates` updates from the start checkpoint with a fresh optimizer over `ds`, from a seeded state: rows and digests."""
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
    random.seed(c.SEED)
    np.random.seed(c.SEED)
    torch.manual_seed(c.SEED)
    config = c.make_config('cpu')
    policy, params = load_start_policy(config)
    policy = policy.to('cuda')
    config.device = 'cuda'
    pre, _ = make_xvla_pre_post_processors(config, stats)
    optimizer = fresh_optimizer(config, params)
    policy.train()
    fetch = Prefetcher(LockedDataset(ds), c.sample_batches(starts))
    rows, seconds = [], []
    try:
        for update in range(1, updates+1):
            tick = time.monotonic()
            rows.append(one_update(update, policy, params, optimizer, pre, fetch, verify_precision_fast))
            seconds.append(time.monotonic()-tick)
    finally:
        fetch.close()
    result = dict(rows=rows, model=state_digest(policy), optimizer=optimizer_digest(optimizer),
                  seconds_per_update=float(np.median(seconds[min(5, len(seconds)-1):])))
    del policy, params, optimizer, pre
    gc.collect()
    torch.cuda.empty_cache()
    return result


def compare_runs(a, b):
    return dict(metrics_equal=a['rows'] == b['rows'], model_state_equal=a['model'] == b['model'],
                optimizer_state_equal=a['optimizer'] == b['optimizer'])


def check(name, output):
    """The CPU admission of one variant, plus the strict load of the start weights and a fresh optimizer: no GPU work."""
    variant = c.load_variant(name)
    report = dict(status='preparing', stage='full_cpu_admission', variant=c.variant_record(variant))
    try:
        c.write_json(output/'result.json', report)
        admitted = admit(variant)
        config = c.make_config('cpu')
        policy, params = load_start_policy(config)
        fresh_optimizer(config, params)
        del policy, params
        report = dict(status='passed', gpu_executed=False, optimizer_updates=0, variant=c.variant_record(variant),
                      binding=admitted['binding'], semantic_gate=admitted['semantic'], training_rows=c.TRAINING_ROWS,
                      complete_training_chunks=c.COMPLETE_STARTS, parent_fingerprint_recomputed=True,
                      sidecar_fingerprint_recomputed=True, start_policy_loaded=True, optimizer_fresh=True,
                      action_statistics_equal_start=True, training_stats=c.statistics_json(admitted['stats']),
                      processor_check=admitted['evidence'])
        c.write_json(output/'result.json', report)
    except BaseException as exc:
        failure(output, report, exc)
        raise
    print('CPU admission passed; no GPU work', flush=True)


def zero_equivalence(name, output):
    """`name` with every extra forced to zero, for ZERO_EQUIVALENCE_UPDATES updates, against the control: byte-equal or fail."""
    variant = c.load_variant(name)
    report = dict(status='preparing', stage='zero_equivalence', updates=ZERO_EQUIVALENCE_UPDATES,
                  variant=c.variant_record(variant), extras_forced_zero=True)
    try:
        c.write_json(output/'result.json', report)
        admitted = admit(c.load_variant('control'))
        ds, starts, stats, binding = admitted['ds'], admitted['starts'], admitted['stats'], admitted['binding']
        builder = c.BUILDERS[variant['spec']['state_builder']]
        forced = c.VariantDataset(ds.ds, builder, zero_extras=True)
        processor_check(forced, [starts[0], starts[421], starts[422], starts[-1]], stats, builder, True)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 required; no fallback')
        require_resources(output)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        runs = {label: run_updates(dataset, starts, stats, ZERO_EQUIVALENCE_UPDATES)
                for label, dataset in (('control', ds), ('forced_zero', forced))}
        result = compare_runs(runs['control'], runs['forced_zero'])
        if not all(result.values()):
            raise ValueError(f'Extras forced to zero are not byte-equal to the control: {result}')
        report.update(status='complete', binding=binding, **result,
                      control_seconds_per_update=runs['control']['seconds_per_update'],
                      forced_zero_seconds_per_update=runs['forced_zero']['seconds_per_update'],
                      model_sha256=runs['forced_zero']['model'], final_loss=runs['forced_zero']['rows'][-1]['loss'])
        c.write_json(output/'result.json', report)
    except BaseException as exc:
        failure(output, report, exc)
        raise
    print('ZERO_EQUIVALENCE', json.dumps({k: v for k, v in report.items() if k != 'binding'}), flush=True)


def train(output, variant, report, updates=c.UPDATES):
    """The continuation: `updates` updates of the pilot schedule from the start checkpoint, then the one checkpoint."""
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors

    def progress(**fields):
        report.update(fields)
        c.write_json(output/'result.json', report)

    admitted = admit(variant)
    ds, starts, stats, binding, builder = (admitted[key] for key in ('ds', 'starts', 'stats', 'binding', 'builder'))
    zero = ds.zero_extras
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required; no fallback')
    resources = require_resources(output)
    torch.cuda.reset_peak_memory_stats()
    random.seed(c.SEED)
    np.random.seed(c.SEED)
    torch.manual_seed(c.SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    source = c.ROOT/variant['path']
    if c.sha256(source) != variant['sha256']:
        raise ValueError('The variant file changed since it was loaded')
    (output/'variant.json').write_bytes(source.read_bytes())
    c.write_json(output/'training_stats.json', c.statistics_json(stats))
    progress(status='training', stage='start_load', binding=binding, scratch_used_for_initialization=False,
             training_rows=c.TRAINING_ROWS, complete_training_chunks=len(starts), resource_admission=resources,
             training_stats_sha256=c.sha256(output/'training_stats.json'), checkpoints=[], checkpoint_files={},
             completed_updates=0, rollouts=0,
             device=dict(name=torch.cuda.get_device_name(), properties=str(torch.cuda.get_device_properties(0))),
             initial_memory=c.p.memory(), start=binding['start'])
    config = c.make_config('cpu')
    policy, params = load_start_policy(config)
    policy = policy.to('cuda')
    config.device = 'cuda'
    config.save_pretrained(output/'initial_config')
    c.p.verify_precision(params)
    pre, post = make_xvla_pre_post_processors(config, stats)
    optimizer = fresh_optimizer(config, params)
    checked_rows = [starts[0], starts[421], starts[422], starts[-1]]
    raw = c.collate_rows([ds[i] for i in checked_rows])
    checked = pre(raw)
    progress(model_input_precheck=c.check_model_inputs(policy, checked, raw['observation.state'], checked_rows, builder, zero),
             parameter_tensors=len(params), parameter_count=sum(x.numel() for x in params.values()),
             optimizer_groups=[dict(name=g['name'], parameter_tensors=len(g['params']),
                                    weight_decay=g['weight_decay']) for g in optimizer.param_groups],
             after_load_memory=c.p.memory(), stage='training')
    del raw, checked
    samples = c.sample_batches(starts)
    began = time.monotonic()
    policy.train()
    data = LockedDataset(ds)
    fetch = Prefetcher(data, samples)
    try:
        with (output/'metrics.jsonl').open('x') as log:
            for update in range(1, updates+1):
                tick = time.monotonic()
                row = one_update(update, policy, params, optimizer, pre, fetch, verify_precision_fast)
                row.update(seconds=time.monotonic()-tick, memory=c.p.memory(), sampled_chunks=update*8)
                if update == 1 or update % TELEMETRY_EVERY == 0:
                    row['gpu'] = gpu_telemetry()
                    if too_hot(row['gpu']):
                        slept = cool_down()
                        if slept is None:
                            raise RuntimeError(f'GPU above {COOL_C} C for {HOT_LIMIT_S} s; re-run the pilot')
                        row['thermal_pause_seconds'] = slept
                        print('THERMAL', json.dumps(dict(update=update, slept=slept, gpu=row['gpu'])), flush=True)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                report['completed_updates'] = update
                if update == 1 or update % 20 == 0 or update == updates:
                    progress(last_metrics=row, elapsed_seconds=time.monotonic()-began)
                    print('TRAIN', json.dumps(row, allow_nan=False), flush=True)
    finally:
        fetch.close()
    if c.current_binding() != binding:
        raise ValueError('Post-training dataset/source/cache/start proof changed')
    if c.sha256(output/'training_stats.json') != report['training_stats_sha256']:
        raise ValueError('Training statistics bytes changed')
    marker = save_checkpoint(policy, pre, post, optimizer, data, starts[0], output, variant, updates)
    c.require_marker(marker, variant, updates)
    key = c.checkpoint_directory(updates)
    report['checkpoints'].append(marker)
    report['checkpoint_files'][key] = c.inventory(output/'checkpoints'/key)
    progress(startup_save_reload_passed=True)
    c.verify_checkpoint(report, output, variant, updates)
    final_config = json.loads((output/'checkpoints'/key/'config.json').read_text())
    initial_config = json.loads((output/'initial_config/config.json').read_text())
    if final_config != initial_config:
        raise ValueError('Final X-VLA config differs from initial saved config')
    progress(status='complete', stage='training_complete', completed_updates=updates, final_data_binding_passed=True,
             selected_checkpoint_update=updates, rollouts=0, final_memory=c.p.memory())


def main(argv=None):
    args = parse_args(argv)
    variant = c.load_variant(args.variant)  # a name that is not a variant file stops here, before anything is claimed
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    torch.set_num_threads(4)
    output = Path(args.out)
    c.claim_output(output)
    if args.check:
        check(variant['name'], output)
        return
    if args.zero_equivalence:
        zero_equivalence(variant['name'], output)
        return
    report = dict(status='preparing', stage='admission', **c.recipe(args.updates), variant=c.variant_record(variant))
    try:
        c.write_json(output/'result.json', report)
        train(output, variant, report, updates=args.updates)
    except BaseException as exc:
        failure(output, report, exc)
        raise
    gc.collect()
    torch.cuda.empty_cache()
    # Evaluation/rollout is a separate protocol; --train never launches it.


if __name__ == '__main__':
    raise SystemExit(main())
