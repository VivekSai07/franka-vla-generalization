"""Memory probe, CPU admission, two-update scratch, resume proof, or the pi0.5 fine-tune on the sealed props P2 dataset.

The loop, prefetcher, snapshot/resume, thermal guard and pause-at-snapshot follow `xvla_props_p2/runner.py` (copied and
adapted, not imported). What is pi0.5's: three memory strategies behind one interface (`strategy.py`), a tie-aware weight
loader instead of `from_pretrained` (`weights.py`), microbatches of one effective batch of 8, and a weights check that
compares the saved file tensor by tensor with the live policy (a second 3.6 B model does not fit next to the first).

`--probe-memory` is not a gate: it writes scratch under `outputs/train/pi05_props_p2_probe`, is never run after `--check`,
and its measurements are what the user decides the memory strategy from. Every other mode refuses while
`contract.DECISION` is None.
"""
import argparse
import contextlib
import gc
import json
import os
import queue
import random
import resource
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

from . import assets
from . import contract as c
from . import strategy as st

TELEMETRY_EVERY = 20
HOT_C, COOL_C, HOT_LIMIT_S = 83, 75, 1800  # firmware target is 85 C on this card (300 W hard cap)
SNAPSHOT_WRITE_S = 600  # a 14 to 43 GB write; the probe measures it and the pause decision keeps this margin
SNAPSHOT_FILES = ('model.pt', 'optimizer.pt', 'rng.pt', 'report.json')
PROBE_UPDATES = 20
PROBE_CPU_THREADS = 24  # of the CPU optimizer in `bf16_cpu_master`; the host has 48 cores
PEAK_LR, FLOOR_LR = 2.5e-5, 2.5e-6
X_VLA_EFFECTIVE_OVER_RAW = 0.9/0.524  # what thermal pauses and data waits made of X-VLA's raw pace


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('probe-memory', 'check', 'preflight', 'resume-proof', 'train'):
        mode.add_argument('--'+name, action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='continue a paused or failed --train from its newest verified snapshot')
    parser.add_argument('--stop-at', type=float, default=None,
                        help='unix time by which --train must have paused at a snapshot')
    parser.add_argument('--configs', default=None, help='probe only: strategy:microbatch,... (default all three at 8)')
    parser.add_argument('--updates', type=int, default=None, help=f'probe only: timed updates (default {PROBE_UPDATES})')
    parser.add_argument('--cpu-threads', type=int, default=None, help='probe only: threads of the CPU optimizer')
    args = parser.parse_args(argv)
    if args.resume and not args.train:
        parser.error('--resume only continues --train')
    if args.stop_at is not None and not args.train:
        parser.error('--stop-at only applies to --train')
    if not args.probe_memory and (args.configs or args.updates or args.cpu_threads):
        parser.error('--configs, --updates and --cpu-threads only apply to --probe-memory')
    return args


def parse_configs(text):
    configs = []
    for item in text.split(','):
        name, _, microbatch = item.partition(':')
        if name not in c.STRATEGIES or not microbatch.isdigit() or int(microbatch) not in (1, 2, 4, 8):
            raise ValueError(f'Bad probe config {item!r}: strategy:microbatch with microbatch in 1, 2, 4, 8')
        configs.append((name, int(microbatch)))
    if len(set(configs)) != len(configs):
        raise ValueError('Duplicate probe config')
    return configs


# --- plumbing that does not touch the math ------------------------------------------------------

class LockedDataset:
    """One re-entrant lock for every dataset read (LeRobot's video decoder cache is not thread-safe)."""

    def __init__(self, ds):
        self.ds, self.lock = ds, threading.RLock()

    def __getitem__(self, i):
        with self.lock:
            return self.ds[i]

    def __getattr__(self, name):
        return getattr(self.ds, name)


def serial_fetch(ds, samples, microbatch):
    """For each update, the group of 8 window starts and its microbatches, decoded and collated."""
    lock = getattr(ds, 'lock', None) or contextlib.nullcontext()
    for group in samples:
        with lock:
            raws = [c.collate_rows([ds[i] for i in group[k:k+microbatch]]) for k in range(0, len(group), microbatch)]
        yield group, raws


class Prefetcher:
    """Decodes updates ahead in a thread, in exactly the sampler's order; errors re-raise in the caller."""

    def __init__(self, ds, samples, microbatch, depth=3):
        self.queue, self.stop = queue.Queue(maxsize=depth), threading.Event()
        self.thread = threading.Thread(target=self._run, args=(ds, samples, microbatch), daemon=True)
        self.thread.start()

    def _run(self, ds, samples, microbatch):
        try:
            for item in serial_fetch(ds, samples, microbatch):
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


def one_update(update, strategy, policy, pre, fetch, microbatch, lr=None):
    """One effective batch of 8: `8 / microbatch` forward/backward passes, clip, step. All three strategies share it."""
    accumulation = c.EFFECTIVE_BATCH//microbatch
    waited = time.perf_counter()
    group, raws = next(fetch)
    waited = time.perf_counter()-waited
    if len(raws) != accumulation:
        raise ValueError('Fetched microbatch count does not match the accumulation')
    strategy.zero_grad()
    losses = []
    for raw in raws:
        batch = pre(raw)
        if tuple(batch['action'].shape) != (microbatch, c.CHUNK_SIZE, 8):
            raise ValueError('Processed microbatch shape changed')
        with strategy.autocast():
            loss, _ = policy(batch)
        (loss/accumulation).backward()
        losses.append(float(loss.detach()))
    info = strategy.step(update, lr)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return dict(update=update, loss=sum(losses)/len(losses), dataset_rows=group,
                learning_rate=c.learning_rate(update) if lr is None else lr, data_wait_seconds=waited, **info)


def gpu_telemetry():
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


def should_pause(now, stop_at, seconds_per_update, updates_to_next_snapshot):
    if stop_at is None:
        return False
    return now+seconds_per_update*updates_to_next_snapshot+SNAPSHOT_WRITE_S > stop_at


def memory():
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())


def host_rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])*1024
    return None


def require_disk(path, nbytes):
    if shutil.disk_usage(path).free < nbytes:
        raise RuntimeError(f'Require {nbytes/1024**3:.1f} GiB free at {path}; no deletion/fallback')


def require_gpu(nbytes):
    free, total = torch.cuda.mem_get_info()
    if free < nbytes:
        raise RuntimeError(f'Require {nbytes/1024**3:.1f} GiB GPU free, {free/1024**3:.1f} are; '
                           'do not interfere with other workloads')
    return dict(gpu_free=free, gpu_total=total)


def require_resources(output, preflight):
    require_disk(output, c.disk_requirement_bytes(preflight))
    return dict(disk_free=shutil.disk_usage(output).free, **require_gpu(c.gpu_requirement_bytes()))


# --- saving ---------------------------------------------------------------------------------------------------

def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None)


def restore_rng(python, numpy, tensors):
    random.setstate((python[0], tuple(python[1]), python[2]))
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(tensors['torch'])
    if 'cuda' in tensors:
        torch.cuda.set_rng_state(tensors['cuda'])


def file_bytes(directory):
    return {p.name: p.stat().st_size for p in sorted(Path(directory).iterdir()) if p.is_file()}


def write_snapshot_files(strategy, report, tmp):
    """The four files of a snapshot in `tmp`; returns the RNG captured with them."""
    torch.save(strategy.model_state(), tmp/'model.pt')
    torch.save(strategy.optimizer_state(), tmp/'optimizer.pt')
    rng = capture_rng()
    torch.save({'torch': rng['torch'], **({'cuda': rng['cuda']} if rng['cuda'] is not None else {})}, tmp/'rng.pt')
    c.write_json(tmp/'report.json', report)
    return rng


def save_snapshot(strategy, report, output, update, binding):
    """Atomic: written under tmp_, hashed, renamed; at most KEEP_SNAPSHOTS survive, plus the one being written."""
    began = time.monotonic()
    root = output/'resume'
    root.mkdir(exist_ok=True)
    tmp = root/f'tmp_{update:06d}'
    tmp.mkdir()
    rng = write_snapshot_files(strategy, report, tmp)
    meta = dict(update=update, binding_digest=c.digest_json(binding), files={n: c.sha256(tmp/n) for n in SNAPSHOT_FILES},
                python_rng=[rng['python'][0], list(rng['python'][1]), rng['python'][2]],
                numpy_rng=[rng['numpy'][0], rng['numpy'][1].tolist(), *rng['numpy'][2:]])
    c.write_json(tmp/'snapshot.json', meta)
    tmp.rename(root/f'snap_{update:06d}')
    for old in sorted(root.glob('snap_*'))[:-c.KEEP_SNAPSHOTS]:
        shutil.rmtree(old)  # own regenerable intermediate; the newer snapshot covers it
    seconds = time.monotonic()-began
    print('SNAPSHOT', json.dumps(dict(update=update, seconds=seconds, bytes=sum(file_bytes(root/f'snap_{update:06d}').values()))),
          flush=True)
    return seconds


class BindingChanged(Exception):
    """The sources or data differ from what the snapshot was written under; no snapshot is usable."""


def load_snapshot(source, binding):
    """Newest snapshot whose bytes still hash to its record; a damaged newest one falls back."""
    problems = []
    for directory in sorted((source/'resume').glob('snap_*'), reverse=True):
        try:
            meta = json.loads((directory/'snapshot.json').read_text())
            if meta['binding_digest'] != c.digest_json(binding):
                raise BindingChanged(directory.name)
            if meta['update'] != int(directory.name[len('snap_'):]) or set(meta['files']) != set(SNAPSHOT_FILES):
                raise ValueError('snapshot record does not match its directory')
            for name, digest in meta['files'].items():
                if c.sha256(directory/name) != digest:
                    raise ValueError(f'{name} does not hash to its record')
            return directory, meta, problems
        except BindingChanged:
            raise ValueError('Sources/data differ from the run the snapshot belongs to; refusing to resume')
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f'{directory.name}: {exc}')
    raise RuntimeError(f'No verifiable snapshot in {source/"resume"}: {problems}')


def quarantine(output, number, restored, update):
    """Move aside, never delete, whatever the chosen snapshot does not vouch for."""
    where = output/'quarantine'/f'resume_{number:02d}'
    where.mkdir(parents=True, exist_ok=True)

    def move(path, group=''):
        target = where/group
        target.mkdir(parents=True, exist_ok=True)
        path.rename(target/path.name)
        return f'{group}/{path.name}'.strip('/')

    moved = []
    kept = {c.checkpoint_directory(m['update']) for m in restored['checkpoints']}
    for directory in sorted((output/'checkpoints').glob('*')):
        if directory.name not in kept:
            moved.append(move(directory, 'checkpoints'))
    for directory in sorted((output/'resume').glob('*')):
        if directory.name.startswith('tmp_') or (directory.name.startswith('snap_')
                                                 and int(directory.name[len('snap_'):]) > update):
            moved.append(move(directory, 'resume'))
    for name in ('failure.json', 'result.json'):
        if (output/name).exists():
            moved.append(move(output/name))
    metrics = output/'metrics.jsonl'
    rows = metrics.read_text().splitlines()
    if len(rows) < update or [json.loads(row)['update'] for row in rows[:update]] != list(range(1, update+1)):
        raise ValueError('metrics.jsonl does not cover updates 1..snapshot update')
    moved.append(move(metrics))
    metrics.write_text(''.join(row+'\n' for row in rows[:update]))
    return moved


def compare_with_reference(output, reference, update):
    """Resume proof: the resumed run's logged updates and final files equal the uninterrupted run's."""
    def rows(root):
        return [json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
    keys = ('update', 'loss', 'grad_norm', 'dataset_rows', 'learning_rate')
    resumed, original = rows(output), rows(reference)
    key = c.checkpoint_directory(update)
    sha = lambda root, name: c.sha256(root/'checkpoints'/key/name)
    return dict(from_update=update-1, compared_update=update,
                metrics_equal=(len(resumed) == len(original) == update
                               and all({k: a[k] for k in keys} == {k: b[k] for k in keys}
                                       for a, b in zip(resumed, original))),
                model_sha256_equal=sha(output, 'model.safetensors') == sha(reference, 'model.safetensors'),
                training_state_sha256_equal=sha(output, 'training_state.pt') == sha(reference, 'training_state.pt'),
                scratch_model_sha256=sha(reference, 'model.safetensors'))


def verify_saved_weights(policy, path):
    """Every tensor in the saved file equals the live one (dtype included); omitted names must be ties of stored ones."""
    from safetensors import safe_open
    state = policy.state_dict()
    with safe_open(str(path), framework='pt', device='cpu') as stored:
        keys = set(stored.keys())
        for key in sorted(keys):
            saved, live = stored.get_tensor(key), state[key].detach().cpu()
            if saved.dtype != live.dtype or not torch.equal(saved, live):
                raise ValueError(f'Saved tensor {key} differs from the live policy')
    pointers = {state[k].data_ptr() for k in keys}
    unexplained = [k for k in state if k not in keys and state[k].data_ptr() not in pointers]
    if unexplained:
        raise ValueError(f'Saved file lacks {unexplained[:3]}')
    return len(keys)


def save_checkpoint(policy, strategy, output, update, final):
    """step_ directory: weights, config, marker; the optimizer state only on the final one. Weights are verified in place."""
    require_disk(output, c.disk_requirement_bytes(False)//4)
    checkpoint = output/'checkpoints'/c.checkpoint_directory(update)
    checkpoint.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(checkpoint)
    if final:
        torch.save(strategy.optimizer_state(), checkpoint/'training_state.pt')
    torch.cuda.synchronize()
    tensors = verify_saved_weights(policy, checkpoint/'model.safetensors')
    marker = {'status': 'complete', 'update': update, 'reload_weights_exact': True, 'reload_max_action_error': 0.0,
              'verified_tensors': tensors, 'model_sha256': c.sha256(checkpoint/'model.safetensors'),
              'optimizer_saved': final, 'learning_rate': c.learning_rate(update),
              'completed_microbatches': update*c.accumulation(), 'sampled_windows': update*c.EFFECTIVE_BATCH}
    c.write_json(checkpoint/'checkpoint_result.json', marker)
    print('CHECKPOINT', json.dumps(marker), flush=True)
    return marker


def failure(output, report, exc):
    report.update(status='failed', error_type=type(exc).__name__, error=str(exc), retries=False)
    if torch.cuda.is_initialized():
        report['failure_memory'] = memory()
    try:
        path, number = output/'failure.json', 1
        while path.exists():
            number += 1
            path = output/f'failure_{number}.json'
        with path.open('x') as stream:
            json.dump(report, stream, allow_nan=False)
        c.write_json(output/'result.json', report)
    except (OSError, ValueError):
        pass  # Preserve the original failure, never clean up or retry.
    print(json.dumps(dict(status='failed', stage=report.get('stage'), error_type=type(exc).__name__)), flush=True)


# --- memory probe ---------------------------------------------------------------------------------------------

def tensor_group(name):
    if 'vision_tower' in name or 'multi_modal_projector' in name:
        return 'vision'
    if 'gemma_expert' in name:
        return 'expert'
    if 'paligemma' in name:
        return 'language'
    return 'projections'


def group_summaries(rows):
    groups = {}
    for row in rows:
        groups.setdefault(tensor_group(row['name']), []).append(row)
    return {group: st.summarize_changes(items) for group, items in sorted(groups.items())}


def measure_changes(strategy, step, label, lr):
    """One update at a forced learning rate; the fraction of each trainable tensor whose stored value changed."""
    before = {n: p.detach().clone() for n, p in strategy.params.items()}
    small = None
    if getattr(strategy, 'masters', None) is not None:
        small = {n: m.detach().clone() for n, m in strategy.masters.items() if m.numel() <= 20_000_000}
    step(lr)
    rows = st.changed_fractions(before, {n: p.detach() for n, p in strategy.params.items()})
    result = dict(label=label, forced_learning_rate=lr, view='policy tensors as the forward pass reads them',
                  overall=st.summarize_changes(rows), by_group=group_summaries(rows),
                  by_dtype={dtype: st.summarize_changes([r for r in rows if r['dtype'] == dtype])
                            for dtype in sorted({r['dtype'] for r in rows})}, per_tensor=rows)
    if small is not None:
        master_rows = st.changed_fractions(small, {n: strategy.masters[n].detach() for n in small})
        result['fp32_masters_small_tensors'] = st.summarize_changes(master_rows)
    del before, small
    return result


def measure_sizes(policy, strategy, scratch):
    """Write one checkpoint and one snapshot into `scratch`, record bytes and seconds, remove them (own scratch)."""
    scratch.mkdir(parents=True)
    sizes = {}
    began = time.monotonic()
    policy.save_pretrained(scratch/'checkpoint')
    sizes['checkpoint_write_seconds'] = time.monotonic()-began
    sizes['checkpoint_files'] = file_bytes(scratch/'checkpoint')
    sizes['checkpoint_bytes'] = sizes['checkpoint_files']['model.safetensors']
    (scratch/'snapshot').mkdir()
    began = time.monotonic()
    write_snapshot_files(strategy, {}, scratch/'snapshot')
    sizes['snapshot_write_seconds'] = time.monotonic()-began
    sizes['snapshot_files'] = file_bytes(scratch/'snapshot')
    sizes['snapshot_bytes'] = sum(sizes['snapshot_files'].values())
    sizes['final_state_bytes'] = sizes['snapshot_files']['optimizer.pt']
    if getattr(strategy, 'masters', None) is not None:
        sizes['fp32_checkpoint_bytes_if_masters_saved'] = sum(m.numel() for m in strategy.masters.values())*4
    shutil.rmtree(scratch)
    return sizes


def project(name, microbatch, seconds_per_update, sizes, peak_bytes):
    """Time and disk for the full run from the measured pace and sizes; the decision itself is the user's."""
    decision = dict(strategy=name, microbatch=microbatch, snapshot_every=1, gpu_peak_bytes=peak_bytes,
                    **{k: sizes[k] for k in ('checkpoint_bytes', 'snapshot_bytes', 'final_state_bytes')})
    raw_hours = seconds_per_update*c.TOTAL_UPDATES/3600
    checkpoints = len(c.checkpoint_updates(False))
    return dict(updates=c.TOTAL_UPDATES, raw_hours=raw_hours, raw_days=raw_hours/24,
                hours_at_x_vla_effective_over_raw=raw_hours*X_VLA_EFFECTIVE_OVER_RAW,
                disk_peak_bytes=c.disk_requirement_bytes(False, decision),
                disk_peak_gib=c.disk_requirement_bytes(False, decision)/1024**3,
                disk_at_end_bytes=checkpoints*sizes['checkpoint_bytes']+sizes['final_state_bytes']
                + c.KEEP_SNAPSHOTS*sizes['snapshot_bytes'],
                # --preflight and --resume-proof outputs stay on disk (the proof re-verifies the scratch run)
                gate_artifacts_bytes=2*(len(c.checkpoint_updates(True))*sizes['checkpoint_bytes']
                                        + sizes['final_state_bytes'] + c.KEEP_SNAPSHOTS*sizes['snapshot_bytes']),
                assets_bytes=sum(size for size, _ in assets.FILES.values()), checkpoints=checkpoints)


def probe_one(name, microbatch, updates, ds, starts, stats, out, threads):
    result = dict(strategy=name, microbatch=microbatch, accumulation=c.EFFECTIVE_BATCH//microbatch, status='running',
                  timed_updates=updates)
    fetch = policy = strategy = pre = post = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        began = time.monotonic()
        policy, config, masters, load_report = st.build_policy(name)
        result['load'] = dict(seconds=time.monotonic()-began, report=load_report, host_rss_after_build=host_rss_bytes())
        policy = policy.to('cuda')
        config.device = 'cuda'
        torch.cuda.reset_peak_memory_stats()
        strategy = st.make_strategy(name, policy, masters, 'cuda')
        if threads and name == 'bf16_cpu_master':
            torch.set_num_threads(threads)
        pre, post = c.make_processors(config, stats)
        policy.train()
        result['memory_after_load'] = memory()
        result['trainable_parameters'] = sum(p.numel() for p in strategy.params.values())
        result['gradient_checkpointing'] = config.gradient_checkpointing
        result['torch_threads'] = torch.get_num_threads()
        fetch = Prefetcher(LockedDataset(ds), c.sample_updates(starts), microbatch)
        rows = []
        for update in range(1, updates+1):
            tick = time.perf_counter()
            row = one_update(update, strategy, policy, pre, fetch, microbatch)
            row['seconds'] = time.perf_counter()-tick
            row['memory_peak_allocated'] = torch.cuda.max_memory_allocated()
            if update in (1, updates//2, updates):
                row['gpu'] = gpu_telemetry()
            rows.append({k: v for k, v in row.items() if k != 'dataset_rows'})
            if update == 1:
                result['dtype_report'] = strategy.dtype_report()
                result['memory_after_first_update'] = memory()
        peak = memory()
        compute = [r['seconds'] for r in rows[2:]]
        result.update(memory_peak=peak, peak_allocated_gib=peak['peak_allocated']/1024**3,
                      peak_reserved_gib=peak['peak_reserved']/1024**3, host_rss_peak_bytes=resource.getrusage(
                          resource.RUSAGE_SELF).ru_maxrss*1024,
                      seconds_per_update=dict(median=statistics.median(compute), mean=statistics.fmean(compute),
                                              min=min(compute), max=max(compute), warmup_excluded=2),
                      data_wait_seconds_median=statistics.median(r['data_wait_seconds'] for r in rows[2:]),
                      losses=[r['loss'] for r in rows], grad_norms=[r['grad_norm'] for r in rows], rows=rows)
        if name == 'bf16_cpu_master':
            result['phase_seconds_median'] = {k: statistics.median(r['phase_seconds'][k] for r in rows[2:])
                                              for k in rows[-1]['phase_seconds']}
        torch.cuda.reset_peak_memory_stats()
        counter = updates
        changes = {}
        for label, lr in (('peak_lr', PEAK_LR), ('floor_lr', FLOOR_LR)):
            def step(lr, label=label):
                nonlocal counter
                counter += 1
                row = one_update(counter, strategy, policy, pre, fetch, microbatch, lr=lr)
                changes.setdefault('rows', []).append(dict(label=label, loss=row['loss'], grad_norm=row['grad_norm']))
            changes[label] = measure_changes(strategy, step, label, lr)
        result['changed_per_update'] = changes
        result['memory_during_diagnostics_peak'] = memory()
        result['sizes'] = measure_sizes(policy, strategy, out/f'tmp_{name}_mb{microbatch}')
        result['projection'] = project(name, microbatch, result['seconds_per_update']['median'], result['sizes'],
                                       peak['peak_allocated'])
        result['status'] = 'complete'
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status='out_of_memory', error=str(exc).splitlines()[0], memory_at_failure=memory())
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:2000])
        raise
    finally:
        if fetch is not None:
            fetch.close()
        del fetch, policy, strategy, pre, post
        gc.collect()
        torch.cuda.empty_cache()
        with (out/f'probe_{name}_mb{microbatch}.json').open('x') as stream:
            json.dump(result, stream, allow_nan=False)
    return result


def probe_memory(args):
    if c.ADMISSION.exists():
        raise RuntimeError('--probe-memory is never run after --check: the decision is part of what --check binds')
    configs = parse_configs(args.configs or ','.join(f'{n}:8' for n in c.STRATEGIES))
    updates = args.updates or PROBE_UPDATES
    if updates < 6:
        raise ValueError('A probe needs at least 6 updates (two warm-up, then the median)')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required; no fallback')
    out = c.PROBE
    out.mkdir(parents=True, exist_ok=True)
    require_disk(out, 70*1024**3)
    resources = require_gpu(34*1024**3)
    ds, starts, stats = c.load_data()
    summary = dict(status='running', resources=resources, device=torch.cuda.get_device_name(),
                   telemetry_before=gpu_telemetry(), image=c.IMAGE, results=[])
    for name, microbatch in configs:
        thermal = gpu_telemetry()
        if too_hot(thermal):
            summary.setdefault('thermal_pauses', []).append(cool_down())
        result = probe_one(name, microbatch, updates, ds, starts, stats, out, args.cpu_threads or PROBE_CPU_THREADS)
        summary['results'].append({k: v for k, v in result.items()
                                   if k not in ('rows', 'changed_per_update', 'losses', 'grad_norms')})
        print('PROBE', json.dumps({k: result.get(k) for k in ('strategy', 'microbatch', 'status', 'peak_allocated_gib',
                                                              'seconds_per_update', 'projection')}), flush=True)
    summary['status'] = 'complete'
    with (out/f'probe_{int(time.time())}.json').open('x') as stream:
        json.dump(summary, stream, allow_nan=False)


# --- CPU admission, scratch, proof and training ---------------------------------------------------------------

def processor_check(ds, selected, stats):
    """Real CPU processors on real rows: shapes, camera order, token layout, the physical-action round trip."""
    from transformers import AutoTokenizer
    from experiments.pi05_props_p2 import assets
    config = c.make_config('cpu')
    pre, post = c.make_processors(config, stats)
    raw = c.collate_rows([ds[i] for i in selected])
    batch = pre(raw)
    tokenizer = AutoTokenizer.from_pretrained(str(assets.TOKENIZER_DIR))
    tokens, masks = batch['observation.language.tokens'], batch['observation.language.attention_mask']
    if tuple(tokens.shape) != (len(selected), config.tokenizer_max_length) or int(masks.sum(1).max()) >= tokens.shape[1]:
        raise ValueError('Prompt tokens missing or truncated')
    prompts = [tokenizer.decode(t[m], skip_special_tokens=True) for t, m in zip(tokens, masks)]
    for prompt, text in zip(prompts, raw['task']):
        if not prompt.startswith(f'Task: {text}, State: ') or not prompt.endswith(';\nAction: '):
            raise ValueError('Prompt does not carry the full instruction')
    if list(config.image_features) != list(c.CAMERAS) or tuple(batch['action'].shape) != (len(selected), c.CHUNK_SIZE, 8):
        raise ValueError('Camera order or action shape changed')
    if not torch.allclose(post(batch['action']), raw['action'], rtol=1e-5, atol=1e-5):
        raise ValueError('Physical action normalization roundtrip failed')
    return dict(physical_action_roundtrip=True, dataset_rows=selected, prompts=prompts, camera_order=list(c.CAMERAS),
                prompt_token_counts=[int(n) for n in masks.sum(1)], max_tokens=int(tokens.shape[1]),
                action_shape=list(batch['action'].shape), whole_policy_constructed=False, backward_executed=False)


def check():
    c.require_decision()
    c.claim_output(c.ADMISSION)
    report = dict(status='preparing', stage='full_cpu_semantic_admission')
    try:
        c.write_json(c.CPU_ADMISSION, report)
        recomputed = c.parent_fingerprint()  # first: nothing else is trusted until the seal reproduces
        if recomputed != c.SEAL:
            raise ValueError('Props P2 parent fingerprint differs from the seal')
        before = c.current_binding()
        semantic = c.semantic_gate()
        ds, starts, stats = c.load_data()
        evidence = processor_check(ds, [starts[0], starts[len(starts)//480-1], starts[len(starts)//480], starts[-1]], stats)
        if c.current_binding() != before:
            raise ValueError('Source/data/assets changed during CPU admission')
        report = c.cpu_proof(before, semantic, c.statistics_json(stats), evidence, parent_fingerprint=recomputed)
        c.validate_cpu(report, before)
        c.write_json(c.CPU_ADMISSION, report)
    except BaseException as exc:
        failure(c.ADMISSION, report, exc)
        raise
    print('CPU data/processor/semantic admission passed; no GPU work', flush=True)


def prepare_data():
    binding = c.admission()
    ds, starts, stats = c.load_data()
    if c.statistics_json(stats) != json.loads(c.CPU_ADMISSION.read_text())['training_stats']:
        raise ValueError('Fresh train-only statistics differ from current CPU admission')
    return ds, starts, stats, binding


def train(output, preflight, report, resume=0, reference=None, stop_at=None):
    """The training loop for the decided strategy; `resume` is the resume number (0 = fresh)."""
    decision = c.require_decision()
    name, microbatch = decision['strategy'], decision['microbatch']

    def progress(**fields):
        report.update(fields)
        c.write_json(output/'result.json', report)

    ds, starts, stats, binding = prepare_data()
    encoded_stats, cpu_hash = c.statistics_json(stats), c.sha256(c.CPU_ADMISSION)
    scratch_hash = proof_hash = None
    if not preflight:
        scratch_hash = c.sha256(c.SCRATCH/'result.json')
        proof = json.loads((c.SCRATCH/'result.json').read_text())
        c.validate_scratch(proof, binding, c.SCRATCH, encoded_stats, cpu_hash)
        proof_hash = c.sha256(c.PROOF/'result.json')
        c.validate_resume_proof(json.loads((c.PROOF/'result.json').read_text()), binding, proof, c.PROOF)
    elif reference is not None:
        c.validate_scratch(json.loads((reference/'result.json').read_text()), binding, reference, encoded_stats, cpu_hash)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required; no fallback')
    resources = require_resources(output, preflight)
    torch.cuda.reset_peak_memory_stats()
    random.seed(c.SEED)
    np.random.seed(c.SEED)
    torch.manual_seed(c.SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    rows, verified = c.indices(ds.meta.episodes)
    if starts != verified or (len(rows), len(starts)) != (c.TRAINING_ROWS, len(c.windows())):
        raise ValueError('Derived training counts changed')
    snapshot = None
    if resume:
        snapshot, meta, skipped = load_snapshot(output, binding)
        restored = json.loads((snapshot/'report.json').read_text())
        if (restored.get('binding') != binding or restored['completed_updates'] != meta['update']
                or json.loads((output/'training_stats.json').read_text()) != encoded_stats):
            raise ValueError('Snapshot report does not belong to this binding/statistics')
        moved = quarantine(output, resume, restored, meta['update'])
        report.clear()
        report.update(restored)
        report.setdefault('resumes', []).append(dict(
            number=resume, from_update=meta['update'], snapshot=snapshot.name, skipped_snapshots=skipped,
            quarantined=moved, resource_admission=resources, resumed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        progress(status='training', stage='resumed', scratch_result_sha256=scratch_hash, resume_proof_sha256=proof_hash)
    else:
        c.write_json(output/'training_stats.json', encoded_stats)
        progress(status='training', stage='fresh_base_load', binding=binding, cpu_admission_sha256=cpu_hash,
                 scratch_result_sha256=scratch_hash, resume_proof_sha256=proof_hash,
                 training_rows=len(rows), complete_training_windows=len(starts), resource_admission=resources,
                 training_stats_sha256=c.sha256(output/'training_stats.json'),
                 checkpoints=[], checkpoint_files={}, completed_updates=0, rollouts=0,
                 device=dict(name=torch.cuda.get_device_name(), properties=str(torch.cuda.get_device_properties(0))),
                 initial_memory=memory())
    policy, config, masters, load_report = st.build_policy(name)
    policy = policy.to('cuda')
    config.device = 'cuda'
    config.save_pretrained(output/'initial_config')
    strategy = st.make_strategy(name, policy, masters, 'cuda')
    pre, _ = c.make_processors(config, stats)
    progress(load_report=load_report, dtype_report=strategy.dtype_report(), after_load_memory=memory(), stage='training')
    samples = c.sample_updates(starts)
    done, rng_tensors = 0, None
    if resume:
        done = meta['update']
        strategy.restore(torch.load(snapshot/'model.pt', map_location='cpu', weights_only=True),
                         torch.load(snapshot/'optimizer.pt', map_location='cpu', weights_only=True))
        rng_tensors = torch.load(snapshot/'rng.pt', map_location='cpu', weights_only=True)
        for _ in range(done):  # the sampler is a pure function of the seed: replay to the position
            next(samples)
    began = time.monotonic()
    elapsed_before = report.get('elapsed_seconds', 0.)
    policy.train()
    if resume:  # last: nothing after this may draw from the RNG before the first update
        restore_rng(meta['python_rng'], meta['numpy_rng'], rng_tensors)
    # Horizon comes from the contract; hardcoding it once silently truncated a run.
    maximum = 2 if preflight else c.recipe(False)['max_optimizer_updates']
    snapshots = c.snapshot_updates(preflight)
    fetch = Prefetcher(LockedDataset(ds), samples, microbatch)
    session_began, session_updates = time.monotonic(), 0
    try:
        with (output/'metrics.jsonl').open('a' if resume else 'x') as log:
            for update in range(done+1, maximum+1):
                tick = time.monotonic()
                row = one_update(update, strategy, policy, pre, fetch, microbatch)
                row.update(seconds=time.monotonic()-tick, memory=memory(), sampled_windows=update*c.EFFECTIVE_BATCH)
                if update == done+1 or update % TELEMETRY_EVERY == 0:
                    row['gpu'] = gpu_telemetry()
                    if too_hot(row['gpu']):
                        slept = cool_down()
                        if slept is None:
                            raise RuntimeError(f'GPU above {COOL_C} C for {HOT_LIMIT_S} s; resume from the last snapshot')
                        row['thermal_pause_seconds'] = slept
                        print('THERMAL', json.dumps(dict(update=update, slept=slept, gpu=row['gpu'])), flush=True)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                session_updates += 1
                report['completed_updates'] = update
                if update == 1 or update % 20 == 0 or update == maximum:
                    progress(last_metrics=row, elapsed_seconds=elapsed_before+time.monotonic()-began)
                    print('TRAIN', json.dumps(row, allow_nan=False), flush=True)
                if update in c.checkpoint_updates(preflight):
                    marker = save_checkpoint(policy, strategy, output, update, final=(update == maximum))
                    c.require_marker(marker, update)
                    key = c.checkpoint_directory(update)
                    report['checkpoints'].append(marker)
                    report['checkpoint_files'][key] = c.inventory(output/'checkpoints'/key)
                    progress(startup_save_reload_passed=True)
                if update in snapshots:
                    report.setdefault('snapshot_updates_written', []).append(update)
                    progress(last_snapshot_seconds=save_snapshot(strategy, report, output, update, binding))
                    later = [u for u in snapshots if u > update]
                    pace = (time.monotonic()-session_began)/session_updates
                    if should_pause(time.time(), stop_at, pace, (later[0] if later else maximum)-update):
                        progress(status='paused', stage='paused_for_deadline', paused_at_update=update,
                                 session_updates=session_updates, session_seconds_per_update=pace,
                                 elapsed_seconds=elapsed_before+time.monotonic()-began)
                        print('PAUSED', json.dumps(dict(update=update, seconds_per_update=pace)), flush=True)
                        return
    finally:
        fetch.close()
    if c.admission() != binding or c.sha256(c.CPU_ADMISSION) != cpu_hash:
        raise ValueError('Post-training dataset/source/cache/CPU proof changed')
    if c.sha256(output/'training_stats.json') != report['training_stats_sha256']:
        raise ValueError('Training statistics bytes changed')
    if not preflight and (c.sha256(c.SCRATCH/'result.json') != scratch_hash or c.sha256(c.PROOF/'result.json') != proof_hash):
        raise ValueError('Scratch or resume proof changed during main')
    if reference is not None:
        progress(resume_proof=compare_with_reference(output, reference, maximum))
    c.verify_checkpoints(report, output, preflight)
    progress(status='complete', stage='training_complete', completed_updates=maximum, final_data_binding_passed=True,
             selected_checkpoint_update=None if preflight else maximum, rollouts=0, final_memory=memory())


def main(argv=None):
    args = parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    torch.set_num_threads(4)
    if args.probe_memory:
        probe_memory(args)
        return
    if args.check:
        check()
        return
    c.require_decision()
    preflight = args.preflight or args.resume_proof
    output = c.PROOF if args.resume_proof else c.SCRATCH if args.preflight else c.OUTPUT
    resume = 0
    if args.resume:
        resume = c.claim_resume(output)
        # A hard kill leaves no failure.json; result.json is then the only record of where it stopped.
        shutil.copy2(output/'result.json', output/f'result_before_resume_{resume:02d}.json')
    else:
        c.claim_output(output)
        if args.resume_proof:  # a fresh dir seeded with the scratch run's files up to snapshot 1
            shutil.copytree(c.SCRATCH, output, dirs_exist_ok=True, ignore=shutil.ignore_patterns(
                'step_000002', 'failure*.json', 'result.json', 'run_claim.json', 'resume_claim_*.json'))
            resume = c.claim_resume(output)
    report = dict(status='preparing', stage='admission', **c.recipe(preflight))
    try:
        if not resume:
            c.write_json(output/'result.json', report)
        train(output, preflight, report, resume=resume, reference=c.SCRATCH if args.resume_proof else None,
              stop_at=args.stop_at)
    except BaseException as exc:
        failure(output, report, exc)
        raise
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    raise SystemExit(main())
