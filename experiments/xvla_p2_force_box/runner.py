"""CPU admission, two-update scratch, resume proof, loop equivalence, or X-VLA fine-tune with force/torque in the state and
the target's box drawn on both camera frames.

The update math is M2's (and so P2's), unchanged, in `one_update`; the run differs from M2's only in its data
(`contract.py`: the loader draws the target's box on both frames). What differs from P2's runner is plumbing only (measured on
a real P1 update, 0.896 s: per-tensor finiteness checks 0.321 s, serial decoding with the GPU idle 0.129 s):
`verify_precision_fast` runs the same checks with one fused GPU pass, and `Prefetcher` decodes the next
update's rows in a background thread, in the sampler's order. `--equivalence` proves the two loops give
byte-identical weights before `--train` is allowed; `--zero-equivalence` proves this loop with the overlay replaced by the
identity reproduces M2's runner byte for byte (and that the overlay itself is detected as a difference). `--check` also writes
`colour_check.json` and `overlay_samples/*.png` for the user's approval. `--stop-at` pauses cleanly at a snapshot before a
deadline; a thermal guard sleeps (no effect on the math) when the GPU runs hot.
"""
import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from experiments.xvla_trial.assets import safe_error
from . import contract as c

EXPECTED_PARAM_TENSORS = 901
EXPECTED_PARAM_NUMEL = 879482456
EQUIVALENCE_UPDATES = 30
ZERO_EQUIVALENCE_UPDATES = 30
# M2's runner: the reference loop of `--zero-equivalence`, imported only there. Its files are bound (`contract.REFERENCE_FILES`).
REFERENCE_RUNNER = 'experiments.xvla_p2_force.runner'
TELEMETRY_EVERY = 20
HOT_C, COOL_C, HOT_LIMIT_S = 83, 75, 1800  # firmware target is 85 C on this card (300 W hard cap)
SNAPSHOT_WRITE_S = 60  # measured ~16 s; margin for the pause decision


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('check', 'preflight', 'resume-proof', 'equivalence', 'zero-equivalence', 'train'):
        mode.add_argument('--'+name, action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='continue a paused or failed --train from its newest verified snapshot')
    parser.add_argument('--stop-at', type=float, default=None,
                        help='unix time by which --train must have paused at a snapshot')
    args = parser.parse_args(argv)
    if args.resume and not args.train:
        parser.error('--resume only continues --train')
    if args.stop_at is not None and not args.train:
        parser.error('--stop-at only applies to --train')
    return args


# --- plumbing that does not touch the math ------------------------------------------------------

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


def should_pause(now, stop_at, seconds_per_update, updates_to_next_snapshot):
    """Pause now unless the next snapshot can be written before the deadline."""
    if stop_at is None:
        return False
    return now+seconds_per_update*updates_to_next_snapshot+SNAPSHOT_WRITE_S > stop_at


def state_digest(policy):
    hasher = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def require_disk(output, gib):
    if shutil.disk_usage(output).free < gib*1024**3:
        raise RuntimeError(f'Require {gib} GiB free; no deletion/fallback')


def require_resources(output, preflight):
    # Train: 7 checkpoints (6x3.3 + 9.9 = 29.7 GB), plus the resume snapshot working set (2 kept +
    # 1 being written = 3x~10.5 GB = 31.5 GB) peaks at ~61.2 GB: require 65 GiB. Preflight / resume
    # proof: step 1 and 2 plus one snapshot, well under 28 GiB.
    require_disk(output, 28 if preflight else 65)
    free, total = torch.cuda.mem_get_info()
    if free < 28*1024**3:
        raise RuntimeError('Require 28 GiB GPU free; do not interfere with other workloads')
    return dict(disk_free=shutil.disk_usage(output).free, gpu_free=free, gpu_total=total)


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
    evidence = c.check_model_inputs(policy, batch, raw['observation.state'], selected, {key: raw[key] for key in c.CAMERAS})
    if not torch.allclose(post(batch['action']), raw['action'], rtol=1e-5, atol=1e-5):
        raise ValueError('Physical action normalization roundtrip failed')
    return {**evidence, 'physical_action_roundtrip': True, 'dataset_rows': selected,
            'whole_policy_constructed': False, 'backward_executed': False}


def colour_check(ds):
    """How much of the scene is already the band's colour, measured on UNDRAWN decoded frames of the training episodes.

    `ds` is the joined dataset; its wrapped dataset's rows carry no overlay. Every 10th training episode at frames 0, 225 and 451,
    both cameras. Returns `c.colour_report`; the caller stops the run when it did not pass.
    """
    near = pixels = frames = 0
    for episode in c.COLOUR_EPISODES:
        for frame in c.COLOUR_FRAMES:
            row = ds.ds[episode*c.FRAMES_PER_EPISODE+frame]
            for key in c.CAMERAS:
                image = row[key]
                if image.dtype != torch.float32 or tuple(image.shape) != (3, 512, 512):
                    raise ValueError('Colour check needs float32 (3, 512, 512) frames')
                near += c.overlay.near_magenta(image, c.COLOUR_TOLERANCE)
                pixels += image.shape[1]*image.shape[2]
                frames += 1
    return c.colour_report(near, pixels, frames)


SAMPLE_FRAMES = (('first', None), ('closing', 1), ('closed_and_lifted', 2))


def overlay_samples(ds, directory, episodes=None):
    """One PNG sheet per episode for the user's approval: the front camera on top, the gripper camera below, three frames across
    (the first, the first `closing` frame and the first `closed_and_lifted` frame), all through the training loader.

    One training episode per object (the first of each), 6 sheets, plus `index.json` naming every panel.
    """
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=False)
    objects, segments = c.boxes.labels()
    if episodes is None:
        episodes = [min(i for i in c.TRAIN if objects[i] == name) for name in sorted(set(objects[i] for i in c.TRAIN))]
    index = []
    for episode in episodes:
        frames = [0 if value is None else c.boxes.first_frame(segments[episode], value) for _, value in SAMPLE_FRAMES]
        rows = [ds[episode*c.FRAMES_PER_EPISODE+frame] for frame in frames]
        sheet = np.zeros((2*512, len(frames)*512, 3), dtype=np.uint8)
        for column, row in enumerate(rows):
            for line, key in enumerate(c.CAMERAS):
                panel = (row[key].permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
                sheet[line*512:(line+1)*512, column*512:(column+1)*512] = panel
        name = f'episode_{episode:03d}_{objects[episode]}.png'
        Image.fromarray(sheet).save(directory/name)
        index.append(dict(file=name, episode=episode, object=objects[episode],
                          columns=[dict(label=label, frame=frame) for (label, _), frame in zip(SAMPLE_FRAMES, frames)],
                          rows=['front camera (observation.images.image)', 'gripper camera (observation.images.image2)']))
    c.write_json(directory/'index.json', index)
    return index


def check():
    c.claim_output(c.ADMISSION)
    report = dict(status='preparing', stage='full_cpu_semantic_admission')
    try:
        c.write_json(c.CPU_ADMISSION, report)
        recomputed = c.parent_fingerprint()  # first: nothing else is trusted until the seal reproduces
        if recomputed != c.SEAL:
            raise ValueError('Props P2 parent fingerprint differs from the seal')
        sidecar = c.forces.verify_sidecar()['fingerprint']  # and S0's seal, before the binding or any data
        before = c.current_binding()
        semantic = c.semantic_gate()  # the collection's own proof, bound by digest in `before`
        ds, starts, stats = c.load_data()
        evidence = processor_check(ds, [starts[0], starts[421], starts[422], starts[-1]], stats)
        colour = colour_check(ds)
        c.write_json(c.ADMISSION/'colour_check.json', colour)
        if not colour['passed']:
            raise ValueError(f'The band colour is already in the scene ({colour["fraction"]:.6%} of undrawn pixels, '
                             f'limit {c.COLOUR_THRESHOLD:.4%}); the user must pick another colour')
        overlay_samples(ds, c.ADMISSION/'overlay_samples')
        if c.current_binding() != before:
            raise ValueError('Source/data/assets changed during CPU admission')
        report = c.cpu_proof(before, semantic, statistics_json(stats), evidence, parent_fingerprint=recomputed,
                             sidecar_fingerprint=sidecar, colour=colour)
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


SNAPSHOT_FILES = ('model.pt', 'optimizer.pt', 'rng.pt', 'report.json')


class BindingChanged(Exception):
    """The sources or data differ from what the snapshot was written under; no snapshot is usable."""


def capture_rng():
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=numpy_state, torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None)


def restore_rng(python, numpy, tensors):
    random.setstate((python[0], tuple(python[1]), python[2]))
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(tensors['torch'])
    if 'cuda' in tensors:
        torch.cuda.set_rng_state(tensors['cuda'])


def save_snapshot(policy, optimizer, report, output, update, binding):
    """Atomic: written under tmp_, hashed, renamed; only the newest KEEP_SNAPSHOTS survive."""
    require_disk(output, 12)
    began = time.monotonic()
    root = output/'resume'
    root.mkdir(exist_ok=True)
    tmp = root/f'tmp_{update:06d}'
    tmp.mkdir()
    torch.save(policy.state_dict(), tmp/'model.pt')
    torch.save(optimizer.state_dict(), tmp/'optimizer.pt')
    rng = capture_rng()
    torch.save({'torch': rng['torch'], **({'cuda': rng['cuda']} if rng['cuda'] is not None else {})},
               tmp/'rng.pt')
    c.write_json(tmp/'report.json', report)
    meta = dict(update=update, binding_digest=c.digest_json(binding),
                files={name: c.sha256(tmp/name) for name in SNAPSHOT_FILES},
                python_rng=[rng['python'][0], list(rng['python'][1]), rng['python'][2]],
                numpy_rng=[rng['numpy'][0], rng['numpy'][1].tolist(), *rng['numpy'][2:]])
    c.write_json(tmp/'snapshot.json', meta)
    tmp.rename(root/f'snap_{update:06d}')
    for old in sorted(root.glob('snap_*'))[:-c.KEEP_SNAPSHOTS]:
        shutil.rmtree(old)  # own regenerable intermediate; the newer snapshots cover it
    seconds = time.monotonic()-began
    print('SNAPSHOT', json.dumps(dict(update=update, seconds=seconds,
          bytes=sum(f.stat().st_size for f in (root/f'snap_{update:06d}').iterdir()))), flush=True)
    return seconds


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
    keys = ('update', 'loss', 'grad_norm', 'dataset_rows', 'learning_rates')
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


def train(output, preflight, report, resume=0, reference=None, stop_at=None):
    """Training loop shared with the other X-VLA packages, bound to the props P2 dataset.

    `resume` is the resume number (0 = fresh). The snapshot is read from `output`. `stop_at` (unix time)
    pauses at the first snapshot after which the next one could not finish in time.
    """
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors

    def progress(**fields):
        report.update(fields)
        c.write_json(output/'result.json', report)

    ds, starts, stats, binding = c.prepare_data()
    encoded_stats, cpu_hash = statistics_json(stats), c.sha256(c.CPU_ADMISSION)
    scratch_hash = proof_hash = None
    if not preflight:
        scratch_hash = c.sha256(c.SCRATCH/'result.json')
        proof = json.loads((c.SCRATCH/'result.json').read_text())
        c.validate_scratch(proof, binding, c.SCRATCH, encoded_stats, cpu_hash)
        proof_hash = c.sha256(c.PROOF/'result.json')
        c.validate_resume_proof(json.loads((c.PROOF/'result.json').read_text()), binding, proof, c.PROOF)
        c.validate_equivalence(json.loads((c.EQUIVALENCE/'result.json').read_text()), binding)
        c.validate_zero_equivalence(json.loads((c.ZERO_EQUIVALENCE/'result.json').read_text()), binding)
        if json.loads((c.SCRATCH/'training_stats.json').read_text()) != encoded_stats:
            raise ValueError('Scratch training statistics changed')
    elif reference is not None:
        c.validate_scratch(json.loads((reference/'result.json').read_text()), binding, reference,
                           encoded_stats, cpu_hash)
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
    if starts != verified or (len(rows), len(starts)) != (c.TRAINING_ROWS, c.COMPLETE_STARTS):
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
        progress(status='training', stage='resumed', scratch_result_sha256=scratch_hash,
                 resume_proof_sha256=proof_hash)
    else:
        c.write_json(output/'training_stats.json', encoded_stats)
        progress(status='training', stage='fresh_base_load', binding=binding, cpu_admission_sha256=cpu_hash,
                 scratch_result_sha256=scratch_hash, resume_proof_sha256=proof_hash,
                 scratch_used_for_initialization=False,
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
    checked_rows = [starts[0], starts[421], starts[422], starts[-1]]
    raw = c.collate_rows([ds[i] for i in checked_rows])
    checked = pre(raw)
    progress(model_input_precheck=c.check_model_inputs(policy, checked, raw['observation.state'], checked_rows,
                                                       {key: raw[key] for key in c.CAMERAS}),
             parameter_tensors=len(params), parameter_count=sum(x.numel() for x in params.values()),
             optimizer_groups=[dict(name=g['name'], parameter_tensors=len(g['params']),
                                    weight_decay=g['weight_decay']) for g in optimizer.param_groups],
             after_load_memory=c.p.memory(), stage='training')
    del raw, checked
    samples = c.sample_batches(starts)
    done, rng_tensors = 0, None
    if resume:
        done = meta['update']
        policy.load_state_dict(torch.load(snapshot/'model.pt', map_location='cpu', weights_only=True), strict=True)
        saved = torch.load(snapshot/'optimizer.pt', map_location='cpu', weights_only=True)
        optimizer.load_state_dict(saved)
        c.p.verify_precision(params, optimizer)
        for group, stored in zip(optimizer.param_groups, saved['param_groups'], strict=True):
            for p, index in zip(group['params'], stored['params'], strict=True):
                if any(not torch.equal(optimizer.state[p][key].cpu(), saved['state'][index][key])
                       for key in ('exp_avg', 'exp_avg_sq', 'step')):
                    raise ValueError('Restored Adam state differs from the snapshot')
        del saved
        rng_tensors = torch.load(snapshot/'rng.pt', map_location='cpu', weights_only=True)
        for _ in range(2*done):  # the sampler is a pure function of the seed: replay to the position
            next(samples)
    began = time.monotonic()
    elapsed_before = report.get('elapsed_seconds', 0.)
    policy.train()
    if resume:  # last: nothing after this may draw from the RNG before the first update
        restore_rng(meta['python_rng'], meta['numpy_rng'], rng_tensors)
    # Horizon comes from the contract; hardcoding it here silently truncated a 16700-update run.
    maximum = 2 if preflight else c.recipe(False)['max_optimizer_updates']
    snapshots = c.snapshot_updates(preflight)
    data = LockedDataset(ds)
    fetch = Prefetcher(data, samples)
    session_began, session_updates = time.monotonic(), 0
    try:
        with (output/'metrics.jsonl').open('a' if resume else 'x') as log:
            for update in range(done+1, maximum+1):
                tick = time.monotonic()
                row = one_update(update, policy, params, optimizer, pre, fetch, verify_precision_fast)
                row.update(seconds=time.monotonic()-tick, memory=c.p.memory(), sampled_chunks=update*8)
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
                    marker = save_checkpoint(policy, pre, post, optimizer, data, starts[0], output, update,
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
                if update in snapshots:
                    report.setdefault('snapshot_updates_written', []).append(update)
                    progress(last_snapshot_seconds=save_snapshot(policy, optimizer, report, output, update, binding))
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
    if not preflight and (c.sha256(c.SCRATCH/'result.json') != scratch_hash
                          or c.sha256(c.PROOF/'result.json') != proof_hash):
        raise ValueError('Scratch or resume proof changed during main')
    if reference is not None:
        progress(resume_proof=compare_with_reference(output, reference, maximum))
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


def equivalence():
    """30 updates with P1's loop, then 30 with the fast loop from the same fresh state: byte-equal or fail."""
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
    c.claim_output(c.EQUIVALENCE)
    report = dict(status='preparing', stage='equivalence', updates=EQUIVALENCE_UPDATES)
    try:
        c.write_json(c.EQUIVALENCE/'result.json', report)
        ds, starts, stats, binding = c.prepare_data()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 required; no fallback')
        require_resources(c.EQUIVALENCE, True)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        runs = {}
        for name in ('p1_loop', 'fast_loop'):
            random.seed(c.SEED)
            np.random.seed(c.SEED)
            torch.manual_seed(c.SEED)
            config = c.make_config('cpu')
            policy, params = load_fresh_policy(config)
            policy = policy.to('cuda')
            config.device = 'cuda'
            pre, _ = make_xvla_pre_post_processors(config, stats)
            optimizer = config.get_optimizer_preset().build(params)
            policy.train()
            samples = c.sample_batches(starts)
            if name == 'p1_loop':
                fetch, verify, closer = serial_fetch(ds, samples), c.p.verify_precision, None
            else:
                fetch = closer = Prefetcher(LockedDataset(ds), samples)
                verify = verify_precision_fast
            rows, seconds = [], []
            try:
                for update in range(1, EQUIVALENCE_UPDATES+1):
                    tick = time.monotonic()
                    rows.append(one_update(update, policy, params, optimizer, pre, fetch, verify))
                    seconds.append(time.monotonic()-tick)
            finally:
                if closer is not None:
                    closer.close()
            optimizer_digest = hashlib.sha256()
            for group in optimizer.param_groups:
                for p in group['params']:
                    for key in ('exp_avg', 'exp_avg_sq'):
                        optimizer_digest.update(optimizer.state[p][key].cpu().numpy().tobytes())
            runs[name] = dict(rows=rows, model=state_digest(policy), optimizer=optimizer_digest.hexdigest(),
                              seconds_per_update=float(np.median(seconds[5:])))
            del policy, params, optimizer, pre
            gc.collect()
            torch.cuda.empty_cache()
        a, b = runs['p1_loop'], runs['fast_loop']
        report.update(status='complete', binding=binding,
                      metrics_equal=a['rows'] == b['rows'], model_state_equal=a['model'] == b['model'],
                      optimizer_state_equal=a['optimizer'] == b['optimizer'],
                      p1_loop_seconds_per_update=a['seconds_per_update'],
                      fast_loop_seconds_per_update=b['seconds_per_update'],
                      model_sha256=b['model'], final_loss=b['rows'][-1]['loss'])
        c.write_json(c.EQUIVALENCE/'result.json', report)
        c.validate_equivalence(report, binding)
    except BaseException as exc:
        failure(c.EQUIVALENCE, report, exc)
        raise
    print('EQUIVALENCE', json.dumps({k: v for k, v in report.items() if k != 'binding'}), flush=True)


@contextlib.contextmanager
def overlay_identity():
    """The loader's overlay replaced by the identity for the scope (rows come back exactly as the dataset yields them)."""
    original = c.OVERLAY
    c.OVERLAY = lambda row, box: row
    try:
        yield
    finally:
        c.OVERLAY = original


def optimizer_digest(optimizer):
    digest = hashlib.sha256()
    for group in optimizer.param_groups:
        for p in group['params']:
            for key in ('exp_avg', 'exp_avg_sq'):
                digest.update(optimizer.state[p][key].cpu().numpy().tobytes())
    return digest.hexdigest()


def thirty_updates(mod, ds, starts, stats, wrap):
    """30 updates of `mod`'s own loop (its `one_update`, prefetcher, sampler and config) from a fresh seeded state."""
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
    random.seed(mod.c.SEED)
    np.random.seed(mod.c.SEED)
    torch.manual_seed(mod.c.SEED)
    config = mod.c.make_config('cpu')
    policy, params = mod.load_fresh_policy(config)
    policy = policy.to('cuda')
    config.device = 'cuda'
    pre, _ = make_xvla_pre_post_processors(config, stats)
    pre = wrap(pre)
    optimizer = config.get_optimizer_preset().build(params)
    policy.train()
    fetch = mod.Prefetcher(mod.LockedDataset(ds), mod.c.sample_batches(starts))
    rows, seconds = [], []
    try:
        for update in range(1, ZERO_EQUIVALENCE_UPDATES+1):
            tick = time.monotonic()
            rows.append(mod.one_update(update, policy, params, optimizer, pre, fetch, mod.verify_precision_fast))
            seconds.append(time.monotonic()-tick)
    finally:
        fetch.close()
    result = dict(rows=rows, model=state_digest(policy), optimizer=optimizer_digest(optimizer),
                  seconds_per_update=float(np.median(seconds[5:])))
    del policy, params, optimizer, pre
    gc.collect()
    torch.cuda.empty_cache()
    return result


def zero_equivalence():
    """M2's own loop on M2's own dataset class, then this loop with the overlay replaced by the identity, 30 updates each from the
    same fresh state: byte-equal metrics, weights and Adam moments, or fail. Then this loop again with the overlay ON, which
    must be reported unequal (the mutation check).

    With the overlay the identity this loop's data are M2's rows, so the real run differs from A2's only by the drawn box. The
    reference is M2's runner itself (`REFERENCE_RUNNER`, bound), fed by M2's `JoinedDataset` over the raw dataset, so the
    comparison involves none of this package's loader.
    """
    reference = importlib.import_module(REFERENCE_RUNNER)
    c.claim_output(c.ZERO_EQUIVALENCE)
    report = dict(status='preparing', stage='zero_equivalence', updates=ZERO_EQUIVALENCE_UPDATES, overlay='identity',
                  reference=REFERENCE_RUNNER)
    try:
        c.write_json(c.ZERO_EQUIVALENCE/'result.json', report)
        ds, starts, stats, binding = c.prepare_data()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 required; no fallback')
        require_resources(c.ZERO_EQUIVALENCE, True)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        a = thirty_updates(reference, reference.c.JoinedDataset(ds.ds), starts, stats, lambda pre: pre)
        with overlay_identity():
            b = thirty_updates(sys.modules[__name__], ds, starts, stats, lambda pre: pre)
        mutant = thirty_updates(sys.modules[__name__], ds, starts, stats, lambda pre: pre)  # the real overlay, on
        report.update(status='complete', binding=binding,
                      metrics_equal=a['rows'] == b['rows'], model_state_equal=a['model'] == b['model'],
                      optimizer_state_equal=a['optimizer'] == b['optimizer'],
                      overlay_on_metrics_equal=a['rows'] == mutant['rows'],
                      overlay_on_model_state_equal=a['model'] == mutant['model'],
                      overlay_on_optimizer_state_equal=a['optimizer'] == mutant['optimizer'],
                      m2_loop_seconds_per_update=a['seconds_per_update'],
                      box_loop_seconds_per_update=b['seconds_per_update'],
                      overlay_on_seconds_per_update=mutant['seconds_per_update'],
                      model_sha256=b['model'], overlay_on_model_sha256=mutant['model'], final_loss=b['rows'][-1]['loss'],
                      overlay_on_final_loss=mutant['rows'][-1]['loss'])
        c.write_json(c.ZERO_EQUIVALENCE/'result.json', report)
        c.validate_zero_equivalence(report, binding)
    except BaseException as exc:
        failure(c.ZERO_EQUIVALENCE, report, exc)
        raise
    print('ZERO_EQUIVALENCE', json.dumps({k: v for k, v in report.items() if k != 'binding'}), flush=True)


def main(argv=None):
    args = parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    torch.set_num_threads(4)
    if args.check:
        check()
        return
    if args.equivalence:
        equivalence()
        return
    if args.zero_equivalence:
        zero_equivalence()
        return
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
    # Evaluation/rollout is a separate protocol; --train never launches it.


if __name__ == '__main__':
    raise SystemExit(main())
