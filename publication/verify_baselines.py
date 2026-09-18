"""Independent CPU reference and relocated-release workers; no simulator required."""
import argparse
import importlib.util
import json
from pathlib import Path

from .load_baseline import offline_mode
from .package_baselines import (IMAGE, MODELS, RELEASE_RELATIVE, audit_json,
                                compare_safetensors, inventory, sha256, write_json)


def read_observation(path):
    import numpy as np
    import torch
    # Read individual observation entries only. Never materialize action or trace.
    with np.load(path, allow_pickle=False) as stored:
        result = {'observation.state': torch.from_numpy(stored['observation.state'].copy()),
                  'task': str(stored['task'].item())}
        for key in ('observation.images.image', 'observation.images.image2'):
            image = stored[key]
            if image.shape != (512, 512, 3) or image.dtype != np.uint8:
                raise ValueError('Recorded RGB uint8 HWC observation required')
            result[key] = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).contiguous()
    return result


def check_inventory(root):
    expected = json.loads((Path(root) / 'inventory.json').read_text())
    if inventory(root, exclude=('inventory.json',)) != expected:
        raise ValueError('Release inventory mismatch (added, removed or changed file)')
    return expected


def reference_load(checkpoint):
    """Original config/assets and original strict-copy algorithm, not release loader."""
    import torch
    from safetensors.torch import load_model
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True, token=False)
    config.device = 'cpu'
    if config.type == 'smolvla':
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        config.load_vlm_weights = False
        policy = SmolVLAPolicy(config)
        with torch.no_grad():
            for parameter in policy.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
    elif config.type == 'xvla':
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        policy = XVLAPolicy(config)
        if any(p.dtype != torch.float32 for p in policy.parameters()):
            raise ValueError('Reference X-VLA construction is not FP32')
    else:
        raise ValueError('Unexpected reference policy')
    load_model(policy, str(checkpoint / 'model.safetensors'), strict=True, device='cpu')
    policy.eval()
    pre, post = make_pre_post_processors(
        config, pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor': {'device': 'cpu'}},
        postprocessor_overrides={'device_processor': {'device': 'cpu'}})
    return policy, pre, post


def model_storage_check(policy, checkpoint):
    import torch
    from safetensors import safe_open
    state = policy.state_dict()
    dtypes = {}
    with safe_open(str(checkpoint / 'model.safetensors'), framework='pt', device='cpu') as stored:
        for key in stored.keys():
            a, b = state[key].detach(), stored.get_tensor(key)
            if (a.device.type != 'cpu' or a.dtype != b.dtype or a.shape != b.shape
                    or not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))):
                raise ValueError('Reference/release loaded storage mismatch: ' + key)
            dtypes[str(b.dtype)] = dtypes.get(str(b.dtype), 0) + 1
        return dict(stored_tensors=len(stored.keys()), state_dict_entries=len(state),
                    dtypes=dtypes, all_stored_tensors_bit_exact=True,
                    trainable_tensors=sum(1 for p in policy.parameters() if p.requires_grad),
                    trainables_fp32=all(p.dtype == torch.float32 for p in policy.parameters() if p.requires_grad))


def worker(mode, checkpoint, raw, output):
    offline_mode()
    import torch
    from safetensors.torch import save_file

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    observation = read_observation(raw)
    if mode == 'source':
        policy, pre, post = reference_load(checkpoint)
        release_loader = None
    else:
        check_inventory(checkpoint)
        spec = importlib.util.spec_from_file_location('standalone_baseline', checkpoint / 'load_baseline.py')
        release_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(release_loader)
        policy, pre, post = release_loader.load_baseline(checkpoint)
    storage = model_storage_check(policy, checkpoint)
    tensors = {}
    metadata = {}
    for prefix, pipeline in (('pre_state', pre), ('post_state', post)):
        for index, step in enumerate(pipeline.steps):
            for key, value in step.state_dict().items():
                tensors[f'{prefix}/{index}/{key}'] = value.detach().cpu().clone().contiguous()
    # Same seed immediately before preprocessing and prediction in both workers.
    policy.reset()
    with torch.inference_mode():
        torch.manual_seed(20260918)
        batch = pre({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in observation.items()})
        if batch.get('action') is not None or batch.get('action_is_pad') is not None:
            raise ValueError('Targets unexpectedly entered prediction inputs')
        batch = {k: v for k, v in batch.items() if k not in ('action', 'action_is_pad')}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                tensors['processed/' + key] = value.detach().cpu().clone().contiguous()
            else:
                metadata[key] = value
        normalized = policy.predict_action_chunk(batch)
        # Clone first: the processor must not make a normalized/physical alias.
        tensors['action/normalized'] = normalized.detach().cpu().clone().contiguous()
        physical = post(normalized.clone())
        tensors['action/physical'] = physical.detach().cpu().clone().contiguous()
    if not all(torch.isfinite(v).all() for v in tensors.values()):
        raise ValueError('Non-finite processor state/input/output')
    helper_matches = None
    if release_loader is not None:
        helper = release_loader.predict(policy, pre, post, observation, seed=20260918)
        helper_matches = torch.equal(helper, physical)
        if not helper_matches:
            raise ValueError('Public prediction helper differs from direct pipeline')
    save_file(tensors, str(output / 'probe.safetensors'))
    record = dict(mode=mode, model=policy.config.type, device='cpu', seed=20260918,
                  raw_tick_sha256=sha256(raw), observation_keys=sorted(observation),
                  observation_task=observation['task'], processed_metadata=metadata,
                  tensor_shapes={k: list(v.shape) for k, v in tensors.items()}, storage=storage,
                  physical_action_shape=list(physical.shape), public_helper_matches=helper_matches,
                  weights_sha256=sha256(checkpoint / 'model.safetensors'),
                  standalone_release_loader=release_loader is not None,
                  probe_sha256=sha256(output / 'probe.safetensors'),
                  threads=2, deterministic_algorithms=True)
    write_json(output / 'result.json', record)
    print('CPU_PROBE_COMPLETE', mode, policy.config.type, list(physical.shape), flush=True)


def report(repo):
    root = repo / RELEASE_RELATIVE
    results = {}
    for model in MODELS:
        release = root / 'packages' / model
        check_inventory(release)
        manifest = json.loads((release / 'release_manifest.json').read_text())
        source = repo / manifest['source_checkpoint']
        for name, expected in manifest['source_files'].items():
            if sha256(source / name) != expected['sha256']:
                raise ValueError('Source changed after packaging')
        a, b = (root / 'verification' / f'{model}-{mode}' for mode in ('source', 'portable'))
        reference, portable = (json.loads((p / 'result.json').read_text()) for p in (a, b))
        if reference['mode'] != 'source' or portable['mode'] != 'portable':
            raise ValueError('Probe modes are not independent source/release')
        for key in ('model', 'device', 'seed', 'raw_tick_sha256', 'observation_keys',
                    'observation_task', 'processed_metadata', 'storage', 'weights_sha256',
                    'physical_action_shape', 'tensor_shapes', 'threads', 'deterministic_algorithms'):
            if reference[key] != portable[key]:
                raise ValueError('Probe metadata differs: ' + key)
        if not portable['public_helper_matches']:
            raise ValueError('Public helper was not verified')
        for probe, metadata in ((a, reference), (b, portable)):
            if sha256(probe / 'probe.safetensors') != metadata['probe_sha256']:
                raise ValueError('Probe artifact hash mismatch')
        compared = compare_safetensors(a / 'probe.safetensors', b / 'probe.safetensors')
        if portable['weights_sha256'] != manifest['source_files']['model.safetensors']['sha256']:
            raise ValueError('Probe checkpoint does not match release source')
        results[model] = dict(status='PASS', source_checkpoint=manifest['source_checkpoint'],
                              weights_sha256=portable['weights_sha256'],
                              raw_tick_sha256=portable['raw_tick_sha256'],
                              storage=portable['storage'], comparison=compared,
                              prediction_max_absolute_error=0.0,
                              physical_action_shape=portable['physical_action_shape'],
                              release_files=len(check_inventory(release)) + 1,
                              release_bytes=sum(p.stat().st_size for p in release.rglob('*') if p.is_file()))
    evidence = dict(status='PASS', scope='CPU only; one raw observation and seed; no new rollout',
                    image=IMAGE, network='none', runtime='runc', repo_mount='read-only',
                    portable_worker='release-only mount plus verification code and one raw tick; no source caches',
                    source_raw_tick='outputs/fixed_layout/fixed_layout_20260918/attempts/000000/ticks/000060.npz',
                    results=results, test_runner='python -m unittest discover -s publication/tests -t . -v')
    audit_json(evidence)
    write_json(root / 'packaging-verification.json', evidence)
    lines = ['# Portable baseline packaging report', '',
             '**Local packaging and CPU equivalence: PASS. No upload or authentication performed.**', '',
             '## Generated bundles', '']
    for model, item in results.items():
        lines += [f'### {model}', f'- Release: packages/{model}',
                  f'- Source: {item["source_checkpoint"]}',
                  f'- Final weights SHA-256: `{item["weights_sha256"]}`',
                  f'- Release: {item["release_files"]} files, {item["release_bytes"]} bytes.',
                  f'- All {item["storage"]["stored_tensors"]} saved model tensors verified bit-exact in CPU memory, '
                  f'dtypes: {item["storage"]["dtypes"]}.',
                  f'- {item["comparison"]["tensors"]} processor-state/input/output tensors bit-exact; '
                  f'physical chunk shape {item["physical_action_shape"]}; maximum action error 0.', '']
    lines += ['## Method and safety boundary', '',
              f'Pinned image: `{IMAGE}`. CPU-only runc, network none, read-only root/repository, '
              'new publication outputs only writable, temporary HOME/HF caches; no host Python/install.',
              'Original-reference workers use original configs, local assets and strict safetensors copy. '
              'Relocated workers load the copied standalone loader with only the release, test harness and '
              'the single raw tick mounted read-only; original repository/caches are absent.',
              'Each worker uses the same CPU thread count, seed 20260918 and deterministic algorithms. '
              'Only measured state, two RGB observations and instruction are extracted from recorded attempt '
              '000000/tick000060. No action targets or simulator truth enter preprocessing or prediction.',
              'Copied model and normalizer files are SHA-identical and tensor-by-tensor byte-identical via '
              'streamed safe_open. Loaded weights (all tensors, not just trainables), processor states, '
              'processed state/images/tokens and normalized/physical actions are compared. The public predict '
              'helper is also checked against direct pipeline inference.',
              'Config changes are restricted to explicit relative asset paths and CPU device. Scheduler '
              'defaults remain untouched. Each release_manifest.json records source config/file SHA-256, '
              'rewrites with original absolute path value hashes, actual training recipe and local asset pins. '
              'inventory.json hashes all release files except itself; original files are rehashed after verification.', '',
              '## Scope and unresolved publication work', '',
              '- Candidate cards correctly state SmolVLA 18/20 and X-VLA 20/20, same geometry/20 sampling seeds; '
              '20 train/4 same-geometry validation. No spatial generalization or architecture-only comparison.',
              '- Actual schedules use 500 warmup updates, 5000 total, cosine floor 10% of peak. '
              'Completed-report recipe is authoritative over stale serialized defaults.',
              '- No credentials, optimizer/training state, reload pickle, full dataset, simulator assets, '
              'cache internals or source-tree archive are released. Sources were copied, not hardlinked.',
              '- License metadata is absent in the locally pinned manifests. No upstream license is invented; '
              'redistribution clearance remains unestablished and requires publisher review.',
              '- CPU equivalence covers one recorded observation/seed only, not GPU equivalence, new task '
              'evaluation, all inputs, hardware safety, or full environment/simulation reproducibility.',
              '- Runtime dependency versions are recorded but dependencies are not vendored. '
              'No install was performed; inference requires those dependencies.',
              '- Publisher must finalize model cards, refresh affected inventory hashes, perform its own '
              'upload allowlist/security review, authenticate and verify PRIVATE remote publication. '
              'No network call, upload, commit, frozen-source edit or GPU worker was performed here.', '',
              'Detailed evidence: packaging-verification.json and verification/*/{result.json,probe.safetensors}.', '']
    with (root / 'packaging-report.md').open('x') as stream:
        stream.write('\n'.join(lines))
    print('PACKAGING_VERIFICATION_PASS', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    probe = sub.add_parser('worker')
    probe.add_argument('--mode', choices=('source', 'portable'), required=True)
    probe.add_argument('--checkpoint', type=Path, required=True)
    probe.add_argument('--raw', type=Path, required=True)
    probe.add_argument('--output', type=Path, required=True)
    summary = sub.add_parser('report')
    summary.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    if args.command == 'worker':
        worker(args.mode, args.checkpoint, args.raw, args.output)
    else:
        report(args.repo.resolve())


if __name__ == '__main__':
    main()