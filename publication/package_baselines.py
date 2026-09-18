"""Allowlisted immutable release copies. Offline only; never handles credentials."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil

SMOL_ASSETS = ('config.json', 'tokenizer.json', 'tokenizer_config.json',
               'vocab.json', 'merges.txt', 'added_tokens.json', 'special_tokens_map.json',
               'preprocessor_config.json', 'processor_config.json', 'chat_template.json')
XVLA_ASSETS = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')
MODELS = ('smolvla', 'xvla')
RELEASE_RELATIVE = Path('outputs/publication/baselines_20260918')
IMAGE = 'sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def claim_packages(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Output cannot contain symlinks')
    path.mkdir(parents=True, exist_ok=False)
    return path


def audit_json(value):
    """Fail closed on host paths/credential-like strings, without printing them."""
    if isinstance(value, dict):
        for child in value.values():
            audit_json(child)
    elif isinstance(value, list):
        for child in value:
            audit_json(child)
    elif isinstance(value, str):
        if (value.startswith(('/', '~/', 'file:')) or re.match(r'^[A-Za-z]:[\\/]', value)
                or re.search(r'\bhf_[A-Za-z0-9]{20,}\b|\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b', value)):
            raise ValueError('Unreviewed absolute path or credential-like value')


def audit_text(text):
    patterns = (r'\bhf_[A-Za-z0-9]{20,}\b', r'\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b',
                r'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----', r'/vla/',
                r'/home/[^\s"\']+/', r'/root/\.cache/', r'[A-Za-z]:\\Users\\')
    if any(re.search(pattern, text) for pattern in patterns):
        raise ValueError('Release text contains a machine path or credential-like pattern')


def audit_release(root):
    """Scan only exact release allowlists; unexpected files are never opened."""
    root = Path(root)
    manifest = json.loads((root / 'release_manifest.json').read_text())
    model = manifest['model']
    if model not in MODELS:
        raise ValueError('Unsupported release model')
    assets = SMOL_ASSETS if model == 'smolvla' else XVLA_ASSETS
    asset_dir = 'assets/vlm/' if model == 'smolvla' else 'assets/tokenizer/'
    allowed = {'config.json', 'model.safetensors', 'policy_preprocessor.json',
               'policy_postprocessor.json', 'load_baseline.py', 'README.md',
               'release_manifest.json', 'runtime.json', 'inventory.json'}
    # Finalized cards may add these exact legal documents, not arbitrary extras.
    allowed.update(name for name in ('LICENSE', 'NOTICE') if (root / name).is_file())
    allowed.update(asset_dir + name for name in assets)
    for document in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        for step in json.loads((root / document).read_text())['steps']:
            if 'state_file' in step:
                name = step['state_file']
                if not re.fullmatch(r'policy_(?:pre|post)processor_step_\d+_(?:un)?normalizer_processor\.safetensors', name):
                    raise ValueError('Unapproved processor state')
                allowed.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != allowed or any(p.is_symlink() or '.cache' in p.parts for p in root.rglob('*')):
        raise ValueError('Release file allowlist mismatch')
    text_count, header_count = 0, 0
    for name in sorted(allowed):
        path = root / name
        if path.suffix == '.safetensors':
            with path.open('rb') as stream:
                length = int.from_bytes(stream.read(8), 'little')
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError('Unexpected safetensors header length')
                text = stream.read(length).decode('utf-8')
            json.loads(text)
            header_count += 1
        else:
            text = path.read_text()
            text_count += 1
        audit_text(text)
    expected = json.loads((root / 'inventory.json').read_text())
    if inventory(root, exclude=('inventory.json',)) != expected:
        raise ValueError('Release inventory mismatch')
    return dict(status='PASS', files=len(allowed), text_files_scanned=text_count,
                safetensors_headers_scanned=header_count,
                note='Allowlist and recognizable-pattern scan, not proof of all possible secret formats')


def portable_documents(config, pre, post):
    docs = copy.deepcopy({'config.json': config, 'policy_preprocessor.json': pre,
                          'policy_postprocessor.json': post})
    model_type = config['type']
    if model_type not in MODELS:
        raise ValueError('Unsupported model type')
    asset = 'assets/vlm' if model_type == 'smolvla' else 'assets/tokenizer'
    changes = []

    def rewrite(filename, parent, key, after, pointer):
        before = parent[key]
        if before != after:
            # Record exact source-value hash without releasing its machine path.
            old = {'redacted_absolute_path': True, 'utf8_sha256': hashlib.sha256(before.encode()).hexdigest()} \
                if isinstance(before, str) and before.startswith('/') else before
            changes.append(dict(file=filename, pointer='#' + pointer, before=old, after=after))
            parent[key] = after

    cfg = docs['config.json']
    if model_type == 'smolvla' and cfg.get('load_vlm_weights') is not False:
        raise ValueError('Source must already disable separate VLM weight loading')
    key = 'vlm_model_name' if model_type == 'smolvla' else 'tokenizer_name'
    rewrite('config.json', cfg, key, asset, '/' + key)
    rewrite('config.json', cfg, 'device', 'cpu', '/device')
    for filename in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        for index, step in enumerate(docs[filename]['steps']):
            if step['registry_name'] == 'tokenizer_processor':
                rewrite(filename, step['config'], 'tokenizer_name', asset, f'/steps/{index}/config/tokenizer_name')
            if step['registry_name'] == 'device_processor':
                rewrite(filename, step['config'], 'device', 'cpu', f'/steps/{index}/config/device')
    audit_json(docs)
    return docs, changes


def copy_assets(source, target, names):
    source, target = Path(source), Path(target)
    for name in names:
        path = source / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise FileNotFoundError('Missing/non-regular required asset: ' + name)
    target.mkdir(parents=True, exist_ok=False)
    for name in names:
        shutil.copyfile(source / name, target / name)


def compare_safetensors(source, target):
    import torch
    from safetensors import safe_open
    dtypes = {}
    with safe_open(str(source), framework='pt', device='cpu') as a, \
            safe_open(str(target), framework='pt', device='cpu') as b:
        if set(a.keys()) != set(b.keys()):
            raise ValueError('Tensor key sets differ')
        for key in a.keys():
            x, y = a.get_tensor(key), b.get_tensor(key)
            if (x.dtype != y.dtype or x.shape != y.shape
                    or not torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))):
                raise ValueError('Tensor bytes differ: ' + key)
            dtype = str(x.dtype)
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
        return dict(tensors=len(a.keys()), dtypes=dtypes, bit_exact=True)


def inventory(directory, exclude=()):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink():
            raise ValueError('Release cannot contain symlinks')
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            if name not in exclude:
                result[name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    return result


def provenance(repo, model):
    manifest_path = (repo / '.cache/benchmark_models/manifest.json' if model == 'smolvla'
                     else repo / '.cache/xvla_trial/bundle/manifest.json')
    raw = json.loads(manifest_path.read_text())
    assets = raw if model == 'smolvla' else raw['assets']
    sources = []
    for role, entry in assets.items():
        sources.append(dict(role=role, repo_id=entry.get('repo_id', entry.get('repo')),
                            revision=entry['revision'], license=entry.get('license'),
                            license_status='not_recorded_in_local_manifest' if not entry.get('license') else 'local_manifest'))
    return dict(sources=sources, local_manifest_sha256=sha256(manifest_path),
                redistribution_clearance='not_established; no license inferred')


def card_template(model, recipe):
    result = '18/20 (90%)' if model == 'smolvla' else '20/20 (100%)'
    return f'''---
library_name: lerobot
pipeline_tag: robotics
tags:
- fixed-layout
- franka
---
# {model.upper()} fixed-layout baseline — candidate card

Prepared for PRIVATE publication; final card/legal review belongs to the publisher.
Measured final policy success: **{result}**, 20 policy sampling seeds on the
same selected geometry/reset. 20 training demonstrations and 4 same-geometry
validation demonstrations. This is NOT spatial generalization, a hardware safety
claim, or an architecture-only controlled comparison. SmolVLA: 18/20; X-VLA: 20/20.

## Authoritative actual training recipe

5,000 fresh optimizer updates; seed 20260917; no resume. Both actual custom
schedules use **500 warmup updates**, cosine decay through update 5000 to
10% of peak. The saved scheduler fields are preserved for provenance and may
describe stale defaults, NOT the actual training schedule. Actual recipe below
is selected from the completed local training report, not inferred from config.

```json
{json.dumps(recipe, indent=2)}
```

## Offline inference (CPU verified only)

Install compatible dependencies separately (exact observed versions in
runtime.json); this release is not a vendored Python environment. Put this release
directory on Python's import path, then use:

```python
from load_baseline import load_baseline, predict
policy, pre, post = load_baseline()  # defaults to directory beside this loader
# observation: unbatched float32 CPU tensors; no targets or simulator truth
# observation.state: (9,) = 7 measured arm joints + 2 finger positions, radians/meters
# observation.images.image: (3,512,512) RGB float32 in [0,1], overview
# observation.images.image2: (3,512,512) RGB float32 in [0,1], wrist
# task: exact string "pick up the red cube and place it on top of the green cube"
actions = predict(policy, pre, post, observation, seed=20260918)
```

Actions: (1, native_chunk, 8), seven **absolute** joint positions in radians plus
gripper command (0 closed, 255 open). State is measured pre-command state, not
action targets. Native chunks are 50 SmolVLA / 30 X-VLA; evaluations used ten
actions before replanning at 30 Hz. No clipping or robot adapter is supplied;
predicted gripper values can exceed its nominal range. Do not send directly to
hardware. The historical simulator applied arm bounds and binary gripper control.

The standalone loader resolves assets relative to this release from both config
and preprocessor overrides, forbids separate VLM weight loading and strict-loads
all final weights. SmolVLA trainables are promoted to FP32 BEFORE copying, frozen
dtypes retained; X-VLA is FP32. Do NOT substitute generic from_pretrained loading.
No original project, simulator, cache, credentials or remote download is needed
at inference, once the recorded software dependencies are available.

## Provenance, legal status and scope

release_manifest.json records source checkpoint hashes, pinned base/asset
repositories, explicit rewrites and actual recipe. inventory.json hashes every
release file except itself. Absolute source paths are represented in the rewrite
map by source field plus hash, not leaked as host paths. Original configs are
not silently retuned. No optimizer, training state, reload pickle, cache internals,
dataset or simulator assets are included.

Local manifests do not record base/tokenizer/VLM license metadata. License is
**unknown from locally available evidence**; no license is granted or invented
here. Publisher must resolve upstream/model/asset redistribution terms before
sharing. Private visibility does not itself resolve licensing.

This is a portable inference release, NOT a full environment/simulation
reproduction package. CPU packaging equivalence uses one recorded raw observation
and one sampling seed; it is not a repeat rollout, GPU-equivalence proof, broad
input validation, or full task reproduction. Final card and remote verification
remain the publisher's responsibility. No upload is performed by this code.
'''


def stage_model(repo, packages, model):
    source = repo / 'outputs/train' / f'{model}_fixed_layout_20260918/checkpoints/005000'
    target = packages / model
    target.mkdir(exist_ok=False)
    config_names = ('config.json', 'policy_preprocessor.json', 'policy_postprocessor.json')
    original = [json.loads((source / name).read_text()) for name in config_names]
    docs, rewrites = portable_documents(*original)
    tensor_names = ['model.safetensors']
    for processor in original[1:]:
        for step in processor['steps']:
            if 'state_file' in step:
                name = step['state_file']
                if Path(name).name != name or not re.fullmatch(r'policy_(?:pre|post)processor_step_\d+_(?:un)?normalizer_processor\.safetensors', name):
                    raise ValueError('Unapproved processor state file')
                tensor_names.append(name)
    sources = {}
    for name in (*config_names, *tensor_names):
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Source must be a regular file')
        sources[name] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    for name, document in docs.items():
        write_json(target / name, document)
    comparisons = {}
    for name in tensor_names:
        shutil.copyfile(source / name, target / name)
        if sha256(target / name) != sources[name]['sha256']:
            raise ValueError('Copy hash mismatch')
        comparisons[name] = compare_safetensors(source / name, target / name)
    asset_source = (repo / '.cache/benchmark_models/vlm' if model == 'smolvla'
                    else repo / '.cache/xvla_trial/bundle/tokenizer')
    asset_target = target / ('assets/vlm' if model == 'smolvla' else 'assets/tokenizer')
    names = SMOL_ASSETS if model == 'smolvla' else XVLA_ASSETS
    copy_assets(asset_source, asset_target, names)
    # Model/tokenizer vocab JSON can contain slash tokens; inspect metadata only.
    for name in ('config.json', 'tokenizer_config.json', 'preprocessor_config.json', 'processor_config.json'):
        if (asset_target / name).exists():
            audit_json(json.loads((asset_target / name).read_text()))
    asset_hashes = {name: dict(bytes=(asset_source / name).stat().st_size,
                             sha256=sha256(asset_source / name)) for name in names}
    if asset_hashes != inventory(asset_target):
        raise ValueError('Bundled assets differ from originals')
    result = json.loads((source.parents[1] / 'result.json').read_text())
    fields = ('seed', 'microbatch', 'accumulation', 'effective_batch', 'chunk_size', 'n_action_steps',
              'precision', 'optimizer', 'schedule', 'total_parameters', 'trainable_parameters')
    recipe = {key: result[key] for key in fields if key in result}
    if result.get('completed_updates') != 5000 or recipe['schedule']['warmup'] != 500:
        raise ValueError('Unexpected actual training recipe')
    manifest = dict(schema_version=1, model=model, checkpoint_update=5000,
                    source_checkpoint=f'outputs/train/{model}_fixed_layout_20260918/checkpoints/005000',
                    source_files=sources, rewrites=rewrites, asset_files=asset_hashes,
                    provenance=provenance(repo, model), actual_training_recipe=recipe,
                    comparison= comparisons, runtime='runtime.json',
                    source_training_report_sha256=sha256(source.parents[1] / 'result.json'))
    audit_json(manifest)
    write_json(target / 'release_manifest.json', manifest)
    shutil.copyfile(Path(__file__).with_name('load_baseline.py'), target / 'load_baseline.py')
    runtime = dict(python=__import__('platform').python_version(), tested_image=IMAGE,
                   packages={name: importlib.metadata.version(name) for name in
                             ('torch', 'torchvision', 'lerobot', 'transformers', 'safetensors',
                              'tokenizers', 'huggingface-hub', 'numpy', 'Pillow')},
                   execution='CPU only; no GPU equivalence asserted', dependencies_vendored=False)
    write_json(target / 'runtime.json', runtime)
    (target / 'README.md').write_text(card_template(model, recipe))
    write_json(target / 'inventory.json', inventory(target))
    audit_release(target)
    for name, info in sources.items():
        if sha256(source / name) != info['sha256']:
            raise ValueError('Source changed during staging')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    packages = claim_packages(repo / RELEASE_RELATIVE / 'packages')
    for model in MODELS:
        stage_model(repo, packages, model)
        print('STAGED', model, flush=True)
    print('Staging complete; independent CPU prediction verification still required.', flush=True)


if __name__ == '__main__':
    main()