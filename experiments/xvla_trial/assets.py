"""Public pinned HF assets only. Never logs network exception text or URLs."""
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path

from . import contract as c

ASSETS = {
    'model': dict(repo='lerobot/xvla-base', revision='cdb7964e4fe842935d671bfab5a5ebe00a96648c',
                  files={'config.json': 5463, 'policy_preprocessor.json': 2078,
                         'policy_postprocessor.json': 575, 'model.safetensors': 3519073692}),
    'tokenizer': dict(repo='facebook/bart-large', revision='cb48c1365bd826bd521f650dc2e0940aee54720c',
                      files={'config.json': 1628, 'tokenizer_config.json': 26,
                             'tokenizer.json': 1355863, 'vocab.json': 898822, 'merges.txt': 456318}),
}
MODEL_SHA = 'f05bc0fab1c9523d7f5d6b41a651313641ca2227a88822249845da8e20e036c9'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def claim_directory(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink output/cache forbidden')
    path.mkdir(parents=True, exist_ok=False)


def verify_file(path, size, digest=None):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size != size:
        raise ValueError('Asset size/path mismatch')
    result = dict(bytes=size, sha256=sha256(path))
    if digest and result['sha256'] != digest:
        raise ValueError('Asset hash mismatch')
    return result


def safe_error(exc):
    return dict(error_type=type(exc).__name__,
                http_status=getattr(getattr(exc, 'response', None), 'status_code', None))


def download(root):
    root = Path(root)
    claim_directory(root)
    os.environ.update(HF_HOME=str(root/'hf_home'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                      HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_XET='1',
                      HF_HUB_DISABLE_PROGRESS_BARS='1')
    report = dict(status='preparing', assets=ASSETS, files={}, metadata={})
    write_json(root/'manifest.json', report)
    try:
        logging.disable(logging.CRITICAL)
        with open(os.devnull, 'w') as quiet, contextlib.redirect_stderr(quiet), contextlib.redirect_stdout(quiet):
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi(token=False)
            for group, asset in ASSETS.items():
                info = api.model_info(asset['repo'], revision=asset['revision'], files_metadata=True)
                if info.sha != asset['revision']:
                    raise ValueError('Revision mismatch')
                siblings = {s.rfilename: s for s in info.siblings}
                for name, size in asset['files'].items():
                    sibling = siblings[name]
                    if sibling.size != size:
                        raise ValueError('Metadata size mismatch')
                    report['metadata'][f'{group}/{name}'] = dict(
                        bytes=size, blob_id=sibling.blob_id,
                        sha256=getattr(sibling.lfs, 'sha256', None))
            write_json(root/'manifest.json', report)
            # Configs/tokenizer first. One standard full-weight download last; no fallback routes.
            names = sorted(report['metadata'], key=lambda n: n.endswith('.safetensors'))
            for relative in names:
                group, name = relative.split('/', 1)
                asset, meta = ASSETS[group], report['metadata'][relative]
                path = Path(hf_hub_download(repo_id=asset['repo'], filename=name,
                            revision=asset['revision'], token=False, local_dir=root/group))
                if path.absolute() != (root/relative).absolute():
                    raise ValueError('Download escaped exclusive cache')
                record = verify_file(path, meta['bytes'], MODEL_SHA if name == 'model.safetensors' else meta['sha256'])
                if not meta['sha256']:
                    blob = path.read_bytes()
                    if hashlib.sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest() != meta['blob_id']:
                        raise ValueError('Git blob mismatch')
                report['files'][relative] = record
                write_json(root/'manifest.json', report)
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='blocked', **safe_error(exc))
    write_json(root/'manifest.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'metadata'}), flush=True)
    return report['status'] == 'complete'


def require_assets(root=c.CACHE):
    root = Path(root)
    report = json.loads((root/'manifest.json').read_text())
    expected = {f'{g}/{n}' for g, a in ASSETS.items() for n in a['files']}
    if report['status'] != 'complete' or report['assets'] != ASSETS or set(report['files']) != expected:
        raise ValueError('Complete pinned asset manifest required')
    for relative, record in report['files'].items():
        group, name = relative.split('/', 1)
        if record['bytes'] != ASSETS[group]['files'][name]:
            raise ValueError('Pinned asset size mismatch')
        verify_file(root/relative, record['bytes'], MODEL_SHA if name == 'model.safetensors' else record['sha256'])
    return report


if __name__ == '__main__':
    try:
        raise SystemExit(0 if download(c.CACHE) else 1)
    except Exception as exc:
        print(json.dumps(dict(status='blocked', **safe_error(exc))))
        raise SystemExit(1) from None