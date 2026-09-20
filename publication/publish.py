"""Explicit, fail-closed remote publication. No remote work at import time."""
import argparse
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
from urllib.parse import parse_qs, urlsplit
import uuid

from publication.package_baselines import RELEASE_RELATIVE, audit_release, inventory, sha256, write_json

HF_ENDPOINT = 'https://huggingface.co'
WB_ENDPOINT = 'https://api.wandb.ai'
# Only the native presigned blob destination observed in the earlier HF attempt;
# this is not an alternate API endpoint or a proxy bypass. Other storage fails closed.
HF_SIGNED_STORAGE = 'hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com'
CREDENTIAL_MOUNT = Path('/run/secrets/publishing.env')
REPOS = {m: f'VivekSai07/{m}-franka-stacking-fixed-layout-v1' for m in ('smolvla', 'xvla')}
TIMEOUT = (10, 60)


class ProxyBlocked(RuntimeError):
    """Deliberately not requests.ConnectionError: Hub must not retry a 407."""
    def __init__(self):
        super().__init__('HTTP407')


def safe_error(exc):
    """Never return exception messages, requests, URLs, headers or payloads."""
    status = None
    current = exc
    for _ in range(5):
        code = getattr(getattr(current, 'response', None), 'status_code', None)
        if isinstance(code, int) and 100 <= code <= 599:
            status = code
            break
        # CONNECT proxy failures often have no Response. Extract only this code.
        if isinstance(current, ProxyBlocked) or re.search(r'\b407\b', str(current)):
            status = 407
            break
        current = current.__cause__ or current.__context__
        if current is None:
            break
    name = type(exc).__name__
    return {'type': name if re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', name) else 'Error',
            'http_status': status}


@contextlib.contextmanager
def quiet():
    """Discard Python and inherited subprocess output, not capture it in reports."""
    previous = logging.root.manager.disable
    saved = []
    with open(os.devnull, 'w') as sink:
        try:
            for fd in (1, 2):
                saved.append((fd, os.dup(fd)))
                os.dup2(sink.fileno(), fd)
            logging.disable(sys.maxsize)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
        finally:
            for fd, backup in saved:
                os.dup2(backup, fd)
                os.close(backup)
            logging.disable(previous)


@contextlib.contextmanager
def deadline(seconds):
    def expired(*_):
        raise TimeoutError('Operation deadline')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@contextlib.contextmanager
def private_runtime():
    """CLI-only isolation, before importing SDKs. No host auth/config caches."""
    original_env = dict(os.environ)
    original_cwd = Path.cwd()
    old_umask = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix='baseline-publisher-', dir='/tmp') as directory:
            root = Path(directory)
            keep = ('PATH', 'LANG', 'LC_ALL', 'SSL_CERT_FILE', 'SSL_CERT_DIR',
                    'REQUESTS_CA_BUNDLE', 'HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY',
                    'https_proxy', 'http_proxy', 'all_proxy', 'NO_PROXY', 'no_proxy')
            os.environ.clear()
            os.environ.update({k: original_env[k] for k in keep if k in original_env})
            # Proxy settings are preserved verbatim, never bypassed or printed.
            os.environ.update({
                'HOME': str(root / 'home'), 'HF_HOME': str(root / 'hf'),
                'XDG_CACHE_HOME': str(root / 'cache'), 'XDG_CONFIG_HOME': str(root / 'config'),
                'WANDB_DIR': str(root / 'wandb'), 'WANDB_CACHE_DIR': str(root / 'wb-cache'),
                'WANDB_CONFIG_DIR': str(root / 'wb-config'), 'WANDB_DATA_DIR': str(root / 'wb-data'),
                'WANDB_BASE_URL': WB_ENDPOINT, 'HF_ENDPOINT': HF_ENDPOINT,
                'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                'HF_HUB_DISABLE_XET': '1', 'HF_HUB_ENABLE_HF_TRANSFER': '0',
                'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'DO_NOT_TRACK': '1',
                'WANDB_SILENT': 'true', 'WANDB_CONSOLE': 'off',
                'WANDB_DISABLE_CODE': 'true', 'WANDB_DISABLE_GIT': 'true',
                'WANDB_SAVE_CODE': 'false', 'WANDB_ERROR_REPORTING': 'false',
                'WANDB_ANONYMOUS': 'never', 'CUDA_VISIBLE_DEVICES': '',
                'NVIDIA_VISIBLE_DEVICES': 'void', 'PYTHONDONTWRITEBYTECODE': '1',
            })
            for key in ('HOME', 'HF_HOME', 'WANDB_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR'):
                Path(os.environ[key]).mkdir(mode=0o700, parents=True)
            os.chdir(root)
            try:
                yield
            finally:
                os.chdir(original_cwd)
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        os.umask(old_umask)


def credentials(services):
    """Only this reviewed process opens the fixed, main-mounted secret file."""
    from dotenv import dotenv_values
    if CREDENTIAL_MOUNT.is_symlink() or not CREDENTIAL_MOUNT.is_file():
        raise ValueError('Credential mount unavailable')
    parsed = dotenv_values(CREDENTIAL_MOUNT, interpolate=False)
    names = ({'HF_TOKEN'} if 'hf' in services else set()) | (
        {'WANDB_API_KEY', 'WANDB_ENTITY'} if 'wandb' in services else set())
    selected = {k: parsed[k] for k in names if parsed.get(k)}
    parsed.clear()
    for service, key in (('hf', 'HF_TOKEN'), ('wandb', 'WANDB_API_KEY')):
        if service in services and (not isinstance(selected.get(key), str) or not selected[key].strip()):
            raise ValueError('Required credential unavailable')
    return selected


def bounded_session(service):
    import requests

    class OfficialSession(requests.Session):
        def request(self, method, url, **kwargs):
            kwargs['timeout'] = TIMEOUT
            return super().request(method, url, **kwargs)

        def send(self, request, **kwargs):
            parsed = urlsplit(request.url)
            host = parsed.hostname or ''
            allowed = host == ('huggingface.co' if service == 'hf' else 'api.wandb.ai')
            if service == 'hf':
                allowed = allowed or host.endswith('.huggingface.co') or host.endswith('.hf.co')
                allowed = allowed or (host == HF_SIGNED_STORAGE and
                                      bool(parse_qs(parsed.query).get('X-Amz-Signature')))
            if not allowed or parsed.scheme != 'https' or parsed.port not in (None, 443):
                raise ValueError('Unapproved service destination')
            if parsed.username or parsed.password:
                raise ValueError('Unapproved URL credentials')
            if service == 'hf' and host != 'huggingface.co':
                request.headers.pop('Authorization', None)
            kwargs['timeout'] = TIMEOUT
            try:
                response = super().send(request, **kwargs)
            except requests.exceptions.ProxyError as exc:
                if safe_error(exc)['http_status'] == 407:
                    raise ProxyBlocked() from None
                raise
            if getattr(response, 'status_code', None) == 407:
                response.close()
                raise ProxyBlocked()
            return response

    return OfficialSession()


def validate_release_path(repo, model, root):
    if model not in REPOS:
        raise ValueError('Unapproved model')
    root = Path(root).absolute()
    expected = Path(repo).absolute() / RELEASE_RELATIVE / 'packages' / model
    if (root != expected or root.resolve() != expected
            or any(p.is_symlink() for p in (root, *root.parents))):
        raise ValueError('Only the exact reviewed release directory is allowed')
    return root


def reviewed_files(repo, model):
    root = validate_release_path(repo, model, Path(repo) / RELEASE_RELATIVE / 'packages' / model)
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Release symlink rejected before reading metadata')
    proof = json.loads((Path(repo) / RELEASE_RELATIVE / 'packaging-verification.json').read_text())['results'][model]
    if (proof['status'] != 'PASS' or proof['prediction_max_absolute_error'] != 0
            or proof['storage']['all_stored_tensors_bit_exact'] is not True):
        raise ValueError('Offline equivalence gate missing')
    audit_release(root)
    files = inventory(root)
    if files['model.safetensors']['sha256'] != proof['weights_sha256']:
        raise ValueError('Verified weight hash mismatch')
    return files


def operation_add(name, path):
    from huggingface_hub import CommitOperationAdd
    return CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))


def verify_remote(api, repo_id, revision, expected, download):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Unverified commit identifier')
    info = api.model_info(repo_id, revision=revision, files_metadata=True, timeout=60)
    if info.private is not True or info.sha != revision:
        raise ValueError('Remote privacy or revision mismatch')
    files = {f.rfilename: f for f in info.siblings}
    # The Hub initializes .gitattributes itself; it is not a local upload input.
    if set(files) - set(expected) not in (set(), {'.gitattributes'}) or not set(expected) <= set(files):
        raise ValueError('Remote file inventory mismatch')
    verified = {}
    for name, remote in files.items():
        local = expected.get(name)
        if local and remote.size != local['bytes']:
            raise ValueError('Remote size mismatch')
        if remote.lfs is not None:
            lfs = remote.lfs
            digest = lfs.get('sha256') if isinstance(lfs, dict) else lfs.sha256
            size = lfs.get('size') if isinstance(lfs, dict) else lfs.size
            if not local or digest != local['sha256'] or size != local['bytes']:
                raise ValueError('Remote LFS mismatch')
            verified[name] = dict(bytes=size, sha256=digest, method='LFS SHA256 and size')
        else:
            if not isinstance(remote.size, int) or not 0 <= remote.size <= 32 * 1024 * 1024:
                raise ValueError('Unexpected non-LFS blob size')
            data = download(name, revision)
            digest = hashlib.sha256(data).hexdigest()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if len(data) != remote.size or blob != remote.blob_id or (local and digest != local['sha256']):
                raise ValueError('Remote Git blob mismatch')
            if not local and (name != '.gitattributes' or len(data) > 65536):
                raise ValueError('Unexpected server metadata')
            verified[name] = dict(bytes=len(data), sha256=digest, git_blob_sha1=blob,
                                  method='download SHA256 and Git blob SHA1', server_generated=local is None)
    head = api.model_info(repo_id, timeout=60)
    if head.private is not True or head.sha != revision:
        raise ValueError('Remote HEAD or privacy changed')
    return verified


def publish_model(repo, model, api, download, report, save=lambda: None):
    report.update(model=model, status='preflight', repo_id=REPOS[model], commit_verified=False)
    try:
        with quiet():
            report['phase'] = 'release_audit'
            root = validate_release_path(repo, model, Path(repo) / RELEASE_RELATIVE / 'packages' / model)
            files = reviewed_files(repo, model)
            report['phase'] = 'identity_check'
            if api.whoami().get('name') != 'VivekSai07':
                raise ValueError('Unexpected HF identity')
            report['phase'] = 'repository_absence_check'
            try:
                api.model_info(REPOS[model], timeout=60)
            except Exception as exc:
                if safe_error(exc)['http_status'] != 404:
                    raise
            else:
                raise ValueError('Existing repository refused, including empty repositories')
            report.update(phase='private_repository_creation', status='repo_creation_unconfirmed')
            save()
            api.create_repo(REPOS[model], private=True, repo_type='model', exist_ok=False)
            report['status'] = 'repo_created_no_modelcommit'
            save()
            report['phase'] = 'new_repository_privacy_check'
            initial = api.model_info(REPOS[model], files_metadata=True, timeout=60)
            if initial.private is not True:
                raise ValueError('New repository is not private')
            if {f.rfilename for f in initial.siblings} - {'.gitattributes'}:
                raise ValueError('New repository unexpectedly contains model files')
            if not re.fullmatch(r'[0-9a-f]{40}', initial.sha):
                raise ValueError('Initial commit not verifiable')
            operations = [operation_add(name, root / name) for name in sorted(files)]
            # Native API only; no folder upload, shell git, or parent-directory traversal.
            report['phase'] = 'lfs_preupload'
            save()
            api.preupload_lfs_files(REPOS[model], additions=operations, repo_type='model',
                                   revision='main', free_memory=False, num_threads=1)
            # Recheck reviewed bytes before committing; main mounts releases read-only.
            if inventory(root) != files:
                raise ValueError('Release changed during upload')
            report.update(status='repo_created_modelcommit_unconfirmed', phase='model_commit')
            save()
            commit = api.create_commit(REPOS[model], operations=operations, repo_type='model',
                                       commit_message='Publish validated fixed-layout baseline',
                                       parent_commit=initial.sha, num_threads=1)
            if not re.fullmatch(r'[0-9a-f]{40}', commit.oid):
                raise ValueError('Invalid commit response')
            report.update(status='modelcommit_unverified', phase='remote_file_verification', commit_sha=commit.oid)
            save()
            report['files'] = verify_remote(api, REPOS[model], commit.oid, files, download)
            report.update(status='verified', commit_verified=True, private=True)
    except Exception as exc:
        report['error'] = safe_error(exc)
        if report['status'] == 'preflight':
            report['status'] = 'blocked'
    finally:
        save()
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for flag in ('hf', 'wandb', 'registry'):
        result.add_argument('--' + flag, action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    services = [k for k in ('hf', 'wandb') if getattr(args, k)]
    if not services and not args.registry:
        print('Select --registry, --hf, or --wandb.')
        return 2
    repo = Path(__file__).resolve().parents[1]
    output = repo / RELEASE_RELATIVE
    attempt = uuid.uuid4().hex
    report_path = output / ('publisher-attempt-' + attempt + '.json')
    report = {'schema_version': 1, 'attempt_id': attempt, 'remote_requested': services,
              'status': 'started', 'hf': [], 'wandb': []}
    claimed = False

    def save():
        # Only the exclusively claimed report for this attempt is ever updated.
        if not claimed or report_path.is_symlink():
            raise ValueError('Attempt report not exclusively owned')
        temporary = output / ('.publisher-report-' + uuid.uuid4().hex + '.tmp')
        with temporary.open('x') as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        temporary.replace(report_path)

    try:
        if any(p.is_symlink() for p in (output, *output.parents)):
            raise ValueError('Output cannot contain symlinks')
        write_json(report_path, report)
        claimed = True
        with private_runtime(), quiet():
            from publication.registry import build_registry
            rows = build_registry(repo)
            if args.registry:
                path = output / ('baseline-registry-' + attempt + '.json')
                write_json(path, {'schema_version': 1, 'baselines': rows})
                report['registry'] = {'file': path.name, 'sha256': sha256(path), 'rows': len(rows)}
                save()
            if services:
                keys = credentials(services)
                try:
                    if args.hf:
                        from huggingface_hub import HfApi, configure_http_backend, hf_hub_download
                        configure_http_backend(backend_factory=lambda: bounded_session('hf'))
                        api = HfApi(endpoint=HF_ENDPOINT, token=keys['HF_TOKEN'])
                        for model in REPOS:
                            item = {}
                            report['hf'].append(item)

                            def download(name, revision):
                                return Path(hf_hub_download(REPOS[model], name, revision=revision,
                                                          token=keys['HF_TOKEN'], endpoint=HF_ENDPOINT)).read_bytes()

                            with deadline(1800):
                                publish_model(repo, model, api, download, item, save)
                            if item['status'] != 'verified':
                                break
                    if args.wandb:
                        import wandb
                        from publication.tracking import WandbControl, import_history
                        control = WandbControl(keys['WANDB_API_KEY'])
                        try:
                            for row in rows:
                                item = {}
                                report['wandb'].append(item)
                                with deadline(900):
                                    import_history(row, control, wandb, keys['WANDB_API_KEY'], keys.get('WANDB_ENTITY'), item, save)
                                if item['status'] != 'imported':
                                    break
                        finally:
                            control.session.auth = None
                            control.session.close()
                            # Stop SDK services before restoring HOME/environment or output.
                            with deadline(30):
                                wandb.teardown()
                finally:
                    keys.clear()
        ok = (not args.hf or len(report['hf']) == 2 and all(r['status'] == 'verified' for r in report['hf'])) and (
            not args.wandb or len(report['wandb']) == 2 and all(r['status'] == 'imported' for r in report['wandb']))
        report['status'] = 'complete' if ok else 'blocked_or_partial'
    except Exception as exc:
        report.update(status='blocked_or_partial', error=safe_error(exc))
    finally:
        if claimed:
            try:
                save()
            except Exception as exc:
                report.update(status='blocked_or_partial', error=safe_error(exc))
    print(json.dumps({'status': report['status'], 'report': report_path.name}))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())