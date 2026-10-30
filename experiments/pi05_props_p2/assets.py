"""Pinned pi0.5 assets: the `pi05_base` weights and the PaliGemma tokenizer, hashed file by file.

Both repositories are downloaded on the host (anonymously; the hub's own preprocessor files are pinned but never used,
processors are built from the local config and P2 statistics). Nothing here touches the network.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = ROOT/'outputs/research/pi05_assets'
WEIGHTS_REPO = 'lerobot/pi05_base'
# The latest revision's preprocessor has a step LeRobot 0.4.4 lacks; this one predates 0.4.4 and has the same weights.
WEIGHTS_REVISION = '9e55186ad36e66b95cda57bc47818d9e6237ae30'
TOKENIZER_REPO = 'google/paligemma-3b-pt-224'
TOKENIZER_REVISION = '35e4f46485b4d07967e7e9935bc3786aad50687c'
WEIGHTS_DIR = ASSETS_ROOT/'pi05_base'
TOKENIZER_DIR = ASSETS_ROOT/'paligemma-3b-pt-224'

# relative path -> (bytes, sha256)
FILES = {
    'pi05_base/.gitattributes': (1519, '11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361'),
    'pi05_base/README.md': (5058, '00eb52c2acb6c70d9acb1a812e5fd800268fb8c9bf31debf9e64447943444978'),
    'pi05_base/config.json': (1896, '367869712a2847c27e95c431ecb03f17bec4eee01a63995e2fb2d91940752b53'),
    'pi05_base/model.safetensors': (14_467_165_872,
                                    '0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f'),
    'pi05_base/policy_postprocessor.json': (450, '938c90a56a3ab7298efe4ee16c0c7f0f3027e8defab6a9b26db52511f257e596'),
    'pi05_base/policy_preprocessor.json': (1035, 'f43c5d8fb768edd95d580eaadb1ab6a2d1d5f7c78247aa2d871f0930654f6587'),
    'paligemma-3b-pt-224/added_tokens.json': (24, '7d0bad90030d638a4bf89a82e91206e25f1fbe0012f14bca21666a64b643bc49'),
    'paligemma-3b-pt-224/special_tokens_map.json': (
        607, '5ef37093ae4236587b6e8266acb815b46e2db8ce656c66552bfa574d32880405'),
    'paligemma-3b-pt-224/tokenizer.json': (
        17_549_604, 'ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563'),
    'paligemma-3b-pt-224/tokenizer.model': (
        4_264_023, '8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6'),
    'paligemma-3b-pt-224/tokenizer_config.json': (
        39_968, '3259402b1d1802e02417d7bff75a889ec61d359d15be6050a957b307c48edbbe'),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def require_assets(root=ASSETS_ROOT, files=FILES):
    """Every pinned file present with its size and hash, and nothing else (the hub client's `.cache` is ignored)."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f'Asset root {root} is not a real directory')
    present = set()
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if '.cache' in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f'Asset symlink forbidden: {relative}')
        if path.is_file():
            present.add(relative.as_posix())
    if present - set(files):
        raise ValueError(f'Asset tree has unlisted files: {sorted(present - set(files))}')
    if set(files) - present:
        raise ValueError(f'Asset tree is missing files: {sorted(set(files) - present)}')
    found = {}
    for name, (size, expected) in sorted(files.items()):
        if (root/name).stat().st_size != size:
            raise ValueError(f'Asset size mismatch: {name}')
        found[name] = sha256(root/name)
        if found[name] != expected:
            raise ValueError(f'Asset hash mismatch: {name}')
    return found
