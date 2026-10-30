"""S0 binding: the sealed P2 root this sidecar is built from, and `provenance()` over everything it depends on.

Pure: imports no simulator. The sidecar is bound to the P2 seal fingerprint, to every file the sealed P2
`provenance.json` lists (so the frozen scene, expert, controllers and `src/` are covered by what P2 already
proved, re-hashed here), and to this package's own `.py` files and README. After the seal: fork, do not edit.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "experiments/vla_inputs_signals"
P2_ROOT = ROOT / "outputs/props_p2_data/props_p2_20261013"
P2_SEAL = "b1d642b3edeebe15d462709ebeb8b92afa6d7eabcca3be5514e5deeecaf4d898"
SIDECAR_ROOT = ROOT / "outputs/vla_inputs_signals"
IMAGE = "sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e"
MUJOCO_VERSION = "3.4.0"
EPISODES = 540
FRAMES = 452
SHARDS = 6
# Imported by frozen P2 code (expert_v1.py) but missing from P2's own provenance: bound here.
EXTRA_FILES = (
    "data_collection/collect_libero_dataset.py",
    "data_collection/pilot_integrity.py",
    "data_collection/pilot_plan.py",
    "experiments/paired_demonstrations/plan.py",
)


def _file_hash(path):
    if path.is_symlink():
        raise ValueError(f"Symlink source forbidden: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_paths(root=ROOT):
    """This package's `.py` files (tests included) and its README, relative to `root`."""
    package = Path(root) / PACKAGE
    paths = [p for p in sorted(package.rglob("*.py")) if "__pycache__" not in p.parts]
    return [p.relative_to(root).as_posix() for p in paths] + [f"{PACKAGE}/README.md"]


SOURCE_FILES = tuple(source_paths())


def sealed_p2_provenance():
    """What P2's seal says it was built from: {relative path: sha256}."""
    return json.loads((P2_ROOT / "provenance.json").read_text())


def provenance(root=ROOT):
    """{relative path: sha256} over this package, every file in P2's sealed provenance and `EXTRA_FILES`."""
    root = Path(root)
    paths = set(source_paths(root)) | set(sealed_p2_provenance()) | set(EXTRA_FILES)
    return {rel: _file_hash(root / rel) for rel in sorted(paths)}


def check_p2_frozen(root=ROOT):
    """Raises unless every file P2 sealed is byte for byte what it was, and the P2 seal is the one this binds to."""
    seal = json.loads((P2_ROOT / "seal.json").read_text())
    if seal["fingerprint"] != P2_SEAL:
        raise ValueError(f"P2 seal fingerprint {seal['fingerprint']} is not the bound {P2_SEAL}")
    sealed = sealed_p2_provenance()
    changed = sorted(rel for rel, digest in sealed.items() if _file_hash(Path(root) / rel) != digest)
    if changed:
        raise ValueError(f"files sealed by P2 changed: {changed[:5]}")
