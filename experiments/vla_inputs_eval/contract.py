"""M0 shared-core binding. Pure; admission image.

The shared core is what every arm's comparison must agree on: this package's modules (not its tests or README), the P2
evaluation modules it imports unchanged, and the whole S0 signal source (`vla_inputs_signals.contract.provenance()`, which
covers P2's frozen scene, expert, controllers and `src/`). `core_provenance()` hashes all of it. An arm adapter, its weights,
processors and runtime are bound separately by `prepare` (arm manifest) and may differ between arms; a core hash that
differs from A0's rejects the comparison. A core correction is a new protocol and a new control.
"""
import hashlib
import json
from pathlib import Path

from experiments.vla_inputs_signals import contract as s0

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "experiments/vla_inputs_eval"
SIDECAR = ROOT / "outputs/vla_inputs_signals/p2_signals_20261030"
# Repository modules the core imports that are in neither this package nor S0's provenance.
EXTRA_CORE = (
    "experiments/xvla_props_p2_eval/analysis.py",
    "experiments/xvla_props_p2_retest/plan.py",
    "validation/rollout_diagnostics.py",
)


def file_hash(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Symlink source forbidden: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def package_files(root=ROOT):
    """This package's non-test modules, relative to `root`."""
    package = Path(root) / PACKAGE
    return sorted(p.relative_to(root).as_posix() for p in package.glob("*.py"))


def core_provenance(root=ROOT):
    """{relative path: sha256} over the shared core."""
    root = Path(root)
    paths = set(package_files(root)) | set(EXTRA_CORE) | set(s0.provenance(root))
    return {rel: file_hash(root / rel) for rel in sorted(paths)}


def core_sha256(root=ROOT):
    return digest(core_provenance(root))


def check_signal_source(root=ROOT, sidecar=SIDECAR):
    """The S0 signal source is the sealed one: the sidecar's recorded provenance equals the current S0 provenance.
    Returns the sidecar's seal fingerprint."""
    sidecar = Path(sidecar)
    sealed = json.loads((sidecar / "seal.json").read_text())
    if json.loads((sidecar / "provenance.json").read_text()) != s0.provenance(root):
        raise ValueError("the S0 signal source differs from the one the sidecar was sealed with")
    if sealed["p2_seal"] != s0.P2_SEAL:
        raise ValueError("the sidecar is not bound to the P2 seal")
    return sealed["fingerprint"]
