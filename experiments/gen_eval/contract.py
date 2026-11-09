"""G0 shared-core binding. Pure; admission image.

The shared core is what every arm's measurement must agree on: this package's modules (not its tests or README), the
randomization engine `gen_scene` (modules and the texture manifest), the M0 modules and P2 evaluation modules it imports unchanged,
and the whole S0 signal source (`vla_inputs_signals.contract.provenance()`, which covers P2's frozen scene, expert, controllers
and `src/`). `core_provenance()` hashes all of it. The texture files are bound separately, by `prepare` (every file's sha256, from
the manifest), and an arm adapter, its weights, processors and runtime by the arm manifest. A core hash that differs between two
prepared arms rejects the campaign: a core change is a new protocol and a new measurement of every arm.

Forked from `experiments/vla_inputs_eval/contract.py` (frozen, unchanged).
"""
import hashlib
import json
from pathlib import Path

from experiments.gen_scene import pools
from experiments.vla_inputs_signals import contract as s0

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "experiments/gen_eval"
ENGINE = "experiments/gen_scene"
SIDECAR = ROOT / "outputs/vla_inputs_signals/p2_signals_20261030"
# Repository files the core imports that are in neither this package, the engine nor S0's provenance.
EXTRA_CORE = (
    "experiments/gen_scene/assets_manifest.json",
    "experiments/vla_inputs_eval/adapters.py",
    "experiments/vla_inputs_eval/contract.py",      # the identity gate verifies M0's reference through M0's own evaluate module
    "experiments/vla_inputs_eval/evaluate.py",
    "experiments/vla_inputs_eval/feasibility.py",
    "experiments/vla_inputs_eval/plan.py",
    "experiments/vla_inputs_eval/protocol.py",
    "experiments/vla_inputs_eval/stats.py",
    "experiments/vla_inputs_pilot_plan/plan.py",
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
    """This package's and the engine's non-test modules, relative to `root`."""
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for package in (PACKAGE, ENGINE) for p in (root / package).glob("*.py"))


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


def texture_files(root=ROOT):
    """{name: {sha256, bytes}} of the texture folder, verified against the committed manifest (raises on any difference)."""
    root = Path(root)
    manifest = pools.load_manifest(root / ENGINE / "assets_manifest.json")
    pools.verify_files(manifest, root / ".cache/gen_textures")
    return {t["name"]: dict(sha256=t["sha256"], bytes=t["bytes"]) for t in manifest["textures"]}
