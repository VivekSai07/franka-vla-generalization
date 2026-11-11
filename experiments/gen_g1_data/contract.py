"""G1 binding: images, plan digest and `provenance()` over everything the collection depends on.

Pure: imports no robosuite, MuJoCo, torch or LeRobot. Recorded when collection starts; any drift before the seal aborts the run.
Bound: this package (every `.py` including tests, and its README), the frozen scene and expert it calls (`props_scene`,
`robosuite_scene`, `gen_scene` with its texture manifest, P2's package), S0's `signals.py`, the shared collection helpers and `src/`.
NOT bound: the live design documents (the G1 spec and plan). A collection that bound them could not take the docs pass the programme
requires after the result without breaking its own seal (FAILURES #52); the spec's mapping note is dated before the first episode,
and what the plan digest fixes is the plan itself.
"""
import hashlib
from pathlib import Path

from experiments.gen_g1_data import plan as p

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e"
ADMISSION_IMAGE = "sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc"
OUTPUT = "outputs/gen_g1_data"
MIN_FREE_BYTES = 60 * 1024**3
SHARDS = 16
PACKAGE = "experiments/gen_g1_data"

SHARED_FILES = (
    "data_collection/benchmark.py",
    "data_collection/collect_libero_dataset.py",
    "data_collection/pilot_integrity.py",
    "data_collection/pilot_plan.py",
    "experiments/paired_demonstrations/expert_v1.py",
    "experiments/paired_demonstrations/media.py",
    "experiments/paired_demonstrations/plan.py",
    "experiments/pose_snapshot_v2/expert.py",
    "experiments/pose_snapshot_v2/planner.py",
    "experiments/vla_inputs_signals/signals.py",
    "experiments/gen_scene/assets_manifest.json",
)
SOURCE_SUFFIXES = (".py", ".xml", ".urdf", ".obj", ".stl", ".png", ".jpg")
PACKAGES = (PACKAGE, "experiments/props_p2_data", "experiments/gen_scene")


def _file_hash(path):
    if path.is_symlink():
        raise ValueError(f"Symlink source forbidden: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_paths(root=ROOT):
    root = Path(root)
    paths = []
    for package in PACKAGES:
        paths += sorted(q for q in (root / package).rglob("*.py") if "__pycache__" not in q.parts)
    paths += [root / PACKAGE / "README.md", root / "experiments/props_p2_data/README.md"]
    paths += sorted((root / "experiments/props_scene").glob("*.py"))
    paths += sorted((root / "experiments/robosuite_scene").glob("*.py"))
    paths += [root / name for name in SHARED_FILES]
    paths += sorted(q for q in (root / "src").rglob("*")
                    if q.is_file() and "__pycache__" not in q.parts and q.suffix in SOURCE_SUFFIXES)
    return paths


def provenance(root=ROOT):
    root = Path(root)
    return dict(sorted((str(q.relative_to(root)), _file_hash(q)) for q in set(source_paths(root))))


def check_plan(plan):
    """Raises ValueError unless `plan` is the plan of record (the digest is pinned in `plan.py`)."""
    p.validate_plan(plan)
