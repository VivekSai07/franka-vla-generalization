"""B3 binding: image, spec, plan, and `provenance()` over everything the collection depends on.

Pure: imports no robosuite, MuJoCo, torch or LeRobot. Recorded when collection starts; any drift
before the seal aborts the run.
"""
import hashlib
from pathlib import Path

from experiments.b3_robosuite_data.plan import PLAN, SPEC

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
IMAGE = "sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e"
ADMISSION_IMAGE = "sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc"
OUTPUT = "outputs/b3_robosuite_data"
MIN_FREE_BYTES = 20 * 1024**3

# Modules this package imports (directly or through the expert/scene), by repo-relative path.
SHARED_FILES = (
    "data_collection/benchmark.py",
    "experiments/paired_demonstrations/plan.py",
    "experiments/paired_demonstrations/media.py",
    "experiments/paired_demonstrations/expert_v1.py",
    "experiments/pose_snapshot_v2/expert.py",
    "experiments/pose_snapshot_v2/planner.py",
    "experiments/b2_distractor_data/expert.py",
    "experiments/b2_distractor_data/plan.py",
    "experiments/fixed_layout/contract.py",
)
SOURCE_SUFFIXES = (".py", ".xml", ".urdf", ".obj", ".stl", ".png", ".jpg")


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
    paths = sorted((root / "experiments/b3_robosuite_data").rglob("*.py"))
    paths.append(root / "experiments/b3_robosuite_data/README.md")
    paths += sorted((root / "experiments/robosuite_scene").glob("*.py"))
    paths += [root / name for name in SHARED_FILES]
    paths += sorted(p for p in (root / "src").rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix in SOURCE_SUFFIXES)
    paths += [root / SPEC, root / PLAN]
    return paths


def provenance(root=ROOT):
    root = Path(root)
    return dict(sorted((str(p.relative_to(root)), _file_hash(p)) for p in source_paths(root)))
