"""What the box detector is bound to: the sealed P2 parent, the S0 sidecar and its box table, the plan split, the pinned constants and its own
source. A training run, the gate and the cache all record this binding and refuse to continue if it changes.

Not the policy packages' full contract: the detector has no policy-training admission of its own, but the same rule holds, that a result
which cannot reproduce what it was built against is not used. `--check` writes the admission record; `--train` and the gate re-derive the
binding and compare.
"""
import hashlib
import json
from pathlib import Path

from experiments.vla_inputs_box_detector import constants as k
from experiments.vla_inputs_box_detector import predictor

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "experiments/vla_inputs_box_detector"
OUTPUT_ROOT = ROOT / "outputs/vla_inputs_box_detector"
CACHE = OUTPUT_ROOT / "cache_v1"
ADMISSION = OUTPUT_ROOT / "admission_v1"
RUN = OUTPUT_ROOT / "run_v1"
FINAL = RUN / "final"
# The files the binding hashes: this package's modules and README (tests are not bound). `tests/test_contract.py` checks the list equals the directory.
SOURCE_FILES = tuple(f"{PACKAGE}/{name}" for name in ("README.md", "__init__.py", "constants.py", "contract.py", "data.py", "gate.py", "inputs.py",
                                                       "model.py", "predictor.py", "runner.py"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root=ROOT):
    return {name: file_sha256(Path(root) / name) for name in SOURCE_FILES}


def check_split(plan):
    """The sealed plan's split is the one the detector trains and is gated on: episodes 0..479 train, 480..539 validation."""
    if list(plan["train_ids"]) != list(k.TRAIN_EPISODES) or list(plan["validation_ids"]) != list(k.VALIDATION_EPISODES):
        raise ValueError("the sealed plan's train/validation split is not the detector's")


def binding(root=ROOT):
    """The parent seal, plan, S0 sidecar and box table, the split, the constants and the sources, all re-derived now."""
    from experiments.xvla_p2_force_box import boxes, contract
    parent = Path(contract.PARENT)
    if json.loads((parent / "seal.json").read_text()) != contract.SEAL_RECORD:
        raise ValueError("the P2 parent seal record changed")
    plan = json.loads((parent / "plan.json").read_text())
    check_split(plan)
    return dict(p2_seal=contract.SEAL, p2_plan_sha256=contract.SEAL_RECORD["plan_sha256"], p2_plan_file_sha256=file_sha256(parent / "plan.json"),
                s0=contract.forces.binding(), boxes=boxes.binding(), constants_sha256=predictor.constants_digest(), sources=source_hashes(root))


def binding_sha256(root=ROOT):
    return digest(binding(root))
