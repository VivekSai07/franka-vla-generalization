"""Fresh evaluation and cryptographically bound validation results."""
import hashlib
import json
from pathlib import Path

import mujoco

from data_collection.benchmark import capture_state, restore_state, stack_metrics, source_provenance, runtime_versions


def fresh_stack_metrics(model, data, scratch, pick, place):
    # mj_step integrates qpos after computing xpos/contact; score a forwarded
    # copy so post-step geometry is current without altering replay warm starts.
    restore_state(model, scratch, capture_state(model, data))
    return stack_metrics(model, scratch, pick, place)


def dataset_fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    paths = sorted(p for directory in ("meta", "data", "videos", "replay")
                   for p in (root / directory).rglob("*") if p.is_file()
                   and p.name not in ("replay_result.json", "replay_result.json.tmp"))
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def require_replay_pass(root, source_root):
    root = Path(root)
    result = json.loads((root / "replay/replay_result.json").read_text())
    manifest = json.loads((root / "replay/manifest.json").read_text())
    if (result.get("passed") is not True or manifest["source_hashes"] != source_provenance(source_root)
            or result.get("dataset_fingerprint") != dataset_fingerprint(root)
            or manifest["versions"] != runtime_versions()):
        raise ValueError("Replay result is absent, failed or stale")
    if manifest.get("validation_required"):
        from validation.validate_pilot import check_final_binding
        check_final_binding(root, result["dataset_fingerprint"])
    return result