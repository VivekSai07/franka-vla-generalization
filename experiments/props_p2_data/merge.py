"""Merge the P2 shards into one LeRobot dataset, prove it equals the shards, then validate and seal.

`lerobot.datasets.aggregate.aggregate_datasets` (LeRobot 0.4.4) with file size 0 writes one data and one
video file per episode: videos are copied byte for byte, data parquet is rewritten with shifted
`episode_index` / `index` / `task_index`, statistics are re-aggregated from the shards'. `verify`
checks every merged episode against its shard episode: length, task string, every data column
(indices checked as shifted), each video file's sha256 and its timestamps.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.props_p2_data import contract
from experiments.props_p2_data import plan as p
from experiments.props_p2_data.record import (
    _require_pinned_runtime, _write_json, read_journal, shard_root, substitutions_of,
)

MERGED_REPO_ID = "props_p2_data"
VIDEO_KEYS = ("observation.images.image", "observation.images.image2")
DATA_PATH = "data/chunk-{chunk:03d}/file-{file:03d}.parquet"
VIDEO_PATH = "videos/{key}/chunk-{chunk:03d}/file-{file:03d}.mp4"


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _episodes(dataset_root):
    import pyarrow.parquet as pq
    rows = []
    for path in sorted((Path(dataset_root) / "meta/episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return sorted(rows, key=lambda e: e["episode_index"])


def _tasks(dataset_root):
    import pandas as pd
    table = pd.read_parquet(Path(dataset_root) / "meta/tasks.parquet")
    return {int(i): name for name, i in zip(table.index, table["task_index"])}


def _data(dataset_root, episode):
    import pandas as pd
    path = Path(dataset_root) / DATA_PATH.format(chunk=episode["data/chunk_index"], file=episode["data/file_index"])
    frame = pd.read_parquet(path)
    return frame[frame["episode_index"] == episode["episode_index"]].reset_index(drop=True)


def verify(shard_dataset_roots, merged_root):
    """Every merged episode equals its shard episode; raises on the first difference, returns a summary."""
    merged = _episodes(merged_root)
    merged_tasks = _tasks(merged_root)
    expected = [(k, e) for k, root in enumerate(shard_dataset_roots) for e in _episodes(root)]
    if [e["episode_index"] for e in merged] != list(range(len(expected))):
        raise ValueError(f"merged holds {len(merged)} episodes, shards hold {len(expected)}")
    frames = 0
    for i, (m, (k, s)) in enumerate(zip(merged, expected)):
        root = shard_dataset_roots[k]
        if m["length"] != s["length"] or list(m["tasks"]) != list(s["tasks"]):
            raise ValueError(f"episode {i}: length or task differs from shard {k} episode {s['episode_index']}")
        md, sd = _data(merged_root, m), _data(root, s)
        if len(md) != m["length"] or len(sd) != s["length"]:
            raise ValueError(f"episode {i}: data rows differ from the episode length")
        for column in ("observation.state", "action", "timestamp", "frame_index"):
            if not np.array_equal(np.stack(md[column].to_numpy()), np.stack(sd[column].to_numpy())):
                raise ValueError(f"episode {i}: column {column} differs from shard {k}")
        shard_tasks = _tasks(root)
        if [merged_tasks[t] for t in md["task_index"]] != [shard_tasks[t] for t in sd["task_index"]]:
            raise ValueError(f"episode {i}: task strings differ from shard {k}")
        if (md["episode_index"] != i).any() or not np.array_equal(md["index"].to_numpy(), np.arange(frames, frames + len(md))):
            raise ValueError(f"episode {i}: episode_index/index not contiguous after the merge")
        for key in VIDEO_KEYS:
            mv = Path(merged_root) / VIDEO_PATH.format(key=key, chunk=m[f"videos/{key}/chunk_index"], file=m[f"videos/{key}/file_index"])
            sv = Path(root) / VIDEO_PATH.format(key=key, chunk=s[f"videos/{key}/chunk_index"], file=s[f"videos/{key}/file_index"])
            if _sha256(mv) != _sha256(sv):
                raise ValueError(f"episode {i}: video {key} bytes differ from shard {k}")
            for t in ("from_timestamp", "to_timestamp"):
                if m[f"videos/{key}/{t}"] != s[f"videos/{key}/{t}"]:
                    raise ValueError(f"episode {i}: video {key} {t} differs from shard {k}")
        frames += len(md)
    info = json.loads((Path(merged_root) / "meta/info.json").read_text())
    if info["total_episodes"] != len(merged) or info["total_frames"] != frames:
        raise ValueError("merged info.json totals differ from the episodes")
    return {"episodes": len(merged), "frames": frames, "shards": len(shard_dataset_roots),
            "videos_verified": len(merged) * len(VIDEO_KEYS)}


def aggregate(shard_dataset_roots, merged_root):
    from lerobot.datasets.aggregate import aggregate_datasets
    aggregate_datasets(repo_ids=[f"props_p2_shard_{k:02d}" for k in range(len(shard_dataset_roots))],
                       aggr_repo_id=MERGED_REPO_ID, roots=[Path(r) for r in shard_dataset_roots],
                       aggr_root=Path(merged_root), data_files_size_in_mb=0, video_files_size_in_mb=0)


def merge_and_seal(shards_dir, out, shards, pilot=False):
    """Pilot: merge whatever shards exist and verify only. Full: every shard complete, then validate and seal."""
    _require_pinned_runtime()
    from experiments.props_p2_data.validate import seal, validate_dataset
    shards_dir, out = Path(shards_dir), Path(out)
    roots = sorted(shards_dir.glob("shard_*")) if pilot else [shard_root(shards_dir, k) for k in range(shards)]
    plan, provenance = p.make_plan(), contract.provenance()
    journal = []
    for k, root in enumerate(roots):
        if json.loads((root / "plan.json").read_text()) != json.loads(json.dumps(plan)):
            raise SystemExit(f"{root.name}: plan differs")
        if json.loads((root / "provenance.json").read_text()) != provenance:
            raise SystemExit(f"{root.name}: provenance differs from the current source")
        lines = read_journal(root / "journal.jsonl")
        slots = json.loads((root / "slots.json").read_text())
        if not pilot and (slots != p.shard_slots(k, shards) or not (root / "done.json").exists()
                          or [j["slot"] for j in lines if j["admitted"]] != slots):
            raise SystemExit(f"{root.name}: incomplete or not shard {k}")
        journal += lines
    if out.exists():
        raise SystemExit(f"No overwrite: {out}")
    out.mkdir(parents=True)
    aggregate([r / "dataset" for r in roots], out / "dataset")
    report = verify([r / "dataset" for r in roots], out / "dataset")
    with (out / "journal.jsonl").open("w") as stream:
        for line in journal:
            stream.write(json.dumps(line, allow_nan=False) + "\n")
    (out / "episodes").mkdir()
    for root in roots:
        for npz in sorted((root / "episodes").glob("*.npz")):
            os.link(npz, out / "episodes" / npz.name)
    _write_json(out / "plan.json", plan)
    _write_json(out / "provenance.json", provenance)
    _write_json(out / "substitutions.json", substitutions_of(journal))
    _write_json(out / "merge.json", {**report, "shard_roots": [r.name for r in roots]})
    print(f"P2 MERGE VERIFIED {report}", flush=True)
    if pilot:
        return report
    validation = validate_dataset(out, plan, journal)
    _write_json(out / "validation.json", validation)
    sealed = seal(out, plan, provenance)
    print(f"P2 SEALED episodes={validation['episodes']} fingerprint={sealed['fingerprint']}", flush=True)
    return report
