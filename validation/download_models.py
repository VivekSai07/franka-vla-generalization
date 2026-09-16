"""Fetch pinned public policy weights and VLM metadata; no tokens printed."""
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    destination = Path(".cache/benchmark_models")
    destination.mkdir(parents=True, exist_ok=True)
    targets = [
        ("policy", "lerobot/smolvla_base", "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
         ["config.json", "model.safetensors"]),
        ("vlm", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
         ["*.json", "*.txt"]),
    ]
    manifest = {}
    for key, repo_id, revision, patterns in targets:
        print(f"Acquiring {repo_id}@{revision}", flush=True)
        try:
            path = snapshot_download(repo_id, revision=revision, allow_patterns=patterns,
                                     local_dir=destination / key, max_workers=2, token=False)
        except Exception as exc:
            # Hub exceptions can contain signed URLs. Keep them out of reports.
            print(f"BLOCKED: {repo_id} download failed ({type(exc).__name__}); no training started.", flush=True)
            raise SystemExit(1) from None
        manifest[key] = {"repo_id": repo_id, "revision": revision, "path": str(Path(path).resolve())}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Model acquisition complete")


if __name__ == "__main__":
    main()