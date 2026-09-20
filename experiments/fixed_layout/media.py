"""Bind raw -> ordered lossless PNG inputs -> pinned lossy encoder -> media bytes.

This is creation-time chain-of-custody evidence, not a raw/AV1 pixel-equality
test or an independent proof of encoder correctness. No re-encoding is needed.
"""
import hashlib
import io

import numpy as np
from PIL import Image

from experiments.paired_demonstrations.media import CAMERAS, ENCODING, PairedDataset, pixels_digest


class BoundPairedDataset(PairedDataset):
    """Keep the installed serial encoder and persistence lifecycle unchanged."""

    def _encode_temporary_episode_video(self, video_key, episode_index):
        from experiments.fixed_layout.record import file_hash
        check = getattr(self, "fixed_layout_check", lambda: None)
        # For fixed_layout runs our attempts index always equals LeRobot's episode_index;
        # broad_workspace retries make them diverge, so the active-attempt path (set by
        # record_attempt) is authoritative when present.
        attempt = getattr(self, "_active_attempt_path", None)
        if attempt is None:
            attempt = self.root.parent / "attempts" / f"{episode_index:06d}"
        raw_paths = sorted((attempt / "ticks").glob("*.npz"))
        images = self._get_image_file_dir(episode_index, video_key)
        expected_names = [f"frame-{i:06d}.png" for i in range(len(raw_paths))]
        if (not raw_paths or [p.name for p in raw_paths] != [f"{i:06d}.npz" for i in range(len(raw_paths))]
                or sorted(p.name for p in images.iterdir()) != expected_names):
            raise ValueError("Encoder input PNG/raw frame order or coverage mismatch")
        camera = list(CAMERAS).index(video_key)
        frames = []
        for i, raw_path in enumerate(raw_paths):
            check()
            with np.load(raw_path, allow_pickle=False) as raw:
                digest = pixels_digest(raw[video_key])
                if digest != raw["trace/image_sha256"][camera]:
                    raise ValueError("Encoder raw source/trace pixel hash mismatch")
            png = images / expected_names[i]
            if png.is_symlink():
                raise ValueError("Encoder input PNG cannot be a symlink")
            payload = png.read_bytes()
            with Image.open(io.BytesIO(payload)) as image:
                if image.mode != "RGB" or pixels_digest(np.asarray(image)) != digest:
                    raise ValueError("Encoder input PNG differs from saved raw source/camera/frame")
            frames.append({"frame_index": i, "raw_pixel_sha256": digest,
                           "png_sha256": hashlib.sha256(payload).hexdigest()})
        check()
        # The inherited worker consumes these exact canonical PNG names, serially.
        encoded = super()._encode_temporary_episode_video(video_key, episode_index)
        self._fixed_encoding = {"episode_index": episode_index, "video_key": video_key,
                                "encoding": dict(ENCODING), "frames": frames,
                                "video_sha256": file_hash(encoded, check)}
        return encoded

    def _save_episode_video(self, video_key, episode_index, temp_path=None):
        from experiments.fixed_layout.record import file_hash, write_json
        if temp_path is not None or self.meta.video_files_size_in_mb != 0:
            raise ValueError("Fixed-layout requires independently bound serial camera files")
        metadata = super()._save_episode_video(video_key, episode_index)
        check = getattr(self, "fixed_layout_check", lambda: None)
        prefix = "videos/" + video_key
        relative = self.meta.video_path.format(video_key=video_key,
            chunk_index=metadata[prefix + "/chunk_index"], file_index=metadata[prefix + "/file_index"])
        binding = self._fixed_encoding
        if file_hash(self.root / relative, check) != binding["video_sha256"]:
            raise ValueError("Encoder output changed while persisting camera media")
        directory = self.root.parent / "attempts" / f"{episode_index:06d}" / "media"
        directory.mkdir(exist_ok=True)
        write_json(directory / f"{video_key}.json", {**binding, "video_path": str(relative),
            "interval": [metadata[prefix + "/from_timestamp"], metadata[prefix + "/to_timestamp"]]})
        check()
        return metadata