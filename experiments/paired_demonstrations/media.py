"""Pinned LeRobot serial AV1/yuv420p encoding and full lossy-video validation."""
import hashlib

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMERAS = {"observation.images.image": "front_cam", "observation.images.image2": "gripper_cam"}
FEATURES = {
    "action": {"dtype": "float32", "shape": (8,),
               "names": [f"panda_joint{i}.pos" for i in range(1, 8)] + ["gripper.pos"]},
    "observation.state": {"dtype": "float32", "shape": (9,),
                          "names": [f"panda_joint{i}.pos" for i in range(1, 8)] + ["finger1.pos", "finger2.pos"]},
    **{key: {"dtype": "video", "shape": (512, 512, 3), "names": ["height", "width", "channels"]}
       for key in CAMERAS},
}
ENCODING = {"vcodec": "libsvtav1", "pix_fmt": "yuv420p", "crf": 30, "g": 2, "encoder_threads": 1}


def pixels_digest(image):
    if image.dtype != np.uint8 or image.shape != (512, 512, 3):
        raise ValueError("Expected 512x512 RGB uint8 observation")
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def decode_video(path):
    """Decode every frame; lossy pixels must not be compared to raw RGB hashes."""
    count = 0
    with av.open(str(path)) as video:
        if len(video.streams.video) != 1:
            raise ValueError("Expected exactly one video stream")
        stream = video.streams.video[0]
        if stream.codec.canonical_name != "av1" or stream.pix_fmt != "yuv420p":
            raise ValueError("Expected AV1/yuv420p camera video")
        if float(stream.average_rate) != 30 or (stream.width, stream.height) != (512, 512):
            raise ValueError("Wrong camera size/FPS")
        for tick, frame in enumerate(video.decode(stream)):
            if (frame.width, frame.height) != (512, 512):
                raise ValueError("Wrong decoded camera dimensions")
            if frame.pts is None or abs(float(frame.pts * frame.time_base) - tick / 30) > 1e-5:
                raise ValueError("Video timestamp or observation boundary mismatch")
            frame.to_ndarray(format="rgb24")  # Decode pixels, not just packet headers.
            count += 1
    if not count:
        raise ValueError("Empty video")
    return count


class PairedDataset(LeRobotDataset):
    """Use the original collector's pinned encoder, without global patches."""

    def save_episode(self, episode_data=None, parallel_encoding=False):
        if parallel_encoding or self.batch_encoding_size != 1 or self._streaming_encoder is not None:
            raise ValueError("Paired dataset requires serial immediate AV1 encoding")
        if self.vcodec != ENCODING["vcodec"] or self._encoder_threads != ENCODING["encoder_threads"]:
            raise ValueError("Paired dataset requires the pinned AV1 encoder configuration")
        return super().save_episode(episode_data, parallel_encoding=False)