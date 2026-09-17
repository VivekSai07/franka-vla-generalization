"""Exclusive durable tick/state data and lossless diagnostic camera videos."""
from pathlib import Path
import av
import numpy as np

from experiments.paired_demonstrations.media import CAMERAS
from .diagnostic import save_json
from .solver import PlanningFailure


def save_npz(path, arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


class TraceSink:
    def __init__(self, out, instruction, prefix_reference=None):
        self.out, self.instruction = Path(out), instruction
        self.prefix_reference = prefix_reference
        (self.out/'ticks').mkdir(exist_ok=False)
        self.count, self.videos, self.streams = 0, {}, {}
        for camera in CAMERAS.values():
            handle = (self.out/f'{camera}.mkv').open('xb')
            video = av.open(handle, mode='w', format='matroska')
            stream = video.add_stream('ffv1', rate=30)
            stream.width, stream.height, stream.pix_fmt = 512, 512, 'bgr0'
            stream.thread_count = 1
            self.videos[camera], self.streams[camera] = (video, handle), stream

    def __call__(self, frame, trace):
        if self.count >= 1200 or frame['task'] != self.instruction:
            raise PlanningFailure('sink_tick_or_instruction')
        # Nested diagnostic decisions are JSON; numeric raw controls/states stay
        # native float64 NPZ, not object arrays or pickle.
        nested = {key: value for key, value in trace.items() if isinstance(value, (dict, list)) or value is None}
        arrays = {key: np.asarray(value).copy() for key, value in trace.items() if key not in nested}
        arrays.update(action=frame['action'].copy(), observation_state=frame['observation.state'].copy())
        save_npz(self.out/'ticks'/f'tick_{self.count:06d}.npz', arrays)
        save_json(self.out/'ticks'/f'tick_{self.count:06d}.json', nested)
        for key, camera in CAMERAS.items():
            image = av.VideoFrame.from_ndarray(frame[key], format='rgb24')
            image.pts = self.count
            for packet in self.streams[camera].encode(image):
                self.videos[camera][0].mux(packet)
        if self.prefix_reference is not None and self.count < 197:
            # Persist the failing tick first, then block the placement planner.
            for key, expected in self.prefix_reference.items():
                np.testing.assert_array_equal(arrays[key], expected[self.count],
                                              err_msg=f'v2 prefix {key} tick {self.count}')
        self.count += 1

    def finish(self):
        for camera, (video, handle) in self.videos.items():
            try:
                for packet in self.streams[camera].encode():
                    video.mux(packet)
            finally:
                video.close()
                handle.close()
        self.videos.clear()