"""Contact sheets: frames from a trial's `cameras.mp4`, one row per requested tick, in one PNG.

`cameras.mp4` holds one frame per tick plus the final state (frames = ticks + 1); frame k is the observation before
action k. Each frame is already both camera views side by side.
"""
from pathlib import Path

import numpy as np


def frames_at(video, ticks):
    """{tick: HxWx3 uint8} for the requested ticks. A tick beyond the last frame raises (never silently clamps)."""
    import av
    wanted = sorted(set(int(t) for t in ticks))
    out = {}
    with av.open(str(video)) as container:
        for k, frame in enumerate(container.decode(video=0)):
            if k in wanted:
                out[k] = frame.to_ndarray(format='rgb24')
            if k >= wanted[-1]:
                break
    missing = [t for t in wanted if t not in out]
    if missing:
        raise ValueError(f'ticks {missing} are beyond the last frame of {video}')
    return out


def sheet(folder, ticks, out, scale=2):
    """One PNG with the requested ticks stacked top to bottom (each row both views side by side), downscaled by `scale`."""
    from PIL import Image
    got = frames_at(Path(folder)/'cameras.mp4', ticks)
    rows = [got[t][::scale, ::scale] for t in sorted(got)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    return sorted(got)
