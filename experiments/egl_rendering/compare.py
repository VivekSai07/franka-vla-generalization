"""Pure comparisons for the EGL side-by-side check. numpy only, so they test in either image.

An *image* is one camera at one tick (a P1 attempt of 452 ticks has 904), a *pixel* is one position in one image whatever
its channels.
"""
import numpy as np

MAX_LEVELS = 2             # criterion C3: largest pixel difference between two EGL runs, in levels of 255
MAX_IMAGE_FRACTION = 0.01  # criterion C3: share of images that may differ at all


def _diff(a, b):
    """|a - b| per element as int16, so uint8 values never wrap."""
    if a.shape != b.shape:
        raise ValueError(f'shape mismatch: {a.shape} against {b.shape}')
    return np.abs(a.astype(np.int16) - b.astype(np.int16))


def _summary(d):
    pixel = d.max(axis=-1) > 0                               # (..., H, W)
    return dict(pixels_differing=int(pixel.sum()), max_abs=int(d.max()) if d.size else 0)


def compare_images(a, b):
    """Differences between two uint8 stacks shaped (ticks, cameras, H, W, 3)."""
    d = _diff(a, b)
    pixel = d.max(axis=-1) > 0
    per_image = pixel.reshape(pixel.shape[0], pixel.shape[1], -1).any(axis=-1)           # (ticks, cameras)
    return dict(images=int(per_image.size), images_differing=int(per_image.sum()), pixels_differing=int(pixel.sum()),
                pixel_fraction=float(pixel.mean()), max_abs=int(d.max()), mean_abs=float(d.mean()),
                per_camera=[_summary(d[:, c]) for c in range(d.shape[1])])


def physics_differences(a, b):
    """Names of the entries (arrays or plain values) that are not bit-identical, including missing ones, sorted."""
    bad = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            bad.append(key)
            continue
        x, y = a[key], b[key]
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x, y = np.asarray(x), np.asarray(y)
            same = x.dtype == y.dtype and x.shape == y.shape and (x.tobytes() == y.tobytes())
        else:
            same = x == y
        if not same:
            bad.append(key)
    return bad


def repeatable(pair_stats):
    """Criterion C3 over the `compare_images` results of every pair of repeated EGL runs. An empty set is not a pass."""
    return bool(pair_stats) and all(
        s['max_abs'] <= MAX_LEVELS and s['images_differing'] <= MAX_IMAGE_FRACTION * s['images'] for s in pair_stats)
