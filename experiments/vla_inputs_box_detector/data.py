"""The detector's frame cache and training set.

**The cache** (`extract`): every `STRIDE_K`-th frame (0, 8, ..., 448) of every sealed P2 episode, decoded from the sealed AV1 videos by
LeRobot's own decoder at the timestamps its loader uses, preprocessed by `inputs.preprocess` (the front camera's fixed crop, the
gripper camera 2 x 2 averaged) and stored as uint8 arrays with S0's `target_box` of that (episode, frame) as the label, the object the
episode's instruction names, and a manifest with the sha256 of each array and of each episode's two source videos. Written once into a
new directory (never reused), after a disk check: the cache must leave at least `CACHE_DISK_GUARD_GIB` free.

**The training set** (`TrainingSet`): rows of the 480 training episodes only; a sample is a deterministic function of (seed, epoch, row), with
an exact translation per camera (`inputs.shift_image` / `shift_box`: the label moves with the frame and is absent when nothing of it
remains), so the detector learns absence from frames in which the object really is out of view.
"""
import hashlib
import json
import shutil
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from experiments.vla_inputs_box_detector import constants as k
from experiments.vla_inputs_box_detector import inputs, model

Cache = namedtuple("Cache", "directory front gripper labels index objects manifest")
ARRAYS = ("front", "gripper", "labels", "index", "objects")
RESERVE_BYTES = 2**26
CAMERAS = ("observation.images.image", "observation.images.image2")
TOLERANCE_S = 1e-4  # LeRobot's own default


def constants_record():
    return dict(stride=k.STRIDE_K, frames=list(k.CACHE_FRAMES), front_crop=list(k.FRONT_CROP), front_shape=list(k.FRONT_SHAPE),
                gripper_shape=list(k.GRIPPER_SHAPE), gripper_factor=k.GRIPPER_FACTOR)


def _shapes(n):
    return dict(front=((n, *k.FRONT_SHAPE, 3), np.uint8), gripper=((n, *k.GRIPPER_SHAPE, 3), np.uint8), labels=((n, 2, 5), np.float64),
                index=((n, 2), np.int32), objects=((n,), np.int8))


def needed_bytes(rows):
    per_row = int(np.prod(k.FRONT_SHAPE)) * 3 + int(np.prod(k.GRIPPER_SHAPE)) * 3 + 2 * 5 * 8 + 2 * 4 + 1
    return rows * per_row + RESERVE_BYTES


def create_arrays(directory, rows):
    """A new cache directory holding empty memory-mapped arrays for `rows` rows; an existing directory is an error."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    return {name: np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape) for name, (shape, dtype) in _shapes(rows).items()}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def finish(directory, arrays, extra):
    """Flush the arrays, write the manifest (hashes of every array file) and, last, the completion marker."""
    directory = Path(directory)
    rows = len(arrays["objects"])
    for array in arrays.values():
        array.flush()
    manifest = dict(constants_record(), constants=constants_record(), rows=rows,
                    arrays={name: dict(sha256=_sha256(directory / f"{name}.npy"), shape=list(arrays[name].shape), dtype=str(arrays[name].dtype)) for name in ARRAYS},
                    **extra)
    text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    (directory / "manifest.json").write_text(text)
    (directory / "complete").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
    return manifest


def open_cache(directory, verify=False):
    """The cache read-only. `verify=True` hashes every array against the manifest; either way the completion marker, the pinned
    constants (stride, crop, shapes) and the array shapes are checked."""
    directory = Path(directory)
    if not (directory / "complete").is_file():
        raise ValueError(f"{directory} has no 'complete' marker: the extraction did not finish")
    text = (directory / "manifest.json").read_text()
    manifest = json.loads(text)
    if manifest.get("stride") != k.STRIDE_K or manifest.get("constants") != constants_record():
        raise ValueError(f"{directory} was made under other constants (stride {manifest.get('stride')}, not {k.STRIDE_K})")
    if (directory / "complete").read_text().strip() != hashlib.sha256(text.encode()).hexdigest():
        raise ValueError(f"{directory}: the manifest does not match its completion marker")
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
    for name, (shape, dtype) in _shapes(manifest["rows"]).items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype or manifest["arrays"][name]["shape"] != list(shape):
            raise ValueError(f"{directory}: array {name} has the wrong shape or dtype")
        if verify and _sha256(directory / f"{name}.npy") != manifest["arrays"][name]["sha256"]:
            raise ValueError(f"{directory}: array {name} does not match its recorded sha256")
    return Cache(directory, arrays["front"], arrays["gripper"], arrays["labels"], arrays["index"], arrays["objects"], manifest)


def rows_of(cache, split):
    """Row numbers of the plan's `train` (episodes 0..479) or `validation` (480..539) episodes."""
    episodes = {"train": k.TRAIN_EPISODES, "validation": k.VALIDATION_EPISODES}.get(split)
    if episodes is None:
        raise ValueError(f"split must be 'train' or 'validation', not {split!r}")
    return np.flatnonzero(np.isin(cache.index[:, 0], episodes))


# --- extraction ----------------------------------------------------------------------------------------------------------------

def to_uint8(pixels):
    """Decoded float32 frames (N, 3, H, W) in [0, 1] -> uint8 (N, H, W, 3): rounded, never truncated (k / 255 * 255 may fall just below k)."""
    return (pixels * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def _extract_episode(task):
    """(episode, front (n, 288, 480, 3), gripper (n, 256, 256, 3), {video key: sha256}) for one episode's cached frames."""
    episode, frames, root = task
    torch.set_num_threads(1)
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import decode_video_frames
    root = Path(root)
    meta = LeRobotDatasetMetadata(root.name, root=root)
    entry, decoded, hashes = meta.episodes[episode], {}, {}
    for key in CAMERAS:
        path = root / meta.get_video_file_path(episode, key)
        stamps = [entry[f"videos/{key}/from_timestamp"] + frame / meta.fps for frame in frames]
        pixels = decode_video_frames(path, stamps, TOLERANCE_S, backend="torchcodec")
        decoded[key] = to_uint8(pixels)
        hashes[key] = _sha256(path)
    pairs = [inputs.preprocess([decoded[CAMERAS[0]][i], decoded[CAMERAS[1]][i]]) for i in range(len(frames))]
    return episode, np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs]), hashes


def extract(directory, episodes=None, workers=6, dataset_root=None, free_bytes=None, guard_gib=k.CACHE_DISK_GUARD_GIB):
    """Build the cache for `episodes` (default all 540) in the new directory `directory`; refuses unless at least `guard_gib` GiB stay free."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"{directory} exists; a cache directory is never reused")
    from experiments.xvla_p2_force_box import boxes, contract
    episodes = tuple(range(540)) if episodes is None else tuple(episodes)
    rows = len(episodes) * len(k.CACHE_FRAMES)
    need = needed_bytes(rows)
    free = (free_bytes or (lambda path: shutil.disk_usage(path).free))(directory.parent if directory.parent.exists() else Path("/"))
    if free - need < guard_gib * 2**30:
        raise RuntimeError(f"refusing to extract: {rows} rows need {need / 2**30:.1f} GiB and only {free / 2**30:.1f} GiB are free (guard {guard_gib} GiB after)")
    root = Path(dataset_root or contract.DATA)
    table, instructions = boxes.table(), contract.planned_instructions()
    arrays = create_arrays(directory, rows)
    tasks = [(episode, list(k.CACHE_FRAMES), str(root)) for episode in episodes]
    records = []
    if workers == 1:
        results = map(_extract_episode, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        results = executor.map(_extract_episode, tasks)
    try:
        for position, (episode, front, gripper, hashes) in enumerate(results):
            lo, hi = position * len(k.CACHE_FRAMES), (position + 1) * len(k.CACHE_FRAMES)
            arrays["front"][lo:hi], arrays["gripper"][lo:hi] = front, gripper
            arrays["labels"][lo:hi] = table[episode, list(k.CACHE_FRAMES)]
            arrays["index"][lo:hi] = [(episode, f) for f in k.CACHE_FRAMES]
            arrays["objects"][lo:hi] = inputs.parse_instruction(instructions[episode])
            records.append(dict(episode=episode, instruction=instructions[episode], videos=hashes, first_row=lo))
            print("EXTRACTED", episode, f"{position + 1}/{len(tasks)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    return finish(directory, arrays, dict(episodes=records, dataset=str(root.name)))


# --- training samples ----------------------------------------------------------------------------------------------------------

class TrainingSet(torch.utils.data.Dataset):
    """Samples of the training rows. A sample is addressed by a global id `epoch * len + position` and is a pure function of
    (seed, epoch, row): the same id always gives the same augmentation."""

    def __init__(self, cache, rows, shift_probability=k.SHIFT_PROBABILITY, seed=k.SEED):
        rows = np.asarray(rows, dtype=np.int64)
        if not np.isin(cache.index[rows, 0], k.TRAIN_EPISODES).all():
            raise ValueError("a validation episode is among the training rows")
        self.cache, self.rows, self.shift_probability, self.seed = cache, rows, shift_probability, seed

    def __len__(self):
        return len(self.rows)

    def sample_row(self, global_id):
        return self.rows[int(global_id) % len(self.rows)]

    def episode(self, global_id):
        return int(self.cache.index[self.sample_row(global_id), 0])

    def _camera(self, rng, image, box, shape):
        dx = dy = 0
        if rng.random() < self.shift_probability:
            dx = int(rng.integers(-int(k.SHIFT_MAX_FRACTION * shape[1]), int(k.SHIFT_MAX_FRACTION * shape[1]) + 1))
            dy = int(rng.integers(-int(k.SHIFT_MAX_FRACTION * shape[0]), int(k.SHIFT_MAX_FRACTION * shape[0]) + 1))
        image = np.array(image)
        if (dx, dy) != (0, 0):
            image = inputs.shift_image(image, dx, dy)
            if box is not None:
                box, present = inputs.shift_box(box, dx, dy, shape)
                box = box if present else None
        return image, box, (dx, dy)

    def __getitem__(self, global_id):
        epoch, position = divmod(int(global_id), len(self.rows))
        row = int(self.rows[position])
        rng = np.random.default_rng([self.seed, epoch, row])
        label = self.cache.labels[row]
        front, front_box, front_shift = self._camera(rng, self.cache.front[row], inputs.front_box_to_input(label[0]), k.FRONT_SHAPE)
        gripper, gripper_box, gripper_shift = self._camera(rng, self.cache.gripper[row], inputs.gripper_box_to_input(label[1]), k.GRIPPER_SHAPE)
        return dict(front=front, gripper=gripper, front_box=front_box, gripper_box=gripper_box, cls=int(self.cache.objects[row]),
                    front_shift=front_shift, gripper_shift=gripper_shift, row=row)


def collate(samples):
    cls = [s["cls"] for s in samples]
    return dict(front=torch.from_numpy(np.stack([s["front"] for s in samples])), gripper=torch.from_numpy(np.stack([s["gripper"] for s in samples])),
                front_targets=model.collate_targets([model.make_target(s["front_box"], k.FRONT_SHAPE) for s in samples], cls),
                gripper_targets=model.collate_targets([model.make_target(s["gripper_box"], k.GRIPPER_SHAPE) for s in samples], cls),
                cls=torch.tensor(cls, dtype=torch.long), rows=torch.tensor([s["row"] for s in samples]))


def photometric(x, generator):
    """Brightness, contrast and noise jitter on float images (N, 3, H, W) in [0, 1] (pinned strengths); reproducible from `generator`."""
    n = x.shape[0]
    brightness = (torch.rand(n, 1, 1, 1, generator=generator) * 2 - 1) * k.BRIGHTNESS
    contrast = (torch.rand(n, 1, 1, 1, generator=generator) * 2 - 1) * k.CONTRAST
    noise = torch.randn(x.shape, generator=generator) * k.NOISE
    out = (x - 0.5) * (1 + contrast.to(x.device)) + 0.5 + brightness.to(x.device) + noise.to(x.device)
    return out.clamp(0.0, 1.0)
