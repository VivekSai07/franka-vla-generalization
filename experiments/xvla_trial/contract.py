"""Fixed admission contract; no model construction or CUDA calls."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/franka_pilot_20260915'
CACHE = ROOT / '.cache/xvla_trial/bundle'
OUTPUT = ROOT / 'outputs/validation/xvla_preflight_20260917'
TRAIN, VAL = tuple(range(9)), tuple(range(300, 304))
TASK = 'pick up the red cube and place it on top of the green cube'
CAMERAS = ('observation.images.image', 'observation.images.image2')
OBS = {'observation.state': (9,), **{k: (3, 512, 512) for k in CAMERAS}}
SEED = 20260917


def select_entries(manifest):
    entries = {e['episode_index']: e for e in manifest['episodes']}
    train, val = manifest['splits']['train'], manifest['splits']['validation']
    if (len(manifest['episodes']) != 420 or set(entries) != set(range(420))
            or set(train) != set(range(300)) or len(train) != 300
            or set(val) != set(range(300, 420)) or len(val) != 120
            or manifest['fps'] != 30 or manifest['camera_size'] != [512, 512]
            or manifest['camera_names'] != ['front_cam', 'gripper_cam']):
        raise ValueError('Unexpected manifest/split/schema')
    selected = [entries[i] for i in TRAIN + VAL]
    for e in selected:
        if (e['split'] != ('train' if e['episode_index'] in TRAIN else 'validation')
                or (e['pick'], e['place']) != ('red', 'green')
                or len(e['colors']) != 2 or set(e['colors']) != {'red', 'green'}
                or e['instruction'] != TASK or e['frames'] != 528):
            raise ValueError('Selected episode identity mismatch')
    return selected


def indices(episodes):
    for i in TRAIN + VAL:
        e = episodes[i]
        if (e['episode_index'] != i or int(e['dataset_from_index']) != i*528
                or int(e['dataset_to_index']) != (i+1)*528):
            raise ValueError('Episode offsets/identity mismatch')
    rows = [j for i in TRAIN for j in range(i*528, (i+1)*528)]
    starts = [j for i in TRAIN for j in range(i*528, (i+1)*528-29)]
    return rows, starts