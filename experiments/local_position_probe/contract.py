"""The only authorized schedule; no runtime budget/model/checkpoint overrides."""
import hashlib
import json
from pathlib import Path

from experiments.fixed_layout.contract import IMAGE, SOURCE_SCENE_SHA256, _same

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'outputs/diagnostics/local_position_probe_20260919'
PARENT = ROOT/'outputs/fixed_layout/fixed_layout_20260918'
RESET_SHA256 = '6c2b6b49b7b3eccd4c37a4ac134f85865eba15745d3629bd98300d3e00652f62'
MODELS = ('smolvla', 'xvla')
SEEDS = (20260918, 20260923)
MIN_DISK_BYTES = 20*1024**3
MIN_GPU_BYTES = 12*1024**3
MAX_SECONDS = 4*60*60
SPEC = 'docs/superpowers/specs/2026-09-18-local-position-probe-design.md'
PLAN = 'docs/superpowers/plans/2026-09-19-local-position-probe.md'
PACKAGE_FILES = ('__init__.py', 'contract.py', 'scenes.py', 'runner.py', 'telemetry.py',
                 'tests/test_contract.py', 'tests/test_scenes.py', 'tests/test_runner.py', 'README.md')
BASELINE_FILES = {
    'smolvla': {
        'result.json': '94616901fde79428f2b64e5ce1f55c983bf29bd9c272ba12e68610758462b130',
        'trial_01_seed_20260918/episode.json': 'a180cea0d71046d84ae68854ac2b38ae46a28a6008afd24b0712523a86f9d19e',
        'trial_01_seed_20260918/trace.npz': 'd61177a5d2eb9b942f5c230bddc486e209f0152b4fd305999f3abd5ca02b5be2',
        'trial_06_seed_20260923/episode.json': '4b6409ceb4fbede9f941422cf22b43f2cdfbfdd6b5bbdd495cb343e17909c5d7',
        'trial_06_seed_20260923/trace.npz': 'ef27991f8fe570b8fdf5edd12eaf6b78159114406a738783969b433b902b618f',
    },
    'xvla': {
        'result.json': '1caf9e212df5cdfcb5d0d2ba902b24169e83e8ed7c9f8d301875a7c66186b53b',
        'trial_01_seed_20260918/episode.json': '653aa88958669d3bfb6d98cd60be796f5c67e547337ed57ad448697af7447bc0',
        'trial_01_seed_20260918/trace.npz': 'f3c24f38bf866dc45870b360ac852243e9f98e7f885b2d18988f4379dfb6a4f5',
        'trial_06_seed_20260923/episode.json': 'de1b8223e47fbbc5e37dafc4c2086a79165f50d3710d305e74fcb16324bb01ed',
        'trial_06_seed_20260923/trace.npz': 'e145e8e7ec286e852a00f477692c2a70069fd5f6cd7637c317a135ab084eac25',
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_baselines():
    files = {}
    for model, inventory in BASELINE_FILES.items():
        for relative, expected in inventory.items():
            path = ROOT/f'outputs/train/{model}_fixed_layout_20260918/evaluation'/relative
            if any(p.is_symlink() for p in (path, *path.parents)) or sha256(path) != expected:
                raise ValueError('Pinned original evaluated baseline bytes changed')
            files[str(path.relative_to(ROOT))] = expected
    return files


def make_plan() -> dict:
    conditions = [dict(condition_id='control', object=None, axis=None, magnitude_mm=0, delta_m=0.)]
    for color in ('red', 'green'):
        for mm in (5, 10):
            for direction, axis, sign in (('px', 0, 1), ('nx', 0, -1), ('py', 1, 1), ('ny', 1, -1)):
                conditions.append(dict(condition_id=f'{color}_{mm}mm_{direction}', object=color,
                                       axis=axis, magnitude_mm=mm, delta_m=sign*mm/1000))
    trials = []
    for model in MODELS:
        for condition in conditions:
            for seed, ordinal in ((20260918, 1), (20260923, 6)):
                trials.append(dict(study_index=len(trials)+1, trial_id=f"{model}_{condition['condition_id']}_{seed}",
                                   model=model, condition_id=condition['condition_id'], seed=seed,
                                   baseline_ordinal=ordinal, source_episode_index=7))
    return dict(version='local_position_probe_v1', conditions=conditions, trials=trials,
                source_episode_index=7, source_reset_sha256=RESET_SHA256,
                source_scene_sha256=SOURCE_SCENE_SHA256, image=IMAGE,
                models=list(MODELS), seeds=list(SEEDS), planned_trials=68,
                max_control_ticks=1200, max_sim_seconds=40, strict_score_ticks=30,
                fps=30, substeps=20, min_disk_bytes=MIN_DISK_BYTES, min_gpu_bytes=MIN_GPU_BYTES,
                max_wall_seconds=MAX_SECONDS, reset_policy='translate_one_qpos_scalar_no_steps_no_resettle',
                resume=False, retries=False, training=False,
                seed_selection='development_selected_including_known_smolvla_seed23_failure',
                scope='descriptive_two_samples_per_condition_not_spatial_generalization')


def validate_plan(plan):
    if not _same(plan, make_plan()): raise ValueError('Noncanonical local-position plan')


def validate_condition(condition):
    if not any(_same(condition, item) for item in make_plan()['conditions']):
        raise ValueError('Only the seventeen predeclared conditions are authorized')


def package_hashes():
    directory = Path(__file__).parent
    if {str(p.relative_to(directory)) for p in directory.rglob('*.py')} != {n for n in PACKAGE_FILES if n.endswith('.py')}:
        raise ValueError('Unexpected/missing experiment Python source')
    paths = [directory/name for name in PACKAGE_FILES] + [ROOT/SPEC, ROOT/PLAN]
    if any(p.is_symlink() for p in paths): raise ValueError('Symlink source forbidden')
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}