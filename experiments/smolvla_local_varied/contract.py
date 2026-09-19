"""Local-varied60 data, train-only statistics, local assets and phase admission contracts."""
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate

from training import smolvla_pilot as h
from experiments.single_task.contract import learning_rate
from experiments.fixed_layout.contract import TASK, IMAGE, _same
from experiments.local_varied_data.plan import validate_plan

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'outputs/local_varied_data/local_varied_20260919'
DATA = PARENT/'dataset'
CACHE = ROOT/'.cache/benchmark_models'
OUTPUT = ROOT/'outputs/train/smolvla_franka_stacking_localvaried_v1_20260919'
SCRATCH = OUTPUT.with_name(OUTPUT.name+'_preflight')
ADMISSION = OUTPUT.with_name(OUTPUT.name+'_admission')
CPU_ADMISSION = ADMISSION/'result.json'
SEAL = 'f8b10a498798217a63564416ab6561e25bd7e9c87e4fdddcf42f288cdb9e9a28'
REVISION = 'c83c3163b8ca9b7e67c509fffd9121e66cb96205'
VLM_REVISION = '7b375e1b73b11138ff12fe22c8f2822d8fe03467'
# Same offline SmolVLA cache as fixed_layout baseline; recheck full byte inventory.
CACHE_DIGEST = '50a731a4147092eeaea3c0f8d46c6e2f2dd5279c18b815ebbad860de247022ab'
TRAIN, VAL, SEED = tuple(range(48)), tuple(range(48, 60)), 20260917
TRAINING_ROWS = 48*528  # 25344
COMPLETE_STARTS = 48*479  # 22992, last complete chunk begins528-50+1=479 offsets in
VALIDATION_ROWS = 12*528  # 6336
TOTAL_FRAMES = 60*528  # 31680
CAMERAS = ('observation.images.image', 'observation.images.image2')
OBS = {'observation.state': (9,), **{key: (3, 512, 512) for key in CAMERAS}}
# Rows spanning both edges of an episode and one deep in training set.
PROCESSOR_ROWS = [0, 60, 107, 347, 478, 528, 47*528, 47*528+478]
sha256, write_json = h.file_hash, h.write_json

# Explicit consumed helpers; local-varied provenance separately binds expert/physical assets.
# Do not call old recursive experiment scans or import an X-VLA adapter.
SOURCE_FILES = (
    'training/smolvla_pilot.py', 'training/pilot_protocol.py',
    'experiments/single_task/contract.py',
    'data_collection/benchmark.py', 'data_collection/pilot_integrity.py', 'data_collection/pilot_plan.py',
    'validation/precision.py', 'validation/smolvla_smoke.py', 'validation/gates.py',
    'validation/evaluate_pilot.py', 'validation/rollout_diagnostics.py', 'validation/validate_pilot.py',
    'src/controllers/grasp_controller.py', 'src/robots/franka_panda/config.py',
    'experiments/local_varied_data/__init__.py', 'experiments/local_varied_data/plan.py',
    'experiments/local_varied_data/record.py', 'experiments/local_varied_data/scenes.py',
    'experiments/local_varied_data/validate.py', 'experiments/local_varied_data/README.md',
    'experiments/local_position_probe/scenes.py', 'experiments/local_position_probe/contract.py',
    'experiments/smolvla_local_varied/__init__.py', 'experiments/smolvla_local_varied/contract.py',
    'experiments/smolvla_local_varied/runner.py',
    'experiments/smolvla_local_varied/tests/test_contract.py',
    'experiments/smolvla_local_varied/tests/test_runner.py',
    'experiments/smolvla_local_varied/README.md',
) + tuple('installed/lerobot/'+name for name in (
    'policies/smolvla/modeling_smolvla.py', 'policies/smolvla/smolvlm_with_expert.py',
    'policies/smolvla/configuration_smolvla.py', 'policies/smolvla/processor_smolvla.py',
    'policies/rtc/modeling_rtc.py', 'policies/rtc/configuration_rtc.py', 'policies/rtc/debug_tracker.py',
    'processor/__init__.py', 'processor/batch_processor.py', 'processor/converters.py',
    'processor/core.py', 'processor/delta_action_processor.py', 'processor/device_processor.py',
    'processor/env_processor.py', 'processor/factory.py', 'processor/gym_action_processor.py',
    'processor/hil_processor.py', 'processor/migrate_policy_normalization.py',
    'processor/normalize_processor.py', 'processor/observation_processor.py', 'processor/pipeline.py',
    'processor/policy_robot_bridge.py', 'processor/rename_processor.py', 'processor/tokenizer_processor.py',
    'policies/pretrained.py', 'policies/utils.py', 'policies/factory.py', 'optim/optimizers.py', 'optim/schedulers.py',
    'configs/policies.py', 'configs/types.py', 'datasets/lerobot_dataset.py',
    'datasets/utils.py', 'datasets/video_utils.py', 'utils/constants.py', 'utils/utils.py',
)) + tuple('installed/transformers/'+name for name in (
    'models/smolvlm/__init__.py', 'models/smolvlm/configuration_smolvlm.py',
    'models/smolvlm/modeling_smolvlm.py', 'models/smolvlm/processing_smolvlm.py',
    'models/smolvlm/image_processing_smolvlm.py', 'models/smolvlm/image_processing_smolvlm_fast.py',
    'models/smolvlm/video_processing_smolvlm.py',
    'models/llama/configuration_llama.py', 'models/llama/modeling_llama.py',
    'models/gpt2/tokenization_gpt2.py', 'models/gpt2/tokenization_gpt2_fast.py',
    'models/auto/configuration_auto.py', 'models/auto/modeling_auto.py', 'models/auto/auto_factory.py',
    'models/auto/processing_auto.py', 'models/auto/tokenization_auto.py', 'models/auto/image_processing_auto.py',
    'configuration_utils.py', 'modeling_utils.py', 'processing_utils.py',
    'tokenization_utils.py', 'tokenization_utils_base.py', 'tokenization_utils_fast.py',
))


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def inventory(directory):
    directory = Path(directory)
    if not directory.is_dir() or any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError('Expected immutable real directory')
    paths = sorted(directory.rglob('*'))
    if any(p.is_symlink() for p in paths): raise ValueError('Symlink artifact forbidden')
    return {str(p.relative_to(directory)): sha256(p) for p in paths if p.is_file()}


def source_hashes():
    import lerobot
    import transformers
    installed = {'lerobot': Path(lerobot.__file__).parent, 'transformers': Path(transformers.__file__).parent}
    result = {}
    for name in SOURCE_FILES:
        if name.startswith('installed/'):
            _, package, relative = name.split('/', 2)
            path = installed[package]/relative
        else:
            path = ROOT/name
        result[name] = sha256(path)
    return result


def require_assets():
    expected = {'policy': dict(repo_id='lerobot/smolvla_base', revision=REVISION, path=str(CACHE/'policy')),
                'vlm': dict(repo_id='HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
                            revision=VLM_REVISION, path=str(CACHE/'vlm'))}
    manifest = json.loads((CACHE/'manifest.json').read_text())
    files = inventory(CACHE)
    required = {'manifest.json', 'policy/config.json', 'policy/model.safetensors',
                'vlm/config.json', 'vlm/tokenizer.json', 'vlm/tokenizer_config.json', 'vlm/preprocessor_config.json'}
    if manifest != expected or not required <= files.keys() or digest_json(files) != CACHE_DIGEST:
        raise ValueError('Pinned complete SmolVLA manifest/cache bytes changed')
    return dict(manifest=manifest, files=files, inventory_digest=digest_json(files))


def is_mountpoint(path):
    # Same-filesystem Docker bind mounts are not reliably detected by os.path.ismount.
    return str(path) in {line.split()[4] for line in Path('/proc/self/mountinfo').read_text().splitlines()}


def claim_output(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)): raise ValueError('Symlink output forbidden')
    if path.exists():
        if not path.is_dir() or not is_mountpoint(path) or any(path.iterdir()):
            raise FileExistsError('Preserve existing output; no overwrite/resume/retry')
    else:
        path.mkdir(parents=True, exist_ok=False)
    with (path/'run_claim.json').open('x') as stream:
        json.dump(dict(purpose='smolvla_local_varied', required_image=IMAGE, resume=False), stream)


def require_fields(actual, expected, message):
    if any(not _same(actual.get(key), value) for key, value in expected.items()): raise ValueError(message)


def indices(episodes):
    if len(episodes) != 60: raise ValueError('Exactly 60 local-varied episodes; old fixed/pilot metadata forbidden')
    for i in TRAIN+VAL:
        require_fields(episodes[i], dict(episode_index=i, length=528, tasks=[TASK],
                       dataset_from_index=i*528, dataset_to_index=(i+1)*528), 'Local-varied episode identity/boundary mismatch')
    return list(range(TRAINING_ROWS)), [i*528+j for i in TRAIN for j in range(479)]


def selected_stats(rows, selected):
    if selected != list(range(TRAINING_ROWS)): raise ValueError('Statistics require every training row exactly once')
    for i in selected:
        row = rows[i]
        if int(row['episode_index']) != i//528 or int(row['task_index']) != 0:
            raise ValueError('Raw training membership/task mismatch')
        for key, size in (('observation.state', 9), ('action', 8)):
            value = torch.as_tensor(row[key])
            if value.shape != (size,) or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('Finite unpadded real FP32 state9/action8 required')
    return h.compute_training_stats(rows, selected)


def observation(row):
    if row.get('task') != TASK or {k for k in row if k.startswith('observation.')} != set(OBS):
        raise ValueError('Unapproved instruction/privileged observation')
    for key, shape in OBS.items():
        value = row.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Real finite FP32 observation schema required')
        if key in CAMERAS and (value.min() < 0 or value.max() > 1): raise ValueError('Image outside [0,1]')
    return {k: row[k] for k in (*OBS, 'task')}


def policy_row(row, training):
    episode = row.get('episode_index')
    if isinstance(episode, torch.Tensor):
        if episode.ndim != 0 or episode.dtype != torch.int64: raise ValueError('Integer episode scalar required')
        episode = episode.item()
    if type(episode) is not int or episode not in (TRAIN if training else TRAIN+VAL):
        raise ValueError('Unapproved training/probe episode')
    obs = observation(row)
    action, pad = row.get('action'), row.get('action_is_pad')
    if (not isinstance(action, torch.Tensor) or action.shape != (50, 8) or action.dtype != torch.float32
            or not torch.isfinite(action).all() or not isinstance(pad, torch.Tensor)
            or pad.shape != (50,) or pad.dtype != torch.bool or pad.any()):
        raise ValueError('Complete finite50x8 physical actions required')
    return {**obs, 'action': action, 'action_is_pad': pad}


class PolicyRows:
    def __init__(self, dataset): self.dataset = dataset
    def __getitem__(self, index): return policy_row(self.dataset[index], training=True)


def collate_rows(rows):
    if len(rows) != 8: raise ValueError('Exactly8 samples per microbatch required')
    batch = default_collate([policy_row(row, training=True) for row in rows])
    h.check_batch(batch)
    return batch


def make_config(device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    config = PreTrainedConfig.from_pretrained(CACHE/'policy', local_files_only=True)
    if (config.type, config.chunk_size, config.n_obs_steps, config.max_state_dim, config.max_action_dim,
            config.expert_width_multiplier, config.tokenizer_max_length) != ('smolvla', 50, 1, 32, 32, .75, 48):
        raise ValueError('Unexpected cached SmolVLA architecture/temporal schema')
    config.device, config.vlm_model_name = device, str(CACHE/'vlm')
    config.load_vlm_weights = config.push_to_hub = False
    config.train_expert_only = config.train_state_proj = True
    config.n_action_steps = 10
    config.input_features = {k: PolicyFeature(type=FeatureType.STATE if k=='observation.state' else FeatureType.VISUAL,
                                             shape=shape) for k, shape in OBS.items()}
    config.output_features = {'action': PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    return config


def load_data():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    info = json.loads((DATA/'meta/info.json').read_text())
    require_fields(info, dict(total_episodes=60, total_frames=TOTAL_FRAMES, total_tasks=1, fps=30,
                   splits={'train': '0:48', 'validation': '48:60'}), 'Local-varied dataset metadata mismatch')
    ds = LeRobotDataset(repo_id=DATA.name, root=DATA, delta_timestamps={'action': [i/30 for i in range(50)]})
    features = dataset_to_policy_features(ds.features)
    if ({k: tuple(v.shape) for k, v in features.items() if k.startswith('observation.')} != OBS
            or tuple(features['action'].shape) != (8,) or len(ds) != TOTAL_FRAMES
            or list(ds.meta.tasks.index) != [TASK] or int(ds.meta.tasks.loc[TASK, 'task_index']) != 0):
        raise ValueError('Real dataset state/action/task schema mismatch')
    rows, starts = indices(ds.meta.episodes)
    ds._ensure_hf_dataset_loaded()
    for i in range(TOTAL_FRAMES):
        if int(ds.hf_dataset[i]['episode_index']) != i//528 or int(ds.hf_dataset[i]['task_index']) != 0:
            raise ValueError('Raw episode/task membership mismatch')
    return ds, starts, selected_stats(ds.hf_dataset, rows)


def statistics_json(stats):
    return {key: {name: value.tolist() for name, value in values.items()} for key, values in stats.items()}


def current_binding():
    from experiments.local_varied_data.scenes import provenance
    if json.loads((PARENT/'seal.json').read_text()) != {'fingerprint': SEAL}:
        raise ValueError('Approved local-varied parent fingerprint changed')
    plan = json.loads((PARENT/'plan.json').read_text()); validate_plan(plan)
    semantic = json.loads((PARENT/'validation.json').read_text())
    current = provenance()
    if (semantic.get('status') != 'passed' or semantic.get('training_ready') is not True
            or semantic.get('automatic_training') is not False or semantic.get('provenance') != current
            or semantic.get('train_ids') != list(TRAIN) or semantic.get('validation_ids') != list(VAL)
            or semantic.get('statistics_episode_ids') != list(TRAIN)):
        raise ValueError('Local-varied provenance/semantic proof changed')
    return dict(dataset_fingerprint=SEAL, seal_sha256=sha256(PARENT/'seal.json'),
                semantic_sha256=digest_json(semantic), local_varied_provenance=current,
                source_reset_sha256=sha256(PARENT/'source.npz'), plan_sha256=sha256(PARENT/'plan.json'),
                snapshots_sha256=sha256(PARENT/'snapshots.json'), sources=source_hashes(),
                assets=require_assets(), required_image=IMAGE)


def cpu_proof(binding, semantic, stats, processor):
    return dict(status='passed', gpu_executed=False, optimizer_updates=0, binding=binding,
                semantic_gate=semantic, training_rows=TRAINING_ROWS, complete_training_chunks=COMPLETE_STARTS,
                train_episodes=list(TRAIN), validation_episodes=list(VAL), training_stats=stats,
                training_stats_digest=digest_json(stats), processor_check=processor,
                processor_digest=digest_json(processor))


def validate_cpu(report, binding):
    require_fields(report, dict(status='passed', gpu_executed=False, optimizer_updates=0, binding=binding,
                   training_rows=TRAINING_ROWS, complete_training_chunks=COMPLETE_STARTS, train_episodes=list(TRAIN),
                   validation_episodes=list(VAL)), 'Current full CPU admission required')
    if (digest_json(report.get('semantic_gate')) != binding.get('semantic_sha256')
            or not report.get('training_stats') or not report.get('processor_check')
            or digest_json(report['training_stats']) != report.get('training_stats_digest')
            or digest_json(report['processor_check']) != report.get('processor_digest')):
        raise ValueError('CPU semantic/statistics/processor proof changed')


def admission():
    binding = current_binding()
    validate_cpu(json.loads(CPU_ADMISSION.read_text()), binding)
    return binding


def prepare_data():
    binding = admission()
    ds, starts, stats = load_data()
    if statistics_json(stats) != json.loads(CPU_ADMISSION.read_text())['training_stats']:
        raise ValueError('Fresh training statistics differ from admitted train-only statistics')
    return ds, starts, stats, binding


def recipe(preflight):
    return dict(purpose='smolvla_local_varied_preflight' if preflight else 'smolvla_local_varied',
                seed=SEED, max_optimizer_updates=2 if preflight else 5000,
                microbatch=8, accumulation=4, effective_batch=32, chunk_size=50, n_action_steps=10,
                precision='fp32_trainable_bf16_training_autocast', loss='unchanged_stock_flow_matching',
                total_parameters=450046176, trainable_parameters=99880992,
                optimizer=dict(betas=[.9, .95], eps=1e-8, weight_decay=1e-10, clip=10.),
                schedule=dict(warmup=500, total=5000, peak=1e-4, floor=1e-5),
                train_episodes=list(TRAIN), validation_episodes=list(VAL),
                resume=False, autotune=False, retries=False)


def checkpoint_updates(preflight):
    return [1, 2] if preflight else [1, 1000, 2000, 3000, 4000, 5000]


def checkpoint_directory(update):
    if type(update) is not int or update < 1: raise ValueError('Positive integer update required')
    return f'step_{update:06d}'


def require_marker(marker, update):
    require_fields(marker, dict(status='complete', update=update, reload_fp32_weights_exact=True), 'Strict FP32 checkpoint marker required')
    error = marker.get('reload_max_action_error')
    if type(error) not in (int, float) or not 0 <= error <= 1e-5:
        raise ValueError('Absolute reload action error must be finite and <=1e-5')


def verify_checkpoints(report, output, preflight):
    updates = checkpoint_updates(preflight)
    keys = [checkpoint_directory(i) for i in updates]
    if [m.get('update') for m in report.get('checkpoints', [])] != updates:
        raise ValueError('Checkpoint schedule changed')
    if set(report.get('checkpoint_files', {})) != set(keys):
        raise ValueError('All checkpoint inventories required')
    present = {p.name for p in (output/'checkpoints').iterdir()}
    # Legacy `{update:06d}` naming is a regression; keys carry the required step_ prefix.
    if present & {f'{i:06d}' for i in updates}:
        raise ValueError('Legacy checkpoint directory naming forbidden; expected step_ prefix')
    if present != set(keys):
        raise ValueError('Unlisted/missing checkpoint')
    required = {'model.safetensors', 'config.json', 'training_state.pt', 'checkpoint_result.json',
                'policy_preprocessor.json', 'policy_postprocessor.json'}
    for marker in report['checkpoints']:
        update = marker['update']; require_marker(marker, update)
        key = checkpoint_directory(update)
        directory = output/'checkpoints'/key
        files = inventory(directory)
        if (files != report['checkpoint_files'][key] or not required <= files.keys()
                or files['model.safetensors'] != marker.get('model_sha256')
                or json.loads((directory/'checkpoint_result.json').read_text()) != marker):
            raise ValueError('Checkpoint/processor/training-state byte inventory changed')
        if preflight:
            if 'adam_reload.json' not in files: raise ValueError('Scratch Adam roundtrip evidence missing')
            evidence = json.loads((directory/'adam_reload.json').read_text())
            require_fields(evidence, dict(adam_reload_fp32_exact=True, update=update,
                           training_state_sha256=files['training_state.pt']), 'Scratch Adam proof changed')


def validate_scratch(report, binding, output, stats, cpu_hash):
    require_fields(report, {**recipe(True), 'status': 'complete', 'completed_updates': 2, 'binding': binding,
                   'training_rows': TRAINING_ROWS, 'complete_training_chunks': COMPLETE_STARTS,
                   'cpu_admission_sha256': cpu_hash, 'final_data_binding_passed': True,
                   'adam_reload_fp32_exact': True, 'selected_checkpoint_update': None, 'rollouts': 0},
                   'Exact current two-update scratch proof required')
    if (report.get('training_stats_sha256') != sha256(output/'training_stats.json')
            or json.loads((output/'training_stats.json').read_text()) != stats):
        raise ValueError('Scratch training statistics changed')
    verify_checkpoints(report, output, True)
