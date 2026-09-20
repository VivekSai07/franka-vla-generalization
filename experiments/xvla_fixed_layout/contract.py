"""Immutable split, train-only data adapter and exact phase admission bindings."""
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate

from experiments.fixed_layout.contract import TASK, IMAGE, validate_plan, _same
from experiments.xvla_trial import preflight as p
from experiments.xvla_trial.assets import claim_directory, require_assets, sha256, write_json
from experiments.xvla_single_task.contract import (
    make_config, learning_rate, set_learning_rate, backward_microbatch,
    observation, check_model_inputs, inventory, source_path, verify_listed,
)

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'outputs/fixed_layout/fixed_layout_20260918'
DATA = PARENT/'dataset'
CACHE = ROOT/'.cache/xvla_trial/bundle'
OUTPUT = ROOT/'outputs/train/xvla_fixed_layout_20260918'
SCRATCH = OUTPUT.with_name(OUTPUT.name+'_preflight')
ADMISSION = OUTPUT.with_name(OUTPUT.name+'_admission')
CPU_ADMISSION = ADMISSION/'result.json'
SEAL = '403b8ec2d16094863faf611a13768e770dbf19a363c477f0f509fb67d8d8fba5'
TRAIN, VAL, SEED = tuple(range(20)), tuple(range(20,24)), 20260917
CAMERAS = ('observation.images.image','observation.images.image2')
OBS = {'observation.state':(9,),**{key:(3,512,512) for key in CAMERAS}}

# Explicit individual imports, NOT old reports or a recursive experiments scan.
# Fixed-layout provenance separately binds its own package, physical assets and expert.
SOURCE_FILES = (
    'experiments/xvla_trial/contract.py','experiments/xvla_trial/preflight.py',
    'experiments/xvla_trial/assets.py','experiments/xvla_single_task/contract.py',
    'data_collection/benchmark.py','data_collection/pilot_integrity.py','data_collection/pilot_plan.py',
    'validation/evaluate_pilot.py','validation/gates.py','validation/rollout_diagnostics.py',
    'validation/validate_pilot.py','validation/precision.py','validation/smolvla_smoke.py',
    'training/smolvla_pilot.py','training/pilot_protocol.py',
    'src/controllers/grasp_controller.py','src/robots/franka_panda/config.py',
    'experiments/xvla_fixed_layout/__init__.py','experiments/xvla_fixed_layout/contract.py',
    'experiments/xvla_fixed_layout/runner.py','experiments/xvla_fixed_layout/evaluate.py',
    'experiments/xvla_fixed_layout/tests/test_contract.py',
    'experiments/xvla_fixed_layout/tests/test_runner.py','experiments/xvla_fixed_layout/tests/test_evaluate.py',
    'experiments/xvla_fixed_layout/README.md',
    'docs/superpowers/specs/2026-09-18-xvla-fixed-layout-training-design.md',
) + tuple('installed/lerobot/'+name for name in (
    'policies/xvla/__init__.py','policies/xvla/action_hub.py','policies/xvla/configuration_florence2.py',
    'policies/xvla/configuration_xvla.py','policies/xvla/modeling_florence2.py',
    'policies/xvla/modeling_xvla.py','policies/xvla/processor_xvla.py',
    'policies/xvla/soft_transformer.py','policies/xvla/utils.py',
    'processor/__init__.py','processor/batch_processor.py','processor/converters.py',
    'processor/core.py','processor/delta_action_processor.py','processor/device_processor.py',
    'processor/env_processor.py','processor/factory.py','processor/gym_action_processor.py',
    'processor/hil_processor.py','processor/migrate_policy_normalization.py',
    'processor/normalize_processor.py','processor/observation_processor.py','processor/pipeline.py',
    'processor/policy_robot_bridge.py','processor/rename_processor.py','processor/tokenizer_processor.py',
    'policies/pretrained.py','policies/utils.py','policies/factory.py','optim/optimizers.py',
    'configs/policies.py','configs/types.py','datasets/lerobot_dataset.py',
    'datasets/utils.py','datasets/video_utils.py','utils/constants.py',
))


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {name:sha256(source_path(name)) for name in SOURCE_FILES}


def is_mountpoint(path):
    # os.path.ismount misses same-filesystem bind mounts. These fixed paths have no escapes.
    return str(path) in {line.split()[4] for line in Path('/proc/self/mountinfo').read_text().splitlines()}


def claim_output(path):
    """Exclusive token permits a newly created empty Docker output bind mount only."""
    path=Path(path).absolute()
    if any(item.is_symlink() for item in (path,*path.parents)):
        raise ValueError('Symlink output forbidden')
    if path.exists():
        if not path.is_dir() or not is_mountpoint(path) or any(path.iterdir()):
            raise FileExistsError('Preserve existing output; no overwrite/resume/retry')
    else:
        claim_directory(path)
    with (path/'run_claim.json').open('x') as stream:
        json.dump({'purpose':'xvla_fixed_layout','resume':False,'required_image':IMAGE},stream)


def require_fields(actual,expected,message):
    if any(not _same(actual.get(key),value) for key,value in expected.items()):
        raise ValueError(message)


def indices(episodes):
    if len(episodes) != 24:
        raise ValueError('Exactly 24 fixed-layout episodes, not pilot metadata')
    for i in TRAIN+VAL:
        require_fields(episodes[i],dict(episode_index=i,length=528,tasks=[TASK],
                       dataset_from_index=i*528,dataset_to_index=(i+1)*528),
                       'Fixed episode identity/task/length/boundary mismatch')
    return list(range(20*528)),[i*528+j for i in TRAIN for j in range(499)]


def selected_stats(rows,indices):
    if indices != list(range(10560)):
        raise ValueError('Statistics require exactly every row of train0..19 once')
    values={key:[] for key in ('observation.state','action')}
    for i in indices:
        row=rows[i]
        if int(row['episode_index']) != i//528 or int(row['task_index']) != 0:
            raise ValueError('Wrong training row/task identity')
        for key,size in (('observation.state',9),('action',8)):
            value=torch.as_tensor(row[key])
            if value.dtype != torch.float32 or value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError('Raw training values must be finite unpadded FP32')
            values[key].append(value)
    return {key:{'mean':(x:=torch.stack(value)).mean(0),'std':x.std(0,unbiased=False),
                 'min':x.amin(0),'max':x.amax(0)} for key,value in values.items()}


def policy_row(row,training):
    ids=TRAIN if training else TRAIN+VAL
    episode=row.get('episode_index')
    if isinstance(episode,torch.Tensor):
        if episode.ndim != 0 or episode.dtype != torch.int64:
            raise ValueError('Episode identity must be a scalar integer')
        episode=episode.item()
    if (type(episode) is not int or episode not in ids or row.get('task') != TASK
            or {key for key in row if key.startswith('observation.')} != set(OBS)):
        raise ValueError('Unapproved episode/instruction/observation')
    observation({key:row[key] for key in (*OBS,'task')})
    action,pad=row.get('action'),row.get('action_is_pad')
    if (not isinstance(action,torch.Tensor) or action.shape != (30,8)
            or action.dtype != torch.float32 or not torch.isfinite(action).all()
            or not isinstance(pad,torch.Tensor) or pad.shape != (30,)
            or pad.dtype != torch.bool or pad.any()):
        raise ValueError('Finite real30x8 actions and complete unpadded chunk required')
    return {key:row[key] for key in (*OBS,'task','action','action_is_pad')}


def collate_rows(rows):
    if len(rows) != 4:
        raise ValueError('Exactly four training rows per microbatch')
    return default_collate([policy_row(row,True) for row in rows])


def sample_batches(starts):
    expected=[i*528+j for i in TRAIN for j in range(499)]
    if starts != expected:
        raise ValueError('Exactly 9980 complete ordered training starts required')
    generator=torch.Generator().manual_seed(SEED)
    while True:
        order=torch.randperm(len(starts),generator=generator).tolist()
        for offset in range(0,len(order),4):
            yield [starts[j] for j in order[offset:offset+4]]


def current_binding():
    """Rehash full parent + actual asset bytes + current provenance; no semantic rerun."""
    from experiments.fixed_layout.source import provenance
    from experiments.fixed_layout.validate import fingerprint
    seal=json.loads((PARENT/'seal.json').read_text())
    if seal != {'fingerprint':SEAL} or fingerprint(PARENT) != SEAL:
        raise ValueError('Approved complete parent seal/fingerprint changed')
    validate_plan(json.loads((PARENT/'plan.json').read_text()))
    semantic=json.loads((PARENT/'validation.json').read_text())
    current=provenance()
    if (semantic.get('status') != 'passed' or semantic.get('training_ready') is not True
            or semantic.get('provenance') != current):
        raise ValueError('Fixed-layout current provenance/semantic evidence mismatch')
    return dict(dataset_fingerprint=SEAL,seal_sha256=sha256(PARENT/'seal.json'),
                semantic_sha256=digest_json(semantic),fixed_provenance=current,
                reset_sha256=sha256(PARENT/'reset.npz'),sources=source_hashes(),
                assets=require_assets(),required_image=IMAGE)


def cpu_proof(binding,semantic,stats,processor):
    return dict(status='passed',gpu_executed=False,optimizer_updates=0,binding=binding,
                semantic_gate=semantic,training_rows=10560,complete_training_chunks=9980,
                train_episodes=list(TRAIN),validation_episodes=list(VAL),
                training_stats=stats,training_stats_digest=digest_json(stats),
                processor_check=processor,processor_digest=digest_json(processor))


def validate_cpu(report,binding):
    require_fields(report,dict(status='passed',gpu_executed=False,optimizer_updates=0,binding=binding,
                   training_rows=10560,complete_training_chunks=9980,
                   train_episodes=list(TRAIN),validation_episodes=list(VAL)),
                   'Current fixed-layout CPU admission required')
    if (digest_json(report.get('semantic_gate')) != binding.get('semantic_sha256')
            or not report.get('training_stats') or not report.get('processor_check')
            or digest_json(report['training_stats']) != report.get('training_stats_digest')
            or digest_json(report['processor_check']) != report.get('processor_digest')):
        raise ValueError('CPU semantic/statistics/processor proof changed')


def admission():
    binding=current_binding()
    validate_cpu(json.loads(CPU_ADMISSION.read_text()),binding)
    return binding


def load_data():
    """Dataset-only reader. Never library/global stats, pilot manifest or old adapter."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    info=json.loads((DATA/'meta/info.json').read_text())
    require_fields(info,dict(total_episodes=24,total_frames=12672,total_tasks=1,fps=30,
                   splits={'train':'0:20','validation':'20:24'}),'Fixed-layout metadata mismatch')
    ds=LeRobotDataset(repo_id=DATA.name,root=DATA,delta_timestamps={'action':[i/30 for i in range(30)]})
    features=dataset_to_policy_features(ds.features)
    if ({k:tuple(v.shape) for k,v in features.items() if k.startswith('observation.')} != OBS
            or tuple(features['action'].shape) != (8,) or len(ds) != 12672
            or list(ds.meta.tasks.index) != [TASK] or int(ds.meta.tasks.loc[TASK,'task_index']) != 0):
        raise ValueError('Real dataset schema/task/count mismatch')
    rows,starts=indices(ds.meta.episodes)
    ds._ensure_hf_dataset_loaded()
    stats=selected_stats(ds.hf_dataset,rows)
    return ds,starts,stats


def prepare_data():
    gate=admission()
    ds,starts,stats=load_data()
    encoded={key:{name:value.tolist() for name,value in values.items()} for key,values in stats.items()}
    if encoded != json.loads(CPU_ADMISSION.read_text())['training_stats']:
        raise ValueError('Fresh train-only statistics differ from current CPU admission')
    return ds,starts,stats,gate


def recipe(preflight):
    return dict(purpose='xvla_fixed_layout_preflight' if preflight else 'xvla_fixed_layout',
                seed=SEED,max_optimizer_updates=2 if preflight else 5000,
                microbatch=4,accumulation=2,effective_batch=8,chunk_size=30,n_action_steps=10,
                precision='fp32_parameters_gradients_adam_bf16_autocast',
                optimizer=dict(peak_lr=1e-4,vlm_scale=.1,soft_prompt_scale=1.,weight_decay=1e-4,
                               vlm_weight_decay=1e-5,betas=[.9,.95],eps=1e-8,clip=10.),
                schedule=dict(warmup=500,total=5000,floor_fraction=.1),
                train_episodes=list(TRAIN),validation_episodes=list(VAL),
                optimizer_storage='final_only',resume=False,autotune=False,retries=False)


def checkpoint_updates(preflight):
    return [1,2] if preflight else [1,1000,2000,3000,4000,5000]


def validate_scratch(report,binding,files,stats_hash,cpu_hash):
    require_fields(report,{**recipe(True),'status':'complete','completed_updates':2,'binding':binding,
                'training_rows':10560,'complete_training_chunks':9980,'cpu_admission_sha256':cpu_hash,
                'final_data_binding_passed':True,'strict_reload_fp32_exact':True,
                'adam_reload_fp32_exact':True,'reload_max_absolute_error':0.,
                'final_checkpoint_files':files,'training_stats_sha256':stats_hash},
                'Exact current two-update scratch/CPU/statistics/checkpoint proof required')
    if [r.get('update') for r in report.get('checkpoints',[])] != checkpoint_updates(True):
        raise ValueError('Scratch checkpoint schedule changed')