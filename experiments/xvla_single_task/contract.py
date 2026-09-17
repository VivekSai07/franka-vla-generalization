"""Fixed approved contract. Never call the old recursive source_hashes()."""
import json
import math
from pathlib import Path

import torch
from torch.utils.data import default_collate

from experiments.xvla_trial import contract as old, preflight as p
from experiments.xvla_trial.assets import claim_directory, require_assets, sha256, write_json

ROOT, DATA, CACHE = old.ROOT, old.DATA, old.CACHE
TRAIN, VAL, TASK, OBS, CAMERAS, SEED = old.TRAIN, old.VAL, old.TASK, old.OBS, old.CAMERAS, old.SEED
PACKAGE = ROOT/'experiments/xvla_single_task'
OUTPUT = ROOT/'outputs/train/xvla_single_task_20260918'
SCRATCH = OUTPUT.with_name(OUTPUT.name+'_preflight')
CPU_ADMISSION = PACKAGE/'CPU_ADMISSION_20260918.json'
OLD_RESULT = old.OUTPUT/'result.json'
OLD_CPU = p.CPU_REPORT
PRIOR_FILES = {
    'outputs/validation/xvla_preflight_20260917/result.json': 'e27eea195717d50d7f242f3445588e452826f720173ad39b7daeb9ae3d09080a',
    'experiments/xvla_trial/CPU_REPORT_REVIEWED.json': '967b576116f144ca3620200aeab37586386a1c8b3f0d8fbe1a887121cf313fff',
}
NEW_FILES = tuple('experiments/xvla_single_task/'+name for name in
                  ('contract.py','runner.py','evaluate.py','tests/test_contract.py','PLAN.md'))
# Additional directly-used installed code, without a recursive experiments scan.
INSTALLED_FILES = ('datasets/lerobot_dataset.py','datasets/utils.py','datasets/video_utils.py',
                   'policies/factory.py','configs/policies.py','configs/types.py')


def source_path(name, root=ROOT):
    if name.startswith('installed/lerobot/'):
        import lerobot
        root, name = Path(lerobot.__file__).parent, name.removeprefix('installed/lerobot/')
    root = Path(root).resolve()
    relative = Path(name)
    path = root/relative
    if (relative.is_absolute() or '..' in relative.parts or not path.is_file()
            or any(x.is_symlink() for x in (path,*path.parents)) or root not in path.resolve().parents):
        raise ValueError('Bound file missing, symlinked, or outside allowed root: '+name)
    return path


def verify_listed(recorded, root=ROOT):
    if not recorded:
        raise ValueError('Empty source binding')
    for name, digest in recorded.items():
        if sha256(source_path(name,root)) != digest:
            raise ValueError('Recorded listed file changed: '+name)
    return dict(recorded)


def validate_prior(report):
    expected = dict(status='complete',purpose='xvla_two_update_preflight',seed=SEED,
                    max_optimizer_updates=2,completed_updates=2,microbatch=1,accumulation=1,
                    strict_reload_fp32_exact=True,adam_reload_fp32_exact=True,
                    reload_max_absolute_error=0.,final_data_binding_passed=True,autotune=False,resume=False)
    if any(report.get(k) != v for k,v in expected.items()):
        raise ValueError('Original exact completed micro1 preflight evidence required')


def admission(gate=None):
    from validation.gates import require_replay_pass
    from data_collection.benchmark import require_complete
    verify_listed(PRIOR_FILES)
    prior, cpu = json.loads(OLD_RESULT.read_text()), json.loads(OLD_CPU.read_text())
    validate_prior(prior)
    verify_listed(prior['source_hashes'])
    verify_listed(cpu['source_hashes'])
    if (cpu['status'] != 'cpu_compatible_gpu_unmeasured' or not cpu['strict_full_checkpoint_load']
            or cpu['parameter_count'] != 879482456 or cpu['source_hashes'] != prior['source_hashes']):
        raise ValueError('Prior CPU/model binding mismatch')
    assets = require_assets()
    manifest = require_complete(DATA)
    old.select_entries(manifest)
    gate = require_replay_pass(DATA,ROOT) if gate is None else gate
    if (assets != prior['assets'] or assets != cpu['assets']
            or gate['dataset_fingerprint'] != prior['dataset_fingerprint']
            or gate['dataset_fingerprint'] != cpu['dataset_fingerprint']):
        raise ValueError('Prior full data/cache binding changed')
    for name,key in (('model.safetensors','checkpoint_sha256'),('optimizer.pt','optimizer_sha256')):
        if sha256(old.OUTPUT/'checkpoint'/name) != prior[key]:
            raise ValueError('Original proof checkpoint bytes changed')
    explicit = NEW_FILES + tuple('installed/lerobot/'+x for x in INSTALLED_FILES)
    sources = {name:sha256(source_path(name)) for name in explicit}
    return dict(prior_reports=PRIOR_FILES,prior_sources=prior['source_hashes'],new_sources=sources,
                assets=assets,dataset_gate=gate,timing=manifest['timing'])


def inventory(directory):
    directory = Path(directory)
    if any(path.is_symlink() for path in (directory,*directory.parents)):
        raise ValueError('Artifact directory symlinks forbidden')
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Artifact symlinks forbidden')
        if path.is_file():
            result[str(path.relative_to(directory))] = sha256(path)
    if not result:
        raise ValueError('Empty artifact inventory')
    return result


def recipe(preflight):
    return dict(purpose='xvla_single_task_preflight' if preflight else 'xvla_single_task',
                seed=SEED,max_optimizer_updates=2 if preflight else 5000,
                microbatch=4,accumulation=2,effective_batch=8,chunk_size=30,n_action_steps=10,
                precision='fp32_parameters_gradients_adam_bf16_autocast',
                optimizer=dict(peak_lr=1e-4,vlm_scale=.1,soft_prompt_scale=1.,weight_decay=1e-4,
                               vlm_weight_decay=1e-5,betas=[.9,.95],eps=1e-8,clip=10.),
                schedule=dict(warmup=500,total=5000,floor_fraction=.1),
                train_episodes=list(TRAIN),validation_episodes=list(VAL),
                optimizer_storage='final_only',resume=False,autotune=False,retries=False)


def validate_cpu(report,binding):
    expected = dict(status='passed',gpu_executed=False,optimizer_updates=0,binding=binding)
    if any(report.get(k) != v for k,v in expected.items()):
        raise ValueError('Current source/data/cache must match the saved CPU admission')


def validate_scratch(report,binding,files,stats_hash):
    expected = {**recipe(True),'status':'complete','completed_updates':2,'binding':binding,
                'final_data_binding_passed':True,'strict_reload_fp32_exact':True,
                'adam_reload_fp32_exact':True,'reload_max_absolute_error':0.,
                'final_checkpoint_files':files,'training_stats_sha256':stats_hash}
    if (any(report.get(k) != v for k,v in expected.items())
            or [r['update'] for r in report.get('checkpoints',[])] != [1,2]):
        raise ValueError('Matching exact micro4/accum2 two-update scratch proof required; never resumed')


def learning_rate(update):
    if type(update) is not int or not 1 <= update <= 5000:
        raise ValueError('Update must be integer in [1,5000]')
    return (1e-4*update/500 if update <= 500 else
            1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*(update-500)/4500)))


def make_config(device):
    config = p.make_config(device)
    config.optimizer_lr,config.optimizer_betas = 1e-4,(.9,.95)
    config.optimizer_weight_decay,config.optimizer_eps,config.optimizer_grad_clip_norm = 1e-4,1e-8,10.
    config.optimizer_soft_prompt_lr_scale,config.optimizer_soft_prompt_warmup_lr_scale = 1.,None
    config.scheduler_warmup_steps,config.scheduler_decay_steps,config.scheduler_decay_lr = 500,5000,1e-5
    if (config.normalization_mapping['STATE'].value != 'IDENTITY'
            or config.normalization_mapping['ACTION'].value != 'MEAN_STD'
            or config.normalization_mapping['VISUAL'].value != 'IDENTITY'
            or not config.use_proprio or config.n_obs_steps != 1):
        raise ValueError('Native normalization/proprio contract changed')
    return config


def set_learning_rate(optimizer,update):
    lr = learning_rate(update)
    if [g['name'] for g in optimizer.param_groups] != ['vlm','soft_prompts','other']:
        raise ValueError('Expected all three native named optimizer groups')
    for group in optimizer.param_groups:
        scale = .1 if group['name'] == 'vlm' else 1.
        if (not math.isclose(group['weight_decay'],1e-4*scale,rel_tol=0,abs_tol=1e-15)
                or tuple(group['betas']) != (.9,.95) or group['eps'] != 1e-8):
            raise ValueError('Native optimizer recipe changed')
        group['lr'] = lr*scale


def collate_rows(rows):
    if len(rows) != 4:
        raise ValueError('Exact four-row microbatch required')
    batch = default_collate([p.policy_row(row,training=True) for row in rows])
    if batch['action'].shape != (4,30,8):
        raise ValueError('Expected B4xT30xA8, not AddBatchDimension on a bare chunk')
    return batch


def observation(frame):
    if set(frame) != set(OBS)|{'task'} or frame['task'] != TASK:
        raise ValueError('Inference accepts only real observations and the approved instruction')
    for key,shape in OBS.items():
        value = frame[key]
        if value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Invalid physical observation')
        if key in CAMERAS and (value.min() < 0 or value.max() > 1):
            raise ValueError('RGB must be float32 [0,1], without LIBERO rotation')
    return dict(frame)


def check_model_inputs(policy,batch,raw_state):
    inputs = policy._build_model_inputs(batch)
    size = raw_state.shape[0]
    expected_mask = torch.tensor([[True,True,False]]*size,device=inputs['image_mask'].device)
    if (inputs['image_input'].shape != (size,3,3,224,224)
            or inputs['proprio'].shape != (size,20) or inputs['input_ids'].shape != (size,64)
            or not torch.equal(inputs['image_mask'],expected_mask)
            or torch.count_nonzero(inputs['image_input'][:,2]) != 0
            or torch.count_nonzero(inputs['domain_id']) != 0
            or not torch.equal(inputs['proprio'][:,:9].cpu(),raw_state.cpu())
            or torch.count_nonzero(inputs['proprio'][:,9:]) != 0
            or any(not torch.isfinite(x).all() for x in inputs.values())):
        raise ValueError('Model image/proprio/token/domain precheck failed')
    if batch.get('action') is not None and batch['action'].shape != (size,30,8):
        raise ValueError('Real action batch shape changed')
    return dict(image_shape=list(inputs['image_input'].shape),image_mask=inputs['image_mask'].tolist(),
                proprio_shape=list(inputs['proprio'].shape),real_state_dim=9,real_action_dim=8,
                state_identity_exact=True,domain=0,token_shape=list(inputs['input_ids'].shape),
                privileged_inputs=False,libero_rotation=False)


def backward_microbatch(loss):
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError('Finite scalar native loss required')
    (loss/2).backward()


def sample_batches(starts):
    # CPU generator isolated from model/noise RNG; complete train-only chunks, drop final3.
    if len(starts) != 4491 or len(set(starts)) != 4491 or any(
            i//528 not in TRAIN or not 0 <= i%528 <= 498 for i in starts):
        raise ValueError('Expected exactly 4491 approved training chunks')
    generator = torch.Generator().manual_seed(SEED)
    while True:
        order = torch.randperm(len(starts),generator=generator).tolist()
        for offset in range(0,len(order)-3,4):
            yield [starts[j] for j in order[offset:offset+4]]