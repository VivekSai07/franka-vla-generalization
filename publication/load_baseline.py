"""Self-contained, offline CPU inference for the two fixed-layout releases.

Requires the versions recorded in runtime.json. No simulator or project imports.
Do not replace this with policy.from_pretrained: that can round FP32 trainables.
"""
import os
from pathlib import Path


def offline_mode():
    # Set before importing HF consumers; constructors only receive local paths.
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE',
                'HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key] = '1'


def bundled_path(root, value):
    root = Path(root).resolve()
    relative = Path(value)
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:1] != ('assets',):
        raise ValueError('Asset must be a release-relative assets path')
    target = root / relative
    if not target.is_dir() or not target.resolve().is_relative_to(root):
        raise ValueError('Missing or escaping bundled asset directory')
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError('Symlinked assets are not supported')
    return str(target)


def strict_load_weights(policy, filename, model_type):
    """Promote BEFORE copy, reject casting, strict-load and verify every byte."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_model

    if model_type not in ('smolvla', 'xvla'):
        raise ValueError('Unsupported baseline')
    if any(p.device.type != 'cpu' for p in policy.parameters()):
        raise ValueError('CPU construction required before weight loading')
    if model_type == 'smolvla':
        with torch.no_grad():
            for parameter in policy.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
    else:
        policy.float()
    state = policy.state_dict()
    with safe_open(str(filename), framework='pt', device='cpu') as stored:
        for name in stored.keys():
            if name not in state:
                raise ValueError('Unexpected checkpoint tensor: ' + name)
            tensor = stored.get_tensor(name)
            if state[name].dtype != tensor.dtype or state[name].shape != tensor.shape:
                raise ValueError('Refusing implicit weight cast/reshape: ' + name)
            if model_type == 'xvla' and tensor.dtype != torch.float32:
                raise ValueError('X-VLA checkpoint must be entirely FP32')
    load_model(policy, str(filename), strict=True, device='cpu')
    # safe_open holds a mapping, not a second eager full state_dict allocation.
    with safe_open(str(filename), framework='pt', device='cpu') as stored:
        for name in stored.keys():
            a, b = state[name].detach(), stored.get_tensor(name)
            if not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
                raise ValueError('Loaded checkpoint bytes differ: ' + name)
    policy.eval()
    return policy


def load_baseline(release=None):
    """Return (policy, observation_preprocessor, physical_action_postprocessor).

    Assets are resolved from release, never the caller's working directory.
    CPU is the only verified execution target. No policy/base-weight downloads.
    """
    offline_mode()
    import json
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    root = Path(release or Path(__file__).parent).resolve()
    config = PreTrainedConfig.from_pretrained(str(root), local_files_only=True, token=False)
    config.device = 'cpu'
    if config.type == 'smolvla':
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        if config.load_vlm_weights:
            raise ValueError('Separate VLM weight loading is forbidden')
        config.vlm_model_name = bundled_path(root, config.vlm_model_name)
        policy = SmolVLAPolicy(config)
    elif config.type == 'xvla':
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        if config.dtype != 'float32':
            raise ValueError('X-VLA config must preserve FP32')
        config.tokenizer_name = bundled_path(root, config.tokenizer_name)
        policy = XVLAPolicy(config)
    else:
        raise ValueError('Unsupported baseline')
    strict_load_weights(policy, root / 'model.safetensors', config.type)
    processor = json.loads((root / 'policy_preprocessor.json').read_text())
    tokenizers = [s for s in processor['steps'] if s['registry_name'] == 'tokenizer_processor']
    if len(tokenizers) != 1:
        raise ValueError('Exactly one tokenizer processor required')
    tokenizer = bundled_path(root, tokenizers[0]['config']['tokenizer_name'])
    pre, post = make_pre_post_processors(
        config, pretrained_path=str(root),
        preprocessor_overrides={'device_processor': {'device': 'cpu'},
                                'tokenizer_processor': {'tokenizer_name': tokenizer}},
        postprocessor_overrides={'device_processor': {'device': 'cpu'}},
    )
    return policy, pre, post


def without_targets(batch):
    """Native converters emit action=None even for observation-only inputs."""
    if batch.get('action') is not None or batch.get('action_is_pad') is not None:
        raise ValueError('Target data must not enter inference')
    return {key: value for key, value in batch.items() if key not in ('action', 'action_is_pad')}


def predict(policy, preprocessor, postprocessor, observation, seed=20260918):
    """Predict a full physical chunk from observation only; resets policy RNG/state.

    Pass unbatched float32 state (9,), RGB CHW images (3,512,512) in [0,1],
    and task as a string. Seven absolute joint positions (radians) plus gripper
    command (0 closed, 255 open) are returned. No actuator clipping is applied.
    """
    import torch
    expected = {'observation.state': (9,), 'observation.images.image': (3, 512, 512),
                'observation.images.image2': (3, 512, 512)}
    if set(observation) != {*expected, 'task'} or not isinstance(observation['task'], str):
        raise ValueError('Only two RGB views, measured state and instruction are allowed')
    for key, shape in expected.items():
        value = observation[key]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.device.type != 'cpu'
                or not torch.isfinite(value).all()):
            raise ValueError('Invalid CPU observation: ' + key)
        if key.startswith('observation.images.') and not (value.min() >= 0 and value.max() <= 1):
            raise ValueError('RGB images must be in [0,1]')
    inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in observation.items()}
    policy.eval()
    policy.reset()
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        batch = without_targets(preprocessor(inputs))
        normalized = policy.predict_action_chunk(batch)
        physical = postprocessor(normalized)
    if not torch.isfinite(physical).all():
        raise ValueError('Non-finite predicted action')
    return physical