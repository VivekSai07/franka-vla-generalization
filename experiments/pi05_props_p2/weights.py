"""Loading `pi05_base` into a `PI05Policy` without `from_pretrained`.

The installed loader reports the tied `embed_tokens.weight` as missing, prints a warning and carries on, and any other
failure inside it also only prints. This loader checks instead: every stored tensor maps to exactly one policy tensor with
the same shape, nothing is skipped or left over, and a policy tensor that is not stored is accepted only if it shares storage
with one that is (a tie). It copies into whatever dtype the policy tensor has, so a bf16 policy receives the rounded values
while `keep_masters` returns the exact FP32 checkpoint values of every trainable parameter.
"""
import torch

from experiments.pi05_props_p2 import assets

RENAMED = ('state_proj.', 'action_time_mlp_in.', 'action_time_mlp_out.')


def map_key(key):
    """Stored key -> policy state_dict key. The checkpoint needs none of the installed loader's renames or skips."""
    if key.startswith(RENAMED):
        raise ValueError(f'Stored key {key!r} needs a rename or skip this loader does not implement')
    return key if key.startswith('model.') else 'model.'+key


def load_base_weights(policy, path=None, keep_masters=False):
    from safetensors import safe_open
    path = path or assets.WEIGHTS_DIR/'model.safetensors'
    state = policy.state_dict()
    trainable = {p.data_ptr() for p in policy.parameters() if p.requires_grad}
    copied, dtypes, masters_by_ptr, targets = set(), {}, {}, {}
    with safe_open(str(path), framework='pt', device='cpu') as stored, torch.no_grad():
        for key in stored.keys():
            name = map_key(key)
            if name not in state:
                raise ValueError(f'Stored key {key!r} has no policy tensor {name!r}')
            if name in targets:
                raise ValueError(f'Two stored keys map to {name!r}: {targets[name]!r} and {key!r}')
            targets[name] = key
            tensor = stored.get_tensor(key)
            if tensor.dtype != torch.float32:
                raise ValueError(f'Stored tensor {key!r} is {tensor.dtype}, not float32')
            if tuple(tensor.shape) != tuple(state[name].shape):
                raise ValueError(f'Shape mismatch for {key!r}: {tuple(tensor.shape)} vs {tuple(state[name].shape)}')
            state[name].copy_(tensor)
            copied.add(state[name].data_ptr())
            dtypes[str(state[name].dtype)] = dtypes.get(str(state[name].dtype), 0)+1
            if keep_masters and state[name].data_ptr() in trainable:
                masters_by_ptr[state[name].data_ptr()] = tensor.clone()
    aliases = sorted(name for name in set(state)-set(targets) if state[name].data_ptr() in copied)
    missing = sorted(name for name in set(state)-set(targets) if state[name].data_ptr() not in copied)
    if missing:
        raise ValueError(f'Policy tensors neither stored nor tied to a stored one: {missing[:5]}')
    masters = {}
    if keep_masters:
        for name, parameter in policy.named_parameters():
            if parameter.requires_grad:
                masters[name] = masters_by_ptr[parameter.data_ptr()]
    report = dict(stored_tensors=len(targets), tied_aliases=aliases, loaded_dtypes=dtypes,
                  trainable_masters=len(masters))
    return report, masters
