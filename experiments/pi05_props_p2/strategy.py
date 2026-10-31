"""The three D5 memory strategies behind one interface: build the policy, update it, save and restore its state.

* `bf16_all`: parameters, gradients and Adam moments all live in the policy's dtype (bf16 except the few tensors LeRobot
  keeps FP32); updates smaller than a bf16 step are lost.
* `bf16_cpu_master`: the policy computes in bf16 on the GPU; FP32 master weights, FP32 gradients and the FP32 moments live
  on the CPU, the CPU optimizer steps them and the rounded result is copied back.
* `expert_only`: the PaliGemma side is frozen in bf16; the action expert and the projections are FP32 with FP32 Adam,
  run under bf16 autocast.

Gradient clipping is the same in all three: the global norm is accumulated in FP32 (`dtype=`), the gradients are scaled in
place by min(1, clip / (norm + 1e-6)).
"""
import contextlib
import time

import torch

from experiments.pi05_props_p2 import contract as c
from experiments.pi05_props_p2 import weights

BETAS, EPS, WEIGHT_DECAY, CLIP = (.9, .95), 1e-8, .01, 1.0


def build_policy(name, keep_masters=None):
    """CPU policy for strategy `name` with the pinned base weights loaded; returns (policy, config, masters, report)."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    config = c.make_config('cpu', name)
    policy = PI05Policy(config)
    if name == 'expert_only':
        policy.model.paligemma_with_expert.gemma_expert.float()  # before loading: the expert gets exact FP32 values
    keep = name == 'bf16_cpu_master' if keep_masters is None else keep_masters
    report, masters = weights.load_base_weights(policy, keep_masters=keep)
    return policy, config, masters, report


def trainable(policy):
    return {name: p for name, p in policy.named_parameters() if p.requires_grad}


def clip_gradients(params, clip=CLIP):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        raise ValueError('No trainable parameter has a gradient')
    norms = torch.stack([torch.linalg.vector_norm(g, dtype=torch.float32) for g in grads])
    norm = torch.linalg.vector_norm(norms)
    if not bool(torch.isfinite(norm)):
        raise ValueError('Non-finite gradient norm')
    coefficient = torch.clamp(clip/(norm+1e-6), max=1.0)
    torch._foreach_mul_(grads, coefficient.to(grads[0].device))
    return float(norm)


def dtype_counts(tensors):
    counts = {}
    for t in tensors:
        counts[str(t.dtype)] = counts.get(str(t.dtype), 0)+t.numel()
    return counts


class Strategy:
    name = None

    def __init__(self, policy):
        self.policy, self.params = policy, trainable(policy)
        self.timings = {}

    def zero_grad(self):
        for p in self.params.values():
            p.grad = None

    def check_unused(self):
        """Trainable tensors the loss never reaches (pi0.5's unused heads and last prefix layer) get no gradient and, as in
        LeRobot's own training, no Adam step and no weight decay. The set must be the same on every update."""
        names = sorted(n for n, p in self.params.items() if p.grad is None)
        if getattr(self, 'unused', None) is None:
            self.unused = names
        elif names != self.unused:
            raise ValueError('The set of trainable tensors without a gradient changed between updates')
        return names

    def autocast(self):
        return contextlib.nullcontext()

    def dtype_report(self):
        raise NotImplementedError


class GpuAdam(Strategy):
    """`bf16_all` and `expert_only`: the optimizer runs on the parameters where they are."""

    def __init__(self, policy, name, device):
        super().__init__(policy)
        self.name, self.device = name, device
        self.optimizer = torch.optim.AdamW(list(self.params.values()), lr=c.learning_rate(1), betas=BETAS, eps=EPS,
                                           weight_decay=WEIGHT_DECAY, foreach=True)

    def autocast(self):
        if self.name == 'expert_only':
            return torch.autocast('cuda', dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def step(self, update, lr=None):
        c.set_learning_rate(self.optimizer, update, lr)
        self.check_unused()
        norm = clip_gradients(list(self.params.values()))
        self.optimizer.step()
        return dict(grad_norm=norm)

    def model_state(self):
        return self.policy.state_dict()

    def optimizer_state(self):
        return self.optimizer.state_dict()

    def restore(self, model_state, optimizer_state):
        self.policy.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(optimizer_state)

    def dtype_report(self):
        moments = [s[k] for s in self.optimizer.state.values() for k in ('exp_avg', 'exp_avg_sq') if k in s]
        grads = [p.grad for p in self.params.values() if p.grad is not None]
        return dict(parameters=dtype_counts(self.params.values()), gradients=dtype_counts(grads),
                    moments=dtype_counts(moments), masters=None)


class CpuMasterAdam(Strategy):
    """`bf16_cpu_master`: FP32 masters, gradients and moments on the CPU; the GPU holds only what the forward needs."""
    name = 'bf16_cpu_master'

    def __init__(self, policy, masters, threads=None, fused=True):
        super().__init__(policy)
        if set(masters) != set(self.params):
            raise ValueError('Masters must cover exactly the trainable parameters')
        self.masters = {}
        for name, value in masters.items():
            if value.dtype != torch.float32 or value.device.type != 'cpu' or value.shape != self.params[name].shape:
                raise ValueError(f'Master {name} must be a CPU FP32 tensor of the parameter\'s shape')
            self.masters[name] = torch.nn.Parameter(value, requires_grad=True)
            self.grad_buffers = getattr(self, 'grad_buffers', {})
            self.grad_buffers[name] = torch.zeros_like(value)
        if threads:
            torch.set_num_threads(threads)
        self.optimizer = torch.optim.AdamW(list(self.masters.values()), lr=c.learning_rate(1), betas=BETAS, eps=EPS,
                                           weight_decay=WEIGHT_DECAY, fused=fused)

    def _tick(self, key, began):
        if self.params and next(iter(self.params.values())).is_cuda:
            torch.cuda.synchronize()
        self.timings[key] = time.perf_counter()-began
        return time.perf_counter()

    def step(self, update, lr=None):
        began = time.perf_counter()
        c.set_learning_rate(self.optimizer, update, lr)
        self.check_unused()
        norm = clip_gradients(list(self.params.values()))
        began = self._tick('clip', began)
        with torch.no_grad():
            for name, p in self.params.items():
                if p.grad is None:
                    self.masters[name].grad = None  # skipped by the optimizer, like a parameter without a gradient
                else:
                    self.grad_buffers[name].copy_(p.grad)
                    self.masters[name].grad = self.grad_buffers[name]
            began = self._tick('grad_to_cpu', began)
            self.optimizer.step()
            began = self._tick('cpu_adam', began)
            for name, p in self.params.items():
                p.copy_(self.masters[name].detach())
            self._tick('weights_to_gpu', began)
        return dict(grad_norm=norm, phase_seconds=dict(self.timings))

    def model_state(self):
        """What the file `model.pt` of a snapshot holds: the exact masters, from which the GPU copy follows."""
        return {name: m.detach() for name, m in self.masters.items()}

    def optimizer_state(self):
        return self.optimizer.state_dict()

    def restore(self, model_state, optimizer_state):
        with torch.no_grad():
            for name, value in model_state.items():
                self.masters[name].copy_(value)
                self.params[name].copy_(value)
        self.optimizer.load_state_dict(optimizer_state)

    def dtype_report(self):
        moments = [s[k] for s in self.optimizer.state.values() for k in ('exp_avg', 'exp_avg_sq') if k in s]
        grads = [g for n, g in self.grad_buffers.items() if n not in (self.unused or ())]
        return dict(parameters=dtype_counts(self.params.values()), gradients=dtype_counts(grads),
                    moments=dtype_counts(moments), masters=dtype_counts(self.masters.values()),
                    gpu_gradients=dtype_counts([p.grad for p in self.params.values() if p.grad is not None]))


def make_strategy(name, policy, masters, device):
    if name == 'bf16_cpu_master':
        return CpuMasterAdam(policy, masters)
    if name in ('bf16_all', 'expert_only'):
        return GpuAdam(policy, name, device)
    raise ValueError(f'Unknown strategy {name!r}')


def changed_fractions(before, after):
    """Per tensor: the fraction of elements whose stored value differs between two snapshots of the same tensors."""
    rows = []
    for name, old in before.items():
        new = after[name]
        rows.append(dict(name=name, numel=old.numel(), dtype=str(old.dtype),
                         fraction_changed=float((new != old).sum())/old.numel()))
    return rows


def summarize_changes(rows):
    total = sum(r['numel'] for r in rows)
    fractions = sorted(r['fraction_changed'] for r in rows)
    return dict(tensors=len(rows), elements=total,
                element_weighted=sum(r['fraction_changed']*r['numel'] for r in rows)/total,
                tensor_min=fractions[0], tensor_median=fractions[len(fractions)//2], tensor_max=fractions[-1],
                tensors_with_no_change=sum(1 for f in fractions if f == 0.0))
