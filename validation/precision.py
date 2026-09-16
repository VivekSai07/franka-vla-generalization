"""Explicit FP32 trainable storage; never changes the frozen VLM or loss."""
import torch


def promote_trainable_parameters(model):
    counts = {"promoted_tensors": 0, "trainable_parameters": 0}
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                counts["trainable_parameters"] += parameter.numel()
                counts["promoted_tensors"] += int(parameter.dtype != torch.float32)
                parameter.data = parameter.data.float()
                if parameter.grad is not None:
                    parameter.grad = parameter.grad.float()
    return counts


def verify_optimizer_precision(optimizer):
    if not optimizer.state:
        raise ValueError("Optimizer state has not been initialized")
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.dtype != torch.float32:
                raise ValueError("Trainable parameter storage is not FP32")
            for key in ("exp_avg", "exp_avg_sq"):
                state = optimizer.state[parameter].get(key)
                if state is None or state.dtype != torch.float32 or not torch.isfinite(state).all():
                    raise ValueError(f"Invalid FP32 AdamW {key}")
    return True


def load_fp32_checkpoint(checkpoint):
    """Promote before copying saved FP32 weights; avoid rounding them into BF16."""
    from pathlib import Path
    from safetensors.torch import load_model
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.load_vlm_weights = False
    policy = SmolVLAPolicy(config)
    promote_trainable_parameters(policy)
    policy.to(config.device)
    load_model(policy, str(Path(checkpoint) / "model.safetensors"), strict=True, device=config.device)
    policy.eval()
    return policy