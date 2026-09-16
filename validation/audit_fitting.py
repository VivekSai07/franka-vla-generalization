"""Read-only fitting audit: no optimizer, rollout, or collection commands."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parameter_group(name):
    if ".vlm_with_expert.vlm." in name:
        return "frozen_vlm"
    if ".vlm_with_expert.lm_expert." in name:
        return "action_expert"
    return name.split(".")[1]


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from data_collection.benchmark import write_json
    from validation.gates import require_replay_pass, dataset_fingerprint
    from validation.replanning_probe import checkpoint_hashes
    from validation.smolvla_smoke import compute_training_stats, valid_chunk_starts
    from torch.utils.data import default_collate

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, checkpoint, output = args.root.resolve(), args.checkpoint.resolve(), args.output.resolve()
    require_replay_pass(root, ROOT)
    provenance = checkpoint_hashes(checkpoint)
    training = json.loads((checkpoint.parent / "result.json").read_text())
    if training["status"] != "complete" or training["protocol"]["train_episodes"] != [0]:
        raise ValueError("Expected completed single-episode overfit checkpoint")
    if Path(training["dataset_root"]) != root:
        raise ValueError("Checkpoint/dataset mismatch")
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "incomplete", "optimizer_updates": 0, "rollouts": 0,
              "dataset_fingerprint": dataset_fingerprint(root), "checkpoint_hashes": provenance,
              "checkpoint": str(checkpoint), "dataset_root": str(root)}
    write_json(output / "result.json", result)
    torch.set_num_threads(8)
    torch.manual_seed(20260915)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    ds = LeRobotDataset(root.name, root=root,
                        delta_timestamps={"action": [i / 30 for i in range(policy.config.chunk_size)]})
    ds._ensure_hf_dataset_loaded()
    episode = ds.meta.episodes[0]
    start, end = episode["dataset_from_index"], episode["dataset_to_index"]
    indices = list(range(start, end))
    stats = compute_training_stats(ds.hf_dataset, indices)
    recorded_stats = json.loads((checkpoint.parent / "training_stats.json").read_text())
    errors = {}
    for key, values in stats.items():
        errors[key] = {}
        for stat, value in values.items():
            saved = torch.tensor(recorded_stats[key][stat], dtype=value.dtype)
            errors[key][stat] = float((value - saved).abs().max())
            torch.testing.assert_close(value, saved, rtol=0, atol=0)
    result["training_stats_recomputed_max_error"] = errors
    result["statistics"] = recorded_stats
    result["near_constant_dimensions"] = {key: torch.where(values["std"] < 1e-5)[0].tolist()
                                           for key, values in stats.items()}
    inputs = policy.config.input_features
    result["input_features"] = {key: list(value.shape) for key, value in inputs.items()}
    result["output_features"] = {key: list(value.shape) for key, value in policy.config.output_features.items()}
    result["config"] = {key: getattr(policy.config, key) for key in
                        ("train_expert_only", "train_state_proj", "freeze_vision_encoder",
                         "max_state_dim", "max_action_dim", "chunk_size", "n_action_steps")}

    # Compare every saved parameter against the original pinned base snapshot.
    base = Path(training["model_assets"]["policy"]["path"]) / "model.safetensors"
    if hashlib.sha256(base.read_bytes()).hexdigest() != training["policy_weights_sha256"]:
        raise ValueError("Base checkpoint differs from training provenance")
    groups, changed_frozen = {}, []
    with safe_open(base, framework="pt", device="cpu") as original:
        for name, parameter in policy.named_parameters():
            group = groups.setdefault(parameter_group(name), {"parameters": 0, "trainable": 0,
                                    "tensors": 0, "changed_tensors": 0, "max_change": 0.})
            group["parameters"] += parameter.numel()
            group["trainable"] += parameter.numel() if parameter.requires_grad else 0
            group["tensors"] += 1
            old = original.get_tensor(name)
            current = parameter.detach().cpu()
            change = float((current.float() - old.float()).abs().max())
            group["changed_tensors"] += int(change > 0)
            group["max_change"] = max(group["max_change"], change)
            if not parameter.requires_grad and change:
                changed_frozen.append(name)
    result["parameter_groups"] = groups
    result["changed_frozen_parameters"] = changed_frozen

    action_rows = np.stack([ds.hf_dataset[i]["action"].numpy() for i in indices])
    transitions = (np.flatnonzero(np.diff(action_rows[:, 7]) != 0) + 1).tolist()
    result["gripper_transitions"] = transitions
    result["gripper_values"] = np.unique(action_rows[:, 7]).tolist()
    starts = valid_chunk_starts(ds.meta.episodes, [0], 50)
    result["valid_chunk_starts"] = len(starts)
    result["training_chunk_max_frame"] = max(starts) + 49

    target_checks = []
    for index in (start, start + transitions[0], start + transitions[-1], end - 50):
        frame = ds[index]
        target = frame["action"].clone()
        expected_target = torch.stack([ds.hf_dataset[i]["action"] for i in range(index, index + 50)])
        torch.testing.assert_close(target, expected_target, rtol=0, atol=0)
        batch = pre(frame)
        expected_norm = (target - stats["action"]["mean"]) / (stats["action"]["std"] + 1e-8)
        normalized = batch["action"].squeeze(0).cpu()
        norm_error = float((normalized - expected_norm).abs().max())
        recovered = post(batch["action"]).squeeze(0).cpu()
        action_error = float((recovered - target).abs().max())
        raw_state = ds.hf_dataset[index]["observation.state"]
        expected_state = (raw_state - stats["observation.state"]["mean"]) / (stats["observation.state"]["std"] + 1e-8)
        state_error = float((batch["observation.state"].squeeze(0).cpu() - expected_state).abs().max())
        target_checks.append({"frame": index, "normalization_max_error": norm_error,
                              "action_roundtrip_max_error": action_error,
                              "state_normalization_max_error": state_error,
                              "raw_padding_keys": [key for key in frame if "pad" in key],
                              "processed_padding_keys": [key for key in batch if "pad" in key]})
    result["target_checks"] = target_checks

    # Observe the actual unreduced flow matching loss with fixed time/noise.
    loss_checks, captured = [], []
    def capture_loss(module, arguments, values):
        captured.append(values.detach().float().cpu())
    # Policy calls model.forward directly, bypassing module-level forward hooks.
    # Observe the called output projection and reconstruct its exact MSE target.
    hook = policy.model.action_out_proj.register_forward_hook(capture_loss)
    try:
        with torch.inference_mode():
            for index in (start + transitions[0], start + transitions[-1]):
                batch = pre(default_collate([ds[index]]))
                padded = policy.prepare_action(batch)
                assert tuple(padded.shape) == (1, 50, 32)
                assert torch.count_nonzero(padded[:, :, 8:]) == 0
                noise = torch.randn(padded.shape, device=padded.device,
                                    generator=torch.Generator(device=padded.device).manual_seed(987))
                flow_time = torch.full((1,), .5, device=padded.device)
                loss, _ = policy(batch, noise=noise, time=flow_time)
                elementwise = (captured[-1] - (noise - padded).detach().float().cpu()).square()
                active = float(elementwise[:, :, :8].mean())
                padding = float(elementwise[:, :, 8:].mean())
                loss_checks.append({"frame": index, "returned_loss": float(loss),
                                    "active_8d_mean": active, "padded_24d_mean": padding,
                                    "active_contribution": active * .25, "padding_contribution": padding * .75,
                                    "matches_32d_mean": bool(torch.isclose(loss.cpu(), elementwise.mean()))})
            # Artificial all-padding masks isolate whether the spelling is consumed.
            index = end - 1
            raw = ds[index]
            boundary = pre(default_collate([raw]))
            padded = policy.prepare_action(boundary)
            noise = torch.zeros_like(padded)
            flow_time = torch.full((1,), .5, device=padded.device)
            standard = {**boundary, "action_is_pad": torch.ones((1, 50), device=padded.device, dtype=torch.bool)}
            alternate = {**boundary, "actions_id_pad": torch.ones((1, 50), device=padded.device, dtype=torch.bool)}
            standard_loss, _ = policy(standard, noise=noise, time=flow_time)
            alternate_loss, _ = policy(alternate, noise=noise, time=flow_time)
            result["temporal_mask_probe"] = {"actual_last_frame_padding_count": int(raw["action_is_pad"].sum()),
                                              "loss_all_action_is_pad": float(standard_loss),
                                              "loss_all_actions_id_pad": float(alternate_loss),
                                              "training_samples_have_padding": False}
    finally:
        hook.remove()
    result["loss_decomposition"] = loss_checks
    result["installed_loss_source"] = inspect.getsource(SmolVLAPolicy.forward)

    # One backward pass audits gradient reachability, not an optimizer update.
    policy.zero_grad(set_to_none=True)
    batch = pre(default_collate([ds[start + transitions[0]]]))
    torch.manual_seed(12345)
    loss, _ = policy(batch)
    loss.backward()
    gradients = {}
    for name, parameter in policy.named_parameters():
        group = gradients.setdefault(parameter_group(name), {"trainable_tensors": 0, "with_gradient": 0,
                                      "nonzero_gradient_tensors": 0, "max_abs_gradient": 0.})
        group["trainable_tensors"] += int(parameter.requires_grad)
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise ValueError("Nonfinite audit gradient")
            maximum = float(parameter.grad.abs().max())
            group["with_gradient"] += 1
            group["nonzero_gradient_tensors"] += int(maximum > 0)
            group["max_abs_gradient"] = max(group["max_abs_gradient"], maximum)
    policy.zero_grad(set_to_none=True)
    result["gradient_groups"] = gradients
    result["backward_passes_without_optimizer"] = 1
    if checkpoint_hashes(checkpoint) != provenance:
        raise AssertionError("Checkpoint artifacts changed")
    require_replay_pass(root, ROOT)
    result["status"] = "complete"
    write_json(output / "result.json", result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("statistics", "checkpoint_hashes", "installed_loss_source")}, indent=2), flush=True)


if __name__ == "__main__":
    main()