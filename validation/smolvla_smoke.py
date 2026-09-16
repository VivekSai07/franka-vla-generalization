"""Bounded train/save/reload/rollout integration test, not a benchmark score."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_collection.benchmark import require_complete, write_json
from validation.gates import fresh_stack_metrics, require_replay_pass


def compute_training_stats(rows, indices):
    stats = {}
    for key in ("observation.state", "action"):
        values = torch.stack([torch.as_tensor(rows[i][key]) for i in indices]).float()
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite training data")
        stats[key] = {"mean": values.mean(0), "std": values.std(0, unbiased=False),
                      "min": values.amin(0), "max": values.amax(0),
                      "q01": torch.quantile(values, .01, dim=0),
                      "q99": torch.quantile(values, .99, dim=0)}
    return stats


def valid_chunk_starts(episodes, selected, horizon):
    return [i for ep_id in selected
            for i in range(episodes[ep_id]["dataset_from_index"],
                           episodes[ep_id]["dataset_to_index"] - horizon + 1)]


def observation_only(frame):
    return {key: value for key, value in frame.items()
            if key.startswith("observation.") or key == "task"}


def experiment_protocol(manifest, overfit):
    train = list(manifest["splits"]["train"])
    if not train:
        raise ValueError("No training episodes")
    return {"train_episodes": train[:1] if overfit else train,
            "eval_episodes": train[:1] if overfit else list(manifest["splits"]["validation"]),
            "steps": 2000 if overfit else 100,
            "purpose": "single_training_scene_overfit" if overfit else "integration_only_not_benchmark_score"}


def fixed_training_probes(policy, preprocessor, postprocessor, ds, indices):
    """Compare identical physical action chunks with fixed noise, not random losses."""
    was_training = policy.training
    policy.eval()
    predictions, targets = [], []
    # Preserve training RNG; probes must not alter subsequent training samples/noise.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
        for index in indices:
            frame = ds[index]
            torch.manual_seed(4242 + index)
            predicted = postprocessor(policy.predict_action_chunk(preprocessor(observation_only(frame)))).squeeze(0).cpu()
            if not torch.isfinite(predicted).all():
                raise ValueError("Nonfinite probe predictions")
            predictions.append(predicted)
            targets.append(frame["action"].cpu())
    policy.train(was_training)
    predicted, target = torch.stack(predictions), torch.stack(targets)
    error = (predicted - target).abs()
    return {"frame_indices": indices,
            "arm_mae_rad": error[:, :, :7].mean().item(),
            "first_10_arm_mae_rad": error[:, :10, :7].mean().item(),
            "gripper_mae_command_units": error[:, :, 7].mean().item(),
            "gripper_binary_accuracy": ((predicted[:, :, 7] > 127) == (target[:, :, 7] > 127)).float().mean().item(),
            "arm_mae_by_probe_rad": error[:, :, :7].mean(dim=(1, 2)).tolist()}


def rollout(policy, preprocessor, postprocessor, manifest, dataset_root, output, episode_ids=None):
    import mujoco
    import av
    from data_collection.benchmark import apply_action, restore_state, step_control, restore_timing, synchronize_observation
    from validation.rollout_diagnostics import bound_action, target_diagnostics
    from src.controllers.grasp_controller import GraspController
    from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF
    results = []
    for ep_id in (manifest["splits"]["validation"] if episode_ids is None else episode_ids):
        entry = manifest["episodes"][ep_id]
        model = mujoco.MjModel.from_xml_path(str(ROOT / manifest["scene_path"]))
        timing = restore_timing(model, manifest)
        data = mujoco.MjData(model)
        scratch = mujoco.MjData(model)
        with np.load(dataset_root / "replay" / entry["file"], allow_pickle=False) as trace:
            restore_state(model, data, trace["initial_state"])
        renderer = mujoco.Renderer(model, height=512, width=512)
        policy.reset()
        grasp = GraspController(model, data, target=f"target_cube_{entry['pick']}")
        initial_pick_z = float(data.xpos[grasp.target_id, 2])
        torch.manual_seed(20260915 + ep_id)
        stable, intervention_steps, success = 0, 0, False
        joint_clip_counts = np.zeros(7, dtype=int)
        gripper_clip_ticks = threshold_ticks = 0
        max_joint_clip = 0.
        trace = {key: [] for key in ("sim_time", "state", "raw_action", "applied_action",
                        "joint_clip_mask", "gripper_clipped", "gripper_thresholded",
                        "finger_midpoint_to_pick_m", "pick_lift_m", "bilateral_target_contact",
                        "pick_position", "place_position", "xy_m", "z_m")}
        record = av.open(str(output / f"rollout_{ep_id}.mp4"), "w")
        stream = record.add_stream("libx264", rate=manifest["fps"])
        stream.width, stream.height, stream.pix_fmt = 1024, 512, "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}
        started = time.monotonic()
        try:
            for tick in range(600):
                synchronize_observation(model, data)
                trace["sim_time"].append(float(data.time))
                trace["state"].append(data.qpos[:9].copy())
                images = []
                for camera in manifest["camera_names"]:
                    renderer.update_scene(data, camera=camera)
                    images.append(renderer.render().copy())
                video_frame = av.VideoFrame.from_ndarray(np.concatenate(images, axis=1), format="rgb24")
                for packet in stream.encode(video_frame):
                    record.mux(packet)
                frame = {"observation.state": torch.tensor(data.qpos[:9].copy(), dtype=torch.float32),
                         "task": entry["instruction"]}
                for key, image in zip(("observation.images.image", "observation.images.image2"), images):
                    frame[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
                with torch.inference_mode():
                    action = postprocessor(policy.select_action(preprocessor(frame))).squeeze(0).cpu().numpy()
                bounded, diagnostics = bound_action(action, model.actuator_ctrlrange[list(ACTIVE_JOINTS)])
                if any(diagnostics["joint_clip_mask"]) or diagnostics["gripper_clipped"]:
                    intervention_steps += 1
                joint_clip_counts += diagnostics["joint_clip_mask"]
                gripper_clip_ticks += int(diagnostics["gripper_clipped"])
                threshold_ticks += int(diagnostics["gripper_thresholded"])
                max_joint_clip = max(max_joint_clip, diagnostics["max_joint_clip_rad"])
                trace["raw_action"].append(action.copy())
                trace["applied_action"].append(bounded.copy())
                for key in ("joint_clip_mask", "gripper_clipped", "gripper_thresholded"):
                    trace[key].append(diagnostics[key])
                apply_action(model, data, bounded, ACTIVE_JOINTS)
                step_control(model, data, manifest["control_dt_requested"], ARM_DOF)
                metrics = fresh_stack_metrics(model, data, scratch, entry["pick"], entry["place"])
                progress = target_diagnostics(model, data, entry["pick"], entry["place"], grasp, initial_pick_z)
                for key, value in progress.items():
                    trace[key].append(value)
                for key in ("xy_m", "z_m"):
                    trace[key].append(metrics[key])
                stable = stable + 1 if metrics["released_stack"] else 0
                if stable >= manifest["fps"]:
                    success = True
                    break
            result = {"episode": ep_id, "instruction": entry["instruction"], "ticks": tick + 1,
                      "wall_seconds": time.monotonic() - started,
                      "task_success": success, "intervention_steps": intervention_steps,
                      "gripper_adapter": "threshold_at_127_not_counted_as_clipping",
                      "joint_clipped_ticks_per_dimension": joint_clip_counts.tolist(),
                      "joint_clipped_ticks": int(np.any(trace["joint_clip_mask"], axis=1).sum()),
                      "gripper_clipped_ticks": gripper_clip_ticks,
                      "gripper_thresholded_ticks": threshold_ticks,
                      "max_joint_clip_rad": max_joint_clip,
                      "bilateral_target_contact_ticks": int(np.sum(trace["bilateral_target_contact"])),
                      "max_pick_lift_m": float(np.max(trace["pick_lift_m"])),
                      "closest_finger_midpoint_to_pick_m": float(np.min(trace["finger_midpoint_to_pick_m"])),
                      "timing": timing,
                      "unassisted_success": success and intervention_steps == 0,
                      "final_metrics": metrics}
            print("ROLLOUT", result, flush=True)
            results.append(result)
            np.savez_compressed(output / f"rollout_{ep_id}_trace.npz", **{key: np.asarray(value) for key, value in trace.items()})
        finally:
            for packet in stream.encode():
                record.mux(packet)
            record.close()
            renderer.close()
        write_json(output / "rollouts.json", results)
    return results


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from torch.utils.data import DataLoader, Subset

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overfit", action="store_true", help="Fixed 2000-update single-training-scene diagnostic")
    parser.add_argument("--fp32-trainable", action="store_true", help="FP32 parameter and AdamW storage; unchanged BF16 training autocast")
    args = parser.parse_args()
    if args.fp32_trainable and not args.overfit:
        parser.error("Precision experiment is bounded to --overfit")
    dataset_root, output = args.root.resolve(), args.output.resolve()
    manifest = require_complete(dataset_root)
    require_replay_pass(dataset_root, ROOT)
    protocol = experiment_protocol(manifest, args.overfit)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "training_steps": protocol["steps"], "batch_size": 2,
              "dataset_root": str(dataset_root), "splits": manifest["splits"],
              "seed": 20260915, "chunk_size": 50, "n_action_steps": 10,
              "purpose": protocol["purpose"], "protocol": protocol}
    report["timing"] = manifest["timing"]
    report["precision_mode"] = "fp32_trainable_bf16_training_autocast" if args.fp32_trainable else "checkpoint_native"
    write_json(output / "result.json", report)
    random.seed(20260915)
    np.random.seed(20260915)
    torch.manual_seed(20260915)
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    assets = json.loads((ROOT / ".cache/benchmark_models/manifest.json").read_text())
    policy_path, vlm_path = Path(assets["policy"]["path"]), Path(assets["vlm"]["path"])
    report["model_assets"] = assets
    report["policy_weights_sha256"] = hashlib.sha256((policy_path / "model.safetensors").read_bytes()).hexdigest()
    config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    config.device = "cuda"
    config.vlm_model_name = str(vlm_path)
    config.load_vlm_weights = False  # All policy/VLM weights are in the base policy snapshot.
    config.n_action_steps = 10
    config.push_to_hub = False
    config.train_expert_only = True
    config.train_state_proj = True
    ds = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root,
                        delta_timestamps={"action": [i / manifest["fps"] for i in range(config.chunk_size)]})
    features = dataset_to_policy_features(ds.features)
    config.output_features = {key: value for key, value in features.items() if value.type is FeatureType.ACTION}
    config.input_features = {key: value for key, value in features.items() if key not in config.output_features}
    train_eps = protocol["train_episodes"]
    train_indices = [i for e in train_eps for i in range(ds.meta.episodes[e]["dataset_from_index"],
                                                        ds.meta.episodes[e]["dataset_to_index"])]
    # Use raw parquet values for training-only statistics; never whole-dataset meta.stats.
    ds._ensure_hf_dataset_loaded()
    stats = compute_training_stats(ds.hf_dataset, train_indices)
    write_json(output / "training_stats.json", {k: {s: v.tolist() for s, v in values.items()} for k, values in stats.items()})
    starts = valid_chunk_starts(ds.meta.episodes, train_eps, config.chunk_size)
    assert starts
    policy = SmolVLAPolicy.from_pretrained(policy_path, config=config, local_files_only=True, strict=True)
    normalization_before = {}
    if args.fp32_trainable:
        from validation.precision import promote_trainable_parameters, verify_optimizer_precision
        report["precision_promotion"] = promote_trainable_parameters(policy)
        normalization_before = {name: p.detach().cpu().clone() for name, p in policy.named_parameters()
                                if p.requires_grad and ".lm_expert." in name and "norm" in name}
        report["expert_normalization_tensor_count"] = len(normalization_before)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    probe_indices = sorted(set(starts[i] for i in np.linspace(0, len(starts) - 1, 9, dtype=int)))
    report["training_frames"] = len(train_indices)
    report["valid_training_chunk_starts"] = len(starts)
    report["fixed_training_probes"] = []
    if args.overfit:
        probe = {"update": 0, **fixed_training_probes(policy, preprocessor, postprocessor, ds, probe_indices)}
        report["fixed_training_probes"].append(probe)
        print("FIXED_TRAINING_PROBES", probe, flush=True)
        write_json(output / "result.json", report)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    report["trainable_parameters"] = sum(p.numel() for p in parameters)
    report["total_parameters"] = sum(p.numel() for p in policy.parameters())
    tracked_name, tracked = next((name, p) for name, p in policy.named_parameters()
                                if p.requires_grad and "action_out_proj.weight" in name)
    initial_weight = tracked.detach().cpu().clone()
    loader = DataLoader(Subset(ds, starts), batch_size=2, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(20260915))
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .95), weight_decay=1e-10)
    # A bounded fixed-LR integration check, not the benchmark tuning schedule.
    policy.train()
    losses = []
    training_start = time.monotonic()
    for step in range(protocol["steps"]):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = preprocessor(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
        optimizer.step()
        if args.fp32_trainable and step == 0:
            report["adamw_fp32_state_verified"] = verify_optimizer_precision(optimizer)
            changes = {name: float((p.detach().cpu() - normalization_before[name]).abs().max())
                       for name, p in policy.named_parameters() if name in normalization_before}
            report["first_step_normalization_changes"] = changes
            if not any(value > 0 for value in changes.values()):
                raise AssertionError("No normalization weight update survived the first FP32 step")
            print("FP32_FIRST_STEP", {"changed_norm_tensors": sum(v > 0 for v in changes.values()),
                                      "max_change": max(changes.values())}, flush=True)
        losses.append(float(loss.detach()))
        if step % (100 if args.overfit else 10) == 0 or step == protocol["steps"] - 1:
            print(f"TRAIN {step + 1}/{protocol['steps']} loss={losses[-1]:.6f} grad_norm={float(norm):.6f}", flush=True)
        if args.overfit and step + 1 in (100, 500, 1000, 2000):
            probe = {"update": step + 1, **fixed_training_probes(policy, preprocessor, postprocessor, ds, probe_indices)}
            report["fixed_training_probes"].append(probe)
            report["completed_training_steps"] = step + 1
            print("FIXED_TRAINING_PROBES", probe, flush=True)
            write_json(output / "result.json", report)
    change = float((tracked.detach().cpu() - initial_weight).abs().max())
    if args.fp32_trainable:
        report["adamw_fp32_state_verified"] = verify_optimizer_precision(optimizer)
        report["final_normalization_changes"] = {
            name: float((p.detach().cpu() - normalization_before[name]).abs().max())
            for name, p in policy.named_parameters() if name in normalization_before}
    if not change > 0:
        raise AssertionError("Optimizer did not update the tracked action projection")
    report.update(training_seconds=time.monotonic() - training_start, losses=losses,
                  updated_parameter=tracked_name, max_parameter_change=change,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    report["last_completed_gate"] = "training"
    write_json(output / "result.json", report)
    policy.eval()
    validation_index = ds.meta.episodes[protocol["eval_episodes"][0]]["dataset_from_index"]
    observation = observation_only(ds[validation_index])
    torch.manual_seed(123)
    with torch.inference_mode():
        expected = postprocessor(policy.predict_action_chunk(preprocessor(observation))).cpu()
    assert expected.shape == (1, 50, 8) and torch.isfinite(expected).all()
    checkpoint = output / "pretrained_model"
    policy.save_pretrained(checkpoint)
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)
    del optimizer, parameters, tracked, policy, preprocessor, postprocessor, loss, batch
    gc.collect()
    torch.cuda.empty_cache()
    if args.fp32_trainable:
        from validation.precision import load_fp32_checkpoint
        reloaded = load_fp32_checkpoint(checkpoint)
        from safetensors import safe_open
        # Full check: saved FP32 values must not be rounded during reloading.
        with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as saved:
            for name, parameter in reloaded.named_parameters():
                if parameter.requires_grad:
                    if parameter.dtype != torch.float32:
                        raise AssertionError("Reload lost FP32 trainable storage")
                    torch.testing.assert_close(parameter.detach().cpu(), saved.get_tensor(name), rtol=0, atol=0)
        report["reload_fp32_weights_exact"] = True
    else:
        reloaded = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    preprocessor, postprocessor = make_pre_post_processors(reloaded.config, pretrained_path=str(checkpoint))
    torch.manual_seed(123)
    with torch.inference_mode():
        actual = postprocessor(reloaded.predict_action_chunk(preprocessor(observation))).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    report["reload_max_action_error"] = float((actual - expected).abs().max())
    report["last_completed_gate"] = "save_reload"
    write_json(output / "result.json", report)
    print("PASS: local save/reload prediction agreement", report["reload_max_action_error"], flush=True)
    report["rollouts"] = rollout(reloaded, preprocessor, postprocessor, manifest, dataset_root, output,
                                 episode_ids=protocol["eval_episodes"])
    report["status"] = "complete"
    write_json(output / "result.json", report)
    print("PASS: bounded training, save/reload and requested rollouts executed; see measured success separately")


if __name__ == "__main__":
    main()