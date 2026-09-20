"""Fresh local pilot training; no uploads, automatic resume or scene evaluation."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_collection.benchmark import require_complete, write_json
from training.pilot_protocol import (backward_microbatch, check_batch, checked_splits,
                                     frame_indices, learning_rate, probe_indices)
from validation.gates import require_replay_pass
from validation.precision import (load_fp32_checkpoint, promote_trainable_parameters,
                                  verify_optimizer_precision)
from validation.smolvla_smoke import (compute_training_stats, fixed_training_probes,
                                     observation_only, valid_chunk_starts)

SEED = 20260916


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_source_hashes():
    paths = ("training/smolvla_pilot.py", "training/pilot_protocol.py",
             "validation/precision.py", "validation/smolvla_smoke.py")
    return {path: file_hash(ROOT / path) for path in paths}


def save_checkpoint(policy, pre, post, optimizer, ds, probe, output, update,
                    generator, epoch_generator_state, epoch, batches_in_epoch):
    """A checkpoint is complete only after exact FP32 reload and action agreement."""
    from lerobot.policies.factory import make_pre_post_processors
    from safetensors import safe_open

    checkpoint = output / "checkpoints" / f"{update:06d}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    was_training = policy.training
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        policy.eval()
        observation = observation_only(ds[probe])
        torch.manual_seed(123)
        with torch.inference_mode():
            expected = post(policy.predict_action_chunk(pre(observation))).cpu()
        if expected.shape != (1, 50, 8) or not torch.isfinite(expected).all():
            raise ValueError("Invalid pre-save prediction")
        policy.save_pretrained(checkpoint)
        pre.save_pretrained(checkpoint)
        post.save_pretrained(checkpoint)
        reloaded = load_fp32_checkpoint(checkpoint)
        with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as saved:
            for name, parameter in reloaded.named_parameters():
                if parameter.requires_grad:
                    if parameter.dtype != torch.float32:
                        raise ValueError("Reload lost FP32 storage")
                    torch.testing.assert_close(parameter.detach().cpu(), saved.get_tensor(name), rtol=0, atol=0)
        rpre, rpost = make_pre_post_processors(reloaded.config, pretrained_path=str(checkpoint))
        torch.manual_seed(123)
        with torch.inference_mode():
            actual = rpost(reloaded.predict_action_chunk(rpre(observation))).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        error = (actual - expected).abs().max().item()
        del reloaded, rpre, rpost, actual, expected
        gc.collect()
        torch.cuda.empty_cache()
    policy.train(was_training)
    torch.save({"optimizer": optimizer.state_dict(), "update": update,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "loader_generator": generator.get_state(), "epoch_generator_state": epoch_generator_state,
                "epoch": epoch, "batches_in_epoch": batches_in_epoch,
                "resume_supported": False}, checkpoint / "training_state.pt")
    marker = {"status": "complete", "update": update, "reload_fp32_weights_exact": True,
              "reload_max_action_error": error, "model_sha256": file_hash(checkpoint / "model.safetensors")}
    write_json(checkpoint / "checkpoint_result.json", marker)
    print("CHECKPOINT", json.dumps(marker), flush=True)
    return marker


def run(args, report):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from torch.utils.data import DataLoader, Subset

    root, output = args.root.resolve(), args.output.resolve()
    def progress(**fields):
        report.update(fields)
        write_json(output / "result.json", report)

    manifest = require_complete(root)
    gate = require_replay_pass(root, ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if shutil.disk_usage(output).free < 35 * 1024**3:
        raise RuntimeError("Less than 35 GiB free for checkpoints")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    initial_sources = training_source_hashes()
    progress(status="preparing", dataset_fingerprint=gate["dataset_fingerprint"],
             training_source_hashes=initial_sources, timing=manifest["timing"],
             gpu=torch.cuda.get_device_name(), checkpoints=[], evaluations=[])
    assets = json.loads((ROOT / ".cache/benchmark_models/manifest.json").read_text())
    policy_path, vlm_path = Path(assets["policy"]["path"]), Path(assets["vlm"]["path"])
    if assets["policy"]["revision"] != "c83c3163b8ca9b7e67c509fffd9121e66cb96205":
        raise ValueError("Unexpected pretrained policy revision")
    config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    config.device = "cuda"
    config.vlm_model_name = str(vlm_path)
    config.load_vlm_weights = False
    config.n_action_steps = 10
    config.push_to_hub = False
    config.train_expert_only = True
    config.train_state_proj = True
    if config.chunk_size != 50:
        raise ValueError("Unexpected chunk length")
    ds = LeRobotDataset(repo_id=root.name, root=root,
                        delta_timestamps={"action": [i / manifest["fps"] for i in range(50)]})
    train_eps, val_eps = checked_splits(manifest, ds.meta.total_episodes)
    features = dataset_to_policy_features(ds.features)
    config.output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    config.input_features = {k: v for k, v in features.items() if v.type in (FeatureType.STATE, FeatureType.VISUAL)}
    if (set(config.input_features) != {"observation.state", "observation.images.image", "observation.images.image2"}
            or tuple(config.input_features["observation.state"].shape) != (9,)
            or tuple(config.output_features["action"].shape) != (8,)):
        raise ValueError("Unexpected policy input/output schema")
    train_rows = frame_indices(ds.meta.episodes, train_eps)
    starts = valid_chunk_starts(ds.meta.episodes, train_eps, 50)
    if len(train_rows) != 158400 or len(starts) != 143700:
        raise ValueError("Unexpected pilot frame/chunk counts")
    ds._ensure_hf_dataset_loaded()
    print("Computing train-only statistics", flush=True)
    stats = compute_training_stats(ds.hf_dataset, train_rows)
    write_json(output / "training_stats.json", {k: {s: v.tolist() for s, v in values.items()} for k, values in stats.items()})
    policy = SmolVLAPolicy.from_pretrained(policy_path, config=config, local_files_only=True, strict=True)
    promotion = promote_trainable_parameters(policy)
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    params = [p for p in policy.parameters() if p.requires_grad]
    tracked = next(p for name, p in policy.named_parameters() if p.requires_grad and "action_out_proj.weight" in name)
    before = tracked.detach().cpu().clone()
    frozen = [name for name, p in policy.named_parameters() if not p.requires_grad]
    if promotion["trainable_parameters"] != 99880992 or sum(p.numel() for p in policy.parameters()) != 450046176:
        raise ValueError("Unexpected pretrained architecture or trainable modules")
    progress(model_assets=assets, policy_weights_sha256=file_hash(policy_path / "model.safetensors"),
             precision_promotion=promotion, train_episodes=train_eps, validation_episodes=val_eps,
             training_frames=len(train_rows), valid_training_chunk_starts=len(starts),
             frozen_parameter_names=frozen, full_config=config.to_dict() if hasattr(config, "to_dict") else None)
    config.save_pretrained(output / "initial_config")
    train_probes = sorted(set(starts[i] for i in np.linspace(0, len(starts) - 1, 9, dtype=int)))
    val_probes = probe_indices(ds.meta.episodes, val_eps)
    if args.preflight:
        val_probes = val_probes[:3]
    def evaluate(update):
        progress(status="evaluating", completed_training_steps=update)
        metrics = {"update": update,
                   "training": fixed_training_probes(policy, pre, post, ds, train_probes),
                   "validation": fixed_training_probes(policy, pre, post, ds, val_probes)}
        report["evaluations"].append(metrics)
        if update > 0 and ("best_validation_arm_mae_rad" not in report or metrics["validation"]["arm_mae_rad"] < report["best_validation_arm_mae_rad"]):
            report["best_validation_arm_mae_rad"] = metrics["validation"]["arm_mae_rad"]
            report["best_checkpoint_update"] = update
        print("EVALUATION", json.dumps({"update": update, "train_arm_mae_rad": metrics["training"]["arm_mae_rad"],
                                       "validation_arm_mae_rad": metrics["validation"]["arm_mae_rad"],
                                       "validation_gripper_accuracy": metrics["validation"]["gripper_binary_accuracy"]}), flush=True)
        progress(status="training")
    evaluate(0)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(Subset(ds, starts), batch_size=8, shuffle=True, drop_last=True,
                        num_workers=0, generator=generator)
    epoch_generator_state = generator.get_state()
    iterator = iter(loader)
    epoch, batches_in_epoch = 0, 0
    optimizer = torch.optim.AdamW(params, lr=learning_rate(1), betas=(.9, .95), weight_decay=1e-10)
    policy.train()
    max_updates = 2 if args.preflight else 20000
    optimizer_wall = 0.
    started = time.monotonic()
    with (output / "metrics.jsonl").open("x") as metrics_log:
        for update in range(1, max_updates + 1):
            tick = time.monotonic()
            lr = learning_rate(update)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(4):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    batches_in_epoch = 0
                    epoch_generator_state = generator.get_state()
                    iterator = iter(loader)
                    batch = next(iterator)
                batches_in_epoch += 1
                check_batch(batch)
                batch = pre(batch)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = policy(batch)
                backward_microbatch(loss)
                losses.append(loss.detach().item())
            norm = torch.nn.utils.clip_grad_norm_(params, 10., error_if_nonfinite=True)
            optimizer.step()
            if update == 1:
                if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
                    raise ValueError("Frozen VLM received gradients")
                verify_optimizer_precision(optimizer)
                delta = (tracked.detach().cpu() - before).abs().max().item()
                if delta <= 0:
                    raise ValueError("First optimizer update did not change action projection")
                progress(first_step_fp32_optimizer_verified=True, first_step_projection_delta=delta)
            torch.cuda.synchronize()
            seconds = time.monotonic() - tick
            optimizer_wall += seconds
            row = {"update": update, "loss": sum(losses) / 4, "grad_norm": float(norm),
                   "lr": lr, "seconds": seconds, "epoch": epoch,
                   "sampled_chunks": update * 32, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
            metrics_log.write(json.dumps(row) + "\n")
            metrics_log.flush()
            if update == 1 or update % 20 == 0 or update == max_updates:
                progress(status="training", completed_training_steps=update, last_metrics=row,
                         mean_update_seconds=optimizer_wall / update,
                         estimated_remaining_training_seconds=(max_updates - update) * optimizer_wall / update,
                         elapsed_loop_seconds=time.monotonic() - started)
                print("TRAIN", json.dumps(row), flush=True)
            if update == 1 or update % 2000 == 0 or update == max_updates:
                progress(status="checkpointing", completed_training_steps=update)
                marker = save_checkpoint(policy, pre, post, optimizer, ds, val_probes[0], output,
                                         update, generator, epoch_generator_state, epoch, batches_in_epoch)
                report["checkpoints"].append(marker)
                progress(status="training", startup_save_reload_passed=True)
            if update % 2000 == 0 or update == max_updates:
                evaluate(update)
    verify_optimizer_precision(optimizer)
    after = require_replay_pass(root, ROOT)
    if after["dataset_fingerprint"] != report["dataset_fingerprint"] or training_source_hashes() != initial_sources:
        raise ValueError("Dataset or training source changed during run")
    progress(status="complete", completed_training_steps=max_updates,
             final_data_binding_passed=True, total_loop_seconds=time.monotonic() - started,
             task_success_evaluated=False)
    print("COMPLETE: training and offline checks; no closed-loop success claim", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="Two scratch updates, same physical batch and precision; never resumed")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "preflight", "purpose": "two_update_preflight" if args.preflight else "smolvla_balanced_pilot",
              "training_steps": 2 if args.preflight else 20000, "seed": SEED,
              "microbatch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32,
              "chunk_size": 50, "n_action_steps": 10, "dataset_root": str(args.root.resolve()),
              "precision_mode": "fp32_trainable_bf16_training_autocast",
              "loss": "unchanged_stock_flow_matching_including_24_padded_action_dimensions",
              "temporal_padding": "excluded_complete_chunks_and_asserted_empty_mask",
              "schedule": {"peak_lr": 1e-4, "warmup_updates": 1000, "end_lr": 1e-5, "total_updates": 20000},
              "resume": False, "upload": False}
    write_json(args.output / "result.json", report)
    try:
        run(args, report)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(args.output / "result.json", report)
        raise


if __name__ == "__main__":
    main()