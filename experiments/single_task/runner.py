"""Exact fresh single-task recipe. --preflight is two scratch updates, no rollouts."""
import argparse
import gc
import json
import os
import random
import shutil
import time

import numpy as np
import torch

from . import contract as c, evaluate


def train(output, preflight, report):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from torch.utils.data import DataLoader, Subset

    def progress(**fields):
        report.update(fields)
        c.h.write_json(output/"result.json", report)

    manifest = c.h.require_complete(c.DATA)
    c.select_entries(manifest)
    gate = c.h.require_replay_pass(c.DATA, c.ROOT)
    binding = c.sources()
    cache = c.ROOT/".cache/benchmark_models"
    asset_files = c.inventory(cache)
    if not preflight:
        scratch = json.loads((c.SCRATCH/"result.json").read_text())
        if (scratch.get("status") != "complete" or scratch.get("purpose") != "single_task_preflight"
                or scratch.get("completed_training_steps") != 2 or not scratch.get("final_data_binding_passed")
                or scratch.get("source_hashes") != binding or scratch.get("dataset_fingerprint") != gate["dataset_fingerprint"]
                or scratch.get("asset_files") != asset_files
                or [m.get("update") for m in scratch.get("checkpoints", [])] != [1, 2]
                or scratch.get("final_checkpoint_files") != c.inventory(c.SCRATCH/"checkpoints/000002")):
            raise ValueError("Matching completed two-update scratch preflight required; never resumed")
        for marker in scratch["checkpoints"]:
            c.require_absolute_reload(marker)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 autocast support required")
    if shutil.disk_usage(output).free < 35*1024**3:
        raise RuntimeError("Less than 35 GiB checkpoint budget")
    random.seed(c.SEED)
    np.random.seed(c.SEED)
    torch.manual_seed(c.SEED)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    assets = json.loads((cache/"manifest.json").read_text())
    if (assets["policy"]["repo_id"] != "lerobot/smolvla_base" or assets["policy"]["revision"] != c.REVISION
            or assets["vlm"]["revision"] != "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
            or assets["policy"]["path"] != str(cache/"policy") or assets["vlm"]["path"] != str(cache/"vlm")):
        raise ValueError("Unexpected cached pinned base assets")
    config = PreTrainedConfig.from_pretrained(cache/"policy", local_files_only=True)
    config.device, config.vlm_model_name = "cuda", str(cache/"vlm")
    config.load_vlm_weights = config.push_to_hub = False
    config.train_expert_only = config.train_state_proj = True
    config.n_action_steps = 10
    if (config.chunk_size, config.n_obs_steps) != (50, 1):
        raise ValueError("Unexpected base temporal schema")
    raw = LeRobotDataset(repo_id=c.DATA.name, root=c.DATA, delta_timestamps={"action": [i/30 for i in range(50)]})
    features = dataset_to_policy_features(raw.features)
    config.input_features = {k: v for k, v in features.items() if v.type in (FeatureType.STATE, FeatureType.VISUAL)}
    config.output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    if ({k: tuple(v.shape) for k, v in config.input_features.items()} != c.OBS
            or {k: tuple(v.shape) for k, v in config.output_features.items()} != {"action": (8,)}):
        raise ValueError("Unexpected policy input/output schema")
    rows, starts, probes = c.indices(raw.meta.episodes)
    raw._ensure_hf_dataset_loaded()
    for i in c.TRAIN+c.VAL:
        for index in c.h.frame_indices(raw.meta.episodes, [i]):
            if int(raw.hf_dataset[index]["episode_index"]) != i:
                raise ValueError("Raw frame episode membership mismatch")
    stats = c.h.compute_training_stats(raw.hf_dataset, rows)
    c.h.write_json(output/"training_stats.json", {k: {s: v.tolist() for s, v in d.items()} for k, d in stats.items()})
    ds = c.PolicyRows(raw)
    policy = SmolVLAPolicy.from_pretrained(cache/"policy", config=config, local_files_only=True, strict=True)
    promotion = c.h.promote_trainable_parameters(policy)
    if promotion["trainable_parameters"] != 99880992 or sum(p.numel() for p in policy.parameters()) != 450046176:
        raise ValueError("Unexpected trainable expert/state projection architecture")
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    config.save_pretrained(output/"initial_config")
    params = [p for p in policy.parameters() if p.requires_grad]
    tracked = next(p for n, p in policy.named_parameters() if p.requires_grad and "action_out_proj.weight" in n)
    original = tracked.detach().clone()
    progress(status="training", dataset_fingerprint=gate["dataset_fingerprint"], source_hashes=binding,
             timing=manifest["timing"], train_episodes=list(c.TRAIN), validation_episodes=list(c.VAL),
             training_frames=len(rows), valid_training_chunk_starts=len(starts), model_assets=assets,
             asset_files=asset_files, precision_promotion=promotion, checkpoints=[], probes=[])

    def probe(update):
        metrics = {name: c.h.fixed_training_probes(policy, pre, post, ds, selected) for name, selected in probes.items()}
        report["probes"].append(dict(update=update, **metrics))
        progress(completed_training_steps=update)
        print("PROBE", json.dumps(report["probes"][-1], allow_nan=False), flush=True)

    probe(0)
    generator = torch.Generator().manual_seed(c.SEED)
    loader = DataLoader(Subset(ds, starts), batch_size=8, shuffle=True, drop_last=True, num_workers=0, generator=generator)
    epoch_state, epoch, batches = generator.get_state(), 0, 0
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(params, lr=c.learning_rate(1), betas=(.9, .95), weight_decay=1e-10)
    policy.train()
    maximum, started = (2 if preflight else 5000), time.monotonic()
    with (output/"metrics.jsonl").open("x") as log:
        for update in range(1, maximum+1):
            tick = time.monotonic()
            for group in optimizer.param_groups:
                group["lr"] = c.learning_rate(update)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(4):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch, batches, epoch_state = epoch+1, 0, generator.get_state()
                    iterator = iter(loader)
                    batch = next(iterator)
                batches += 1
                c.h.check_batch(batch)
                batch = pre(batch)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = policy(batch)
                c.h.backward_microbatch(loss)
                losses.append(loss.detach().item())
            norm = torch.nn.utils.clip_grad_norm_(params, 10., error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in params):
                raise ValueError("Nonfinite trainable parameter")
            if update == 1:
                if any(p.grad is not None for p in policy.parameters() if not p.requires_grad) or torch.equal(original, tracked):
                    raise ValueError("Frozen gradient or absent first projection update")
                c.h.verify_optimizer_precision(optimizer)
            torch.cuda.synchronize()
            row = dict(update=update, loss=sum(losses)/4, grad_norm=float(norm), lr=c.learning_rate(update),
                       seconds=time.monotonic()-tick, sampled_chunks=update*32, epoch=epoch,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            log.write(json.dumps(row, allow_nan=False)+"\n")
            log.flush()
            if update == 1 or update % 20 == 0 or update == maximum:
                progress(completed_training_steps=update, last_metrics=row, elapsed_seconds=time.monotonic()-started)
                print("TRAIN", json.dumps(row), flush=True)
            if update == 1 or update % 1000 == 0 or update == maximum:
                c.h.verify_optimizer_precision(optimizer)
                if shutil.disk_usage(output).free < 5*1024**3:
                    raise RuntimeError("Less than 5 GiB checkpoint save headroom")
                marker = c.h.save_checkpoint(policy, pre, post, optimizer, ds, probes["training"][0], output,
                                             update, generator, epoch_state, epoch, batches)
                c.require_absolute_reload(marker)
                report["checkpoints"].append(marker)
                progress(startup_save_reload_passed=True)
            if update % 1000 == 0 or update == maximum:
                probe(update)
    after = c.h.require_replay_pass(c.DATA, c.ROOT)
    if gate != after or c.sources() != binding or c.inventory(cache) != asset_files:
        raise ValueError("Data/source/cached base changed during training")
    progress(status="complete", completed_training_steps=maximum, final_data_binding_passed=True,
             selected_checkpoint_update=None if preflight else 5000,
             final_checkpoint_files=c.inventory(output/f"checkpoints/{maximum:06d}"),
             training_stats_sha256=c.h.file_hash(output/"training_stats.json"), task_success_evaluated=False)
    del policy, optimizer, pre, post, raw, ds, params, tracked, original, batch, loss
    gc.collect()
    torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Two fresh scratch updates/save/reload; zero rollouts")
    args = parser.parse_args(argv)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    output = c.SCRATCH if args.preflight else c.OUTPUT
    c.claim_output(output, output)
    report = dict(status="preparing", purpose="single_task_preflight" if args.preflight else "single_task",
                  training_steps=2 if args.preflight else 5000, seed=c.SEED, dataset_root=str(c.DATA.resolve()),
                  microbatch_size=8, gradient_accumulation=4, effective_batch_size=32, chunk_size=50, n_action_steps=10,
                  precision="fp32_trainable_bf16_autocast", loss="unchanged_stock_flow_matching",
                  schedule=dict(warmup=500, peak=1e-4, floor=1e-5, total=5000), resume=False, autotune=False)
    c.h.write_json(output/"result.json", report)
    try:
        train(output, args.preflight, report)
    except BaseException as exc:
        c.save_failure(output, report, exc)
        raise
    if not args.preflight:
        evaluate.run(output)


if __name__ == "__main__":
    main()