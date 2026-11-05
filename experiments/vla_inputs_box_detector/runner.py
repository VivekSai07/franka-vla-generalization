"""The detector's commands: `extract`, `check`, `train`, `gate-decoded`, `gate-raw`, `gate-decide`.

    extract       decode the sealed videos into the frame cache (CPU; refuses below the 65 GiB disk guard)
    check         CPU admission: seals, split, the cache re-hashed, the network, a real batch through the loss; writes `admission_v1/result.json`
    train         the training run (GPU); requires a current admission; writes `run_v1/` with per-epoch validation IoU and `final/`
    gate-decoded  gate measurements (a) on the cached validation frames and (c) on synthetic out-of-view frames
    gate-raw      gate measurement (b), one shard of the 60 validation episodes replayed and rendered raw (robosuite image)
    gate-decide   merge the shards, apply the pinned thresholds, write `run_v1/gate/result.json`; exit code 1 when the gate fails

The checkpoint is chosen by validation IoU only (never by rollouts, never on a layout of any M0 tier); offline IoU selects a perception module
and is not capability. All constants are `constants.py`'s; nothing here decides a threshold.
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from experiments.vla_inputs_box_detector import constants as k
from experiments.vla_inputs_box_detector import contract, data, gate, model, predictor


def make_net(widths=k.WIDTHS, head_channels=k.HEAD_CHANNELS):
    return model.BoxDetector(widths, head_channels)


def learning_rate(step, steps_per_epoch, epochs=k.EPOCHS, base=k.LEARNING_RATE, warmup_epochs=k.WARMUP_EPOCHS):
    """Linear warm-up for `warmup_epochs`, then cosine to zero at the last step."""
    warmup, total = warmup_epochs * steps_per_epoch, epochs * steps_per_epoch
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def validation_score(report):
    """The mean of the two cameras' mean IoU on the validation frames: the only thing a checkpoint is chosen by."""
    return float(np.mean([report[name]["mean_iou"] for name in gate.CAMERAS]))


def disk_guard(path, guard_gib=k.CACHE_DISK_GUARD_GIB):
    free = shutil.disk_usage(path).free
    if free < guard_gib * 2**30:
        raise RuntimeError(f"refusing: {free / 2**30:.1f} GiB free, guard {guard_gib} GiB")


def _prepare(batch, device, generator, augment):
    out = []
    for key in ("front", "gripper"):
        x = batch[key].to(device).permute(0, 3, 1, 2).float() / 255.0
        out.append(model.standardise(data.photometric(x, generator) if augment else x))
    return out


def train(cache, run_dir, device="cpu", epochs=k.EPOCHS, batch=k.BATCH, widths=k.WIDTHS, head_channels=k.HEAD_CHANNELS, workers=0, binding=None,
          log=print):
    """Train from scratch on the training rows; validate on the validation rows after every epoch; keep the best by `validation_score`.

    Returns the final directory (`detector.pt`, `detector.json`). `run_dir` must not exist.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True)
    rows_train, rows_val = data.rows_of(cache, "train"), data.rows_of(cache, "validation")
    training = data.TrainingSet(cache, rows_train)
    torch.manual_seed(k.SEED)
    net = make_net(widths, head_channels).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=k.LEARNING_RATE, weight_decay=k.WEIGHT_DECAY)
    steps_per_epoch = len(training) // batch
    if steps_per_epoch < 1:
        raise ValueError("fewer training rows than one batch")
    generator = torch.Generator().manual_seed(k.SEED)
    history, best, step, started = [], None, 0, time.time()
    for epoch in range(epochs):
        order = np.random.default_rng([k.SEED, epoch]).permutation(len(training)) + epoch * len(training)
        loader = torch.utils.data.DataLoader(training, batch_size=batch, sampler=[int(i) for i in order], collate_fn=data.collate, num_workers=workers,
                                             drop_last=True)
        net.train()
        losses = []
        for samples in loader:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, steps_per_epoch, epochs)
            front, gripper = _prepare(samples, device, generator, augment=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                front_out, gripper_out = net(front, gripper)
            front_loss = model.loss(front_out, samples["front_targets"])
            gripper_loss = model.loss(gripper_out, samples["gripper_targets"])
            total = front_loss["total"] + gripper_loss["total"]
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            losses.append(float(total))
            step += 1
        net.eval()
        report = gate.decoded_report(predictor.Predictor(net, device), cache, rows_val, batch=64)
        score = validation_score(report)
        entry = dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_score=score, validation=report, seconds=time.time() - started)
        history.append(entry)
        if best is None or score > best["validation_score"]:
            best = dict(entry, state={key: value.detach().cpu() for key, value in net.state_dict().items()})
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
        log("TRAIN", epoch + 1, f"loss={entry['train_loss']:.4f}", f"val_iou={score:.4f}", {n: round(report[n]["mean_iou"], 4) for n in gate.CAMERAS}, flush=True)
    final = run_dir / "final"
    final.mkdir()
    torch.save(dict(state=best["state"], epoch=best["epoch"], validation=best["validation"]), final / predictor.WEIGHTS)
    description = dict(constants_sha256=predictor.constants_digest(), widths=list(widths), head_channels=head_channels, epoch=best["epoch"],
                       validation_score=best["validation_score"], validation=best["validation"], parameters=model.count_parameters(net),
                       cache_manifest_sha256=contract.file_sha256(Path(cache.directory) / "manifest.json"), binding_sha256=binding, epochs_run=epochs,
                       chosen_by="validation IoU only; never by rollouts, never on an M0 layout")
    (final / predictor.DESCRIPTION).write_text(json.dumps(description, indent=1, sort_keys=True) + "\n")
    return final


def admission_result(cache_dir=contract.CACHE, expect_episodes=540):
    """CPU admission: the binding, the cache hashed in full, the network's size, and a real batch through the loss. Returns the record."""
    binding = contract.binding()
    cache = data.open_cache(cache_dir, verify=True)
    if len(cache.manifest["episodes"]) != expect_episodes or cache.manifest["rows"] != expect_episodes * len(k.CACHE_FRAMES):
        raise ValueError("the cache does not hold every sealed episode at every cached frame")
    net = make_net()
    parameters = model.count_parameters(net)
    if parameters > k.MAX_PARAMETERS:
        raise ValueError(f"{parameters} parameters exceed the budget {k.MAX_PARAMETERS}")
    training = data.TrainingSet(cache, data.rows_of(cache, "train"))
    batch = data.collate([training[i] for i in range(4)])
    torch.manual_seed(0)
    front, gripper = _prepare(batch, "cpu", torch.Generator().manual_seed(0), augment=True)
    front_out, gripper_out = net(front, gripper)
    total = model.loss(front_out, batch["front_targets"])["total"] + model.loss(gripper_out, batch["gripper_targets"])["total"]
    if not torch.isfinite(total):
        raise ValueError("the loss of a real batch is not finite")
    return dict(binding=binding, binding_sha256=contract.digest(binding), cache_manifest_sha256=contract.file_sha256(Path(cache_dir) / "manifest.json"),
                parameters=parameters, batch_loss=float(total), rows=cache.manifest["rows"])


def require_admission(admission_dir=contract.ADMISSION, cache_dir=contract.CACHE):
    """The admission record must exist and equal what is derived now."""
    record = json.loads((Path(admission_dir) / "result.json").read_text())
    if record["binding_sha256"] != contract.binding_sha256():
        raise ValueError("the binding changed since the admission; run --check again")
    if record["cache_manifest_sha256"] != contract.file_sha256(Path(cache_dir) / "manifest.json"):
        raise ValueError("the cache changed since the admission")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("extract").add_argument("--workers", type=int, default=6)
    commands.add_parser("check")
    t = commands.add_parser("train")
    t.add_argument("--workers", type=int, default=8)
    commands.add_parser("gate-decoded")
    r = commands.add_parser("gate-raw")
    r.add_argument("--shard", type=int, required=True)
    r.add_argument("--shards", type=int, required=True)
    commands.add_parser("gate-decide")
    args = parser.parse_args(argv)
    if args.command == "extract":
        data.extract(contract.CACHE, workers=args.workers)
    elif args.command == "check":
        record = admission_result()
        contract.ADMISSION.mkdir(parents=True)
        (contract.ADMISSION / "result.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print("CHECK PASSED", record["binding_sha256"][:12], record["parameters"], flush=True)
    elif args.command == "train":
        record = require_admission()
        if not torch.cuda.is_available():
            raise RuntimeError("training needs the GPU")
        disk_guard(contract.OUTPUT_ROOT)
        cache = data.open_cache(contract.CACHE, verify=False)
        final = train(cache, contract.RUN, device="cuda", workers=args.workers, binding=record["binding_sha256"])
        print("TRAINED", final, flush=True)
    else:
        require_admission()
        final = contract.FINAL
        out = contract.RUN / "gate"
        if args.command == "gate-decoded":
            cache = data.open_cache(contract.CACHE, verify=False)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pred = predictor.Predictor.load(final, device)
            rows = data.rows_of(cache, "validation")
            out.mkdir(exist_ok=True)
            report = dict(decoded=gate.decoded_report(pred, cache, rows, batch=64), synthetic=gate.synthetic_report(pred, cache, rows, batch=64))
            (out / "decoded.json").write_text(json.dumps(report, indent=1) + "\n")
            print("GATE_DECODED", json.dumps(report), flush=True)
        elif args.command == "gate-raw":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pred = predictor.Predictor.load(final, device)
            episodes = [e for e in k.VALIDATION_EPISODES if (e - k.VALIDATION_EPISODES[0]) % args.shards == args.shard]
            out.mkdir(exist_ok=True)
            records = gate.raw_report(pred, episodes)
            (out / f"raw_shard_{args.shard}_of_{args.shards}.json").write_text(json.dumps(records) + "\n")
        else:
            records = [r for p in sorted(out.glob("raw_shard_*.json")) for r in json.loads(p.read_text())]
            if sorted(r["episode"] for r in records) != list(k.VALIDATION_EPISODES):
                raise ValueError("the raw-render shards do not cover exactly the 60 validation episodes")
            decoded = json.loads((out / "decoded.json").read_text())
            verdict = gate.decide(decoded["decoded"], gate.merge_raw(records), decoded["synthetic"])
            verdict["raw"] = gate.merge_raw(records)
            verdict.update(decoded)
            (out / "result.json").write_text(json.dumps(verdict, indent=1) + "\n")
            print("GATE", "PASSED" if verdict["passed"] else "FAILED", flush=True)
            for c in verdict["checks"]:
                print("  ", "ok  " if c["ok"] else "FAIL", c["name"], c["value"], "threshold", c["threshold"], flush=True)
            return 0 if verdict["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
