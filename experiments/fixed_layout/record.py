"""Single-use fixed-reset recording. No retries, source overrides or training."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from shutil import disk_usage
import signal
import sys
from time import monotonic

import numpy as np

from experiments.fixed_layout.contract import (
    ROOT, OUTPUT, SOURCE, MIN_FREE_BYTES, MAX_SECONDS, make_plan, validate_plan,
)
from experiments.fixed_layout.source import Expert, load_reset, preflight, provenance, storage_path
from experiments.paired_demonstrations.media import FEATURES, ENCODING
from experiments.fixed_layout.media import BoundPairedDataset as PairedDataset


def write_json(path, value):
    """Evidence is write-once, flushed before advancing the lifecycle."""
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_npz(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def file_hash(path, check=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            check()
            digest.update(block)
    return digest.hexdigest()


class Budget:
    def __init__(self, root):
        self.root, self.started = Path(root), monotonic()

    def check(self):
        if monotonic() - self.started >= MAX_SECONDS:
            raise TimeoutError("Fixed-layout 3h wall deadline exceeded")
        if disk_usage(storage_path(self.root)).free < MIN_FREE_BYTES:
            raise ValueError("At least 30 GiB free space required; no reserve override")

    def __enter__(self):
        self.check()
        if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
            raise RuntimeError("Cannot replace an existing process deadline")
        self.old_handler = signal.getsignal(signal.SIGALRM)
        def expire(signum, frame):
            raise TimeoutError("Fixed-layout 3h wall deadline exceeded")
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, max(.001, MAX_SECONDS - (monotonic() - self.started)))
        return self

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old_handler)


def snapshot_hashes(root):
    return {name: file_hash(Path(root) / name) for name in ("plan.json", "provenance.json", "reset.npz")}


def check_snapshots(root, hashes):
    root = Path(root)
    required = {"plan.json", "provenance.json", "reset.npz"}
    if not required <= hashes.keys() or not hashes.keys() <= required | {"configuration.json", "source.npz"}:
        raise ValueError("Frozen snapshot inventory has missing/unknown paths")
    if any((root / name).is_symlink() for name in hashes):
        raise ValueError("Frozen snapshots cannot be symlinks")
    if any(file_hash(root / name) != sha for name, sha in hashes.items()):
        raise ValueError("Frozen plan/reset/source snapshot changed")
    validate_plan(json.loads((root / "plan.json").read_text()))
    if json.loads((root / "provenance.json").read_text()) != provenance():
        raise ValueError("Source/runtime changed before next operation")


def run_schedule(plan, initial, execute, validate, journal, before=lambda: None):
    """One call per immutable slot; validation completes before the next attempt."""
    validate_plan(plan)
    frozen, reset = copy.deepcopy(plan), initial.copy()
    if reset.dtype != np.float64 or not np.isfinite(reset).all():
        raise ValueError("Expected finite full-precision reset")
    admitted = []
    for slot in frozen["episodes"]:
        index = slot["episode_index"]
        try:
            validate_plan(plan)
            if not np.array_equal(initial, reset):
                raise ValueError("Initial state mutated between attempts")
            before()
            journal({"event": "started", **slot, "training_ready": False})
            outcome = execute(copy.deepcopy(slot), reset.copy())
            if outcome.get("success") is not True or outcome.get("frames") != 528:
                raise RuntimeError("First failed or wrong-duration expert attempt; no retry")
            validate(copy.deepcopy(slot), outcome)
            journal({"event": "admitted", "episode_index": index, "training_ready": False})
            admitted.append(index)
            if index == 2:
                journal({"event": "preparation_pass", "episode_ids": [0, 1, 2], "training_ready": False})
        except BaseException as exc:
            try:
                journal({"event": "failed", "episode_index": index, "training_ready": False,
                         "error": f"{type(exc).__name__}: {exc}"})
            except BaseException as secondary:
                print(f"Failure journal unavailable: {secondary!r}", file=sys.stderr)
            raise
    return admitted


class TickSink:
    """Bounded memory: each original callback is durable before media or gates."""
    def __init__(self, root, dataset, check):
        self.root, self.dataset, self.check = Path(root), dataset, check
        (self.root / "ticks").mkdir(exist_ok=False)
        self.count = 0
        self.error = None
        self.terminal_available = False

    def __call__(self, frame, trace):
        try:
            arrays = {key: np.asarray(value).copy() for key, value in frame.items()}
            arrays.update({"trace/" + key: np.asarray(value).copy() for key, value in trace.items()})
            write_npz(self.root / "ticks" / f"{self.count:06d}.npz", arrays)
            self.count += 1
            # Check AFTER preserving this already-delivered callback, BEFORE more work.
            self.check()
            if self.count > 528:
                raise ValueError("Expert emitted more than the fixed 528 frames")
            self.dataset.add_frame(frame)
            self.check()
        except BaseException as exc:
            self.error = exc
            raise

    def interrupted(self, error):
        write_json(self.root / "interrupted.json", {
            "success": False, "training_ready": False, "frames_preserved": self.count,
            "error": f"{type(error).__name__}: {error}",
            "last_known_tick": self.count - 1 if self.count else None,
            "state_scope": "returned_terminal_state" if self.terminal_available else (
                "last_emitted_pre_step_observation" if self.count else "frozen_reset_only"),
            "terminal_integration_state": "terminal.npz:final_state" if self.terminal_available else None,
            "unavailable": [] if self.terminal_available else [
                "actual_post_interruption_state", "unemitted_frames", "expert_return_value"],
        })


def record_attempt(root, expert, dataset, episode, initial, check):
    root = Path(root)
    sink = TickSink(root, dataset, check)
    primary = None
    dataset._active_attempt_path = root
    try:
        check()
        outcome = expert.run(initial.copy(), copy.deepcopy(episode), sink)
        # Preserve the returned terminal state even when the expert reports failure.
        write_npz(root / "terminal.npz", {key: outcome[key] for key in ("final_qpos", "final_state")})
        sink.terminal_available = True
        scalar = {key: value for key, value in outcome.items() if key not in ("final_qpos", "final_state")}
        write_json(root / "outcome.json", scalar)
        check()
        if scalar.get("success") is not True or scalar.get("frames") != 528 or sink.count != 528:
            raise RuntimeError("Failed expert or noncanonical duration; raw attempt preserved")
        dataset.save_episode()
        check()
        return scalar
    except BaseException as exc:
        primary = sink.error if sink.error is not None else exc
        try:
            sink.interrupted(primary)
            if primary is not exc:
                write_json(root / "expert-cleanup-error.json", {"error": repr(exc), "original": repr(primary)})
        except BaseException as secondary:
            print(f"Interrupted-state marker unavailable: {secondary!r}", file=sys.stderr)
        if primary is not exc:
            raise primary from exc
        raise
    finally:
        dataset._active_attempt_path = None
        try:
            dataset.finalize()  # Only close writers; never save a partial imitation episode.
        except BaseException as secondary:
            if primary is None:
                raise
            try:
                write_json(root / "finalization-error.json", {"error": repr(secondary), "original": repr(primary)})
            except BaseException:
                print(f"Finalization also failed: {secondary!r}; original preserved", file=sys.stderr)


def collect():
    """Canonical root only. Calling this executes physics; NEVER called by preflight."""
    from experiments.fixed_layout.validate import validate_episode, validate_dataset, fingerprint
    output = OUTPUT
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"No overwrite/resume/relaunch: {output}")
    if any(p.is_symlink() for p in output.parents):
        raise ValueError("Output parents must not be symlinks")
    budget = Budget(output)
    budget.check()  # Before constructor, output mkdir or any expert call.
    with budget:
        expert = Expert()
        plan, bound, reset = make_plan(), provenance(), load_reset(expert)
        budget.check()
        if provenance() != bound:
            raise ValueError("Source changed during reset preflight")
        output.mkdir(parents=True, exist_ok=False)
        dataset = None
        try:
            budget.check()  # The newly created destination may be a different filesystem.
            write_json(output / "plan.json", plan)
            write_json(output / "provenance.json", bound)
            write_json(output / "configuration.json", expert.config)
            write_npz(output / "reset.npz", reset)
            with (output / "source.npz").open("xb") as stream:
                stream.write((ROOT / SOURCE).read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            hashes = snapshot_hashes(output)
            hashes.update({name: file_hash(output / name) for name in ("configuration.json", "source.npz")})
            write_json(output / "snapshots.json", hashes)
            (output / "attempts").mkdir()
            dataset = PairedDataset.create(repo_id="fixed_layout", root=output / "dataset", fps=30,
                features=copy.deepcopy(FEATURES), vcodec=ENCODING["vcodec"],
                encoder_threads=ENCODING["encoder_threads"], metadata_buffer_size=1)
            # Independent, closed files per episode: never append over validated media/rows.
            dataset.meta.info["data_files_size_in_mb"] = 0
            dataset.meta.info["video_files_size_in_mb"] = 0
            dataset.fixed_layout_check = budget.check

            def journal(event):
                with (output / "journal.jsonl").open("a") as stream:
                    stream.write(json.dumps({**event, "elapsed_seconds": monotonic() - budget.started}, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())

            def before():
                budget.check()
                check_snapshots(output, hashes)
                fresh = load_reset(expert)  # Source + compiled model + no-step reset, never resettle.
                if expert.config != json.loads((output / "configuration.json").read_text()):
                    raise ValueError("Expert configuration changed")
                for key in reset:
                    if not np.array_equal(fresh[key], reset[key]):
                        raise ValueError("Fixed reset changed")
                budget.check()

            def execute(episode, initial):
                path = output / "attempts" / f"{episode['episode_index']:06d}"
                path.mkdir(exist_ok=False)
                write_json(path / "slot.json", episode)
                result = record_attempt(path, expert, dataset, episode, initial, budget.check)
                dataset._writer_closed_for_reading = True
                return result

            def validate(episode, outcome):
                before()
                validate_episode(output, episode, expert, reset, budget.check)
                before()

            run_schedule(plan, reset["initial_state"], execute, validate, journal, before)
            dataset.finalize()
            dataset = None
            before()
            # LeRobot defaults all rows to train; persist the approved split explicitly.
            info_path = output / "dataset/meta/info.json"
            info = json.loads(info_path.read_text())
            info["splits"] = {"train": "0:20", "validation": "20:24"}
            temporary = info_path.with_suffix(".fixed-layout.json")
            write_json(temporary, info)
            temporary.replace(info_path)
            report = validate_dataset(output, expert, reset, budget.check)
            before()
            write_json(output / "validation.json", report)
            budget.check()
            write_json(output / "seal.json", {"fingerprint": fingerprint(output, budget.check)})
            return report
        except BaseException as exc:
            try:
                write_json(output / "failure.json", {"training_ready": False, "error": repr(exc),
                           "elapsed_seconds": monotonic() - budget.started})
            except BaseException as secondary:
                print(f"Failure marker unavailable: {secondary!r}", file=sys.stderr)
            raise
        finally:
            if dataset is not None:
                try:
                    dataset.finalize()
                except BaseException as secondary:
                    # A failure marker always blocks admission; keep the triggering exception.
                    print(f"Dataset cleanup failed: {secondary!r}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--preflight", action="store_true")
    choice.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    result = preflight() if args.preflight else collect()
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if (not args.preflight or result["reset_valid"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())