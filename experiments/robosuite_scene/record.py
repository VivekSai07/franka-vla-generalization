"""Recording entry point.

Modes: --trace-only (deterministic reset trace), --expert-trace (one scripted episode),
--collect (the EPISODE_COUNT-episode proof-of-concept dataset, sealed).
"""
import argparse
import copy
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from experiments.robosuite_scene import contract

TRACE_KEYS = ("qpos", "ctrl", "sim_time", "substeps", "phase_index")


def trace_only(seed, out):
    from experiments.robosuite_scene.scene import build_sim, overwrite_reset
    _, model, data = build_sim()
    reset = overwrite_reset(model, data)
    np.savez(out / "trace.npz", qpos=reset["qpos"], initial_state=reset["initial_state"])


def _save_trace(path, ticks, result):
    np.savez(path, **{key: np.asarray(values) for key, values in ticks.items()},
             initial_state=result["initial_state"], final_qpos=result["final_qpos"])


def expert_trace(seed, out):
    from experiments.robosuite_scene.expert import run_episode
    ticks = {key: [] for key in TRACE_KEYS}

    def sink(frame, trace):
        for key in TRACE_KEYS:
            ticks[key].append(np.copy(trace[key]))

    result = run_episode(seed, sink)
    _save_trace(out / "trace.npz", ticks, result)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def collect(out):
    """Seeds 0..EPISODE_COUNT-1, one episode each. A failed seed aborts before any seal is written."""
    from experiments.paired_demonstrations.media import CAMERAS, ENCODING, FEATURES, PairedDataset
    from experiments.robosuite_scene.expert import run_episode
    from experiments.robosuite_scene.validate import fingerprint
    dataset = PairedDataset.create(
        repo_id="robosuite_bringup", root=out / "dataset", fps=30, features=copy.deepcopy(FEATURES),
        vcodec=ENCODING["vcodec"], encoder_threads=ENCODING["encoder_threads"], metadata_buffer_size=1)
    episodes, first_images = [], None
    try:
        for seed in range(contract.EPISODE_COUNT):
            started = time.monotonic()
            frames, ticks = [], {key: [] for key in TRACE_KEYS}

            def sink(frame, trace):
                # The expert owns and reuses what it hands over: buffer copies only.
                frames.append({key: np.copy(value) if isinstance(value, np.ndarray) else value
                               for key, value in frame.items()})
                if seed == 0:
                    for key in TRACE_KEYS:
                        ticks[key].append(np.copy(trace[key]))

            result = run_episode(seed, sink)
            if not result["success"]:
                raise RuntimeError(f"seed {seed} failed: {result['reason']}; nothing sealed")
            # The reset has no RNG, so every seed's first observation is the same picture. A second
            # OSMesa context in the process can corrupt renders; this catches that before encoding.
            images = {key: frames[0][key] for key in CAMERAS}
            if first_images is None:
                first_images = images
            elif any(not np.array_equal(images[key], first_images[key]) for key in CAMERAS):
                raise RuntimeError(f"seed {seed}: first frame differs from seed 0; nothing sealed")
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            if seed == 0:
                _save_trace(out / "trace_seed_0.npz", ticks, result)
            episodes.append({"seed": seed, "frames": len(frames), "final_xy_m": float(result["final_xy_m"]),
                             "final_z_m": float(result["final_z_m"]), "stable_ticks": int(result["stable_ticks"])})
            print(f"EPISODE seed={seed} success=True frames={len(frames)} "
                  f"final_xy_m={episodes[-1]['final_xy_m']:.5f} final_z_m={episodes[-1]['final_z_m']:.5f} "
                  f"stable_ticks={episodes[-1]['stable_ticks']} wall_s={time.monotonic() - started:.1f}", flush=True)
    finally:
        dataset.finalize()  # close parquet footers; a partial episode is never saved
    _write_json(out / "episodes.json", episodes)
    seal = {"fingerprint": fingerprint(out), "image": contract.IMAGE,
            "grasp_z_offset_m": contract.GRASP_Z_OFFSET_M}
    _write_json(out / "seal.json", seal)
    print(f"SEALED episodes={len(episodes)} fingerprint={seal['fingerprint']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trace-only", action="store_true")
    mode.add_argument("--expert-trace", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.collect and args.seed is not None:
        parser.error("--collect runs seeds 0..EPISODE_COUNT-1; omit --seed")
    if not args.collect and args.seed is None:
        parser.error("--seed is required with --trace-only and --expert-trace")
    out = Path(args.out)
    if args.collect:
        if mujoco.__version__ != contract.MUJOCO_VERSION:
            raise SystemExit(f"MuJoCo {mujoco.__version__} is not the pinned {contract.MUJOCO_VERSION}; "
                             "run in the robosuite image")
        if out.exists():
            raise SystemExit(f"No overwrite/resume: {out} exists")
    out.mkdir(parents=True)  # must not already exist
    if args.trace_only:
        trace_only(args.seed, out)
    elif args.expert_trace:
        expert_trace(args.seed, out)
    else:
        collect(out)


if __name__ == "__main__":
    main()
