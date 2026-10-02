"""Recording entry point. Modes: --trace-only (deterministic reset trace), --expert-trace (one scripted episode)."""
import argparse
from pathlib import Path

import numpy as np

TRACE_KEYS = ("qpos", "ctrl", "sim_time", "substeps", "phase_index")


def trace_only(seed, out):
    from experiments.robosuite_scene.scene import build_sim, overwrite_reset
    _, model, data = build_sim()
    reset = overwrite_reset(model, data)
    np.savez(out / "trace.npz", qpos=reset["qpos"], initial_state=reset["initial_state"])


def expert_trace(seed, out):
    from experiments.robosuite_scene.expert import run_episode
    ticks = {key: [] for key in TRACE_KEYS}

    def sink(frame, trace):
        for key in TRACE_KEYS:
            ticks[key].append(np.copy(trace[key]))

    result = run_episode(seed, sink)
    np.savez(out / "trace.npz", **{key: np.asarray(values) for key, values in ticks.items()},
             initial_state=result["initial_state"], final_qpos=result["final_qpos"])


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trace-only", action="store_true")
    mode.add_argument("--expert-trace", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True)  # must not already exist
    if args.trace_only:
        trace_only(args.seed, out)
    else:
        expert_trace(args.seed, out)


if __name__ == "__main__":
    main()
