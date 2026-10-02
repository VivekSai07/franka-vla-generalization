"""Recording entry point. Modes: --trace-only (deterministic reset trace)."""
import argparse
from pathlib import Path

import numpy as np


def trace_only(seed, out):
    from experiments.robosuite_scene.scene import build_sim, overwrite_reset
    _, model, data = build_sim()
    reset = overwrite_reset(model, data)
    np.savez(out / "trace.npz", qpos=reset["qpos"], initial_state=reset["initial_state"])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True)  # must not already exist
    if args.trace_only:
        trace_only(args.seed, out)
    else:
        parser.error("no mode selected (supported: --trace-only)")


if __name__ == "__main__":
    main()
