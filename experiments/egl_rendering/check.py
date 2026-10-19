"""EGL side-by-side check: run one unmodified P1 attempt under a named renderer, store everything, then compare.

    python -m experiments.egl_rendering.check run --renderer {osmesa,egl} --label NAME --out DIR [--slot 0]
    python -m experiments.egl_rendering.check report --out DIR --osmesa-base L --osmesa-new L --egl L L L --layers-added N

`run` needs the robosuite image (the EGL one for `--renderer egl`); `report` needs numpy only. Design and criteria:
docs/superpowers/specs/2026-10-19-egl-rendering-design.md. The criteria are fixed there; nothing here may move them.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from experiments.egl_rendering import compare

RENDERERS = ("osmesa", "egl")


def require_renderer(requested, env):
    """Refuse to run under an environment that is not the renderer the label claims (the whole comparison rests on it)."""
    if requested not in RENDERERS:
        raise SystemExit(f"renderer must be one of {RENDERERS}, not {requested!r}")
    if env.get("MUJOCO_GL") != requested:
        raise SystemExit(f"--renderer {requested} but MUJOCO_GL={env.get('MUJOCO_GL')!r}")
    if requested == "egl" and env.get("PYOPENGL_PLATFORM") != "egl":
        raise SystemExit("EGL needs PYOPENGL_PLATFORM=egl as well (PyOpenGL binds to another platform otherwise)")


def verdict(osmesa_pair, physics_differences, egl_pairs, layers_added):
    """Criteria C1 to C3 of the design. C4 and C5 are recorded, never gated."""
    # C1 as written covers the OSMesa pixels here; its physics half is part of C2, which compares every run including that pair
    c1 = layers_added == 1 and osmesa_pair["images_differing"] == 0 and osmesa_pair["max_abs"] == 0
    c2 = not physics_differences
    c3 = compare.repeatable(egl_pairs)
    return dict(C1=c1, C2=c2, C3=c3, admitted=c1 and c2 and c3)


def run(renderer, label, out, slot=0):
    require_renderer(renderer, os.environ)
    import mujoco
    from experiments.paired_demonstrations.expert_v1 import CAMERAS
    from experiments.props_p1_data import plan as p
    from experiments.props_p1_data.episode import run_attempt

    episode = p.make_plan()["episodes"][slot]
    draw = p.attempt_draw(episode, 0)
    images, physics = [], {}

    def sink(frame, trace):
        images.append(np.stack([frame[key] for key in CAMERAS]))
        for key, value in trace.items():
            if key != "image_sha256":
                physics.setdefault(key, []).append(value)
        physics.setdefault("action", []).append(frame["action"])
        physics.setdefault("observation.state", []).append(frame["observation.state"])

    started = time.monotonic()
    result = run_attempt(episode, draw, sink)
    wall = time.monotonic() - started
    arrays = {key: np.asarray(values) for key, values in physics.items()}
    arrays["success"], arrays["reason"], arrays["ticks"] = (str(result.get("success")), str(result.get("reason")),
                                                           np.asarray(len(images)))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stack = np.stack(images)
    np.save(out / f"{label}_images.npy", stack)
    np.savez(out / f"{label}_physics.npz", **arrays)
    gl = None
    try:                                           # the driver string, to see that EGL ran on the GPU and OSMesa on llvmpipe
        import OpenGL.GL as gl_api
        gl = str(gl_api.glGetString(gl_api.GL_RENDERER))
    except Exception:
        pass
    info = dict(label=label, renderer=renderer, slot=slot, wall_seconds=wall, ticks=len(images), images=stack.shape[0] * stack.shape[1],
                mujoco=mujoco.__version__, gl_renderer=gl, success=result.get("success"), reason=result.get("reason"),
                image_id=os.environ.get("IMAGE_ID"), lp_num_threads=os.environ.get("LP_NUM_THREADS"))
    (out / f"{label}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


def _load(out, label):
    return np.load(Path(out) / f"{label}_images.npy", mmap_mode="r"), dict(np.load(Path(out) / f"{label}_physics.npz"))


def _physics_plain(arrays):
    return {k: (v.item() if v.dtype.kind in "US" and v.shape == () else v) for k, v in arrays.items()}


def report(out, osmesa_base, osmesa_new, egl, layers_added):
    out = Path(out)
    loaded = {label: _load(out, label) for label in [osmesa_base, osmesa_new, *egl]}
    infos = {label: json.loads((out / f"{label}.json").read_text()) for label in loaded}
    names = list(loaded)
    differences = sorted({key for other in names[1:] for key in compare.physics_differences(
        _physics_plain(loaded[names[0]][1]), _physics_plain(loaded[other][1]))})
    osmesa_pair = compare.compare_images(np.asarray(loaded[osmesa_base][0]), np.asarray(loaded[osmesa_new][0]))
    egl_pairs = [compare.compare_images(np.asarray(loaded[a][0]), np.asarray(loaded[b][0]))
                 for i, a in enumerate(egl) for b in egl[i + 1:]]
    c4 = {label: compare.compare_images(np.asarray(loaded[osmesa_new][0]), np.asarray(loaded[label][0])) for label in egl}
    middle = loaded[osmesa_new][0].shape[0] // 2
    c4_frames = {}
    for tick in (0, middle, loaded[osmesa_new][0].shape[0] - 1):
        c4_frames[tick] = compare.compare_images(np.asarray(loaded[osmesa_new][0][tick:tick + 1]),
                                                 np.asarray(loaded[egl[0]][0][tick:tick + 1]))
    result = dict(verdict=verdict(osmesa_pair, differences, egl_pairs, layers_added), physics_differences=differences,
                  c1_osmesa_new_vs_base=osmesa_pair, c3_egl_pairs=egl_pairs, c4_egl_vs_osmesa=c4, c4_frames=c4_frames,
                  c5_wall_seconds={label: infos[label]["wall_seconds"] for label in names}, runs=infos)
    (out / "report.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps(result["verdict"]), "physics differences:", differences, flush=True)
    return 0 if result["verdict"]["admitted"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--renderer", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--slot", type=int, default=0)
    q = sub.add_parser("report")
    q.add_argument("--out", required=True)
    q.add_argument("--osmesa-base", required=True)
    q.add_argument("--osmesa-new", required=True)
    q.add_argument("--egl", nargs="+", required=True)
    q.add_argument("--layers-added", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.renderer, args.label, args.out, args.slot)
        return 0
    return report(args.out, args.osmesa_base, args.osmesa_new, args.egl, args.layers_added)


if __name__ == "__main__":
    sys.exit(main())
