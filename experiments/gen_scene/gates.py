"""Equality gates of the randomization engine (G0 spec D9 i and ii). Robosuite image, CPU.

    equality   the engine's nominal condition composes the model, the initial state and the two 512 px renders that
               `props_scene` does, for sealed P2 episodes: every compiled array equal, initial state equal, renders byte equal
    expert     the forked expert, on the engine's nominal model, reproduces the sealed episodes' qpos, ctrl and prop-pose traces
               and final qpos byte for byte, from the sealed initial state, with the sealed action count

Nothing here is tolerant: a single differing byte fails. The sealed episodes are read, never written.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from data_collection.benchmark import restore_state
from experiments.gen_scene import compose
from experiments.gen_scene.conditions import NOMINAL, resolve
from experiments.gen_scene.expert import GenExpert
from experiments.props_scene import scene as props_scene

ROOT = Path(__file__).resolve().parents[2]
P2_ROOT = ROOT / "outputs/props_p2_data/props_p2_20261013"
GATE_EPISODES = (0, 1, 5)   # cereal with 2 distractors, milk with 1, lemon with 2; all admitted at their first attempt
CAMERAS = ("front_cam", "gripper_cam")


def sealed_episode(index, root=P2_ROOT):
    plan = json.loads((Path(root) / "plan.json").read_text())
    row = plan["episodes"][index]
    with np.load(Path(root) / "episodes" / f"episode_{index:03d}.npz") as npz:
        arrays = {key: npz[key] for key in ("qpos", "ctrl", "props_pose", "initial_state", "final_qpos")}
    return row, arrays


def _same(x, y):
    """Equal in shape and bits; a NaN equals a NaN (MuJoCo leaves NaNs in `mesh_polynormal`, also between two compiles of one XML)."""
    return x.shape == y.shape and np.array_equal(x, y, equal_nan=x.dtype.kind == "f")


def differing_arrays(a, b):
    """(count, names) of the compiled-model arrays of `a` and `b` that are not equal (all array attributes, bit for bit), plus
    the headlight and the physics timestep, which live in structs."""
    names = [n for n in dir(mujoco.MjModel) if not n.startswith("_") and isinstance(getattr(a, n, None), np.ndarray)]
    bad = [n for n in names if not _same(getattr(a, n), getattr(b, n))]
    structs = {f"vis.headlight.{k}": (np.asarray(getattr(a.vis.headlight, k)), np.asarray(getattr(b.vis.headlight, k)))
               for k in ("ambient", "diffuse", "specular", "active")}
    structs["opt.timestep"] = (np.asarray(a.opt.timestep), np.asarray(b.opt.timestep))
    bad += [k for k, (x, y) in structs.items() if not _same(x, y)]
    return len(names) + len(structs), bad


def render(model, data):
    out = []
    with mujoco.Renderer(model, height=512, width=512) as renderer:
        for camera in CAMERAS:
            renderer.update_scene(data, camera=camera)
            out.append(renderer.render().copy())
    if any(image.std() == 0 for image in out):
        raise ValueError("a render is blank")
    return out


def equality(index, root=P2_ROOT):
    """Gate D9 (i) on one sealed episode."""
    row, sealed = sealed_episode(index, root)
    names = [row["object"], *(d["object"] for d in row["distractors"])]
    poses = {row["object"]: tuple(row["start"]), **{d["object"]: tuple(d["pose"]) for d in row["distractors"]}}
    reference_model = props_scene.build_model(names, row["blue_slot"])[1]
    reference = props_scene.reset(reference_model, poses, row["home_pose_jitter_deg"])
    data = mujoco.MjData(reference_model)
    restore_state(reference_model, data, reference["initial_state"])
    mujoco.mj_forward(reference_model, data)
    reference_images = render(reference_model, data)
    model, _ = compose.model_for(row, NOMINAL)
    scene = compose.reset(model, resolve(row, NOMINAL)["poses"], row["home_pose_jitter_deg"])
    data = mujoco.MjData(model)
    restore_state(model, data, scene["initial_state"])
    mujoco.mj_forward(model, data)
    images = render(model, data)
    count, bad = differing_arrays(reference_model, model)
    result = dict(episode=index, object=row["object"], arrays_compared=count, differing_arrays=bad,
                  initial_state_equal_reference=bool(np.array_equal(reference["initial_state"], scene["initial_state"])),
                  initial_state_equal_sealed=bool(np.array_equal(sealed["initial_state"], scene["initial_state"])),
                  renders_byte_equal=[bool(a.tobytes() == b.tobytes()) for a, b in zip(reference_images, images)],
                  render_std=[float(i.std()) for i in images])
    result["identical"] = (not bad and result["initial_state_equal_reference"] and result["initial_state_equal_sealed"]
                           and all(result["renders_byte_equal"]))
    props_scene._built.cache_clear()
    compose.clear_cache()
    gc.collect()
    return result


def expert(index, root=P2_ROOT):
    """Gate D9 (ii) on one sealed episode."""
    row, sealed = sealed_episode(index, root)
    model, resolved = compose.model_for(row, NOMINAL)
    gen = GenExpert(model, resolved["primitive_names"])
    scene = compose.reset(model, resolved["poses"], row["home_pose_jitter_deg"])
    traces = []
    result = gen.run(sealed["initial_state"], {"object": row["object"], "bin": row["bin"], "seed": row["seed"],
                                                "home_pose_jitter_deg": list(row["home_pose_jitter_deg"])},
                     lambda frame, trace: traces.append(trace), render=False)
    mine = {key: np.stack([t[key] for t in traces]) for key in ("qpos", "ctrl", "props_pose")}
    same = {key: bool(mine[key].shape == sealed[key].shape and np.array_equal(mine[key], sealed[key])) for key in mine}
    first = {key: None for key in mine}
    for key in mine:
        if not same[key] and mine[key].shape[0] and sealed[key].shape[0]:
            n = min(len(mine[key]), len(sealed[key]))
            differing = np.flatnonzero((mine[key][:n] != sealed[key][:n]).reshape(n, -1).any(axis=1))
            first[key] = int(differing[0]) if len(differing) else n
    final_equal = bool(np.array_equal(result["final_qpos"], sealed["final_qpos"]))
    out = dict(episode=index, object=row["object"], ticks=len(traces), sealed_ticks=len(sealed["qpos"]), trace_equal=same,
               first_difference_tick=first, final_qpos_equal=final_equal, success=result["success"],
               initial_state_equal_sealed=bool(np.array_equal(sealed["initial_state"], scene["initial_state"])))
    out["identical"] = all(same.values()) and final_equal and result["success"] and out["initial_state_equal_sealed"]
    compose.clear_cache()
    gc.collect()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("gate", choices=("equality", "expert"))
    parser.add_argument("--episodes", type=int, nargs="+", default=list(GATE_EPISODES))
    parser.add_argument("--out", help="write the verdict here (a new file)")
    args = parser.parse_args(argv)
    run = equality if args.gate == "equality" else expert
    results = [run(i) for i in args.episodes]
    verdict = dict(gate=args.gate, identical=all(r["identical"] for r in results), results=results,
                   mujoco=mujoco.__version__)
    print("GATE", json.dumps(verdict), flush=True)
    if args.out:
        target = Path(args.out)
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n")
    return 0 if verdict["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
