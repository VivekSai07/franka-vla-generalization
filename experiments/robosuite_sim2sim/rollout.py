"""Stage B (robosuite image, GPU): roll the MuJoCo-trained B2 checkpoint on two scenes, paired.

Arm "mujoco": the scene the policy was trained on (`SCENE_PATH`). Arm "robosuite": the robosuite
bring-up scene. Both arms share the starts exported by `starts.py`, the rollout seed, the policy,
the processors and this loop, and both run on this image's MuJoCo, so the scene is the only
difference between the arms. The loop mirrors `xvla_b2_distractors_eval.evaluate._rollout`.
"""
import argparse
import json
import time
from pathlib import Path

import av
import mujoco
import numpy as np
import torch

SCENES = ("mujoco", "robosuite")
MAX_TICKS = 1200
CUBES = ("red", "green")


def _cube_address(model, color):
    return int(model.jnt_qposadr[model.body(f"target_cube_{color}").jntadr[0]])


def mujoco_start(model, qpos, qvel, ctrl):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, ctrl
    mujoco.mj_forward(model, data)
    return data


def robosuite_start(model, source_model, qpos, ctrl):
    """Same arm pose and cube poses as the MuJoCo start, settled on the robosuite table."""
    from data_collection.benchmark import synchronize_observation
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:9] = qpos[:9]
    for color in CUBES:
        src, dst = _cube_address(source_model, color), _cube_address(model, color)
        data.qpos[dst:dst + 7] = qpos[src:src + 7]
    data.ctrl[:] = ctrl
    for _ in range(round(0.2 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    synchronize_observation(model, data)
    return data


def rollout(policy, pre, post, model, data, renderer, trial, cameras, folder):
    from data_collection.benchmark import apply_action, step_control, synchronize_observation
    from experiments.xvla_b2_distractors import contract as c
    from experiments.xvla_b2_distractors_eval.evaluate import seed_trial
    from src.controllers.grasp_controller import GraspController
    from src.robots.franka_panda.config import ACTIVE_JOINTS, ARM_DOF
    from validation.evaluate_pilot import advance_streak
    from validation.gates import fresh_stack_metrics
    from validation.rollout_diagnostics import bound_action, target_diagnostics

    pick, place = trial["pick"], trial["place"]
    scratch = mujoco.MjData(model)
    policy.reset()
    seed_trial(trial["seed"])
    grasp = GraspController(model, data, target=f"target_cube_{pick}")
    synchronize_observation(model, data)
    initial_z = float(data.xpos[grasp.target_id, 2])
    trace = {key: [] for key in ("state", "raw_action", "applied_action", "pick_position", "place_position",
                                 "pick_lift_m", "bilateral_target_contact", "released_stack", "xy_m", "z_m")}
    stable, success, outcome, started = 0, False, "timeout", time.monotonic()
    with av.open(str(folder / "cameras.mp4"), "w") as video:
        stream = video.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 1024, 512, "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}

        def observe():
            images = []
            for camera in cameras:
                renderer.update_scene(data, camera=camera)
                images.append(renderer.render().copy())
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.concatenate(images, axis=1), format="rgb24")):
                video.mux(packet)
            return images

        for _ in range(MAX_TICKS):
            synchronize_observation(model, data)
            state = data.qpos[:9].copy()
            frame = {"observation.state": torch.tensor(state, dtype=torch.float32), "task": trial["instruction"]}
            for key, image in zip(("observation.images.image", "observation.images.image2"), observe()):
                frame[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
            with c.approved_instruction(trial["instruction"]):
                processed = pre(c.observation(frame))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                action = post(policy.select_action(processed)).squeeze(0).float().cpu().numpy()
            if action.shape != (8,):
                raise ValueError("Policy action shape is not eight physical commands")
            trace["state"].append(state)
            trace["raw_action"].append(action.copy())
            if not np.isfinite(action).all():
                outcome = "nonfinite_action"
                break
            bounded, _ = bound_action(action, model.actuator_ctrlrange[list(ACTIVE_JOINTS)])
            apply_action(model, data, bounded, ACTIVE_JOINTS)
            if step_control(model, data, 1 / 30, ARM_DOF) != 20:
                raise ValueError("Physics substep count changed")
            metrics = fresh_stack_metrics(model, data, scratch, pick, place)
            diagnostics = target_diagnostics(model, data, pick, place, grasp, initial_z)
            trace["applied_action"].append(bounded.copy())
            for key in ("pick_position", "place_position", "pick_lift_m", "bilateral_target_contact"):
                trace[key].append(diagnostics[key])
            for key in ("released_stack", "xy_m", "z_m"):
                trace[key].append(metrics[key])
            stable, success = advance_streak(stable, metrics)
            if success:
                outcome = "success"
                break
        observe()
        for packet in stream.encode():
            video.mux(packet)
    np.savez_compressed(folder / "trace.npz", **{key: np.asarray(value) for key, value in trace.items()})
    lift = np.asarray(trace["pick_lift_m"]) if trace["pick_lift_m"] else np.zeros(1)
    result = dict(trial, outcome=outcome, task_success=bool(success), ticks=len(trace["raw_action"]),
                  max_pick_lift_m=float(lift.max()), ever_bilateral_contact=bool(np.any(trace["bilateral_target_contact"])),
                  final_xy_m=float(trace["xy_m"][-1]) if trace["xy_m"] else None,
                  final_z_m=float(trace["z_m"][-1]) if trace["z_m"] else None,
                  wall_seconds=time.monotonic() - started)
    (folder / "episode.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def summarize(rows):
    table = {scene: {r["ordinal"]: r["task_success"] for r in rows if r["scene"] == scene} for scene in SCENES}
    out = {scene: dict(trials=len(done), successes=sum(done.values())) for scene, done in table.items() if done}
    both = sorted(set(table["mujoco"]) & set(table["robosuite"]))
    if both:
        out["paired"] = dict(
            trials=len(both),
            both=sum(table["mujoco"][o] and table["robosuite"][o] for o in both),
            mujoco_only=sum(table["mujoco"][o] and not table["robosuite"][o] for o in both),
            robosuite_only=sum(table["robosuite"][o] and not table["mujoco"][o] for o in both),
            neither=sum(not table["mujoco"][o] and not table["robosuite"][o] for o in both))
    mujoco_rows = [r for r in rows if r["scene"] == "mujoco"]
    if mujoco_rows:
        out["mujoco_arm_agrees_with_b2_report"] = sum(r["task_success"] == r["b2_task_success"] for r in mujoco_rows)
    return out


def run(run_dir, scenes, limit):
    from lerobot.policies.factory import make_pre_post_processors
    from data_collection.benchmark import configure_timing
    from experiments.robosuite_scene.contract import MUJOCO_VERSION
    from experiments.xvla_b2_distractors import contract as c
    from experiments.xvla_b2_distractors_eval.evaluate import load_counted
    from src.robots.franka_panda.config import SCENE_PATH

    if mujoco.__version__ != MUJOCO_VERSION:
        raise RuntimeError(f"Run in the robosuite image (MuJoCo {MUJOCO_VERSION}), not {mujoco.__version__}")
    run_dir = Path(run_dir)
    plan = json.loads((run_dir / "trials.json").read_text())
    trials = plan["trials"][:limit] if limit else plan["trials"]
    starts = np.load(run_dir / "starts.npz")
    rollouts = run_dir / "rollouts"
    rollouts.mkdir(exist_ok=False)

    checkpoint = c.OUTPUT / "checkpoints" / c.checkpoint_directory(plan["checkpoint"])
    torch.use_deterministic_algorithms(True)
    policy = load_counted(checkpoint)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))

    source = mujoco.MjModel.from_xml_path(SCENE_PATH)
    configure_timing(source, 30)
    rows = []

    def arm(scene, model, start):
        # One renderer per arm, opened after the previous arm's is closed: OSMesa contexts conflict.
        with mujoco.Renderer(model, height=512, width=512) as renderer:
            for trial in trials:
                key = f"t{trial['ordinal']:03d}"
                data = start(model, starts[f"{key}_qpos"], starts[f"{key}_qvel"], starts[f"{key}_ctrl"])
                folder = rollouts / f"{scene}_{key}_{trial['layout_id']}"
                folder.mkdir()
                row = rollout(policy, pre, post, model, data, renderer, dict(trial, scene=scene),
                              plan["camera_names"], folder)
                rows.append(row)
                (run_dir / "result.json").write_text(json.dumps(dict(
                    checkpoint=plan["checkpoint"], mujoco=mujoco.__version__, starts_mujoco=plan["mujoco"],
                    summary=summarize(rows), episodes=rows), indent=2) + "\n")
                print("SIM2SIM", scene, key, trial["layout_id"], row["outcome"], row["ticks"],
                      f"lift={row['max_pick_lift_m']:.3f}", f"b2={trial['b2_task_success']}", flush=True)

    if "mujoco" in scenes:
        arm("mujoco", source, mujoco_start)
    if "robosuite" in scenes:
        from experiments.robosuite_scene.scene import build_sim
        _, model, _ = build_sim()
        arm("robosuite", model, lambda m, qpos, qvel, ctrl: robosuite_start(m, source, qpos, ctrl))
    print("SIM2SIM_DONE", json.dumps(summarize(rows)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="directory written by starts.py")
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--limit", type=int, default=0, help="first N trials only (smoke)")
    args = parser.parse_args()
    run(args.run, args.scenes, args.limit)
