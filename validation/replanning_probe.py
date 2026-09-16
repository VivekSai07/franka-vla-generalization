"""Evaluation-only paired horizon test and expert-observation grasp probes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def expected_chunk_calls(ticks, horizon):
    if ticks < 0 or horizon <= 0:
        raise ValueError("Invalid tick count or execution horizon")
    return (ticks + horizon - 1) // horizon


def transition_probes(actions, chunk_size):
    binary = np.asarray(actions)[:, 7] > 127
    transitions = np.flatnonzero(binary[1:] != binary[:-1]) + 1
    result = []
    for index in transitions:
        for offset in (-15, -5, -1, 0, 1, 5, 15):
            frame = int(index + offset)
            if 0 <= frame <= len(actions) - chunk_size:
                result.append({"transition": "release" if binary[index] else "close",
                               "transition_index": int(index), "offset": offset, "frame_index": frame})
    return result


def checkpoint_hashes(checkpoint):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(checkpoint.iterdir()) if p.is_file()}


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from data_collection.benchmark import require_complete, write_json
    from validation.gates import require_replay_pass, dataset_fingerprint
    from validation.smolvla_smoke import observation_only, rollout

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, checkpoint, output = args.root.resolve(), args.checkpoint.resolve(), args.output.resolve()
    manifest = require_complete(root)
    require_replay_pass(root, ROOT)
    prior = json.loads((checkpoint.parent / "result.json").read_text())
    if (prior["status"] != "complete" or prior["protocol"]["train_episodes"] != [0]
            or Path(prior["dataset_root"]) != root):
        raise ValueError("Expected the completed single-episode overfit checkpoint for this dataset")
    output.mkdir(parents=True, exist_ok=False)
    hashes = checkpoint_hashes(checkpoint)
    result = {"status": "incomplete", "checkpoint": str(checkpoint), "checkpoint_hashes": hashes,
              "dataset_root": str(root), "dataset_fingerprint": dataset_fingerprint(root),
              "episode": 0, "rollout_seed": 20260915, "horizons": [10, 1],
              "max_control_ticks": 600, "training_updates": 0, "rollouts": [],
              "purpose": "training_scene_execution_horizon_diagnostic"}
    write_json(output / "result.json", result)
    torch.set_num_threads(8)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    ds = LeRobotDataset(root.name, root=root,
                        delta_timestamps={"action": [i / manifest["fps"] for i in range(policy.config.chunk_size)]})
    ds._ensure_hf_dataset_loaded()
    ep = ds.meta.episodes[0]
    start, end = ep["dataset_from_index"], ep["dataset_to_index"]
    actions = np.stack([ds.hf_dataset[i]["action"].numpy() for i in range(start, end)])
    probes, predicted_chunks, expert_chunks = [], [], []
    with torch.inference_mode():
        for descriptor in transition_probes(actions, policy.config.chunk_size):
            index = start + descriptor["frame_index"]
            frame = ds[index]
            torch.manual_seed(4242 + index)
            predicted = postprocessor(policy.predict_action_chunk(preprocessor(observation_only(frame)))).squeeze(0).cpu().numpy()
            expert = frame["action"].cpu().numpy()
            if predicted.shape != expert.shape or not np.isfinite(predicted).all():
                raise ValueError("Invalid predicted probe chunk")
            error = np.abs(predicted - expert)
            probes.append({**descriptor, "sim_elapsed_s": descriptor["frame_index"] / manifest["fps"],
                           "arm_first_action_mae_rad": float(error[0, :7].mean()),
                           "arm_first_10_mae_rad": float(error[:10, :7].mean()),
                           "arm_first_action_error_rad": (predicted[0, :7] - expert[0, :7]).tolist(),
                           "expert_gripper_first": float(expert[0, 7]),
                           "predicted_gripper_first": float(predicted[0, 7]),
                           "first_gripper_correct": bool((predicted[0, 7] > 127) == (expert[0, 7] > 127)),
                           "first_10_gripper_accuracy": float(np.mean((predicted[:10, 7] > 127) == (expert[:10, 7] > 127)))})
            predicted_chunks.append(predicted)
            expert_chunks.append(expert)
    result["grasp_phase_probes"] = probes
    np.savez_compressed(output / "grasp_phase_chunks.npz", predicted=np.array(predicted_chunks), expert=np.array(expert_chunks))
    write_json(output / "result.json", result)
    print("GRASP_PHASE_PROBES", probes, flush=True)

    for horizon in (10, 1):
        policy.config.n_action_steps = horizon
        policy.reset()  # Recreate the action deque with the selected max length.
        directory = output / f"horizon_{horizon}"
        directory.mkdir(exist_ok=False)
        calls = 0
        get_chunk = policy._get_action_chunk

        def counted_chunk(*arguments, **kwargs):
            nonlocal calls
            calls += 1
            return get_chunk(*arguments, **kwargs)

        # Observe actual calls, not an estimate based on requested configuration.
        policy._get_action_chunk = counted_chunk
        try:
            metrics = rollout(policy, preprocessor, postprocessor, manifest, root, directory, episode_ids=[0])[0]
        finally:
            policy._get_action_chunk = get_chunk
        if calls != expected_chunk_calls(metrics["ticks"], horizon):
            raise AssertionError("Actual replanning count differs from configured horizon")
        metrics.update(n_action_steps=horizon, predicted_chunk_size=policy.config.chunk_size,
                       actual_chunk_predictions=calls, rollout_seed=20260915)
        result["rollouts"].append(metrics)
        write_json(output / "result.json", result)
        print("PAIRED_RESULT", metrics, flush=True)

    # The first physical action must be identical: same initialization and first noise draw.
    with np.load(output / "horizon_10/rollout_0_trace.npz") as a, np.load(output / "horizon_1/rollout_0_trace.npz") as b:
        np.testing.assert_array_equal(a["state"][0], b["state"][0])
        np.testing.assert_allclose(a["raw_action"][0], b["raw_action"][0], rtol=0, atol=1e-6)
        result["first_action_max_difference"] = float(np.max(np.abs(a["raw_action"][0] - b["raw_action"][0])))
    with np.load(checkpoint.parent / "rollout_0_trace.npz") as previous, np.load(output / "horizon_10/rollout_0_trace.npz") as current:
        # Diagnostic only: device nondeterminism can prevent a bit-identical prior rerun.
        if previous["raw_action"].shape == current["raw_action"].shape:
            result["horizon_10_vs_previous_max_action_difference"] = float(np.max(np.abs(previous["raw_action"] - current["raw_action"])))
            result["horizon_10_vs_previous_max_state_difference"] = float(np.max(np.abs(previous["state"] - current["state"])))
    if hashes != checkpoint_hashes(checkpoint):
        raise AssertionError("Checkpoint changed during evaluation")
    require_replay_pass(root, ROOT)
    result["status"] = "complete"
    write_json(output / "result.json", result)
    print("Paired evaluation complete; no training performed", flush=True)


if __name__ == "__main__":
    main()