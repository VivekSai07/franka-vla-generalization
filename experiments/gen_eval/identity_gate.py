"""The live identity gate of G0 (spec D9 iii): with the nominal condition, `gen_eval` re-runs two of A2's recorded M0 comparison
trials, bit for bit.

M0's comparison trials 1 and 291 of arm A2 (checkpoint 88,830) are re-run by THIS package's rollout (`default_runner`: the scene
from the randomization engine, the forked rollout loop) with A2's own adapter, the M0 layouts given directly, the nominal
condition and the M0 tick cap. Every trace array is compared with A2's recorded `trace.npz` byte for byte (dtype, shape and
bytes; a NaN equals a NaN in the same place), as are the outcome fields of `episode.json` (everything M0 recorded except the wall
clock). Anything else is a failed gate and the run stops: until the fork is proven to change nothing under nominal, no G0 trial
counts. The video is recorded, not compared (encoders may differ in container details).

The gate runs under the runtime of M0's worker (`configure_runtime`: offline mode, the cuBLAS workspace, deterministic torch
algorithms) and, from the driver, with the comparison workers' CPU thread count (FAILURES #70: a bit-for-bit gate is only a gate
if the harness reproduces the reference's runtime flags). M0's campaign is mounted read-only and only read; the re-runs and the
verdict go to a scratch `--out` (`<out>/identity_gate/`), which must not be the campaign. Robosuite image with a GPU, from the
dedicated worktree, through `scripts/gen_eval.sh identity_gate`. Adapted from `vla_inputs_m3b_analysis/identity_gate.py`.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.gen_eval import evaluate, protocol
from experiments.vla_inputs_eval import adapters
from experiments.vla_inputs_eval import evaluate as m0_evaluate
from experiments.vla_inputs_eval import protocol as m0_protocol

IDENTIFIER = "experiments.vla_inputs_arm_xvla_force.adapter:XVLAForceAdapter"
ARM, CHECKPOINT, ORDINALS = "A2", 88830, (1, 291)
M0_CORE_SHA256 = "c4f6c303f2ae37509530323efe7d190686b1b5258910863f041241d30bc5b39d"
IGNORED_EPISODE_FIELDS = ("wall_seconds",)
OUTCOME_FIELDS = ("outcome", "task_success", "ticks", "placed_bin", "final_gripper_open", "cause", "first_lifted", "wrong_object",
                  "distractor_displacement_m", "names")


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and (a.tobytes() == b.tobytes() or (a.dtype.kind == "f" and np.array_equal(a, b, equal_nan=True)))


def _first_difference_tick(a, b):
    if a.shape != b.shape or a.ndim == 0:
        return 0
    differing = (a != b) if a.dtype.kind != "f" else ~((a == b) | (np.isnan(a) & np.isnan(b)))
    ticks = np.flatnonzero(differing.reshape(len(a), -1).any(axis=1)) if len(a) else []
    return int(ticks[0]) if len(ticks) else 0


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_trial(recorded, rerun):
    """Compare two trial folders: every trace array byte for byte (the same arrays on both sides), the recorded `episode.json`'s
    fields (all but the wall clock) against the re-run's, the video recorded only."""
    with np.load(Path(recorded) / "trace.npz") as a, np.load(Path(rerun) / "trace.npz") as b:
        keys_a, keys_b = set(a.files), set(b.files)
        arrays = {key: bool(_same(a[key], b[key])) for key in sorted(keys_a & keys_b)}
        differing = [key for key, same in arrays.items() if not same]
        first = min((_first_difference_tick(a[key], b[key]) for key in differing), default=None)
    episode_a = json.loads((Path(recorded) / "episode.json").read_text())
    episode_b = json.loads((Path(rerun) / "episode.json").read_text())
    shared = [k for k in episode_a if k not in IGNORED_EPISODE_FIELDS]
    missing = [k for k in shared if k not in episode_b]
    fields = sorted(k for k in shared if k in episode_b and episode_a[k] != episode_b[k])
    mismatch = sorted(keys_a ^ keys_b)
    video = [Path(f) / "cameras.mp4" for f in (recorded, rerun)]
    return dict(identical=not differing and not mismatch and not fields and not missing and all(k in episode_a for k in OUTCOME_FIELDS),
                arrays=arrays, differing_arrays=differing, key_mismatch=mismatch, first_difference_tick=first,
                differing_episode_fields=fields, episode_fields_missing_from_rerun=missing,
                video_identical=all(v.is_file() for v in video) and _sha256(video[0]) == _sha256(video[1]))


def exit_code(result):
    return 0 if result["identical"] else 1


def _read(path):
    return json.loads(Path(path).read_text())


def run(campaign, out, checkpoint=CHECKPOINT, ordinals=ORDINALS, *, repo=evaluate.ROOT, verify=True, factory=None, runner=None,
        manifests=None, evaluate_module=m0_evaluate):
    """Re-run `ordinals` of A2's M0 comparison tier at `checkpoint` through gen_eval's rollout, nominal, and compare each trial."""
    campaign, out = Path(campaign), Path(out)
    if campaign.resolve() == out.resolve():
        raise ValueError("--out must be a scratch directory, not the campaign")
    prepared = _read(campaign / "arms" / ARM / "prepared.json")
    if prepared["core"]["sha256"] != M0_CORE_SHA256:
        raise ValueError(f"{ARM} was prepared with a shared core other than the pinned {M0_CORE_SHA256[:12]}")
    if str(checkpoint) not in prepared["checkpoints"]:
        raise ValueError(f"checkpoint {checkpoint} was not prepared for {ARM}")
    if prepared["arm"]["adapter"] != IDENTIFIER:
        raise ValueError(f"{ARM} was prepared with adapter {prepared['arm']['adapter']}, not {IDENTIFIER}")
    if verify:
        evaluate_module.verify_core(prepared, repo)
        evaluate_module.verify_manifests(campaign, prepared)
        evaluate_module.verify_arm(prepared, repo)
        evaluate_module.verify_checkpoint(prepared, checkpoint)
    plan, feasibility = (manifests or m0_evaluate.load_manifests)(campaign)
    by_ordinal = {t["ordinal"]: t for t in m0_protocol.stage_trials(plan, feasibility, "comparison")}
    unknown = [o for o in ordinals if o not in by_ordinal]
    if unknown:
        raise ValueError(f"no comparison trial with ordinal {unknown} (a dropped layout leaves a gap)")
    recorded_dir = m0_protocol.stage_dir(campaign, ARM, "comparison", checkpoint)
    for ordinal in ordinals:
        if not m0_protocol.trial_is_complete(m0_protocol.trial_folder(recorded_dir, by_ordinal[ordinal])):
            raise RuntimeError(f"{ARM}'s recorded trial {ordinal} is not complete in {recorded_dir}")
    gate = out / "identity_gate"
    gate.mkdir(parents=True)
    adapter = (factory or (lambda p, c: adapters.load_adapter(IDENTIFIER, Path(p["checkpoints"][str(c)]["path"]))))(prepared, checkpoint)
    adapters.check_adapter(adapter)
    state_builder = getattr(adapter, "state_builder", adapters.state_p2)
    run_trial = runner or evaluate.default_runner
    trials = []
    for ordinal in ordinals:
        recorded_trial = by_ordinal[ordinal]
        layout = plan["layouts"][recorded_trial["layout_id"]]
        trial = protocol.trial_of(dict(layout, axes={}), "nominal", ordinal)
        folder = protocol.trial_folder(gate, trial)
        folder.mkdir()
        run_trial(adapter, state_builder, layout, trial, folder)
        verdict = compare_trial(m0_protocol.trial_folder(recorded_dir, recorded_trial), folder)
        trials.append(dict(ordinal=ordinal, layout_id=trial["layout_id"], **verdict))
        print("IDENTITY_TRIAL", ordinal, trial["layout_id"], "identical" if verdict["identical"] else f"DIFFERENT {verdict['differing_arrays']}", flush=True)
    result = dict(identical=all(t["identical"] for t in trials), adapter=IDENTIFIER, arm=ARM, checkpoint=checkpoint, ordinals=list(ordinals),
                  m0_core_sha256=prepared["core"]["sha256"], trials=trials)
    (gate / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("IDENTITY_GATE", "PASSED" if result["identical"] else "FAILED", flush=True)
    return result


def configure_runtime():
    """What M0's `worker` sets before any trial (`vla_inputs_eval.evaluate.main`): offline mode, the cuBLAS workspace and deterministic
    torch algorithms. Without them the policy's tick-0 action already differs from the recorded one (FAILURES #70)."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign", required=True, help="M0's campaign directory (read only)")
    parser.add_argument("--out", required=True, help="a scratch directory for the re-runs and the verdict")
    parser.add_argument("--checkpoint", type=int, default=CHECKPOINT)
    parser.add_argument("--ordinals", type=int, nargs="+", default=list(ORDINALS))
    args = parser.parse_args(argv)
    configure_runtime()
    return exit_code(run(args.campaign, args.out, args.checkpoint, tuple(args.ordinals)))


if __name__ == "__main__":
    raise SystemExit(main())
