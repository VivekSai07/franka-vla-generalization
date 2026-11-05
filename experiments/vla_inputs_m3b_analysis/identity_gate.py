"""The live identity gate of M3b (spec D7 iv): with every corruption off, A3's comparison trials 1 and 291 reproduce bit for bit.

`XVLANoCorruption` (A3's frozen policy and force state, the oracle box drawn unchanged) re-runs the two recorded comparison trials with
M0's own rollout and every trace array is compared with A3's recorded `trace.npz` byte for byte (dtype, shape and bytes; a NaN equals a
NaN in the same place), as are `episode.json`'s fields except the wall clock. Anything else is a failed gate and the run stops: until
the only difference from A3 is proven to be the box, no arm A5-A9 trial counts. The video is recorded, not compared (encoders may
differ in container details).

The campaign (A3's recorded stage, the plan, the feasibility record) is mounted read-only and only read; the re-runs and the verdict
go to a scratch `--out` (`<out>/identity_gate/`), which must not be the campaign. Robosuite image with a GPU, from the dedicated
worktree, through `scripts/vla_inputs_m3b_eval.sh identity_gate`. Not part of an arm's closure or of M0's core.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.vla_inputs_eval import adapters, evaluate, protocol

IDENTIFIER = "experiments.vla_inputs_arm_xvla_force_box_noisy.adapter:XVLANoCorruption"
ARM, CHECKPOINT, ORDINALS = "A3", 101520, (1, 291)
CORE_SHA256 = "c4f6c303f2ae37509530323efe7d190686b1b5258910863f041241d30bc5b39d"
IGNORED_EPISODE_FIELDS = ("wall_seconds",)


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
    """Compare two trial folders: every trace array byte for byte, `episode.json` without the wall clock, the video recorded only."""
    with np.load(Path(recorded) / "trace.npz") as a, np.load(Path(rerun) / "trace.npz") as b:
        keys_a, keys_b = set(a.files), set(b.files)
        arrays = {key: bool(_same(a[key], b[key])) for key in sorted(keys_a & keys_b)}
        differing = [key for key, same in arrays.items() if not same]
        first = min((_first_difference_tick(a[key], b[key]) for key in differing), default=None)
    episode_a = json.loads((Path(recorded) / "episode.json").read_text())
    episode_b = json.loads((Path(rerun) / "episode.json").read_text())
    fields = sorted(k for k in set(episode_a) | set(episode_b) if k not in IGNORED_EPISODE_FIELDS and episode_a.get(k) != episode_b.get(k))
    mismatch = sorted(keys_a ^ keys_b)
    video = [Path(f) / "cameras.mp4" for f in (recorded, rerun)]
    return dict(identical=not differing and not mismatch and not fields, arrays=arrays, differing_arrays=differing, key_mismatch=mismatch,
                first_difference_tick=first, differing_episode_fields=fields,
                video_identical=all(v.is_file() for v in video) and _sha256(video[0]) == _sha256(video[1]))


def exit_code(result):
    return 0 if result["identical"] else 1


def _read(path):
    return json.loads(Path(path).read_text())


def run(campaign, out, checkpoint=CHECKPOINT, ordinals=ORDINALS, *, repo=evaluate.ROOT, verify=True, factory=None, runner=None,
        manifests=None, evaluate_module=evaluate):
    """Re-run `ordinals` of A3's comparison tier at `checkpoint` with the identity adapter and compare each with A3's recorded trial."""
    campaign, out = Path(campaign), Path(out)
    if campaign.resolve() == out.resolve():
        raise ValueError("--out must be a scratch directory, not the campaign")
    prepared = _read(campaign / "arms" / ARM / "prepared.json")
    if prepared["core"]["sha256"] != CORE_SHA256:
        raise ValueError(f"{ARM} was prepared with a shared core other than the pinned {CORE_SHA256[:12]}")
    if str(checkpoint) not in prepared["checkpoints"]:
        raise ValueError(f"checkpoint {checkpoint} was not prepared for {ARM}")
    if verify:
        evaluate_module.verify_core(prepared, repo)
        evaluate_module.verify_manifests(campaign, prepared)
        evaluate_module.verify_arm(prepared, repo)
        evaluate_module.verify_checkpoint(prepared, checkpoint)
    plan, feasibility = (manifests or evaluate.load_manifests)(campaign)
    by_ordinal = {t["ordinal"]: t for t in protocol.stage_trials(plan, feasibility, "comparison")}
    unknown = [o for o in ordinals if o not in by_ordinal]
    if unknown:
        raise ValueError(f"no comparison trial with ordinal {unknown} (a dropped layout leaves a gap)")
    recorded_dir = protocol.stage_dir(campaign, ARM, "comparison", checkpoint)
    for ordinal in ordinals:
        if not protocol.trial_is_complete(protocol.trial_folder(recorded_dir, by_ordinal[ordinal])):
            raise RuntimeError(f"{ARM}'s recorded trial {ordinal} is not complete in {recorded_dir}")
    gate = out / "identity_gate"
    gate.mkdir(parents=True)
    adapter = (factory or (lambda p, c: adapters.load_adapter(IDENTIFIER, Path(p["checkpoints"][str(c)]["path"]))))(prepared, checkpoint)
    adapters.check_adapter(adapter)
    state_builder = getattr(adapter, "state_builder", adapters.state_p2)
    run_trial = runner or evaluate.default_runner
    trials = []
    for ordinal in ordinals:
        trial = by_ordinal[ordinal]
        folder = protocol.trial_folder(gate, trial)
        folder.mkdir()
        run_trial(adapter, state_builder, plan["layouts"][trial["layout_id"]], trial, folder)
        verdict = compare_trial(protocol.trial_folder(recorded_dir, trial), folder)
        trials.append(dict(ordinal=ordinal, layout_id=trial["layout_id"], **verdict))
        print("IDENTITY_TRIAL", ordinal, trial["layout_id"], "identical" if verdict["identical"] else f"DIFFERENT {verdict['differing_arrays']}", flush=True)
    result = dict(identical=all(t["identical"] for t in trials), adapter=IDENTIFIER, arm=ARM, checkpoint=checkpoint, ordinals=list(ordinals),
                  core_sha256=prepared["core"]["sha256"], trials=trials)
    (gate / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("IDENTITY_GATE", "PASSED" if result["identical"] else "FAILED", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign", required=True, help="M0's campaign directory (read only)")
    parser.add_argument("--out", required=True, help="a scratch directory for the re-runs and the verdict")
    parser.add_argument("--checkpoint", type=int, default=CHECKPOINT)
    parser.add_argument("--ordinals", type=int, nargs="+", default=list(ORDINALS))
    args = parser.parse_args(argv)
    return exit_code(run(args.campaign, args.out, args.checkpoint, tuple(args.ordinals)))


if __name__ == "__main__":
    raise SystemExit(main())
