"""The detector gate (spec D5), applied before any A5 trial. A failed gate means no A5 run; thresholds are in `constants.py` and never move.

Three measurements, per camera:

(a) **decoded frames**: the 60 validation episodes' cached frames (video-decoded, as in training): mean IoU with S0's box and the share of
    frames below IoU 0.5; an absent prediction scores 0;
(b) **raw renders**: the same validation frames replayed exactly (S0's `replay.frames`) and rendered 512 x 512 as a rollout renders them,
    through the very `Predictor` an arm A5 rollout uses: the same two numbers (this measures the video-codec shift);
(c) **synthetic out-of-view frames**: validation frames translated by exact shifts (the training augmentation's own functions, a fixed seed),
    labelled present or absent by what remains of the box, then the smaller of the F1 of the "present" and of the "absent" class.

Everything reads a `predictor.predict_arrays`-like object; `decide` applies the thresholds and records every number next to its threshold.
"""
import numpy as np

from experiments.vla_inputs_box_detector import constants as k
from experiments.vla_inputs_box_detector import data, inputs, model

CAMERAS = ("front", "gripper")


def iou(predicted, truth):
    """IoU of two (5,) boxes in the full frame; a prediction that is absent (or has no area) scores 0 against a present truth."""
    predicted, truth = np.asarray(predicted, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if predicted[4] == 0.0 or truth[4] == 0.0:
        return 1.0 if predicted[4] == truth[4] else 0.0
    return model.pixel_iou(predicted[:4], truth[:4])


def _summary(ious, predicted_absent):
    ious = np.asarray(ious, dtype=np.float64)
    return dict(frames=int(len(ious)), mean_iou=float(ious.mean()), share_below_half=float((ious < k.GATE_IOU_FLOOR).mean()),
                share_predicted_absent=float(np.mean(predicted_absent)))


def _batches(rows, batch):
    for start in range(0, len(rows), batch):
        yield rows[start:start + batch]


def decoded_report(predictor, cache, rows, batch=32):
    """(a): per camera, IoU of the predictor's boxes against S0's on the cached validation frames `rows`."""
    rows = np.asarray(rows)
    if not np.isin(cache.index[rows, 0], k.VALIDATION_EPISODES).all():
        raise ValueError("the gate measures validation episodes only")
    ious, absent = {0: [], 1: []}, {0: [], 1: []}
    for chunk in _batches(rows, batch):
        boxes, _ = predictor.predict_arrays(np.asarray(cache.front[chunk]), np.asarray(cache.gripper[chunk]), cache.objects[chunk])
        for camera in (0, 1):
            for i, row in enumerate(chunk):
                ious[camera].append(iou(boxes[i, camera], cache.labels[row, camera]))
                absent[camera].append(boxes[i, camera, 4] == 0.0)
    return {name: _summary(ious[c], absent[c]) for c, name in enumerate(CAMERAS)}


def merge_raw(records):
    """(b): merge per-episode records `{episode, front: [IoU...], gripper: [IoU...]}` (from the replay shards) into one report."""
    episodes = [r["episode"] for r in records]
    if len(set(episodes)) != len(episodes):
        raise ValueError("an episode appears twice among the raw-render records")
    out = {}
    for camera in CAMERAS:
        values = np.array([v for r in records for v in r[camera]], dtype=np.float64)
        out[camera] = dict(frames=int(len(values)), mean_iou=float(values.mean()), share_below_half=float((values < k.GATE_IOU_FLOOR).mean()))
    out["episodes"] = sorted(episodes)
    return out


def raw_report(predictor, episodes, log=print):
    """(b), one shard: replay each validation episode exactly, render both cameras at the cached frames as a rollout does, predict from the
    raw renders and the plan's instruction, and score against S0's boxes. Robosuite image."""
    import mujoco
    from experiments.vla_inputs_eval.rollout import CAMERA_NAMES
    from experiments.vla_inputs_signals import replay
    from experiments.xvla_p2_force_box import boxes, contract
    table, instructions, records = boxes.table(), contract.planned_instructions(), []
    for episode in episodes:
        scores = {name: [] for name in CAMERAS}
        holder = {}

        def on_frame(tick, sim_model, sim_data, episode=episode, scores=scores, holder=holder):
            if tick not in k.CACHE_FRAMES:
                return
            if "renderer" not in holder:
                holder["renderer"] = mujoco.Renderer(sim_model, height=512, width=512)
            images = []
            for camera in CAMERA_NAMES:
                holder["renderer"].update_scene(sim_data, camera=camera)
                images.append(holder["renderer"].render().copy())
            predicted = predictor.predict(images, instructions[episode])
            for camera, name in enumerate(CAMERAS):
                scores[name].append(iou(predicted[camera], table[episode, tick, camera]))

        try:
            replay.frames(episode, on_frame)
        finally:
            if "renderer" in holder:
                holder["renderer"].close()
        records.append(dict(episode=episode, **scores))
        log("RAW_EPISODE", episode, {name: round(float(np.mean(v)), 4) for name, v in scores.items()}, flush=True)
    return records


def f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def synthetic_set(cache, rows, seed=k.SYNTHETIC_SEED):
    """Validation frames translated by exact shifts (the training augmentation's functions), deterministic from `seed`: per camera, about
    `SYNTHETIC_ABSENT_FRACTION` of the frames are pushed until the box is absent where that is reachable within the pinned shift range;
    every label is whatever remains after the shift, never a hoped-for one. Returns frames and `present` (N, 2) bool."""
    rows = np.asarray(rows)
    fronts, grippers = np.empty((len(rows), *k.FRONT_SHAPE, 3), np.uint8), np.empty((len(rows), *k.GRIPPER_SHAPE, 3), np.uint8)
    present = np.zeros((len(rows), 2), dtype=bool)
    specs = ((0, cache.front, k.FRONT_SHAPE, inputs.front_box_to_input, fronts), (1, cache.gripper, k.GRIPPER_SHAPE, inputs.gripper_box_to_input, grippers))
    for i, row in enumerate(rows):
        for camera, frames, shape, to_input, out in specs:
            rng = np.random.default_rng([seed, int(row), camera])
            want_absent = rng.random() < k.SYNTHETIC_ABSENT_FRACTION
            box, chosen = to_input(cache.labels[row, camera]), None
            for _ in range(200):
                dx = int(rng.integers(-int(k.SHIFT_MAX_FRACTION * shape[1]), int(k.SHIFT_MAX_FRACTION * shape[1]) + 1))
                dy = int(rng.integers(-int(k.SHIFT_MAX_FRACTION * shape[0]), int(k.SHIFT_MAX_FRACTION * shape[0]) + 1))
                _, visible = inputs.shift_box(box, dx, dy, shape)
                chosen = (dx, dy, visible)
                if visible != want_absent:
                    break
            dx, dy, visible = chosen
            out[i] = inputs.shift_image(np.asarray(frames[row]), dx, dy)
            present[i, camera] = visible
    return dict(front=fronts, gripper=grippers, present=present, cls=np.asarray(cache.objects[rows]))


def synthetic_report(predictor, cache, rows, batch=32):
    """(c): per camera, the F1 of the present and of the absent class and the smaller of the two."""
    rows = np.asarray(rows)
    synthetic = synthetic_set(cache, rows)
    said = np.zeros_like(synthetic["present"])
    for start in range(0, len(rows), batch):
        part = slice(start, start + batch)
        boxes, _ = predictor.predict_arrays(synthetic["front"][part], synthetic["gripper"][part], synthetic["cls"][part])
        said[part] = boxes[:, :, 4] == 1.0
    out = {}
    for camera, name in enumerate(CAMERAS):
        truth, pred = synthetic["present"][:, camera], said[:, camera]
        tp, fp, fn, tn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum()), int((~truth & ~pred).sum())
        if tp + fn == 0 or tn + fp == 0:
            raise ValueError(f"the synthetic set lacks a class for the {name} camera (present {tp + fn}, absent {tn + fp})")
        present_f1, absent_f1 = f1(tp, fp, fn), f1(tn, fn, fp)
        out[name] = dict(tp=tp, fp=fp, fn=fn, tn=tn, f1_present=present_f1, f1_absent=absent_f1, min_f1=min(present_f1, absent_f1))
    return out


def decide(decoded, raw, synthetic):
    """Apply the pinned thresholds; every check is recorded with its value and threshold. A missing measurement fails the gate."""
    checks = []
    for label, report in (("decoded", decoded), ("raw", raw)):
        for camera in CAMERAS:
            entry = None if report is None else report[camera]
            checks.append(dict(name=f"{label} {camera} mean iou", value=None if entry is None else entry["mean_iou"], threshold=k.GATE_MEAN_IOU,
                               ok=entry is not None and entry["mean_iou"] >= k.GATE_MEAN_IOU))
            checks.append(dict(name=f"{label} {camera} share of frames below iou {k.GATE_IOU_FLOOR}",
                               value=None if entry is None else entry["share_below_half"], threshold=k.GATE_SHARE_BELOW_HALF,
                               ok=entry is not None and entry["share_below_half"] <= k.GATE_SHARE_BELOW_HALF))
    for camera in CAMERAS:
        entry = None if synthetic is None else synthetic[camera]
        checks.append(dict(name=f"synthetic {camera} presence f1 (smaller class)", value=None if entry is None else entry["min_f1"],
                           threshold=k.GATE_PRESENCE_F1, ok=entry is not None and entry["min_f1"] >= k.GATE_PRESENCE_F1))
    return dict(passed=all(c["ok"] for c in checks), checks=checks)
