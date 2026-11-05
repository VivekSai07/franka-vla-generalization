# vla_inputs_box_detector: the box from pixels (M3b, arm A5)

M3b ([spec](../../docs/superpowers/specs/2026-11-05-vla-inputs-m3b-predicted-box-design.md) D4, D5, D6) asks how much of A3's gain over A2 survives
when the box drawn on the images is predicted from the images instead of read from the simulator. This package is the predictor: a
closed-set detector that, given the two camera renders and the instruction, returns S0's `(u_min, v_min, u_max, v_max, in_frame)` for the
object the instruction names, in both cameras. **No policy is trained**; A3's policy is used as it is (arm A5, `experiments/vla_inputs_arm_xvla_force_box_pred`).

It is hash-bound (`contract.SOURCE_FILES`, this README included): do not edit anything here after `--check`. A retrained or changed detector is a
**new arm** reported as such, never a replacement for a detector A5 was run with.

## What it sees and what it must output

| camera | input to the network | why |
|---|---|---|
| front | a fixed native-resolution crop of the 512 x 512 render: rows 128..415, columns 16..495 (288 x 480) | all 540 sealed episodes' target boxes lie within rows 175..365, columns 51..476; the object is small (median 27 px wide), so resolution matters |
| gripper | the render averaged 2 x 2 to 256 x 256 | its boxes are large (median 128 px) and half of them touch the image edge |

Output per camera: the box in [0, 1] of the full frame, clipped, or the absent box (all zeros) when the peak logit of the instructed object's channel
is below the presence threshold (sigmoid 0.5). The instruction is parsed from `put the <object> in the <colour> bin` (six objects) and selects the
heat-map channel; the network itself does not read text. The S0 box is amodal and tight to the rendered geometry, so against a visible-pixels
detector IoU cannot reach 1 (S0: mean 0.949 against a segmentation render).

## The network (`model.py`)

Per camera, its own ResNet-18-shaped backbone (from scratch: no weights are downloaded and none are loaded), a top-down neck to stride 4 and three heads:
a heat map with one channel per object, a sub-cell centre offset and a box size (CenterNet-style). Loss: focal loss on the instructed object's channel
only (a frame without the object has an all-negative target) plus L1 on offset and size at the centre cell of a present box. About 24 M parameters
(limit 50 M, asserted). Float32 inference.

## Data (`data.py`) and training (`runner.py`)

* **Cache.** Every 8th frame (0, 8, ..., 448; `STRIDE_K`, fixed with the gate thresholds before extraction) of every sealed P2 episode, decoded from the sealed AV1
  videos by LeRobot's own decoder at the timestamps its loader uses, preprocessed by `inputs.preprocess` (the same code the rollout adapter runs) and stored as
  uint8 arrays, with S0's `target_box` of that (episode, frame) as the label and the object the plan's instruction names. 30,780 rows, about 19 GB; the extraction
  refuses unless 65 GiB stay free; the manifest holds the sha256 of every array and of each episode's two source videos.
* **Training rows** are the plan's 480 training episodes only; the 60 validation episodes are used for the checkpoint choice and the gate; no frame of any
  M0 tier layout is ever read (tests compare the plans).
* **Absence is learned from exact geometry.** S0's own `in_frame` is 1 on every training frame, so the augmentation translates each camera's frame by a random
  integer shift (probability 0.5, up to 0.6 of the frame), moves the label with it, clips it, and calls it absent when less than 2 px remain in either dimension
  (`inputs.shift_image`, `shift_box`; a test draws a rectangle, shifts both and compares). Brightness, contrast and noise jitter (5%, 5%, 0.01) on top.
* **Recipe:** 60 epochs, batch 32, AdamW 1e-3 (weight decay 1e-4), one warm-up epoch then cosine, bf16 autocast on the GPU, seed 20261105. The checkpoint is
  the epoch with the best mean over the two cameras of the mean validation IoU: **never chosen by rollouts, never on any M0 layout**. Offline IoU selects a
  perception module; it says nothing about capability.

## The gate (`gate.py`, spec D5), fixed before training and never moved

Per camera: mean IoU at least 0.85 and at most 3% of frames below IoU 0.5, on (a) the validation episodes' decoded frames and (b) **raw renders** of the
replayed validation episodes (the rollout sees raw renders; the cache is video-decoded; (b) runs the same `Predictor` the arm uses), and presence F1 at
least 0.90 on (c) synthetic out-of-view frames (validation frames translated with a fixed seed; the F1 is the smaller of the "present" and the "absent" class).
A failed gate means no A5 run; the detector is then fixed using training and validation data only, and no threshold is touched. Caveat, stated now: the checkpoint is
chosen on the same 60 validation episodes the gate measures (as the spec says), so (a) and (b) are not independent of the selection; the gate is a go/no-go for
GPU time, not an estimate of rollout performance.

## Commands

```
python -m experiments.vla_inputs_box_detector.runner extract         # CPU, admission image, cache_v1
python -m experiments.vla_inputs_box_detector.runner check           # CPU admission record
python -m experiments.vla_inputs_box_detector.runner train           # GPU, run_v1/, final/
python -m experiments.vla_inputs_box_detector.runner gate-decoded    # (a) and (c)
python -m experiments.vla_inputs_box_detector.runner gate-raw --shard K --shards N   # (b), robosuite image
python -m experiments.vla_inputs_box_detector.runner gate-decide     # thresholds; exit code 1 on failure
```

Outputs under `outputs/vla_inputs_box_detector/`: `cache_v1/`, `admission_v1/`, `run_v1/` (`history.json`, `gate/`, `final/` = `detector.pt` and
`detector.json`, the directory the arm binds as an input). Nothing is ever overwritten.

## Tests

`tests/test_inputs.py` (preprocessing, geometry, exact shifts, the parser), `test_model.py` (shapes, budget, targets, loss, decoding, and that a small copy learns),
`test_data.py` (the cache format and its hashes, determinism, the split, the disk guard, and on the sealed data: frames equal what the dataset decodes, labels
equal S0's, objects equal the plan's), `test_gate.py` (perfect, shifted, never-firing and always-firing stand-in detectors, the thresholds and their boundary),
`test_runner.py` (a tiny training run, the schedule, best-epoch choice, the saved detector's round trip, the guards, the binding against the real seals, a
symtable scan). Each was confirmed against deliberate mutations.
