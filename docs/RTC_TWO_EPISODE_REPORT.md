# RTC two-episode collection and alignment verification report

**Date:** 2026-09-14

**Result:** Two-episode runtime verification passed.

**Decision status:** Full collection is stopped and awaits explicit user approval.

**Source baseline:** `ea641ffbdcfa5f2938e06ae511cbe9cb1daa523e`

## 1. Purpose and scope

Execute the small-batch gate in [RTC_VERIFICATION.md](RTC_VERIFICATION.md)
before committing resources to a 300-episode collection. The fixed
[collector](../data_collection/collect_libero_dataset.py) records the arm target
computed and sent during the current tick, rather than the preceding tick's
stale target. The required test checks whether recorded actions are closer to
the following joint state than to the current joint state.

The checkout includes alignment-fix commit `0709993`. The collector, verifier,
original Dockerfile and original Compose configuration were not changed for
this test. The results below concern the LIBERO-style collector only; the other
collectors were not runtime-tested.

## 2. Environment and setup findings

| Component | Observed configuration |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation, 49,140 MiB reported VRAM |
| NVIDIA driver | 570.211.01 |
| Container base | NVIDIA CUDA 12.8.1 + cuDNN, Ubuntu 22.04 |
| Container Python | 3.10 |
| Docker / Compose | Docker 29.1.3 / user-local Compose 2.40.3 |
| PyTorch | 2.7.1+cu128 |
| TorchVision / TorchAudio | 0.22.1+cu128 / 2.7.1+cu128 |
| TorchCodec | 0.4.0, CPU video decoding |
| LeRobot | 0.4.4, installed with SmolVLA extras |
| Pinocchio | 3.8.0 |
| Native URDF dependencies | cmeel-urdfdom 4.0.1, cmeel-tinyxml2 10.0.0 |
| MuJoCo | 3.13.0 |
| Rendering | Headless OSMesa, two 512 × 512 cameras |
| Container UID:GID | 348064031:348000513, matching the host account |

Validated image identifier:

```text
sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
```

### Problems encountered and resolutions

1. **Compose unavailable:** installed the Ubuntu Compose package's executable
   under the user's Docker CLI plugin directory, without changing system packages.
2. **Bind-mount ownership mismatch:** the original service uses UID:GID
   `1001:1001`. A collection-only overlay uses the actual host identity instead.
3. **Build networking:** forwarding the host's standard proxy build arguments
   enabled Ubuntu package installation. PyTorch's download host still returned
   `407 Proxy Authentication Required`; the CPU wheel index referenced the same
   inaccessible download host.
4. **Local wheel installation:** the user transferred the three matched stable
   CUDA wheels. All three SHA-256 checks matched the official index. The alternate
   build installs these local files and constrains their versions during later
   dependency resolution. Other dependencies were downloaded normally through
   the configured proxy; this is not a fully offline build.
5. **Native ABI mismatch:** the initial Pinocchio installation selected URDFDOM 6.
   Although `pip check` passed, importing Pinocchio failed because
   `liburdfdom_sensor.so.4.0` was missing. Pinning URDFDOM 4.0.1 and TinyXML2 10.0.0
   resolved the import failure. Package metadata checks alone are insufficient
   for native robotics libraries.

After correction, `pip check`, Torch/TorchVision/TorchAudio/TorchCodec imports,
Pinocchio/MuJoCo/LeRobot imports, CUDA tensor computation and checkout write access
all passed. CUDA was tested explicitly; collection itself uses scripted control
and CPU rendering rather than a learned GPU policy.

The alternate build files are local, uncommitted implementation artifacts at
report time. This report-only commit does not make a fresh checkout reproduce
the alternate build automatically. No credentials or wheel binaries belong in Git.

## 3. Collection procedure

With the local build overlay available, the test used:

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose -f docker-compose.yml -f compose.offline.yml run --rm -T collect \
  python data_collection/collect_libero_dataset.py --episodes 2 --headless
```

The printed output directory was `/vla/datasets/franka_libero_stacking_v1`
inside the container, corresponding to `datasets/franka_libero_stacking_v1`
in the checkout. The test did not change FPS, robot speed, playback speed,
instructions, spawn logic, success thresholds or the alignment implementation.
The script does not expose a seed argument and no deterministic seed was set;
repeating collection may produce different tasks and numerical results.

## 4. Observed dataset and episode results

| Property | Result |
|---|---|
| Saved episodes | 2 |
| Recorded frames | 1,040 total; 520 per episode |
| FPS | 30 |
| LeRobot codebase version | v3.0 |
| Distinct instructions | 2 of the 12 possible variants |
| Action | 8 float32 values: 7 commanded arm targets + gripper command |
| Observation state | 9 float32 values: 7 arm positions + 2 finger positions |
| Images | Front and wrist streams, 512 × 512 RGB |
| Video metadata | AV1, yuv420p, 30 FPS, no audio |
| On-disk size | Approximately 11 MiB reported by `du -sh` |

| Episode | Instruction | Cubes present | Frames | Logged final XY offset | Logged final Z separation |
|---|---|---|---:|---:|---:|
| 1 | Pick yellow, place on green | Yellow, green | 520 | 0.2 cm | 4.2 cm |
| 2 | Pick green, place on red | Green, red, yellow | 520 | 1.6 cm | 4.2 cm |

Both attempts logged grasp verification at approximately 6.76 seconds of
simulation time and success at 17.85 seconds. No discarded attempt appeared in
the two-episode test output. The acceptance rule is XY distance below 2 cm and
Z separation strictly between 3.5 and 4.5 cm, as evaluated by the collector.
Distances above are rounded console values, not a separate geometric audit.

The run printed `Data collection complete: 2 episodes saved.` and the final
dataset path. The final metadata independently confirms two episodes and 1,040
frames. No manual visual review of the saved videos was performed.

Wall time averaged approximately 85 seconds per successful episode, or roughly
170 seconds for this test. A simple 300-episode extrapolation is approximately
7.1 hours, excluding additional failed-attempt overhead and machine-load changes.
Two episodes are too small a sample for a reliable production-duration estimate.

## 5. Alignment verification

The original [verifier](../data_collection/verify_action_alignment.py) was run
against the exact test dataset, and rerun during report preparation:

```bash
docker compose -f docker-compose.yml -f compose.offline.yml run --rm -T collect \
  python data_collection/verify_action_alignment.py \
  --root datasets/franka_libero_stacking_v1 \
  --repo-id franka_libero_stacking_v1
```

Both executions returned exit code **0** and the same numerical output:

```text
mean |action_t - state_t|     (current, should be LARGER): 0.057231
mean |action_t - state_t+1|   (next, should be SMALLER):   0.050660
PASS: recorded actions are closer to the NEXT state than the current one — alignment fix confirmed.
```

The verifier compares only the first seven dimensions (arm joints). For each
episode, it excludes the last action because there is no following frame,
averages absolute error over joint dimensions and remaining timesteps, then
averages the episode-level errors. Here, 519 timestep pairs per episode give
1,038 within-episode comparisons. No cross-episode transition is included.

The next-state mean error is lower by **0.006571 rad**, approximately **11.5%**
relative to the current-state mean, using the rounded printed values. This meets
the documented strict inequality gate and provides runtime evidence consistent
with the intended action/observation fix on these two trajectories.

### What this result does not prove

- It is not a controlled before/after experiment: no pre-fix dataset was generated
  with matching seeds for comparison.
- The aggregate inequality is not a frame-by-frame proof of exact command timing,
  nor does it establish statistical significance or a minimum safety margin.
- The gripper command is excluded from the numerical comparison. Camera timing,
  task grounding and image quality were not independently audited.
- Two successful attempts do not establish a general collection success rate,
  balanced coverage of all 12 instructions, robustness to all spawn conditions,
  or trained-policy performance.
- No fine-tuning, inference evaluation or other collector runtime test was run.

## 6. Full-collection hold and preserved artifacts

After the first PASS, a 300-episode job was launched under the earlier approval.
The user's subsequent instruction reserved that decision, so the job was stopped
at **2026-09-14 20:28:15 UTC**. Its state was independently checked afterward:

```text
Container: smolvla-rtc-collect-300
Status: exited
Running: false
ExitCode: 137
```

Exit code 137 followed the requested Docker stop; it is not a successful bulk
completion and is not, by itself, evidence of an out-of-memory failure. The job
had logged one saved blue-on-green episode. Its preserved v2 metadata reports
**1 episode / 520 frames**. Since collection was interrupted before normal
finalization, this v2 directory is incomplete and must not be treated as a
validated training dataset. The container and files have not been deleted.

The validated two-episode **v1** dataset is separate and remains intact.
No further collection will be started without explicit user approval. Restarting
the current script does not resume an interrupted dataset: its auto-versioning
selects the next unused directory. A future run must identify its actual output
path rather than assuming v2.

## 7. Conclusion and next decision

**The requested two-episode alignment gate passed on the RTC machine.** The
local-wheel approach resolved the blocked PyTorch download, and runtime testing
identified and corrected an additional native-library compatibility problem.

The next decision is whether to authorize full collection. If authorized,
confirm the intended image and output directory, preserve the smoke dataset,
and require normal completion plus matching episode metadata before upload.
The existing push/train defaults must not select the smoke or interrupted data
by accident. No dataset upload or training has been performed.

**Never resume training from pre-fix checkpoints.** Any later training must use
the corrected, completed dataset and an explicitly fresh output directory.