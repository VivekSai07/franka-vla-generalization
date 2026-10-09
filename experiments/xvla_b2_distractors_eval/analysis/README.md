# B2 evaluation analysis

The analyzed policy is [X-VLA B2](https://huggingface.co/VivekSai07/xvla-franka-b2-distractors-50000), trained on the
[B2 dataset](https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-b2-distractors-200). See the [artifact index](../../../docs/HUGGING_FACE_ARTIFACTS.md#b2-distractors)
for earlier checkpoints and archive gaps; keep the report's selection caveats.

Scripts behind [docs/B2_DISTRACTOR_EVALUATION_20260930.md](../../../docs/B2_DISTRACTOR_EVALUATION_20260930.md),
in run order. Each reads the evaluation outputs under
`outputs/train/xvla_franka_stacking_b2_distractors_v1_20260928/` and writes to `/out`.

`run.sh <script>` runs one inside the pinned image with the repo mounted read-only and
`/home/utxowqr/b2_analysis_out` mounted writable. Paths are hardcoded to this machine.

| script | produces |
|---|---|
| `01_extract.py` | per-trial feature table (`/out/trials.json`) |
| `02`–`03` | headline statistics, paired contrasts, layout concentration, predictors |
| `04` | validation of the measures (positive control for distractor displacement) |
| `05`–`07`, `09` | cube-identity behaviour, offsets, determinism, release-precision cliff |
| `08` | video contact sheets for the odd trials |
| `10`–`12` | hand-frame grasp error, stall/reach, final mechanism table and failure CSV |
| `13` | scripted-expert feasibility of the failing layouts |
| `14` | expert placement precision from the sealed dataset vs the policy |
| `15`–`16` | wrong-cube check (which cube was lifted first) and per-trial timelines |

Stages 02 onward read the output of the one before, so run them in order. Several early scripts
contain measures that turned out uninformative (see section 6 of the report); they are kept so the
record is complete.
