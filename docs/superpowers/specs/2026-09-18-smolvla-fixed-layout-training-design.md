# SmolVLA fixed-layout comparison

User approved SmolVLA on the same dataset/reset/evaluation seeds after X-VLA20/20.
Keep the previously verified SmolVLA recipe, not an architecture-only comparison.

- Dataset parent outputs/fixed_layout/fixed_layout_20260918, seal
  403b8ec2d16094863faf611a13768e770dbf19a363c477f0f509fb67d8d8fba5.
- Train0-19, validation20-23;10560trainingframes,9580complete50action starts.
  Fresh training-only statistics, no datasetglobalstats or validationoptimization.
- Fresh pinned lerobot/smolvla_base c83c3163b8ca9b7e67c509fffd9121e66cb96205,
  existing local VLM/tokenizer revision7b375e1b73b11138ff12fe22c8f2822d8fe03467.
  No downloads, oldcheckpoint resume, or initialization from scratch-preflight.
-450046176total/99880992trainable parameters; frozenVLM, expert+stateprojection
  FP32 before Adam, BF16trainingautocast, verifiedFP32loader before copying weights.
-5000updates, micro8xaccum4/effective32,160000sampledchunks; seed20260917.
  AdamW betas(.9,.95),eps1e-8,weightdecay1e-10,clip10;500warmup1e-4peak,
  cosine1e-5floor at5000. Stockflowmatchingloss unchanged, no paddedboundarychunks.
- Nativechunk50,execution10,state9/action8,two512RGB,30Hz/600Hz physics.
  Same normalization/config as earlier SmolVLA, no X-VLA-specific processors.
- CPUfresh semantic admission then2scratchupdates at exactbatch, checkpoints1/2
  strictFP32save/reload absoluteactionerror<=1e-5 and finiteFP32Adam. Main freshbase
  only after matching scratch/cache/data/stats/source checks. No scratchrollouts.
- Main checkpoints1/1000..5000, original save/reload helper/training_state artifacts
  retained. Final5000only evaluated, no selection by evaluation success.
- Final72offlineprobes at60/107/347per20train+4val, physicalfull50/first10armMAE
  and gripperbinaryaccuracy, RNGpreserved, inferenceobservationsonly. Additional
  prior-style training probes optional only if fixed beforelaunch and notselection.
-20trials seeds20260918..20260937, exactsamefullreset sourceepisode7 asXVLA,
 40s/1200ticks,30strictconsecutive releasedstackticks, identicalaction/governor/
 scoring. All20complete, gate>=18/20; closure/contact/lift/release/clipping separately.
-Samegeometry validation/seed trials do not establish spatialgeneralization.
 SmolVLAchunk50/160kchunks/frozenVLM vsXVLAchunk30/40kchunks/alltrainable;
 retain those differences explicitly, never claim compute-matched architecturetest.

Newpackage experiments/smolvla_fixed_layout; exclusive outputs/train/
smolvla_fixed_layout_20260918 with siblings_admission,_preflight. No oldbound
source/spec/artifact changes, no retries/overwrite/resume/autotune/newcollection.
Only mainagent launchesGPU afterCPUtests/independentreview/admission/scratch.
AllPython pinnedDockeroffline,no secrets/install/delete/push. Existingrepo/data/
cache readonly, onlynewoutput writable. Hostdisk>=35GiB andGPUfree>=12GiB before
start; preserve checkpoints and unrelatedcontainers. Laterjobs require decision.
