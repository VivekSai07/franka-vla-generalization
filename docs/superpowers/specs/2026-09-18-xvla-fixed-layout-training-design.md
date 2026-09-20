# X-VLA fixed-layout training — approved direction

User approved the preceding recommendation on2026-09-18: X-VLA first, fresh base,
5000updates using the prior verified recipe, final-checkpoint20fixed-reset trials,
initial capability gate>=18/20. SmolVLA is a later decision, not an automatic job.

## Immutable experiment

- Dataset `outputs/fixed_layout/fixed_layout_20260918/dataset`; parent seal
  `403b8ec2d16094863faf611a13768e770dbf19a363c477f0f509fb67d8d8fba5`.
- Training IDs0-19, validation20-23.20x528=10560training rows;20x499=9980complete
  30-action starts. No episode boundary padding. Never use library global stats.
  Fit fresh mean/std/min/max on training rows only; state IDENTITY, action MEAN_STD.
- Fresh pinned `lerobot/xvla-base` cdb7964e4fe842935d671bfab5a5ebe00a96648c,
  existing offline bundle/BART tokenizer; no downloads or old fine-tuned weights.
- Identical prior model recipe:901FP32trainable tensors/879482456parameters,
  BF16CUDAautocast/FP32Adam, native auto real8MSE, chunk30/horizon10, domain0,
  two512input views resized224 + masked zero third, state9/padded20,64tokens,
  no LIBEROrotation/privilegedinputs.
- Prior seed20260917 retained for training; micro4 xaccum2 effective8,
  5000updates/40000sampledchunks.500warmup,1e-4peak cosine1e-5floor; VLM0.1x,
  softprompt1x; Adam betas(.9,.95),eps1e-8,weightdecay1e-4/VLM1e-5,clip10.
- Checkpoints1/1000/2000/3000/4000/5000; optimizer only final. Final5000selected,
  no best-of-rollouts selection. Small scratch exactly2updates at same microbatch,
  strictFP32weights+Adam+prediction0 reload; never initialize main from scratch.
- Fixed train/validation offline probes at frames60/107/347 in each respective
  episode on final checkpoint only, seeded without changing later rollout seeds.
  Report physical arm/first10MAE and gripperbinary accuracy; no heldoutsceneclaim.
- Evaluation20distinct seeds20260918..20260937, same saved full reset every time.
  One rollout/seed,40s/1200ticks,30consecutive released-stack ticks with existing
  strict geometric/contact/open-gripper score. Gate>=18/20; descriptive small-n
  sampling reliability at one intentionally selected geometry, not generalization.
- Log first/sustainedclosure, bilateralcontact, lift, release, correctstable stack
  and clipping separately. No oraclepolicyinput. Videos/raw/appliedactiontraces.
  All20mustfinish; no optional stopping after18successes, no inference retries.

## Safety and admission

New package `experiments/xvla_fixed_layout/`; old bound source/spec/data/report
untouched. Output `outputs/train/xvla_fixed_layout_20260918`, separate `_preflight`
scratch and `_admission` directory for CPUgate. Exclusive creation,nooverwrite/
resume/retry/fallback/automaticbudgetextension. Cache/source/data/statistics/
checkpoint bindings rechecked; source hashed explicitly, no oldrecursive rescan.

Run the existing `require_fixed_layout_pass` read-only CPU gate once to establish
fresh full semantic admission. Save report bound to exact data seal, source and
runtime. Later phases rehash full dataset/sources/assets and compare CPUadmission
bytes; do not redo expensive full media audit on every model reload. Cache a gate
only with exact immutable fingerprint equivalence, never trusting a bare PASS.

CPUtests → fresh CPUdataadmission → GPU2update scratch → exact reload checks →
fresh base5000training → finalofflineprobes →20rollouts. A gate failure stops;
no silent repair/retry/relaunch. Main agent reviews before actualGPUjobs.

All Python in existing pinned Docker image, no network/secrets/install. Repo/data
read-only, only each new output root writable; retain logs/container. Check free
disk>=60GiB and GPUfree>=28GiB before main (previous peak~21.34GB), no interference
with unrelated running `moveitFrancesco`. Checkpoints preserved; no new deletion.
No claim of X-VLA superiority or randomization-only causal effect: expert/dataset
count changed too. Training only is authorized, not laterSmolVLA/collection/upload.
