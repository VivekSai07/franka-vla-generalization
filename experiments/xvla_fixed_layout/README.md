# Fixed-layout X-VLA (isolated adaptation)

Approved contract: fresh offline XVLA base, train episodes 0–19 only; validation
20–23 is the **same geometry**, not a spatial holdout. 10,560 training rows,
9,980 complete 30-action starts. Fresh train-only statistics, identity state,
native real8 MSE, FP32 weights/Adam, BF16 autocast, micro4 × accumulation2,
seed20260917, exactly5000 updates. No resume, overrides, retries or autotuning.

## Ordered entry points (main agent only for GPU)

Module `experiments.xvla_fixed_layout.runner` accepts exactly one of:

- `--check`: claims the exclusive `_admission` directory, performs one fresh
  `require_fixed_layout_pass`, actual data/statistics and CPU micro4 processor
  checks without constructing the backbone. This can take minutes. Persists the
  semantic report and bindings in that directory, never in bound source.
- `--preflight`: requires that CPU proof; fresh base, two updates, checkpoints1/2,
  exact FP32 weights/Adam and zero prediction-error reload, no probes/rollouts.
- `--train`: requires byte-bound scratch/CPU/stats proofs; starts **fresh base**,
  not scratch. Checkpoints1/1000/2000/3000/4000/5000, optimizer only5000. Final
  strict reload, 72 observation-only offline probes, then all20 fixed-reset seeds
  20260918–20260937. Gate18/20 cannot pass early. No automatic budget extension.

Output roots beneath outputs/train are xvla_fixed_layout_20260918_admission,
xvla_fixed_layout_20260918_preflight, and xvla_fixed_layout_20260918. Roots are
exclusively claimed. For the read-only repository sandbox, main may exclusively
create the fresh host root, then bind-mount **only that root** writable at the
same container path. The runner permits an existing directory only when it is
an actual empty mountpoint, then atomically creates run_claim.json with exclusive
open. Any existing claim/content/failed attempt or ordinary existing directory
is rejected; no overwrite/resume. Other output roots, repository/data/model cache
stay read-only. CPU admission writes only its new admission root.

Pinned image: sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc.
All Python in Docker; no network/install/credential files. CPU tests use runc,
no visible GPU, read-only root/repo, temporary HF/XDG caches,16GiB/4CPU. Never
interfere with unrelated moveitFrancesco. Main-start reserve60GiB disk/28GiB GPU
free (prior peak21.34GB), checkpoints preserved.

## Deliberate reuse and copies

Reused dataset-independent functions: model config, LR/group scheduler,
precision/Adam/strict weight loaders, observation/input checks, file hashing.
The training loop is an explicit small port from xvla_single_task/runner because
its global contract and data calls are not injectable. Differences: fixed dataset
adapter, fresh CPU proof/statistics, derived counts, resource floors and20trials.
Never call old prepare_data/policy_row/selected_stats/sample_batches/admission.

The rollout loop is copied literally from xvla_single_task/evaluate; a regression
compares its entire1200tick loop AST. Outside that loop, reset comes directly from
the sealed reset.npz, a genuine explicit seed argument initializes all RNGs,
and a subclass counts real chunk calls (no method/global monkeypatch). Episode7
identity is unchanged across all20trials. Existing30Hz/20substeps/40s/30tick
strict contact/open-gripper scoring, command clipping and traces are unchanged.
Stage reports expose closure/contact/lift/release/stack separately. Offline
probes preserve Python/NumPy/Torch RNG and mode, never send action targets to
prediction, and report real-unit errors separately from normalized training loss.

The full semantic gate is CPU-only and called exactly once per fresh admission.
Every subsequent phase recomputes the full parent fingerprint (including media,
attempts, resets and reports), current fixed-layout source/runtime provenance,
actual model/tokenizer bytes, and an explicit list of imported helpers, installed
XVLA/processor files, new source/tests/README and approved spec. New files do not
invalidate an old recursive report; those reports are never admission inputs.
Changes after CPU admission require stopping, not silently regenerating a proof.

## Evidence limits

Software/synthetic CPU verification does not establish GPU compatibility,
successful training, physical stacking, or generalization. Full fresh semantic
readmission, two-update GPU scratch and actual5000+20 are main-agent stages.