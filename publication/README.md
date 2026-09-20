# Offline baseline release tooling

Scope: only portable inference packaging/verification for the two final
fixed-layout baselines. No credential loading, HTTP, Hub client, upload, commit,
training, GPU worker, or simulator execution is implemented.

## Modules and tests

- `package_baselines.py`: copy-only allowlists, exclusive output claims, explicit
  config rewrites, source/asset SHA-256, byte-exact streamed safetensors checks,
  local pinned provenance, candidate model cards, inventories and release audit.
- `load_baseline.py`: copied verbatim into each release. CPU-only, local assets,
  no project imports; preserves frozen dtypes and promotes SmolVLA trainables
  before strict safetensors loading. Config and tokenizer processor paths are
  both resolved relative to the release, independent of working directory.
- `verify_baselines.py`: separate original/reference and portable workers and
  exact numerical comparison. Raw tick observation fields only, no targets or
  simulator truth. Native converter `action=None` placeholders are removed.
- `tests/`: importable unittest suite; deterministic tiny real fixtures, no
  downloads or model-weight dependency for unit tests.

All Python commands below are **inside** the pinned Docker sandbox, never host
Python. Runtime image:
`sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`.
Use `--runtime runc --network none --read-only --memory 16g --cpus 4`, current
user UID/GID, temporary `/tmp`, `HOME=/tmp/home`, `HF_HOME=/tmp/huggingface`,
`XDG_CACHE_HOME=/tmp/cache`, HF/Transformers/Datasets offline flags, empty
`CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES=void`, `OMP_NUM_THREADS=2` and
`PYTHONDONTWRITEBYTECODE=1`. Mount repository read-only; only new publication
output directories writable. Never use an env-file or mount credential stores.

Unit runner (working directory: repository root):

```sh
python -m unittest discover -s publication/tests -t . -v
```

Fresh staging, only when the exact packages root does not exist:

```sh
python -m publication.package_baselines
```

The CLI only targets `outputs/publication/baselines_20260918/packages/`.
An existing packages root, including a partial root, fails closed. Do not remove
or overwrite it automatically. Diagnose failures and preserve/quarantine the
owned partial attempt before deliberately choosing a fresh execution.

## Verification workers

Run `python -m publication.verify_baselines worker --mode source --checkpoint
<original-005000> --raw <recorded-tick> --output <new-source-evidence-directory>`
with original repository/assets read-only. Then run mode `portable` in another
fresh container with only one release mounted read-only at `/release`, this
verification package mounted at `/tools/publication`, `PYTHONPATH=/tools`, the
single raw tick at `/observation.npz`, and only the evidence output parent writable.
Set working directory `/tmp`; do **not** mount the repository, original caches or
checkpoint there. Pass `--checkpoint /release --raw /observation.npz`.

The recorded observation used here is attempt 000000/tick000060, not a training
target or a synthesized observation. Source and portable workers use the same
CPU settings and seed 20260918. Expected evidence directory names beneath the
publication root are `verification/{smolvla,xvla}-{source,portable}`.

After all four workers succeed, run `python -m publication.verify_baselines report`
with repository read-only and publication output writable. This rehashes source
files and releases and writes new (never overwrites) JSON/Markdown reports.
Release-wide text/allowlist audit is available as `audit_release(path)`.

## Handoff boundaries

Only `packages/smolvla/` and `packages/xvla/` are release candidates. Never upload
the parent publication directory, verification inputs/outputs or failed staging
directories. Candidate cards are not final legal/model-card review. Source
manifests lack license metadata: record unknown, do not infer a license.

The actual recipe uses **500 warmup / 5000 total** updates. Preserve source
scheduler defaults; explain the mismatch instead of silently changing history.
Model cards report SmolVLA 18/20 vs X-VLA 20/20 on one selected geometry and 20
sampling seeds, not generalization. Releases omit datasets/simulator/environment
assets and are not a full simulation/environment reproduction package.

If the publisher finalizes a card, review the exact diff and update **only its**
inventory entry using its measured size/hash, then rerun inventory/text audits.
Do not bless unexpected files by blindly regenerating the inventory. Changed
config, tokenizer, processor, loader or weights require new numerical verification.
No remote publication or authentication is implemented here.