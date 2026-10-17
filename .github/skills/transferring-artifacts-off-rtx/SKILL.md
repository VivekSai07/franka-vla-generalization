---
name: transferring-artifacts-off-rtx
description: "Use when moving a sealed dataset or trained checkpoint off this RTX machine to Windows for safekeeping, building a download bundle for an axis, publishing to Hugging Face after a verified finding, or reclaiming RTX disk space afterward."
---

# Transferring artifacts off RTX

The recurring lifecycle: **stage** (RTX, zero extra disk) → **transfer** (RTX → Windows) →
**verify** (Windows) → **publish** (Windows → HF, only after a legit finding) → **reclaim**
(delete the RTX originals). Do the steps in this order; do not delete anything before the Windows
copy is hash-verified.

## 1. Stage a hardlinked bundle on RTX

### What a bundle is

A bundle is a **complete, self-sufficient archive of one axis of work**. It is not an upload
payload. The test it must pass: *with this directory and nothing else, could someone retrain a model
on the data, and re-evaluate a kept checkpoint?* An upload layout is a **view inside** the bundle
(`hf_upload/`), never the bundle itself.

A bundle for `<axis>` has these sections, each one hardlinked from its original:

```
outputs/downloads_<YYYYMMDD>/<axis>/
├── dataset/              the LeRobot payload                        → retrain
├── dataset_provenance/   plan, provenance, seal, validation, bank   → verify the seal, read the splits
├── dataset_episodes/     per-episode expert traces (if collected)   → inspect the demonstrations
├── checkpoints/          one dir per kept checkpoint                → use the model
├── evaluation/           every trial of every run, with the videos, → re-read any result
│                         traces and per-trial JSON, plus the
│                         reserved-scene bank                        → re-evaluate
├── training/             result, metrics, stats, initial config     → read the recipe
├── runtime/              base model + tokenizer bundle, pip freeze   → load the checkpoint anywhere
├── code/                 `git archive HEAD` + COMMIT                 → the evaluator and scene code
├── reevaluate/           bundle-only evaluation entry + verifier     → re-evaluate, verify by hash
├── hf_upload/            the upload layout + model card (a view)    → publish
└── README.md             inventory, hashes, retrain / re-eval / upload recipes
```

**"Re-evaluate" means from the bundle alone, on a fresh machine.** The B3 bundle (2026-10-05)
passed every hash check and was still not re-evaluable off RTX (Windows audit, 2026-10-06): every
checkpoint's `config.json` and `policy_preprocessor.json` name the tokenizer by its RTX path
`/vla/.cache/xvla_trial/bundle/tokenizer`, and the evaluator and scene code lived only in the repo.
So a bundle also carries `runtime/xvla_base_bundle/` (hardlink of `.cache/xvla_trial/bundle/`,
mounted back at that path, which leaves the checkpoint files unedited), `code/` (with an empty
`code/.cache/xvla_trial/bundle/` as the mount point, because the code mount is read-only), and a
re-evaluation entry point. The repo's evaluators verify the whole training run and the sealed
root directory, which a bundle does not hold in that form. So the entry point replaces only those two
checks: weights against `checkpoint_result.json`, and the seal recomputed over the bundle's
sections. Everything else stays the repo's code. **Run it in a container that mounts only
the bundle** (`--network none`, plus `-e CUBLAS_WORKSPACE_CONFIG=:4096:8`; deterministic evaluators
stop without it) and compare its trials with the originals. Worked example and recipe:
`outputs/downloads_20261009/props_p1/` (smoke reproduced bit for bit). The Docker images are not
bundled (about 20 GB each); the README gives the `docker save` command.

Omit a section only when it does not exist for that axis (e.g. no `dataset_episodes/` when the
collector wrote none) — say so in the README rather than leaving a silent gap, and fold any small
stray files from the sealed root into `dataset_provenance/`. "We only want to publish the weights"
is not a reason to omit a section — the bundle is the archive, and `hf_upload/` is how you publish
from it.

`hf_upload/` holds the winning checkpoint's files at its root (so the repo root is the model), the
other kept checkpoints under `checkpoints/step_NNNNNN/`, the evaluation `result.json` summaries, the
training record, and `README.md` — the model card, whose numbers come from the axis's dated report
and `docs/RESULTS.md`, never re-derived.

The README's hash table lists every `model.safetensors` in the bundle with its sha256, because that
is what the Windows side re-checks after the transfer.

Worked examples (`b2_distractor` was deleted from RTX on 2026-10-17 after the Windows copy was verified; the layout is in the
Windows copy): `outputs/downloads_20261001/b2_distractor/`,
`outputs/downloads_20261005/b3_robosuite/` (no `runtime/` or `code/`, so not portable),
`outputs/downloads_20261009/props_p1/` (complete),
`outputs/downloads_20261017/props_p2/` (complete; adds `reevaluate/infer_example.py`, a one-observation inference
check, and `bundle_evaluate.py --shard/--shards/--collect-only` for a parallel re-evaluation. Its smoke re-run in a clean
container, run alone, matched originals that had run in 8-way parallel workers bit for bit).

**Never rename trial folders in an evaluation output** to make them easier to browse: the
evaluators find trials by `trial_NNN_<layout_id>` for `--resume` and re-analysis. Build a
hardlinked view next to it instead (`outputs/xvla_props_p1_eval_browse/`: held-out vs trained,
object, bin, side and outcome in the name).

### Build it

Never copy — hardlink, so the staging directory costs zero extra disk:

```bash
mkdir -p outputs/downloads_<YYYYMMDD>/<axis>/checkpoints
cp -al outputs/train/<run>/checkpoints/step_NNNNNN outputs/downloads_<YYYYMMDD>/<axis>/checkpoints/
cp -al outputs/<axis>_data/<sealed_root>/dataset outputs/downloads_<YYYYMMDD>/<axis>/dataset
```

`cp -al` fails on an existing target, so build into a fresh directory.

Within each section:

- **`dataset/`**: the LeRobot payload only — not `attempts/`, `rejected/`, `resets/` (15–17 GB of
  raw admission evidence, needed only to re-verify the sealed admission, not to reuse the data).
- **`checkpoints/`**: `model.safetensors` + `config.json` + processor files +
  `checkpoint_result.json` per `step_NNNNNN/`. **Never `training_state.pt`** (optimizer/RNG state,
  ~7 GB, only needed to *resume* training — it is not needed to use or re-evaluate the model).
- **`evaluation/`**: every run (report, screen, smoke, feasibility) with every trial directory
  intact — `cameras.mp4`, `trace.npz`, `episode.json`. These are what make a result re-readable a
  month later. Include the reserved-scene bank (`bank.npz` or equivalent): without it the
  evaluation cannot be re-run.
- **`README.md`**: inventory, the weight hashes, and the three recipes (retrain / re-evaluate /
  upload). It is disposable staging documentation, gitignored like everything under `outputs/`;
  this skill is the durable copy of the *process*.

### Verify the bundle before calling it staged

```bash
D=outputs/downloads_<YYYYMMDD>/<axis>
for s in dataset dataset_provenance checkpoints evaluation training README.md; do
  [ -e "$D/$s" ] || echo "MISSING SECTION: $s"
done
du -sh --apparent-size $D          # apparent size of the bundle
df -h .                            # unchanged vs before: hardlinks cost nothing
```

Then, in Python: every `model.safetensors` hashes to its `checkpoint_result.json`; every bundled
dataset file is byte-identical to the sealed original (comparing inodes is enough for hardlinks);
every trial directory has all three files. Report those counts — not "staged".

## 2. Transfer to Windows

Three interchangeable methods — pick based on what's available and how big the transfer is:

**`rsync`** (preferred for anything large or repeated — resumable, only re-sends what changed):

```bash
rsync -avh --progress utxowqr@<host>:/home/utxowqr/smolvla_franka/outputs/downloads_<date>/ D:\research\smolvla_franka\downloads_<date>\
```

**`scp`** (simplest one-shot copy, no resume):

```powershell
scp -r utxowqr@<host>:/home/utxowqr/smolvla_franka/outputs/downloads_<date> D:\research\smolvla_franka\downloads_<date>
```

**Interactive `sftp` `get`** — **the user's preferred method**; also what an SFTP GUI/CLI client
(e.g. WinSCP) generates as a session script. `get -r` pulls a whole bundle; plain `get` is for a
hand-picked subset:

```
sftp utxowqr@<host>
get /home/utxowqr/smolvla_franka/outputs/downloads_<date>/<run>/checkpoints/step_NNNNNN/model.safetensors "D:/RTX trained models/downloads_<date>/<run>/checkpoints/step_NNNNNN/model.safetensors"
...
bye
```

**`hf_upload/` is hardlinked to `checkpoints/`, but no transfer tool preserves that.** `sftp`,
`scp` and `rsync` all resend those bytes as a second copy, so pulling the whole bundle sends the
winning checkpoint's weights twice. Pull the bundle *excluding* `hf_upload/`, and rebuild that view
on Windows (or pull it separately only when you are about to upload).

Keep the destination path structure identical to the source (`downloads_<date>/<axis>/...`) so the
Windows side mirrors the staging bundle exactly — this is what makes the verify step below a
direct path-for-path comparison. `sftp get` does not resume a partial file the way `rsync` does;
prefer `rsync` for a fresh multi-GB pull and reserve `sftp get` for topping up specific files.

## 3. Verify integrity on Windows before trusting the copy

```powershell
Get-FileHash -Algorithm SHA256 "D:\RTX trained models\downloads_<date>\<run>\checkpoints\step_NNNNNN\model.safetensors"
```

Compare against the `model_sha256` field in that checkpoint's `checkpoint_result.json`. For a
dataset, re-derive the seal fingerprint (see the dataset package's `validate.py`/`fingerprint`
helper) and compare against `seal.json`. Do not proceed to publish or delete until every hash
matches.

## 4. Publish to Hugging Face — only after a legit, measured finding

"Legit finding" means: the result is measured (rollouts, not offline metrics — see
[evaluating-a-checkpoint](../evaluating-a-checkpoint/SKILL.md)), documented in `docs/RESULTS.md`,
and the user has agreed it's publication-worthy. Publishing an unmeasured or superseded run wastes
the same bandwidth as a measured one and is harder to retract than to not do.

- One dataset repo + one model repo per run/axis — do not split one ablation's checkpoints across
  separate model repos; upload the winning checkpoint as the repo root and the rest under
  `checkpoints/step_NNNNNN/` subpaths in the same repo.
- Card content (numbers, contrasts, caveats, licensing) comes from the point-in-time publication
  doc for that run (`docs/<TOPIC>_PUBLICATION_<date>.md`) — copy from there, don't re-derive.
- If direct upload from RTX hits a proxy/network block, the Windows copy is also the workaround:
  publish from Windows once verified, not from inside this repo's sandbox.

## 5. Reclaim RTX disk — only after verification, and only with explicit go-ahead

**Never delete a dataset or checkpoint without the user's explicit confirmation for that specific
path**, per this repo's non-negotiable rules. Two things to get right:

- The hardlinked staging directory (`outputs/downloads_<date>/`) **frees nothing by itself** —
  it shares inodes with the original. You must delete the *original* under `outputs/train/` or
  `outputs/<dataset>_data/` to reclaim space; deleting only the staging copy is a no-op on disk.
- The raw collection audit evidence (`attempts/`, `rejected/`, `resets/`) is the largest single
  piece (15-17 GB per dataset) and is separable from the `dataset/` payload — it's fine to keep
  the sealed dataset and drop the audit trail once the seal has been independently re-verified, or
  to keep both if disk allows. Confirm which the user wants before deleting either.

## See also

[training-a-policy](../training-a-policy/SKILL.md) and
[evaluating-a-checkpoint](../evaluating-a-checkpoint/SKILL.md) for what makes a checkpoint worth
transferring in the first place; `docs/OPERATIONS.md` for the current census of what's staged.
