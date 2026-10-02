---
name: transferring-artifacts-off-rtx
description: "Use when moving a sealed dataset or trained checkpoint off this RTX machine to Windows for safekeeping, publishing it to Hugging Face after a verified finding, or reclaiming RTX disk space afterward."
---

# Transferring artifacts off RTX

The recurring lifecycle: **stage** (RTX, zero extra disk) → **transfer** (RTX → Windows) →
**verify** (Windows) → **publish** (Windows → HF, only after a legit finding) → **reclaim**
(delete the RTX originals). Do the steps in this order; do not delete anything before the Windows
copy is hash-verified.

## 1. Stage a hardlinked bundle on RTX

Never copy — hardlink, so the staging directory costs zero extra disk:

```bash
mkdir -p outputs/downloads_<YYYYMMDD>/<run_name>/checkpoints
cp -al outputs/train/<run>/checkpoints/step_NNNNNN outputs/downloads_<YYYYMMDD>/<run_name>/checkpoints/
```

Confirm zero extra disk before/after: `du -sh outputs/train outputs/downloads_<date>` should not
double-count (hardlinked files share one inode; `du` on the parent still reports them once if you
check total filesystem usage, not the sum of the two directory trees).

What to include / exclude:

- **Datasets**: only the `dataset/` subdirectory (LeRobot payload, ~1.1 GB), plus the small
  `plan.json`/`provenance.json`/`seal.json`/`validation.json` from the parent — not `attempts/`,
  `rejected/`, `resets/` (15-17 GB of raw per-episode audit evidence, needed only to re-verify the
  sealed admission, not to reuse the data).
- **Checkpoints**: `model.safetensors` + `config.json` + processor files + `checkpoint_result.json`
  per `step_NNNNNN/`. **Never include `training_state.pt`** (optimizer/RNG state, ~7 GB, only
  needed to resume training — roughly doubles transfer size for zero inference value).
- Write a `README.md` inside the staging directory recording exactly what's there and why (see
  `outputs/downloads_20260926/README.md` for the convention) — that file is disposable staging
  documentation, gitignored along with everything else under `outputs/`; this skill is the durable
  copy of the *process*.

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

**Interactive `sftp` `get`** (what an SFTP GUI/CLI client — e.g. WinSCP — generates as a session
script; useful for a hand-picked subset of files rather than a whole tree):

```
sftp utxowqr@<host>
get /home/utxowqr/smolvla_franka/outputs/downloads_<date>/<run>/checkpoints/step_NNNNNN/model.safetensors "D:/RTX trained models/downloads_<date>/<run>/checkpoints/step_NNNNNN/model.safetensors"
...
bye
```

Keep the destination path structure identical to the source (`downloads_<date>/<run>/...`) so the
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
