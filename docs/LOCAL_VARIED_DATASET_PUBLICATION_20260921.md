# Local-varied dataset publication — 2026-09-21

## Status: blocked before any remote mutation

The user requested publication of `outputs/local_varied_data` to their own
Hugging Face account, using credentials from the local environment file and a
descriptive dataset name. No repository was created and no files were uploaded.
No dataset, seal, collector source, credentials or existing user changes were
modified. No training, collection, simulation replay or policy evaluation ran.

Recommended repository ID:
**VivekSai07/franka-panda-cube-stacking-local-varied-60**.
Use **private** visibility, matching the existing uploader and previous user
publication preference. Public release requires a separate visibility decision
and review of dataset licensing; no license has been inferred or assigned.

## Correct publication payload

The requested parent contains one run. The actual LeRobot root is
[the nested dataset directory](../outputs/local_varied_data/local_varied_20260919/dataset/),
not the parent experiment directory.

- Fresh inventory: **243 files, 341,017,635 bytes** (about 341 MB / 325 MiB).
- File types: JSON, Parquet and MP4.
- Metadata: LeRobot `v3.0`, 60 episodes, 31,680 frames, 30 Hz, one task.
- Explicit episode splits: training `[0, 48)`, validation `[48, 60)`.
- State: nine measured joint/finger positions; action: eight commanded
  joint/gripper coordinates. Two 512 x 512 AV1 camera streams.
- Collection design: 40 varied training demonstrations plus eight center
  demonstrations; 12 validation demonstrations. Independent cube XY offsets
  bounded by 10 mm per axis around the reference layout.

Upload only the reviewed dataset payload and a new descriptive dataset card.
Exclude the approximately 4.95 GB full experiment evidence tree, credentials,
raw attempts/replays, checkpoints, and the reserved evaluation scene bank.
Do not imply this is broad-workspace coverage, real-robot data, or evidence of
learned-policy success. The recorded collection reports 120 successful control
replays and 63,360 decoded camera frames; those expensive checks were not rerun
during this publication preflight.

## Blocker 1: storage route requires proxy authentication

Fresh authenticated Hub checks in a temporary CPU Docker container established:

- Account `VivekSai07` matches the configured username.
- Hugging Face token is accepted and reports the `write` role.
- The proposed dataset repository does not exist.
- An unauthenticated HTTPS HEAD request to the previously used storage origin
  `hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com` fails with
  **ProxyError / HTTP 407 (Proxy Authentication Required)**.

The HEAD request checks origin connectivity, not a signed upload or per-repo
authorization. No actual upload was attempted. Nevertheless, the same storage
route that blocked previous publications is currently unavailable through the
configured proxy. An accepted Hub token does not authenticate that proxy.

Proxy settings were preserved; no alternate route, proxy bypass, login-cache
write or credential disclosure was attempted. Only the authenticated username,
token role and sanitized error type/status were reported. Credentials were
loaded inside the container from a read-only mount, not printed or supplied as
literal command arguments.

**Required action:** user/IT restores authorized proxy access to this storage
host and any official storage destinations negotiated by the Hub. Enter proxy
credentials privately, never in assistant chat. Recheck the real upload path
after connectivity is repaired.

## Blocker 2: strict current-source provenance does not match

Fresh hashing of the complete collection evidence matches its **current** seal:

`ae8828eab71c6709222c6fdcc1af4e026f07bcac2aa33b105da4830c25dcf52c`.

However, the publication preflight stopped at comparison of recorded versus
current source/runtime provenance. Exactly two entries differ:

| File | Recorded SHA256 | Current SHA256 |
| --- | --- | --- |
| `experiments/fixed_layout/media.py` | `e468b3e356fabb74a808fa4fb1e4a3116c62210937c19e821450599587966d28` | `665ec87dfa6bf56654e9f171a76c0106e882720086fcaa8d02ddef2270b7afe1` |
| `experiments/fixed_layout/record.py` | `9f95047aaae7b08034dd1ba2d10378ee5190306b74af1f6fe0391d87f9dcdee7` | `af5de7ac6ccc560d5a6ae2c1362a1e206d8e1b29b1715054fed2e944be1f235e` |

Git commit `2435dca5b534446818f02aeb01017c04809a0933` on September 20 changed
these shared helpers to route encoder ticks through the active attempt during
broad-workspace retries. Hashing both files from that commit's parent reproduces
the recorded hashes exactly. This identifies recoverable historical source,
not evidence of corrupt dataset videos or actions.

There is a separate evidence-history discrepancy: the September 19 completion
report records seal
`f8b10a498798217a63564416ab6561e25bd7e9c87e4fdddcf42f288cdb9e9a28`, whereas the
current seal is the value above. Filesystem modification times show validation
and seal files updated on September 20. The current seal verifies internal
consistency, but does not explain that historical change. No matching reseal
explanation was found in the searched project documentation.

**Required continuation:** reconcile the September 20 validation/seal history
and validate with an isolated, hash-matching historical source snapshot. Do not
reset the user's checkout, rewrite the saved provenance/seal, or relax the
validator to make it pass. The later metadata, bank and replay-report checks in
the attempted fresh audit were not reached after the provenance assertion;
there is no fresh full-admission claim.

## Existing uploader assessment and safe continuation

[The existing uploader](../huggingface/push_dataset_to_hub.py) is a useful API
example but should not be run with its defaults for this dataset:

1. It expects `meta/info.json` directly under `DATASET_DIR`; explicitly select
   the nested dataset root above and the new repository ID, not a stale value
   from the environment file.
2. It does not perform the collection admission or upload-payload audit, add a
   dataset card, or verify remote file hashes.
3. It uploads only to the version branch, leaving the default branch without
   the dataset/card. Prefer publishing to `main`, then creating `v3.0` at the
   verified resulting commit.
4. It persists a login token and prints raw exception messages. The latter can
   expose signed storage URLs. It also catches upload failures without a
   failing exit status. Use process-local credentials and sanitized errors.

After both blockers are resolved: recheck identity and target absence, audit
the exact allowlisted payload for secrets, add an accurate card, create a private
repository, upload with collision protection, and verify every remote filename,
size and LFS SHA256/Git blob hash. Verify `main` and `v3.0` point to the same
immutable commit and record that commit here. Recheck local integrity afterward.
Do not claim publication success from repository creation or an SDK call alone.