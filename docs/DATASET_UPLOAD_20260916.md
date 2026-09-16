# Private dataset upload — 2026-09-16

## Authorization and status

The user authorized uploading the validated dataset to their Hugging Face
account using credentials loaded privately from the environment file, choosing
an appropriate repository name, then preparing the SmolVLA pilot training
protocol after upload completion. Training execution is not authorized.

**BLOCKED: repository created privately; dataset upload did not complete.**

- Repository: https://huggingface.co/datasets/VivekSai07/franka-panda-cube-stacking-pilot-420
- Authenticated account matched configured username `VivekSai07`.
- The target did not exist before creation; no existing dataset was overwritten.
- Intended payload: 442 validated dataset files, 2,399,107,881 bytes, plus a
  descriptive dataset card. Separate tuning/final scene banks were excluded.
- Fresh `require_replay_pass` succeeded before upload, including final pilot
  validation binding, source/runtime and dataset fingerprint checks.
- Dataset fingerprint:
  `94d3a6a717c74995c7688a835ccccfee2e27754aad14fb3affb1083e60ee4b28`.
- Dataset was mounted read-only. Local metadata, data, videos and sidecars were
  not renamed or rewritten. Credentials were loaded inside the container and
  were not printed or included in the payload. JSON/JSONL metadata was checked
  for configured credential values; upload paths/types were allowlisted.

## Failure and confirmed remote state

Hugging Face account/repository API calls succeeded. Large-file upload failed
with **407 Proxy Authentication Required** at:

`hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com`

This is a proxy authentication failure, not evidence of an invalid Hugging Face
token. No alternate route or proxy bypass was attempted. Exception details and
signed upload URLs must not be copied into reports.

After failure, a fresh remote inspection confirmed:

- Repository remains **private**.
- Only the initial Git attributes file is committed; no dataset/card commit.
- Only `main` exists; the intended `v3.0` branch was not created.
- Initial HEAD: `48e306f2c9965e0d77ea8db88f4c8356dbad584c`.

Some uncommitted large-file objects may have transferred before failure; the
repository is not a usable uploaded dataset. No remote objects were deleted.

## Required continuation

1. User/IT restores authorized proxy access to the named storage endpoint.
   Any proxy secrets must be entered privately, never into assistant chat.
2. Recheck account, repository privacy, initial HEAD and inventory before
   reusing this now-existing repository. Do not blindly overwrite newer work.
3. Recheck the local validation binding. Upload the unchanged allowlisted
   payload and card to `main`, then create `v3.0` at the resulting commit.
4. Verify every remote filename, size and LFS SHA256/Git blob hash against the
   local payload; verify privacy and both branch targets. Record immutable SHA.
5. Only then prepare the SmolVLA pilot protocol, as the user requested.

Protocol preparation must preserve explicit 300 training / 120 validation IDs,
train-only normalization, independent tuning/final scenes and fresh pretrained
initialization. No automatic training, pre-fix resume or model upload.