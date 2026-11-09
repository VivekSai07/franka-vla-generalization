#!/usr/bin/env bash
# Host-side texture fetch for G0/G1 (never run inside a container; containers stay --network none).
# Downloads the pinned Hugging Face mirror of ambientCG (about 6 GB, to a staging folder outside outputs/ and datasets/),
# verifies every file against the Hub's published sha256, extracts the chosen colour maps into $OUT and writes the manifest.
#   OUT=/home/utxowqr/smolvla_franka/.cache/gen_textures  STAGE=/tmp/gen_stage  scripts/gen_fetch_textures.sh
# Run from the repository (or worktree) root. The staging folder can be deleted afterwards.
set -euo pipefail
OUT="${OUT:?set OUT to the texture folder (.cache/gen_textures)}"
STAGE="${STAGE:-/tmp/gen_stage}"
REV=50329ad52d825bfb740080546c30c04014983ec8
BASE="https://huggingface.co/datasets/nyuuzyou/ambientcg/resolve/$REV"
mkdir -p "$STAGE"
for f in metadata.jsonl.zst dataset_00.tar dataset_01.tar dataset_02.tar dataset_03.tar dataset_04.tar dataset_05.tar dataset_06.tar dataset_07.tar; do
  [ -s "$STAGE/$f" ] || curl -fsSL -C - -o "$STAGE/$f" "$BASE/$f"
done
python3 -m experiments.gen_scene.fetch --stage "$STAGE" --out "$OUT" --manifest experiments/gen_scene/assets_manifest.json
