# gen_scene — randomization engine for the generalization programme (G)

Shared by G0 (the benchmark's test conditions) and G1/G2 (training randomization). Design:
[G0 spec](../../docs/superpowers/specs/2026-11-06-g0-generalization-benchmark-design.md) (D2, D5, D6, D9),
[programme](../../docs/superpowers/specs/2026-11-06-generalization-programme-design.md). Not hash-bound until a contract is
written (G0 binds it through `gen_eval`'s core hash). It imports `props_scene` and `robosuite_scene` read-only; nothing in them is edited.

## Textures (Task 1)

- 120 CC0 colour maps: 60 table-like, 60 floor/wall, 12 families each, 5 materials per family
  (the five with the smallest sha256 of the id among those with a square 1024 px colour map).
- **Source caveat (2026-11-09):** ambientcg.com and polyhaven.com answered `407` through the proxy. The maps come from the Hugging Face
  mirror `nyuuzyou/ambientcg` (cc0-1.0), pinned at revision `50329ad5…`, where each material's 1K colour map was
  **re-compressed to JPEG q90**. Every shard and the metadata are checked against the sha256 the Hub published for that revision;
  the extracted maps are pinned by their own sha256 in [assets_manifest.json](assets_manifest.json) (committed). Poly Haven is not used.
- Files: `.cache/gen_textures/<AmbientCG id>.jpg` (gitignored; mounted read-only at `/vla/.cache/gen_textures`). Fetch on the HOST with
  `OUT=<repo>/.cache/gen_textures scripts/gen_fetch_textures.sh`; containers stay `--network none`.
- **Split (fixed, committed before any G1 episode exists):** per surface, sort by sha256 of the file name; every 5th (index 4, 9, ...)
  is the **test pool** (12 per surface, G0 only), the rest the **train pool** (48). `tests/test_pools.py` pins the manifest and the
  split by digest. Moving one file across pools voids every G0 number.
- Primitive distractors ([pools.py](pools.py)): pool A (train) boxes and cylinders in red, green, yellow, purple, white, brown;
  pool B (test) spheres, ellipsoids and capsules in pink, lime, black, grey, olive. No shape family or colour is shared, and no colour
  is within 0.25 (RGB distance) of the other pool or of a bin.

Tests (admission image): see the commands in [docs/OPERATIONS.md](../../docs/OPERATIONS.md); the texture folder must be mounted.
