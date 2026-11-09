# gen_scene — randomization engine for the generalization programme (G)

Shared by G0 (the benchmark's test conditions) and G1/G2 (training randomization). Design:
[G0 spec](../../docs/superpowers/specs/2026-11-06-g0-generalization-benchmark-design.md) (D2, D5, D6, D9),
[programme](../../docs/superpowers/specs/2026-11-06-generalization-programme-design.md). Not hash-bound by a contract of its own: G0's
core hash (`gen_eval/contract.py`) binds every module here and the texture manifest. It imports `props_scene` and `robosuite_scene`
read-only; nothing in them is edited. Tests: `tests/` (admission image for the pure ones, robosuite image for the rest; the texture
folder and the sealed P2 data must be mounted, see [docs/OPERATIONS.md](../../docs/OPERATIONS.md)).

| module | what | image |
|---|---|---|
| `pools.py` | texture manifest, the fixed train/test split, primitive pools A and B | either |
| `fetch.py` | choose, verify and extract the textures (host, `scripts/gen_fetch_textures.sh`) | host |
| `conditions.py` | `Condition`: every departure from P2's scene as one JSON record, `pool` checked against the pools, `NOMINAL`, `resolve()` | either |
| `levels.py` | every level of the five axes with its numbers, the placement draws, the rotation that gives a layout one level per axis | either |
| `compose.py` | the scene under a condition (XML edits before compiling, light values after), `reset`, bounded model cache | robosuite |
| `expert.py` | `GenExpert`: the sealed P2 expert reading the table height and the bins from the model | robosuite |
| `metrics.py` | `placement` measured against the bin's own height | either |
| `gates.py` | gates D9 (i) and (ii) as a CLI (`equality`, `expert`) | robosuite |
| `tools/expert_probe.py` | expert reach evidence for the geometry levels; not part of any protocol | robosuite |

## Textures (Task 1)

- 120 CC0 colour maps: 60 table-like, 60 floor/wall, 12 families each, 5 materials per family
  (the five with the smallest sha256 of the id among those with a square 1024 px colour map).
- **Source caveat (2026-11-09):** ambientcg.com and polyhaven.com answered `407` through the proxy. With the user's approval the maps
  come from the Hugging Face mirror `nyuuzyou/ambientcg` (cc0-1.0), pinned at revision `50329ad5…`, where each material's 1K colour
  map was **re-compressed to JPEG q90**. Every shard and the metadata are checked against the sha256 the Hub published for that revision;
  the extracted maps are pinned by their own sha256 in [assets_manifest.json](assets_manifest.json) (committed). Poly Haven is not used.
- Files: `.cache/gen_textures/<AmbientCG id>.jpg` (gitignored; mounted read-only at `/vla/.cache/gen_textures`). Fetch on the HOST with
  `OUT=<repo>/.cache/gen_textures scripts/gen_fetch_textures.sh`; containers stay `--network none`. MuJoCo reads PNG only, so
  `compose.texture_png` decodes the JPEG and hands MuJoCo the PNG bytes in memory.
- **Split (fixed, committed before any G1 episode exists):** per surface, sort by sha256 of the file name; every 5th (index 4, 9, ...)
  is the **test pool** (12 per surface, G0 only), the rest the **train pool** (48). `tests/test_pools.py` pins the manifest and the
  split by digest. Moving one file across pools voids every G0 number.
- Primitive distractors ([pools.py](pools.py)): pool A (train) boxes and cylinders in red, green, yellow, purple, white, brown;
  pool B (test) spheres, ellipsoids and capsules in pink, lime, black, grey, olive. No shape family or colour is shared, and no colour
  is within 0.25 (RGB distance) of the other pool or of a bin. A `Condition` declares `pool="train"` or `"test"` and `check()` refuses
  anything outside it, so a test texture or a pool B primitive cannot reach a training scene (and the reverse).

## The engine (Task 2)

Under the nominal condition the XML is the very string `props_scene` compiles: **gate D9 (i)** (`python -m experiments.gen_scene.gates
equality`) compares the compiled model (every array, bit for bit, a NaN equal to a NaN: `mesh_polynormal` holds NaNs even between two
compiles of one XML), the headlight, the initial state (also against the sealed one) and both 512 px renders for three sealed P2 episodes.

Levels (all numbers are pinned by `tests/test_levels.py`; layout `n` of a tier gets level `(n + offset) % levels` on each axis):

| axis | level | what |
|---|---|---|
| lighting | `dark` | every light's diffuse and ambient x0.3 (headlight included) |
| | `bright` | every light's diffuse and specular x2.5 |
| | `tinted` | every light's diffuse, specular and ambient x (1, .75, .5) warm or (.6, .75, 1) cool; alternates within the level |
| | `low_angle` | the scene light 20 degrees above the horizon from +y, shadows on, headlight x0.3 |
| background | `table` / `room` / `both` | test-pool textures on the table / on the floor and the six walls (two different maps) / all |
| distractors | `three` / `four` / `five` | the layout's own distractors kept, props of the six added to three, then 0 / 1 / 2 pool B primitives; region x 0.25-0.52, y +-0.20 |
| spatial | `table_up5` / `table_down5` / `table_up10` | table top, both bins and the drop height move together; the slab reaches the floor, which drops with a lower table |
| | `outer_start` | target start x 0.42-0.50, y +-0.12 (P2's region ends at x = 0.40); both bins +-4 cm in y, apart or together (together flips to apart if a bin would come within 2 cm of an object) |
| disturbance | `move` / `drop` | target moved 8 cm / gripper forced open for 10 ticks (triggers in `gen_eval`) |

Where the spec was not literal: `bright` scales the specular as well (the scene light's specular is 0.3, not off); `dark` leaves the
specular alone; "warm or cool by layout parity" alternates within the level because layout parity is constant inside a level;
a disturbance axis has two levels, so each has 72 layouts, not 36-48; added objects are placed one at a time under P2's spacing and
hand-clearance rules (primitives are only ever the neighbour); the `move` destination is 8 cm from the start under P2's full spacing,
else under P2's distance only, else only without overlap (the expert pass, D8, decides what remains); the outer-start and distractor
regions were chosen with the expert's reach evidence below, before any policy trial.

## The expert (Task 3)

`GenExpert` is a fork, not a subclass, of `PropsExpert` with `P2Expert`'s carry: every absolute height is `table_z` plus the original
offset, so with `table_z` = 0.0 each target is the original's float. **Gate D9 (ii)** (`gates expert`): from the sealed initial state the
fork reproduces sealed episodes 0, 1 and 5 (cereal, milk, lemon): `qpos`, `ctrl` and `props_pose` for all 452 ticks and the final `qpos`
byte for byte.

Expert reach (unrendered, `tools/expert_probe.py`, 48 fresh layouts per level, 2026-11-09): nominal 48/48; `table_up5`, `table_down5`,
`table_up10` 48/48 each; `outer_start` 47/48 (one failed grasp); `three`, `four`, `five` 48/48 each; a fresh episode from the `move`
destination 47/48 (one failed grasp). Nothing was tuned to raise these.
