# Props P2 — collection Implementation Plan

> **For agentic workers:** executed inline in the session that wrote it (user asked to launch the
> same night). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `experiments/props_p2_data/`, pass the expert re-test and a sharded pilot, then
collect, merge and seal 540 P2 episodes with 12 parallel CPU-rendered shards.

**Architecture:** A new collection package on top of the accepted, frozen `experiments/props_scene/`
(imported, never edited). New: the P2 plan (distractors, spacing, 120 reserved layouts), a `PropsExpert`
subclass with a 0.19 m carry, a bounded model cache, a `distractors` admission gate, a sharded
scheduler, a verified shard merge and the seal.

**Tech Stack:** Python 3.10, robosuite 1.5.2 / MuJoCo 3.4.0 image, LeRobot 0.4.4, numpy.

**Spec:** [../specs/2026-10-13-props-p2-distractors-design.md](../specs/2026-10-13-props-p2-distractors-design.md).

## Global Constraints

- Never edit: `experiments/props_scene/**`, `experiments/props_p1_data/**`, `src/**`,
  `data_collection/**`, `experiments/paired_demonstrations/**`, `experiments/pose_snapshot_v2/**`,
  `experiments/robosuite_scene/**`, the 2026-10-05 props spec, the P1 plan. Import read-only.
- P2 must not import `experiments.props_p1_data` (keeps P2's provenance independent of P1's package).
- Images: robosuite `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e` (scene,
  collection, merge); admission `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`
  for tests that import no simulator. Flags: `--read-only --network none --tmpfs /tmp:rw,size=16g
  --ulimit nofile=65536:65536 -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/vla`.
- Counts: 480 train (20 cells × 24: 12 one-distractor, 12 two), 60 validation (20 × 3; 30/30), 540
  admitted; reserved 120 = screen 20 + report 60 + control 20 + confirm 20.
- Seeds `sha256("props_p2_v1:20261013:<identity>")[:4]` big-endian; balance by shuffled decks.
- Carry 0.19 m; spacing `r_a + r_b + 0.03`, target–distractor ≥ `0.065 + r_d`; distractor gate 0.01 m /
  5°. Success rule and P1 gates never weakened.
- Commits: user identity, no trailers, short messages; never commit `outputs/`; do not push.
- Output root: `outputs/props_p2_data/props_p2_<YYYYMMDD>/` (shards under `shards/`).

## Review Focus

1. A held-out pair is collected as a target (test_plan, validate_dataset).
2. Colour predicts slot, or distractor count/identity predicts the target (test_plan: per-colour slot
   balance; distractor identity deck balance; count split exact).
3. A layout violates spacing or the region (test_plan over every episode, resample and reserved layout).
4. A rejected attempt leaves frames in a shard, or a shard skips/duplicates a slot (test_record).
5. The merge reorders, drops or alters an episode (test_merge on two synthetic shards; pilot on real ones).
6. Forked residue: no `props_p1_data` import anywhere in the package (test_contract).

## File map

| file | responsibility |
|---|---|
| `plan.py` | cells, distractor decks, spacing sampler, resample/attempt draws, 120 reserved layouts, gate attempts; pure |
| `contract.py` | `IMAGE`, `OUTPUT`, `provenance()` over this package + `props_scene` + shared + `src` + spec/plan; pure |
| `expert.py` | `P2Expert(PropsExpert)`: 0.19 m carry; robosuite image |
| `episode.py` | one attempt with distractors, bounded expert cache, distractor displacement |
| `validate.py` | `admit` (P1 gates + `distractors`), `fingerprint`, `validate_dataset`, `seal` |
| `record.py` | sharded scheduler; `--pilot/--collect/--resume/--status` per shard; `--merge`; `--gate` |
| `merge.py` | `aggregate_datasets` over shards + verification against the shards |
| `README.md` | what, images, gate order, commands |
| `tests/` | `test_plan`, `test_contract`, `test_validate`, `test_record` (admission image); `test_merge`, `test_episode` (robosuite image) |

## Tasks

- [ ] **1. Plan** — `plan.py` + `test_plan.py` (Review Focus 1–3). Run in the admission image.
- [ ] **2. Contract** — `contract.py` + `test_contract.py` (Focus 6; provenance stable; spec/plan bound).
- [ ] **3. Expert, episode, validate** — `expert.py`, `episode.py`, `validate.py`; `test_validate.py`
  (admission image, fake replay), `test_episode.py` (robosuite image: carry level reaches 0.19 m, a
  2-distractor episode succeeds with distractors undisturbed, bounded cache rebuild is bit-identical,
  footprint constants ≥ compiled half-diagonals, instruction wording equals the expert's).
- [ ] **4. Sharded scheduler** — `record.py` + `test_record.py` (Focus 4; shard slot blocks partition
  0..539 exactly; resume per shard).
- [ ] **5. Merge** — `merge.py` + `test_merge.py` (Focus 5; two tiny synthetic shards).
- [ ] **6. Expert re-test** — `record --gate`: 288 attempts, 12 workers, no rendering; 12 rendered
  attempts re-run in a second process; `LP_NUM_THREADS` pixel check. Report numbers; stop if a bar fails.
- [ ] **7. Pilot** — 2 shards × 6 slots → merge → validate on a scratch root. Watch the first episodes.
- [ ] **8. Launch** — 12 shards `--collect`, poll; `--merge`; seal; independent re-verification.
- [ ] **9. Report + docs pass** — `docs/PROPS_P2_COLLECTION_<date>.md`, TODO, BENCHMARK_STATUS, INDEX.

## Running things

```bash
MAIN=/home/utxowqr/smolvla_franka
OLD=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
NEW=sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
run() { local image=$1; shift; docker run --rm --runtime runc --network none --read-only \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=16g --ulimit nofile=65536:65536 \
  -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/vla -v "$MAIN:/vla:ro" -v "$MAIN/outputs/props_p2_data:/out:rw" -w /vla \
  --entrypoint python "$image" "$@"; }
run $OLD -m unittest discover -s experiments/props_p2_data/tests -t experiments/props_p2_data/tests -p 'test_[pcvr]*.py'
run $NEW -m unittest discover -s experiments/props_p2_data/tests -t experiments/props_p2_data/tests -p 'test_[me]*.py'
```
