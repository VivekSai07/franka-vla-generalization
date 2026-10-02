# B3 robosuite collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `experiments/b3_robosuite_data/` and use it to collect and seal the B3 dataset: 500 training and 60 validation scripted episodes in the robosuite scene, plus a banked set of 60 reserved evaluation layouts.

**Architecture:** A new collection package on top of `experiments/robosuite_scene/` (imported, extended to four cubes; not copied). `plan.py` is pure and decides everything in advance; `scenes.py` turns a planned layout into a settled simulator state; `expert.py` runs the scripted expert; `validate.py` admits or rejects one episode and seals the dataset; `record.py` schedules slots with a journal so the 12-hour run can resume. Frames reach the dataset only after an episode is admitted.

**Tech Stack:** Python 3.10, robosuite 1.5.2, MuJoCo 3.4.0, numpy 2.2.6, LeRobot 0.4.4, pinocchio. CPU `runc` containers. Two images: the robosuite image for anything that builds the scene, the admission image for the pure tests.

**Spec:** [../specs/2026-10-03-b3-robosuite-cubes-design.md](../specs/2026-10-03-b3-robosuite-cubes-design.md) — read sections "What the user wants", 1 and 2 before starting. This plan implements sections 1 and 2 only. Training and evaluation get their own plans after the dataset is sealed.

**Branch / worktree:** `feat/b3-robosuite-cubes`, worktree `.worktrees/feat-b3-robosuite-cubes`, cut from `main`. Load the `working-on-the-robosuite-path` and `collecting-a-dataset` skills before running anything.

## Global Constraints

- Do not edit `src/**`, `data_collection/**`, `validation/**`, `experiments/fixed_layout/**`, `experiments/paired_demonstrations/**`, `experiments/pose_snapshot_v2/**`, or any existing `experiments/*_data/**`, `experiments/xvla_*/**`, `experiments/smolvla_*/**`. `experiments/robosuite_scene/` may be extended, with its existing tests kept green.
- Never weaken a gate, the `stack_metrics` window (xy < 0.02, 0.035 < z < 0.045), or a test tolerance to get an episode admitted. An infeasible layout is substituted, not forced.
- Never call `env.step()`. One `mujoco.Renderer` at a time. No `env.sim.render`.
- Images: robosuite `sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e` (MuJoCo 3.4.0); admission `sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc`. Never retag or prune images; the machine is shared.
- Colours `("red", "green", "blue", "yellow")`; rgba `0.8 0.2 0.2`, `0.2 0.8 0.2`, `0.2 0.2 0.8`, `0.8 0.8 0.2`; cube half-size 0.02, mass 0.05.
- Held-out pairs, never in a training or validation episode: `("yellow", "red")`, `("green", "blue")`.
- Regions: A = x ∈ [0.20, 0.35], y ∈ [−0.45, −0.20]; B = x ∈ [0.20, 0.35], y ∈ [0.20, 0.45]. Distractor band: x ∈ [0.20, 0.35], y ∈ [−0.45, 0.45].
- Edge bands for the pick cube: inner corner x ∈ [0.20, 0.25], |y| ∈ [0.20, 0.25]; outer corner x ∈ [0.32, 0.35], |y| ∈ [0.40, 0.45].
- Counts: 20 cells (10 trained pairs × 2 sides); 25 training + 3 validation episodes per cell; 500 + 60 = 560 admitted episodes; 60 reserved layouts (20 screen, 40 report).
- Clearances: `MIN_CLEARANCE_M` starts at 0.08 and is frozen by the pilot (ladder 0.08 → 0.09 → 0.10); near distractor 0.08–0.12 m from its target (lower bound = `MIN_CLEARANCE_M`); far distractor ≥ 0.15 m from both targets; distractor moved > 0.02 m ⇒ reject.
- Jitter: home pose ±1.0° per arm joint; hover z ±0.03 m on `pre_grasp` / `pre_place`. Episode: 528 frames, 30 Hz.
- Instruction text comes only from `experiments.paired_demonstrations.plan.instruction(pick, place)`. LeRobot schema comes only from `experiments.paired_demonstrations.media` (`FEATURES`, `ENCODING`, `CAMERAS`, `PairedDataset`).
- Seeds: `int.from_bytes(sha256(f"{VERSION}:{MASTER}:{identity}").digest()[:4], "big")`, `VERSION = "b3_robosuite_data_v1"`, `MASTER = 20261003`. Balance is by construction (shuffled decks), never by sampling.
- `plan.py` and `contract.py` import nothing from robosuite, mujoco, torch or lerobot.
- Commits: user identity only, no trailers, short messages; no outputs, images or videos. The user pushes; do not push.
- Outputs: `outputs/b3_robosuite_data/b3_robosuite_<YYYYMMDD>/` in the main checkout, mounted at `/out`.

## Unattended execution (the user's instruction, 2026-10-03)

This plan is meant to be executed overnight by an agent with nobody watching, ending with a
detached container that finishes the collection on its own. So:

- **No step waits for a human.** Where a decision is needed the executor decides, in favour of the
  spec, and records it in `STATUS.md` under "Decisions taken" with what it costs if wrong.
- **What the executor may never decide:** loosening or skipping a gate or test tolerance; changing a
  count, a region, a band, the held-out pairs or the success window; editing a path the Global
  Constraints forbid; raising `MIN_CLEARANCE_M` above 0.10 or lowering it below 0.08; deleting or
  overwriting anything under `outputs/` or `datasets/` except its own scratch under
  `outputs/b3_robosuite_data/`; touching any Docker image or container that is not this project's;
  pushing to a remote. If the only way forward needs one of these: stop and write why.
- **`STATUS.md`** at `outputs/b3_robosuite_data/STATUS.md` (main checkout) is rewritten at every
  stage boundary and whenever something stops. It is what the user reads in the morning. Always:
  time, branch and commit, current stage, what is running (container name) and how to check it
  (`docker ps -a --filter name=b3-collect`, `docker logs <name> 2>&1 | grep '^B3 ' | tail`,
  `--status`), what passed with its numbers, decisions taken, and — if stopped — the exact reason
  and the last journal lines.
- **The collection container is detached and self-finishing.** `--collect` runs to the seal by
  itself, so it does not depend on the agent's session staying alive. The executor checks it at 5,
  15 and 60 minutes and then as long as its session lasts, updating `STATUS.md` each time.
- **What is left for the morning** if the session ends before the container does: Task 8 Steps 4–7
  (verify the seal, write the report, documentation pass, final commit). Any agent can do those
  from `STATUS.md` and the output directory.
- **Expected timing:** Tasks 1–6 are implementation; the pilot is under an hour; the collection is
  about 11–12 hours from launch. A launch at 01:00 seals around 13:00.

## Review Focus

1. **Colour predicts position.** If a colour were more often in one region, a policy could skip the instruction. `test_plan` asserts every colour is the pick cube equally often in region A and region B, and likewise as place cube, over the training set.
2. **A held-out pair leaks into training.** `test_plan` asserts no training or validation episode has a held-out (pick, place), and that every reserved held-out layout does.
3. **A rejected attempt leaves frames in the dataset.** `test_record` asserts the dataset's episode count equals the journal's admitted count after a run with injected rejections, and that the pending buffer is empty.
4. **Resume duplicates or skips a slot.** `test_record` kills a fake run mid-way, resumes, and asserts every slot is admitted exactly once in plan order.
5. **A distractor is disturbed but the episode is admitted.** `test_validate` feeds a trace where a distractor moved 2.5 cm and asserts rejection; and one where the *pick* cube moved (it must) and asserts that alone does not reject.

## File map

| file | responsibility |
|---|---|
| `experiments/robosuite_scene/scene.py`, `xml_contract.py`, `validate.py` | extended: any subset of the four colours; replay takes pick/place |
| `experiments/b3_robosuite_data/plan.py` | every count, seed, layout, episode and reserved layout; pure |
| `experiments/b3_robosuite_data/contract.py` | `IMAGE`, `SPEC`, `PLAN`, `OUTPUT`, `provenance()`; pure |
| `experiments/b3_robosuite_data/scenes.py` | `build_model`, `reset`, `validate_geometry`, `make_bank` |
| `experiments/b3_robosuite_data/expert.py` | `B3Expert`, `run_attempt` |
| `experiments/b3_robosuite_data/validate.py` | `admit`, `distractor_disturbance`, `fingerprint`, `validate_dataset`, `seal` |
| `experiments/b3_robosuite_data/record.py` | `run_schedule`, journal, `--pilot` / `--collect` / `--resume` / `--status` |
| `experiments/b3_robosuite_data/README.md` | what it is, gate order, commands |
| `experiments/b3_robosuite_data/tests/` | `test_plan`, `test_contract`, `test_expert_fork`, `test_record`, `test_validate` (admission image); `test_scenes`, `test_expert` (robosuite image) |
| `docs/B3_ROBOSUITE_COLLECTION_<YYYYMMDD>.md` | collection report (Task 8) |

Shared names later tasks must not rename:

```python
# plan.py
COLOURS, HELD_OUT_PAIRS, TRAINED_PAIRS, SIDES, CELLS, MIN_CLEARANCE_M
def seed(identity: str) -> int
def make_plan() -> dict            # keys below
def validate_plan(plan: dict) -> None
def attempt_draw(episode: dict, k: int) -> dict   # {"seed", "home_pose_jitter_deg"} for attempt k; k=0 is the planned draw
def resample_layout(episode: dict, r: int) -> dict  # r=0 returns the episode; r>=1 the same slot with freshly drawn positions
# plan dict keys: version, master_seed, rng_seeds, layouts, episodes, train_ids, validation_ids,
#   evaluation_holdout, evaluation_screen_ids, evaluation_report_ids, min_clearance_m, ...
# episode keys: episode_index, split, cell, pick, place, side, colours (sorted, present in scene),
#   distractors, profile, edge, layout_id, cube_xy_m {colour: [x, y]}, instruction, task_index,
#   seed, home_pose_jitter_deg, hover_z_offset {"pre_grasp", "pre_place"}
# scenes.py
def build_model(colours: tuple) -> mujoco.MjModel            # cached per sorted colour tuple
def reset(model, cube_xy_m: dict, home_pose_jitter_deg: list) -> dict   # {"qpos", "initial_state"}
def make_bank(plan: dict) -> tuple                            # (arrays: dict[str, ndarray], manifest: dict)
# expert.py
def run_attempt(episode: dict, draw: dict, sink) -> dict      # run_episode-style result + "model", "reset"
# validate.py
def admit(episode, result, frames, traces) -> dict            # {"admitted": bool, "gates": {...}, "reason": str}
def fingerprint(root) -> str
def seal(root, plan, provenance) -> dict
# record.py
def run_schedule(plan, execute, journal_path, sink_factory, resume=False) -> dict
```

## Running things

From the worktree root. Shell state does not persist between calls; redefine `run` each time.

```bash
MAIN=/home/utxowqr/smolvla_franka
OLD=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
NEW=sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
mkdir -p "$MAIN/outputs/b3_robosuite_data"
run() {  # run <image> <python args...>
  local image=$1; shift
  docker run --rm --runtime runc --network none --read-only \
    --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=8g \
    -e MUJOCO_GL=osmesa -e HF_HOME=/tmp/hf -e XDG_CACHE_HOME=/tmp/cache -e HOME=/tmp \
    -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e OMP_NUM_THREADS=1 -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONPATH=/vla \
    -e NUMBA_CACHE_DIR=/tmp/numba -e GIT_OPTIONAL_LOCKS=0 \
    -v "$PWD:/vla:ro" -v "$MAIN/.git:$MAIN/.git:ro" \
    -v "$MAIN/outputs/b3_robosuite_data:/out:rw" \
    -w /vla --entrypoint python "$image" "$@"
}
# tests:  run "$OLD" -m unittest experiments.b3_robosuite_data.tests.test_plan -v
```

Long runs: `docker run -d --name b3-collect-<date> ...` (same flags without `--rm`), then poll `--status` and `docker logs`. Never rely on `docker wait`.

---

### Task 1: the scene takes any subset of the four colours

**Files:**
- Modify: `experiments/robosuite_scene/scene.py`, `xml_contract.py`, `validate.py`
- Modify: `experiments/robosuite_scene/tests/test_model_contract.py`
- Create: `experiments/robosuite_scene/tests/test_colours.py`

**Interfaces:**
- Produces: `build_sim(colours=("red", "green")) -> (env, model, data)`; `xml_contract.CUBE_RGBA: dict[str, str]`; `replay_recorded(result, frames, traces, pick="red", place="green")` and `replay_reference(...)` with the same two keyword arguments.

- [ ] **Step 1: Write the failing test** (`test_colours.py`, robosuite image)

```python
class ColourTests(unittest.TestCase):
    def test_four_cube_model_matches_the_current_scene_cubes(self):
        from experiments.robosuite_scene.scene import build_sim
        from src.robots.franka_panda.config import SCENE_PATH
        _, model, _ = build_sim(colours=("red", "green", "blue", "yellow"))
        current = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.assertEqual(model.nq, 9 + 4 * 7)
        for colour in ("red", "green", "blue", "yellow"):
            body = model.body(f"target_cube_{colour}")
            self.assertAlmostEqual(float(model.body_mass[body.id]), 0.05, places=6)
            ours = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body.id and model.geom_contype[g]]
            theirs = current.geom(f"cube_geom_{colour}").id
            self.assertEqual(len(ours), 1)
            for field in ("geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_margin"):
                np.testing.assert_allclose(getattr(model, field)[ours[0]], getattr(current, field)[theirs])
            visible = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body.id and model.geom_group[g] < 3]
            np.testing.assert_allclose(model.geom_rgba[visible[0]][:3], current.geom_rgba[theirs][:3], atol=1e-6)

    def test_only_requested_cubes_exist(self):
        from experiments.robosuite_scene.scene import build_sim
        _, model, _ = build_sim(colours=("blue", "yellow"))
        names = {model.body(i).name for i in range(model.nbody)}
        self.assertEqual({n for n in names if n.startswith("target_cube_")},
                         {"target_cube_blue", "target_cube_yellow"})
        self.assertEqual(model.nq, 9 + 2 * 7)

    def test_default_is_still_red_and_green(self):
        from experiments.robosuite_scene.scene import build_sim
        _, model, _ = build_sim()
        self.assertEqual(model.nq, 23)
```

- [ ] **Step 2: Run it** — `run "$NEW" -m unittest experiments.robosuite_scene.tests.test_colours -v` → FAIL (`build_sim() got an unexpected keyword argument`).

- [ ] **Step 3: Implement**
  - `xml_contract.py`: add `CUBE_RGBA` for the four colours; the cube rename and contact rewrite iterate over every body named `target_cube_<colour>_main` found in the XML (raise `ValueError` if none, or if a name is not one of the four colours) instead of the two hardcoded names.
  - `scene.py`: `BringupEnv.__init__(self, colours=("red", "green"), **kwargs)` stores `colours` before `super().__init__`; `_load_model` creates one `BoxObject` per colour in the given order with its rgba; `build_sim(colours=("red", "green"))` passes `colours` to `robosuite.make` and rejects duplicates, unknown colours and fewer than two. `overwrite_reset` is unchanged.
  - `validate.py`: `_replay` takes `pick`, `place` and passes them to `stack_metrics`; both public replay functions gain the two keyword arguments with the old defaults.

- [ ] **Step 4: Run** — `test_colours` PASS, and every existing robosuite_scene test still PASS: `test_r2_render test_model_contract test_r7_cameras test_r4_schema test_r5_reset test_r3_stack test_r6_drift` in `$NEW`; `test_r1_image test_seal test_expert_fork` in `$OLD`.

- [ ] **Step 5: Commit** — `git commit -m "Let the robosuite scene hold any of the four cube colours"`.

---

### Task 2: the plan

**Files:**
- Create: `experiments/b3_robosuite_data/__init__.py`, `plan.py`, `tests/test_plan.py`

**Interfaces:** produces everything under `# plan.py` in the shared-names block. Pure Python + numpy.

**What `make_plan()` decides** (each is a named constant or a small function; tests pin each):

- `TRAINED_PAIRS`: the 12 ordered pairs minus `HELD_OUT_PAIRS`, sorted. `CELLS`: `(pick, place, side)` for trained pairs × `SIDES = ("A", "B")`, where `side` is the region of the **pick** cube; the place cube is in the other region.
- **Profiles.** Each training cell gets a shuffled deck of 25 cards `(distractor_count, placement)`: 5 × `(0, "none")`, 5 × `(1, "far")`, 5 × `(1, near)`, 5 × `(2, "far")`, 5 × `(2, near)`, where the ten `near` cards are 5 × `"near_pick"` and 5 × `"near_place"`; cells with even index give the one-distractor cards 3 `near_pick` + 2 `near_place` and the two-distractor cards 2 + 3, odd cells the reverse. In a two-distractor `near_*` profile the first distractor is near and the second is far. Validation: 3 cards per cell, `(0, "none")`, `(1, "far")`, `(2, "near_pick")`.
- **Distractor colours.** The two colours that are neither pick nor place, sorted. One-distractor episodes alternate between them through a per-cell deck.
- **Edge.** Each training cell gets a shuffled deck of 25: 20 × `"interior"` and 5 edge cards — even cells 3 × `"inner"` + 2 × `"outer"`, odd cells 2 + 3. Validation is all `"interior"`. Edge and profile decks are shuffled independently.
- **Layouts.** Interior: a 4-D Latin hypercube over (pick x, pick |y|, place x, place |y|) inside the regions, one hypercube per split (`train`: 400 points, `validation`: 60, `evaluation`: 45), using B2's `_lhs` construction (permutation stream + jitter stream per axis). Edge: pick xy uniform in the band on the pick side, place xy uniform in its region, from seeded streams. Sign of y comes from the side.
- **Distractor positions.** Rejection sampling in the distractor band, at most 500 draws per cube, raising `ValueError` on exhaustion (never a silent redraw of the layout). Constraints: `near_pick` / `near_place`: `MIN_CLEARANCE_M ≤ d(target) ≤ 0.12` and `≥ MIN_CLEARANCE_M` from the other target; `far`: `≥ 0.15` from both targets; every distractor `≥ MIN_CLEARANCE_M` from every other cube.
- **Episodes.** Training episodes are ordered by a seeded shuffle of all 500 (so a partial run is roughly balanced), then the four riskiest profiles are swapped to the front (two-distractor `near_pick` at an outer corner; two-distractor `near_place`; one-distractor `near_pick` at an inner corner; a `none` control). Validation episodes follow as indices 500–559. `task_index` is the index of the instruction in the sorted list of the 10 trained instructions.
- **Reserved layouts** (`evaluation_holdout`, keyed `evaluation_000`…`059`, never an episode): 20 screen (each trained pair × each side), 20 report trained (same), 20 report held-out (each held-out pair × 5 per side). In each tier a quarter are edge layouts (screen 5, report 10; alternating inner/outer), the rest from the evaluation hypercube. Each stores `pick`, `place`, `side`, `edge`, `cube_xy_m`, home and hover jitter, and `conditions`: `"control"` (no distractors) for report layouts, and `"distractor"` (both other colours; first near — pick on even layout indices, place on odd — second far) for all.
- `attempt_draw(episode, k)`: `k = 0` returns the planned seed and home jitter; `k ≥ 1` derives both from `seed(f"attempt:{episode_index}:{k}:...")`.
- `resample_layout(episode, r)`: `r = 0` returns the episode unchanged; `r ≥ 1` returns the same slot — same cell, profile, edge class, colours, instruction — with cube and distractor positions redrawn from `seed(f"resample:{episode_index}:{r}:...")` under the same constraints (an edge slot stays in its band), and a new `layout_id` `f"{old}_r{r}"`. This is the substitution an infeasible layout gets; the collector applies it automatically and records it.
- All seeds are recorded in `rng_seeds`; a duplicate seed value raises.

- [ ] **Step 1: Write the failing tests** (`test_plan.py`, admission image). One test per bullet; the assertions that matter most:

```python
    def setUpClass(cls): cls.plan = make_plan(); cls.train = [e for e in cls.plan["episodes"] if e["split"] == "train"]

    def test_counts(self):
        self.assertEqual(len(self.train), 500)
        self.assertEqual(len(self.plan["episodes"]), 560)
        self.assertEqual(Counter((e["pick"], e["place"], e["side"]) for e in self.train), {c: 25 for c in CELLS})
        self.assertEqual(Counter(len(e["distractors"]) for e in self.train), {0: 100, 1: 200, 2: 200})
        self.assertEqual(Counter(e["edge"] for e in self.train), {"interior": 400, "inner": 50, "outer": 50})

    def test_held_out_pairs_never_collected(self):
        for e in self.plan["episodes"]:
            self.assertNotIn((e["pick"], e["place"]), HELD_OUT_PAIRS)

    def test_colour_does_not_predict_region(self):
        for role in ("pick", "place"):
            for colour in COLOURS:
                sides = Counter(e["side"] for e in self.train if e[role] == colour)
                self.assertEqual(sides["A"], sides["B"])

    def test_every_colour_trained_in_both_roles(self):
        self.assertEqual({e["pick"] for e in self.train}, set(COLOURS))
        self.assertEqual({e["place"] for e in self.train}, set(COLOURS))

    def test_geometry(self):          # every episode and every reserved condition
        # targets inside their regions with pick on `side`; edge picks inside their band;
        # near distractor within [MIN_CLEARANCE_M, 0.12] of its target; far >= 0.15 from both;
        # all pairwise centre distances >= MIN_CLEARANCE_M; distractors inside the band

    def test_near_profiles_are_split_between_pick_and_place(self):
        near = Counter(e["profile"][1] for e in self.train if e["profile"][1].startswith("near"))
        self.assertEqual(near, {"near_pick": 100, "near_place": 100})

    def test_reserved_layouts(self):
        hold = self.plan["evaluation_holdout"]
        self.assertEqual((len(self.plan["evaluation_screen_ids"]), len(self.plan["evaluation_report_ids"])), (20, 40))
        report = [hold[i] for i in self.plan["evaluation_report_ids"]]
        self.assertEqual(sum((h["pick"], h["place"]) in HELD_OUT_PAIRS for h in report), 20)
        self.assertEqual(Counter(h["edge"] != "interior" for h in report)[True], 10)
        used = {tuple(e["cube_xy_m"][e["pick"]]) for e in self.plan["episodes"]}
        self.assertFalse(used & {tuple(h["cube_xy_m"][h["pick"]]) for h in hold.values()})

    def test_deterministic_and_canonical(self):
        self.assertEqual(make_plan(), make_plan()); validate_plan(make_plan())
        bad = make_plan(); bad["episodes"][0]["seed"] += 1
        with self.assertRaises(ValueError): validate_plan(bad)

    def test_riskiest_profiles_come_first(self): ...   # the four profiles named above, in slots 0-3
    def test_attempt_draw(self): ...                   # k=0 equals the planned draw; k=1,2 differ, are stable, jitter within +/-1.0
    def test_resample_layout(self): ...                # r=0 identity; r=1..3 keep cell/profile/edge/colours/instruction, move the cubes, satisfy test_geometry, are stable
    def test_no_simulator_imports(self):               # ast-parse plan.py: no import of robosuite, mujoco, torch, lerobot
```

- [ ] **Step 2: Run** — `run "$OLD" -m unittest experiments.b3_robosuite_data.tests.test_plan -v` → FAIL (module missing).
- [ ] **Step 3: Implement `plan.py`.** Follow `experiments/b2_distractor_data/plan.py` for the seed, deck and hypercube idioms (read it; do not import it). `validate_plan` compares against a fresh `make_plan()`.
- [ ] **Step 4: Run** — PASS, and the whole module runs in under 10 s.
- [ ] **Step 5: Commit** — `git commit -m "Add the B3 collection plan"`.

---

### Task 3: scenes — models, reset, reserved bank

**Files:**
- Create: `experiments/b3_robosuite_data/scenes.py`, `tests/test_scenes.py`

**Interfaces:**
- Consumes: `build_sim(colours)`, plan episodes and reserved layouts.
- Produces: `build_model(colours)`, `reset(model, cube_xy_m, home_pose_jitter_deg)`, `validate_geometry(model, data, cube_xy_m)`, `make_bank(plan)`.

- [ ] **Step 1: Write the failing tests** (`test_scenes.py`, robosuite image): for the first planned episode with two distractors —
  - `build_model(colours)` returns the same object for the same colours in any order, and its cube bodies are exactly the episode's `colours`;
  - `reset` twice gives array-equal `qpos` and `initial_state`; every cube settles within 1e-3 m of its planned xy and within 1e-3 of z = 0.02 + 0.001; arm joints equal `Q_HOME[:7] + deg2rad(jitter)` within 5e-3 rad after settling (report the measured maximum); fingers within 1e-3 of 0.04; `ctrl[:7]` equals the jittered home and `ctrl[-1] == 255`;
  - `validate_geometry` raises `ValueError` when a cube is moved 5 mm after reset;
  - `make_bank(plan)` returns one entry per reserved (layout, condition) — 100 — each with `qpos`, `initial_state`, and a manifest hash; calling it twice gives equal hashes.
- [ ] **Step 2: Run in `$NEW`** — FAIL.
- [ ] **Step 3: Implement.** `build_model` wraps `build_sim` with an `functools.lru_cache` on the sorted colour tuple and keeps the `env` alive beside the model (robosuite frees its sim otherwise). `reset` follows `robosuite_scene.scene.overwrite_reset` with planned xy (z = 0.02), identity quaternions, the jittered arm, a 0.2 s `mj_step` settle and `synchronize_observation`; no RNG. Bank keys are `f"{layout_id}:{condition}"`.
- [ ] **Step 4: Run** — PASS. Then the two-process check: run a one-line script twice that prints `sha256` of `make_bank(make_plan())[1]` serialized with sorted keys; the two lines must match.
- [ ] **Step 5: Commit** — `git commit -m "Reset and bank B3 layouts in the robosuite scene"`.

---

### Task 4: the expert

**Files:**
- Create: `experiments/b3_robosuite_data/expert.py`, `tests/test_expert_fork.py`, `tests/test_expert.py`

**Interfaces:**
- Consumes: `build_model`, `reset`, `robosuite_scene.contract.GRASP_Z_OFFSET_M`.
- Produces: `B3Expert(colours)` and `run_attempt(episode, draw, sink) -> dict` with keys `success`, `reason`, `released_stack`, `final_xy_m`, `final_z_m`, `stable_ticks`, `frames`, `model`, `reset` (the dict from `scenes.reset`), `final_state`, `final_qpos`.

- [ ] **Step 1: Fork test** (admission image, source text only): the AST of `B3Expert.run` equals the AST of `experiments/b2_distractor_data/expert.py::Expert.run` after replacing the single `Name("GRASP_Z_OFFSET_M")` in the fork with `Constant(0.105)`; exactly one such name; and the fork imports `_apply_hover_offset`, `_validate_home_jitter`, `_validate_hover_offset` from `experiments.b2_distractor_data.expert` rather than redefining them.
- [ ] **Step 2: Episode test** (robosuite image, about 4 minutes): `run_attempt` on the first four planned episodes (the riskiest profiles) with `draw = attempt_draw(episode, 0)`; assert each returns a result with `frames == 528`; assert at least the `none` control succeeds with `final_xy_m < 0.02` and `0.035 < final_z_m < 0.045`. The three distractor profiles may legitimately fail at `MIN_CLEARANCE_M = 0.08`: the test prints their outcome and asserts only that a failure carries a `reason`. Task 7 decides the clearance from real rejection rates.
- [ ] **Step 3: Run both** — FAIL.
- [ ] **Step 4: Implement.** `B3Expert(colours)` mirrors `robosuite_scene.expert.RobosuiteExpert.__init__` but takes its model from `build_model(colours)`; `expert_version = "b3_robosuite_expert_v1"`. `run` is the verbatim copy described in Step 1. `run_attempt` builds the episode dict the expert needs (the planned episode with `seed` and `home_pose_jitter_deg` replaced by the draw), resets with the draw's jitter, runs, and returns the keys above. One `B3Expert` per colour set is cached.
- [ ] **Step 5: Run** — fork test PASS in `$OLD`; episode test PASS in `$NEW`. Record each of the four outcomes in the task report.
- [ ] **Step 6: Commit** — `git commit -m "Run the scripted expert on B3 layouts"`.

---

### Task 5: admission and the seal

**Files:**
- Create: `experiments/b3_robosuite_data/validate.py`, `contract.py`, `tests/test_validate.py`, `tests/test_contract.py`

**Interfaces:**
- Produces: `admit(episode, result, frames, traces) -> {"admitted", "gates", "reason"}`; `distractor_disturbance(model, initial_qpos, final_qpos, distractors) -> dict[str, float]`; `fingerprint(root)`; `validate_dataset(root, plan, journal) -> dict`; `seal(root, plan, provenance) -> dict`; `contract.provenance() -> dict[str, str]`, `contract.IMAGE`, `contract.OUTPUT`, `contract.SPEC`, `contract.PLAN`.

**Gates in `admit`, in this order, all recorded even after the first failure:**

| gate | passes when |
|---|---|
| `expert_success` | `result["success"]` and `stable_ticks ≥ 30` |
| `frames` | `len(frames) == 528 == result["frames"]` |
| `stack` | `final_xy_m < 0.02` and `0.035 < final_z_m < 0.045` (read from the result; the window itself lives in shared code) |
| `distractors_undisturbed` | every distractor moved ≤ 0.02 m in xy between reset and final state |
| `replay` | `robosuite_scene.validate.replay_recorded(result, actions, traces, pick, place)["max_policy_state_error"] < 1e-3` |
| `labels` | every frame's `task` equals the planned instruction; state is float32 `(9,)`, action float32 `(8,)` |

`reason` is the first failing gate's name, or `"admitted"`.

- [ ] **Step 1: Write the failing tests** (admission image; `admit` is tested with synthetic results and traces, with `replay_recorded` injected as a parameter so no simulator is needed): each gate fails alone and names itself; all pass ⇒ admitted; a distractor moved 2.5 cm rejects, the pick cube moving does not; `fingerprint` ignores `seal.json`, changes with any other file, rejects symlinks, **and is unchanged when a `.json` file's keys are reordered** but changes when a JSON value changes; `provenance()` lists this package's `.py` files and README, `experiments/robosuite_scene/*.py`, the spec and this plan, and every value is a 64-hex digest; `validate_dataset` takes the substitutions into account (episode `i` must match `resample_layout(plan episode i, r)`) and rejects a wrong episode count, a wrong cell balance, and an instruction that disagrees with its plan slot (use a tiny fake dataset directory and a stub reader).
- [ ] **Step 2: Run in `$OLD`** — FAIL.
- [ ] **Step 3: Implement.** `fingerprint`: sorted `{relative path: digest}` over every file under root except `root/seal.json`; a `.json` file's digest is the sha256 of `json.dumps(json.load(f), sort_keys=True, separators=(",", ":"))`, any other file's is the sha256 of its bytes. `seal` writes `seal.json` = `{"fingerprint", "provenance_sha256", "plan_sha256", "substitutions_sha256", "image", "min_clearance_m", "episodes"}` last. `validate.py` imports robosuite-dependent modules only inside the functions that need them.
- [ ] **Step 4: Run** — PASS in `$OLD`.
- [ ] **Step 5: Commit** — `git commit -m "Admit, validate and seal B3 episodes"`.

---

### Task 6: the collector

**Files:**
- Create: `experiments/b3_robosuite_data/record.py`, `tests/test_record.py`, `README.md`

**Interfaces:**
- Produces: `run_schedule(plan, execute, journal_path, commit, resume=False) -> dict` and the CLI `python -m experiments.b3_robosuite_data.record --pilot | --collect | --resume | --status` with `--out DIR`.
- `execute(episode, draw) -> (result, frames, traces, verdict)` and `commit(episode, frames) -> None` are injected, so the scheduler is tested without a simulator.

**Scheduler rules:**

- Slots are the plan's episodes in order. A slot tries layout `resample_layout(episode, r)` for `r = 0, 1, 2, 3` (`r = 0` is the planned layout), and on each layout up to `ATTEMPTS_PER_LAYOUT = 3` attempts with `attempt_draw(episode, k)`. So one bad layout is substituted automatically instead of stopping the run. `REJECTION_BUDGET = 150` rejected attempts overall (the pilot accepts up to 20% rejection on crowded layouts, which is up to about 45 on its own).
- An admitted attempt: `commit` (which calls `add_frame` for each buffered frame, then `save_episode`), write `episodes/episode_<i>.npz` (qpos, ctrl, initial_state, final_qpos), append a journal line `{"slot", "attempt", "admitted": true, "gates", "seed", "wall_seconds"}`, move to the next slot.
- A rejected attempt: append a journal line with `admitted: false`, `r`, `k` and the reason; nothing is written to the dataset; retry the same slot. Every admitted journal line also carries `r`. When the layout index moves past 0 the collector rewrites `substitutions.json` = `{episode_index: r}` for every admitted slot with `r > 0`. A slot that fails all 4 layouts × 3 attempts stops the run with `SlotExhausted`; the 151st rejection overall stops it with `BudgetExhausted`; journal intact in both cases.
- `--resume`: read the journal; admitted slots are skipped; the first non-admitted slot continues at its next attempt index. The dataset is reopened for appending with the pinned encoder settings (see `open_dataset` in `experiments/b2_distractor_data/record.py` for the attributes LeRobot drops on reopen; reproduce the pattern, do not import it). Refuse to resume if the dataset's episode count differs from the journal's admitted count, or if `plan.json` or `provenance.json` in the output differ from fresh values. Substitutions never change `plan.json`: the plan of record is `plan.json` plus `substitutions.json`.
- `--collect` refuses an existing output directory; it first writes `plan.json`, `provenance.json`, `bank.npz` + `bank.json` (from `make_bank`), then runs. After the last slot: `validate_dataset`, `validation.json`, `seal`.
- At start and every 25 admits: require ≥ 20 GB free under the output root, and re-hash `provenance()`; drift aborts.
- `--pilot` (Task 7) takes `--slots I J K ...` and `--attempts N` (default 1), runs those slots into a scratch output and never seals.
- `--status` prints admitted / rejected counts, the current slot and the last five journal lines.
- Log one line per attempt starting with `B3 ` so `docker logs | grep '^B3 '` shows progress.

- [ ] **Step 1: Write the failing tests** (admission image, fake `execute` / `commit`): all slots admitted in order; an injected rejection retries the same slot with attempt 1 and its different draw; twelve rejections on one slot raise `SlotExhausted` and leave a readable journal; the 151st rejection raises `BudgetExhausted`; a run interrupted after slot 7 and resumed admits every slot exactly once and calls `commit` exactly once per slot; resume refuses a dataset/journal count mismatch; a rejected attempt never calls `commit`; a slot whose planned layout is rejected three times is admitted on layout `r = 1`, the journal and `substitutions.json` both say so, and resume after that point keeps `r`; a slot failing all 12 attempts raises `SlotExhausted`.
- [ ] **Step 2: Run in `$OLD`** — FAIL.
- [ ] **Step 3: Implement** `record.py` and `README.md` (what the package is, images, gate order, the commands, "hash-bound once sealed: fork, do not edit").
- [ ] **Step 4: Run** — `test_record` PASS in `$OLD`. Then a real 3-slot end-to-end check in `$NEW`: `--pilot --slots 0 3 500 --out /out/e2e_check` must leave a dataset with one episode per admitted slot that loads with `LeRobotDataset` and whose videos pass `media.decode_video`.
- [ ] **Step 5: Commit** — `git commit -m "Schedule, journal and resume B3 collection"`.

---

### Task 7: pilot — freeze the clearance, look at the data

Operational, no new code unless the pilot exposes a defect.

- [ ] **Step 1: Clearance ladder.** For `MIN_CLEARANCE_M` = 0.08: run `--pilot` on the first 20 training slots whose profile is `near_pick` or `near_place` (10 of each), single attempt each. Record admitted / rejected and each rejection's gate. If more than 4 of 20 are rejected, set the constant to 0.09 (the plan changes, so the slot list is recomputed) and repeat; then 0.10. If 0.10 still rejects more than 4 of 20, stop and report to the user with the rejection gates — do not go higher or loosen a gate.
- [ ] **Step 2: Corners.** `--pilot` on the first 5 `inner` and 5 `outer` edge slots. Any corner with more than 2 of 5 rejected is reported to the user before collection.
- [ ] **Step 3: Look.** Extract a first, middle and last frame of both cameras from three pilot episodes (a control, a two-distractor near-pick, an outer-corner) and view them. Check: the right cubes are present and coloured, distractors sit where planned, nothing is clipped by the camera, the stack is on the instructed place cube. Describe what is seen in the task report.
- [ ] **Step 4: Freeze.** Commit the final `MIN_CLEARANCE_M` (and `test_plan` still passing) as `git commit -m "Freeze B3 distractor clearance at <value> m after the pilot"`. Write the pilot numbers into `README.md`. From this commit on the package is frozen until the seal.
- [ ] **Step 5: Go / no-go, decided by the executor, not by a human** (see *Unattended execution*). **Go** only if all hold: a clearance on the ladder met ≤ 4 of 20; each corner had ≤ 2 of 5 rejected; the three inspected episodes show the instructed pick cube on the instructed place cube with every planned cube visible and correctly coloured; every test is green; the end-to-end check from Task 6 loaded and decoded. Anything else is **no-go**: do not launch, write the numbers and the reason into `STATUS.md`, stop.

---

### Task 8: collect, seal, report

- [ ] **Step 1: Pre-launch.** Work through section 1 of `.github/skills/launching-a-run/SKILL.md`. In particular: worktree at the frozen commit, `df -h` shows ≥ 30 GB free, all tests green (`$OLD`: `test_plan test_contract test_expert_fork test_record test_validate`; `$NEW`: `test_scenes test_expert` and the robosuite_scene suite), nothing in `provenance()` will be edited during the run.
- [ ] **Step 2: Launch** `docker run -d --name b3-collect-<YYYYMMDD> ... -m experiments.b3_robosuite_data.record --collect --out /out/b3_robosuite_<YYYYMMDD>`. Check `--status` after 5, 15 and 60 minutes, then at least hourly. Expected rate: about 72 s per admitted episode, about 11–12 h in total.
- [ ] **Step 3: If it stops.** A crash or a killed container with an intact journal: `--resume`, once; if it stops again at the same slot, do not loop. `SlotExhausted`, `BudgetExhausted` or a provenance drift: do not resume and do not restart from scratch; write the journal's tail and the rejection gates into `STATUS.md` and leave it for the user.
- [ ] **Step 4: Verify the seal.** `seal.json` present; recompute `fingerprint` in a second process and compare; load the dataset with `LeRobotDataset` and read first and last frame; `validation.json` shows 560 episodes, 25 + 3 per cell, 100 / 200 / 200 by distractor count in training, 400 / 50 / 50 by edge class, and no held-out pair.
- [ ] **Step 5: Report** `docs/B3_ROBOSUITE_COLLECTION_<YYYYMMDD>.md`: what was collected (the table of counts as measured), the frozen clearance and pilot numbers, rejections by gate and by profile, any substituted layout, wall-clock, the seal fingerprint and image digest, the per-episode placement offsets (min / median / max `final_xy_m`), and anything odd seen in the frames. Caveats beside the figures.
- [ ] **Step 6: Documentation pass.** `TODO.md` (B3 collection done, next: training plan), `docs/BENCHMARK_STATUS.md`, `docs/OPERATIONS.md` (output census row), `docs/INDEX.md`, `docs/FAILURES.md` for anything that went wrong and could recur, `CLAUDE.md` and `.github/instructions/experiments.instructions.md` (the new package is hash-bound: fork, do not edit), the `working-on-the-robosuite-path` skill if a new trap was hit.
- [ ] **Step 7: Commit** — `git commit -m "Seal the B3 robosuite dataset and report the collection"`; `git status --short` shows nothing under `outputs/`.

---

## Self-review

- Spec coverage: section 1 (scene, colours, layout, edge, distractors, admission, reserved layouts) → Tasks 1–5; section 2 (package layout, journal, resume, hash binding, reproducible fingerprint, gate order) → Tasks 5–8; the pilot ladder → Task 7. Sections 3 and 4 are deliberately not in this plan.
- Review Focus items map to tests: (1) (2) `test_plan`; (3) (4) `test_record`; (5) `test_validate`.
- Names used across tasks match the shared-names block; `replay_recorded`'s new keyword arguments are produced in Task 1 and consumed in Task 5.
- Plan-mandated duplication a reviewer will see: `B3Expert.run` is a verbatim copy (the repo's expert-versioning pattern, pinned by the AST test), and the dataset-reopen attributes are reproduced from B2's collector (that module cannot be imported without its MuJoCo-scene dependencies).
- Checked before hand-over (2026-10-03): the Phase 1 expert stacks red on green from both edge corners on both sides, 14 of 14 (`outputs/robosuite_scene/spike_b3_corners/result.json`; no distractors, two cubes). So a corner failure in the pilot points at the new code, not at reach.
- Not known when this plan was written, and found out by Tasks 4 and 7: whether the expert clears a distractor at 8 cm, and the real rejection rate.
