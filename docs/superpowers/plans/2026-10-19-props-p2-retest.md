# Props P2 fresh-layout re-test — Implementation Plan

> **For agentic workers:** executed inline in the session that wrote it. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run checkpoint 101,520 on 240 fresh paired layouts (480 rollouts) and report the pooled paired unseen-pair penalty with
the pre-registered reading, without touching the sealed P2 plan, the evaluator or any hash-bound file.

**Architecture:** A new, not hash-bound package `experiments/xvla_props_p2_retest/`: a pure plan generator (P2's own samplers,
imported read-only, a new identity namespace), a pure paired analysis, and a thin driver that points the unchanged P2 evaluator
(`experiments/xvla_props_p2_eval/`: rollout, scoring, classifier) at the new plan through the same module-global rebinding the
archive bundle's `bundle_evaluate.py` uses.

**Spec:** [../specs/2026-10-19-props-p2-retest-design.md](../specs/2026-10-19-props-p2-retest-design.md).

## Global Constraints

- Never edit: `experiments/props_p2_data/**`, `experiments/props_scene/**`, `experiments/robosuite_scene/**`,
  `experiments/xvla_props_p2_eval/*.py` (the evaluator; `diagnostics/` is separate), `src/**`, the P2 sealed plan, the bundle.
- Images: robosuite `sha256:54d861aa…ccf27e` for feasibility, workers; admission `sha256:08de1e09…714abc` for tests that import no
  simulator. **OSMesa only** (`MUJOCO_GL=osmesa`); not the EGL image.
- Weights: the bundle's `checkpoints/step_101520`, verified by `model.safetensors` sha256 against `checkpoint_result.json` (the
  bundle route); `verify_checkpoints` cannot pass in the repo.
- Workers pinned to cores (5 each), `CUBLAS_WORKSPACE_CONFIG=:4096:8`, sized by GPU memory (about 7.3 GB each after a switch;
  FAILURES #57). Poll with `docker ps`; do not trust `docker wait`.
- Rules fixed in the spec: 120 layouts per object, one seed, the pre-registered reading. A failed gate is investigated, not moved.
- Commits: user identity, no trailers, short; never commit `outputs/`; do not push.
- Output root: `outputs/xvla_props_p2_retest/` (plan, feasibility, prepared, `report_p2_step_101520/`, analysis).

## Review Focus

1. A re-test layout coincides with a P2 layout (any seed or pose) — test over every P2 episode and reserved layout.
2. The two arms of a pair differ in anything but the bin instruction — test geometry equality field by field.
3. Imbalance: per object, counts 1/2 and sides left/right each 60, every combination 30; arms swap bin consistently
   (milk: orange trained / blue held-out; bread: blue trained / orange held-out).
4. A pair silently loses an arm in the analysis — a missing arm must raise, not shrink n.
5. The bootstrap is not stratified by object, or resamples trials instead of pairs.
6. The driver rebinds a P2 global and forgets to restore it, or reads the sealed plan after rebinding.

## Tasks

- [ ] **1. Plan generator** — `plan.py` + `tests/test_plan.py` (Focus 1–3). Admission image; needs the sealed P2 `plan.json`
  mounted read-only for the no-overlap test (a missing file is an error, not a skip).
- [ ] **2. Paired analysis** — `analysis.py` + `tests/test_analysis.py` (Focus 4–5); hand-computed expectations; mutation checks.
- [ ] **3. Driver** — `retest.py`: `plan`, `feasibility` (+ `--merge`), `prepare`, `worker`, `collect`, `analyze`; `tests/test_retest.py`
  (global rebinding restored, no overwrite, a pair with an infeasible arm is dropped as a pair) (Focus 6).
- [ ] **4. Run the checks:** both test suites; `plan` (digest recorded); `feasibility` on 480 entries, no rendering.
- [ ] **5. Prepare and smoke:** `prepare` from the bundle; 2-trial smoke run; confirm the trial folders are laid out as P2's.
- [ ] **6. Run** the 480 rollouts with pinned workers; `collect`; retry only infrastructure deaths via resume.
- [ ] **7. Analyze** with the pre-registered reading; report, docs pass (TODO, RESULTS, BENCHMARK_STATUS, INDEX, README,
  FAILURES if a mistake recurred); commit locally.

## Running things

```bash
MAIN=/home/utxowqr/smolvla_franka
B=$MAIN/outputs/downloads_20261017/props_p2
ROBO=sha256:54d861aac79c1c8ceed09734a00bbc5d8cbf566c427e47ff23089f9bb0ccf27e
ADMIT=sha256:08de1e09200539a26f7564bbf095fb846b8104ba07a72564b8232feb58714abc
# tests, admission image
docker run --rm --runtime runc --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=4g \
  -e PYTHONPATH=/vla -v "$MAIN:/vla:ro" -v "$B/dataset_provenance:/bundle_prov:ro" -w /vla --entrypoint python $ADMIT \
  -m unittest discover -s experiments/xvla_props_p2_retest/tests -t experiments/xvla_props_p2_retest/tests
```
