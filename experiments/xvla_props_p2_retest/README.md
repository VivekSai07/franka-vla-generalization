# Props P2 fresh-layout re-test

Not a dataset, training or sealed package: no contract, no seal, nothing here is hashed by another package. It re-evaluates
checkpoint 101,520 from the archive bundle on 240 fresh paired layouts (milk and bread, each under its trained and its held-out
bin), to measure the pooled unseen-pair penalty. Design and the pre-registered reading:
`docs/superpowers/specs/2026-10-19-props-p2-retest-design.md`; plan: `docs/superpowers/plans/2026-10-19-props-p2-retest.md`.

| file | what |
|---|---|
| `plan.py` | the 480 entries (240 layouts × 2 arms) in the sealed P2 plan's format, drawn with P2's own samplers under `retest:` identities; pure |
| `analysis.py` | pairs, pooled penalty (bootstrap over pairs, stratified by object), the reading, tables; pure |
| `retest.py` | driver: `plan`, `feasibility`, `merge`, `prepare`, `worker`, `collect`, `analyze`; points the unchanged P2 evaluator (`experiments/xvla_props_p2_eval/`) at the re-test plan through `bind`, which restores every global it rebinds |
| `tests/` | admission image; `test_plan` needs the sealed P2 `plan.json` (mount `outputs/props_p2_data` or the bundle's `dataset_provenance` at `/bundle_prov`) and fails without it |

Run in the robosuite image (feasibility, workers) with the bundle mounted at `/bundle`, repo at `/vla`, OSMesa. Never the EGL image:
the checkpoint was trained on OSMesa pixels.
