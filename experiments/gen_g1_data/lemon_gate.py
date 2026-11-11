"""The lemon-only expert gate (spec note 2026-11-10): 96 fresh lemon attempts, pass at >= 90% first-attempt admission.

The attempts are drawn with the plan's own scene sampler under a separate namespace, so the plan and its digest are untouched.
Run through `record --lemon-gate`; the first-attempt admission is the gate's (expert success, undisturbed distractors, exact replay,
labels, schema, signals).
"""
import json
from collections import Counter
from pathlib import Path

from experiments.gen_g1_data import plan as p

NAMESPACE = f"{p.NAMESPACE}:lemongate"
ATTEMPTS = 96
BAR = 0.90
LEMON_CELLS = tuple(c for c in p.TRAINED_CELLS if c[0] == "lemon")
SHA256 = "5186ea7a92574fca84799e98fe4002cbb3a3d2442511fac6b48c451864a44c7d"   # pinned 2026-11-10 before any attempt ran; test_lemon_gate
REPORTED = ("distractor_count", "quintile", "bin", "region")


def _strata(seeds):
    cells = p._shuffled([LEMON_CELLS[i % len(LEMON_CELLS)] for i in range(ATTEMPTS)], "lemongate:cells", seeds)
    counts = p._shuffled([n for n in p.COUNT_WEIGHTS for _ in range(ATTEMPTS // len(p.COUNT_WEIGHTS))], "lemongate:counts", seeds)
    quintiles = p._shuffled([q for q in range(p.QUINTILES) for _ in range(ATTEMPTS // p.QUINTILES)]
                            + [0] * (ATTEMPTS % p.QUINTILES), "lemongate:quintiles", seeds)
    outer = round(ATTEMPTS * p.OUTER_SHARE)
    regions = p._shuffled(["outer"] * outer + ["p2"] * (ATTEMPTS - outer), "lemongate:regions", seeds)
    return [dict(cell=cell, distractor_count=n, quintile=q, region=r) for cell, n, q, r in zip(cells, counts, quintiles, regions)]


def make_attempts(manifest=None):
    seeds = {}
    attempts = []
    for index, stratum in enumerate(_strata(seeds)):
        identity = f"{NAMESPACE}:{index}"
        attempts.append(dict(episode_index=index, split="gate", layout_id=f"lemongate_{index:03d}", identity=identity,
                             **p._episode(identity, stratum, seeds, manifest)))
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("Seed collision; stop without redraw")
    return attempts


def verdict(rows):
    """Pass iff at least BAR of the attempts were admitted at the first attempt; the strata are reported, never barred. Pure."""
    if sorted(r["index"] for r in rows) != list(range(ATTEMPTS)):
        raise ValueError(f"lemon gate incomplete or repeated: {len(rows)} rows")
    admitted = sum(r["admitted"] for r in rows)
    out = {"attempts": ATTEMPTS, "admitted": admitted, "overall": admitted / ATTEMPTS,
           "expert_success_overall": sum(r["expert_success"] for r in rows) / ATTEMPTS, "bar": BAR}
    for field in REPORTED:
        out[f"by_{field}"] = {str(v): {"n": sum(r[field] == v for r in rows), "admitted": sum(r["admitted"] for r in rows if r[field] == v)}
                              for v in sorted({r[field] for r in rows})}
    out["reasons"] = dict(Counter(r["reason"] for r in rows if not r["admitted"]))
    out["expert_reasons"] = dict(Counter(r["expert_reason"] for r in rows if not r["admitted"]))
    out["passed"] = out["overall"] >= BAR
    return out


def summary(out):
    from experiments.gen_g1_data import record
    out = Path(out)
    rows = [r for path in sorted(out.glob("lemongate_*.jsonl")) for r in record.read_journal(path)]
    result = verdict(rows)
    record._write_json(out / "lemon_gate_summary.json", result)
    print(json.dumps(result, indent=2))
