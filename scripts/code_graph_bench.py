#!/usr/bin/env python3
"""Reproducible proxy for what the code graph saves an agent. Read-only, stdlib only.

Counts what lands in an agent's context (tokens ~ bytes/4) for the same navigation question answered
with scripts/code_graph.py versus plain git/grep/head, and checks the graph's answers against grep.
It does not measure real agent sessions; those expose no per-session token counts.
"""
import json
import random
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOK = lambda s: len(s.encode()) / 4


def sh(cmd):
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, executable="/bin/bash").stdout


def helper(*args):
    return sh("python3 scripts/code_graph.py " + " ".join(args))


py_files = [f for f in sh("git ls-files '*.py'").splitlines() if not f.startswith(".")]

print("== 1. Orientation: what areas exist and what are they for")
base = sh("ls experiments; for d in experiments/*/; do head -n 6 \"$d/README.md\" 2>/dev/null; done")
g = helper("layers")
print(f"   grep/head baseline {TOK(base):7.0f} tok   graph 'layers' {TOK(g):7.0f} tok   ratio {TOK(base) / TOK(g):.1f}x")

print("== 2. Importers of a module (accuracy against grep). Targets: non-test modules with a repo-unique stem, so grep is unambiguous")
stems = [Path(f).stem for f in py_files]
cands = [f for f in py_files if stems.count(Path(f).stem) == 1 and "/tests/" not in f and Path(f).stem not in ("__init__", "main")]
src = {f: (ROOT / f).read_text(errors="ignore").splitlines() for f in py_files}


def grep_truth(target):
    pat = re.compile(rf"^\s*(from|import)\s+.*\b{Path(target).stem}\b")
    return {f for f, lines in src.items() if f != target and any(pat.search(l) for l in lines)}


truths = {f: grep_truth(f) for f in cands}
random.seed(1)
used = [f for f in cands if truths[f]]  # modules something imports
picked = sorted(used, key=lambda f: -len(truths[f]))[:3] + random.sample(used, 12)
precs, recs = [], []
for target in dict.fromkeys(picked):
    truth = truths[target]
    out = helper("importers", target)
    got = {l.split("file:")[1] for l in out.splitlines() if l.startswith("imported by")}
    if got:
        precs.append(len(got & truth) / len(got))
    recs.append(len(got & truth) / len(truth))
    stem = Path(target).stem
    base_out = sh(f"git grep -lE '^\\s*(from|import)\\s+.*\\b{stem}\\b' -- '*.py'")
    if target in picked[:3]:
        print(f"   top-3 by importers  {target}: grep {len(truth)}, graph {len(got)}, recall {len(got & truth) / len(truth):.0%}; "
              f"tokens grep {TOK(base_out):.0f} vs graph {TOK(out):.0f}")
print(f"   over {len(recs)} modules: median recall {statistics.median(recs):.0%} (min {min(recs):.0%}, max {max(recs):.0%}); "
      f"median precision {statistics.median(precs):.0%} over {len(precs)} with any graph hit; "
      f"modules where graph found none: {sum(1 for r in recs if r == 0)}")

print("== 3. Triage: summary of a file versus opening it (60 sampled experiment .py files, seed 1)")
random.seed(1)
sample = random.sample([f for f in py_files if f.startswith("experiments/") and "/tests/" not in f], 60)
file_t, show_t, covered = [], [], 0
for f in sample:
    s = helper("show", f)
    if not s.startswith("file:"):
        continue
    covered += 1
    file_t.append(TOK((ROOT / f).read_text(errors="ignore")))
    show_t.append(TOK(s))
print(f"   covered {covered}/60 files; median file {statistics.median(file_t):.0f} tok, median 'show' {statistics.median(show_t):.0f} tok; "
      f"saved per file not opened: median {statistics.median([a - b for a, b in zip(file_t, show_t)]):.0f} tok")

print("== 4. Concept search: which files mention a topic")
for q in ["disk guard", "checkpoint schedule", "seal", "EGL"]:
    base_out = sh(f"git grep -il '{q}' -- experiments src data_collection validation docs scripts")
    out = helper("find", q)
    print(f"   '{q}': grep lists {len(base_out.splitlines())} files ({TOK(base_out):.0f} tok), graph 'find' {len(out.splitlines())} lines ({TOK(out):.0f} tok)")

print("== 5. Fixed cost the integration adds to every session")
diff = sh("git diff f5754cd 7b1e410 -- .github/copilot-instructions.md AGENTS.md | grep '^+[^+]'")
print(f"   always-loaded lines added: {TOK(diff):.0f} tok; skill loaded only on demand: {TOK((ROOT / '.github/skills/using-the-code-graph/SKILL.md').read_text()):.0f} tok")
print(f"   whole graph if (wrongly) read in full: {TOK((ROOT / '.ua/knowledge-graph.json').read_text()):.0f} tok")
