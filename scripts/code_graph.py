#!/usr/bin/env python3
"""Cheap queries over .ua/knowledge-graph.json so agents never load the whole 2 MB file.

  code_graph.py status             graph commit and how far main has moved since
  code_graph.py layers             layer names, sizes, descriptions
  code_graph.py show PATH          summary, tags, layer, edges for one file
  code_graph.py importers PATH     files that import PATH, and its tests
  code_graph.py find TEXT          nodes whose name or summary mention TEXT
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = ROOT / ".ua"
FILE_TYPES = {"file", "config", "document", "service", "pipeline", "table", "schema", "resource", "endpoint"}
ANALYSED = ["experiments", "src", "data_collection", "validation", "training", "inference", "scripts", "tests", "docs", ".github", "*.md"]
STALE_AT = 10
PLUGIN_FULL_AT = 30  # the plugin's own hardcoded limit for an incremental refresh


def load():
    g = json.loads((UA / "knowledge-graph.json").read_text())
    layer_of = {i: l["name"] for l in g["layers"] for i in l["nodeIds"]}
    return g, {n["id"]: n for n in g["nodes"]}, layer_of


def file_node(nodes, path):
    hits = [n for n in nodes.values() if n.get("filePath") == path and n["type"] in FILE_TYPES]
    if not hits:
        sys.exit(f"no file node for {path!r} (graph may be stale or the path is excluded by .ua/.understandignore)")
    return hits[0]


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    cmd, arg = argv[1], " ".join(argv[2:])
    if cmd == "status":
        meta = json.loads((UA / "meta.json").read_text())
        h = meta["gitCommitHash"]
        changed = subprocess.run(
            ["git", "-C", str(ROOT), "diff", "--name-only", h, "HEAD", "--", *ANALYSED],
            capture_output=True, text=True).stdout.split()
        n = len(changed)
        print(f"graph built at {h[:8]} ({meta['lastAnalyzedAt']}), {meta['analyzedFiles']} files; {n} analysed files changed since")
        # Judgment call, not a measured limit: a few files behind is normal after merging a refresh itself.
        print(f"STALE ({n} files behind): refresh per skill using-the-code-graph, incremental cost grows with this count" if n >= STALE_AT else "up to date enough")
        if n > PLUGIN_FULL_AT:
            print(f"past the plugin's {PLUGIN_FULL_AT}-file limit: it will demand a full rebuild; see the skill for the cheaper override")
        return
    g, nodes, layer_of = load()
    if cmd == "layers":
        for l in g["layers"]:
            print(f"{l['name']} ({len(l['nodeIds'])}): {l['description']}")
    elif cmd == "show":
        n = file_node(nodes, arg)
        print(f"{n['id']}  [{layer_of.get(n['id'], '-')}]\n{n['summary']}\ntags: {', '.join(n['tags'])}")
        for e in g["edges"]:
            if e["type"] in ("contains", "exports"):
                continue
            if e["source"] == n["id"]:
                print(f"  -> {e['type']} {e['target']}")
            elif e["target"] == n["id"]:
                print(f"  <- {e['type']} {e['source']}")
    elif cmd == "importers":
        n = file_node(nodes, arg)
        print("# lower bound only: measured median import recall 63% (0-100%) vs grep; confirm with git grep")
        for e in g["edges"]:
            if e["target"] == n["id"] and e["type"] == "imports":
                print(f"imported by  {e['source']}")
            elif e["source"] == n["id"] and e["type"] == "tested_by":
                print(f"tested by    {e['target']}")
    elif cmd == "find":
        q = arg.lower()
        hits = [n for n in nodes.values() if q in n["name"].lower() or q in n["summary"].lower()]
        for n in hits[:25]:
            print(f"{n['id']}: {n['summary'][:110]}")
        if len(hits) > 25:
            print(f"... {len(hits) - 25} more, narrow the query")
    else:
        sys.exit(__doc__)


main(sys.argv)
