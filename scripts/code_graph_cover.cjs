// Coverage check for an incremental refresh, run after merge-batch-graphs.py and before finalize-incremental.mjs:
// every file in the plan has a node, nothing dangles, and nodes of unchanged files keep identical summaries.
// Usage: node scripts/code_graph_cover.cjs <repo root>   (reads .ua/intermediate/ and the committed .ua/knowledge-graph.json)
const fs = require('fs');
const root = process.argv[2];
const ua = root + '/.ua';
const rd = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
const plan = rd(ua + '/intermediate/incremental-plan.json');
const asm = rd(ua + '/intermediate/assembled-graph.json');
const old = rd(ua + '/knowledge-graph.json');
const scan = rd(ua + '/intermediate/scan-result.json');

const byPath = new Map();
for (const n of asm.nodes) if (n.filePath) byPath.set(n.filePath, (byPath.get(n.filePath) || 0) + 1);

const missing = plan.filesToReanalyze.filter((p) => !byPath.has(p));
console.log('plan files', plan.filesToReanalyze.length, '| without any node:', missing.length, missing.slice(0, 5).join(', '));

const scanPaths = new Set(scan.files.map((f) => f.path));
const noNode = [...scanPaths].filter((p) => !byPath.has(p));
console.log('scanned files without node:', noNode.length, noNode.slice(0, 5).join(', '));

const ids = new Set(asm.nodes.map((n) => n.id));
console.log('dangling edges:', asm.edges.filter((e) => !ids.has(e.source) || !ids.has(e.target)).length);
const dup = asm.nodes.length - ids.size;
console.log('duplicate node ids:', dup);

// Unchanged files must keep exactly their old nodes (summaries untouched).
const changed = new Set(plan.filesToReanalyze);
const oldById = new Map(old.nodes.map((n) => [n.id, n]));
let kept = 0;
let drifted = 0;
for (const n of asm.nodes) {
  if (changed.has(n.filePath)) continue;
  const o = oldById.get(n.id);
  if (!o) continue;
  kept++;
  if (o.summary !== n.summary) drifted++;
}
const oldUnchanged = old.nodes.filter((n) => !changed.has(n.filePath)).length;
console.log('old nodes in unchanged files:', oldUnchanged, '| still present:', kept, '| summary drifted:', drifted);
console.log('totals: nodes', old.nodes.length, '->', asm.nodes.length, ' edges', old.edges.length, '->', asm.edges.length);
