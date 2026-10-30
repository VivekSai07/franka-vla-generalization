// Verifies .ua/knowledge-graph.json with the plugin's schema, plus dangling edges, layer coverage and excluded-path leaks.
// Usage: node scripts/code_graph_verify.mjs <repo root>   (needs the plugin installed; see skill using-the-code-graph)
import fs from 'fs';
import os from 'os';

const { validateGraph } = await import(`${os.homedir()}/.understand-anything-plugin/packages/core/dist/schema.js`);

const root = process.argv[2] ?? '.';
const g = JSON.parse(fs.readFileSync(root + '/.ua/knowledge-graph.json', 'utf8'));
const v = validateGraph(g);
const ids = new Set(g.nodes.map((n) => n.id));
const dangling = g.edges.filter((e) => !ids.has(e.source) || !ids.has(e.target)).length;
const fileTypes = new Set(['file', 'config', 'document', 'service', 'pipeline', 'table', 'schema', 'resource', 'endpoint']);
const layered = new Set(g.layers.flatMap((l) => l.nodeIds));
const unlayered = g.nodes.filter((n) => fileTypes.has(n.type) && !layered.has(n.id)).length;
const leaked = g.nodes.filter((n) => /^(outputs|datasets|huggingface|publication|\.worktrees)\//.test(n.filePath || '')).length;
const tourBad = g.tour.reduce((a, s) => a + s.nodeIds.filter((id) => !ids.has(id)).length, 0);
const out = {
  schemaOk: v.success, nodes: g.nodes.length, edges: g.edges.length, layers: g.layers.length, tour: g.tour.length,
  danglingEdges: dangling, unlayered, leaked, tourDanglingIds: tourBad, commit: g.project.gitCommitHash.slice(0, 8),
};
console.log(JSON.stringify(out));
console.log(g.tour.map((s) => `${s.order}.${s.title}`).join(' | '));
process.exit(out.schemaOk && !dangling && !unlayered && !leaked && !tourBad ? 0 : 1);
