"""Driver for the P2 fresh-layout re-test: points the unchanged P2 evaluator at the re-test plan and runs the stages.

    retest.py plan         --out DIR                       write plan.json (+ sha256); refuses to overwrite
    retest.py feasibility  --out DIR --shard K --shards N  expert on every entry, no rendering (CPU, robosuite image)
    retest.py merge        --out DIR                       combine the shards into feasibility/feasibility.json
    retest.py prepare      --out DIR --candidate 101520    verify the bundle checkpoint and seal, write prepared.json
    retest.py worker       --out DIR --candidate 101520 --shard K --shards N     rollouts (GPU, robosuite image)
    retest.py collect      --out DIR --candidate 101520
    retest.py analyze      --out DIR --candidate 101520    paired penalty and the pre-registered reading

The rollout loop, scene restore, scoring and failure classifier are the P2 evaluator's, untouched
(`experiments/xvla_props_p2_eval/`, identical to the archive bundle's code snapshot). Only module globals it reads are rebound,
inside `bind`, and always restored. Design: docs/superpowers/specs/2026-10-19-props-p2-retest-design.md.
"""
import argparse
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from experiments.xvla_props_p2_eval import evaluate, feasibility, protocol
from experiments.xvla_props_p2_retest import analysis, plan

MODE, POLICY = 'report', 'p2'


@contextmanager
def bind(out):
    """Rebind what the P2 evaluator reads (plan, output root, one rollout seed per layout) and restore it afterwards."""
    out = Path(out)
    saved = [(protocol, n, getattr(protocol, n)) for n in ('PLAN_JSON', 'PARENT', 'EVAL_ROOT', 'REPORT_SEEDS')] + \
            [(evaluate, n, getattr(evaluate, n)) for n in ('PREPARED', 'FEASIBILITY', 'DECISIONS')]
    try:
        protocol.PLAN_JSON, protocol.PARENT, protocol.EVAL_ROOT, protocol.REPORT_SEEDS = out/'plan.json', out, out, 1
        evaluate.PREPARED, evaluate.FEASIBILITY = out/'prepared.json', out/'feasibility'/'feasibility.json'
        evaluate.DECISIONS = out/'decisions'
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


def _write_once(path, text):
    path = Path(path)
    if path.exists():
        raise SystemExit(f'No overwrite: {path} exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_plan(out):
    value = plan.make_plan()
    digest = plan.digest(value)
    _write_once(Path(out)/'plan.json', json.dumps(value, indent=1) + '\n')
    _write_once(Path(out)/'plan.sha256', digest + '\n')
    return digest


def feasibility_shard(out, shard, shards):
    import mujoco
    if mujoco.__version__ != feasibility.MUJOCO_VERSION:
        raise SystemExit(f'MuJoCo {mujoco.__version__} is not {feasibility.MUJOCO_VERSION}; run in the robosuite image')
    target = Path(out)/'feasibility'/f'feasibility.shard_{shard}.json'
    if target.exists():
        raise SystemExit(f'No overwrite: {target} exists')
    with bind(out):
        holdout = protocol._plan()['evaluation_holdout']
    results = {}
    for key in feasibility.keys_of_shard(shard, shards, holdout):
        results[key] = feasibility.run_one(holdout[key])
        print('FEAS', key, results[key]['success'], results[key]['reason'], results[key]['seconds'], flush=True)
    _write_once(target, json.dumps(feasibility.summarize(results), indent=2) + '\n')
    print('FEAS_SHARD_DONE', shard, shards, flush=True)


def merge_feasibility(out):
    out = Path(out)
    results = {}
    for path in sorted((out/'feasibility').glob('feasibility.shard_*.json')):
        results.update(json.loads(path.read_text())['results'])
    keys = set(json.loads((out/'plan.json').read_text())['evaluation_holdout'])
    if set(results) != keys:
        raise SystemExit(f'Shards cover {len(results)} layouts, the plan has {len(keys)}; every shard must finish, and '
                         f'only plan layouts may appear')
    summary = feasibility.summarize(results)
    _write_once(out/'feasibility'/'feasibility.json', json.dumps(summary, indent=2) + '\n')
    print('FEAS_DONE', summary['succeeded'], 'of', summary['keys'], 'failed', summary['failed_layouts'], flush=True)
    return summary


def load_feasibility(out):
    """The merged feasibility summary, checked against the re-test plan's own layouts (the evaluator's loader insists on P2's 120)."""
    path = Path(out)/'feasibility'/'feasibility.json'
    if not path.is_file():
        raise RuntimeError(f'Run the feasibility pass first: {path} is missing')
    summary = json.loads(path.read_text())
    keys = set(json.loads((Path(out)/'plan.json').read_text())['evaluation_holdout'])
    if set(summary['results']) != keys:
        raise ValueError('Feasibility pass does not cover exactly the re-test plan layouts')
    return summary


def _bundle_module():
    bundle = Path(os.environ.get('BUNDLE', '/bundle'))
    spec = importlib.util.spec_from_file_location('bundle_evaluate', bundle/'reevaluate'/'bundle_evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(out, candidate):
    """Verify the bundle's seal and the checkpoint exactly as the bundle's own evaluator does, then write prepared.json."""
    from experiments.props_p2_data.plan import digest as p2_digest
    b = _bundle_module()
    seal = json.loads((b.BUNDLE/'dataset_provenance/seal.json').read_text())
    if p2_digest(json.loads((b.BUNDLE/'dataset_provenance/plan.json').read_text())) != seal['plan_sha256']:
        raise ValueError('bundle plan.json does not match the seal')
    fingerprint = b.bundle_fingerprint()
    if fingerprint != seal['fingerprint']:
        raise ValueError('Bundle dataset fingerprint differs from the seal')
    facts = b.verify_checkpoint(candidate)
    with bind(out):
        feas = load_feasibility(out)
        evaluate.write_json(evaluate.PREPARED, dict(
            excluded_layouts=sorted(feas['failed_layouts']), feasibility_succeeded=feas['succeeded'],
            p2=dict(seal=seal['fingerprint'], parent_fingerprint=fingerprint, candidates={str(candidate): facts}),
            p1=dict(seal='', parent_fingerprint='', candidates={})))
    print('PREPARED', candidate, facts['model_sha256'], flush=True)


def worker(out, candidate, shard, shards):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with bind(out):
        evaluate.worker(MODE, POLICY, (candidate,), shard, shards)


def collect(out, candidate):
    with bind(out):
        return evaluate.collect(MODE, POLICY, candidate)


def analyze(out, candidate, n=10_000):
    out = Path(out)
    with bind(out):
        result = json.loads((protocol.output_dir(MODE, POLICY, candidate)/'result.json').read_text())
        prepared = evaluate.load_prepared()
    if result['status'] != 'complete':
        raise SystemExit(f'{result["status"]}: missing trials {result["missing_ordinals"]}')
    pairs, dropped = analysis.pairs_from_rows(result['episodes'], excluded=prepared['excluded_layouts'])
    pen = analysis.penalty(pairs, n=n)
    report = dict(candidate=candidate, model_sha256=result['model_sha256'], plan_sha256=(out/'plan.sha256').read_text().strip(),
                  parent_fingerprint=result['parent_fingerprint'], pairs=len(pairs), dropped_pairs=dropped, penalty=pen,
                  reading=analysis.reading(pen['lo'], pen['hi']), pair_table=analysis.pair_table(pairs),
                  by=analysis.penalty_by(pairs), causes=analysis.causes(pairs), bootstrap=dict(n=n, seed=0))
    _write_once(out/'analysis.json', json.dumps(report, indent=2, default=float) + '\n')
    print(json.dumps(report, indent=2, default=float), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'feasibility', 'merge', 'prepare', 'worker', 'collect', 'analyze'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True)
        if name in ('prepare', 'worker', 'collect', 'analyze'):
            p.add_argument('--candidate', type=int, required=True)
        if name in ('feasibility', 'worker'):
            p.add_argument('--shard', type=int, default=0)
            p.add_argument('--shards', type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        print('PLAN', write_plan(args.out))
    elif args.command == 'feasibility':
        feasibility_shard(args.out, args.shard, args.shards)
    elif args.command == 'merge':
        merge_feasibility(args.out)
    elif args.command == 'prepare':
        prepare(args.out, args.candidate)
    elif args.command == 'worker':
        worker(args.out, args.candidate, args.shard, args.shards)
    elif args.command == 'collect':
        collect(args.out, args.candidate)
    else:
        analyze(args.out, args.candidate)
    return 0


if __name__ == '__main__':
    sys.exit(main())
