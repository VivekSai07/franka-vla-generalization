"""Offline CPU test evidence and protected-file inventories. No actual trials."""
import contextlib
import json
import os
from pathlib import Path
import unittest

from data_collection.benchmark import source_provenance
from .diagnostic import ROOT, OUT, PAIRED, IMAGE, save_json, sha
from .solver import PlanningFailure


def implementation_hashes():
    paths = sorted((ROOT/'experiments/bounded_placement_v3').glob('*.py'))
    paths += sorted((ROOT/'tests').glob('test_bounded_placement_v3*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def protected_hashes():
    paths = set(ROOT/p for p in source_provenance(ROOT))
    for directory in ('experiments/paired_demonstrations', 'experiments/pose_snapshot_v2',
                      'outputs/paired_demonstrations/paired_20260917',
                      'outputs/diagnostics/pose_snapshot_v2_20260917'):
        paths.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update(p for p in (ROOT/'tests').glob('*.py') if not p.name.startswith('test_bounded_placement_v3'))
    paths.add(ROOT/'docs/superpowers/specs/2026-09-17-bounded-placement-design.md')
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def require_cpu_runtime():
    if (os.environ.get('DIAGNOSTIC_IMAGE_ID') != IMAGE
            or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or list(Path('/dev').glob('nvidia*'))
            or os.environ.get('HF_HUB_OFFLINE') != '1'
            or not os.environ.get('HF_HOME', '').startswith('/tmp/')):
        raise PlanningFailure('pinned_offline_cpu_runtime_required')


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def require_test_evidence(out, sources):
    """Require the refreshed evidence; never fall back to historical passes."""
    path = Path(out)/'tests_passed_reviewed.json'
    if not path.is_file():
        raise PlanningFailure('current_cpu_test_evidence_required')
    tests = json.loads(path.read_text())
    suites = [tests.get(name, {}) for name in ('legacy', 'new')]
    if (tests.get('source_hashes') != sources or tests.get('image') != IMAGE
            or tests.get('actual_expert_invocations') != 0 or suites[0].get('count') != 137
            or suites[1].get('count', 0) <= 0
            or not all(s.get('passed') is True and all(s.get(k) == 0 for k in ('failures', 'errors', 'skipped'))
                       for s in suites)
            or tests.get('protected_hashes_unchanged') is not True
            or tests.get('implementation_hashes_unchanged') is not True):
        raise PlanningFailure('current_cpu_test_evidence_required')
    return tests


def test_evidence():
    require_cpu_runtime()
    before, sources = protected_hashes(), implementation_hashes()
    if json.loads((OUT/'hashes_before.json').read_text()) != before:
        raise PlanningFailure('protected_hashes_changed_before_tests')
    historical = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    save_json(OUT/'hashes_before_reviewed.json', before)
    tests = list(flatten(unittest.defaultTestLoader.discover(str(ROOT/'tests'))))
    old = [t for t in tests if not t.id().startswith('test_bounded_placement_v3')]
    new = [t for t in tests if t.id().startswith('test_bounded_placement_v3')]
    if len(old) != 137 or not new:
        raise PlanningFailure('test_inventory', old=len(old), new=len(new))
    summary = dict(source_hashes=sources, image=IMAGE, actual_expert_invocations=0,
                   historical_evidence_hashes=historical)
    for name, cases in (('legacy', old), ('new', new)):
        with (OUT/f'tests_{name}_reviewed.log').open('x') as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestSuite(cases))
        summary[name] = dict(count=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                             skipped=len(result.skipped), passed=result.wasSuccessful() and not result.skipped)
        print(f'{name}: {summary[name]}', flush=True)
    after = protected_hashes()
    save_json(OUT/'hashes_after_tests_reviewed.json', after)
    summary['protected_hashes_unchanged'] = before == after
    summary['implementation_hashes_unchanged'] = sources == implementation_hashes()
    summary['historical_evidence_unchanged'] = all(sha(OUT/name) == digest for name, digest in historical.items())
    save_json(OUT/'tests_passed_reviewed.json', summary)
    if not (summary['legacy']['passed'] and summary['new']['passed']
            and summary['protected_hashes_unchanged'] and summary['implementation_hashes_unchanged']
            and summary['historical_evidence_unchanged']):
        raise PlanningFailure('test_or_integrity_failure')
    return summary


if __name__ == '__main__':
    test_evidence()