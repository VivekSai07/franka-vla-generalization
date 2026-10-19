import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from experiments.xvla_props_p2_eval import evaluate, feasibility, protocol
from experiments.xvla_props_p2_retest import plan as rt_plan
from experiments.xvla_props_p2_retest import retest

GLOBALS = [(protocol, n) for n in ('PLAN_JSON', 'PARENT', 'EVAL_ROOT', 'REPORT_SEEDS')] + \
          [(evaluate, n) for n in ('PREPARED', 'FEASIBILITY', 'DECISIONS')]


def snapshot():
    return {(m.__name__, n): getattr(m, n) for m, n in GLOBALS}


class BindTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_bind_points_the_unchanged_evaluator_at_the_retest_and_restores_everything(self):
        before = snapshot()
        with retest.bind(self.tmp):
            self.assertEqual(protocol.PLAN_JSON, self.tmp/'plan.json')
            self.assertEqual(protocol.EVAL_ROOT, self.tmp)
            self.assertEqual(protocol.REPORT_SEEDS, 1)
            self.assertEqual(evaluate.PREPARED, self.tmp/'prepared.json')
            self.assertEqual(evaluate.FEASIBILITY, self.tmp/'feasibility'/'feasibility.json')
        self.assertEqual(snapshot(), before)

    def test_bind_restores_even_when_the_body_raises(self):
        before = snapshot()
        with self.assertRaises(RuntimeError):
            with retest.bind(self.tmp):
                raise RuntimeError('boom')
        self.assertEqual(snapshot(), before)

    def test_the_p2_evaluator_selects_exactly_the_480_retest_trials_under_the_binding(self):
        retest.write_plan(self.tmp)
        with retest.bind(self.tmp):
            trials = protocol.select_trials('report')
            jobs = protocol.jobs('report', 'p2', (101520,))
        self.assertEqual(len(trials), 480)
        self.assertEqual(len(jobs), 480)
        self.assertEqual(sorted(t['layout_id'] for t in trials), sorted(rt_plan.make_plan()['evaluation_report_ids']))
        self.assertEqual({t['kind'] for t in trials}, {'trained', 'held_out'})
        self.assertEqual(sum(t['kind'] == 'held_out' for t in trials), 240)

    def test_trial_seeds_are_distinct(self):
        retest.write_plan(self.tmp)
        with retest.bind(self.tmp):
            seeds = [t['seed'] for t in protocol.select_trials('report')]
        self.assertEqual(len(set(seeds)), 480)


class PlanFileTests(unittest.TestCase):
    def test_write_plan_records_the_digest_and_refuses_to_overwrite(self):
        tmp = Path(tempfile.mkdtemp())
        digest = retest.write_plan(tmp)
        self.assertEqual((tmp/'plan.sha256').read_text().strip(), digest)
        self.assertEqual(digest, rt_plan.digest(json.loads((tmp/'plan.json').read_text())))
        with self.assertRaises(SystemExit):
            retest.write_plan(tmp)


class LoadFeasibilityTests(unittest.TestCase):
    """The P2 evaluator's own loader insists on 120 layouts (a P2 constant) and must stay unedited; the re-test checks the
    count against its own plan."""

    def write(self, tmp, keys):
        (tmp/'feasibility').mkdir(exist_ok=True)
        results = {k: dict(success=True, reason='success', frames=452, stable_ticks=30, seconds=1.0) for k in keys}
        (tmp/'feasibility'/'feasibility.json').write_text(json.dumps(feasibility.summarize(results)))

    def test_accepts_a_summary_covering_exactly_the_retest_plan(self):
        tmp = Path(tempfile.mkdtemp())
        retest.write_plan(tmp)
        self.write(tmp, sorted(rt_plan.make_plan()['evaluation_holdout']))
        self.assertEqual(retest.load_feasibility(tmp)['keys'], 480)

    def test_refuses_a_summary_that_covers_a_different_set(self):
        tmp = Path(tempfile.mkdtemp())
        retest.write_plan(tmp)
        keys = sorted(rt_plan.make_plan()['evaluation_holdout'])
        self.write(tmp, keys[:-1])
        with self.assertRaises(ValueError):
            retest.load_feasibility(tmp)
        (tmp/'feasibility'/'feasibility.json').unlink()
        self.write(tmp, keys + ['not_a_plan_layout'])                 # a true superset: every plan layout plus one more
        with self.assertRaises(ValueError):
            retest.load_feasibility(tmp)

    def test_missing_file_is_an_error(self):
        tmp = Path(tempfile.mkdtemp())
        retest.write_plan(tmp)
        with self.assertRaises(RuntimeError):
            retest.load_feasibility(tmp)


class FeasibilityMergeTests(unittest.TestCase):
    def shard(self, out, k, keys, failed=()):
        (out/'feasibility').mkdir(exist_ok=True)
        results = {key: dict(success=key not in failed, reason='success', frames=452, stable_ticks=30, seconds=1.0) for key in keys}
        (out/'feasibility'/f'feasibility.shard_{k}.json').write_text(json.dumps(feasibility.summarize(results)))

    def test_merge_needs_every_layout_and_lists_the_failures(self):
        tmp = Path(tempfile.mkdtemp())
        retest.write_plan(tmp)
        keys = sorted(rt_plan.make_plan()['evaluation_holdout'])
        self.shard(tmp, 0, keys[:200])
        with self.assertRaises(SystemExit):
            retest.merge_feasibility(tmp)
        self.shard(tmp, 1, keys[200:], failed=[keys[250]])
        summary = retest.merge_feasibility(tmp)
        self.assertEqual((summary['keys'], summary['succeeded'], summary['failed_layouts']), (480, 479, [keys[250]]))
        with self.assertRaises(SystemExit):
            retest.merge_feasibility(tmp)

    def test_merge_does_not_accept_the_wrong_number_of_layouts_even_if_they_match(self):
        tmp = Path(tempfile.mkdtemp())
        retest.write_plan(tmp)
        keys = sorted(rt_plan.make_plan()['evaluation_holdout'])
        self.shard(tmp, 0, keys + ['extra_layout'])
        with self.assertRaises(SystemExit):
            retest.merge_feasibility(tmp)


if __name__ == '__main__':
    unittest.main()
