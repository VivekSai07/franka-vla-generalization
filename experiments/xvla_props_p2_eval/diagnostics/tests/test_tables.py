import unittest

from experiments.xvla_props_p2_eval.diagnostics import tables


def trial(layout, cls, kind='trained', obj='can', cell_bin='blue', success=None, distractor_distance=0.15,
          first_lifted=None, names=None, cause=None, **extra):
    ok = cls == 'success' if success is None else success
    cause = cause or ('success' if ok else {'wrong_object': 'wrong_object', 'lifted_then_failed': 'timeout_carrying'}.get(cls, 'never_lifted'))
    return dict(layout_id=layout, **{'class': cls}, kind=kind, object=obj, bin=cell_bin, task_success=ok, cause=cause,
                min_distractor_distance_m=distractor_distance, first_lifted=first_lifted, names=names or [obj, 'milk'],
                grasp_inside=True if ok else None, held=False, closest_approach_m=0.05, **extra)


class TaxonomyTableTests(unittest.TestCase):
    def test_counts_by_class_and_the_total(self):
        rows = [trial('a', 'success'), trial('b', 'never_closed'), trial('c', 'never_closed'), trial('d', 'closed_off_target')]
        out = tables.taxonomy_table({'run': rows})
        self.assertEqual(out['run']['success'], 1)
        self.assertEqual(out['run']['never_closed'], 2)
        self.assertEqual(out['run']['closed_off_target'], 1)
        self.assertEqual(out['run']['n'], 4)
        self.assertEqual(sum(v for k, v in out['run'].items() if k != 'n'), 4)

    def test_every_class_key_is_present_even_when_zero(self):
        out = tables.taxonomy_table({'run': [trial('a', 'success')]})
        for cls in ('success', 'wrong_object', 'lifted_then_failed', 'never_closed', 'closed_on_distractor',
                    'closed_off_target', 'closed_not_held', 'held_not_lifted'):
            self.assertIn(cls, out['run'])


class LayoutTests(unittest.TestCase):
    def test_wrong_object_layout_test_uses_the_layout_minimum_distance_as_the_unit(self):
        # Two trials of layout 'p' (positive) with different distances: the layout's value is the minimum.
        rows = [trial('p', 'wrong_object', distractor_distance=0.12), trial('p', 'success', distractor_distance=0.30),
                trial('q', 'success', distractor_distance=0.20), trial('r', 'success', distractor_distance=0.22),
                trial('s', 'success', distractor_distance=0.19)]
        out = tables.wrong_object_layout_test(rows)
        self.assertEqual((out['positive_layouts'], out['negative_layouts']), (1, 3))
        self.assertAlmostEqual(out['positive_median_m'], 0.12)
        self.assertAlmostEqual(out['negative_median_m'], 0.20)
        self.assertAlmostEqual(out['p'], 0.5, places=12)  # 1 of 4 layouts positive and it is the lowest: 2 of 4 splits as extreme

    def test_concentration_of_never_lifted(self):
        rows = [trial('a', 'never_closed'), trial('a', 'never_closed'), trial('b', 'success'), trial('b', 'success'),
                trial('c', 'success'), trial('c', 'success'), trial('d', 'success'), trial('d', 'success')]
        out = tables.never_lifted_concentration(rows)
        self.assertEqual(out['failed_trials'], 2)
        self.assertEqual(out['layouts_with_failures'], 1)
        self.assertEqual(out['statistic'], 4)
        self.assertLess(out['p'], 0.5)


class MilkTests(unittest.TestCase):
    def test_milk_rows_split_by_cell_with_classes(self):
        rows = [trial('a', 'success', obj='milk', kind='held_out', cell_bin='blue'),
                trial('b', 'closed_off_target', obj='milk', kind='held_out', cell_bin='blue'),
                trial('c', 'closed_not_held', obj='milk', kind='trained', cell_bin='orange'),
                trial('d', 'success', obj='can')]
        out = tables.milk_table(rows)
        self.assertEqual(out['milk->blue (held_out)']['n'], 2)
        self.assertEqual(out['milk->blue (held_out)']['success'], 1)
        self.assertEqual(out['milk->blue (held_out)']['closed_off_target'], 1)
        self.assertEqual(out['milk->orange (trained)']['closed_not_held'], 1)
        self.assertEqual(sorted(out), ['milk->blue (held_out)', 'milk->orange (trained)'])
        self.assertEqual(sum(v['n'] for v in out.values()), 3, 'the can trial must not be counted')


def raw(layout, cause, success=False, closed=True, held=False, approach=0.05, obj='can', kind='trained', cell_bin='blue',
        first_closure=100, touched=(), sample=(0.0, 0.0, 0.12, 0.0), distance=0.15, first_lifted=None):
    """A merged evaluation row plus RAW measures (no envelope, no class): what analyze has when Gate 1 fails."""
    return dict(layout_id=layout, cause='success' if success else cause, task_success=success, closed=closed, held=held,
                closest_approach_m=approach, object=obj, kind=kind, bin=cell_bin, first_closure=first_closure if closed else None,
                touched_distractors=list(touched), grasp_sample=list(sample) if closed else None,
                min_distractor_distance_m=distance, first_lifted=first_lifted, names=[obj, 'milk'], lifted=success)


class DescriptiveTests(unittest.TestCase):
    def test_never_lifted_detail_splits_by_what_the_gripper_did(self):
        rows = [raw('a', 'never_lifted', closed=False), raw('b', 'never_lifted', closed=True, held=False),
                raw('c', 'never_lifted', closed=True, held=True), raw('d', 'never_lifted', closed=False),
                raw('e', 'success', success=True), raw('f', 'timeout_carrying')]
        out = tables.never_lifted_detail(rows)
        self.assertEqual((out['n'], out['never_closed'], out['closed_not_held'], out['held']), (4, 2, 1, 1))

    def test_closest_approach_summary_compares_with_successes(self):
        rows = [raw('a', 'never_lifted', approach=0.20), raw('b', 'never_lifted', approach=0.10),
                raw('c', 'success', success=True, approach=0.01), raw('d', 'success', success=True, approach=0.03),
                raw('e', 'timeout_carrying', approach=0.0), raw('f', 'wrong_object', approach=0.0)]  # not never-lifted
        out = tables.closest_approach_summary(rows)
        self.assertAlmostEqual(out['never_lifted_median_m'], 0.15)
        self.assertAlmostEqual(out['success_median_m'], 0.02)
        self.assertEqual((out['never_lifted_n'], out['success_n']), (2, 2))

    def test_wrong_object_test_works_on_raw_rows_through_the_evaluations_cause(self):
        rows = [raw('p', 'wrong_object', distance=0.12), raw('p', 'success', success=True, distance=0.30),
                raw('q', 'success', success=True, distance=0.20), raw('r', 'success', success=True, distance=0.22),
                raw('s', 'success', success=True, distance=0.19)]
        out = tables.wrong_object_layout_test(rows)
        self.assertEqual((out['positive_layouts'], out['negative_layouts']), (1, 3))
        self.assertAlmostEqual(out['positive_median_m'], 0.12)
        cases = tables.wrong_object_cases(rows)
        self.assertEqual(len(cases), 1)

    def test_milk_raw_table_has_samples_and_no_inside_outside_label(self):
        rows = [raw('a', 'never_lifted', obj='milk', kind='held_out', cell_bin='blue', sample=(0.01, 0.0, 0.17, -3.0)),
                raw('b', 'success', success=True, obj='milk', kind='held_out', cell_bin='blue'),
                raw('c', 'never_lifted', obj='milk', kind='trained', cell_bin='orange', closed=False),
                raw('d', 'success', success=True, obj='can')]
        out = tables.milk_raw_table(rows)
        self.assertEqual(sorted(out), ['milk->blue (held_out)', 'milk->orange (trained)'])
        blue = out['milk->blue (held_out)']
        self.assertEqual((blue['n'], blue['success'], blue['failed']), (2, 1, 1))
        self.assertEqual(blue['failures'][0]['grasp_sample'], [0.01, 0.0, 0.17, -3.0])
        self.assertNotIn('inside', ' '.join(blue['failures'][0]))
        self.assertIsNone(out['milk->orange (trained)']['failures'][0]['grasp_sample'])

    def test_build_descriptive_and_render(self):
        data = {n: [raw('a', 'never_lifted', closed=False), raw('b', 'success', success=True)]
                for n in ('screen_p2_step_101520', 'report_p2_step_101520', 'control_p2_step_101520',
                          'confirm_p2_step_101520', 'report_p2_step_126900', 'report_p1_step_126900')}
        result = tables.build_descriptive(data)
        text = tables.render_descriptive(result)
        self.assertIn('| run | never lifted | never closed | closed, never held | held, not lifted |', text)
        self.assertIn('| report_p2_step_101520 | 1 | 1 | 0 | 0 |', text)
        self.assertIn('| 101,520 all four tiers | 4 | 4 | 0 | 0 |', text)
        self.assertNotIn('closed_off_target', text, 'the envelope taxonomy must not appear')


class RenderTests(unittest.TestCase):
    def test_build_and_render_produce_markdown_with_the_run_names(self):
        data = {'report_p2_step_101520': [trial('a', 'success'), trial('b', 'never_closed', kind='held_out', obj='milk')],
                'report_p2_step_126900': [trial('a', 'success')], 'report_p1_step_126900': [trial('a', 'never_closed')],
                'screen_p2_step_101520': [], 'control_p2_step_101520': [], 'confirm_p2_step_101520': []}
        result = tables.build(data)
        text = tables.render(result)
        self.assertIn('report_p2_step_101520', text)
        self.assertIn('| class |', text)
        self.assertIn('never_closed', text)


if __name__ == '__main__':
    unittest.main()
