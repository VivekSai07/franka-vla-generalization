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
