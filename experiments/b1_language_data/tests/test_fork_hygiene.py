"""Fork hygiene: catch references the 2x2 rewrite left behind, without executing collection.

`xvla_placement_modes` shipped twice with bugs that only surfaced at runtime — a deleted name
still referenced in `run()`, and an inherited roster guard. Both were statically detectable.
"""
import builtins
import importlib
import symtable
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
MODULES = ('plan.py', 'scenes.py', 'validate.py', 'record.py', 'expert.py')
# Injected by the import machinery, so symtable never sees them bound.
MODULE_DUNDERS = {'__file__', '__name__', '__doc__', '__package__', '__spec__', '__loader__',
                  '__builtins__', '__path__', '__debug__'}


class UndefinedNameTests(unittest.TestCase):
    def test_no_module_reads_a_name_that_is_never_bound(self):
        offenders = []
        for name in MODULES:
            source = (PACKAGE / name).read_text()
            top = symtable.symtable(source, name, 'exec')
            module_names = set(top.get_identifiers()) | MODULE_DUNDERS

            def walk(table):
                for child in table.get_children():
                    if child.get_type() == 'function':
                        for sym in child.get_symbols():
                            ident = sym.get_name()
                            if (sym.is_global() and not sym.is_assigned()
                                    and ident not in module_names and not hasattr(builtins, ident)):
                                offenders.append(f'{name}:{child.get_name()}() reads undefined {ident!r}')
                    walk(child)

            walk(top)
        self.assertEqual(offenders, [], '; '.join(offenders))

    def test_no_module_still_references_the_source_package(self):
        """Imports, not prose — the fork source is legitimately named in comments about provenance."""
        for name in MODULES:
            source = (PACKAGE / name).read_text()
            for banned in ('from experiments.broad_workspace_data',
                           'import experiments.broad_workspace_data',
                           'experiments.broad_workspace_data.'):
                self.assertNotIn(banned, source, f'{name} still imports the fork source')

    def test_no_colour_keyed_envelope_constants_survive(self):
        """Colour-keyed envelopes were the confound; regions replaced them everywhere."""
        for name in MODULES:
            source = (PACKAGE / name).read_text()
            for banned in ('RED_LOW', 'RED_HIGH', 'GREEN_LOW', 'GREEN_HIGH'):
                self.assertNotIn(banned, source, f'{name} still uses {banned}')


class PlanContractTests(unittest.TestCase):
    def setUp(self):
        self.p = importlib.import_module('experiments.b1_language_data.plan')

    def test_instruction_is_never_hardcoded_at_a_call_site(self):
        source = (PACKAGE / 'plan.py').read_text()
        body = source.split('def instruction_for', 1)[1].split('\ndef ', 1)[0]
        self.assertIn('place it on top of', body)
        rest = source.replace(body, '')
        self.assertNotIn('place it on top of the green cube', rest,
                         'instruction text must come from instruction_for() only')

    def test_every_cell_round_trips_through_the_public_helpers(self):
        for cell, spec in self.p.CELLS.items():
            instruction = self.p.instruction_for(spec['pick'], spec['place'])
            self.assertIn(instruction, self.p.INSTRUCTIONS, cell)
            self.assertEqual(self.p.task_index_for(instruction),
                             self.p.INSTRUCTIONS.index(instruction))
            a, b = (0.30, -0.30), (0.30, 0.30)
            placed = self.p.colour_positions(spec['arrangement'], a, b)
            self.assertEqual(len(placed), 4)
            red, green = tuple(placed[:2]), tuple(placed[2:])
            self.assertEqual({red, green}, {a, b}, 'colours must occupy the two regions')

    def test_helpers_reject_unapproved_input(self):
        for bad in (('red', 'red'), ('blue', 'green'), ('red', 'blue')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.p.instruction_for(*bad)
        with self.assertRaises(ValueError):
            self.p.task_index_for('pick up the blue cube and place it on top of the red cube')
        with self.assertRaises(ValueError):
            self.p.colour_positions('mirrored', (0.3, -0.3), (0.3, 0.3))
        with self.assertRaises(ValueError):
            self.p.assign_cells('train', 7, {})  # not divisible by four


if __name__ == '__main__':
    unittest.main()
