import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.gen_g1_data import plan as p
from experiments.gen_g1_data import sheet as s
from experiments.props_scene import constants as c

PLAN = p.make_plan()


class SelectionTests(unittest.TestCase):
    def test_twenty_four_gate_attempts_four_per_object_in_gate_order(self):
        indices = s.sheet_indices(PLAN)
        self.assertEqual(len(indices), 24)
        self.assertEqual(indices, sorted(set(indices)))
        by_object = Counter(PLAN["gate_attempts"][i]["object"] for i in indices)
        self.assertEqual(by_object, {o: 4 for o in p.OBJECTS})
        for obj in p.OBJECTS:
            first = [a["episode_index"] for a in PLAN["gate_attempts"] if a["object"] == obj][:4]
            self.assertTrue(set(first) <= set(indices), obj)

    def test_the_selection_spans_distractor_counts_and_heights(self):
        chosen = [PLAN["gate_attempts"][i] for i in s.sheet_indices(PLAN)]
        self.assertGreaterEqual(len({a["distractor_count"] for a in chosen}), 4)
        self.assertGreaterEqual(len({a["quintile"] for a in chosen}), 4)

    def test_the_carry_tick_is_inside_the_move_phase(self):
        move = c.PHASE_NAMES.index("move")
        phases = [0] * 10 + [move] * 31 + [move + 1] * 5
        self.assertEqual(s.carry_tick(phases), 25)
        self.assertEqual(c.PHASE_NAMES[phases[s.carry_tick(phases)]], "move")
        self.assertEqual(s.carry_tick([0] * 7), 6)


class TileTests(unittest.TestCase):
    def test_cells_are_laid_out_row_major_and_the_last_row_is_padded(self):
        cells = [np.full((4, 6, 3), 10 * (i + 1), np.uint8) for i in range(5)]
        sheet = s.tile(cells, columns=3)
        self.assertEqual(sheet.shape, (8, 18, 3))
        self.assertEqual(sheet[0, 0, 0], 10)
        self.assertEqual(sheet[0, 12, 0], 30)
        self.assertEqual(sheet[4, 0, 0], 40)
        self.assertEqual(sheet[4, 6, 0], 50)
        self.assertEqual(int(sheet[4:, 12:].sum()), 0)

    def test_cells_must_be_the_same_size(self):
        with self.assertRaises(ValueError):
            s.tile([np.zeros((4, 6, 3), np.uint8), np.zeros((5, 6, 3), np.uint8)])
        with self.assertRaises(ValueError):
            s.tile([])

    def test_a_cell_is_both_cameras_with_a_label_strip(self):
        front, gripper = np.full((512, 512, 3), 90, np.uint8), np.full((512, 512, 3), 200, np.uint8)
        cell = s.make_cell(front, gripper, "07 can n=3 h=+5cm OK")
        self.assertEqual(cell.shape, (s.LABEL_HEIGHT + s.CELL, 2 * s.CELL, 3))
        self.assertTrue((cell[s.LABEL_HEIGHT:, :s.CELL] == 90).all())
        self.assertTrue((cell[s.LABEL_HEIGHT:, s.CELL:] == 200).all())
        self.assertGreater(int(cell[:s.LABEL_HEIGHT].sum()), 0)

    def test_the_label_names_the_slot_object_count_height_and_verdict(self):
        attempt = PLAN["gate_attempts"][5]
        text = s.label(attempt, "OK")
        self.assertIn(f"{attempt['episode_index']:02d} {attempt['object']} n={attempt['distractor_count']}", text)
        self.assertTrue(text.endswith(" OK"))
        self.assertIn("cm", text)


class AssembleTests(unittest.TestCase):
    def test_the_png_has_the_cameras_colours_not_swapped(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for index in s.sheet_indices(PLAN):
                rgb = np.zeros((512, 512, 3), np.uint8)
                rgb[:] = (200, 100, 50)
                np.savez(out / f"cell_{index:03d}.npz", **{f"{n}_{k}": rgb for n in ("scene", "carry") for k in (0, 1)})
                (out / f"cell_{index:03d}.json").write_text(json.dumps({"label": "x"}))
            s.assemble(out)
            bgr = cv2.imread(str(out / "sheet_scene.png"))[s.LABEL_HEIGHT + 5, 5]
            self.assertEqual(list(bgr), [50, 100, 200])

    def test_assemble_refuses_missing_cells_and_writes_both_sheets(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaises(SystemExit):
                s.assemble(out)
            for index in s.sheet_indices(PLAN):
                saved = {f"{name}_{k}": np.full((512, 512, 3), 40 + k * 80, np.uint8) for name in ("scene", "carry") for k in (0, 1)}
                np.savez(out / f"cell_{index:03d}.npz", **saved)
                (out / f"cell_{index:03d}.json").write_text(json.dumps({"label": f"cell {index}"}))
            s.assemble(out)
            import cv2
            for name in ("scene", "carry"):
                image = cv2.imread(str(out / f"sheet_{name}.png"))
                self.assertEqual(image.shape, (8 * (s.LABEL_HEIGHT + s.CELL), s.COLUMNS * 2 * s.CELL, 3))
            (out / f"cell_{s.sheet_indices(PLAN)[0]:03d}.npz").unlink()
            with self.assertRaises(SystemExit):
                s.assemble(out)


if __name__ == "__main__":
    unittest.main()
