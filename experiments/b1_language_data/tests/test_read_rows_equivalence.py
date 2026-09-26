"""Pin the local `read_rows` copy to the shared original.

`fixed_layout.validate` is inside the sealed provenance of `broad_workspace_data` and cannot be
edited, so B1 copies `read_rows` with one line changed. A copied validator that is accidentally
*weaker* than the original would still pass, which is the worst failure mode in this repository —
these tests exist so the copy cannot silently drift.
"""
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.fixed_layout import validate as generic
from experiments.b1_language_data import plan as p
from experiments.b1_language_data import validate as v

N = p.EXPECTED_FRAMES


def _table(task_index=0, episode=0, **overrides):
    data = {
        'action': pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(N*8, np.float32)), 8),
        'observation.state': pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(N*9, np.float32)), 9),
        'timestamp': pa.array(np.arange(N)/30, pa.float32()),
        'frame_index': pa.array(np.arange(N), pa.int64()),
        'episode_index': pa.array(np.full(N, episode), pa.int64()),
        'index': pa.array(np.arange(N) + episode*N, pa.int64()),
        'task_index': pa.array(np.full(N, task_index), pa.int64()),
    }
    data.update(overrides)
    return pa.table(data)


class ReadRowsEquivalenceTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'rows.parquet'

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, table):
        pq.write_table(table, self.path)
        return self.path

    def test_read_rows_matches_shared_on_single_task(self):
        """With task_index 0 the copy must agree with the original, value for value."""
        path = self._write(_table(task_index=0))
        mine = v.read_rows_for_task(path, 0, 0)
        theirs = generic.read_rows(path, 0)
        self.assertEqual(len(mine), len(theirs))
        for a, b in zip(mine, theirs):
            np.testing.assert_array_equal(a, b)

    def test_copy_accepts_a_nonzero_task_index_the_original_rejects(self):
        path = self._write(_table(task_index=1))
        actions, states, times = v.read_rows_for_task(path, 0, 1)
        self.assertEqual(actions.shape, (N, 8))
        with self.assertRaises(ValueError):
            generic.read_rows(path, 0)  # the original hardcodes zeros

    def test_copy_still_rejects_the_wrong_task_index(self):
        path = self._write(_table(task_index=1))
        with self.assertRaises(ValueError):
            v.read_rows_for_task(path, 0, 0)

    def test_copy_retains_every_other_rejection_of_the_original(self):
        """Each corruption the shared version rejects must still be rejected by the copy."""
        cases = {
            'wrong frame_index': dict(frame_index=pa.array(np.arange(N)[::-1].copy(), pa.int64())),
            'wrong episode_index': dict(episode_index=pa.array(np.full(N, 3), pa.int64())),
            'wrong global index': dict(index=pa.array(np.arange(N), pa.int64())),
            'int32 indices': dict(frame_index=pa.array(np.arange(N), pa.int32())),
            'float64 timestamps': dict(timestamp=pa.array(np.arange(N)/30, pa.float64())),
            'timestamp gap': dict(timestamp=pa.array(np.arange(N)/15, pa.float32())),
        }
        for name, override in cases.items():
            with self.subTest(case=name):
                path = self._write(_table(task_index=0, episode=1, **override))
                with self.assertRaises(ValueError):
                    generic.read_rows(path, 1)
                with self.assertRaises(ValueError):
                    v.read_rows_for_task(path, 1, 0)

    def test_copy_rejects_missing_and_extra_columns(self):
        table = _table()
        dropped = table.drop(['task_index'])
        path = self._write(dropped)
        with self.assertRaises(ValueError):
            v.read_rows_for_task(path, 0, 0)
        extra = table.append_column('privileged', pa.array(np.zeros(N), pa.float64()))
        path = self._write(extra)
        with self.assertRaises(ValueError):
            v.read_rows_for_task(path, 0, 0)


class _FakeModel:
    def body(self, name):
        return type('Body', (), {'id': 0})()


class _FakeExpert:
    model = _FakeModel()


class TaskAwareContextTests(unittest.TestCase):
    RESET = {'body_xpos': np.zeros((1, 3))}

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'rows.parquet'
        pq.write_table(_table(task_index=1), self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def _enter(self, instruction, task_index):
        return v.task_aware(instruction, task_index, 'red', 'green', _FakeExpert(), self.RESET)

    def test_context_restores_the_shared_module(self):
        before_rows, before_task = generic.read_rows, generic.TASK
        with self._enter('pick up the green cube and place it on top of the red cube', 1):
            self.assertIsNot(generic.read_rows, before_rows)
            self.assertNotEqual(generic.TASK, before_task)
        self.assertIs(generic.read_rows, before_rows)
        self.assertEqual(generic.TASK, before_task)

    def test_context_restores_even_when_the_body_raises(self):
        before_rows, before_task = generic.read_rows, generic.TASK
        with self.assertRaises(RuntimeError):
            with self._enter('anything', 0):
                raise RuntimeError('boom')
        self.assertIs(generic.read_rows, before_rows)
        self.assertEqual(generic.TASK, before_task)

    def test_patched_read_rows_expects_the_contexts_task_index(self):
        with self._enter('anything', 1):
            generic.read_rows(self.path, 0)
        with self._enter('anything', 0):
            with self.assertRaises(ValueError):
                generic.read_rows(self.path, 0)


if __name__ == '__main__':
    unittest.main()
