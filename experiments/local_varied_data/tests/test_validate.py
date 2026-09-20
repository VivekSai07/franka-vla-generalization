"""Own count/split/admission wrappers, shared validators remain unchanged."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.paired_demonstrations.media import FEATURES
from experiments.fixed_layout.contract import TASK


class ValidateTests(unittest.TestCase):
    def setUp(self):
        try:
            self.v = importlib.import_module('experiments.local_varied_data.validate')
        except ModuleNotFoundError as exc: self.fail(f'Missing validator: {exc}')

    def test_60_metadata_48_train_only_not_24_or_all_train(self):
        root = Path(tempfile.mkdtemp()); meta = root / 'dataset/meta'
        (meta / 'episodes').mkdir(parents=True)
        features = json.loads(json.dumps(FEATURES))
        for key in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index'):
            features[key] = {'dtype': 'float32' if key == 'timestamp' else 'int64', 'shape': [1], 'names': None}
        info = dict(total_episodes=60, total_frames=31680, total_tasks=1, fps=30,
                    splits={'train': '0:48', 'validation': '48:60'}, features=features)
        pd.DataFrame({'task_index': [0]}, index=[TASK]).to_parquet(meta / 'tasks.parquet')
        entries = [dict(episode_index=i, length=528, tasks=[TASK], dataset_from_index=i*528,
                        dataset_to_index=(i+1)*528) for i in range(60)]
        pq.write_table(pa.Table.from_pylist(entries), meta / 'episodes/episodes.parquet')
        (meta / 'info.json').write_text(json.dumps(info))
        self.assertEqual(len(self.v.metadata(root, 60, final=True)[1]), 60)
        for key, value in (('total_episodes', 24), ('total_frames', 31679), ('splits', {'train': '0:60'})):
            (meta / 'info.json').write_text(json.dumps({**info, key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError): self.v.metadata(root, 60, final=True)

    def test_exact_120_replay_coverage_no_regeneration_duplicates_or_eval(self):
        results = [dict(episode=i, mode=m, passed=True) for i in range(60) for m in ('reference', 'persisted_float32')]
        self.v.check_replay_coverage(results)
        for bad in (results[:-1], results+results[:1], results[:-1]+[results[0]],
                    [{**r, 'mode': 'expert_regeneration'} for r in results], [{**r, 'episode': False} for r in results]):
            with self.assertRaises(ValueError): self.v.check_replay_coverage(bad)

    def test_journal_requires_preparation_after_12_before_13(self):
        from experiments.local_varied_data.plan import make_plan
        p = make_plan(); events = []
        for slot in p['episodes']:
            events += [dict(event='started', **slot, training_ready=False, elapsed_seconds=float(len(events))),
                       dict(event='admitted', episode_index=slot['episode_index'], training_ready=False, elapsed_seconds=float(len(events)+1))]
            if slot['episode_index'] == 11:
                events.append(dict(event='preparation_pass', episode_ids=list(range(12)), training_ready=False, elapsed_seconds=float(len(events))))
        self.v.validate_journal(events, p)
        for mutate in ('missing', 'early', 'seed', 'deadline'):
            bad = copy.deepcopy(events)
            if mutate == 'missing': bad.pop(24)
            elif mutate == 'early': bad.insert(6, bad.pop(24))
            elif mutate == 'seed': bad[0]['seed'] += 1
            else: bad[-1]['elapsed_seconds'] = 21600.
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): self.v.validate_journal(bad, p)

    def test_fresh_admission_rejects_stale_seal_before_any_initialization(self):
        root = Path(tempfile.mkdtemp(dir='/vla/outputs/implementation/local_varied_data_20260919'))
        (root / 'validation.json').write_text(json.dumps({'status': 'passed', 'training_ready': True}))
        (root / 'seal.json').write_text(json.dumps({'fingerprint': 'stale'}))
        with self.assertRaisesRegex(ValueError, 'stale'): self.v.require_local_varied_pass(root)

    def test_episode_gate_rejects_noncanonical_eval_or_out_of_order_slot_before_io(self):
        from experiments.local_varied_data.plan import make_plan
        slot = make_plan()['episodes'][0]
        for field, value in (('layout_id', 'evaluation_000'), ('episode_index', 60), ('seed', 1)):
            bad = {**slot, field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'slot'):
                self.v.validate_episode(Path('/tmp/nonexistent-local-varied-test'), bad, None, None, lambda: None)


if __name__ == '__main__': unittest.main()