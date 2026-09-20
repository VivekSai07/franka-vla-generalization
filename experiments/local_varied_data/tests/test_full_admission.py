"""Full 60-slot adapter audit, real rows/metadata/bank/seal, cheap boundary doubles.

Raw/media/replay interpretation is exercised by the real two-episode integration
test. Here only those expensive boundaries are numeric doubles; no 60 encodes.
Empty replay token files are explicitly synthetic, never physics evidence.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.fixed_layout import validate as g
from experiments.fixed_layout.source import Expert, load_reset
from experiments.fixed_layout.record import write_json, write_npz, file_hash
from experiments.paired_demonstrations.media import FEATURES, CAMERAS
from experiments.local_varied_data import plan as p, record as r, scenes as s, validate as v


class FullAdmissionTests(unittest.TestCase):
    def test_full_60_seal_fresh_admission_and_resealed_mutations(self):
        root = Path(tempfile.mkdtemp(prefix='full-60-', dir='/vla/outputs/implementation/local_varied_data_20260919'))
        plan = p.make_plan(); r.claim_output(root)
        with ExitStack() as guards:
            for name in ('mj_step', 'mj_step1', 'mj_step2', 'mj_Euler', 'mj_RungeKutta', 'Renderer'):
                guards.enter_context(patch.object(mujoco, name, side_effect=AssertionError('No native execution')))
            guards.enter_context(patch.object(Expert, 'run', side_effect=AssertionError('No real expert')))
            expert = Expert(); source = load_reset(expert); bank = s.make_bank(expert.model, source, plan)
            for name, value in (('plan.json', plan), ('provenance.json', s.provenance()),
                                ('configuration.json', expert.config), ('bank.json', s.bank_manifest(plan, bank))):
                write_json(root / name, value)
            (root / 'source.npz').write_bytes((p.ROOT / p.SOURCE).read_bytes()); (root / 'scenes').mkdir()
            for name, arrays in bank.items(): write_npz(root / 'scenes' / f'{name}.npz', arrays)
            write_json(root / 'snapshots.json', {name: file_hash(root / name) for name in r.frozen_names(plan)})
            ds = root / 'dataset'; meta = ds / 'meta'; (meta / 'episodes').mkdir(parents=True)
            (ds / 'data').mkdir(); (ds / 'videos').mkdir(); (root / 'attempts').mkdir()
            features = json.loads(json.dumps(FEATURES))
            for key in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index'):
                features[key] = dict(dtype='float32' if key == 'timestamp' else 'int64', shape=[1], names=None)
            info = dict(total_episodes=60, total_frames=31680, total_tasks=1, fps=30,
                        splits={'train': '0:48', 'validation': '48:60'}, features=features,
                        data_path='data/{file_index:06d}.parquet', video_path='videos/{video_key}/{file_index:06d}.mp4')
            write_json(meta / 'info.json', info)
            pd.DataFrame({'task_index': [0]}, index=[p.TASK]).to_parquet(meta / 'tasks.parquet')
            entries, events, sequences = [], [], []
            expected_reports = {}; boundary_calls = []
            for slot in plan['episodes']:
                i = slot['episode_index']; path = root / 'attempts' / f'{i:06d}'; path.mkdir()
                for directory in ('ticks', 'media', 'replays'): (path / directory).mkdir()
                for name in ('slot.json', 'outcome.json', 'terminal.npz'): (path / name).write_text('synthetic boundary fixture')
                actions = np.full((528, 8), i/100, np.float32); states = np.full((528, 9), i/100, np.float32)
                sequences.append((actions, states))
                pq.write_table(pa.table({
                    'action': pa.array(actions.tolist(), type=pa.list_(pa.float32(), 8)),
                    'observation.state': pa.array(states.tolist(), type=pa.list_(pa.float32(), 9)),
                    'timestamp': pa.array(np.arange(528, dtype=np.float32)/30),
                    'frame_index': pa.array(np.arange(528)), 'episode_index': pa.array(np.full(528, i)),
                    'index': pa.array(np.arange(528)+528*i), 'task_index': pa.array(np.zeros(528, np.int64)),
                }), ds / 'data' / f'{i:06d}.parquet')
                videos = []
                for key in CAMERAS:
                    video = ds / 'videos' / key / f'{i:06d}.mp4'; video.parent.mkdir(exist_ok=True)
                    video.write_bytes(b'explicit synthetic media boundary')
                    (path / 'media' / f'{key}.json').write_text('{}')
                    videos.append(str(video.relative_to(ds)))
                media = dict(rows=f'data/{i:06d}.parquet', videos=videos, decoded_frames=1056)
                replays = []
                for mode in g.MODES:
                    directory = path / 'replays' / mode; directory.mkdir()
                    for tick in range(528): (directory / f'{tick:06d}.npz').touch()
                    result = dict(mode=mode, passed=True, frames=528, synthetic_boundary=True)
                    write_json(directory / 'result.json', result); replays.append(result)
                report = v.episode_report(i, {}, media, replays, {})
                write_json(path / 'validation.json', report); expected_reports[i] = report
                write_json(path / 'duplicates.json', g.duplicate_report(sequences))
                write_json(path / 'resources.json', dict(wall_seconds=0., process_cpu_seconds=0.,
                    process_maxrss_kib=0, free_bytes=p.FLOOR_BYTES))
                entries.append(dict(episode_index=i, length=528, tasks=[p.TASK], dataset_from_index=i*528,
                                    dataset_to_index=(i+1)*528, **{'data/chunk_index': 0, 'data/file_index': i}))
                events.extend([dict(event='started', **slot, training_ready=False, elapsed_seconds=float(i)),
                               dict(event='admitted', episode_index=i, training_ready=False, elapsed_seconds=float(i))])
                if i == 11: events.append(dict(event='preparation_pass', episode_ids=list(range(12)), training_ready=False, elapsed_seconds=float(i)))
            pq.write_table(pa.Table.from_pylist(entries), meta / 'episodes/episodes.parquet')
            journal = ''.join(json.dumps(e)+'\n' for e in events); (root / 'journal.jsonl').write_text(journal)

            def inputs(root_arg, slot, expert_arg, reset, check, info_arg, entries_arg):
                i = slot['episode_index']; boundary_calls.append(i)
                self.assertEqual(slot, plan['episodes'][i])
                for key in reset: self.assertEqual(reset[key].tobytes(), bank[slot['layout_id']][key].tobytes())
                actions, states, _ = g.read_rows(ds / 'data' / f'{i:06d}.parquet', i)
                return actions, states, {}, {}, expected_reports[i]['media']

            def replay(directory, expert_arg, reset, trace, actions, mode, check):
                return dict(mode=mode, passed=True, frames=528, synthetic_boundary=True)

            guards.enter_context(patch.object(g, '_episode_inputs', side_effect=inputs))
            guards.enter_context(patch.object(g, 'audit_replay', side_effect=replay))
            guards.enter_context(patch.object(v, 'diagnostics', return_value={}))
            # The terminal archive is normally read before diagnostics; the
            # synthetic boundary still writes a valid numeric archive for it.
            for i in range(60):
                with (root / 'attempts' / f'{i:06d}' / 'terminal.npz').open('wb') as stream:
                    np.savez_compressed(stream, final_qpos=np.zeros(37), final_state=np.zeros(242))
            report = v.validate_dataset(root, expert, source, bank, lambda: None)
            self.assertEqual((report['episode_count'], report['frame_count'], report['replay_count'], report['decoded_camera_frames']),
                             (60, 31680, 120, 63360))
            self.assertEqual(report['statistics_episode_ids'], list(range(48)))
            self.assertEqual(boundary_calls, list(range(60)))
            write_json(root / 'validation.json', report); write_json(root / 'seal.json', {'fingerprint': g.fingerprint(root)})
            self.assertEqual(v.require_local_varied_pass(root), report)
            self.assertEqual(boundary_calls, list(range(60))*2)
            for kind in ('count', 'preparation', 'extra_media'):
                if kind == 'count': (meta / 'info.json').write_text(json.dumps({**info, 'total_episodes': 24}))
                elif kind == 'preparation': (root / 'journal.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events if e['event'] != 'preparation_pass'))
                else: (root / 'attempts/000000/media/unexpected.json').write_text('{}')
                with self.subTest(kind=kind, seal='stale'), self.assertRaisesRegex(ValueError, 'stale'):
                    v.require_local_varied_pass(root)
                (root / 'seal.json').write_text(json.dumps({'fingerprint': g.fingerprint(root)}))
                with self.subTest(kind=kind, seal='fresh'), self.assertRaises(ValueError):
                    v.require_local_varied_pass(root)
                if kind == 'count': (meta / 'info.json').write_text(json.dumps(info))
                elif kind == 'preparation': (root / 'journal.jsonl').write_text(journal)
                (root / 'seal.json').write_text(json.dumps({'fingerprint': g.fingerprint(root)}))
            print(f'SYNTHETIC_FULL_ADMISSION root={root} slots=60 rows=31680 replays=120 camera_frames=63360 native_steps=0 expert_calls=0 expensive_boundaries_doubled=3', flush=True)


if __name__ == '__main__': unittest.main()