"""Synthetic model/hardware boundaries, real scheduler, bank, files and comparers."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from experiments.local_position_probe import contract as c, runner as r, scenes as s, telemetry as t

EVIDENCE = c.ROOT/'outputs/implementation/local_position_probe_20260919'


def folder():
    # Preserve evidence, including failed attempts: no cleanup/deletion.
    return Path(tempfile.mkdtemp(prefix='synthetic-', dir=EVIDENCE))


def baseline(model, ordinal):
    return c.ROOT/f'outputs/train/{model}_fixed_layout_20260918/evaluation/trial_{ordinal:02d}_seed_{20260917+ordinal}'


class ComparisonTests(unittest.TestCase):
    def test_real_four_baseline_controls_self_match_including_known_failure(self):
        for model in ('smolvla', 'xvla'):
            for ordinal in (1, 6):
                b = baseline(model, ordinal); row = json.loads((b/'episode.json').read_text())
                self.assertTrue(r.compare_control(row, b/'trace.npz', b)['matched'])
                self.assertEqual(row['task_success'], not (model == 'smolvla' and ordinal == 6))

    def test_every_array_mutation_dtype_shape_key_and_signed_zero_detected(self):
        b = baseline('smolvla', 1); row = json.loads((b/'episode.json').read_text()); dest = folder()/'trace.npz'
        with np.load(b/'trace.npz', allow_pickle=False) as z: original = {k: z[k] for k in z.files}
        for key in original:
            arrays = {k: v.copy() for k, v in original.items()}
            a = arrays[key]
            if a.dtype.kind == 'b': a.flat[0] = not a.flat[0]
            else: a.flat[0] = np.nextafter(a.flat[0], np.inf, dtype=a.dtype)
            np.savez(dest, **arrays)
            with self.subTest(key=key), self.assertRaises(ValueError): r.compare_control(row, dest, b)
        for arrays in ({**original, 'extra': np.array(1)}, {k: v for k, v in original.items() if k != 'state'},
                       {**original, 'state': original['state'].astype('f4')},
                       {**original, 'state': original['state'].reshape(-1)},
                       {**original, 'max_joint_clip_rad': -original['max_joint_clip_rad']}):
            np.savez(dest, **arrays)
            with self.assertRaises(ValueError): r.compare_control(row, dest, b)

    def test_each_semantic_field_checked_only_wall_and_artifact_ids_ignored(self):
        b = baseline('smolvla', 6); row = json.loads((b/'episode.json').read_text())
        for key in r.SEMANTIC_FIELDS:
            bad = copy.deepcopy(row); bad[key] = 'corrupted'
            with self.subTest(key=key), self.assertRaises(ValueError): r.compare_control(bad, b/'trace.npz', b)
        for key in r.ARTIFACT_FIELDS:
            changed = copy.deepcopy(row); changed[key] = 1. if key == 'wall_seconds' else 'new-artifact'
            self.assertTrue(r.compare_control(changed, b/'trace.npz', b)['matched'])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.output = folder()/'job'
        self.calls, self.loads, self.releases = [], [], []
        self.failure = None
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        def use(target, **kwargs): return self.stack.enter_context(patch(target, **kwargs))
        use('experiments.local_position_probe.contract.OUTPUT', new=self.output)
        use('experiments.local_position_probe.runner.preflight', return_value={'status': 'passed', 'snapshot': {'fixture': 'fixed'}})
        use('experiments.local_position_probe.runner.source_snapshot', return_value={'fixture': 'fixed'})
        use('experiments.local_position_probe.runner.require_disk', return_value=30*1024**3)
        use('experiments.local_position_probe.runner.require_gpu', return_value={'free_bytes': 16*1024**3})
        use('experiments.local_position_probe.runner.load_policy', side_effect=self.load)
        use('experiments.local_position_probe.runner.release_policy', side_effect=lambda bundle: self.releases.append(bundle[0]))
        use('experiments.local_position_probe.runner.invoke_trial', side_effect=self.trial)
        use('experiments.local_position_probe.telemetry.query', return_value=[])
        use('mujoco.mj_step', side_effect=AssertionError('Never step'))
        use('mujoco.Renderer', side_effect=AssertionError('Never render'))

    def load(self, model):
        self.assertEqual(len(list((self.output/'bank').glob('*.npz'))), 17)
        self.assertTrue((self.output/'plan.json').is_file())
        self.loads.append(model)
        return (model, None, None)

    def trial(self, model_name, bundle, model, manifest, entry, bank, output, ordinal, seed):
        self.calls.append((model_name, entry.copy(), bank, ordinal, seed))
        self.assertEqual(entry['episode_index'], 7)
        self.assertEqual(ordinal, {20260918: 1, 20260923: 6}[seed])
        if self.failure == ('exception', len(self.calls)): raise RuntimeError('synthetic infrastructure failure')
        b = baseline(model_name, ordinal)
        row = json.loads((b/'episode.json').read_text()); row.update(entry)
        row['reset_sha256'] = entry['sha256']; row['wall_seconds'] = .01
        with np.load(b/'trace.npz', allow_pickle=False) as z: arrays = {k: z[k] for k in z.files}
        with np.load(bank/entry['file'], allow_pickle=False) as z: arrays['initial_integration_state'] = z['initial_state']
        if self.failure == ('mismatch', len(self.calls)): arrays['raw_action'].flat[0] += .01
        if self.failure == ('nonfinite', len(self.calls)): arrays['raw_action'].flat[0] = np.nan
        if self.failure == ('seed', len(self.calls)): row['seed'] += 1
        np.savez_compressed(output/'trace.npz', **arrays)
        (output/'episode.json').write_text(json.dumps(row))
        return row

    def test_exact_68_controls_first_load_once_shared_17_bank(self):
        result = r.run()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['finished_trials'], 68)
        self.assertEqual(self.loads, ['smolvla', 'xvla']); self.assertEqual(self.releases, self.loads)
        self.assertEqual(len(self.calls), 68)
        for first, second in zip(self.calls[:34], self.calls[34:]): self.assertEqual(first[1:], second[1:])
        journal = [json.loads(line) for line in (self.output/'journal.jsonl').read_text().splitlines()]
        self.assertEqual([v['event'] for v in journal], ['trial_start', 'trial_end']*68)
        self.assertEqual(result['per_model']['smolvla']['controls']['successes'], 1)
        self.assertEqual(result['per_model']['smolvla']['perturbations']['planned'], 32)
        self.assertFalse(any(thread.name == 'local-position-telemetry' for thread in threading.enumerate()))

    def test_reproduction_failure_stops_before_any_perturbation(self):
        self.failure = ('mismatch', 2)
        with self.assertRaises(ValueError): r.run()
        self.assertEqual(len(self.calls), 2); self.assertEqual(self.loads, ['smolvla'])
        self.assertEqual(json.loads((self.output/'result.json').read_text())['status'], 'failed')
        with self.assertRaises(FileExistsError): r.run()
        self.assertEqual(len(self.calls), 2)

    def test_nonfinite_stops_without_next_call(self):
        self.failure = ('nonfinite', 3)
        with self.assertRaises(ValueError): r.run()
        self.assertEqual(len(self.calls), 3)

    def test_infrastructure_failure_preserves_start_no_end_and_stops(self):
        self.failure = ('exception', 4)
        with self.assertRaises(RuntimeError): r.run()
        events = [json.loads(line)['event'] for line in (self.output/'journal.jsonl').read_text().splitlines()]
        self.assertEqual(events, ['trial_start', 'trial_end']*3+['trial_start'])
        self.assertEqual(len(self.calls), 4); self.assertEqual(self.releases, ['smolvla'])

    def test_seed_failure_stops(self):
        self.failure = ('seed', 1)
        with self.assertRaises(ValueError): r.run()
        self.assertEqual(len(self.calls), 1)

    def test_existing_output_not_resumed(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError): r.run()
        self.assertEqual(self.loads, [])

    def test_source_drift_marks_failure_even_after_last_trial(self):
        with patch.object(r, 'source_snapshot', side_effect=[{'fixture': 'fixed'}, {'fixture': 'changed'}]):
            with self.assertRaises(ValueError): r.run()
        self.assertEqual(json.loads((self.output/'result.json').read_text())['status'], 'failed')

    def test_runtime_matches_each_baseline_before_loading(self):
        import os
        def checked_load(model):
            self.assertEqual(r.torch.get_num_threads(), 8 if model == 'smolvla' else 4)
            self.assertFalse(r.torch.backends.cudnn.benchmark)
            self.assertEqual(os.environ.get('CUBLAS_WORKSPACE_CONFIG'), ':4096:8')
            return self.load(model)
        with patch.object(r, 'load_policy', side_effect=checked_load): r.run()

    def test_second_model_control_failure_does_not_start_its_perturbations(self):
        self.failure = ('mismatch', 36)
        with self.assertRaises(ValueError): r.run()
        self.assertEqual(len(self.calls), 36)
        self.assertEqual(self.loads, ['smolvla', 'xvla'])
        self.assertEqual(self.releases, self.loads)

    def test_final_bank_mutation_prevents_complete(self):
        def mutate(*args):
            row = self.trial(*args)
            if len(self.calls) == 68:
                path = self.output/'bank/control.npz'
                with path.open('ab') as stream: stream.write(b'changed-after-use')
            return row
        with patch.object(r, 'invoke_trial', side_effect=mutate), self.assertRaises(ValueError): r.run()
        self.assertEqual(json.loads((self.output/'result.json').read_text())['status'], 'failed')

    def test_trial_wall_overrun_cannot_complete_even_if_alarm_dispatch_is_delayed(self):
        # Native CUDA/encoder calls may postpone Python signal delivery. Check wall time after return too.
        original = r.time.monotonic
        def clock(): return original()+(14401 if len(self.calls) == 68 else 0)
        with patch.object(r.time, 'monotonic', side_effect=clock), self.assertRaises(TimeoutError): r.run()
        self.assertEqual(json.loads((self.output/'result.json').read_text())['status'], 'failed')


class GateAndSummaryTests(unittest.TestCase):
    def test_partial_duplicate_and_extra_rows_never_complete(self):
        rows = []
        for slot in c.make_plan()['trials']:
            result = json.loads((baseline(slot['model'], slot['baseline_ordinal'])/'episode.json').read_text())
            rows.append({**slot, 'result': result})
        self.assertTrue(r.summarize(rows)['all_trials_valid'])
        for invalid in (rows[:-1], rows+[rows[0]], rows[:1]*68, list(reversed(rows))):
            self.assertFalse(r.summarize(invalid)['all_trials_valid'])
        summary = r.summarize(rows[:-1])
        self.assertIsNone(summary['per_model']['xvla']['perturbations']['success_rate'])
        self.assertEqual(len(r.summarize(rows)['paired_outcomes']), 34)

    def test_disk_and_gpu_floors_and_bounded_query(self):
        with patch.object(r.shutil, 'disk_usage', return_value=type('D', (), {'free': 20*1024**3-1})()):
            with self.assertRaises(ValueError): r.require_disk(c.ROOT)
        with patch.object(r.torch.cuda, 'is_available', return_value=True), patch.object(r.torch.cuda, 'mem_get_info', return_value=(12*1024**3-1, 48*1024**3)):
            with self.assertRaises(ValueError): r.require_gpu()

    def test_claim_rejects_symlink_and_nonmount_existing_directory(self):
        root = folder(); target = root/'existing'; target.mkdir(); link = root/'link'; link.symlink_to(target)
        for path in (target, link):
            with self.assertRaises((ValueError, FileExistsError)): r.claim_output(path)

    def test_wall_guard_restores_alarm_on_failure(self):
        with patch.object(r.signal, 'signal') as handler, patch.object(r.signal, 'setitimer') as timer:
            with self.assertRaises(RuntimeError):
                with r.wall_guard(): raise RuntimeError('stop')
            self.assertEqual(timer.call_args_list[0].args[1], 14400)
            self.assertEqual(timer.call_args_list[-1].args[1], 0)
            self.assertEqual(handler.call_count, 2)

    def test_summary_rejects_wrong_result_seed_even_with_correct_outer_slot(self):
        rows = []
        for slot in c.make_plan()['trials']:
            result = json.loads((baseline(slot['model'], slot['baseline_ordinal'])/'episode.json').read_text())
            rows.append({**slot, 'result': result})
        rows[-1]['result']['seed'] = 123
        self.assertFalse(r.summarize(rows)['all_trials_valid'])

    def test_preflight_nonstepping_and_readonly_and_no_model_loading(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(r, 'source_snapshot', return_value={'package_sources': c.package_hashes()}))
            stack.enter_context(patch.object(Path, 'mkdir', side_effect=AssertionError('No mkdir in check')))
            stack.enter_context(patch.object(r, 'load_policy', side_effect=AssertionError('No model in check')))
            for name in ('mj_step', 'mj_step1', 'mj_step2', 'Renderer'):
                stack.enter_context(patch.object(s.mujoco, name, side_effect=AssertionError('No stepping/rendering')))
            original_open = Path.open
            def readonly(path, mode='r', *args, **kwargs):
                if any(flag in mode for flag in 'wax+'): raise AssertionError('No write in preflight')
                return original_open(path, mode, *args, **kwargs)
            stack.enter_context(patch.object(Path, 'open', readonly))
            proof = r.preflight()
        self.assertEqual(proof['bank_conditions'], 17)
        self.assertEqual(proof['physics_steps'], 0)
        self.assertEqual(proof['policy_loads'], 0)


class TelemetryTests(unittest.TestCase):
    def test_sampler_uses_five_second_cadence_not_query_duration_plus_five(self):
        class StopAfterWait:
            stopped = False
            waits = []
            def is_set(self): return self.stopped
            def wait(self, seconds):
                self.waits.append(seconds); self.stopped = True
                return True
        sampler = t.Sampler(folder()/'cadence.jsonl', lambda: 'loading')
        sampler.stop = StopAfterWait()
        with patch.object(t, 'query', return_value=[]), patch.object(t.time, 'monotonic', side_effect=[100., 100.5, 101.]):
            sampler._loop()
        self.assertEqual(sampler.stop.waits, [4.])

    def test_parse_devicewide_missing_and_nonfinite_fields(self):
        rows = t.parse('GPU-test, RTX test, 12, 1024, 49140, [N/A], nan\n')
        self.assertEqual(rows[0]['uuid'], 'GPU-test')
        self.assertEqual(rows[0]['memory_used_mib'], 1024.)
        self.assertIsNone(rows[0]['power_w']); self.assertIsNone(rows[0]['temperature_c'])
        self.assertEqual(set(rows[0]['missing']), {'power_w', 'temperature_c'})
        with self.assertRaises(ValueError): t.parse('bad,row')

    def test_subprocess_is_fixed_bounded_no_shell_or_environment_capture(self):
        with patch.object(t.subprocess, 'run', return_value=type('P', (), {'stdout': 'GPU-a, RTX, 0, 0, 49140, 10, 30\n'})()) as run:
            self.assertEqual(len(t.query()), 1)
        args, kwargs = run.call_args
        self.assertEqual(args[0][0], 'nvidia-smi')
        self.assertEqual(kwargs['timeout'], 3)
        self.assertNotIn('shell', kwargs); self.assertNotIn('env', kwargs)
        self.assertNotIn('process', ' '.join(args[0]))

    def test_sampler_failure_nonfatal_recorded_and_thread_joined_rng_unchanged(self):
        import random
        import torch
        root = folder(); py = random.getstate(); npstate = np.random.get_state(); ts = torch.get_rng_state().clone()
        with patch.object(t, 'query', side_effect=subprocess.TimeoutExpired('nvidia-smi', 3)):
            with t.Sampler(root/'telemetry.jsonl', lambda: 'control') as sampler:
                sampler.sample()
        self.assertFalse(sampler.thread.is_alive())
        rows = [json.loads(line) for line in (root/'telemetry.jsonl').read_text().splitlines()]
        self.assertTrue(rows); self.assertTrue(all(v['error'] == 'TimeoutExpired' for v in rows))
        self.assertTrue(all(v['phase'] == 'control' and v['scope'] == 'devicewide_shared_gpu' for v in rows))
        self.assertEqual(random.getstate(), py)
        self.assertTrue(np.array_equal(np.random.get_state()[1], npstate[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), ts))


if __name__ == '__main__': unittest.main()