"""Synthetic failure/evidence regressions; no real expert run or approval."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from experiments.bounded_placement_v3 import diagnostic as d, verification as v
from experiments.bounded_placement_v3.solver import PlanningFailure, finite


def strict_json(path):
    def reject(token):
        raise AssertionError(f'Nonstandard JSON token: {token}')
    return json.loads(path.read_text(), parse_constant=reject)


def test_summary(sources):
    return dict(source_hashes=sources, image=d.IMAGE, actual_expert_invocations=0,
                legacy=dict(count=137, failures=0, errors=0, skipped=0, passed=True),
                new=dict(count=31, failures=0, errors=0, skipped=0, passed=True),
                protected_hashes_unchanged=True, implementation_hashes_unchanged=True)


class NonfiniteEvidenceTests(unittest.TestCase):
    def test_nested_json_marks_all_nonfinite_types_without_mutating_raw_values(self):
        raw = np.array([[np.nan, np.inf, -np.inf, 1.25]])
        value = dict(array=raw, scalar=np.float32(np.inf), zero_dim=np.array(np.nan),
                     nested=({'native': float('-inf'), 'numpy': np.float64(np.nan)},),
                     finite=[np.int64(2), np.bool_(False), None, 'unchanged'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'evidence.json'
            d.save_json(path, value)
            saved = strict_json(path)
            self.assertEqual(saved, dict(
                array=[[{'nonfinite': 'NaN'}, {'nonfinite': '+Infinity'},
                        {'nonfinite': '-Infinity'}, 1.25]],
                scalar={'nonfinite': '+Infinity'}, zero_dim={'nonfinite': 'NaN'},
                nested=[{'native': {'nonfinite': '-Infinity'}, 'numpy': {'nonfinite': 'NaN'}}],
                finite=[2, False, None, 'unchanged']))
            np.testing.assert_array_equal(raw, [[np.nan, np.inf, -np.inf, 1.25]])
            with self.assertRaises(PlanningFailure) as error:
                finite(raw)
            self.assertEqual(error.exception.gate, 'nonfinite')
            with self.assertRaises(FileExistsError):
                d.save_json(path, {})

    def test_failure_json_keeps_original_gate_and_contaminated_journal(self):
        raw = np.array([np.nan, np.inf, -np.inf])
        failure = PlanningFailure('nonfinite', candidate=raw)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            d.save_failure(out, failure, {'terminal_state': raw}, 1)
            saved = strict_json(out/'failure.json')
            self.assertEqual(saved['first_failed_gate'], 'nonfinite')
            self.assertFalse(saved['success'])
            self.assertTrue(saved['no_retry'])
            self.assertEqual(saved['actual_expert_invocations'], 1)
            self.assertEqual(saved['details']['candidate'], saved['plan']['terminal_state'])
            self.assertEqual(saved['plan']['terminal_state'][0], {'nonfinite': 'NaN'})

    def synthetic_execution(self, directory, *, json_failure=False):
        # Actual orchestration and writers, but tiny synthetic state, fake expert
        # and no renderer/physics stepping. All admission patches are temp-only.
        from data_collection.benchmark import capture_state
        out = Path(directory)
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom size=".01"/></body></worldbody></mujoco>')
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        initial = capture_state(model, data)
        np.savez(out/'initial.npz', initial_state=initial, qpos=data.qpos,
                 initial_ctrl=data.ctrl, body_xpos=data.xpos)
        np.savez(out/'trace.npz', initial_state=initial, qpos=data.qpos[None])
        sources = {'synthetic': 'hash'}
        d.save_json(out/'hashes_before.json', {})
        # Both filenames allow the pre-fix writer to reach the target regression.
        d.save_json(out/'tests_passed.json', test_summary(sources))
        d.save_json(out/'tests_passed_reviewed.json', test_summary(sources))
        raw = np.array([np.nan, np.inf, -np.inf])
        result = dict(success=False, reason='nonfinite', final_state=raw.copy(),
                      final_qpos=raw.copy(), final_ctrl=raw.copy(), frames=0,
                      stable_ticks=0, actual_substeps=0, final_metrics={'xy_m': np.nan})
        calls = []

        def run(state, entry, sink, journal):
            calls.append(entry['episode_index'])
            journal.update(terminal_state=raw.copy(), terminal_ctrl=raw.copy(),
                           failure=dict(first_failed_gate='nonfinite', details={'candidate': raw.copy()},
                                        message='nonfinite'))
            return result

        entry = dict(episode_index=0, state_file='initial.npz', file='trace.npz',
                     instruction='synthetic only', seed=1)
        original_save = d.save_json

        def save(path, value):
            if Path(path).name == 'result.json':
                self.assertTrue((Path(path).parent/'terminal_state.npz').is_file(),
                                'Raw terminal evidence must precede JSON serialization')
                if json_failure:
                    raise OSError('synthetic JSON I/O failure')
            original_save(path, value)

        with ExitStack() as stack:
            for name, value in (('OUT', out), ('PAIRED', out)):
                stack.enter_context(patch.object(d, name, value))
            stack.enter_context(patch.object(d, 'require_review', return_value={}))
            stack.enter_context(patch.object(d, 'preflight', return_value=[entry]))
            stack.enter_context(patch.object(d, 'saved_cases', return_value=[entry, entry]))
            stack.enter_context(patch.object(d, 'save_json', side_effect=save))
            stack.enter_context(patch.object(v, 'implementation_hashes', return_value=sources))
            stack.enter_context(patch.object(v, 'protected_hashes', return_value={}))
            stack.enter_context(patch('experiments.bounded_placement_v3.expert.Expert',
                                      return_value=SimpleNamespace(model=model, run=run)))
            stack.enter_context(patch('experiments.bounded_placement_v3.artifacts.TraceSink',
                                      return_value=SimpleNamespace(finish=lambda: None)))
            stack.enter_context(patch('experiments.bounded_placement_v3.replay.read_ticks', return_value=[]))
            stack.enter_context(patch('experiments.bounded_placement_v3.replay.verify_reference',
                                      side_effect=PlanningFailure('reference_nonfinite')))
            with self.assertRaises(PlanningFailure) as error:
                d.execute_reviewed()
            self.assertEqual(error.exception.gate, 'nonfinite')
        self.assertEqual(calls, [0])
        episode = out/'episode_000000'
        with np.load(episode/'terminal_state.npz', allow_pickle=False) as terminal:
            for key in ('final_state', 'final_qpos', 'final_ctrl'):
                np.testing.assert_array_equal(terminal[key], raw)
                self.assertEqual(terminal[key].dtype, np.dtype('float64'))
        for path in (out/'failure.json', episode/'failure.json'):
            saved = strict_json(path)
            self.assertEqual(saved['first_failed_gate'], 'nonfinite')
            self.assertFalse(saved['success'])
        journal = strict_json(episode/'plan_and_events.json')
        self.assertEqual(journal['failure']['first_failed_gate'], 'nonfinite')
        self.assertEqual(journal['terminal_state'][0], {'nonfinite': 'NaN'})
        summary = strict_json(out/'execution_summary.json')
        self.assertEqual(summary['actual_expert_invocations'], 1)  # Fake call only.
        self.assertFalse(summary['all_five_passed'])
        self.assertFalse(summary['training_ready'])
        self.assertFalse((out/'independent_review.json').exists())

    def test_nonfinite_terminal_survives_and_replay_cannot_mask_expert_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            self.synthetic_execution(directory)
            result = strict_json(Path(directory)/'episode_000000/result.json')
            self.assertFalse(result['success'])
            self.assertEqual(result['final_metrics']['xy_m'], {'nonfinite': 'NaN'})

    def test_raw_terminal_and_original_gate_survive_result_json_io_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            self.synthetic_execution(directory, json_failure=True)


class ReviewedTestAdmissionTests(unittest.TestCase):
    def test_historical_evidence_alone_cannot_admit_current_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            d.save_json(out/'tests_passed.json', test_summary({'source': 'current'}))
            with self.assertRaises(PlanningFailure) as error:
                v.require_test_evidence(out, {'source': 'current'})
            self.assertEqual(error.exception.gate, 'current_cpu_test_evidence_required')

    def test_reviewed_evidence_requires_current_hashes_and_passing_suites(self):
        sources = {'source': 'current'}
        for change in ('none', 'stale', 'legacy_failure', 'new_skip', 'changed_protected'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                out = Path(directory); summary = test_summary(sources)
                if change == 'stale': summary['source_hashes'] = {'source': 'old'}
                if change == 'legacy_failure': summary['legacy']['passed'] = False
                if change == 'new_skip': summary['new']['skipped'] = 1
                if change == 'changed_protected': summary['protected_hashes_unchanged'] = False
                d.save_json(out/'tests_passed_reviewed.json', summary)
                if change == 'none':
                    v.require_test_evidence(out, sources)
                else:
                    with self.assertRaises(PlanningFailure):
                        v.require_test_evidence(out, sources)

    def test_missing_independent_review_still_blocks_before_marker(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(d, 'OUT', Path(directory)):
            with self.assertRaises(PlanningFailure) as error:
                d.execute_reviewed()
            self.assertEqual(error.exception.gate, 'independent_review_required')
            self.assertFalse((Path(directory)/'execution_started.json').exists())


if __name__ == '__main__':
    unittest.main()