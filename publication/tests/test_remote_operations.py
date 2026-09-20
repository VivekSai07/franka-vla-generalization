"""Offline fake-service security tests; never use credentials or a network."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from publication import publish, tracking, registry


class HTTPFailure(Exception):
    def __init__(self, status=407):
        super().__init__('SECRET https://untrusted.invalid/?signature=SECRET')
        self.response = NS(status_code=status)


class FakeHF:
    def __init__(self, files):
        self.files = files
        self.exists = False
        self.private = True
        self.identity = 'VivekSai07'
        self.calls = []
        self.oid = 'a' * 40
        self.fail = None
        self.corrupt = False
        self.extra = False
        self.committed = False

    def whoami(self):
        self.calls.append('whoami')
        return {'name': self.identity}

    def model_info(self, repo_id, **kwargs):
        if not self.exists:
            raise HTTPFailure(404)
        siblings = []
        for name, data in (self.files.items() if self.committed else []):
            lfs = NS(sha256=hashlib.sha256(data).hexdigest(), size=len(data)) if name.endswith('.safetensors') else None
            if self.corrupt and lfs:
                lfs.sha256 = '0' * 64
            siblings.append(NS(rfilename=name, size=len(data), lfs=lfs,
                               blob_id=hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()))
        if self.extra and self.committed:
            siblings.append(NS(rfilename='unreviewed.txt', size=1, lfs=None, blob_id='a' * 40))
        return NS(private=self.private, sha=self.oid, siblings=siblings)

    def create_repo(self, repo_id, **kwargs):
        assert kwargs['private'] is True and kwargs['exist_ok'] is False
        self.calls.append('create')
        self.exists = True

    def preupload_lfs_files(self, repo_id, **kwargs):
        self.calls.append('preupload')
        if self.fail:
            print('SECRET noisy SDK')
            raise self.fail

    def create_commit(self, repo_id, operations, **kwargs):
        self.calls.append('commit')
        self.operations = list(operations)
        self.committed = True
        return NS(oid=self.oid)


class HFTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.root = self.repo / publish.RELEASE_RELATIVE / 'packages/smolvla'
        self.root.mkdir(parents=True)
        self.files = {'model.safetensors': b'weights', 'README.md': b'reviewed'}
        for name, data in self.files.items():
            (self.root / name).write_bytes(data)
        self.sdk = FakeHF(dict(self.files))

    def invoke(self):
        report = {}
        with patch.object(publish, 'reviewed_files', return_value={
            name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            for name, data in self.files.items()
        }), patch.object(publish, 'operation_add', side_effect=lambda name, path: NS(path_in_repo=name, path_or_fileobj=path)):
            publish.publish_model(self.repo, 'smolvla', self.sdk,
                                  lambda name, revision: self.files[name], report)
        return report

    def test_identity_blocks_creation(self):
        self.sdk.identity = 'someone-else'
        result = self.invoke()
        self.assertNotEqual(result['status'], 'verified')
        self.assertNotIn('create', self.sdk.calls)

    def test_any_existing_repo_including_empty_is_refused(self):
        for files in ({}, self.files):
            self.sdk.files = files
            self.sdk.exists = True
            self.assertNotEqual(self.invoke()['status'], 'verified')
            self.assertNotIn('commit', self.sdk.calls)

    def test_private_creation_and_every_file_verified(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['commit_sha'], self.sdk.oid)
        self.assertEqual(set(result['files']), set(self.files))
        self.assertEqual({op.path_in_repo for op in self.sdk.operations}, set(self.files))

    def test_visibility_failure_blocks_upload(self):
        self.sdk.private = False
        self.assertNotEqual(self.invoke()['status'], 'verified')
        self.assertNotIn('preupload', self.sdk.calls)

    def test_corrupt_lfs_and_extra_files_rejected(self):
        for field in ('corrupt', 'extra'):
            self.sdk.exists = False
            self.sdk.committed = False
            setattr(self.sdk, field, True)
            self.assertNotEqual(self.invoke()['status'], 'verified')
            self.assertIn('commit', self.sdk.calls)
            setattr(self.sdk, field, False)

    def test_407_keeps_created_state_without_leaking_error(self):
        self.sdk.fail = HTTPFailure()
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = self.invoke()
        self.assertEqual(result['status'], 'repo_created_no_modelcommit')
        self.assertEqual(result['phase'], 'lfs_preupload')
        self.assertEqual(result['error']['http_status'], 407)
        self.assertNotIn('commit', self.sdk.calls)
        self.assertNotIn('SECRET', json.dumps(result) + stdout.getvalue() + stderr.getvalue())

    def test_non_lfs_blob_download_mismatch_rejected(self):
        self.sdk.files['README.md'] = b'changed'
        self.assertNotEqual(self.invoke()['status'], 'verified')
        self.assertIn('commit', self.sdk.calls)

    def test_broad_parent_or_symlink_release_rejected(self):
        for root in (self.root.parent, self.repo, self.root / '..'):
            with self.assertRaises(ValueError):
                publish.validate_release_path(self.repo, 'smolvla', root)
        alias = self.repo / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            publish.validate_release_path(self.repo, 'smolvla', alias)

    def test_error_sanitization(self):
        self.assertEqual(publish.safe_error(HTTPFailure()), {'type': 'HTTPFailure', 'http_status': 407})

    def test_fixed_cli_has_no_key_or_path_override(self):
        parser = publish.parser()
        self.assertTrue(parser.parse_args(['--hf', '--wandb', '--registry']).hf)
        flags = {flag for action in parser._actions for flag in action.option_strings}
        self.assertFalse(flags & {'--token', '--api-key', '--repo', '--package', '--endpoint'})


class FakeControl:
    def __init__(self):
        self.identity = {'username': 'authenticated-user',
                         'entity': 'vivekchaitanya07-university-of-stuttgart'}
        self.visibility = None
        self.created = False
        self.duplicate = False
        self.schema_ok = True
        self.calls = []

    def viewer(self):
        return self.identity

    def project_visibility(self, entity):
        self.calls.append('privacy')
        return self.visibility

    def create_private_project(self, entity):
        if not self.schema_ok:
            raise ValueError('privacy schema unavailable')
        self.created = True
        self.visibility = 'PRIVATE'

    def run_exists(self, entity, run_id):
        return self.duplicate


class FakeRun:
    def __init__(self):
        self.logs = []
        self.summary = {}
        self.finished = False

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, data, step):
        self.logs.append((step, data))

    def finish(self, exit_code=0):
        self.finished = exit_code == 0


class FakeWandb:
    def __init__(self):
        self.runs = []

    def Settings(self, **kwargs):
        return kwargs

    def Table(self, columns, data):
        return {'columns': columns, 'data': data}

    def init(self, **kwargs):
        self.kwargs = kwargs
        run = FakeRun()
        self.runs.append(run)
        return run


def historical_row():
    return {
        'model': 'smolvla', 'source_hash': 'b' * 64,
        'config': {'logging_mode': 'historical_import', 'completed_updates': 5000},
        'summary': {'successes': 18, 'failures': 2, 'gpu_utilization': None, 'gpu_power_watts': None},
        'trials': [{'ordinal': n, 'task_success': n not in (6, 16)} for n in range(1, 21)],
        'history': [{'update': n, 'train/loss': 1 / n} for n in range(1, 5001)],
    }


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.control = FakeControl()
        self.sdk = FakeWandb()
        self.row = historical_row()

    def invoke(self, requested_entity=None, api_key='dummy-key'):
        report = {}
        tracking.import_history(self.row, self.control, self.sdk, api_key, requested_entity, report)
        return report

    def test_approved_default_entity_allows_different_authenticated_username(self):
        for requested in (None, 'vivekchaitanya07-university-of-stuttgart'):
            with self.subTest(requested=requested):
                result = self.invoke(requested_entity=requested)
                self.assertEqual(result['status'], 'imported')
                self.assertEqual(result['entity'], 'vivekchaitanya07-university-of-stuttgart')
                self.assertEqual(self.sdk.kwargs['entity'], 'vivekchaitanya07-university-of-stuttgart')

    def test_unapproved_personal_entity_blocks_before_project_access(self):
        self.control.identity = {'username': 'personal', 'entity': 'personal'}
        result = self.invoke()
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['phase'], 'approved_entity_check')
        self.assertEqual(self.control.calls, [])
        self.assertFalse(self.control.created)
        self.assertEqual(self.sdk.runs, [])

    def test_requested_entity_cannot_override_approved_default(self):
        for requested in ('personal', 'other-team', '', 'vivekchaitanya07-university-of-stuttgart-extra'):
            with self.subTest(requested=requested):
                result = self.invoke(requested_entity=requested)
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual(result['phase'], 'approved_entity_check')
                self.assertEqual(self.control.calls, [])
                self.assertFalse(self.control.created)
                self.assertEqual(self.sdk.runs, [])

    def test_missing_authenticated_viewer_blocks_before_project_access(self):
        approved = 'vivekchaitanya07-university-of-stuttgart'
        for identity in (None, {}, {'entity': approved}, {'username': None, 'entity': approved},
                         {'username': '', 'entity': approved}, {'username': '   ', 'entity': approved}):
            with self.subTest(identity=identity):
                self.control.identity = identity
                result = self.invoke()
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual(result['error']['type'], 'ValueError')
                self.assertEqual(self.control.calls, [])
                self.assertFalse(self.control.created)
                self.assertEqual(self.sdk.runs, [])

    def test_null_or_empty_key_blocks_before_viewer_or_mutation(self):
        for key in (None, '', '   '):
            with self.subTest(key=key), patch.object(self.control, 'viewer') as viewer:
                result = self.invoke(api_key=key)
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual(result['phase'], 'credential_check')
                viewer.assert_not_called()
                self.assertEqual(self.control.calls, [])
                self.assertFalse(self.control.created)
                self.assertEqual(self.sdk.runs, [])

    def test_private_project_before_init_and_exact_historical_steps(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'imported')
        self.assertTrue(self.control.created)
        self.assertEqual(self.sdk.kwargs['resume'], 'never')
        self.assertEqual([step for step, _ in self.sdk.runs[0].logs], list(range(1, 5001)))
        self.assertEqual(self.sdk.runs[0].summary['failures'], 2)
        self.assertEqual(len(self.sdk.runs[0].summary['trials']['data']), 20)
        settings = self.sdk.kwargs['settings']
        for name in ('disable_git', 'disable_code', 'x_disable_stats', 'x_disable_meta', 'x_disable_machine_info'):
            self.assertIs(settings[name], True)
        self.assertEqual(settings['console'], 'off')
        self.assertIs(settings['save_code'], False)

    def test_public_project_never_mutated(self):
        self.control.visibility = 'PUBLIC'
        self.assertNotEqual(self.invoke()['status'], 'imported')
        self.assertFalse(self.control.created)
        self.assertEqual(self.sdk.runs, [])

    def test_unverifiable_privacy_blocks_init(self):
        self.control.schema_ok = False
        self.assertNotEqual(self.invoke()['status'], 'imported')
        self.assertEqual(self.sdk.runs, [])

    def test_other_default_team_blocks_even_when_approved_entity_requested(self):
        self.control.identity['entity'] = 'team'
        for requested in (None, 'vivekchaitanya07-university-of-stuttgart'):
            result = self.invoke(requested_entity=requested)
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(result['phase'], 'approved_entity_check')
        self.assertEqual(self.control.calls, [])
        self.assertFalse(self.control.created)
        self.assertEqual(self.sdk.runs, [])

    def test_duplicate_id_blocks_even_private_project(self):
        self.control.visibility = 'PRIVATE'
        self.control.duplicate = True
        self.assertNotEqual(self.invoke()['status'], 'imported')
        self.assertEqual(self.sdk.runs, [])

    def test_ids_repeat_only_for_same_model_and_source_hash(self):
        a = tracking.run_id(self.row)
        self.assertEqual(a, tracking.run_id(dict(self.row)))
        self.assertNotEqual(a, tracking.run_id(dict(self.row, model='xvla')))
        self.assertNotEqual(a, tracking.run_id(dict(self.row, source_hash='c' * 64)))

    def test_secret_exception_is_quiet_and_reported(self):
        self.control.viewer = lambda: (_ for _ in ()).throw(HTTPFailure())
        result = self.invoke()
        self.assertNotIn('SECRET', json.dumps(result))
        self.assertEqual(result['error']['http_status'], 407)

    def test_noncontiguous_history_blocks_init(self):
        self.row['history'].pop(20)
        self.assertNotEqual(self.invoke()['status'], 'imported')
        self.assertEqual(self.sdk.runs, [])


class RegistryTests(unittest.TestCase):
    def test_metrics_are_numeric_allowlist_not_sampled_rows(self):
        raw = dict(update=1, loss=0.5, seconds=1, sampled_chunks=8,
                   grad_norm=2, learning_rates={'vlm': 1e-6, 'secret': 'SECRET'},
                   memory={'allocated': 123}, dataset_rows=[[1]], env='SECRET')
        record = registry.metric_record(raw)
        self.assertEqual(record['update'], 1)
        self.assertEqual(record['train/lr/vlm'], 1e-6)
        self.assertNotIn('SECRET', json.dumps(record))
        self.assertNotIn('dataset_rows', record)

    def test_nan_metrics_are_rejected(self):
        with self.assertRaises(ValueError):
            registry.metric_record({'update': 1, 'loss': float('nan')})

    def test_exact_two_real_readonly_histories(self):
        repo = Path(__file__).resolve().parents[2]
        rows = registry.build_registry(repo)
        self.assertEqual([r['model'] for r in rows], ['smolvla', 'xvla'])
        for row, successes in zip(rows, (18, 20)):
            self.assertEqual(len(row['history']), 5000)
            self.assertEqual(len(row['trials']), 20)
            self.assertEqual(row['summary']['successes'], successes)
            self.assertEqual(sum(t['task_success'] for t in row['trials']), successes)
            self.assertEqual(row['config']['logging_mode'], 'historical_import')
            self.assertIsNone(row['summary']['gpu_power_watts'])
            self.assertIsNone(row['summary']['gpu_utilization'])
            self.assertTrue(row['source_files'])
            text = json.dumps(row)
            self.assertNotIn('/vla/', text)
            self.assertNotIn('binding', row['config'])
            self.assertNotIn('pretrained_path', row['config'])
        self.assertEqual(rows[0]['config']['trainable_scope'], 'frozen_vlm_expert_and_state_projection')
        self.assertEqual(rows[1]['config']['trainable_scope'], 'all_parameters')
        self.assertEqual(rows[0]['summary']['sampled_chunks'], 160000)
        self.assertEqual(rows[1]['summary']['sampled_chunks'], 40000)
        self.assertEqual(rows[0]['config']['objective'], 'stock_flow_matching')
        self.assertEqual(rows[1]['config']['objective'], 'native_MSE')
        self.assertEqual(rows[0]['config']['original_finish'], '2026-09-18T19:32:07Z')
        self.assertEqual(rows[1]['config']['original_finish'], '2026-09-18T16:56:38Z')


if __name__ == '__main__':
    unittest.main()