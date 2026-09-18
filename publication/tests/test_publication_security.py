"""Offline boundary tests with synthetic values, not real credential files."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from publication import publish, tracking


class SecurityTests(unittest.TestCase):
    def test_null_credential_rejected_without_ambient_fallback(self):
        with patch.dict(os.environ, {'WANDB_API_KEY': 'dummy-from-env'}), \
                patch('dotenv.dotenv_values', return_value={'WANDB_API_KEY': None}), \
                patch.object(Path, 'is_symlink', return_value=False), patch.object(Path, 'is_file', return_value=True):
            with self.assertRaises(ValueError):
                publish.credentials(['wandb'])

    def test_control_rejects_null_or_empty_key_before_creating_session(self):
        for key in (None, '', '   '):
            with self.subTest(key=key), patch.object(tracking, 'bounded_session') as session:
                with self.assertRaises(ValueError):
                    tracking.WandbControl(key)
                session.assert_not_called()

    def test_credentials_use_only_requested_keys_without_interpolation(self):
        with patch('dotenv.dotenv_values', return_value={
            'HF_TOKEN': 'dummy-hf', 'WANDB_API_KEY': 'dummy-wb', 'WANDB_ENTITY': 'personal',
            'UNRELATED_SECRET': 'not-selected',
        }) as parse, patch.object(Path, 'is_symlink', return_value=False), patch.object(Path, 'is_file', return_value=True):
            selected = publish.credentials(['hf'])
        self.assertEqual(selected, {'HF_TOKEN': 'dummy-hf'})
        parse.assert_called_once_with(Path('/run/secrets/publishing.env'), interpolate=False)

    def test_environment_key_is_never_a_credential_fallback(self):
        with patch.dict(os.environ, {'HF_TOKEN': 'dummy-from-env'}), patch('dotenv.dotenv_values', return_value={}), \
                patch.object(Path, 'is_symlink', return_value=False), patch.object(Path, 'is_file', return_value=True):
            with self.assertRaises(ValueError):
                publish.credentials(['hf'])

    def test_isolation_drops_inherited_capture_and_service_config(self):
        cwd = Path.cwd()
        with patch.dict(os.environ, {'WANDB_API_KEY': 'fake', 'AWS_SECRET_ACCESS_KEY': 'fake',
                                     'WANDB_BASE_URL': 'https://evil.invalid', 'HF_ENDPOINT': 'https://evil.invalid'}):
            with publish.private_runtime():
                self.assertNotIn('WANDB_API_KEY', os.environ)
                self.assertNotIn('AWS_SECRET_ACCESS_KEY', os.environ)
                self.assertEqual(os.environ['WANDB_BASE_URL'], publish.WB_ENDPOINT)
                self.assertEqual(os.environ['HF_ENDPOINT'], publish.HF_ENDPOINT)
                home = Path(os.environ['HOME'])
                self.assertEqual(home.stat().st_mode & 0o777, 0o700)
                self.assertNotEqual(Path.cwd(), cwd)
            self.assertFalse(home.exists())
        self.assertEqual(Path.cwd(), cwd)

    def test_quiet_discards_python_and_logging_output(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with publish.quiet():
                print('dummy-secret')
                __import__('logging').critical('dummy-secret')
        self.assertEqual(out.getvalue(), '')

    def test_transport_rejects_nonservice_urls_before_network(self):
        import requests
        session = publish.bounded_session('hf')
        for url in ('https://evil.invalid', 'http://huggingface.co', 'https://huggingface.co:444',
                    'https://user:password@huggingface.co'):
            with self.assertRaises(ValueError):
                session.send(requests.Request('GET', url).prepare())

    def test_proxy_407_is_not_a_retryable_connection_error(self):
        import requests
        session = publish.bounded_session('hf')
        request = requests.Request('GET', publish.HF_ENDPOINT).prepare()
        with patch('requests.Session.send', side_effect=requests.exceptions.ProxyError('Tunnel failed: 407 dummy-secret')):
            with self.assertRaises(Exception) as caught:
                session.send(request)
        self.assertNotIsInstance(caught.exception, requests.exceptions.ConnectionError)
        self.assertEqual(publish.safe_error(caught.exception)['http_status'], 407)
        self.assertNotIn('dummy-secret', str(caught.exception))

    def test_transport_bounds_timeout_and_does_not_forward_hf_key_to_storage(self):
        import requests
        session = publish.bounded_session('hf')
        request = requests.Request('GET', 'https://cdn-lfs.hf.co/blob', headers={'Authorization': 'Bearer dummy'}).prepare()
        with patch('requests.Session.send', return_value=object()) as send:
            session.send(request, timeout=None)
        self.assertNotIn('Authorization', request.headers)
        self.assertEqual(send.call_args.kwargs['timeout'], (10, 60))

    def test_installed_wandb_0242_settings_accept_all_capture_flags(self):
        import wandb
        with publish.quiet():
            settings = wandb.Settings(**tracking.historical_settings('dummy-key'))
        self.assertTrue(settings.x_disable_stats)
        self.assertTrue(settings.x_disable_meta)
        self.assertTrue(settings.disable_git)
        self.assertFalse(settings.save_code)
        self.assertEqual(settings.base_url, publish.WB_ENDPOINT)
        self.assertEqual(settings.x_graphql_retry_max, 0)
        self.assertEqual(settings.x_file_stream_retry_max, 0)
        self.assertEqual(settings.x_file_transfer_retry_max, 0)

    def test_known_hf_presigned_storage_never_receives_bearer_token(self):
        import requests
        session = publish.bounded_session('hf')
        url = 'https://hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com/asset?X-Amz-Signature=dummy'
        request = requests.Request('PUT', url, headers={'Authorization': 'Bearer dummy'}).prepare()
        with patch('requests.Session.send', return_value=object()):
            session.send(request)
        self.assertNotIn('Authorization', request.headers)
        with self.assertRaises(ValueError):
            session.send(requests.Request('PUT', url.split('?')[0]).prepare())

    def test_release_symlinks_rejected_before_audit_opens_any_file(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            root = repo / publish.RELEASE_RELATIVE / 'packages/smolvla'
            root.mkdir(parents=True)
            (root / 'release_manifest.json').symlink_to('/unapproved/input')
            with patch.object(publish, 'audit_release') as audit:
                with self.assertRaises(ValueError):
                    publish.reviewed_files(repo, 'smolvla')
            audit.assert_not_called()


class GraphQLTests(unittest.TestCase):
    def setUp(self):
        self.control = tracking.WandbControl('dummy-key')
        self.calls = []
        self.visibility = None
        self.missing_schema = False
        self.project_fields = ('name', 'access')
        self.input_fields = ('name', 'entityName', 'access')
        self.privacy_type = {'kind': 'SCALAR', 'name': 'String'}
        self.enum_values = ('PRIVATE', 'PUBLIC', 'OPEN')
        self.mutation_model = None

        def execute(query, variables=None):
            self.calls.append((query, variables))
            if 'PrivacySchema' in query:
                if self.missing_schema:
                    return {'__type': None}
                name = variables['name']
                if name not in ('Project', 'UpsertModelInput'):
                    return {'__type': {'enumValues': [{'name': n} for n in self.enum_values]}}
                names = self.project_fields if name == 'Project' else self.input_fields
                fields = [{'name': n, 'type': self.privacy_type if n in ('access', 'visibility')
                           else {'kind': 'SCALAR', 'name': 'String'}} for n in names]
                return {'__type': {'name': name, 'fields': fields if name == 'Project' else None,
                                  'inputFields': fields if name == 'UpsertModelInput' else None}}
            if 'PrivatePublicationProject' in query:
                self.visibility = variables['visibility']
                model = self.mutation_model if self.mutation_model is not None else {
                    'name': tracking.PROJECT, 'visibility': self.visibility}
                return {'upsertModel': {'model': model}}
            if 'PublishingPrivacy' in query:
                return {'project': None if self.visibility is None else
                        {'name': tracking.PROJECT, 'visibility': self.visibility}}
            if 'PublishingIdentity' in query:
                return {'viewer': {'username': 'authenticated-user', 'entity': tracking.APPROVED_ENTITY}}
            if 'HistoricalRunCollision' in query:
                return {'project': {'run': None}}
            raise AssertionError('Unexpected test query')
        self.control.execute = execute

    def test_private_creation_is_schema_checked_and_explicit(self):
        self.control.create_private_project(tracking.APPROVED_ENTITY)
        mutations = [(q, v) for q, v in self.calls if 'mutation ' in q]
        self.assertEqual(len(mutations), 1)
        query, variables = mutations[0]
        self.assertEqual(variables, {'entity': tracking.APPROVED_ENTITY,
                                    'project': tracking.PROJECT, 'visibility': 'PRIVATE'})
        self.assertIn('$visibility: String!', query)
        self.assertIn('entityName: $entity, access: $visibility', query)
        self.assertIn('model { name visibility: access }', query)
        self.assertEqual(self.control.project_visibility(tracking.APPROVED_ENTITY), 'PRIVATE')
        privacy = [q for q, _ in self.calls if 'PublishingPrivacy' in q]
        self.assertEqual(len(privacy), 2)
        self.assertTrue(all('{ name visibility: access }' in q for q in privacy))

    def test_access_preferred_when_both_fields_exist(self):
        self.project_fields += ('visibility',)
        self.input_fields += ('visibility',)
        self.test_private_creation_is_schema_checked_and_explicit()

    def test_visibility_only_schema_remains_supported(self):
        self.project_fields = ('name', 'visibility')
        self.input_fields = ('name', 'entityName', 'visibility')
        self.control.create_private_project(tracking.APPROVED_ENTITY)
        mutation = next(q for q, _ in self.calls if 'mutation ' in q)
        self.assertIn('visibility: $visibility', mutation)
        self.assertIn('model { name visibility }', mutation)
        self.assertNotIn('access', mutation)

    def test_no_shared_privacy_field_blocks_before_lookup_or_mutation(self):
        self.input_fields = ('name', 'entityName', 'visibility')
        with self.assertRaises(ValueError):
            self.control.project_visibility(tracking.APPROVED_ENTITY)
        self.assertTrue(all('PrivacySchema' in q for q, _ in self.calls))

    def test_non_null_enum_access_uses_introspected_type(self):
        self.privacy_type = {'kind': 'NON_NULL', 'name': None,
                             'ofType': {'kind': 'ENUM', 'name': 'ProjectAccess'}}
        self.control.create_private_project(tracking.APPROVED_ENTITY)
        mutation = next(q for q, _ in self.calls if 'mutation ' in q)
        self.assertIn('$visibility: ProjectAccess!', mutation)
        self.assertIn('access: $visibility', mutation)

    def test_enum_without_private_blocks_before_lookup_or_mutation(self):
        self.privacy_type = {'kind': 'ENUM', 'name': 'ProjectAccess'}
        self.enum_values = ('PUBLIC',)
        with self.assertRaises(ValueError):
            self.control.project_visibility(tracking.APPROVED_ENTITY)
        self.assertTrue(all('PrivacySchema' in q for q, _ in self.calls))

    def test_mutation_requires_exact_private_and_project_name(self):
        # Isolate response validation from the new access-schema support.
        self.project_fields = ('name', 'visibility')
        self.input_fields = ('name', 'entityName', 'visibility')
        for model in ({}, {'name': 'wrong', 'visibility': 'PRIVATE'},
                      *({'name': tracking.PROJECT, 'visibility': v}
                        for v in ('PUBLIC', 'OPEN', 'private', None))):
            with self.subTest(model=model):
                self.visibility = None
                self.mutation_model = model
                with self.assertRaises(ValueError):
                    self.control.create_private_project(tracking.APPROVED_ENTITY)

    def test_unknown_or_lowercase_access_is_not_private(self):
        for value in ('private', 'UNKNOWN', '', False):
            with self.subTest(value=value):
                self.visibility = value
                with self.assertRaises(ValueError):
                    self.control.project_visibility(tracking.APPROVED_ENTITY)

    def test_actual_schema_import_queries_privacy_before_and_after_run(self):
        from publication.tests.test_remote_operations import FakeWandb, historical_row
        sdk = FakeWandb()
        original_init = sdk.init

        def init(**kwargs):
            self.assertEqual(self.calls[-2][1]['project'], tracking.PROJECT)
            self.assertIn('PublishingPrivacy', self.calls[-2][0])
            self.calls.append(('SDK init', None))
            return original_init(**kwargs)

        sdk.init = init
        report = tracking.import_history(historical_row(), self.control, sdk, 'dummy-key', None, {})
        self.assertEqual(report['status'], 'imported')
        queries = [q for q, _ in self.calls]
        mutation = next(i for i, q in enumerate(queries) if 'mutation ' in q)
        init_index = queries.index('SDK init')
        self.assertIn('PublishingPrivacy', queries[mutation - 1])
        self.assertIn('PublishingPrivacy', queries[mutation + 1])
        self.assertLess(mutation, init_index)
        self.assertIn('PublishingPrivacy', queries[-1])
        self.assertTrue(sdk.runs[0].finished)

    def test_actual_schema_existing_nonprivate_blocks_import_and_mutation(self):
        from publication.tests.test_remote_operations import FakeWandb, historical_row
        for value in ('PUBLIC', 'OPEN', 'private', 'UNKNOWN'):
            with self.subTest(value=value):
                self.visibility = value
                sdk = FakeWandb()
                report = tracking.import_history(historical_row(), self.control, sdk, 'dummy-key', None, {})
                self.assertEqual(report['status'], 'blocked')
                self.assertEqual(report['phase'], 'project_privacy_check')
                self.assertEqual(sdk.runs, [])
                self.assertFalse(any('mutation ' in q for q, _ in self.calls))

    def test_existing_public_project_blocks_mutation(self):
        self.visibility = 'PUBLIC'
        with self.assertRaises(ValueError):
            self.control.create_private_project('personal')
        self.assertFalse(any('mutation ' in q for q, _ in self.calls))

    def test_missing_introspection_blocks_even_project_lookup(self):
        self.missing_schema = True
        with self.assertRaises(ValueError):
            self.control.project_visibility('personal')
        self.assertFalse(any('mutation ' in q for q, _ in self.calls))


if __name__ == '__main__':
    unittest.main()