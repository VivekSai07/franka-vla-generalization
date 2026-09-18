"""Private historical W&B import; never attaches to or modifies frozen trainers."""
from datetime import datetime, timezone
import hashlib
import re

from publication.publish import WB_ENDPOINT, bounded_session, deadline, quiet, safe_error

PROJECT = 'franka-fixed-layout-baselines'
APPROVED_ENTITY = 'vivekchaitanya07-university-of-stuttgart'


def run_id(row):
    if row['model'] not in ('smolvla', 'xvla') or not re.fullmatch(r'[0-9a-f]{64}', row['source_hash']):
        raise ValueError('Invalid historical identity')
    return row['model'] + '-' + hashlib.sha256((row['model'] + ':' + row['source_hash']).encode()).hexdigest()[:24]


def historical_settings(api_key):
    """Names verified against installed wandb.Settings 0.24.2, not legacy flags."""
    return dict(
        api_key=api_key, base_url=WB_ENDPOINT, mode='online', anonymous='never',
        console='off', quiet=True, silent=True, disable_code=True, disable_git=True,
        save_code=False, disable_job_creation=True, sagemaker_disable=True,
        x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True,
        x_disable_viewer=True, host='historical-import', program='historical_import',
        init_timeout=60, login_timeout=30, summary_timeout=60,
        x_graphql_timeout_seconds=30, x_file_stream_timeout_seconds=60,
        x_file_transfer_timeout_seconds=60,
        x_graphql_retry_max=0, x_file_stream_retry_max=0, x_file_transfer_retry_max=0,
    )


class WandbControl:
    """Bounded official GraphQL control plane; no private SDK API assumptions.

    SDK 0.24.2 public Project._load/GetProject omits visibility and its project
    creation helpers cannot specify PRIVATE. Introspection must prove the field
    and mutation input exist. No fallback to account defaults or public projects.
    """
    def __init__(self, api_key):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError('Required credential unavailable')
        self.session = bounded_session('wandb')
        self.session.auth = ('api', api_key)
        self.visibility_type = None
        self.privacy_field = None

    def execute(self, query, variables=None):
        response = self.session.post(WB_ENDPOINT + '/graphql',
                                     json={'query': query, 'variables': variables or {}},
                                     allow_redirects=False)
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError('Unexpected GraphQL transport status')
        payload = response.json()
        if payload.get('errors') or not isinstance(payload.get('data'), dict):
            raise ValueError('GraphQL operation not verified')
        return payload['data']

    def type_info(self, name):
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
            raise ValueError('Invalid GraphQL type')
        return self.execute('''query PrivacySchema($name: String!) {
            __type(name: $name) {
                kind name enumValues { name }
                fields { name type { kind name ofType { kind name } } }
                inputFields { name type { kind name ofType { kind name } } }
            }
        }''', {'name': name})['__type']

    def privacy_schema(self):
        if self.visibility_type:
            return
        project = self.type_info('Project')
        inputs = self.type_info('UpsertModelInput')
        if not project or not inputs:
            raise ValueError('Privacy schema unavailable')
        project_fields = {f['name']: f for f in project['fields'] or []}
        input_fields = {f['name']: f for f in inputs['inputFields'] or []}
        privacy_field = next((name for name in ('access', 'visibility')
                              if name in project_fields and name in input_fields), None)
        if privacy_field is None or not {'name', 'entityName'} <= set(input_fields):
            raise ValueError('Privacy controls unavailable')
        for field in (project_fields[privacy_field], input_fields[privacy_field]):
            kind = field['type']
            if kind['kind'] == 'NON_NULL':
                kind = kind['ofType']
            if kind['kind'] == 'ENUM':
                enum = self.type_info(kind['name'])
                if not enum or 'PRIVATE' not in {v['name'] for v in enum['enumValues'] or []}:
                    raise ValueError('PRIVATE visibility unavailable')
            elif kind['kind'] != 'SCALAR' or kind['name'] != 'String':
                raise ValueError('Unsupported visibility type')
        # Mutation variable type is introspected, not an invented SDK parameter.
        field_type = input_fields[privacy_field]['type']
        visibility_type = field_type['ofType']['name'] if field_type['kind'] == 'NON_NULL' else field_type['name']
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', visibility_type):
            raise ValueError('Invalid visibility type')
        self.privacy_field = privacy_field
        self.visibility_type = visibility_type

    def viewer(self):
        return self.execute('query PublishingIdentity { viewer { username entity } }')['viewer']

    def project_visibility(self, entity):
        self.privacy_schema()
        selection = 'visibility: access' if self.privacy_field == 'access' else 'visibility'
        data = self.execute('''query PublishingPrivacy($entity: String!, $project: String!) {
            project(name: $project, entityName: $entity) { name %s }
        }''' % selection, {'entity': entity, 'project': PROJECT})['project']
        if data is None:
            return None
        if data.get('name') != PROJECT or data.get('visibility') not in ('PRIVATE', 'PUBLIC', 'OPEN'):
            raise ValueError('Project visibility not verified')
        return data['visibility']

    def create_private_project(self, entity):
        self.privacy_schema()
        # Repeat absence check immediately before the only allowed mutation.
        if self.project_visibility(entity) is not None:
            raise ValueError('Project appeared concurrently; refuse mutation')
        selection = 'visibility: access' if self.privacy_field == 'access' else 'visibility'
        data = self.execute('''mutation PrivatePublicationProject($entity: String!, $project: String!,
                         $visibility: %s!) {
            upsertModel(input: {name: $project, entityName: $entity, %s: $visibility}) {
                model { name %s }
            }
        }''' % (self.visibility_type, self.privacy_field, selection),
                            {'entity': entity, 'project': PROJECT, 'visibility': 'PRIVATE'})
        model = data['upsertModel']['model']
        if not isinstance(model, dict) or model.get('name') != PROJECT or model.get('visibility') != 'PRIVATE':
            raise ValueError('Created project privacy not verified')

    def run_exists(self, entity, identifier):
        data = self.execute('''query HistoricalRunCollision($entity: String!, $project: String!, $run: String!) {
            project(name: $project, entityName: $entity) { run(name: $run) { id } }
        }''', {'entity': entity, 'project': PROJECT, 'run': identifier})['project']
        if data is None or 'run' not in data:
            raise ValueError('Run absence not established')
        return data['run'] is not None


def import_history(row, control, sdk, api_key, requested_entity, report, save=lambda: None):
    report.update(model=row['model'], status='preflight', project=PROJECT)
    run = None
    try:
        with quiet():
            report['phase'] = 'historical_input_validation'
            if ([m['update'] for m in row['history']] != list(range(1, 5001))
                    or len(row['trials']) != 20 or row['config']['logging_mode'] != 'historical_import'):
                raise ValueError('Historical input coverage mismatch')
            report['phase'] = 'credential_check'
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError('Required credential unavailable')
            report['phase'] = 'approved_entity_check'
            identity = control.viewer()
            if (not isinstance(identity, dict) or not isinstance(identity.get('username'), str)
                    or not identity['username'].strip()):
                raise ValueError('Authenticated viewer not verified')
            entity = identity.get('entity')
            # Native viewer.entity is the authenticated default, not necessarily
            # the username. Non-default memberships are deliberately unsupported.
            if entity != APPROVED_ENTITY or requested_entity not in (None, APPROVED_ENTITY):
                raise ValueError('Approved authenticated default entity not verified')
            identifier = run_id(row)
            report.update(entity=entity, run_id=identifier)
            report['phase'] = 'project_privacy_check'
            visibility = control.project_visibility(entity)
            if visibility is None:
                control.create_private_project(entity)
                report['status'] = 'private_project_created_no_run'
                save()
            elif visibility != 'PRIVATE':
                raise ValueError('Existing project is not private; never change visibility')
            if control.project_visibility(entity) != 'PRIVATE':
                raise ValueError('Project privacy not verified')
            report['phase'] = 'run_collision_check'
            if control.run_exists(entity, identifier):
                raise ValueError('Historical run already exists; never resume')
            settings = sdk.Settings(**historical_settings(api_key))
            config = dict(row['config'], source_hash=row['source_hash'],
                          import_started_utc=datetime.now(timezone.utc).isoformat())
            report.update(status='import_unconfirmed', phase='historical_import')
            save()
            run = sdk.init(entity=entity, project=PROJECT, id=identifier, name=row['model'] + '-5000-historical',
                           job_type='historical_import', resume='never', config=config, settings=settings)
            run.define_metric('update')
            run.define_metric('train/*', step_metric='update')
            for metric in row['history']:
                run.log(metric, step=metric['update'])
            run.summary.update(row['summary'])
            columns = sorted(row['trials'][0])
            run.summary['trial_count'] = len(row['trials'])
            run.summary['trials'] = sdk.Table(columns=columns,
                                              data=[[trial[column] for column in columns] for trial in row['trials']])
            run.finish()
            run = None
            report['phase'] = 'final_privacy_check'
            if control.project_visibility(entity) != 'PRIVATE':
                raise ValueError('Final project privacy not verified')
            report.update(status='imported', private=True, logged_updates=5000, trial_count=20,
                          completion_evidence='SDK finish returned; project privacy rechecked; no remote history readback')
    except Exception as exc:
        report['error'] = safe_error(exc)
        if report['status'] == 'preflight':
            report['status'] = 'blocked'
        if run is not None:
            with quiet():
                try:
                    with deadline(30):
                        run.finish(exit_code=1)
                except Exception:
                    pass
    finally:
        save()
    return report