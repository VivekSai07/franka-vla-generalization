"""Read-only, credential-safe schema diagnosis; no project/run mutation."""
import json

from publication.publish import credentials, deadline, private_runtime, quiet, safe_error
from publication.tracking import WandbControl, PROJECT


def main():
    result = {"operation": "readonly_privacy_schema", "project": PROJECT}
    try:
        with private_runtime(), quiet(), deadline(90):
            values = credentials({'wandb'})
            control = WandbControl(values['WANDB_API_KEY'])
            values.clear()
            try:
                for name in ('Project', 'UpsertModelInput'):
                    info = control.type_info(name)
                    if info is None:
                        result[name] = None
                        continue
                    fields = info.get('inputFields') or info.get('fields') or []
                    result[name] = {f['name']: f['type'] for f in fields
                                    if f['name'] in ('name', 'entityName', 'visibility', 'access')}
                    for field in fields:
                        if field['name'] in ('visibility', 'access'):
                            kind = field['type']
                            if kind['kind'] == 'NON_NULL':
                                kind = kind['ofType']
                            if kind['kind'] == 'ENUM':
                                enum = control.type_info(kind['name'])
                                result['enum_' + kind['name']] = [v['name'] for v in enum['enumValues']]
            finally:
                control.session.auth = None
                control.session.close()
    except Exception as exc:
        result['error'] = safe_error(exc)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()