import os
import json

with open(os.path.join(os.path.dirname(__file__), 'datamart-services.json')) as input:
    service_defs = json.load(input)


def _get_service_def(service_name) -> dict:
    for definition in service_defs['services']:
        if service_name == definition['name']:
            return definition
    return None


def get_service_url(service_name) -> str:
    definition = _get_service_def(service_name)
    if definition is None:
        raise ValueError(f'Service name not found: {service_name}')
    if service_defs['server']['mode'] == 'single_host':
        host = service_defs['server']['host']
    else:
        host = definition['host']
    port = definition['port']
    path = definition.get('path', '')
    if path:
        return f'http://{host}:{port}/{path}'
    else:
        return f'http://{host}:{port}'


def get_default_datamart_url() -> str:
    return get_service_url('isi_datamart')
