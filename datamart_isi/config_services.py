from pathlib import Path

import os
import json
import typing

config_file = Path(os.path.join(os.path.dirname(__file__), 'datamart-services.json'))

if not config_file.exists():
    raise FileNotFoundError(f'Must define service location: {config_file}\nFor example, see datamart-services-dsbox01.json')

with open(config_file) as input:
    service_defs = json.load(input)


def _get_service_def(service_name) -> dict:
    for definition in service_defs['services']:
        if service_name == definition['name']:
            return definition
    return None


def get_host_port_path(service_name) -> typing.Tuple[str, str, int, str]:
    definition = _get_service_def(service_name)
    if definition is None:
        print('get_service_url missing definition: ', service_name)
        raise ValueError(f'Service name not found: {service_name}')

    default_host = service_defs['server'].get('default_host', '')
    host = definition.get('host', default_host)
    if host == '':
        raise ValueError(f'Host for service {service_name} not defined')

    port = int(definition['port'])
    path = definition.get('path', '')
    connection_type = definition.get('connection_type', 'http')
    return (connection_type, host, port, path)


def get_service_url(service_name, as_url=True) -> str:
    connection_type, host, port, path = get_host_port_path(service_name)
    if as_url:
        if path:
            url = f'{connection_type}://{host}:{port}/{path}'
        else:
            url = f'{connection_type}://{host}:{port}'
    else:
        if path:
            url = f'{host}:{port}/{path}'
        else:
            url = f'{host}:{port}'
    print('get_service_url', service_name, url)
    return url


def get_default_datamart_url() -> str:
    return get_service_url('isi_datamart')
