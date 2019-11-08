import typing

from datamart_isi import config_services


def get_memcache_server_url() -> str:
    return config_services.get_service_url('memcached', as_url=False)


def get_wikidata_server_url() -> str:
    return config_services.get_service_url('wikidata')


def get_general_search_server_url() -> str:
    return config_services.get_service_url('general_search')


def get_wikifier_identifier_server_url() -> str:
    return config_services.get_service_url('wikifier_identifier')


def get_wikifier_knowledge_graph_server_url() -> str:
    return config_services.get_service_url('wikifier_knowledge_graph')


def get_es_fb_embedding_server_url() -> str:
    return config_services.get_service_url('elasticsearch_fb_embeddings', as_url=True)


def get_general_search_test_server_url() -> str:
    return config_services.get_service_url('general_search')


def get_redis_host_port() -> typing.Tuple[str, int]:
    _, host, port, _ = config_services.get_host_port_path('redis')
    return (host, port)
