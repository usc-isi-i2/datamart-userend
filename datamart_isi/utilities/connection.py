from datamart_isi import config_services


def get_memcache_server_url() -> str:
    return config_services.get_service_url('memcached')


def get_wikidata_server_url() -> str:
    return config_services.get_service_url('wikidata')


def get_general_search_server_url() -> str:
    return config_services.get_service_url('general_search')


def get_wikifier_identifier_server_url() -> str:
    return config_services.get_service_url('wikifier_identifier')
