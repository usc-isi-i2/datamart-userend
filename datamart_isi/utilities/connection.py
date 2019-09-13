from datamart_isi import config


def get_memcache_server_url(connection_url: str = config.default_datamart_url) -> str:
    if connection_url.startswith("http://"):
        connection_url = connection_url[7:]
    elif connection_url.startswith("https://"):
        connection_url = connection_url[8:]
    if connection_url.endswith(config.rest_api_suffix):
        connection_url = connection_url[:-5]
    memcache_url = connection_url + config.memcache_server_suffix
    return memcache_url


def get_wikidata_server_url(connection_url: str = config.default_datamart_url) -> str:
    if not connection_url.startswith("http://"):
        connection_url = "http://" + connection_url
    if connection_url.endswith(config.rest_api_suffix):
        connection_url = connection_url[:-5]
    wikidata_url = connection_url + config.wikidata_server_suffix
    return wikidata_url


def get_general_search_server_url(connection_url: str = config.default_datamart_url) -> str:
    if not connection_url.startswith("http://"):
        connection_url = "http://" + connection_url
    if connection_url.endswith(config.rest_api_suffix):
        connection_url = connection_url[:-5]
    general_server_url = connection_url + config.general_search_server_suffix
    return general_server_url


def get_wikifier_identifier_server_url(connection_url: str = config.default_datamart_url) -> str:
    if not connection_url.startswith("http://"):
        connection_url = "http://" + connection_url
    if connection_url.endswith(config.rest_api_suffix):
        connection_url = connection_url[:-5]
    general_server_url = connection_url + config.wikifier_server_suffix
    return general_server_url


def get_general_search_test_server_url(connection_url: str = config.default_datamart_url) -> str:
    if not connection_url.startswith("http://"):
        connection_url = "http://" + connection_url
    if connection_url.endswith(config.rest_api_suffix):
        connection_url = connection_url[:-5]
    general_server_url = connection_url + config.general_search_test_server_suffix
    return general_server_url
