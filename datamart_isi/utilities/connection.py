from datamart_isi import config


def get_memcache_server_url(connection_url: str) -> str:
    if connection_url.startswith("http://"):
        connection_url = connection_url[7:]
    memcache_url = connection_url + config.memcache_server_suffix
    return memcache_url


def get_wikidata_server_url(connection_url: str) -> str:
    if not connection_url.startswith("http://"):
        connection_url = "http://" + connection_url
    wikidata_url = connection_url + config.wikidata_server_suffix
    return wikidata_url


def get_genearl_search_server_url(connection_url: str) -> str:
    if not connection_url.startswith("http://"):
        connection_url = "http://" + connection_url
    general_server_url = connection_url + config.general_search_server_suffix
    return general_server_url
