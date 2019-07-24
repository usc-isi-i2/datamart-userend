default_datamart_url = "dsbox02.isi.edu"

wikidata_server = "http://kg2018a.isi.edu:8888/bigdata/namespace/wdq/sparql"
# wikidata_server_test = "http://sitaware.isi.edu:8080/bigdata/namespace/datamart3/sparql"
general_search_server = "http://dsbox02.isi.edu:9001/blazegraph/namespace/datamart3/sparql"
wikifier_server = "http://dsbox02.isi.edu/wikifier/get_identifiers"
memcache_server = "dsbox02.isi.edu:11211"

cache_file_storage_base_loc = "/nfs1/dsbox-repo/datamart/datamart_new/memcache_storage"
wikidata_server_suffix = "/wikidata"
memcache_server_suffix = ":11211"
wikifier_server_suffix = "/wikifier/get_identifiers"
general_search_server_suffix = ":9001/blazegraph/namespace/datamart3/sparql"

# current memcache server's max size is 100MB
memcache_max_value_size = 1024*1024*100
# current cache expiration time is 24 hours
cache_expire_time = 3600*24

# following are datamart detail configs, usually these should not be changed
augmented_column_semantic_type = "https://metadata.datadrivendiscovery.org/types/Datamart_augmented_column"
augmented_resource_id = "learningData"
d3m_container_version = "https://metadata.datadrivendiscovery.org/schemas/v0/container.json"
p_nodes_ignore_list = {"P1549"}
q_node_semantic_type = "http://wikidata.org/qnode"
# this is used to add some speical part in query to constrain p nodes search results
special_request_for_p_nodes = {"P1813": "FILTER(strlen(str(?P1813)) = 2)"}
time_column_mark = "%^&*SPECIAL_TIME_TYPE%^&*"
max_entities_length = 10000

skip_wikifier_column_type_list = {"https://metadata.datadrivendiscovery.org/types/PrimaryKey",
                                  "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                                  "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                                  "https://metadata.datadrivendiscovery.org/types/Target",
                                  "https://metadata.datadrivendiscovery.org/types/SuggestedTarget"}
