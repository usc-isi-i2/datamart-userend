from . import config_services

import os

home_dir = os.getenv("HOME")
# in the case that no correct home dir found (e.g. in docker)
if home_dir == "/":
    home_dir = "/tmp"


default_datamart_url = config_services.get_default_datamart_url()

cache_file_storage_base_loc = os.path.join(home_dir, "memcache_storage")

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
wikifier_column_mark = "%^$#@wikifier@%^$#"
max_entities_length = 10000

need_wikifier_column_type_list = {"https://metadata.datadrivendiscovery.org/types/CategoricalData",
                                  "http://schema.org/Text"
                                  }

skip_wikifier_column_type_list = {"https://metadata.datadrivendiscovery.org/types/PrimaryKey",
                                  "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                                  "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                                  "https://metadata.datadrivendiscovery.org/types/Target",
                                  "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                                  "http://schema.org/Float",
                                  "http://schema.org/Integer"}
default_temp_path = "/tmp"

# elastic search to fetch FB embeddings
wikidata_uri_template = '<http://www.wikidata.org/entity/{}>'
# em_es_url = "http://kg2018a.isi.edu:9200"
# em_es_url = "http://sitaware.isi.edu:9200"
# em_es_index = "wiki_fb_embeddings_1"
# em_es_type = "vectors"

# new wikifier server
# new_wikifier_server = "http://dsbox02.isi.edu:8396/wikify"
# new_wikifier_server = "http://minds03.isi.edu:8396/wikify"


max_longitude_val = 180
min_longitude_val = -180
max_latitude_val = 90
min_latitude_val = -90

maximum_accept_wikifier_size = 2000000
