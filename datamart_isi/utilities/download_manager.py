import requests
import pandas
import logging
import os
from multiprocessing import Pool
from d3m.container import Dataset as d3m_Dataset
from d3m.container import DataFrame as d3m_DataFrame
from d3m.base import utils as d3m_utils
from datamart_isi.cache.general_search_cache import GeneralSearchCache
from datamart_isi.cache.metadata_cache import MetadataCache
from datamart_isi import config
from datamart_isi.utilities import connection
from SPARQLWrapper import SPARQLWrapper, JSON, POST, URLENCODED

WIKIDATA_URI_TEMPLATE = config.wikidata_uri_template
EM_ES_URL = config.em_es_url
EM_ES_INDEX = config.em_es_index
EM_ES_TYPE = config.em_es_type
DEFAULT_DATAMART_URL = config.default_datamart_url
AUGMENT_RESOURCE_ID = config.augmented_resource_id
logger = logging.getLogger(__name__)



class DownloadManager:
    @staticmethod
    def fetch_fb_embeddings(q_nodes_list, target_q_node_column_name):
        # add vectors columns in wikifier_res
        qnodes = list(filter(None, q_nodes_list))
        qnode_uris = [WIKIDATA_URI_TEMPLATE.format(qnode) for qnode in qnodes]
        # do elastic search
        num_of_try = int(len(qnode_uris)/1024) + 1 if len(qnode_uris)%1024 != 0 else int(len(qnode_uris)/1024)
        res = dict()
        for i in range(num_of_try):
            query = {
                'query': {
                    'terms': {
                        'key.keyword': qnode_uris[1024*i:1024*i+1024]
                    }
                },
                "size": len(qnode_uris[1024*i:1024*i+1024])
            }
            url = '{}/{}/{}/_search'.format(EM_ES_URL, EM_ES_INDEX, EM_ES_TYPE)
            resp = requests.get(url, json=query)
            if resp.status_code == 200:
                result = resp.json()
                hits = result['hits']['hits']
                for hit in hits:
                    source = hit['_source']
                    _qnode = source['key'].split('/')[-1][:-1]
                    res[_qnode] = ",".join(source['value'])

        # change to dataframe
        return_df = pandas.DataFrame()
        for key, val in res.items():
            each_result = dict()
            each_result["q_node"] = key
            vectors = val.split(',')
            for i in range(len(vectors)):
                if i < 10:
                    s = '00' + str(i)
                elif i < 100:
                    s = '0' + str(i)
                else:
                    s = str(i)
                v_name = "vector_" + s + "_of_qnode_with_" + target_q_node_column_name
                each_result[v_name] = float(vectors[i])
            return_df = return_df.append(each_result, ignore_index=True)

        return return_df

    @staticmethod
    def parse_geospatial_query(geo_variable):
        geo_gra_dict = {'country': 'Q6256', 'state': 'Q7275', 'city': 'Q515', 'county': 'Q28575',
                        'postal_code': 'Q37447'}

        wikidata_server = connection.get_wikidata_server_url(DEFAULT_DATAMART_URL)
        qm_wikidata = SPARQLWrapper(wikidata_server)
        qm_wikidata.setReturnFormat(JSON)
        qm_wikidata.setMethod(POST)
        qm_wikidata.setRequestMethod(URLENCODED)

        if "latitude" in geo_variable.keys() and "longitude" in geo_variable.keys():
            granularity = geo_gra_dict[geo_variable["granularity"]]
            radius = geo_variable["radius"]
            x, y = geo_variable["longitude"], geo_variable["latitude"]

            if x and y:
                # find closest Q nodes around a geospatial point from wikidata query
                sparql_query = "select distinct ?place where \n{\n  ?place wdt:P31/wdt:P279* wd:" + granularity + " .\n" \
                               + "SERVICE wikibase:around {\n ?place wdt:P625 ?location .\n" \
                               + "bd:serviceParam wikibase:center " + "\"Point(" + str(x) + " " + str(
                    y) + ")\"^^geo:wktLiteral .\n" \
                               + "bd:serviceParam wikibase:radius " + "\"" + str(radius) + "\" .\n" \
                               + "bd:serviceParam wikibase:distance ?dist. \n}\n" \
                               + "SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\" }\n}\n" \
                               + "ORDER BY ASC(?dist) \n Limit 1 \n"
                qm_wikidata.setQuery(sparql_query)
                results = qm_wikidata.query().convert()['results']['bindings']
                if results:
                    value = results[0]["place"]["value"]
                    return value.split('/')[-1]
                else:
                    return ''

    @staticmethod
    def query_geospatial_wikidata(supplied_dataset, search_result, endpoint) -> d3m_Dataset:
        general_search_cache_manager = GeneralSearchCache(connection_url=endpoint)
        _, supplied_dataframe = d3m_utils.get_tabular_resource(dataset=supplied_dataset,
                                                               resource_id=None,
                                                               has_hyperparameter=False)
        # try cache first
        try:
            cache_key = general_search_cache_manager.get_hash_key(supplied_dataframe=supplied_dataframe,
                                                                  search_result_serialized=search_result)
            cache_result = general_search_cache_manager.get_cache_results(cache_key)
            if cache_result is not None:
                return cache_result
        except Exception as e:
            cache_key = None

        # find geospatial data's qnodes
        latitude_index = search_result['metadata']['search_result']['latitude_index']
        longitude_index = search_result['metadata']['search_result']['longitude_index']
        radius = search_result['metadata']['search_result']['radius']
        gran = search_result['metadata']['search_result']['granularity']

        geo_variables_list = []
        for latitude, longitude in zip(supplied_dataframe.iloc[:, latitude_index],supplied_dataframe.iloc[:, longitude_index]):
            geo_variable = {"latitude": latitude, "longitude": longitude, "radius": radius, "granularity": gran}
            geo_variables_list.append(geo_variable)

        # get qnodes from multiple processing
        qnodes = []
        with Pool(os.cpu_count()) as p:
            qnodes.append(p.map(DownloadManager.parse_geospatial_query, geo_variables_list))

        # generate augment_results
        res_df = supplied_dataframe
        lat_name, long_name = supplied_dataframe.columns[latitude_index], supplied_dataframe.columns[longitude_index]
        res_df["Geo_" + lat_name + "_" + long_name + "_" + gran + "_wikidata"] = qnodes

        return_df = d3m_DataFrame(res_df, generate_metadata=False)
        resources = {AUGMENT_RESOURCE_ID: return_df}
        res_ds = d3m_Dataset(resources=resources, generate_metadata=False)
        res_ds[AUGMENT_RESOURCE_ID].fillna('', inplace=True)
        res_ds[AUGMENT_RESOURCE_ID] = res_ds[AUGMENT_RESOURCE_ID].astype(str)

        # save to cache file
        if cache_key:
            response = general_search_cache_manager.add_to_memcache(supplied_dataframe=supplied_dataframe,
                                                                    search_result_serialized=search_result,
                                                                    augment_results=res_ds,
                                                                    hash_key=cache_key
                                                                    )
            # save the augmented result's metadata if second augment is conducted
            MetadataCache.save_metadata_from_dataset(res_ds)
            if not response:
                logger.warning("Push augment results to results failed!")
            else:
                logger.info("Push augment results to memcache success!")
        return res_ds
