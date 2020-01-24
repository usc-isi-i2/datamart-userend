import logging
import hashlib
import pandas as pd
import wikifier
import os
import traceback

from ast import literal_eval
from io import StringIO
from datamart_isi.cache.wikidata_cache import QueryCache
from datamart_isi.config import cache_file_storage_base_loc
from datamart_isi.utilities import connection
from datamart_isi.utilities.utils import Utils
from datamart_isi.utilities.d3m_wikifier import check_has_q_node_columns
from d3m.container import DataFrame as d3m_DataFrame


WIKIDATA_CACHE_MANAGER = QueryCache()
WIKIDATA_SERVER = connection.get_wikidata_server_url()
_logger = logging.getLogger(__name__)


class MaterializerCache(object):
    @staticmethod
    def materialize(metadata, run_wikifier=True) -> pd.DataFrame:
        # general type materializer
        if 'url' in metadata:
            loaded_data = MaterializerCache.get_data(metadata=metadata)

            has_q_nodes = check_has_q_node_columns(loaded_data)
            if has_q_nodes:
                _logger.warning("The original data already has Q nodes! Will not run wikifier")
            else:
                if run_wikifier:
                    loaded_data = wikifier.produce(loaded_data)

            return loaded_data

        elif "p_nodes_needed" in metadata:
            # wikidata materializer
            label_part = "  ?itemLabel \n"
            where_part = ""
            length = metadata.get("length", 100)
            for i, each_p_node in enumerate(metadata["p_nodes_needed"]):
                label_part += "  ?value" + str(i) + "Label\n"
                where_part += "  ?item wdt:" + each_p_node + " ?value" + str(i) + ".\n"

            sparql_query = """PREFIX wikibase: <http://wikiba.se/ontology#>
                              PREFIX wd: <http://www.wikidata.org/entity/>
                              prefix bd: <http://www.bigdata.com/rdf#>
                              PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                              SELECT \n""" + label_part + "WHERE \n {\n" + where_part \
                           + """  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }\n}\n""" \
                           + "LIMIT " + str(length)

            results = WIKIDATA_CACHE_MANAGER.get_result(sparql_query)
            all_res = {}
            for i, result in enumerate(results):
                each_res = {}
                for each_key in result.keys():
                    each_res[each_key] = result[each_key]['value']
                all_res[i] = each_res
            df_res = pd.DataFrame.from_dict(all_res, "index")
            column_names = df_res.columns.tolist()
            column_names = column_names[1:]
            column_names_replaced = dict()
            for each in zip(column_names, metadata["p_nodes_needed"]):
                column_names_replaced[each[0]] = Utils.get_node_name(each[1])
            df_res.rename(columns=column_names_replaced, inplace=True)
            df_res = d3m_DataFrame(df_res, generate_metadata=True)
            return df_res

        else:
            raise ValueError("Unknown type for materialize!")


    @staticmethod
    def materialize_for_wikitable(dataset_url: str, file_type: str, extra_information: str) -> pd.DataFrame:
        from datamart_isi.materializers.wikitables_materializer import WikitablesMaterializer
        materializer = WikitablesMaterializer()
        loaded_data = materializer.get_one(dataset_url, extra_information['xpath'])
        return loaded_data

    @staticmethod
    def materialize_for_general(dataset_url: str, file_type: str) -> pd.DataFrame:
        from datamart_isi.materializers.general_materializer import GeneralMaterializer
        general_materializer = GeneralMaterializer()
        file_metadata = {
            "materialization": {
                "arguments": {
                    "url": dataset_url,
                    "file_type": file_type
                }
            }
        }

        try:
            result = general_materializer.get(metadata=file_metadata).to_csv(index=False)
            # remove last \n so that we will not get an extra useless row
            if result[-1] == "\n":
                result = result[:-1]

            loaded_data = StringIO(result)
            loaded_data = pd.read_csv(loaded_data, dtype="str")
            return loaded_data
        except:
            traceback.print_exc()
            raise ValueError("Materializing from " + dataset_url + " failed!")

    @staticmethod
    def get_data(metadata) -> pd.DataFrame:
        """
        Main function for get the data through cache
        :param metadata:
        :return:
        """
        dataset_url = metadata['url']['value']
        # updated v2019.10.14: add local storage cache file
        hash_generator = hashlib.md5()
        hash_generator.update(dataset_url.encode('utf-8'))
        hash_url_key = hash_generator.hexdigest()
        dataset_cache_loc = os.path.join(cache_file_storage_base_loc, "datasets_cache", hash_url_key + ".h5")
        _logger.debug("Try to check whether cache file exist or not at " + dataset_cache_loc)
        if os.path.exists(dataset_cache_loc):
            _logger.info("Found exist cached dataset file")
            loaded_data = pd.read_hdf(dataset_cache_loc)
        else:
            _logger.info("Cached dataset file does not find, will run materializer.")
            file_type = metadata.get("file_type") or ""
            if file_type == "":
                # no file type get, try to guess
                file_type = dataset_url.split(".")[-1]
            else:
                file_type = file_type['value']

            if file_type == "wikitable":
                extra_information = literal_eval(metadata['extra_information']['value'])
                loaded_data = MaterializerCache.materialize_for_wikitable(dataset_url, file_type, extra_information)
            else:
                loaded_data = MaterializerCache.materialize_for_general(dataset_url, file_type)
                try:
                    # save the loaded data
                    loaded_data.to_hdf(dataset_cache_loc, key='df', mode='w', format='fixed')
                    _logger.debug("Saving dataset cache success!")
                except Exception as e:
                    _logger.warning("Saving dataset cache failed!")
                    _logger.debug(e, exc_info=True)
        return loaded_data
