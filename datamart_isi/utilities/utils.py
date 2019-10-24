import typing
import pandas as pd
import wikifier
import traceback
import os
import logging
import json
import hashlib
from SPARQLWrapper import SPARQLWrapper, JSON, POST, URLENCODED
from d3m.metadata.base import ALL_ELEMENTS
from io import StringIO
from ast import literal_eval
from d3m.container import DataFrame as d3m_DataFrame
from datamart_isi.config import cache_file_storage_base_loc
from datamart_isi.utilities import connection
from dsbox.datapreprocessing.cleaner.data_profile import Profiler, Hyperparams as ProfilerHyperparams
from dsbox.datapreprocessing.cleaner.cleaning_featurizer import CleaningFeaturizer, CleaningFeaturizerHyperparameter

WIKIDATA_SERVER = connection.get_wikidata_server_url()

_logger = logging.getLogger(__name__)
seed_dataset_store_location = os.path.join(cache_file_storage_base_loc, "datasets_cache")


class Utils:
    DEFAULT_DESCRIPTION = {
        "materialization": {
            "python_path": "default_materializer"
        },
        "variables": []
    }

    @staticmethod
    def materialize(metadata, run_wikifier=True) -> pd.DataFrame:
        # general type materializer
        if 'url' in metadata:
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
                    loaded_data = Utils.materialize_for_wikitable(dataset_url, file_type, extra_information)
                else:
                    loaded_data = Utils.materialize_for_general(dataset_url, file_type)
                    try:
                        # save the loaded data
                        loaded_data.to_hdf(dataset_cache_loc, key='df', mode='w', format='fixed')
                        _logger.debug("Saving dataset cache success!")
                    except Exception as e:
                        _logger.warning("Saving dataset cache failed!")
                        _logger.debug(e, exc_info=True)
            # run dsbox's profiler and cleaner
            # from dsbox.datapreprocessing.cleaner.data_profile import Profiler, Hyperparams as ProfilerHyperparams
            # from dsbox.datapreprocessing.cleaner.cleaning_featurizer import CleaningFeaturizer, CleaningFeaturizerHyperparameter
            # hyper1 = ProfilerHyperparams.defaults()
            # profiler = Profiler(hyperparams=hyper1)
            # profiled_df = profiler.produce(inputs=loaded_data).value
            # hyper2 = CleaningFeaturizerHyperparameter.defaults()
            # clean_f = CleaningFeaturizer(hyperparams=hyper2)
            # clean_f.set_training_data(inputs=profiled_df)
            # clean_f.fit()
            # cleaned_df = pd.DataFrame(clean_f.produce(inputs=profiled_df).value)
            if run_wikifier:
                loaded_data = wikifier.produce(loaded_data)

            return loaded_data

        elif "p_nodes_needed" in metadata:
            # wikidata materializer
            label_part = "  ?itemLabel \n"
            where_part = ""
            for i, each_p_node in enumerate(metadata["p_nodes_needed"]):
                label_part += "  ?value" + str(i) + "Label\n"
                where_part += "  ?item wdt:" + each_p_node + " ?value" + str(i) + ".\n"
            try:
                sparql_query = """PREFIX wikibase: <http://wikiba.se/ontology#>
                                  PREFIX wd: <http://www.wikidata.org/entity/>
                                  prefix bd: <http://www.bigdata.com/rdf#>
                                  PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                                  SELECT \n""" + label_part + "WHERE \n {\n" + where_part \
                               + """  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }\n}\n""" \
                               + "LIMIT 100"

                sparql = SPARQLWrapper(WIKIDATA_SERVER)
                sparql.setQuery(sparql_query)
                sparql.setReturnFormat(JSON)
                # sparql.setMethod(POST)
                # sparql.setRequestMethod(URLENCODED)
            except:
                print("[ERROR] Wikidata query failed!")
                traceback.print_exc()

            results = sparql.query().convert()["results"]["bindings"]
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
    def get_node_name(node_code) -> str:
        """
        Function used to get the properties(P nodes) names with given P node
        :param node_code: a str indicate the P node (e.g. "P123")
        :return: a str indicate the P node label (e.g. "inception")
        """
        sparql_query = "SELECT DISTINCT ?x WHERE \n { \n" + \
                       "wd:" + node_code + " rdfs:label ?x .\n FILTER(LANG(?x) = 'en') \n} "
        try:
            sparql = SPARQLWrapper(WIKIDATA_SERVER)
            sparql.setQuery(sparql_query)
            sparql.setReturnFormat(JSON)
            sparql.setMethod(POST)
            sparql.setRequestMethod(URLENCODED)
            results = sparql.query().convert()
            return results['results']['bindings'][0]['x']['value']
        except:
            return node_code

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
                    # one example here: "url": "http://insight.dev.schoolwires.com/HelpAssets/C2Assets/C2Files/C2ImportFamRelSample.csv",
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
    def calculate_dsbox_features(data: pd.DataFrame, metadata: typing.Union[dict, None],
                                 selected_columns: typing.Set[int] = None) -> dict:
        """Calculate dsbox features, add to metadata dictionary

         Args:
             data: dataset as a pandas dataframe
             metadata: metadata dict

         Returns:
              updated metadata dict
         """

        from datamart_isi.profilers.dsbox_profiler import DSboxProfiler
        if not metadata:
            return metadata
        return DSboxProfiler().profile(inputs=data, metadata=metadata, selected_columns=selected_columns)

    @classmethod
    def generate_metadata_from_dataframe(cls, data: pd.DataFrame, original_meta: dict = None) -> dict:
        """Generate a default metadata just from the data, without the dataset schema

         Args:
             data: pandas DataFrame

         Returns:
              metadata dict
         """
        from datamart_isi.profilers.basic_profiler import BasicProfiler, VariableMetadata, GlobalMetadata

        global_metadata = GlobalMetadata.construct_global(description=cls.DEFAULT_DESCRIPTION)
        global_metadata = BasicProfiler.basic_profiling_entire(global_metadata=global_metadata, data=data)
        metadata_dict = global_metadata.value

        # for col_offset in range(data.shape[1]):
        #     variable_metadata = BasicProfiler.basic_profiling_column(
        #         description={},
        #         variable_metadata=VariableMetadata.construct_variable(description={}),
        #         column=data.iloc[:, col_offset]
        #     )
        #     global_metadata.add_variable_metadata(variable_metadata)
        hyper1 = ProfilerHyperparams.defaults()
        hyper2 = CleaningFeaturizerHyperparameter.defaults()
        clean_f = CleaningFeaturizer(hyperparams=hyper2)
        profiler = Profiler(hyperparams=hyper1)
        profiled_df = profiler.produce(inputs=data).value
        clean_f.set_training_data(inputs=profiled_df)
        clean_f.fit()
        cleaned_df = clean_f.produce(inputs=profiled_df).value
        cleaned_df_metadata = cleaned_df.metadata

        for i in range(data.shape[1]):
            each_column_metadata = cleaned_df_metadata.query((ALL_ELEMENTS, i))
            column_name = data.columns[i]
            if "datetime" in data.iloc[:, i].dtype.name:
                semantic_type = ("http://schema.org/DateTime", 'https://metadata.datadrivendiscovery.org/types/Attribute')
            else:
                semantic_type = each_column_metadata['semantic_types']
            variable_metadata = {'datamart_id': None,
                                 'semantic_type': semantic_type,
                                 'name': column_name,
                                 'description': 'column name: {}, dtype: {}'.format(column_name, cleaned_df.iloc[:, i].dtype.name)
                                 }
            metadata_dict['variables'].append(variable_metadata)

        if original_meta:
            metadata_dict.update(original_meta)
        return metadata_dict
